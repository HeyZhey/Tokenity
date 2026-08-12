from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from tokenity.paths import log_root, state_root


DEFAULT_STATE_PATH = state_root() / "node-agent-watchdog.json"
DEFAULT_MAINTENANCE_PATH = state_root() / "maintenance.json"
DEFAULT_LOG_PATH = log_root() / "node-agent-watchdog.log"
DEFAULT_LOCK_PATH = state_root() / "node-agent-watchdog.lock"


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    status_code: int | None = None
    status: str | None = None
    error: str | None = None

    @classmethod
    def healthy(cls, status: str = "healthy") -> "ProbeResult":
        return cls(ok=True, status_code=200, status=status)

    @classmethod
    def failed(cls, reason: str, status_code: int | None = None) -> "ProbeResult":
        return cls(ok=False, status_code=status_code, error=reason)


@dataclass(frozen=True)
class AgentSnapshot:
    process_exists: bool
    port_listening: bool
    launchd_running: bool


@dataclass(frozen=True)
class WatchdogConfig:
    agent_port: int = 9100
    probe_timeout_seconds: float = 2.0
    probe_interval_seconds: float = 5.0
    failure_threshold: int = 3
    base_backoff_seconds: float = 10.0
    max_backoff_seconds: float = 300.0
    restart_window_seconds: float = 900.0
    max_restarts_in_window: int = 4
    keepalive_grace_seconds: float = 20.0
    launchd_label: str = "ai.tokenity.node-agent"
    launchd_domain: str = "system"

    def __post_init__(self) -> None:
        if not 1 <= self.agent_port <= 65_535:
            raise ValueError("agent_port must be a valid TCP port")
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        if self.max_restarts_in_window < 1:
            raise ValueError("max_restarts_in_window must be positive")
        for name in (
            "probe_timeout_seconds",
            "probe_interval_seconds",
            "base_backoff_seconds",
            "max_backoff_seconds",
            "restart_window_seconds",
            "keepalive_grace_seconds",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class WatchdogState:
    phase: str = "online"
    consecutive_failures: int = 0
    restart_count: int = 0
    restart_times: list[float] = field(default_factory=list)
    last_failure_reason: str | None = None
    last_restart_time: float | None = None
    last_recovery_duration_seconds: float | None = None
    last_success_time: float | None = None
    last_probe_time: float | None = None
    backoff_until: float | None = None
    process_missing_since: float | None = None
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def from_mapping(cls, payload: object) -> "WatchdogState":
        if not isinstance(payload, dict):
            return cls()
        fields = cls.__dataclass_fields__
        values = {key: value for key, value in payload.items() if key in fields}
        try:
            return cls(**values)
        except (TypeError, ValueError):
            return cls()


class SingleInstanceLock:
    def __init__(self, path: Path):
        self.path = path
        self._handle = None

    def acquire(self) -> bool:
        if self._handle is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def bounded_append(path: Path, message: str, *, max_bytes: int = 1_048_576) -> None:
    """Append one line while retaining only the newest bounded diagnostics."""

    if max_bytes < 256:
        raise ValueError("max_bytes must be at least 256")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (message.rstrip("\n") + "\n").encode("utf-8", errors="replace")
    if len(line) > max_bytes:
        line = line[-max_bytes:]
    existing = b""
    try:
        if path.exists() and path.stat().st_size + len(line) > max_bytes:
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size - (max_bytes - len(line))))
                existing = handle.read()
            newline = existing.find(b"\n")
            if newline >= 0:
                existing = existing[newline + 1 :]
    except OSError:
        existing = b""
    if existing:
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_bytes((existing + line)[-max_bytes:])
        os.replace(temporary, path)
    else:
        mode = "ab" if path.exists() and path.stat().st_size + len(line) <= max_bytes else "wb"
        with path.open(mode) as handle:
            handle.write(line)


class AgentWatchdog:
    def __init__(
        self,
        *,
        config: WatchdogConfig = WatchdogConfig(),
        state_path: Path = DEFAULT_STATE_PATH,
        maintenance_path: Path = DEFAULT_MAINTENANCE_PATH,
        probe_local: Callable[[str, float], ProbeResult] | None = None,
        inspect_local: Callable[[], AgentSnapshot] | None = None,
        restart_agent: Callable[[], bool] | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.state_path = state_path
        self.maintenance_path = maintenance_path
        self.probe_local = probe_local or _probe
        self.inspect_local = inspect_local or (
            lambda: inspect_agent(
                config.agent_port,
                config.launchd_label,
                config.launchd_domain,
            )
        )
        self.restart_agent = restart_agent or (
            lambda: kickstart_agent(config.launchd_label, config.launchd_domain)
        )
        self.monotonic_clock = monotonic_clock
        self.wall_clock = wall_clock
        self.health_url = (
            f"http://127.0.0.1:{self.config.agent_port}/v1/node/health"
        )
        self.state = self._load_state()

    def _load_state(self) -> WatchdogState:
        try:
            return WatchdogState.from_mapping(
                json.loads(self.state_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError):
            return WatchdogState()

    def _persist(self) -> WatchdogState:
        self.state.updated_at = self.wall_clock()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(
            self.state_path.suffix + f".{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(asdict(self.state), sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)
        return WatchdogState.from_mapping(asdict(self.state))

    def _maintenance_active(self) -> bool:
        try:
            payload = json.loads(self.maintenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        until = payload.get("until") if isinstance(payload, dict) else None
        return isinstance(until, (int, float)) and until > self.wall_clock()

    def tick(self) -> WatchdogState:
        # Backoff and restart-window timestamps are persisted across watchdog
        # and machine restarts, so they must share the wall-clock domain.
        # A persisted monotonic timestamp would survive the file but not a
        # reboot and could leave the circuit open indefinitely.
        now = self.wall_clock()
        self.state.last_probe_time = self.wall_clock()
        if self._maintenance_active():
            self.state.phase = "maintenance"
            self.state.consecutive_failures = 0
            self.state.last_failure_reason = None
            return self._persist()

        result = self.probe_local(self.health_url, self.config.probe_timeout_seconds)
        if result.ok:
            recovery_started = self.state.last_restart_time
            was_recovering = self.state.phase in {"recovering", "restarting"}
            self.state.phase = "online" if result.status != "degraded" else "degraded"
            self.state.consecutive_failures = 0
            self.state.last_success_time = self.wall_clock()
            self.state.last_failure_reason = None
            self.state.backoff_until = None
            self.state.process_missing_since = None
            if recovery_started is not None and was_recovering:
                self.state.last_recovery_duration_seconds = max(
                    0.0, self.wall_clock() - recovery_started
                )
            return self._persist()

        self.state.consecutive_failures += 1
        self.state.last_failure_reason = result.error or (
            f"health_http_{result.status_code}" if result.status_code else "health_failed"
        )
        if self.state.consecutive_failures < self.config.failure_threshold:
            self.state.phase = (
                "recovering"
                if self.state.backoff_until is not None and now < self.state.backoff_until
                else "degraded"
            )
            return self._persist()

        snapshot = self.inspect_local()
        if not snapshot.process_exists and snapshot.launchd_running:
            if self.state.process_missing_since is None:
                self.state.process_missing_since = now
            if now - self.state.process_missing_since < self.config.keepalive_grace_seconds:
                self.state.phase = "recovering"
                self.state.last_failure_reason = "process_exited_keepalive_pending"
                return self._persist()
        else:
            self.state.process_missing_since = None

        cutoff = now - self.config.restart_window_seconds
        self.state.restart_times = [
            value for value in self.state.restart_times if value >= cutoff
        ]
        if len(self.state.restart_times) >= self.config.max_restarts_in_window:
            self.state.phase = "circuit_open"
            return self._persist()
        if self.state.backoff_until is not None and now < self.state.backoff_until:
            self.state.phase = "recovering"
            return self._persist()

        restarted = self.restart_agent()
        if not restarted:
            self.state.phase = "recovering"
            self.state.last_failure_reason = "kickstart_failed"
            return self._persist()
        self.state.restart_times.append(now)
        self.state.restart_count += 1
        self.state.last_restart_time = self.wall_clock()
        backoff = min(
            self.config.base_backoff_seconds
            * (2 ** max(0, len(self.state.restart_times) - 1)),
            self.config.max_backoff_seconds,
        )
        self.state.backoff_until = now + backoff
        self.state.phase = "restarting"
        self.state.consecutive_failures = 0
        return self._persist()


def _probe(url: str, timeout: float) -> ProbeResult:
    request = Request(url, method="GET")
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            data = response.read(32_768)
            payload = json.loads(data) if data else {}
            status = payload.get("status") if isinstance(payload, dict) else None
            return ProbeResult(
                ok=response.status == 200 and status in {"healthy", "degraded"},
                status_code=response.status,
                status=str(status) if status else None,
                error=None if response.status == 200 else f"health_http_{response.status}",
            )
    except HTTPError as exc:
        return ProbeResult.failed(f"health_http_{exc.code}", exc.code)
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return ProbeResult.failed(str(exc))


def inspect_agent(
    port: int,
    launchd_label: str,
    launchd_domain: str = "system",
) -> AgentSnapshot:
    process = subprocess.run(
        [
            "/usr/bin/pgrep",
            "-f",
            rf"tokenity node-agent .*--port[ =]{port}([[:space:]]|$)",
        ],
        capture_output=True,
        timeout=2,
    )
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            port_listening = True
    except OSError:
        port_listening = False
    launchd = subprocess.run(
        ["/bin/launchctl", "print", f"{launchd_domain}/{launchd_label}"],
        capture_output=True,
        timeout=3,
    )
    return AgentSnapshot(
        process_exists=process.returncode == 0,
        port_listening=port_listening,
        launchd_running=launchd.returncode == 0,
    )


def kickstart_agent(
    launchd_label: str,
    launchd_domain: str = "system",
) -> bool:
    completed = subprocess.run(
        ["/bin/launchctl", "kickstart", "-k", f"{launchd_domain}/{launchd_label}"],
        capture_output=True,
        timeout=15,
    )
    return completed.returncode == 0


def _status_server(
    watchdog: AgentWatchdog,
    host: str,
    port: int,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in {"/health", "/v1/watchdog/status"}:
                self.send_error(404)
                return
            payload = json.dumps(asdict(watchdog.state), sort_keys=True).encode("utf-8")
            self.send_response(503 if watchdog.state.phase == "circuit_open" else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tokenity Node Agent liveness watchdog")
    parser.add_argument("--agent-port", type=int, default=9100)
    parser.add_argument("--status-host", default="0.0.0.0")
    parser.add_argument("--status-port", type=int, default=9101)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--maintenance-path", type=Path, default=DEFAULT_MAINTENANCE_PATH)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--launchd-label", default="ai.tokenity.node-agent")
    parser.add_argument("--launchd-domain", default="system")
    arguments = parser.parse_args(argv)
    config = WatchdogConfig(
        agent_port=arguments.agent_port,
        launchd_label=arguments.launchd_label,
        launchd_domain=arguments.launchd_domain,
    )
    lock = SingleInstanceLock(arguments.lock_path)
    if not lock.acquire():
        return 0
    watchdog = AgentWatchdog(
        config=config,
        state_path=arguments.state_path,
        maintenance_path=arguments.maintenance_path,
    )
    server = _status_server(watchdog, arguments.status_host, arguments.status_port)
    try:
        while True:
            previous = watchdog.state.phase
            state = watchdog.tick()
            if state.phase != previous or state.phase in {"restarting", "circuit_open"}:
                bounded_append(arguments.log_path, json.dumps(asdict(state), sort_keys=True))
            time.sleep(config.probe_interval_seconds)
    except KeyboardInterrupt:
        return 0
    finally:
        server.shutdown()
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
