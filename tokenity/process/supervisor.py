from __future__ import annotations

import os
import re
import select
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RoleStatus:
    role: str
    state: str
    pid: int | None = None
    return_code: int | None = None
    command: list[str] | None = None
    log_path: str | None = None
    message: str | None = None
    log_tail: str | None = None


class RoleSupervisor:
    def __init__(self, log_dir: Path | None = None) -> None:
        self.log_dir = log_dir or Path.home() / "Library" / "Application Support" / "Tokenity" / "logs"
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._commands: dict[str, list[str]] = {}
        self._logs: dict[str, Path] = {}
        self._new_sessions: dict[str, bool] = {}
        self._lock = threading.RLock()

    def start(
        self,
        role: str,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        start_new_session: bool = True,
    ) -> RoleStatus:
        with self._lock:
            existing = self._processes.get(role)
            if existing and existing.poll() is None:
                if self._commands.get(role) == command:
                    return self.status(role)
                # A same-role request for a different model/configuration must
                # replace the old backend instead of silently reusing it.
                self.stop(role, timeout=10)

            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"{role}.log"
            merged_env = os.environ.copy()
            if env:
                merged_env.update(env)
            log_path.write_bytes(b"")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                env=merged_env,
                cwd=str(cwd) if cwd is not None else None,
                start_new_session=start_new_session,
            )
            threading.Thread(
                target=_copy_process_output,
                args=(process, log_path),
                daemon=True,
            ).start()
            self._processes[role] = process
            self._commands[role] = command
            self._logs[role] = log_path
            self._new_sessions[role] = start_new_session
            return self.status(role)

    def stop(self, role: str, timeout: float = 10.0) -> RoleStatus:
        with self._lock:
            process = self._processes.get(role)
            command = self._commands.get(role)
            if not process or process.poll() is not None:
                if command:
                    _terminate_related_processes(command, signal.SIGTERM)
                    time.sleep(min(timeout, 0.5))
                    _terminate_related_processes(command, signal.SIGKILL)
                return self.status(role)  # type: ignore[return-value]

            if self._new_sessions.get(role, True):
                _terminate_process_group(process, signal.SIGTERM)
            else:
                process.terminate()
            _terminate_related_processes(command, signal.SIGTERM)
            deadline = time.monotonic() + timeout
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            if process.poll() is None:
                if self._new_sessions.get(role, True):
                    _terminate_process_group(process, signal.SIGKILL)
                else:
                    process.kill()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            _terminate_related_processes(command, signal.SIGKILL)
            return self.status(role)

    def request_stop(self, role: str) -> RoleStatus:
        """Ask a role to stop without waiting or escalating to SIGKILL.

        Distributed ranks need a short coordination phase where every rank has
        observed the stop request before any one process is reaped.  Keeping
        this separate from ``stop`` lets the Node Agent fan the request out to
        all ranks first and only then wait for them to leave their collectives.
        """

        with self._lock:
            process = self._processes.get(role)
            if not process or process.poll() is not None:
                return self.status(role)  # type: ignore[return-value]
            if self._new_sessions.get(role, True):
                _terminate_process_group(process, signal.SIGTERM)
            else:
                process.terminate()
            return self.status(role)  # type: ignore[return-value]

    def wait(self, role: str, timeout: float = 10.0) -> RoleStatus:
        """Wait for a role to exit without sending another signal."""

        with self._lock:
            process = self._processes.get(role)
        if process is None or process.poll() is not None:
            return self.status(role)  # type: ignore[return-value]
        deadline = time.monotonic() + timeout
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        return self.status(role)  # type: ignore[return-value]

    def stop_all(self, roles: list[str] | tuple[str, ...] | None = None, timeout: float = 10.0) -> list[RoleStatus]:
        with self._lock:
            selected = list(roles) if roles is not None else sorted(set(self._processes) | set(self._commands))
        return [self.stop(role, timeout=timeout) for role in selected]

    def cleanup_orphaned_model_processes(self) -> None:
        """Terminate model ranks left behind by a previous Agent process."""

        _terminate_matching_processes(
            ["tokenity distributed-openai serve --model"],
            signal.SIGTERM,
        )
        time.sleep(0.25)
        _terminate_matching_processes(
            ["tokenity distributed-openai serve --model"],
            signal.SIGKILL,
        )

    def status(self, role: str | None = None) -> RoleStatus | list[RoleStatus]:
        with self._lock:
            if role is not None:
                process = self._processes.get(role)
                if not process:
                    log_path = self._logs.get(role)
                    return RoleStatus(
                        role=role,
                        state="stopped",
                        command=self._commands.get(role),
                        log_path=_str(log_path),
                    )
                return_code = process.poll()
                state = "running" if return_code is None else "stopped"
                log_path = self._logs.get(role)
                log_tail = _tail_log(log_path) if state == "stopped" else _failure_log_tail(log_path)
                if state == "running" and log_tail is not None:
                    state = "failed"
                return RoleStatus(
                    role=role,
                    state=state,
                    pid=process.pid if state in {"running", "failed"} else None,
                    return_code=return_code,
                    command=self._commands.get(role),
                    log_path=_str(log_path),
                    message=_status_message(role, return_code, log_tail),
                    log_tail=log_tail,
                )
            roles = sorted(set(self._processes) | set(self._commands))
            return [self.status(item) for item in roles]  # type: ignore[misc]


def _str(path: Path | None) -> str | None:
    return str(path) if path else None


def _terminate_process_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    if process.stdin is not None:
        try:
            process.stdin.close()
        except OSError:
            pass
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        process.send_signal(sig)


def _terminate_related_processes(command: list[str] | None, sig: signal.Signals) -> None:
    markers = _related_process_markers(command)
    if not markers:
        return
    _terminate_matching_processes(markers, sig)


def _terminate_matching_processes(markers: list[str], sig: signal.Signals) -> None:
    for pgid in _matching_process_groups(markers):
        if pgid in {os.getpid(), os.getpgrp()}:
            continue
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def _related_process_markers(command: list[str] | None) -> list[str]:
    if not command:
        return []
    command_text = " ".join(command)
    markers: list[str] = []
    for match in re.finditer(r"--hostfile\s+('([^']+)'|\"([^\"]+)\"|(\S+))", command_text):
        marker = next(group for group in match.groups()[1:] if group)
        if "tokenity-hostfiles" in marker:
            markers.append(marker)
    for match in re.finditer(
        r"tokenity\s+distributed-openai\s+serve\s+--model\s+('([^']+)'|\"([^\"]+)\"|(\S+))",
        command_text,
    ):
        model = next(group for group in match.groups()[1:] if group)
        markers.append(f"tokenity distributed-openai serve --model {model}")
    return markers


def _matching_process_groups(markers: list[str]) -> set[int]:
    try:
        completed = subprocess.run(
            ["/bin/ps", "-axo", "pid=,pgid=,command="],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    groups: set[int] = set()
    current_pid = os.getpid()
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            pgid = int(parts[1])
        except ValueError:
            continue
        if pid == current_pid:
            continue
        process_command = parts[2]
        if any(marker in process_command for marker in markers):
            groups.add(pgid)
    return groups


def _copy_process_output(process: subprocess.Popen[bytes], log_path: Path) -> None:
    if process.stdout is None:
        return
    fd = process.stdout.fileno()
    os.set_blocking(fd, False)
    with log_path.open("ab") as handle:
        while True:
            readable, _, _ = select.select([fd], [], [], 0.5)
            if not readable:
                if process.poll() is not None:
                    try:
                        chunk = os.read(fd, 8192)
                    except BlockingIOError:
                        break
                    except OSError:
                        break
                    if not chunk:
                        break
                    handle.write(chunk)
                    handle.flush()
                continue
            try:
                chunk = os.read(fd, 8192)
            except BlockingIOError:
                continue
            except OSError:
                break
            if not chunk:
                break
            handle.write(chunk)
            handle.flush()
        try:
            process.stdout.close()
        except OSError:
            pass


def _status_message(role: str, return_code: int | None, log_tail: str | None) -> str | None:
    if return_code is None and log_tail:
        return f"{role} reported a child-rank failure.\n{log_tail}"
    if return_code is None:
        return None
    message = f"{role} exited with code {return_code}."
    if log_tail:
        return f"{message}\n{log_tail}"
    return message


def _tail_log(path: Path | None, *, max_bytes: int = 4096, max_lines: int = 20) -> str | None:
    if path is None or not path.exists():
        return None
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes), os.SEEK_SET)
        chunk = handle.read().decode("utf-8", errors="replace")
    lines = [line for line in chunk.splitlines() if line.strip()]
    if not lines:
        return None
    return "\n".join(lines[-max_lines:])


def _failure_log_tail(path: Path | None) -> str | None:
    tail = _tail_log(path, max_bytes=16384, max_lines=40)
    if tail is None:
        return None
    fatal_markers = (
        "GPU Timeout",
        "libc++abi: terminating",
        "Node with rank",
        "exited with code 255",
        "Traceback (most recent call last)",
        "SIGBUS",
        "Bus error",
        "Broken pipe",
        "Tokenity distributed runtime failed",
        "Changing queue pair",
        "failed with errno",
    )
    if any(marker in tail for marker in fatal_markers):
        return tail
    return None
