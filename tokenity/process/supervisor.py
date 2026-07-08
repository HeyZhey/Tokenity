from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
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

    def start(self, role: str, command: list[str], env: dict[str, str] | None = None) -> RoleStatus:
        existing = self._processes.get(role)
        if existing and existing.poll() is None:
            return self.status(role)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{role}.log"
        log_handle = log_path.open("wb")
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=log_handle,
            stdin=subprocess.DEVNULL,
            env=merged_env,
            start_new_session=True,
        )
        self._processes[role] = process
        self._commands[role] = command
        self._logs[role] = log_path
        return self.status(role)

    def stop(self, role: str, timeout: float = 10.0) -> RoleStatus:
        process = self._processes.get(role)
        if not process or process.poll() is not None:
            return self.status(role)  # type: ignore[return-value]

        _terminate_process_group(process, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            _terminate_process_group(process, signal.SIGKILL)
        return self.status(role)

    def status(self, role: str | None = None) -> RoleStatus | list[RoleStatus]:
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
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        process.send_signal(sig)


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
