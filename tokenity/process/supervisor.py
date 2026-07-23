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
    instance_id: str | None = None
    operation_id: str | None = None
    pid: int | None = None
    return_code: int | None = None
    command: list[str] | None = None
    log_path: str | None = None
    message: str | None = None
    log_tail: str | None = None
    process_resident_bytes: int | None = None
    sampled_at: float | None = None


class RoleSupervisor:
    def __init__(self, log_dir: Path | None = None) -> None:
        self.log_dir = log_dir or Path.home() / "Library" / "Application Support" / "Tokenity" / "logs"
        self._processes: dict[tuple[str, str | None], subprocess.Popen[bytes]] = {}
        self._commands: dict[tuple[str, str | None], list[str]] = {}
        self._logs: dict[tuple[str, str | None], Path] = {}
        self._new_sessions: dict[tuple[str, str | None], bool] = {}
        self._operation_ids: dict[tuple[str, str | None], str | None] = {}
        self._lock = threading.RLock()

    def start(
        self,
        role: str,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        start_new_session: bool = True,
        instance_id: str | None = None,
        operation_id: str | None = None,
    ) -> RoleStatus:
        key = (role, instance_id)
        with self._lock:
            existing = self._processes.get(key)
            if existing and existing.poll() is None:
                if self._commands.get(key) == command and self._operation_ids.get(key) == operation_id:
                    return self.status(role, instance_id=instance_id)
                # A same-role request for a different model/configuration must
                # replace only that instance instead of another model process.
                self.stop(role, timeout=10, instance_id=instance_id)

            self.log_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"-{_safe_log_component(instance_id)}" if instance_id else ""
            log_path = self.log_dir / f"{role}{suffix}.log"
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
            self._processes[key] = process
            self._commands[key] = command
            self._logs[key] = log_path
            self._new_sessions[key] = start_new_session
            self._operation_ids[key] = operation_id
            return self.status(role, instance_id=instance_id)

    def stop(self, role: str, timeout: float = 10.0, *, instance_id: str | None = None) -> RoleStatus:
        key = (role, instance_id)
        with self._lock:
            process = self._processes.get(key)
            command = self._commands.get(key)
            if not process or process.poll() is not None:
                if command and instance_id is None:
                    _terminate_related_processes(command, signal.SIGTERM)
                    time.sleep(min(timeout, 0.5))
                    _terminate_related_processes(command, signal.SIGKILL)
                return self.status(role, instance_id=instance_id)  # type: ignore[return-value]

            if self._new_sessions.get(key, True):
                _terminate_process_group(process, signal.SIGTERM)
            else:
                process.terminate()
            if instance_id is None:
                _terminate_related_processes(command, signal.SIGTERM)
            deadline = time.monotonic() + timeout
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            if process.poll() is None:
                if self._new_sessions.get(key, True):
                    _terminate_process_group(process, signal.SIGKILL)
                else:
                    process.kill()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            if instance_id is None:
                _terminate_related_processes(command, signal.SIGKILL)
            return self.status(role, instance_id=instance_id)

    def request_stop(self, role: str, *, instance_id: str | None = None) -> RoleStatus:
        """Ask a role to stop without waiting or escalating to SIGKILL.

        Distributed ranks need a short coordination phase where every rank has
        observed the stop request before any one process is reaped.  Keeping
        this separate from ``stop`` lets the Node Agent fan the request out to
        all ranks first and only then wait for them to leave their collectives.
        """

        key = (role, instance_id)
        with self._lock:
            process = self._processes.get(key)
            if not process or process.poll() is not None:
                return self.status(role, instance_id=instance_id)  # type: ignore[return-value]
            if self._new_sessions.get(key, True):
                _terminate_process_group(process, signal.SIGTERM)
            else:
                process.terminate()
            return self.status(role, instance_id=instance_id)  # type: ignore[return-value]

    def wait(self, role: str, timeout: float = 10.0, *, instance_id: str | None = None) -> RoleStatus:
        """Wait for a role to exit without sending another signal."""

        key = (role, instance_id)
        with self._lock:
            process = self._processes.get(key)
        if process is None or process.poll() is not None:
            return self.status(role, instance_id=instance_id)  # type: ignore[return-value]
        deadline = time.monotonic() + timeout
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        return self.status(role, instance_id=instance_id)  # type: ignore[return-value]

    def stop_all(self, roles: list[str] | tuple[str, ...] | None = None, timeout: float = 10.0) -> list[RoleStatus]:
        with self._lock:
            keys = sorted(set(self._processes) | set(self._commands), key=lambda item: (item[0], item[1] or ""))
            if roles is not None:
                requested = set(roles)
                keys = [key for key in keys if key[0] in requested]
        return [self.stop(role, timeout=timeout, instance_id=instance_id) for role, instance_id in keys]

    def cleanup_orphaned_model_processes(self) -> list[dict[str, object]]:
        """Reconcile, but never silently kill, ranks from a previous Agent.

        A freshly restarted Agent is not the parent of those processes and
        therefore cannot prove their instance lease or request count. They are
        reported as orphaned for an explicit operator decision instead of being
        destroyed during Agent startup.
        """

        return _matching_processes(["tokenity distributed-openai serve --model"])

    def status(
        self,
        role: str | None = None,
        *,
        instance_id: str | None = None,
    ) -> RoleStatus | list[RoleStatus]:
        with self._lock:
            if role is not None:
                key = (role, instance_id)
                process = self._processes.get(key)
                if not process:
                    log_path = self._logs.get(key)
                    return RoleStatus(
                        role=role,
                        state="stopped",
                        instance_id=instance_id,
                        operation_id=self._operation_ids.get(key),
                        command=self._commands.get(key),
                        log_path=_str(log_path),
                        sampled_at=time.time(),
                    )
                return_code = process.poll()
                state = "running" if return_code is None else "stopped"
                log_path = self._logs.get(key)
                log_tail = _tail_log(log_path) if state == "stopped" else _failure_log_tail(log_path)
                if state == "running" and log_tail is not None:
                    state = "failed"
                return RoleStatus(
                    role=role,
                    state=state,
                    instance_id=instance_id,
                    operation_id=self._operation_ids.get(key),
                    pid=process.pid if state in {"running", "failed"} else None,
                    return_code=return_code,
                    command=self._commands.get(key),
                    log_path=_str(log_path),
                    message=_status_message(role, return_code, log_tail),
                    log_tail=log_tail,
                    process_resident_bytes=(
                        _process_resident_bytes(process.pid) if return_code is None else None
                    ),
                    sampled_at=time.time(),
                )
            keys = sorted(set(self._processes) | set(self._commands), key=lambda item: (item[0], item[1] or ""))
            return [
                self.status(item_role, instance_id=item_instance)
                for item_role, item_instance in keys
            ]  # type: ignore[misc]


def _str(path: Path | None) -> str | None:
    return str(path) if path else None


def _safe_log_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)[:80]


def _process_resident_bytes(pid: int) -> int | None:
    try:
        completed = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        kibibytes = int(completed.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return kibibytes * 1024


def _matching_processes(markers: list[str]) -> list[dict[str, object]]:
    try:
        completed = subprocess.run(
            ["/bin/ps", "-axo", "pid=,command="],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    matches: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == os.getpid() or not any(marker in command for marker in markers):
            continue
        matches.append(
            {
                "pid": pid,
                "command": command,
                "state": "orphaned",
                "reason": "Process predates the current Node Agent and has no verified request lease.",
            }
        )
    return matches


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
