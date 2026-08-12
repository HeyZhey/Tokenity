from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from tokenity.node_agent.watchdog import (
    AgentSnapshot,
    AgentWatchdog,
    SingleInstanceLock,
    WatchdogConfig,
    _status_server,
    bounded_append,
)


class IsolatedAgentSupervisor:
    """Unprivileged hardware-test supervisor for the isolated port-9200 Agent."""

    def __init__(self, worktree: Path, state_root: Path) -> None:
        self.worktree = worktree
        self.state_root = state_root
        self.log_handle = None
        self.process: subprocess.Popen[bytes] | None = None
        self.stopping = False

    def start_agent(self) -> bool:
        if self.stopping:
            return False
        self.state_root.mkdir(parents=True, exist_ok=True)
        (self.state_root / "instances").mkdir(parents=True, exist_ok=True)
        (self.state_root / "logs").mkdir(parents=True, exist_ok=True)
        if self.log_handle is None:
            self.log_handle = (self.state_root / "logs" / "agent.log").open("ab")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.worktree)
        environment["TOKENITY_INSTANCE_STATE_ROOT"] = str(
            self.state_root / "instances"
        )
        environment["TOKENITY_DISABLE_UNJOURNALED_ORPHAN_SCAN"] = "1"
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "tokenity",
                "node-agent",
                "--host",
                "0.0.0.0",
                "--port",
                "9200",
            ],
            cwd=self.worktree,
            env=environment,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return True

    def inspect(self) -> AgentSnapshot:
        process_exists = self.process is not None and self.process.poll() is None
        try:
            with socket.create_connection(("127.0.0.1", 9200), timeout=0.5):
                port_listening = True
        except OSError:
            port_listening = False
        return AgentSnapshot(
            process_exists=process_exists,
            port_listening=port_listening,
            launchd_running=True,
        )

    def restart_agent(self) -> bool:
        self.kill_agent()
        return self.start_agent()

    def kill_agent(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            os.kill(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return

    def stop(self) -> None:
        self.stopping = True
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.kill_agent()
        if self.log_handle is not None:
            self.log_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("worktree", type=Path)
    parser.add_argument("state_root", type=Path)
    arguments = parser.parse_args()
    supervisor = IsolatedAgentSupervisor(arguments.worktree, arguments.state_root)
    lock = SingleInstanceLock(arguments.state_root / "supervisor.lock")
    if not lock.acquire():
        return 0
    supervisor.start_agent()
    watchdog = AgentWatchdog(
        config=WatchdogConfig(agent_port=9200),
        state_path=arguments.state_root / "watchdog.json",
        maintenance_path=arguments.state_root / "maintenance.json",
        inspect_local=supervisor.inspect,
        restart_agent=supervisor.restart_agent,
    )
    server = _status_server(watchdog, "0.0.0.0", 9201)

    def request_stop(*_args: object) -> None:
        supervisor.stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        while not supervisor.stopping:
            if supervisor.process is None or supervisor.process.poll() is not None:
                supervisor.start_agent()
            previous = watchdog.state.phase
            state = watchdog.tick()
            if state.phase != previous or state.phase in {"restarting", "circuit_open"}:
                bounded_append(
                    arguments.state_root / "logs" / "watchdog.log",
                    str(state),
                )
            time.sleep(watchdog.config.probe_interval_seconds)
    finally:
        server.shutdown()
        supervisor.stop()
        lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
