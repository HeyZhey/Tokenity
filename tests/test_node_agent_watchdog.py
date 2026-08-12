from __future__ import annotations

import json
from pathlib import Path

from tokenity.node_agent.watchdog import (
    AgentSnapshot,
    AgentWatchdog,
    ProbeResult,
    SingleInstanceLock,
    WatchdogConfig,
    bounded_append,
)


class FakeClock:
    def __init__(self, value: float = 1_000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def make_watchdog(
    tmp_path: Path,
    *,
    results: list[ProbeResult],
    snapshot: AgentSnapshot | None = None,
    config: WatchdogConfig | None = None,
    clock: FakeClock | None = None,
):
    restarts: list[float] = []
    observed_urls: list[str] = []
    clock = clock or FakeClock()

    def probe(url: str, _timeout: float) -> ProbeResult:
        observed_urls.append(url)
        return results.pop(0) if results else ProbeResult.failed("still unavailable")

    watchdog = AgentWatchdog(
        config=config
        or WatchdogConfig(
            failure_threshold=3,
            base_backoff_seconds=10,
            max_backoff_seconds=60,
            restart_window_seconds=300,
            max_restarts_in_window=4,
            keepalive_grace_seconds=20,
        ),
        state_path=tmp_path / "watchdog-state.json",
        maintenance_path=tmp_path / "maintenance.json",
        probe_local=probe,
        inspect_local=lambda: snapshot
        or AgentSnapshot(process_exists=True, port_listening=True, launchd_running=True),
        restart_agent=lambda: restarts.append(clock()) or True,
        monotonic_clock=clock,
        wall_clock=clock,
    )
    return watchdog, restarts, observed_urls, clock


def test_healthy_agent_never_restarts_and_probe_is_strictly_loopback(tmp_path: Path):
    watchdog, restarts, urls, _ = make_watchdog(
        tmp_path,
        results=[ProbeResult.healthy() for _ in range(4)],
    )

    states = [watchdog.tick() for _ in range(4)]

    assert restarts == []
    assert all(state.phase == "online" for state in states)
    assert urls == ["http://127.0.0.1:9100/v1/node/health"] * 4


def test_transient_failures_do_not_restart_and_threshold_restarts_only_once(tmp_path: Path):
    watchdog, restarts, _, _ = make_watchdog(
        tmp_path,
        results=[ProbeResult.failed("timeout") for _ in range(5)],
    )

    first = watchdog.tick()
    second = watchdog.tick()
    third = watchdog.tick()
    fourth = watchdog.tick()

    assert first.consecutive_failures == 1
    assert second.consecutive_failures == 2
    assert restarts == [1_000.0]
    assert third.phase == "restarting"
    assert fourth.phase == "recovering"
    assert fourth.restart_count == 1


def test_process_exit_is_left_to_launchd_keepalive_during_grace_window(tmp_path: Path):
    watchdog, restarts, _, _ = make_watchdog(
        tmp_path,
        results=[ProbeResult.failed("connection refused") for _ in range(4)],
        snapshot=AgentSnapshot(
            process_exists=False,
            port_listening=False,
            launchd_running=True,
        ),
    )

    state = None
    for _ in range(4):
        state = watchdog.tick()

    assert restarts == []
    assert state is not None
    assert state.phase == "recovering"
    assert state.last_failure_reason == "process_exited_keepalive_pending"


def test_hung_process_uses_exponential_backoff(tmp_path: Path):
    clock = FakeClock()
    config = WatchdogConfig(
        failure_threshold=1,
        base_backoff_seconds=10,
        max_backoff_seconds=60,
        restart_window_seconds=300,
        max_restarts_in_window=5,
    )
    watchdog, restarts, _, _ = make_watchdog(
        tmp_path,
        results=[ProbeResult.failed("hung") for _ in range(8)],
        config=config,
        clock=clock,
    )

    watchdog.tick()
    clock.advance(9)
    watchdog.tick()
    clock.advance(1)
    watchdog.tick()
    clock.advance(19)
    watchdog.tick()
    clock.advance(1)
    watchdog.tick()

    assert restarts == [1_000.0, 1_010.0, 1_030.0]


def test_restart_window_opens_circuit_and_persists_diagnostics(tmp_path: Path):
    clock = FakeClock()
    config = WatchdogConfig(
        failure_threshold=1,
        base_backoff_seconds=1,
        max_backoff_seconds=2,
        restart_window_seconds=60,
        max_restarts_in_window=2,
    )
    watchdog, restarts, _, _ = make_watchdog(
        tmp_path,
        results=[ProbeResult.failed("hung") for _ in range(5)],
        config=config,
        clock=clock,
    )

    watchdog.tick()
    clock.advance(1)
    watchdog.tick()
    clock.advance(2)
    state = watchdog.tick()

    assert len(restarts) == 2
    assert state.phase == "circuit_open"
    persisted = json.loads((tmp_path / "watchdog-state.json").read_text(encoding="utf-8"))
    assert persisted["phase"] == "circuit_open"
    assert persisted["restart_count"] == 2
    assert persisted["last_failure_reason"] == "hung"


def test_recovery_duration_and_restart_window_survive_watchdog_restart(tmp_path: Path):
    clock = FakeClock()
    config = WatchdogConfig(
        failure_threshold=1,
        base_backoff_seconds=1,
        max_backoff_seconds=1,
        restart_window_seconds=60,
        max_restarts_in_window=2,
    )
    state_path = tmp_path / "watchdog-state.json"
    maintenance_path = tmp_path / "maintenance.json"
    first_restarts: list[float] = []
    first = AgentWatchdog(
        config=config,
        state_path=state_path,
        maintenance_path=maintenance_path,
        probe_local=lambda *_: ProbeResult.failed("hung"),
        inspect_local=lambda: AgentSnapshot(True, True, True),
        restart_agent=lambda: first_restarts.append(clock()) or True,
        monotonic_clock=clock,
        wall_clock=clock,
    )
    first.tick()
    clock.advance(1)
    first.tick()

    recovered_watchdog = AgentWatchdog(
        config=config,
        state_path=state_path,
        maintenance_path=maintenance_path,
        probe_local=lambda *_: ProbeResult.healthy(),
        inspect_local=lambda: AgentSnapshot(True, True, True),
        restart_agent=lambda: True,
        monotonic_clock=clock,
        wall_clock=clock,
    )
    recovered = recovered_watchdog.tick()
    clock.advance(10)
    stable = recovered_watchdog.tick()

    assert first_restarts == [1_000.0, 1_001.0]
    assert recovered.phase == "online"
    assert recovered.last_recovery_duration_seconds == 0
    assert stable.last_recovery_duration_seconds == 0
    assert recovered.restart_times == [1_000.0, 1_001.0]


def test_maintenance_window_suppresses_failure_count_and_restart(tmp_path: Path):
    clock = FakeClock()
    maintenance = tmp_path / "maintenance.json"
    maintenance.write_text(json.dumps({"until": clock() + 120}), encoding="utf-8")
    watchdog, restarts, _, _ = make_watchdog(
        tmp_path,
        results=[ProbeResult.failed("upgrade") for _ in range(4)],
        clock=clock,
    )

    state = watchdog.tick()

    assert restarts == []
    assert state.phase == "maintenance"
    assert state.consecutive_failures == 0


def test_watchdog_single_instance_lock_and_bounded_log(tmp_path: Path):
    first = SingleInstanceLock(tmp_path / "watchdog.lock")
    second = SingleInstanceLock(tmp_path / "watchdog.lock")

    assert first.acquire() is True
    assert second.acquire() is False
    first.release()
    assert second.acquire() is True
    second.release()

    log_path = tmp_path / "watchdog.log"
    for index in range(200):
        bounded_append(log_path, f"event-{index}-" + ("x" * 80), max_bytes=2_048)

    assert log_path.stat().st_size <= 2_048
    assert "event-199" in log_path.read_text(encoding="utf-8")
