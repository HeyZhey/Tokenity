from __future__ import annotations

import threading
import time

import pytest

from tokenity.control.instances import (
    GenerationQueueFull,
    GenerationSlotCancelled,
    GenerationSlotScheduler,
    GenerationSlotTimeout,
    InstanceConflict,
    InstanceLifecycle,
    InstanceRegistry,
    InstanceRouter,
    InvalidLifecycleTransition,
    MemoryReservationBreakdown,
    ModelInstance,
    ResourceAdmissionError,
    ResourceLedger,
    StaleInstanceUpdate,
    live_system_available_memory_bytes,
)


def _instance_values(**overrides):
    values = {
        "instance_id": "instance-a",
        "operation_id": "operation-a",
        "requested_model_id": "qwen",
        "resolved_path": "/models/qwen",
        "model_revision": "revision-a",
        "tokenizer_identity": "tokenizer-a",
        "execution_mode": "single",
        "selected_nodes": ["mac-a"],
        "rank_mapping": {"mac-a": 0},
        "world_size": 1,
        "connection_mode": "single",
        "coordinator": "mac-a",
        "http_port": 8000,
        "starting_port": 29500,
        "memory_reservation_bytes": 1024,
    }
    values.update(overrides)
    return values


def _ready_instance(
    registry: InstanceRegistry,
    instance_id: str,
    *,
    selected_nodes: list[str] | None = None,
):
    instance, _ = registry.create(
        **_instance_values(
            instance_id=instance_id,
            operation_id=f"operation-{instance_id}",
            selected_nodes=selected_nodes or ["mac-a"],
        )
    )
    instance.transition(InstanceLifecycle.LAUNCHING)
    instance.transition(InstanceLifecycle.LOADING_METADATA)
    instance.transition(InstanceLifecycle.MATERIALIZING_WEIGHTS)
    instance.transition(InstanceLifecycle.COMPILING_WARMING)
    instance.transition(InstanceLifecycle.READY)
    return instance


def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true before the deadline")
        time.sleep(0.005)


def test_lifecycle_rejects_illegal_and_stale_updates():
    registry = InstanceRegistry()
    instance, _ = registry.create(**_instance_values())

    version = instance.transition(InstanceLifecycle.LAUNCHING)
    with pytest.raises(StaleInstanceUpdate):
        instance.transition(InstanceLifecycle.LOADING_METADATA, expected_version=version - 1)
    with pytest.raises(InvalidLifecycleTransition):
        instance.transition(InstanceLifecycle.READY, expected_version=version)


def test_request_lease_blocks_unload_until_release():
    registry = InstanceRegistry()
    instance, _ = registry.create(**_instance_values())
    instance.transition(InstanceLifecycle.LAUNCHING)
    instance.transition(InstanceLifecycle.LOADING_METADATA)
    instance.transition(InstanceLifecycle.MATERIALIZING_WEIGHTS)
    instance.transition(InstanceLifecycle.COMPILING_WARMING)
    instance.transition(InstanceLifecycle.READY)

    instance.acquire_request_lease()
    with pytest.raises(InstanceConflict, match="active request"):
        instance.transition(InstanceLifecycle.UNLOADING)
    instance.release_request_lease()
    instance.transition(InstanceLifecycle.UNLOADING)
    instance.transition(InstanceLifecycle.STOPPED)

    assert instance.state == InstanceLifecycle.STOPPED


def test_registry_start_is_idempotent_by_instance_and_operation():
    registry = InstanceRegistry()
    first, created = registry.create(**_instance_values())
    same, created_again = registry.create(**_instance_values())

    assert created is True
    assert created_again is False
    assert same is first
    with pytest.raises(InstanceConflict):
        registry.create(**_instance_values(operation_id="operation-b"))


def test_stopped_instance_clears_live_health_and_memory_fields():
    instance = ModelInstance(**_instance_values())
    instance.actual_memory_bytes = 4_096
    instance.health_ready = True
    instance.health_issues = []
    instance.transition(InstanceLifecycle.LAUNCHING)
    instance.transition(InstanceLifecycle.LOADING_METADATA)
    instance.transition(InstanceLifecycle.UNLOADING)
    instance.transition(InstanceLifecycle.STOPPED)

    assert instance.actual_memory_bytes is None
    assert instance.health_ready is False
    assert instance.health_sampled_at is not None
    assert instance.health_issues == ["Instance is stopped."]


def test_resource_admission_reserves_and_releases_memory_and_ports():
    ledger = ResourceLedger(10_000, minimum_headroom_ratio=0.1)
    ledger.reserve("instance-a", 6_000, [8000, 29500])

    with pytest.raises(ResourceAdmissionError):
        ledger.reserve("instance-b", 4_000, [8010, 29600])
    with pytest.raises(ResourceAdmissionError, match="Ports already reserved"):
        ledger.reserve("instance-b", 1_000, [8000])

    ledger.release("instance-a")
    ledger.reserve("instance-b", 4_000, [8000])
    assert ledger.snapshot()["reservations"] == {"instance-b": 4_000}


def test_released_ports_are_quarantined_before_collective_epoch_reuse():
    now = [100.0]
    ledger = ResourceLedger(
        20_000,
        minimum_headroom_ratio=0,
        port_reuse_delay_seconds=30,
        monotonic_clock=lambda: now[0],
    )
    ledger.reserve("instance-a", 1_000, [8000, 29500, 29501])
    ledger.release("instance-a")

    snapshot = ledger.snapshot()
    assert snapshot["ports"] == {}
    assert set(snapshot["quarantined_ports"]) == {"8000", "29500", "29501"}
    with pytest.raises(ResourceAdmissionError, match="Ports already reserved"):
        ledger.reserve("instance-b", 1_000, [29500])

    now[0] += 30
    ledger.reserve("instance-b", 1_000, [29500])
    assert ledger.snapshot()["ports"] == {"29500": "instance-b"}


def test_resource_admission_honors_live_system_memory_headroom():
    ledger = ResourceLedger(10_000, minimum_headroom_ratio=0.1)

    with pytest.raises(ResourceAdmissionError, match="only 1,?500|only 1500"):
        ledger.reserve(
            "instance-a",
            2_000,
            [8000],
            system_available_memory_bytes=1_500,
        )

    ledger.reserve(
        "instance-a",
        1_500,
        [8000],
        system_available_memory_bytes=1_500,
    )
    snapshot = ledger.snapshot(system_available_memory_bytes=750)
    assert snapshot["available_memory_bytes"] == 750
    assert snapshot["ledger_available_memory_bytes"] == 7_500
    assert snapshot["system_available_memory_bytes"] == 750


def test_resource_admission_allows_a_per_load_headroom_override():
    ledger = ResourceLedger(10_000, minimum_headroom_ratio=0.25)
    ledger.reserve("instance-a", 6_000, [8000])

    with pytest.raises(ResourceAdmissionError):
        ledger.reserve("instance-b", 2_000, [8010])

    ledger.reserve(
        "instance-b",
        2_000,
        [8010],
        minimum_headroom_ratio=0.15,
    )
    snapshot = ledger.snapshot(minimum_headroom_ratio=0.1)
    assert snapshot["reserved_memory_bytes"] == 8_000
    assert snapshot["available_memory_bytes"] == 1_000
    assert snapshot["minimum_headroom_ratio"] == 0.1


def test_live_system_available_memory_accounts_for_wired_residue_and_pressure():
    gib = 1_073_741_824

    available = live_system_available_memory_bytes(
        {
            "total_bytes": 512 * gib,
            "in_use_bytes": 443 * gib,
            "pressure_available_ratio": 0.19,
        }
    )

    assert available == int(512 * gib * 0.9) - 443 * gib


def test_disabled_fixed_headroom_still_enforces_live_pressure_and_in_use_memory():
    available = live_system_available_memory_bytes(
        {
            "total_bytes": 10_000,
            "in_use_bytes": 2_000,
            "pressure_available_ratio": 0.5,
        },
        minimum_headroom_ratio=0.0,
    )

    assert available == 5_000


def test_live_system_available_memory_falls_back_when_signals_are_unavailable():
    assert live_system_available_memory_bytes({"total_bytes": 10_000}) is None
    assert live_system_available_memory_bytes({"total_bytes": None}) is None


def test_resource_resize_uses_live_headroom_only_for_growth():
    ledger = ResourceLedger(20_000, minimum_headroom_ratio=0)
    ledger.reserve("instance-a", 8_000, [8000])

    assert (
        ledger.resize(
            "instance-a",
            9_000,
            system_available_memory_bytes=1_000,
        )
        == 9_000
    )
    with pytest.raises(ResourceAdmissionError):
        ledger.resize(
            "instance-a",
            11_000,
            system_available_memory_bytes=1_000,
        )
    assert (
        ledger.resize(
            "instance-a",
            7_000,
            system_available_memory_bytes=0,
        )
        == 7_000
    )


def test_resource_ledger_atomically_resizes_and_reconciles_without_changing_ports():
    ledger = ResourceLedger(20_000, minimum_headroom_ratio=0)
    ledger.reserve("instance-a", 8_000, [8_000, 29_500, 29_501])
    ledger.reserve("instance-b", 4_000, [8_010])
    original_ports = ledger.snapshot()["ports"]

    assert ledger.resize("instance-a", 10_000) == 10_000
    assert ledger.snapshot()["ports"] == original_ports

    reconciled = ledger.reconcile(
        "instance-a",
        observed_peak_bytes=5_000,
        observed_footprint_bytes=6_000,
        safety_margin_ratio=0.25,
        minimum_reservation_bytes=4_000,
    )
    assert reconciled == 7_500
    assert ledger.snapshot()["reservations"]["instance-a"] == 7_500
    assert ledger.snapshot()["ports"] == original_ports

    assert ledger.reconcile(
        "instance-a",
        observed_peak_bytes=10_000,
        observed_footprint_bytes=9_000,
        safety_margin_ratio=0.25,
    ) == 12_500
    before_failed_resize = ledger.snapshot()
    with pytest.raises(ResourceAdmissionError):
        ledger.reconcile(
            "instance-a",
            observed_peak_bytes=16_000,
            observed_footprint_bytes=15_000,
            safety_margin_ratio=0.25,
        )
    assert ledger.snapshot() == before_failed_resize


def test_memory_reservation_breakdown_is_structured_and_legacy_constructor_stays_compatible():
    legacy = ModelInstance(**_instance_values(memory_reservation_bytes=1_024))
    assert legacy.memory_breakdown().estimated_total_bytes == 1_024

    breakdown = MemoryReservationBreakdown(
        weights_bytes=3_000,
        kv_cache_bytes=1_000,
        prompt_cache_bytes=500,
        mlx_cache_bytes=500,
        runtime_peak_bytes=2_000,
        os_headroom_bytes=1_000,
    )
    observed = breakdown.with_observation(
        peak_bytes=4_000,
        footprint_bytes=5_500,
    )
    instance = ModelInstance(
        **_instance_values(
            memory_reservation_bytes=breakdown.estimated_total_bytes,
            memory_reservation_breakdown=observed,
        )
    )

    assert breakdown.estimated_total_bytes == 8_000
    assert observed.observed_local_bytes == 5_500
    assert observed.recommended_reservation_bytes(safety_margin_ratio=0.2) == 6_600
    assert instance.memory_breakdown() == observed
    assert instance.to_dict()["memory_reservation_breakdown"]["weights_bytes"] == 3_000
    ledger = ResourceLedger(20_000, minimum_headroom_ratio=0)
    ledger.reserve(instance.instance_id, instance.memory_reservation_bytes, [8_000])
    assert ledger.reconcile(
        instance.instance_id,
        breakdown=observed,
        safety_margin_ratio=0.2,
    ) == 6_600


def test_router_round_robins_ready_aliases_and_honors_exact_instance_id():
    registry = InstanceRegistry()
    first, _ = registry.create(**_instance_values(instance_id="instance-a"))
    second, _ = registry.create(**_instance_values(instance_id="instance-b"))
    for instance in (first, second):
        instance.transition(InstanceLifecycle.LAUNCHING)
        instance.transition(InstanceLifecycle.LOADING_METADATA)
        instance.transition(InstanceLifecycle.MATERIALIZING_WEIGHTS)
        instance.transition(InstanceLifecycle.COMPILING_WARMING)
        instance.transition(InstanceLifecycle.READY)
    router = InstanceRouter(registry)

    assert router.resolve("qwen").instance_id == "instance-a"
    assert router.resolve("qwen").instance_id == "instance-b"
    assert router.resolve("instance-b").instance_id == "instance-b"


def test_router_filters_same_model_alias_by_exact_revision():
    registry = InstanceRegistry()
    old_a = _ready_instance(registry, "instance-old-a")
    old_b = _ready_instance(registry, "instance-old-b")
    new_a = _ready_instance(registry, "instance-new-a")
    new_b = _ready_instance(registry, "instance-new-b")
    for instance in (old_a, old_b):
        instance.model_revision = "revision-old"
    for instance in (new_a, new_b):
        instance.model_revision = "revision-new"
    router = InstanceRouter(registry)

    assert router.resolve("qwen", revision="revision-old") is old_a
    assert router.resolve("qwen", revision="revision-new") is new_a
    assert router.resolve("qwen", revision="revision-old") is old_b
    assert router.resolve("qwen", revision="revision-new") is new_b
    with pytest.raises(InstanceConflict, match="revision-missing"):
        router.resolve("qwen", revision="revision-missing")


def test_router_prefers_the_replica_with_less_active_and_queued_work():
    registry = InstanceRegistry()
    first = _ready_instance(registry, "instance-a")
    second = _ready_instance(registry, "instance-b")
    router = InstanceRouter(registry)

    first.acquire_request_lease()
    assert router.resolve("qwen") is second

    second.enqueue_request()
    first.acquire_request_lease()
    assert router.resolve("qwen") is second

    second.enqueue_request()
    assert router.resolve("qwen") is first
    assert router.resolve("instance-b") is second


def test_router_excludes_an_unhealthy_replica_until_quorum_recovers():
    registry = InstanceRegistry()
    first = _ready_instance(registry, "instance-a")
    second = _ready_instance(registry, "instance-b")
    router = InstanceRouter(registry)

    first.health_ready = False
    first.health_issues = ["rank quorum is stale"]
    assert router.resolve("qwen") is second

    first.health_ready = True
    second.acquire_request_lease()
    assert router.resolve("qwen") is first


def test_registry_serializes_request_leases_and_blocks_unload():
    registry = InstanceRegistry()
    instance, _ = registry.create(**_instance_values())
    instance.transition(InstanceLifecycle.LAUNCHING)
    instance.transition(InstanceLifecycle.LOADING_METADATA)
    instance.transition(InstanceLifecycle.MATERIALIZING_WEIGHTS)
    instance.transition(InstanceLifecycle.COMPILING_WARMING)
    instance.transition(InstanceLifecycle.READY)

    registry.acquire_request_lease(instance.instance_id)
    assert instance.state == InstanceLifecycle.BUSY
    with pytest.raises(InstanceConflict, match="active request"):
        instance.transition(InstanceLifecycle.UNLOADING)
    registry.release_request_lease(instance.instance_id)
    assert instance.state == InstanceLifecycle.READY


def test_generation_slots_serialize_overlapping_nodes_fifo_and_allow_disjoint_parallelism():
    registry = InstanceRegistry()
    first = _ready_instance(registry, "instance-a", selected_nodes=["mac-a"])
    second = _ready_instance(registry, "instance-b", selected_nodes=["mac-a", "mac-b"])
    third = _ready_instance(registry, "instance-c", selected_nodes=["mac-a"])
    _ready_instance(registry, "instance-d", selected_nodes=["mac-c"])
    scheduler = GenerationSlotScheduler(max_queue_depth=4, registry=registry)

    active = scheduler.acquire(first.instance_id, first.selected_nodes)
    order: list[str] = []
    second_acquired = threading.Event()
    third_acquired = threading.Event()
    release_second = threading.Event()
    release_third = threading.Event()

    def wait_for_slot(instance_id, selected_nodes, acquired, release):
        lease = scheduler.acquire(instance_id, selected_nodes, timeout=1)
        order.append(instance_id)
        acquired.set()
        release.wait(1)
        lease.release()

    second_thread = threading.Thread(
        target=wait_for_slot,
        args=(second.instance_id, second.selected_nodes, second_acquired, release_second),
    )
    third_thread = threading.Thread(
        target=wait_for_slot,
        args=(third.instance_id, third.selected_nodes, third_acquired, release_third),
    )
    second_thread.start()
    _wait_until(lambda: scheduler.queue_depth(second.instance_id) == 1)
    third_thread.start()
    _wait_until(lambda: scheduler.queue_depth(third.instance_id) == 1)

    disjoint = scheduler.acquire("instance-d", ["mac-c"], timeout=0.1)
    assert scheduler.active_count == 2
    assert registry.get(second.instance_id).queued_request_count == 1
    assert registry.get(second.instance_id).state == InstanceLifecycle.BUSY
    disjoint.release()

    active.release()
    assert second_acquired.wait(1)
    assert not third_acquired.wait(0.05)
    release_second.set()
    assert third_acquired.wait(1)
    release_third.set()
    second_thread.join(1)
    third_thread.join(1)

    assert order == ["instance-b", "instance-c"]
    assert scheduler.queue_depth() == 0
    assert scheduler.active_count == 0
    assert registry.get(second.instance_id).queued_request_count == 0
    assert registry.get(second.instance_id).state == InstanceLifecycle.READY


def test_generation_slot_queue_is_bounded_cancellable_timed_and_release_is_idempotent():
    scheduler = GenerationSlotScheduler(max_queue_depth=1)
    active = scheduler.acquire("instance-a", ["mac-a"])
    cancel = threading.Event()
    cancelled: list[type[BaseException]] = []

    def wait_until_cancelled():
        try:
            scheduler.acquire(
                "instance-b",
                ["mac-a"],
                timeout=1,
                cancel_event=cancel,
            )
        except BaseException as exc:
            cancelled.append(type(exc))

    waiter = threading.Thread(target=wait_until_cancelled)
    waiter.start()
    _wait_until(lambda: scheduler.queue_depth() == 1)

    with pytest.raises(GenerationQueueFull):
        scheduler.acquire("instance-c", ["mac-a"], timeout=0.1)
    cancel.set()
    waiter.join(1)
    assert cancelled == [GenerationSlotCancelled]
    assert scheduler.queue_depth() == 0

    with pytest.raises(GenerationSlotTimeout):
        scheduler.acquire("instance-c", ["mac-a"], timeout=0.02)
    assert active.release() is True
    assert active.release() is False
