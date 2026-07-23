from __future__ import annotations

import pytest

from tokenity.control.instances import (
    InstanceConflict,
    InstanceLifecycle,
    InstanceRegistry,
    InstanceRouter,
    InvalidLifecycleTransition,
    ResourceAdmissionError,
    ResourceLedger,
    StaleInstanceUpdate,
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
