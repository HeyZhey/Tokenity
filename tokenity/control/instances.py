from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class InstanceLifecycle(str, Enum):
    DISCOVERED = "discovered"
    AVAILABLE = "available"
    QUEUED = "queued"
    LAUNCHING = "launching"
    DISTRIBUTED_INITIALIZING = "distributed_initializing"
    LOADING_METADATA = "loading_metadata"
    MATERIALIZING_WEIGHTS = "materializing_weights"
    COMPILING_WARMING = "compiling_warming"
    READY = "ready"
    BUSY = "busy"
    UNLOADING = "unloading"
    STOPPED = "stopped"
    FAILED = "failed"
    ORPHANED = "orphaned"


_TRANSITIONS: dict[InstanceLifecycle, set[InstanceLifecycle]] = {
    InstanceLifecycle.DISCOVERED: {InstanceLifecycle.AVAILABLE, InstanceLifecycle.FAILED},
    InstanceLifecycle.AVAILABLE: {InstanceLifecycle.QUEUED, InstanceLifecycle.STOPPED, InstanceLifecycle.FAILED},
    InstanceLifecycle.QUEUED: {InstanceLifecycle.LAUNCHING, InstanceLifecycle.UNLOADING, InstanceLifecycle.FAILED},
    InstanceLifecycle.LAUNCHING: {
        InstanceLifecycle.DISTRIBUTED_INITIALIZING,
        InstanceLifecycle.LOADING_METADATA,
        InstanceLifecycle.UNLOADING,
        InstanceLifecycle.FAILED,
    },
    InstanceLifecycle.DISTRIBUTED_INITIALIZING: {
        InstanceLifecycle.LOADING_METADATA,
        InstanceLifecycle.UNLOADING,
        InstanceLifecycle.FAILED,
    },
    InstanceLifecycle.LOADING_METADATA: {
        InstanceLifecycle.MATERIALIZING_WEIGHTS,
        InstanceLifecycle.UNLOADING,
        InstanceLifecycle.FAILED,
    },
    InstanceLifecycle.MATERIALIZING_WEIGHTS: {
        InstanceLifecycle.COMPILING_WARMING,
        InstanceLifecycle.UNLOADING,
        InstanceLifecycle.FAILED,
    },
    InstanceLifecycle.COMPILING_WARMING: {
        InstanceLifecycle.READY,
        InstanceLifecycle.UNLOADING,
        InstanceLifecycle.FAILED,
    },
    InstanceLifecycle.READY: {
        InstanceLifecycle.BUSY,
        InstanceLifecycle.UNLOADING,
        InstanceLifecycle.FAILED,
        InstanceLifecycle.ORPHANED,
    },
    InstanceLifecycle.BUSY: {
        InstanceLifecycle.READY,
        InstanceLifecycle.UNLOADING,
        InstanceLifecycle.FAILED,
        InstanceLifecycle.ORPHANED,
    },
    InstanceLifecycle.UNLOADING: {InstanceLifecycle.STOPPED, InstanceLifecycle.FAILED},
    InstanceLifecycle.STOPPED: {InstanceLifecycle.QUEUED},
    InstanceLifecycle.FAILED: {InstanceLifecycle.QUEUED, InstanceLifecycle.UNLOADING, InstanceLifecycle.STOPPED},
    InstanceLifecycle.ORPHANED: {InstanceLifecycle.UNLOADING, InstanceLifecycle.STOPPED, InstanceLifecycle.FAILED},
}


class InvalidLifecycleTransition(RuntimeError):
    pass


class StaleInstanceUpdate(RuntimeError):
    pass


class InstanceConflict(RuntimeError):
    pass


class ResourceAdmissionError(RuntimeError):
    pass


@dataclass
class ModelInstance:
    instance_id: str
    operation_id: str
    requested_model_id: str
    resolved_path: str
    model_revision: str | None
    tokenizer_identity: str | None
    execution_mode: str
    selected_nodes: list[str]
    rank_mapping: dict[str, int]
    world_size: int
    connection_mode: str
    coordinator: str
    http_port: int
    starting_port: int
    memory_reservation_bytes: int
    state: InstanceLifecycle = InstanceLifecycle.QUEUED
    version: int = 0
    generation: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    deadline: float | None = None
    heartbeat_at: float | None = None
    actual_memory_bytes: int | None = None
    active_request_count: int = 0
    readiness_evidence: dict[str, Any] = field(default_factory=dict)
    last_error: dict[str, Any] | None = None
    log_paths: dict[str, str] = field(default_factory=dict)
    process_identities: dict[str, int] = field(default_factory=dict)

    def transition(
        self,
        state: InstanceLifecycle,
        *,
        expected_version: int | None = None,
        readiness_evidence: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> int:
        if expected_version is not None and expected_version != self.version:
            raise StaleInstanceUpdate(
                f"Instance {self.instance_id} is version {self.version}, not {expected_version}."
            )
        if state == self.state:
            return self.version
        allowed = _TRANSITIONS.get(self.state, set())
        if state not in allowed:
            raise InvalidLifecycleTransition(f"Cannot transition {self.state.value} -> {state.value}.")
        if state == InstanceLifecycle.UNLOADING and self.active_request_count:
            raise InstanceConflict(
                f"Instance {self.instance_id} has {self.active_request_count} active request lease(s)."
            )
        self.state = state
        self.version += 1
        self.updated_at = time.time()
        if readiness_evidence is not None:
            self.readiness_evidence = dict(readiness_evidence)
        if error is not None:
            self.last_error = dict(error)
        return self.version

    def acquire_request_lease(self) -> int:
        if self.state not in {InstanceLifecycle.READY, InstanceLifecycle.BUSY}:
            raise InstanceConflict(f"Instance {self.instance_id} is not ready for requests.")
        self.active_request_count += 1
        if self.state == InstanceLifecycle.READY:
            self.state = InstanceLifecycle.BUSY
            self.version += 1
        self.updated_at = time.time()
        return self.active_request_count

    def release_request_lease(self) -> int:
        if self.active_request_count <= 0:
            raise InstanceConflict(f"Instance {self.instance_id} has no active request lease.")
        self.active_request_count -= 1
        if self.active_request_count == 0 and self.state == InstanceLifecycle.BUSY:
            self.state = InstanceLifecycle.READY
            self.version += 1
        self.updated_at = time.time()
        return self.active_request_count

    def heartbeat(self, ttl_seconds: float) -> None:
        now = time.time()
        self.heartbeat_at = now
        self.deadline = now + ttl_seconds
        self.updated_at = now

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload


class ResourceLedger:
    """Per-node reservations and ports used for admission before process launch."""

    def __init__(self, total_memory_bytes: int, *, minimum_headroom_ratio: float = 0.1) -> None:
        if total_memory_bytes < 0:
            raise ValueError("total_memory_bytes must be non-negative")
        self.total_memory_bytes = total_memory_bytes
        self.minimum_headroom_ratio = min(max(minimum_headroom_ratio, 0.0), 0.9)
        self._reservations: dict[str, int] = {}
        self._ports: dict[int, str] = {}
        self._lock = threading.RLock()

    @property
    def reserved_memory_bytes(self) -> int:
        with self._lock:
            return sum(self._reservations.values())

    @property
    def available_memory_bytes(self) -> int:
        usable = int(self.total_memory_bytes * (1.0 - self.minimum_headroom_ratio))
        return max(0, usable - self.reserved_memory_bytes)

    def reserve(self, instance_id: str, memory_bytes: int, ports: list[int]) -> None:
        if memory_bytes < 0:
            raise ResourceAdmissionError("Memory reservation cannot be negative.")
        with self._lock:
            current = self._reservations.get(instance_id, 0)
            additional = max(0, memory_bytes - current)
            if additional > self.available_memory_bytes:
                raise ResourceAdmissionError(
                    f"Instance {instance_id} needs {memory_bytes} bytes but only "
                    f"{self.available_memory_bytes + current} bytes are admissible."
                )
            conflicts = [port for port in ports if port in self._ports and self._ports[port] != instance_id]
            if conflicts:
                raise ResourceAdmissionError(f"Ports already reserved: {sorted(conflicts)}")
            self._reservations[instance_id] = memory_bytes
            for port in ports:
                self._ports[port] = instance_id

    def release(self, instance_id: str) -> None:
        with self._lock:
            self._reservations.pop(instance_id, None)
            for port, owner in list(self._ports.items()):
                if owner == instance_id:
                    del self._ports[port]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_memory_bytes": self.total_memory_bytes,
                "reserved_memory_bytes": sum(self._reservations.values()),
                "available_memory_bytes": self.available_memory_bytes,
                "minimum_headroom_ratio": self.minimum_headroom_ratio,
                "reservations": dict(self._reservations),
                "ports": {str(port): owner for port, owner in sorted(self._ports.items())},
            }


class InstanceRegistry:
    def __init__(self) -> None:
        self._instances: dict[str, ModelInstance] = {}
        self._lock = threading.RLock()

    def create(self, **values: Any) -> tuple[ModelInstance, bool]:
        with self._lock:
            instance_id = str(values.get("instance_id") or uuid.uuid4())
            operation_id = str(values.get("operation_id") or uuid.uuid4())
            existing = self._instances.get(instance_id)
            if existing is not None:
                if existing.operation_id == operation_id:
                    return existing, False
                raise InstanceConflict(
                    f"Instance {instance_id} already belongs to operation {existing.operation_id}."
                )
            values["instance_id"] = instance_id
            values["operation_id"] = operation_id
            instance = ModelInstance(**values)
            self._instances[instance_id] = instance
            return instance, True

    def get(self, instance_id: str) -> ModelInstance | None:
        with self._lock:
            return self._instances.get(instance_id)

    def remove(self, instance_id: str) -> ModelInstance | None:
        with self._lock:
            return self._instances.pop(instance_id, None)

    def snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._instances[key].to_dict() for key in sorted(self._instances)]

    def acquire_request_lease(self, instance_id: str) -> ModelInstance:
        with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                raise InstanceConflict(f"Unknown model instance: {instance_id}")
            instance.acquire_request_lease()
            return instance

    def release_request_lease(self, instance_id: str) -> ModelInstance:
        with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                raise InstanceConflict(f"Unknown model instance: {instance_id}")
            instance.release_request_lease()
            return instance


class InstanceRouter:
    """Deterministic round-robin routing over independently ready instances."""

    def __init__(self, registry: InstanceRegistry) -> None:
        self.registry = registry
        self._cursor: dict[str, int] = {}
        self._lock = threading.RLock()

    def resolve(self, model_or_instance_id: str) -> ModelInstance:
        key = model_or_instance_id.strip()
        if not key:
            raise InstanceConflict("A model alias or instance_id is required.")
        candidates: list[ModelInstance] = []
        for snapshot in self.registry.snapshots():
            instance_id = str(snapshot["instance_id"])
            instance = self.registry.get(instance_id)
            if instance is None:
                continue
            if instance.state not in {InstanceLifecycle.READY, InstanceLifecycle.BUSY}:
                continue
            if instance.instance_id == key:
                return instance
            if instance.requested_model_id == key:
                candidates.append(instance)
        if not candidates:
            raise InstanceConflict(f"No ready model instance matches {key}.")
        candidates.sort(key=lambda item: item.instance_id)
        with self._lock:
            index = self._cursor.get(key, 0) % len(candidates)
            self._cursor[key] = (index + 1) % len(candidates)
        return candidates[index]
