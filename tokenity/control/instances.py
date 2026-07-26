from __future__ import annotations

import math
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field, replace
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


class GenerationQueueFull(InstanceConflict):
    pass


class GenerationSlotTimeout(InstanceConflict):
    pass


class GenerationSlotCancelled(InstanceConflict):
    pass


@dataclass(frozen=True)
class MemoryReservationBreakdown:
    """Local-node memory inputs used to explain and correct a reservation.

    The legacy reservation remains authoritative when this optional structure is
    absent. Observed peak/footprint values replace uncertain runtime overhead,
    but never the deterministic weights/cache/headroom floor.
    """

    weights_bytes: int = 0
    kv_cache_bytes: int = 0
    prompt_cache_bytes: int = 0
    mlx_cache_bytes: int = 0
    runtime_peak_bytes: int = 0
    os_headroom_bytes: int = 0
    legacy_unattributed_bytes: int = 0
    observed_peak_bytes: int | None = None
    observed_footprint_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "weights_bytes",
            "kv_cache_bytes",
            "prompt_cache_bytes",
            "mlx_cache_bytes",
            "runtime_peak_bytes",
            "os_headroom_bytes",
            "legacy_unattributed_bytes",
            "observed_peak_bytes",
            "observed_footprint_bytes",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "MemoryReservationBreakdown":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})

    @property
    def estimated_total_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.kv_cache_bytes
            + self.prompt_cache_bytes
            + self.mlx_cache_bytes
            + self.runtime_peak_bytes
            + self.os_headroom_bytes
            + self.legacy_unattributed_bytes
        )

    @property
    def protected_floor_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.kv_cache_bytes
            + self.prompt_cache_bytes
            + self.mlx_cache_bytes
            + self.os_headroom_bytes
        )

    @property
    def observed_local_bytes(self) -> int | None:
        observations = [
            value
            for value in (self.observed_peak_bytes, self.observed_footprint_bytes)
            if value is not None
        ]
        return max(observations) if observations else None

    def with_observation(
        self,
        *,
        peak_bytes: int | None = None,
        footprint_bytes: int | None = None,
    ) -> "MemoryReservationBreakdown":
        return replace(
            self,
            observed_peak_bytes=peak_bytes,
            observed_footprint_bytes=footprint_bytes,
        )

    def recommended_reservation_bytes(self, *, safety_margin_ratio: float = 0.1) -> int:
        if safety_margin_ratio < 0:
            raise ValueError("safety_margin_ratio must be non-negative")
        observed = self.observed_local_bytes
        if observed is None:
            return self.estimated_total_bytes
        observed_with_margin = math.ceil(observed * (1.0 + safety_margin_ratio))
        return max(self.protected_floor_bytes, observed_with_margin)


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
    queued_request_count: int = 0
    memory_reservation_breakdown: MemoryReservationBreakdown | dict[str, Any] | None = None
    health_ready: bool | None = None
    health_sampled_at: float | None = None
    health_issues: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.memory_reservation_bytes < 0:
            raise ValueError("memory_reservation_bytes must be non-negative")
        if isinstance(self.memory_reservation_breakdown, dict):
            self.memory_reservation_breakdown = MemoryReservationBreakdown.from_mapping(
                self.memory_reservation_breakdown
            )

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
        if state == InstanceLifecycle.UNLOADING and (
            self.active_request_count or self.queued_request_count
        ):
            raise InstanceConflict(
                f"Instance {self.instance_id} has {self.active_request_count} active request "
                f"lease(s) and {self.queued_request_count} queued request(s)."
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
        if (
            self.active_request_count == 0
            and self.queued_request_count == 0
            and self.state == InstanceLifecycle.BUSY
        ):
            self.state = InstanceLifecycle.READY
            self.version += 1
        self.updated_at = time.time()
        return self.active_request_count

    def enqueue_request(self) -> int:
        if self.state not in {InstanceLifecycle.READY, InstanceLifecycle.BUSY}:
            raise InstanceConflict(f"Instance {self.instance_id} is not ready for requests.")
        self.queued_request_count += 1
        if self.state == InstanceLifecycle.READY:
            self.state = InstanceLifecycle.BUSY
            self.version += 1
        self.updated_at = time.time()
        return self.queued_request_count

    def dequeue_request(self) -> int:
        if self.queued_request_count <= 0:
            raise InstanceConflict(f"Instance {self.instance_id} has no queued request.")
        self.queued_request_count -= 1
        if (
            self.queued_request_count == 0
            and self.active_request_count == 0
            and self.state == InstanceLifecycle.BUSY
        ):
            self.state = InstanceLifecycle.READY
            self.version += 1
        self.updated_at = time.time()
        return self.queued_request_count

    def memory_breakdown(self) -> MemoryReservationBreakdown:
        if isinstance(self.memory_reservation_breakdown, MemoryReservationBreakdown):
            return self.memory_reservation_breakdown
        if isinstance(self.memory_reservation_breakdown, dict):
            return MemoryReservationBreakdown.from_mapping(self.memory_reservation_breakdown)
        return MemoryReservationBreakdown(
            legacy_unattributed_bytes=self.memory_reservation_bytes
        )

    def heartbeat(self, ttl_seconds: float) -> None:
        now = time.time()
        self.heartbeat_at = now
        # A slower control client must not shorten a lease that was explicitly
        # granted for a long distributed load. This is especially important
        # while different UI/background refresh loops overlap.
        self.deadline = max(self.deadline or 0.0, now + ttl_seconds)
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
        with self._lock:
            usable = int(self.total_memory_bytes * (1.0 - self.minimum_headroom_ratio))
            return max(0, usable - sum(self._reservations.values()))

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

    def resize(self, instance_id: str, memory_bytes: int) -> int:
        """Atomically replace memory for an existing reservation.

        Port ownership is deliberately untouched. A failed growth leaves both
        memory and ports at their previous values.
        """

        if memory_bytes < 0:
            raise ResourceAdmissionError("Memory reservation cannot be negative.")
        with self._lock:
            if instance_id not in self._reservations:
                raise ResourceAdmissionError(f"Unknown memory reservation: {instance_id}")
            current = self._reservations[instance_id]
            usable = int(self.total_memory_bytes * (1.0 - self.minimum_headroom_ratio))
            admissible = usable - (sum(self._reservations.values()) - current)
            if memory_bytes > admissible:
                raise ResourceAdmissionError(
                    f"Instance {instance_id} needs {memory_bytes} bytes but only "
                    f"{max(0, admissible)} bytes are admissible."
                )
            self._reservations[instance_id] = memory_bytes
            return memory_bytes

    def reconcile(
        self,
        instance_id: str,
        *,
        observed_peak_bytes: int | None = None,
        observed_footprint_bytes: int | None = None,
        safety_margin_ratio: float = 0.1,
        minimum_reservation_bytes: int = 0,
        breakdown: MemoryReservationBreakdown | None = None,
    ) -> int:
        """Resize from local observed memory while retaining a safety floor."""

        if safety_margin_ratio < 0:
            raise ValueError("safety_margin_ratio must be non-negative")
        if minimum_reservation_bytes < 0:
            raise ValueError("minimum_reservation_bytes must be non-negative")
        for name, value in (
            ("observed_peak_bytes", observed_peak_bytes),
            ("observed_footprint_bytes", observed_footprint_bytes),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

        if breakdown is not None:
            observed_breakdown = breakdown.with_observation(
                peak_bytes=(
                    observed_peak_bytes
                    if observed_peak_bytes is not None
                    else breakdown.observed_peak_bytes
                ),
                footprint_bytes=(
                    observed_footprint_bytes
                    if observed_footprint_bytes is not None
                    else breakdown.observed_footprint_bytes
                ),
            )
            target = observed_breakdown.recommended_reservation_bytes(
                safety_margin_ratio=safety_margin_ratio
            )
        else:
            observations = [
                value
                for value in (observed_peak_bytes, observed_footprint_bytes)
                if value is not None
            ]
            if not observations:
                raise ValueError("At least one local memory observation is required")
            target = math.ceil(max(observations) * (1.0 + safety_margin_ratio))
        return self.resize(instance_id, max(minimum_reservation_bytes, target))

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

    def enqueue_request(self, instance_id: str) -> ModelInstance:
        with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                raise InstanceConflict(f"Unknown model instance: {instance_id}")
            instance.enqueue_request()
            return instance

    def dequeue_request(self, instance_id: str) -> ModelInstance:
        with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                raise InstanceConflict(f"Unknown model instance: {instance_id}")
            instance.dequeue_request()
            return instance


@dataclass(frozen=True)
class _GenerationSlotRecord:
    lease_id: str
    instance_id: str
    selected_nodes: frozenset[str]
    sequence: int
    requested_at: float
    acquired_at: float | None = None


class GenerationSlotLease:
    """An idempotently releasable shared-hardware generation lease."""

    def __init__(
        self,
        scheduler: "GenerationSlotScheduler",
        record: _GenerationSlotRecord,
    ) -> None:
        self._scheduler = scheduler
        self._record = record

    @property
    def instance_id(self) -> str:
        return self._record.instance_id

    @property
    def selected_nodes(self) -> frozenset[str]:
        return self._record.selected_nodes

    @property
    def waited_seconds(self) -> float:
        acquired_at = self._record.acquired_at or self._record.requested_at
        return max(0.0, acquired_at - self._record.requested_at)

    @property
    def released(self) -> bool:
        return not self._scheduler._is_active(self._record.lease_id)

    def release(self) -> bool:
        return self._scheduler._release(self._record.lease_id)

    def __enter__(self) -> "GenerationSlotLease":
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


class GenerationSlotScheduler:
    """FIFO exclusion for heavyweight generation on overlapping node sets.

    Requests whose selected nodes are disjoint may run concurrently. Unknown or
    empty node sets are treated conservatively as overlapping every request.
    """

    _UNKNOWN_NODE = "*"

    def __init__(
        self,
        *,
        max_queue_depth: int = 32,
        registry: InstanceRegistry | None = None,
    ) -> None:
        if max_queue_depth < 0:
            raise ValueError("max_queue_depth must be non-negative")
        self.max_queue_depth = max_queue_depth
        self.registry = registry
        self._condition = threading.Condition(threading.RLock())
        self._waiting: deque[_GenerationSlotRecord] = deque()
        self._active: dict[str, _GenerationSlotRecord] = {}
        self._sequence = 0

    @property
    def active_count(self) -> int:
        with self._condition:
            return len(self._active)

    def queue_depth(self, instance_id: str | None = None) -> int:
        with self._condition:
            if instance_id is None:
                return len(self._waiting)
            return sum(1 for waiter in self._waiting if waiter.instance_id == instance_id)

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            queued_by_instance: dict[str, int] = {}
            for waiter in self._waiting:
                queued_by_instance[waiter.instance_id] = (
                    queued_by_instance.get(waiter.instance_id, 0) + 1
                )
            active_by_instance: dict[str, int] = {}
            for lease in self._active.values():
                active_by_instance[lease.instance_id] = (
                    active_by_instance.get(lease.instance_id, 0) + 1
                )
            return {
                "max_queue_depth": self.max_queue_depth,
                "queue_depth": len(self._waiting),
                "active_count": len(self._active),
                "queued_by_instance": dict(sorted(queued_by_instance.items())),
                "active_by_instance": dict(sorted(active_by_instance.items())),
            }

    def acquire(
        self,
        instance_id: str,
        selected_nodes: list[str] | tuple[str, ...] | set[str] | frozenset[str],
        *,
        timeout: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> GenerationSlotLease:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationSlotCancelled(
                f"Generation slot request for {instance_id} was cancelled."
            )
        requested_at = time.monotonic()
        deadline = requested_at + timeout if timeout is not None else None
        nodes = self._normalize_nodes(selected_nodes)
        with self._condition:
            self._sequence += 1
            record = _GenerationSlotRecord(
                lease_id=uuid.uuid4().hex,
                instance_id=instance_id,
                selected_nodes=nodes,
                sequence=self._sequence,
                requested_at=requested_at,
            )
            if self._can_grant(record):
                return self._grant(record)
            if len(self._waiting) >= self.max_queue_depth:
                raise GenerationQueueFull(
                    f"Generation slot queue is full ({self.max_queue_depth})."
                )
            self._waiting.append(record)
            try:
                self._mark_queued(instance_id, enqueue=True)
            except BaseException:
                self._waiting.remove(record)
                raise

            while True:
                if cancel_event is not None and cancel_event.is_set():
                    self._remove_waiter(record)
                    raise GenerationSlotCancelled(
                        f"Generation slot request for {instance_id} was cancelled."
                    )
                if self._can_grant(record):
                    self._remove_waiter(record)
                    return self._grant(record)
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    self._remove_waiter(record)
                    raise GenerationSlotTimeout(
                        f"Generation slot request for {instance_id} timed out."
                    )
                poll_interval = 0.05 if cancel_event is not None else None
                wait_seconds = remaining
                if poll_interval is not None:
                    wait_seconds = (
                        poll_interval
                        if remaining is None
                        else min(poll_interval, remaining)
                    )
                self._condition.wait(wait_seconds)

    def _normalize_nodes(
        self,
        selected_nodes: list[str] | tuple[str, ...] | set[str] | frozenset[str],
    ) -> frozenset[str]:
        nodes = frozenset(str(node).strip() for node in selected_nodes if str(node).strip())
        return nodes or frozenset({self._UNKNOWN_NODE})

    def _overlaps(self, first: frozenset[str], second: frozenset[str]) -> bool:
        return (
            self._UNKNOWN_NODE in first
            or self._UNKNOWN_NODE in second
            or not first.isdisjoint(second)
        )

    def _can_grant(self, record: _GenerationSlotRecord) -> bool:
        if any(
            self._overlaps(record.selected_nodes, active.selected_nodes)
            for active in self._active.values()
        ):
            return False
        for waiter in self._waiting:
            if waiter.lease_id == record.lease_id:
                break
            if self._overlaps(record.selected_nodes, waiter.selected_nodes):
                return False
        return True

    def _grant(self, record: _GenerationSlotRecord) -> GenerationSlotLease:
        granted = replace(record, acquired_at=time.monotonic())
        self._active[record.lease_id] = granted
        return GenerationSlotLease(self, granted)

    def _remove_waiter(self, record: _GenerationSlotRecord) -> None:
        try:
            self._waiting.remove(record)
        except ValueError:
            return
        self._mark_queued(record.instance_id, enqueue=False)
        self._condition.notify_all()

    def _mark_queued(self, instance_id: str, *, enqueue: bool) -> None:
        if self.registry is None:
            return
        if enqueue:
            self.registry.enqueue_request(instance_id)
        else:
            self.registry.dequeue_request(instance_id)

    def _is_active(self, lease_id: str) -> bool:
        with self._condition:
            return lease_id in self._active

    def _release(self, lease_id: str) -> bool:
        with self._condition:
            if self._active.pop(lease_id, None) is None:
                return False
            self._condition.notify_all()
            return True


class InstanceRouter:
    """Deterministic round-robin routing over independently ready instances."""

    def __init__(self, registry: InstanceRegistry) -> None:
        self.registry = registry
        self._cursor: dict[tuple[str, str | None], int] = {}
        self._lock = threading.RLock()

    def resolve(
        self,
        model_or_instance_id: str,
        *,
        revision: str | None = None,
        exclude_instance_ids: set[str] | frozenset[str] | None = None,
    ) -> ModelInstance:
        key = model_or_instance_id.strip()
        if not key:
            raise InstanceConflict("A model alias or instance_id is required.")
        excluded = exclude_instance_ids or frozenset()
        candidates: list[ModelInstance] = []
        for snapshot in self.registry.snapshots():
            instance_id = str(snapshot["instance_id"])
            instance = self.registry.get(instance_id)
            if instance is None:
                continue
            if instance.instance_id in excluded:
                continue
            if instance.state not in {InstanceLifecycle.READY, InstanceLifecycle.BUSY}:
                continue
            if instance.health_ready is False:
                continue
            instance_revision = instance.model_revision or "unversioned"
            if revision is not None and instance_revision != revision:
                continue
            if instance.instance_id == key:
                return instance
            if instance.requested_model_id == key:
                candidates.append(instance)
        if not candidates:
            revision_detail = f" at revision {revision}" if revision is not None else ""
            raise InstanceConflict(
                f"No ready model instance matches {key}{revision_detail}."
            )
        minimum_load = min(
            (
                instance.active_request_count + instance.queued_request_count,
                instance.queued_request_count,
                instance.active_request_count,
            )
            for instance in candidates
        )
        candidates = [
            instance
            for instance in candidates
            if (
                instance.active_request_count + instance.queued_request_count,
                instance.queued_request_count,
                instance.active_request_count,
            )
            == minimum_load
        ]
        candidates.sort(key=lambda item: item.instance_id)
        cursor_key = (key, revision)
        with self._lock:
            index = self._cursor.get(cursor_key, 0) % len(candidates)
            self._cursor[cursor_key] = (index + 1) % len(candidates)
        return candidates[index]
