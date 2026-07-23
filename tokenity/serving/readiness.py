from __future__ import annotations

import time
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ReadinessPhase(str, Enum):
    LAUNCHING = "launching"
    DISTRIBUTED_INIT = "distributed_init"
    LOADING_MODEL = "loading_model"
    COMPILING = "compiling"
    READY = "ready"
    PREFILL_PENDING = "prefill_pending"
    GENERATING = "generating"
    FAILED = "failed"
    STOPPING = "stopping"


_CANONICAL_LIFECYCLE = {
    ReadinessPhase.LAUNCHING: "launching",
    ReadinessPhase.DISTRIBUTED_INIT: "distributed_initializing",
    ReadinessPhase.LOADING_MODEL: "materializing_weights",
    ReadinessPhase.COMPILING: "compiling_warming",
    ReadinessPhase.READY: "ready",
    ReadinessPhase.PREFILL_PENDING: "busy",
    ReadinessPhase.GENERATING: "busy",
    ReadinessPhase.FAILED: "failed",
    ReadinessPhase.STOPPING: "unloading",
}


@dataclass
class ReadinessState:
    phase: ReadinessPhase = ReadinessPhase.LAUNCHING
    rank: int = 0
    world_size: int = 1
    backend: str = "single"
    model: str | None = None
    message: str | None = None
    progress: float | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    native_mtp: dict[str, object] | None = None
    instance_id: str | None = None
    operation_id: str | None = None
    generation: int = 0
    version: int = 0
    execution_mode: str = "single"
    cluster_id: str | None = None
    connection_mode: str | None = None
    model_revision: str | None = None
    tokenizer_identity: str | None = None
    ready_evidence: dict[str, object] = field(default_factory=dict)
    last_error: dict[str, object] | None = None
    last_request: dict[str, object] | None = None
    memory: dict[str, object] | None = None
    updated_at: float = field(default_factory=time.time)
    status_path: str | None = field(default=None, repr=False)

    @property
    def lifecycle_state(self) -> str:
        return _CANONICAL_LIFECYCLE[self.phase]

    def transition(
        self,
        phase: ReadinessPhase,
        *,
        message: str | None = None,
        expected_version: int | None = None,
        error: dict[str, object] | None = None,
        evidence: dict[str, object] | None = None,
    ) -> bool:
        """Publish a monotonic state update and reject stale writers.

        Existing callers can still assign ``phase`` directly. New lifecycle
        code should use this method so an older poll/operation cannot overwrite
        a newer instance state.
        """

        if expected_version is not None and expected_version != self.version:
            return False
        self.phase = phase
        if message is not None:
            self.message = message
        if error is not None:
            self.last_error = error
        if evidence is not None:
            self.ready_evidence = dict(evidence)
        self.version += 1
        self.updated_at = time.time()
        self.publish()
        return True

    def record_request_event(self, request_id: str, event: str, at: float | None = None) -> None:
        timestamp = at if at is not None else time.time()
        if self.last_request is None or self.last_request.get("request_id") != request_id:
            self.last_request = {"request_id": request_id}
        self.last_request[event] = timestamp
        self.updated_at = timestamp
        self.publish()

    def publish(self) -> None:
        if not self.status_path:
            return
        path = Path(self.status_path)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(self.to_dict(), sort_keys=True), encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, Any] = {
            "phase": self.phase.value,
            "lifecycle_state": self.lifecycle_state,
            "rank": self.rank,
            "world_size": self.world_size,
            "backend": self.backend,
            "model": self.model,
            "message": self.message,
            "progress": self.progress,
            "progress_current": self.progress_current,
            "progress_total": self.progress_total,
            "native_mtp": self.native_mtp,
            "instance_id": self.instance_id,
            "operation_id": self.operation_id,
            "generation": self.generation,
            "version": self.version,
            "execution_mode": self.execution_mode,
            "cluster_id": self.cluster_id,
            "connection_mode": self.connection_mode,
            "model_revision": self.model_revision,
            "tokenizer_identity": self.tokenizer_identity,
            "ready_evidence": self.ready_evidence,
            "last_error": self.last_error,
            "last_request": self.last_request,
            "memory": self.memory,
            "updated_at": self.updated_at,
        }
        return payload
