from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase.value,
            "rank": self.rank,
            "world_size": self.world_size,
            "backend": self.backend,
            "model": self.model,
            "message": self.message,
            "progress": self.progress,
            "progress_current": self.progress_current,
            "progress_total": self.progress_total,
            "native_mtp": self.native_mtp,
        }
