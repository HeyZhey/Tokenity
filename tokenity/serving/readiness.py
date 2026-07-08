from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ReadinessPhase(StrEnum):
    LAUNCHING = "launching"
    DISTRIBUTED_INIT = "distributed_init"
    LOADING_MODEL = "loading_model"
    COMPILING = "compiling"
    READY = "ready"
    PREFILL_PENDING = "prefill_pending"
    GENERATING = "generating"
    FAILED = "failed"
    STOPPING = "stopping"


@dataclass(slots=True)
class ReadinessState:
    phase: ReadinessPhase = ReadinessPhase.LAUNCHING
    rank: int = 0
    world_size: int = 1
    backend: str = "single"
    model: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase.value,
            "rank": self.rank,
            "world_size": self.world_size,
            "backend": self.backend,
            "model": self.model,
            "message": self.message,
        }

