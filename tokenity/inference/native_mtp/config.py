# SPDX-License-Identifier: Apache-2.0
"""Validated server-level configuration for Tokenity Native MTP.

Adapted for Tokenity from the configuration boundary described by
ml-explore/mlx-lm PR #990.  Tokenity modifications: explicit rollout modes,
fixed depth/placement, and a default-off compatibility contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


NativeMTPMode = Literal["off", "auto", "required"]


@dataclass(frozen=True)
class NativeMTPConfig:
    """Immutable Native MTP launch configuration.

    The MVP deliberately has no hidden clamping.  A request outside the
    supported boundary is rejected before any runtime patch or model
    construction can occur.
    """

    mode: NativeMTPMode = "off"
    max_depth: int = 1
    head_placement: Literal["replicated"] = "replicated"

    def __post_init__(self) -> None:
        if self.mode not in {"off", "auto", "required"}:
            raise ValueError("native_mtp.mode must be off, auto, or required")
        if self.max_depth != 1:
            raise ValueError("native_mtp.max_depth must be 1 for the MVP")
        if self.head_placement != "replicated":
            raise ValueError("native_mtp.head_placement must be replicated for the MVP")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "NativeMTPConfig":
        if value is None:
            return cls()
        return cls(
            mode=str(value.get("mode", "off")),  # type: ignore[arg-type]
            max_depth=int(value.get("max_depth", 1)),
            head_placement=str(value.get("head_placement", "replicated")),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "max_depth": self.max_depth,
            "head_placement": self.head_placement,
        }

    @classmethod
    def forced_off(cls) -> "NativeMTPConfig":
        """Configuration used by unsupported product backends."""

        return cls(mode="off", max_depth=1, head_placement="replicated")

