# SPDX-License-Identifier: Apache-2.0
"""Tokenity's Qwen3.5/Qwen3.6 native MTP integration.

This package is a Tokenity-specific adaptation of ml-explore/mlx-lm PR #990
and selected parts of oMLX's Apache-2.0 runtime patch.  It intentionally
supports only the singleton, depth-one Qwen text path used by Tokenity's
distributed OpenAI backend.
"""

from .config import NativeMTPConfig
from .detector import (
    NATIVE_MTP_PATCH_ABI,
    NativeMTPCapability,
    NativeMTPDecision,
    inspect_native_mtp,
    scan_native_mtp_capability,
)

__all__ = [
    "NATIVE_MTP_PATCH_ABI",
    "NativeMTPCapability",
    "NativeMTPConfig",
    "NativeMTPDecision",
    "inspect_native_mtp",
    "scan_native_mtp_capability",
]
