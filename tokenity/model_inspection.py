from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


DRAFT_ONLY_MODEL_TYPES = frozenset({"qwen3_5_mtp"})
DISTRIBUTED_QUARANTINED_MODEL_TYPES = frozenset({"qwen3_moe"})
QWEN35_MTP_LOAD_BLOCK_REASON = (
    "Qwen3.5 MTP weights are a speculative-decoding draft model and cannot be "
    "loaded as a standalone chat model. Load the matching Qwen3.5 base model instead."
)
QWEN3_MOE_DISTRIBUTED_LOAD_BLOCK_REASON = (
    "The current Tokenity runtime does not support this model in two-Mac mode. "
    "Its Qwen3 MoE pipeline path is quarantined after repeated-request watchdog "
    "resets on MLX 0.32.0 / mlx-lm 0.31.3. Choose Load on one Mac instead."
)


def read_model_config(model: str | Path) -> dict[str, Any]:
    path = Path(model)
    config_path = path if path.name == "config.json" else path / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def standalone_model_issue(
    model: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> str | None:
    payload = dict(config) if config is not None else read_model_config(model)
    model_type = str(payload.get("model_type", "")).strip().lower()
    if model_type in DRAFT_ONLY_MODEL_TYPES:
        return QWEN35_MTP_LOAD_BLOCK_REASON
    return None


def distributed_model_issue(
    model: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> str | None:
    """Return a capability-derived multi-node block reason, if any.

    Model names are deliberately not considered. The pinned runtime is only
    allowed to start topologies that have been requalified for the exact model
    architecture advertised by config.json.
    """

    payload = dict(config) if config is not None else read_model_config(model)
    model_type = str(payload.get("model_type", "")).strip().lower()
    if model_type in DISTRIBUTED_QUARANTINED_MODEL_TYPES:
        return QWEN3_MOE_DISTRIBUTED_LOAD_BLOCK_REASON
    return None


def model_usage_metadata(config: Mapping[str, Any]) -> dict[str, object]:
    model_type = str(config.get("model_type", "")).strip() or None
    issue = standalone_model_issue("", config=config)
    distributed_issue = distributed_model_issue("", config=config)
    return {
        "model_type": model_type,
        "standalone_loadable": issue is None,
        "load_block_reason": issue,
        "distributed_loadable": distributed_issue is None,
        "distributed_load_block_reason": distributed_issue,
    }
