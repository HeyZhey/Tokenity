from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


DRAFT_ONLY_MODEL_TYPES = frozenset({"qwen3_5_mtp"})
QWEN35_MTP_LOAD_BLOCK_REASON = (
    "Qwen3.5 MTP weights are a speculative-decoding draft model and cannot be "
    "loaded as a standalone chat model. Load the matching Qwen3.5 base model instead."
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


def model_usage_metadata(config: Mapping[str, Any]) -> dict[str, object]:
    model_type = str(config.get("model_type", "")).strip() or None
    issue = standalone_model_issue("", config=config)
    return {
        "model_type": model_type,
        "standalone_loadable": issue is None,
        "load_block_reason": issue,
    }
