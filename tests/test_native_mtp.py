from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tokenity.inference.native_mtp.config import NativeMTPConfig
from tokenity.inference.native_mtp.detector import (
    REASON_INCOMPLETE_WEIGHTS,
    REASON_MISSING_WEIGHTS,
    REASON_MODEL_DOES_NOT_DECLARE,
    REASON_UNSUPPORTED_DECODE_CONCURRENCY,
    REASON_UNSUPPORTED_MODEL_TYPE,
    inspect_native_mtp,
    scan_native_mtp_capability,
)
from tokenity.inference.native_mtp.runtime import (
    NativeMTPRuntimeController,
    NativeMTPStartupError,
    PatchTransaction,
    native_mtp_construction_scope,
    is_native_mtp_construction_active,
)
from tokenity.node_agent.agent import (
    NativeMTPRequest,
    RankStartRequest,
    StartRequest,
    _rank_command,
    create_app,
)


def _write_model(
    root: Path,
    *,
    model_type: str = "qwen3_5_moe_text",
    declared: int = 1,
    keys: list[str] | None = None,
) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "text_config": {
                    "model_type": model_type,
                    "mtp_num_hidden_layers": declared,
                    "num_experts": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors" for key in keys or []}}),
        encoding="utf-8",
    )
    return root


def _complete_dense_keys() -> list[str]:
    prefix = "model.language_model.mtp."
    keys = [
        prefix + "pre_fc_norm_hidden.weight",
        prefix + "pre_fc_norm_embedding.weight",
        prefix + "fc.weight",
        prefix + "norm.weight",
    ]
    layer = prefix + "layers.0."
    keys.extend(
        layer + suffix
        for suffix in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        )
    )
    return keys


def test_native_mtp_config_is_strict_and_default_off():
    assert NativeMTPConfig().to_dict() == {
        "mode": "off",
        "max_depth": 1,
        "head_placement": "replicated",
    }
    with pytest.raises(ValueError, match="max_depth"):
        NativeMTPConfig(max_depth=2)
    with pytest.raises(ValueError, match="head_placement"):
        NativeMTPConfig(head_placement="sharded")  # type: ignore[arg-type]


def test_static_detector_distinguishes_capability_states(tmp_path: Path):
    unsupported = _write_model(tmp_path / "unsupported", model_type="llama", declared=1)
    undeclared = _write_model(tmp_path / "undeclared", declared=0)
    missing = _write_model(tmp_path / "missing", keys=["model.embed_tokens.weight"])
    incomplete = _write_model(tmp_path / "incomplete", keys=_complete_dense_keys()[:-1])
    supported = _write_model(tmp_path / "supported", keys=_complete_dense_keys())

    assert scan_native_mtp_capability(unsupported).reason == REASON_UNSUPPORTED_MODEL_TYPE
    assert scan_native_mtp_capability(undeclared).reason == REASON_MODEL_DOES_NOT_DECLARE
    assert scan_native_mtp_capability(missing).reason == REASON_MISSING_WEIGHTS
    assert scan_native_mtp_capability(incomplete).reason == REASON_INCOMPLETE_WEIGHTS
    assert scan_native_mtp_capability(incomplete).status == "incomplete_weights"
    assert scan_native_mtp_capability(supported).status == "supported"
    assert scan_native_mtp_capability(supported).weights_present is True


def test_detector_reads_safetensors_header_without_tensor_payload(tmp_path: Path):
    root = tmp_path / "header"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "mtp_num_hidden_layers": 1}),
        encoding="utf-8",
    )
    header = json.dumps(
        {
            **{
                key: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
                for key in _complete_dense_keys()
            },
            "__metadata__": {"format": "pt"},
        }
    ).encode("utf-8")
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"xxxx")

    capability = scan_native_mtp_capability(root)
    assert capability.status == "supported"
    assert capability.tensor_format == "safetensors-header"


def test_mode_semantics_and_fingerprint_are_deterministic(tmp_path: Path):
    model = _write_model(tmp_path / "model", keys=_complete_dense_keys())
    first = inspect_native_mtp(model, NativeMTPConfig(mode="auto"))
    second = inspect_native_mtp(model, NativeMTPConfig(mode="auto"))
    off = inspect_native_mtp(model, NativeMTPConfig(mode="off"))
    assert first.enabled is True
    assert first.fingerprint == second.fingerprint
    assert off.enabled is False
    assert off.reason == "disabled_by_user"
    assert off.fingerprint != first.fingerprint


def test_required_missing_weights_fails_before_runtime_import(tmp_path: Path):
    model = _write_model(tmp_path / "model", keys=["model.weight"])
    controller = NativeMTPRuntimeController(
        model=str(model),
        config=NativeMTPConfig(mode="required"),
        decode_concurrency=1,
    )
    with pytest.raises(NativeMTPStartupError, match=REASON_MISSING_WEIGHTS):
        controller.prepare(_WorldOneMX(), _WorldOneGroup())


def test_auto_decode_concurrency_falls_back_with_structured_reason(tmp_path: Path):
    model = _write_model(tmp_path / "model", keys=_complete_dense_keys())
    controller = NativeMTPRuntimeController(
        model=str(model),
        config=NativeMTPConfig(mode="auto"),
        decode_concurrency=2,
    )
    decision = controller.prepare(_WorldOneMX(), _WorldOneGroup())
    assert decision.enabled is False
    assert decision.reason == REASON_UNSUPPORTED_DECODE_CONCURRENCY
    assert controller.telemetry["effective_mode"] == "standard"


def test_patch_transaction_and_construction_scope_restore_prior_state():
    class Target:
        existing = "before"

    transaction = PatchTransaction()
    transaction.set(Target, "existing", "after")
    transaction.set(Target, "new", 1)
    assert (Target.existing, Target.new) == ("after", 1)
    transaction.rollback()
    assert Target.existing == "before"
    assert not hasattr(Target, "new")

    assert is_native_mtp_construction_active() is False
    with native_mtp_construction_scope(True):
        assert is_native_mtp_construction_active() is True
        with native_mtp_construction_scope(False):
            assert is_native_mtp_construction_active() is False
        assert is_native_mtp_construction_active() is True
    assert is_native_mtp_construction_active() is False


def test_node_models_and_rank_propagation_expose_nested_native_mtp(tmp_path: Path):
    _write_model(tmp_path / "qwen", keys=["model.weight"])
    client = TestClient(create_app())
    model_payload = client.get("/v1/node/models", params={"root": str(tmp_path)}).json()
    assert model_payload["models"][0]["native_mtp"]["status"] == "missing_weights"

    request = StartRequest(
        model="/models/qwen",
        native_mtp=NativeMTPRequest(mode="required"),
    )
    rank = RankStartRequest(
        cluster_id="abcdefgh",
        model=request.model,
        rank=0,
        world_size=1,
        python="/runtime/python",
        native_mtp=request.native_mtp,
    )
    command = _rank_command(rank)
    assert command[command.index("--native-mtp-mode") + 1] == "required"
    assert command[command.index("--native-mtp-max-depth") + 1] == "1"
    assert command[command.index("--native-mtp-head-placement") + 1] == "replicated"

    with pytest.raises(ValidationError):
        StartRequest(model="/models/qwen", native_mtp={"mode": "auto", "max_depth": 2})


class _WorldOneGroup:
    def size(self) -> int:
        return 1


class _WorldOneMX:
    pass
