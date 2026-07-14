# SPDX-License-Identifier: Apache-2.0
"""Static Qwen Native MTP checkpoint inspection and load decisions.

Source basis: ml-explore/mlx-lm PR #990 and oMLX's Apache-2.0
``omlx/utils/model_loading.py``.  Modified by Tokenity to inspect complete
parameter groups, avoid tensor materialization, distinguish missing from
incomplete checkpoints, and produce a deterministic cross-rank fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import NativeMTPConfig


NATIVE_MTP_PATCH_ABI = "tokenity-native-mtp-qwen-singleton-depth1-v1"

REASON_DISABLED_BY_USER = "disabled_by_user"
REASON_UNSUPPORTED_BACKEND = "unsupported_backend"
REASON_UNSUPPORTED_MODEL_TYPE = "unsupported_model_type"
REASON_MODEL_DOES_NOT_DECLARE = "model_does_not_declare_mtp"
REASON_MISSING_WEIGHTS = "checkpoint_missing_mtp_weights"
REASON_INCOMPLETE_WEIGHTS = "checkpoint_incomplete_mtp_weights"
REASON_UNSUPPORTED_DECODE_CONCURRENCY = "unsupported_decode_concurrency"
REASON_UNSUPPORTED_TOPOLOGY = "unsupported_topology"
REASON_UNSUPPORTED_RUNTIME = "unsupported_mlx_lm_runtime_shape"
REASON_RANK_MISMATCH = "rank_decision_mismatch"
REASON_PATCH_INSTALL_FAILED = "patch_install_failed"
REASON_LOADED_INSTANCE_INVALID = "loaded_instance_invalid"
REASON_SEEDED_SEQUENTIAL_PATH = "seeded_sequential_path"


_MESSAGES = {
    REASON_DISABLED_BY_USER: "Native MTP is disabled by the launch configuration.",
    REASON_UNSUPPORTED_BACKEND: "Native MTP is not supported by this inference backend.",
    REASON_UNSUPPORTED_MODEL_TYPE: "The model is not a supported Qwen3.5/Qwen3.6 text model.",
    REASON_MODEL_DOES_NOT_DECLARE: "The model configuration does not declare an MTP head.",
    REASON_MISSING_WEIGHTS: "The checkpoint declares MTP but contains no MTP tensors.",
    REASON_INCOMPLETE_WEIGHTS: "The checkpoint contains only part of the required MTP tensor set.",
    REASON_UNSUPPORTED_DECODE_CONCURRENCY: "Native MTP requires decode_concurrency=1.",
    REASON_UNSUPPORTED_TOPOLOGY: "Native MTP requires tensor parallelism without pipeline parallelism.",
    REASON_UNSUPPORTED_RUNTIME: "The installed MLX-LM runtime does not match the required internal API shape.",
    REASON_RANK_MISMATCH: "Native MTP preflight decisions differ across ranks.",
    REASON_PATCH_INSTALL_FAILED: "Native MTP runtime patch installation failed.",
    REASON_LOADED_INSTANCE_INVALID: "The loaded model does not expose a valid active MTP head.",
    REASON_SEEDED_SEQUENTIAL_PATH: "The request used the seeded sequential decoding path.",
}


@dataclass(frozen=True)
class NativeMTPCapability:
    status: str
    model_type: str | None
    declared_layers: int
    weights_present: bool
    reason: str | None
    message: str | None
    tensor_format: str
    tensor_key_digest: str
    missing_groups: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "model_type": self.model_type,
            "declared_layers": self.declared_layers,
            "weights_present": self.weights_present,
            "reason": self.reason,
            "message": self.message,
            "tensor_format": self.tensor_format,
            "tensor_key_digest": self.tensor_key_digest,
            "missing_groups": list(self.missing_groups),
        }


@dataclass(frozen=True)
class NativeMTPDecision:
    requested_mode: str
    enabled: bool
    status: str
    reason: str | None
    message: str | None
    model_type: str | None
    declared_layers: int
    max_depth: int
    head_placement: str
    weights_present: bool
    tensor_format: str
    tensor_key_digest: str
    config_fingerprint: str
    index_fingerprint: str
    patch_abi: str = NATIVE_MTP_PATCH_ABI
    missing_groups: tuple[str, ...] = ()

    @property
    def required_failure(self) -> bool:
        return self.requested_mode == "required" and not self.enabled

    @property
    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "reason": self.reason,
            "model_type": self.model_type,
            "declared_layers": self.declared_layers,
            "max_depth": self.max_depth,
            "head_placement": self.head_placement,
            "weights_present": self.weights_present,
            "tensor_format": self.tensor_format,
            "tensor_key_digest": self.tensor_key_digest,
            "config_fingerprint": self.config_fingerprint,
            "index_fingerprint": self.index_fingerprint,
            "patch_abi": self.patch_abi,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def with_fallback(self, reason: str, *, message: str | None = None) -> "NativeMTPDecision":
        return NativeMTPDecision(
            requested_mode=self.requested_mode,
            enabled=False,
            status="node_mismatch" if reason == REASON_RANK_MISMATCH else self.status,
            reason=reason,
            message=message or reason_message(reason),
            model_type=self.model_type,
            declared_layers=self.declared_layers,
            max_depth=self.max_depth,
            head_placement=self.head_placement,
            weights_present=self.weights_present,
            tensor_format=self.tensor_format,
            tensor_key_digest=self.tensor_key_digest,
            config_fingerprint=self.config_fingerprint,
            index_fingerprint=self.index_fingerprint,
            patch_abi=self.patch_abi,
            missing_groups=self.missing_groups,
        )

    def to_readiness_dict(self) -> dict[str, object]:
        return {
            "requested_mode": self.requested_mode,
            "enabled": self.enabled,
            "model_type": self.model_type,
            "declared_layers": self.declared_layers,
            "max_depth": self.max_depth,
            "head_placement": self.head_placement,
            "weights_present": self.weights_present,
            "status": self.status,
            "fallback_reason": self.reason if not self.enabled else None,
            "message": self.message,
            "decision_fingerprint": self.fingerprint,
            "patch_abi": self.patch_abi,
        }


def reason_message(reason: str | None) -> str | None:
    return _MESSAGES.get(reason) if reason else None


def inspect_native_mtp(
    model_path: str | Path,
    config: NativeMTPConfig | None = None,
) -> NativeMTPDecision:
    """Inspect config and safetensors metadata without loading tensor data."""

    options = config or NativeMTPConfig()
    root = Path(model_path)
    config_path = root / "config.json"
    raw_config = _read_bytes(config_path)
    parsed = _decode_object(raw_config)
    model_config = _text_config(parsed)
    model_type = _string(model_config.get("model_type")) or _string(parsed.get("model_type"))
    declared_layers = _declared_layers(parsed, model_config)

    tensor_keys, tensor_format, index_bytes = _checkpoint_tensor_keys(root)
    tensor_digest = _digest_lines(tensor_keys)
    config_digest = hashlib.sha256(raw_config).hexdigest()
    index_digest = hashlib.sha256(index_bytes).hexdigest()

    capability = _capability(
        model_type=model_type,
        declared_layers=declared_layers,
        model_config=model_config,
        tensor_keys=tensor_keys,
        tensor_format=tensor_format,
        tensor_digest=tensor_digest,
    )

    if options.mode == "off":
        enabled = False
        reason = REASON_DISABLED_BY_USER
    elif capability.status == "supported":
        enabled = True
        reason = None
    else:
        enabled = False
        reason = capability.reason

    return NativeMTPDecision(
        requested_mode=options.mode,
        enabled=enabled,
        status=capability.status,
        reason=reason,
        message=reason_message(reason),
        model_type=model_type,
        declared_layers=declared_layers,
        max_depth=options.max_depth,
        head_placement=options.head_placement,
        weights_present=capability.weights_present,
        tensor_format=tensor_format,
        tensor_key_digest=tensor_digest,
        config_fingerprint=config_digest,
        index_fingerprint=index_digest,
        missing_groups=capability.missing_groups,
    )


def scan_native_mtp_capability(model_path: str | Path) -> NativeMTPCapability:
    """Return the static model-library capability independent of rollout mode."""

    decision = inspect_native_mtp(
        model_path,
        NativeMTPConfig(mode="auto"),
    )
    return NativeMTPCapability(
        status=decision.status,
        model_type=decision.model_type,
        declared_layers=decision.declared_layers,
        weights_present=decision.weights_present,
        reason=decision.reason,
        message=decision.message,
        tensor_format=decision.tensor_format,
        tensor_key_digest=decision.tensor_key_digest,
        missing_groups=decision.missing_groups,
    )


def _capability(
    *,
    model_type: str | None,
    declared_layers: int,
    model_config: Mapping[str, Any],
    tensor_keys: tuple[str, ...],
    tensor_format: str,
    tensor_digest: str,
) -> NativeMTPCapability:
    if not model_type:
        return _cap("unknown", model_type, declared_layers, False, None, tensor_format, tensor_digest)
    if not model_type.startswith(("qwen3_5", "qwen3_6")):
        return _cap(
            "unsupported",
            model_type,
            declared_layers,
            False,
            REASON_UNSUPPORTED_MODEL_TYPE,
            tensor_format,
            tensor_digest,
        )
    if declared_layers <= 0:
        return _cap(
            "unsupported",
            model_type,
            declared_layers,
            False,
            REASON_MODEL_DOES_NOT_DECLARE,
            tensor_format,
            tensor_digest,
        )

    mtp_keys = tuple(key for key in tensor_keys if _mtp_relative_key(key) is not None)
    if not mtp_keys:
        status = "unknown" if tensor_format == "unavailable" else "missing_weights"
        return _cap(
            status,
            model_type,
            declared_layers,
            False,
            REASON_MISSING_WEIGHTS,
            tensor_format,
            tensor_digest,
        )

    missing = _missing_required_groups(
        mtp_keys,
        declared_layers=declared_layers,
        num_experts=_int(model_config.get("num_experts")),
    )
    if missing:
        return _cap(
            "incomplete_weights",
            model_type,
            declared_layers,
            False,
            REASON_INCOMPLETE_WEIGHTS,
            tensor_format,
            tensor_digest,
            missing,
        )
    return _cap(
        "supported",
        model_type,
        declared_layers,
        True,
        None,
        tensor_format,
        tensor_digest,
    )


def _cap(
    status: str,
    model_type: str | None,
    declared_layers: int,
    weights_present: bool,
    reason: str | None,
    tensor_format: str,
    tensor_digest: str,
    missing: tuple[str, ...] = (),
) -> NativeMTPCapability:
    return NativeMTPCapability(
        status=status,
        model_type=model_type,
        declared_layers=declared_layers,
        weights_present=weights_present,
        reason=reason,
        message=reason_message(reason),
        tensor_format=tensor_format,
        tensor_key_digest=tensor_digest,
        missing_groups=missing,
    )


def _missing_required_groups(
    keys: Iterable[str],
    *,
    declared_layers: int,
    num_experts: int,
) -> tuple[str, ...]:
    relative = {_mtp_relative_key(key) for key in keys}
    relative.discard(None)
    names = {str(key) for key in relative}
    missing: list[str] = []

    for suffix in (
        "pre_fc_norm_hidden.weight",
        "pre_fc_norm_embedding.weight",
        "fc.weight",
        "norm.weight",
    ):
        if suffix not in names:
            missing.append(f"mtp.{suffix}")

    for layer in range(declared_layers):
        prefix = f"layers.{layer}."
        for suffix in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        ):
            if prefix + suffix not in names:
                missing.append(f"mtp.{prefix}{suffix}")
        if not _has_complete_mlp(names, prefix, num_experts):
            missing.append(f"mtp.{prefix}mlp.<complete-dense-or-moe-group>")

    return tuple(sorted(missing))


def _has_complete_mlp(names: set[str], layer_prefix: str, num_experts: int) -> bool:
    prefix = layer_prefix + "mlp."
    dense = all(prefix + item in names for item in (
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    ))
    if dense:
        return True

    shared = all(prefix + item in names for item in (
        "gate.weight",
        "shared_expert.gate_proj.weight",
        "shared_expert.up_proj.weight",
        "shared_expert.down_proj.weight",
        "shared_expert_gate.weight",
    ))
    if not shared:
        return False

    switch = all(prefix + item in names for item in (
        "switch_mlp.gate_proj.weight",
        "switch_mlp.up_proj.weight",
        "switch_mlp.down_proj.weight",
    ))
    fused = all(prefix + item in names for item in (
        "experts.gate_up_proj",
        "experts.down_proj",
    ))
    if switch or fused:
        return True

    if num_experts <= 0:
        return False
    return all(
        prefix + f"experts.{expert}.{projection}.weight" in names
        for expert in range(num_experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
    )


def _mtp_relative_key(key: str) -> str | None:
    segments = key.split(".")
    try:
        index = segments.index("mtp")
    except ValueError:
        return None
    if index + 1 >= len(segments):
        return None
    return ".".join(segments[index + 1 :])


def _checkpoint_tensor_keys(root: Path) -> tuple[tuple[str, ...], str, bytes]:
    index_path = root / "model.safetensors.index.json"
    raw_index = _read_bytes(index_path)
    if raw_index:
        index = _decode_object(raw_index)
        weight_map = index.get("weight_map")
        if isinstance(weight_map, dict):
            return tuple(sorted(str(key) for key in weight_map)), "safetensors-index", raw_index

    keys: set[str] = set()
    fingerprints: list[bytes] = []
    for safetensor in sorted(root.glob("*.safetensors")):
        header_bytes = _read_safetensors_header(safetensor)
        if header_bytes is None:
            continue
        fingerprints.append(safetensor.name.encode("utf-8") + b"\0" + header_bytes)
        header = _decode_object(header_bytes)
        keys.update(str(key) for key in header if key != "__metadata__")
    if fingerprints:
        return tuple(sorted(keys)), "safetensors-header", b"\n".join(fingerprints)
    return (), "unavailable", b""


def _read_safetensors_header(path: Path) -> bytes | None:
    try:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return None
            length = struct.unpack("<Q", raw_length)[0]
            if length <= 0 or length > 256 * 1024 * 1024:
                return None
            header = handle.read(length)
            return header if len(header) == length else None
    except OSError:
        return None


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def _declared_layers(config: Mapping[str, Any], text_config: Mapping[str, Any]) -> int:
    return max(
        _int(config.get("mtp_num_hidden_layers")),
        _int(text_config.get("mtp_num_hidden_layers")),
    )


def _decode_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def _digest_lines(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
