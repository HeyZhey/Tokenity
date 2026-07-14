# SPDX-License-Identifier: Apache-2.0
"""Distributed preflight and atomic activation for Tokenity Native MTP.

Adapted from the activation boundary in oMLX's Apache-2.0 MTP patch and
ml-explore/mlx-lm PR #990. Tokenity adds deterministic all-rank agreement,
an explicit mlx-lm ABI guard, rollback of every monkey-patch mutation, and a
construction-only activation scope so ordinary model loads stay unchanged.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from .config import NativeMTPConfig
from .detector import (
    REASON_LOADED_INSTANCE_INVALID,
    REASON_PATCH_INSTALL_FAILED,
    REASON_RANK_MISMATCH,
    REASON_UNSUPPORTED_DECODE_CONCURRENCY,
    REASON_UNSUPPORTED_RUNTIME,
    REASON_UNSUPPORTED_TOPOLOGY,
    NativeMTPDecision,
    inspect_native_mtp,
    reason_message,
)


logger = logging.getLogger(__name__)


class NativeMTPStartupError(RuntimeError):
    """Structured startup failure whose reason survives Node Agent logs."""

    def __init__(self, reason: str, message: str | None = None):
        self.reason = reason
        self.detail = message or reason_message(reason) or reason
        super().__init__(f"native_mtp_reason={reason} {self.detail}")


class PatchTransaction:
    """Record foreign attribute mutations and restore them in reverse order."""

    def __init__(self) -> None:
        self._changes: list[tuple[object, str, bool, object | None]] = []
        self._closed = False

    def set(self, owner: object, name: str, value: object) -> None:
        if self._closed:
            raise RuntimeError("Native MTP patch transaction is already closed")
        namespace = getattr(owner, "__dict__", {})
        existed = name in namespace
        previous = namespace.get(name)
        self._changes.append((owner, name, existed, previous))
        setattr(owner, name, value)

    def rollback(self) -> None:
        if self._closed:
            return
        for owner, name, existed, previous in reversed(self._changes):
            if existed:
                setattr(owner, name, previous)
            elif hasattr(owner, name):
                delattr(owner, name)
        self._closed = True

    def commit(self) -> "InstalledPatch":
        if self._closed:
            raise RuntimeError("Native MTP patch transaction is already closed")
        self._closed = True
        return InstalledPatch(self._changes)


@dataclass
class InstalledPatch:
    _changes: list[tuple[object, str, bool, object | None]]
    _rolled_back: bool = False

    def rollback(self) -> None:
        if self._rolled_back:
            return
        for owner, name, existed, previous in reversed(self._changes):
            if existed:
                setattr(owner, name, previous)
            elif hasattr(owner, name):
                delattr(owner, name)
        self._rolled_back = True


_CONSTRUCTION_LOCK = threading.RLock()
_CONSTRUCTION_ACTIVE = False


def is_native_mtp_construction_active() -> bool:
    return _CONSTRUCTION_ACTIVE


@contextmanager
def native_mtp_construction_scope(enabled: bool) -> Iterator[None]:
    """Serialize model construction and restore the prior activation flag."""

    global _CONSTRUCTION_ACTIVE
    with _CONSTRUCTION_LOCK:
        previous = _CONSTRUCTION_ACTIVE
        _CONSTRUCTION_ACTIVE = bool(enabled)
        try:
            yield
        finally:
            _CONSTRUCTION_ACTIVE = previous


@dataclass
class NativeMTPRuntimeController:
    model: str
    config: NativeMTPConfig
    decode_concurrency: int
    pipeline_parallel: bool = False

    def __post_init__(self) -> None:
        self.decision = inspect_native_mtp(Path(self.model), self.config)
        self.installed_patch: InstalledPatch | None = None
        self.telemetry = self.decision.to_readiness_dict()
        self.telemetry.update(
            {
                "effective_mode": "native_mtp" if self.decision.enabled else "standard",
                "proposed_tokens": 0,
                "accepted_tokens": 0,
                "acceptance_rate": None,
                "seeded_sequential_requests": 0,
            }
        )

    @property
    def enabled(self) -> bool:
        return self.decision.enabled

    def prepare(self, mx: Any, group: Any) -> NativeMTPDecision:
        """Run read-only checks, all-rank agreement, then atomic patching."""

        if not _rank_fingerprints_match(mx, group, self.decision.fingerprint):
            return self._fallback_or_raise(REASON_RANK_MISMATCH)

        if not self.decision.enabled:
            if self.decision.required_failure:
                raise NativeMTPStartupError(
                    self.decision.reason or REASON_UNSUPPORTED_RUNTIME,
                    self.decision.message,
                )
            self._refresh_telemetry()
            return self.decision

        if self.decode_concurrency != 1:
            return self._fallback_or_raise(REASON_UNSUPPORTED_DECODE_CONCURRENCY)
        if self.pipeline_parallel:
            return self._fallback_or_raise(REASON_UNSUPPORTED_TOPOLOGY)

        supported, detail = runtime_shape_supported()
        if not supported:
            return self._fallback_or_raise(REASON_UNSUPPORTED_RUNTIME, detail)

        # Runtime shape is local state too. Include it in a second final
        # fingerprint before the first mutation so heterogeneous ranks stop.
        final_fingerprint = hashlib.sha256(
            f"{self.decision.fingerprint}\0runtime-ok".encode("utf-8")
        ).hexdigest()
        if not _rank_fingerprints_match(mx, group, final_fingerprint):
            return self._fallback_or_raise(REASON_RANK_MISMATCH)

        transaction = PatchTransaction()
        local_error: Exception | None = None
        try:
            from . import cache, generation, qwen

            cache.install(transaction)
            qwen.install(transaction)
            generation.install(transaction, self.telemetry)
        except Exception as exc:  # keep every rank in the status collective
            local_error = exc

        all_succeeded = _all_ranks_succeeded(mx, group, local_error is None)
        if local_error is not None or not all_succeeded:
            transaction.rollback()
            message = reason_message(REASON_PATCH_INSTALL_FAILED) or REASON_PATCH_INSTALL_FAILED
            if local_error is not None:
                message = f"{message} {type(local_error).__name__}: {local_error}"
            # Once mutation was attempted, even auto is fatal: continuing in
            # a potentially heterogeneous Python process is not safe.
            raise NativeMTPStartupError(REASON_PATCH_INSTALL_FAILED, message)

        self.installed_patch = transaction.commit()
        self._refresh_telemetry()
        logger.warning(
            "Tokenity Native MTP patch active: model_type=%s depth=1 placement=replicated fingerprint=%s",
            self.decision.model_type,
            self.decision.fingerprint[:12],
        )
        return self.decision

    def construction_scope(self) -> Iterator[None]:
        return native_mtp_construction_scope(self.enabled)

    def validate_loaded_model(self, model: Any) -> None:
        if not self.enabled:
            return
        inner = getattr(model, "language_model", model)
        reasons: list[str] = []
        if not getattr(inner, "_tokenity_native_mtp_decode_enabled", False):
            reasons.append("construction marker missing")
        mtp = getattr(inner, "mtp", None)
        if mtp is None:
            reasons.append("MTP head missing")
        elif len(getattr(mtp, "layers", ())) != 1:
            reasons.append("MTP depth is not exactly one")
        for name in ("mtp_forward", "make_mtp_cache"):
            if not callable(getattr(model, name, None)):
                reasons.append(f"{name} missing")

        loaded_keys = set(getattr(model, "_tokenity_native_mtp_loaded_keys", ()))
        try:
            from mlx.utils import tree_flatten  # type: ignore

            parameter_keys = {
                key for key, _ in tree_flatten(model.parameters()) if ".mtp." in f".{key}."
            }
        except Exception as exc:
            reasons.append(f"could not inspect loaded parameters: {exc}")
            parameter_keys = set()
        if not parameter_keys:
            reasons.append("MTP parameter tree is empty")
        else:
            missing_loaded = sorted(parameter_keys - loaded_keys)
            if missing_loaded:
                reasons.append(
                    "MTP tensors were not supplied to load_weights: "
                    + ", ".join(missing_loaded[:4])
                )

        if reasons:
            raise NativeMTPStartupError(
                REASON_LOADED_INSTANCE_INVALID,
                "; ".join(reasons),
            )

    def record_seeded_fallback(self) -> None:
        self.telemetry["seeded_sequential_requests"] = int(
            self.telemetry.get("seeded_sequential_requests", 0)
        ) + 1

    def _fallback_or_raise(
        self,
        reason: str,
        detail: str | None = None,
    ) -> NativeMTPDecision:
        self.decision = self.decision.with_fallback(reason, message=detail)
        self._refresh_telemetry()
        if self.config.mode == "required":
            raise NativeMTPStartupError(reason, self.decision.message)
        logger.warning("Native MTP auto fallback: native_mtp_reason=%s %s", reason, self.decision.message)
        return self.decision

    def _refresh_telemetry(self) -> None:
        counters = {
            key: self.telemetry.get(key)
            for key in (
                "proposed_tokens",
                "accepted_tokens",
                "acceptance_rate",
                "seeded_sequential_requests",
            )
        }
        self.telemetry.clear()
        self.telemetry.update(self.decision.to_readiness_dict())
        self.telemetry["effective_mode"] = "native_mtp" if self.enabled else "standard"
        self.telemetry.update(counters)


def runtime_shape_supported() -> tuple[bool, str | None]:
    """Validate the exact mlx-lm internals patched by the MVP."""

    try:
        generate = importlib.import_module("mlx_lm.generate")
        server = importlib.import_module("mlx_lm.server")
        qwen = importlib.import_module("mlx_lm.models.qwen3_5")
    except Exception as exc:
        return False, f"required mlx-lm module import failed: {exc}"

    required = (
        (generate, "GenerationBatch"),
        (generate, "BatchGenerator"),
        (server, "load"),
        (server, "sharded_load"),
        (qwen, "TextModelArgs"),
        (qwen, "GatedDeltaNet"),
        (qwen, "DecoderLayer"),
        (qwen, "Qwen3_5TextModel"),
        (qwen, "TextModel"),
        (qwen, "Model"),
    )
    missing = [f"{owner.__name__}.{name}" for owner, name in required if not hasattr(owner, name)]
    if missing:
        return False, "missing runtime symbols: " + ", ".join(missing)

    generation_batch = generate.GenerationBatch
    batch_generator = generate.BatchGenerator
    signatures = {
        "GenerationBatch.__init__": (
            generation_batch.__init__,
            {"self", "model", "uids", "inputs", "prompt_cache", "tokens", "samplers", "fallback_sampler", "logits_processors", "state_machines", "max_tokens"},
        ),
        "GenerationBatch.next": (generation_batch.next, {"self"}),
        "GenerationBatch.extend": (generation_batch.extend, {"self", "batch"}),
        "GenerationBatch.filter": (generation_batch.filter, {"self", "keep"}),
        "BatchGenerator._next": (batch_generator._next, {"self"}),
    }
    for label, (callable_value, expected) in signatures.items():
        try:
            actual = set(inspect.signature(callable_value).parameters)
        except (TypeError, ValueError) as exc:
            return False, f"could not inspect {label}: {exc}"
        if actual != expected:
            return False, f"{label} parameters changed: expected {sorted(expected)}, got {sorted(actual)}"

    server_signatures = {
        "server.load": (
            server.load,
            {
                "path_or_hf_repo",
                "tokenizer_config",
                "model_config",
                "adapter_path",
                "lazy",
                "return_config",
                "revision",
            },
        ),
        "server.sharded_load": (
            server.sharded_load,
            {"repo", "pipeline_group", "tensor_group", "return_config", "tokenizer_config"},
        ),
    }
    for label, (callable_value, expected) in server_signatures.items():
        actual = set(inspect.signature(callable_value).parameters)
        if actual != expected:
            return False, f"{label} parameters changed: expected {sorted(expected)}, got {sorted(actual)}"
    return True, None


def _rank_fingerprints_match(mx: Any, group: Any, fingerprint: str) -> bool:
    world_size = int(group.size())
    if world_size <= 1:
        return True
    raw = bytes.fromhex(fingerprint)
    one_hot = [[0] * 256 for _ in raw]
    for index, value in enumerate(raw):
        one_hot[index][value] = 1
    encoded = mx.array(one_hot, dtype=mx.int32)
    summed = mx.distributed.all_sum(encoded, group=group)
    mx.eval(summed)
    rows = summed.tolist()
    return all(sum(row) == world_size and max(row) == world_size for row in rows)


def _all_ranks_succeeded(mx: Any, group: Any, local_success: bool) -> bool:
    world_size = int(group.size())
    if world_size <= 1:
        return local_success
    value = mx.array(1 if local_success else 0, dtype=mx.int32)
    total = mx.distributed.all_sum(value, group=group)
    mx.eval(total)
    return int(total.item()) == world_size


def attach_controller(server: ModuleType, controller: NativeMTPRuntimeController) -> None:
    """Publish the controller for Tokenity's load wrappers."""

    server._tokenity_native_mtp_controller = controller


def controller_for(server: ModuleType) -> NativeMTPRuntimeController | None:
    value = getattr(server, "_tokenity_native_mtp_controller", None)
    return value if isinstance(value, NativeMTPRuntimeController) else None
