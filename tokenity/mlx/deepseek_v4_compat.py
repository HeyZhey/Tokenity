"""Process-local DeepSeek V4 support for Tokenity's pinned MLX-LM runtime.

Tokenity pins MLX-LM 0.31.3, while DeepSeek V4 support was developed in
ml-explore/mlx-lm PR #1189 but not released. The upstream model modules are
vendored byte-for-byte and loaded under ``mlx_lm.models`` only for the current
process. The user's site-packages are never modified.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import struct
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Any


UPSTREAM_PR = "https://github.com/ml-explore/mlx-lm/pull/1189"
UPSTREAM_COMMIT = "63a26625c7ba2ffb8159ff430e630321446c7df4"
SUPPORTED_MLX_LM_VERSION = "0.31.3"
_VENDORED_MODULES = ("sinkhorn", "hyper_connection", "deepseek_v4")
_VENDOR_ROOT = Path(__file__).with_name("_upstream_pr1189")
_MARKER = "_tokenity_deepseek_v4_pr1189"
_QUANTIZATION_MARKER = "_tokenity_deepseek_v4_mixed_quantization"
_SWITCH_PROJECTIONS = frozenset(("gate_proj", "up_proj", "down_proj"))
_MXFP4_QUANTIZATION = {"group_size": 32, "bits": 4, "mode": "mxfp4"}


def _installed_mlx_lm_version() -> str | None:
    try:
        return version("mlx-lm")
    except PackageNotFoundError:
        return None


def _import_existing_deepseek_v4() -> ModuleType | None:
    name = "mlx_lm.models.deepseek_v4"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        return None


def _safetensors_header(path: Path) -> dict[str, Any]:
    """Read tensor metadata without mapping or materializing checkpoint data."""

    try:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return {}
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 0 or header_length > 256 * 1024 * 1024:
                return {}
            decoded = json.loads(handle.read(header_length))
    except (OSError, json.JSONDecodeError, struct.error):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _is_mxfp4_tensor_pair(weight: Any, scales: Any) -> bool:
    if not isinstance(weight, dict) or not isinstance(scales, dict):
        return False
    weight_shape = weight.get("shape")
    scales_shape = scales.get("shape")
    if not isinstance(weight_shape, list) or not isinstance(scales_shape, list):
        return False
    if not weight_shape or not scales_shape:
        return False
    packed_width = weight_shape[-1]
    scale_width = scales_shape[-1]
    return (
        weight.get("dtype") == "U32"
        and scales.get("dtype") == "U8"
        and isinstance(packed_width, int)
        and isinstance(scale_width, int)
        and scale_width > 0
        and packed_width == scale_width * 4
    )


def _inferred_mxfp4_switch_paths(model_path: Path) -> tuple[str, ...]:
    """Find routed-expert projections whose files encode MXFP4 metadata."""

    index_path = model_path / "model.safetensors.index.json"
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict):
        return ()

    model_root = model_path.resolve()
    headers: dict[str, dict[str, Any]] = {}

    def metadata(key: str) -> Any:
        shard = weight_map.get(key)
        if not isinstance(shard, str):
            return None
        if shard not in headers:
            shard_path = (model_path / shard).resolve()
            if not shard_path.is_relative_to(model_root):
                headers[shard] = {}
            else:
                headers[shard] = _safetensors_header(shard_path)
        return headers[shard].get(key)

    inferred: list[str] = []
    for key in weight_map:
        if not isinstance(key, str) or not key.endswith(".weight"):
            continue
        layer_path = key.removesuffix(".weight")
        if ".switch_mlp." not in layer_path:
            continue
        if layer_path.rsplit(".", 1)[-1] not in _SWITCH_PROJECTIONS:
            continue
        scales_key = f"{layer_path}.scales"
        if scales_key not in weight_map or f"{layer_path}.biases" in weight_map:
            continue
        if _is_mxfp4_tensor_pair(metadata(key), metadata(scales_key)):
            inferred.append(layer_path)
    return tuple(sorted(inferred))


def _augment_mixed_quantization_config(
    model_path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Supply omitted per-layer metadata for hybrid DeepSeek V4 exports."""

    if config.get("model_type") != "deepseek_v4":
        return config, 0
    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        return config, 0

    inferred_paths = _inferred_mxfp4_switch_paths(model_path)
    missing_paths = [path for path in inferred_paths if path not in quantization]
    if not missing_paths:
        return config, 0

    augmented = dict(config)
    augmented_quantization = dict(quantization)
    for path in missing_paths:
        augmented_quantization[path] = dict(_MXFP4_QUANTIZATION)
    augmented["quantization"] = augmented_quantization
    return augmented, len(missing_paths)


def _install_mixed_quantization_compat() -> bool:
    """Patch MLX-LM's config reader for under-specified hybrid checkpoints."""

    utils = importlib.import_module("mlx_lm.utils")
    if getattr(utils, _QUANTIZATION_MARKER, False):
        return False
    original_load_config = utils.load_config

    def load_config_with_mxfp4_switch_metadata(model_path: Path) -> dict[str, Any]:
        config = original_load_config(model_path)
        augmented, count = _augment_mixed_quantization_config(Path(model_path), config)
        if count:
            logging.warning(
                "Tokenity inferred MXFP4 quantization for %d DeepSeek V4 "
                "Switch-MLP projections omitted from config.json.",
                count,
            )
        return augmented

    utils.load_config = load_config_with_mxfp4_switch_metadata
    setattr(utils, _QUANTIZATION_MARKER, True)
    return True


def install_deepseek_v4_compat() -> bool:
    """Install PR #1189 modules and return whether this call applied them.

    A future MLX-LM release with native ``deepseek_v4`` support wins. The
    vendored implementation is intentionally ABI-gated to the exact upstream
    version on which the PR was based.
    """

    existing = _import_existing_deepseek_v4()
    if existing is not None:
        return False

    installed_version = _installed_mlx_lm_version()
    if installed_version != SUPPORTED_MLX_LM_VERSION:
        raise RuntimeError(
            "Tokenity DeepSeek V4 compatibility requires mlx-lm "
            f"{SUPPORTED_MLX_LM_VERSION}; found {installed_version or 'not installed'}."
        )

    models_package = importlib.import_module("mlx_lm.models")
    installed_names: list[str] = []
    previous_attributes: dict[str, Any] = {}
    try:
        for leaf_name in _VENDORED_MODULES:
            qualified_name = f"mlx_lm.models.{leaf_name}"
            source_path = _VENDOR_ROOT / f"{leaf_name}.py"
            spec = importlib.util.spec_from_file_location(qualified_name, source_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load vendored module {source_path}.")
            module = importlib.util.module_from_spec(spec)
            setattr(module, _MARKER, UPSTREAM_COMMIT)
            if leaf_name == "deepseek_v4":
                # PR #1189's final MTP commit uses ``Any`` in annotations but
                # omitted the typing import. Seed the module namespace while
                # keeping the vendored upstream file byte-identical.
                module.Any = Any
            if hasattr(models_package, leaf_name):
                previous_attributes[leaf_name] = getattr(models_package, leaf_name)
            sys.modules[qualified_name] = module
            setattr(models_package, leaf_name, module)
            installed_names.append(qualified_name)
            spec.loader.exec_module(module)
    except BaseException:
        for qualified_name in reversed(installed_names):
            leaf_name = qualified_name.rsplit(".", 1)[-1]
            sys.modules.pop(qualified_name, None)
            if leaf_name in previous_attributes:
                setattr(models_package, leaf_name, previous_attributes[leaf_name])
            elif getattr(models_package, leaf_name, None) is not None:
                delattr(models_package, leaf_name)
        raise
    _install_mixed_quantization_compat()
    return True


def install_deepseek_v4_server_compat(server: Any) -> bool:
    """Disable MLX-LM batch generation for the PR's hybrid cache model."""

    provider = server.ModelProvider
    if getattr(provider, _MARKER, None) == UPSTREAM_COMMIT:
        return False

    original_load = provider._load

    def load_without_unsafe_batching(
        self: Any,
        model_path: Any,
        adapter_path: Any = None,
        draft_model_path: Any = None,
    ) -> None:
        original_load(self, model_path, adapter_path, draft_model_path)
        if getattr(self.model, "model_type", None) == "deepseek_v4":
            self.is_batchable = False

    provider._load = load_without_unsafe_batching
    setattr(provider, _MARKER, UPSTREAM_COMMIT)
    return True
