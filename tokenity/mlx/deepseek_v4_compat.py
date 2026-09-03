"""Process-local DeepSeek V4 support for Tokenity's pinned MLX-LM runtime.

Tokenity pins MLX-LM 0.31.3, while DeepSeek V4 support was developed in
ml-explore/mlx-lm PR #1189 but not released. The upstream model modules are
vendored byte-for-byte and loaded under ``mlx_lm.models`` only for the current
process. The user's site-packages are never modified.
"""

from __future__ import annotations

import importlib
import importlib.util
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
