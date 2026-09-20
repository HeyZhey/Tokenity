"""The separately packaged MLX-VLM interpreter; never upgrade the live LM venv."""

from __future__ import annotations

import os
from pathlib import Path


BACKEND_NAME = "mlx-vlm-macos-arm64-0.7.1"
PACKAGES = {"mlx": "0.32.2", "mlx-vlm": "0.7.1", "transformers": "5.16.1"}


def runtime_python(agent_python: str) -> str:
    override = os.environ.get("TOKENITY_MLX_VLM_PYTHON")
    if override:
        return override
    # Keep the venv path: resolving its python symlink loses the backend root.
    for parent in Path(agent_python).absolute().parents:
        candidate = parent / "backends" / BACKEND_NAME / ".venv/bin/python"
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(
        "The MLX-VLM runtime is missing. Install the Tokenity package with "
        "MLX-VLM support, or set TOKENITY_MLX_VLM_PYTHON to its isolated interpreter."
    )


def validate_packages() -> None:
    from importlib.metadata import PackageNotFoundError, version

    for name, expected in PACKAGES.items():
        try:
            found = version(name)
        except PackageNotFoundError:
            found = "not installed"
        if found != expected:
            raise RuntimeError(f"MLX-VLM backend requires {name} {expected}; found {found}.")
