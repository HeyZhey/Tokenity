"""Host-independent filesystem configuration for Tokenity.

Every deployment path can be overridden with an environment variable.  The
fallbacks intentionally use the current account's platform data directory so
that importing Tokenity never binds a checkout to one developer or machine.
System packages set the same variables in their launchd jobs when a shared
installation is desired.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _configured_path(
    name: str,
    fallback: Path,
    environ: Mapping[str, str] | None = None,
) -> Path:
    value = _environment(environ).get(name)
    if not value:
        return fallback
    configured = Path(value).expanduser()
    if not configured.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return configured


def data_root(environ: Mapping[str, str] | None = None) -> Path:
    env = _environment(environ)
    configured = env.get("TOKENITY_DATA_ROOT")
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            raise ValueError("TOKENITY_DATA_ROOT must be an absolute path")
        return root
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Tokenity"
    xdg_data_home = env.get("XDG_DATA_HOME")
    if xdg_data_home:
        base = Path(xdg_data_home).expanduser()
        if not base.is_absolute():
            raise ValueError("XDG_DATA_HOME must be an absolute path")
    else:
        base = Path.home() / ".local" / "share"
    return base / "tokenity"


def model_root(environ: Mapping[str, str] | None = None) -> Path:
    return _configured_path(
        "TOKENITY_MODEL_ROOT", data_root(environ) / "Models", environ
    )


def runtime_root(environ: Mapping[str, str] | None = None) -> Path:
    return _configured_path(
        "TOKENITY_RUNTIME_ROOT", data_root(environ) / "Runtime", environ
    )


def code_root(environ: Mapping[str, str] | None = None) -> Path:
    return _configured_path("TOKENITY_CODE_ROOT", data_root(environ) / "Code", environ)


def state_root(environ: Mapping[str, str] | None = None) -> Path:
    return _configured_path("TOKENITY_STATE_ROOT", data_root(environ) / "State", environ)


def log_root(environ: Mapping[str, str] | None = None) -> Path:
    return _configured_path("TOKENITY_LOG_ROOT", data_root(environ) / "Logs", environ)


def runtime_python(environ: Mapping[str, str] | None = None) -> Path:
    return _configured_path(
        "TOKENITY_RUNTIME_PYTHON",
        runtime_root(environ) / "current" / ".venv" / "bin" / "python",
        environ,
    )


def h3_binary(environ: Mapping[str, str] | None = None) -> Path:
    return _configured_path(
        "TOKENITY_H3_BINARY_PATH",
        runtime_root(environ) / "current" / "bin" / "mlx-serve",
        environ,
    )
