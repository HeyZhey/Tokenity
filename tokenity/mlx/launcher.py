from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hostfile import ClusterNode, ConnectionMode, build_hostfile
from tokenity.inference.native_mtp import NativeMTPConfig


@dataclass
class LaunchPlan:
    role: str
    backend: str
    experimental: bool
    command: list[str]
    hostfile: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)

    @property
    def command_string(self) -> str:
        return shlex.join(self.command)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "backend": self.backend,
            "experimental": self.experimental,
            "command": self.command,
            "command_string": self.command_string,
            "hostfile": self.hostfile,
            "warnings": self.warnings,
        }


def build_official_mlx_lm_launch_plan(
    *,
    nodes: list[ClusterNode],
    connection_mode: ConnectionMode,
    model: str,
    python: str,
    starting_port: int = 29500,
    host: str = "0.0.0.0",
    port: int = 8000,
    hostfile_path: str = "<generated-hostfile>",
) -> LaunchPlan:
    hostfile = build_hostfile(nodes, connection_mode)
    backend = _mlx_backend(connection_mode)
    command = [
        _mlx_launch_executable(python),
        "--hostfile",
        hostfile_path,
        "--backend",
        backend,
        "--starting-port",
        str(starting_port),
        "--",
        _tool_executable("mlx_lm.server", python),
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
    ]
    return LaunchPlan(
        role="official-mlx-lm",
        backend="Official mlx_lm server",
        experimental=True,
        command=command,
        hostfile=hostfile,
        warnings=[
            "Experimental: official mlx_lm server has not been stable for the target A/B multi-node chat path.",
        ],
    )


def build_distributed_openai_launch_plan(
    *,
    nodes: list[ClusterNode],
    connection_mode: ConnectionMode,
    model: str,
    python: str,
    starting_port: int = 29500,
    host: str = "0.0.0.0",
    port: int = 8000,
    hostfile_path: str = "<generated-hostfile>",
    api_identifier: str | None = None,
    max_tokens: int = 32_768,
    prompt_cache_size: int = 4,
    prefill_step_size: int = 2_048,
    decode_concurrency: int = 1,
    prompt_concurrency: int = 1,
    trust_remote_code: bool = False,
    native_mtp: NativeMTPConfig | None = None,
) -> LaunchPlan:
    hostfile = build_hostfile(nodes, connection_mode)
    backend = _mlx_backend(connection_mode)
    native_mtp = native_mtp or NativeMTPConfig()
    command = [
        _mlx_launch_executable(python),
        "--hostfile",
        hostfile_path,
        "--backend",
        backend,
        "--starting-port",
        str(starting_port),
        "--env",
        f"PATH={_distributed_path(python)}",
        "--env",
        f"PYTHONPATH={_distributed_code_root()}",
        "--env",
        f"MLX_METAL_FAST_SYNCH={os.environ.get('MLX_METAL_FAST_SYNCH', '1')}",
        "--env",
        f"TOKENITY_MLX_LOAD_EVAL_CHUNK_SIZE={os.environ.get('TOKENITY_MLX_LOAD_EVAL_CHUNK_SIZE', '1')}",
        "--env",
        f"TOKENITY_MLX_LOAD_EVAL_LOG_INTERVAL={os.environ.get('TOKENITY_MLX_LOAD_EVAL_LOG_INTERVAL', '100')}",
        "--env",
        f"TOKENITY_MLX_LOAD_EVAL_SLEEP_SECONDS={os.environ.get('TOKENITY_MLX_LOAD_EVAL_SLEEP_SECONDS', '0.05')}",
        "--env",
        f"TOKENITY_MLX_LOAD_POST_BARRIER={os.environ.get('TOKENITY_MLX_LOAD_POST_BARRIER', '0')}",
        "--env",
        "TOKENITY_MLX_DISTRIBUTED_INIT_RANK0_DELAY_SECONDS="
        f"{os.environ.get('TOKENITY_MLX_DISTRIBUTED_INIT_RANK0_DELAY_SECONDS', '0')}",
        "--cwd",
        _distributed_code_root(),
        "--no-verify-script",
        "--",
        "python",
        "-m",
        "tokenity",
        "distributed-openai",
        "serve",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--max-tokens",
        str(max_tokens),
        "--prompt-cache-size",
        str(prompt_cache_size),
        "--prefill-step-size",
        str(prefill_step_size),
        "--decode-concurrency",
        str(decode_concurrency),
        "--prompt-concurrency",
        str(prompt_concurrency),
        "--native-mtp-mode",
        native_mtp.mode,
        "--native-mtp-max-depth",
        str(native_mtp.max_depth),
        "--native-mtp-head-placement",
        native_mtp.head_placement,
    ]
    if api_identifier:
        command.extend(["--api-identifier", api_identifier])
    if trust_remote_code:
        command.append("--trust-remote-code")
    return LaunchPlan(
        role="distributed-openai",
        backend="Tokenity distributed OpenAI server",
        experimental=False,
        command=_maybe_wrap_local_ssh(command, hostfile),
        hostfile=hostfile,
        warnings=["Tokenity server uses the shared runtime Python and Tokenity-owned OpenAI-compatible serving path."],
    )


def _mlx_backend(connection_mode: ConnectionMode) -> str:
    if connection_mode == ConnectionMode.RING:
        return "ring"
    if connection_mode in {ConnectionMode.JACCL, ConnectionMode.JACCL_RING}:
        return "jaccl"
    raise ValueError(f"Unsupported connection mode: {connection_mode}")


def _mlx_launch_executable(python: str) -> str:
    return _tool_executable("mlx.launch", python)


def _tool_executable(name: str, python: str) -> str:
    shared_tool_dir = os.environ.get("TOKENITY_TOOL_DIR")
    if shared_tool_dir:
        candidate = Path(shared_tool_dir) / name
        if candidate.exists():
            return str(candidate)
    if python:
        candidate = Path(python).with_name(name)
        if candidate.exists():
            return str(candidate)
    return name


def _distributed_path(python: str) -> str:
    path_items: list[str] = []
    tool_dir = os.environ.get("TOKENITY_TOOL_DIR")
    if tool_dir:
        path_items.append(tool_dir)
    if python:
        path_items.append(str(Path(python).parent))
    existing = os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    path_items.extend([item for item in existing.split(os.pathsep) if item])
    deduped = list(dict.fromkeys(path_items))
    return os.pathsep.join(deduped)


def _distributed_code_root() -> str:
    return os.environ.get("TOKENITY_CODE_ROOT") or "/Users/Shared/TokenityCode"


def _maybe_wrap_local_ssh(command: list[str], hostfile: list[dict[str, Any]]) -> list[str]:
    mode = os.environ.get("TOKENITY_MLX_LAUNCH_VIA_LOCAL_SSH", "auto").strip().lower()
    enabled = mode in {"1", "true", "yes", "on"} or (mode == "auto" and len(hostfile) > 1)
    if not enabled:
        return command
    host = os.environ.get("TOKENITY_MLX_LOCAL_SSH_HOST", "127.0.0.1")
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        host,
        shlex.join(command),
    ]
