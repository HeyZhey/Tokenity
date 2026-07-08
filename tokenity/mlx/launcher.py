from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any

from .hostfile import ClusterNode, ConnectionMode, build_hostfile


@dataclass(slots=True)
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
        "mlx.launch",
        "--hostfile",
        hostfile_path,
        "--backend",
        backend,
        "--python",
        python,
        "--starting-port",
        str(starting_port),
        "--",
        "mlx_lm",
        "server",
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
) -> LaunchPlan:
    hostfile = build_hostfile(nodes, connection_mode)
    backend = _mlx_backend(connection_mode)
    command = [
        "mlx.launch",
        "--hostfile",
        hostfile_path,
        "--backend",
        backend,
        "--python",
        python,
        "--starting-port",
        str(starting_port),
        "--",
        python,
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
    ]
    return LaunchPlan(
        role="distributed-openai",
        backend="Tokenity distributed OpenAI server",
        experimental=False,
        command=command,
        hostfile=hostfile,
        warnings=["Skeleton: distributed request loop and sharded generation are not complete yet."],
    )


def _mlx_backend(connection_mode: ConnectionMode) -> str:
    if connection_mode == ConnectionMode.RING:
        return "ring"
    if connection_mode in {ConnectionMode.JACCL, ConnectionMode.JACCL_RING}:
        return "jaccl"
    raise ValueError(f"Unsupported connection mode: {connection_mode}")
