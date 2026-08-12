from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any

from .hostfile import ClusterNode, ConnectionMode
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


class LegacyLauncherDisabled(RuntimeError):
    pass


def _legacy_launcher_disabled() -> None:
    raise LegacyLauncherDisabled(
        "Legacy hostfile launch is permanently disabled. Use the Node Agent typed HTTP instance API."
    )


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
    del nodes, connection_mode, model, python, starting_port, host, port, hostfile_path
    _legacy_launcher_disabled()


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
    del (
        nodes,
        connection_mode,
        model,
        python,
        starting_port,
        host,
        port,
        hostfile_path,
        api_identifier,
        max_tokens,
        prompt_cache_size,
        prefill_step_size,
        decode_concurrency,
        prompt_concurrency,
        trust_remote_code,
        native_mtp,
    )
    _legacy_launcher_disabled()
