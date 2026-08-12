from __future__ import annotations

from pathlib import Path

import pytest

from tokenity.mlx.hostfile import ClusterNode, ConnectionMode
from tokenity.mlx.launcher import (
    LegacyLauncherDisabled,
    build_distributed_openai_launch_plan,
    build_official_mlx_lm_launch_plan,
)


@pytest.mark.parametrize(
    "builder",
    [build_official_mlx_lm_launch_plan, build_distributed_openai_launch_plan],
)
def test_legacy_hostfile_launchers_are_permanently_disabled(builder):
    with pytest.raises(LegacyLauncherDisabled, match="typed HTTP instance API"):
        builder(
            nodes=[ClusterNode(id="local", agent_url="http://127.0.0.1:9100", lan_ip="127.0.0.1")],
            connection_mode=ConnectionMode.RING,
            model="/models/qwen",
            python="/runtime/bin/python",
        )


def test_production_python_contains_no_remote_shell_invocation_path():
    root = Path(__file__).parents[1] / "tokenity"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))

    assert '["ssh"' not in source
    assert "TOKENITY_MLX_LAUNCH_VIA_LOCAL_SSH" not in source
    assert "TOKENITY_MLX_LOCAL_SSH_HOST" not in source
