from __future__ import annotations

import pytest

from tokenity.mlx.hostfile import ClusterNode, ConnectionMode, HostfileError, build_hostfile
from tokenity.mlx.launcher import build_distributed_openai_launch_plan, build_official_mlx_lm_launch_plan


def test_known_two_mac_jaccl_hostfile_shape():
    nodes = [
        ClusterNode(
            id="mac-a",
            ssh="127.0.0.1",
            lan_ip="192.168.5.23",
            rdma_ip="192.168.0.1",
            rdma_devices=["rdma_en4"],
        ),
        ClusterNode(
            id="mac-b",
            ssh="probriefing@192.168.5.75",
            lan_ip="192.168.5.75",
            rdma_ip="192.168.0.2",
            rdma_devices=["rdma_en5"],
        ),
    ]

    assert build_hostfile(nodes, ConnectionMode.JACCL) == [
        {"ssh": "127.0.0.1", "ips": ["192.168.0.1"], "rdma": [None, "rdma_en4"]},
        {"ssh": "probriefing@192.168.5.75", "ips": [], "rdma": ["rdma_en5", None]},
    ]


def test_jaccl_blocks_missing_rdma_data():
    nodes = [
        ClusterNode(id="mac-a", ssh="127.0.0.1", lan_ip="192.168.5.23"),
        ClusterNode(id="mac-b", ssh="probriefing@192.168.5.75", lan_ip="192.168.5.75"),
    ]

    with pytest.raises(HostfileError, match="JACCL readiness failed"):
        build_hostfile(nodes, ConnectionMode.JACCL)


def test_launch_plan_prefers_python_venv_mlx_launch(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python3"
    mlx_launch = bin_dir / "mlx.launch"
    mlx_lm_server = bin_dir / "mlx_lm.server"
    python.write_text("", encoding="utf-8")
    mlx_launch.write_text("", encoding="utf-8")
    mlx_lm_server.write_text("", encoding="utf-8")

    plan = build_official_mlx_lm_launch_plan(
        nodes=[ClusterNode(id="mac-a", ssh="127.0.0.1", lan_ip="192.168.5.23")],
        connection_mode=ConnectionMode.RING,
        model="/models/qwen",
        python=str(python),
    )

    assert plan.command[0] == str(mlx_launch)
    assert "--python" not in plan.command
    assert plan.command[plan.command.index("--") + 1] == str(mlx_lm_server)


def test_distributed_plan_uses_path_resolved_python(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python3"
    mlx_launch = bin_dir / "mlx.launch"
    python.write_text("", encoding="utf-8")
    mlx_launch.write_text("", encoding="utf-8")

    plan = build_distributed_openai_launch_plan(
        nodes=[ClusterNode(id="mac-a", ssh="127.0.0.1", lan_ip="192.168.5.23")],
        connection_mode=ConnectionMode.RING,
        model="/models/qwen",
        python=str(python),
    )

    assert plan.command[0] == str(mlx_launch)
    assert plan.command[plan.command.index("--") + 1] == "python"
    assert "--no-verify-script" in plan.command
    assert any(str(python.parent) in item for item in plan.command if item.startswith("PATH="))
