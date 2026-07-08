from __future__ import annotations

from tokenity.mlx.hostfile import ClusterNode, ConnectionMode
from tokenity.mlx.launcher import build_official_mlx_lm_launch_plan


def test_official_plan_is_marked_experimental():
    plan = build_official_mlx_lm_launch_plan(
        nodes=[ClusterNode(id="local", ssh="127.0.0.1", lan_ip="127.0.0.1")],
        connection_mode=ConnectionMode.RING,
        model="/Users/Shared/TokenityModels/Qwen",
        python="/usr/bin/python3",
    )

    assert plan.experimental is True
    assert "mlx_lm" in plan.command
    assert "server" in plan.command
    assert "<generated-hostfile>" in plan.command
    assert plan.command[plan.command.index("--backend") + 1] == "ring"


def test_jaccl_plan_uses_jaccl_backend():
    plan = build_official_mlx_lm_launch_plan(
        nodes=[
            ClusterNode(id="mac-a", ssh="127.0.0.1", rdma_ip="192.168.0.1", rdma_devices=["rdma_en4"]),
            ClusterNode(id="mac-b", ssh="probriefing@192.168.5.75", rdma_ip="192.168.0.2", rdma_devices=["rdma_en5"]),
        ],
        connection_mode=ConnectionMode.JACCL,
        model="/Users/Shared/TokenityModels/Qwen",
        python="/usr/bin/python3",
    )

    assert plan.command[plan.command.index("--backend") + 1] == "jaccl"
