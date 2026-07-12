from __future__ import annotations

from tokenity.mlx.hostfile import ClusterNode, ConnectionMode
from tokenity.mlx.launcher import (
    build_distributed_openai_launch_plan,
    build_official_mlx_lm_launch_plan,
)


def test_official_plan_is_marked_experimental():
    plan = build_official_mlx_lm_launch_plan(
        nodes=[ClusterNode(id="local", ssh="127.0.0.1", lan_ip="127.0.0.1")],
        connection_mode=ConnectionMode.RING,
        model="/Users/Shared/TokenityModels/Qwen",
        python="/usr/bin/python3",
    )

    assert plan.experimental is True
    assert any("mlx_lm" in item for item in plan.command)
    assert any("server" in item for item in plan.command)
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


def test_distributed_plan_uses_runtime_python_for_tokenity_server():
    plan = build_distributed_openai_launch_plan(
        nodes=[ClusterNode(id="local", ssh="127.0.0.1", lan_ip="127.0.0.1")],
        connection_mode=ConnectionMode.RING,
        model="/Users/Shared/TokenityModels/Qwen",
        python="/Users/Shared/TokenityRuntime/current/.venv/bin/python",
    )

    separator = plan.command.index("--")
    assert plan.command[separator + 1] == "python"
    assert "--no-verify-script" in plan.command
    env_values = [plan.command[index + 1] for index, item in enumerate(plan.command) if item == "--env"]
    path_env = env_values[0]
    assert path_env.startswith("PATH=/Users/Shared/TokenityRuntime/current/.venv/bin")
    assert "MLX_METAL_FAST_SYNCH=1" in env_values
    assert "TOKENITY_MLX_LOAD_EVAL_CHUNK_SIZE=1" in env_values
    assert "TOKENITY_MLX_LOAD_EVAL_LOG_INTERVAL=100" in env_values
    assert "TOKENITY_MLX_LOAD_EVAL_SLEEP_SECONDS=0.05" in env_values
    assert "TOKENITY_MLX_LOAD_POST_BARRIER=0" in env_values
    assert "TOKENITY_MLX_DISTRIBUTED_INIT_RANK0_DELAY_SECONDS=0" in env_values
    assert plan.command[plan.command.index("--backend") + 1] == "ring"


def test_distributed_plan_can_wrap_launcher_in_local_ssh(monkeypatch):
    monkeypatch.setenv("TOKENITY_MLX_LAUNCH_VIA_LOCAL_SSH", "1")

    plan = build_distributed_openai_launch_plan(
        nodes=[ClusterNode(id="local", ssh="127.0.0.1", lan_ip="127.0.0.1")],
        connection_mode=ConnectionMode.RING,
        model="/Users/Shared/TokenityModels/Qwen",
        python="/Users/Shared/TokenityRuntime/current/.venv/bin/python",
    )

    assert plan.command[:5] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
    ]
    assert plan.command[5] == "127.0.0.1"
    assert "mlx.launch" in plan.command[6]
    assert "distributed-openai serve" in plan.command[6]


def test_distributed_plan_auto_wraps_multi_node_launcher():
    plan = build_distributed_openai_launch_plan(
        nodes=[
            ClusterNode(id="mac-a", ssh="127.0.0.1", lan_ip="192.168.0.1"),
            ClusterNode(id="mac-b", ssh="probriefing@192.168.5.75", lan_ip="192.168.0.2"),
        ],
        connection_mode=ConnectionMode.RING,
        model="/Users/Shared/TokenityModels/Qwen",
        python="/Users/Shared/TokenityRuntime/current/.venv/bin/python",
    )

    assert plan.command[0] == "ssh"
    assert plan.command[5] == "127.0.0.1"
    assert "--hostfile" in plan.command[6]
