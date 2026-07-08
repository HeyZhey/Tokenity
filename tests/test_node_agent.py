from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

from tokenity.mlx.rdma_probe import RDMAProbeResult
from tokenity.node_agent.agent import create_app
from tokenity.process.supervisor import RoleSupervisor


def fake_rdma_probe():
    return RDMAProbeResult(
        rdma_enabled=True,
        rdma_devices=["rdma_en4"],
        rdma_port_state={"rdma_en4": "active"},
        thunderbolt_ip="192.168.0.1",
        rdma_errors=[],
    )


def test_node_info_reports_rdma():
    client = TestClient(create_app(rdma_probe_fn=fake_rdma_probe))

    response = client.get("/v1/node/info")

    assert response.status_code == 200
    assert response.json()["rdma"]["rdma_devices"] == ["rdma_en4"]


def test_running_role_reports_failed_when_child_rank_crashes(tmp_path: Path):
    supervisor = RoleSupervisor(log_dir=tmp_path)
    status = supervisor.start(
        "distributed-openai",
        [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    assert status.log_path is not None
    Path(status.log_path).write_text(
        "Tokenity distributed runtime failed: [jaccl] Changing queue pair to RTR failed with errno 96\n"
        "libc++abi: terminating due to uncaught exception\n"
        "[WARN] Node with rank 1 exited with code 255\n"
        "[METAL] Command buffer execution failed: Caused GPU Timeout Error\n",
        encoding="utf-8",
    )

    try:
        failed = supervisor.status("distributed-openai")
        assert not isinstance(failed, list)
        assert failed.state == "failed"
        assert failed.pid is not None
        assert failed.message is not None
        assert "rank 1 exited" in failed.message
    finally:
        supervisor.stop("distributed-openai", timeout=1)


def test_model_scan(tmp_path: Path):
    model = tmp_path / "Qwen3.5-122B-A10B-4bit"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")

    client = TestClient(create_app(rdma_probe_fn=fake_rdma_probe))
    response = client.get("/v1/node/models", params={"root": str(tmp_path)})

    assert response.status_code == 200
    assert response.json()["models"][0]["id"] == "Qwen3.5-122B-A10B-4bit"


def test_official_dry_run_blocks_bad_jaccl():
    client = TestClient(create_app(rdma_probe_fn=fake_rdma_probe))
    response = client.post(
        "/v1/node/start-official-mlx-lm",
        json={
            "model": "/Users/Shared/TokenityModels/Qwen",
            "connection_mode": "jaccl",
            "nodes": [
                {"id": "mac-a", "ssh": "127.0.0.1"},
                {"id": "mac-b", "ssh": "probriefing@192.168.5.75"},
            ],
        },
    )

    assert response.status_code == 400
    assert "JACCL readiness failed" in response.json()["detail"]
