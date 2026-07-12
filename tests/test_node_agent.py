from __future__ import annotations

import subprocess
import sys
import time
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
    assert "memory" in response.json()


def test_node_status_reports_memory():
    client = TestClient(create_app(rdma_probe_fn=fake_rdma_probe))

    response = client.get("/v1/node/status")

    assert response.status_code == 200
    assert "memory" in response.json()


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


def test_supervisor_captures_process_output(tmp_path: Path):
    supervisor = RoleSupervisor(log_dir=tmp_path)
    status = supervisor.start(
        "probe",
        [sys.executable, "-c", "import time; print('pipe-log-ok', flush=True); time.sleep(30)"],
    )
    assert status.log_path is not None

    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if "pipe-log-ok" in Path(status.log_path).read_text(encoding="utf-8"):
                break
            time.sleep(0.05)
        else:
            raise AssertionError("process output was not written to the role log")
    finally:
        supervisor.stop("probe", timeout=1)


def test_supervisor_keeps_child_stdin_open(tmp_path: Path):
    supervisor = RoleSupervisor(log_dir=tmp_path)
    status = supervisor.start(
        "stdin-probe",
        [
            sys.executable,
            "-c",
            (
                "import select, sys, time; "
                "time.sleep(0.2); "
                "ready = bool(select.select([sys.stdin], [], [], 0)[0]); "
                "print('stdin-readable' if ready else 'stdin-open', flush=True); "
                "time.sleep(30)"
            ),
        ],
    )
    assert status.log_path is not None

    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if "stdin-open" in Path(status.log_path).read_text(encoding="utf-8"):
                break
            time.sleep(0.05)
        else:
            raise AssertionError("child stdin looked closed or readable at startup")
    finally:
        supervisor.stop("stdin-probe", timeout=1)


def test_supervisor_stop_cleans_orphaned_hostfile_process_group(tmp_path: Path):
    hostfile = tmp_path / "tokenity-hostfiles" / "distributed-openai-test.json"
    hostfile.parent.mkdir()
    hostfile.write_text("[]", encoding="utf-8")
    orphan = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            "--hostfile",
            str(hostfile),
        ],
        start_new_session=True,
    )
    supervisor = RoleSupervisor(log_dir=tmp_path)
    supervisor._commands["distributed-openai"] = [  # noqa: SLF001 - targeted supervisor cleanup regression
        "ssh",
        "-o",
        "BatchMode=yes",
        "127.0.0.1",
        f"mlx.launch --hostfile {hostfile} -- python -m tokenity distributed-openai serve",
    ]

    try:
        supervisor.stop("distributed-openai", timeout=1)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if orphan.poll() is not None:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("orphaned hostfile process was not terminated")
    finally:
        if orphan.poll() is None:
            orphan.kill()
            orphan.wait(timeout=5)


def test_supervisor_stop_kills_related_process_after_wrapper_exits(tmp_path: Path):
    hostfile = tmp_path / "tokenity-hostfiles" / "distributed-openai-test.json"
    hostfile.parent.mkdir()
    hostfile.write_text("[]", encoding="utf-8")
    stubborn = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, lambda *_: None); time.sleep(30)",
            "--hostfile",
            str(hostfile),
        ],
        start_new_session=True,
    )
    supervisor = RoleSupervisor(log_dir=tmp_path)
    supervisor.start(
        "distributed-openai",
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            "--hostfile",
            str(hostfile),
        ],
    )

    try:
        supervisor.stop("distributed-openai", timeout=1)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if stubborn.poll() is not None:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("related hostfile process survived final SIGKILL")
    finally:
        if stubborn.poll() is None:
            stubborn.kill()
            stubborn.wait(timeout=5)


def test_supervisor_stop_kills_related_tokenity_serve_process(tmp_path: Path):
    model = tmp_path / "Qwen"
    model.mkdir()
    rank = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, lambda *_: None); time.sleep(30)",
            "tokenity",
            "distributed-openai",
            "serve",
            "--model",
            str(model),
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        start_new_session=True,
    )
    supervisor = RoleSupervisor(log_dir=tmp_path)
    supervisor._commands["distributed-openai"] = [  # noqa: SLF001 - targeted supervisor cleanup regression
        "ssh",
        "127.0.0.1",
        f"mlx.launch -- python -m tokenity distributed-openai serve --model {model} --host 0.0.0.0 --port 8000",
    ]

    try:
        supervisor.stop("distributed-openai", timeout=1)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if rank.poll() is not None:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("related Tokenity serve process survived final SIGKILL")
    finally:
        if rank.poll() is None:
            rank.kill()
            rank.wait(timeout=5)


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
