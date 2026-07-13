from __future__ import annotations

import subprocess
import sys
import time
import json
from pathlib import Path

from fastapi.testclient import TestClient

from tokenity.mlx.rdma_probe import RDMAProbeResult
from tokenity.mlx.glm_moe_dsa_compat import derive_indexer_types
from tokenity.node_agent.agent import RankStartRequest, _rank_command_and_environment, create_app
from tokenity.process.supervisor import RoleStatus, RoleSupervisor


def fake_rdma_probe():
    return RDMAProbeResult(
        rdma_enabled=True,
        rdma_devices=["rdma_en4"],
        rdma_port_state={"rdma_en4": "active"},
        thunderbolt_ip="192.168.0.1",
        rdma_errors=[],
    )


def test_glm_52_indexer_schedule_matches_upstream_pr_1410():
    schedule = derive_indexer_types(
        num_hidden_layers=78,
        frequency=4,
        skip_offset=3,
    )

    assert schedule[:15] == [
        "full",
        "full",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
    ]
    assert len(schedule) == 78
    assert schedule.count("full") == 21


def test_glm_indexer_pattern_accepts_explicit_full_shared_schedule():
    assert derive_indexer_types(
        num_hidden_layers=6,
        pattern="FSFSFS",
    ) == ["full", "shared", "full", "shared", "full", "shared"]


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


def test_supervisor_replaces_a_running_role_when_the_model_command_changes(tmp_path: Path):
    supervisor = RoleSupervisor(log_dir=tmp_path)
    first = supervisor.start(
        "distributed-openai",
        [sys.executable, "-c", "import time; time.sleep(30)", "--model", "GLM"],
    )
    second = supervisor.start(
        "distributed-openai",
        [sys.executable, "-c", "import time; time.sleep(30)", "--model", "Qwen"],
    )

    try:
        assert first.pid is not None
        assert second.pid is not None
        assert second.pid != first.pid
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            process_exists = subprocess.run(
                ["/bin/ps", "-p", str(first.pid)],
                capture_output=True,
                check=False,
            ).returncode == 0
            if not process_exists:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("the old model role survived replacement")
    finally:
        supervisor.stop("distributed-openai", timeout=1)


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
    (model / "config.json").write_text(
        '{"architectures":["Qwen3_5MoeForCausalLM"],"quantization":{"bits":4,"group_size":64}}',
        encoding="utf-8",
    )
    (model / "model-00001-of-00002.safetensors").write_bytes(b"a" * 128)
    (model / "model-00002-of-00002.safetensors").write_bytes(b"b" * 256)

    client = TestClient(create_app(rdma_probe_fn=fake_rdma_probe))
    response = client.get("/v1/node/models", params={"root": str(tmp_path)})

    assert response.status_code == 200
    payload = response.json()["models"][0]
    assert payload["id"] == "Qwen3.5-122B-A10B-4bit"
    assert payload["format"] == "MLX"
    assert payload["quantization"] == "4-bit · group 64"
    assert payload["architecture"] == "Qwen3_5MoeForCausalLM"
    assert payload["shard_count"] == 2
    assert payload["size_bytes"] >= 384


def test_legacy_official_backend_is_disabled_because_it_requires_ssh():
    client = TestClient(create_app(rdma_probe_fn=fake_rdma_probe))
    response = client.post(
        "/v1/node/start-official-mlx-lm",
        json={
            "model": "/Users/Shared/TokenityModels/Qwen",
            "connection_mode": "jaccl",
            "nodes": [
                {"id": "mac-a", "agent_url": "http://192.168.5.23:9100", "rdma_ip": "192.168.0.1", "rdma_devices": ["rdma_en4"]},
                {"id": "mac-b", "agent_url": "http://192.168.5.75:9100", "rdma_ip": "192.168.0.2", "rdma_devices": ["rdma_en5"]},
            ],
        },
    )

    assert response.status_code == 410
    assert "requires SSH" in response.json()["detail"]
    assert "start-distributed-openai" in response.json()["detail"]


def test_distributed_dry_run_forwards_runtime_configuration():
    client = TestClient(create_app(rdma_probe_fn=fake_rdma_probe))
    response = client.post(
        "/v1/node/start-distributed-openai",
        json={
            "model": "/Users/Shared/TokenityModels/Qwen",
            "connection_mode": "ring",
            "nodes": [
                {
                    "id": "local",
                    "agent_url": "http://127.0.0.1:9100",
                    "lan_ip": "127.0.0.1",
                }
            ],
            "api_identifier": "tokenity/qwen",
            "max_tokens": 65_536,
            "prompt_cache_size": 8,
            "prefill_step_size": 4_096,
            "decode_concurrency": 2,
            "prompt_concurrency": 3,
            "trust_remote_code": True,
            "dry_run": True,
        },
    )

    assert response.status_code == 200
    plan = response.json()["launch_plan"]
    assert plan["transport"] == "http"
    command = plan["ranks"][0]["command"]
    assert command[command.index("--api-identifier") + 1] == "tokenity/qwen"
    assert command[command.index("--max-tokens") + 1] == "65536"
    assert command[command.index("--prompt-cache-size") + 1] == "8"
    assert command[command.index("--prefill-step-size") + 1] == "4096"
    assert command[command.index("--decode-concurrency") + 1] == "2"
    assert command[command.index("--prompt-concurrency") + 1] == "3"
    assert "--trust-remote-code" in command


def test_cluster_payload_rejects_legacy_ssh_fields():
    client = TestClient(create_app(rdma_probe_fn=fake_rdma_probe))
    response = client.post(
        "/v1/node/start-distributed-openai",
        json={
            "model": "/models/qwen",
            "dry_run": True,
            "nodes": [{"id": "mac-b", "ssh": "user@example", "lan_ip": "192.168.5.75"}],
        },
    )

    assert response.status_code == 422
    assert "ssh" in response.text


def test_cluster_payload_rejects_credentials_in_agent_url():
    client = TestClient(create_app(rdma_probe_fn=fake_rdma_probe))
    response = client.post(
        "/v1/node/start-distributed-openai",
        json={
            "model": "/models/qwen",
            "dry_run": True,
            "nodes": [
                {
                    "id": "mac-b",
                    "agent_url": "http://user:password@192.168.5.75:9100",
                    "lan_ip": "192.168.5.75",
                }
            ],
        },
    )

    assert response.status_code == 400
    assert "must not contain a username or password" in response.json()["detail"]


def test_rank_environment_matches_mlx_jaccl_contract(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tokenity.node_agent.agent.tempfile.gettempdir", lambda: str(tmp_path))
    request = RankStartRequest(
        cluster_id="cluster-test",
        model="/models/qwen",
        rank=1,
        world_size=2,
        coordinator=False,
        connection_mode="jaccl",
        python="/runtime/bin/python",
        coordinator_ip="192.168.0.1",
        starting_port=30_020,
        rdma_matrix=[[None, "rdma_en4"], ["rdma_en5", None]],
    )

    command, env = _rank_command_and_environment(request)

    assert command[:4] == ["/runtime/bin/python", "-m", "tokenity", "distributed-openai"]
    assert env["MLX_RANK"] == "1"
    assert env["MLX_JACCL_COORDINATOR"] == "192.168.0.1:30020"
    assert json.loads(Path(env["MLX_IBV_DEVICES"]).read_text()) == [
        [None, "rdma_en4"],
        ["rdma_en5", None],
    ]


def test_single_node_ring_uses_mlx_singleton_without_an_empty_hostfile():
    request = RankStartRequest(
        cluster_id="cluster-test",
        model="/models/qwen",
        rank=0,
        world_size=1,
        coordinator=True,
        connection_mode="ring",
        python=sys.executable,
        ring_hosts=[["127.0.0.1:29500"]],
    )

    _, env = _rank_command_and_environment(request)

    assert env["MLX_RANK"] == "0"
    assert "MLX_HOSTFILE" not in env


class _FakeSupervisor:
    def __init__(self):
        self.starts = []
        self.stops = []

    def start(self, role, command, env=None, cwd=None, start_new_session=True):
        self.starts.append((role, command, env or {}, cwd, start_new_session))
        return RoleStatus(role=role, state="running", pid=123, command=command)

    def stop(self, role, timeout=10):
        self.stops.append((role, timeout))
        return RoleStatus(role=role, state="stopped")

    def status(self, role=None):
        if role is None:
            return []
        return RoleStatus(role=role, state="running", pid=123)


def test_distributed_start_fans_out_worker_rank_over_http():
    supervisor = _FakeSupervisor()
    posts = []

    def post_json(url, payload, timeout):
        posts.append((url, payload, timeout))
        return {"status": {"state": "running"}}

    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=supervisor,
            post_json_fn=post_json,
            rank_ready_fn=lambda pid, port, timeout: True,
            rank_stabilize_fn=lambda seconds: None,
            rank_connected_fn=lambda pid, request, timeout: True,
        )
    )
    response = client.post(
        "/v1/node/start-distributed-openai",
        json={
            "model": "/models/qwen",
            "connection_mode": "jaccl",
            "starting_port": 30_020,
            "dry_run": False,
            "nodes": [
                {
                    "id": "mac-a",
                    "agent_url": "http://192.168.5.23:9100",
                    "lan_ip": "192.168.5.23",
                    "rdma_ip": "192.168.0.1",
                    "rdma_devices": ["rdma_en4"],
                },
                {
                    "id": "mac-b",
                    "agent_url": "http://192.168.5.75:9100",
                    "lan_ip": "192.168.5.75",
                    "rdma_ip": "192.168.0.2",
                    "rdma_devices": ["rdma_en5"],
                },
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["launch_plan"]["transport"] == "http"
    assert supervisor.starts[0][0] == "distributed-openai"
    assert supervisor.starts[0][2]["MLX_RANK"] == "0"
    assert posts[0][0] == "http://192.168.5.75:9100/v1/node/start-distributed-rank"
    assert posts[0][1]["rank"] == 1
    assert "ssh" not in posts[0][1]

    stop = client.post(
        "/v1/node/stop-role",
        json={"role": "distributed-openai", "timeout": 5},
    )
    assert stop.status_code == 200
    assert posts[-1][0] == "http://192.168.5.75:9100/v1/node/stop-all"
    assert posts[-1][1]["timeout"] == 5


def test_stop_all_stops_every_model_role_and_heartbeat_renews_lease():
    supervisor = _FakeSupervisor()
    client = TestClient(create_app(rdma_probe_fn=fake_rdma_probe, supervisor=supervisor))

    heartbeat = client.post("/v1/node/heartbeat", json={"ttl_seconds": 45})
    stop = client.post("/v1/node/stop-all", json={"timeout": 2})

    assert heartbeat.status_code == 200
    assert heartbeat.json()["lease_seconds"] == 45
    assert stop.status_code == 200
    assert {role for role, _ in supervisor.stops} == {
        "distributed-openai",
        "distributed-openai-rank",
        "single-node-openai",
        "official-mlx-lm",
    }


def test_remote_rank_rejects_a_caller_selected_python_executable():
    client = TestClient(create_app(rdma_probe_fn=fake_rdma_probe))
    response = client.post(
        "/v1/node/start-distributed-rank",
        json={
            "cluster_id": "cluster-test",
            "model": "/models/qwen",
            "rank": 1,
            "world_size": 2,
            "connection_mode": "jaccl",
            "python": "/tmp/untrusted-python",
            "coordinator_ip": "192.168.0.1",
            "rdma_matrix": [[None, "rdma_en4"], ["rdma_en5", None]],
        },
    )

    assert response.status_code == 400
    assert "installed Python runtime" in response.json()["detail"]


def test_remote_rank_rolls_back_when_the_data_plane_does_not_connect():
    supervisor = _FakeSupervisor()
    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=supervisor,
            rank_connected_fn=lambda pid, request, timeout: False,
        )
    )
    response = client.post(
        "/v1/node/start-distributed-rank",
        json={
            "cluster_id": "cluster-test",
            "model": "/models/qwen",
            "rank": 1,
            "world_size": 2,
            "connection_mode": "jaccl",
            "python": sys.executable,
            "coordinator_ip": "192.168.0.1",
            "rdma_matrix": [[None, "rdma_en4"], ["rdma_en5", None]],
        },
    )

    assert response.status_code == 504
    assert supervisor.stops == [("distributed-openai-rank", 5)]


def test_distributed_runtime_configuration_is_validated():
    client = TestClient(create_app(rdma_probe_fn=fake_rdma_probe))
    response = client.post(
        "/v1/node/start-distributed-openai",
        json={
            "model": "/models/qwen",
            "max_tokens": 0,
            "dry_run": True,
        },
    )

    assert response.status_code == 422
