from __future__ import annotations

import subprocess
import sys
import time
import json
from pathlib import Path

from fastapi.testclient import TestClient

import tokenity.node_agent.agent as agent_module
from tokenity.mlx.rdma_probe import RDMAProbeResult
from tokenity.mlx.glm_moe_dsa_compat import derive_indexer_types
from tokenity.node_agent.agent import RankStartRequest, _rank_command_and_environment, create_app, scan_models
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


def test_node_info_advertises_stable_agent_contract():
    client = TestClient(create_app(rdma_probe_fn=fake_rdma_probe))

    response = client.get("/v1/node/info")

    assert response.status_code == 200
    assert response.json()["agent_contract"] == {
        "version": 1,
        "capabilities": [
            "cluster_runtime",
            "instance_quorum",
            "instance_runtimes",
            "managed_instances",
            "native_mtp",
        ],
    }


def test_node_status_reports_memory():
    client = TestClient(create_app(rdma_probe_fn=fake_rdma_probe))

    response = client.get("/v1/node/status")

    assert response.status_code == 200
    assert "memory" in response.json()


def test_runtime_status_marks_old_memory_sample_stale(tmp_path: Path):
    status_path = tmp_path / "runtime.json"
    status_path.write_text(
        json.dumps(
            {
                "phase": "ready",
                "updated_at": time.time() - 30,
                "memory": {"mlx_active_bytes": 123, "stale": False},
            }
        ),
        encoding="utf-8",
    )

    status = agent_module._read_runtime_status(str(status_path))

    assert status is not None
    assert status["status_stale"] is True
    assert status["memory"]["stale"] is True


def test_memory_stats_prefers_physical_vm_pages_over_pressure_percentage(monkeypatch):
    gib = 1_073_741_824
    total = 512 * gib
    page_size = 16_384
    free_pages = (125 * gib) // page_size
    monkeypatch.setattr(agent_module, "_total_memory_bytes", lambda: total)
    monkeypatch.setattr(
        agent_module,
        "_vm_stat_pages",
        lambda: {
            "page_size": page_size,
            "pages": {
                "Pages free": free_pages - 100,
                "Pages speculative": 100,
                "Pages wired down": (5 * gib) // page_size,
                "Pages occupied by compressor": 0,
                "Anonymous pages": (3 * gib) // page_size,
                "Pages inactive": (370 * gib) // page_size,
                "File-backed pages": (368 * gib) // page_size,
            },
        },
    )
    monkeypatch.setattr(agent_module, "_memory_pressure_available_ratio", lambda: 0.99)

    memory = agent_module._memory_stats()

    assert memory["free_bytes"] == 125 * gib
    assert memory["used_bytes"] == 387 * gib
    assert memory["used_ratio"] == 387 / 512
    assert memory["physical_used_bytes"] == 387 * gib
    assert memory["physical_used_ratio"] == 387 / 512
    assert memory["in_use_bytes"] == 8 * gib
    assert memory["in_use_ratio"] == 8 / 512
    assert memory["reclaimable_bytes"] == 368 * gib
    assert memory["file_backed_bytes"] == 368 * gib
    assert memory["pressure_available_ratio"] == 0.99


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


def test_supervisor_can_request_then_wait_for_a_clean_distributed_exit(tmp_path: Path):
    supervisor = RoleSupervisor(log_dir=tmp_path)
    started = supervisor.start(
        "distributed-openai-rank",
        [
            sys.executable,
            "-c",
            (
                "import signal, sys, time; "
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
                "print('ready', flush=True); "
                "time.sleep(30)"
            ),
        ],
    )
    assert started.log_path is not None
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if "ready" in Path(started.log_path).read_text(encoding="utf-8"):
            break
        time.sleep(0.05)
    else:
        raise AssertionError("distributed exit probe did not start")

    supervisor.request_stop("distributed-openai-rank")
    status = supervisor.wait("distributed-openai-rank", timeout=3)

    assert status.state == "stopped"
    assert status.return_code == 0


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


def test_supervisor_manages_two_same_role_instances_independently(tmp_path: Path):
    supervisor = RoleSupervisor(log_dir=tmp_path)
    first = supervisor.start(
        "single-node-openai",
        [sys.executable, "-c", "import time; time.sleep(30)"],
        instance_id="instance-a",
        operation_id="operation-a",
    )
    second = supervisor.start(
        "single-node-openai",
        [sys.executable, "-c", "import time; time.sleep(30)"],
        instance_id="instance-b",
        operation_id="operation-b",
    )

    try:
        assert first.pid is not None and second.pid is not None and first.pid != second.pid
        supervisor.stop("single-node-openai", timeout=1, instance_id="instance-a")
        assert supervisor.status("single-node-openai", instance_id="instance-a").state == "stopped"
        assert supervisor.status("single-node-openai", instance_id="instance-b").state == "running"
    finally:
        supervisor.stop("single-node-openai", timeout=1, instance_id="instance-b")


def test_agent_restart_reports_unverified_process_as_orphan_instead_of_killing_it():
    class OrphanAwareSupervisor(_FakeSupervisor):
        def cleanup_orphaned_model_processes(self):
            return [
                {
                    "pid": 42,
                    "command": "python -m tokenity distributed-openai serve --model /models/qwen",
                    "state": "orphaned",
                    "reason": "unverified",
                }
            ]

    with TestClient(
        create_app(rdma_probe_fn=fake_rdma_probe, supervisor=OrphanAwareSupervisor())
    ) as client:
        payload = client.get("/v1/node/status").json()

    assert payload["orphaned_processes"][0]["state"] == "orphaned"
    assert payload["orphaned_processes"][0]["pid"] == 42


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
    assert plan["transport"] == "local-process"
    assert plan["execution_mode"] == "single"
    command = plan["ranks"][0]["command"]
    assert command[command.index("--api-identifier") + 1] == "tokenity/qwen"
    assert command[command.index("--max-tokens") + 1] == "65536"
    assert command[command.index("--prompt-cache-size") + 1] == "8"
    assert command[command.index("--prefill-step-size") + 1] == "4096"
    assert command[command.index("--decode-concurrency") + 1] == "2"
    assert command[command.index("--prompt-concurrency") + 1] == "3"
    assert "--trust-remote-code" in command


def test_model_scan_marks_qwen35_mtp_as_draft_only_and_start_rejects_it(tmp_path: Path):
    draft = tmp_path / "Qwen3.5-4B-MTP-4bit"
    draft.mkdir()
    (draft / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_mtp", "architectures": ["Qwen3_5MTPForCausalLM"]}),
        encoding="utf-8",
    )
    (draft / "model.safetensors").write_bytes(b"draft")

    scanned = scan_models(tmp_path)

    assert scanned[0]["model_type"] == "qwen3_5_mtp"
    assert scanned[0]["standalone_loadable"] is False
    assert "draft model" in str(scanned[0]["load_block_reason"])

    client = TestClient(create_app(rdma_probe_fn=fake_rdma_probe))
    response = client.post(
        "/v1/node/start-distributed-openai",
        json={"model": str(draft), "connection_mode": "ring", "dry_run": True},
    )

    assert response.status_code == 400
    assert "cannot be loaded as a standalone chat model" in response.json()["detail"]


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
    assert env["TOKENITY_MLX_LOAD_POLICY"] == "adaptive"
    assert env["TOKENITY_MLX_LOAD_ADAPTIVE_MAX_LEAVES"] == "64"
    assert env["TOKENITY_MLX_LOAD_ADAPTIVE_TARGET_BYTES"] == "268435456"
    assert env["MLX_JACCL_COORDINATOR"] == "192.168.0.1:30020"
    assert json.loads(Path(env["MLX_IBV_DEVICES"]).read_text()) == [
        [None, "rdma_en4"],
        ["rdma_en5", None],
    ]


def test_single_node_uses_local_runtime_without_distributed_environment():
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

    command, env = _rank_command_and_environment(request)

    assert "MLX_RANK" not in env
    assert "MLX_HOSTFILE" not in env
    assert "MLX_JACCL_COORDINATOR" not in env
    assert command[command.index("--execution-mode") + 1] == "single"


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
            runtime_preflight_fn=lambda *_: [],
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
    assert "native_mtp" not in posts[0][1]

    runtime = client.get("/v1/node/status").json()["cluster_runtime"]
    assert runtime["rank"] == 0
    assert runtime["world_size"] == 2
    assert runtime["connection_mode"] == "jaccl"
    assert runtime["role"] == "controller"

    stop = client.post(
        "/v1/node/stop-role",
        json={"role": "distributed-openai", "timeout": 5},
    )
    assert stop.status_code == 200
    assert any(url.endswith("/v1/node/request-stop-all") for url, _, _ in posts)
    assert posts[-1][0] == "http://192.168.5.75:9100/v1/node/stop-all"
    assert posts[-1][1]["timeout"] == 5
    assert client.get("/v1/node/status").json()["cluster_runtime"] is None


def test_instance_quorum_requires_matching_ready_rank_evidence(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    supervisor = _FakeSupervisor()
    worker_payloads = []

    def post_json(url, payload, timeout):
        worker_payloads.append(payload)
        return {"status": {"state": "running"}}

    def get_json(url, timeout):
        instance_id = worker_payloads[0]["instance_id"]
        return {
            "instance": {"instance_id": instance_id},
            "process": {"state": "running"},
            "runtime": {
                "instance_id": instance_id,
                "operation_id": worker_payloads[0]["operation_id"],
                "rank": 1,
                "world_size": 2,
                "connection_mode": "jaccl",
                "model_revision": None,
                "phase": "ready",
                "updated_at": time.time(),
                "tokenizer_identity": "Tokenizer",
                "ready_evidence": {
                    "weights_materialized": True,
                    "tokenizer_ready": True,
                    "generation_engine_ready": True,
                    "one_token_probe": False,
                    "warmup_cache_isolated": False,
                },
            },
        }

    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=supervisor,
            post_json_fn=post_json,
            get_json_fn=get_json,
            rank_ready_fn=lambda pid, port, timeout: True,
            rank_stabilize_fn=lambda seconds: None,
            rank_connected_fn=lambda pid, request, timeout: True,
            runtime_preflight_fn=lambda *_: [],
        )
    )
    start = client.post(
        "/v1/node/start-distributed-openai",
        json={
            "model": "/models/qwen",
            "connection_mode": "jaccl",
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
    assert start.status_code == 200
    instance_id = start.json()["instance_id"]
    operation_id = start.json()["operation_id"]
    local_path = agent_module._runtime_status_path(instance_id, 0)
    local_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "operation_id": operation_id,
                "rank": 0,
                "world_size": 2,
                "connection_mode": "jaccl",
                "model_revision": None,
                "phase": "ready",
                "updated_at": time.time(),
                "tokenizer_identity": "Tokenizer",
                "ready_evidence": {
                    "weights_materialized": True,
                    "tokenizer_ready": True,
                    "generation_engine_ready": True,
                    "one_token_probe": True,
                    "warmup_cache_isolated": True,
                },
            }
        ),
        encoding="utf-8",
    )

    quorum = client.get(f"/v1/node/instances/{instance_id}/quorum")

    assert quorum.status_code == 200
    assert quorum.json()["ready"] is True
    assert quorum.json()["rank_quorum"] == "2/2"
    assert quorum.json()["instance"]["state"] == "ready"


class _FakeGatewayConnection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeGatewayResponse:
    def __init__(self, body: bytes, *, content_type: str):
        self.status = 200
        self._body = body
        self._offset = 0
        self._content_type = content_type

    def getheader(self, name):
        return self._content_type if name.lower() == "content-type" else None

    def read1(self, size):
        if self._offset >= len(self._body):
            return b""
        end = min(len(self._body), self._offset + min(size, 13))
        result = self._body[self._offset:end]
        self._offset = end
        return result

    def read(self):
        result = self._body[self._offset:]
        self._offset = len(self._body)
        return result


def test_stable_gateway_routes_multiple_instances_and_releases_request_leases(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    supervisor = _FakeSupervisor()
    upstream_urls = []

    def gateway_open(url, body, headers, timeout):
        upstream_urls.append(url)
        payload = json.loads(body)
        if payload.get("stream"):
            body_bytes = b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n'
            content_type = "text/event-stream"
        else:
            body_bytes = b'{"choices":[{"message":{"role":"assistant","content":"OK"}}]}'
            content_type = "application/json"
        return _FakeGatewayConnection(), _FakeGatewayResponse(body_bytes, content_type=content_type)

    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=supervisor,
            runtime_preflight_fn=lambda *_: [],
            gateway_open_fn=gateway_open,
        )
    )

    def start_ready(instance_id: str, operation_id: str, port: int):
        response = client.post(
            "/v1/node/start-distributed-openai",
            json={
                "model": "/models/qwen",
                "api_identifier": "qwen-shared",
                "instance_id": instance_id,
                "operation_id": operation_id,
                "port": port,
                "starting_port": port + 1_000,
                "connection_mode": "ring",
                "dry_run": False,
                "nodes": [
                    {
                        "id": "mac-a",
                        "agent_url": "http://127.0.0.1:9100",
                        "lan_ip": "127.0.0.1",
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        runtime_path = agent_module._runtime_status_path(instance_id, 0)
        runtime_path.write_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "operation_id": operation_id,
                    "rank": 0,
                    "world_size": 1,
                    "connection_mode": "single",
                    "model_revision": None,
                    "phase": "ready",
                    "updated_at": time.time(),
                    "tokenizer_identity": "Tokenizer",
                    "ready_evidence": {
                        "weights_materialized": True,
                        "tokenizer_ready": True,
                        "generation_engine_ready": True,
                        "one_token_probe": True,
                        "warmup_cache_isolated": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        quorum = client.get(f"/v1/node/instances/{instance_id}/quorum")
        assert quorum.status_code == 200 and quorum.json()["ready"] is True
        return response.json()["instance"]["http_port"]

    first_port = start_ready("instance-a", "operation-a", 18_000)
    second_port = start_ready("instance-b", "operation-b", 18_000)

    first = client.post(
        "/v1/chat/completions",
        json={"model": "qwen-shared", "messages": [], "stream": True},
    )
    second = client.post(
        "/v1/chat/completions",
        json={"model": "qwen-shared", "messages": [], "stream": False},
    )

    assert first.status_code == 200
    assert first.headers["x-tokenity-instance-id"] == "instance-a"
    assert first.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in first.text
    assert second.status_code == 200
    assert second.headers["x-tokenity-instance-id"] == "instance-b"
    assert second.json()["choices"][0]["message"]["content"] == "OK"
    assert upstream_urls == [
        f"http://127.0.0.1:{first_port}/v1/chat/completions",
        f"http://127.0.0.1:{second_port}/v1/chat/completions",
    ]
    routes = client.get("/v1/gateway/routes").json()["data"]
    assert {item["instance_id"] for item in routes} == {"instance-a", "instance-b"}
    assert all(item["active_request_count"] == 0 for item in routes)

    runtime_status = client.get("/v1/node/status").json()
    assert {
        item["instance_id"] for item in runtime_status["cluster_runtimes"]
    } == {"instance-a", "instance-b"}
    assert runtime_status["cluster_runtime"]["instance_id"] == "instance-b"

    stopped = client.post("/v1/node/instances/instance-b/stop", json={"timeout": 1})
    assert stopped.status_code == 200
    runtime_status = client.get("/v1/node/status").json()
    assert [
        item["instance_id"] for item in runtime_status["cluster_runtimes"]
    ] == ["instance-a"]
    assert runtime_status["cluster_runtime"]["instance_id"] == "instance-a"
    remaining = client.post(
        "/v1/chat/completions",
        json={"model": "qwen-shared", "messages": [], "stream": False},
    )
    assert remaining.status_code == 200
    assert remaining.headers["x-tokenity-instance-id"] == "instance-a"


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


def test_remote_rank_preflight_rejects_mismatched_tokenity_code_revision(monkeypatch):
    monkeypatch.setattr(agent_module, "_tokenity_code_revision", lambda: "local-revision")
    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            runtime_preflight_fn=lambda *_: [],
        )
    )
    response = client.post(
        "/v1/node/start-distributed-rank",
        json={
            "cluster_id": "cluster-test",
            "model": "/models/qwen",
            "rank": 1,
            "world_size": 2,
            "connection_mode": "ring",
            "python": sys.executable,
            "ring_hosts": [["127.0.0.1:29500"], ["127.0.0.1:29501"]],
            "tokenity_code_revision": "remote-revision",
        },
    )

    assert response.status_code == 412
    issues = response.json()["detail"]["issues"]
    assert any("code revision mismatch" in issue for issue in issues)


def test_remote_rank_rolls_back_when_the_data_plane_does_not_connect():
    supervisor = _FakeSupervisor()
    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=supervisor,
            rank_connected_fn=lambda pid, request, timeout: False,
            runtime_preflight_fn=lambda *_: [],
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
