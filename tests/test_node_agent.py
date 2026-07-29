from __future__ import annotations

import asyncio
import platform
import subprocess
import sys
import time
import json
from pathlib import Path

import pytest
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


def test_direct_jaccl_is_upgraded_to_ring_fallback_for_all_models(tmp_path):
    glm = tmp_path / "GLM-5.2-mxfp4"
    qwen = tmp_path / "Qwen3.5-122B"
    glm.mkdir()
    qwen.mkdir()
    (glm / "config.json").write_text(
        json.dumps({"model_type": "glm_moe_dsa"}),
        encoding="utf-8",
    )
    (qwen / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe"}),
        encoding="utf-8",
    )

    assert agent_module._stable_connection_mode_for_model(  # noqa: SLF001
        str(glm),
        agent_module.ConnectionMode.JACCL,
    ) == agent_module.ConnectionMode.JACCL_RING
    assert agent_module._stable_connection_mode_for_model(  # noqa: SLF001
        str(qwen),
        agent_module.ConnectionMode.JACCL,
    ) == agent_module.ConnectionMode.JACCL_RING
    assert agent_module._stable_connection_mode_for_model(  # noqa: SLF001
        str(glm),
        agent_module.ConnectionMode.RING,
    ) == agent_module.ConnectionMode.RING


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


def test_node_info_reports_pinned_runtime_identity(tmp_path, monkeypatch):
    manifest = {
        "runtime_id": "runtime-test",
        "architecture": "arm64",
        "minimum_macos": "26.2",
        "python_version": platform.python_version(),
        "packages": {
            "mlx": agent_module._package_version("mlx"),  # noqa: SLF001
            "mlx-lm": agent_module._package_version("mlx-lm"),  # noqa: SLF001
        },
        "payload": {"tree_sha256": "a" * 64},
    }
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("TOKENITY_RUNTIME_MANIFEST", str(manifest_path))
    monkeypatch.setattr(agent_module.platform, "machine", lambda: "arm64")

    client = TestClient(create_app(rdma_probe_fn=fake_rdma_probe))
    runtime = client.get("/v1/node/info").json()["runtime"]

    assert runtime["runtime_id"] == "runtime-test"
    assert runtime["payload_sha256"] == "a" * 64
    assert runtime["install_state"] == "ready"
    assert len(runtime["manifest_sha256"]) == 64


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


def test_start_rejects_model_when_live_wired_memory_exhausts_headroom(
    monkeypatch,
):
    gib = 1_073_741_824
    supervisor = _FakeSupervisor()
    monkeypatch.setattr(agent_module, "_total_memory_bytes", lambda: 512 * gib)
    monkeypatch.setattr(
        agent_module,
        "_memory_stats",
        lambda: {
            "total_bytes": 512 * gib,
            "in_use_bytes": 443 * gib,
            "pressure_available_ratio": 0.19,
        },
    )
    monkeypatch.setattr(agent_module, "_tcp_port_available", lambda port: True)

    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=supervisor,
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
            "connection_mode": "ring",
            "dry_run": False,
            "memory_reservation_bytes": 20 * gib,
            "nodes": [
                {
                    "id": "local",
                    "agent_url": "http://127.0.0.1:9100",
                    "lan_ip": "127.0.0.1",
                }
            ],
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["stage"] == "resource_admission"
    assert supervisor.starts == []


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


@pytest.mark.parametrize(
    ("model", "expected_events"),
    [
        (
            "/models/glm",
            [("wait", "distributed-openai-rank", "coordinated-rank-stop")],
        ),
        (
            "/models/qwen",
            [("wait", "distributed-openai-rank", "coordinated-rank-stop")],
        ),
    ],
)
def test_worker_instance_stop_waits_for_validated_jaccl_stop_sentinel(
    monkeypatch,
    model,
    expected_events,
):
    events = []

    class CoordinatedExitSupervisor:
        running = True

        def start(
            self,
            role,
            command,
            env=None,
            cwd=None,
            start_new_session=True,
            instance_id=None,
            operation_id=None,
        ):
            self.running = True
            return RoleStatus(
                role=role,
                state="running",
                instance_id=instance_id,
                operation_id=operation_id,
                pid=123,
                command=command,
            )

        def status(self, role=None, instance_id=None):
            return RoleStatus(
                role=role or "distributed-openai-rank",
                state="running" if self.running else "stopped",
                instance_id=instance_id,
                pid=123 if self.running else None,
                return_code=None if self.running else 0,
            )

        def wait(self, role, timeout=10, instance_id=None):
            events.append(("wait", role, instance_id))
            self.running = False
            return self.status(role, instance_id=instance_id)

        def request_stop(self, role, instance_id=None):
            events.append(("request_stop", role, instance_id))
            self.running = False
            return self.status(role, instance_id=instance_id)

        def stop(self, role, timeout=10, instance_id=None):
            events.append(("stop", role, instance_id))
            self.running = False
            return self.status(role, instance_id=instance_id)

    monkeypatch.setattr(agent_module, "_post_json", lambda *_: None)
    supervisor = CoordinatedExitSupervisor()
    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=supervisor,
            rank_connected_fn=lambda pid, request, timeout: True,
            runtime_preflight_fn=lambda *_: [],
        )
    )
    started = client.post(
        "/v1/node/start-distributed-rank",
        json={
            "cluster_id": "coordinated-rank-stop",
            "instance_id": "coordinated-rank-stop",
            "operation_id": "coordinated-rank-stop-op",
            "model": model,
            "rank": 1,
            "world_size": 2,
            "connection_mode": "jaccl-ring",
            "python": sys.executable,
            "coordinator_ip": "127.0.0.1",
            "starting_port": 30_200,
            "ring_hosts": [
                ["127.0.0.1:30200", "127.0.0.1:30201"],
                ["127.0.0.1:30201", "127.0.0.1:30200"],
            ],
            "rdma_matrix": [
                [None, "rdma_en4"],
                ["rdma_en5", None],
            ],
        },
    )
    assert started.status_code == 200, started.text

    stopped = client.post(
        "/v1/node/instances/coordinated-rank-stop/stop",
        json={"timeout": 5},
    )

    assert stopped.status_code == 200, stopped.text
    assert events == expected_events


def test_coordinator_instance_stop_waits_for_admin_requested_natural_exit(
    monkeypatch,
):
    events = []

    class NaturalExitSupervisor:
        running = True

        def start(
            self,
            role,
            command,
            env=None,
            cwd=None,
            start_new_session=True,
            instance_id=None,
            operation_id=None,
        ):
            self.running = True
            return RoleStatus(
                role=role,
                state="running",
                instance_id=instance_id,
                operation_id=operation_id,
                pid=123,
                command=command,
            )

        def status(self, role=None, instance_id=None):
            return RoleStatus(
                role=role or "single-node-openai",
                state="running" if self.running else "stopped",
                instance_id=instance_id,
                pid=123 if self.running else None,
                return_code=None if self.running else 0,
            )

        def wait(self, role, timeout=10, instance_id=None):
            events.append(("wait", role, instance_id))
            self.running = False
            return self.status(role, instance_id=instance_id)

        def request_stop(self, role, instance_id=None):
            events.append(("request_stop", role, instance_id))
            self.running = False
            return self.status(role, instance_id=instance_id)

        def stop(self, role, timeout=10, instance_id=None):
            events.append(("stop", role, instance_id))
            self.running = False
            return self.status(role, instance_id=instance_id)

    monkeypatch.setattr(agent_module, "_post_json", lambda *_: {})
    supervisor = NaturalExitSupervisor()
    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=supervisor,
            rank_ready_fn=lambda pid, port, timeout: True,
            rank_stabilize_fn=lambda seconds: None,
            runtime_preflight_fn=lambda *_: [],
        )
    )
    _start_ready_gateway_model(
        client,
        model_id="coordinator-natural-stop",
        instance_id="coordinator-natural-stop",
        operation_id="coordinator-natural-stop-op",
        port=23_700,
    )

    stopped = client.post(
        "/v1/node/instances/coordinator-natural-stop/stop",
        json={"timeout": 5},
    )

    assert stopped.status_code == 200, stopped.text
    assert events == [
        ("wait", "single-node-openai", "coordinator-natural-stop")
    ]


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
    assert env["TOKENITY_MLX_LOAD_POST_BARRIER"] == "1"
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
    assert env["TOKENITY_MLX_LOAD_POST_BARRIER"] == "0"
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


def test_empty_agent_shutdown_never_probes_a_default_model_port(
    tmp_path: Path,
    monkeypatch,
):
    stop_requests = []
    monkeypatch.setattr(
        agent_module,
        "_post_json",
        lambda url, payload, timeout: stop_requests.append((url, payload, timeout)),
    )

    app = create_app(
        rdma_probe_fn=fake_rdma_probe,
        supervisor=RoleSupervisor(log_dir=tmp_path),
    )
    with TestClient(app):
        pass

    assert stop_requests == []


def test_shutdown_requests_stop_only_for_a_running_owned_coordinator(
    tmp_path: Path,
    monkeypatch,
):
    stop_requests = []
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(agent_module, "_tcp_port_available", lambda port: True)
    monkeypatch.setattr(
        agent_module,
        "_post_json",
        lambda url, payload, timeout: stop_requests.append((url, payload, timeout)),
    )

    app = create_app(
        rdma_probe_fn=fake_rdma_probe,
        supervisor=_FakeSupervisor(),
        rank_ready_fn=lambda pid, port, timeout: True,
        rank_stabilize_fn=lambda seconds: None,
        rank_connected_fn=lambda pid, request, timeout: True,
        runtime_preflight_fn=lambda *_: [],
    )
    with TestClient(app) as client:
        port = _start_ready_gateway_model(
            client,
            model_id="shutdown-owned-model",
            instance_id="instance-shutdown-owned",
            operation_id="operation-shutdown-owned",
            port=23_500,
        )

    assert stop_requests == [
        (f"http://127.0.0.1:{port}/v1/tokenity/stop", {}, 2.0)
    ]


def test_shutdown_does_not_stop_a_registered_but_exited_coordinator(
    tmp_path: Path,
    monkeypatch,
):
    class ExitableSupervisor(_FakeSupervisor):
        running = True

        def status(self, role=None):
            if role is None:
                return []
            if not self.running:
                return RoleStatus(
                    role=role,
                    state="stopped",
                    return_code=0,
                )
            return super().status(role)

    supervisor = ExitableSupervisor()
    stop_requests = []
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(agent_module, "_tcp_port_available", lambda port: True)
    monkeypatch.setattr(
        agent_module,
        "_post_json",
        lambda url, payload, timeout: stop_requests.append((url, payload, timeout)),
    )

    app = create_app(
        rdma_probe_fn=fake_rdma_probe,
        supervisor=supervisor,
        rank_ready_fn=lambda pid, port, timeout: True,
        rank_stabilize_fn=lambda seconds: None,
        rank_connected_fn=lambda pid, request, timeout: True,
        runtime_preflight_fn=lambda *_: [],
    )
    with TestClient(app) as client:
        _start_ready_gateway_model(
            client,
            model_id="shutdown-exited-model",
            instance_id="instance-shutdown-exited",
            operation_id="operation-shutdown-exited",
            port=23_600,
        )
        supervisor.running = False

    assert stop_requests == []


def _managed_agent_info():
    return {
        "agent_contract": {
            "version": agent_module.NODE_AGENT_CONTRACT_VERSION,
            "capabilities": list(agent_module.NODE_AGENT_CAPABILITIES),
        }
    }


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
            get_json_fn=lambda url, timeout: _managed_agent_info(),
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
            "lease_seconds": 300,
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

    initial_deadline = response.json()["instance"]["deadline"]
    heartbeat = client.post(
        "/v1/node/heartbeat",
        json={
            "ttl_seconds": 30,
            "instance_id": response.json()["instance_id"],
        },
    )
    assert heartbeat.status_code == 200, heartbeat.text
    assert heartbeat.json()["deadline_epoch"] >= initial_deadline
    assert heartbeat.json()["worker_renewals"][0]["status"] == "ok"
    assert posts[1][0] == "http://192.168.5.75:9100/v1/node/heartbeat"
    assert posts[1][1]["ttl_seconds"] == 30

    runtime = client.get("/v1/node/status").json()["cluster_runtime"]
    assert runtime["rank"] == 0
    assert runtime["world_size"] == 2
    assert runtime["connection_mode"] == "jaccl-ring"
    assert supervisor.starts[0][2]["MLX_JACCL_RING"] == "1"
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


@pytest.mark.parametrize(
    "missing_capability",
    sorted(agent_module.MANAGED_INSTANCE_CAPABILITIES),
)
def test_distributed_start_requires_managed_instance_capabilities_without_side_effects(
    missing_capability,
):
    supervisor = _FakeSupervisor()
    posts = []

    def get_json(url, timeout):
        info = _managed_agent_info()
        info["agent_contract"]["capabilities"].remove(missing_capability)
        return info

    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=supervisor,
            post_json_fn=lambda url, payload, timeout: posts.append((url, payload, timeout)),
            get_json_fn=get_json,
            runtime_preflight_fn=lambda *_: [],
        )
    )

    response = client.post(
        "/v1/node/start-distributed-openai",
        json={
            "model": "/models/qwen",
            "instance_id": "instance-capability",
            "operation_id": "operation-capability",
            "connection_mode": "ring",
            "dry_run": False,
            "nodes": [
                {
                    "id": "mac-a",
                    "agent_url": "http://192.168.5.23:9100",
                    "lan_ip": "192.168.5.23",
                },
                {
                    "id": "mac-b",
                    "agent_url": "http://192.168.5.75:9100",
                    "lan_ip": "192.168.5.75",
                },
            ],
        },
    )

    assert response.status_code == 412
    assert response.json()["detail"]["stage"] == "agent_capabilities"
    assert response.json()["detail"]["issues"][0]["missing_capabilities"] == [
        missing_capability
    ]
    assert supervisor.starts == []
    assert supervisor.stops == []
    assert posts == []


def test_distributed_start_rollback_never_falls_back_to_remote_stop_all(monkeypatch):
    monkeypatch.setattr(agent_module, "_tcp_port_available", lambda port: True)
    supervisor = _FakeSupervisor()
    posts = []

    def post_json(url, payload, timeout):
        posts.append((url, payload, timeout))
        if url.endswith("192.168.5.76:9100/v1/node/start-distributed-rank"):
            raise RuntimeError("rank start failed")
        if "/v1/node/instances/instance-rollback/stop" in url:
            raise RuntimeError("instance stop unavailable")
        return {"status": {"state": "running"}}

    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=supervisor,
            post_json_fn=post_json,
            get_json_fn=lambda url, timeout: _managed_agent_info(),
            runtime_preflight_fn=lambda *_: [],
        )
    )

    response = client.post(
        "/v1/node/start-distributed-openai",
        json={
            "model": "/models/qwen",
            "instance_id": "instance-rollback",
            "operation_id": "operation-rollback",
            "connection_mode": "ring",
            "dry_run": False,
            "nodes": [
                {
                    "id": "mac-a",
                    "agent_url": "http://192.168.5.23:9100",
                    "lan_ip": "192.168.5.23",
                },
                {
                    "id": "mac-b",
                    "agent_url": "http://192.168.5.75:9100",
                    "lan_ip": "192.168.5.75",
                },
                {
                    "id": "mac-c",
                    "agent_url": "http://192.168.5.76:9100",
                    "lan_ip": "192.168.5.76",
                },
            ],
        },
    )

    assert response.status_code == 502
    assert any(
        url.endswith("/v1/node/instances/instance-rollback/stop")
        for url, _, _ in posts
    )
    assert not any(url.endswith("/v1/node/stop-all") for url, _, _ in posts)


def test_global_stop_fans_out_to_union_of_instance_workers(monkeypatch):
    monkeypatch.setattr(agent_module, "_tcp_port_available", lambda port: True)
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
            get_json_fn=lambda url, timeout: _managed_agent_info(),
            runtime_preflight_fn=lambda *_: [],
        )
    )

    for instance_id, operation_id, worker_ip in (
        ("instance-union-a", "operation-union-a", "192.168.5.75"),
        ("instance-union-b", "operation-union-b", "192.168.5.76"),
    ):
        response = client.post(
            "/v1/node/start-distributed-openai",
            json={
                "model": "/models/qwen",
                "instance_id": instance_id,
                "operation_id": operation_id,
                "connection_mode": "ring",
                "dry_run": False,
                "nodes": [
                    {
                        "id": "mac-a",
                        "agent_url": "http://192.168.5.23:9100",
                        "lan_ip": "192.168.5.23",
                    },
                    {
                        "id": f"worker-{instance_id}",
                        "agent_url": f"http://{worker_ip}:9100",
                        "lan_ip": worker_ip,
                    },
                ],
            },
        )
        assert response.status_code == 200, response.text

    stop = client.post("/v1/node/stop-all", json={"timeout": 1})

    assert stop.status_code == 200
    request_stop_hosts = {
        url.split("/v1/", 1)[0]
        for url, _, _ in posts
        if url.endswith("/v1/node/request-stop-all")
    }
    stop_all_hosts = {
        url.split("/v1/", 1)[0]
        for url, _, _ in posts
        if url.endswith("/v1/node/stop-all")
    }
    assert request_stop_hosts == {
        "http://192.168.5.75:9100",
        "http://192.168.5.76:9100",
    }
    assert stop_all_hosts == request_stop_hosts


def test_instance_quorum_requires_matching_ready_rank_evidence(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    supervisor = _FakeSupervisor()
    worker_payloads = []

    def post_json(url, payload, timeout):
        worker_payloads.append(payload)
        return {"status": {"state": "running"}}

    def get_json(url, timeout):
        if url.endswith("/v1/node/info"):
            return _managed_agent_info()
        instance_id = worker_payloads[0]["instance_id"]
        return {
            "instance": {"instance_id": instance_id},
            "process": {"state": "running"},
            "runtime": {
                "instance_id": instance_id,
                "operation_id": worker_payloads[0]["operation_id"],
                "rank": 1,
                "world_size": 2,
                "connection_mode": "jaccl-ring",
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
                "connection_mode": "jaccl-ring",
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
        self.close_count = 0

    def close(self):
        self.closed = True
        self.close_count += 1


class _FakeGatewayResponse:
    def __init__(self, body: bytes, *, content_type: str, status: int = 200, fail_read1=False):
        self.status = status
        self._body = body
        self._offset = 0
        self._content_type = content_type
        self._fail_read1 = fail_read1

    def getheader(self, name):
        return self._content_type if name.lower() == "content-type" else None

    def read1(self, size):
        if self._fail_read1:
            raise RuntimeError("stream read failed")
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


def _start_ready_gateway_model(
    client,
    *,
    model_id: str,
    instance_id: str,
    operation_id: str,
    port: int,
    model_path: str | None = None,
    model_revision: str | None = None,
):
    response = client.post(
        "/v1/node/start-distributed-openai",
        json={
            "model": model_path or f"/models/{model_id}",
            "api_identifier": model_id,
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
                "model_revision": model_revision,
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


def test_stable_gateway_routes_multiple_instances_and_releases_request_leases(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    supervisor = _FakeSupervisor()
    upstream_urls = []
    upstream_payloads = []

    def gateway_open(url, body, headers, timeout):
        upstream_urls.append(url)
        payload = json.loads(body)
        upstream_payloads.append(payload)
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
                    "memory": {
                        "mlx_active_bytes": 800,
                        "mlx_peak_bytes": 1_000,
                        "mlx_cache_bytes": 100,
                        "process_phys_footprint_bytes": 900,
                    },
                }
            ),
            encoding="utf-8",
        )
        quorum = client.get(f"/v1/node/instances/{instance_id}/quorum")
        assert quorum.status_code == 200 and quorum.json()["ready"] is True
        assert quorum.json()["instance"]["memory_reservation_bytes"] == 1_100
        ledger = client.get("/v1/node/instances").json()["resource_ledger"]
        assert ledger["reservations"][instance_id] == 1_100
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
    assert all(item["model_revision"] == "unversioned" for item in routes)
    assert all(
        item["capabilities"]
        == {
            "tools": True,
            "json": True,
            "thinking": False,
            "modalities": ["text"],
            "task_tags": ["general"],
        }
        for item in routes
    )
    assert all(item["warm_ttft_p50_ms"] == 300.0 for item in routes)
    assert all(item["warm_ttft_p95_ms"] == 450.0 for item in routes)
    assert "messages" not in json.dumps(routes)
    assert "prompt" not in json.dumps(routes).lower()
    models = client.get("/v1/models").json()["data"]
    assert [item["id"] for item in models] == ["tokenity-auto", "qwen-shared"]
    assert models[1]["tokenity_instance_ids"] == ["instance-a", "instance-b"]

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

    exact = client.post(
        "/v1/chat/completions",
        json={"model": "instance-a", "messages": [], "stream": False},
    )
    assert exact.status_code == 200
    assert exact.headers["x-tokenity-instance-id"] == "instance-a"
    assert upstream_payloads[-1]["model"] == "qwen-shared"


def test_auto_decision_chat_headers_defaults_private_fields_and_session_sticky(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    upstream_payloads = []

    def gateway_open(url, body, headers, timeout):
        upstream_payloads.append(json.loads(body))
        return (
            _FakeGatewayConnection(),
            _FakeGatewayResponse(
                b'{"choices":[{"message":{"role":"assistant","content":"OK"}}]}',
                content_type="application/json",
            ),
        )

    profiles = [
        agent_module.ModelCapabilityProfile(
            model_id="fast-model",
            revision="unversioned",
            task_tags=frozenset({"general", "fast-chat"}),
            task_quality={"general": 0.75, "coding": 0.4},
            warm_ttft_p95_ms=20,
            sampling_defaults={"temperature": 0.8, "top_p": 0.9},
        ),
        agent_module.ModelCapabilityProfile(
            model_id="coder-model",
            revision="unversioned",
            task_tags=frozenset({"general", "coding", "tool-use"}),
            task_quality={"general": 0.8, "coding": 0.99, "tool-use": 0.98},
            supports_tools=True,
            supports_json=True,
            warm_ttft_p95_ms=250,
            sampling_defaults={"temperature": 0.2, "top_p": 0.95},
            chat_template_defaults={"enable_thinking": False},
        ),
    ]
    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=_FakeSupervisor(),
            runtime_preflight_fn=lambda *_: [],
            gateway_open_fn=gateway_open,
            capability_profiles=profiles,
        )
    )
    _start_ready_gateway_model(
        client,
        model_id="fast-model",
        instance_id="instance-auto-fast",
        operation_id="operation-auto-fast",
        port=20_000,
    )
    _start_ready_gateway_model(
        client,
        model_id="coder-model",
        instance_id="instance-auto-coder",
        operation_id="operation-auto-coder",
        port=20_010,
    )
    secret_prompt = "PRIVATE-7391: fix this Python compiler error"
    request_payload = {
        "model": "tokenity-auto",
        "messages": [{"role": "user", "content": secret_prompt}],
        "stream": False,
        "top_p": 0.5,
        "tokenity_route_policy": "balanced",
        "tokenity_session_id": "session-auto",
        "tokenity_lock_model": False,
        "tokenity_constraints": {"allowed_model_ids": ["fast-model", "coder-model"]},
    }

    decision = client.post("/v1/router/decision", json=request_payload)

    assert decision.status_code == 200, decision.text
    assert decision.json()["model_id"] == "coder-model"
    assert decision.json()["selected_instance_id"] == "instance-auto-coder"
    assert decision.json()["derived_features"]["has_compiler_error"] is True
    assert secret_prompt not in decision.text
    assert upstream_payloads == []

    response = client.post("/v1/chat/completions", json=request_payload)

    assert response.status_code == 200, response.text
    assert response.headers["x-tokenity-routed-model"] == "coder-model"
    assert response.headers["x-tokenity-instance-id"] == "instance-auto-coder"
    assert response.headers["x-tokenity-model-revision"] == "unversioned"
    assert response.headers["x-tokenity-route-reason"] == "deterministic_rule"
    assert float(response.headers["x-tokenity-route-confidence"]) > 0
    assert float(response.headers["x-tokenity-routing-latency-ms"]) >= 0
    assert response.headers["x-tokenity-model"] == "coder-model"
    forwarded = upstream_payloads[-1]
    assert forwarded["model"] == "coder-model"
    assert forwarded["temperature"] == 0.2
    assert forwarded["top_p"] == 0.5
    assert forwarded["chat_template_kwargs"] == {"enable_thinking": False}
    assert not agent_module.TOKENITY_ROUTE_FIELDS.intersection(forwarded)

    sticky = client.post(
        "/v1/chat/completions",
        json={
            "model": "tokenity-auto",
            "messages": [{"role": "user", "content": "hello"}],
            "tokenity_route_policy": "fast",
            "tokenity_session_id": "session-auto",
        },
    )
    assert sticky.status_code == 200, sticky.text
    assert sticky.headers["x-tokenity-routed-model"] == "coder-model"
    assert sticky.headers["x-tokenity-route-reason"] == "session_sticky"
    assert upstream_payloads[-1]["model"] == "coder-model"

    constrained = client.post(
        "/v1/router/decision",
        json={
            "model": "tokenity-auto",
            "messages": [{"role": "user", "content": "call a tool"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        },
    )
    assert constrained.status_code == 200
    assert constrained.json()["model_id"] == "coder-model"
    assert constrained.json()["derived_features"]["requires_tools"] is True


def test_auto_allowed_instance_ids_constrain_primary_and_retry_routes(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    attempted_urls = []

    def gateway_open(url, body, headers, timeout):
        attempted_urls.append(url)
        raise ConnectionError("allowed replica unavailable")

    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=_FakeSupervisor(),
            runtime_preflight_fn=lambda *_: [],
            gateway_open_fn=gateway_open,
        )
    )
    _start_ready_gateway_model(
        client,
        model_id="shared-model",
        instance_id="instance-disabled",
        operation_id="operation-disabled",
        port=20_000,
    )
    _start_ready_gateway_model(
        client,
        model_id="shared-model",
        instance_id="instance-allowed",
        operation_id="operation-allowed",
        port=20_010,
    )
    payload = {
        "model": "tokenity-auto",
        "messages": [{"role": "user", "content": "route this request"}],
        "tokenity_constraints": {
            "allowed_model_ids": ["shared-model"],
            "allowed_instance_ids": ["instance-allowed"],
        },
    }

    decision = client.post("/v1/router/decision", json=payload)

    assert decision.status_code == 200, decision.text
    assert decision.json()["selected_instance_id"] == "instance-allowed"

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 502, response.text
    assert attempted_urls == [
        "http://127.0.0.1:20010/v1/chat/completions",
    ]

    invalid = client.post(
        "/v1/router/decision",
        json={
            **payload,
            "tokenity_constraints": {"allowed_instance_ids": "instance-allowed"},
        },
    )
    assert invalid.status_code == 422
    assert "allowed_instance_ids" in invalid.text


def test_auto_allowed_instance_ids_filter_models_before_scoring(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    attempted_urls = []

    def gateway_open(url, body, headers, timeout):
        attempted_urls.append(url)
        return (
            _FakeGatewayConnection(),
            _FakeGatewayResponse(b'{"choices":[]}', content_type="application/json"),
        )

    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=_FakeSupervisor(),
            runtime_preflight_fn=lambda *_: [],
            gateway_open_fn=gateway_open,
        )
    )
    _start_ready_gateway_model(
        client,
        model_id="model-a",
        instance_id="instance-a",
        operation_id="operation-a",
        port=20_000,
    )
    _start_ready_gateway_model(
        client,
        model_id="model-b",
        instance_id="instance-b",
        operation_id="operation-b",
        port=20_010,
    )
    payload = {
        "model": "tokenity-auto",
        "messages": [{"role": "user", "content": "route only to model b"}],
        "tokenity_constraints": {"allowed_instance_ids": ["instance-b"]},
    }

    decision = client.post("/v1/router/decision", json=payload)

    assert decision.status_code == 200, decision.text
    assert decision.json()["model_id"] == "model-b"
    assert decision.json()["selected_instance_id"] == "instance-b"
    assert [candidate["model_id"] for candidate in decision.json()["candidates"]] == [
        "model-b"
    ]

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200, response.text
    assert response.headers["x-tokenity-routed-model"] == "model-b"
    assert attempted_urls == ["http://127.0.0.1:20010/v1/chat/completions"]


def test_auto_routes_and_sticks_to_the_exact_capable_revision(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        agent_module,
        "_model_revision",
        lambda model: (
            "revision-tool-capable"
            if model.endswith("tool-capable")
            else "revision-tool-incapable"
        ),
    )
    opened = []

    def gateway_open(url, body, headers, timeout):
        opened.append((url, json.loads(body)))
        return (
            _FakeGatewayConnection(),
            _FakeGatewayResponse(b'{"choices":[]}', content_type="application/json"),
        )

    profiles = [
        agent_module.ModelCapabilityProfile(
            model_id="shared-model",
            revision="revision-tool-incapable",
            task_tags=frozenset({"general"}),
            supports_tools=False,
            task_quality={"general": 0.99, "tool-use": 0.99},
        ),
        agent_module.ModelCapabilityProfile(
            model_id="shared-model",
            revision="revision-tool-capable",
            task_tags=frozenset({"general", "tool-use"}),
            supports_tools=True,
            task_quality={"general": 0.7, "tool-use": 0.9},
        ),
    ]
    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=_FakeSupervisor(),
            runtime_preflight_fn=lambda *_: [],
            gateway_open_fn=gateway_open,
            capability_profiles=profiles,
        )
    )
    _start_ready_gateway_model(
        client,
        model_id="shared-model",
        model_path="/models/tool-incapable",
        model_revision="revision-tool-incapable",
        instance_id="instance-revision-a-old",
        operation_id="operation-revision-old",
        port=20_100,
    )
    _start_ready_gateway_model(
        client,
        model_id="shared-model",
        model_path="/models/tool-capable",
        model_revision="revision-tool-capable",
        instance_id="instance-revision-z-new",
        operation_id="operation-revision-new",
        port=20_110,
    )
    request_payload = {
        "model": "tokenity-auto",
        "messages": [{"role": "user", "content": "call a tool"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tokenity_session_id": "revision-session",
    }

    decision = client.post("/v1/router/decision", json=request_payload)
    response = client.post("/v1/chat/completions", json=request_payload)
    sticky = client.post(
        "/v1/chat/completions",
        json={
            "model": "tokenity-auto",
            "messages": [{"role": "user", "content": "continue"}],
            "tokenity_session_id": "revision-session",
        },
    )

    assert decision.status_code == 200, decision.text
    assert decision.json()["revision"] == "revision-tool-capable"
    assert decision.json()["selected_instance_id"] == "instance-revision-z-new"
    assert response.status_code == 200, response.text
    assert response.headers["x-tokenity-instance-id"] == "instance-revision-z-new"
    assert response.headers["x-tokenity-model-revision"] == "revision-tool-capable"
    assert sticky.status_code == 200, sticky.text
    assert sticky.headers["x-tokenity-instance-id"] == "instance-revision-z-new"
    assert sticky.headers["x-tokenity-route-reason"] == "session_sticky"
    assert all("20100" not in url for url, _ in opened)


def test_auto_gateway_enforces_profile_runtime_parameter_support(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    profiles = [
        agent_module.ModelCapabilityProfile(
            model_id="no-effort",
            revision="unversioned",
            task_quality={"general": 0.99},
            supported_runtime_parameters=frozenset(),
        ),
        agent_module.ModelCapabilityProfile(
            model_id="effort-capable",
            revision="unversioned",
            task_quality={"general": 0.7},
            supported_runtime_parameters=frozenset({"reasoning_effort"}),
        ),
    ]
    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=_FakeSupervisor(),
            runtime_preflight_fn=lambda *_: [],
            capability_profiles=profiles,
        )
    )
    for model_id, instance_id, port in (
        ("no-effort", "instance-no-effort", 20_200),
        ("effort-capable", "instance-effort-capable", 20_210),
    ):
        _start_ready_gateway_model(
            client,
            model_id=model_id,
            instance_id=instance_id,
            operation_id=f"operation-{instance_id}",
            port=port,
        )

    decision = client.post(
        "/v1/router/decision",
        json={
            "model": "tokenity-auto",
            "messages": [],
            "reasoning_effort": "high",
        },
    )

    assert decision.status_code == 200, decision.text
    assert decision.json()["model_id"] == "effort-capable"
    rejected = next(
        candidate
        for candidate in decision.json()["candidates"]
        if candidate["model_id"] == "no-effort"
    )
    assert rejected["reasons"] == [
        "runtime_parameter_unsupported:reasoning_effort"
    ]


def test_auto_open_fallback_releases_each_attempt_once_and_never_uses_wrong_model(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    attempts = []
    release_counts = {}
    original_release = agent_module.InstanceRegistry.release_request_lease

    def release_once(self, instance_id):
        release_counts[instance_id] = release_counts.get(instance_id, 0) + 1
        return original_release(self, instance_id)

    monkeypatch.setattr(
        agent_module.InstanceRegistry,
        "release_request_lease",
        release_once,
    )

    def gateway_open(url, body, headers, timeout):
        attempts.append((url, json.loads(body)))
        if len(attempts) <= 2:
            raise ConnectionError("runtime unavailable")
        return (
            _FakeGatewayConnection(),
            _FakeGatewayResponse(b'{"choices":[]}', content_type="application/json"),
        )

    profiles = [
        agent_module.ModelCapabilityProfile(
            model_id="coder-primary",
            revision="unversioned",
            task_tags=frozenset({"general", "coding"}),
            task_quality={"coding": 0.99},
            warm_ttft_p95_ms=100,
        ),
        agent_module.ModelCapabilityProfile(
            model_id="coder-fallback",
            revision="unversioned",
            task_tags=frozenset({"general", "coding"}),
            task_quality={"coding": 0.9},
            warm_ttft_p95_ms=200,
        ),
    ]
    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=_FakeSupervisor(),
            runtime_preflight_fn=lambda *_: [],
            gateway_open_fn=gateway_open,
            capability_profiles=profiles,
        )
    )
    for model_id, instance_id, port in (
        ("coder-primary", "instance-fallback-a", 21_000),
        ("coder-primary", "instance-fallback-b", 21_010),
        ("coder-fallback", "instance-fallback-c", 21_020),
    ):
        _start_ready_gateway_model(
            client,
            model_id=model_id,
            instance_id=instance_id,
            operation_id=f"operation-{instance_id}",
            port=port,
        )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tokenity-auto",
            "messages": [{"role": "user", "content": "fix this Python function"}],
        },
    )

    assert response.status_code == 200, response.text
    assert response.headers["x-tokenity-routed-model"] == "coder-fallback"
    assert response.headers["x-tokenity-route-reason"] == "fallback_same_capability_model"
    assert [payload["model"] for _, payload in attempts] == [
        "coder-primary",
        "coder-primary",
        "coder-fallback",
    ]
    assert release_counts == {
        "instance-fallback-a": 1,
        "instance-fallback-b": 1,
        "instance-fallback-c": 1,
    }
    routes = client.get("/v1/node/instances").json()["data"]
    assert all(item["active_request_count"] == 0 for item in routes)


def test_auto_fallback_skips_generic_model_and_uses_implicit_default_last(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    attempts = []

    def gateway_open(url, body, headers, timeout):
        payload = json.loads(body)
        attempts.append(payload["model"])
        if payload["model"] == "coder-primary":
            raise ConnectionError("primary unavailable")
        return (
            _FakeGatewayConnection(),
            _FakeGatewayResponse(b'{"choices":[]}', content_type="application/json"),
        )

    profiles = [
        agent_module.ModelCapabilityProfile(
            model_id="coder-primary",
            revision="unversioned",
            task_tags=frozenset({"general", "coding"}),
            task_quality={"general": 0.8, "coding": 1.0},
            warm_ttft_p95_ms=10,
        ),
        agent_module.ModelCapabilityProfile(
            model_id="generic-imposter",
            revision="unversioned",
            task_tags=frozenset({"general"}),
            task_quality={"general": 0.7, "coding": 0.8},
            warm_ttft_p95_ms=20,
        ),
        agent_module.ModelCapabilityProfile(
            model_id="strong-default",
            revision="unversioned",
            task_tags=frozenset({"reasoning"}),
            task_quality={"general": 1.0, "coding": 0.1},
            user_priority=10,
            warm_ttft_p95_ms=500,
        ),
    ]
    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=_FakeSupervisor(),
            runtime_preflight_fn=lambda *_: [],
            gateway_open_fn=gateway_open,
            capability_profiles=profiles,
        )
    )
    for model_id, instance_id, port in (
        ("coder-primary", "instance-strict-primary", 21_100),
        ("generic-imposter", "instance-strict-generic", 21_110),
        ("strong-default", "instance-strict-default", 21_120),
    ):
        _start_ready_gateway_model(
            client,
            model_id=model_id,
            instance_id=instance_id,
            operation_id=f"operation-{instance_id}",
            port=port,
        )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tokenity-auto",
            "messages": [{"role": "user", "content": "fix this Python function"}],
        },
    )

    assert response.status_code == 200, response.text
    assert attempts == ["coder-primary", "strong-default"]
    assert response.headers["x-tokenity-routed-model"] == "strong-default"
    assert response.headers["x-tokenity-route-reason"] == "fallback_default_model"


def test_stream_read_failure_does_not_fallback_and_releases_once(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    attempts = []
    release_counts = {}
    original_release = agent_module.InstanceRegistry.release_request_lease

    def release_once(self, instance_id):
        release_counts[instance_id] = release_counts.get(instance_id, 0) + 1
        return original_release(self, instance_id)

    monkeypatch.setattr(agent_module.InstanceRegistry, "release_request_lease", release_once)

    def gateway_open(url, body, headers, timeout):
        attempts.append(url)
        return (
            _FakeGatewayConnection(),
            _FakeGatewayResponse(
                b"data: partial\n\n",
                content_type="text/event-stream",
                fail_read1=True,
            ),
        )

    profile = agent_module.ModelCapabilityProfile(
        model_id="stream-model",
        revision="unversioned",
    )
    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=_FakeSupervisor(),
            runtime_preflight_fn=lambda *_: [],
            gateway_open_fn=gateway_open,
            capability_profiles=[profile],
        ),
        raise_server_exceptions=False,
    )
    for instance_id, port in (
        ("instance-stream-a", 22_000),
        ("instance-stream-b", 22_010),
    ):
        _start_ready_gateway_model(
            client,
            model_id="stream-model",
            instance_id=instance_id,
            operation_id=f"operation-{instance_id}",
            port=port,
        )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tokenity-auto",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert len(attempts) == 1
    assert release_counts == {"instance-stream-a": 1}
    instances_payload = client.get("/v1/node/instances").json()["data"]
    assert all(item["active_request_count"] == 0 for item in instances_payload)


def test_generation_slot_acquire_is_cancel_safe():
    scheduler = agent_module.GenerationSlotScheduler(max_queue_depth=2)
    active = scheduler.acquire("instance-active", ["mac-a"])

    async def scenario():
        task = asyncio.create_task(
            agent_module._acquire_generation_slot(
                scheduler,
                "instance-waiting",
                ["mac-a"],
                timeout=5,
            )
        )
        deadline = time.monotonic() + 1
        while scheduler.queue_depth("instance-waiting") != 1:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        deadline = time.monotonic() + 1
        while scheduler.queue_depth("instance-waiting") != 0:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.005)

    asyncio.run(scenario())
    active.release()
    assert scheduler.active_count == 0


def test_gateway_stream_response_cleans_up_when_response_start_fails():
    cleanup_calls = []
    response = agent_module._GatewayStreamingResponse(
        iter([b"data: ignored\n\n"]),
        on_close=lambda: cleanup_calls.append("closed"),
        media_type="text/event-stream",
    )

    async def send(message):
        if message["type"] == "http.response.start":
            raise RuntimeError("client disconnected before response start")

    async def receive():
        return {"type": "http.disconnect"}

    async def invoke():
        await response(
            {"type": "http", "asgi": {"spec_version": "2.4"}},
            receive,
            send,
        )

    with pytest.raises(RuntimeError, match="before response start"):
        asyncio.run(invoke())
    assert cleanup_calls == ["closed"]


def test_gateway_base_exception_during_open_releases_slot_and_request_lease(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))

    class GatewayCancelled(BaseException):
        pass

    calls = 0

    def gateway_open(url, body, headers, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise GatewayCancelled()
        return (
            _FakeGatewayConnection(),
            _FakeGatewayResponse(b'{"choices":[]}', content_type="application/json"),
        )

    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=_FakeSupervisor(),
            runtime_preflight_fn=lambda *_: [],
            gateway_open_fn=gateway_open,
            generation_slot_timeout=0.05,
        )
    )
    _start_ready_gateway_model(
        client,
        model_id="cancel-model",
        instance_id="instance-cancel-open",
        operation_id="operation-cancel-open",
        port=22_100,
    )

    with pytest.raises(GatewayCancelled):
        client.post(
            "/v1/chat/completions",
            json={"model": "cancel-model", "messages": []},
        )
    instance = client.get("/v1/node/instances/instance-cancel-open").json()["instance"]
    assert instance["active_request_count"] == 0

    response = client.post(
        "/v1/chat/completions",
        json={"model": "cancel-model", "messages": []},
    )
    assert response.status_code == 200, response.text


def test_auto_queue_depth_is_counted_once_in_router_score(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=_FakeSupervisor(),
            runtime_preflight_fn=lambda *_: [],
        )
    )
    _start_ready_gateway_model(
        client,
        model_id="queued-model",
        instance_id="instance-queued-score",
        operation_id="operation-queued-score",
        port=22_200,
    )
    original_snapshots = agent_module.InstanceRegistry.snapshots

    def queued_snapshots(self):
        snapshots = original_snapshots(self)
        for snapshot in snapshots:
            if snapshot["instance_id"] == "instance-queued-score":
                snapshot["queued_request_count"] = 1
        return snapshots

    monkeypatch.setattr(agent_module.InstanceRegistry, "snapshots", queued_snapshots)
    monkeypatch.setattr(
        agent_module.GenerationSlotScheduler,
        "snapshot",
        lambda self: {
            "max_queue_depth": 32,
            "queue_depth": 1,
            "active_count": 0,
            "queued_by_instance": {"instance-queued-score": 1},
            "active_by_instance": {},
        },
    )

    decision = client.post(
        "/v1/router/decision",
        json={"model": "tokenity-auto", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert decision.status_code == 200, decision.text
    candidate = decision.json()["candidates"][0]
    assert candidate["queue_penalty"] == pytest.approx(1.75)


def test_gateway_routing_latency_excludes_upstream_connect_time(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))

    def slow_gateway_open(url, body, headers, timeout):
        time.sleep(0.08)
        return (
            _FakeGatewayConnection(),
            _FakeGatewayResponse(b'{"choices":[]}', content_type="application/json"),
        )

    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=_FakeSupervisor(),
            runtime_preflight_fn=lambda *_: [],
            gateway_open_fn=slow_gateway_open,
        )
    )
    _start_ready_gateway_model(
        client,
        model_id="latency-model",
        instance_id="instance-latency",
        operation_id="operation-latency",
        port=22_300,
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "latency-model", "messages": []},
    )

    assert response.status_code == 200
    assert float(response.headers["x-tokenity-routing-latency-ms"]) < 40


def test_glm_profile_strips_qwen_private_thinking_switch():
    glm = agent_module._infer_gateway_model_profile(
        "glm-reasoning",
        "revision-glm",
        "/models/glm-reasoning",
    )

    class Selected:
        requested_model_id = "glm-reasoning"

    upstream = agent_module._gateway_upstream_payload(
        {
            "model": "tokenity-auto",
            "messages": [],
            "chat_template_kwargs": {
                "enable_thinking": True,
                "qwen_private_value": "must-not-cross-models",
            },
        },
        Selected(),
        glm,
    )

    assert "chat_template_kwargs" not in upstream


def test_inferred_four_billion_parameter_model_is_a_fast_route_candidate():
    qwen = agent_module._infer_gateway_model_profile(
        "Qwen3.5-4B-Native-MTP-4bit",
        "revision-qwen",
        "/models/Qwen3.5-4B-Native-MTP-4bit",
    )

    assert qwen.warm_ttft_p95_ms == 120.0
    assert qwen.task_quality["general"] == 0.72


def test_inferred_qwen_profile_reads_nested_text_context_capacity(tmp_path: Path):
    model = tmp_path / "Qwen3.5-4B-Native-MTP-4bit"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "max_position_embeddings": 262_144,
                    "max_new_tokens": 16_384,
                },
            }
        ),
        encoding="utf-8",
    )

    profile = agent_module._infer_gateway_model_profile(
        model.name,
        "revision-qwen",
        str(model),
    )

    assert profile.context_length == 262_144
    assert profile.max_output_length == 16_384
    assert "long-context" in profile.task_tags


def test_inferred_profiles_route_fast_to_qwen_four_b_and_quality_to_glm(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=_FakeSupervisor(),
            runtime_preflight_fn=lambda *_: [],
        )
    )
    _start_ready_gateway_model(
        client,
        model_id="GLM-5.2-mxfp4",
        model_path="/models/GLM-5.2-mxfp4",
        instance_id="instance-glm-profile",
        operation_id="operation-glm-profile",
        port=23_000,
    )
    _start_ready_gateway_model(
        client,
        model_id="Qwen3.5-4B-Native-MTP-4bit",
        model_path="/models/Qwen3.5-4B-Native-MTP-4bit",
        instance_id="instance-qwen-profile",
        operation_id="operation-qwen-profile",
        port=23_010,
    )

    fast = client.post(
        "/v1/router/decision",
        json={
            "model": "tokenity-auto",
            "messages": [{"role": "user", "content": "Reply briefly."}],
            "tokenity_route_policy": "fast",
        },
    )
    quality = client.post(
        "/v1/router/decision",
        json={
            "model": "tokenity-auto",
            "messages": [{"role": "user", "content": "Solve this carefully."}],
            "tokenity_route_policy": "quality",
        },
    )

    assert fast.status_code == 200, fast.text
    assert fast.json()["model_id"] == "Qwen3.5-4B-Native-MTP-4bit"
    assert quality.status_code == 200, quality.text
    assert quality.json()["model_id"] == "GLM-5.2-mxfp4"


def test_gateway_returns_retry_after_when_generation_queue_is_full(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    opened = []
    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=_FakeSupervisor(),
            runtime_preflight_fn=lambda *_: [],
            gateway_open_fn=lambda *args: opened.append(args),
        )
    )
    _start_ready_gateway_model(
        client,
        model_id="queue-model",
        instance_id="instance-queue",
        operation_id="operation-queue",
        port=23_000,
    )

    def queue_full(self, instance_id, selected_nodes, **kwargs):
        raise agent_module.GenerationQueueFull("queue full")

    monkeypatch.setattr(agent_module.GenerationSlotScheduler, "acquire", queue_full)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "queue-model", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"
    assert opened == []
    instance = client.get("/v1/node/instances/instance-queue").json()["instance"]
    assert instance["active_request_count"] == 0


def test_managed_instances_reject_legacy_global_heartbeat_and_keep_instance_lease(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(agent_module.tempfile, "gettempdir", lambda: str(tmp_path))
    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=_FakeSupervisor(),
            runtime_preflight_fn=lambda *_: [],
        )
    )
    assert client.post("/v1/node/heartbeat", json={"ttl_seconds": 30}).status_code == 200
    _start_ready_gateway_model(
        client,
        model_id="managed-model",
        instance_id="instance-managed-lease",
        operation_id="operation-managed-lease",
        port=23_100,
    )

    legacy = client.post("/v1/node/heartbeat", json={"ttl_seconds": 30})
    managed = client.post(
        "/v1/node/heartbeat",
        json={"ttl_seconds": 30, "instance_id": "instance-managed-lease"},
    )

    assert legacy.status_code == 409
    assert "instance_id" in legacy.text
    assert managed.status_code == 200
    instance = client.get("/v1/node/instances/instance-managed-lease").json()["instance"]
    assert instance["state"] == "ready"


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


def test_collective_port_allocator_checks_every_port_in_candidate_range(monkeypatch):
    unavailable = {30_001}
    monkeypatch.setattr(
        agent_module,
        "_tcp_port_available",
        lambda port: port not in unavailable,
    )

    starting_port = agent_module._allocate_collective_port(
        30_000,
        3,
        {"ports": {}},
    )

    assert starting_port == 30_002


def test_collective_port_allocator_includes_last_valid_starting_port(monkeypatch):
    monkeypatch.setattr(agent_module, "_tcp_port_available", lambda port: True)

    starting_port = agent_module._allocate_collective_port(
        65_533,
        3,
        {"ports": {}},
    )

    assert starting_port == 65_533


def test_start_excludes_allocated_http_port_from_collective_range(monkeypatch):
    monkeypatch.setattr(agent_module, "_tcp_port_available", lambda port: True)
    client = TestClient(
        create_app(
            rdma_probe_fn=fake_rdma_probe,
            supervisor=_FakeSupervisor(),
            runtime_preflight_fn=lambda *_: [],
        )
    )

    response = client.post(
        "/v1/node/start-distributed-openai",
        json={
            "model": "/models/qwen",
            "instance_id": "instance-port-overlap",
            "operation_id": "operation-port-overlap",
            "connection_mode": "ring",
            "port": 18_000,
            "starting_port": 18_000,
            "dry_run": False,
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["instance"]["http_port"] == 18_000
    assert response.json()["instance"]["starting_port"] == 18_001


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
