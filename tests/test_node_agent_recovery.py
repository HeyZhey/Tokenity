from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

import tokenity.node_agent.agent as agent_module
import tokenity.process.supervisor as supervisor_module
from tokenity.node_agent.agent import create_app
from tokenity.node_agent.recovery import (
    InstanceStateStore,
    RecoveryAssessment,
    assess_recovery_record,
    process_start_identity,
)
from tokenity.process.supervisor import RoleStatus, RoleSupervisor


def _instance_payload() -> dict[str, object]:
    return {
        "instance_id": "instance-recovered",
        "operation_id": "operation-recovered",
        "requested_model_id": "qwen-4b",
        "resolved_path": "/models/qwen-4b",
        "model_revision": "revision-recovered",
        "tokenizer_identity": "tokenizer-recovered",
        "execution_mode": "single_node",
        "selected_nodes": ["mac-a"],
        "rank_mapping": {"mac-a": 0},
        "world_size": 1,
        "connection_mode": "ring",
        "coordinator": "mac-a",
        "http_port": 8234,
        "starting_port": 31_234,
        "memory_reservation_bytes": 1_024,
        "state": "loading_metadata",
        "version": 3,
        "generation": 1,
        "created_at": 1.0,
        "updated_at": 1.0,
        "deadline": None,
        "heartbeat_at": None,
        "actual_memory_bytes": None,
        "active_request_count": 0,
        "readiness_evidence": {},
        "last_error": None,
        "log_paths": {"rank-0": "/tmp/recovered.log"},
        "process_identities": {"mac-a": 4321},
        "queued_request_count": 0,
        "memory_reservation_breakdown": None,
        "health_ready": None,
        "health_sampled_at": None,
        "health_issues": [],
    }


def _runtime_payload(*, updated_at: float) -> dict[str, object]:
    return {
        "version": 9,
        "phase": "ready",
        "lifecycle_state": "ready",
        "instance_id": "instance-recovered",
        "operation_id": "operation-recovered",
        "model_revision": "revision-recovered",
        "rank": 0,
        "world_size": 1,
        "connection_mode": "ring",
        "updated_at": updated_at,
        "ready_evidence": {
            "one_token_probe": True,
            "warmup_cache_isolated": True,
        },
    }


def test_recovery_requires_process_start_command_runtime_and_revision(tmp_path: Path):
    status_path = tmp_path / "runtime.json"
    now = time.time()
    status_path.write_text(json.dumps(_runtime_payload(updated_at=now)), encoding="utf-8")
    record = {
        "instance": _instance_payload(),
        "process": {"pid": 4321, "start_identity": "identity-a"},
        "rank": 0,
        "status_path": str(status_path),
    }

    assessment = assess_recovery_record(
        record,
        now=now,
        start_identity_fn=lambda pid: "identity-a",
        command_fn=lambda pid: (
            f"python -m tokenity distributed-openai serve --model "
            f"{record['instance']['resolved_path']}"
        ),
    )

    assert assessment.safe_to_adopt is True
    assert assessment.reason == "identity_revision_epoch_and_runtime_verified"

    reused = assess_recovery_record(
        record,
        now=now,
        start_identity_fn=lambda pid: "identity-b",
        command_fn=lambda pid: "unrelated process",
    )
    assert reused.safe_to_adopt is False
    assert reused.reason == "process_start_identity_mismatch"

    stale_runtime = _runtime_payload(updated_at=now - 60)
    status_path.write_text(json.dumps(stale_runtime), encoding="utf-8")
    stale = assess_recovery_record(
        record,
        now=now,
        start_identity_fn=lambda pid: "identity-a",
        command_fn=lambda pid: (
            "python -m tokenity distributed-openai serve --model /models/qwen-4b"
        ),
    )
    assert stale.safe_to_adopt is False
    assert stale.reason == "runtime_heartbeat_stale"


def test_recovery_rejects_unhashable_identity_fields_without_crashing(tmp_path: Path):
    record = {
        "instance": _instance_payload() | {"world_size": []},
        "process": {"pid": 4321, "start_identity": "identity-a"},
        "rank": 0,
        "status_path": str(tmp_path / "runtime.json"),
    }

    assessment = assess_recovery_record(record)

    assert assessment.safe_to_adopt is False
    assert assessment.reason == "process_missing"


def test_instance_state_store_is_atomic_and_removable(tmp_path: Path):
    store = InstanceStateStore(tmp_path / "instances")
    store.write("instance/a", {"instance": {"instance_id": "instance/a"}})

    records = store.records()
    assert len(records) == 1
    assert records[0]["instance"]["instance_id"] == "instance/a"
    assert list(store.root.glob("*.tmp")) == []

    store.remove("instance/a")
    assert store.records() == []


def test_supervisor_adopts_and_stops_only_the_fenced_process(tmp_path: Path):
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    supervisor = RoleSupervisor(log_dir=tmp_path)
    try:
        identity = process_start_identity(process.pid)
        assert identity is not None
        adopted = supervisor.adopt(
            "single-node-openai",
            instance_id="instance-adopted",
            operation_id="operation-adopted",
            pid=process.pid,
            start_identity=identity,
            command=["python", "-m", "tokenity", "distributed-openai", "serve"],
        )
        assert adopted.state == "running"
        assert adopted.pid == process.pid

        stopped = supervisor.stop(
            "single-node-openai",
            timeout=2,
            instance_id="instance-adopted",
        )
        assert stopped.state == "stopped"
        assert process.wait(timeout=2) != 999
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)


def test_unjournaled_scan_does_not_claim_h3_owned_by_an_isolated_agent(
    tmp_path: Path,
    monkeypatch,
):
    class ProcessList:
        stdout = """\
101 python -m tokenity distributed-openai serve --model /models/text
202 python -m tokenity minimax-h3-video serve --binary /runtime/mlx-serve --model /models/h3
203 /runtime/mlx-serve --model /models/h3 --serve
"""

    monkeypatch.setattr(
        supervisor_module.subprocess,
        "run",
        lambda *args, **kwargs: ProcessList(),
    )

    matches = RoleSupervisor(log_dir=tmp_path).cleanup_orphaned_model_processes()

    assert [match["pid"] for match in matches] == [101]


def test_agent_restart_adopts_verified_instance_and_restores_resources(
    tmp_path: Path,
    monkeypatch,
):
    now = time.time()
    status_path = tmp_path / "runtime.json"
    runtime = _runtime_payload(updated_at=now)
    status_path.write_text(json.dumps(runtime), encoding="utf-8")
    record = {
        "schema_version": 1,
        "instance": _instance_payload(),
        "role": "single-node-openai",
        "rank": 0,
        "command": [
            sys.executable,
            "-m",
            "tokenity",
            "distributed-openai",
            "serve",
            "--model",
            "/models/qwen-4b",
        ],
        "status_path": str(status_path),
        "ports": [8234, 31_234],
        "worker_agent_urls": [],
        "cluster_runtime": {
            "instance_id": "instance-recovered",
            "operation_id": "operation-recovered",
            "rank": 0,
            "world_size": 1,
            "connection_mode": "ring",
            "model_revision": "revision-recovered",
            "role": "single-node-openai",
        },
        "process": {"pid": 4321, "start_identity": "identity-recovered"},
    }
    store = InstanceStateStore(tmp_path / "instances")
    store.write("instance-recovered", record)

    monkeypatch.setattr(
        agent_module,
        "assess_recovery_record",
        lambda value: RecoveryAssessment(
            True,
            "identity_revision_epoch_and_runtime_verified",
            value,
            runtime,
        ),
    )
    monkeypatch.setattr(agent_module, "_total_memory_bytes", lambda: 1_000_000_000)
    monkeypatch.setattr(
        agent_module,
        "_post_json",
        lambda *args, **kwargs: {"status": "stopping"},
    )

    class AdoptSupervisor:
        def __init__(self):
            self.running = True
            self.adoptions: list[dict[str, object]] = []

        def adopt(self, role, **kwargs):
            self.adoptions.append({"role": role, **kwargs})
            return self.status(role, instance_id=kwargs["instance_id"])

        def status(self, role=None, *, instance_id=None):
            if role is None:
                return []
            return RoleStatus(
                role=role,
                state="running" if self.running else "stopped",
                instance_id=instance_id,
                operation_id="operation-recovered",
                pid=4321 if self.running else None,
                command=record["command"],
            )

        def stop(self, role, timeout=10, *, instance_id=None):
            self.running = False
            return self.status(role, instance_id=instance_id)

    supervisor = AdoptSupervisor()
    app = create_app(supervisor=supervisor, instance_state_store=store)
    with TestClient(app) as client:
        instance_response = client.get("/v1/node/instances/instance-recovered")
        status_response = client.get("/v1/node/status")

    assert instance_response.status_code == 200
    assert instance_response.json()["instance"]["state"] == "ready"
    assert instance_response.json()["instance"]["version"] >= runtime["version"]
    ledger = status_response.json()["resource_ledger"]
    assert ledger["reservations"] == {"instance-recovered": 1_024}
    assert ledger["ports"] == {
        "8234": "instance-recovered",
        "31234": "instance-recovered",
    }
    assert len(supervisor.adoptions) == 1


def test_unadoptable_but_fenced_process_is_cleaned_without_touching_siblings(
    tmp_path: Path,
    monkeypatch,
):
    status_path = tmp_path / "runtime.json"
    status_path.write_text(
        json.dumps(_runtime_payload(updated_at=time.time() - 60)),
        encoding="utf-8",
    )
    record = {
        "schema_version": 1,
        "instance": _instance_payload(),
        "role": "distributed-openai-rank",
        "rank": 0,
        "command": [
            "python",
            "-m",
            "tokenity",
            "distributed-openai",
            "serve",
            "--model",
            "/models/qwen-4b",
        ],
        "status_path": str(status_path),
        "ports": [31_234],
        "worker_agent_urls": [],
        "process": {"pid": 4321, "start_identity": "identity-recovered"},
    }
    store = InstanceStateStore(tmp_path / "instances")
    store.write("instance-recovered", record)
    alive = True
    signals: list[tuple[int, int]] = []

    monkeypatch.setattr(
        agent_module,
        "assess_recovery_record",
        lambda value: RecoveryAssessment(False, "runtime_heartbeat_stale", value),
    )
    monkeypatch.setattr(
        agent_module,
        "process_start_identity",
        lambda pid: "identity-recovered",
    )
    monkeypatch.setattr(
        agent_module,
        "process_command",
        lambda pid: (
            "python -m tokenity distributed-openai serve --model /models/qwen-4b"
        ),
    )

    class ProcessState:
        stdout = "S"

    def fake_process_state(*args, **kwargs):
        result = ProcessState()
        result.stdout = "S" if alive else "Z"
        return result

    def fake_killpg(group: int, sent_signal: int):
        nonlocal alive
        signals.append((group, sent_signal))
        alive = False

    monkeypatch.setattr(agent_module.subprocess, "run", fake_process_state)
    monkeypatch.setattr(agent_module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(agent_module.os, "killpg", fake_killpg)

    with TestClient(create_app(instance_state_store=store)) as client:
        status = client.get("/v1/node/status").json()
        health = client.get("/v1/node/health").json()

    orphan = status["orphaned_processes"][0]
    assert orphan["state"] == "cleaned"
    assert orphan["cleanup_required"] is False
    assert signals == [(4321, 15)]
    assert store.records() == []
    assert health["status"] == "healthy"


def test_h3_restart_uses_journaled_fenced_cleanup_instead_of_unsafe_adoption(
    tmp_path: Path,
    monkeypatch,
):
    instance = _instance_payload() | {
        "instance_id": "h3-restart-instance",
        "operation_id": "h3-restart-operation",
        "requested_model_id": "MiniMax-H3",
        "resolved_path": "/models/MiniMax-H3",
        "execution_mode": "tp2",
        "world_size": 2,
        "connection_mode": "jaccl-ring",
        "http_port": 24_200,
        "starting_port": 30_096,
    }
    record = {
        "schema_version": 1,
        "instance": instance,
        "role": "minimax-h3-video",
        "rank": 0,
        "command": [
            "python",
            "-m",
            "tokenity",
            "minimax-h3-video",
            "serve",
            "--binary",
            "/runtime/bin/mlx-serve",
            "--model",
            "/models/MiniMax-H3",
        ],
        "status_path": None,
        "recovery_policy": "cleanup",
        "ports": [24_200, 30_096, 30_097],
        "worker_agent_urls": ["http://worker:9200"],
        "process": {"pid": 4321, "start_identity": "identity-h3"},
    }
    store = InstanceStateStore(tmp_path / "instances")
    store.write("h3-restart-instance", record)
    alive = True
    signals: list[tuple[int, int]] = []
    remote_stops: list[str] = []

    monkeypatch.setenv("TOKENITY_DISABLE_UNJOURNALED_ORPHAN_SCAN", "1")
    monkeypatch.setattr(agent_module, "process_start_identity", lambda pid: "identity-h3")
    monkeypatch.setattr(
        agent_module,
        "process_command",
        lambda pid: "/runtime/bin/mlx-serve --model /models/MiniMax-H3 --serve",
    )
    monkeypatch.setattr(agent_module, "_post_json", lambda *args, **kwargs: {})

    class ProcessState:
        @property
        def stdout(self):
            return "S" if alive else "Z"

    monkeypatch.setattr(agent_module.subprocess, "run", lambda *args, **kwargs: ProcessState())
    monkeypatch.setattr(agent_module.os, "getpgid", lambda pid: pid)

    def fake_killpg(group: int, sent_signal: int):
        nonlocal alive
        signals.append((group, sent_signal))
        alive = False

    monkeypatch.setattr(agent_module.os, "killpg", fake_killpg)

    def post_json(url, payload, timeout):
        remote_stops.append(url)
        return {"status": "stopped"}

    with TestClient(create_app(instance_state_store=store, post_json_fn=post_json)) as client:
        status = client.get("/v1/node/status").json()
        health = client.get("/v1/node/health").json()

    orphan = status["orphaned_processes"][0]
    assert orphan["reason"] == "native_runtime_requires_fenced_restart_cleanup"
    assert orphan["state"] == "cleaned"
    assert orphan["cleanup_required"] is False
    assert signals == [(4321, 15)]
    assert remote_stops == [
        "http://worker:9200/v1/node/instances/h3-restart-instance/stop"
    ]
    assert store.records() == []
    assert health["status"] == "healthy"
