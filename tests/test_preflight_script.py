from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "preflight-tokenity-cluster.py"
SPEC = importlib.util.spec_from_file_location("tokenity_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preflight_requires_explicit_node_and_deployment_paths(monkeypatch):
    monkeypatch.delenv("TOKENITY_PREFLIGHT_MODEL_PATH", raising=False)
    monkeypatch.delenv("TOKENITY_RUNTIME_PYTHON", raising=False)
    try:
        MODULE.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("deployment-specific preflight values must be explicit")


def test_preflight_uses_semantic_version_comparison():
    assert MODULE.version_release("0.31.10") > MODULE.version_release("0.31.3")
    assert MODULE.version_release("0.31.3+local") == (0, 31, 3)


def test_node_evaluation_requires_runtime_model_and_rdma_evidence():
    fixture = {
        "id": "test-node",
        "agent_url": "http://node.invalid:9100",
        "lan_ip": "198.51.100.10",
        "rdma_device": "rdma_test0",
        "rdma_ip": "203.0.113.10",
    }
    info = {
        "node_id": "mac-b",
        "hostname": "Kiwi",
        "ips": ["198.51.100.10"],
        "architecture": "arm64",
        "python_path": "/runtime/python",
        "mlx_version": "0.32.0",
        "mlx_lm_version": "0.31.10",
        "tokenity_code_revision": "same-revision",
        "rdma": {
            "rdma_devices": ["rdma_test0"],
            "rdma_port_state": {"rdma_test0": "active"},
            "thunderbolt_ip": "203.0.113.10",
        },
    }
    models = {
        "models": [
            {"path": "/models/qwen", "revision": "model-revision", "size_bytes": 123}
        ]
    }

    result = MODULE.evaluate_node(
        fixture,
        info,
        models,
        model_path="/models/qwen",
        python_path="/runtime/python",
    )

    assert result["ok"] is True
    assert result["observed"]["model_revision"] == "model-revision"
