from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "preflight-tokenity-cluster.py"
SPEC = importlib.util.spec_from_file_location("tokenity_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_default_fixture_uses_only_current_mac_b_addresses():
    mac_b = next(item for item in MODULE.DEFAULT_NODES if item["id"] == "mac-b")
    assert mac_b == {
        "id": "mac-b",
        "agent_url": "http://192.168.5.75:9100",
        "lan_ip": "192.168.5.75",
        "rdma_device": "rdma_en5",
        "rdma_ip": "192.168.0.2",
    }


def test_preflight_uses_semantic_version_comparison():
    assert MODULE.version_release("0.31.10") > MODULE.version_release("0.31.3")
    assert MODULE.version_release("0.31.3+local") == (0, 31, 3)


def test_node_evaluation_requires_runtime_model_and_rdma_evidence():
    fixture = MODULE.DEFAULT_NODES[1]
    info = {
        "node_id": "mac-b",
        "hostname": "Kiwi",
        "ips": ["192.168.5.75"],
        "architecture": "arm64",
        "python_path": "/runtime/python",
        "mlx_version": "0.31.2",
        "mlx_lm_version": "0.31.10",
        "tokenity_code_revision": "same-revision",
        "rdma": {
            "rdma_devices": ["rdma_en5"],
            "rdma_port_state": {"rdma_en5": "active"},
            "thunderbolt_ip": "192.168.0.2",
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
