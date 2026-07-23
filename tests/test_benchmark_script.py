from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark-openai-stream.py"
SPEC = importlib.util.spec_from_file_location("tokenity_benchmark_stream", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_percentile_interpolates_and_handles_empty_input():
    assert MODULE.percentile([], 0.95) is None
    assert MODULE.percentile([1.0, 2.0, 3.0], 0.50) == 2.0
    assert MODULE.percentile([1.0, 2.0], 0.95) == 1.95


def test_aggregate_keeps_failures_out_of_latency_distributions():
    result = MODULE.aggregate(
        [
            {"ok": True, "headers_seconds": 0.1, "ttft_content_seconds": 1.0, "total_seconds": 2.0},
            {"ok": True, "headers_seconds": 0.3, "ttft_content_seconds": 3.0, "total_seconds": 4.0},
            {"ok": False, "total_seconds": 100.0},
        ]
    )

    assert result["successful"] == 2
    assert result["failed"] == 1
    assert result["headers_seconds"]["mean"] == 0.2
    assert result["total_seconds"]["mean"] == 3.0


def test_readiness_url_supports_direct_runtime_and_gateway_base_paths():
    assert MODULE.readiness_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1/readiness"
    assert MODULE.readiness_url("http://127.0.0.1:9100/v1/") == "http://127.0.0.1:9100/v1/readiness"
