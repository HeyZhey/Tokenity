from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark-openai-stream.py"
EXAMPLE_MATRIX = (
    Path(__file__).parents[1]
    / "scripts"
    / "benchmark-auto-router-matrix.example.json"
)
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


def test_checked_in_example_matrix_matches_the_supported_schema():
    matrix = MODULE.load_matrix(EXAMPLE_MATRIX)
    MODULE.validate_matrix(matrix)
    assert {scenario["mode"] for scenario in matrix["scenarios"]} == MODULE.MATRIX_MODES


def _matrix():
    prompts = {
        "short": "private short prompt",
        "medium": "private medium prompt with more detail",
        "long": "private long prompt " * 100,
    }
    return {
        "schema_version": 1,
        "base_url": "http://127.0.0.1:9100/v1",
        "repeats": 1,
        "prompt_profiles": {
            name: {"prompt": prompt, "max_tokens": 32}
            for name, prompt in prompts.items()
        },
        "scenarios": [
            {"name": "strongest", "mode": "strongest", "model": "model-quality"},
            {"name": "fastest", "mode": "fastest", "model": "model-fast"},
            {
                "name": "auto",
                "mode": "auto",
                "model": "tokenity-auto",
                "route_policy": "balanced",
            },
            {
                "name": "resident-serial",
                "mode": "resident-serial",
                "models": ["model-quality", "model-fast"],
                "concurrency": 1,
            },
            {
                "name": "resident-concurrent",
                "mode": "resident-concurrent",
                "models": ["model-quality", "model-fast"],
                "concurrency": 2,
                "quality_by_profile": {
                    "short": {
                        "outcome": "tie",
                        "route_regret": 0.0,
                        "source": "human-review",
                    }
                },
            },
        ],
    }


def test_matrix_validation_requires_all_comparison_modes_and_prompt_profiles():
    matrix = _matrix()
    MODULE.validate_matrix(matrix)

    incomplete = _matrix()
    incomplete["scenarios"] = incomplete["scenarios"][:-1]
    with pytest.raises(ValueError, match="resident-concurrent"):
        MODULE.validate_matrix(incomplete)

    incomplete = _matrix()
    del incomplete["prompt_profiles"]["long"]
    with pytest.raises(ValueError, match="long"):
        MODULE.validate_matrix(incomplete)


def test_matrix_execution_is_reproducible_and_never_serializes_raw_prompts():
    calls = []

    def sample_runner(**kwargs):
        calls.append(kwargs)
        return {
            "sample_index": kwargs["sample_index"],
            "ok": True,
            "session_id": kwargs["session_id"],
            "routed_model": kwargs["model"],
            "routed_model_revision": "revision-a",
            "routed_instance_id": f"{kwargs['model']}-instance",
            "route_reason": "benchmark-test",
            "router_latency_ms": 1.5,
            "queue_wait_ms": 2.0,
            "fallback": False,
            "headers_seconds": 0.01,
            "first_sse_byte_seconds": 0.02,
            "ttft_content_seconds": 0.03,
            "total_seconds": 0.04,
            "prefill_tokens_per_second": 100.0,
            "decode_tokens_per_second": 20.0,
            "quality_outcome": kwargs["quality_outcome"],
            "quality_source": kwargs["quality_source"],
            "route_regret": kwargs["route_regret"],
        }

    telemetry_calls = []

    def telemetry_collector(_base_url):
        telemetry_calls.append(True)
        return {
            "macos": {
                "memory_pressure": None,
                "compressed_bytes": None,
                "swap": None,
                "thermal": None,
                "errors": ["test telemetry"],
            },
            "agent": {"memory": None, "error": "test telemetry"},
        }

    result = MODULE.execute_matrix(
        _matrix(),
        sample_runner=sample_runner,
        telemetry_collector=telemetry_collector,
    )
    rendered = json.dumps(result, sort_keys=True)

    assert len(result["results"]) == 15
    assert len(calls) == 21
    assert len(telemetry_calls) == 32
    assert result["matrix"]["prompt_profiles"]["short"]["characters"] == 20
    assert len(result["matrix"]["prompt_profiles"]["short"]["sha256"]) == 64
    assert "private short prompt" not in rendered
    assert "private medium prompt" not in rendered
    assert "private long prompt" not in rendered
    assert result["aggregate"]["quality"]["tie"] == 2
    assert result["aggregate"]["route_regret"]["count"] == 2
    profiles_by_session = {}
    for call in calls:
        profiles_by_session.setdefault(call["session_id"], set()).add(
            call["prompt_profile"]
        )
    assert all(profiles == {"short", "medium", "long"} for profiles in profiles_by_session.values())
    auto_calls = [call for call in calls if call["scenario"] == "auto"]
    assert all(call["route_policy"] == "balanced" for call in auto_calls)


def test_route_headers_are_captured_with_typed_null_safe_values():
    metadata = MODULE.route_metadata_from_headers(
        {
            "X-Tokenity-Model": "qwen",
            "X-Tokenity-Model-Revision": "revision-a",
            "X-Tokenity-Instance-ID": "instance-a",
            "X-Tokenity-Route-Reason": "coding",
            "X-Tokenity-Router-Latency-Ms": "3.25",
            "X-Tokenity-Queue-Wait-Ms": "7.5",
            "X-Tokenity-Fallback": "true",
        }
    )

    assert metadata == {
        "routed_model": "qwen",
        "routed_model_revision": "revision-a",
        "routed_instance_id": "instance-a",
        "route_reason": "coding",
        "router_latency_ms": 3.25,
        "queue_wait_ms": 7.5,
        "fallback": True,
    }
    empty = MODULE.route_metadata_from_headers({})
    assert empty["routed_model"] is None
    assert empty["router_latency_ms"] is None
    assert empty["fallback"] is False


def test_route_headers_support_gateway_latency_name_and_reason_based_fallback():
    metadata = MODULE.route_metadata_from_headers(
        {
            "X-Tokenity-Model": "qwen",
            "X-Tokenity-Route-Reason": "fallback_primary_failed",
            "X-Tokenity-Routing-Latency-Ms": "4.5",
        }
    )

    assert metadata["router_latency_ms"] == 4.5
    assert metadata["fallback"] is True


def test_aggregate_reports_cancel_fallback_switches_quality_and_route_regret():
    result = MODULE.aggregate(
        [
            {
                "ok": True,
                "sample_index": 0,
                "session_id": "session-a",
                "routed_model": "fast",
                "fallback": False,
                "quality_outcome": "win",
                "quality_source": "human",
                "route_regret": 0.0,
                "router_latency_ms": 1.0,
                "queue_wait_ms": 0.0,
            },
            {
                "ok": True,
                "sample_index": 1,
                "session_id": "session-a",
                "routed_model": "quality",
                "fallback": True,
                "quality_outcome": "loss",
                "quality_source": "human",
                "route_regret": 0.5,
                "router_latency_ms": 2.0,
                "queue_wait_ms": 4.0,
            },
            {"ok": False, "cancelled": True},
            {"ok": False, "cancelled": False},
        ]
    )

    assert result["failed"] == 2
    assert result["cancelled"] == 1
    assert result["fallback"] == 1
    assert result["model_switches"] == 1
    assert result["quality"] == {"win": 1, "tie": 0, "loss": 1, "unrated": 2}
    assert result["route_regret"]["mean"] == 0.25
    assert result["router_latency_ms"]["p95"] == 1.95
    assert result["queue_wait_ms"]["mean"] == 2.0


def test_model_switches_are_counted_only_within_a_session():
    result = MODULE.aggregate(
        [
            {
                "ok": True,
                "sample_index": 0,
                "session_id": "session-a",
                "routed_model": "fast",
            },
            {
                "ok": True,
                "sample_index": 1,
                "session_id": "session-b",
                "routed_model": "quality",
            },
            {
                "ok": True,
                "sample_index": 2,
                "session_id": "session-a",
                "routed_model": "quality",
            },
        ]
    )

    assert result["session_count"] == 2
    assert result["model_switches"] == 1


def test_quality_annotations_must_be_explicit_and_are_never_inferred():
    matrix = _matrix()
    matrix["scenarios"][-1]["quality_by_profile"]["short"] = {
        "outcome": "win",
        "route_regret": 0.1,
    }
    with pytest.raises(ValueError, match="source"):
        MODULE.validate_matrix(matrix)

    assert MODULE.quality_annotation({}) == {
        "quality_outcome": None,
        "quality_source": None,
        "route_regret": None,
    }


def test_macos_telemetry_parsers_return_values_or_explicit_nulls():
    vm = MODULE.parse_vm_stat(
        'Mach Virtual Memory Statistics: (page size of 16384 bytes)\n'
        '"Pages occupied by compressor": 10.\n'
    )
    swap = MODULE.parse_swapusage(
        "vm.swapusage: total = 4096.00M  used = 512.00M  free = 3584.00M"
    )
    thermal = MODULE.parse_thermal(
        "CPU_Speed_Limit = 80\nScheduler_Limit = 70\nThermal_Level = 1\n"
    )

    assert vm["compressed_bytes"] == 163_840
    assert swap == {
        "total_bytes": 4_294_967_296,
        "used_bytes": 536_870_912,
        "free_bytes": 3_758_096_384,
    }
    assert thermal == {
        "cpu_speed_limit_percent": 80,
        "scheduler_limit_percent": 70,
        "thermal_level": 1,
    }
    assert (
        MODULE.parse_memory_pressure("System-wide memory free percentage: 42%")
        == 42.0
    )
    unsupported = MODULE.collect_macos_telemetry(system_name="Linux")
    assert unsupported["memory_pressure_percent"] is None
    assert unsupported["compressed_bytes"] is None
    assert unsupported["swap"] is None
    assert unsupported["thermal"] is None
    assert unsupported["errors"] == ["macOS telemetry is unavailable on Linux."]
