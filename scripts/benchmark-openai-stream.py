#!/usr/bin/env python3
"""Reproducible OpenAI-compatible SSE latency/throughput sampler.

The sampler uses HTTP/1.1 without environment proxies and timestamps bytes as
they arrive. It distinguishes response headers, first SSE byte, first content
token, completion, and (when Tokenity readiness is directly reachable) the
server-side accepted/prefill/first-token timeline.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener


MATRIX_MODES = frozenset(
    {"strongest", "fastest", "auto", "resident-serial", "resident-concurrent"}
)
PROMPT_PROFILES = frozenset({"short", "medium", "long"})
QUALITY_OUTCOMES = frozenset({"win", "tie", "loss"})


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def readiness_url(base_url: str) -> str:
    parsed = urlsplit(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/v1/readiness", "", ""))


def agent_status_url(base_url: str) -> str:
    parsed = urlsplit(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/v1/node/status", "", ""))


def prompt_descriptor(prompt: str, profile: str) -> dict[str, Any]:
    encoded = prompt.encode("utf-8")
    return {
        "profile": profile,
        "characters": len(prompt),
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _header_value(headers: Any, *names: str) -> str | None:
    if headers is None:
        return None
    normalized: dict[str, Any] = {}
    if hasattr(headers, "items"):
        normalized = {str(key).lower(): value for key, value in headers.items()}
    for name in names:
        value = normalized.get(name.lower())
        if value is None and hasattr(headers, "get"):
            value = headers.get(name)
        if value is not None:
            rendered = str(value).strip()
            return rendered or None
    return None


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def route_metadata_from_headers(headers: Any) -> dict[str, Any]:
    fallback = _header_value(headers, "X-Tokenity-Fallback")
    route_reason = _header_value(headers, "X-Tokenity-Route-Reason")
    return {
        "routed_model": _header_value(
            headers,
            "X-Tokenity-Model",
            "X-Tokenity-Routed-Model",
        ),
        "routed_model_revision": _header_value(
            headers,
            "X-Tokenity-Model-Revision",
            "X-Tokenity-Revision",
        ),
        "routed_instance_id": _header_value(headers, "X-Tokenity-Instance-ID"),
        "route_reason": route_reason,
        "router_latency_ms": _optional_float(
            _header_value(
                headers,
                "X-Tokenity-Routing-Latency-Ms",
                "X-Tokenity-Router-Latency-Ms",
                "X-Tokenity-Route-Latency-Ms",
            )
        ),
        "queue_wait_ms": _optional_float(
            _header_value(
                headers,
                "X-Tokenity-Queue-Wait-Ms",
                "X-Tokenity-Queue-Ms",
            )
        ),
        "fallback": (
            fallback is not None and fallback.lower() in {"1", "true", "yes", "on"}
        )
        or bool(route_reason and route_reason.lower().startswith("fallback_")),
    }


def quality_annotation(values: dict[str, Any] | None) -> dict[str, Any]:
    values = values or {}
    outcome = values.get("outcome")
    regret = values.get("route_regret")
    source = values.get("source")
    if outcome is not None and outcome not in QUALITY_OUTCOMES:
        raise ValueError(f"quality outcome must be one of {sorted(QUALITY_OUTCOMES)}")
    if regret is not None and not isinstance(regret, (int, float)):
        raise ValueError("route_regret must be numeric")
    if (outcome is not None or regret is not None) and not (
        isinstance(source, str) and source.strip()
    ):
        raise ValueError("Explicit quality annotations require a non-empty source.")
    return {
        "quality_outcome": outcome,
        "quality_source": source.strip() if isinstance(source, str) and source.strip() else None,
        "route_regret": float(regret) if isinstance(regret, (int, float)) else None,
    }


def fetch_request_timeline(base_url: str, request_id: str | None) -> dict[str, Any] | None:
    if not request_id:
        return None
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(Request(readiness_url(base_url)), timeout=3) as response:
            payload = json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None
    last_request = payload.get("last_request") if isinstance(payload, dict) else None
    if not isinstance(last_request, dict) or last_request.get("request_id") != request_id:
        return None
    return last_request


def run_sample(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    sample_index: int,
    scenario: str = "manual",
    prompt_profile: str = "manual",
    route_policy: str | None = None,
    session_id: str | None = None,
    quality_outcome: str | None = None,
    quality_source: str | None = None,
    route_regret: float | None = None,
) -> dict[str, Any]:
    request_payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if route_policy is not None:
        request_payload["tokenity_route_policy"] = route_policy
    if session_id is not None:
        request_payload["tokenity_session_id"] = session_id
    body = json.dumps(request_payload, separators=(",", ":")).encode("utf-8")
    url = f"{base_url.rstrip('/')}/chat/completions"
    opener = build_opener(ProxyHandler({}))
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    started_epoch = time.time()
    headers_at: float | None = None
    first_byte_at: float | None = None
    first_content_at: float | None = None
    completed_at: float | None = None
    content_event_times: list[float] = []
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    request_id: str | None = None
    route_metadata = route_metadata_from_headers({})
    content_fragments: list[str] = []
    reasoning_fragments: list[str] = []
    buffer = b""

    try:
        with opener.open(request, timeout=timeout) as response:
            headers_at = time.perf_counter()
            request_id = response.headers.get("X-Tokenity-Request-ID")
            route_metadata = route_metadata_from_headers(response.headers)
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" not in content_type.lower():
                raise RuntimeError(f"Expected text/event-stream, received {content_type or 'unknown'}")
            while True:
                chunk = response.read1(64 * 1024)
                received_at = time.perf_counter()
                if not chunk:
                    break
                first_byte_at = first_byte_at or received_at
                buffer += chunk
                while b"\n" in buffer:
                    raw_line, buffer = buffer.split(b"\n", 1)
                    line = raw_line.decode("utf-8", errors="strict").rstrip("\r")
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        completed_at = received_at
                        continue
                    if not data:
                        continue
                    payload = json.loads(data)
                    usage = payload.get("usage")
                    if isinstance(usage, dict):
                        if isinstance(usage.get("completion_tokens"), int):
                            completion_tokens = usage["completion_tokens"]
                        if isinstance(usage.get("prompt_tokens"), int):
                            prompt_tokens = usage["prompt_tokens"]
                    choices = payload.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    delta = choices[0].get("delta")
                    if not isinstance(delta, dict):
                        continue
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_fragments.append(reasoning)
                        content_event_times.append(received_at)
                        first_content_at = first_content_at or received_at
                    if isinstance(content, str) and content:
                        content_fragments.append(content)
                        content_event_times.append(received_at)
                        first_content_at = first_content_at or received_at
            completed_at = completed_at or time.perf_counter()
    except Exception as exc:
        completed_at = time.perf_counter()
        if isinstance(exc, HTTPError):
            route_metadata = route_metadata_from_headers(exc.headers)
        cancelled = isinstance(exc, concurrent.futures.CancelledError) or (
            isinstance(exc, HTTPError) and exc.code == 499
        )
        return {
            "sample_index": sample_index,
            "scenario": scenario,
            "prompt_profile": prompt_profile,
            "route_policy": route_policy,
            "session_id": session_id,
            "prompt": prompt_descriptor(prompt, prompt_profile),
            "ok": False,
            "cancelled": cancelled,
            "error": f"{type(exc).__name__}: {exc}",
            "started_at": started_epoch,
            "total_seconds": completed_at - started,
            "quality_outcome": quality_outcome,
            "quality_source": quality_source,
            "route_regret": route_regret,
            **route_metadata,
        }

    intervals = [
        later - earlier
        for earlier, later in zip(content_event_times, content_event_times[1:])
    ]
    total_seconds = completed_at - started
    generation_seconds = (
        completed_at - first_content_at if first_content_at is not None else None
    )
    decode_tokens_per_second = (
        completion_tokens / generation_seconds
        if completion_tokens is not None and generation_seconds and generation_seconds > 0
        else None
    )
    timeline = fetch_request_timeline(base_url, request_id)
    prefill_tokens_per_second = None
    if timeline and prompt_tokens is not None:
        prefill_start = timeline.get("prefill_start")
        prefill_end = timeline.get("prefill_end")
        if isinstance(prefill_start, (int, float)) and isinstance(prefill_end, (int, float)):
            duration = prefill_end - prefill_start
            if duration > 0:
                prefill_tokens_per_second = prompt_tokens / duration
    return {
        "sample_index": sample_index,
        "scenario": scenario,
        "prompt_profile": prompt_profile,
        "route_policy": route_policy,
        "session_id": session_id,
        "prompt": prompt_descriptor(prompt, prompt_profile),
        "ok": True,
        "cancelled": False,
        "started_at": started_epoch,
        "request_id": request_id,
        **route_metadata,
        "headers_seconds": headers_at - started if headers_at is not None else None,
        "first_sse_byte_seconds": first_byte_at - started if first_byte_at is not None else None,
        "ttft_content_seconds": first_content_at - started if first_content_at is not None else None,
        "total_seconds": total_seconds,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_tokens_per_second": prefill_tokens_per_second,
        "decode_tokens_per_second": decode_tokens_per_second,
        "inter_content_seconds_p50": percentile(intervals, 0.50),
        "inter_content_seconds_p95": percentile(intervals, 0.95),
        "inter_content_seconds_p99": percentile(intervals, 0.99),
        "content_event_count": len(content_event_times),
        "output_characters": len("".join(content_fragments)),
        "reasoning_characters": len("".join(reasoning_fragments)),
        "server_timeline": timeline,
        "quality_outcome": quality_outcome,
        "quality_source": quality_source,
        "route_regret": route_regret,
    }


def aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [sample for sample in samples if sample.get("ok") is True]

    def metric(name: str) -> dict[str, float | None]:
        values = [float(sample[name]) for sample in successful if isinstance(sample.get(name), (int, float))]
        return {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
        }

    routed_models = [
        str(sample["routed_model"])
        for sample in successful
        if isinstance(sample.get("routed_model"), str) and sample["routed_model"]
    ]
    model_counts: dict[str, int] = {}
    for model in routed_models:
        model_counts[model] = model_counts.get(model, 0) + 1
    route_reason_counts: dict[str, int] = {}
    for sample in samples:
        reason = sample.get("route_reason")
        if isinstance(reason, str) and reason:
            route_reason_counts[reason] = route_reason_counts.get(reason, 0) + 1
    quality = {"win": 0, "tie": 0, "loss": 0, "unrated": 0}
    for sample in samples:
        outcome = sample.get("quality_outcome")
        if outcome in QUALITY_OUTCOMES and sample.get("quality_source"):
            quality[str(outcome)] += 1
        else:
            quality["unrated"] += 1
    session_samples: dict[str, list[dict[str, Any]]] = {}
    for sample in successful:
        session_id = sample.get("session_id")
        routed_model = sample.get("routed_model")
        if not isinstance(session_id, str) or not session_id:
            continue
        if not isinstance(routed_model, str) or not routed_model:
            continue
        session_samples.setdefault(session_id, []).append(sample)
    model_switches = 0
    for session in session_samples.values():
        ordered = sorted(session, key=lambda sample: int(sample.get("sample_index", 0)))
        models = [str(sample["routed_model"]) for sample in ordered]
        model_switches += sum(
            current != previous
            for previous, current in zip(models, models[1:])
        )

    return {
        "requested": len(samples),
        "successful": len(successful),
        "failed": len(samples) - len(successful),
        "cancelled": sum(sample.get("cancelled") is True for sample in samples),
        "fallback": sum(sample.get("fallback") is True for sample in samples),
        "model_switches": model_switches,
        "session_count": len(session_samples),
        "selected_models": dict(sorted(model_counts.items())),
        "route_reasons": dict(sorted(route_reason_counts.items())),
        "quality": quality,
        "route_regret": metric("route_regret"),
        "router_latency_ms": metric("router_latency_ms"),
        "queue_wait_ms": metric("queue_wait_ms"),
        "headers_seconds": metric("headers_seconds"),
        "first_sse_byte_seconds": metric("first_sse_byte_seconds"),
        "ttft_content_seconds": metric("ttft_content_seconds"),
        "total_seconds": metric("total_seconds"),
        "prefill_tokens_per_second": metric("prefill_tokens_per_second"),
        "decode_tokens_per_second": metric("decode_tokens_per_second"),
    }


def parse_vm_stat(output: str) -> dict[str, int | None]:
    page_match = re.search(r"page size of\s+(\d+)\s+bytes", output, re.IGNORECASE)
    compressed_match = re.search(
        r'"?Pages occupied by compressor"?\s*:\s*(\d+)',
        output,
        re.IGNORECASE,
    )
    page_size = int(page_match.group(1)) if page_match else None
    compressed_pages = int(compressed_match.group(1)) if compressed_match else None
    return {
        "page_size": page_size,
        "compressed_pages": compressed_pages,
        "compressed_bytes": (
            page_size * compressed_pages
            if page_size is not None and compressed_pages is not None
            else None
        ),
    }


def _binary_size(value: str, unit: str) -> int:
    multipliers = {
        "B": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
    }
    return int(float(value) * multipliers[unit.upper()])


def parse_swapusage(output: str) -> dict[str, int | None]:
    values: dict[str, int | None] = {}
    for field in ("total", "used", "free"):
        match = re.search(
            rf"\b{field}\s*=\s*([0-9.]+)([BKMGT])",
            output,
            re.IGNORECASE,
        )
        values[f"{field}_bytes"] = (
            _binary_size(match.group(1), match.group(2)) if match else None
        )
    return values


def parse_memory_pressure(output: str) -> float | None:
    match = re.search(
        r"(?:free|available)(?:\s+memory)?\s+percentage\s*:\s*([0-9.]+)%",
        output,
        re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def parse_thermal(output: str) -> dict[str, int | None]:
    def value(name: str) -> int | None:
        match = re.search(rf"\b{name}\s*=\s*(\d+)", output, re.IGNORECASE)
        return int(match.group(1)) if match else None

    return {
        "cpu_speed_limit_percent": value("CPU_Speed_Limit"),
        "scheduler_limit_percent": value("Scheduler_Limit"),
        "thermal_level": value("Thermal_Level"),
    }


def _run_command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return completed.stdout


def collect_macos_telemetry(
    *,
    system_name: str | None = None,
    command_runner: Callable[[list[str]], str] = _run_command,
) -> dict[str, Any]:
    system_name = system_name or platform.system()
    telemetry: dict[str, Any] = {
        "memory_pressure_percent": None,
        "compressed_bytes": None,
        "swap": None,
        "thermal": None,
        "sampled_at": time.time(),
        "errors": [],
    }
    if system_name != "Darwin":
        telemetry["errors"].append(
            f"macOS telemetry is unavailable on {system_name}."
        )
        return telemetry

    commands = {
        "memory_pressure": ["/usr/bin/memory_pressure", "-Q"],
        "vm_stat": ["/usr/bin/vm_stat"],
        "swap": ["/usr/sbin/sysctl", "vm.swapusage"],
        "thermal": ["/usr/bin/pmset", "-g", "therm"],
    }
    outputs: dict[str, str] = {}
    for name, command in commands.items():
        try:
            outputs[name] = command_runner(command)
        except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
            telemetry["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
    if "memory_pressure" in outputs:
        telemetry["memory_pressure_percent"] = parse_memory_pressure(
            outputs["memory_pressure"]
        )
        if telemetry["memory_pressure_percent"] is None:
            telemetry["errors"].append("memory_pressure: value unavailable")
    if "vm_stat" in outputs:
        telemetry["compressed_bytes"] = parse_vm_stat(outputs["vm_stat"])[
            "compressed_bytes"
        ]
        if telemetry["compressed_bytes"] is None:
            telemetry["errors"].append("vm_stat: compressor value unavailable")
    if "swap" in outputs:
        telemetry["swap"] = parse_swapusage(outputs["swap"])
        if any(value is None for value in telemetry["swap"].values()):
            telemetry["errors"].append("swap: one or more values unavailable")
    if "thermal" in outputs:
        telemetry["thermal"] = parse_thermal(outputs["thermal"])
        if all(value is None for value in telemetry["thermal"].values()):
            telemetry["errors"].append("thermal: values unavailable")
    return telemetry


def collect_agent_memory(base_url: str) -> dict[str, Any]:
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(Request(agent_status_url(base_url)), timeout=3) as response:
            payload = json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"memory": None, "error": f"{type(exc).__name__}: {exc}"}
    memory = payload.get("memory") if isinstance(payload, dict) else None
    if not isinstance(memory, dict):
        return {"memory": None, "error": "Agent response did not include memory telemetry."}
    return {"memory": memory, "error": None}


def collect_telemetry(base_url: str) -> dict[str, Any]:
    return {
        "macos": collect_macos_telemetry(),
        "agent": collect_agent_memory(base_url),
    }


def validate_matrix(matrix: dict[str, Any]) -> None:
    if matrix.get("schema_version") != 1:
        raise ValueError("matrix schema_version must be 1")
    if not isinstance(matrix.get("base_url"), str) or not matrix["base_url"].strip():
        raise ValueError("matrix base_url is required")
    repeats = matrix.get("repeats", 1)
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("matrix repeats must be a positive integer")

    profiles = matrix.get("prompt_profiles")
    if not isinstance(profiles, dict):
        raise ValueError("matrix prompt_profiles must be an object")
    missing_profiles = PROMPT_PROFILES - set(profiles)
    if missing_profiles:
        raise ValueError(
            f"matrix is missing prompt profiles: {sorted(missing_profiles)}"
        )
    for name, profile in profiles.items():
        if not isinstance(profile, dict) or not isinstance(profile.get("prompt"), str):
            raise ValueError(f"prompt profile {name} requires a prompt string")
        max_tokens = profile.get("max_tokens")
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError(f"prompt profile {name} max_tokens must be positive")

    scenarios = matrix.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("matrix scenarios must be a list")
    modes = {
        scenario.get("mode")
        for scenario in scenarios
        if isinstance(scenario, dict)
    }
    missing_modes = MATRIX_MODES - modes
    if missing_modes:
        raise ValueError(f"matrix is missing scenarios: {sorted(missing_modes)}")
    names: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("every matrix scenario must be an object")
        name = scenario.get("name")
        mode = scenario.get("mode")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("scenario names must be non-empty and unique")
        names.add(name)
        if mode not in MATRIX_MODES:
            raise ValueError(f"unsupported matrix mode: {mode}")
        models = _scenario_models(scenario)
        if not models:
            raise ValueError(f"scenario {name} requires at least one model")
        concurrency = scenario.get(
            "concurrency",
            2 if mode == "resident-concurrent" else 1,
        )
        if not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError(f"scenario {name} concurrency must be positive")
        if mode == "resident-serial" and concurrency != 1:
            raise ValueError("resident-serial concurrency must be 1")
        if mode == "resident-concurrent" and concurrency < 2:
            raise ValueError("resident-concurrent concurrency must be at least 2")
        route_policy = scenario.get("route_policy")
        if route_policy is not None and route_policy not in {"fast", "balanced", "quality"}:
            raise ValueError(
                f"scenario {name} route_policy must be fast, balanced, or quality"
            )
        annotations = scenario.get("quality_by_profile", {})
        if not isinstance(annotations, dict):
            raise ValueError(f"scenario {name} quality_by_profile must be an object")
        if "quality" in scenario:
            if not isinstance(scenario["quality"], dict):
                raise ValueError(f"scenario {name} quality must be an object")
            quality_annotation(scenario["quality"])
        for profile_name, annotation in annotations.items():
            if profile_name not in profiles:
                raise ValueError(f"unknown quality prompt profile: {profile_name}")
            if not isinstance(annotation, dict):
                raise ValueError(
                    f"scenario {name} quality annotation for {profile_name} must be an object"
                )
            quality_annotation(annotation)


def _scenario_models(scenario: dict[str, Any]) -> list[str]:
    model = scenario.get("model")
    models = scenario.get("models")
    if isinstance(model, str) and model.strip():
        return [model.strip()]
    if isinstance(models, list):
        return [str(item).strip() for item in models if str(item).strip()]
    return []


def _scenario_quality(
    scenario: dict[str, Any],
    profile_name: str,
) -> dict[str, Any]:
    by_profile = scenario.get("quality_by_profile") or {}
    annotation = by_profile.get(profile_name, scenario.get("quality"))
    return quality_annotation(annotation)


def _sanitized_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    scenarios = []
    for scenario in matrix["scenarios"]:
        scenarios.append(
            {
                "name": scenario["name"],
                "mode": scenario["mode"],
                "models": _scenario_models(scenario),
                "concurrency": scenario.get(
                    "concurrency",
                    2 if scenario["mode"] == "resident-concurrent" else 1,
                ),
                "timeout": scenario.get("timeout", matrix.get("timeout", 600)),
                "route_policy": scenario.get("route_policy"),
                "quality": quality_annotation(scenario.get("quality")),
                "quality_by_profile": {
                    profile_name: quality_annotation(annotation)
                    for profile_name, annotation in sorted(
                        (scenario.get("quality_by_profile") or {}).items()
                    )
                },
            }
        )
    return {
        "schema_version": matrix["schema_version"],
        "base_url": matrix["base_url"],
        "repeats": matrix.get("repeats", 1),
        "prompt_profiles": {
            name: {
                **prompt_descriptor(profile["prompt"], name),
                "max_tokens": profile["max_tokens"],
            }
            for name, profile in sorted(matrix["prompt_profiles"].items())
        },
        "scenarios": scenarios,
    }


def _environment() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "recorded_at": time.time(),
    }


def _matrix_session_id(
    scenario_name: str,
    repeat_index: int,
    model_lane: int,
) -> str:
    identity = f"{scenario_name}\0{repeat_index}\0{model_lane}".encode("utf-8")
    return f"benchmark-{hashlib.sha256(identity).hexdigest()[:20]}"


def execute_matrix(
    matrix: dict[str, Any],
    *,
    sample_runner: Callable[..., dict[str, Any]] = run_sample,
    telemetry_collector: Callable[[str], dict[str, Any]] = collect_telemetry,
) -> dict[str, Any]:
    validate_matrix(matrix)
    base_url = matrix["base_url"]
    repeats = matrix.get("repeats", 1)
    all_samples: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    sample_index = 0
    global_before = telemetry_collector(base_url)

    for scenario in matrix["scenarios"]:
        models = _scenario_models(scenario)
        concurrency = scenario.get(
            "concurrency",
            2 if scenario["mode"] == "resident-concurrent" else 1,
        )
        for profile_name in ("short", "medium", "long"):
            profile = matrix["prompt_profiles"][profile_name]
            quality = _scenario_quality(scenario, profile_name)
            telemetry_before = telemetry_collector(base_url)
            jobs: list[dict[str, Any]] = []
            for repeat_index in range(repeats):
                for model_lane, model in enumerate(models):
                    jobs.append(
                        {
                            "base_url": base_url,
                            "model": model,
                            "prompt": profile["prompt"],
                            "max_tokens": profile["max_tokens"],
                            "timeout": float(scenario.get("timeout", matrix.get("timeout", 600))),
                            "sample_index": sample_index,
                            "scenario": scenario["name"],
                            "prompt_profile": profile_name,
                            "route_policy": scenario.get("route_policy"),
                            "session_id": _matrix_session_id(
                                scenario["name"],
                                repeat_index,
                                model_lane,
                            ),
                            **quality,
                        }
                    )
                    sample_index += 1
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(concurrency, len(jobs))
            ) as executor:
                futures = [executor.submit(sample_runner, **job) for job in jobs]
                samples = [future.result() for future in futures]
            telemetry_after = telemetry_collector(base_url)
            all_samples.extend(samples)
            results.append(
                {
                    "scenario": scenario["name"],
                    "mode": scenario["mode"],
                    "prompt_profile": profile_name,
                    "prompt": prompt_descriptor(profile["prompt"], profile_name),
                    "models": models,
                    "concurrency": concurrency,
                    "aggregate": aggregate(samples),
                    "telemetry": {
                        "before": telemetry_before,
                        "after": telemetry_after,
                    },
                    "samples": samples,
                }
            )

    return {
        "schema_version": 2,
        "benchmark_mode": "matrix",
        "matrix": _sanitized_matrix(matrix),
        "environment": _environment(),
        "limitations": [
            "The sampler never loads models; prepare and verify the requested resident pool before running the matrix.",
            "Telemetry is sampled before and after each case; it does not capture an in-case peak.",
            "Quality and route regret are reported only when supplied with an explicit source.",
            "Null telemetry fields mean the platform or Agent did not expose that measurement.",
        ],
        "telemetry": {
            "before": global_before,
            "after": telemetry_collector(base_url),
        },
        "aggregate": aggregate(all_samples),
        "results": results,
    }


def load_matrix(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load benchmark matrix: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("benchmark matrix root must be an object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", help="OpenAI base URL ending in /v1")
    parser.add_argument("--model")
    parser.add_argument("--matrix", type=Path, help="Reproducible JSON comparison matrix")
    parser.add_argument("--prompt", default="Reply with a concise explanation of unified memory.")
    parser.add_argument("--prompt-profile", default="manual")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--label", default="manual")
    parser.add_argument("--route-policy", choices=["fast", "balanced", "quality"])
    parser.add_argument("--session-id")
    parser.add_argument("--quality-outcome", choices=sorted(QUALITY_OUTCOMES))
    parser.add_argument("--quality-source")
    parser.add_argument("--route-regret", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _write_result(result: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(output)
    print(rendered)


def main() -> int:
    args = parse_args()
    if args.matrix is not None:
        matrix = load_matrix(args.matrix)
        if args.base_url:
            matrix = dict(matrix)
            matrix["base_url"] = args.base_url
        result = execute_matrix(matrix)
        _write_result(result, args.output)
        return 0 if result["aggregate"]["failed"] == 0 else 1

    if not args.base_url or not args.model:
        raise SystemExit("--base-url and --model are required unless --matrix is used")
    if args.repeats < 1 or args.concurrency < 1 or args.max_tokens < 1:
        raise SystemExit("repeats, concurrency, and max-tokens must be positive")
    try:
        quality = quality_annotation(
            {
                "outcome": args.quality_outcome,
                "source": args.quality_source,
                "route_regret": args.route_regret,
            }
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    telemetry_before = collect_telemetry(args.base_url)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_sample,
                base_url=args.base_url,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                sample_index=index,
                scenario=args.label,
                prompt_profile=args.prompt_profile,
                route_policy=args.route_policy,
                session_id=args.session_id,
                **quality,
            )
            for index in range(args.repeats)
        ]
        samples = [future.result() for future in futures]
    result = {
        "schema_version": 2,
        "benchmark_mode": "single",
        "label": args.label,
        "configuration": {
            "base_url": args.base_url,
            "model": args.model,
            "prompt": prompt_descriptor(args.prompt, args.prompt_profile),
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "concurrency": args.concurrency,
            "timeout": args.timeout,
            "route_policy": args.route_policy,
            "session_id": args.session_id,
            **quality,
        },
        "environment": _environment(),
        "limitations": [
            "The sampler never loads models; prepare and verify the requested resident pool before running this benchmark.",
            "Telemetry is sampled before and after the run; it does not capture an in-run peak.",
            "Quality and route regret are reported only when supplied with an explicit source.",
            "Null telemetry fields mean the platform or Agent did not expose that measurement.",
        ],
        "telemetry": {
            "before": telemetry_before,
            "after": collect_telemetry(args.base_url),
        },
        "aggregate": aggregate(samples),
        "samples": samples,
    }
    _write_result(result, args.output)
    return 0 if result["aggregate"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
