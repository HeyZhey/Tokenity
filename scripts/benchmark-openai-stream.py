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
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener


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
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        separators=(",", ":"),
    ).encode("utf-8")
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
    routed_instance_id: str | None = None
    content_fragments: list[str] = []
    reasoning_fragments: list[str] = []
    buffer = b""

    try:
        with opener.open(request, timeout=timeout) as response:
            headers_at = time.perf_counter()
            request_id = response.headers.get("X-Tokenity-Request-ID")
            routed_instance_id = response.headers.get("X-Tokenity-Instance-ID")
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
        return {
            "sample_index": sample_index,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "started_at": started_epoch,
            "total_seconds": completed_at - started,
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
        "ok": True,
        "started_at": started_epoch,
        "request_id": request_id,
        "routed_instance_id": routed_instance_id,
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

    return {
        "requested": len(samples),
        "successful": len(successful),
        "failed": len(samples) - len(successful),
        "headers_seconds": metric("headers_seconds"),
        "first_sse_byte_seconds": metric("first_sse_byte_seconds"),
        "ttft_content_seconds": metric("ttft_content_seconds"),
        "total_seconds": metric("total_seconds"),
        "prefill_tokens_per_second": metric("prefill_tokens_per_second"),
        "decode_tokens_per_second": metric("decode_tokens_per_second"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="OpenAI base URL ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Reply with a concise explanation of unified memory.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--label", default="manual")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 1 or args.concurrency < 1 or args.max_tokens < 1:
        raise SystemExit("repeats, concurrency, and max-tokens must be positive")
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
            )
            for index in range(args.repeats)
        ]
        samples = [future.result() for future in futures]
    result = {
        "schema_version": 1,
        "label": args.label,
        "configuration": {
            "base_url": args.base_url,
            "model": args.model,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "concurrency": args.concurrency,
            "timeout": args.timeout,
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "recorded_at": time.time(),
        },
        "aggregate": aggregate(samples),
        "samples": samples,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(rendered)
    return 0 if result["aggregate"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
