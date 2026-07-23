#!/usr/bin/env python3
"""Read-only HTTP preflight for Tokenity Node Agents and MLX data-plane fixtures."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener


DEFAULT_NODES = [
    {
        "id": "mac-a",
        "agent_url": "http://192.168.5.23:9100",
        "lan_ip": "192.168.5.23",
        "rdma_device": "rdma_en4",
        "rdma_ip": "192.168.0.1",
    },
    {
        "id": "mac-b",
        "agent_url": "http://192.168.5.75:9100",
        "lan_ip": "192.168.5.75",
        "rdma_device": "rdma_en5",
        "rdma_ip": "192.168.0.2",
    },
]


def version_release(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    release = value.split("+", 1)[0].split("-", 1)[0]
    parts: list[int] = []
    for raw in release.split("."):
        match = re.match(r"(\d+)", raw)
        if match is None:
            break
        parts.append(int(match.group(1)))
    return tuple(parts)


def fetch_json(url: str, timeout: float) -> dict[str, Any]:
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(Request(url), timeout=timeout) as response:
            payload = json.loads(response.read())
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Node Agent is unreachable at {url}: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    return payload


def evaluate_node(
    fixture: dict[str, str],
    info: dict[str, Any],
    models: dict[str, Any],
    *,
    model_path: str,
    python_path: str,
) -> dict[str, Any]:
    issues: list[str] = []
    if info.get("python_path") != python_path:
        issues.append(f"Runtime Python mismatch: {info.get('python_path') or 'unknown'}")
    if info.get("mlx_version") != "0.31.2":
        issues.append(f"mlx 0.31.2 required: {info.get('mlx_version') or 'unknown'}")
    if version_release(info.get("mlx_lm_version")) < (0, 31, 3):
        issues.append(f"mlx-lm >= 0.31.3 required: {info.get('mlx_lm_version') or 'unknown'}")
    if info.get("architecture") not in {"arm64", "arm64e"}:
        issues.append(f"Apple Silicon architecture required: {info.get('architecture') or 'unknown'}")
    ips = info.get("ips") if isinstance(info.get("ips"), list) else []
    if fixture["lan_ip"] not in ips:
        issues.append(f"Expected LAN IP {fixture['lan_ip']} is not advertised")

    rdma = info.get("rdma") if isinstance(info.get("rdma"), dict) else {}
    devices = rdma.get("rdma_devices") if isinstance(rdma.get("rdma_devices"), list) else []
    states = rdma.get("rdma_port_state") if isinstance(rdma.get("rdma_port_state"), dict) else {}
    if fixture["rdma_device"] not in devices:
        issues.append(f"Expected RDMA device {fixture['rdma_device']} is missing")
    if states.get(fixture["rdma_device"]) != "active":
        issues.append(f"RDMA device {fixture['rdma_device']} is not active")
    if rdma.get("thunderbolt_ip") != fixture["rdma_ip"]:
        issues.append(
            f"Expected RDMA IP {fixture['rdma_ip']}, found {rdma.get('thunderbolt_ip') or 'unknown'}"
        )

    inventory = models.get("models") if isinstance(models.get("models"), list) else []
    model = next((item for item in inventory if isinstance(item, dict) and item.get("path") == model_path), None)
    if model is None:
        issues.append(f"Model is unavailable: {model_path}")
    elif not model.get("revision"):
        issues.append("Model revision is unavailable")
    return {
        "id": fixture["id"],
        "agent_url": fixture["agent_url"],
        "ok": not issues,
        "issues": issues,
        "observed": {
            "node_id": info.get("node_id"),
            "hostname": info.get("hostname"),
            "ips": ips,
            "python_path": info.get("python_path"),
            "mlx_version": info.get("mlx_version"),
            "mlx_lm_version": info.get("mlx_lm_version"),
            "tokenity_version": info.get("tokenity_version"),
            "tokenity_code_revision": info.get("tokenity_code_revision"),
            "rdma": rdma,
            "model_revision": model.get("revision") if isinstance(model, dict) else None,
            "model_size_bytes": model.get("size_bytes") if isinstance(model, dict) else None,
        },
    }


def parse_node(value: str) -> dict[str, str]:
    fields = value.split(",")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError("node must be id,agent_url,lan_ip,rdma_device,rdma_ip")
    return dict(zip(("id", "agent_url", "lan_ip", "rdma_device", "rdma_ip"), fields))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", action="append", type=parse_node, help="Override default A/B fixture")
    parser.add_argument(
        "--model-path",
        default="/Users/Shared/TokenityModels/Qwen3.5-122B-A10B-4bit",
    )
    parser.add_argument(
        "--python-path",
        default="/Users/Shared/TokenityRuntime/current/.venv/bin/python",
    )
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixtures = args.node or DEFAULT_NODES
    results = []
    for fixture in fixtures:
        try:
            info = fetch_json(f"{fixture['agent_url']}/v1/node/info", args.timeout)
            query = urlencode({"root": str(Path(args.model_path).parent)})
            models = fetch_json(f"{fixture['agent_url']}/v1/node/models?{query}", args.timeout)
            result = evaluate_node(
                fixture,
                info,
                models,
                model_path=args.model_path,
                python_path=args.python_path,
            )
        except Exception as exc:
            result = {
                "id": fixture["id"],
                "agent_url": fixture["agent_url"],
                "ok": False,
                "issues": [f"{type(exc).__name__}: {exc}"],
                "observed": {},
            }
        results.append(result)

    code_revisions = {
        result["observed"].get("tokenity_code_revision")
        for result in results
        if result["observed"].get("tokenity_code_revision")
    }
    model_revisions = {
        result["observed"].get("model_revision")
        for result in results
        if result["observed"].get("model_revision")
    }
    cluster_issues: list[str] = []
    if len(code_revisions) != 1:
        cluster_issues.append("Node Agent code revisions are missing or inconsistent")
    if len(model_revisions) != 1:
        cluster_issues.append("Model revisions are missing or inconsistent")
    payload = {
        "schema_version": 1,
        "recorded_at": time.time(),
        "model_path": args.model_path,
        "python_path": args.python_path,
        "ok": all(result["ok"] for result in results) and not cluster_issues,
        "cluster_issues": cluster_issues,
        "nodes": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
