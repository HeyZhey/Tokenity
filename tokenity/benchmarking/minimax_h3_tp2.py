from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Protocol
from urllib.request import ProxyHandler, Request, build_opener


FIXED_REQUEST: dict[str, object] = {
    "prompt": "A cinematic tracking shot of a paper boat crossing a rainy neon street",
    "width": 512,
    "height": 256,
    "num_frames": 124,
    "steps": 28,
    "seed": 42,
    "fast": False,
    "stream": True,
}


class Transport(Protocol):
    def get_json(self, url: str, timeout: float = 10.0) -> dict[str, object]: ...

    def post_json(
        self, url: str, payload: dict[str, object], timeout: float = 10.0
    ) -> dict[str, object]: ...

    def post_sse(
        self, url: str, payload: dict[str, object], *, timeout: float
    ) -> tuple[int, Iterable[bytes]]: ...


class HttpTransport:
    def __init__(self) -> None:
        self._opener = build_opener(ProxyHandler({}))

    def get_json(self, url: str, timeout: float = 10.0) -> dict[str, object]:
        with self._opener.open(Request(url), timeout=timeout) as response:
            return json.loads(response.read())

    def post_json(
        self, url: str, payload: dict[str, object], timeout: float = 10.0
    ) -> dict[str, object]:
        request = Request(
            url,
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._opener.open(request, timeout=timeout) as response:
            return json.loads(response.read())

    def post_sse(
        self, url: str, payload: dict[str, object], *, timeout: float
    ) -> tuple[int, Iterable[bytes]]:
        request = Request(
            url,
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        response = self._opener.open(request, timeout=timeout)

        def chunks() -> Iterable[bytes]:
            try:
                while chunk := response.read(1024 * 1024):
                    yield chunk
            finally:
                response.close()

        return response.status, chunks()


@dataclass(frozen=True)
class BenchmarkConfig:
    output_dir: Path
    test_id: str
    coordinator_agent: str
    worker_agent: str
    nodes: list[dict[str, object]]
    model: str
    binary: str
    optimization_profile: str = "stock-qmm"
    runs: int = 4
    instance_id: str = field(default_factory=lambda: f"h3-bench-{uuid.uuid4().hex[:12]}")
    operation_id: str = field(default_factory=lambda: f"h3-bench-op-{uuid.uuid4().hex[:12]}")
    port: int = 11_260
    starting_port: int = 30_096
    minimum_free_disk_bytes: int = 2 * 1024**3
    lease_seconds: float = 300.0
    heartbeat_interval_seconds: float = 30.0
    request_timeout_seconds: float = 900.0
    readiness_timeout_seconds: float = 300.0
    dry_run_only: bool = False
    request: dict[str, object] = field(default_factory=lambda: dict(FIXED_REQUEST))


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_sse_and_archive(
    chunks: Iterable[bytes], raw_path: Path, artifact_prefix: Path
) -> dict[str, object]:
    progress = 0
    complete: dict[str, object] | None = None
    pending = bytearray()
    with raw_path.open("wb") as raw:
        for chunk in chunks:
            raw.write(chunk)
            pending.extend(chunk)
            while b"\n" in pending:
                line, _, rest = pending.partition(b"\n")
                pending = bytearray(rest)
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("type") == "progress":
                    if complete is not None:
                        raise RuntimeError("MiniMax H3 emitted progress after complete.")
                    progress += 1
                elif event.get("type") == "complete":
                    if complete is not None:
                        raise RuntimeError("MiniMax H3 emitted more than one complete event.")
                    complete = event
        raw.flush()
    if complete is None:
        raise RuntimeError("MiniMax H3 SSE stream ended without a complete event.")
    video = base64.b64decode(str(complete["data"]), validate=True)
    audio = base64.b64decode(str(complete.get("audio_data", "")), validate=True)
    frames = int(complete["frames"])
    width = int(complete["width"])
    height = int(complete["height"])
    expected_video_bytes = frames * width * height * 3
    if len(video) != expected_video_bytes:
        raise RuntimeError(
            f"RGB payload length mismatch: expected {expected_video_bytes}, got {len(video)}."
        )
    video_path = artifact_prefix.with_suffix(".rgb")
    audio_path = artifact_prefix.with_suffix(".pcm")
    video_path.write_bytes(video)
    audio_path.write_bytes(audio)
    return {
        "progress_events": progress,
        "complete": {
            key: value
            for key, value in complete.items()
            if key not in {"data", "audio_data"}
        },
        "video_bytes": len(video),
        "video_sha256": hashlib.sha256(video).hexdigest(),
        "video_path": str(video_path),
        "audio_bytes": len(audio),
        "audio_sha256": hashlib.sha256(audio).hexdigest(),
        "audio_path": str(audio_path),
        "sse_path": str(raw_path),
    }


def _preflight(config: BenchmarkConfig, transport: Transport) -> dict[str, object]:
    nodes: dict[str, object] = {}
    for rank, node in enumerate(config.nodes):
        role = f"rank-{rank}"
        fallback_url = config.coordinator_agent if rank == 0 else config.worker_agent
        url = str(node.get("agent_url") or fallback_url).rstrip("/")
        health = transport.get_json(f"{url}/health", 10.0)
        info = transport.get_json(f"{url}/v1/node/info", 10.0)
        if health.get("status") != "healthy":
            raise RuntimeError(f"{role} Node Agent is not healthy.")
        contract = info.get("agent_contract")
        capabilities = contract.get("capabilities") if isinstance(contract, dict) else []
        if "minimax_h3_video" not in capabilities:
            raise RuntimeError(f"{role} does not advertise minimax_h3_video.")
        disk = info.get("disk")
        free = disk.get("free_bytes") if isinstance(disk, dict) else None
        if not isinstance(free, int) or free < config.minimum_free_disk_bytes:
            raise RuntimeError(
                f"{role} disk gate failed: need {config.minimum_free_disk_bytes} free bytes, got {free}."
            )
        rdma = info.get("rdma")
        if len(config.nodes) > 1 and (
            not isinstance(rdma, dict) or rdma.get("rdma_enabled") is not True
        ):
            raise RuntimeError(f"{role} RDMA is not enabled.")
        nodes[role] = {"agent_url": url, "health": health, "info": info}
    return {"sampled_at_epoch": time.time(), "nodes": nodes}


def run_benchmark(
    config: BenchmarkConfig, *, transport: Transport | None = None
) -> dict[str, object]:
    if config.runs < 1:
        raise ValueError("runs must be at least one")
    if len(config.nodes) not in {1, 2}:
        raise ValueError("MiniMax H3 benchmark requires one or two typed nodes")
    transport = transport or HttpTransport()
    root = config.output_dir / config.test_id
    root.mkdir(parents=True, exist_ok=False)
    report: dict[str, object] = {
        "schema_version": 1,
        "test_id": config.test_id,
        "created_at_epoch": time.time(),
        "config": asdict(config) | {"output_dir": str(config.output_dir)},
        "request": dict(config.request),
        "runs": [],
    }
    preflight = _preflight(config, transport)
    report["preflight"] = preflight
    _write_json(root / "preflight.json", preflight)

    start_payload: dict[str, object] = {
        "model": config.model,
        "binary": config.binary,
        "nodes": config.nodes,
        "connection_mode": "ring" if len(config.nodes) == 1 else "jaccl-ring",
        "port": config.port,
        "starting_port": config.starting_port,
        "api_identifier": config.instance_id,
        "instance_id": config.instance_id,
        "operation_id": config.operation_id,
        "optimization_profile": config.optimization_profile,
        "minimum_free_disk_bytes": config.minimum_free_disk_bytes,
        "lease_seconds": config.lease_seconds,
        "dry_run": True,
    }
    coordinator = config.coordinator_agent.rstrip("/")
    start_url = f"{coordinator}/v1/node/start-minimax-h3-video"
    dry_run = transport.post_json(start_url, start_payload, 60.0)
    report["dry_run"] = dry_run
    _write_json(root / "dry-run.json", dry_run)
    if config.dry_run_only:
        _write_json(root / "result.json", report)
        return report

    launch_attempted = False
    heartbeat_stop = threading.Event()
    heartbeat_errors: list[str] = []
    failure: Exception | None = None
    start_payload["dry_run"] = False
    try:
        # A timeout can happen after the Agent accepted the instance. Always
        # attempt a precise instance stop once the mutating request was sent.
        launch_attempted = True
        launch = transport.post_json(start_url, start_payload, config.readiness_timeout_seconds)
        report["launch"] = launch
        _write_json(root / "launch.json", launch)

        def heartbeat() -> None:
            while not heartbeat_stop.wait(config.heartbeat_interval_seconds):
                try:
                    transport.post_json(
                        f"{coordinator}/v1/node/heartbeat",
                        {"instance_id": config.instance_id, "ttl_seconds": config.lease_seconds},
                        10.0,
                    )
                except Exception as exc:  # retained in the final evidence
                    heartbeat_errors.append(str(exc))

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        readiness_deadline = time.monotonic() + config.readiness_timeout_seconds
        while True:
            quorum = transport.get_json(
                f"{coordinator}/v1/node/instances/{config.instance_id}/quorum", 15.0
            )
            if quorum.get("ready") is True:
                report["readiness"] = quorum
                break
            if time.monotonic() >= readiness_deadline:
                raise RuntimeError(f"MiniMax H3 readiness deadline exceeded: {quorum}")
            time.sleep(1.0)
        _write_json(root / "readiness.json", report["readiness"])

        request_payload = dict(config.request)
        request_payload["tokenity_instance_id"] = config.instance_id
        for index in range(config.runs):
            started = time.monotonic()
            status, chunks = transport.post_sse(
                f"{coordinator}/v1/video/generations",
                request_payload,
                timeout=config.request_timeout_seconds,
            )
            if status != 200:
                raise RuntimeError(f"MiniMax H3 request returned HTTP {status}.")
            prefix = root / f"run-{index + 1:02d}"
            parsed = _parse_sse_and_archive(chunks, prefix.with_suffix(".sse"), prefix)
            # H3 reports one event per denoiser forward. Prompt plus
            # video/audio decode add three more progress events.
            expected_progress = int(config.request["steps"]) + 3
            if parsed["progress_events"] != expected_progress:
                raise RuntimeError(
                    f"Expected {expected_progress} progress events, got {parsed['progress_events']}."
                )
            rank_snapshots = {
                "rank-0": transport.get_json(
                    f"{coordinator}/v1/node/instances/{config.instance_id}", 15.0
                )
            }
            if len(config.nodes) == 2:
                rank_snapshots["rank-1"] = transport.get_json(
                    f"{config.worker_agent.rstrip('/')}/v1/node/instances/{config.instance_id}",
                    15.0,
                )
            run = {
                "run": index + 1,
                "temperature": "cold" if index == 0 else "warm",
                "http_status": status,
                "end_to_end_seconds": time.monotonic() - started,
                **parsed,
                "rank_snapshots": rank_snapshots,
            }
            report["runs"].append(run)  # type: ignore[union-attr]
            _write_json(root / f"run-{index + 1:02d}.json", run)

        runs = report["runs"]
        video_hashes = [str(item["video_sha256"]) for item in runs]  # type: ignore[index]
        audio_hashes = [str(item["audio_sha256"]) for item in runs]  # type: ignore[index]
        warm = [float(item["end_to_end_seconds"]) for item in runs[1:]]  # type: ignore[index]
        report["determinism"] = {
            "video_hashes_identical": len(set(video_hashes)) == 1,
            "audio_hashes_identical": len(set(audio_hashes)) == 1,
            "video_sha256": video_hashes[0],
            "audio_sha256": audio_hashes[0],
        }
        if len(set(video_hashes)) != 1 or len(set(audio_hashes)) != 1:
            raise RuntimeError("Fixed-seed MiniMax H3 outputs were not deterministic.")
        report["aggregate"] = {
            "warm_count": len(warm),
            "warm_end_to_end_mean_seconds": statistics.mean(warm) if warm else None,
            "warm_end_to_end_median_seconds": statistics.median(warm) if warm else None,
        }
    except Exception as exc:
        failure = exc
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        heartbeat_stop.set()
        if "heartbeat_thread" in locals():
            heartbeat_thread.join(timeout=5.0)
        report["heartbeat_errors"] = heartbeat_errors
        shutdown: dict[str, object] = {"requested": False}
        if launch_attempted:
            try:
                shutdown["response"] = transport.post_json(
                    f"{coordinator}/v1/node/instances/{config.instance_id}/stop",
                    {"timeout": 10.0},
                    30.0,
                )
                shutdown["requested"] = True
            except Exception as exc:
                shutdown["error"] = str(exc)
            snapshot_agents = [("rank-0", coordinator)]
            if len(config.nodes) == 2:
                snapshot_agents.append(("rank-1", config.worker_agent.rstrip("/")))
            for role, url in snapshot_agents:
                try:
                    shutdown[role] = transport.get_json(
                        f"{url}/v1/node/instances/{config.instance_id}", 15.0
                    )
                    shutdown[f"{role}_node_info_after"] = transport.get_json(
                        f"{url}/v1/node/info", 15.0
                    )
                except Exception as exc:
                    shutdown[f"{role}_snapshot_error"] = str(exc)
        report["shutdown"] = shutdown
        _write_json(root / "result.json", report)
    if failure is not None:
        raise RuntimeError(f"MiniMax H3 benchmark failed; evidence: {root}: {failure}") from failure
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and archive the fixed MiniMax H3 single- or two-Mac cold/warm benchmark."
    )
    parser.add_argument("--coordinator-agent", required=True)
    parser.add_argument("--worker-agent")
    parser.add_argument("--nodes-json", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-id", required=True)
    parser.add_argument(
        "--optimization-profile",
        choices=("baseline", "block-fusions", "stock-qmm"),
        default="stock-qmm",
    )
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--port", type=int, default=11_260)
    parser.add_argument("--starting-port", type=int, default=30_096)
    parser.add_argument("--minimum-free-disk-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    nodes = json.loads(args.nodes_json.read_text(encoding="utf-8"))
    if not isinstance(nodes, list):
        parser.error("--nodes-json must contain a JSON array")
    config = BenchmarkConfig(
        output_dir=args.output_dir,
        test_id=args.test_id,
        coordinator_agent=args.coordinator_agent,
        worker_agent=args.worker_agent or args.coordinator_agent,
        nodes=nodes,
        model=args.model,
        binary=args.binary,
        optimization_profile=args.optimization_profile,
        runs=args.runs,
        port=args.port,
        starting_port=args.starting_port,
        minimum_free_disk_bytes=args.minimum_free_disk_bytes,
        dry_run_only=args.dry_run,
    )
    result = run_benchmark(config)
    print(json.dumps({"test_id": result["test_id"], "result": str(args.output_dir / args.test_id / "result.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
