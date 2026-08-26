from __future__ import annotations

import base64
import json
from pathlib import Path

from tokenity.benchmarking.minimax_h3_tp2 import (
    BenchmarkConfig,
    _build_parser,
    run_benchmark,
)


class FakeTransport:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, object]]] = []
        self.sse_calls = 0

    def get_json(self, url: str, timeout: float = 10.0) -> dict[str, object]:
        if url.endswith("/health"):
            return {"status": "healthy"}
        if url.endswith("/v1/node/info"):
            return {
                "hostname": "fixture",
                "architecture": "arm64",
                "macos_version": "26.5.1",
                "tokenity_code_revision": "code",
                "agent_contract": {"capabilities": ["minimax_h3_video"]},
                "disk": {"free_bytes": 100 * 1024**3},
                "memory": {"total_bytes": 512 * 1024**3},
                "rdma": {"rdma_enabled": True, "rdma_devices": ["rdma_en4"]},
            }
        if url.endswith("/quorum"):
            return {"ready": True, "rank_quorum": "2/2", "ranks": []}
        if "/v1/node/instances/" in url:
            return {
                "instance": {"state": "ready", "active_request_count": 0},
                "process": {"state": "running", "log_tail": "sampling done in 10 ms"},
                "runtime": {"mlx_peak_bytes": 1234},
            }
        raise AssertionError(url)

    def post_json(
        self, url: str, payload: dict[str, object], timeout: float = 10.0
    ) -> dict[str, object]:
        self.posts.append((url, payload))
        if url.endswith("/v1/node/start-minimax-h3-video"):
            if payload["dry_run"]:
                return {"dry_run": True, "launch_plan": {"execution_mode": "tp2"}}
            return {
                "dry_run": False,
                "instance_id": payload["instance_id"],
                "instance": {
                    "state": "ready",
                    "readiness_evidence": {
                        "rank_quorum": "2/2",
                        "runtime_fingerprint": {"contract_sha256": "a" * 64},
                    },
                },
            }
        if url.endswith("/heartbeat"):
            return {"status": "ok"}
        if url.endswith("/stop"):
            return {"instance": {"state": "stopped"}}
        raise AssertionError(url)

    def post_sse(
        self,
        url: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> tuple[int, list[bytes]]:
        self.sse_calls += 1
        video = bytes(range(24))
        audio = b"\x00\x01\x02\x03"
        events = [
            {
                "type": "progress",
                "stage": "Generating" if index else "Encoding prompt",
                "step": index,
                "total": 28,
            }
            for index in range(31)
        ] + [
            {
                "type": "complete",
                "format": "rgb8",
                "frames": 1,
                "width": 4,
                "height": 2,
                "data": base64.b64encode(video).decode(),
                "audio_format": "pcm_s16le",
                "audio_channels": 2,
                "audio_sample_rate": 32000,
                "audio_data": base64.b64encode(audio).decode(),
            },
        ]
        return 200, [("data: " + json.dumps(event) + "\n\n").encode() for event in events]


def test_benchmark_default_profile_is_stock_fused_qmm(tmp_path: Path):
    config = BenchmarkConfig(
        output_dir=tmp_path,
        test_id="default-profile",
        coordinator_agent="http://mac-a:9100",
        worker_agent="http://mac-b:9200",
        nodes=[{}, {}],
        model="/models/h3",
        binary="/runtime/bin/mlx-serve",
    )
    assert config.optimization_profile == "stock-qmm"
    assert config.starting_port == 30_096


def test_benchmark_harness_supports_single_mac(tmp_path: Path):
    transport = FakeTransport()
    config = BenchmarkConfig(
        output_dir=tmp_path,
        test_id="single-mac",
        coordinator_agent="http://mac-a:9100",
        worker_agent="http://mac-a:9100",
        nodes=[
            {
                "id": "mac-a",
                "agent_url": "http://mac-a:9100",
                "lan_ip": "mac-a",
                "rdma_devices": [],
            }
        ],
        model="/models/h3",
        binary="/runtime/bin/mlx-serve",
        runs=1,
        heartbeat_interval_seconds=3600,
    )

    report = run_benchmark(config, transport=transport)

    assert len(report["runs"]) == 1
    assert set(report["runs"][0]["rank_snapshots"]) == {"rank-0"}
    start_payloads = [
        payload for url, payload in transport.posts
        if url.endswith("/v1/node/start-minimax-h3-video")
    ]
    assert start_payloads[0]["connection_mode"] == "ring"
    assert set(report["shutdown"]) >= {"rank-0", "rank-0_node_info_after"}
    assert "rank-1" not in report["shutdown"]


def test_benchmark_cli_uses_the_reserved_h3_collective_range():
    args = _build_parser().parse_args(
        [
            "--coordinator-agent", "http://mac-a:9100",
            "--worker-agent", "http://mac-b:9200",
            "--nodes-json", "nodes.json",
            "--model", "/models/h3",
            "--binary", "/runtime/bin/mlx-serve",
            "--output-dir", "evidence",
            "--test-id", "default-port",
        ]
    )

    assert args.starting_port == 30_096


def test_benchmark_harness_archives_sse_hashes_snapshots_and_stops(tmp_path: Path):
    transport = FakeTransport()
    config = BenchmarkConfig(
        output_dir=tmp_path,
        test_id="fixture-run",
        coordinator_agent="http://mac-a:9100",
        worker_agent="http://mac-b:9200",
        nodes=[
            {
                "id": "mac-a",
                "agent_url": "http://mac-a:9100",
                "lan_ip": "mac-a",
                "rdma_ip": "203.0.113.1",
                "rdma_devices": ["rdma_en4"],
            },
            {
                "id": "mac-b",
                "agent_url": "http://mac-b:9200",
                "lan_ip": "mac-b",
                "rdma_ip": "203.0.113.2",
                "rdma_devices": ["rdma_en5"],
            },
        ],
        model="/models/h3",
        binary="/runtime/bin/mlx-serve",
        optimization_profile="block-fusions",
        runs=2,
        heartbeat_interval_seconds=3600,
    )

    report = run_benchmark(config, transport=transport)

    assert len(report["runs"]) == 2
    assert report["runs"][0]["temperature"] == "cold"
    assert report["runs"][1]["temperature"] == "warm"
    assert report["runs"][0]["progress_events"] == 31
    assert len(report["runs"][0]["video_sha256"]) == 64
    assert report["determinism"]["video_hashes_identical"] is True
    assert report["shutdown"]["requested"] is True
    assert (tmp_path / "fixture-run" / "result.json").is_file()
    assert (tmp_path / "fixture-run" / "run-01.sse").is_file()
    assert (tmp_path / "fixture-run" / "run-01.rgb").read_bytes() == bytes(range(24))
    assert any(url.endswith("/stop") for url, _ in transport.posts)
