from __future__ import annotations

import json
import hashlib
import os
import subprocess
from pathlib import Path

from tokenity.serving.minimax_h3_video import (
    H3_DISTRIBUTED_PROTOCOL_VERSION,
    build_native_h3_command,
    exec_native_h3,
    h3_optimization_environment,
    h3_model_issues,
    h3_runtime_fingerprint,
    h3_runtime_preflight,
    probe_h3_backend,
)


def _h3_model(tmp_path: Path) -> Path:
    model = tmp_path / "MiniMax-H3-FL2VA-MLX-Serve-8bit"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "minimax_h3", "partition": "fl2va"}),
        encoding="utf-8",
    )
    for name in (
        "transformer.safetensors",
        "text_encoder.safetensors",
        "video_vae.safetensors",
        "audio_vae.safetensors",
    ):
        (model / name).touch()
    return model


def _add_tp2_layout(model: Path) -> None:
    tp2 = model / "tp2"
    for rank in (0, 1):
        rank_dir = tp2 / f"rank-{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        (rank_dir / "transformer.safetensors").touch()
    rank_files = [
        tp2 / "rank-0" / "transformer.safetensors",
        tp2 / "rank-1" / "transformer.safetensors",
    ]
    (tp2 / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "minimax_h3",
                "protocol": H3_DISTRIBUTED_PROTOCOL_VERSION,
                "world_size": 2,
                "rank_files": [
                    "rank-0/transformer.safetensors",
                    "rank-1/transformer.safetensors",
                ],
                "rank_artifacts": [
                    {
                        "path": f"rank-{rank}/transformer.safetensors",
                        "size_bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for rank, path in enumerate(rank_files)
                ],
                "sharding": {
                    "main_blocks": 50,
                    "attention_heads": 56,
                    "attention_heads_per_rank": 28,
                    "ffn_hidden": 14336,
                    "ffn_hidden_per_rank": 7168,
                },
            }
        ),
        encoding="utf-8",
    )


def _binary(tmp_path: Path) -> Path:
    binary = tmp_path / "mlx-serve"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


def _runtime(tmp_path: Path) -> Path:
    root = tmp_path / "runtime"
    binary = root / "bin" / "mlx-serve"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary-v1")
    binary.chmod(0o755)
    mlx = root / "lib" / "mlx" / "lib"
    mlx.mkdir(parents=True)
    (mlx / "libmlx.dylib").write_bytes(b"mlx-v1")
    (mlx / "libmlxc.dylib").write_bytes(b"mlxc-v1")
    (mlx / "mlx.metallib").write_bytes(b"metal-v1")
    return binary


def test_optimization_profiles_are_typed_and_expand_only_allowlisted_flags():
    assert h3_optimization_environment("baseline") == {
        "MLX_SERVE_MF_DQ_GEMM": "2048",
        "MINIMAX_H3_FUSED_SWIGLU": "0",
        "MINIMAX_H3_FUSED_GATE_RESIDUAL": "0",
        "MINIMAX_H3_FUSED_RMS_ADALN": "0",
    }
    fused = h3_optimization_environment("block-fusions")
    assert fused["MINIMAX_H3_FUSED_SWIGLU"] == "1"
    assert fused["MINIMAX_H3_FUSED_GATE_RESIDUAL"] == "1"
    assert fused["MINIMAX_H3_FUSED_RMS_ADALN"] == "1"
    try:
        h3_optimization_environment("MLX_EVIL=1")
    except ValueError as exc:
        assert "Unknown MiniMax H3 optimization profile" in str(exc)
    else:
        raise AssertionError("arbitrary environment input was accepted")


def test_runtime_fingerprint_covers_binary_mlx_metallib_manifest_and_rank_contract(
    tmp_path: Path,
):
    model = _h3_model(tmp_path)
    _add_tp2_layout(model)
    binary = _runtime(tmp_path)

    first = h3_runtime_fingerprint(
        binary=binary,
        model=model,
        optimization_profile="block-fusions",
        tokenity_code_revision="code-revision",
        verify_rank=0,
    )
    second = h3_runtime_fingerprint(
        binary=binary,
        model=model,
        optimization_profile="block-fusions",
        tokenity_code_revision="code-revision",
        verify_rank=0,
    )

    assert first == second
    assert set(first["artifacts"]) == {
        "binary",
        "libmlx",
        "libmlxc",
        "metallib",
        "tp2_manifest",
    }
    assert first["rank_shards"]["rank-0"]["verified"] is True
    assert first["optimization_profile"] == "block-fusions"
    assert len(first["contract_sha256"]) == 64

    metallib = binary.parent.parent / "lib" / "mlx" / "lib" / "mlx.metallib"
    metallib.write_bytes(b"metal-v2")
    changed = h3_runtime_fingerprint(
        binary=binary,
        model=model,
        optimization_profile="block-fusions",
        tokenity_code_revision="code-revision",
        verify_rank=0,
    )
    assert changed["contract_sha256"] != first["contract_sha256"]


def test_runtime_contract_ignores_rank_local_absolute_roots_but_not_content(
    tmp_path: Path,
):
    contracts = []
    for name in ("mac-a", "mac-b"):
        root = tmp_path / name
        root.mkdir()
        model = _h3_model(root)
        _add_tp2_layout(model)
        binary = _runtime(root)
        contracts.append(
            h3_runtime_fingerprint(
                binary=binary,
                model=model,
                optimization_profile="block-fusions",
                tokenity_code_revision="same-code",
                verify_rank=0,
            )
        )

    assert contracts[0]["artifacts"]["binary"]["path"] != contracts[1]["artifacts"]["binary"]["path"]
    assert contracts[0]["contract_sha256"] == contracts[1]["contract_sha256"]


def test_single_node_command_matches_native_mlx_serve_contract():
    command = build_native_h3_command(
        binary="/runtime/bin/mlx-serve",
        model="/models/MiniMax-H3",
        host="0.0.0.0",
        port=11_241,
    )

    assert command == [
        "/runtime/bin/mlx-serve",
        "--model",
        "/models/MiniMax-H3",
        "--serve",
        "--host",
        "0.0.0.0",
        "--port",
        "11241",
    ]


def test_distributed_command_is_explicit_and_versioned():
    command = build_native_h3_command(
        binary="/runtime/bin/mlx-serve",
        model="/models/MiniMax-H3-tp2-rank1",
        host="0.0.0.0",
        port=11_241,
        rank=1,
        world_size=2,
    )

    assert command[-6:] == [
        "--h3-distributed-rank",
        "1",
        "--h3-distributed-world-size",
        "2",
        "--h3-distributed-protocol",
        str(H3_DISTRIBUTED_PROTOCOL_VERSION),
    ]


def test_model_validation_rejects_non_h3_and_missing_components(tmp_path: Path):
    model = tmp_path / "not-h3"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )

    issues = h3_model_issues(model)

    assert "Expected config.json model_type 'minimax_h3'" in issues[0]
    assert any("transformer.safetensors" in issue for issue in issues)


def test_backend_probe_requires_all_distributed_protocol_flags(tmp_path: Path):
    binary = _binary(tmp_path)

    def single_runner(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="--model --serve", stderr="")

    def distributed_runner(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout=(
                "--h3-distributed-rank --h3-distributed-world-size "
                "--h3-distributed-protocol"
            ),
            stderr="",
        )

    assert not probe_h3_backend(binary, runner=single_runner).supports_distributed_h3
    assert probe_h3_backend(binary, runner=distributed_runner).supports_distributed_h3


def test_two_mac_preflight_fails_closed_for_current_single_node_binary(tmp_path: Path):
    model = _h3_model(tmp_path)
    binary = _binary(tmp_path)

    issues, capabilities = h3_runtime_preflight(
        binary=str(binary),
        model=str(model),
        world_size=2,
        backend_probe=lambda _: probe_h3_backend(
            binary,
            runner=lambda *args, **kwargs: subprocess.CompletedProcess(
                args[0], 0, stdout="--model --serve", stderr=""
            ),
        ),
    )

    assert not capabilities.supports_distributed_h3
    assert any("distributed protocol v1" in issue for issue in issues)


def test_two_mac_preflight_requires_a_versioned_tp2_checkpoint(tmp_path: Path):
    model = _h3_model(tmp_path)
    binary = _binary(tmp_path)
    distributed = lambda _: probe_h3_backend(
        binary,
        runner=lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=(
                "--h3-distributed-rank --h3-distributed-world-size "
                "--h3-distributed-protocol"
            ),
            stderr="",
        ),
    )

    issues, _ = h3_runtime_preflight(
        binary=str(binary),
        model=str(model),
        world_size=2,
        backend_probe=distributed,
    )
    assert any("tp2/manifest.json" in issue for issue in issues)

    _add_tp2_layout(model)
    issues, capabilities = h3_runtime_preflight(
        binary=str(binary),
        model=str(model),
        world_size=2,
        backend_probe=distributed,
    )
    assert issues == []
    assert capabilities.supports_distributed_h3

    (model / "tp2" / "rank-0" / "transformer.safetensors").write_bytes(b"tampered")
    issues, _ = h3_runtime_preflight(
        binary=str(binary),
        model=str(model),
        world_size=2,
        rank=0,
        backend_probe=distributed,
    )
    assert any("rank-0/transformer.safetensors" in issue for issue in issues)


def test_rank_one_preflight_requires_only_its_local_tp2_shard(tmp_path: Path):
    model = _h3_model(tmp_path)
    _add_tp2_layout(model)
    binary = _binary(tmp_path)
    distributed = lambda _: probe_h3_backend(
        binary,
        runner=lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=(
                "--h3-distributed-rank --h3-distributed-world-size "
                "--h3-distributed-protocol"
            ),
            stderr="",
        ),
    )

    (model / "tp2" / "rank-0" / "transformer.safetensors").unlink()
    for name in (
        "transformer.safetensors",
        "text_encoder.safetensors",
        "video_vae.safetensors",
        "audio_vae.safetensors",
    ):
        (model / name).unlink()

    issues, capabilities = h3_runtime_preflight(
        binary=str(binary),
        model=str(model),
        world_size=2,
        rank=1,
        backend_probe=distributed,
    )

    assert issues == []
    assert capabilities.supports_distributed_h3


def test_exec_adapter_replaces_process_without_a_shell(tmp_path: Path):
    model = _h3_model(tmp_path)
    binary = _binary(tmp_path)
    calls = []

    def exec_fn(executable, argv, env):
        calls.append((executable, list(argv), dict(env)))
        raise SystemExit(0)

    try:
        exec_native_h3(
            binary=str(binary),
            model=str(model),
            host="127.0.0.1",
            port=11_241,
            rank=0,
            world_size=1,
            environ={"PATH": os.environ.get("PATH", "")},
            exec_fn=exec_fn,
        )
    except SystemExit as exc:
        assert exc.code == 0

    executable, argv, env = calls[0]
    assert executable == str(binary)
    assert argv[0] == str(binary)
    assert argv[1:3] == ["--model", str(model)]
    assert "--serve" in argv
    assert env["MLX_METAL_FAST_SYNCH"] == "1"
