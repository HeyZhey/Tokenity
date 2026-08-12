from __future__ import annotations

import pytest

from tokenity.cli import _build_parser, main


def test_version(capsys):
    assert main(["--version"]) == 0
    assert "tokenity 0.1.0" in capsys.readouterr().out


def test_native_mtp_cli_defaults_off_and_rejects_depth_greater_than_one():
    parser = _build_parser()
    args = parser.parse_args(["distributed-openai", "serve", "--model", "/models/qwen"])
    assert args.native_mtp_mode == "off"
    assert args.native_mtp_max_depth == 1
    assert args.native_mtp_head_placement == "replicated"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "distributed-openai",
                "serve",
                "--model",
                "/models/qwen",
                "--native-mtp-max-depth",
                "2",
            ]
        )


def test_minimax_h3_video_cli_exposes_native_runtime_adapter():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "minimax-h3-video",
            "serve",
            "--binary",
            "/runtime/bin/mlx-serve",
            "--model",
            "/models/MiniMax-H3",
            "--rank",
            "1",
            "--world-size",
            "2",
        ]
    )

    assert args.binary == "/runtime/bin/mlx-serve"
    assert args.model == "/models/MiniMax-H3"
    assert args.rank == 1
    assert args.world_size == 2
