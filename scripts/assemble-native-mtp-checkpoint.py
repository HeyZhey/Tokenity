#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assemble a Tokenity-native Qwen checkpoint from target and MTP repos.

The mlx-community Qwen MTP repositories are split draft models.  Tokenity's
Native MTP runtime intentionally does not load an external draft model, so
this tool produces a self-contained checkpoint by namespacing the draft
weights under ``language_model.mtp`` and extending the target index.

The MTP fusion projection is stored quantized by the split repository while
the mlx-lm Qwen model keeps that projection in full precision.  The tool
therefore dequantizes only ``fc`` to BF16; every other draft projection keeps
its original MLX affine quantization tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def validate_configs(target: dict[str, Any], draft: dict[str, Any]) -> None:
    target_text = text_config(target)
    draft_text = text_config(draft)
    if not str(target_text.get("model_type", "")).startswith(("qwen3_5", "qwen3_6")):
        raise ValueError("Target is not a supported Qwen3.5/Qwen3.6 text checkpoint")
    if draft.get("model_type") != "qwen3_5_mtp":
        raise ValueError("Draft is not the expected split qwen3_5_mtp checkpoint")
    # Split drafter repos express the maximum *total* speculative block here
    # (confirmed token plus one or more drafts).  Tokenity deliberately uses
    # only one draft, so any repository capable of at least that depth is
    # compatible; the larger configured ceiling is not copied into runtime
    # configuration and does not enable adaptive depth.
    if int(draft.get("block_size", 0) or 0) < 2:
        raise ValueError("Tokenity depth-one Native MTP requires draft block_size>=2")
    if int(target_text.get("mtp_num_hidden_layers", 0) or 0) != 1:
        raise ValueError("Target must declare exactly one MTP hidden layer")
    if int(draft_text.get("mtp_num_hidden_layers", 0) or 0) != 1:
        raise ValueError("Draft must declare exactly one MTP hidden layer")

    compatible_fields = (
        "hidden_size",
        "intermediate_size",
        "head_dim",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "rms_norm_eps",
        "tie_word_embeddings",
    )
    mismatches = [
        name
        for name in compatible_fields
        if target_text.get(name) != draft_text.get(name)
    ]
    if mismatches:
        raise ValueError(f"Target and draft configs disagree: {', '.join(mismatches)}")

    target_quant = target.get("quantization") or target.get("quantization_config")
    draft_quant = draft.get("quantization") or draft.get("quantization_config")
    if target_quant != draft_quant:
        raise ValueError("Target and draft quantization configs differ")


def copy_target(target: Path, destination: Path) -> None:
    for source in target.iterdir():
        if source.name == ".cache":
            continue
        output = destination / source.name
        if source.is_dir():
            shutil.copytree(source, output)
        elif source.name == "model.safetensors":
            try:
                os.link(source, output)
            except OSError:
                shutil.copy2(source, output)
        else:
            shutil.copy2(source, output)


def assemble(target: Path, draft: Path, output: Path) -> dict[str, Any]:
    import mlx.core as mx  # Imported lazily so --help works without MLX.

    target = target.resolve()
    draft = draft.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    target_config = read_json(target / "config.json")
    draft_config = read_json(draft / "config.json")
    validate_configs(target_config, draft_config)

    target_index = read_json(target / "model.safetensors.index.json")
    draft_index = read_json(draft / "model.safetensors.index.json")
    target_map = target_index.get("weight_map")
    draft_map = draft_index.get("weight_map")
    if not isinstance(target_map, dict) or not isinstance(draft_map, dict):
        raise ValueError("Both checkpoints must have a safetensors weight map")
    if any("mtp" in str(key).lower() for key in target_map):
        raise ValueError("Target index already contains MTP tensors")

    weights = mx.load(str(draft / "model.safetensors"))
    if set(weights) != set(draft_map):
        raise ValueError("Draft index keys do not exactly match model.safetensors")
    for key in ("fc.weight", "fc.scales", "fc.biases"):
        if key not in weights:
            raise ValueError(f"Draft is missing required quantized tensor: {key}")

    quantization = draft_config.get("quantization") or draft_config["quantization_config"]
    fc = mx.dequantize(
        weights.pop("fc.weight"),
        scales=weights.pop("fc.scales"),
        biases=weights.pop("fc.biases"),
        group_size=int(quantization["group_size"]),
        bits=int(quantization["bits"]),
        mode=str(quantization.get("mode", "affine")),
        dtype=mx.bfloat16,
    )
    mx.eval(fc)
    hidden_size = int(text_config(target_config)["hidden_size"])
    if tuple(fc.shape) != (hidden_size, hidden_size * 2):
        raise ValueError(f"Unexpected dequantized fc shape: {tuple(fc.shape)}")
    weights["fc.weight"] = fc

    namespaced = {f"language_model.mtp.{key}": value for key, value in weights.items()}
    mtp_bytes = sum(int(value.nbytes) for value in namespaced.values())

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        copy_target(target, temporary)
        # Keep the head outside mlx-lm's ordinary ``model*.safetensors`` glob.
        # Tokenity exposes this sidecar only inside an enabled construction
        # scope, which keeps ``mode=off`` identical to the original target.
        mtp_file_name = "mtp.safetensors"
        mx.save_safetensors(
            str(temporary / mtp_file_name),
            namespaced,
            metadata={"format": "mlx", "tokenity_native_mtp": "depth1"},
        )

        merged_map = dict(target_map)
        merged_map.update({key: mtp_file_name for key in namespaced})
        metadata = dict(target_index.get("metadata") or {})
        metadata["total_size"] = int(metadata.get("total_size", 0)) + mtp_bytes
        merged_index = {**target_index, "metadata": metadata, "weight_map": merged_map}
        (temporary / "model.safetensors.index.json").write_text(
            json.dumps(merged_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(draft / "README.md", temporary / "MTP_DRAFT_README.md")

        provenance = {
            "format_version": 1,
            "target": {
                "path": str(target),
                "model_sha256": sha256(target / "model.safetensors"),
            },
            "draft": {
                "path": str(draft),
                "model_sha256": sha256(draft / "model.safetensors"),
                "configured_block_size": int(draft_config["block_size"]),
            },
            "tokenity_max_depth": 1,
            "mtp_tensor_count": len(namespaced),
            "mtp_tensor_bytes": mtp_bytes,
            "fc_storage": "bfloat16",
            "other_mtp_linear_storage": "mlx-affine-4bit",
        }
        (temporary / "tokenity-native-mtp.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return provenance
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    try:
        provenance = assemble(args.target, args.draft, args.output)
    except Exception as exc:
        print(f"assemble-native-mtp-checkpoint: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
