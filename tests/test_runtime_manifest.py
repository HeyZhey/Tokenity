from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "tokenity-runtime-manifest.py"
SPEC = importlib.util.spec_from_file_location("tokenity_runtime_manifest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime_manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_manifest)


def test_tree_identity_is_stable_across_mtime_and_ignores_manifest(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    executable = payload / "tool"
    executable.write_bytes(b"runtime")
    executable.chmod(0o755)
    (payload / "alias").symlink_to("tool")

    first = runtime_manifest.tree_identity(payload)
    os.utime(executable, (1_000_000, 1_000_000))
    (payload / runtime_manifest.MANIFEST_NAME).write_text(
        '{"generated": true}\n', encoding="utf-8"
    )
    second = runtime_manifest.tree_identity(payload)

    assert first == second
    assert first["file_count"] == 2
    assert first["size_bytes"] == len(b"runtime")


def test_artifact_catalog_pins_filename_size_and_sha256(tmp_path: Path) -> None:
    artifact = tmp_path / "Runtime.pkg"
    artifact.write_bytes(b"package-bytes")
    lock = {
        "runtime_id": "runtime-test",
        "tokenity_version": "0.1.0",
        "platform": "macos",
        "architecture": "arm64",
        "minimum_macos": "14.0",
        "python_version": "3.12.13",
        "packages": {"mlx": "0.32.0", "mlx-lm": "0.31.3"},
        "package_identifier": "ai.tokenity.runtime.test",
        "artifact_filename": artifact.name,
    }
    runtime = {"payload": {"tree_sha256": "a" * 64}}

    catalog = runtime_manifest.build_artifact_catalog(
        lock,
        runtime,
        artifact,
        "https://example.invalid/Runtime.pkg",
    )

    assert catalog["artifact"]["filename"] == "Runtime.pkg"
    assert catalog["artifact"]["size_bytes"] == len(b"package-bytes")
    assert len(catalog["artifact"]["sha256"]) == 64
    assert catalog["artifact"]["urls"] == ["https://example.invalid/Runtime.pkg"]

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    runtime_manifest.verify_catalog(catalog_path, artifact)


def test_artifact_catalog_rejects_unlocked_filename(tmp_path: Path) -> None:
    artifact = tmp_path / "unexpected.pkg"
    artifact.write_bytes(b"package")
    lock = {
        "runtime_id": "runtime-test",
        "tokenity_version": "0.1.0",
        "platform": "macos",
        "architecture": "arm64",
        "minimum_macos": "14.0",
        "python_version": "3.12.13",
        "packages": {},
        "package_identifier": "ai.tokenity.runtime.test",
        "artifact_filename": "expected.pkg",
    }

    try:
        runtime_manifest.build_artifact_catalog(
            lock,
            {"payload": {"tree_sha256": "a" * 64}},
            artifact,
            "https://example.invalid/unexpected.pkg",
        )
    except runtime_manifest.RuntimeValidationError:
        pass
    else:
        raise AssertionError("Expected an unlocked artifact filename to fail")
