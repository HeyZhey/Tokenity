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


def test_native_bundle_must_include_libraries_and_locked_turbo_contract(tmp_path, monkeypatch):
    import subprocess
    import pytest
    binary = tmp_path / "current/bin/mlx-serve"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary")
    binary.chmod(0o755)
    lock = {"minimum_macos": "26.2", "native_h3": {"binary": "current/bin/mlx-serve", "distributed_protocol": 1, "turbo_protocol_version": 1, "modules": 259, "strength": 1.0}}
    with pytest.raises(runtime_manifest.RuntimeValidationError, match="incomplete"):
        runtime_manifest.inspect_native_h3(tmp_path, lock)
    for relative in ("lib/mlx/lib/libmlx.dylib", "lib/mlx/lib/libmlxc.dylib", "lib/mlx/lib/libjaccl.dylib", "lib/mlx/lib/mlx.metallib", "lib/llama/lib/libllama.dylib", "lib/libwebp.7.dylib"):
        path = tmp_path / "current" / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"library")
    declared = dict(turbo_protocol_version=1, modules=259, strength=1.0)
    def probe(command, **kwargs):
        output = ("--h3-distributed-rank --h3-distributed-world-size --h3-distributed-protocol 1"
                  if command[-1] == "--help" else json.dumps(declared))
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
    monkeypatch.setattr(subprocess, "run", probe)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "Mach-O arm64")
    monkeypatch.setattr(runtime_manifest, "macho_minimum_macos", lambda path: "26.2")
    observed = runtime_manifest.inspect_native_h3(tmp_path, lock)
    assert len(observed["artifacts"]) == 7
    declared["modules"] = 258
    with pytest.raises(runtime_manifest.RuntimeValidationError, match="modules"):
        runtime_manifest.inspect_native_h3(tmp_path, lock)


def test_macos_minimum_is_read_without_developer_tools(tmp_path):
    import struct
    import pytest
    binary = tmp_path / "runtime"
    def macho(cpu=0x0100000C, version=0x1A0200, command=0x32):
        load = (struct.pack("<6I", command, 24, 1, version, version, 0)
                if command == 0x32 else struct.pack("<4I", command, 16, version, version))
        return struct.pack("<8I", 0xFEEDFACF, cpu, 0, 2, 1, len(load), 0, 0) + load
    binary.write_bytes(macho())
    assert runtime_manifest.macho_minimum_macos(binary) == "26.2"
    binary.write_bytes(macho(version=0x0E0001, command=0x24))
    assert runtime_manifest.macho_minimum_macos(binary) == "14.0.1"
    binary.write_bytes(macho(cpu=0x01000007))
    with pytest.raises(runtime_manifest.RuntimeValidationError, match="arm64"):
        runtime_manifest.macho_minimum_macos(binary)
    binary.write_bytes(macho()[:-4])
    with pytest.raises(runtime_manifest.RuntimeValidationError, match="length"):
        runtime_manifest.macho_minimum_macos(binary)


def test_python_entrypoints_are_relocated_from_any_previous_install(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_manifest, "EXPECTED_INSTALL_ROOT", Path("/Library/Tokenity/Runtime"))
    script = tmp_path / "mlx_lm.generate"
    script.write_bytes(b"#!/old/runtime/current/.venv/bin/python\nprint('ok')\n")
    shell = tmp_path / "shell"
    shell.write_bytes(b"#!/bin/sh\necho ok\n")
    runtime_manifest.normalize_entrypoint_shebangs(tmp_path)
    expected = b"#!/Library/Tokenity/Runtime/current/.venv/bin/python\nprint('ok')\n"
    assert script.read_bytes() == expected
    runtime_manifest.normalize_entrypoint_shebangs(tmp_path)
    assert script.read_bytes() == expected
    assert shell.read_bytes() == b"#!/bin/sh\necho ok\n"
