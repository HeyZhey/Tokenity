#!/usr/bin/env python3
"""Normalize, identify, and verify a distributable Tokenity runtime tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


MANIFEST_NAME = "runtime-manifest.json"
IGNORED_DIRECTORY_NAMES = {".git", "__pycache__", "cache"}
IGNORED_FILE_NAMES = {".DS_Store", ".lock", "CACHEDIR.TAG", MANIFEST_NAME}
IGNORED_FILE_SUFFIXES = {".pyc", ".pyo"}
RECEIPT_FILE_NAMES = {"INSTALLER", "RECORD", "REQUESTED", "direct_url.json"}
DEFAULT_DATA_ROOT = Path.home() / ".tokenity"
EXPECTED_INSTALL_ROOT = Path(
    os.environ.get("TOKENITY_RUNTIME_ROOT", DEFAULT_DATA_ROOT / "Runtime")
)
TOKENITY_CODE_PATH = os.environ.get(
    "TOKENITY_CODE_ROOT", str(DEFAULT_DATA_ROOT / "Code")
)


class RuntimeValidationError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeValidationError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def find_backend(root: Path, lock: dict[str, Any]) -> tuple[str, Path]:
    expected_name = str(lock["backend_name"])
    expected = root / "backends" / expected_name
    if expected.is_dir():
        return expected_name, expected

    current = root / "current"
    if current.is_symlink():
        candidate_name = Path(os.readlink(current)).name
        candidate = root / "backends" / candidate_name
        if candidate.is_dir():
            return candidate_name, candidate

    raise RuntimeValidationError(
        f"runtime backend {expected_name!r} was not found below {root / 'backends'}"
    )


def find_python_release(root: Path, python_version: str) -> tuple[str, Path]:
    expected_name = f"cpython-{python_version}-macos-aarch64-none"
    expected = root / "pythons" / expected_name
    if expected.is_dir():
        return expected_name, expected
    raise RuntimeValidationError(f"embedded CPython {expected_name!r} was not found")


def replace_symlink(path: Path, target: str) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(target)


def remove_transient_files(root: Path) -> None:
    for directory, directory_names, file_names in os.walk(root, topdown=True):
        directory_path = Path(directory)
        kept_directories: list[str] = []
        for name in directory_names:
            candidate = directory_path / name
            if name in IGNORED_DIRECTORY_NAMES:
                shutil.rmtree(candidate)
            elif name.startswith("tokenity-") and name.endswith(".dist-info"):
                shutil.rmtree(candidate)
            else:
                kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in file_names:
            candidate = directory_path / name
            if (
                name in IGNORED_FILE_NAMES
                or candidate.suffix in IGNORED_FILE_SUFFIXES
                or name.startswith("__editable__") and "tokenity" in name
                or candidate.parent.name.endswith(".dist-info")
                and name in RECEIPT_FILE_NAMES
            ):
                candidate.unlink(missing_ok=True)


def normalize_entrypoint_shebangs(bin_directory: Path) -> None:
    if any(character.isspace() for character in str(EXPECTED_INSTALL_ROOT)):
        raise RuntimeValidationError(
            "TOKENITY_RUNTIME_ROOT cannot contain whitespace because packaged "
            "entrypoint shebangs must be directly executable"
        )
    expected = f"#!{EXPECTED_INSTALL_ROOT}/current/.venv/bin/python\n".encode()
    for path in sorted(bin_directory.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        first_line, separator, remainder = data.partition(b"\n")
        if (
            separator
            and first_line.startswith(b"#!")
            and b"TokenityRuntime/" in first_line
            and first_line.endswith(b"/.venv/bin/python")
            and first_line + b"\n" != expected
        ):
            path.write_bytes(expected + remainder)

    wheel_entrypoint = bin_directory / "wheel"
    if wheel_entrypoint.is_file():
        wheel_entrypoint.write_text(
            f"#!{EXPECTED_INSTALL_ROOT}/current/.venv/bin/python\n"
            "import sys\n"
            "from wheel._commands import main\n"
            "\n"
            "if __name__ == \"__main__\":\n"
            "    sys.exit(main())\n",
            encoding="utf-8",
        )


def normalize_runtime(root: Path, lock: dict[str, Any]) -> None:
    root = root.resolve()
    if root == EXPECTED_INSTALL_ROOT:
        raise RuntimeValidationError(
            f"refusing to normalize the live runtime at {EXPECTED_INSTALL_ROOT}; import it into staging first"
        )
    if root == Path("/") or len(root.parts) < 3:
        raise RuntimeValidationError(f"refusing to normalize unsafe path {root}")

    remove_transient_files(root)
    backend_name, backend = find_backend(root, lock)
    python_release_name, _ = find_python_release(root, str(lock["python_version"]))
    python_alias_name = "cpython-3.12-macos-aarch64-none"

    replace_symlink(root / "current", f"backends/{backend_name}")
    replace_symlink(root / "pythons" / python_alias_name, python_release_name)
    replace_symlink(
        backend / ".venv" / "bin" / "python",
        f"../../../../pythons/{python_alias_name}/bin/python3.12",
    )
    replace_symlink(backend / ".venv" / "bin" / "python3", "python")
    replace_symlink(backend / ".venv" / "bin" / "python3.12", "python")

    site_packages = backend / ".venv" / "lib" / "python3.12" / "site-packages"
    (site_packages / "tokenity-code.pth").write_text(
        f"{TOKENITY_CODE_PATH}\n", encoding="utf-8"
    )
    normalize_entrypoint_shebangs(backend / ".venv" / "bin")


def runtime_python(root: Path) -> Path:
    executable = root / "current" / ".venv" / "bin" / "python"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeValidationError(f"runtime Python is missing or not executable: {executable}")
    return executable


def inspect_python(root: Path, lock: dict[str, Any]) -> dict[str, Any]:
    package_names = sorted(str(name) for name in lock["packages"])
    program = """
import json
import platform
from importlib.metadata import version

names = json.loads(%r)
print(json.dumps({
    "architecture": platform.machine(),
    "python_version": platform.python_version(),
    "packages": {name: version(name) for name in names},
}, sort_keys=True))
""" % json.dumps(package_names)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [str(runtime_python(root)), "-c", program],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(completed.stdout)


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(piece) for piece in value.split("."))


def macho_minimum_macos(path: Path) -> str:
    completed = subprocess.run(
        ["/usr/bin/otool", "-l", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    versions = re.findall(r"^\s*minos\s+([0-9.]+)\s*$", completed.stdout, re.MULTILINE)
    if not versions:
        raise RuntimeValidationError(f"LC_BUILD_VERSION minos is missing from {path}")
    return max(versions, key=version_tuple)


def inspect_mlx_binaries(root: Path) -> dict[str, Any]:
    backend = root / "current"
    site_packages = backend / ".venv" / "lib" / "python3.12" / "site-packages"
    candidates = [
        *site_packages.glob("mlx/core*.so"),
        site_packages / "mlx" / "lib" / "libmlx.dylib",
        site_packages / "mlx" / "lib" / "libjaccl.dylib",
    ]
    binaries = [path for path in candidates if path.is_file()]
    if len(binaries) < 3:
        raise RuntimeValidationError("the MLX core, libmlx, and libjaccl binaries are required")

    minimum_versions: dict[str, str] = {}
    for binary in sorted(binaries):
        file_output = subprocess.run(
            ["/usr/bin/file", str(binary)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if "Mach-O" not in file_output or "arm64" not in file_output:
            raise RuntimeValidationError(f"{binary} is not an arm64 Mach-O binary")
        minimum_versions[str(binary.relative_to(root))] = macho_minimum_macos(binary)
    return {
        "minimum_macos": max(minimum_versions.values(), key=version_tuple),
        "binaries": minimum_versions,
    }


def tree_identity(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0

    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.name == MANIFEST_NAME:
            continue
        if path.is_symlink():
            target = os.readlink(path)
            digest.update(b"L\0")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(target.encode("utf-8"))
            digest.update(b"\0")
            file_count += 1
            continue
        if not path.is_file():
            continue

        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                file_digest.update(chunk)
        mode = stat.S_IMODE(path.stat().st_mode)
        size = path.stat().st_size
        digest.update(b"F\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(f"{mode:o}".encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.hexdigest().encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        total_bytes += size

    return {
        "file_count": file_count,
        "size_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def build_runtime_manifest(root: Path, lock: dict[str, Any]) -> dict[str, Any]:
    observed = inspect_python(root, lock)
    if observed["architecture"] != lock["architecture"]:
        raise RuntimeValidationError(
            f"expected architecture {lock['architecture']}, found {observed['architecture']}"
        )
    if observed["python_version"] != lock["python_version"]:
        raise RuntimeValidationError(
            f"expected Python {lock['python_version']}, found {observed['python_version']}"
        )
    if observed["packages"] != lock["packages"]:
        raise RuntimeValidationError(
            f"expected packages {lock['packages']}, found {observed['packages']}"
        )

    mlx_binaries = inspect_mlx_binaries(root)
    if version_tuple(mlx_binaries["minimum_macos"]) > version_tuple(
        str(lock["minimum_macos"])
    ):
        raise RuntimeValidationError(
            "runtime binaries require macOS "
            f"{mlx_binaries['minimum_macos']}, above lock {lock['minimum_macos']}"
        )

    return {
        "schema_version": 1,
        "runtime_id": lock["runtime_id"],
        "tokenity_version": lock["tokenity_version"],
        "platform": lock["platform"],
        "architecture": observed["architecture"],
        "minimum_macos": lock["minimum_macos"],
        "observed_mlx_minimum_macos": mlx_binaries["minimum_macos"],
        "python_version": observed["python_version"],
        "backend_name": lock["backend_name"],
        "packages": observed["packages"],
        "payload": tree_identity(root),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_catalog(
    lock: dict[str, Any],
    runtime_manifest: dict[str, Any],
    artifact: Path,
    download_url: str,
) -> dict[str, Any]:
    if artifact.name != lock["artifact_filename"]:
        raise RuntimeValidationError(
            f"expected artifact filename {lock['artifact_filename']}, found {artifact.name}"
        )
    return {
        "schema_version": 1,
        "runtime_id": lock["runtime_id"],
        "tokenity_version": lock["tokenity_version"],
        "platform": lock["platform"],
        "architecture": lock["architecture"],
        "minimum_macos": lock["minimum_macos"],
        "python_version": lock["python_version"],
        "packages": lock["packages"],
        "runtime_payload_sha256": runtime_manifest["payload"]["tree_sha256"],
        "artifact": {
            "filename": artifact.name,
            "package_identifier": lock["package_identifier"],
            "size_bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
            "urls": [download_url],
        },
    }


def verify_runtime(root: Path, manifest_path: Path, lock: dict[str, Any]) -> None:
    expected = load_json(manifest_path)
    observed = build_runtime_manifest(root, lock)
    if observed != expected:
        raise RuntimeValidationError(
            "runtime manifest mismatch:\n"
            f"expected={json.dumps(expected, sort_keys=True)}\n"
            f"observed={json.dumps(observed, sort_keys=True)}"
        )


def verify_catalog(catalog_path: Path, artifact: Path) -> None:
    catalog = load_json(catalog_path)
    expected = catalog["artifact"]
    if artifact.name != expected["filename"]:
        raise RuntimeValidationError(
            f"expected artifact {expected['filename']}, found {artifact.name}"
        )
    if artifact.stat().st_size != expected["size_bytes"]:
        raise RuntimeValidationError("artifact size does not match the catalog")
    if sha256_file(artifact) != expected["sha256"]:
        raise RuntimeValidationError("artifact SHA-256 does not match the catalog")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("root", type=Path)
    prepare.add_argument("--lock", type=Path, required=True)
    prepare.add_argument("--output", type=Path)

    verify = subparsers.add_parser("verify")
    verify.add_argument("root", type=Path)
    verify.add_argument("--lock", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)

    catalog = subparsers.add_parser("catalog")
    catalog.add_argument("--lock", type=Path, required=True)
    catalog.add_argument("--runtime-manifest", type=Path, required=True)
    catalog.add_argument("--artifact", type=Path, required=True)
    catalog.add_argument("--download-url", required=True)
    catalog.add_argument("--output", type=Path, required=True)

    verify_artifact = subparsers.add_parser("verify-artifact")
    verify_artifact.add_argument("--catalog", type=Path, required=True)
    verify_artifact.add_argument("--artifact", type=Path, required=True)

    arguments = parser.parse_args()
    try:
        if arguments.command == "prepare":
            root = arguments.root.resolve()
            lock = load_json(arguments.lock)
            normalize_runtime(root, lock)
            output = arguments.output or root / MANIFEST_NAME
            write_json(output, build_runtime_manifest(root, lock))
        elif arguments.command == "verify":
            verify_runtime(
                arguments.root.resolve(),
                arguments.manifest.resolve(),
                load_json(arguments.lock),
            )
        elif arguments.command == "catalog":
            lock = load_json(arguments.lock)
            runtime_manifest = load_json(arguments.runtime_manifest)
            value = build_artifact_catalog(
                lock,
                runtime_manifest,
                arguments.artifact.resolve(),
                arguments.download_url,
            )
            write_json(arguments.output, value)
        elif arguments.command == "verify-artifact":
            verify_catalog(arguments.catalog.resolve(), arguments.artifact.resolve())
    except (OSError, subprocess.CalledProcessError, RuntimeValidationError) as error:
        print(f"runtime validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
