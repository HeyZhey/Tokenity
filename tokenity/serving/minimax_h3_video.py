from __future__ import annotations

import json
import hashlib
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from tokenity.paths import h3_binary


H3_MODEL_TYPE = "minimax_h3"
H3_DISTRIBUTED_PROTOCOL_VERSION = 1
H3_DISTRIBUTED_HELP_MARKERS = (
    "--h3-distributed-rank",
    "--h3-distributed-world-size",
    "--h3-distributed-protocol",
)
DEFAULT_MLX_SERVE_BINARY = str(h3_binary())

H3_OPTIMIZATION_PROFILES: dict[str, dict[str, str]] = {
    # Explicit values make the readiness contract auditable and avoid a future
    # native default silently changing one rank but not the other.
    "baseline": {
        "MLX_SERVE_MF_DQ_GEMM": "2048",
        "MINIMAX_H3_FUSED_SWIGLU": "0",
        "MINIMAX_H3_FUSED_GATE_RESIDUAL": "0",
        "MINIMAX_H3_FUSED_RMS_ADALN": "0",
    },
    "block-fusions": {
        "MLX_SERVE_MF_DQ_GEMM": "2048",
        "MINIMAX_H3_FUSED_SWIGLU": "1",
        "MINIMAX_H3_FUSED_GATE_RESIDUAL": "1",
        "MINIMAX_H3_FUSED_RMS_ADALN": "1",
    },
    "stock-qmm": {
        "MLX_SERVE_MF_DQ_GEMM": "0",
        "MINIMAX_H3_FUSED_SWIGLU": "0",
        "MINIMAX_H3_FUSED_GATE_RESIDUAL": "0",
        "MINIMAX_H3_FUSED_RMS_ADALN": "0",
    },
}


def h3_optimization_environment(profile: str) -> dict[str, str]:
    """Expand a typed profile; arbitrary environment input is never accepted."""

    try:
        return dict(H3_OPTIMIZATION_PROFILES[profile])
    except KeyError as exc:
        allowed = ", ".join(sorted(H3_OPTIMIZATION_PROFILES))
        raise ValueError(
            f"Unknown MiniMax H3 optimization profile {profile!r}; allowed: {allowed}."
        ) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprinted_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"path": str(path), "present": False, "size_bytes": None, "sha256": None}
    return {
        "path": str(path),
        "present": True,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def h3_runtime_fingerprint(
    *,
    binary: str | Path,
    model: str | Path,
    optimization_profile: str,
    tokenity_code_revision: str | None,
    verify_rank: int | None = None,
) -> dict[str, object]:
    """Fingerprint the complete native/runtime/checkpoint/profile contract.

    The contract digest intentionally contains both manifest-declared shard
    hashes, so it is identical on rank 0 and rank 1 even though each node only
    has to read and verify its own 25 GB shard.
    """

    flags = h3_optimization_environment(optimization_profile)
    binary_path = Path(binary).resolve()
    runtime_roots = (binary_path.parent.parent, binary_path.parent.parent.parent)

    def runtime_artifact(relative: str) -> Path:
        candidates = [root / relative for root in runtime_roots]
        return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])

    model_root = Path(model).resolve()
    manifest_path = model_root / "tp2" / "manifest.json"
    artifacts = {
        "binary": _fingerprinted_file(binary_path),
        "libmlx": _fingerprinted_file(runtime_artifact("lib/mlx/lib/libmlx.dylib")),
        "libmlxc": _fingerprinted_file(runtime_artifact("lib/mlx/lib/libmlxc.dylib")),
        "metallib": _fingerprinted_file(runtime_artifact("lib/mlx/lib/mlx.metallib")),
        "tp2_manifest": _fingerprinted_file(manifest_path),
    }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        manifest = {}
    declared: dict[str, dict[str, object]] = {}
    rank_artifacts = manifest.get("rank_artifacts") if isinstance(manifest, dict) else None
    if isinstance(rank_artifacts, list):
        for item in rank_artifacts:
            if not isinstance(item, dict):
                continue
            relative = str(item.get("path", ""))
            if relative in {
                "rank-0/transformer.safetensors",
                "rank-1/transformer.safetensors",
            }:
                declared[relative.split("/", 1)[0]] = {
                    "path": relative,
                    "size_bytes": item.get("size_bytes"),
                    "sha256": item.get("sha256"),
                }
    rank_shards: dict[str, dict[str, object]] = {}
    for rank in (0, 1):
        name = f"rank-{rank}"
        expected = dict(declared.get(name, {}))
        local_path = model_root / "tp2" / name / "transformer.safetensors"
        verified: bool | None = None
        if verify_rank == rank:
            expected_sha = expected.get("sha256")
            expected_size = expected.get("size_bytes")
            verified = bool(
                local_path.is_file()
                and isinstance(expected_sha, str)
                and local_path.stat().st_size == expected_size
                and _sha256_file(local_path) == expected_sha
            )
        rank_shards[name] = expected | {"verified": verified}

    canonical_artifacts = {
        role: {
            "present": artifact["present"],
            "size_bytes": artifact["size_bytes"],
            "sha256": artifact["sha256"],
        }
        for role, artifact in artifacts.items()
    }
    contract = {
        "schema_version": 1,
        # Absolute paths are rank-local deployment details.  The contract is
        # about content and must remain identical when the same runtime/model
        # is installed under different roots on the two Macs.
        "artifacts": canonical_artifacts,
        "rank_shards": declared,
        "optimization_profile": optimization_profile,
        "optimization_flags": flags,
        "tokenity_code_revision": tokenity_code_revision,
        "distributed_protocol": H3_DISTRIBUTED_PROTOCOL_VERSION,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return contract | {
        "artifacts": artifacts,
        "rank_shards": rank_shards,
        "contract_sha256": hashlib.sha256(encoded).hexdigest(),
    }


@dataclass(frozen=True)
class H3BackendCapabilities:
    binary: str
    executable: bool
    help_available: bool
    distributed_protocol_version: int | None
    detail: str | None = None

    @property
    def supports_distributed_h3(self) -> bool:
        return self.distributed_protocol_version == H3_DISTRIBUTED_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {"supports_distributed_h3": self.supports_distributed_h3}


def native_h3_environment(
    binary: str | Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if environ is None else environ)
    path = Path(binary).resolve()
    roots = (path.parent.parent, path.parent.parent.parent)
    libraries: list[str] = []
    for root in roots:
        for relative in ("lib", "lib/llama/lib", "lib/mlx/lib"):
            candidate = root / relative
            if candidate.is_dir() and str(candidate) not in libraries:
                libraries.append(str(candidate))
    existing = env.get("DYLD_LIBRARY_PATH")
    if existing:
        libraries.append(existing)
    if libraries:
        env["DYLD_LIBRARY_PATH"] = os.pathsep.join(libraries)
    return env


def build_native_h3_command(
    *,
    binary: str,
    model: str,
    host: str,
    port: int,
    rank: int = 0,
    world_size: int = 1,
) -> list[str]:
    if rank < 0 or world_size < 1 or rank >= world_size:
        raise ValueError("MiniMax H3 rank must be in [0, world_size).")
    command = [
        binary,
        "--model",
        model,
        "--serve",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if world_size > 1:
        command.extend(
            [
                "--h3-distributed-rank",
                str(rank),
                "--h3-distributed-world-size",
                str(world_size),
                "--h3-distributed-protocol",
                str(H3_DISTRIBUTED_PROTOCOL_VERSION),
            ]
        )
    return command


def read_h3_config(model: str | Path) -> dict[str, object]:
    try:
        payload = json.loads((Path(model) / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def h3_model_issues(model: str | Path, *, require_complete: bool = True) -> list[str]:
    root = Path(model)
    issues: list[str] = []
    if not root.is_dir():
        return [f"MiniMax H3 model directory does not exist: {root}"]
    config = read_h3_config(root)
    if str(config.get("model_type", "")).lower() != H3_MODEL_TYPE:
        issues.append(
            f"Expected config.json model_type '{H3_MODEL_TYPE}', got "
            f"'{config.get('model_type') or 'unknown'}'."
        )
    if require_complete:
        for name in (
            "transformer.safetensors",
            "text_encoder.safetensors",
            "video_vae.safetensors",
            "audio_vae.safetensors",
        ):
            if not (root / name).is_file():
                issues.append(f"MiniMax H3 checkpoint is missing {name}.")
    return issues


def h3_tp2_model_issues(
    model: str | Path,
    *,
    verify_rank: int | None = None,
) -> list[str]:
    root = Path(model)
    manifest_path = root / "tp2" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ["MiniMax H3 TP2 checkpoint is missing tp2/manifest.json."]
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        return [f"MiniMax H3 TP2 manifest is invalid: {exc}."]
    if not isinstance(manifest, dict):
        return ["MiniMax H3 TP2 manifest must be a JSON object."]

    issues: list[str] = []
    expected = {
        "schema_version": 1,
        "model_type": H3_MODEL_TYPE,
        "protocol": H3_DISTRIBUTED_PROTOCOL_VERSION,
        "world_size": 2,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            issues.append(
                f"MiniMax H3 TP2 manifest {field} must be {value!r}, "
                f"got {manifest.get(field)!r}."
            )

    expected_rank_files = [
        "rank-0/transformer.safetensors",
        "rank-1/transformer.safetensors",
    ]
    if manifest.get("rank_files") != expected_rank_files:
        issues.append(
            "MiniMax H3 TP2 manifest rank_files must name the exact rank-0 and "
            "rank-1 transformer shards."
        )
    required_rank_files = expected_rank_files
    if verify_rank is not None:
        if verify_rank not in (0, 1):
            issues.append(f"MiniMax H3 TP2 verify_rank must be 0 or 1, got {verify_rank}.")
            required_rank_files = []
        else:
            required_rank_files = [expected_rank_files[verify_rank]]
    for relative in required_rank_files:
        if not (root / "tp2" / relative).is_file():
            issues.append(f"MiniMax H3 TP2 checkpoint is missing tp2/{relative}.")

    artifacts = manifest.get("rank_artifacts")
    artifact_by_path: dict[str, dict[str, object]] = {}
    if isinstance(artifacts, list):
        artifact_by_path = {
            str(item.get("path")): item
            for item in artifacts
            if isinstance(item, dict)
        }
    if sorted(artifact_by_path) != expected_rank_files:
        issues.append(
            "MiniMax H3 TP2 manifest rank_artifacts must describe both exact rank shards."
        )
    else:
        for rank, relative in enumerate(expected_rank_files):
            artifact = artifact_by_path[relative]
            path = root / "tp2" / relative
            expected_size = artifact.get("size_bytes")
            expected_sha = artifact.get("sha256")
            if not isinstance(expected_size, int) or expected_size < 0:
                issues.append(f"MiniMax H3 TP2 artifact {relative} has an invalid size_bytes.")
            elif path.is_file() and path.stat().st_size != expected_size:
                issues.append(f"MiniMax H3 TP2 artifact {relative} size does not match its manifest.")
            if (
                not isinstance(expected_sha, str)
                or len(expected_sha) != 64
                or any(char not in "0123456789abcdef" for char in expected_sha)
            ):
                issues.append(f"MiniMax H3 TP2 artifact {relative} has an invalid sha256.")
            elif verify_rank == rank and path.is_file():
                digest = hashlib.sha256()
                try:
                    with path.open("rb") as handle:
                        while chunk := handle.read(16 * 1024 * 1024):
                            digest.update(chunk)
                except OSError as exc:
                    issues.append(f"Could not verify MiniMax H3 TP2 artifact {relative}: {exc}.")
                else:
                    if digest.hexdigest() != expected_sha:
                        issues.append(
                            f"MiniMax H3 TP2 artifact {relative} sha256 does not match its manifest."
                        )

    sharding = manifest.get("sharding")
    expected_geometry = {
        "main_blocks": 50,
        "attention_heads": 56,
        "attention_heads_per_rank": 28,
        "ffn_hidden": 14_336,
        "ffn_hidden_per_rank": 7_168,
    }
    if not isinstance(sharding, dict):
        issues.append("MiniMax H3 TP2 manifest is missing sharding geometry.")
    else:
        for field, value in expected_geometry.items():
            if sharding.get(field) != value:
                issues.append(
                    f"MiniMax H3 TP2 sharding {field} must be {value}, "
                    f"got {sharding.get(field)!r}."
                )
    return issues


def probe_h3_backend(
    binary: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> H3BackendCapabilities:
    path = Path(binary)
    if not path.is_file():
        return H3BackendCapabilities(
            binary=str(path),
            executable=False,
            help_available=False,
            distributed_protocol_version=None,
            detail="Native mlx-serve binary does not exist.",
        )
    if not os.access(path, os.X_OK):
        return H3BackendCapabilities(
            binary=str(path),
            executable=False,
            help_available=False,
            distributed_protocol_version=None,
            detail="Native mlx-serve binary is not executable.",
        )
    try:
        completed = runner(
            [str(path), "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=native_h3_environment(path),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return H3BackendCapabilities(
            binary=str(path),
            executable=True,
            help_available=False,
            distributed_protocol_version=None,
            detail=f"Could not inspect native mlx-serve capabilities: {exc}",
        )
    help_text = f"{completed.stdout}\n{completed.stderr}"
    supports_protocol = all(marker in help_text for marker in H3_DISTRIBUTED_HELP_MARKERS)
    return H3BackendCapabilities(
        binary=str(path),
        executable=True,
        help_available=completed.returncode == 0 or bool(help_text.strip()),
        distributed_protocol_version=(
            H3_DISTRIBUTED_PROTOCOL_VERSION if supports_protocol else None
        ),
        detail=(
            None
            if supports_protocol
            else "Native backend has no Tokenity MiniMax H3 distributed-rank protocol."
        ),
    )


def h3_runtime_preflight(
    *,
    binary: str,
    model: str,
    world_size: int,
    rank: int | None = None,
    backend_probe: Callable[[str | Path], H3BackendCapabilities] = probe_h3_backend,
) -> tuple[list[str], H3BackendCapabilities]:
    # A TP2 worker only tokenizes the canonical prompt and loads its local DiT
    # shard. Rank 0 owns the full transformer fallback, text encoder, VAEs, and
    # media response path. Requiring every large component on rank 1 defeats
    # per-rank checkpoint deployment and needlessly doubles worker storage.
    require_complete = world_size == 1 or rank in (None, 0)
    issues = h3_model_issues(model, require_complete=require_complete)
    capabilities = backend_probe(binary)
    if not capabilities.executable:
        issues.append(capabilities.detail or "Native mlx-serve binary is unavailable.")
    if world_size > 1 and not capabilities.supports_distributed_h3:
        issues.append(
            "Two-Mac MiniMax H3 requires native backend distributed protocol v1; "
            "this binary only supports single-node video inference."
        )
    if world_size > 1:
        issues.extend(h3_tp2_model_issues(model, verify_rank=rank))
    return issues, capabilities


def exec_native_h3(
    *,
    binary: str,
    model: str,
    host: str,
    port: int,
    rank: int,
    world_size: int,
    environ: Mapping[str, str] | None = None,
    exec_fn: Callable[[str, Sequence[str], Mapping[str, str]], object] = os.execvpe,
) -> None:
    issues, _ = h3_runtime_preflight(
        binary=binary,
        model=model,
        world_size=world_size,
        rank=rank,
    )
    if issues:
        raise RuntimeError("MiniMax H3 runtime preflight failed: " + "; ".join(issues))
    command = build_native_h3_command(
        binary=binary,
        model=model,
        host=host,
        port=port,
        rank=rank,
        world_size=world_size,
    )
    env = native_h3_environment(binary, environ)
    env.setdefault("MLX_METAL_FAST_SYNCH", "1")
    exec_fn(binary, command, env)
    raise RuntimeError("Native MiniMax H3 exec unexpectedly returned.")
