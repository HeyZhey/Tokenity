from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class RecoveryAssessment:
    safe_to_adopt: bool
    reason: str
    record: dict[str, object]
    runtime: dict[str, object] | None = None


class InstanceStateStore:
    """Small atomic journals used to recover Agent ownership after a restart."""

    def __init__(self, root: Path):
        self.root = root

    @property
    def readable(self) -> bool:
        parent = self.root if self.root.exists() else self.root.parent
        return os.access(parent, os.R_OK | os.W_OK)

    def write(self, instance_id: str, record: dict[str, object]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{_safe_component(instance_id)}.json"
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
        return path

    def remove(self, instance_id: str) -> None:
        (self.root / f"{_safe_component(instance_id)}.json").unlink(missing_ok=True)

    def records(self) -> list[dict[str, object]]:
        if not self.root.is_dir():
            return []
        records: list[dict[str, object]] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payload["_journal_path"] = str(path)
                records.append(payload)
        return records


def process_start_identity(pid: int) -> str | None:
    try:
        completed = subprocess.run(
            ["/bin/ps", "-o", "lstart=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = " ".join(completed.stdout.split())
    return value or None


def process_command(pid: int) -> str | None:
    try:
        completed = subprocess.run(
            ["/bin/ps", "-o", "command=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def assess_recovery_record(
    record: dict[str, object],
    *,
    now: float | None = None,
    maximum_status_age_seconds: float = 15.0,
    start_identity_fn: Callable[[int], str | None] = process_start_identity,
    command_fn: Callable[[int], str | None] = process_command,
) -> RecoveryAssessment:
    now = time.time() if now is None else now
    instance = record.get("instance")
    process = record.get("process")
    status_path = record.get("status_path")
    if not isinstance(instance, dict) or not isinstance(process, dict):
        return RecoveryAssessment(False, "journal_schema_invalid", record)
    instance_id = instance.get("instance_id")
    operation_id = instance.get("operation_id")
    model_revision = instance.get("model_revision")
    resolved_path = instance.get("resolved_path")
    required_instance_fields = (
        instance_id,
        operation_id,
        resolved_path,
        instance.get("world_size"),
        instance.get("connection_mode"),
    )
    if any(value is None or value == "" for value in required_instance_fields):
        return RecoveryAssessment(False, "journal_identity_incomplete", record)
    pid = process.get("pid")
    recorded_start = process.get("start_identity")
    if not isinstance(pid, int) or pid <= 0 or not isinstance(recorded_start, str):
        return RecoveryAssessment(False, "process_identity_incomplete", record)
    observed_start = start_identity_fn(pid)
    if observed_start is None:
        return RecoveryAssessment(False, "process_missing", record)
    if observed_start != recorded_start:
        return RecoveryAssessment(False, "process_start_identity_mismatch", record)
    observed_command = command_fn(pid)
    if observed_command is None:
        return RecoveryAssessment(False, "process_command_unavailable", record)
    if (
        "tokenity distributed-openai serve --model" not in observed_command
        or str(resolved_path) not in observed_command
    ):
        return RecoveryAssessment(False, "process_command_mismatch", record)
    if not isinstance(status_path, str):
        return RecoveryAssessment(False, "runtime_status_path_missing", record)
    try:
        runtime = json.loads(Path(status_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return RecoveryAssessment(False, "runtime_status_unreadable", record)
    if not isinstance(runtime, dict):
        return RecoveryAssessment(False, "runtime_status_invalid", record)
    updated_at = runtime.get("updated_at")
    if not isinstance(updated_at, (int, float)) or now - updated_at > maximum_status_age_seconds:
        return RecoveryAssessment(False, "runtime_heartbeat_stale", record, runtime)
    expected = {
        "instance_id": instance_id,
        "operation_id": operation_id,
        "model_revision": model_revision,
        "world_size": instance.get("world_size"),
        "connection_mode": instance.get("connection_mode"),
    }
    for key, value in expected.items():
        if runtime.get(key) != value:
            return RecoveryAssessment(False, f"runtime_{key}_mismatch", record, runtime)
    rank = record.get("rank")
    if not isinstance(rank, int) or runtime.get("rank") != rank:
        return RecoveryAssessment(False, "runtime_rank_mismatch", record, runtime)
    if runtime.get("phase") in {"failed", "stopping"}:
        return RecoveryAssessment(False, f"runtime_{runtime.get('phase')}", record, runtime)
    return RecoveryAssessment(True, "identity_revision_epoch_and_runtime_verified", record, runtime)


def _safe_component(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_.-" else "_" for character in value)[:96]
