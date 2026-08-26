from __future__ import annotations

import asyncio
import ctypes
import getpass
import hashlib
import http.client
import importlib.metadata
import json
import logging
import os
import platform
import re
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from fastapi import FastAPI, HTTPException, Query, Request as FastAPIRequest
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from tokenity import __version__
from tokenity.control.instances import (
    GenerationQueueFull,
    GenerationSlotCancelled,
    GenerationSlotScheduler,
    GenerationSlotTimeout,
    InstanceConflict,
    InstanceLifecycle,
    InstanceRegistry,
    InstanceRouter,
    MemoryReservationBreakdown,
    ResourceAdmissionError,
    ResourceLedger,
    live_system_available_memory_bytes,
)
from tokenity.control.routing import (
    AutoRouter,
    CapabilityRegistry,
    ModelCapabilityProfile,
    ModelRuntimeState,
    RouteContext,
    RouteDecision,
    RoutePolicy,
    RouteReason,
)
from tokenity.inference.native_mtp import scan_native_mtp_capability
from tokenity.mlx.hostfile import (
    ClusterNode,
    ConnectionMode,
    HostfileError,
    build_hostfile,
    is_loopback_host,
)
from tokenity.mlx.rdma_probe import RDMAProbeResult, probe_rdma
from tokenity.model_inspection import (
    distributed_model_issue,
    model_usage_metadata,
    standalone_model_issue,
)
from tokenity.paths import code_root, model_root, runtime_root
from tokenity.node_agent.recovery import (
    InstanceStateStore,
    assess_recovery_record,
    process_command,
    process_start_identity,
)
from tokenity.process.supervisor import RoleSupervisor
from tokenity.serving.minimax_h3_video import (
    DEFAULT_MLX_SERVE_BINARY,
    H3BackendCapabilities,
    H3_DISTRIBUTED_PROTOCOL_VERSION,
    h3_optimization_environment,
    h3_runtime_fingerprint,
    h3_runtime_preflight,
)


DEFAULT_MODEL_ROOT = str(model_root())
DEFAULT_RUNTIME_MANIFEST = str(runtime_root() / "runtime-manifest.json")
MINIMUM_MEMORY_HEADROOM_RATIO = 0.25
PORT_REUSE_QUARANTINE_SECONDS = 30.0
NODE_AGENT_CONTRACT_VERSION = 1
NODE_AGENT_CAPABILITIES = (
    "agent_health",
    "agent_recovery",
    "cluster_runtime",
    "minimax_h3_video",
    "instance_quorum",
    "instance_runtimes",
    "managed_instances",
    "native_mtp",
)
MANAGED_INSTANCE_CAPABILITIES = frozenset(
    {
        "cluster_runtime",
        "instance_quorum",
        "instance_runtimes",
        "managed_instances",
    }
)
TOKENITY_AUTO_MODEL_ID = "tokenity-auto"
# The H3 TP2 data plane is validated on a dedicated communicator range.  Text
# runtimes retain their existing 30020 deployments; H3 must not reuse them.
H3_COLLECTIVE_STARTING_PORT = 30_096
TOKENITY_ROUTE_FIELDS = frozenset(
    {
        "tokenity_route_policy",
        "tokenity_session_id",
        "tokenity_lock_model",
        "tokenity_constraints",
    }
)
COMMON_CHAT_RUNTIME_PARAMETERS = frozenset(
    {
        "frequency_penalty",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "metadata",
        "min_p",
        "n",
        "presence_penalty",
        "repetition_penalty",
        "seed",
        "service_tier",
        "stop",
        "stream",
        "stream_options",
        "temperature",
        "top_k",
        "top_logprobs",
        "top_p",
        "user",
    }
)
MODEL_ROLES = (
    "distributed-openai",
    "distributed-openai-rank",
    "minimax-h3-video",
    "minimax-h3-video-rank",
    "single-node-openai",
)
NATIVE_ADMIN_STOP_ROLES = frozenset(
    {"distributed-openai", "minimax-h3-video", "single-node-openai"}
)
COORDINATED_WORKER_ROLES = frozenset(
    {"distributed-openai-rank", "minimax-h3-video-rank"}
)
VIDEO_MODEL_ROLES = frozenset(
    {"minimax-h3-video", "minimax-h3-video-rank"}
)


class _LaunchCancelled(RuntimeError):
    pass


class _GatewayStreamingResponse(StreamingResponse):
    """Streaming response whose cleanup covers the complete ASGI lifecycle."""

    def __init__(self, *args, on_close, on_disconnect=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._on_close = on_close
        self._on_disconnect = on_disconnect or on_close

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._on_close()

    async def listen_for_disconnect(self, receive) -> None:
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                self._on_disconnect()
                return


class ClusterNodePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    agent_url: Optional[str] = None
    lan_ip: Optional[str] = None
    rdma_ip: Optional[str] = None
    rdma_devices: List[str] = Field(default_factory=list)
    rdma_matrix_row: Optional[List[Optional[str]]] = None

    def to_cluster_node(self) -> ClusterNode:
        return ClusterNode(
            id=self.id,
            agent_url=self.agent_url,
            lan_ip=self.lan_ip,
            rdma_ip=self.rdma_ip,
            rdma_devices=self.rdma_devices,
            rdma_matrix_row=self.rdma_matrix_row,
        )


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    nodes: List[ClusterNodePayload] = Field(default_factory=list)
    connection_mode: ConnectionMode = ConnectionMode.RING
    python: str = Field(default_factory=lambda: sys.executable)
    starting_port: int = 29500
    host: str = "0.0.0.0"
    port: int = 8000
    dry_run: bool = True
    api_identifier: Optional[str] = None
    max_tokens: int = Field(default=32_768, ge=1, le=262_144)
    prompt_cache_size: int = Field(default=4, ge=1, le=64)
    prefill_step_size: int = Field(default=2_048, ge=128, le=8_192)
    decode_concurrency: int = Field(default=1, ge=1, le=8)
    prompt_concurrency: int = Field(default=1, ge=1, le=8)
    trust_remote_code: bool = False
    lease_seconds: float = Field(default=30.0, ge=15.0, le=300.0)
    native_mtp: "NativeMTPRequest" = Field(default_factory=lambda: NativeMTPRequest())
    instance_id: Optional[str] = Field(default=None, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    operation_id: Optional[str] = Field(default=None, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    memory_reservation_bytes: Optional[int] = Field(default=None, ge=0)


class NativeMTPRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["off", "auto", "required"] = "off"
    max_depth: Literal[1] = 1
    head_placement: Literal["replicated"] = "replicated"


class H3VideoStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    binary: str = DEFAULT_MLX_SERVE_BINARY
    nodes: List[ClusterNodePayload] = Field(default_factory=list, max_length=2)
    connection_mode: ConnectionMode = ConnectionMode.JACCL
    python: str = Field(default_factory=lambda: sys.executable)
    starting_port: int = Field(default=H3_COLLECTIVE_STARTING_PORT, ge=1, le=65_535)
    host: str = "0.0.0.0"
    port: int = Field(default=11_241, ge=1, le=65_535)
    dry_run: bool = True
    api_identifier: Optional[str] = None
    lease_seconds: float = Field(default=30.0, ge=15.0, le=300.0)
    instance_id: Optional[str] = Field(
        default=None,
        min_length=8,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    operation_id: Optional[str] = Field(
        default=None,
        min_length=8,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    memory_reservation_bytes: Optional[int] = Field(default=None, ge=0)
    optimization_profile: Literal["baseline", "block-fusions", "stock-qmm"] = "stock-qmm"
    minimum_free_disk_bytes: int = Field(default=2 * 1024**3, ge=0)


class H3VideoRankStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    model: str
    binary: str = DEFAULT_MLX_SERVE_BINARY
    rank: int = Field(ge=0)
    world_size: int = Field(ge=1, le=2)
    coordinator: bool = False
    connection_mode: ConnectionMode = ConnectionMode.JACCL_RING
    python: str
    coordinator_ip: Optional[str] = None
    starting_port: int = Field(default=H3_COLLECTIVE_STARTING_PORT, ge=1, le=65_535)
    ring_hosts: List[List[str]] = Field(default_factory=list)
    rdma_matrix: List[List[Optional[str]]] = Field(default_factory=list)
    host: str = "0.0.0.0"
    port: int = Field(default=11_241, ge=1, le=65_535)
    api_identifier: Optional[str] = None
    lease_seconds: float = Field(default=30.0, ge=15.0, le=300.0)
    instance_id: Optional[str] = Field(
        default=None,
        min_length=8,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    operation_id: Optional[str] = Field(
        default=None,
        min_length=8,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    model_revision: Optional[str] = Field(default=None, min_length=8, max_length=128)
    tokenity_code_revision: Optional[str] = Field(default=None, min_length=8, max_length=128)
    memory_reservation_bytes: Optional[int] = Field(default=None, ge=0)
    optimization_profile: Literal["baseline", "block-fusions", "stock-qmm"] = "stock-qmm"
    minimum_free_disk_bytes: int = Field(default=2 * 1024**3, ge=0)
    dry_run: bool = False
    runtime_contract_sha256: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class StopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal[
        "distributed-openai",
        "distributed-openai-rank",
        "minimax-h3-video",
        "minimax-h3-video-rank",
        "single-node-openai",
        "node-agent",
    ]
    timeout: float = 10.0
    instance_id: Optional[str] = None


class StopAllRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout: float = Field(default=10.0, ge=0.1, le=30.0)


class HeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ttl_seconds: float = Field(default=30.0, ge=15.0, le=300.0)
    instance_id: Optional[str] = None


class RankStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    model: str
    rank: int = Field(ge=0)
    world_size: int = Field(ge=1)
    coordinator: bool = False
    connection_mode: ConnectionMode = ConnectionMode.RING
    python: str
    coordinator_ip: Optional[str] = None
    starting_port: int = Field(default=29_500, ge=1, le=65_535)
    ring_hosts: List[List[str]] = Field(default_factory=list)
    rdma_matrix: List[List[Optional[str]]] = Field(default_factory=list)
    host: str = "0.0.0.0"
    port: int = 8_000
    api_identifier: Optional[str] = None
    max_tokens: int = Field(default=32_768, ge=1, le=262_144)
    prompt_cache_size: int = Field(default=4, ge=1, le=64)
    prefill_step_size: int = Field(default=2_048, ge=128, le=8_192)
    decode_concurrency: int = Field(default=1, ge=1, le=8)
    prompt_concurrency: int = Field(default=1, ge=1, le=8)
    trust_remote_code: bool = False
    lease_seconds: float = Field(default=30.0, ge=15.0, le=300.0)
    native_mtp: NativeMTPRequest = Field(default_factory=NativeMTPRequest)
    instance_id: Optional[str] = Field(default=None, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    operation_id: Optional[str] = Field(default=None, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    model_revision: Optional[str] = Field(default=None, min_length=8, max_length=128)
    tokenity_code_revision: Optional[str] = Field(default=None, min_length=8, max_length=128)
    memory_reservation_bytes: Optional[int] = Field(default=None, ge=0)


def _disk_stats(path: str | Path = "/") -> dict[str, object]:
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    return {
        "path": str(target),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def create_app(
    *,
    rdma_probe_fn=probe_rdma,
    supervisor: RoleSupervisor | None = None,
    post_json_fn=None,
    get_json_fn=None,
    rank_ready_fn=None,
    rank_stabilize_fn=None,
    rank_connected_fn=None,
    runtime_preflight_fn=None,
    h3_runtime_preflight_fn=None,
    h3_ready_fn=None,
    gateway_open_fn=None,
    capability_profiles=None,
    generation_queue_depth: int = 32,
    generation_slot_timeout: float = 30.0,
    instance_state_store: InstanceStateStore | None = None,
) -> FastAPI:
    app = FastAPI(title="Tokenity Node Agent", version=__version__)
    roles = supervisor or RoleSupervisor()
    post_json = post_json_fn or _post_json
    get_json = get_json_fn or _get_json
    wait_for_rank = rank_ready_fn or _wait_for_listening_port
    stabilize_rank = rank_stabilize_fn or time.sleep
    wait_for_rank_connection = rank_connected_fn or _wait_for_rank_connection
    runtime_preflight = runtime_preflight_fn or _runtime_preflight
    h3_preflight = h3_runtime_preflight_fn or h3_runtime_preflight
    wait_for_h3 = h3_ready_fn or _wait_for_h3_health
    gateway_open = gateway_open_fn or _open_gateway_upstream
    instances = InstanceRegistry()
    instance_router = InstanceRouter(instances)
    generation_scheduler = GenerationSlotScheduler(
        max_queue_depth=generation_queue_depth,
        registry=instances,
    )
    capability_registry = CapabilityRegistry()
    for profile in capability_profiles or ():
        capability_registry.register(profile)
    auto_router = AutoRouter()
    session_routes: dict[str, tuple[str, str]] = {}
    resource_ledger = ResourceLedger(
        _total_memory_bytes() or 0,
        minimum_headroom_ratio=MINIMUM_MEMORY_HEADROOM_RATIO,
        port_reuse_delay_seconds=PORT_REUSE_QUARANTINE_SECONDS,
    )
    instance_workers: dict[str, list[str]] = {}
    instance_roles: dict[str, str] = {}
    instance_ports: dict[str, int] = {}
    instance_status_paths: dict[str, str] = {}
    lifecycle_lock = threading.RLock()
    watchdog_stop = threading.Event()
    launch_generation = 0
    launch_tokens: dict[str, int] = {}
    lease_deadline: float | None = None
    cluster_runtime: dict[str, object] | None = None
    cluster_runtimes: dict[str, dict[str, object]] = {}
    startup_orphans: list[dict[str, object]] = []
    startup_orphan_records: dict[int, dict[str, object]] = {}
    if instance_state_store is None:
        configured_state_root = os.environ.get("TOKENITY_INSTANCE_STATE_ROOT")
        if configured_state_root:
            instance_state_store = InstanceStateStore(Path(configured_state_root))
    process_started_at = time.time()
    event_loop_heartbeat = time.monotonic()
    event_loop_heartbeat_task: asyncio.Task[None] | None = None
    last_fatal_internal_error: dict[str, object] | None = None
    instance_runtime_versions: dict[str, int] = {}

    def persist_instance_state(
        instance,
        *,
        role: str,
        rank: int,
        command: list[str],
        status_path: str | None,
        pid: int | None,
        worker_agent_urls: list[str] | None = None,
        recovery_policy: str = "adopt",
    ) -> None:
        if instance_state_store is None or pid is None:
            return
        start_identity = process_start_identity(pid)
        if start_identity is None:
            return
        ports = list(
            range(instance.starting_port, instance.starting_port + max(1, instance.world_size))
        )
        if role in NATIVE_ADMIN_STOP_ROLES:
            ports.insert(0, instance.http_port)
        runtime_snapshot = cluster_runtimes.get(instance.instance_id)
        instance_state_store.write(
            instance.instance_id,
            {
                "schema_version": 1,
                "instance": instance.to_dict(),
                "role": role,
                "rank": rank,
                "command": list(command),
                "status_path": status_path,
                "recovery_policy": recovery_policy,
                "ports": ports,
                "worker_agent_urls": list(worker_agent_urls or []),
                "cluster_runtime": dict(runtime_snapshot) if runtime_snapshot else None,
                "process": {
                    "pid": pid,
                    "start_identity": start_identity,
                },
                "journaled_at": time.time(),
            },
        )

    def remove_instance_state(instance_id: str) -> None:
        if instance_state_store is not None:
            instance_state_store.remove(instance_id)

    def advance_recovered_instance(instance, runtime: dict[str, object]) -> None:
        try:
            instance.transition(InstanceLifecycle.LAUNCHING)
            if instance.world_size > 1:
                instance.transition(InstanceLifecycle.DISTRIBUTED_INITIALIZING)
            instance.transition(InstanceLifecycle.LOADING_METADATA)
        except RuntimeError:
            return
        role = instance_roles.get(instance.instance_id)
        status = supervisor_status(role, instance.instance_id) if role else None
        reconcile_instance_runtime(instance, runtime, status, role)

    def recover_persisted_instances() -> set[int]:
        nonlocal cluster_runtime
        if instance_state_store is None:
            return set()
        adopted_pids: set[int] = set()
        for record in instance_state_store.records():
            instance_payload = record.get("instance")
            process_payload = record.get("process")
            if not isinstance(instance_payload, dict) or not isinstance(process_payload, dict):
                continue
            instance_id = str(instance_payload.get("instance_id") or "")
            pid = process_payload.get("pid")
            if record.get("recovery_policy") == "cleanup":
                # Native H3 does not publish Tokenity's epoch/status heartbeat,
                # so adopting it after an Agent restart would invent authority.
                # A PID/start-time/command-fenced journal makes precise cleanup
                # safe while leaving unrelated and PID-reused processes alone.
                if isinstance(pid, int) and process_start_identity(pid) is None:
                    if instance_id:
                        remove_instance_state(instance_id)
                    continue
                cleanup_safe = orphan_process_identity_matches(record)
                if cleanup_safe and isinstance(pid, int):
                    startup_orphan_records[pid] = record
                startup_orphans.append(
                    {
                        "pid": pid,
                        "instance_id": instance_id or None,
                        "operation_id": instance_payload.get("operation_id"),
                        "state": "orphaned",
                        "reason": "native_runtime_requires_fenced_restart_cleanup",
                        "cleanup_required": True,
                        "cleanup_safe": cleanup_safe,
                        "journal_path": record.get("_journal_path"),
                    }
                )
                continue
            assessment = assess_recovery_record(record)
            if not assessment.safe_to_adopt:
                if assessment.reason == "process_missing" and instance_id:
                    remove_instance_state(instance_id)
                    continue
                cleanup_safe = orphan_process_identity_matches(record)
                if cleanup_safe and isinstance(pid, int):
                    startup_orphan_records[pid] = record
                startup_orphans.append(
                    {
                        "pid": pid,
                        "instance_id": instance_id or None,
                        "operation_id": instance_payload.get("operation_id"),
                        "state": "orphaned",
                        "reason": assessment.reason,
                        "cleanup_required": True,
                        "cleanup_safe": cleanup_safe,
                        "journal_path": record.get("_journal_path"),
                    }
                )
                continue
            role = str(record.get("role") or "")
            command = record.get("command")
            status_path = record.get("status_path")
            start_identity = process_payload.get("start_identity")
            if (
                not instance_id
                or not isinstance(pid, int)
                or not isinstance(start_identity, str)
                or not isinstance(command, list)
                or not all(isinstance(item, str) for item in command)
                or not isinstance(status_path, str)
                or role not in MODEL_ROLES
            ):
                cleanup_safe = orphan_process_identity_matches(record)
                if cleanup_safe and isinstance(pid, int):
                    startup_orphan_records[pid] = record
                startup_orphans.append(
                    {
                        "pid": pid,
                        "instance_id": instance_id or None,
                        "state": "orphaned",
                        "reason": "recovery_schema_or_role_invalid",
                        "cleanup_required": True,
                        "cleanup_safe": cleanup_safe,
                    }
                )
                continue
            adopt = getattr(roles, "adopt", None)
            if adopt is None:
                cleanup_safe = orphan_process_identity_matches(record)
                if cleanup_safe:
                    startup_orphan_records[pid] = record
                startup_orphans.append(
                    {
                        "pid": pid,
                        "instance_id": instance_id,
                        "state": "orphaned",
                        "reason": "supervisor_does_not_support_adoption",
                        "cleanup_required": True,
                        "cleanup_safe": cleanup_safe,
                    }
                )
                continue
            try:
                instance, created = instances.create(
                    instance_id=instance_id,
                    operation_id=str(instance_payload["operation_id"]),
                    requested_model_id=str(instance_payload["requested_model_id"]),
                    resolved_path=str(instance_payload["resolved_path"]),
                    model_revision=instance_payload.get("model_revision"),
                    tokenizer_identity=instance_payload.get("tokenizer_identity"),
                    execution_mode=str(instance_payload["execution_mode"]),
                    selected_nodes=[str(value) for value in instance_payload.get("selected_nodes") or []],
                    rank_mapping={
                        str(key): int(value)
                        for key, value in dict(instance_payload.get("rank_mapping") or {}).items()
                    },
                    world_size=int(instance_payload["world_size"]),
                    connection_mode=str(instance_payload["connection_mode"]),
                    coordinator=str(instance_payload["coordinator"]),
                    http_port=int(instance_payload["http_port"]),
                    starting_port=int(instance_payload["starting_port"]),
                    memory_reservation_bytes=int(instance_payload["memory_reservation_bytes"]),
                    memory_reservation_breakdown=instance_payload.get("memory_reservation_breakdown"),
                )
                if not created:
                    continue
                ports = [int(value) for value in record.get("ports") or []]
                resource_ledger.reserve(
                    instance_id,
                    instance.memory_reservation_bytes,
                    ports,
                    system_available_memory_bytes=system_available_memory_bytes(),
                )
                log_paths = instance_payload.get("log_paths")
                log_path = None
                if isinstance(log_paths, dict):
                    log_path = next(
                        (
                            str(value)
                            for value in log_paths.values()
                            if isinstance(value, str)
                        ),
                        None,
                    )
                adopt(
                    role,
                    instance_id=instance_id,
                    operation_id=instance.operation_id,
                    pid=pid,
                    start_identity=start_identity,
                    command=list(command),
                    log_path=log_path,
                )
            except Exception as exc:
                instances.remove(instance_id)
                resource_ledger.release(instance_id)
                cleanup_safe = orphan_process_identity_matches(record)
                if cleanup_safe:
                    startup_orphan_records[pid] = record
                startup_orphans.append(
                    {
                        "pid": pid,
                        "instance_id": instance_id,
                        "state": "orphaned",
                        "reason": f"adoption_failed:{exc}",
                        "cleanup_required": True,
                        "cleanup_safe": cleanup_safe,
                    }
                )
                continue
            instance_roles[instance_id] = role
            instance_ports[instance_id] = instance.http_port
            instance_status_paths[instance_id] = status_path
            instance_workers[instance_id] = [
                str(value) for value in record.get("worker_agent_urls") or []
            ]
            runtime_snapshot = record.get("cluster_runtime")
            if isinstance(runtime_snapshot, dict):
                cluster_runtimes[instance_id] = dict(runtime_snapshot)
                cluster_runtime = dict(runtime_snapshot)
            instance.process_identities[
                str(record.get("rank") if role == "distributed-openai-rank" else instance.coordinator)
            ] = pid
            instance.heartbeat(30.0)
            if assessment.runtime is not None:
                advance_recovered_instance(instance, assessment.runtime)
            adopted_pids.add(pid)
        # Fenced cleanup records are already owned by startup recovery.  Treat
        # them as known here so the broad unjournaled scan cannot register a
        # duplicate orphan that remains degraded after the fenced copy is
        # cleaned.
        return adopted_pids | set(startup_orphan_records)

    def orphan_process_identity_matches(record: dict[str, object]) -> bool:
        instance_payload = record.get("instance")
        process_payload = record.get("process")
        command = record.get("command")
        if (
            not isinstance(instance_payload, dict)
            or not isinstance(process_payload, dict)
            or not isinstance(command, list)
        ):
            return False
        pid = process_payload.get("pid")
        recorded_start = process_payload.get("start_identity")
        resolved_path = instance_payload.get("resolved_path")
        if (
            not isinstance(pid, int)
            or not isinstance(recorded_start, str)
            or not isinstance(resolved_path, str)
            or process_start_identity(pid) != recorded_start
        ):
            return False
        observed_command = process_command(pid)
        if observed_command is None or resolved_path not in observed_command:
            return False
        journal_command = " ".join(str(value) for value in command)
        if resolved_path not in journal_command:
            return False
        role = str(record.get("role") or "")
        if role in {"minimax-h3-video", "minimax-h3-video-rank"}:
            return (
                "tokenity minimax-h3-video serve --binary" in observed_command
                or "mlx-serve --model" in observed_command
            ) and "tokenity minimax-h3-video serve --binary" in journal_command
        return (
            role in {"distributed-openai", "distributed-openai-rank", "single-node-openai"}
            and "tokenity distributed-openai serve --model" in observed_command
            and "tokenity distributed-openai serve --model" in journal_command
        )

    def exact_orphan_process_running(record: dict[str, object]) -> bool:
        if not orphan_process_identity_matches(record):
            return False
        process_payload = record.get("process")
        assert isinstance(process_payload, dict)
        pid = int(process_payload["pid"])
        try:
            state = subprocess.run(
                ["/bin/ps", "-o", "stat=", "-p", str(pid)],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return False
        return bool(state) and not state.startswith("Z")

    def update_orphan_audit(pid: int, **values: object) -> dict[str, object]:
        for orphan in startup_orphans:
            if orphan.get("pid") == pid:
                orphan.update(values)
                return orphan
        orphan = {"pid": pid, **values}
        startup_orphans.append(orphan)
        return orphan

    def cleanup_startup_orphan(pid: int) -> dict[str, object]:
        """Stop one journaled process only after revalidating its PID fence."""

        record = startup_orphan_records.get(pid)
        if record is None:
            return update_orphan_audit(
                pid,
                state="orphaned",
                cleanup_required=True,
                cleanup_safe=False,
                cleanup_result="No fenced instance journal is available.",
            )
        if not orphan_process_identity_matches(record):
            startup_orphan_records.pop(pid, None)
            return update_orphan_audit(
                pid,
                state="orphaned",
                cleanup_required=True,
                cleanup_safe=False,
                cleanup_result="Process identity changed; no signal was sent.",
            )

        instance_payload = record.get("instance")
        assert isinstance(instance_payload, dict)
        instance_id = str(instance_payload.get("instance_id") or "")
        role = str(record.get("role") or "")
        if role in NATIVE_ADMIN_STOP_ROLES:
            for agent_url in record.get("worker_agent_urls") or []:
                if not isinstance(agent_url, str) or not instance_id:
                    continue
                try:
                    post_json(
                        f"{agent_url}/v1/node/instances/{instance_id}/stop",
                        {"timeout": 3},
                        6,
                    )
                except Exception:
                    pass
            port = instance_payload.get("http_port")
            if isinstance(port, int):
                try:
                    _post_json(f"http://127.0.0.1:{port}/v1/tokenity/stop", {}, 2.0)
                except Exception:
                    pass

        signals_sent: list[str] = []
        try:
            if exact_orphan_process_running(record):
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                signals_sent.append("SIGTERM")
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + 3.0
        while exact_orphan_process_running(record) and time.monotonic() < deadline:
            time.sleep(0.05)
        try:
            if exact_orphan_process_running(record):
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                signals_sent.append("SIGKILL")
        except (ProcessLookupError, PermissionError):
            pass
        if exact_orphan_process_running(record):
            return update_orphan_audit(
                pid,
                state="orphaned",
                cleanup_required=True,
                cleanup_safe=True,
                cleanup_result="Exact process did not exit after fenced cleanup.",
                signals_sent=signals_sent,
            )

        if instance_id:
            remove_instance_state(instance_id)
        startup_orphan_records.pop(pid, None)
        return update_orphan_audit(
            pid,
            state="cleaned",
            cleanup_required=False,
            cleanup_safe=True,
            cleanup_result="Exact journaled process stopped and journal removed.",
            signals_sent=signals_sent,
            cleaned_at=time.time(),
        )

    def system_available_memory_bytes(
        memory: dict[str, object] | None = None,
    ) -> int | None:
        return live_system_available_memory_bytes(
            memory or _memory_stats(),
            minimum_headroom_ratio=resource_ledger.minimum_headroom_ratio,
        )

    def resource_ledger_snapshot(
        memory: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return resource_ledger.snapshot(
            system_available_memory_bytes=system_available_memory_bytes(memory)
        )

    def begin_launch(instance_id: str) -> int:
        nonlocal launch_generation
        with lifecycle_lock:
            launch_generation += 1
            launch_tokens[instance_id] = launch_generation
            return launch_generation

    def launch_is_current(instance_id: str, generation: int) -> bool:
        with lifecycle_lock:
            return launch_tokens.get(instance_id) == generation

    def cancel_launch(instance_id: str | None = None) -> None:
        with lifecycle_lock:
            if instance_id is None:
                launch_tokens.clear()
            else:
                launch_tokens.pop(instance_id, None)

    def renew_lease(ttl_seconds: float) -> float:
        nonlocal lease_deadline
        with lifecycle_lock:
            lease_deadline = time.monotonic() + ttl_seconds
            return lease_deadline

    def clear_legacy_lease() -> None:
        nonlocal lease_deadline
        with lifecycle_lock:
            lease_deadline = None

    def set_cluster_runtime(request: RankStartRequest, role: str) -> None:
        nonlocal cluster_runtime
        instance_id = request.instance_id or request.cluster_id
        with lifecycle_lock:
            snapshot: dict[str, object] = {
                "cluster_id": request.cluster_id,
                "instance_id": instance_id,
                "operation_id": request.operation_id,
                "rank": request.rank,
                "world_size": request.world_size,
                "connection_mode": request.connection_mode.value,
                "model_revision": request.model_revision,
                "epoch": launch_generation,
                "role": role,
            }
            cluster_runtimes[instance_id] = snapshot
            cluster_runtime = snapshot

    def remove_cluster_runtime(instance_id: str) -> None:
        nonlocal cluster_runtime
        with lifecycle_lock:
            cluster_runtimes.pop(instance_id, None)
            cluster_runtime = (
                cluster_runtimes[next(reversed(cluster_runtimes))]
                if cluster_runtimes
                else None
            )

    def cluster_runtime_snapshot() -> dict[str, object] | None:
        with lifecycle_lock:
            return dict(cluster_runtime) if cluster_runtime is not None else None

    def cluster_runtimes_snapshot() -> list[dict[str, object]]:
        with lifecycle_lock:
            return [
                dict(cluster_runtimes[instance_id])
                for instance_id in sorted(cluster_runtimes)
            ]

    def process_role_snapshots() -> list[dict[str, object]]:
        snapshots: list[dict[str, object]] = []
        for status in roles.status():  # type: ignore[union-attr]
            payload = dict(status.__dict__)
            instance_id = payload.get("instance_id")
            instance = instances.get(str(instance_id)) if instance_id else None
            if (
                instance is not None
                and instance.last_error is not None
                and payload.get("return_code") == 0
                and payload.get("state") == "stopped"
            ):
                payload["state"] = "failed"
                message = instance.last_error.get("message") or "Runtime validation failed."
                payload["message"] = f"Runtime failed before a coordinated stop: {message}"
            snapshots.append(payload)
        return snapshots

    def reconcile_instance_memory(
        instance,
        runtime: dict[str, object] | None,
    ) -> str | None:
        if not isinstance(runtime, dict):
            return None
        memory = runtime.get("memory")
        if not isinstance(memory, dict):
            return None
        peak_candidates = [
            value
            for value in (
                memory.get("mlx_peak_bytes"),
                memory.get("model_resident_observed_bytes"),
            )
            if isinstance(value, int)
        ]
        active = memory.get("mlx_active_bytes")
        cache = memory.get("mlx_cache_bytes")
        if isinstance(active, int) and isinstance(cache, int):
            peak_candidates.append(active + cache)
        observed_peak = max(peak_candidates) if peak_candidates else None
        footprint = memory.get("process_phys_footprint_bytes")
        observed_footprint = footprint if isinstance(footprint, int) else None
        if observed_peak is None and observed_footprint is None:
            return None
        breakdown = instance.memory_breakdown()
        try:
            reservation = resource_ledger.reconcile(
                instance.instance_id,
                observed_peak_bytes=observed_peak,
                observed_footprint_bytes=observed_footprint,
                safety_margin_ratio=0.1,
                breakdown=breakdown,
                system_available_memory_bytes=system_available_memory_bytes(),
            )
        except ResourceAdmissionError as exc:
            return str(exc)
        instance.memory_reservation_bytes = reservation
        instance.memory_reservation_breakdown = breakdown.with_observation(
            peak_bytes=observed_peak,
            footprint_bytes=observed_footprint,
        )
        return None

    def reconcile_instance_runtime(
        instance,
        runtime: dict[str, object] | None,
        status,
        role: str | None,
    ) -> bool:
        """Apply only identity-valid, monotonic runtime truth to the registry.

        A worker owns its local rank lifecycle. The coordinator owns global
        quorum and therefore reaches READY only through ``instance_quorum``.
        This prevents a rank-local observation from masquerading as cluster
        readiness while still allowing a worker registry to converge.
        """

        if not isinstance(runtime, dict):
            return False
        runtime_version = runtime.get("version")
        if not isinstance(runtime_version, int):
            runtime_version = 0
        previous_version = instance_runtime_versions.get(instance.instance_id, -1)
        if runtime_version < previous_version:
            return False
        expected = {
            "instance_id": instance.instance_id,
            "operation_id": instance.operation_id,
            "world_size": instance.world_size,
            "connection_mode": instance.connection_mode,
        }
        for key, value in expected.items():
            if runtime.get(key) != value:
                instance.health_ready = False
                instance.health_sampled_at = time.time()
                instance.health_issues = [f"Runtime {key} does not match the registry."]
                return False
        if instance.model_revision and runtime.get("model_revision") != instance.model_revision:
            instance.health_ready = False
            instance.health_sampled_at = time.time()
            instance.health_issues = ["Runtime model revision does not match the registry."]
            return False
        if runtime.get("status_stale") is True:
            instance.health_ready = False
            instance.health_sampled_at = time.time()
            instance.health_issues = ["Runtime heartbeat is stale."]
            return False

        instance_runtime_versions[instance.instance_id] = runtime_version

        def merge_runtime_clock() -> None:
            # Registry transitions and runtime telemetry share a monotonic
            # externally visible version fence. An Agent restart reconstructs
            # several registry transitions from the journal; never publish a
            # value lower than the already-running runtime process reported.
            instance.version = max(instance.version, runtime_version)
            runtime_updated_at = runtime.get("updated_at")
            if isinstance(runtime_updated_at, (int, float)):
                instance.updated_at = max(instance.updated_at, float(runtime_updated_at))

        phase = str(runtime.get("phase") or "")
        if phase == "failed":
            if instance.state not in {
                InstanceLifecycle.FAILED,
                InstanceLifecycle.UNLOADING,
                InstanceLifecycle.STOPPED,
            }:
                error = runtime.get("last_error")
                if not isinstance(error, dict):
                    error = {
                        "stage": "runtime",
                        "message": str(runtime.get("message") or "Runtime failed."),
                    }
                try:
                    instance.transition(InstanceLifecycle.FAILED, error=error)
                except RuntimeError:
                    pass
            instance.health_ready = False
            instance.health_sampled_at = time.time()
            instance.health_issues = [str(runtime.get("message") or "Runtime failed.")]
            merge_runtime_clock()
            return True

        process_running = status is not None and getattr(status, "state", None) == "running"
        local_rank_is_authoritative = role == "distributed-openai-rank" or instance.world_size == 1
        if phase == "ready" and process_running and local_rank_is_authoritative:
            try:
                if instance.state == InstanceLifecycle.DISTRIBUTED_INITIALIZING:
                    instance.transition(InstanceLifecycle.LOADING_METADATA)
                if instance.state == InstanceLifecycle.LOADING_METADATA:
                    instance.transition(InstanceLifecycle.MATERIALIZING_WEIGHTS)
                if instance.state == InstanceLifecycle.MATERIALIZING_WEIGHTS:
                    instance.transition(InstanceLifecycle.COMPILING_WARMING)
                if instance.state == InstanceLifecycle.COMPILING_WARMING:
                    evidence = dict(runtime.get("ready_evidence") or {})
                    evidence["scope"] = "local_rank" if instance.world_size > 1 else "single_node"
                    instance.transition(
                        InstanceLifecycle.READY,
                        readiness_evidence=evidence,
                    )
            except RuntimeError:
                merge_runtime_clock()
                return False
            instance.tokenizer_identity = (
                str(runtime["tokenizer_identity"])
                if runtime.get("tokenizer_identity")
                else instance.tokenizer_identity
            )
            instance.health_ready = True
            instance.health_sampled_at = time.time()
            instance.health_issues = []
        merge_runtime_clock()
        return True

    def supervisor_start(
        role: str,
        command: list[str],
        *,
        env: dict[str, str],
        instance_id: str,
        operation_id: str,
    ):
        try:
            return roles.start(
                role,
                command,
                env=env,
                cwd=_distributed_code_root(),
                instance_id=instance_id,
                operation_id=operation_id,
            )
        except TypeError as exc:
            if "instance_id" not in str(exc) and "operation_id" not in str(exc):
                raise
            return roles.start(role, command, env=env, cwd=_distributed_code_root())

    def advance_h3_instance_to_ready(
        instance,
        readiness_evidence: dict[str, object],
    ) -> None:
        """Finish H3 startup without racing concurrent quorum health probes."""

        transitions = {
            InstanceLifecycle.DISTRIBUTED_INITIALIZING: InstanceLifecycle.LOADING_METADATA,
            InstanceLifecycle.LOADING_METADATA: InstanceLifecycle.MATERIALIZING_WEIGHTS,
            InstanceLifecycle.MATERIALIZING_WEIGHTS: InstanceLifecycle.COMPILING_WARMING,
            InstanceLifecycle.COMPILING_WARMING: InstanceLifecycle.READY,
        }
        with lifecycle_lock:
            while instance.state not in {
                InstanceLifecycle.READY,
                InstanceLifecycle.BUSY,
            }:
                next_state = transitions.get(instance.state)
                if next_state is None:
                    raise RuntimeError(
                        f"Cannot complete MiniMax H3 readiness from {instance.state.value}."
                    )
                instance.transition(
                    next_state,
                    readiness_evidence=(
                        readiness_evidence
                        if next_state == InstanceLifecycle.READY
                        else None
                    ),
                )
            # A concurrent quorum request may have completed the transition
            # first with generic evidence. Preserve the richer launch contract.
            instance.readiness_evidence.update(readiness_evidence)

    def supervisor_status(role: str, instance_id: str | None = None):
        try:
            return roles.status(role, instance_id=instance_id)
        except TypeError:
            return roles.status(role)

    def supervisor_stop(role: str, timeout: float, instance_id: str | None = None):
        try:
            return roles.stop(role, timeout=timeout, instance_id=instance_id)
        except TypeError:
            return roles.stop(role, timeout=timeout)

    def supervisor_request_stop(role: str, instance_id: str | None = None):
        request_stop = getattr(roles, "request_stop", None)
        if request_stop is None:
            return supervisor_status(role, instance_id)
        try:
            return request_stop(role, instance_id=instance_id)
        except TypeError:
            return request_stop(role)

    def supervisor_wait(role: str, timeout: float, instance_id: str | None = None):
        wait = getattr(roles, "wait", None)
        if wait is None:
            return supervisor_status(role, instance_id)
        try:
            return wait(role, timeout, instance_id=instance_id)
        except TypeError:
            return wait(role, timeout)

    def clear_lease() -> None:
        nonlocal cluster_runtime
        with lifecycle_lock:
            clear_legacy_lease()
            cluster_runtime = None
            cluster_runtimes.clear()

    def owned_running_coordinator_ports() -> dict[str, int]:
        with lifecycle_lock:
            candidates = [
                (instance_id, instance_roles.get(instance_id), port)
                for instance_id, port in instance_ports.items()
            ]

        owned: dict[str, int] = {}
        for instance_id, role, port in candidates:
            if role not in NATIVE_ADMIN_STOP_ROLES:
                continue
            status = supervisor_status(role, instance_id)
            if (
                getattr(status, "pid", None) is None
                or getattr(status, "return_code", None) is not None
            ):
                continue
            owned[instance_id] = port
        return owned

    def request_local_model_roles_stop() -> list[dict[str, object]]:
        coordinator_requested: set[str | None] = set()
        for instance_id, port in owned_running_coordinator_ports().items():
            try:
                _post_json(f"http://127.0.0.1:{port}/v1/tokenity/stop", {}, 2.0)
                coordinator_requested.add(instance_id)
            except Exception:
                pass

        statuses: list[object] = []
        all_statuses = roles.status()
        targets = [
            status
            for status in all_statuses
            if getattr(status, "role", None) in MODEL_ROLES
        ] if isinstance(all_statuses, list) else []
        if not targets:
            targets = [supervisor_status(role) for role in MODEL_ROLES]
        for target in targets:
            role = target.role
            instance_id = getattr(target, "instance_id", None)
            # The coordinator's admin endpoint asks Uvicorn to exit naturally.
            # Sending SIGTERM immediately would turn a clean exit into -15 and
            # can tear down JACCL while the worker is still in a collective.
            if role in NATIVE_ADMIN_STOP_ROLES and instance_id in coordinator_requested:
                statuses.append(supervisor_status(role, instance_id))
            else:
                statuses.append(supervisor_request_stop(role, instance_id))
        return [status.__dict__ for status in statuses]

    def stop_local_model_roles(
        timeout: float,
        *,
        request_first: bool = True,
    ) -> list[dict[str, object]]:
        if request_first:
            request_local_model_roles_stop()
        deadline = time.monotonic() + timeout
        statuses: list[object] = []
        all_statuses = roles.status()
        targets = [
            status
            for status in all_statuses
            if getattr(status, "role", None) in MODEL_ROLES
        ] if isinstance(all_statuses, list) else []
        if not targets:
            targets = [supervisor_status(role) for role in MODEL_ROLES]
        for target in targets:
            role = target.role
            instance_id = getattr(target, "instance_id", None)
            remaining = max(0.0, deadline - time.monotonic())
            status = supervisor_wait(role, remaining, instance_id)
            if getattr(status, "pid", None) is not None:
                status = supervisor_stop(role, max(0.5, min(2.0, remaining)), instance_id)
            statuses.append(status)
        return [status.__dict__ for status in statuses]

    def worker_urls(*, clear: bool = False) -> list[str]:
        with lifecycle_lock:
            urls = sorted(
                {
                    agent_url
                    for workers in instance_workers.values()
                    for agent_url in workers
                }
            )
            if clear:
                instance_workers.clear()
        return urls

    def require_managed_instance_capabilities(
        nodes: list[ClusterNode],
        *,
        additional: frozenset[str] = frozenset(),
    ) -> None:
        required = MANAGED_INSTANCE_CAPABILITIES.union(additional)
        issues: list[dict[str, object]] = []
        local_missing = sorted(
            required.difference(NODE_AGENT_CAPABILITIES)
        )
        if local_missing:
            issues.append(
                {
                    "node": nodes[0].id,
                    "agent_url": _agent_url(nodes[0]),
                    "missing_capabilities": local_missing,
                }
            )
        for node in nodes[1:]:
            agent_url = _agent_url(node, remote=True)
            try:
                info = get_json(f"{agent_url}/v1/node/info", 3.0)
            except Exception as exc:
                issues.append(
                    {
                        "node": node.id,
                        "agent_url": agent_url,
                        "error": str(exc),
                        "missing_capabilities": sorted(required),
                    }
                )
                continue
            contract = info.get("agent_contract") if isinstance(info, dict) else None
            capabilities = contract.get("capabilities") if isinstance(contract, dict) else None
            advertised = (
                {str(capability) for capability in capabilities}
                if isinstance(capabilities, list)
                else set()
            )
            missing = sorted(required.difference(advertised))
            if missing:
                issues.append(
                    {
                        "node": node.id,
                        "agent_url": agent_url,
                        "missing_capabilities": missing,
                    }
                )
        if issues:
            raise HTTPException(
                status_code=412,
                detail={
                    "stage": "agent_capabilities",
                    "required_capabilities": sorted(required),
                    "issues": issues,
                },
            )

    def request_known_workers_stop() -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for agent_url in worker_urls():
            try:
                result = post_json(
                    f"{agent_url}/v1/node/request-stop-all",
                    {},
                    3.0,
                )
                results.append({"agent_url": agent_url, "result": result})
            except Exception as exc:
                results.append({"agent_url": agent_url, "error": str(exc)})
        return results

    def stop_known_workers(timeout: float) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for agent_url in worker_urls(clear=True):
            try:
                result = post_json(
                    f"{agent_url}/v1/node/stop-all",
                    {"timeout": timeout},
                    timeout + 3,
                )
                results.append({"agent_url": agent_url, "result": result})
            except Exception as exc:
                results.append({"agent_url": agent_url, "error": str(exc)})
        return results

    def stop_everything(timeout: float, *, notify_workers: bool = True) -> dict[str, object]:
        cancel_launch()
        statuses: list[dict[str, object]] = []

        # Phase 1: every rank observes the stop request before any process is
        # forcibly reaped.  This keeps JACCL/MLX collectives symmetric.
        requested_workers: list[dict[str, object]] = []

        def request_workers() -> None:
            if notify_workers:
                requested_workers.extend(request_known_workers_stop())

        worker_request_thread = threading.Thread(target=request_workers)
        local_request_thread = threading.Thread(target=request_local_model_roles_stop)
        worker_request_thread.start()
        local_request_thread.start()
        worker_request_thread.join(4)
        local_request_thread.join(4)

        def stop_local() -> None:
            statuses.extend(stop_local_model_roles(timeout, request_first=False))

        local_thread = threading.Thread(target=stop_local)
        local_thread.start()
        workers = stop_known_workers(timeout) if notify_workers else []
        local_thread.join(timeout + 3)
        clear_lease()
        for snapshot in instances.snapshots():
            instance_id = str(snapshot["instance_id"])
            resource_ledger.release(instance_id)
            instance = instances.get(instance_id)
            if instance is not None and instance.state != InstanceLifecycle.STOPPED:
                try:
                    if instance.state != InstanceLifecycle.UNLOADING:
                        instance.transition(InstanceLifecycle.UNLOADING)
                    instance.transition(InstanceLifecycle.STOPPED)
                except (InstanceConflict, RuntimeError):
                    pass
        return {
            "statuses": statuses,
            "requested_workers": requested_workers,
            "workers": workers,
        }

    def lease_watchdog() -> None:
        next_health_refresh = 0.0
        while not watchdog_stop.wait(1.0):
            with lifecycle_lock:
                expired = lease_deadline is not None and time.monotonic() >= lease_deadline
            if expired:
                stop_everything(3.0, notify_workers=False)
            now = time.time()
            monotonic_now = time.monotonic()
            for snapshot in instances.snapshots():
                deadline = snapshot.get("deadline")
                if not isinstance(deadline, (int, float)) or deadline > now:
                    continue
                if snapshot.get("state") in {"stopped", "unloading", "failed"}:
                    continue
                try:
                    stop_instance(str(snapshot["instance_id"]), StopAllRequest(timeout=3.0))
                except Exception:
                    continue
            if monotonic_now >= next_health_refresh:
                next_health_refresh = monotonic_now + 5.0
                for snapshot in instances.snapshots():
                    if snapshot.get("state") in {"stopped", "failed", "orphaned", "unloading"}:
                        continue
                    try:
                        instance_id = str(snapshot["instance_id"])
                        role = instance_roles.get(instance_id)
                        if role in NATIVE_ADMIN_STOP_ROLES or int(snapshot.get("world_size") or 1) == 1:
                            instance_quorum(instance_id)
                        else:
                            get_instance(instance_id)
                    except Exception:
                        instance = instances.get(str(snapshot["instance_id"]))
                        if instance is not None:
                            instance.health_ready = False
                            instance.health_sampled_at = time.time()
                            instance.health_issues = ["Health refresh failed."]

    async def event_loop_heartbeat_loop() -> None:
        nonlocal event_loop_heartbeat
        while True:
            event_loop_heartbeat = time.monotonic()
            await asyncio.sleep(1.0)

    @app.on_event("startup")
    async def startup_cleanup() -> None:
        nonlocal event_loop_heartbeat_task
        adopted_pids = recover_persisted_instances()
        cleanup = getattr(roles, "cleanup_orphaned_model_processes", None)
        scans_unjournaled_processes = (
            os.environ.get("TOKENITY_DISABLE_UNJOURNALED_ORPHAN_SCAN") != "1"
        )
        if cleanup is not None and scans_unjournaled_processes:
            reconciled = cleanup()
            if isinstance(reconciled, list):
                startup_orphans.extend(
                    item
                    for item in reconciled
                    if isinstance(item, dict)
                    and (
                        not isinstance(item.get("pid"), int)
                        or item.get("pid") not in adopted_pids
                    )
                )
        threading.Thread(target=lease_watchdog, daemon=True).start()
        event_loop_heartbeat_task = asyncio.create_task(event_loop_heartbeat_loop())
        for pid in sorted(startup_orphan_records):
            await asyncio.to_thread(cleanup_startup_orphan, pid)

    @app.on_event("shutdown")
    async def shutdown_cleanup() -> None:
        watchdog_stop.set()
        if event_loop_heartbeat_task is not None:
            event_loop_heartbeat_task.cancel()
        stop_everything(3.0, notify_workers=False)

    @app.get("/health")
    @app.get("/v1/node/health")
    async def node_health():
        """Fast Agent-core liveness contract for the external watchdog.

        This endpoint intentionally performs no model discovery, distributed
        probe, log read, or remote-node request. Individual model failures do
        not make the Agent unavailable as a control service.
        """

        await asyncio.sleep(0)
        now_monotonic = time.monotonic()
        heartbeat_age = max(0.0, now_monotonic - event_loop_heartbeat)
        state_root = Path(tempfile.gettempdir())
        state_store_readable = (
            instance_state_store.readable
            if instance_state_store is not None
            else os.access(state_root, os.R_OK | os.W_OK)
        )
        active_instances = sum(
            1
            for snapshot in instances.snapshots()
            if snapshot.get("state") not in {"stopped", "failed", "orphaned"}
        )
        if heartbeat_age >= 5.0 or not state_store_readable:
            status = "unhealthy"
            status_code = 503
        elif last_fatal_internal_error is not None or any(
            orphan.get("cleanup_required") is True
            or orphan.get("state") == "orphaned"
            for orphan in startup_orphans
        ):
            status = "degraded"
            status_code = 200
        else:
            status = "healthy"
            status_code = 200
        payload = {
            "status": status,
            "service": "tokenity-node-agent",
            "agent_revision": _tokenity_code_revision(),
            "process_start_time": process_started_at,
            "uptime_seconds": max(0.0, time.time() - process_started_at),
            "event_loop_heartbeat_age_seconds": heartbeat_age,
            "active_instance_count": active_instances,
            "watchdog_compatible_monotonic": now_monotonic,
            "state_store_readable": state_store_readable,
            "last_fatal_internal_error": last_fatal_internal_error,
        }
        if status_code != 200:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=status_code, content=payload)
        return payload

    @app.get("/v1/node/info")
    def node_info() -> dict[str, object]:
        rdma: RDMAProbeResult = rdma_probe_fn()
        memory = _memory_stats()
        return {
            "node_id": _node_id(),
            "machine_id": _machine_id(),
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "ips": _local_ipv4s(),
            "architecture": platform.machine(),
            "macos_version": platform.mac_ver()[0] or platform.platform(),
            "python_path": sys.executable,
            "venv": sys.prefix if sys.prefix != getattr(sys, "base_prefix", sys.prefix) else None,
            "mlx_version": _package_version("mlx"),
            "mlx_lm_version": _package_version("mlx-lm"),
            "tokenity_version": __version__,
            "tokenity_code_revision": _tokenity_code_revision(),
            "runtime": _runtime_identity(),
            "agent_contract": {
                "version": NODE_AGENT_CONTRACT_VERSION,
                "capabilities": list(NODE_AGENT_CAPABILITIES),
            },
            "available_ports": {"agent": 9100, "openai": 8000, "mlx_starting_port": 29500},
            "process_roles": process_role_snapshots(),
            "cluster_runtime": cluster_runtime_snapshot(),
            "cluster_runtimes": cluster_runtimes_snapshot(),
            "instances": instances.snapshots(),
            "resource_ledger": resource_ledger_snapshot(memory),
            "orphaned_processes": list(startup_orphans),
            "memory": memory,
            "disk": _disk_stats(DEFAULT_MODEL_ROOT),
            "rdma": rdma.to_dict(),
        }

    @app.get("/v1/node/models")
    def node_models(root: str = Query(DEFAULT_MODEL_ROOT)) -> dict[str, object]:
        return {"root": root, "models": scan_models(Path(root))}

    @app.get("/v1/node/status")
    def node_status() -> dict[str, object]:
        memory = _memory_stats()
        return {
            "roles": process_role_snapshots(),
            "cluster_runtime": cluster_runtime_snapshot(),
            "cluster_runtimes": cluster_runtimes_snapshot(),
            "instances": instances.snapshots(),
            "resource_ledger": resource_ledger_snapshot(memory),
            "orphaned_processes": list(startup_orphans),
            "memory": memory,
        }

    @app.post("/v1/node/orphans/{pid}/cleanup")
    def cleanup_orphan(pid: int) -> dict[str, object]:
        result = cleanup_startup_orphan(pid)
        if result.get("cleanup_safe") is not True and result.get("state") != "cleaned":
            raise HTTPException(
                status_code=409,
                detail=str(result.get("cleanup_result") or "Orphan identity is not safe to clean."),
            )
        return result

    @app.get("/v1/node/instances")
    def list_instances() -> dict[str, object]:
        return {
            "data": instances.snapshots(),
            "resource_ledger": resource_ledger_snapshot(),
        }

    @app.get("/v1/node/instances/{instance_id}")
    def get_instance(instance_id: str) -> dict[str, object]:
        instance = instances.get(instance_id)
        if instance is None:
            raise HTTPException(status_code=404, detail=f"Unknown model instance: {instance_id}")
        role = instance_roles.get(instance_id)
        status = supervisor_status(role, instance_id) if role else None
        runtime = _read_runtime_status(instance_status_paths.get(instance_id))
        runtime_update_accepted = reconcile_instance_runtime(instance, runtime, status, role)
        resource_issue = reconcile_instance_memory(instance, runtime)
        if status is not None and getattr(status, "state", None) in {"failed", "stopped"}:
            if instance.state not in {InstanceLifecycle.FAILED, InstanceLifecycle.STOPPED, InstanceLifecycle.UNLOADING}:
                try:
                    instance.transition(
                        InstanceLifecycle.FAILED,
                        error={
                            "stage": "runtime_process",
                            "message": getattr(status, "message", None) or "Runtime process exited.",
                            "log_path": getattr(status, "log_path", None),
                        },
                    )
                except RuntimeError:
                    pass
        return {
            "instance": instance.to_dict(),
            "process": status.__dict__ if status is not None else None,
            "runtime": runtime,
            "runtime_update_accepted": runtime_update_accepted,
            "resource_issue": resource_issue,
        }

    @app.get("/v1/node/instances/{instance_id}/quorum")
    def instance_quorum(instance_id: str) -> dict[str, object]:
        instance = instances.get(instance_id)
        if instance is None:
            raise HTTPException(status_code=404, detail=f"Unknown model instance: {instance_id}")
        role = instance_roles.get(instance_id)
        if role in {"minimax-h3-video", "minimax-h3-video-rank"}:
            local_process = supervisor_status(role, instance_id)
            ranks: list[dict[str, object]] = [
                {
                    "node": instance.coordinator,
                    "rank": instance.rank_mapping.get(instance.coordinator, 0),
                    "process": local_process.__dict__,
                }
            ]
            for agent_url in instance_workers.get(instance_id, []):
                try:
                    remote = get_json(
                        f"{agent_url}/v1/node/instances/{instance_id}",
                        3.0,
                    )
                    ranks.append({"node": agent_url, **remote})
                except Exception as exc:
                    ranks.append({"node": agent_url, "error": str(exc)})

            issues: list[str] = []
            for item in ranks:
                process = item.get("process")
                if not isinstance(process, dict) or process.get("state") != "running":
                    issues.append(f"{item.get('node')}: native H3 process is not running.")
            if role == "minimax-h3-video":
                pid = getattr(local_process, "pid", None)
                if not isinstance(pid, int) or not wait_for_h3(
                    pid,
                    instance.http_port,
                    1.0,
                ):
                    issues.append("Rank 0 native H3 /health probe failed.")
            if len(ranks) != instance.world_size:
                issues.append(
                    f"Rank quorum mismatch: expected {instance.world_size}, got {len(ranks)}."
                )
            ready = not issues
            instance.health_ready = ready
            instance.health_sampled_at = time.time()
            instance.health_issues = list(issues)
            if ready and instance.state not in {
                InstanceLifecycle.READY,
                InstanceLifecycle.BUSY,
            }:
                try:
                    advance_h3_instance_to_ready(
                        instance,
                        {
                            "rank_quorum": f"{len(ranks)}/{instance.world_size}",
                            "model_revision": instance.model_revision,
                            "connection_mode": instance.connection_mode,
                            "all_ranks_healthy": True,
                            "native_video_health": True,
                            "distributed_protocol": (
                                H3_DISTRIBUTED_PROTOCOL_VERSION
                                if instance.world_size > 1
                                else None
                            ),
                        },
                    )
                except RuntimeError as exc:
                    issues.append(str(exc))
                    ready = False
            return {
                "instance_id": instance_id,
                "ready": ready,
                "issues": issues,
                "rank_quorum": f"{len(ranks)}/{instance.world_size}",
                "ranks": ranks,
                "instance": instance.to_dict(),
            }
        local = get_instance(instance_id)
        ranks: list[dict[str, object]] = [
            {"node": instance.coordinator, **local}
        ]
        for agent_url in instance_workers.get(instance_id, []):
            try:
                remote = get_json(f"{agent_url}/v1/node/instances/{instance_id}", 3.0)
                ranks.append({"node": agent_url, **remote})
            except Exception as exc:
                ranks.append({"node": agent_url, "error": str(exc)})

        issues: list[str] = []
        seen_ranks: set[int] = set()
        tokenizer_identities: set[str] = set()
        rank0_probe_succeeded = False
        observed_runtime_bytes = 0
        observed_rank_count = 0
        stale_memory_ranks = 0
        for item in ranks:
            runtime = item.get("runtime")
            process = item.get("process")
            node = str(item.get("node"))
            resource_issue = item.get("resource_issue")
            if isinstance(resource_issue, str) and resource_issue:
                issues.append(f"{node}: memory reservation reconciliation failed: {resource_issue}")
            if not isinstance(runtime, dict):
                issues.append(f"{node}: runtime readiness is unavailable.")
                continue
            if runtime.get("instance_id") != instance_id:
                issues.append(f"{node}: instance identity mismatch.")
            if runtime.get("operation_id") != instance.operation_id:
                issues.append(f"{node}: operation identity mismatch.")
            if runtime.get("phase") not in {"ready", "prefill_pending", "generating"}:
                issues.append(f"{node}: runtime phase is {runtime.get('phase') or 'unknown'}.")
            if runtime.get("status_stale") is True:
                issues.append(f"{node}: runtime heartbeat is stale.")
            memory = runtime.get("memory")
            if isinstance(memory, dict):
                if memory.get("stale") is True:
                    stale_memory_ranks += 1
                observed = memory.get("model_resident_observed_bytes")
                if not isinstance(observed, int):
                    observed = memory.get("mlx_active_bytes")
                if isinstance(observed, int):
                    observed_runtime_bytes += observed
                    observed_rank_count += 1
            if runtime.get("world_size") != instance.world_size:
                issues.append(f"{node}: world size mismatch.")
            if runtime.get("connection_mode") != instance.connection_mode:
                issues.append(f"{node}: connection mode mismatch.")
            if instance.model_revision and runtime.get("model_revision") != instance.model_revision:
                issues.append(f"{node}: model revision mismatch.")
            tokenizer_identity = runtime.get("tokenizer_identity")
            if isinstance(tokenizer_identity, str) and tokenizer_identity:
                tokenizer_identities.add(tokenizer_identity)
            else:
                issues.append(f"{node}: tokenizer identity is unavailable.")
            evidence = runtime.get("ready_evidence")
            if not isinstance(evidence, dict):
                issues.append(f"{node}: readiness evidence is unavailable.")
            else:
                for field in ("weights_materialized", "tokenizer_ready", "generation_engine_ready"):
                    if evidence.get(field) is not True:
                        issues.append(f"{node}: readiness evidence {field} is not true.")
            rank = runtime.get("rank")
            if isinstance(rank, int):
                if rank in seen_ranks:
                    issues.append(f"{node}: duplicate rank {rank}.")
                seen_ranks.add(rank)
                if rank == 0 and isinstance(evidence, dict):
                    rank0_probe_succeeded = (
                        evidence.get("one_token_probe") is True
                        and evidence.get("warmup_cache_isolated") is True
                    )
            if not isinstance(process, dict) or process.get("state") != "running":
                issues.append(f"{node}: runtime process is not running.")

        if seen_ranks != set(range(instance.world_size)):
            issues.append(
                f"Rank quorum mismatch: expected {list(range(instance.world_size))}, got {sorted(seen_ranks)}."
            )
        if len(tokenizer_identities) > 1:
            issues.append(f"Tokenizer identity mismatch: {sorted(tokenizer_identities)}.")
        if not rank0_probe_succeeded:
            issues.append("Rank 0 one-token isolated readiness probe did not succeed.")
        ready = not issues and len(ranks) == instance.world_size
        instance.health_ready = ready
        instance.health_sampled_at = time.time()
        instance.health_issues = list(issues)
        instance.actual_memory_bytes = (
            observed_runtime_bytes if observed_rank_count == instance.world_size else None
        )
        if ready and instance.state != InstanceLifecycle.READY:
            try:
                if instance.state == InstanceLifecycle.LOADING_METADATA:
                    instance.transition(InstanceLifecycle.MATERIALIZING_WEIGHTS)
                if instance.state == InstanceLifecycle.MATERIALIZING_WEIGHTS:
                    instance.transition(InstanceLifecycle.COMPILING_WARMING)
                if instance.state == InstanceLifecycle.COMPILING_WARMING:
                    instance.transition(
                        InstanceLifecycle.READY,
                        readiness_evidence={
                            "rank_quorum": f"{len(seen_ranks)}/{instance.world_size}",
                            "model_revision": instance.model_revision,
                            "connection_mode": instance.connection_mode,
                            "all_ranks_healthy": True,
                            "one_token_probe": True,
                            "warmup_cache_isolated": True,
                            "tokenizer_identity": next(iter(tokenizer_identities), None),
                        },
                    )
            except RuntimeError as exc:
                issues.append(str(exc))
                ready = False
        return {
            "instance_id": instance_id,
            "ready": ready,
            "issues": issues,
            "rank_quorum": f"{len(seen_ranks)}/{instance.world_size}",
            "ranks": ranks,
            "instance": instance.to_dict(),
            "memory_aggregate": {
                "model_weights_estimated_bytes": _directory_size(Path(instance.resolved_path)),
                "model_resident_observed_bytes": instance.actual_memory_bytes,
                "observed_rank_count": observed_rank_count,
                "expected_rank_count": instance.world_size,
                "stale_rank_count": stale_memory_ranks,
                "sampled_at": time.time(),
            },
        }

    def refresh_gateway_candidates(model_or_instance_id: str) -> None:
        for snapshot in instances.snapshots():
            if (
                model_or_instance_id != TOKENITY_AUTO_MODEL_ID
                and model_or_instance_id
                not in {
                    str(snapshot.get("instance_id") or ""),
                    str(snapshot.get("requested_model_id") or ""),
                }
            ):
                continue
            try:
                instance_quorum(str(snapshot["instance_id"]))
            except HTTPException:
                continue

    def routing_inputs(
        allowed_instance_ids: frozenset[str] | None = None,
    ) -> tuple[
        list[ModelCapabilityProfile],
        list[ModelRuntimeState],
    ]:
        grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
        for snapshot in instances.snapshots():
            if instance_roles.get(str(snapshot["instance_id"])) in VIDEO_MODEL_ROLES:
                continue
            if (
                allowed_instance_ids is not None
                and str(snapshot["instance_id"]) not in allowed_instance_ids
            ):
                continue
            model_id = str(snapshot["requested_model_id"])
            revision = _routing_revision(snapshot.get("model_revision"))
            grouped.setdefault((model_id, revision), []).append(snapshot)

        profiles: list[ModelCapabilityProfile] = []
        runtimes: list[ModelRuntimeState] = []
        now = time.time()
        for (model_id, revision), snapshots in sorted(grouped.items()):
            profile = capability_registry.get(model_id, revision)
            if profile is None:
                profile = _infer_gateway_model_profile(
                    model_id,
                    revision,
                    str(snapshots[0].get("resolved_path") or ""),
                )
                capability_registry.register(profile)
            profiles.append(profile)
            routable = [
                snapshot
                for snapshot in snapshots
                if snapshot.get("state") in {"ready", "busy"}
                and snapshot.get("health_ready") is not False
            ]
            load_source = routable or snapshots
            best = min(
                load_source,
                key=lambda item: (
                    int(item.get("active_request_count") or 0)
                    + int(item.get("queued_request_count") or 0),
                    str(item["instance_id"]),
                ),
            )
            sampled_at = best.get("health_sampled_at") or best.get("heartbeat_at")
            heartbeat_age = (
                max(0.0, now - float(sampled_at))
                if isinstance(sampled_at, (int, float))
                else 0.0
            )
            # GenerationSlotScheduler mirrors each waiter into the registry, so
            # queued_request_count is the single authoritative routing signal.
            queue_depth = int(best.get("queued_request_count") or 0)
            runtimes.append(
                ModelRuntimeState(
                    model_id=model_id,
                    revision=revision,
                    ready=bool(routable),
                    quorum_ready=bool(routable),
                    state=str(best.get("state") or "stopped"),
                    queue_depth=queue_depth,
                    active_request_count=int(best.get("active_request_count") or 0),
                    heartbeat_age_seconds=heartbeat_age,
                    actual_memory_bytes=(
                        int(best["actual_memory_bytes"])
                        if isinstance(best.get("actual_memory_bytes"), int)
                        else None
                    ),
                    recent_failure_rate=1.0 if best.get("last_error") else 0.0,
                    predicted_queue_wait_ms=float(queue_depth * 1_000),
                )
            )
        return profiles, runtimes

    def parse_route_request(payload: object) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            raise HTTPException(status_code=400, detail="A model alias or instance_id is required.")
        policy_name = payload.get("tokenity_route_policy", "balanced")
        if not isinstance(policy_name, str) or policy_name not in {"fast", "balanced", "quality"}:
            raise HTTPException(
                status_code=422,
                detail="tokenity_route_policy must be fast, balanced, or quality.",
            )
        session_id = payload.get("tokenity_session_id")
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id.strip()
        ):
            raise HTTPException(status_code=422, detail="tokenity_session_id must be a non-empty string.")
        normalized_session_id = session_id.strip() if isinstance(session_id, str) else None
        lock_model = payload.get("tokenity_lock_model", False)
        if not isinstance(lock_model, (bool, str)):
            raise HTTPException(
                status_code=422,
                detail="tokenity_lock_model must be a boolean or model ID.",
            )
        constraints = payload.get("tokenity_constraints") or {}
        if not isinstance(constraints, dict):
            raise HTTPException(status_code=422, detail="tokenity_constraints must be an object.")
        allowed_instance_ids_raw = constraints.get("allowed_instance_ids")
        if allowed_instance_ids_raw is not None and (
            not isinstance(allowed_instance_ids_raw, list)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in allowed_instance_ids_raw
            )
        ):
            raise HTTPException(
                status_code=422,
                detail="tokenity_constraints.allowed_instance_ids must be a list of non-empty strings.",
            )
        allowed_instance_ids = (
            frozenset(value.strip() for value in allowed_instance_ids_raw)
            if isinstance(allowed_instance_ids_raw, list)
            else None
        )
        with lifecycle_lock:
            sticky_route = (
                session_routes.get(normalized_session_id)
                if normalized_session_id is not None
                else None
            )
        sticky_model_id = sticky_route[0] if sticky_route is not None else None
        sticky_revision = sticky_route[1] if sticky_route is not None else None
        locked_model_id = (
            lock_model.strip()
            if isinstance(lock_model, str) and lock_model.strip()
            else sticky_model_id if lock_model is True else None
        )
        locked_revision = sticky_revision if lock_model is True else None
        context, features = _derive_gateway_route_context(
            payload,
            constraints,
            session_id=normalized_session_id,
            sticky_model_id=sticky_model_id,
            sticky_revision=sticky_revision,
            locked_model_id=locked_model_id,
            locked_revision=locked_revision,
        )
        return {
            "requested_model": model.strip(),
            "is_auto": model.strip() == TOKENITY_AUTO_MODEL_ID,
            "policy": RoutePolicy.from_name(policy_name),
            "policy_name": policy_name,
            "session_id": normalized_session_id,
            "lock_model": lock_model,
            "constraints": constraints,
            "allowed_instance_ids": allowed_instance_ids,
            "context": context,
            "features": features,
        }

    def excluded_instance_ids_for_route(
        parsed: dict[str, object],
    ) -> frozenset[str]:
        allowed = parsed.get("allowed_instance_ids")
        if parsed["is_auto"] is not True or not isinstance(allowed, frozenset):
            return frozenset()
        return frozenset(
            str(snapshot["instance_id"])
            for snapshot in instances.snapshots()
            if str(snapshot["instance_id"]) not in allowed
        )

    def decide_route(parsed: dict[str, object]) -> tuple[
        RouteDecision,
        object,
        ModelCapabilityProfile,
    ]:
        requested_model = str(parsed["requested_model"])
        refresh_gateway_candidates(requested_model)
        excluded_instance_ids = excluded_instance_ids_for_route(parsed)
        parsed["excluded_instance_ids"] = excluded_instance_ids
        allowed_instance_ids = parsed.get("allowed_instance_ids")
        profiles, runtimes = routing_inputs(
            allowed_instance_ids
            if parsed["is_auto"] is True
            and isinstance(allowed_instance_ids, frozenset)
            else None
        )
        profile_by_key = {
            (profile.model_id, profile.revision): profile for profile in profiles
        }
        runtime_by_key = {
            (runtime.model_id, runtime.revision): runtime for runtime in runtimes
        }
        if parsed["is_auto"] is True:
            context = parsed["context"]
            assert isinstance(context, RouteContext)
            constraints = parsed["constraints"]
            assert isinstance(constraints, dict)
            configured_default = constraints.get("default_model_id")
            default_model_id = (
                configured_default
                if isinstance(configured_default, str) and configured_default
                else _default_gateway_model_id(profiles)
            )
            context = RouteContext(
                **{
                    **context.__dict__,
                    "default_model_id": default_model_id,
                }
            )
            policy = parsed["policy"]
            assert isinstance(policy, RoutePolicy)
            decision = auto_router.decide(profiles, runtimes, context, policy)
            if decision.model_id is None or decision.revision is None:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "stage": "route_decision",
                        **decision.to_dict(),
                    },
                )
            default_candidates = [
                candidate
                for candidate in decision.candidates
                if candidate.eligible and candidate.model_id == default_model_id
            ]
            default_candidates.sort(
                key=lambda candidate: (
                    -(candidate.score if candidate.score is not None else float("-inf")),
                    candidate.revision,
                )
            )
            parsed["effective_default_model_id"] = default_model_id
            parsed["effective_default_revision"] = (
                default_candidates[0].revision if default_candidates else None
            )
            selected = instance_router.resolve(
                decision.model_id,
                revision=decision.revision,
                exclude_instance_ids=excluded_instance_ids,
            )
            profile = profile_by_key[(decision.model_id, decision.revision)]
            actual_reasons = auto_router.hard_constraint_reasons(
                profile,
                runtime_by_key.get((decision.model_id, decision.revision)),
                context,
            )
            if actual_reasons:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "stage": "selected_instance_constraints",
                        "model_id": decision.model_id,
                        "revision": decision.revision,
                        "instance_id": selected.instance_id,
                        "reasons": list(actual_reasons),
                    },
                )
            return decision, selected, profile

        video_target = next(
            (
                snapshot
                for snapshot in instances.snapshots()
                if requested_model
                in {
                    str(snapshot["instance_id"]),
                    str(snapshot["requested_model_id"]),
                }
                and instance_roles.get(str(snapshot["instance_id"])) in VIDEO_MODEL_ROLES
            ),
            None,
        )
        if video_target is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "MiniMax H3 is a video runtime. "
                    "Use /v1/video/generations instead of the chat gateway."
                ),
            )
        started = time.perf_counter()
        parsed["effective_default_model_id"] = None
        parsed["effective_default_revision"] = None
        selected = instance_router.resolve(requested_model)
        revision = _routing_revision(selected.model_revision)
        profile = profile_by_key.get((selected.requested_model_id, revision))
        if profile is None:
            profile = _infer_gateway_model_profile(
                selected.requested_model_id,
                revision,
                selected.resolved_path,
            )
            capability_registry.register(profile)
        decision = RouteDecision(
            model_id=selected.requested_model_id,
            revision=revision,
            category="explicit",
            reason=RouteReason.EXPLICIT_MODEL,
            confidence=1.0,
            routing_latency_ms=(time.perf_counter() - started) * 1_000,
            candidates=(),
            hard_constraints_satisfied=True,
        )
        return decision, selected, profile

    def decision_payload(
        parsed: dict[str, object],
        decision: RouteDecision,
        selected,
    ) -> dict[str, object]:
        return {
            **decision.to_dict(),
            "selected_instance_id": selected.instance_id,
            "routed_model": selected.requested_model_id,
            "model_revision": selected.model_revision,
            "policy": parsed["policy_name"],
            "session_sticky": decision.reason == RouteReason.SESSION_STICKY,
            "derived_features": parsed["features"],
        }

    def fallback_targets(
        parsed: dict[str, object],
        decision: RouteDecision,
    ) -> list[tuple[str, str, str]]:
        if decision.model_id is None or decision.revision is None:
            return []
        result = [(decision.model_id, decision.revision, "primary")]
        if parsed["is_auto"] is not True:
            return result
        profiles, _ = routing_inputs()
        profiles_by_key = {
            (profile.model_id, profile.revision): profile for profile in profiles
        }
        eligible = sorted(
            (
                candidate
                for candidate in decision.candidates
                if candidate.eligible and candidate.model_id != decision.model_id
            ),
            key=lambda candidate: (
                -(candidate.score if candidate.score is not None else float("-inf")),
                candidate.model_id,
                candidate.revision,
            ),
        )
        for candidate in eligible:
            profile = profiles_by_key.get((candidate.model_id, candidate.revision))
            if profile is None:
                continue
            if decision.category in profile.task_tags:
                target = (
                    candidate.model_id,
                    candidate.revision,
                    "same_capability",
                )
                if target[:2] not in [item[:2] for item in result]:
                    result.append(target)
        default_model_id = parsed.get("effective_default_model_id")
        default_revision = parsed.get("effective_default_revision")
        if (
            isinstance(default_model_id, str)
            and default_model_id
            and isinstance(default_revision, str)
            and (default_model_id, default_revision)
            not in [item[:2] for item in result]
        ):
            result.append((default_model_id, default_revision, "default"))
        return result

    @app.get("/v1/gateway/routes")
    def gateway_routes() -> dict[str, object]:
        routes = []
        scheduler_snapshot = generation_scheduler.snapshot()
        queued_by_instance = scheduler_snapshot["queued_by_instance"]
        for snapshot in instances.snapshots():
            if instance_roles.get(str(snapshot["instance_id"])) in VIDEO_MODEL_ROLES:
                continue
            if (
                snapshot.get("state") not in {"ready", "busy"}
                or snapshot.get("health_ready") is False
            ):
                continue
            revision = _routing_revision(snapshot.get("model_revision"))
            model_id = str(snapshot["requested_model_id"])
            profile = capability_registry.get(model_id, revision)
            if profile is None:
                profile = _infer_gateway_model_profile(
                    model_id,
                    revision,
                    str(snapshot.get("resolved_path") or ""),
                )
                capability_registry.register(profile)
            routes.append(
                {
                    "model": model_id,
                    "instance_id": snapshot["instance_id"],
                    "model_revision": revision,
                    "execution_mode": snapshot["execution_mode"],
                    "state": snapshot["state"],
                    "active_request_count": snapshot["active_request_count"],
                    "queue_depth": queued_by_instance.get(snapshot["instance_id"], 0),
                    "api_base_url": f"http://127.0.0.1:{snapshot['http_port']}/v1",
                    "capabilities": {
                        "tools": profile.supports_tools,
                        "json": profile.supports_json,
                        "thinking": profile.supports_thinking,
                        "modalities": sorted(profile.modalities),
                        "task_tags": sorted(profile.task_tags),
                    },
                    "warm_ttft_p50_ms": profile.warm_ttft_p50_ms,
                    "warm_ttft_p95_ms": profile.warm_ttft_p95_ms,
                }
            )
        return {"data": routes}

    @app.get("/v1/models")
    def gateway_models() -> dict[str, object]:
        now = int(time.time())
        ready_models: dict[str, dict[str, object]] = {}
        for snapshot in instances.snapshots():
            if instance_roles.get(str(snapshot["instance_id"])) in VIDEO_MODEL_ROLES:
                continue
            if (
                snapshot.get("state") not in {"ready", "busy"}
                or snapshot.get("health_ready") is False
            ):
                continue
            model_id = str(snapshot["requested_model_id"])
            item = ready_models.setdefault(
                model_id,
                {
                    "id": model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": "tokenity",
                    "tokenity_instance_id": snapshot["instance_id"],
                    "tokenity_instance_ids": [],
                    "tokenity_execution_mode": snapshot["execution_mode"],
                },
            )
            instance_ids = item["tokenity_instance_ids"]
            if isinstance(instance_ids, list):
                instance_ids.append(snapshot["instance_id"])
        data = [
            {
                "id": "tokenity-auto",
                "object": "model",
                "created": now,
                "owned_by": "tokenity",
                "tokenity_virtual": True,
            },
            *[ready_models[model_id] for model_id in sorted(ready_models)],
        ]
        return {"object": "list", "data": data}

    @app.post("/v1/router/decision")
    async def gateway_router_decision(request: FastAPIRequest) -> dict[str, object]:
        try:
            payload = json.loads(await request.body())
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc
        parsed = parse_route_request(payload)
        try:
            decision, selected, _ = decide_route(parsed)
        except InstanceConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return decision_payload(parsed, decision, selected)

    @app.post("/v1/chat/completions")
    async def gateway_chat_completions(request: FastAPIRequest) -> Response:
        try:
            payload = json.loads(await request.body())
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc
        parsed = parse_route_request(payload)
        try:
            decision, primary, _ = decide_route(parsed)
        except InstanceConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        upstream_headers = {"Content-Type": "application/json"}
        authorization = request.headers.get("authorization")
        if authorization:
            upstream_headers["Authorization"] = authorization
        attempted_instances: set[str] = set()
        excluded_instance_ids = parsed.get("excluded_instance_ids")
        if not isinstance(excluded_instance_ids, frozenset):
            excluded_instance_ids = frozenset()
        candidate_targets = fallback_targets(parsed, decision)
        selected = primary
        last_open_error: Exception | None = None
        open_failures = 0
        exact_instance_request = (
            parsed["is_auto"] is not True
            and str(parsed["requested_model"]) == primary.instance_id
        )

        for candidate_model_id, candidate_revision, fallback_kind in candidate_targets:
            while True:
                if (
                    selected.requested_model_id != candidate_model_id
                    or _routing_revision(selected.model_revision) != candidate_revision
                    or selected.instance_id in attempted_instances
                ):
                    try:
                        selected = instance_router.resolve(
                            candidate_model_id,
                            revision=candidate_revision,
                            exclude_instance_ids=attempted_instances | excluded_instance_ids,
                        )
                    except InstanceConflict:
                        break
                if exact_instance_request and attempted_instances:
                    break
                attempted_instances.add(selected.instance_id)
                revision = _routing_revision(selected.model_revision)
                profile = capability_registry.get(selected.requested_model_id, revision)
                if profile is None:
                    profile = _infer_gateway_model_profile(
                        selected.requested_model_id,
                        revision,
                        selected.resolved_path,
                    )
                    capability_registry.register(profile)
                try:
                    slot = await _acquire_generation_slot(
                        generation_scheduler,
                        selected.instance_id,
                        selected.selected_nodes,
                        timeout=generation_slot_timeout,
                        disconnect_checker=request.is_disconnected,
                    )
                except (GenerationQueueFull, GenerationSlotTimeout) as exc:
                    raise HTTPException(
                        status_code=429,
                        detail=str(exc),
                        headers={"Retry-After": "1"},
                    ) from exc
                except GenerationSlotCancelled as exc:
                    raise HTTPException(status_code=499, detail=str(exc)) from exc
                try:
                    instances.acquire_request_lease(selected.instance_id)
                except InstanceConflict as exc:
                    slot.release()
                    raise HTTPException(status_code=409, detail=str(exc)) from exc

                release_lock = threading.Lock()
                released = False
                selected_instance_id = selected.instance_id

                def release_gateway_resources() -> None:
                    nonlocal released
                    with release_lock:
                        if released:
                            return
                        released = True
                    try:
                        instances.release_request_lease(selected_instance_id)
                    except InstanceConflict:
                        pass
                    finally:
                        slot.release()

                upstream_payload = _gateway_upstream_payload(payload, selected, profile)
                upstream_body = json.dumps(
                    upstream_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                try:
                    if await request.is_disconnected():
                        raise GenerationSlotCancelled(
                            f"Client disconnected before opening {selected_instance_id}."
                        )
                    connection, upstream = await _run_gateway_open(
                        gateway_open,
                        f"http://127.0.0.1:{selected.http_port}/v1/chat/completions",
                        upstream_body,
                        upstream_headers,
                        600.0,
                    )
                    if upstream.status >= 500:
                        connection.close()
                        raise RuntimeError(f"Upstream returned HTTP {upstream.status}.")
                except BaseException as exc:
                    release_gateway_resources()
                    if not isinstance(exc, Exception):
                        raise
                    if isinstance(exc, GenerationSlotCancelled):
                        raise HTTPException(status_code=499, detail=str(exc)) from exc
                    last_open_error = exc
                    open_failures += 1
                    selected.health_ready = False
                    selected.health_sampled_at = time.time()
                    selected.health_issues = [f"Gateway open failed: {type(exc).__name__}."]
                    if exact_instance_request:
                        break
                    try:
                        selected = instance_router.resolve(
                            candidate_model_id,
                            revision=candidate_revision,
                            exclude_instance_ids=attempted_instances | excluded_instance_ids,
                        )
                    except InstanceConflict:
                        break
                    continue

                if open_failures:
                    auto_router.metrics.fallback_count += 1
                with lifecycle_lock:
                    session_id = parsed["session_id"]
                    if isinstance(session_id, str):
                        session_routes[session_id] = (
                            selected.requested_model_id,
                            revision,
                        )
                route_reason = decision.reason.value
                if open_failures:
                    if fallback_kind == "primary":
                        route_reason = "fallback_same_model_replica"
                    elif fallback_kind == "default":
                        route_reason = "fallback_default_model"
                    else:
                        route_reason = "fallback_same_capability_model"
                response_headers = {
                    "X-Tokenity-Routed-Model": selected.requested_model_id,
                    "X-Tokenity-Instance-ID": selected.instance_id,
                    "X-Tokenity-Model-Revision": selected.model_revision or revision,
                    "X-Tokenity-Route-Reason": route_reason,
                    "X-Tokenity-Route-Confidence": f"{decision.confidence:.3f}",
                    "X-Tokenity-Routing-Latency-Ms": f"{decision.routing_latency_ms:.3f}",
                    "X-Tokenity-Model": selected.requested_model_id,
                    "X-Tokenity-Queue-Wait-Ms": f"{slot.waited_seconds * 1_000:.3f}",
                }
                try:
                    content_type = upstream.getheader("Content-Type")
                    if content_type:
                        response_headers["Content-Type"] = content_type
                    upstream_request_id = upstream.getheader("X-Tokenity-Request-ID")
                    if upstream_request_id:
                        response_headers["X-Tokenity-Request-ID"] = upstream_request_id
                    is_stream = bool(payload.get("stream"))
                    if is_stream:
                        response_headers.update(
                            {
                                "Cache-Control": "no-cache, no-transform",
                                "X-Accel-Buffering": "no",
                            }
                        )
                        stream_close_lock = threading.Lock()
                        stream_read_lock = threading.Lock()
                        stream_closed = False
                        drain_started = False
                        client_disconnected = threading.Event()

                        def finish_gateway_stream() -> None:
                            nonlocal stream_closed
                            with stream_close_lock:
                                if stream_closed:
                                    return
                                stream_closed = True
                            connection.close()
                            release_gateway_resources()

                        def drain_gateway_stream() -> None:
                            try:
                                _drain_gateway_response(upstream, stream_read_lock)
                            except Exception:
                                logging.exception(
                                    "Tokenity could not finish draining a disconnected "
                                    "distributed chat stream."
                                )
                            finally:
                                finish_gateway_stream()

                        def close_gateway_stream() -> None:
                            nonlocal drain_started
                            with stream_close_lock:
                                if stream_closed:
                                    return
                                if client_disconnected.is_set():
                                    if drain_started:
                                        return
                                    drain_started = True
                                    drain_thread = threading.Thread(
                                        target=drain_gateway_stream,
                                        name=f"tokenity-stream-drain-{selected_instance_id}",
                                        daemon=True,
                                    )
                                    drain_thread.start()
                                    return
                            finish_gateway_stream()

                        preserve_collective = (
                            selected.world_size > 1
                            and str(selected.connection_mode).lower()
                            in {"jaccl", "jaccl-ring"}
                        )

                        def disconnect_gateway_stream() -> None:
                            if preserve_collective:
                                # A client disconnect is rank-local. Keep reading the
                                # upstream response to its synchronized terminal frame
                                # so JACCL remains reusable for the next chat request.
                                client_disconnected.set()
                            else:
                                close_gateway_stream()

                        return _GatewayStreamingResponse(
                            _gateway_body_chunks(
                                connection,
                                upstream,
                                on_close=close_gateway_stream,
                                read_lock=stream_read_lock,
                            ),
                            on_close=close_gateway_stream,
                            on_disconnect=disconnect_gateway_stream,
                            status_code=upstream.status,
                            headers=response_headers,
                            media_type=None,
                        )

                    try:
                        response_body = await run_in_threadpool(upstream.read)
                    finally:
                        connection.close()
                        release_gateway_resources()
                    return Response(
                        content=response_body,
                        status_code=upstream.status,
                        headers=response_headers,
                        media_type=None,
                    )
                except BaseException:
                    connection.close()
                    release_gateway_resources()
                    raise

        raise HTTPException(
            status_code=502,
            detail=(
                f"No routed model instance could accept the request: {last_open_error}"
                if last_open_error is not None
                else "No routed model instance could accept the request."
            ),
        )

    @app.post("/v1/video/generations")
    async def gateway_video_generations(request: FastAPIRequest) -> Response:
        try:
            payload = json.loads(await request.body())
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail="Request body must be valid JSON.",
            ) from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

        requested_target = payload.get("tokenity_instance_id") or payload.get("model")
        candidates = []
        for snapshot in instances.snapshots():
            if instance_roles.get(str(snapshot["instance_id"])) != "minimax-h3-video":
                continue
            if snapshot.get("state") not in {"ready", "busy"}:
                continue
            if snapshot.get("health_ready") is False:
                continue
            if requested_target and requested_target not in {
                snapshot.get("instance_id"),
                snapshot.get("requested_model_id"),
            }:
                continue
            candidates.append(snapshot)
        if not candidates:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"No ready MiniMax H3 instance matches {requested_target}."
                    if requested_target
                    else "No ready MiniMax H3 video instance is available."
                ),
            )
        candidates.sort(
            key=lambda item: (
                int(item.get("active_request_count") or 0),
                float(item.get("created_at") or 0),
                str(item.get("instance_id")),
            )
        )
        selected = instances.get(str(candidates[0]["instance_id"]))
        if selected is None:
            raise HTTPException(status_code=409, detail="Selected H3 instance disappeared.")
        quorum = instance_quorum(selected.instance_id)
        if quorum.get("ready") is not True:
            raise HTTPException(
                status_code=503,
                detail={
                    "stage": "minimax_h3_readiness",
                    "issues": quorum.get("issues") or [],
                },
            )
        try:
            slot = await _acquire_generation_slot(
                generation_scheduler,
                selected.instance_id,
                selected.selected_nodes,
                timeout=generation_slot_timeout,
                disconnect_checker=request.is_disconnected,
            )
        except (GenerationQueueFull, GenerationSlotTimeout) as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": "1"},
            ) from exc
        except GenerationSlotCancelled as exc:
            raise HTTPException(status_code=499, detail=str(exc)) from exc
        try:
            instances.acquire_request_lease(selected.instance_id)
        except InstanceConflict as exc:
            slot.release()
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        release_lock = threading.Lock()
        released = False

        def release_video_resources() -> None:
            nonlocal released
            with release_lock:
                if released:
                    return
                released = True
            try:
                instances.release_request_lease(selected.instance_id)
            except InstanceConflict:
                pass
            finally:
                slot.release()

        upstream_payload = dict(payload)
        upstream_payload.pop("tokenity_instance_id", None)
        if upstream_payload.get("model") in {
            selected.instance_id,
            selected.requested_model_id,
        }:
            # The target native server is already bound to this exact model.
            # Removing the Tokenity routing identifier avoids depending on the
            # backend's optional multi-model registry.
            upstream_payload.pop("model", None)
        upstream_body = json.dumps(
            upstream_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        upstream_headers = {"Content-Type": "application/json"}
        authorization = request.headers.get("authorization")
        if authorization:
            upstream_headers["Authorization"] = authorization
        try:
            connection, upstream = await _run_gateway_open(
                gateway_open,
                f"http://127.0.0.1:{selected.http_port}/v1/video/generations",
                upstream_body,
                upstream_headers,
                14_400.0,
            )
        except BaseException as exc:
            release_video_resources()
            if not isinstance(exc, Exception):
                raise
            selected.health_ready = False
            selected.health_sampled_at = time.time()
            selected.health_issues = [f"Video gateway open failed: {type(exc).__name__}."]
            raise HTTPException(
                status_code=502,
                detail=f"MiniMax H3 upstream could not accept the request: {exc}",
            ) from exc

        response_headers = {
            "X-Tokenity-Model": selected.requested_model_id,
            "X-Tokenity-Instance-ID": selected.instance_id,
            "X-Tokenity-Model-Revision": selected.model_revision or "unversioned",
            "X-Tokenity-Queue-Wait-Ms": f"{slot.waited_seconds * 1_000:.3f}",
        }
        content_type = upstream.getheader("Content-Type")
        if content_type:
            response_headers["Content-Type"] = content_type
        is_stream = bool(payload.get("stream")) or (
            isinstance(content_type, str)
            and content_type.lower().startswith("text/event-stream")
        )
        try:
            if is_stream:
                response_headers.update(
                    {
                        "Cache-Control": "no-cache, no-transform",
                        "X-Accel-Buffering": "no",
                    }
                )
                close_lock = threading.Lock()
                closed = False

                def close_video_stream() -> None:
                    nonlocal closed
                    with close_lock:
                        if closed:
                            return
                        closed = True
                    connection.close()
                    release_video_resources()

                return _GatewayStreamingResponse(
                    _gateway_body_chunks(
                        connection,
                        upstream,
                        on_close=close_video_stream,
                    ),
                    on_close=close_video_stream,
                    status_code=upstream.status,
                    headers=response_headers,
                    media_type=None,
                )
            try:
                body = await run_in_threadpool(upstream.read)
            finally:
                connection.close()
                release_video_resources()
            return Response(
                content=body,
                status_code=upstream.status,
                headers=response_headers,
                media_type=None,
            )
        except BaseException:
            connection.close()
            release_video_resources()
            raise

    @app.post("/v1/node/heartbeat")
    def heartbeat(request: HeartbeatRequest) -> dict[str, object]:
        worker_renewals: list[dict[str, object]] = []
        if request.instance_id:
            instance = instances.get(request.instance_id)
            if instance is None:
                raise HTTPException(status_code=404, detail=f"Unknown model instance: {request.instance_id}")
            instance.heartbeat(request.ttl_seconds)
            deadline = instance.deadline
            deadline_kind = "epoch"
            for agent_url in instance_workers.get(request.instance_id, []):
                try:
                    result = post_json(
                        f"{agent_url}/v1/node/heartbeat",
                        request.model_dump(mode="json"),
                        3.0,
                    )
                    worker_renewals.append(
                        {"agent_url": agent_url, "status": "ok", "result": result}
                    )
                except Exception as exc:
                    worker_renewals.append(
                        {"agent_url": agent_url, "status": "error", "error": str(exc)}
                    )
        else:
            managed_instances = [
                snapshot
                for snapshot in instances.snapshots()
                if snapshot.get("state") not in {"stopped", "failed", "orphaned"}
            ]
            if managed_instances:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "A managed instance is active; heartbeat must include its "
                        "instance_id instead of renewing the legacy global lease."
                    ),
                )
            deadline = renew_lease(request.ttl_seconds)
            deadline_kind = "monotonic"
        return {
            "status": "ok",
            "lease_seconds": request.ttl_seconds,
            f"deadline_{deadline_kind}": deadline,
            "worker_renewals": worker_renewals,
        }

    @app.post("/v1/node/start-minimax-h3-video")
    def start_minimax_h3_video(
        request: H3VideoStartRequest,
    ) -> dict[str, object]:
        if not request.dry_run and Path(request.python).resolve() != Path(sys.executable).resolve():
            raise HTTPException(
                status_code=400,
                detail="MiniMax H3 startup must use the coordinator Agent's installed Python runtime.",
            )
        nodes = _h3_request_nodes(request)
        instance_id = request.instance_id or uuid.uuid4().hex
        operation_id = request.operation_id or uuid.uuid4().hex
        request.instance_id = instance_id
        request.operation_id = operation_id
        if len(nodes) == 1:
            request.connection_mode = ConnectionMode.RING
        else:
            request.connection_mode = _stable_connection_mode_for_model(
                request.model,
                request.connection_mode,
            )
        model_revision = _model_revision(request.model)
        try:
            rank_requests = _h3_rank_requests(
                request,
                nodes,
                model_revision=model_revision,
            )
        except HostfileError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        plan = _h3_launch_plan(nodes, rank_requests)
        runtime_fingerprint = h3_runtime_fingerprint(
            binary=request.binary,
            model=request.model,
            optimization_profile=request.optimization_profile,
            tokenity_code_revision=_tokenity_code_revision(),
        )
        disk = _disk_stats(request.model)
        if int(disk["free_bytes"]) < request.minimum_free_disk_bytes:
            raise HTTPException(
                status_code=412,
                detail={
                    "stage": "minimax_h3_disk_preflight",
                    "required_free_bytes": request.minimum_free_disk_bytes,
                    "disk": disk,
                    "suggestion": "Archive prior benchmark traces or remove an obsolete staged runtime before deployment.",
                },
            )

        issues, capabilities = h3_preflight(
            binary=request.binary,
            model=request.model,
            world_size=len(nodes),
            rank=0 if len(nodes) > 1 else None,
        )
        if issues:
            raise HTTPException(
                status_code=412,
                detail={
                    "stage": "minimax_h3_preflight",
                    "instance_id": instance_id,
                    "issues": issues,
                    "backend": _h3_capabilities_payload(capabilities),
                },
            )
        if request.dry_run:
            if len(nodes) > 1:
                require_managed_instance_capabilities(
                    nodes,
                    additional=frozenset({"minimax_h3_video"}),
                )
                try:
                    for node, rank_request in zip(nodes[1:], rank_requests[1:]):
                        post_json(
                            f"{_agent_url(node)}/v1/node/start-minimax-h3-video-rank",
                            rank_request.model_dump(mode="json"),
                            120.0,
                        )
                except Exception as exc:
                    raise HTTPException(
                        status_code=412,
                        detail={
                            "stage": "minimax_h3_preflight",
                            "issues": [f"The second Mac failed video preflight: {exc}"],
                        },
                    ) from exc
            return {
                "dry_run": True,
                "instance_id": instance_id,
                "operation_id": operation_id,
                "execution_mode": "single" if len(nodes) == 1 else "tp2",
                "model_revision": model_revision,
                "runtime_fingerprint": runtime_fingerprint,
                "launch_plan": plan,
            }
        if len(nodes) > 1:
            require_managed_instance_capabilities(
                nodes,
                additional=frozenset({"minimax_h3_video"}),
            )
            ledger_snapshot = resource_ledger_snapshot()
            request.port = _allocate_instance_port(request.port, ledger_snapshot)
            request.starting_port = _allocate_collective_port(
                request.starting_port,
                len(nodes),
                ledger_snapshot,
                excluded_ports={request.port},
            )
            try:
                rank_requests = _h3_rank_requests(
                    request,
                    nodes,
                    model_revision=model_revision,
                )
            except HostfileError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            plan = _h3_launch_plan(nodes, rank_requests)

            reservation_breakdown = _estimated_h3_memory_reservation_breakdown(
                request.model,
                world_size=2,
                rank=0,
            )
            reservation = request.memory_reservation_bytes
            if reservation is None:
                reservation = reservation_breakdown.estimated_total_bytes
            else:
                reservation_breakdown = MemoryReservationBreakdown(
                    legacy_unattributed_bytes=reservation
                )
            clear_legacy_lease()
            try:
                instance, created = instances.create(
                    instance_id=instance_id,
                    operation_id=operation_id,
                    requested_model_id=request.api_identifier or Path(request.model).name,
                    resolved_path=request.model,
                    model_revision=model_revision,
                    tokenizer_identity=None,
                    execution_mode="tp2",
                    selected_nodes=[node.id for node in nodes],
                    rank_mapping={node.id: rank for rank, node in enumerate(nodes)},
                    world_size=2,
                    connection_mode=request.connection_mode.value,
                    coordinator=nodes[0].id,
                    http_port=request.port,
                    starting_port=request.starting_port,
                    memory_reservation_bytes=reservation,
                    memory_reservation_breakdown=reservation_breakdown,
                )
            except InstanceConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            if not created:
                role = instance_roles.get(instance_id)
                status = supervisor_status(role, instance_id) if role else None
                return {
                    "dry_run": False,
                    "idempotent": True,
                    "instance": instance.to_dict(),
                    "status": status.__dict__ if status is not None else None,
                }
            ports = [
                request.port,
                *range(request.starting_port, request.starting_port + 2),
            ]
            try:
                resource_ledger.reserve(
                    instance_id,
                    reservation,
                    ports,
                    system_available_memory_bytes=system_available_memory_bytes(),
                )
            except ResourceAdmissionError as exc:
                instances.remove(instance_id)
                raise HTTPException(
                    status_code=409,
                    detail={"stage": "resource_admission", "message": str(exc)},
                ) from exc

            instance.transition(InstanceLifecycle.LAUNCHING)
            instance.transition(InstanceLifecycle.DISTRIBUTED_INITIALIZING)
            generation = begin_launch(instance_id)
            local_request = rank_requests[0]
            try:
                local_command, local_env = _h3_rank_command_and_environment(
                    local_request,
                    actual_world_size=2,
                )
            except HostfileError as exc:
                resource_ledger.release(instance_id)
                instance.transition(
                    InstanceLifecycle.FAILED,
                    error={"stage": "rank_environment", "message": str(exc)},
                )
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            role = "minimax-h3-video"
            instance_roles[instance_id] = role
            instance_ports[instance_id] = request.port
            instance.heartbeat(request.lease_seconds)
            local_status = supervisor_start(
                role,
                local_command,
                env=local_env,
                instance_id=instance_id,
                operation_id=operation_id,
            )
            persist_instance_state(
                instance,
                role=role,
                rank=0,
                command=local_command,
                status_path=None,
                pid=local_status.pid,
                recovery_policy="cleanup",
            )
            started_workers: list[str] = []
            try:
                if local_status.pid is None:
                    raise RuntimeError("Native MiniMax H3 coordinator process did not start.")
                for node, rank_request in zip(nodes[1:], rank_requests[1:]):
                    if not launch_is_current(instance_id, generation):
                        raise _LaunchCancelled("MiniMax H3 loading was cancelled.")
                    agent_url = _agent_url(node)
                    post_json(
                        f"{agent_url}/v1/node/start-minimax-h3-video-rank",
                        rank_request.model_dump(mode="json"),
                        45.0,
                    )
                    started_workers.append(agent_url)
                if not launch_is_current(instance_id, generation):
                    raise _LaunchCancelled("MiniMax H3 loading was cancelled.")
                if not wait_for_h3(local_status.pid, request.port, 180.0):
                    failed = supervisor_status(role, instance_id)
                    raise RuntimeError(
                        getattr(failed, "message", None)
                        or "Native MiniMax H3 rank 0 /health deadline exceeded."
                    )
            except Exception as exc:
                for agent_url in started_workers:
                    try:
                        post_json(
                            f"{agent_url}/v1/node/instances/{instance_id}/stop",
                            {"timeout": 5},
                            8.0,
                        )
                    except Exception:
                        pass
                supervisor_stop(role, 5, instance_id)
                resource_ledger.release(instance_id)
                remove_instance_state(instance_id)
                instance.transition(
                    InstanceLifecycle.FAILED,
                    error={"stage": "minimax_h3_tp2_launch", "message": str(exc)},
                )
                if isinstance(exc, _LaunchCancelled):
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                raise HTTPException(
                    status_code=502,
                    detail=f"Could not start MiniMax H3 TP2 ranks: {exc}",
                ) from exc

            with lifecycle_lock:
                instance_workers[instance_id] = list(started_workers)
            persist_instance_state(
                instance,
                role=role,
                rank=0,
                command=local_command,
                status_path=None,
                pid=local_status.pid,
                worker_agent_urls=started_workers,
                recovery_policy="cleanup",
            )
            instance.process_identities[nodes[0].id] = local_status.pid
            if local_status.log_path:
                instance.log_paths[nodes[0].id] = local_status.log_path
            advance_h3_instance_to_ready(
                instance,
                {
                    "rank_quorum": "2/2",
                    "model_revision": model_revision,
                    "connection_mode": request.connection_mode.value,
                    "all_ranks_healthy": True,
                    "native_video_health": True,
                    "distributed_protocol": H3_DISTRIBUTED_PROTOCOL_VERSION,
                    "optimization_profile": request.optimization_profile,
                    "optimization_flags": h3_optimization_environment(request.optimization_profile),
                    "runtime_fingerprint": runtime_fingerprint,
                },
            )
            instance.health_ready = True
            instance.health_sampled_at = time.time()
            instance.health_issues = []
            return {
                "dry_run": False,
                "instance_id": instance_id,
                "operation_id": operation_id,
                "instance": instance.to_dict(),
                "api_base_url": (
                    f"http://{nodes[0].lan_ip or '127.0.0.1'}:{request.port}/v1"
                ),
                "launch_plan": plan,
                "backend": _h3_capabilities_payload(capabilities),
                "status": local_status.__dict__,
                "workers": started_workers,
                "runtime_fingerprint": runtime_fingerprint,
            }

        ledger_snapshot = resource_ledger_snapshot()
        request.port = _allocate_instance_port(request.port, ledger_snapshot)
        reservation_breakdown = _estimated_h3_memory_reservation_breakdown(
            request.model,
            world_size=1,
            rank=0,
        )
        reservation = request.memory_reservation_bytes
        if reservation is None:
            reservation = reservation_breakdown.estimated_total_bytes
        else:
            reservation_breakdown = MemoryReservationBreakdown(
                legacy_unattributed_bytes=reservation
            )
        clear_legacy_lease()
        try:
            instance, created = instances.create(
                instance_id=instance_id,
                operation_id=operation_id,
                requested_model_id=request.api_identifier or Path(request.model).name,
                resolved_path=request.model,
                model_revision=model_revision,
                tokenizer_identity=None,
                execution_mode="single",
                selected_nodes=[nodes[0].id],
                rank_mapping={nodes[0].id: 0},
                world_size=1,
                connection_mode="single",
                coordinator=nodes[0].id,
                http_port=request.port,
                starting_port=request.starting_port,
                memory_reservation_bytes=reservation,
                memory_reservation_breakdown=reservation_breakdown,
            )
        except InstanceConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not created:
            role = instance_roles.get(instance_id)
            status = supervisor_status(role, instance_id) if role else None
            return {
                "dry_run": False,
                "idempotent": True,
                "instance": instance.to_dict(),
                "status": status.__dict__ if status is not None else None,
            }
        try:
            resource_ledger.reserve(
                instance_id,
                reservation,
                [request.port],
                system_available_memory_bytes=system_available_memory_bytes(),
            )
        except ResourceAdmissionError as exc:
            instances.remove(instance_id)
            raise HTTPException(
                status_code=409,
                detail={"stage": "resource_admission", "message": str(exc)},
            ) from exc

        instance.transition(InstanceLifecycle.LAUNCHING)
        instance.transition(InstanceLifecycle.LOADING_METADATA)
        rank_request = rank_requests[0]
        rank_request.port = request.port
        command, env = _h3_rank_command_and_environment(
            rank_request,
            actual_world_size=1,
        )
        role = "minimax-h3-video"
        instance_roles[instance_id] = role
        instance_ports[instance_id] = request.port
        instance.heartbeat(request.lease_seconds)
        status = supervisor_start(
            role,
            command,
            env=env,
            instance_id=instance_id,
            operation_id=operation_id,
        )
        persist_instance_state(
            instance,
            role=role,
            rank=0,
            command=command,
            status_path=None,
            pid=status.pid,
            recovery_policy="cleanup",
        )
        if status.pid is None or not wait_for_h3(status.pid, request.port, 120.0):
            failed = supervisor_status(role, instance_id)
            supervisor_stop(role, 5, instance_id)
            resource_ledger.release(instance_id)
            remove_instance_state(instance_id)
            instance.transition(
                InstanceLifecycle.FAILED,
                error={
                    "stage": "native_video_readiness",
                    "message": getattr(failed, "message", None)
                    or "Native MiniMax H3 /health deadline exceeded.",
                },
            )
            raise HTTPException(
                status_code=504,
                detail="Native MiniMax H3 server did not become healthy.",
            )

        instance.process_identities[nodes[0].id] = status.pid
        if status.log_path:
            instance.log_paths[nodes[0].id] = status.log_path
        advance_h3_instance_to_ready(
            instance,
            {
                "rank_quorum": "1/1",
                "model_revision": model_revision,
                "connection_mode": "single",
                "all_ranks_healthy": True,
                "native_video_health": True,
                "distributed_protocol": None,
            },
        )
        instance.health_ready = True
        instance.health_sampled_at = time.time()
        instance.health_issues = []
        return {
            "dry_run": False,
            "instance_id": instance_id,
            "operation_id": operation_id,
            "instance": instance.to_dict(),
            "api_base_url": f"http://{nodes[0].lan_ip or '127.0.0.1'}:{request.port}/v1",
            "launch_plan": plan,
            "backend": _h3_capabilities_payload(capabilities),
            "status": status.__dict__,
        }

    @app.post("/v1/node/start-minimax-h3-video-rank")
    def start_minimax_h3_video_rank(
        request: H3VideoRankStartRequest,
    ) -> dict[str, object]:
        if request.coordinator:
            raise HTTPException(
                status_code=400,
                detail="Remote H3 rank endpoint cannot start the coordinator role.",
            )
        if request.rank >= request.world_size:
            raise HTTPException(status_code=400, detail="Rank must be smaller than world size.")
        if Path(request.python).resolve() != Path(sys.executable).resolve():
            raise HTTPException(
                status_code=400,
                detail="Remote H3 ranks must use the Node Agent's installed Python runtime.",
            )
        issues, capabilities = h3_preflight(
            binary=request.binary,
            model=request.model,
            world_size=request.world_size,
            rank=request.rank,
        )
        if issues:
            raise HTTPException(
                status_code=412,
                detail={
                    "stage": "minimax_h3_preflight",
                    "rank": request.rank,
                    "issues": issues,
                    "backend": _h3_capabilities_payload(capabilities),
                },
            )
        disk = _disk_stats(request.model)
        if int(disk["free_bytes"]) < request.minimum_free_disk_bytes:
            raise HTTPException(
                status_code=412,
                detail={
                    "stage": "minimax_h3_disk_preflight",
                    "rank": request.rank,
                    "required_free_bytes": request.minimum_free_disk_bytes,
                    "disk": disk,
                    "suggestion": "Archive prior benchmark traces or remove an obsolete staged runtime before deployment.",
                },
            )
        instance_id = request.instance_id or request.cluster_id
        operation_id = request.operation_id or request.cluster_id
        local_revision = _model_revision(request.model)
        revision_issues: list[str] = []
        if request.model_revision and local_revision != request.model_revision:
            revision_issues.append(
                f"Model revision mismatch: expected {request.model_revision}, "
                f"local {local_revision or 'unknown'}."
            )
        local_code_revision = _tokenity_code_revision()
        if request.tokenity_code_revision and local_code_revision != request.tokenity_code_revision:
            revision_issues.append(
                "Tokenity code revision mismatch: expected "
                f"{request.tokenity_code_revision}, local {local_code_revision or 'unknown'}."
            )
        if revision_issues:
            raise HTTPException(
                status_code=412,
                detail={
                    "stage": "minimax_h3_revision_preflight",
                    "instance_id": instance_id,
                    "rank": request.rank,
                    "issues": revision_issues,
                },
            )
        local_runtime_fingerprint = h3_runtime_fingerprint(
            binary=request.binary,
            model=request.model,
            optimization_profile=request.optimization_profile,
            tokenity_code_revision=local_code_revision,
            verify_rank=request.rank,
        )
        local_contract = str(local_runtime_fingerprint["contract_sha256"])
        if request.runtime_contract_sha256 and local_contract != request.runtime_contract_sha256:
            raise HTTPException(
                status_code=412,
                detail={
                    "stage": "minimax_h3_runtime_fingerprint",
                    "instance_id": instance_id,
                    "rank": request.rank,
                    "expected": request.runtime_contract_sha256,
                    "local": local_contract,
                    "fingerprint": local_runtime_fingerprint,
                },
            )
        if request.dry_run:
            return {
                "dry_run": True,
                "instance_id": instance_id,
                "rank": request.rank,
                "model_revision": local_revision,
                "runtime_fingerprint": local_runtime_fingerprint,
            }
        try:
            command, env = _h3_rank_command_and_environment(
                request,
                actual_world_size=request.world_size,
            )
        except HostfileError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        reservation_breakdown = _estimated_h3_memory_reservation_breakdown(
            request.model,
            world_size=request.world_size,
            rank=request.rank,
        )
        reservation = request.memory_reservation_bytes
        if reservation is None:
            reservation = reservation_breakdown.estimated_total_bytes
        else:
            reservation_breakdown = MemoryReservationBreakdown(
                legacy_unattributed_bytes=reservation
            )
        try:
            instance, created = instances.create(
                instance_id=instance_id,
                operation_id=operation_id,
                requested_model_id=request.api_identifier or Path(request.model).name,
                resolved_path=request.model,
                model_revision=local_revision,
                tokenizer_identity=None,
                execution_mode="tp2",
                selected_nodes=[f"rank-{request.rank}"],
                rank_mapping={f"rank-{request.rank}": request.rank},
                world_size=request.world_size,
                connection_mode=request.connection_mode.value,
                coordinator="rank-0",
                http_port=request.port,
                starting_port=request.starting_port,
                memory_reservation_bytes=reservation,
                memory_reservation_breakdown=reservation_breakdown,
            )
            if created:
                resource_ledger.reserve(
                    instance_id,
                    reservation,
                    list(
                        range(
                            request.starting_port,
                            request.starting_port + request.world_size,
                        )
                    ),
                    system_available_memory_bytes=system_available_memory_bytes(),
                )
                instance.transition(InstanceLifecycle.LAUNCHING)
                instance.transition(InstanceLifecycle.DISTRIBUTED_INITIALIZING)
            instance.heartbeat(request.lease_seconds)
        except (InstanceConflict, ResourceAdmissionError) as exc:
            instances.remove(instance_id)
            resource_ledger.release(instance_id)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not created:
            role = instance_roles.get(instance_id)
            status = supervisor_status(role, instance_id) if role else None
            return {
                "idempotent": True,
                "rank": request.rank,
                "cluster_id": request.cluster_id,
                "instance": instance.to_dict(),
                "status": status.__dict__ if status is not None else None,
            }

        generation = begin_launch(instance_id)
        role = "minimax-h3-video-rank"
        instance_roles[instance_id] = role
        instance_ports[instance_id] = request.port
        status = supervisor_start(
            role,
            command,
            env=env,
            instance_id=instance_id,
            operation_id=operation_id,
        )
        persist_instance_state(
            instance,
            role=role,
            rank=request.rank,
            command=command,
            status_path=None,
            pid=status.pid,
            recovery_policy="cleanup",
        )
        connected = (
            status.pid is not None
            and wait_for_rank_connection(status.pid, request, 30.0)
        )
        if not connected or not launch_is_current(instance_id, generation):
            failed = supervisor_status(role, instance_id)
            supervisor_stop(role, 5, instance_id)
            resource_ledger.release(instance_id)
            remove_instance_state(instance_id)
            instance.transition(
                InstanceLifecycle.FAILED,
                error={
                    "stage": "minimax_h3_tp2_data_plane",
                    "message": getattr(failed, "message", None)
                    or "MiniMax H3 worker rank connection failed.",
                },
            )
            if not launch_is_current(instance_id, generation):
                raise HTTPException(status_code=409, detail="MiniMax H3 loading was cancelled.")
            raise HTTPException(
                status_code=504,
                detail=(
                    getattr(failed, "message", None)
                    or getattr(failed, "log_tail", None)
                    or "MiniMax H3 worker rank could not establish its MLX data plane."
                ),
            )
        instance.process_identities[f"rank-{request.rank}"] = status.pid or 0
        if status.log_path:
            instance.log_paths[f"rank-{request.rank}"] = status.log_path
        instance.transition(InstanceLifecycle.LOADING_METADATA)
        return {
            "status": status.__dict__,
            "rank": request.rank,
            "cluster_id": request.cluster_id,
            "instance": instance.to_dict(),
            "backend": _h3_capabilities_payload(capabilities),
            "runtime_fingerprint": local_runtime_fingerprint,
        }

    @app.post("/v1/node/start-distributed-openai")
    def start_distributed(request: StartRequest) -> dict[str, object]:
        if issue := standalone_model_issue(request.model):
            raise HTTPException(status_code=400, detail=issue)
        if not request.dry_run and Path(request.python).resolve() != Path(sys.executable).resolve():
            raise HTTPException(
                status_code=400,
                detail="Cluster startup must use the coordinator Agent's installed Python runtime.",
            )
        nodes = _request_nodes(request)
        if len(nodes) > 1 and (issue := distributed_model_issue(request.model)):
            # Capability validation is deliberately before operation identity,
            # admission, port allocation, reservation, or process creation.
            raise HTTPException(status_code=400, detail=issue)
        instance_id = request.instance_id or uuid.uuid4().hex
        operation_id = request.operation_id or uuid.uuid4().hex
        request.instance_id = instance_id
        request.operation_id = operation_id
        if len(nodes) == 1:
            request.connection_mode = ConnectionMode.RING
        else:
            requested_connection_mode = request.connection_mode
            request.connection_mode = _stable_connection_mode_for_model(
                request.model,
                request.connection_mode,
            )
            if request.connection_mode != requested_connection_mode:
                logging.warning(
                    "Tokenity selected %s instead of %s because the direct "
                    "JACCL mesh can fail with Recv error -12 after readiness "
                    "or when another resident communicator is active.",
                    request.connection_mode.value,
                    requested_connection_mode.value,
                )
            if not request.dry_run:
                require_managed_instance_capabilities(nodes)
        model_revision = _model_revision(request.model)
        if not request.dry_run:
            issues = runtime_preflight(request.python, request.model, model_revision)
            if issues:
                raise HTTPException(
                    status_code=412,
                    detail={"stage": "preflight", "instance_id": instance_id, "issues": issues},
                )
            ledger_snapshot = resource_ledger_snapshot()
            request.port = _allocate_instance_port(request.port, ledger_snapshot)
            request.starting_port = _allocate_collective_port(
                request.starting_port,
                len(nodes),
                ledger_snapshot,
                excluded_ports={request.port},
            )
        try:
            rank_requests = _http_rank_requests(request, nodes, model_revision=model_revision)
        except HostfileError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        plan = _http_launch_plan(nodes, rank_requests)
        if request.dry_run:
            return {
                "dry_run": True,
                "instance_id": instance_id,
                "operation_id": operation_id,
                "execution_mode": "single" if len(nodes) == 1 else request.connection_mode.value,
                "model_revision": model_revision,
                "launch_plan": plan,
            }

        reservation_breakdown = _estimated_memory_reservation_breakdown(
            request.model,
            len(nodes),
            max_tokens=request.max_tokens,
            prompt_cache_size=request.prompt_cache_size,
            prefill_step_size=request.prefill_step_size,
            decode_concurrency=request.decode_concurrency,
            prompt_concurrency=request.prompt_concurrency,
        )
        reservation = request.memory_reservation_bytes
        if reservation is None:
            reservation = reservation_breakdown.estimated_total_bytes
        else:
            reservation_breakdown = MemoryReservationBreakdown(
                legacy_unattributed_bytes=reservation
            )
        # Managed instances own independent epoch leases. A stale legacy lease
        # must never retain authority to stop every sibling instance.
        clear_legacy_lease()
        try:
            instance, created = instances.create(
                instance_id=instance_id,
                operation_id=operation_id,
                requested_model_id=request.api_identifier or Path(request.model).name,
                resolved_path=request.model,
                model_revision=model_revision,
                tokenizer_identity=None,
                execution_mode="single" if len(nodes) == 1 else request.connection_mode.value,
                selected_nodes=[node.id for node in nodes],
                rank_mapping={node.id: rank for rank, node in enumerate(nodes)},
                world_size=len(nodes),
                connection_mode="single" if len(nodes) == 1 else request.connection_mode.value,
                coordinator=nodes[0].id,
                http_port=request.port,
                starting_port=request.starting_port,
                memory_reservation_bytes=reservation,
                memory_reservation_breakdown=reservation_breakdown,
            )
        except InstanceConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not created:
            role = instance_roles.get(instance_id)
            status = supervisor_status(role, instance_id) if role else None
            return {
                "dry_run": False,
                "idempotent": True,
                "instance": instance.to_dict(),
                "status": status.__dict__ if status is not None else None,
            }
        ports = [request.port, *range(request.starting_port, request.starting_port + max(1, len(nodes)))]
        try:
            resource_ledger.reserve(
                instance_id,
                reservation,
                ports,
                system_available_memory_bytes=system_available_memory_bytes(),
            )
        except ResourceAdmissionError as exc:
            instances.remove(instance_id)
            raise HTTPException(status_code=409, detail={"stage": "resource_admission", "message": str(exc)}) from exc
        instance.transition(InstanceLifecycle.LAUNCHING)

        generation = begin_launch(instance_id)
        local_request = rank_requests[0]
        set_cluster_runtime(local_request, "controller")
        local_command, local_env = _rank_command_and_environment(local_request)
        local_role = "single-node-openai" if len(nodes) == 1 else "distributed-openai"
        instance_roles[instance_id] = local_role
        instance_ports[instance_id] = request.port
        instance_status_paths[instance_id] = local_env["TOKENITY_STATUS_PATH"]
        instance.heartbeat(request.lease_seconds)
        instance.transition(
            InstanceLifecycle.LOADING_METADATA
            if len(nodes) == 1
            else InstanceLifecycle.DISTRIBUTED_INITIALIZING
        )
        local_status = supervisor_start(
            local_role,
            local_command,
            env=local_env,
            instance_id=instance_id,
            operation_id=operation_id,
        )
        persist_instance_state(
            instance,
            role=local_role,
            rank=0,
            command=local_command,
            status_path=local_env["TOKENITY_STATUS_PATH"],
            pid=local_status.pid,
        )
        if len(nodes) > 1 and request.connection_mode != ConnectionMode.RING:
            if local_status.pid is None or not wait_for_rank(local_status.pid, request.starting_port, 45.0):
                supervisor_stop(local_role, 5, instance_id)
                resource_ledger.release(instance_id)
                remove_cluster_runtime(instance_id)
                remove_instance_state(instance_id)
                instance.transition(
                    InstanceLifecycle.FAILED,
                    error={"stage": "distributed_initializing", "message": "Coordinator JACCL port deadline exceeded."},
                )
                raise HTTPException(
                    status_code=504,
                    detail="Coordinator rank did not open the JACCL port before the startup deadline.",
                )
            stabilize_rank(3.0)
        if not launch_is_current(instance_id, generation):
            supervisor_stop(local_role, 5, instance_id)
            resource_ledger.release(instance_id)
            remove_cluster_runtime(instance_id)
            remove_instance_state(instance_id)
            raise HTTPException(status_code=409, detail="Model loading was cancelled.")
        started_workers: list[str] = []
        try:
            for node, rank_request in zip(nodes[1:], rank_requests[1:]):
                if not launch_is_current(instance_id, generation):
                    raise _LaunchCancelled("Model loading was cancelled.")
                agent_url = _agent_url(node)
                worker_payload = rank_request.model_dump(mode="json")
                # Native MTP was added after the typed worker-rank endpoint.
                # Omitting its no-op default keeps rolling upgrades compatible
                # with an older worker Agent whose strict schema does not know
                # this field yet. Non-default requests must remain explicit.
                if rank_request.native_mtp.mode == "off":
                    worker_payload.pop("native_mtp", None)
                post_json(
                    f"{agent_url}/v1/node/start-distributed-rank",
                    worker_payload,
                    25.0,
                )
                started_workers.append(agent_url)
                if not launch_is_current(instance_id, generation):
                    raise _LaunchCancelled("Model loading was cancelled.")
        except Exception as exc:
            for agent_url in started_workers:
                try:
                    post_json(
                        f"{agent_url}/v1/node/instances/{instance_id}/stop",
                        {"timeout": 5},
                        8.0,
                    )
                except Exception:
                    pass
            supervisor_stop(local_role, 5, instance_id)
            resource_ledger.release(instance_id)
            remove_cluster_runtime(instance_id)
            remove_instance_state(instance_id)
            instance.transition(
                InstanceLifecycle.FAILED,
                error={"stage": "launching", "message": str(exc)},
            )
            if isinstance(exc, _LaunchCancelled):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise HTTPException(status_code=502, detail=f"Could not start worker rank over HTTP: {exc}") from exc

        with lifecycle_lock:
            instance_workers[instance_id] = list(started_workers)
        persist_instance_state(
            instance,
            role=local_role,
            rank=0,
            command=local_command,
            status_path=local_env["TOKENITY_STATUS_PATH"],
            pid=local_status.pid,
            worker_agent_urls=started_workers,
        )
        if len(nodes) > 1:
            instance.transition(InstanceLifecycle.LOADING_METADATA)
        instance.process_identities[nodes[0].id] = local_status.pid or 0
        if local_status.log_path:
            instance.log_paths[nodes[0].id] = local_status.log_path
        return {
            "dry_run": False,
            "instance_id": instance_id,
            "operation_id": operation_id,
            "instance": instance.to_dict(),
            "api_base_url": f"http://{nodes[0].lan_ip or '127.0.0.1'}:{request.port}/v1",
            "launch_plan": plan,
            "status": local_status.__dict__,
            "workers": started_workers,
        }

    @app.post("/v1/node/start-distributed-rank")
    def start_distributed_rank(request: RankStartRequest) -> dict[str, object]:
        if issue := standalone_model_issue(request.model):
            raise HTTPException(status_code=400, detail=issue)
        if request.world_size > 1 and (issue := distributed_model_issue(request.model)):
            raise HTTPException(status_code=400, detail=issue)
        if request.rank >= request.world_size:
            raise HTTPException(status_code=400, detail="Rank must be smaller than world size.")
        if request.coordinator:
            raise HTTPException(status_code=400, detail="Remote rank endpoint cannot start the coordinator role.")
        if Path(request.python).resolve() != Path(sys.executable).resolve():
            raise HTTPException(
                status_code=400,
                detail="Remote ranks must use the Node Agent's installed Python runtime.",
            )
        instance_id = request.instance_id or request.cluster_id
        operation_id = request.operation_id or request.cluster_id
        local_revision = _model_revision(request.model)
        issues = runtime_preflight(request.python, request.model, request.model_revision or local_revision)
        if request.model_revision and local_revision != request.model_revision:
            issues.append(
                f"Model revision mismatch: expected {request.model_revision}, local {local_revision or 'unknown'}."
            )
        local_code_revision = _tokenity_code_revision()
        if request.tokenity_code_revision and local_code_revision != request.tokenity_code_revision:
            issues.append(
                "Tokenity code revision mismatch: expected "
                f"{request.tokenity_code_revision}, local {local_code_revision or 'unknown'}."
            )
        if issues:
            raise HTTPException(
                status_code=412,
                detail={"stage": "preflight", "instance_id": instance_id, "rank": request.rank, "issues": issues},
            )
        try:
            command, env = _rank_command_and_environment(request)
        except HostfileError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        generation = begin_launch(instance_id)
        set_cluster_runtime(request, "worker")
        reservation_breakdown = _estimated_memory_reservation_breakdown(
            request.model,
            request.world_size,
            max_tokens=request.max_tokens,
            prompt_cache_size=request.prompt_cache_size,
            prefill_step_size=request.prefill_step_size,
            decode_concurrency=request.decode_concurrency,
            prompt_concurrency=request.prompt_concurrency,
        )
        reservation = request.memory_reservation_bytes
        if reservation is None:
            reservation = reservation_breakdown.estimated_total_bytes
        else:
            reservation_breakdown = MemoryReservationBreakdown(
                legacy_unattributed_bytes=reservation
            )
        try:
            instance, created = instances.create(
                instance_id=instance_id,
                operation_id=operation_id,
                requested_model_id=Path(request.model).name,
                resolved_path=request.model,
                model_revision=local_revision,
                tokenizer_identity=None,
                execution_mode=request.connection_mode.value,
                selected_nodes=[f"rank-{request.rank}"],
                rank_mapping={f"rank-{request.rank}": request.rank},
                world_size=request.world_size,
                connection_mode=request.connection_mode.value,
                coordinator="rank-0",
                http_port=request.port,
                starting_port=request.starting_port,
                memory_reservation_bytes=reservation,
                memory_reservation_breakdown=reservation_breakdown,
            )
            if created:
                resource_ledger.reserve(
                    instance_id,
                    reservation,
                    list(range(request.starting_port, request.starting_port + max(1, request.world_size))),
                    system_available_memory_bytes=system_available_memory_bytes(),
                )
                instance.transition(InstanceLifecycle.LAUNCHING)
                instance.transition(InstanceLifecycle.DISTRIBUTED_INITIALIZING)
            instance.heartbeat(request.lease_seconds)
        except (InstanceConflict, ResourceAdmissionError) as exc:
            instances.remove(instance_id)
            resource_ledger.release(instance_id)
            remove_cluster_runtime(instance_id)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        instance_roles[instance_id] = "distributed-openai-rank"
        instance_ports[instance_id] = request.port
        instance_status_paths[instance_id] = env["TOKENITY_STATUS_PATH"]
        status = supervisor_start(
            "distributed-openai-rank",
            command,
            env=env,
            instance_id=instance_id,
            operation_id=operation_id,
        )
        connected = status.pid is not None and wait_for_rank_connection(status.pid, request, 20.0)
        if not connected or not launch_is_current(instance_id, generation):
            failed = supervisor_status("distributed-openai-rank", instance_id)
            supervisor_stop("distributed-openai-rank", 5, instance_id)
            resource_ledger.release(instance_id)
            remove_cluster_runtime(instance_id)
            remove_instance_state(instance_id)
            instance.transition(
                InstanceLifecycle.FAILED,
                error={"stage": "distributed_initializing", "message": getattr(failed, "message", None) or "Rank connection failed."},
            )
            if not launch_is_current(instance_id, generation):
                raise HTTPException(status_code=409, detail="Model loading was cancelled.")
            message = getattr(failed, "message", None) or getattr(failed, "log_tail", None)
            raise HTTPException(
                status_code=504,
                detail=message or "Worker rank could not establish its MLX data-plane connection.",
            )
        instance.transition(InstanceLifecycle.LOADING_METADATA)
        instance.process_identities[f"rank-{request.rank}"] = status.pid or 0
        if status.log_path:
            instance.log_paths[f"rank-{request.rank}"] = status.log_path
        persist_instance_state(
            instance,
            role="distributed-openai-rank",
            rank=request.rank,
            command=command,
            status_path=env["TOKENITY_STATUS_PATH"],
            pid=status.pid,
        )
        return {
            "status": status.__dict__,
            "rank": request.rank,
            "cluster_id": request.cluster_id,
            "instance": instance.to_dict(),
        }

    @app.post("/v1/node/instances/{instance_id}/stop")
    def stop_instance(instance_id: str, request: StopAllRequest) -> dict[str, object]:
        instance = instances.get(instance_id)
        if instance is None:
            raise HTTPException(status_code=404, detail=f"Unknown model instance: {instance_id}")
        cancel_launch(instance_id)
        role_before_stop = instance_roles.get(instance_id)
        status_before_stop = (
            supervisor_status(role_before_stop, instance_id) if role_before_stop else None
        )
        runtime_before_stop = _read_runtime_status(instance_status_paths.get(instance_id))
        reconcile_instance_runtime(
            instance,
            runtime_before_stop,
            status_before_stop,
            role_before_stop,
        )
        try:
            if instance.state not in {InstanceLifecycle.UNLOADING, InstanceLifecycle.STOPPED}:
                instance.transition(InstanceLifecycle.UNLOADING)
        except InstanceConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        port = instance_ports.get(instance_id, instance.http_port)
        coordinator_stop_requested = False
        try:
            _post_json(f"http://127.0.0.1:{port}/v1/tokenity/stop", {}, 2.0)
            coordinator_stop_requested = True
        except Exception:
            pass

        worker_results: list[dict[str, object]] = []
        worker_stop_timeout = request.timeout + (
            11
            if role_before_stop in VIDEO_MODEL_ROLES
            and instance.world_size > 1
            and _requires_coordinated_rank_stop(instance.connection_mode)
            else 3
        )
        for agent_url in instance_workers.get(instance_id, []):
            try:
                result = post_json(
                    f"{agent_url}/v1/node/instances/{instance_id}/stop",
                    {"timeout": request.timeout},
                    worker_stop_timeout,
                )
                worker_results.append({"agent_url": agent_url, "result": result})
            except Exception as exc:
                worker_results.append({"agent_url": agent_url, "error": str(exc)})

        role = instance_roles.get(instance_id)
        status = None
        if role is not None:
            deadline = time.monotonic() + request.timeout
            if role in COORDINATED_WORKER_ROLES and _requires_coordinated_rank_stop(
                instance.connection_mode,
            ):
                # The coordinator broadcasts a validated stop sentinel through
                # the existing control collective.  Give the worker rank time
                # to receive it, leave the generation loop, and clear its
                # thread-local MLX streams before escalating to SIGTERM.
                status = supervisor_wait(
                    role,
                    min(2.0, request.timeout),
                    instance_id,
                )
                if getattr(status, "pid", None) is not None:
                    supervisor_request_stop(role, instance_id)
                    status = supervisor_wait(
                        role,
                        max(0.0, deadline - time.monotonic()),
                        instance_id,
                    )
            elif (
                role in NATIVE_ADMIN_STOP_ROLES
                and coordinator_stop_requested
            ):
                # The successful admin request already initiated natural
                # Uvicorn/runtime teardown.  Let it finish before using a
                # signal fallback so a clean exit is not reported as -15.
                status = supervisor_wait(
                    role,
                    max(0.0, deadline - time.monotonic()),
                    instance_id,
                )
                if getattr(status, "pid", None) is not None:
                    supervisor_request_stop(role, instance_id)
                    status = supervisor_wait(
                        role,
                        max(0.0, deadline - time.monotonic()),
                        instance_id,
                    )
            else:
                supervisor_request_stop(role, instance_id)
                status = supervisor_wait(
                    role,
                    max(0.0, deadline - time.monotonic()),
                    instance_id,
                )
            if getattr(status, "pid", None) is not None:
                status = supervisor_stop(role, min(2.0, request.timeout), instance_id)
        if (
            role in VIDEO_MODEL_ROLES
            and instance.world_size > 1
            and _requires_coordinated_rank_stop(instance.connection_mode)
        ):
            _request_rdma_link_reset()
        resource_ledger.release(instance_id)
        instance_workers.pop(instance_id, None)
        instance_roles.pop(instance_id, None)
        instance_ports.pop(instance_id, None)
        remove_cluster_runtime(instance_id)
        remove_instance_state(instance_id)
        status_path = instance_status_paths.pop(instance_id, None)
        if status_path:
            try:
                Path(status_path).unlink(missing_ok=True)
            except OSError:
                pass
        if instance.state != InstanceLifecycle.STOPPED:
            instance.transition(InstanceLifecycle.STOPPED)
        return {
            "instance": instance.to_dict(),
            "status": status.__dict__ if status is not None else None,
            "workers": worker_results,
        }

    @app.post("/v1/node/stop-role")
    def stop_role(request: StopRequest) -> dict[str, object]:
        if request.instance_id:
            return stop_instance(request.instance_id, StopAllRequest(timeout=request.timeout))
        cancel_launch()
        worker_results: list[dict[str, object]] = []
        local_status: list[object] = []

        request_threads = [threading.Thread(target=request_local_model_roles_stop)]
        if request.role == "distributed-openai":
            request_threads.append(threading.Thread(target=request_known_workers_stop))
        for thread in request_threads:
            thread.start()
        for thread in request_threads:
            thread.join(4)

        def stop_local() -> None:
            wait_for_role = getattr(roles, "wait", None)
            status = wait_for_role(request.role, request.timeout) if wait_for_role is not None else roles.status(request.role)
            if getattr(status, "pid", None) is not None:
                status = roles.stop(request.role, timeout=2)
            local_status.append(status)

        local_thread = threading.Thread(target=stop_local)
        local_thread.start()
        if request.role == "distributed-openai":
            worker_results = stop_known_workers(request.timeout)
        local_thread.join(request.timeout + 3)
        status = local_status[0] if local_status else roles.status(request.role)
        result = {
            "status": status.__dict__,
            "workers": worker_results,
        }
        clear_lease()
        return result

    @app.post("/v1/node/stop-all")
    def stop_all(request: StopAllRequest) -> dict[str, object]:
        return stop_everything(request.timeout)

    @app.post("/v1/node/request-stop-all")
    def request_stop_all() -> dict[str, object]:
        """Phase-one distributed stop: signal local ranks without waiting."""

        cancel_launch()
        return {"statuses": request_local_model_roles_stop()}

    return app


async def _acquire_generation_slot(
    scheduler: GenerationSlotScheduler,
    instance_id: str,
    selected_nodes,
    *,
    timeout: float | None,
    disconnect_checker=None,
):
    """Acquire in a worker while retaining ownership across cancellation races."""

    cancel_event = threading.Event()
    holder_lock = threading.Lock()
    holder: dict[str, object] = {}

    def acquire():
        lease = scheduler.acquire(
            instance_id,
            selected_nodes,
            timeout=timeout,
            cancel_event=cancel_event,
        )
        with holder_lock:
            if cancel_event.is_set():
                lease.release()
                raise GenerationSlotCancelled(
                    f"Generation slot request for {instance_id} was cancelled."
                )
            holder["lease"] = lease
        return lease

    future = asyncio.get_running_loop().run_in_executor(None, acquire)
    try:
        while True:
            done, _ = await asyncio.wait({future}, timeout=0.05)
            if done:
                lease = future.result()
                break
            if disconnect_checker is not None and await disconnect_checker():
                raise GenerationSlotCancelled(
                    f"Client disconnected while waiting for {instance_id}."
                )
    except BaseException:
        cancel_event.set()
        with holder_lock:
            held = holder.pop("lease", None)
        if held is not None:
            held.release()
        if not future.done():
            future.add_done_callback(
                lambda completed: (
                    None
                    if completed.cancelled()
                    else completed.exception()
                )
            )
        raise
    with holder_lock:
        holder.pop("lease", None)
    return lease


async def _run_gateway_open(open_fn, *args):
    """Close a late upstream connection if its awaiting request is cancelled."""

    cancel_event = threading.Event()
    holder_lock = threading.Lock()
    holder: dict[str, object] = {}

    def open_upstream():
        result = open_fn(*args)
        with holder_lock:
            if cancel_event.is_set():
                result[0].close()
                raise GenerationSlotCancelled(
                    "Gateway open completed after its request was cancelled."
                )
            holder["result"] = result
        return result

    future = asyncio.get_running_loop().run_in_executor(None, open_upstream)
    try:
        result = await future
    except BaseException:
        cancel_event.set()
        with holder_lock:
            held = holder.pop("result", None)
        if held is not None:
            held[0].close()
        raise
    with holder_lock:
        holder.pop("result", None)
    return result


def _open_gateway_upstream(
    url: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float,
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("Tokenity gateway upstream must be a local HTTP URL.")
    connection = http.client.HTTPConnection(
        parsed.hostname,
        parsed.port or 80,
        timeout=timeout,
    )
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"
    try:
        connection.request("POST", path, body=body, headers=headers)
        return connection, connection.getresponse()
    except Exception:
        connection.close()
        raise


def _gateway_body_chunks(
    connection: http.client.HTTPConnection,
    upstream: http.client.HTTPResponse,
    *,
    on_close,
    read_lock: threading.Lock | None = None,
):
    del connection
    try:
        while True:
            # read1 returns currently available bytes instead of waiting to fill
            # a large buffer, preserving the runtime's SSE chunk cadence.
            if read_lock is None:
                chunk = upstream.read1(64 * 1024)
            else:
                with read_lock:
                    chunk = upstream.read1(64 * 1024)
            if not chunk:
                break
            yield chunk
    finally:
        try:
            on_close()
        except InstanceConflict:
            pass


def _drain_gateway_response(
    upstream: http.client.HTTPResponse,
    read_lock: threading.Lock,
) -> None:
    while True:
        with read_lock:
            chunk = upstream.read1(64 * 1024)
        if not chunk:
            return


def _routing_revision(value: object) -> str:
    return str(value) if isinstance(value, str) and value else "unversioned"


def _infer_gateway_model_profile(
    model_id: str,
    revision: str,
    resolved_path: str,
) -> ModelCapabilityProfile:
    config = _read_model_config(Path(resolved_path) / "config.json")
    identity = f"{model_id} {resolved_path}".lower()
    context_length = _first_positive_int(
        config,
        "max_position_embeddings",
        "context_length",
        "model_max_length",
        default=32_768,
    )
    max_output_length = _first_positive_int(
        config,
        "max_output_length",
        "max_new_tokens",
        default=min(8_192, context_length),
    )
    tags = {"general"}
    if any(marker in identity for marker in ("coder", "code", "devstral")):
        tags.add("coding")
    if any(marker in identity for marker in ("reason", "thinking", "r1", "qwq")):
        tags.add("reasoning")
    if any(marker in identity for marker in ("tool", "function")):
        tags.add("tool-use")
    if any(marker in identity for marker in ("fast", "mini", "small", "flash")):
        tags.add("fast-chat")
    if context_length >= 65_536 or any(
        marker in identity for marker in ("long", "128k", "256k", "1m")
    ):
        tags.add("long-context")

    parameter_scales_billions = [
        float(match)
        for match in re.findall(
            r"(?:^|[^0-9])(\d+(?:\.\d+)?)b(?:[^a-z0-9]|$)",
            identity,
        )
    ]
    largest_parameter_scale = (
        max(parameter_scales_billions) if parameter_scales_billions else None
    )
    is_large = any(
        marker in identity
        for marker in ("large", "strong", "70b", "72b", "122b", "235b", "405b")
    ) or bool(largest_parameter_scale is not None and largest_parameter_scale >= 70)
    is_fast = "fast-chat" in tags or any(
        marker in identity for marker in ("1b", "3b", "7b", "8b")
    ) or bool(largest_parameter_scale is not None and largest_parameter_scale <= 8)
    base_quality = 0.92 if is_large else 0.72 if is_fast else 0.82
    warm_ttft_p95_ms = 1_200.0 if is_large else 120.0 if is_fast else 450.0
    task_quality = {
        "general": base_quality,
        "fast-chat": max(0.65, base_quality - (0.08 if is_large else 0.0)),
        "coding": 0.96 if "coding" in tags else max(0.5, base_quality - 0.12),
        "reasoning": 0.96 if "reasoning" in tags else max(0.5, base_quality - 0.12),
        "long-context": 0.94 if "long-context" in tags else max(0.45, base_quality - 0.18),
        "tool-use": 0.94 if "tool-use" in tags else max(0.45, base_quality - 0.18),
    }
    supports_thinking = (
        "reasoning" in tags
        or "qwen3" in identity
        or "glm" in identity
    )
    supports_tools = (
        "tool-use" in tags
        or "qwen" in identity
        or bool(config.get("supports_tools"))
    )
    supports_json = supports_tools or "json" in identity or bool(config.get("supports_json"))
    sampling_defaults: dict[str, object] = {"temperature": 0.7, "top_p": 0.9}
    if "qwen3" in identity:
        sampling_defaults = {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
        }
    elif "coding" in tags:
        sampling_defaults["temperature"] = 0.2
    elif "reasoning" in tags:
        sampling_defaults["temperature"] = 0.6
    configured_template_defaults = config.get("tokenity_chat_template_defaults")
    if isinstance(configured_template_defaults, dict):
        chat_template_defaults = dict(configured_template_defaults)
    elif "qwen3" in identity:
        # Qwen 3.x can spend the entire (often very large) output budget in
        # reasoning when the private switch is left to the tokenizer default.
        # Keep normal chat finite; callers that want reasoning can still
        # explicitly send enable_thinking=true.
        chat_template_defaults = {"enable_thinking": False}
    else:
        # GLM and other reasoning families keep their own template behavior;
        # never inject Qwen's private enable_thinking switch into them.
        chat_template_defaults = {}
    configured_runtime_parameters = config.get(
        "tokenity_supported_runtime_parameters"
    )
    if isinstance(configured_runtime_parameters, list):
        supported_runtime_parameters = {
            str(value)
            for value in configured_runtime_parameters
            if isinstance(value, str) and value
        }
    else:
        supported_runtime_parameters = set(COMMON_CHAT_RUNTIME_PARAMETERS)
        if supports_tools:
            supported_runtime_parameters.update(
                {"parallel_tool_calls", "tool_choice", "tools"}
            )
        if supports_json:
            supported_runtime_parameters.add("response_format")
        if "qwen3" in identity or chat_template_defaults:
            supported_runtime_parameters.add("chat_template_kwargs")
    configured_template_parameters = config.get(
        "tokenity_supported_chat_template_parameters"
    )
    if isinstance(configured_template_parameters, list):
        supported_template_parameters = {
            str(value)
            for value in configured_template_parameters
            if isinstance(value, str) and value
        }
    else:
        supported_template_parameters = {
            str(key) for key in chat_template_defaults
        }
    return ModelCapabilityProfile(
        model_id=model_id,
        revision=revision,
        tokenizer=None,
        context_length=context_length,
        max_output_length=max_output_length,
        task_tags=frozenset(tags),
        task_quality=task_quality,
        supports_tools=supports_tools,
        supports_json=supports_json,
        supports_thinking=supports_thinking,
        allow_auto=bool(config.get("tokenity_allow_auto", True)),
        user_priority=float(config.get("tokenity_user_priority") or 0.0),
        privacy_tier=int(config.get("tokenity_privacy_tier") or 0),
        sampling_defaults=sampling_defaults,
        chat_template_defaults=chat_template_defaults,
        supported_runtime_parameters=frozenset(supported_runtime_parameters),
        supported_chat_template_parameters=frozenset(
            supported_template_parameters
        ),
        warm_ttft_p50_ms=warm_ttft_p95_ms / 1.5,
        warm_ttft_p95_ms=warm_ttft_p95_ms,
    )


def _first_positive_int(
    values: dict[str, object],
    *keys: str,
    default: int,
) -> int:
    candidates = [values]
    candidates.extend(
        nested
        for nested_key in ("text_config", "language_config")
        if isinstance((nested := values.get(nested_key)), dict)
    )
    for candidate in candidates:
        for key in keys:
            value = candidate.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return default


def _derive_gateway_route_context(
    payload: dict[str, object],
    constraints: dict[str, object],
    *,
    session_id: str | None,
    sticky_model_id: str | None,
    sticky_revision: str | None,
    locked_model_id: str | None,
    locked_revision: str | None,
) -> tuple[RouteContext, dict[str, object]]:
    messages = payload.get("messages")
    message_list = messages if isinstance(messages, list) else []
    all_text_parts: list[str] = []
    last_user_text = ""
    modality = str(constraints.get("modality") or "text")
    for message in message_list:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        text, has_image = _gateway_message_text(content)
        if text:
            all_text_parts.append(text)
        if has_image:
            modality = "image"
        if message.get("role") == "user":
            last_user_text = text
    input_tokens = max(
        0,
        int(constraints.get("input_tokens") or 0),
        sum(max(1, (len(text) + 3) // 4) for text in all_text_parts),
    )
    requested_output_tokens = payload.get("max_tokens")
    if not isinstance(requested_output_tokens, int):
        requested_output_tokens = payload.get("max_completion_tokens")
    if not isinstance(requested_output_tokens, int):
        requested_output_tokens = int(constraints.get("requested_output_tokens") or 0)
    lowered = last_user_text.lower()
    contains_code = bool(
        re.search(
            r"```|\b(?:python|swift|rust|javascript|typescript|java|c\+\+|function|class|def)\b",
            lowered,
        )
    )
    has_compiler_error = bool(
        re.search(r"traceback|compiler error|build failed|syntaxerror|typeerror|exception:", lowered)
    )
    has_file_diff = bool(re.search(r"(^|\n)(?:diff --git|@@ |\+\+\+ |--- )", last_user_text))
    multi_step_reasoning = bool(
        re.search(
            r"\b(?:prove|derive|step by step|reasoning|theorem|calculate)\b|证明|推导|逐步|多步",
            lowered,
        )
    )
    is_translation = bool(re.search(r"\btranslat(?:e|ion)\b|翻译|译成", lowered))
    is_rewrite = bool(re.search(r"\brewrite\b|改写|润色", lowered))
    long_document = input_tokens >= int(constraints.get("long_document_tokens") or 8_192)
    tools = payload.get("tools")
    response_format = payload.get("response_format")
    chat_template_kwargs = payload.get("chat_template_kwargs")
    requires_tools = bool(tools) or bool(constraints.get("requires_tools"))
    requires_json = (
        isinstance(response_format, dict)
        and response_format.get("type") in {"json_object", "json_schema"}
    ) or bool(constraints.get("requires_json"))
    requires_thinking = (
        isinstance(chat_template_kwargs, dict)
        and chat_template_kwargs.get("enable_thinking") is True
    ) or bool(constraints.get("requires_thinking"))
    language = "zh" if re.search(r"[\u3400-\u9fff]", last_user_text) else "en"
    allowed_raw = constraints.get("allowed_model_ids")
    allowed_model_ids = {
        str(value)
        for value in allowed_raw
        if isinstance(value, str) and value
    } if isinstance(allowed_raw, list) else set()
    if locked_model_id:
        if allowed_model_ids and locked_model_id not in allowed_model_ids:
            allowed_model_ids = {"__tokenity_no_eligible_locked_model__"}
        else:
            allowed_model_ids = {locked_model_id}
    topic_changed = bool(constraints.get("topic_changed")) and not bool(locked_model_id)
    category = constraints.get("category")
    if not isinstance(category, str):
        category = None
    context = RouteContext(
        category=category,
        input_tokens=input_tokens,
        requested_output_tokens=max(0, requested_output_tokens),
        language=language,
        requires_tools=requires_tools,
        requires_json=requires_json,
        requires_thinking=requires_thinking,
        modality=modality,
        allowed_model_ids=frozenset(allowed_model_ids),
        minimum_privacy_tier=max(0, int(constraints.get("minimum_privacy_tier") or 0)),
        runtime_parameters=frozenset(
            str(key)
            for key in payload
            if key not in TOKENITY_ROUTE_FIELDS and key not in {"model", "messages"}
        ),
        chat_template_parameters=frozenset(
            str(key)
            for key in chat_template_kwargs
        )
        if isinstance(chat_template_kwargs, dict)
        else frozenset(),
        session_id=session_id,
        sticky_model_id=locked_model_id or sticky_model_id,
        sticky_revision=locked_revision if locked_model_id else sticky_revision,
        topic_changed=topic_changed,
        contains_code=contains_code,
        has_compiler_error=has_compiler_error,
        has_file_diff=has_file_diff,
        multi_step_reasoning=multi_step_reasoning,
        long_document=long_document,
        is_translation=is_translation,
        is_rewrite=is_rewrite,
    )
    features = {
        "input_tokens": input_tokens,
        "requested_output_tokens": max(0, requested_output_tokens),
        "language": language,
        "requires_tools": requires_tools,
        "requires_json": requires_json,
        "requires_thinking": requires_thinking,
        "modality": modality,
        "contains_code": contains_code,
        "has_compiler_error": has_compiler_error,
        "has_file_diff": has_file_diff,
        "multi_step_reasoning": multi_step_reasoning,
        "long_document": long_document,
        "is_translation": is_translation,
        "is_rewrite": is_rewrite,
    }
    return context, features


def _gateway_message_text(content: object) -> tuple[str, bool]:
    if isinstance(content, str):
        return content, False
    if not isinstance(content, list):
        return "", False
    texts: list[str] = []
    has_image = False
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"image", "image_url", "input_image"}:
            has_image = True
        text = item.get("text")
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts), has_image


def _default_gateway_model_id(profiles: list[ModelCapabilityProfile]) -> str | None:
    if not profiles:
        return None
    return max(
        profiles,
        key=lambda profile: (
            profile.user_priority,
            profile.quality_for("general"),
            profile.context_length,
            profile.model_id,
        ),
    ).model_id


def _gateway_upstream_payload(
    payload: dict[str, object],
    selected,
    profile: ModelCapabilityProfile,
) -> dict[str, object]:
    upstream = {
        key: value
        for key, value in payload.items()
        if key not in TOKENITY_ROUTE_FIELDS
    }
    upstream["model"] = selected.requested_model_id
    allowed_template_parameters = set(
        profile.supported_chat_template_parameters
    )
    allowed_template_parameters.update(
        str(key) for key in profile.chat_template_defaults
    )
    existing_template = upstream.get("chat_template_kwargs")
    if isinstance(existing_template, dict) and allowed_template_parameters:
        filtered_template = {
            str(key): value
            for key, value in existing_template.items()
            if str(key) in allowed_template_parameters
        }
        if filtered_template:
            upstream["chat_template_kwargs"] = filtered_template
        else:
            upstream.pop("chat_template_kwargs", None)
    else:
        upstream.pop("chat_template_kwargs", None)
    for key, value in profile.sampling_defaults.items():
        upstream.setdefault(str(key), value)
    if profile.chat_template_defaults:
        existing = upstream.get("chat_template_kwargs")
        explicit = dict(existing) if isinstance(existing, dict) else {}
        merged = dict(profile.chat_template_defaults)
        merged.update(explicit)
        upstream["chat_template_kwargs"] = merged
    return upstream


def scan_models(root: Path) -> list[dict[str, object]]:
    if not root.exists() or not root.is_dir():
        return []
    models: list[dict[str, object]] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if child.name.startswith("."):
            continue
        model_path = child if child.is_dir() else _resolve_macos_alias(child)
        if model_path is None or not model_path.is_dir():
            continue
        config = _read_model_config(model_path / "config.json")
        safetensors = list(model_path.glob("*.safetensors"))
        gguf_files = list(model_path.glob("*.gguf"))
        markers = {
            "config": (model_path / "config.json").exists(),
            "tokenizer": (model_path / "tokenizer.json").exists() or (model_path / "tokenizer.model").exists(),
            "safetensors": bool(safetensors),
            "gguf": bool(gguf_files),
        }
        if any(markers.values()):
            architecture = None
            architectures = config.get("architectures")
            if isinstance(architectures, list) and architectures:
                architecture = str(architectures[0])
            elif config.get("model_type"):
                architecture = str(config["model_type"])
            models.append(
                {
                    "id": child.name,
                    "path": str(model_path),
                    "markers": markers,
                    "format": "GGUF" if gguf_files else "MLX" if safetensors else "Transformers",
                    "quantization": _quantization_description(config, model_path),
                    "size_bytes": _directory_size(model_path),
                    "architecture": architecture,
                    "shard_count": len(gguf_files) + len(safetensors),
                    "revision": _model_revision(str(model_path)),
                    "native_mtp": scan_native_mtp_capability(model_path).to_dict(),
                    **model_usage_metadata(config),
                }
            )
    return models


def _resolve_macos_alias(path: Path) -> Path | None:
    if sys.platform != "darwin" or not path.is_file():
        return None
    try:
        with path.open("rb") as alias_file:
            if alias_file.read(4) != b"book":
                return None
        core_foundation = ctypes.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        pointer = ctypes.c_void_p
        core_foundation.CFURLCreateFromFileSystemRepresentation.argtypes = [
            pointer,
            ctypes.c_char_p,
            ctypes.c_long,
            ctypes.c_bool,
        ]
        core_foundation.CFURLCreateFromFileSystemRepresentation.restype = pointer
        core_foundation.CFURLCreateBookmarkDataFromFile.argtypes = [
            pointer,
            pointer,
            ctypes.POINTER(pointer),
        ]
        core_foundation.CFURLCreateBookmarkDataFromFile.restype = pointer
        core_foundation.CFURLCreateByResolvingBookmarkData.argtypes = [
            pointer,
            pointer,
            ctypes.c_ulong,
            pointer,
            pointer,
            ctypes.POINTER(ctypes.c_bool),
            ctypes.POINTER(pointer),
        ]
        core_foundation.CFURLCreateByResolvingBookmarkData.restype = pointer
        core_foundation.CFURLGetFileSystemRepresentation.argtypes = [
            pointer,
            ctypes.c_bool,
            ctypes.c_char_p,
            ctypes.c_long,
        ]
        core_foundation.CFURLGetFileSystemRepresentation.restype = ctypes.c_bool
        core_foundation.CFRelease.argtypes = [pointer]
    except (AttributeError, OSError):
        return None

    references: list[int] = []
    try:
        encoded = os.fsencode(path)
        source = core_foundation.CFURLCreateFromFileSystemRepresentation(
            None, encoded, len(encoded), False
        )
        if not source:
            return None
        references.append(source)
        error = pointer()
        bookmark = core_foundation.CFURLCreateBookmarkDataFromFile(
            None, source, ctypes.byref(error)
        )
        if error.value:
            references.append(error.value)
        if not bookmark:
            return None
        references.append(bookmark)
        stale = ctypes.c_bool()
        error = pointer()
        resolved = core_foundation.CFURLCreateByResolvingBookmarkData(
            None,
            bookmark,
            (1 << 8) | (1 << 9),
            None,
            None,
            ctypes.byref(stale),
            ctypes.byref(error),
        )
        if error.value:
            references.append(error.value)
        if not resolved:
            return None
        references.append(resolved)
        buffer = ctypes.create_string_buffer(4096)
        if not core_foundation.CFURLGetFileSystemRepresentation(
            resolved, True, buffer, len(buffer)
        ):
            return None
        return Path(os.fsdecode(buffer.value))
    finally:
        for reference in reversed(references):
            core_foundation.CFRelease(reference)


def _read_model_config(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _stable_connection_mode_for_model(
    model: str,
    requested: ConnectionMode,
) -> ConnectionMode:
    del model
    if requested != ConnectionMode.JACCL:
        return requested
    # The direct two-rank JACCL mesh can leave receives posted after a
    # completed request. A later request, or a second resident communicator on
    # the same RDMA device, then fails in ibv_post_recv with ENOMEM (-12).
    # The ring implementation drains its work requests and has passed the
    # repeated-request and multi-resident hardware matrix, so make it the
    # stable implementation for every model.
    return ConnectionMode.JACCL_RING


def _requires_coordinated_rank_stop(
    connection_mode: str,
) -> bool:
    return str(connection_mode).lower() in {"jaccl", "jaccl-ring"}


def _request_rdma_link_reset(timeout: float = 12.0) -> bool:
    interface = os.environ.get("TOKENITY_TB_INTERFACE", "").strip()
    raw_path = os.environ.get("TOKENITY_RDMA_RESET_REQUEST_PATH", "").strip()
    if not interface or not raw_path:
        return False

    request_path = Path(raw_path)
    try:
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(f"{time.time()}\n", encoding="utf-8")
    except OSError:
        logging.exception("Could not request an RDMA link reset on %s.", interface)
        return False

    deadline = time.monotonic() + timeout
    while request_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    if request_path.exists():
        logging.warning("Timed out waiting for the RDMA link reset on %s.", interface)
        return False
    logging.info("Completed the RDMA link reset on %s.", interface)
    return True


def _quantization_description(config: dict[str, object], model_path: Path) -> str | None:
    raw = config.get("quantization") or config.get("quantization_config")
    if isinstance(raw, dict):
        bits = raw.get("bits") or raw.get("nbits")
        group_size = raw.get("group_size")
        if isinstance(bits, (int, float)):
            result = f"{int(bits)}-bit"
            if isinstance(group_size, (int, float)):
                result += f" · group {int(group_size)}"
            return result

    for file in model_path.glob("*.gguf"):
        match = re.search(r"\b(Q\d(?:_[A-Z0-9]+)*)\b", file.name.upper())
        if match:
            return match.group(1)
    return None


def _directory_size(path: Path) -> int:
    total = 0
    for directory, _, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += (Path(directory) / name).stat().st_size
            except OSError:
                continue
    return total


def _request_nodes(request: StartRequest) -> list[ClusterNode]:
    if request.nodes:
        return [node.to_cluster_node() for node in request.nodes]
    return [
        ClusterNode(
            id="local",
            agent_url=f"http://127.0.0.1:{_node_agent_port()}",
            lan_ip="127.0.0.1",
        )
    ]


def _h3_request_nodes(request: H3VideoStartRequest) -> list[ClusterNode]:
    if request.nodes:
        return [node.to_cluster_node() for node in request.nodes]
    return [
        ClusterNode(
            id="local",
            agent_url=f"http://127.0.0.1:{_node_agent_port()}",
            lan_ip="127.0.0.1",
        )
    ]


def _agent_url(node: ClusterNode, *, remote: bool = False) -> str:
    if node.agent_url:
        url = node.agent_url.rstrip("/")
    elif node.lan_ip:
        url = f"http://{node.lan_ip}:{_node_agent_port()}"
    else:
        raise HostfileError(f"{node.id}: missing Node Agent URL.")
    parsed = urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise HostfileError(f"{node.id}: Agent URL must use http:// with a valid host.")
    if parsed.username or parsed.password:
        raise HostfileError(f"{node.id}: Agent URL must not contain a username or password.")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise HostfileError(f"{node.id}: Agent URL must be an HTTP origin without a path, query, or fragment.")
    if remote and is_loopback_host(parsed.hostname):
        raise HostfileError(f"{node.id}: remote Node Agent URL must not use loopback.")
    return url


def _node_agent_port() -> int:
    value = os.environ.get("TOKENITY_NODE_AGENT_PORT", "9100")
    try:
        port = int(value)
    except ValueError as exc:
        raise HostfileError("TOKENITY_NODE_AGENT_PORT must be an integer.") from exc
    if not 1 <= port <= 65_535:
        raise HostfileError("TOKENITY_NODE_AGENT_PORT must be a valid TCP port.")
    return port


def _http_rank_requests(
    request: StartRequest,
    nodes: list[ClusterNode],
    *,
    model_revision: str | None = None,
) -> list[RankStartRequest]:
    if not nodes:
        raise HostfileError("At least one node is required.")
    for rank, node in enumerate(nodes):
        _agent_url(node, remote=rank > 0)

    world_size = len(nodes)
    ring_hosts: list[list[str]] = []
    rdma_matrix: list[list[str | None]] = []
    coordinator_ip: str | None = None

    hostfile: list[dict[str, object]] = []
    if world_size == 1:
        # Single-node inference is a first-class local path. It deliberately
        # creates no hostfile, collective group, remote-rank request, or RDMA
        # environment.
        pass
    else:
        hostfile = build_hostfile(nodes, request.connection_mode)
    if world_size > 1 and request.connection_mode == ConnectionMode.RING:
        port = request.starting_port
        for node, row in zip(nodes, hostfile):
            if not row["ips"]:
                raise HostfileError(f"{node.id}: missing standard-network IP.")
            ring_hosts.append([f"{row['ips'][0]}:{port}"])
            port += 1
    elif world_size > 1:
        if not hostfile[0]["ips"]:
            raise HostfileError("Rank 0 needs a Thunderbolt/RDMA coordinator IP.")
        coordinator_ip = str(hostfile[0]["ips"][0])
        rdma_matrix = [list(row["rdma"]) for row in hostfile]

    cluster_id = request.instance_id or uuid.uuid4().hex
    operation_id = request.operation_id or uuid.uuid4().hex
    return [
        RankStartRequest(
            cluster_id=cluster_id,
            model=request.model,
            rank=rank,
            world_size=world_size,
            coordinator=rank == 0,
            connection_mode=request.connection_mode,
            python=request.python,
            coordinator_ip=coordinator_ip,
            starting_port=request.starting_port,
            ring_hosts=ring_hosts,
            rdma_matrix=rdma_matrix,
            host=request.host,
            port=request.port,
            api_identifier=request.api_identifier,
            max_tokens=request.max_tokens,
            prompt_cache_size=request.prompt_cache_size,
            prefill_step_size=request.prefill_step_size,
            decode_concurrency=request.decode_concurrency,
            prompt_concurrency=request.prompt_concurrency,
            trust_remote_code=request.trust_remote_code,
            lease_seconds=request.lease_seconds,
            native_mtp=request.native_mtp,
            instance_id=request.instance_id or cluster_id,
            operation_id=operation_id,
            model_revision=model_revision,
            tokenity_code_revision=_tokenity_code_revision(),
            memory_reservation_bytes=request.memory_reservation_bytes,
        )
        for rank in range(world_size)
    ]


def _h3_rank_requests(
    request: H3VideoStartRequest,
    nodes: list[ClusterNode],
    *,
    model_revision: str | None = None,
) -> list[H3VideoRankStartRequest]:
    if not nodes:
        raise HostfileError("At least one node is required.")
    if len(nodes) > 2:
        raise HostfileError("MiniMax H3 Tensor Parallel currently supports exactly two ranks.")
    for rank, node in enumerate(nodes):
        _agent_url(node, remote=rank > 0)

    world_size = len(nodes)
    if world_size > 1 and request.connection_mode == ConnectionMode.RING:
        raise HostfileError(
            "MiniMax H3 TP2 requires jaccl-ring over RDMA; plain TCP ring is unsupported."
        )
    ring_hosts: list[list[str]] = []
    rdma_matrix: list[list[str | None]] = []
    coordinator_ip: str | None = None
    if world_size > 1:
        hostfile = build_hostfile(nodes, request.connection_mode)
        if not hostfile[0]["ips"]:
            raise HostfileError("Rank 0 needs a Thunderbolt/RDMA coordinator IP.")
        coordinator_ip = str(hostfile[0]["ips"][0])
        rdma_matrix = [list(row["rdma"]) for row in hostfile]

    cluster_id = request.instance_id or uuid.uuid4().hex
    operation_id = request.operation_id or uuid.uuid4().hex
    code_revision = _tokenity_code_revision()
    runtime_contract = h3_runtime_fingerprint(
        binary=request.binary,
        model=request.model,
        optimization_profile=request.optimization_profile,
        tokenity_code_revision=code_revision,
    )
    return [
        H3VideoRankStartRequest(
            cluster_id=cluster_id,
            model=request.model,
            binary=request.binary,
            rank=rank,
            world_size=world_size,
            coordinator=rank == 0,
            connection_mode=(
                request.connection_mode if world_size > 1 else ConnectionMode.RING
            ),
            python=request.python,
            coordinator_ip=coordinator_ip,
            starting_port=request.starting_port,
            ring_hosts=ring_hosts,
            rdma_matrix=rdma_matrix,
            host=request.host,
            port=request.port,
            api_identifier=request.api_identifier,
            lease_seconds=request.lease_seconds,
            instance_id=request.instance_id or cluster_id,
            operation_id=operation_id,
            model_revision=model_revision,
            tokenity_code_revision=code_revision,
            memory_reservation_bytes=request.memory_reservation_bytes,
            optimization_profile=request.optimization_profile,
            minimum_free_disk_bytes=request.minimum_free_disk_bytes,
            runtime_contract_sha256=str(runtime_contract["contract_sha256"]),
            dry_run=request.dry_run,
        )
        for rank in range(world_size)
    ]


def _h3_launch_plan(
    nodes: list[ClusterNode],
    rank_requests: list[H3VideoRankStartRequest],
) -> dict[str, object]:
    distributed = len(rank_requests) > 1
    return {
        "role": "minimax-h3-video",
        "worker_role": "minimax-h3-video-rank" if distributed else None,
        "backend": "native mlx-serve MiniMax H3",
        "execution_mode": "tp2" if distributed else "single",
        "experimental": distributed,
        "transport": "http-control+jaccl-rdma" if distributed else "local-process",
        "required_backend_protocol": (
            H3_DISTRIBUTED_PROTOCOL_VERSION if distributed else None
        ),
        "ranks": [
            {
                "rank": rank_request.rank,
                "agent_url": _agent_url(node),
                "endpoint": (
                    None
                    if rank_request.rank == 0
                    else "/v1/node/start-minimax-h3-video-rank"
                ),
                "command": _h3_rank_command(rank_request),
                "connection_mode": (
                    rank_request.connection_mode.value if distributed else "single"
                ),
            }
            for node, rank_request in zip(nodes, rank_requests)
        ],
        "warnings": (
            [
                "Launch is capability-gated until the native binary advertises "
                "Tokenity MiniMax H3 distributed protocol v1."
            ]
            if distributed
            else []
        ),
    }


def _h3_rank_command_and_environment(
    request: H3VideoRankStartRequest,
    *,
    actual_world_size: int,
) -> tuple[list[str], dict[str, str]]:
    if request.rank >= actual_world_size:
        raise HostfileError("Rank must be smaller than world size.")
    env = {
        "PATH": _distributed_path(request.python),
        "PYTHONPATH": _distributed_code_root(),
        "MLX_METAL_FAST_SYNCH": os.environ.get("MLX_METAL_FAST_SYNCH", "1"),
        "TOKENITY_INSTANCE_ID": request.instance_id or request.cluster_id,
        "TOKENITY_OPERATION_ID": request.operation_id or request.cluster_id,
        "TOKENITY_CLUSTER_ID": request.cluster_id,
        "TOKENITY_CONNECTION_MODE": (
            "single" if actual_world_size == 1 else request.connection_mode.value
        ),
        "TOKENITY_H3_OPTIMIZATION_PROFILE": request.optimization_profile,
    }
    env.update(h3_optimization_environment(request.optimization_profile))
    if request.model_revision:
        env["TOKENITY_MODEL_REVISION"] = request.model_revision
    if actual_world_size == 1:
        return _h3_rank_command(request, actual_world_size=1), env

    if request.connection_mode == ConnectionMode.RING:
        raise HostfileError(
            "MiniMax H3 TP2 requires jaccl-ring over RDMA; plain TCP ring is unsupported."
        )

    env["MLX_RANK"] = str(request.rank)
    if not request.coordinator_ip:
        raise HostfileError("JACCL coordinator IP is required.")
    if len(request.rdma_matrix) != actual_world_size or any(
        len(row) != actual_world_size for row in request.rdma_matrix
    ):
        raise HostfileError("RDMA matrix dimensions must equal world size.")
    env["MLX_JACCL_COORDINATOR"] = (
        f"{request.coordinator_ip}:{request.starting_port}"
    )
    env["MLX_IBV_DEVICES"] = str(
        _write_rank_environment_file(
            request.cluster_id,
            "h3-rdma-devices",
            json.dumps(request.rdma_matrix),
        )
    )
    if request.connection_mode == ConnectionMode.JACCL_RING:
        env["MLX_JACCL_RING"] = "1"
    return _h3_rank_command(request, actual_world_size=actual_world_size), env


def _h3_rank_command(
    request: H3VideoRankStartRequest,
    *,
    actual_world_size: int | None = None,
) -> list[str]:
    world_size = actual_world_size or request.world_size
    return [
        request.python,
        "-m",
        "tokenity",
        "minimax-h3-video",
        "serve",
        "--binary",
        request.binary,
        "--model",
        request.model,
        "--host",
        request.host,
        "--port",
        str(request.port),
        "--rank",
        str(request.rank),
        "--world-size",
        str(world_size),
    ]


def _http_launch_plan(
    nodes: list[ClusterNode],
    rank_requests: list[RankStartRequest],
) -> dict[str, object]:
    if len(rank_requests) == 1:
        rank_request = rank_requests[0]
        return {
            "role": "single-node-openai",
            "backend": "Tokenity single-node OpenAI server",
            "execution_mode": "single",
            "experimental": False,
            "transport": "local-process",
            "ranks": [
                {
                    "rank": 0,
                    "agent_url": _agent_url(nodes[0]),
                    "endpoint": None,
                    "command": _rank_process_command(rank_request),
                    "connection_mode": "single",
                }
            ],
            "warnings": [
                "The selected Node Agent starts one local runtime; no hostfile, remote rank, SSH, JACCL, or MLX distributed group is used."
            ],
        }
    return {
        "role": "distributed-openai",
        "backend": "Tokenity distributed OpenAI server",
        "experimental": False,
        "transport": "http",
        "ranks": [
            {
                "rank": rank_request.rank,
                "agent_url": _agent_url(node),
                "endpoint": None if rank_request.rank == 0 else "/v1/node/start-distributed-rank",
                "command": _rank_process_command(rank_request),
                "connection_mode": rank_request.connection_mode.value,
            }
            for node, rank_request in zip(nodes, rank_requests)
        ],
        "warnings": [
            "Rank lifecycle uses typed Node Agent HTTP requests; no SSH credentials are required."
        ],
    }


def _rank_command_and_environment(request: RankStartRequest) -> tuple[list[str], dict[str, str]]:
    if request.rank >= request.world_size:
        raise HostfileError("Rank must be smaller than world size.")

    post_load_barrier_default = (
        "1"
        if request.world_size > 1
        and request.connection_mode in {ConnectionMode.JACCL, ConnectionMode.JACCL_RING}
        else "0"
    )
    env = {
        "PATH": _distributed_path(request.python),
        "PYTHONPATH": _distributed_code_root(),
        "MLX_METAL_FAST_SYNCH": os.environ.get("MLX_METAL_FAST_SYNCH", "1"),
        "TOKENITY_MLX_LOAD_POLICY": os.environ.get("TOKENITY_MLX_LOAD_POLICY", "adaptive"),
        "TOKENITY_MLX_LOAD_ADAPTIVE_MAX_LEAVES": os.environ.get(
            "TOKENITY_MLX_LOAD_ADAPTIVE_MAX_LEAVES", "64"
        ),
        "TOKENITY_MLX_LOAD_ADAPTIVE_TARGET_BYTES": os.environ.get(
            "TOKENITY_MLX_LOAD_ADAPTIVE_TARGET_BYTES", str(256 * 1024 * 1024)
        ),
        "TOKENITY_MLX_LOAD_EVAL_CHUNK_SIZE": os.environ.get("TOKENITY_MLX_LOAD_EVAL_CHUNK_SIZE", "1"),
        "TOKENITY_MLX_LOAD_EVAL_LOG_INTERVAL": os.environ.get("TOKENITY_MLX_LOAD_EVAL_LOG_INTERVAL", "100"),
        "TOKENITY_MLX_LOAD_EVAL_SLEEP_SECONDS": os.environ.get("TOKENITY_MLX_LOAD_EVAL_SLEEP_SECONDS", "0.05"),
        "TOKENITY_MLX_LOAD_POST_BARRIER": os.environ.get(
            "TOKENITY_MLX_LOAD_POST_BARRIER",
            post_load_barrier_default,
        ),
        "TOKENITY_MLX_DISTRIBUTED_INIT_RANK0_DELAY_SECONDS": os.environ.get(
            "TOKENITY_MLX_DISTRIBUTED_INIT_RANK0_DELAY_SECONDS", "0"
        ),
        "TOKENITY_INSTANCE_ID": request.instance_id or request.cluster_id,
        "TOKENITY_OPERATION_ID": request.operation_id or request.cluster_id,
        "TOKENITY_CLUSTER_ID": request.cluster_id,
        "TOKENITY_CONNECTION_MODE": (
            "single" if request.world_size == 1 else request.connection_mode.value
        ),
        "TOKENITY_STATUS_PATH": str(
            _runtime_status_path(request.instance_id or request.cluster_id, request.rank)
        ),
    }

    if request.model_revision:
        env["TOKENITY_MODEL_REVISION"] = request.model_revision

    if request.world_size == 1:
        return _rank_process_command(request), env

    env["MLX_RANK"] = str(request.rank)

    if request.connection_mode == ConnectionMode.RING:
        if request.world_size > 1 and len(request.ring_hosts) != request.world_size:
            raise HostfileError("Ring host list length must equal world size.")
        if request.world_size > 1:
            env["MLX_HOSTFILE"] = str(
                _write_rank_environment_file(
                    request.cluster_id,
                    "ring-hosts",
                    json.dumps(request.ring_hosts),
                )
            )
    else:
        if not request.coordinator_ip:
            raise HostfileError("JACCL coordinator IP is required.")
        if len(request.rdma_matrix) != request.world_size or any(
            len(row) != request.world_size for row in request.rdma_matrix
        ):
            raise HostfileError("RDMA matrix dimensions must equal world size.")
        env["MLX_JACCL_COORDINATOR"] = f"{request.coordinator_ip}:{request.starting_port}"
        env["MLX_IBV_DEVICES"] = str(
            _write_rank_environment_file(
                request.cluster_id,
                "rdma-devices",
                json.dumps(request.rdma_matrix),
            )
        )
        if request.connection_mode == ConnectionMode.JACCL_RING:
            env["MLX_JACCL_RING"] = "1"

    return _rank_process_command(request), env


def _rank_process_command(request: RankStartRequest) -> list[str]:
    return _rank_command(request)


def _rank_command(request: RankStartRequest) -> list[str]:
    command = [
        request.python,
        "-m",
        "tokenity",
        "distributed-openai",
        "serve",
        "--model",
        request.model,
        "--host",
        request.host,
        "--port",
        str(request.port),
        "--max-tokens",
        str(request.max_tokens),
        "--prompt-cache-size",
        str(request.prompt_cache_size),
        "--prefill-step-size",
        str(request.prefill_step_size),
        "--decode-concurrency",
        str(request.decode_concurrency),
        "--prompt-concurrency",
        str(request.prompt_concurrency),
        "--execution-mode",
        "single" if request.world_size == 1 else "distributed",
        "--native-mtp-mode",
        request.native_mtp.mode,
        "--native-mtp-max-depth",
        str(request.native_mtp.max_depth),
        "--native-mtp-head-placement",
        request.native_mtp.head_placement,
    ]
    if request.api_identifier:
        command.extend(["--api-identifier", request.api_identifier])
    if request.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def _write_rank_environment_file(cluster_id: str, name: str, content: str) -> Path:
    base = Path(tempfile.gettempdir()) / "tokenity-rank-env"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{cluster_id}-{name}.json"
    path.write_text(content, encoding="utf-8")
    return path


def _runtime_status_path(instance_id: str, rank: int) -> Path:
    base = Path(tempfile.gettempdir()) / "tokenity-runtime-status"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{instance_id}-rank-{rank}.json"


def _distributed_path(python: str) -> str:
    existing = os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    items = [str(Path(python).parent), *existing.split(os.pathsep)]
    return os.pathsep.join(dict.fromkeys(item for item in items if item))


def _distributed_code_root() -> str:
    return str(code_root())


def _post_json(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            data = response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            decoded_error = json.loads(body)
        except json.JSONDecodeError:
            decoded_error = None
        detail = decoded_error.get("detail") if isinstance(decoded_error, dict) else body
        raise RuntimeError(f"Node Agent returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Node Agent is unreachable: {exc.reason}") from exc
    decoded = json.loads(data) if data else {}
    if not isinstance(decoded, dict):
        raise RuntimeError("Node Agent returned an invalid JSON response.")
    return decoded


def _get_json(url: str, timeout: float) -> dict[str, object]:
    request = Request(url, method="GET")
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            data = response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Node Agent returned HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Node Agent is unreachable: {exc.reason}") from exc
    decoded = json.loads(data) if data else {}
    if not isinstance(decoded, dict):
        raise RuntimeError("Node Agent returned an invalid JSON response.")
    return decoded


def _read_runtime_status(path: str | None) -> dict[str, object] | None:
    if not path:
        return None
    try:
        decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    updated_at = decoded.get("updated_at")
    stale = not isinstance(updated_at, (int, float)) or time.time() - updated_at > 6.0
    decoded["status_stale"] = stale
    memory = decoded.get("memory")
    if isinstance(memory, dict):
        memory["stale"] = stale
    return decoded


def _wait_for_listening_port(pid: int, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["lsof", "-nP", "-a", "-p", str(pid), f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            return True
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        time.sleep(0.1)
    return False


def _wait_for_rank_connection(pid: int, request: RankStartRequest, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if request.connection_mode == ConnectionMode.RING:
            network_filter = "-iTCP"
        else:
            network_filter = f"-iTCP@{request.coordinator_ip}:{request.starting_port}"
        result = subprocess.run(
            ["lsof", "-nP", "-a", "-p", str(pid), network_filter, "-sTCP:ESTABLISHED"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            return True
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        time.sleep(0.1)
    return False


def _wait_for_h3_health(pid: int, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            payload = _get_json(f"http://127.0.0.1:{port}/health", 1.0)
        except Exception:
            payload = None
        if isinstance(payload, dict) and payload.get("status") == "ok":
            return True
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        time.sleep(0.1)
    return False


def _h3_capabilities_payload(capabilities: object) -> dict[str, object]:
    if isinstance(capabilities, H3BackendCapabilities):
        return capabilities.to_dict()
    to_dict = getattr(capabilities, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, dict):
            return payload
    if isinstance(capabilities, dict):
        return dict(capabilities)
    return {"detail": str(capabilities)}


def _runtime_preflight(
    python: str,
    model: str,
    expected_model_revision: str | None,
) -> list[str]:
    issues: list[str] = []
    try:
        python_matches = Path(python).resolve() == Path(sys.executable).resolve()
    except OSError:
        python_matches = False
    if not python_matches:
        issues.append(
            f"Runtime Python mismatch: requested {python}, Agent uses {sys.executable}."
        )

    mlx_version = _package_version("mlx")
    mlx_lm_version = _package_version("mlx-lm")
    if mlx_version != "0.32.0":
        issues.append(f"mlx 0.32.0 is required; found {mlx_version or 'not installed'}.")
    if mlx_lm_version is None or _version_release(mlx_lm_version) < (0, 31, 3):
        issues.append(f"mlx-lm >= 0.31.3 is required; found {mlx_lm_version or 'not installed'}.")

    model_path = Path(model)
    if not model_path.is_dir():
        issues.append(f"Model directory does not exist: {model}")
    else:
        if not (model_path / "config.json").is_file():
            issues.append("Model config.json is missing.")
        if not (
            (model_path / "tokenizer.json").is_file()
            or (model_path / "tokenizer.model").is_file()
        ):
            issues.append("Model tokenizer files are missing.")
        if not any(model_path.glob("*.safetensors")) and not any(model_path.glob("*.gguf")):
            issues.append("Model weight files are missing.")

    local_revision = _model_revision(model)
    if expected_model_revision and local_revision != expected_model_revision:
        issues.append(
            f"Model revision mismatch: expected {expected_model_revision}, local {local_revision or 'unknown'}."
        )
    return issues


def _version_release(value: str) -> tuple[int, ...]:
    release = value.split("+", 1)[0].split("-", 1)[0]
    result: list[int] = []
    for part in release.split("."):
        match = re.match(r"(\d+)", part)
        if match is None:
            break
        result.append(int(match.group(1)))
    return tuple(result)


def _model_revision(model: str) -> str | None:
    root = Path(model)
    files = [
        root / "config.json",
        root / "model.safetensors.index.json",
        root / "tp2" / "manifest.json",
    ]
    existing = [path for path in files if path.is_file()]
    if not existing:
        return None
    digest = hashlib.sha256()
    for path in existing:
        digest.update(path.name.encode("utf-8"))
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        except OSError:
            return None
    return digest.hexdigest()


def _tokenity_code_revision() -> str | None:
    digest = hashlib.sha256()
    paths = [
        Path(__file__),
        Path(__file__).parents[1] / "serving" / "distributed_openai.py",
        Path(__file__).parents[1] / "serving" / "minimax_h3_video.py",
    ]
    try:
        for path in paths:
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    except OSError:
        return None
    return digest.hexdigest()


def _runtime_identity() -> dict[str, object] | None:
    path = Path(os.environ.get("TOKENITY_RUNTIME_MANIFEST", DEFAULT_RUNTIME_MANIFEST))
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None

    packages = manifest.get("packages")
    payload = manifest.get("payload")
    if not isinstance(packages, dict) or not isinstance(payload, dict):
        return None

    expected_packages = {
        str(name): str(version)
        for name, version in packages.items()
        if isinstance(name, str) and isinstance(version, str)
    }
    observed_packages = {
        name: _package_version(name)
        for name in expected_packages
    }
    compatible = (
        manifest.get("architecture") == platform.machine()
        and manifest.get("python_version") == platform.python_version()
        and observed_packages == expected_packages
    )
    return {
        "runtime_id": manifest.get("runtime_id"),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "payload_sha256": payload.get("tree_sha256"),
        "architecture": manifest.get("architecture"),
        "minimum_macos": manifest.get("minimum_macos"),
        "python_version": manifest.get("python_version"),
        "packages": expected_packages,
        "install_state": "ready" if compatible else "incompatible",
    }


def _estimated_memory_reservation_breakdown(
    model: str,
    world_size: int,
    *,
    max_tokens: int,
    prompt_cache_size: int,
    prefill_step_size: int,
    decode_concurrency: int,
    prompt_concurrency: int,
) -> MemoryReservationBreakdown:
    root = Path(model)
    total = 0
    try:
        for path in root.iterdir():
            if path.is_file() and path.suffix in {".safetensors", ".gguf"}:
                total += path.stat().st_size
    except OSError:
        pass
    ranks = max(1, world_size)
    weights = (total + ranks - 1) // ranks if total else 0

    config: dict[str, object] = {}
    try:
        candidate = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if isinstance(candidate, dict):
            config = candidate
    except (OSError, json.JSONDecodeError):
        pass
    if total == 0 and not config:
        return MemoryReservationBreakdown()
    hidden_size = config.get("hidden_size")
    layer_count = config.get("num_hidden_layers")
    attention_heads = config.get("num_attention_heads")
    kv_heads = config.get("num_key_value_heads", attention_heads)
    head_dim = config.get("head_dim")
    if not isinstance(head_dim, int) and isinstance(hidden_size, int) and isinstance(attention_heads, int):
        head_dim = max(1, hidden_size // max(1, attention_heads))
    if not all(isinstance(value, int) and value > 0 for value in (layer_count, kv_heads, head_dim)):
        # Unknown architectures keep a bounded conservative cache allowance;
        # observed phys_footprint/MLX peak replaces this estimate after warmup.
        kv_cache = max(512 * 1024**2, weights // 20 if weights else 0)
        prompt_cache = max(256 * 1024**2, weights // 40 if weights else 0)
    else:
        bytes_per_token = 2 * int(layer_count) * int(kv_heads) * int(head_dim) * 2
        live_tokens = max_tokens * max(1, decode_concurrency)
        cached_tokens = min(max_tokens, prefill_step_size) * max(1, prompt_cache_size)
        kv_cache = (bytes_per_token * live_tokens * max(1, prompt_concurrency) + ranks - 1) // ranks
        prompt_cache = (bytes_per_token * cached_tokens + ranks - 1) // ranks

    mlx_cache = max(512 * 1024**2, weights // 20 if weights else 0)
    runtime_peak = max(1024**3, weights // 10 if weights else 0)
    os_headroom = max(2 * 1024**3, (weights + kv_cache + prompt_cache) // 20)
    return MemoryReservationBreakdown(
        weights_bytes=weights,
        kv_cache_bytes=kv_cache,
        prompt_cache_bytes=prompt_cache,
        mlx_cache_bytes=mlx_cache,
        runtime_peak_bytes=runtime_peak,
        os_headroom_bytes=os_headroom,
    )


def _estimated_h3_memory_reservation_breakdown(
    model: str,
    *,
    world_size: int,
    rank: int,
) -> MemoryReservationBreakdown:
    root = Path(model)

    def size(relative: str) -> int:
        try:
            return (root / relative).stat().st_size
        except OSError:
            return 0

    text = size("text_encoder.safetensors")
    vaes = size("video_vae.safetensors") + size("audio_vae.safetensors")
    if world_size > 1:
        dit = size(f"tp2/rank-{rank}/transformer.safetensors")
        # Rank 0 uses staged residency: text -> TP DiT -> VAEs. Rank 1 owns
        # only its TP DiT shard and returns before media decoding.
        weights = max(dit, text, vaes) if rank == 0 else dit
    else:
        dit = size("transformer.safetensors")
        weights = max(dit, text, vaes)
    # H3 has no KV cache, but its packed video sequence and VAE staging need a
    # much larger activation allowance than a text decoder. Keep this estimate
    # deliberately conservative until live MLX peak telemetry can replace it.
    runtime_peak = max(16 * 1024**3, weights // 2 if weights else 0)
    mlx_cache = max(8 * 1024**3, weights // 8 if weights else 0)
    os_headroom = max(
        16 * 1024**3,
        (weights + runtime_peak + mlx_cache) // 6,
    )
    return MemoryReservationBreakdown(
        weights_bytes=weights,
        mlx_cache_bytes=mlx_cache,
        runtime_peak_bytes=runtime_peak,
        os_headroom_bytes=os_headroom,
    )


def _allocate_instance_port(preferred: int, ledger: dict[str, object]) -> int:
    reserved = {int(port) for port in dict(ledger.get("ports") or {}).keys()}
    reserved.update(
        int(port)
        for port in dict(ledger.get("quarantined_ports") or {}).keys()
    )
    for port in range(preferred, min(65_536, preferred + 256)):
        if port in reserved:
            continue
        if _tcp_port_available(port):
            return port
    raise HTTPException(status_code=409, detail=f"No free HTTP port near {preferred}.")


def _allocate_collective_port(
    preferred: int,
    world_size: int,
    ledger: dict[str, object],
    *,
    excluded_ports: set[int] | None = None,
) -> int:
    reserved = {int(port) for port in dict(ledger.get("ports") or {}).keys()}
    reserved.update(
        int(port)
        for port in dict(ledger.get("quarantined_ports") or {}).keys()
    )
    reserved.update(excluded_ports or set())
    width = max(1, world_size)
    last_starting_port = min(65_536 - width, preferred + 255)
    for starting_port in range(preferred, last_starting_port + 1):
        ports = tuple(range(starting_port, starting_port + width))
        if any(port in reserved for port in ports):
            continue
        if all(_tcp_port_available(port) for port in ports):
            return starting_port
    raise HTTPException(status_code=409, detail=f"No free collective port range near {preferred}.")


def _tcp_port_available(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("0.0.0.0", port))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _local_ipv4s() -> list[str]:
    ips: set[str] = set()
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except socket.gaierror:
        infos = []
    for info in infos:
        ip = info[4][0]
        if not ip.startswith("127."):
            ips.add(ip)
    return sorted(ips)


def _node_id() -> str:
    return f"{getpass.getuser()}@{socket.gethostname()}"


@lru_cache(maxsize=1)
def _machine_id() -> str:
    """Return a stable pseudonymous identity that survives LAN IP changes."""
    override = os.environ.get("TOKENITY_MACHINE_ID", "").strip()
    if override:
        source = override
    else:
        source = ""
        if sys.platform == "darwin":
            try:
                result = subprocess.run(
                    ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                )
                match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', result.stdout)
                if match:
                    source = match.group(1)
            except (OSError, subprocess.SubprocessError):
                pass
        if not source:
            try:
                source = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
            except OSError:
                pass
        if not source:
            source = f"{platform.system()}|{platform.node()}|{uuid.getnode()}"
    digest_input = f"tokenity-machine|{source}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()[:32]


def _memory_stats() -> dict[str, object]:
    total = _total_memory_bytes()
    vm = _vm_stat_pages()
    if total is not None and vm is not None:
        page_size = int(vm["page_size"])
        pages = vm["pages"]
        free_pages = pages.get("Pages free", 0) + pages.get("Pages speculative", 0)
        free = min(total, max(0, free_pages * page_size))
        physical_used = max(0, total - free)

        def page_bytes(key: str) -> int:
            return max(0, pages.get(key, 0) * page_size)

        wired = page_bytes("Pages wired down")
        compressed = page_bytes("Pages occupied by compressor")
        anonymous = page_bytes("Anonymous pages")
        file_backed = page_bytes("File-backed pages")

        # Model weights are memory-mapped files. macOS keeps their clean pages in
        # RAM after a model exits, but those pages are immediately reclaimable
        # under pressure. Report that cache separately from memory which is still
        # genuinely in use so the UI does not make a stopped model look leaked.
        in_use = min(physical_used, wired + compressed + anonymous)
        reclaimable = min(file_backed, max(0, physical_used - in_use))

        return {
            "total_bytes": total,
            # Match macOS `top` PhysMem: memory which currently occupies
            # physical pages, including reclaimable file-backed model weights.
            "used_bytes": physical_used,
            "free_bytes": free,
            "used_ratio": physical_used / total if total > 0 else None,
            "physical_used_bytes": physical_used,
            "physical_used_ratio": physical_used / total if total > 0 else None,
            "in_use_bytes": in_use,
            "in_use_ratio": in_use / total if total > 0 else None,
            "reclaimable_bytes": reclaimable,
            "wired_bytes": wired,
            "compressed_bytes": compressed,
            "anonymous_bytes": anonymous,
            "active_bytes": page_bytes("Pages active"),
            "inactive_bytes": page_bytes("Pages inactive"),
            "file_backed_bytes": file_backed,
            "pressure_available_ratio": _memory_pressure_available_ratio(),
        }

    # Keep a pressure-based fallback for non-macOS/test environments where
    # vm_stat is unavailable, but never prefer it over physical page counts.
    available_ratio = _memory_pressure_available_ratio()
    if total is not None and available_ratio is not None:
        free = int(total * available_ratio)
        used = max(0, total - free)
        return {
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "used_ratio": used / total if total > 0 else None,
            "in_use_bytes": used,
            "in_use_ratio": used / total if total > 0 else None,
            "pressure_available_ratio": available_ratio,
        }

    return {
        "total_bytes": total,
        "used_bytes": None,
        "free_bytes": None,
        "used_ratio": None,
    }


def _memory_pressure_available_ratio() -> float | None:
    try:
        completed = subprocess.run(
            ["/usr/bin/memory_pressure", "-Q"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", completed.stdout)
    if match is None:
        return None
    return min(max(int(match.group(1)) / 100.0, 0.0), 1.0)


def _total_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _vm_stat_pages() -> dict[str, object] | None:
    try:
        completed = subprocess.run(
            ["/usr/bin/vm_stat"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    page_size = 4096
    pages: dict[str, int] = {}
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip().rstrip(".")
        if "page size of" in line:
            try:
                page_size = int(line.split("page size of", 1)[1].split("bytes", 1)[0].strip())
            except (IndexError, ValueError):
                page_size = 4096
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        digits = "".join(char for char in value if char.isdigit())
        if digits:
            pages[key] = int(digits)
    return {"page_size": page_size, "pages": pages}
