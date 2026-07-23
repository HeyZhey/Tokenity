from __future__ import annotations

import getpass
import hashlib
import http.client
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
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
from tokenity.control import (
    InstanceConflict,
    InstanceLifecycle,
    InstanceRegistry,
    InstanceRouter,
    ResourceAdmissionError,
    ResourceLedger,
)
from tokenity.inference.native_mtp import scan_native_mtp_capability
from tokenity.mlx.hostfile import ClusterNode, ConnectionMode, HostfileError, build_hostfile
from tokenity.mlx.rdma_probe import RDMAProbeResult, probe_rdma
from tokenity.model_inspection import model_usage_metadata, standalone_model_issue
from tokenity.process.supervisor import RoleSupervisor


DEFAULT_MODEL_ROOT = "/Users/Shared/TokenityModels"
NODE_AGENT_CONTRACT_VERSION = 1
NODE_AGENT_CAPABILITIES = (
    "cluster_runtime",
    "instance_quorum",
    "instance_runtimes",
    "managed_instances",
    "native_mtp",
)
MODEL_ROLES = (
    "distributed-openai",
    "distributed-openai-rank",
    "single-node-openai",
    "official-mlx-lm",
)


class _LaunchCancelled(RuntimeError):
    pass


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


class StopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal[
        "official-mlx-lm",
        "distributed-openai",
        "distributed-openai-rank",
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
    gateway_open_fn=None,
) -> FastAPI:
    app = FastAPI(title="Tokenity Node Agent", version=__version__)
    roles = supervisor or RoleSupervisor()
    post_json = post_json_fn or _post_json
    get_json = get_json_fn or _get_json
    wait_for_rank = rank_ready_fn or _wait_for_listening_port
    stabilize_rank = rank_stabilize_fn or time.sleep
    wait_for_rank_connection = rank_connected_fn or _wait_for_rank_connection
    runtime_preflight = runtime_preflight_fn or _runtime_preflight
    gateway_open = gateway_open_fn or _open_gateway_upstream
    instances = InstanceRegistry()
    instance_router = InstanceRouter(instances)
    resource_ledger = ResourceLedger(_total_memory_bytes() or 0)
    instance_workers: dict[str, list[str]] = {}
    instance_roles: dict[str, str] = {}
    instance_ports: dict[str, int] = {}
    instance_status_paths: dict[str, str] = {}
    distributed_workers: list[str] = []
    lifecycle_lock = threading.RLock()
    watchdog_stop = threading.Event()
    launch_generation = 0
    launch_tokens: dict[str, int] = {}
    lease_deadline: float | None = None
    runtime_port = 8_000
    cluster_runtime: dict[str, object] | None = None
    cluster_runtimes: dict[str, dict[str, object]] = {}
    startup_orphans: list[dict[str, object]] = []

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
        nonlocal lease_deadline, cluster_runtime
        with lifecycle_lock:
            lease_deadline = None
            cluster_runtime = None
            cluster_runtimes.clear()

    def request_local_model_roles_stop() -> list[dict[str, object]]:
        coordinator_requested: set[str | None] = set()
        ports = {None: runtime_port, **{key: value for key, value in instance_ports.items()}}
        for instance_id, port in ports.items():
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
            if role in {"distributed-openai", "single-node-openai"} and instance_id in coordinator_requested:
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
            urls = list(distributed_workers)
            if clear:
                distributed_workers.clear()
        return urls

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
        while not watchdog_stop.wait(1.0):
            with lifecycle_lock:
                expired = lease_deadline is not None and time.monotonic() >= lease_deadline
            if expired:
                stop_everything(3.0, notify_workers=False)
            now = time.time()
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

    @app.on_event("startup")
    def startup_cleanup() -> None:
        cleanup = getattr(roles, "cleanup_orphaned_model_processes", None)
        if cleanup is not None:
            reconciled = cleanup()
            if isinstance(reconciled, list):
                startup_orphans[:] = reconciled
        threading.Thread(target=lease_watchdog, daemon=True).start()

    @app.on_event("shutdown")
    def shutdown_cleanup() -> None:
        watchdog_stop.set()
        stop_everything(3.0, notify_workers=False)

    @app.get("/v1/node/info")
    def node_info() -> dict[str, object]:
        rdma: RDMAProbeResult = rdma_probe_fn()
        memory = _memory_stats()
        return {
            "node_id": _node_id(),
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
            "agent_contract": {
                "version": NODE_AGENT_CONTRACT_VERSION,
                "capabilities": list(NODE_AGENT_CAPABILITIES),
            },
            "available_ports": {"agent": 9100, "openai": 8000, "mlx_starting_port": 29500},
            "process_roles": [status.__dict__ for status in roles.status()],  # type: ignore[union-attr]
            "cluster_runtime": cluster_runtime_snapshot(),
            "cluster_runtimes": cluster_runtimes_snapshot(),
            "instances": instances.snapshots(),
            "resource_ledger": resource_ledger.snapshot(),
            "orphaned_processes": list(startup_orphans),
            "memory": memory,
            "rdma": rdma.to_dict(),
        }

    @app.get("/v1/node/models")
    def node_models(root: str = Query(DEFAULT_MODEL_ROOT)) -> dict[str, object]:
        return {"root": root, "models": scan_models(Path(root))}

    @app.get("/v1/node/status")
    def node_status() -> dict[str, object]:
        return {
            "roles": [status.__dict__ for status in roles.status()],  # type: ignore[union-attr]
            "cluster_runtime": cluster_runtime_snapshot(),
            "cluster_runtimes": cluster_runtimes_snapshot(),
            "instances": instances.snapshots(),
            "resource_ledger": resource_ledger.snapshot(),
            "orphaned_processes": list(startup_orphans),
            "memory": _memory_stats(),
        }

    @app.get("/v1/node/instances")
    def list_instances() -> dict[str, object]:
        return {"data": instances.snapshots(), "resource_ledger": resource_ledger.snapshot()}

    @app.get("/v1/node/instances/{instance_id}")
    def get_instance(instance_id: str) -> dict[str, object]:
        instance = instances.get(instance_id)
        if instance is None:
            raise HTTPException(status_code=404, detail=f"Unknown model instance: {instance_id}")
        role = instance_roles.get(instance_id)
        status = supervisor_status(role, instance_id) if role else None
        runtime = _read_runtime_status(instance_status_paths.get(instance_id))
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
        }

    @app.get("/v1/node/instances/{instance_id}/quorum")
    def instance_quorum(instance_id: str) -> dict[str, object]:
        instance = instances.get(instance_id)
        if instance is None:
            raise HTTPException(status_code=404, detail=f"Unknown model instance: {instance_id}")
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
            if not isinstance(runtime, dict):
                issues.append(f"{node}: runtime readiness is unavailable.")
                continue
            if runtime.get("instance_id") != instance_id:
                issues.append(f"{node}: instance identity mismatch.")
            if runtime.get("operation_id") != instance.operation_id:
                issues.append(f"{node}: operation identity mismatch.")
            if runtime.get("phase") != "ready":
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
            if model_or_instance_id not in {
                str(snapshot.get("instance_id") or ""),
                str(snapshot.get("requested_model_id") or ""),
            }:
                continue
            if snapshot.get("state") in {"ready", "busy"}:
                # BUSY is still routable: the runtime owns concurrency and
                # queue limits, while the request lease prevents unload.
                get_instance(str(snapshot["instance_id"]))
                continue
            try:
                instance_quorum(str(snapshot["instance_id"]))
            except HTTPException:
                continue

    @app.get("/v1/gateway/routes")
    def gateway_routes() -> dict[str, object]:
        routes = []
        for snapshot in instances.snapshots():
            if snapshot.get("state") not in {"ready", "busy"}:
                continue
            routes.append(
                {
                    "model": snapshot["requested_model_id"],
                    "instance_id": snapshot["instance_id"],
                    "execution_mode": snapshot["execution_mode"],
                    "state": snapshot["state"],
                    "active_request_count": snapshot["active_request_count"],
                    "api_base_url": f"http://127.0.0.1:{snapshot['http_port']}/v1",
                }
            )
        return {"data": routes}

    @app.get("/v1/models")
    def gateway_models() -> dict[str, object]:
        now = int(time.time())
        data = []
        for snapshot in instances.snapshots():
            if snapshot.get("state") not in {"ready", "busy"}:
                continue
            data.append(
                {
                    "id": snapshot["requested_model_id"],
                    "object": "model",
                    "created": now,
                    "owned_by": "tokenity",
                    "tokenity_instance_id": snapshot["instance_id"],
                    "tokenity_execution_mode": snapshot["execution_mode"],
                }
            )
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def gateway_chat_completions(request: FastAPIRequest) -> Response:
        body = await request.body()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc
        model = payload.get("model") if isinstance(payload, dict) else None
        if not isinstance(model, str) or not model.strip():
            raise HTTPException(status_code=400, detail="A model alias or instance_id is required.")
        refresh_gateway_candidates(model)
        try:
            selected = instance_router.resolve(model)
            instances.acquire_request_lease(selected.instance_id)
        except InstanceConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        upstream_headers = {"Content-Type": "application/json"}
        authorization = request.headers.get("authorization")
        if authorization:
            upstream_headers["Authorization"] = authorization
        try:
            connection, upstream = await run_in_threadpool(
                gateway_open,
                f"http://127.0.0.1:{selected.http_port}/v1/chat/completions",
                body,
                upstream_headers,
                600.0,
            )
        except Exception as exc:
            instances.release_request_lease(selected.instance_id)
            raise HTTPException(
                status_code=502,
                detail=f"Model instance {selected.instance_id} could not accept the request: {exc}",
            ) from exc

        response_headers = {
            "X-Tokenity-Instance-ID": selected.instance_id,
            "X-Tokenity-Model": selected.requested_model_id,
        }
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
            return StreamingResponse(
                _gateway_body_chunks(
                    connection,
                    upstream,
                    on_close=lambda: instances.release_request_lease(selected.instance_id),
                ),
                status_code=upstream.status,
                headers=response_headers,
                media_type=None,
            )

        try:
            response_body = await run_in_threadpool(upstream.read)
        finally:
            connection.close()
            instances.release_request_lease(selected.instance_id)
        return Response(
            content=response_body,
            status_code=upstream.status,
            headers=response_headers,
            media_type=None,
        )

    @app.post("/v1/node/heartbeat")
    def heartbeat(request: HeartbeatRequest) -> dict[str, object]:
        if request.instance_id:
            instance = instances.get(request.instance_id)
            if instance is None:
                raise HTTPException(status_code=404, detail=f"Unknown model instance: {request.instance_id}")
            instance.heartbeat(request.ttl_seconds)
            deadline = instance.deadline
            deadline_kind = "epoch"
        else:
            deadline = renew_lease(request.ttl_seconds)
            deadline_kind = "monotonic"
        return {
            "status": "ok",
            "lease_seconds": request.ttl_seconds,
            f"deadline_{deadline_kind}": deadline,
        }

    @app.post("/v1/node/start-official-mlx-lm")
    def start_official(request: StartRequest) -> dict[str, object]:
        del request
        raise HTTPException(
            status_code=410,
            detail=(
                "The legacy mlx.launch backend is disabled because it requires SSH. "
                "Use /v1/node/start-distributed-openai for HTTP Node Agent orchestration."
            ),
        )

    @app.post("/v1/node/start-distributed-openai")
    def start_distributed(request: StartRequest) -> dict[str, object]:
        nonlocal runtime_port
        if issue := standalone_model_issue(request.model):
            raise HTTPException(status_code=400, detail=issue)
        if not request.dry_run and Path(request.python).resolve() != Path(sys.executable).resolve():
            raise HTTPException(
                status_code=400,
                detail="Cluster startup must use the coordinator Agent's installed Python runtime.",
            )
        nodes = _request_nodes(request)
        instance_id = request.instance_id or uuid.uuid4().hex
        operation_id = request.operation_id or uuid.uuid4().hex
        request.instance_id = instance_id
        request.operation_id = operation_id
        if len(nodes) == 1:
            request.connection_mode = ConnectionMode.RING
        model_revision = _model_revision(request.model)
        if not request.dry_run:
            issues = runtime_preflight(request.python, request.model, model_revision)
            if issues:
                raise HTTPException(
                    status_code=412,
                    detail={"stage": "preflight", "instance_id": instance_id, "issues": issues},
                )
            request.port = _allocate_instance_port(request.port, resource_ledger.snapshot())
            request.starting_port = _allocate_collective_port(
                request.starting_port,
                len(nodes),
                resource_ledger.snapshot(),
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

        reservation = request.memory_reservation_bytes
        if reservation is None:
            reservation = _estimated_memory_reservation(request.model, len(nodes))
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
            resource_ledger.reserve(instance_id, reservation, ports)
        except ResourceAdmissionError as exc:
            instances.remove(instance_id)
            raise HTTPException(status_code=409, detail={"stage": "resource_admission", "message": str(exc)}) from exc
        instance.transition(InstanceLifecycle.LAUNCHING)

        runtime_port = request.port
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
        if len(nodes) > 1 and request.connection_mode != ConnectionMode.RING:
            if local_status.pid is None or not wait_for_rank(local_status.pid, request.starting_port, 45.0):
                supervisor_stop(local_role, 5, instance_id)
                resource_ledger.release(instance_id)
                remove_cluster_runtime(instance_id)
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
                    try:
                        post_json(
                            f"{agent_url}/v1/node/stop-all",
                            {"timeout": 5},
                            8.0,
                        )
                    except Exception:
                        pass
            supervisor_stop(local_role, 5, instance_id)
            resource_ledger.release(instance_id)
            remove_cluster_runtime(instance_id)
            instance.transition(
                InstanceLifecycle.FAILED,
                error={"stage": "launching", "message": str(exc)},
            )
            if isinstance(exc, _LaunchCancelled):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise HTTPException(status_code=502, detail=f"Could not start worker rank over HTTP: {exc}") from exc

        with lifecycle_lock:
            distributed_workers[:] = started_workers
            instance_workers[instance_id] = list(started_workers)
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
        nonlocal runtime_port
        if issue := standalone_model_issue(request.model):
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
        runtime_port = request.port
        generation = begin_launch(instance_id)
        set_cluster_runtime(request, "worker")
        reservation = request.memory_reservation_bytes
        if reservation is None:
            reservation = _estimated_memory_reservation(request.model, request.world_size)
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
            )
            if created:
                resource_ledger.reserve(
                    instance_id,
                    reservation,
                    list(range(request.starting_port, request.starting_port + max(1, request.world_size))),
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
        try:
            if instance.state not in {InstanceLifecycle.UNLOADING, InstanceLifecycle.STOPPED}:
                instance.transition(InstanceLifecycle.UNLOADING)
        except InstanceConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        port = instance_ports.get(instance_id, instance.http_port)
        try:
            _post_json(f"http://127.0.0.1:{port}/v1/tokenity/stop", {}, 2.0)
        except Exception:
            pass

        worker_results: list[dict[str, object]] = []
        for agent_url in instance_workers.get(instance_id, []):
            try:
                result = post_json(
                    f"{agent_url}/v1/node/instances/{instance_id}/stop",
                    {"timeout": request.timeout},
                    request.timeout + 3,
                )
                worker_results.append({"agent_url": agent_url, "result": result})
            except Exception as exc:
                worker_results.append({"agent_url": agent_url, "error": str(exc)})

        role = instance_roles.get(instance_id)
        status = None
        if role is not None:
            supervisor_request_stop(role, instance_id)
            status = supervisor_wait(role, request.timeout, instance_id)
            if getattr(status, "pid", None) is not None:
                status = supervisor_stop(role, min(2.0, request.timeout), instance_id)
        resource_ledger.release(instance_id)
        instance_workers.pop(instance_id, None)
        instance_roles.pop(instance_id, None)
        instance_ports.pop(instance_id, None)
        remove_cluster_runtime(instance_id)
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
):
    try:
        while True:
            # read1 returns currently available bytes instead of waiting to fill
            # a large buffer, preserving the runtime's SSE chunk cadence.
            chunk = upstream.read1(64 * 1024)
            if not chunk:
                break
            yield chunk
    finally:
        connection.close()
        try:
            on_close()
        except InstanceConflict:
            pass


def scan_models(root: Path) -> list[dict[str, object]]:
    if not root.exists() or not root.is_dir():
        return []
    models: list[dict[str, object]] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if child.name.startswith("."):
            continue
        if not child.is_dir():
            continue
        config = _read_model_config(child / "config.json")
        safetensors = list(child.glob("*.safetensors"))
        gguf_files = list(child.glob("*.gguf"))
        markers = {
            "config": (child / "config.json").exists(),
            "tokenizer": (child / "tokenizer.json").exists() or (child / "tokenizer.model").exists(),
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
                    "path": str(child),
                    "markers": markers,
                    "format": "GGUF" if gguf_files else "MLX" if safetensors else "Transformers",
                    "quantization": _quantization_description(config, child),
                    "size_bytes": _directory_size(child),
                    "architecture": architecture,
                    "shard_count": len(gguf_files) + len(safetensors),
                    "revision": _model_revision(str(child)),
                    "native_mtp": scan_native_mtp_capability(child).to_dict(),
                    **model_usage_metadata(config),
                }
            )
    return models


def _read_model_config(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
            agent_url="http://127.0.0.1:9100",
            lan_ip="127.0.0.1",
        )
    ]


def _agent_url(node: ClusterNode) -> str:
    if node.agent_url:
        url = node.agent_url.rstrip("/")
    elif node.lan_ip:
        url = f"http://{node.lan_ip}:9100"
    else:
        raise HostfileError(f"{node.id}: missing Node Agent URL.")
    parsed = urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise HostfileError(f"{node.id}: Agent URL must use http:// with a valid host.")
    if parsed.username or parsed.password:
        raise HostfileError(f"{node.id}: Agent URL must not contain a username or password.")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise HostfileError(f"{node.id}: Agent URL must be an HTTP origin without a path, query, or fragment.")
    return url


def _http_rank_requests(
    request: StartRequest,
    nodes: list[ClusterNode],
    *,
    model_revision: str | None = None,
) -> list[RankStartRequest]:
    if not nodes:
        raise HostfileError("At least one node is required.")
    for node in nodes:
        _agent_url(node)

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
    elif request.connection_mode == ConnectionMode.RING:
        port = request.starting_port
        for node in nodes:
            data_ip = node.lan_ip or node.rdma_ip
            if not data_ip:
                raise HostfileError(f"{node.id}: missing standard-network IP.")
            ring_hosts.append([f"{data_ip}:{port}"])
            port += 1
    else:
        hostfile = build_hostfile(nodes, request.connection_mode)
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
        "TOKENITY_MLX_LOAD_POST_BARRIER": os.environ.get("TOKENITY_MLX_LOAD_POST_BARRIER", "0"),
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
    return os.environ.get("TOKENITY_CODE_ROOT") or "/Users/Shared/TokenityCode"


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
    if mlx_version != "0.31.2":
        issues.append(f"mlx 0.31.2 is required; found {mlx_version or 'not installed'}.")
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
    files = [root / "config.json", root / "model.safetensors.index.json"]
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
    paths = [Path(__file__), Path(__file__).parents[1] / "serving" / "distributed_openai.py"]
    try:
        for path in paths:
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    except OSError:
        return None
    return digest.hexdigest()


def _estimated_memory_reservation(model: str, world_size: int) -> int:
    root = Path(model)
    total = 0
    try:
        for path in root.iterdir():
            if path.is_file() and path.suffix in {".safetensors", ".gguf"}:
                total += path.stat().st_size
    except OSError:
        return 0
    if total == 0:
        return 0
    shard = (total + max(1, world_size) - 1) // max(1, world_size)
    return int(shard * 1.15) + (2 * 1024**3)


def _allocate_instance_port(preferred: int, ledger: dict[str, object]) -> int:
    reserved = {int(port) for port in dict(ledger.get("ports") or {}).keys()}
    for port in range(preferred, min(65_536, preferred + 256)):
        if port in reserved:
            continue
        if _tcp_port_available(port):
            return port
    raise HTTPException(status_code=409, detail=f"No free HTTP port near {preferred}.")


def _allocate_collective_port(preferred: int, world_size: int, ledger: dict[str, object]) -> int:
    reserved = {int(port) for port in dict(ledger.get("ports") or {}).keys()}
    width = max(1, world_size)
    for starting_port in range(preferred, min(65_536 - width, preferred + 256)):
        ports = range(starting_port, starting_port + width)
        if any(port in reserved for port in ports):
            continue
        if _tcp_port_available(starting_port):
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
