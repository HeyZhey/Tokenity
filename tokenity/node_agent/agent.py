from __future__ import annotations

import getpass
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

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from tokenity import __version__
from tokenity.mlx.hostfile import ClusterNode, ConnectionMode, HostfileError, build_hostfile
from tokenity.mlx.rdma_probe import RDMAProbeResult, probe_rdma
from tokenity.process.supervisor import RoleSupervisor


DEFAULT_MODEL_ROOT = "/Users/Shared/TokenityModels"
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


class StopAllRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout: float = Field(default=10.0, ge=0.1, le=30.0)


class HeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ttl_seconds: float = Field(default=30.0, ge=15.0, le=300.0)


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


def create_app(
    *,
    rdma_probe_fn=probe_rdma,
    supervisor: RoleSupervisor | None = None,
    post_json_fn=None,
    rank_ready_fn=None,
    rank_stabilize_fn=None,
    rank_connected_fn=None,
) -> FastAPI:
    app = FastAPI(title="Tokenity Node Agent", version=__version__)
    roles = supervisor or RoleSupervisor()
    post_json = post_json_fn or _post_json
    wait_for_rank = rank_ready_fn or _wait_for_listening_port
    stabilize_rank = rank_stabilize_fn or time.sleep
    wait_for_rank_connection = rank_connected_fn or _wait_for_rank_connection
    distributed_workers: list[str] = []
    lifecycle_lock = threading.RLock()
    watchdog_stop = threading.Event()
    launch_generation = 0
    lease_deadline: float | None = None
    runtime_port = 8_000

    def begin_launch(lease_seconds: float) -> int:
        nonlocal launch_generation, lease_deadline
        with lifecycle_lock:
            launch_generation += 1
            lease_deadline = time.monotonic() + lease_seconds
            return launch_generation

    def launch_is_current(generation: int) -> bool:
        with lifecycle_lock:
            return generation == launch_generation

    def cancel_launch() -> None:
        nonlocal launch_generation
        with lifecycle_lock:
            launch_generation += 1

    def renew_lease(ttl_seconds: float) -> float:
        nonlocal lease_deadline
        with lifecycle_lock:
            lease_deadline = time.monotonic() + ttl_seconds
            return lease_deadline

    def clear_lease() -> None:
        nonlocal lease_deadline
        with lifecycle_lock:
            lease_deadline = None

    def request_local_model_roles_stop() -> list[dict[str, object]]:
        with lifecycle_lock:
            port = runtime_port
        coordinator_requested = False
        try:
            _post_json(f"http://127.0.0.1:{port}/v1/tokenity/stop", {}, 2.0)
            coordinator_requested = True
        except Exception:
            pass

        statuses: list[object] = []
        request_role_stop = getattr(roles, "request_stop", None)
        for role in MODEL_ROLES:
            # The coordinator's admin endpoint asks Uvicorn to exit naturally.
            # Sending SIGTERM immediately would turn a clean exit into -15 and
            # can tear down JACCL while the worker is still in a collective.
            if role == "distributed-openai" and coordinator_requested:
                statuses.append(roles.status(role))
            elif request_role_stop is not None:
                statuses.append(request_role_stop(role))
            else:
                statuses.append(roles.status(role))
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
        wait_for_role = getattr(roles, "wait", None)
        for role in MODEL_ROLES:
            remaining = max(0.0, deadline - time.monotonic())
            status = wait_for_role(role, remaining) if wait_for_role is not None else roles.status(role)
            if getattr(status, "pid", None) is not None:
                status = roles.stop(role, timeout=max(0.5, min(2.0, remaining)))
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

    @app.on_event("startup")
    def startup_cleanup() -> None:
        cleanup = getattr(roles, "cleanup_orphaned_model_processes", None)
        if cleanup is not None:
            cleanup()
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
            "available_ports": {"agent": 9100, "openai": 8000, "mlx_starting_port": 29500},
            "process_roles": [status.__dict__ for status in roles.status()],  # type: ignore[union-attr]
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
            "memory": _memory_stats(),
        }

    @app.post("/v1/node/heartbeat")
    def heartbeat(request: HeartbeatRequest) -> dict[str, object]:
        deadline = renew_lease(request.ttl_seconds)
        return {
            "status": "ok",
            "lease_seconds": request.ttl_seconds,
            "deadline_monotonic": deadline,
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
        if not request.dry_run and Path(request.python).resolve() != Path(sys.executable).resolve():
            raise HTTPException(
                status_code=400,
                detail="Cluster startup must use the coordinator Agent's installed Python runtime.",
            )
        nodes = _request_nodes(request)
        try:
            rank_requests = _http_rank_requests(request, nodes)
        except HostfileError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        plan = _http_launch_plan(nodes, rank_requests)
        if request.dry_run:
            return {"dry_run": True, "launch_plan": plan}

        runtime_port = request.port
        generation = begin_launch(request.lease_seconds)
        local_request = rank_requests[0]
        local_command, local_env = _rank_command_and_environment(local_request)
        local_status = roles.start(
            "distributed-openai",
            local_command,
            env=local_env,
            cwd=_distributed_code_root(),
        )
        if request.connection_mode != ConnectionMode.RING:
            if local_status.pid is None or not wait_for_rank(local_status.pid, request.starting_port, 45.0):
                roles.stop("distributed-openai", timeout=5)
                clear_lease()
                raise HTTPException(
                    status_code=504,
                    detail="Coordinator rank did not open the JACCL port before the startup deadline.",
                )
            stabilize_rank(3.0)
        if not launch_is_current(generation):
            roles.stop("distributed-openai", timeout=5)
            raise HTTPException(status_code=409, detail="Model loading was cancelled.")
        started_workers: list[str] = []
        try:
            for node, rank_request in zip(nodes[1:], rank_requests[1:]):
                if not launch_is_current(generation):
                    raise _LaunchCancelled("Model loading was cancelled.")
                agent_url = _agent_url(node)
                post_json(
                    f"{agent_url}/v1/node/start-distributed-rank",
                    rank_request.model_dump(mode="json"),
                    25.0,
                )
                started_workers.append(agent_url)
                if not launch_is_current(generation):
                    raise _LaunchCancelled("Model loading was cancelled.")
        except Exception as exc:
            for agent_url in started_workers:
                try:
                    post_json(
                        f"{agent_url}/v1/node/stop-all",
                        {"timeout": 5},
                        8.0,
                    )
                except Exception:
                    pass
            roles.stop("distributed-openai", timeout=5)
            clear_lease()
            if isinstance(exc, _LaunchCancelled):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise HTTPException(status_code=502, detail=f"Could not start worker rank over HTTP: {exc}") from exc

        with lifecycle_lock:
            distributed_workers[:] = started_workers
        return {
            "dry_run": False,
            "launch_plan": plan,
            "status": local_status.__dict__,
            "workers": started_workers,
        }

    @app.post("/v1/node/start-distributed-rank")
    def start_distributed_rank(request: RankStartRequest) -> dict[str, object]:
        nonlocal runtime_port
        if request.rank >= request.world_size:
            raise HTTPException(status_code=400, detail="Rank must be smaller than world size.")
        if request.coordinator:
            raise HTTPException(status_code=400, detail="Remote rank endpoint cannot start the coordinator role.")
        if Path(request.python).resolve() != Path(sys.executable).resolve():
            raise HTTPException(
                status_code=400,
                detail="Remote ranks must use the Node Agent's installed Python runtime.",
            )
        try:
            command, env = _rank_command_and_environment(request)
        except HostfileError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        runtime_port = request.port
        generation = begin_launch(request.lease_seconds)
        status = roles.start(
            "distributed-openai-rank",
            command,
            env=env,
            cwd=_distributed_code_root(),
        )
        connected = status.pid is not None and wait_for_rank_connection(status.pid, request, 20.0)
        if not connected or not launch_is_current(generation):
            failed = roles.status("distributed-openai-rank")
            roles.stop("distributed-openai-rank", timeout=5)
            clear_lease()
            if not launch_is_current(generation):
                raise HTTPException(status_code=409, detail="Model loading was cancelled.")
            message = getattr(failed, "message", None) or getattr(failed, "log_tail", None)
            raise HTTPException(
                status_code=504,
                detail=message or "Worker rank could not establish its MLX data-plane connection.",
            )
        return {"status": status.__dict__, "rank": request.rank, "cluster_id": request.cluster_id}

    @app.post("/v1/node/stop-role")
    def stop_role(request: StopRequest) -> dict[str, object]:
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


def _http_rank_requests(request: StartRequest, nodes: list[ClusterNode]) -> list[RankStartRequest]:
    if not nodes:
        raise HostfileError("At least one node is required.")
    for node in nodes:
        _agent_url(node)

    hostfile = build_hostfile(nodes, request.connection_mode)
    world_size = len(nodes)
    ring_hosts: list[list[str]] = []
    rdma_matrix: list[list[str | None]] = []
    coordinator_ip: str | None = None

    if request.connection_mode == ConnectionMode.RING:
        port = request.starting_port
        for node in nodes:
            data_ip = node.lan_ip or node.rdma_ip
            if not data_ip:
                raise HostfileError(f"{node.id}: missing standard-network IP.")
            ring_hosts.append([f"{data_ip}:{port}"])
            port += 1
    else:
        if not hostfile[0]["ips"]:
            raise HostfileError("Rank 0 needs a Thunderbolt/RDMA coordinator IP.")
        coordinator_ip = str(hostfile[0]["ips"][0])
        rdma_matrix = [list(row["rdma"]) for row in hostfile]

    cluster_id = uuid.uuid4().hex
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
        )
        for rank in range(world_size)
    ]


def _http_launch_plan(
    nodes: list[ClusterNode],
    rank_requests: list[RankStartRequest],
) -> dict[str, object]:
    return {
        "role": "distributed-openai",
        "backend": "Tokenity distributed OpenAI server",
        "experimental": False,
        "transport": "http",
        "ranks": [
            {
                "rank": rank_request.rank,
                "agent_url": _agent_url(node),
                "endpoint": "/v1/node/start-distributed-rank",
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
        "TOKENITY_MLX_LOAD_EVAL_CHUNK_SIZE": os.environ.get("TOKENITY_MLX_LOAD_EVAL_CHUNK_SIZE", "1"),
        "TOKENITY_MLX_LOAD_EVAL_LOG_INTERVAL": os.environ.get("TOKENITY_MLX_LOAD_EVAL_LOG_INTERVAL", "100"),
        "TOKENITY_MLX_LOAD_EVAL_SLEEP_SECONDS": os.environ.get("TOKENITY_MLX_LOAD_EVAL_SLEEP_SECONDS", "0.05"),
        "TOKENITY_MLX_LOAD_POST_BARRIER": os.environ.get("TOKENITY_MLX_LOAD_POST_BARRIER", "0"),
        "TOKENITY_MLX_DISTRIBUTED_INIT_RANK0_DELAY_SECONDS": os.environ.get(
            "TOKENITY_MLX_DISTRIBUTED_INIT_RANK0_DELAY_SECONDS", "0"
        ),
        "MLX_RANK": str(request.rank),
    }

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
        used = max(0, total - free)

        def page_bytes(key: str) -> int:
            return max(0, pages.get(key, 0) * page_size)

        return {
            "total_bytes": total,
            # Match macOS `top` PhysMem: memory which currently occupies
            # physical pages, including reclaimable file-backed model weights.
            "used_bytes": used,
            "free_bytes": free,
            "used_ratio": used / total if total > 0 else None,
            "wired_bytes": page_bytes("Pages wired down"),
            "compressed_bytes": page_bytes("Pages occupied by compressor"),
            "active_bytes": page_bytes("Pages active"),
            "inactive_bytes": page_bytes("Pages inactive"),
            "file_backed_bytes": page_bytes("File-backed pages"),
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
