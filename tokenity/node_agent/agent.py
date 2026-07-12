from __future__ import annotations

import getpass
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from tokenity import __version__
from tokenity.mlx.hostfile import ClusterNode, ConnectionMode, HostfileError
from tokenity.mlx.launcher import (
    build_distributed_openai_launch_plan,
    build_official_mlx_lm_launch_plan,
)
from tokenity.mlx.rdma_probe import RDMAProbeResult, probe_rdma
from tokenity.process.supervisor import RoleSupervisor


DEFAULT_MODEL_ROOT = "/Users/Shared/TokenityModels"


class ClusterNodePayload(BaseModel):
    id: str
    ssh: str
    lan_ip: Optional[str] = None
    rdma_ip: Optional[str] = None
    rdma_devices: List[str] = Field(default_factory=list)
    rdma_matrix_row: Optional[List[Optional[str]]] = None

    def to_cluster_node(self) -> ClusterNode:
        return ClusterNode(
            id=self.id,
            ssh=self.ssh,
            lan_ip=self.lan_ip,
            rdma_ip=self.rdma_ip,
            rdma_devices=self.rdma_devices,
            rdma_matrix_row=self.rdma_matrix_row,
        )


class StartRequest(BaseModel):
    model: str
    nodes: List[ClusterNodePayload] = Field(default_factory=list)
    connection_mode: ConnectionMode = ConnectionMode.RING
    python: str = Field(default_factory=lambda: sys.executable)
    starting_port: int = 29500
    host: str = "0.0.0.0"
    port: int = 8000
    dry_run: bool = True


class StopRequest(BaseModel):
    role: Literal["official-mlx-lm", "distributed-openai", "single-node-openai", "node-agent"]
    timeout: float = 10.0


def create_app(
    *,
    rdma_probe_fn=probe_rdma,
    supervisor: RoleSupervisor | None = None,
) -> FastAPI:
    app = FastAPI(title="Tokenity Node Agent", version=__version__)
    roles = supervisor or RoleSupervisor()

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

    @app.post("/v1/node/start-official-mlx-lm")
    def start_official(request: StartRequest) -> dict[str, object]:
        nodes = _request_nodes(request)
        try:
            if request.dry_run:
                plan = build_official_mlx_lm_launch_plan(
                    nodes=nodes,
                    connection_mode=request.connection_mode,
                    model=request.model,
                    python=request.python,
                    starting_port=request.starting_port,
                    host=request.host,
                    port=request.port,
                )
                return {"dry_run": True, "launch_plan": plan.to_dict()}
            hostfile_path = _write_hostfile("official-mlx-lm", request, nodes)
            plan = build_official_mlx_lm_launch_plan(
                nodes=nodes,
                connection_mode=request.connection_mode,
                model=request.model,
                python=request.python,
                starting_port=request.starting_port,
                host=request.host,
                port=request.port,
                hostfile_path=str(hostfile_path),
            )
        except HostfileError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        status = roles.start("official-mlx-lm", plan.command)
        return {"dry_run": False, "launch_plan": plan.to_dict(), "status": status.__dict__}

    @app.post("/v1/node/start-distributed-openai")
    def start_distributed(request: StartRequest) -> dict[str, object]:
        nodes = _request_nodes(request)
        try:
            if request.dry_run:
                plan = build_distributed_openai_launch_plan(
                    nodes=nodes,
                    connection_mode=request.connection_mode,
                    model=request.model,
                    python=request.python,
                    starting_port=request.starting_port,
                    host=request.host,
                    port=request.port,
                )
                return {"dry_run": True, "launch_plan": plan.to_dict()}
            hostfile_path = _write_hostfile("distributed-openai", request, nodes)
            plan = build_distributed_openai_launch_plan(
                nodes=nodes,
                connection_mode=request.connection_mode,
                model=request.model,
                python=request.python,
                starting_port=request.starting_port,
                host=request.host,
                port=request.port,
                hostfile_path=str(hostfile_path),
            )
        except HostfileError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        status = roles.start("distributed-openai", plan.command)
        return {"dry_run": False, "launch_plan": plan.to_dict(), "status": status.__dict__}

    @app.post("/v1/node/stop-role")
    def stop_role(request: StopRequest) -> dict[str, object]:
        return {"status": roles.stop(request.role, timeout=request.timeout).__dict__}

    return app


def scan_models(root: Path) -> list[dict[str, object]]:
    if not root.exists() or not root.is_dir():
        return []
    models: list[dict[str, object]] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir():
            continue
        markers = {
            "config": (child / "config.json").exists(),
            "tokenizer": (child / "tokenizer.json").exists() or (child / "tokenizer.model").exists(),
            "safetensors": any(child.glob("*.safetensors")),
        }
        if any(markers.values()):
            models.append({"id": child.name, "path": str(child), "markers": markers})
    return models


def _request_nodes(request: StartRequest) -> list[ClusterNode]:
    if request.nodes:
        return [node.to_cluster_node() for node in request.nodes]
    return [ClusterNode(id="local", ssh="127.0.0.1", lan_ip="127.0.0.1")]


def _write_hostfile(role: str, request: StartRequest, nodes: list[ClusterNode]) -> Path:
    from tokenity.mlx.hostfile import build_hostfile

    base = Path(tempfile.gettempdir()) / "tokenity-hostfiles"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{role}-{uuid.uuid4().hex}.json"
    hostfile = build_hostfile(nodes, request.connection_mode)
    path.write_text(json.dumps(hostfile, indent=2), encoding="utf-8")
    return path


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
    if total is None or vm is None:
        return {
            "total_bytes": total,
            "used_bytes": None,
            "free_bytes": None,
            "used_ratio": None,
        }

    page_size = vm["page_size"]
    free_pages = vm["pages"].get("Pages free", 0) + vm["pages"].get("Pages speculative", 0)
    free = max(0, free_pages * page_size)
    used = max(0, total - free)
    return {
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": free,
        "used_ratio": used / total if total > 0 else None,
    }


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
