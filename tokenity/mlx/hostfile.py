from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConnectionMode(str, Enum):
    RING = "ring"
    JACCL = "jaccl"
    JACCL_RING = "jaccl-ring"


class HostfileError(ValueError):
    pass


@dataclass
class ClusterNode:
    id: str
    agent_url: str | None = None
    lan_ip: str | None = None
    rdma_ip: str | None = None
    rdma_devices: list[str] = field(default_factory=list)
    rdma_matrix_row: list[str | None] | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ClusterNode":
        return cls(
            id=str(payload.get("id") or payload.get("node_id") or payload.get("hostname") or "node"),
            agent_url=payload.get("agent_url"),
            lan_ip=payload.get("lan_ip") or payload.get("ip"),
            rdma_ip=payload.get("rdma_ip") or payload.get("thunderbolt_ip"),
            rdma_devices=list(payload.get("rdma_devices") or []),
            rdma_matrix_row=payload.get("rdma_matrix_row"),
        )


def build_hostfile(nodes: list[ClusterNode], mode: ConnectionMode) -> list[dict[str, Any]]:
    if not nodes:
        raise HostfileError("At least one node is required.")
    if mode == ConnectionMode.RING:
        rows = _build_ring_hostfile(nodes)
    elif mode in {ConnectionMode.JACCL, ConnectionMode.JACCL_RING}:
        rows = _build_jaccl_hostfile(nodes, include_ring_ips=mode == ConnectionMode.JACCL_RING)
    else:
        raise HostfileError(f"Unsupported connection mode: {mode}")
    if len(nodes) > 1:
        for node, row in zip(nodes, rows):
            if any(is_loopback_host(address) for address in row["ips"]):
                raise HostfileError(f"{node.id}: distributed data address must not use loopback.")
    return rows


def is_loopback_host(host: str) -> bool:
    host = host.strip().rstrip(".").strip("[]")
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        lowered = host.lower()
        if lowered == "localhost" or lowered.endswith(".localhost"):
            return True
        try:
            addresses = socket.getaddrinfo(
                host,
                None,
                type=socket.SOCK_STREAM,
                flags=socket.AI_NUMERICHOST,
            )
        except socket.gaierror:
            return False
        return any(ipaddress.ip_address(address[4][0]).is_loopback for address in addresses)


def validate_jaccl_readiness(nodes: list[ClusterNode]) -> list[str]:
    errors: list[str] = []
    world_size = len(nodes)
    for index, node in enumerate(nodes):
        if not node.rdma_ip:
            errors.append(f"{node.id}: missing Thunderbolt/RDMA IP.")
        if not node.rdma_devices and not node.rdma_matrix_row:
            errors.append(f"{node.id}: missing active RDMA device.")
        if node.rdma_matrix_row is not None:
            if len(node.rdma_matrix_row) != world_size:
                errors.append(f"{node.id}: RDMA matrix row length must equal world size {world_size}.")
            elif node.rdma_matrix_row[index] is not None:
                errors.append(f"{node.id}: RDMA matrix self entry must be null.")
    return errors


def _build_ring_hostfile(nodes: list[ClusterNode]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in nodes:
        data_ip = node.lan_ip or node.rdma_ip
        rows.append(
            {
                "ips": [data_ip] if data_ip else [],
                "rdma": [],
            }
        )
    return rows


def _build_jaccl_hostfile(
    nodes: list[ClusterNode],
    *,
    include_ring_ips: bool = False,
) -> list[dict[str, Any]]:
    errors = validate_jaccl_readiness(nodes)
    if errors:
        raise HostfileError("JACCL readiness failed: " + "; ".join(errors))

    rows: list[dict[str, Any]] = []
    world_size = len(nodes)
    for index, node in enumerate(nodes):
        if node.rdma_matrix_row is not None:
            rdma_row = list(node.rdma_matrix_row)
        else:
            active_device = node.rdma_devices[0]
            rdma_row = [active_device if peer != index else None for peer in range(world_size)]

        ips: list[str] = []
        if include_ring_ips:
            ips = [ip for ip in [node.rdma_ip or node.lan_ip] if ip]
        elif index == 0 and node.rdma_ip:
            # This mirrors the known successful two-node JACCL hostfile:
            # rank 0 advertises its local Thunderbolt IP; worker rows carry
            # RDMA device rows without ordinary IP entries.
            ips = [node.rdma_ip]

        rows.append({"ips": ips, "rdma": rdma_row})
    return rows
