from __future__ import annotations

import pytest

from tokenity.mlx.hostfile import (
    ClusterNode,
    ConnectionMode,
    HostfileError,
    build_hostfile,
    is_loopback_host,
)


def test_known_two_mac_jaccl_data_plane_shape_has_no_control_credentials():
    nodes = [
        ClusterNode(
            id="mac-a",
            agent_url="http://198.51.100.23:9100",
            lan_ip="198.51.100.23",
            rdma_ip="203.0.113.1",
            rdma_devices=["rdma_en4"],
        ),
        ClusterNode(
            id="mac-b",
            agent_url="http://198.51.100.75:9100",
            lan_ip="198.51.100.75",
            rdma_ip="203.0.113.2",
            rdma_devices=["rdma_en5"],
        ),
    ]

    assert build_hostfile(nodes, ConnectionMode.JACCL) == [
        {"ips": ["203.0.113.1"], "rdma": [None, "rdma_en4"]},
        {"ips": [], "rdma": ["rdma_en5", None]},
    ]


def test_jaccl_blocks_missing_rdma_data():
    nodes = [
        ClusterNode(id="mac-a", agent_url="http://198.51.100.23:9100", lan_ip="198.51.100.23"),
        ClusterNode(id="mac-b", agent_url="http://198.51.100.75:9100", lan_ip="198.51.100.75"),
    ]

    with pytest.raises(HostfileError, match="JACCL readiness failed"):
        build_hostfile(nodes, ConnectionMode.JACCL)


def test_ring_data_plane_contains_only_ips():
    rows = build_hostfile(
        [ClusterNode(id="mac-b", agent_url="http://198.51.100.75:9100", lan_ip="198.51.100.75")],
        ConnectionMode.RING,
    )

    assert rows == [{"ips": ["198.51.100.75"], "rdma": []}]


@pytest.mark.parametrize("host", ["127.0.0.1 ", "127.1", "[::1]", "worker.localhost."])
def test_loopback_host_normalizes_equivalent_forms(host):
    assert is_loopback_host(host) is True


@pytest.mark.parametrize(
    ("mode", "rank"),
    [
        (ConnectionMode.RING, 0),
        (ConnectionMode.RING, 1),
        (ConnectionMode.JACCL, 0),
        (ConnectionMode.JACCL_RING, 0),
        (ConnectionMode.JACCL_RING, 1),
    ],
)
def test_distributed_hostfile_rejects_loopback_data_addresses(mode, rank):
    nodes = [
        ClusterNode(
            id="mac-a",
            lan_ip="198.51.100.23",
            rdma_ip="203.0.113.1",
            rdma_devices=["rdma_en4"],
        ),
        ClusterNode(
            id="mac-b",
            lan_ip="198.51.100.75",
            rdma_ip="203.0.113.2",
            rdma_devices=["rdma_en5"],
        ),
    ]
    if mode == ConnectionMode.RING:
        nodes[rank].lan_ip = "127.0.0.1"
    else:
        nodes[rank].rdma_ip = "::1"

    with pytest.raises(HostfileError, match="must not use loopback"):
        build_hostfile(nodes, mode)


def test_hostfile_ignores_loopback_in_addresses_the_mode_does_not_emit():
    ring_nodes = [
        ClusterNode(
            id="mac-a",
            lan_ip="198.51.100.23",
            rdma_ip="127.0.0.1",
            rdma_devices=["rdma_en4"],
        ),
        ClusterNode(
            id="mac-b",
            lan_ip="198.51.100.75",
            rdma_ip="::1",
            rdma_devices=["rdma_en5"],
        ),
    ]
    jaccl_nodes = [
        ClusterNode(
            id="mac-a",
            lan_ip="127.0.0.1",
            rdma_ip="203.0.113.1",
            rdma_devices=["rdma_en4"],
        ),
        ClusterNode(
            id="mac-b",
            lan_ip="::1",
            rdma_ip="203.0.113.2",
            rdma_devices=["rdma_en5"],
        ),
    ]

    assert build_hostfile(ring_nodes, ConnectionMode.RING)[0]["ips"] == ["198.51.100.23"]
    assert build_hostfile(jaccl_nodes, ConnectionMode.JACCL)[0]["ips"] == ["203.0.113.1"]


def test_single_node_hostfile_allows_local_loopback():
    assert build_hostfile([ClusterNode(id="local", lan_ip="127.0.0.1")], ConnectionMode.RING) == [
        {"ips": ["127.0.0.1"], "rdma": []}
    ]
