from __future__ import annotations

import pytest

from tokenity.mlx.hostfile import ClusterNode, ConnectionMode, HostfileError, build_hostfile


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
