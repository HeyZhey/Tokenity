from __future__ import annotations

import pytest

from tokenity.mlx.hostfile import ClusterNode, ConnectionMode, HostfileError, build_hostfile


def test_known_two_mac_jaccl_hostfile_shape():
    nodes = [
        ClusterNode(
            id="mac-a",
            ssh="127.0.0.1",
            lan_ip="192.168.5.23",
            rdma_ip="192.168.0.1",
            rdma_devices=["rdma_en4"],
        ),
        ClusterNode(
            id="mac-b",
            ssh="probriefing@192.168.5.75",
            lan_ip="192.168.5.75",
            rdma_ip="192.168.0.2",
            rdma_devices=["rdma_en5"],
        ),
    ]

    assert build_hostfile(nodes, ConnectionMode.JACCL) == [
        {"ssh": "127.0.0.1", "ips": ["192.168.0.1"], "rdma": [None, "rdma_en4"]},
        {"ssh": "probriefing@192.168.5.75", "ips": [], "rdma": ["rdma_en5", None]},
    ]


def test_jaccl_blocks_missing_rdma_data():
    nodes = [
        ClusterNode(id="mac-a", ssh="127.0.0.1", lan_ip="192.168.5.23"),
        ClusterNode(id="mac-b", ssh="probriefing@192.168.5.75", lan_ip="192.168.5.75"),
    ]

    with pytest.raises(HostfileError, match="JACCL readiness failed"):
        build_hostfile(nodes, ConnectionMode.JACCL)

