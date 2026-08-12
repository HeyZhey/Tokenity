from __future__ import annotations

from tokenity.mlx.rdma_probe import CommandResult, infer_thunderbolt_ip, parse_ifconfig, probe_rdma


RDMA_CTL = """
RDMA enabled
rdma_en4: active
"""

IBV_DEVICES = """
device                 node GUID
------              ----------------
rdma_en4            00:00:00:00:00:00:00:04
"""

IBV_DEVINFO = """
hca_id: rdma_en4
    transport:                      InfiniBand (0)
    port:   1
        state:                  PORT_ACTIVE (4)
"""

IFCONFIG = """
lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
    inet 127.0.0.1 netmask 0xff000000
rdma_en4: flags=8863<UP,BROADCAST,RUNNING,SIMPLEX,MULTICAST> mtu 1500
    inet 203.0.113.1 netmask 0xffffff00 broadcast 203.0.113.255
    status: active
en0: flags=8863<UP,BROADCAST,RUNNING,SIMPLEX,MULTICAST> mtu 1500
    inet 198.51.100.23 netmask 0xffffff00 broadcast 198.51.100.255
    status: active
"""


def test_probe_rdma_merges_tools_and_thunderbolt_ip():
    calls = []

    def runner(command):
        calls.append(command[0])
        name = command[0]
        outputs = {
            "rdma_ctl": RDMA_CTL,
            "ibv_devices": IBV_DEVICES,
            "ibv_devinfo": IBV_DEVINFO,
            "ifconfig": IFCONFIG,
        }
        return CommandResult(tuple(command), 0, stdout=outputs[name])

    result = probe_rdma(runner)

    assert result.rdma_enabled is True
    assert result.rdma_devices == ["rdma_en4"]
    assert result.rdma_port_state == {"rdma_en4": "active"}
    assert result.thunderbolt_ip == "203.0.113.1"
    assert result.rdma_errors == []
    assert calls == ["rdma_ctl", "ibv_devices", "ifconfig"]


def test_probe_rdma_uses_devinfo_when_rdma_ctl_has_no_port_state():
    calls = []

    def runner(command):
        calls.append(command[0])
        outputs = {
            "rdma_ctl": "RDMA enabled\n",
            "ibv_devices": IBV_DEVICES,
            "ibv_devinfo": IBV_DEVINFO,
            "ifconfig": IFCONFIG,
        }
        return CommandResult(tuple(command), 0, stdout=outputs[command[0]])

    result = probe_rdma(runner)

    assert result.rdma_enabled is True
    assert result.rdma_port_state == {"rdma_en4": "active"}
    assert calls == ["rdma_ctl", "ibv_devices", "ibv_devinfo", "ifconfig"]


def test_parse_ifconfig_groups_interfaces():
    interfaces = parse_ifconfig(IFCONFIG)
    assert interfaces["rdma_en4"].ipv4 == ["203.0.113.1"]
    assert interfaces["en0"].status == "active"


def test_infer_thunderbolt_ip_maps_rdma_device_to_network_interface():
    interfaces = parse_ifconfig(
        """
en5: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
    inet 203.0.113.2 netmask 0xfffffffc broadcast 203.0.113.3
    inet 169.254.26.13 netmask 0xffff0000 broadcast 169.254.255.255
    status: active
"""
    )

    assert infer_thunderbolt_ip(interfaces, ["rdma_en5"]) == "203.0.113.2"


def test_infer_thunderbolt_ip_honors_configured_cidrs(monkeypatch):
    monkeypatch.setenv("TOKENITY_RDMA_CIDRS", "203.0.113.0/24")
    interfaces = parse_ifconfig(
        """
en9: flags=8863<UP,BROADCAST,RUNNING,SIMPLEX,MULTICAST> mtu 1500
    inet 203.0.113.9 netmask 0xffffff00 broadcast 203.0.113.255
    status: active
"""
    )

    assert infer_thunderbolt_ip(interfaces, ["rdma_en99"]) == "203.0.113.9"
