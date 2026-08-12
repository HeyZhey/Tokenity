from __future__ import annotations

import ipaddress
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence


CommandRunner = Callable[[Sequence[str]], "CommandResult"]


@dataclass
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.returncode == 0


@dataclass
class NetworkInterface:
    name: str
    ipv4: list[str] = field(default_factory=list)
    status: str = "unknown"


@dataclass
class RDMAProbeResult:
    rdma_enabled: bool
    rdma_devices: list[str]
    rdma_port_state: dict[str, str]
    thunderbolt_ip: str | None
    rdma_errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "rdma_enabled": self.rdma_enabled,
            "rdma_devices": self.rdma_devices,
            "rdma_port_state": self.rdma_port_state,
            "thunderbolt_ip": self.thunderbolt_ip,
            "rdma_errors": self.rdma_errors,
        }


def probe_rdma(runner: CommandRunner | None = None) -> RDMAProbeResult:
    run = runner or default_runner
    rdma_ctl = run(("rdma_ctl", "status"))
    ibv_devices = run(("ibv_devices",))
    rdma_ctl_devices, rdma_ctl_states = parse_rdma_ctl_status(rdma_ctl.stdout)
    # Apple's ibv_devinfo can remain in an uninterruptible kernel wait after a
    # JACCL runtime exits. rdma_ctl is authoritative when it already reports
    # port state, so avoid the redundant probe in the normal macOS path.
    has_authoritative_port_state = bool(rdma_ctl_devices) and all(
        rdma_ctl_states.get(device) in {"active", "down"}
        for device in rdma_ctl_devices
    )
    ibv_devinfo = (
        CommandResult(("ibv_devinfo",), 0)
        if has_authoritative_port_state
        else run(("ibv_devinfo",))
    )
    results = {
        "rdma_ctl": rdma_ctl,
        "ibv_devices": ibv_devices,
        "ibv_devinfo": ibv_devinfo,
        "ifconfig": run(("ifconfig",)),
    }

    errors: list[str] = []
    for name, result in results.items():
        if result.error:
            errors.append(f"{name}: {result.error}")
        elif result.returncode != 0:
            message = result.stderr.strip() or f"exit code {result.returncode}"
            errors.append(f"{name}: {message}")

    devices: set[str] = set()
    states: dict[str, str] = {}

    devices.update(rdma_ctl_devices)
    states.update(rdma_ctl_states)

    devices.update(parse_ibv_devices(results["ibv_devices"].stdout))
    devinfo_states = parse_ibv_devinfo(results["ibv_devinfo"].stdout)
    devices.update(devinfo_states)
    states.update(devinfo_states)

    interfaces = parse_ifconfig(results["ifconfig"].stdout)
    ordered_devices = sorted(devices, key=_natural_device_key)
    active_devices = [dev for dev in ordered_devices if states.get(dev, "unknown") == "active"]
    thunderbolt_ip = infer_thunderbolt_ip(interfaces, active_devices or ordered_devices)

    if not ordered_devices:
        errors.append("No RDMA devices were reported by rdma_ctl or ibv tools.")
    if ordered_devices and not active_devices:
        errors.append("RDMA devices were found, but no active RDMA port was detected.")
    if ordered_devices and not thunderbolt_ip:
        errors.append("No Thunderbolt IPv4 address could be associated with an RDMA device.")

    return RDMAProbeResult(
        rdma_enabled=bool(active_devices and thunderbolt_ip),
        rdma_devices=active_devices or ordered_devices,
        rdma_port_state={dev: states.get(dev, "unknown") for dev in ordered_devices},
        thunderbolt_ip=thunderbolt_ip,
        rdma_errors=errors,
    )


def default_runner(command: Sequence[str]) -> CommandResult:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return CommandResult(tuple(command), 127, error=f"{command[0]} not found")
    except subprocess.TimeoutExpired:
        return CommandResult(tuple(command), 124, error="timed out")
    except OSError as exc:
        return CommandResult(tuple(command), 1, error=str(exc))
    return CommandResult(tuple(command), completed.returncode, completed.stdout, completed.stderr)


def parse_rdma_ctl_status(text: str) -> tuple[set[str], dict[str, str]]:
    devices: set[str] = set()
    states: dict[str, str] = {}
    for line in text.splitlines():
        names = re.findall(r"\brdma_en\d+\b", line)
        if not names:
            continue
        lowered = line.lower()
        state = "unknown"
        if "active" in lowered or re.search(r"\bup\b", lowered):
            state = "active"
        elif "down" in lowered or "inactive" in lowered:
            state = "down"
        for name in names:
            devices.add(name)
            states[name] = state
    return devices, states


def parse_ibv_devices(text: str) -> set[str]:
    devices: set[str] = set()
    for line in text.splitlines():
        devices.update(re.findall(r"\brdma_en\d+\b", line))
    return devices


def parse_ibv_devinfo(text: str) -> dict[str, str]:
    states: dict[str, str] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = re.search(r"hca_id:\s*(rdma_en\d+)", line)
        if match:
            current = match.group(1)
            states.setdefault(current, "unknown")
            continue
        if current and re.search(r"\bstate\s*:", line, flags=re.IGNORECASE):
            lowered = line.lower()
            if "port_active" in lowered or re.search(r"\bactive\b", lowered):
                states[current] = "active"
            elif "down" in lowered or "port_down" in lowered:
                states[current] = "down"
    return states


def parse_ifconfig(text: str) -> dict[str, NetworkInterface]:
    interfaces: dict[str, NetworkInterface] = {}
    current: NetworkInterface | None = None
    for raw_line in text.splitlines():
        header = re.match(r"^([a-zA-Z0-9_.-]+):\s+flags=", raw_line)
        if header:
            current = NetworkInterface(name=header.group(1))
            interfaces[current.name] = current
            continue
        if current is None:
            continue
        stripped = raw_line.strip()
        inet = re.match(r"inet\s+(\d+\.\d+\.\d+\.\d+)\b", stripped)
        if inet and not inet.group(1).startswith("127."):
            current.ipv4.append(inet.group(1))
        status = re.match(r"status:\s*(\w+)", stripped)
        if status:
            current.status = status.group(1).lower()
    return interfaces


def infer_thunderbolt_ip(
    interfaces: dict[str, NetworkInterface],
    rdma_devices: Iterable[str],
) -> str | None:
    for device in rdma_devices:
        iface = interfaces.get(device) or interfaces.get(device.replace("rdma_", ""))
        if iface and iface.ipv4:
            return _prefer_private_link_ip(iface.ipv4)

    candidate_ips: list[str] = []
    for iface in interfaces.values():
        if iface.status not in {"active", "unknown"}:
            continue
        candidate_ips.extend(ip for ip in iface.ipv4 if _looks_like_thunderbolt_link(ip))
    if candidate_ips:
        return _prefer_private_link_ip(candidate_ips)
    return None


def _prefer_private_link_ip(ips: list[str]) -> str:
    private_tb = [ip for ip in ips if _is_private_thunderbolt_ip(ip)]
    if private_tb:
        return sorted(private_tb, key=lambda ip: ipaddress.ip_address(ip))[0]
    link_like = [ip for ip in ips if _looks_like_thunderbolt_link(ip)]
    return sorted(link_like or ips, key=lambda ip: ipaddress.ip_address(ip))[0]


def _looks_like_thunderbolt_link(ip: str) -> bool:
    if _is_private_thunderbolt_ip(ip):
        return True
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return address.is_link_local


def _is_private_thunderbolt_ip(ip: str) -> bool:
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return False
    configured = os.environ.get("TOKENITY_RDMA_CIDRS")
    if configured:
        networks = []
        for value in configured.split(","):
            try:
                networks.append(ipaddress.ip_network(value.strip(), strict=False))
            except ValueError:
                continue
        return any(address in network for network in networks)
    return address.is_private and not (
        address.is_loopback or address.is_link_local or address.is_unspecified
    )


def _natural_device_key(device: str) -> tuple[str, int]:
    match = re.match(r"([a-z_]+)(\d+)$", device)
    if not match:
        return device, -1
    return match.group(1), int(match.group(2))
