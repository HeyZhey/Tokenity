from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .mlx.hostfile import ClusterNode, ConnectionMode, build_hostfile
from .mlx.launcher import (
    build_distributed_openai_launch_plan,
    build_official_mlx_lm_launch_plan,
)
from .mlx.rdma_probe import probe_rdma


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"tokenity {__version__}")
        return 0

    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    return args.handler(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tokenity")
    parser.add_argument("--version", action="store_true", help="Print Tokenity version.")
    subparsers = parser.add_subparsers(dest="command")

    agent = subparsers.add_parser("node-agent", help="Run the lightweight Node Agent.")
    agent.add_argument("--host", default="127.0.0.1")
    agent.add_argument("--port", type=int, default=9100)
    agent.set_defaults(handler=_run_node_agent)

    rdma = subparsers.add_parser("rdma-probe", help="Probe local RDMA/JACCL readiness.")
    rdma.add_argument("--pretty", action="store_true")
    rdma.set_defaults(handler=_run_rdma_probe)

    hostfile = subparsers.add_parser("hostfile", help="Build an MLX hostfile preview.")
    hostfile.add_argument("--connection", choices=[mode.value for mode in ConnectionMode], default="ring")
    hostfile.add_argument("--nodes-json", help="JSON array of cluster nodes.")
    hostfile.add_argument("--ab-jaccl-fixture", action="store_true", help="Use the known A/B JACCL fixture.")
    hostfile.add_argument("--pretty", action="store_true")
    hostfile.set_defaults(handler=_run_hostfile)

    official = subparsers.add_parser("official-mlx-lm", help="Official MLX-LM compatibility tools.")
    official_sub = official.add_subparsers(dest="official_command")
    official_plan = official_sub.add_parser("launch-plan", help="Preview an experimental mlx_lm server launch.")
    _add_launch_plan_args(official_plan)
    official_plan.set_defaults(handler=_run_official_launch_plan)

    distributed = subparsers.add_parser("distributed-openai", help="Tokenity distributed OpenAI server.")
    distributed_sub = distributed.add_subparsers(dest="distributed_command")
    distributed_serve = distributed_sub.add_parser("serve", help="Run the distributed OpenAI server skeleton.")
    distributed_serve.add_argument("--model", required=True)
    distributed_serve.add_argument("--host", default="127.0.0.1")
    distributed_serve.add_argument("--port", type=int, default=8000)
    distributed_serve.add_argument(
        "--require-mlx",
        action="store_true",
        help="Fail readiness if MLX/MLX-LM imports are unavailable.",
    )
    distributed_serve.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow tokenizer remote code when the selected model needs it.",
    )
    distributed_serve.set_defaults(handler=_run_distributed_serve)

    distributed_plan = distributed_sub.add_parser("launch-plan", help="Preview Tokenity distributed launch.")
    _add_launch_plan_args(distributed_plan)
    distributed_plan.set_defaults(handler=_run_distributed_launch_plan)
    return parser


def _add_launch_plan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--connection", choices=[mode.value for mode in ConnectionMode], default="ring")
    parser.add_argument("--nodes-json", help="JSON array of cluster nodes.")
    parser.add_argument("--ab-jaccl-fixture", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--starting-port", type=int, default=29500)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--pretty", action="store_true")


def _run_node_agent(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("tokenity.node_agent.agent:create_app", factory=True, host=args.host, port=args.port)
    return 0


def _run_rdma_probe(args: argparse.Namespace) -> int:
    result = probe_rdma()
    print(_json(result.to_dict(), pretty=args.pretty))
    return 0


def _run_hostfile(args: argparse.Namespace) -> int:
    nodes = _load_nodes(args)
    mode = ConnectionMode(args.connection)
    hostfile = build_hostfile(nodes, mode)
    print(_json(hostfile, pretty=args.pretty))
    return 0


def _run_official_launch_plan(args: argparse.Namespace) -> int:
    nodes = _load_nodes(args)
    plan = build_official_mlx_lm_launch_plan(
        nodes=nodes,
        connection_mode=ConnectionMode(args.connection),
        model=args.model,
        python=args.python,
        starting_port=args.starting_port,
        host=args.host,
        port=args.port,
    )
    print(_json(plan.to_dict(), pretty=args.pretty))
    return 0


def _run_distributed_launch_plan(args: argparse.Namespace) -> int:
    nodes = _load_nodes(args)
    plan = build_distributed_openai_launch_plan(
        nodes=nodes,
        connection_mode=ConnectionMode(args.connection),
        model=args.model,
        python=args.python,
        starting_port=args.starting_port,
        host=args.host,
        port=args.port,
    )
    print(_json(plan.to_dict(), pretty=args.pretty))
    return 0


def _run_distributed_serve(args: argparse.Namespace) -> int:
    from .serving.distributed_openai import serve

    serve(
        model=args.model,
        host=args.host,
        port=args.port,
        require_mlx=args.require_mlx,
        trust_remote_code=args.trust_remote_code,
    )
    return 0


def _load_nodes(args: argparse.Namespace) -> list[ClusterNode]:
    if getattr(args, "ab_jaccl_fixture", False):
        return [
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
    raw = getattr(args, "nodes_json", None)
    if not raw:
        raise SystemExit("--nodes-json or --ab-jaccl-fixture is required")
    if Path(raw).exists():
        payload = json.loads(Path(raw).read_text())
    else:
        payload = json.loads(raw)
    if not isinstance(payload, list):
        raise SystemExit("--nodes-json must be a JSON array")
    return [ClusterNode.from_mapping(item) for item in payload]


def _json(payload: Any, *, pretty: bool = False) -> str:
    return json.dumps(payload, indent=2 if pretty else None, sort_keys=pretty)


if __name__ == "__main__":
    raise SystemExit(main())
