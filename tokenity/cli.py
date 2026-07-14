from __future__ import annotations

import argparse
import json
from typing import Any

from . import __version__
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

    distributed = subparsers.add_parser("distributed-openai", help="Tokenity distributed OpenAI server.")
    distributed_sub = distributed.add_subparsers(dest="distributed_command")
    distributed_serve = distributed_sub.add_parser("serve", help="Run the distributed OpenAI-compatible model server.")
    distributed_serve.add_argument("--model", required=True)
    distributed_serve.add_argument("--host", default="127.0.0.1")
    distributed_serve.add_argument("--port", type=int, default=8000)
    distributed_serve.add_argument("--api-identifier")
    distributed_serve.add_argument("--max-tokens", type=int, default=32_768)
    distributed_serve.add_argument("--prompt-cache-size", type=int, default=4)
    distributed_serve.add_argument("--prefill-step-size", type=int, default=2_048)
    distributed_serve.add_argument("--decode-concurrency", type=int, default=1)
    distributed_serve.add_argument("--prompt-concurrency", type=int, default=1)
    distributed_serve.add_argument(
        "--native-mtp-mode",
        choices=("off", "auto", "required"),
        default="off",
    )
    distributed_serve.add_argument("--native-mtp-max-depth", type=int, choices=(1,), default=1)
    distributed_serve.add_argument(
        "--native-mtp-head-placement",
        choices=("replicated",),
        default="replicated",
    )
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

    return parser


def _run_node_agent(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("tokenity.node_agent.agent:create_app", factory=True, host=args.host, port=args.port)
    return 0


def _run_rdma_probe(args: argparse.Namespace) -> int:
    result = probe_rdma()
    print(_json(result.to_dict(), pretty=args.pretty))
    return 0


def _run_distributed_serve(args: argparse.Namespace) -> int:
    from .serving.distributed_openai import serve

    serve(
        model=args.model,
        host=args.host,
        port=args.port,
        require_mlx=args.require_mlx,
        trust_remote_code=args.trust_remote_code,
        api_identifier=args.api_identifier,
        max_tokens=args.max_tokens,
        prompt_cache_size=args.prompt_cache_size,
        prefill_step_size=args.prefill_step_size,
        decode_concurrency=args.decode_concurrency,
        prompt_concurrency=args.prompt_concurrency,
        native_mtp={
            "mode": args.native_mtp_mode,
            "max_depth": args.native_mtp_max_depth,
            "head_placement": args.native_mtp_head_placement,
        },
    )
    return 0


def _json(payload: Any, *, pretty: bool = False) -> str:
    return json.dumps(payload, indent=2 if pretty else None, sort_keys=pretty)


if __name__ == "__main__":
    raise SystemExit(main())
