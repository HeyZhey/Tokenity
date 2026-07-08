# Tokenity

Tokenity is a clean rebuild of the Mac-native control plane for multi-machine
MLX/MLX-LM inference. This repository intentionally starts from a fresh
workspace and treats the old `MLX-Distributed` tree only as read-only product
and experiment context.

## First Milestone

- Python package and CLI: `tokenity --version`, `tokenity node-agent`
- RDMA/JACCL probe with parser-focused tests
- MLX hostfile builder with the known two-Mac JACCL topology fixture
- Node Agent FastAPI surface for node info, model scan, status, launch-plan
  dry runs, and local process supervision hooks
- Fresh SwiftUI macOS app skeleton in `apps/TokenityControl`
- Experimental official `mlx_lm server` launch-plan preview
- Tokenity-owned distributed OpenAI server skeleton with readiness phases

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

Run the Node Agent:

```bash
tokenity node-agent --host 0.0.0.0 --port 9100
```

Build the SwiftUI console:

```bash
cd apps/TokenityControl
swift build
```

## Safety Notes

- SSH passwords are never stored in source, config, or logs.
- JACCL launch plans are blocked unless each selected node has RDMA device and
  Thunderbolt IP data.
- Official `mlx_lm server` mode is labelled Experimental because the target
  A/B multi-node chat path has not yet been stable for large models.

