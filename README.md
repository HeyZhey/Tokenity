# Tokenity

Tokenity is the Mac-native control plane for multi-machine MLX/MLX-LM
inference. The `stable-baseline` branch combines the current SwiftUI control
app with the distributed Qwen runtime that was validated on the Mango/Kiwi
two-Mac cluster. The old `MLX-Distributed` tree is now read-only migration
history; active development belongs in this repository.

The isolated development worktree for this baseline is:

```text
/Users/zxc/Documents/Tokenity-Stable
```

## First Milestone

- Python package and CLI: `tokenity --version`, `tokenity node-agent`
- RDMA/JACCL probe with parser-focused tests
- MLX hostfile builder with the known two-Mac JACCL topology fixture
- Node Agent FastAPI surface for node info, model scan, status, launch-plan
  dry runs, and local process supervision hooks
- Fresh SwiftUI macOS app skeleton in `apps/TokenityControl`
- Experimental official `mlx_lm server` launch-plan preview
- Tokenity-owned distributed OpenAI server with sharded MLX-LM loading,
  OpenAI-compatible streaming, reasoning output, and readiness phases

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

Run the complete baseline verification (Python tests, Swift tests, and a fresh
app bundle build):

```bash
./scripts/verify-stable-baseline.sh
```

## Model Configuration

Each model row in the Models page has a Configure action. Settings are saved
per model and applied to later loads and chat requests. The default maximum
output is `32768` tokens so reasoning models have room to finish their final
answer. Configurable settings include the API identifier, sampling controls,
prefill step size, prompt cache size, prompt/decode concurrency, and tokenizer
remote-code trust.

## Safety Notes

- SSH passwords are never stored in source, config, or logs.
- JACCL launch plans are blocked unless each selected node has RDMA device and
  Thunderbolt IP data.
- Official `mlx_lm server` mode is labelled Experimental because the target
  A/B multi-node chat path has not yet been stable for large models.
