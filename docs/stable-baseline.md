# Tokenity Stable Baseline

## Purpose

This workspace is the single source of truth for subsequent Tokenity work:

```text
/path/to/Tokenity-Stable
```

It is a Git worktree on branch `stable-baseline`, forked from Tokenity commit
`a39ddc7`. The original `/path/to/Tokenity` worktree and the legacy
`/path/to/MLX-Distributed` directory remain untouched.

## Included Baseline

- The current SwiftUI app from `Tokenity/apps/TokenityControl`.
- The distributed MLX-LM/OpenAI runtime that loaded
  `Qwen3.5-122B-A10B-4bit` across Mango and Kiwi.
- HTTP Node Agent rank orchestration, NodeAgent memory reporting, child-rank
  failure detection, and distributed process cleanup behavior.
- The current DMG/pkg builder, now sourcing both UI and backend from this one
  repository.

The validated cluster topology remains:

- Mango: `<node-a-user>@<node-a-lan-ip>`, `en4`, `rdma_en4`, `<node-a-rdma-ip>`
- Kiwi: `<node-b-user>@<node-b-lan-ip>`, `en5`, `rdma_en5`, `<node-b-rdma-ip>`
- Runtime Python: `${TOKENITY_RUNTIME_PYTHON}`
- Model: `${TOKENITY_MODEL_ROOT}/Qwen3.5-122B-A10B-4bit`

The consolidated backend was checked against `${TOKENITY_CODE_ROOT}` on
both Mango and Kiwi. All three core files matched byte-for-byte:

- `distributed_openai.py`: `bd014e1e2938ed0583b84343850c718170a20c6fd87dff773af554d0f4ebb306`
- `launcher.py`: `a389816795df339193654c743bfa5d309ffb3874c4576ebb21672eed582a8487`
- `agent.py`: `8a9f30703291eaf84ddfc6076d3820a5fc11800a6f8f1a2979e54cdd2fb85d1f`

## Baseline Invariants

1. Do not edit either UI or backend under `MLX-Distributed`.
2. Build and launch the app through scripts in this worktree.
3. Installer payloads must copy backend code from this worktree.
4. `tokenity.serving.distributed_openai.TokenityDistributedRuntime` must be
   present; the skeleton-only server is not a distributable backend.
5. Run `./scripts/verify-stable-baseline.sh` before committing later fixes.

## Start the UI

```bash
cd /path/to/Tokenity-Stable
./scripts/run-tokenity-control-app.sh
```

This rebuilds the app bundle before opening it, preventing stale executable
code from being mistaken for current source.

## Scope of Stability

This baseline consolidates the known-working UI/backend deployment source.
Model-specific runtime and generation settings are available from the Models
page, and non-conversational UI messages are excluded from model context.
State recovery after app/NodeAgent restart and Agent/API authentication remain
explicit follow-up work. Product rank startup no longer uses SSH.
