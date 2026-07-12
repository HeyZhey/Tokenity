# Tokenity Stable Baseline

## Purpose

This workspace is the single source of truth for subsequent Tokenity work:

```text
/Users/zxc/Documents/Tokenity-Stable
```

It is a Git worktree on branch `stable-baseline`, forked from Tokenity commit
`a39ddc7`. The original `/Users/zxc/Documents/Tokenity` worktree and the legacy
`/Users/zxc/Documents/MLX-Distributed` directory remain untouched.

## Included Baseline

- The current SwiftUI app from `Tokenity/apps/TokenityControl`.
- The distributed MLX-LM/OpenAI runtime that loaded
  `Qwen3.5-122B-A10B-4bit` across Mango and Kiwi.
- The matching launcher environment, local-SSH wrapper, NodeAgent memory
  reporting, child-rank failure detection, and process cleanup behavior.
- The current DMG/pkg builder, now sourcing both UI and backend from this one
  repository.

The validated cluster topology remains:

- Mango: `apple@192.168.5.23`, `en4`, `rdma_en4`, `192.168.0.1`
- Kiwi: `probriefing@192.168.5.75`, `en5`, `rdma_en5`, `192.168.0.2`
- Runtime Python: `/Users/Shared/TokenityRuntime/current/.venv/bin/python`
- Model: `/Users/Shared/TokenityModels/Qwen3.5-122B-A10B-4bit`

The consolidated backend was checked against `/Users/Shared/TokenityCode` on
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
cd /Users/zxc/Documents/Tokenity-Stable
./scripts/run-tokenity-control-app.sh
```

This rebuilds the app bundle before opening it, preventing stale executable
code from being mistaken for current source.

## Scope of Stability

This baseline consolidates the known-working UI/backend deployment source. It
does not declare all product bugs fixed. In particular, chat-history semantics,
state recovery after app/NodeAgent restart, installer service identity and
authentication remain explicit follow-up work.
