# Unified H3 and Stability Merge Validation

## Scope

This project merges the MiniMax H3 video branch and the language-inference
stability branch without modifying either source checkout. The merge keeps the
stable language runtime as the lifecycle authority and adds H3 as a first-class
video modality with an isolated worker Agent and collective-port range.

## Deliberate merge contracts

- Language instances retain per-instance leases, admission, readiness quorum,
  gateway routing, process journals, restart adoption, and UI operation fences.
- MiniMax H3 retains typed single-Mac/TP2 launch, native runtime fingerprinting,
  shard validation, SSE generation, artifact writing, benchmark evidence, and
  coordinated stop.
- Text collectives default to `30020+`; H3 collectives default to `30096+`.
- The stable worker Agent remains on `:9100`; an isolated H3 worker may run on
  `:9200` with its own state root and watchdog.
- A native H3 process is never adopted after an Agent restart because it cannot
  prove Tokenity's runtime epoch. Its journal is used for PID/start-time/command
  fenced cleanup, including precise remote-rank cleanup.
- Global unjournaled recovery scans only the language-server marker, preventing
  one Agent from claiming an H3 process owned by another Agent.
- H3 UI refresh checks complete rank quorum, not only the surviving coordinator
  process, and an in-flight video Load/Stop/Load sequence uses an operation fence.

## Model validation

The GLM-5.2 compatibility suite covers its 78-layer cross-layer DSA schedule,
load barriers, repeated-request JACCL failure handling, stream cancellation,
runtime retirement, and recovery behavior.

Read-only cluster validation on 2026-08-11 found the same
`GLM-5.2-mxfp4` checkpoint on both inference Macs:

- revision: `411bb8dd51cbb32be80e3ad12bf3f8dc6f2a0462efead45dd79201d0477b0ce0`
- size: `395114879612` bytes
- architecture: `GlmMoeDsaForCausalLM`
- format: MLX, 4-bit group size 32, 76 shards

A typed two-Mac dry-run succeeded and automatically changed requested direct
`jaccl` to `jaccl-ring`, producing Rank 0/Rank 1 HTTP launch plans with no SSH.
A destructive full GLM load was intentionally not attempted because Rank 0 had
an existing H3 process and reported no active RDMA port at sampling time.

The local development Mac exposes Qwen3.5 4B checkpoints, but its installed
Node Agent runtime does not contain MLX/MLX-LM. The unified Agent correctly
failed the real-load preflight with HTTP 412 rather than starting a partial
runtime.

## Verification commands

```bash
python -m pytest -q
swift test --package-path apps/TokenityControl
./scripts/build-tokenity-control-app.sh
```

For a post-deployment GLM hardware run, first require both `/health` endpoints,
matching model revisions, active RDMA devices, empty conflicting reservations,
and no existing H3 generation. Start with a dry-run, use a unique instance ID,
run repeated fixed prompts (including cancellation), query instance quorum
after every request, and finish with the instance-specific stop endpoint.
