# Tokenity Lean Stable Baseline

This repository is a standalone Tokenity source tree. It does not use a Git
worktree, source-tree remote, linked files, or a runtime `PYTHONPATH` pointing
at another checkout.

## Supported Runtime Surface

- Tokenity Distributed Server and Single-Mac Tokenity Server.
- OpenAI-compatible streaming and non-streaming chat, cancellation, readiness,
  request admission, queues, timeouts, automatic routing, and instance-scoped
  lifecycle management.
- Native MTP capability detection and launch configuration.
- HTTP Node Agent orchestration, stable `machine_id` discovery, watchdog
  recovery, and single- or two-Mac execution. SSH rank launch is unsupported.
- MiniMax H3 single-Mac and TP2/RDMA launch planning, video generation API,
  SwiftUI generation workflow, progress, cancellation, recovery, and artifact
  handling.
- GLM 5.2, MiniMax H3, Qwen, and other MLX checkpoints selected through model
  metadata and capability inspection rather than directory-name allowlists.

The removed Official MLX-LM backend is not a supported mode. The two supported
server modes share the maintained Tokenity runtime paths above.

## Deployment Contract

Set deployment paths through the variables documented in
`deployment-configuration.md`. Node Agent endpoints come from app settings or
`TOKENITY_NODE_AGENT_URLS`. A remote endpoint is never inferred as loopback;
an unconfigured remote Mac remains unbound. Display hostnames do not replace
the saved Agent origin, and two-Mac H3 readiness requires two different stable
machine IDs.

Builds consume the tracked PNG and ICNS resources directly. They do not
regenerate brand assets or depend on a mounted volume being named or rooted at
a fixed path.

## Verification Baseline

Run from the repository root unless noted:

```bash
python -m pytest -q
(cd apps/TokenityControl && swift test)
TOKENITY_APP_BUNDLE_PATH=/tmp/TokenityControl.app \
  scripts/build-tokenity-control-app.sh
python -m tokenity --help
python -m tokenity.benchmarking.minimax_h3_tp2 --help
```

The local lean validation baseline is 258 passing Python tests with 8 hardware
tests skipped, and 143 passing Swift tests with 1 platform-dependent test
skipped. The debug app bundle, Node Agent health/info API, single-Mac LLM
dry-run, two-Mac HTTP launch plan, and single-Mac H3 dry-run also pass.

## Hardware Validation

The current two-Mac hardware baseline has also completed the following real
model checks:

1. GLM 5.2 reached a `2/2` JACCL/RDMA readiness quorum, completed non-streaming,
   streaming, multi-turn, and queued OpenAI-compatible requests, then stopped
   both ranks with exit code zero.
2. Three consecutive 1,024-token GLM streams were disconnected after their
   first content token. Each runtime failed closed, rejected reuse with HTTP
   503, stopped both ranks normally, reclaimed memory immediately, and loaded a
   fresh instance that completed both streaming and non-streaming inference.
3. MiniMax H3 completed real single-Mac and TP2/RDMA generation through the
   Agent SSE gateway. Both paths published progress, released their request
   slot after client cancellation, generated complete RGB8 video plus PCM audio
   on retry, and stopped without retained reservations or ports.
4. The standard 512x256, 124-frame, 28-step H3 request completed four times in
   one TP2 lifecycle with identical video and audio hashes.

The external hardware reports are stored outside the repository under the
operator's validation-artifact directory. Re-run them after changing MLX,
MLX-LM, the native H3 binary, model weights, or RDMA topology.
