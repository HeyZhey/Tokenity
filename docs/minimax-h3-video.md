# MiniMax H3 Video Inference

Tokenity can supervise a native MLX MiniMax H3 video server on one Mac and
expose it through the long-lived Node Agent on port `9100`. Video requests are
forwarded without buffering, so native `text/event-stream` progress events keep
their original cadence.

## Current support boundary

| Path | State | Behavior |
| --- | --- | --- |
| Native CLI, one Mac | Implemented | Validates the H3 checkpoint and executes `mlx-serve` without a shell. |
| Node Agent, one Mac | Implemented | Reserves memory/port, supervises the process, checks `/health`, and publishes an instance. |
| Video gateway | Implemented | Routes `POST /v1/video/generations`, preserves SSE, and accounts for request leases. |
| Two-Mac TP2 plan | Implemented | Produces typed rank requests, JACCL/RDMA topology, and exact commands. |
| Two-Mac TP2 execution | Implemented and hardware validated | Starts Rank 0 locally and Rank 1 through its Node Agent, waits for the JACCL data plane and Rank 0 `/health`, and rolls both ranks back on failure. |

This is block-wise Tensor Parallel rather than two independent full-model
servers. Startup still fails closed with HTTP `412` when either Agent, native
protocol v1, checkpoint manifest, shard hash, or model/code revision is absent
or inconsistent. The target two-Mac hardware passed the then-current
2,800-collective gate, the standard 124-frame generation, and coordinated
shutdown on 2026-08-09.

## TokenityControl Video workspace

The macOS app exposes MiniMax H3 both in the unified **Models** library and as
a first-class **Video** workspace. Models refreshes H3 Agent availability and
uses the same Load/Stop lifecycle as language models; **Open Video** switches
to the modality-specific generation and runtime controls. H3 remains excluded
from Chat auto-routing because it is a video model. The Video workspace owns
the full two-Mac workflow rather than requiring manual `curl` commands:

1. Refresh and validate the coordinator and worker Node Agents, advertised H3
   capability, and Thunderbolt RDMA devices.
2. Start a single-Mac or TP2 runtime through the typed
   `start-minimax-h3-video` contract and show the rank topology and lifecycle.
3. Submit prompt, canvas, frame, step, seed, and fast-mode controls while
   displaying native SSE progress.
4. Convert the completed RGB8 and PCM response into a playable H.264 MOV with
   audio, while preserving raw RGB, PCM/WAV, and request metadata beside it.

The deployment defaults are intentionally unbound: the coordinator is the
loopback Agent and the worker is empty. Both H3 endpoints, model path, native
binary path, and optimization profile are editable in the Video page's
configuration disclosure. This prevents a new installation from contacting a
previously validated lab topology.

Video runtime stop, app shutdown, and lease renewal are instance-scoped. The
coordinator fans the precise operation out to Rank 1, so the UI never widens an
H3 action into a global `stop-all` that could affect a sibling model service.
The UI also verifies the complete H3 rank quorum during refresh; a surviving
Rank 0 can no longer appear ready after Rank 1 has failed.

The deployment defaults can be overridden before starting TokenityControl with
`TOKENITY_H3_COORDINATOR_AGENT`, `TOKENITY_H3_WORKER_AGENT`,
`TOKENITY_H3_MODEL_PATH`, `TOKENITY_H3_BINARY_PATH`, and
`TOKENITY_CONTROL_H3_STARTING_PORT`.

Native H3 does not publish Tokenity's language-runtime epoch status. The Agent
therefore journals its exact PID start identity, command, model, ports, and
worker endpoints, but deliberately does not adopt it after an Agent restart.
It performs fenced, instance-specific cleanup and leaves unrelated processes
untouched. The isolated `:9200` service uses a separate state root and disables
the global unjournaled scan so the stable `:9100` Agent cannot claim it.

H3 uses collective ports beginning at `30096`, independently of the text
runtime range beginning at `30020`. A live 2026-08-10 regression reproduced
the Rank 1 JACCL connection timeout on `30020`; the identical Agents, runtime,
checkpoint, and RDMA topology reached `2/2` Ready on `30096` and then stopped
cleanly. Startup failures are read back from both rank instance records so the
UI reports the failing stage instead of collapsing it to an HTTP 500 label.

## Runtime prerequisites

The model directory must have `model_type: "minimax_h3"` in `config.json`.
Single-node and Rank 0 execution require these component files:

- `transformer.safetensors`
- `text_encoder.safetensors`
- `video_vae.safetensors`
- `audio_vae.safetensors`

Two-Mac execution additionally requires a common manifest describing both
ranks:

- `tp2/manifest.json` with schema/protocol version `1` and world size `2`
- the exact file sizes and SHA-256 values emitted by the offline sharder

Each machine only needs its own `tp2/rank-N/transformer.safetensors` shard.
Rank 1 does not need the full transformer, text encoder, or VAEs; Rank 0 keeps
those components because it owns conditioning and media output.

The default native executable is derived from `TOKENITY_RUNTIME_ROOT`; set
`TOKENITY_H3_BINARY_PATH` to override it.

## Direct native wrapper

```bash
tokenity minimax-h3-video serve \
  --binary "$TOKENITY_H3_BINARY_PATH" \
  --model "$TOKENITY_H3_MODEL_PATH" \
  --host 0.0.0.0 \
  --port 11241
```

The wrapper validates the binary and checkpoint, sets the MLX synchronization
environment, then replaces itself with the native process. It does not invoke a
shell and does not fall back to PyTorch or MPS.

## Start through the Node Agent

First inspect the exact launch plan without changing machine state:

```bash
curl -X POST "${TOKENITY_H3_COORDINATOR_AGENT}/v1/node/start-minimax-h3-video" \
  -H 'Content-Type: application/json' \
  -d @- <<JSON
  {
    "model": "${TOKENITY_H3_MODEL_PATH}",
    "binary": "${TOKENITY_H3_BINARY_PATH}",
    "api_identifier": "minimax-h3-video",
    "dry_run": true
  }
JSON
```

Set `dry_run` to `false` to start it. Tokenity validates the native server,
allocates a private backend port near `11241`, waits for `GET /health`, and only
then marks the instance ready.

## Generate video

Send the native H3 request body to the stable Node Agent gateway. The optional
`tokenity_instance_id` field selects one Tokenity instance and is removed before
the request reaches the native server.

```bash
curl -N http://127.0.0.1:9100/v1/video/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minimax-h3-video",
    "prompt": "A cinematic tracking shot of a paper boat crossing a rainy neon street",
    "width": 768,
    "height": 512,
    "num_frames": 56,
    "steps": 30,
    "seed": 42,
    "stream": true
  }'
```

The native H3 API remains the authority for generation fields. Tokenity only
adds routing, lifecycle, readiness, queueing, and transparent response
forwarding.

## Two-Mac TP2 execution

A two-node request accepts the same node descriptors used by Tokenity's text
models. The data plane is JACCL ring over Thunderbolt RDMA. Before execution,
the native binary must advertise all three CLI arguments in `--help`:

```text
--h3-distributed-rank
--h3-distributed-world-size
--h3-distributed-protocol
```

Tokenity currently requires protocol version `1`. Rank 0 owns the public HTTP
API, text encoder, initial latent construction, VAE/audio decode, and muxing.
Rank 1 owns no public port; it receives conditioning and initial latents, loads
its TP shard, and participates in the 50 main DiT blocks. Each full baseline
block executes one attention output-projection all-sum and one MLP FC2 all-sum:
100 collectives per denoiser forward. H3's `steps` value counts denoiser
forwards; the scheduler adds a terminal zero sigma. Therefore `steps=28` runs
28 forwards and 2,800 collectives with `"fast": false`.

```bash
curl -X POST "${TOKENITY_H3_COORDINATOR_AGENT}/v1/node/start-minimax-h3-video" \
  -H 'Content-Type: application/json' \
  -d @- <<JSON
  {
    "model": "${TOKENITY_H3_MODEL_PATH}",
    "binary": "${TOKENITY_H3_BINARY_PATH}",
    "connection_mode": "jaccl-ring",
    "starting_port": 30096,
    "dry_run": false,
    "lease_seconds": 300,
    "nodes": [
      {"id":"node-a","agent_url":"${TOKENITY_H3_COORDINATOR_AGENT}","lan_ip":"${TOKENITY_NODE_A_LAN_IP}","rdma_ip":"${TOKENITY_NODE_A_RDMA_IP}","rdma_devices":["${TOKENITY_NODE_A_RDMA_DEVICE}"]},
      {"id":"node-b","agent_url":"${TOKENITY_H3_WORKER_AGENT}","lan_ip":"${TOKENITY_NODE_B_LAN_IP}","rdma_ip":"${TOKENITY_NODE_B_RDMA_IP}","rdma_devices":["${TOKENITY_NODE_B_RDMA_DEVICE}"]}
    ]
  }
JSON
```

The standard first hardware run is `256x512`, `124` frames, `28` denoiser
forwards (29 sigma grid points), `"fast": false`, and a fixed seed. Protocol v1
intentionally rejects H3 keyframes in TP2. If an SSE client disconnects during
a distributed generation, both ranks finish that in-flight job to preserve
collective ordering; subsequent jobs remain usable.

## Validated hardware results

On 2026-08-09 the two 512 GiB Macs passed the formal 2,800-call collective
gate in `19.499 s` on the slower rank (`6.943 ms` p50, `6.991 ms` p95,
`7.405 GiB/s`). A fresh paired benchmark on 2026-08-26 used the same fixed
paper-boat request (`512x256`, 124 frames, 28 steps, seed 42, `fast=false`),
the installed `stock-qmm` runtime, and one cold plus three warm requests per
topology:

| Measurement | Single Mac | TP2 | Improvement |
| --- | ---: | ---: | ---: |
| Cold DiT sampling | 316.203 s | 181.385 s | 1.743x |
| Cold end to end | 347.126 s | 201.497 s | 1.723x |
| Warm DiT mean, n=3 | 315.024 s | 182.076 s | 1.730x, 42.2% less time |
| Warm end-to-end median, n=3 | 338.744 s | 204.544 s | 1.656x, 39.6% less time |
| Warm video-decode mean | 9.937 s | 10.002 s | Rank 0 only |
| Warm audio-decode mean | 0.570 s | 0.571 s | Rank 0 only |

All eight outputs carried 124 RGB8 frames plus stereo PCM audio and produced
31 progress events. Video and audio hashes were identical across all four runs
within each topology. The full-weight and sharded paths were not byte-identical
to each other; comparison of all 124 RGB frames measured SSIM `0.910` and PSNR
`27.26 dB`. The end-to-end warm median avoids overstating the result after one
single-Mac gateway transfer outlier (`422.042 s`); single-Mac warm DiT remained
stable at `315.547`, `314.749`, and `314.776 s`.

The final TP2 lifecycle reached `2/2` readiness, completed four jobs with the
ranks in step lock, and stopped both native ranks with return code `0`. Rank 1
received the protocol stop sentinel, and both Agent resource ledgers contained
no retained ports or memory reservations after shutdown.

## Phase 3 profiles, reproducible harness, and current default

The H3 Node Agent accepts only three typed optimization profiles; arbitrary
environment variables and shell fragments are rejected:

| profile | `MLX_SERVE_MF_DQ_GEMM` | block fusions | use |
| --- | ---: | --- | --- |
| `stock-qmm` | `0` | off | **default**, selected by the 512x256 TP2 live A/B |
| `baseline` | `2048` | off | former whole-weight DQ+transpose+Steel path |
| `block-fusions` | `2048` | on | opt-in packed SwiGLU, gate+residual, and RMSNorm+AdaLN evaluation |

Each launch fingerprints the native executable, `libmlx`, `libmlxc`,
`mlx.metallib`, TP2 manifest, both declared rank-shard hashes, Tokenity code
revision, protocol, profile, and expanded flags.  Absolute install paths remain
in local evidence but are excluded from the content contract, so identical
artifacts installed below different rank-local model roots compare equal.
Rank mismatch fails closed with HTTP `412` before JACCL launch.  Both actual
starts and dry-runs enforce a 2 GiB default free-disk gate.

The reusable fixed-request harness replaces manual curl bookkeeping. A
two-node JSON array plus `--worker-agent` selects TP2; a one-node array with no
worker selects the matching single-Mac path:

```bash
python -m tokenity.benchmarking.minimax_h3_tp2 \
  --coordinator-agent "$TOKENITY_H3_COORDINATOR_AGENT" \
  --worker-agent "$TOKENITY_H3_WORKER_AGENT" \
  --nodes-json "$TOKENITY_H3_NODES_JSON" \
  --model "$TOKENITY_H3_MODEL_PATH" \
  --binary "$TOKENITY_H3_BINARY_PATH" \
  --output-dir "$TOKENITY_H3_OUTPUT_DIR" \
  --test-id h3-tp2-run
```

It validates current Agent identity/capability/disk state (plus RDMA for TP2),
dry-runs, launches the requested `1/1` or `2/2` topology, maintains the request
lease, records readiness/fingerprints, performs one cold plus three warm
requests, archives raw SSE/RGB/PCM and rank snapshots, verifies 31 events and
deterministic hashes, then always asks for an instance-scoped stop and records
the post-stop resource ledgers.

On 2026-08-09, the same-runtime paired warm means were `200.393 s` for the
former DQ+Steel baseline, `199.784 s` for all three block fusions, and
`196.019 s` for stock fused qmm.  Stock qmm reduced end to end by `4.374 s`
(`2.18%`) and the two ranks' warm DiT from about `187.07 s` to `182.82 s`.
The fusion bundle added only `0.304%` and remains opt-in.  Every cold/warm
artifact matched video SHA-256 `e5c486ed...759f0` and audio SHA-256
`979488ec...d3850`; all ranks stopped with code `0`, `last_error=null`, and
empty reservations.

The validated run used a separate coordinator and isolated worker Agent.
Topology always comes from the typed request and is revalidated through
`/v1/node/info`; archived measurements are evidence, never business-layer
routing heuristics.
