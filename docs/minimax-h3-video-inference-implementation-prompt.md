# Tokenity MiniMax H3 Video Inference Implementation Prompt

## Mission

Extend Tokenity with a production-honest MiniMax H3 video inference path for
Apple Silicon. Preserve Tokenity's validated two-Mac HTTP control plane and
JACCL/RDMA data plane, use the native-MLX MiniMax H3 engine as the computation
backend, and add Tensor Parallel support incrementally behind explicit
capability gates.

The first delivered slice must be useful without pretending that unfinished
distributed model code exists: Tokenity must be able to launch, observe, proxy,
and stop a real native MiniMax H3 server on one Mac. A two-Mac launch may only
become available after the native engine reports a compatible distributed-H3
rank protocol and passes the collective and numerical validation gates below.

## Working copy

- Repository: `/path/to/MinMaxH3/Tokenity-H3`
- Branch: `codex/minimax-h3-video`
- Baseline commit: `20ea6a2`
- Do not modify `/path/to/Tokenity-Stable` or the legacy
  `/path/to/MLX-Distributed` tree.

## Known hardware

- Mac A Node Agent: `http://<node-a-lan-ip>:9100`
- Mac A JACCL interface: `rdma_en4`, Thunderbolt IP `<node-a-rdma-ip>`
- Mac B Node Agent: `http://<node-b-lan-ip>:9100`
- Mac B JACCL interface: `rdma_en5`, Thunderbolt IP `<node-b-rdma-ip>`
- Both machines have 512 GiB unified memory.
- Use `jaccl-ring` for the stable production path. Direct JACCL previously
  failed with error `-12` after repeated or multi-resident requests.

## Existing native H3 contract

The already measured H3 runtime is the native `mlx-serve` implementation. It
serves `POST /v1/video/generations` and accepts at least:

```json
{
  "prompt": "...",
  "num_frames": 124,
  "width": 512,
  "height": 256,
  "steps": 28,
  "seed": 123,
  "fast": false,
  "stream": true
}
```

Do not use the 115 GB Diffusers/PyTorch MPS conversion as the distributed
runtime. It is a correctness fallback, not the native MLX computation base.

## Required architecture

### Control plane

Reuse Tokenity's typed Node Agent HTTP orchestration, instance registry,
resource ledger, port quarantine, revision checks, readiness/quorum state,
process supervisor, cancellation, fail-closed behavior, and cleanup. HTTP
starts and stops ranks but never carries per-layer model tensors.

Add first-class roles instead of overloading chat roles:

- `minimax-h3-video` for a coordinator or single-node server
- `minimax-h3-video-rank` for a remote distributed worker

Add typed endpoints:

- `POST /v1/node/start-minimax-h3-video`
- `POST /v1/node/start-minimax-h3-video-rank`

Single-node startup must execute a configured native binary directly. The
binary path and model path must be validated without a shell. Never interpolate
user input into a command string.

### Public video gateway

Expose or proxy these coordinator endpoints:

- `POST /v1/video/generations`
- `GET /v1/video/models` or equivalent capability/readiness metadata
- `GET /v1/readiness`
- `POST /v1/tokenity/stop`

Preserve streaming response bytes and media content types. Do not parse and
re-encode generated video payloads in the Node Agent.

### Native distributed engine

Implement TP2 inside the native MLX H3 DiT, not as pipeline parallelism and not
by converting the model back to PyTorch.

For each of the 50 main DiT blocks:

1. Split 56 attention heads into 28 heads per rank.
2. Shard fused QKV by output rows while keeping Q, K, and V ranges aligned.
3. Run SDPA locally on 28 heads.
4. Shard out-projection by input columns and `all_sum` the partial hidden
   output.
5. Shard fused FC1 so each rank receives matching 7,168-wide gate and up
   halves.
6. Shard FC2 by input columns and `all_sum` the partial hidden output.
7. Replicate norms, AdaLN values, residual state, patch projections, and final
   heads.

The reference 28-step path therefore has two model collectives per block,
100 per step, and 2,800 total. Collective ordering is protocol state: all ranks
must enter the same collective with the same shape and dtype.

Use the MLX C distributed API already present in the pinned `mlx-c` dependency.
Add narrow Zig FFI bindings for distributed group init/rank/size/free,
`all_sum`, and the minimum send/receive operations needed by the job protocol.

### Rank protocol

Rank 0 owns the public video API, text conditioning, final VAE/audio decode,
and media mux. Rank 1 owns no public port.

Use a deterministic state machine:

1. idle/control frame
2. canonical request accepted
3. input tensors shared
4. denoise step 0..N
5. result/complete frame
6. idle or coordinated stop

Do not interleave an asynchronous control collective with outstanding model
collectives. Poll cancellation only at agreed step boundaries and send a
versioned multi-word control frame containing magic, version, operation epoch,
sequence, command, and integrity check. A cancelled or protocol-mismatched
runtime must fail closed and require a clean restart.

### Quantized checkpoint sharding

Create an offline TP2 sharder. For every quantized tensor, slice packed
`weight`, `scales`, and `biases` together and validate group-size alignment.
QKV and FC1 require non-contiguous semantic ranges that are concatenated into
rank-local tensors. Out and FC2 require input-column slices. Preserve tensor
names so the native loader can load a rank-local checkpoint without holding the
full DiT shard in memory.

## Delivery phases

### Phase 0 — honest single-node vertical slice

- Typed Node Agent request and dry-run plan.
- Direct native-binary command construction.
- Binary/model preflight.
- Coordinator readiness and stop integration.
- Transparent `/v1/video/generations` proxy.
- Unit tests using fake supervisor and fake upstream server.

### Phase 1 — collective go/no-go

Run Mac A/B `jaccl-ring` with the actual BF16 hidden shape. At 512x256 and 124
frames the non-text packed sequence has 5,150 rows, so benchmark at least
`[5150, 5376]` for 2,800 `all_sum` calls. Record warmup, p50/p95 per collective,
total time, effective GiB/s, memory, and clean shutdown.

- `<= 50 s`: proceed.
- `50-100 s`: proceed only with a measured end-to-end win or for higher
  resolutions where attention dominates.
- `> 100 s`: stop TP2 integration and investigate fused collectives or another
  partition before modifying production behavior.

### Phase 2 — native block TP2

- TP2 checkpoint converter.
- Block fixture tests for QKV, attention, Out, FC1, and FC2.
- Single-block and multi-block numerical comparison against unsharded native
  MLX. Bit identity is not required; document tolerances and changed reduction
  order.
- Rank failure, shape mismatch, cancellation, and repeated-job tests.

### Phase 3 — full H3 generation

- Rank 0 conditioning and rank input sharing.
- 50-block DiT TP2.
- Rank 0-only VAE/audio decode and mux.
- Same-seed single-vs-TP quality evaluation.
- 512x256 reference benchmark followed by 960x544 and 1344x768.

### Phase 4 — product integration

- Tokenity UI video model capability and form.
- Progress, cancellation, output reveal, and history.
- Installer/runtime manifest updates.
- Real hardware restart, repeated request, cancellation, and orphan audit.

## Non-negotiable correctness rules

- Never advertise TP2 readiness when the backend lacks the distributed-H3
  protocol capability.
- Never silently fall back from a requested two-Mac TP job to two independent
  full-model generations.
- A successful response must originate from the requested instance and model
  revision.
- A failed or cancelled job must leave zero active ranks, reservations, and
  orphaned child processes after cleanup.
- Keep chat and video scheduling separate until shared-node admission has an
  explicit memory and generation-slot policy.
- `fast: false` is the correctness baseline. Validate TP first, then test the
  native approximate fast recipe separately.
- Tests must fail before their implementation is added and must exercise the
  public contract rather than private mocks alone.

## Acceptance criteria for the first implementation slice

1. `tokenity minimax-h3-video serve --help` exposes a direct, shell-free native
   runtime adapter.
2. Node Agent dry-run returns exact coordinator/worker command and environment
   plans without starting a process.
3. Single-node non-dry-run preflights the binary/model, supervises the process,
   reports instance state, proxies `/v1/video/generations`, and stops cleanly.
4. A two-node request returns a clear capability error until a backend binary
   positively reports compatible TP2 support.
5. Existing distributed OpenAI tests remain green.
6. New CLI, request validation, command construction, proxy, cancellation, and
   cleanup tests are green.
