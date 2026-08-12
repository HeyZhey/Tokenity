# Tokenity Native MTP MVP

Tokenity Native MTP is an opt-in speculative decoding path for Qwen3.5 and
Qwen3.6 text checkpoints that contain the model's own one-layer
Multi-Token-Prediction head. It is integrated only with the Tokenity
distributed OpenAI backend. The default remains standard decoding.

## Supported boundary

- Qwen3.5/Qwen3.6 text models, including `qwen3_5_moe_text`.
- Tensor-parallel Tokenity distributed serving and the world-size-one baseline.
- `decode_concurrency=1`.
- One MTP layer (`max_depth=1`) with a replicated head.
- Greedy and temperature/top-p/top-k/min-p sampling.
- Standard streaming, maximum-token, EOS/stop-sequence, usage, and
  cancellation behavior.

The MVP does not enable Native MTP for VLM models, external draft models,
row-wise batches, adaptive depth, QMM verify
kernels, or per-request MTP selection. Seeded requests use mlx-lm's existing
sequential generation path and log `seeded_sequential_path`; they do not
change global readiness.

## Configuration

The Node Agent launch request carries a nested server-level object:

```json
{
  "native_mtp": {
    "mode": "off",
    "max_depth": 1,
    "head_placement": "replicated"
  }
}
```

`mode` has three values:

- `off`: never install or activate the runtime patch. This is the default.
- `auto`: enable only when checkpoint metadata, topology, every rank, and the
  installed mlx-lm ABI pass preflight; otherwise use ordinary decoding and
  publish a fallback reason.
- `required`: apply the same checks, but fail startup when Native MTP cannot
  be enabled.

Direct CLI launches use:

```bash
python -m tokenity distributed-openai serve \
  --model /path/to/model \
  --native-mtp-mode auto \
  --native-mtp-max-depth 1 \
  --native-mtp-head-placement replicated
```

When launched through `mlx.launch`, these arguments belong after the `--`
separator with the Tokenity server command.

## Capability and readiness

`GET /v1/node/models` performs read-only inspection of `config.json`, the
safetensors index, or safetensors headers. It does not materialize tensor
payloads. Each model entry includes `native_mtp` with one of:

- `supported`
- `missing_weights`
- `incomplete_weights`
- `unsupported`
- `unknown`
- `node_mismatch` (the Control app's aggregate when selected Macs disagree)

Static model-library capability is intentionally kept out of OpenAI
`GET /v1/models`. Runtime state is published by `GET /v1/readiness` under
`native_mtp`, including requested mode, effective mode, enabled state,
fallback reason, decision fingerprint, patch ABI, acceptance counters,
emitted tokens per verify cycle, and cumulative head/verify/rollback time.

Common reason codes include:

- `disabled_by_user`
- `unsupported_backend`
- `unsupported_model_type`
- `model_does_not_declare_mtp`
- `checkpoint_missing_mtp_weights`
- `checkpoint_incomplete_mtp_weights`
- `unsupported_decode_concurrency`
- `unsupported_topology`
- `unsupported_mlx_lm_runtime_shape`
- `rank_decision_mismatch`
- `patch_install_failed`
- `loaded_instance_invalid`
- `seeded_sequential_path`

## Safety model

Activation follows a fail-closed order:

1. Inspect configuration and safetensors metadata without loading tensors.
2. Compare a full decision fingerprint across every rank.
3. Validate the exact mlx-lm internal signatures used by the patch.
4. Apply all class mutations as one rollback-capable transaction and compare
   patch results across ranks.
5. Hold a serialized construction scope across both model constructions in
   distributed `sharded_load` (or the single real `server.load` call).
6. Shard the backbone only when world size is greater than one; the MTP head
   stays replicated.
7. Validate the loaded instance and prove that every MTP parameter key was
   supplied to `load_weights`. A head left at random initialization by
   `strict=False` is rejected.

Once a patch mutation has been attempted, a patch failure is fatal even in
`auto` mode. Continuing after a partial or heterogeneous monkey patch is not
safe.

During generation the ordinary prefill path is unchanged. Singleton MTP is
activated lazily only when no request can join the decode batch. Each cycle
verifies `[confirmed, draft]`; rejected drafts restore KV, rotating KV, and
GatedDeltaNet state. Stochastic acceptance uses `min(1, p/q)` and samples a
rejection from normalized `max(p-q, 0)`, using the same temperature and
top-p/top-k/min-p filtered distributions as the request sampler.

## Split-checkpoint assembly

The mlx-community Qwen3.5 MTP release is a split drafter rather than a target
checkpoint containing namespaced `mtp.*` tensors. Tokenity does not load an
external draft model at runtime. The repository instead provides an explicit,
offline assembler:

```bash
python scripts/assemble-native-mtp-checkpoint.py \
  --target /path/to/Qwen3.5-4B-MLX-4bit \
  --draft /path/to/Qwen3.5-4B-MTP-4bit \
  --output /path/to/Qwen3.5-4B-Native-MTP-4bit
```

The assembler validates the two configurations, keeps the target checkpoint
unchanged, namespaces the draft tensors under `language_model.mtp`, and
dequantizes only the MTP fusion `fc` projection to BF16 to match mlx-lm's
module definition. The head is written to `mtp.safetensors`, outside stock
mlx-lm's `model*.safetensors` glob. Tokenity exposes that sidecar only inside
an enabled construction scope. Consequently, `mode=off` loads exactly the
original target files and cannot trigger qwen3_5's MTP-aware sanitize path.
The output includes `tokenity-native-mtp.json` with source hashes and storage
provenance, and the assembler refuses to overwrite an existing directory.

## Validation scope

The repository includes control-plane tests plus Apple-Silicon synthetic
runtime tests against MLX 0.31.2 / mlx-lm 0.31.3. The synthetic suite checks
greedy parity, stochastic marginal correctness, hybrid-cache rollback,
maximum tokens, stop sequences, cancellation, lazy activation, late-join
protection, and random-head rejection.

The pre-existing large checkpoint
`Qwen3.5-122B-A10B-4bit` declares one MTP layer but its safetensors index
contains no MTP tensors. Its correct result is therefore:

- `off`: ordinary decoding;
- `auto`: ordinary decoding with `checkpoint_missing_mtp_weights`;
- `required`: startup failure with the same reason.

That checkpoint cannot provide a real Native MTP throughput or acceptance
measurement. It was retained as the missing-weight fail-closed test case.

The 4B split target and MTP repositories have also been exercised with real
model weights at world sizes one and two. Synthetic tests remain clearly
separate from real-checkpoint validation; rerun those checks on the target
hardware before relying on performance or acceptance claims.
