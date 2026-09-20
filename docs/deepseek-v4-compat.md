# DeepSeek V4 Compatibility

Tokenity supports checkpoints with `model_type: deepseek_v4` on its pinned
MLX-LM 0.31.3 runtime by integrating the inference implementation from
[ml-explore/mlx-lm PR #1189](https://github.com/ml-explore/mlx-lm/pull/1189),
commit `63a26625c7ba2ffb8159ff430e630321446c7df4`.

The three upstream model modules are vendored byte-for-byte under
`tokenity/mlx/_upstream_pr1189`. Tokenity loads them into the
`mlx_lm.models` namespace only when the selected checkpoint declares
`deepseek_v4`; it does not edit the installed MLX-LM package. The adapter is
strictly gated to MLX-LM 0.31.3, the exact version on which the upstream PR was
based. If a later MLX-LM release provides its own DeepSeek V4 module, the native
implementation takes precedence.

PR #1189 uses a hybrid compressed KV cache that is unsafe with MLX-LM's
continuous batching path. Tokenity therefore selects sequential generation for
DeepSeek V4 while preserving normal batch eligibility for other architectures.
Both single-host inference and the model's tensor-parallel `shard()` path use
the same compatibility module.

Some community `DeepSeek-V4-Flash` exports declare a global 8-bit affine
quantization mode but store the 129 routed-expert projections as MXFP4 without
the required per-layer declarations. Tokenity detects the MXFP4 tensor layout
from the safetensors headers and augments the configuration in memory before
MLX-LM binds weights. The checkpoint's `config.json` and weight files are not
modified.

The integration intentionally does not install the PR's conversion command or
its experimental global MTP generation changes. Tokenity's existing Native MTP
path remains independently capability-gated.

## Validation

The installed Runtime uses MLX 0.32.0 and MLX-LM 0.31.3. Validation covered:

- constructing a reduced four-layer DeepSeek V4 model with real MLX;
- full prefill followed by cached single-token decode;
- hybrid `RotatingKVCache` and `CompressedKVCache` creation;
- server-side sequential-generation selection; and
- strict lazy loading of both local `DeepSeek-V4-Flash` and
  `DeepSeek-V4-Flash-4bit` checkpoints, including the hybrid MXFP4 expert
  metadata, tokenizer, 43-layer configuration, and all 33 weight shards.

When a DeepSeek V4 export omits its chat template, Tokenity supplies the verified
V4 text-message delimiters in memory after checking the tokenizer vocabulary.
Existing templates and other architectures are preserved. The fallback supports
system/user/assistant text messages and thinking mode; tool calling requires a
checkpoint-provided template. This avoids treating chat requests as plain-text
continuations. Both the single-host runtime and server provider apply the fix.

The local checkpoint contains about 151.5 GB of weight data. Lazy loading
validates model resolution, configuration, tokenizer, shard indexes, weight
names, sanitization, and quantization setup without materializing the complete
checkpoint in Metal memory. In the 0.1.2 merged runtime, the full `DeepSeek-V4-Flash-4bit` checkpoint also
completed a short greedy generation on M3 Ultra (512 GiB), returning the expected
`OK` response with approximately 151.34 GB peak MLX memory. The mixed 8-bit checkpoint also returned
`OK` with the missing-template fix, using approximately 154.80 GB peak MLX memory. This is a functional
smoke test, not a throughput benchmark or validation of every checkpoint.

## Provenance

The vendored files retain the upstream MIT license. Their SHA-256 digests are
locked by `tests/test_deepseek_v4_compat.py`, so an upstream refresh must be an
explicit, reviewable change.
