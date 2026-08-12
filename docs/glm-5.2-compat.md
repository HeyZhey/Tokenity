# GLM-5.2 Compatibility

Tokenity includes an in-memory compatibility patch for GLM-5.2 checkpoints with
`model_type: glm_moe_dsa`. It ports the DSA cross-layer indexer sharing behavior
from [ml-explore/mlx-lm PR #1410](https://github.com/ml-explore/mlx-lm/pull/1410)
for the project's pinned MLX-LM 0.31.x runtime.

GLM-5.2 schedules its 78 decoder layers as 21 `full` indexer layers and 57
`shared` layers. Only full layers contain indexer weights. Shared layers reuse
the most recent full layer's top-k selection and must not allocate an indexer
cache. Without this behavior, the older runtime constructs indexers for every
layer and fails with `Missing 285 parameters`.

The patch is installed before MLX-LM resolves the model class. It affects only
the current Tokenity process and does not edit site-packages. A future MLX-LM
release that already exposes the upstream implementation is left untouched.

## Validated model

```text
Path:         ${TOKENITY_MODEL_ROOT}/GLM-5.2-mxfp4
Architecture: GlmMoeDsaForCausalLM
Quantization: mxfp4, 4-bit, group size 32
Shards:       76
Size:         368 GB
Runtime:      MLX 0.31.2, MLX-LM 0.31.3
```

The model was loaded successfully on a 512 GB M3 Ultra in Tokenity Single Mac
mode and completed a short OpenAI-compatible chat request. Memory is expected
to remain near the machine's wired-memory limit while this model is loaded.
