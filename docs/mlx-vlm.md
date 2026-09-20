# MLX-VLM integration (Tokenity 0.1.2 Build 4)

Tokenity automatically selects MLX-VLM for config.json model_type `glm5_next`
(GLM-5.3-Flash) and `qwen4_exp` (Qwen3.8-Flash-Next). Select **Load on one Mac**.
The model library displays the selected engine. The OpenAI endpoint stays the same.

MLX-LM focuses on text language models. MLX-VLM also implements the vision encoders,
processors, and multimodal architectures needed by these two models, including when
the request only contains text. Both use MLX/Metal; neither backend is universally
faster. Model architecture, precision, input length, and batching determine speed.

The package keeps the existing MLX 0.32.0 / MLX-LM 0.31.3 backend and native H3 binary.
MLX-VLM 0.7.1, MLX 0.32.2, and Transformers 5.16.1 run in a separate interpreter at
`Runtime/backends/mlx-vlm-macos-arm64-0.7.1/.venv/bin/python`. No runtime packages are
installed on first launch. The installer validates both environments.

## Supported request path

- Text chat, streaming/nonstreaming responses, thinking content, stop strings,
  sampling options, cancellation and the existing readiness/stop endpoints.
- Images through OpenAI `image_url` content parts containing base64 image data URLs.
  The desktop chat composer remains text-only. External image URLs, audio, and video
  input are not enabled in this adapter.
- Single-Mac execution. Distributed sharding, VLM MTP, tool calling, structured JSON
  output, and logprobs have not been enabled. Explicit unsupported tool/logprob requests
  return HTTP 400; required MTP and multi-Mac launches fail with a capability message.
- The initial GLM conversion layout (`vision_model`, nested `forget_gate`, fused
  `conv1d`) is mapped without modifying checkpoint files, dequantization, or dropping
  parameters. Upstream sanitize performs its normal projection fusion. Loading remains
  strict so an incomplete/mismatched checkpoint fails rather than using random weights.

## Building

Import the existing LM/H3 runtime using `scripts/import-tokenity-runtime.sh`, then run:

```sh
bash scripts/prepare-tokenity-vlm-runtime.sh
TOKENITY_RUNTIME_ROOT=/Library/Tokenity/Runtime \
TOKENITY_CODE_ROOT=/Library/Tokenity/Code \
bash scripts/package-tokenity-dmg.sh
```

The preparation step uses uv on the build Mac and the exact package versions in
`packaging/runtime/vlm-requirements.txt`. Packaged Python, links, and entrypoints are
normalized for installation. For development, create a separate venv and install
`.[vlm,dev]`; point the development Agent's `TOKENITY_MLX_VLM_PYTHON` at that interpreter.
Do not install the VLM extra into the qualified LM environment.

## Upstream sources

- https://github.com/Blaizzy/mlx-vlm/releases/tag/v0.7.1
- GLM: https://github.com/Blaizzy/mlx-vlm/pull/2127
- Qwen: https://github.com/Blaizzy/mlx-vlm/pull/2032 and https://github.com/Blaizzy/mlx-vlm/pull/2126

Verified with the user's complete 4-bit checkpoints on an M3 Ultra: strict load,
one-token warmup, text response, HTTP streaming, image color recognition, and unload.
