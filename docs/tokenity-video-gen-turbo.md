# MiniMax H3 acceleration in Tokenity 0.1.2

The Video workspace connects SwiftUI → Node Agent → native MLX inference → SSE →
AVFoundation, including a playable MOV, audio and generation metadata.

Choose Normal for the existing 28-step sampler. Choose Turbo for the pinned H3
Turbo LoRA, with 6 steps initially and presets of 4 (fastest), 6 (balanced), or
8 (more refinement). Prompt, size, duration and seed remain editable. Switching
modes restores each mode's chosen steps; Turbo disables Fast/cache optimizations.
Fewer steps can change quality, detail and motion. The presets are a tradeoff,
not a guarantee of identical output.

Turbo currently supports one Apple-silicon Mac. Ordinary H3 generation retains
Single Mac and the existing two-Mac Thunderbolt RDMA path. If Turbo is selected
with two Macs, use the displayed Single Mac action before starting inference.

Install the complete 0.1.2 PKG on participating Macs. It includes the native H3
binary, MLX libraries, Metal shaders, Python and DeepSeek V4 compatibility. Model
weights are separate. Select the local MiniMax H3 model folder in Video, and
place `turbo_lora.safetensors` beside its other weights. The required adapter is
revision `43a74557ac3f6539db8e0f2a959d03feb7a81480`, size 779849816 bytes, SHA-256
`5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`.

Readiness checks run on the execution node and validate the adapter, actual
process identity, single-node topology, Turbo protocol 1, all 259 modules and
strength 1.0. A missing adapter, incompatible binary, TP2 topology or nonzero
`MINIMAX_H3_STEP_CACHE` / `MINIMAX_H3_ATTN_BCAST` override reports an error before
inference. Ordinary generation stays available without the Turbo adapter.

The reproducible hardware harness is `scripts/benchmark-minimax-h3-turbo.py`.
Supply `--native-repo`, `--model`, `--binary`, and a fresh `--out` directory.
It runs the real Store, gateway and AVFoundation writer. The historical H3-only
benchmark on M3 Ultra measured median denoising times of 337.7 seconds (28 steps),
50.8 (Turbo 4), 71.9 (Turbo 6), and 104.6 (Turbo 8); these are reference results,
not measurements of every target Mac or a promise of the same quality.
