# Tokenity 0.1.2

This release adds DeepSeek V4, GLM 5.3 / Qwen4 MLX-VLM support and MiniMax H3 Turbo.
It retains the newer model lifecycle, video history, load progress, memory
admission, and atomic Runtime upgrade behavior.

- MiniMax H3: Normal 28-step generation, plus Turbo presets 4, 6 and 8. Turbo
  initially uses 6 steps and strength 1.0, and currently supports one Mac.
  Normal mode retains the existing Single Mac and two-Mac RDMA paths.
- DeepSeek V4: selected-model compatibility on MLX-LM 0.31.3, hybrid compressed
  KV caching, sequential server generation, and detection of missing MXFP4
  expert metadata in mixed 8-bit checkpoints. Missing text chat templates are
  supplied in memory without changing model files. See `deepseek-v4-compat.md`.
- MLX-VLM: an isolated backend automatically handles `glm5_next` (GLM 5.3)
  and `qwen4_exp`. GLM 5.3 Flash has completed 512–128K input benchmarks.
  This backend currently supports one Mac. MLX-LM remains the backend for
  DeepSeek V4 and other supported language models. See [MLX-VLM](mlx-vlm.md).
- Packaging: app version 0.1.2, build 4, Runtime ID 2026.09.15.2. The complete
  arm64 PKG installs the app, Node Agent, watchdog, embedded Python, MLX,
  native H3 and Metal libraries. It requires macOS 26.2 or newer.
- Portability: installed paths use `/Library/Tokenity`. Python entrypoints are
  rewritten for that installation, the embedded virtual environment no longer
  points to a developer's Python, and installation verification reads Mach-O
  load commands directly using the embedded Python. Xcode, Homebrew and a
  separately installed Python are not needed on the target Mac.

Open `Tokenity-0.1.2-macos-arm64.pkg`, then launch Tokenity from Applications.
Install the same package on each participating Mac. Model weights are separate:
place them below `/Library/Tokenity/Models`, or select the appropriate model
folder in the application. H3 Turbo also needs the pinned adapter described in
`tokenity-video-gen-turbo.md`.

This build is distributed without Developer ID signing or Apple notarization,
as requested. Local ad-hoc code signatures provide executable integrity; they
do not establish a trusted developer identity. If macOS blocks the downloaded installer, use System Settings → Privacy &
Security → Open Anyway, then reopen the PKG. The same approval may be needed
on the app’s first launch. See [installation instructions](installer-dmg.md). No system security settings are
changed by the package.

The package contains an embedded repair component and validates the staged
Runtime before swapping it into place. Existing model weights are preserved.
The previous Runtime is kept until health checks complete and restored if the
new Runtime fails those checks.

Build with `scripts/package-tokenity-dmg.sh`, setting `TOKENITY_RUNTIME_SOURCE`
to a complete runtime tree containing both H3 Turbo and the isolated VLM backend. The runtime lock pins dependency versions,
H3 distributed protocol 1, Turbo protocol 1, 259 adapter modules and strength 1.0.
The generated runtime manifest and package catalog pin file and artifact hashes.
