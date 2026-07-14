# Third-Party Notices

## Tokenity Native MTP adaptation

Files under `tokenity/inference/native_mtp/` that carry the
`SPDX-License-Identifier: Apache-2.0` header are derived from or informed by:

- `ml-explore/mlx-lm`, pull request #990, “Native MTP speculative decoding
  (Qwen3.5/3.6 reference implementation)”.
- The `omlx/patches/mlx_lm_mtp/` and `omlx/utils/model_loading.py` files from
  the oMLX project snapshot reviewed on 2026-07-13.

Those adapted files are licensed under Apache License 2.0. They have been
modified for Tokenity's Qwen-only, singleton, depth-one distributed runtime,
including static checkpoint completeness checks, deterministic distributed
preflight fingerprints, atomic monkey-patch installation, and Tokenity API
integration.

The Apache License 2.0 applies only to the identified adapted files and does
not change the license of Tokenity as a whole.

