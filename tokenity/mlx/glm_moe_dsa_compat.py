"""Compatibility for GLM-5.2 DSA cross-layer indexer sharing.

This ports the model implementation from ml-explore/mlx-lm#1410 so Tokenity's
pinned mlx-lm 0.31.x runtime can load GLM-5.2 checkpoints before that upstream
change is released. The patch is installed in memory and does not modify the
user's site-packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


UPSTREAM_PR = "https://github.com/ml-explore/mlx-lm/pull/1410"


def derive_indexer_types(
    *,
    num_hidden_layers: int,
    pattern: Any = None,
    frequency: int = 1,
    skip_offset: int = 2,
) -> list[str]:
    if pattern is not None:
        if isinstance(pattern, str):
            try:
                return [{"F": "full", "S": "shared"}[value] for value in pattern]
            except KeyError as exc:
                raise ValueError("GLM index_topk_pattern must contain only F and S.") from exc
        return list(pattern)

    frequency = max(frequency, 1)
    return [
        "full" if (max(index - skip_offset + 1, 0) % frequency) == 0 else "shared"
        for index in range(num_hidden_layers)
    ]


def install_glm_moe_dsa_compat() -> bool:
    """Install the PR #1410 model classes and return whether a patch was applied."""

    import mlx.core as mx  # type: ignore
    import mlx_lm.models.glm_moe_dsa as target  # type: ignore
    from mlx_lm.models.base import (  # type: ignore
        create_attention_mask,
        scaled_dot_product_attention,
    )
    from mlx_lm.models.cache import CacheList, KVCache  # type: ignore
    from mlx_lm.models.deepseek_v32 import (  # type: ignore
        DeepseekV32Attention,
        DeepseekV32DecoderLayer,
        DeepseekV32Model,
    )
    from mlx_lm.models.deepseek_v32 import Model as DSV32Model  # type: ignore

    if getattr(target, "_tokenity_cross_layer_indexer_sharing", False):
        return False

    existing_fields = getattr(target.ModelArgs, "__dataclass_fields__", {})
    if "indexer_types" in existing_fields:
        target._tokenity_cross_layer_indexer_sharing = True
        return False

    base_model_args = target.ModelArgs

    @dataclass
    class ModelArgs(base_model_args):
        indexer_types: Optional[List[str]] = None
        index_topk_pattern: Optional[Any] = None
        index_topk_freq: int = 1
        index_skip_topk_offset: int = 2

        def __post_init__(self):
            super().__post_init__()
            self.indexer_types = derive_indexer_types(
                num_hidden_layers=self.num_hidden_layers,
                pattern=self.indexer_types or self.index_topk_pattern,
                frequency=self.index_topk_freq,
                skip_offset=self.index_skip_topk_offset,
            )

    class GlmMoeDsaAttention(DeepseekV32Attention):
        def __init__(self, config: ModelArgs, layer_idx: int):
            super().__init__(config)
            self.skip_topk = config.indexer_types[layer_idx] == "shared"
            if self.skip_topk:
                self.indexer = None

        def __call__(
            self,
            x: Any,
            mask: Optional[Any] = None,
            cache: Optional[Any] = None,
            prev_topk_indices: Optional[Any] = None,
        ):
            batch, length, _ = x.shape

            qr = self.q_a_layernorm(self.q_a_proj(x))
            q = self.q_b_proj(qr)

            q = q.reshape(
                batch, length, self.num_heads, self.q_head_dim
            ).transpose(0, 2, 1, 3)
            q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
            compressed_kv = self.kv_a_proj_with_mqa(x)
            compressed_kv, k_pe = mx.split(
                compressed_kv, [self.kv_lora_rank], axis=-1
            )
            k_pe = k_pe.reshape(
                batch, length, 1, self.qk_rope_head_dim
            ).transpose(0, 2, 1, 3)
            kv_latent = self.kv_a_layernorm(compressed_kv)

            offset = cache[0].offset if cache is not None else 0
            q_pe = self.rope(q_pe, offset)
            k_pe = self.rope(k_pe, offset)

            kv_latent = mx.expand_dims(kv_latent, axis=1)

            if cache is not None:
                kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
            else:
                cache = [None] * 2

            if self.indexer is not None:
                topk_indices = self.indexer(x, qr, mask, cache=cache[1])
            else:
                topk_indices = prev_topk_indices

            if topk_indices is not None:
                if length == 1:
                    indices = topk_indices[:, :, 0, :, None]
                    kv_latent = mx.take_along_axis(
                        kv_latent,
                        mx.broadcast_to(
                            indices,
                            indices.shape[:-1] + (kv_latent.shape[-1],),
                        ),
                        axis=2,
                    )
                    k_pe = mx.take_along_axis(
                        k_pe,
                        mx.broadcast_to(
                            indices,
                            indices.shape[:-1] + (k_pe.shape[-1],),
                        ),
                        axis=2,
                    )
                    if mask is not None:
                        mask = mx.take_along_axis(mask, topk_indices, axis=-1)
                else:
                    shape = list(topk_indices.shape)
                    shape[-1] = kv_latent.shape[2]
                    sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                    sparse_mask = mx.put_along_axis(
                        sparse_mask, topk_indices, mx.array(True), axis=-1
                    )
                    if mask is not None:
                        sparse_mask = sparse_mask & mask
                    mask = sparse_mask

            if self.indexer is not None and cache is not None and cache[0] is not None:
                cache[0].keys = mx.depends(
                    cache[0].keys,
                    (cache[1].keys, cache[1].values),
                )

            pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
            if mask is not None:
                pe_scores = mx.where(
                    mask,
                    pe_scores,
                    mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
                )

            if length == 1:
                q_nope = self.embed_q(q_nope)
                key = value = kv_latent
            else:
                key = self.embed_q(kv_latent, transpose=False)
                value = self.unembed_out(kv_latent)

            output = scaled_dot_product_attention(
                q_nope,
                key,
                value,
                cache=cache,
                scale=self.scale,
                mask=pe_scores,
            )
            if length == 1:
                output = self.unembed_out(output)

            output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
            return self.o_proj(output), topk_indices

    class GlmMoeDsaDecoderLayer(DeepseekV32DecoderLayer):
        def __init__(self, config: ModelArgs, layer_idx: int):
            super().__init__(config, layer_idx)
            self.self_attn = GlmMoeDsaAttention(config, layer_idx)

        def __call__(
            self,
            x: Any,
            mask: Optional[Any] = None,
            cache: Optional[Any] = None,
            prev_topk_indices: Optional[Any] = None,
        ):
            residual, topk_indices = self.self_attn(
                self.input_layernorm(x),
                mask,
                cache,
                prev_topk_indices,
            )
            hidden = x + residual
            residual = self.mlp(self.post_attention_layernorm(hidden))
            return hidden + residual, topk_indices

    class GlmMoeDsaModel(DeepseekV32Model):
        def __init__(self, config: ModelArgs):
            super().__init__(config)
            self.layers = [
                GlmMoeDsaDecoderLayer(config, index)
                for index in range(config.num_hidden_layers)
            ]

        def __call__(self, x: Any, cache: Optional[Any] = None):
            hidden = self.embed_tokens(x)

            pipeline_rank = self.pipeline_rank
            pipeline_size = self.pipeline_size

            if cache is None:
                cache = [None] * self.num_layers
            mask = create_attention_mask(
                hidden,
                cache[0][0] if cache[0] else None,
                return_array=True,
            )

            if pipeline_rank < pipeline_size - 1:
                hidden = mx.distributed.recv_like(hidden, (pipeline_rank + 1))

            prev_topk_indices = None
            for index in range(self.num_layers):
                hidden, prev_topk_indices = self.layers[self.start_idx + index](
                    hidden,
                    mask,
                    cache[index],
                    prev_topk_indices,
                )

            if pipeline_rank != 0:
                hidden = mx.distributed.send(
                    hidden,
                    (pipeline_rank - 1) % pipeline_size,
                )
                if cache[-1] is not None:
                    cache[-1][0].keys = mx.depends(cache[-1][0].keys, hidden)

            if pipeline_size > 1:
                hidden = mx.distributed.all_gather(hidden)[: hidden.shape[0]]

            return self.norm(hidden)

    class Model(DSV32Model):
        def __init__(self, config: ModelArgs):
            super().__init__(config)
            self.model = GlmMoeDsaModel(config)

        def make_cache(self):
            caches = []
            for layer in self.layers:
                if getattr(layer.self_attn, "skip_topk", False):
                    caches.append(CacheList(KVCache()))
                else:
                    caches.append(CacheList(KVCache(), KVCache()))
            return caches

    ModelArgs.__module__ = target.__name__
    GlmMoeDsaAttention.__module__ = target.__name__
    GlmMoeDsaDecoderLayer.__module__ = target.__name__
    GlmMoeDsaModel.__module__ = target.__name__
    Model.__module__ = target.__name__
    target.ModelArgs = ModelArgs
    target.GlmMoeDsaAttention = GlmMoeDsaAttention
    target.GlmMoeDsaDecoderLayer = GlmMoeDsaDecoderLayer
    target.GlmMoeDsaModel = GlmMoeDsaModel
    target.Model = Model
    target._tokenity_cross_layer_indexer_sharing = True
    return True

