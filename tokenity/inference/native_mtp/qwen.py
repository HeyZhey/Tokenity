# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5/Qwen3.6 text-model support for one native MTP head.

Adapted from ml-explore/mlx-lm PR #990 and oMLX's Apache-2.0
``qwen35_model.py``. Tokenity removes VLM, row-wise batching, chained depth,
adaptive depth, and QMM paths; the head remains replicated under tensor
parallelism because mlx-lm's existing ``Model.shard`` visits backbone layers
only.
"""

from __future__ import annotations

from typing import Any

from .runtime import PatchTransaction, is_native_mtp_construction_active


def install(transaction: PatchTransaction) -> None:
    from mlx_lm.models import qwen3_5 as qwen  # type: ignore

    if getattr(qwen.TextModel, "_tokenity_native_mtp_model_patch", False):
        return

    _patch_args(transaction, qwen)
    _register_head(transaction, qwen)
    _patch_gated_delta(transaction, qwen)
    _patch_decoder_layer(transaction, qwen)
    _patch_backbone(transaction, qwen)
    _patch_text_model(transaction, qwen)
    _patch_outer_model(transaction, qwen)
    _patch_moe_sanitize(transaction)
    transaction.set(qwen.TextModel, "_tokenity_native_mtp_model_patch", True)


def _patch_args(transaction: PatchTransaction, qwen: Any) -> None:
    args_cls = qwen.TextModelArgs
    original = args_cls.from_dict.__func__

    def from_dict(cls: type, params: dict[str, Any]) -> Any:
        instance = original(cls, params)
        instance.mtp_num_hidden_layers = int(params.get("mtp_num_hidden_layers", 0) or 0)
        return instance

    transaction.set(args_cls, "from_dict", classmethod(from_dict))


def _register_head(transaction: PatchTransaction, qwen: Any) -> None:
    import mlx.core as mx  # type: ignore
    import mlx.nn as nn  # type: ignore

    class NativeMTPDecoderLayer(nn.Module):
        def __init__(self, args: Any):
            super().__init__()
            self.self_attn = qwen.Attention(args)
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size,
                eps=args.rms_norm_eps,
            )
            if args.num_experts > 0:
                self.mlp = qwen.SparseMoeBlock(args)
            else:
                self.mlp = qwen.MLP(args.hidden_size, args.intermediate_size)

        def __call__(self, x: Any, mask: Any = None, cache: Any = None) -> Any:
            residual = self.self_attn(self.input_layernorm(x), mask, cache)
            hidden = x + residual
            return hidden + self.mlp(self.post_attention_layernorm(hidden))

    class NativeMTPModule(nn.Module):
        def __init__(self, args: Any):
            super().__init__()
            self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.layers = [NativeMTPDecoderLayer(args)]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(
            self,
            hidden_states: Any,
            next_token_ids: Any,
            embed_tokens: Any,
            cache: Any = None,
        ) -> Any:
            embeddings = self.pre_fc_norm_embedding(embed_tokens(next_token_ids))
            hidden = self.pre_fc_norm_hidden(hidden_states)
            fused = self.fc(mx.concatenate([embeddings, hidden], axis=-1))
            caches = cache if cache is not None else [None]
            mask = qwen.create_attention_mask(fused, caches[0])
            fused = self.layers[0](fused, mask, caches[0])
            return self.norm(fused)

    transaction.set(qwen, "NativeMTPDecoderLayer", NativeMTPDecoderLayer)
    transaction.set(qwen, "NativeMTPModule", NativeMTPModule)


def _patch_gated_delta(transaction: PatchTransaction, qwen: Any) -> None:
    import mlx.core as mx  # type: ignore
    import mlx.nn as nn  # type: ignore
    from mlx.nn.layers.distributed import sum_gradients  # type: ignore
    from mlx_lm.models.gated_delta import gated_delta_update  # type: ignore

    cls = qwen.GatedDeltaNet

    def process_chunk(
        self: Any,
        qkv: Any,
        a: Any,
        b: Any,
        conv_state: Any,
        ssm_state: Any,
        mask: Any = None,
        lengths: Any = None,
    ) -> tuple[Any, Any, Any]:
        batch, sequence = qkv.shape[:2]
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        keep = self.conv_kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, sequence)
            positions = (ends[:, None] + mx.arange(keep))[..., None]
            next_conv = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            next_conv = mx.contiguous(conv_input[:, -keep:])
        convolved = nn.silu(self.conv1d(conv_input))
        query, key, value = [
            tensor.reshape(batch, sequence, heads, width)
            for tensor, heads, width in zip(
                mx.split(convolved, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        inverse_scale = key.shape[-1] ** -0.5
        query = (inverse_scale**2) * mx.fast.rms_norm(query, None, 1e-6)
        key = inverse_scale * mx.fast.rms_norm(key, None, 1e-6)
        output, next_ssm = gated_delta_update(
            query,
            key,
            value,
            a,
            b,
            self.A_log,
            self.dt_bias,
            ssm_state,
            mask,
            use_kernel=not self.training,
        )
        return output, next_conv, next_ssm

    def call(
        self: Any,
        inputs: Any,
        mask: Any = None,
        cache: Any = None,
        n_confirmed: int = 0,
    ) -> Any:
        batch, sequence, _ = inputs.shape
        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(
            batch,
            sequence,
            self.num_v_heads,
            self.head_v_dim,
        )
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)
        conv_state = (
            cache[0]
            if cache is not None and cache[0] is not None
            else mx.zeros(
                (batch, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )
        )
        ssm_state = cache[1] if cache else None
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)

        if cache is not None and 0 < n_confirmed < sequence:
            prefix_mask = mask[:, :n_confirmed] if mask is not None else None
            suffix_mask = mask[:, n_confirmed:] if mask is not None else None
            prefix, conv_mid, ssm_mid = process_chunk(
                self,
                qkv[:, :n_confirmed],
                a[:, :n_confirmed],
                b[:, :n_confirmed],
                conv_state,
                ssm_state,
                prefix_mask,
            )
            cache.rollback_state = (conv_mid, ssm_mid)
            suffix, conv_final, ssm_final = process_chunk(
                self,
                qkv[:, n_confirmed:],
                a[:, n_confirmed:],
                b[:, n_confirmed:],
                conv_mid,
                ssm_mid,
                suffix_mask,
            )
            output = mx.concatenate([prefix, suffix], axis=1)
        else:
            lengths = cache.lengths if cache is not None else None
            output, conv_final, ssm_final = process_chunk(
                self,
                qkv,
                a,
                b,
                conv_state,
                ssm_state,
                mask,
                lengths,
            )

        if cache is not None:
            cache[0] = conv_final
            cache[1] = ssm_final
            cache.advance(sequence)

        output = self.norm(output, z)
        output = self.out_proj(output.reshape(batch, sequence, -1))
        if self.sharding_group is not None:
            output = mx.distributed.all_sum(output, group=self.sharding_group)
        return output

    transaction.set(cls, "_tokenity_mtp_process_chunk", process_chunk)
    transaction.set(cls, "__call__", call)


def _patch_decoder_layer(transaction: PatchTransaction, qwen: Any) -> None:
    cls = qwen.DecoderLayer

    def call(
        self: Any,
        hidden: Any,
        mask: Any = None,
        cache: Any = None,
        n_confirmed: int = 0,
    ) -> Any:
        normalized = self.input_layernorm(hidden)
        if self.is_linear:
            residual = self.linear_attn(
                normalized,
                mask,
                cache,
                n_confirmed=n_confirmed,
            )
        else:
            residual = self.self_attn(normalized, mask, cache)
        next_hidden = hidden + residual
        return next_hidden + self.mlp(self.post_attention_layernorm(next_hidden))

    transaction.set(cls, "__call__", call)


def _patch_backbone(transaction: PatchTransaction, qwen: Any) -> None:
    cls = qwen.Qwen3_5TextModel

    def call(
        self: Any,
        inputs: Any,
        cache: Any = None,
        input_embeddings: Any = None,
        n_confirmed: int = 0,
    ) -> Any:
        hidden = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        caches = cache if cache is not None else [None] * len(self.layers)
        attention_mask = qwen.create_attention_mask(hidden, caches[self.fa_idx])
        ssm_mask = qwen.create_ssm_mask(hidden, caches[self.ssm_idx])
        for layer, layer_cache in zip(self.layers, caches):
            layer_mask = ssm_mask if layer.is_linear else attention_mask
            hidden = layer(
                hidden,
                mask=layer_mask,
                cache=layer_cache,
                n_confirmed=n_confirmed,
            )
        return hidden

    transaction.set(cls, "__call__", call)


def _patch_text_model(transaction: PatchTransaction, qwen: Any) -> None:
    from mlx_lm.models.cache import KVCache  # type: ignore

    cls = qwen.TextModel
    original_init = cls.__init__

    def init(self: Any, args: Any) -> None:
        original_init(self, args)
        declared = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
        active = is_native_mtp_construction_active()
        self._tokenity_native_mtp_decode_enabled = bool(active and declared == 1)
        if active and declared != 1:
            raise ValueError(f"Native MTP expected exactly one head layer, got {declared}")
        if self._tokenity_native_mtp_decode_enabled:
            self.mtp = qwen.NativeMTPModule(args)

    def call(
        self: Any,
        inputs: Any,
        cache: Any = None,
        input_embeddings: Any = None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ) -> Any:
        hidden = self.model(
            inputs,
            cache,
            input_embeddings=input_embeddings,
            n_confirmed=n_confirmed,
        )
        normalized = self.model.norm(hidden)
        logits = (
            self.model.embed_tokens.as_linear(normalized)
            if self.args.tie_word_embeddings
            else self.lm_head(normalized)
        )
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(
        self: Any,
        hidden_states: Any,
        next_token_ids: Any,
        mtp_cache: Any,
    ) -> Any:
        head_output = self.mtp(
            hidden_states,
            next_token_ids,
            self.model.embed_tokens,
            mtp_cache,
        )
        return (
            self.model.embed_tokens.as_linear(head_output)
            if self.args.tie_word_embeddings
            else self.lm_head(head_output)
        )

    def make_mtp_cache(self: Any) -> list[Any]:
        return [KVCache()] if hasattr(self, "mtp") else []

    def sanitize(self: Any, weights: dict[str, Any]) -> dict[str, Any]:
        if not hasattr(self, "mtp"):
            weights = {key: value for key, value in weights.items() if "mtp." not in key}

        raw_conv = any(
            "conv1d.weight" in key and getattr(value, "shape", (1,))[-1] != 1
            for key, value in weights.items()
        )
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
            weights.pop("language_model.lm_head.weight", None)
        norm_suffixes = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        )
        for key, value in list(weights.items()):
            if "conv1d.weight" in key and value.shape[-1] != 1:
                weights[key] = value.moveaxis(2, 1)
            if raw_conv and value.ndim == 1 and any(key.endswith(suffix) for suffix in norm_suffixes):
                weights[key] = value + 1.0
        return weights

    def quant_predicate(self: Any) -> Any:
        if self.args.num_experts <= 0 and not hasattr(self, "mtp"):
            return None

        def predicate(path: str, _module: Any) -> bool | dict[str, int]:
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            if path.endswith("mtp.fc"):
                return False
            return True

        return predicate

    transaction.set(cls, "__init__", init)
    transaction.set(cls, "__call__", call)
    transaction.set(cls, "mtp_forward", mtp_forward)
    transaction.set(cls, "make_mtp_cache", make_mtp_cache)
    transaction.set(cls, "sanitize", sanitize)
    transaction.set(cls, "quant_predicate", property(quant_predicate))


def _patch_outer_model(transaction: PatchTransaction, qwen: Any) -> None:
    cls = qwen.Model
    original_load_weights = cls.load_weights

    def call(
        self: Any,
        inputs: Any,
        cache: Any = None,
        input_embeddings: Any = None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ) -> Any:
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            return_hidden=return_hidden,
            n_confirmed=n_confirmed,
        )

    def mtp_forward(
        self: Any,
        hidden_states: Any,
        next_token_ids: Any,
        mtp_cache: Any,
    ) -> Any:
        return self.language_model.mtp_forward(hidden_states, next_token_ids, mtp_cache)

    def make_mtp_cache(self: Any) -> list[Any]:
        return self.language_model.make_mtp_cache()

    def sanitize(self: Any, weights: dict[str, Any]) -> dict[str, Any]:
        converted: dict[str, Any] = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model", 1)
            elif not key.startswith("language_model."):
                key = "language_model." + key
            key = key.replace("language_model.model.mtp.", "language_model.mtp.", 1)
            converted[key] = value
        return self.language_model.sanitize(converted)

    def load_weights(self: Any, weights: Any, strict: bool = True) -> Any:
        materialized = list(weights)
        self._tokenity_native_mtp_loaded_keys = tuple(
            key for key, _ in materialized if ".mtp." in f".{key}."
        )
        return original_load_weights(self, materialized, strict=strict)

    transaction.set(cls, "__call__", call)
    transaction.set(cls, "mtp_forward", mtp_forward)
    transaction.set(cls, "make_mtp_cache", make_mtp_cache)
    transaction.set(cls, "sanitize", sanitize)
    transaction.set(cls, "load_weights", load_weights)


def _patch_moe_sanitize(transaction: PatchTransaction) -> None:
    import mlx.core as mx  # type: ignore
    from mlx_lm.models import qwen3_5_moe as moe  # type: ignore

    cls = moe.Model

    def unfuse(weights: dict[str, Any], prefix: str) -> None:
        key = f"{prefix}.experts.gate_up_proj"
        if key not in weights:
            return
        gate_up = weights.pop(key)
        midpoint = gate_up.shape[-2] // 2
        weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :midpoint, :]
        weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., midpoint:, :]
        weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
            f"{prefix}.experts.down_proj"
        )

    def stack_experts(weights: dict[str, Any], prefix: str, count: int) -> None:
        if f"{prefix}.experts.0.gate_proj.weight" not in weights:
            return
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "scales", "biases"):
                first = f"{prefix}.experts.0.{projection}.{suffix}"
                if first not in weights:
                    continue
                weights[f"{prefix}.switch_mlp.{projection}.{suffix}"] = mx.stack(
                    [
                        weights.pop(f"{prefix}.experts.{index}.{projection}.{suffix}")
                        for index in range(count)
                    ]
                )

    def sanitize(self: Any, weights: dict[str, Any]) -> dict[str, Any]:
        converted: dict[str, Any] = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model", 1)
            elif not key.startswith("language_model."):
                key = "language_model." + key
            key = key.replace("language_model.model.mtp.", "language_model.mtp.", 1)
            converted[key] = value

        num_experts = int(getattr(self.language_model.args, "num_experts", 0) or 0)
        for index in range(self.language_model.args.num_hidden_layers):
            prefix = f"language_model.model.layers.{index}.mlp"
            if f"{prefix}.switch_mlp.gate_proj.weight" not in converted:
                unfuse(converted, prefix)
                stack_experts(converted, prefix, num_experts)
        if hasattr(self.language_model, "mtp"):
            prefix = "language_model.mtp.layers.0.mlp"
            if f"{prefix}.switch_mlp.gate_proj.weight" not in converted:
                unfuse(converted, prefix)
                stack_experts(converted, prefix, num_experts)
        return self.language_model.sanitize(converted)

    transaction.set(cls, "sanitize", sanitize)
