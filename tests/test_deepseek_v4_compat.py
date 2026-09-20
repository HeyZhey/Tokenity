from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tokenity.mlx import deepseek_v4_compat as compat


VENDORED_DIGESTS = {
    "deepseek_v4.py": "9dfd13abd1452a1aa51c3006fb1c2631fc7b8062867cd56a2fa93186e4846af1",
    "hyper_connection.py": "54ef5fc835a4b10480fcedd03d5e0a3890b386a0036a81d35d57458ca582f379",
    "sinkhorn.py": "debf1d280cf2120033938595325dc4b0f668afaaa8b5b8de00149ab93062b7b4",
}


@pytest.fixture
def isolated_mlx_lm_models():
    models = pytest.importorskip("mlx_lm.models")
    module_names = [f"mlx_lm.models.{name}" for name in compat._VENDORED_MODULES]
    previous_modules = {name: sys.modules.get(name) for name in module_names}
    previous_attributes = {
        name: getattr(models, name) for name in compat._VENDORED_MODULES if hasattr(models, name)
    }
    for name in module_names:
        sys.modules.pop(name, None)
    for name in compat._VENDORED_MODULES:
        if hasattr(models, name):
            delattr(models, name)
    yield models
    for name in module_names:
        sys.modules.pop(name, None)
    for name in compat._VENDORED_MODULES:
        if hasattr(models, name):
            delattr(models, name)
    for name, module in previous_modules.items():
        if module is not None:
            sys.modules[name] = module
    for name, module in previous_attributes.items():
        setattr(models, name, module)


def test_vendored_sources_match_pr_1189_head():
    source_root = Path(compat.__file__).with_name("_upstream_pr1189")
    assert compat.UPSTREAM_COMMIT == "63a26625c7ba2ffb8159ff430e630321446c7df4"
    assert {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in VENDORED_DIGESTS
    } == VENDORED_DIGESTS


def test_installer_rejects_an_unsupported_mlx_lm_version(monkeypatch):
    monkeypatch.setattr(compat, "_import_existing_deepseek_v4", lambda: None)
    monkeypatch.setattr(compat, "_installed_mlx_lm_version", lambda: "9.9.9")

    with pytest.raises(RuntimeError, match="requires mlx-lm 0.31.3"):
        compat.install_deepseek_v4_compat()


def test_pr_1189_model_constructs_prefills_and_decodes(isolated_mlx_lm_models):
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import RotatingKVCache

    assert compat.install_deepseek_v4_compat()
    assert not compat.install_deepseek_v4_compat()
    deepseek_v4 = importlib.import_module("mlx_lm.models.deepseek_v4")
    assert getattr(deepseek_v4, compat._MARKER) == compat.UPSTREAM_COMMIT

    args = deepseek_v4.ModelArgs(
        model_type="deepseek_v4",
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        head_dim=16,
        qk_rope_head_dim=4,
        sliding_window=16,
        compress_ratios=[0, 0, 4, 0],
        index_n_heads=4,
        index_head_dim=8,
        index_topk=4,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=1,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        max_position_embeddings=256,
    )
    model = deepseek_v4.Model(args)
    inputs = mx.array([[0, 1, 2, 3, 4]], dtype=mx.int32)

    output = model(inputs)
    mx.eval(output)
    assert output.shape == (1, 5, args.vocab_size)

    cache = model.make_cache()
    assert isinstance(cache[0], RotatingKVCache)
    assert isinstance(cache[2], deepseek_v4.CompressedKVCache)
    prefill = model(inputs[:, :4], cache=cache)
    decode = model(inputs[:, 4:], cache=cache)
    mx.eval(prefill, decode)
    assert decode.shape == (1, 1, args.vocab_size)


def test_hybrid_checkpoint_infers_omitted_mxfp4_switch_metadata(
    isolated_mlx_lm_models,
    tmp_path,
):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model

    assert compat.install_deepseek_v4_compat()
    deepseek_v4 = importlib.import_module("mlx_lm.models.deepseek_v4")
    args = deepseek_v4.ModelArgs(
        model_type="deepseek_v4",
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        head_dim=16,
        qk_rope_head_dim=4,
        sliding_window=16,
        compress_ratios=[0, 0, 4, 0],
        index_n_heads=4,
        index_head_dim=8,
        index_topk=4,
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=1,
        num_nextn_predict_layers=0,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        max_position_embeddings=256,
    )
    source = deepseek_v4.Model(args)
    nn.quantize(
        source,
        class_predicate=lambda path, module: (
            dict(compat._MXFP4_QUANTIZATION)
            if ".switch_mlp." in path and hasattr(module, "to_quantized")
            else False
        ),
    )
    weights = dict(tree_flatten(source.parameters()))
    switch_paths = sorted(
        key.removesuffix(".weight")
        for key in weights
        if ".switch_mlp." in key and key.endswith(".weight")
    )
    assert len(switch_paths) == 12
    assert not any(f"{path}.biases" in weights for path in switch_paths)

    shard_name = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(tmp_path / shard_name), weights)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard_name for key in weights}}),
        encoding="utf-8",
    )
    config = asdict(args)
    config["quantization"] = {"group_size": 64, "bits": 8, "mode": "affine"}
    config["quantization_config"] = dict(config["quantization"])
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    loaded, loaded_config = load_model(tmp_path, lazy=True, strict=True)

    assert all(
        loaded_config["quantization"][path] == compat._MXFP4_QUANTIZATION
        for path in switch_paths
    )
    assert all(
        getattr(getattr(layer.ffn.switch_mlp, projection), "mode") == "mxfp4"
        for layer in loaded.model.layers
        for projection in compat._SWITCH_PROJECTIONS
    )


def test_server_patch_forces_deepseek_v4_to_sequential_generation():
    class Provider:
        def _load(self, model_path, adapter_path=None, draft_model_path=None):
            del adapter_path, draft_model_path
            self.model = SimpleNamespace(model_type=model_path)
            self.is_batchable = True

    server = SimpleNamespace(ModelProvider=Provider)

    assert compat.install_deepseek_v4_server_compat(server)
    assert not compat.install_deepseek_v4_server_compat(server)

    deepseek = Provider()
    deepseek._load("deepseek_v4")
    assert deepseek.is_batchable is False

    qwen = Provider()
    qwen._load("qwen3_5")
    assert qwen.is_batchable is True


def test_missing_chat_template_is_model_scoped_and_preserves_user_template():
    tokens = {"<｜begin▁of▁sentence｜>": 0, "<｜end▁of▁sentence｜>": 1,
              "<｜User｜>": 2, "<｜Assistant｜>": 3, "<think>": 4, "</think>": 5}
    tokenizer = SimpleNamespace(chat_template=None, has_chat_template=False, get_vocab=lambda: tokens)
    assert not compat.install_deepseek_v4_tokenizer_compat(SimpleNamespace(model_type="qwen"), tokenizer)
    model = SimpleNamespace(model_type="deepseek_v4")
    assert compat.install_deepseek_v4_tokenizer_compat(model, tokenizer)
    assert tokenizer.has_chat_template
    assert not compat.install_deepseek_v4_tokenizer_compat(model, tokenizer)
    tokenizer.chat_template = "custom"
    assert not compat.install_deepseek_v4_tokenizer_compat(model, tokenizer)
    assert tokenizer.chat_template == "custom"


def test_v4_chat_fallback_formats_messages_and_thinking_without_plain_text_roles():
    jinja = pytest.importorskip("jinja2")
    template = jinja.Environment().from_string(compat._CHAT_TEMPLATE)
    kwargs = dict(bos_token="<｜begin▁of▁sentence｜>", eos_token="<｜end▁of▁sentence｜>",
                  messages=[{"role": "user", "content": "Hello"}], add_generation_prompt=True)
    assert template.render(**kwargs) == "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"
    assert template.render(**kwargs, enable_thinking=True).endswith("<｜Assistant｜><think>")
    assert template.render(**kwargs, thinking_mode="thinking").endswith("<｜Assistant｜><think>")
    kwargs["messages"] = [{"role": "system", "content": "Be brief."},
                          {"role": "user", "content": "Hello"},
                          {"role": "assistant", "content": "Hi"},
                          {"role": "user", "content": "Again"}]
    assert template.render(**kwargs) == "<｜begin▁of▁sentence｜>Be brief.<｜User｜>Hello<｜Assistant｜></think>Hi<｜end▁of▁sentence｜><｜User｜>Again<｜Assistant｜></think>"


def test_v4_missing_template_rejects_unknown_token_vocabulary():
    tokenizer = SimpleNamespace(chat_template=None, has_chat_template=False, get_vocab=lambda: {})
    with pytest.raises(RuntimeError, match="missing required chat tokens"):
        compat.install_deepseek_v4_tokenizer_compat(SimpleNamespace(model_type="deepseek_v4"), tokenizer)
    assert tokenizer.chat_template is None
