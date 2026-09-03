from __future__ import annotations

import hashlib
import importlib
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
