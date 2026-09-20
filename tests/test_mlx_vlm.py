from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tokenity.mlx.vlm_runtime import BACKEND_NAME, runtime_python
from tokenity.model_inspection import distributed_model_issue, model_backend, model_usage_metadata
from tokenity.serving.distributed_openai import ChatCompletionRequest, TokenityDistributedRuntime, create_app
from tokenity.serving.mlx_vlm_openai import _OutputText, _legacy_glm_weights, begin_generation
from tokenity.serving.readiness import ReadinessPhase, ReadinessState


def test_backend_selection_and_packaged_interpreter(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    for model_type in ("glm5_next", "qwen4_exp"):
        config = {"model_type": model_type}
        (model / "config.json").write_text(json.dumps(config))
        assert model_backend(model) == "mlx-vlm"
        assert "one Mac" in distributed_model_issue(model)
        assert model_usage_metadata(config)["inference_backend"] == "mlx-vlm"
    assert model_backend("", config={"model_type": "deepseek_v4"}) == "mlx-lm"
    monkeypatch.delenv("TOKENITY_MLX_VLM_PYTHON", raising=False)
    python = tmp_path / "Runtime/current/.venv/bin/python"
    vlm = tmp_path / "Runtime/backends" / BACKEND_NAME / ".venv/bin/python"
    with pytest.raises(RuntimeError, match="missing"):
        runtime_python(str(python))
    vlm.parent.mkdir(parents=True)
    vlm.touch()
    assert runtime_python(str(python)) == str(vlm)


def test_thinking_delimiters_and_stop_strings_split_across_tokens():
    text = _OutputText("assistant<think>", ["STOP"])
    parts = []
    for chunk in ("reason</th", "ink>an", "swerST", "OPleak"):
        parts.extend(text.feed(chunk))
    assert parts == [("reasoning", "reason"), ("content", "an"), ("content", "swer")]
    assert text.stopped
    text = _OutputText("", [])
    assert list(text.feed("a<")) == [("content", "a")]
    assert list(text.feed("", final=True)) == [("content", "<")]


def test_glm_legacy_names_preserve_arrays_and_reject_collisions():
    value = object()
    result = _legacy_glm_weights({
        "vision_model.patch_embed.proj.weight": value,
        "language_model.model.layers.0.self_attn.forget_gate.f_a_proj.scales": value,
        "language_model.model.layers.0.self_attn.conv1d.weight": value,
    })
    assert result == {
        "vision_tower.patch_embed.proj.weight": value,
        "language_model.model.layers.0.self_attn.f_a_proj.scales": value,
        "language_model.model.layers.0.self_attn.qkv_conv.conv.weight": value,
    }
    assert _legacy_glm_weights(result) == result
    with pytest.raises(ValueError, match="Duplicate"):
        _legacy_glm_weights({"vision_model.w": value, "vision_tower.w": value})


def test_vlm_stream_usage_final_flush_cancellation_and_http_validation(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type":"qwen4_exp"}')
    runtime = TokenityDistributedRuntime(
        model=str(tmp_path), state=ReadinessState(phase=ReadinessPhase.READY), execution_mode="single",
    )
    runtime._single_model = SimpleNamespace(config={"model_type": "qwen4_exp"})
    runtime._single_tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2])
    vlm = ModuleType("mlx_vlm")
    prompt_utils = ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.apply_chat_template = lambda *a, **kw: "assistant"
    closed = []

    def generate(*args, **kwargs):
        try:
            yield SimpleNamespace(text="OK", token=4, generation_tokens=1, prompt_tokens=7, finish_reason=None)
            yield SimpleNamespace(text="!", token=4, generation_tokens=1, prompt_tokens=7, finish_reason="length")
        finally:
            closed.append(True)

    vlm.stream_generate = generate
    monkeypatch.setitem(sys.modules, "mlx_vlm", vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)
    # This checks the HTTP adapter with a fake generator, without a Metal device.
    monkeypatch.setitem(sys.modules, "mlx", ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", ModuleType("mlx.core"))
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=None))
    request = ChatCompletionRequest(model=runtime.model_id, messages=[{"role": "user", "content": "hello"}])
    result = runtime.complete(request)
    assert result["choices"][0]["message"]["content"] == "OK!"
    assert result["usage"]["completion_tokens"] == 1
    assert result["usage"]["prompt_tokens"] == 7
    ctx, responses = begin_generation(runtime, request)
    next(responses)
    ctx.stop()
    responses.close()
    assert len(closed) == 2
    assert not runtime._single_generation_lock.locked()
    assert not runtime._single_contexts
    client = TestClient(create_app(model=runtime.model, runtime=runtime))
    for extra in ({"tools": [{"type": "function"}]}, {"logprobs": True}, {"response_format": {"type": "json_object"}}):
        response = client.post("/v1/chat/completions", json={**request.model_dump(), "stream": True, **extra})
        assert response.status_code == 400
