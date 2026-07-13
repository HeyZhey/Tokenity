from __future__ import annotations

import threading
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from tokenity.serving.distributed_openai import (
    TokenityDistributedRuntime,
    _requires_process_isolated_shutdown,
    _report_load_progress,
    _set_active_load_state,
    _stream_payload,
    create_app,
)
from tokenity.serving.readiness import ReadinessState


def test_distributed_skeleton_readiness_and_models():
    with TestClient(create_app(model="/models/qwen")) as client:
        readiness = client.get("/v1/readiness").json()
        models = client.get("/v1/models").json()
        info = client.get("/v1/tokenity/info").json()
        health = client.get("/health").json()

    assert readiness["phase"] == "ready"
    assert models["data"][0]["id"] == "/models/qwen"
    assert info["api"] == "openai-compatible"
    assert info["endpoints"] == ["/v1/models", "/v1/chat/completions"]
    assert health == {"status": "ok", "phase": "ready"}


def test_consolidated_runtime_is_not_skeleton_only():
    runtime = TokenityDistributedRuntime(
        model="/models/Qwen3.5-122B-A10B-4bit",
        state=ReadinessState(model="/models/Qwen3.5-122B-A10B-4bit"),
    )

    assert runtime.model_id == "Qwen3.5-122B-A10B-4bit"
    assert runtime.accepts_model("Qwen3.5-122B-A10B-4bit")
    assert runtime.max_tokens == 32_768


def test_runtime_supports_custom_api_identifier_and_runtime_tuning():
    runtime = TokenityDistributedRuntime(
        model="/models/Qwen3.5-122B-A10B-4bit",
        state=ReadinessState(model="/models/Qwen3.5-122B-A10B-4bit"),
        api_identifier="tokenity/qwen",
        max_tokens=65_536,
        prompt_cache_size=8,
        prefill_step_size=4_096,
        decode_concurrency=2,
        prompt_concurrency=3,
    )

    assert runtime.model_id == "tokenity/qwen"
    assert runtime.accepts_model("tokenity/qwen")
    assert runtime.accepts_model("/models/Qwen3.5-122B-A10B-4bit")
    assert runtime.max_tokens == 65_536
    assert runtime.prompt_cache_size == 8
    assert runtime.prefill_step_size == 4_096
    assert runtime.decode_concurrency == 2
    assert runtime.prompt_concurrency == 3


def test_stream_payload_keeps_reasoning_separate_from_answer():
    reasoning = _stream_payload(
        SimpleNamespace(text="check the plan", state="reasoning"),
        "qwen",
        finish_reason=None,
    )
    answer = _stream_payload(
        SimpleNamespace(text="Final answer", state="content"),
        "qwen",
        finish_reason=None,
    )

    assert reasoning is not None
    assert reasoning["choices"][0]["delta"] == {"reasoning_content": "check the plan"}
    assert answer is not None
    assert answer["choices"][0]["delta"] == {"content": "Final answer"}


def test_readiness_reports_real_parameter_load_progress():
    state = ReadinessState(model="/models/qwen")
    _set_active_load_state(state)
    try:
        _report_load_progress(25, 100)
    finally:
        _set_active_load_state(None)

    payload = state.to_dict()
    assert payload["progress_current"] == 25
    assert payload["progress_total"] == 100
    assert 0.3 < payload["progress"] < 0.4


def test_stop_endpoint_requests_natural_server_exit():
    state = ReadinessState(model="/models/glm")
    requested = threading.Event()

    class Runtime:
        def __init__(self):
            self.state = state
            self.stop_requested = False

        def request_stop(self):
            self.stop_requested = True

        def stop(self):
            pass

    runtime = Runtime()
    with TestClient(
        create_app(
            model="/models/glm",
            runtime=runtime,  # type: ignore[arg-type]
            request_server_exit=requested.set,
        )
    ) as client:
        response = client.post("/v1/tokenity/stop")
        assert requested.wait(1)

    assert response.status_code == 200
    assert response.json() == {"status": "stopping"}
    assert runtime.stop_requested is True
    assert state.phase.value == "stopping"


def test_distributed_release_does_not_clear_metal_cache_during_rank_teardown():
    state = ReadinessState(model="/models/glm", rank=0, world_size=2)
    runtime = TokenityDistributedRuntime(model="/models/glm", state=state)
    runtime._generator = object()  # noqa: SLF001 - targeted lifecycle regression
    runtime._provider = object()  # noqa: SLF001 - targeted lifecycle regression
    runtime._symbols = object()  # noqa: SLF001 - targeted lifecycle regression
    runtime._group = object()  # noqa: SLF001 - targeted lifecycle regression

    runtime._release_memory()  # noqa: SLF001 - targeted lifecycle regression

    assert runtime._generator is None  # noqa: SLF001
    assert runtime._provider is None  # noqa: SLF001
    assert runtime._group is None  # noqa: SLF001


def test_glm_uses_process_isolated_shutdown_without_exiting_generation_thread(tmp_path):
    glm = tmp_path / "GLM-5.2-mxfp4"
    glm.mkdir()
    (glm / "config.json").write_text(
        json.dumps({"architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa"}),
        encoding="utf-8",
    )
    qwen = tmp_path / "Qwen"
    qwen.mkdir()
    (qwen / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5MoeForConditionalGeneration"]}),
        encoding="utf-8",
    )

    assert _requires_process_isolated_shutdown(str(glm)) is True
    assert _requires_process_isolated_shutdown(str(qwen)) is False

    runtime = TokenityDistributedRuntime(model=str(glm), state=ReadinessState(model=str(glm)))
    generator = SimpleNamespace(_stop=False)
    runtime._generator = generator  # noqa: SLF001 - targeted GLM teardown regression
    runtime.request_stop()

    assert runtime.process_isolated_shutdown is True
    assert generator._stop is False
    assert runtime.state.phase.value == "stopping"
