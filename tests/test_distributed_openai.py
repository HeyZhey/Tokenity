from __future__ import annotations

import asyncio
import inspect
import json
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tokenity.serving.distributed_openai import (
    ChatCompletionRequest,
    _LoadEvalPolicy,
    _RepetitionDetector,
    TokenityDistributedRuntime,
    _eval_parameters_in_chunks,
    _install_chunked_sharded_load,
    _install_jaccl_server_control_collectives,
    _jaccl_control_header,
    _load_eval_policy,
    _parameter_eval_chunks,
    _configure_mlx_wired_memory,
    _requires_process_isolated_shutdown,
    _report_load_progress,
    _replace_jaccl_sequential_cancel_collective,
    _replace_jaccl_seed_collective,
    _set_active_load_state,
    _should_run_post_load_barrier,
    _single_token_logprobs,
    _stream_payload,
    _validate_jaccl_control_header,
    _validate_jaccl_control_payload,
    create_app,
    serve,
)
from tokenity.serving.readiness import ReadinessState
from tokenity.serving.readiness import ReadinessPhase


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


def test_failed_readiness_uses_503_while_preserving_structured_runtime_state():
    state = ReadinessState(model="/models/qwen")
    state.transition(
        ReadinessPhase.FAILED,
        message="tensor topology is unsupported",
        error={"stage": "model_validation", "message": "tensor topology is unsupported"},
    )

    class FailedRuntime:
        def __init__(self):
            self.state = state

        def memory_metrics(self):
            return None

        def stop(self):
            return None

    with TestClient(create_app(model="/models/qwen", runtime=FailedRuntime())) as client:
        response = client.get("/v1/readiness")

    assert response.status_code == 503
    assert response.json()["phase"] == "failed"
    assert response.json()["lifecycle_state"] == "failed"
    assert response.json()["last_error"]["stage"] == "model_validation"


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


def test_generation_arguments_forward_presence_and_repetition_penalties():
    runtime = TokenityDistributedRuntime(
        model="/models/qwen",
        state=ReadinessState(model="/models/qwen"),
    )
    runtime._symbols = SimpleNamespace(  # noqa: SLF001 - request mapping regression
        GenerationArguments=SimpleNamespace,
        ModelDescription=SimpleNamespace,
        SamplingArguments=SimpleNamespace,
        LogitsProcessorArguments=SimpleNamespace,
    )

    arguments = runtime._generation_args(  # noqa: SLF001
        ChatCompletionRequest(
            model="qwen",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=100,
            max_completion_tokens=3,
            presence_penalty=1.5,
            repetition_penalty=1.1,
            logprobs=True,
            top_logprobs=2,
            chat_template_kwargs={"enable_thinking": False},
        )
    )

    assert arguments.max_tokens == 3
    assert arguments.logprobs is True
    assert arguments.top_logprobs == 2
    assert arguments.logits.presence_penalty == 1.5
    assert arguments.logits.repetition_penalty == 1.1
    assert arguments.chat_template_kwargs == {"enable_thinking": False}


def test_completion_returns_structured_tool_calls_and_logprobs():
    stopped = threading.Event()
    context = SimpleNamespace(
        prompt=[1, 2],
        prompt_cache_count=0,
        stop=stopped.set,
        tool_parser=lambda text, tools: json.loads(text),
    )
    responses = iter(
        [
            SimpleNamespace(
                text="I will check.",
                token=10,
                state="normal",
                logprob=-0.1,
                top_tokens=(
                    {"token": "I", "logprob": -0.1},
                    {"token": "We", "logprob": -1.2},
                ),
                finish_reason=None,
            ),
            SimpleNamespace(
                text='{"name":"weather","arguments":{"city":"Shanghai"}}',
                token=11,
                state="tool",
                logprob=-0.2,
                top_tokens=(),
                finish_reason="stop",
            ),
        ]
    )
    runtime = TokenityDistributedRuntime(
        model="/models/qwen",
        state=ReadinessState(model="/models/qwen", phase=ReadinessPhase.READY),
    )
    runtime._begin_generation = lambda _: (context, responses)  # type: ignore[method-assign]  # noqa: SLF001

    response = runtime.complete(
        ChatCompletionRequest(
            model="qwen",
            messages=[{"role": "user", "content": "weather"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            logprobs=True,
            top_logprobs=2,
        )
    )

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "I will check."
    assert choice["message"]["tool_calls"][0]["function"] == {
        "name": "weather",
        "arguments": '{"city": "Shanghai"}',
    }
    assert choice["logprobs"]["content"][0]["token"] == "I will check."
    assert choice["logprobs"]["content"][0]["top_logprobs"][1]["token"] == "We"
    assert stopped.is_set()


def test_single_token_logprobs_preserve_selected_and_top_probabilities():
    class Scalar:
        def item(self):
            return -0.25

    raw = {7: Scalar()}
    item = SimpleNamespace(token=7, logprobs=raw)
    formatter_calls = []

    token, logprob, top_tokens = _single_token_logprobs(
        item,
        tokenizer="tokenizer",
        include_logprob=True,
        top_n=2,
        top_formatter=lambda values, count, tokenizer: formatter_calls.append(
            (values, count, tokenizer)
        )
        or (
            {"token": "A", "logprob": -0.25},
            {"token": "B", "logprob": -1.0},
        ),
    )

    assert token == 7
    assert logprob == -0.25
    assert top_tokens[1]["token"] == "B"
    assert formatter_calls == [(raw, 2, "tokenizer")]


def test_repetition_detector_stops_contiguous_output_cycle():
    detector = _RepetitionDetector()
    phrase = "wait while I reconsider the instruction. "

    assert detector.observe(phrase) is False
    assert detector.observe(phrase) is False
    assert detector.observe(phrase) is True


def test_stream_reports_repetition_finish_reason_and_stops_generation_context():
    phrase = "wait while I reconsider the instruction. "
    stopped = threading.Event()
    context = SimpleNamespace(stop=stopped.set)
    responses = iter(
        SimpleNamespace(text=phrase, state="content", finish_reason=None)
        for _ in range(3)
    )
    runtime = TokenityDistributedRuntime(
        model="/models/qwen",
        state=ReadinessState(model="/models/qwen"),
    )
    runtime._begin_generation = lambda _: (context, responses)  # type: ignore[method-assign]  # noqa: SLF001

    async def collect() -> list[str]:
        request = ChatCompletionRequest(
            model="qwen",
            messages=[{"role": "user", "content": "loop"}],
            stream=True,
        )
        return [event async for event in runtime.stream(request)]

    events = asyncio.run(collect())

    assert any('"finish_reason": "tokenity_repetition"' in event for event in events)
    final_payload = json.loads(events[-2].removeprefix("data: "))
    assert final_payload["usage"]["completion_tokens"] == 2
    assert final_payload["usage"]["total_tokens"] == 2
    assert events[-1] == "data: [DONE]\n\n"
    assert stopped.is_set()


def test_first_stream_emits_keepalives_then_incremental_content_with_timeline():
    stopped = threading.Event()
    context = SimpleNamespace(prompt=[1, 2], prompt_cache_count=0, stop=stopped.set)

    def begin(_request):
        time.sleep(0.04)

        def responses():
            time.sleep(0.04)
            yield SimpleNamespace(text="first", state="content", finish_reason=None)
            time.sleep(0.02)
            yield SimpleNamespace(text=" second", state="content", finish_reason="stop")

        return context, responses()

    state = ReadinessState(model="/models/qwen", phase=ReadinessPhase.READY)
    runtime = TokenityDistributedRuntime(model="/models/qwen", state=state)
    runtime._begin_generation = begin  # type: ignore[method-assign]  # noqa: SLF001

    async def collect():
        started = time.monotonic()
        events = []
        request = ChatCompletionRequest(
            model="qwen",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        async for event in runtime.stream(request, request_id="request-timing", keepalive_interval=0.01):
            events.append((time.monotonic() - started, event))
        return events

    events = asyncio.run(collect())

    assert events[0][1].startswith(": keep-alive")
    content_events = [(at, event) for at, event in events if event.startswith("data:") and "[DONE]" not in event]
    assert len(content_events) >= 3
    assert '"content": "first"' in content_events[0][1]
    assert '"content": " second"' in content_events[1][1]
    assert content_events[0][0] < content_events[1][0]
    assert state.last_request is not None
    for event in [
        "accepted",
        "first_keepalive",
        "prefill_start",
        "prefill_end",
        "first_content_token",
        "last_token",
        "completed",
    ]:
        assert event in state.last_request
    assert stopped.is_set()


def test_glm_jaccl_stream_cancel_retires_runtime_drains_task_and_returns_503(tmp_path):
    model = tmp_path / "GLM-5.2-mxfp4"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "glm_moe_dsa"}),
        encoding="utf-8",
    )
    state = ReadinessState(
        model=str(model),
        phase=ReadinessPhase.READY,
        connection_mode="jaccl-ring",
    )
    runtime = TokenityDistributedRuntime(model=str(model), state=state)
    stopped = threading.Event()
    next_response_started = threading.Event()
    context = SimpleNamespace(prompt=[], prompt_cache_count=0, stop=stopped.set)

    def responses():
        next_response_started.set()
        assert stopped.wait(1)
        raise NotImplementedError("distributed sequential cancellation")
        yield  # pragma: no cover - make this a generator

    runtime._begin_generation = lambda _: (context, responses())  # type: ignore[method-assign]  # noqa: SLF001

    async def cancel_during_next_response() -> None:
        request = ChatCompletionRequest(
            model=runtime.model_id,
            messages=[{"role": "user", "content": "keep generating"}],
            stream=True,
        )
        stream = runtime.stream(
            request,
            request_id="cancelled-glm-stream",
            keepalive_interval=0.01,
        )
        consumer = asyncio.create_task(anext(stream))
        assert await asyncio.to_thread(next_response_started.wait, 1)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer

    asyncio.run(cancel_during_next_response())

    assert stopped.is_set()
    assert state.phase == ReadinessPhase.FAILED
    assert state.last_error is not None
    assert state.last_error["stage"] == "stream_cancelled"
    assert state.last_error["request_id"] == "cancelled-glm-stream"
    assert state.last_error["requires_reload"] is True
    assert runtime._active_request_count == 0  # noqa: SLF001
    assert runtime._pending_stream_tasks == set()  # noqa: SLF001

    # The failed state is published before cleanup completes, so all subsequent
    # OpenAI requests are rejected instead of reusing the tainted JACCL group.
    runtime.stop = lambda: None  # type: ignore[method-assign]
    with TestClient(create_app(model=str(model), runtime=runtime)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": runtime.model_id,
                "messages": [{"role": "user", "content": "second request"}],
            },
        )

    assert response.status_code == 503
    assert "Stop and reload" in response.json()["detail"]


def test_glm_jaccl_cancel_during_generation_start_recovers_context_and_drains(tmp_path):
    model = tmp_path / "GLM-5.2-mxfp4"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "glm_moe_dsa"}),
        encoding="utf-8",
    )
    state = ReadinessState(
        model=str(model),
        phase=ReadinessPhase.READY,
        connection_mode="jaccl",
    )
    runtime = TokenityDistributedRuntime(model=str(model), state=state)
    begin_started = threading.Event()
    release_begin = threading.Event()
    stopped = threading.Event()
    drained = threading.Event()
    context = SimpleNamespace(prompt=[], prompt_cache_count=0, stop=stopped.set)

    def responses():
        try:
            if False:
                yield None
        finally:
            drained.set()

    def begin(_request):
        begin_started.set()
        assert release_begin.wait(1)
        return context, responses()

    runtime._begin_generation = begin  # type: ignore[method-assign]  # noqa: SLF001

    async def cancel_during_start() -> None:
        request = ChatCompletionRequest(
            model=runtime.model_id,
            messages=[{"role": "user", "content": "cancel during prefill"}],
            stream=True,
        )
        consumer = asyncio.create_task(
            anext(runtime.stream(request, keepalive_interval=1))
        )
        assert await asyncio.to_thread(begin_started.wait, 1)
        consumer.cancel()
        release_begin.set()
        with pytest.raises(asyncio.CancelledError):
            await consumer

    asyncio.run(cancel_during_start())

    assert stopped.is_set()
    assert drained.is_set()
    assert state.phase == ReadinessPhase.FAILED
    assert runtime._pending_stream_tasks == set()  # noqa: SLF001


def test_glm_jaccl_async_generator_close_retires_runtime(tmp_path):
    model = tmp_path / "GLM-5.2-mxfp4"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "glm_moe_dsa"}),
        encoding="utf-8",
    )
    state = ReadinessState(
        model=str(model),
        phase=ReadinessPhase.READY,
        connection_mode="jaccl",
    )
    runtime = TokenityDistributedRuntime(model=str(model), state=state)
    stopped = threading.Event()
    context = SimpleNamespace(prompt=[], prompt_cache_count=0, stop=stopped.set)
    runtime._begin_generation = lambda _: (  # type: ignore[method-assign]  # noqa: SLF001
        context,
        iter([SimpleNamespace(text="first", state="content", finish_reason=None)]),
    )

    async def close_after_first_chunk() -> None:
        request = ChatCompletionRequest(
            model=runtime.model_id,
            messages=[{"role": "user", "content": "close stream"}],
            stream=True,
        )
        stream = runtime.stream(request)
        assert '"content": "first"' in await anext(stream)
        await stream.aclose()

    asyncio.run(close_after_first_chunk())

    assert stopped.is_set()
    assert state.phase == ReadinessPhase.FAILED
    assert state.last_request is not None
    assert "cancelled" in state.last_request


def test_non_glm_stream_cancel_does_not_retire_runtime():
    state = ReadinessState(
        model="/models/qwen",
        phase=ReadinessPhase.READY,
        connection_mode="jaccl",
    )
    runtime = TokenityDistributedRuntime(model="/models/qwen", state=state)
    stopped = threading.Event()
    next_response_started = threading.Event()
    context = SimpleNamespace(prompt=[], prompt_cache_count=0, stop=stopped.set)

    def responses():
        next_response_started.set()
        assert stopped.wait(1)
        return
        yield  # pragma: no cover - make this a generator

    runtime._begin_generation = lambda _: (context, responses())  # type: ignore[method-assign]  # noqa: SLF001

    async def cancel() -> None:
        request = ChatCompletionRequest(
            model=runtime.model_id,
            messages=[{"role": "user", "content": "cancel"}],
            stream=True,
        )
        consumer = asyncio.create_task(anext(runtime.stream(request)))
        assert await asyncio.to_thread(next_response_started.wait, 1)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer

    asyncio.run(cancel())

    assert stopped.is_set()
    assert state.phase == ReadinessPhase.READY
    assert state.last_error is None


def test_stream_records_headers_only_when_asgi_response_start_is_sent():
    state = ReadinessState(model="/models/qwen", phase=ReadinessPhase.READY)
    runtime = TokenityDistributedRuntime(model="/models/qwen", state=state)
    runtime._begin_generation = lambda _: (  # type: ignore[method-assign]  # noqa: SLF001
        SimpleNamespace(prompt=[], prompt_cache_count=0, stop=lambda: None),
        iter([SimpleNamespace(text="OK", state="content", finish_reason="stop")]),
    )

    with TestClient(create_app(model="/models/qwen", runtime=runtime)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-tokenity-request-id"]
    assert state.last_request is not None
    assert state.last_request["headers_sent"] >= state.last_request["accepted"]
    assert "data: [DONE]" in response.text


def test_single_runtime_source_contains_no_distributed_initialization():
    source = inspect.getsource(TokenityDistributedRuntime._start_single)  # noqa: SLF001

    assert "distributed.init" not in source
    assert "mlx_lm import load" in source
    assert "self._warmup_single()" in source
    assert "_run_warmup_with_timeout(self._warmup_single)" not in source


def test_ready_is_published_only_after_materialization_and_warmup():
    state = ReadinessState(model="/models/qwen", phase=ReadinessPhase.LOADING_MODEL)
    runtime = TokenityDistributedRuntime(model="/models/qwen", state=state)
    runtime.state.rank = 0
    runtime.state.world_size = 2
    runtime._provider = SimpleNamespace(model=object(), tokenizer=object())  # noqa: SLF001
    runtime._generator = SimpleNamespace(  # noqa: SLF001
        _generation_thread=SimpleNamespace(is_alive=lambda: True)
    )
    warmed = threading.Event()
    runtime._warmup_distributed = warmed.set  # type: ignore[method-assign]  # noqa: SLF001
    runtime._run_warmup_with_timeout = lambda warmup: warmup()  # type: ignore[method-assign]  # noqa: SLF001

    runtime._monitor_model_load()  # noqa: SLF001

    assert warmed.is_set()
    assert state.phase == ReadinessPhase.READY
    assert state.ready_evidence["weights_materialized"] is True
    assert state.ready_evidence["one_token_probe"] is True


def test_distributed_warmup_drains_response_before_stopping_context():
    events = []
    cache_sizes = []
    runtime = TokenityDistributedRuntime(
        model="/models/qwen",
        state=ReadinessState(model="/models/qwen"),
    )
    runtime._generator = SimpleNamespace(prompt_cache=object())  # noqa: SLF001
    runtime._symbols = SimpleNamespace(  # noqa: SLF001
        LRUPromptCache=lambda size: cache_sizes.append(size) or object(),
    )
    context = SimpleNamespace(stop=lambda: events.append("stopped"))

    def responses():
        yield SimpleNamespace(text="OK")
        events.append("drained")

    runtime._begin_distributed_generation = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda _request: (context, responses())
    )

    runtime._warmup_distributed()  # noqa: SLF001

    assert events == ["drained", "stopped"]
    assert cache_sizes == [runtime.prompt_cache_size]


def test_glm_jaccl_uses_sequential_generation_engine(tmp_path):
    model = tmp_path / "GLM-5.2-mxfp4"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "glm_moe_dsa"}),
        encoding="utf-8",
    )
    state = ReadinessState(model=str(model), connection_mode="jaccl")
    runtime = TokenityDistributedRuntime(model=str(model), state=state)
    runtime._provider = SimpleNamespace(is_batchable=True)  # noqa: SLF001
    runtime._generator = SimpleNamespace(prompt_cache=object())  # noqa: SLF001
    cache_sizes = []
    runtime._symbols = SimpleNamespace(  # noqa: SLF001
        LRUPromptCache=lambda size: cache_sizes.append(size) or object(),
    )

    runtime._configure_distributed_generation_mode()  # noqa: SLF001

    assert runtime._provider.is_batchable is False  # noqa: SLF001
    assert runtime._distributed_prompt_cache_size == 0  # noqa: SLF001
    assert cache_sizes == [0]


def test_qwen_jaccl_uses_sequential_generation_engine(tmp_path):
    model = tmp_path / "Qwen3.5-122B-A10B-4bit"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe"}),
        encoding="utf-8",
    )
    state = ReadinessState(model=str(model), connection_mode="jaccl-ring")
    runtime = TokenityDistributedRuntime(model=str(model), state=state)
    runtime._provider = SimpleNamespace(is_batchable=True)  # noqa: SLF001
    runtime._generator = SimpleNamespace(prompt_cache=object())  # noqa: SLF001
    cache_sizes = []
    runtime._symbols = SimpleNamespace(  # noqa: SLF001
        LRUPromptCache=lambda size: cache_sizes.append(size) or object(),
    )

    runtime._configure_distributed_generation_mode()  # noqa: SLF001

    assert runtime._provider.is_batchable is False  # noqa: SLF001
    assert runtime._distributed_prompt_cache_size == 0  # noqa: SLF001
    assert cache_sizes == [0]


def test_qwen_jaccl_prepares_sequential_mode_before_generation_thread_starts(tmp_path):
    model = tmp_path / "Qwen3.5-122B-A10B-4bit"
    model.mkdir()
    provider = SimpleNamespace(is_batchable=True)

    def load_default():
        provider.is_batchable = True

    provider.load_default = load_default
    runtime = TokenityDistributedRuntime(
        model=str(model),
        state=ReadinessState(model=str(model), connection_mode="jaccl-ring"),
    )
    runtime._provider = provider  # noqa: SLF001

    runtime._prepare_distributed_generation_mode()  # noqa: SLF001
    provider.load_default()

    assert runtime._distributed_prompt_cache_size == 0  # noqa: SLF001
    assert provider.is_batchable is False


def test_glm_jaccl_warmup_does_not_reenable_prompt_cache(tmp_path):
    model = tmp_path / "GLM-5.2-mxfp4"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "glm_moe_dsa"}),
        encoding="utf-8",
    )
    runtime = TokenityDistributedRuntime(
        model=str(model),
        state=ReadinessState(model=str(model), connection_mode="jaccl"),
    )
    runtime._provider = SimpleNamespace(is_batchable=True)  # noqa: SLF001
    runtime._generator = SimpleNamespace(prompt_cache=object())  # noqa: SLF001
    cache_sizes = []
    runtime._symbols = SimpleNamespace(  # noqa: SLF001
        LRUPromptCache=lambda size: cache_sizes.append(size) or object(),
    )
    runtime._configure_distributed_generation_mode()  # noqa: SLF001
    context = SimpleNamespace(stop=lambda: None)
    runtime._begin_distributed_generation = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda _request: (context, iter([SimpleNamespace(text="OK")]))
    )

    runtime._warmup_distributed()  # noqa: SLF001

    assert cache_sizes == [0, 0]


def test_jaccl_post_load_barrier_uses_multi_element_cpu_collective():
    source = inspect.getsource(_install_chunked_sharded_load)

    assert "mx.ones(10)" in source
    assert "stream=mx.cpu" in source
    assert "_should_run_post_load_barrier(repo)" in source


def test_glm_skips_redundant_jaccl_post_load_barrier(tmp_path, monkeypatch):
    glm = tmp_path / "GLM-5.2-mxfp4"
    qwen = tmp_path / "Qwen3.5-122B"
    glm.mkdir()
    qwen.mkdir()
    (glm / "config.json").write_text(
        json.dumps({"model_type": "glm_moe_dsa"}),
        encoding="utf-8",
    )
    (qwen / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TOKENITY_MLX_LOAD_POST_BARRIER", "1")

    assert _should_run_post_load_barrier(str(glm)) is False
    assert _should_run_post_load_barrier(str(qwen)) is True

    monkeypatch.setenv("TOKENITY_MLX_LOAD_POST_BARRIER", "0")
    assert _should_run_post_load_barrier(str(qwen)) is False


def test_glm_jaccl_does_not_wire_the_model_shard(tmp_path):
    glm = tmp_path / "GLM-5.2-mxfp4"
    qwen = tmp_path / "Qwen3.5-122B"
    glm.mkdir()
    qwen.mkdir()
    (glm / "config.json").write_text(
        json.dumps({"model_type": "glm_moe_dsa"}),
        encoding="utf-8",
    )
    (qwen / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe"}),
        encoding="utf-8",
    )
    wired_limits = []
    mx = SimpleNamespace(
        metal=SimpleNamespace(is_available=lambda: True),
        device_info=lambda: {"max_recommended_working_set_size": 123},
        set_wired_limit=wired_limits.append,
    )

    assert _configure_mlx_wired_memory(
        mx,
        model=str(glm),
        execution_mode="distributed",
        connection_mode="jaccl-ring",
    ) == 0
    assert _configure_mlx_wired_memory(
        mx,
        model=str(qwen),
        execution_mode="distributed",
        connection_mode="jaccl-ring",
    ) == 123
    assert _configure_mlx_wired_memory(
        mx,
        model=str(glm),
        execution_mode="single",
        connection_mode="single",
    ) == 123
    assert wired_limits == [0, 123, 123]


def test_jaccl_request_control_avoids_one_element_collectives():
    source = inspect.getsource(_install_jaccl_server_control_collectives)

    assert "mx.zeros((_JACCL_CONTROL_WORDS,)" in source
    assert "_replace_jaccl_sequential_cancel_collective" in source
    assert "_jaccl_control_header(" in source
    assert "_validate_jaccl_control_header(" in source
    assert "_validate_jaccl_control_payload(" in source
    assert "stream=mx.cpu" in source
    assert "all_sum(0)" not in source
    assert "_TOKENITY_DISTRIBUTED_STOP" in source
    assert "set_wired_limit(0)" in source
    assert "clear_streams" in source


def test_jaccl_control_frame_rejects_stale_epoch_sequence_and_checksum():
    epoch = (1, 2, 3, 4)
    payload = b"current-operation"
    header = _jaccl_control_header(
        sequence=7,
        payload=payload,
        epoch_words=epoch,
    )

    assert _validate_jaccl_control_header(
        header,
        expected_sequence=7,
        expected_epoch_words=epoch,
    ) == (1, len(payload), header[-1])
    _validate_jaccl_control_payload(payload, header[-1])

    with pytest.raises(RuntimeError, match="sequence"):
        _validate_jaccl_control_header(
            header,
            expected_sequence=8,
            expected_epoch_words=epoch,
        )
    with pytest.raises(RuntimeError, match="epoch"):
        _validate_jaccl_control_header(
            header,
            expected_sequence=7,
            expected_epoch_words=(5, 6, 7, 8),
        )
    with pytest.raises(RuntimeError, match="checksum"):
        _validate_jaccl_control_payload(b"stale-operation", header[-1])


def test_jaccl_control_frame_rejects_oversized_payload_before_collective(
    monkeypatch,
):
    monkeypatch.setattr(
        "tokenity.serving.distributed_openai._JACCL_CONTROL_MAX_PAYLOAD_BYTES",
        4,
    )
    with pytest.raises(RuntimeError, match="64 MiB"):
        _jaccl_control_header(
            sequence=1,
            payload=b"12345",
            epoch_words=(1, 2, 3, 4),
        )


def test_jaccl_generation_worker_uses_local_synchronized_seed():
    source = """\
def _generate(self):
    if self._is_distributed:
        seed = mx.distributed.all_sum(mx.random.state[0]).view(mx.uint64).item()
        mx.random.seed(seed)
    while not self._stop:
        pass
"""

    patched = _replace_jaccl_seed_collective(source, 12345)

    assert "mx.random.seed(12345)" in patched
    assert "mx.distributed.all_sum(mx.random.state[0])" not in patched


def test_jaccl_seed_patch_rejects_unexpected_mlx_lm_source():
    with pytest.raises(RuntimeError, match="pinned JACCL compatibility contract"):
        _replace_jaccl_seed_collective("def _generate(self): pass", 12345)


def test_jaccl_sequential_cancel_patch_synchronizes_all_ranks():
    source = """\
def _serve_single(self, request):
    try:
        for gen in responses:
            if ctx._should_stop:
                if self._is_distributed:
                    raise NotImplementedError()
                break
"""

    patched = _replace_jaccl_sequential_cancel_collective(source)

    assert "[int(tokenity_cancelled)] * 10" in patched
    assert "stream=mx.cpu" in patched
    assert "tokenity_cancelled = bool(tokenity_cancel_control[0].item())" in patched


def test_jaccl_sequential_cancel_patch_rejects_unexpected_mlx_lm_source():
    with pytest.raises(RuntimeError, match="pinned JACCL compatibility contract"):
        _replace_jaccl_sequential_cancel_collective(
            "def _serve_single(self, request): pass"
        )


def test_runtime_rejects_qwen35_mtp_draft_checkpoint_before_mlx_init(tmp_path):
    draft = tmp_path / "Qwen3.5-4B-MTP-4bit"
    draft.mkdir()
    (draft / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_mtp"}),
        encoding="utf-8",
    )
    state = ReadinessState(model=str(draft))
    runtime = TokenityDistributedRuntime(model=str(draft), state=state)

    try:
        runtime.start()
    except ValueError as exc:
        assert "draft model" in str(exc)
    else:
        raise AssertionError("draft-only MTP checkpoint must be rejected")

    assert state.phase.value == "failed"


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


def test_adaptive_load_policy_bypasses_legacy_agent_throttling(monkeypatch):
    monkeypatch.delenv("TOKENITY_MLX_LOAD_POLICY", raising=False)
    monkeypatch.setenv("TOKENITY_MLX_LOAD_EVAL_CHUNK_SIZE", "1")
    monkeypatch.setenv("TOKENITY_MLX_LOAD_EVAL_SLEEP_SECONDS", "0.05")

    policy = _load_eval_policy()

    assert policy.name == "adaptive"
    assert policy.max_leaves == 64
    assert policy.target_bytes == 256 * 1024 * 1024
    assert policy.sleep_seconds == 0


def test_fixed_load_policy_preserves_explicit_memory_throttling(monkeypatch):
    monkeypatch.setenv("TOKENITY_MLX_LOAD_POLICY", "fixed")
    monkeypatch.setenv("TOKENITY_MLX_LOAD_EVAL_CHUNK_SIZE", "3")
    monkeypatch.setenv("TOKENITY_MLX_LOAD_EVAL_SLEEP_SECONDS", "0.125")

    policy = _load_eval_policy()

    assert policy.name == "fixed"
    assert policy.max_leaves == 3
    assert policy.target_bytes is None
    assert policy.sleep_seconds == 0.125


def test_adaptive_parameter_chunks_respect_leaf_and_byte_limits():
    parameters = [(str(index), SimpleNamespace(nbytes=size)) for index, size in enumerate([4, 4, 4, 20, 1])]
    policy = _LoadEvalPolicy(name="adaptive", max_leaves=4, target_bytes=10, sleep_seconds=0)

    chunks = _parameter_eval_chunks(parameters, policy)

    assert [[value.nbytes for _, value in chunk] for chunk in chunks] == [[4, 4], [4], [20], [1]]


def test_adaptive_parameter_eval_batches_without_per_leaf_sleep(monkeypatch):
    monkeypatch.setenv("TOKENITY_MLX_LOAD_POLICY", "adaptive")
    monkeypatch.setenv("TOKENITY_MLX_LOAD_ADAPTIVE_MAX_LEAVES", "64")
    monkeypatch.setenv("TOKENITY_MLX_LOAD_ADAPTIVE_TARGET_BYTES", str(1024 * 1024))
    monkeypatch.setattr(
        "tokenity.serving.distributed_openai.time.sleep",
        lambda _: (_ for _ in ()).throw(AssertionError("adaptive loading must not sleep")),
    )

    class FakeMLX:
        def __init__(self):
            self.eval_sizes = []

        def eval(self, values):
            self.eval_sizes.append(len(values))

    mx = FakeMLX()
    parameters = [(str(index), SimpleNamespace(nbytes=1)) for index in range(130)]

    _eval_parameters_in_chunks(mx, parameters)

    assert mx.eval_sizes == [64, 64, 2]


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


def test_coordinated_glm_release_clears_metal_cache_before_process_exit(
    tmp_path,
    monkeypatch,
):
    glm = tmp_path / "GLM-5.2-mxfp4"
    glm.mkdir()
    (glm / "config.json").write_text(
        json.dumps({"model_type": "glm_moe_dsa"}),
        encoding="utf-8",
    )
    calls = []
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.__path__ = []  # type: ignore[attr-defined]
    fake_core = types.ModuleType("mlx.core")
    fake_core.set_wired_limit = lambda limit: calls.append(f"set_wired_limit:{limit}")  # type: ignore[attr-defined]
    fake_core.clear_streams = lambda: calls.append("clear_streams")  # type: ignore[attr-defined]
    fake_core.clear_cache = lambda: calls.append("clear_cache")  # type: ignore[attr-defined]
    fake_core.get_active_memory = lambda: 0  # type: ignore[attr-defined]
    fake_core.get_cache_memory = lambda: 0  # type: ignore[attr-defined]
    fake_mlx.core = fake_core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    state = ReadinessState(
        model=str(glm),
        rank=1,
        world_size=2,
        connection_mode="jaccl-ring",
    )
    runtime = TokenityDistributedRuntime(model=str(glm), state=state)
    runtime._generator = object()  # noqa: SLF001 - targeted lifecycle regression
    runtime._provider = object()  # noqa: SLF001 - targeted lifecycle regression
    runtime._symbols = object()  # noqa: SLF001 - targeted lifecycle regression
    runtime._group = object()  # noqa: SLF001 - targeted lifecycle regression

    runtime._release_memory()  # noqa: SLF001 - targeted lifecycle regression

    assert calls == ["set_wired_limit:0", "clear_streams", "clear_cache"]


def test_coordinated_glm_shutdown_leaves_group_teardown_to_interpreter(
    tmp_path,
    monkeypatch,
):
    glm = tmp_path / "GLM-5.2-mxfp4"
    glm.mkdir()
    (glm / "config.json").write_text(
        json.dumps({"model_type": "glm_moe_dsa"}),
        encoding="utf-8",
    )
    exits = []
    releases = []
    generation_thread = SimpleNamespace(
        join=lambda timeout: None,
        is_alive=lambda: False,
    )
    runtime = TokenityDistributedRuntime(
        model=str(glm),
        state=ReadinessState(
            model=str(glm),
            rank=0,
            world_size=2,
            connection_mode="jaccl-ring",
        ),
    )
    runtime._generator = SimpleNamespace(  # noqa: SLF001
        _generation_thread=generation_thread,
        _stop=False,
        _tokenity_shutdown_requested=False,
        stop_and_join=lambda: None,
    )
    runtime._release_memory = lambda: releases.append("released")  # type: ignore[method-assign]  # noqa: SLF001
    monkeypatch.setattr(
        "tokenity.serving.distributed_openai.os._exit",
        lambda code: exits.append(code),
    )

    runtime.stop()

    assert releases == ["released"]
    assert exits == []
    assert "os._exit" not in inspect.getsource(serve)


def test_glm_process_isolated_shutdown_coordinates_generation_thread_exit(tmp_path):
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

    runtime = TokenityDistributedRuntime(
        model=str(glm),
        state=ReadinessState(
            model=str(glm),
            connection_mode="jaccl-ring",
        ),
    )
    generator = SimpleNamespace(
        _stop=False,
        _tokenity_shutdown_requested=False,
    )
    runtime._generator = generator  # noqa: SLF001 - targeted GLM teardown regression
    runtime.request_stop()

    assert runtime.process_isolated_shutdown is True
    assert generator._stop is False
    assert generator._tokenity_shutdown_requested is True
    assert runtime.state.phase.value == "stopping"


def test_glm_compatibility_uses_metadata_not_directory_name(tmp_path):
    disguised = tmp_path / "GLM-5.2-by-name-only"
    disguised.mkdir()
    (disguised / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )

    assert _requires_process_isolated_shutdown(str(disguised)) is False


def test_qwen_jaccl_shutdown_uses_the_validated_stop_sentinel(tmp_path):
    qwen = tmp_path / "Qwen3.5-4B"
    qwen.mkdir()
    (qwen / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )
    runtime = TokenityDistributedRuntime(
        model=str(qwen),
        state=ReadinessState(
            model=str(qwen),
            connection_mode="jaccl-ring",
        ),
    )
    generator = SimpleNamespace(
        _stop=False,
        _tokenity_shutdown_requested=False,
    )
    runtime._generator = generator  # noqa: SLF001 - targeted JACCL teardown regression

    runtime.request_stop()

    assert runtime.process_isolated_shutdown is False
    assert generator._stop is False
    assert generator._tokenity_shutdown_requested is True
