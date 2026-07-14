from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import signal
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tokenity.inference.native_mtp import NativeMTPConfig
from tokenity.inference.native_mtp.runtime import (
    NativeMTPRuntimeController,
    NativeMTPStartupError,
    attach_controller,
    controller_for,
)

from .readiness import ReadinessPhase, ReadinessState


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0, le=1)
    stop: str | list[str] | None = None
    seed: int | None = None
    tools: list[Any] | None = None
    role_mapping: dict[str, Any] | None = None
    chat_template_kwargs: dict[str, Any] | None = None


@dataclass
class _GenerationResult:
    text: str
    reasoning: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    prompt_cache_tokens: int | None


_LOAD_PROGRESS_LOCK = threading.Lock()
_ACTIVE_LOAD_STATE: ReadinessState | None = None
_ACTIVE_LOAD_CANCEL: threading.Event | None = None


class _ModelLoadCancelled(SystemExit):
    pass


def _set_active_load_state(
    state: ReadinessState | None,
    cancel_event: threading.Event | None = None,
) -> None:
    global _ACTIVE_LOAD_STATE, _ACTIVE_LOAD_CANCEL
    with _LOAD_PROGRESS_LOCK:
        _ACTIVE_LOAD_STATE = state
        _ACTIVE_LOAD_CANCEL = cancel_event


def _load_was_cancelled() -> bool:
    with _LOAD_PROGRESS_LOCK:
        cancel_event = _ACTIVE_LOAD_CANCEL
    return cancel_event is not None and cancel_event.is_set()


def _report_load_progress(current: int, total: int) -> None:
    with _LOAD_PROGRESS_LOCK:
        state = _ACTIVE_LOAD_STATE
    if state is None or total <= 0:
        return
    state.progress_current = current
    state.progress_total = total
    # Reserve the first 10% for distributed initialization and the final 5%
    # for provider publication/compilation.
    state.progress = min(0.95, 0.1 + (0.85 * current / total))
    state.message = f"Loading model parameters ({current}/{total})."


@dataclass
class _MLXServerSymbols:
    CompletionRequest: type
    GenerationArguments: type
    LogitsProcessorArguments: type
    LRUPromptCache: type
    ModelDescription: type
    ModelProvider: type
    ResponseGenerator: type
    SamplingArguments: type

    @classmethod
    def load(
        cls,
        native_mtp: NativeMTPRuntimeController | None = None,
    ) -> "_MLXServerSymbols":
        import mlx_lm.server as server  # type: ignore

        from tokenity.mlx.glm_moe_dsa_compat import install_glm_moe_dsa_compat

        if install_glm_moe_dsa_compat():
            logging.warning(
                "Tokenity installed GLM-5.2 cross-layer indexer sharing compatibility from mlx-lm PR #1410."
            )
        if native_mtp is not None:
            attach_controller(server, native_mtp)
        _install_chunked_sharded_load(server)
        from mlx_lm.server import (  # type: ignore
            CompletionRequest,
            GenerationArguments,
            LogitsProcessorArguments,
            LRUPromptCache,
            ModelDescription,
            ModelProvider,
            ResponseGenerator,
            SamplingArguments,
        )

        return cls(
            CompletionRequest=CompletionRequest,
            GenerationArguments=GenerationArguments,
            LogitsProcessorArguments=LogitsProcessorArguments,
            LRUPromptCache=LRUPromptCache,
            ModelDescription=ModelDescription,
            ModelProvider=ModelProvider,
            ResponseGenerator=ResponseGenerator,
            SamplingArguments=SamplingArguments,
        )


def _install_chunked_sharded_load(server: Any) -> None:
    if getattr(server, "_tokenity_chunked_sharded_load", False):
        return

    from mlx.utils import tree_flatten  # type: ignore
    from mlx_lm.utils import _download, load_model, load_tokenizer  # type: ignore

    original_load = server.load

    def tokenity_load(*args: Any, **kwargs: Any) -> Any:
        controller = controller_for(server)
        scope = controller.construction_scope() if controller is not None else nullcontext()
        with scope:
            result = original_load(*args, **kwargs)
        if controller is not None:
            controller.validate_loaded_model(result[0])
        return result

    def tokenity_sharded_load(
        repo: str,
        pipeline_group: Any = None,
        tensor_group: Any = None,
        return_config: bool = False,
        *,
        tokenizer_config: dict[str, Any] | None = None,
    ) -> Any:
        import mlx.core as mx  # type: ignore

        controller = controller_for(server)
        scope = controller.construction_scope() if controller is not None else nullcontext()
        with scope:
            model_path = _download(
                repo,
                allow_patterns=[
                    "*.json",
                    "*.py",
                    "tokenizer.model",
                    "*.tiktoken",
                    "tiktoken.model",
                    "*.txt",
                    "*.jsonl",
                    "*.jinja",
                ],
            )
            model, config = load_model(model_path, lazy=True, strict=False)

            has_pipelining = hasattr(model, "model") and hasattr(model.model, "pipeline")
            has_tensor_parallel = hasattr(model, "shard")
            if pipeline_group is not None and not has_pipelining:
                raise ValueError("The model does not support pipelining but a pipeline_group was provided")
            if tensor_group is not None and not has_tensor_parallel:
                raise ValueError("The model does not support tensor parallelism but a tensor_group was provided")
            if not has_pipelining and not has_tensor_parallel:
                raise ValueError("The model does not support any sharding")
            if pipeline_group is tensor_group is None:
                if has_tensor_parallel:
                    tensor_group = mx.distributed.init()
                elif has_pipelining:
                    pipeline_group = mx.distributed.init()

            if pipeline_group is not None:
                model.model.pipeline(pipeline_group)
                with open(model_path / "model.safetensors.index.json", "r") as handle:
                    weight_index = json.load(handle)["weight_map"]
                local_files = set()
                for key, _ in tree_flatten(model.parameters()):
                    file_name = weight_index.get(key)
                    if file_name is None:
                        raise ValueError("Pipeline loading is only supported for MLX converted models.")
                    local_files.add(file_name)
                _download(repo, allow_patterns=local_files)
            else:
                _download(repo)

            tokenizer = load_tokenizer(
                model_path,
                tokenizer_config or {"trust_remote_code": True},
                eos_token_ids=config.get("eos_token_id", None),
            )
            model, _ = load_model(model_path, lazy=True, strict=False)
            if tensor_group is not None and int(tensor_group.size()) > 1:
                model.shard(tensor_group)
            if pipeline_group is not None:
                model.model.pipeline(pipeline_group)
        _eval_parameters_in_chunks(mx, tree_flatten(model.parameters()))
        if controller is not None:
            controller.validate_loaded_model(model)
        if _bool_env("TOKENITY_MLX_LOAD_POST_BARRIER", False):
            logging.warning("Tokenity chunked sharded_load running post-load distributed all_sum barrier.")
            mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
            logging.warning("Tokenity chunked sharded_load finished post-load distributed all_sum barrier.")
        else:
            logging.warning("Tokenity chunked sharded_load skipped post-load distributed all_sum barrier.")
        if return_config:
            return model, tokenizer, config
        return model, tokenizer

    server.load = tokenity_load
    server.sharded_load = tokenity_sharded_load
    server._tokenity_chunked_sharded_load = True


def _eval_parameters_in_chunks(mx: Any, flattened_parameters: list[tuple[str, Any]]) -> None:
    chunk_size = _positive_int_env("TOKENITY_MLX_LOAD_EVAL_CHUNK_SIZE", 1)
    log_interval = _positive_int_env("TOKENITY_MLX_LOAD_EVAL_LOG_INTERVAL", 100)
    sleep_seconds = _nonnegative_float_env("TOKENITY_MLX_LOAD_EVAL_SLEEP_SECONDS", 0.05)
    total_chunks = (len(flattened_parameters) + chunk_size - 1) // chunk_size
    logging.warning(
        "Tokenity chunked sharded_load evaluating %s parameter leaves in %s chunks of %s.",
        len(flattened_parameters),
        total_chunks,
        chunk_size,
    )
    _report_load_progress(0, total_chunks)
    for start in range(0, len(flattened_parameters), chunk_size):
        if _load_was_cancelled():
            raise _ModelLoadCancelled("Tokenity model loading was cancelled.")
        chunk = flattened_parameters[start : start + chunk_size]
        chunk_index = start // chunk_size + 1
        should_log = chunk_index == 1 or chunk_index == total_chunks or chunk_index % log_interval == 0
        if should_log:
            logging.warning(
                "Tokenity chunked sharded_load eval chunk %s/%s: %s.",
                chunk_index,
                total_chunks,
                _describe_eval_chunk(chunk),
            )
        mx.eval([value for _, value in chunk])
        if _load_was_cancelled():
            raise _ModelLoadCancelled("Tokenity model loading was cancelled.")
        _report_load_progress(chunk_index, total_chunks)
        if should_log:
            logging.warning(
                "Tokenity chunked sharded_load finished chunk %s/%s.",
                chunk_index,
                total_chunks,
            )
        if sleep_seconds:
            time.sleep(sleep_seconds)


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _nonnegative_float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _delay_rank0_distributed_init() -> None:
    delay_seconds = _nonnegative_float_env("TOKENITY_MLX_DISTRIBUTED_INIT_RANK0_DELAY_SECONDS", 0.0)
    if delay_seconds <= 0 or os.environ.get("MLX_RANK") != "0":
        return
    logging.warning(
        "Tokenity delaying MLX distributed init on rank 0 for %.2f seconds.",
        delay_seconds,
    )
    time.sleep(delay_seconds)


def _requires_process_isolated_shutdown(model: str) -> bool:
    """Return whether model teardown must bypass MLX thread destructors.

    MLX 0.31.x can double-free the thread-local CompilerCache when the GLM
    generation thread exits after JACCL inference.  The runtime already lives
    in a disposable subprocess, so an explicit ``os._exit(0)`` is the safest
    ownership boundary: macOS reclaims Metal/JACCL resources without running
    the faulty C++ TLS destructor.
    """

    config_path = Path(model) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        config = {}
    identities = [str(config.get("model_type", "")), Path(model).name]
    architectures = config.get("architectures")
    if isinstance(architectures, list):
        identities.extend(str(item) for item in architectures)
    identity = " ".join(identities).lower()
    return "glm" in identity and ("moe" in identity or "5.2" in identity)


def _describe_eval_chunk(chunk: list[tuple[str, Any]]) -> str:
    descriptions = []
    for name, value in chunk:
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        nbytes = _array_nbytes(value)
        details = [name]
        if shape is not None:
            details.append(f"shape={shape}")
        if dtype is not None:
            details.append(f"dtype={dtype}")
        if nbytes is not None:
            details.append(f"bytes={nbytes}")
        descriptions.append(" ".join(details))
    return "; ".join(descriptions)


def _array_nbytes(value: Any) -> int | None:
    nbytes = getattr(value, "nbytes", None)
    if callable(nbytes):
        nbytes = nbytes()
    if isinstance(nbytes, int):
        return nbytes
    return None


class TokenityDistributedRuntime:
    def __init__(
        self,
        *,
        model: str,
        state: ReadinessState,
        max_tokens: int = 32_768,
        trust_remote_code: bool = False,
        api_identifier: str | None = None,
        prompt_cache_size: int = 4,
        prefill_step_size: int = 2_048,
        decode_concurrency: int = 1,
        prompt_concurrency: int = 1,
        native_mtp: NativeMTPConfig | dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.model_id = (api_identifier or "").strip() or Path(model).name or model
        self.state = state
        self.max_tokens = max_tokens
        self.trust_remote_code = trust_remote_code
        self.prompt_cache_size = prompt_cache_size
        self.prefill_step_size = prefill_step_size
        self.decode_concurrency = decode_concurrency
        self.prompt_concurrency = prompt_concurrency
        self.native_mtp = NativeMTPRuntimeController(
            model=model,
            config=(
                native_mtp
                if isinstance(native_mtp, NativeMTPConfig)
                else NativeMTPConfig.from_mapping(native_mtp)
            ),
            decode_concurrency=decode_concurrency,
        )
        self.state.native_mtp = self.native_mtp.telemetry
        self.process_isolated_shutdown = _requires_process_isolated_shutdown(model)
        self._symbols: _MLXServerSymbols | None = None
        self._provider: Any = None
        self._generator: Any = None
        self._group: Any = None
        self._load_monitor: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self.state.phase = ReadinessPhase.DISTRIBUTED_INIT
        self.state.progress = 0.02
        self.state.message = "Initializing the MLX distributed group."
        _set_active_load_state(self.state, self._stop_event)
        try:
            import mlx.core as mx  # type: ignore

            if mx.metal.is_available():
                mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
            _delay_rank0_distributed_init()
            self._group = mx.distributed.init()
        except Exception as exc:  # pragma: no cover - depends on target MLX runtime
            self.state.phase = ReadinessPhase.FAILED
            self.state.message = f"MLX distributed init failed: {exc}"
            _set_active_load_state(None)
            raise

        self.state.rank = int(self._group.rank())
        self.state.world_size = int(self._group.size())
        self.state.backend = "mlx-distributed" if self.state.world_size > 1 else "single"

        self.native_mtp.prepare(mx, self._group)
        self.state.native_mtp = self.native_mtp.telemetry

        self.state.phase = ReadinessPhase.LOADING_MODEL
        self.state.progress = 0.1
        self._symbols = _MLXServerSymbols.load(self.native_mtp)
        args = self._server_args()
        self._provider = self._symbols.ModelProvider(args)
        self._map_model_aliases(self._provider)
        cache = self._symbols.LRUPromptCache(args.prompt_cache_size)
        self._generator = self._symbols.ResponseGenerator(self._provider, cache)
        self.state.message = "Loading model across MLX ranks."
        self._load_monitor = threading.Thread(target=self._monitor_model_load, daemon=True)
        self._load_monitor.start()

    def is_rank0(self) -> bool:
        return self.state.rank == 0

    def join(self) -> None:
        if self._generator is not None:
            try:
                self._generator.join()
            finally:
                self._release_memory()

    def request_stop(self) -> None:
        self._stop_event.set()
        self.state.phase = ReadinessPhase.STOPPING
        self.state.message = "Stopping model runtime and releasing MLX memory."
        if self._generator is not None and not self.process_isolated_shutdown:
            self._generator._stop = True

    def stop(self) -> None:
        self.request_stop()
        if self.process_isolated_shutdown:
            logging.warning(
                "Tokenity is using process-isolated GLM teardown to bypass the MLX CompilerCache destructor bug."
            )
            os._exit(0)
        if self._generator is not None:
            self.state.phase = ReadinessPhase.STOPPING
            self.state.message = "Stopping model runtime and releasing MLX memory."
            try:
                self._generator.stop_and_join()
            finally:
                self._release_memory()

    def _release_memory(self) -> None:
        distributed_runtime = self.state.world_size > 1
        _set_active_load_state(None)
        self._generator = None
        self._provider = None
        self._symbols = None
        self._group = None
        gc.collect()
        if distributed_runtime:
            # A distributed process exits immediately after shutdown, so the
            # OS will reclaim its Metal allocations. Calling mx.clear_cache()
            # while JACCL/MLX ranks are dismantling their groups can segfault
            # GLM on rank 0 or leave a worker blocked in teardown.
            return
        try:
            import mlx.core as mx  # type: ignore

            mx.clear_cache()
        except Exception:
            logging.exception("Tokenity could not clear the MLX memory cache during shutdown")

    def accepts_model(self, model: str) -> bool:
        return model in {"default_model", self.model, self.model_id}

    def models_payload(self) -> dict[str, object]:
        if not self._is_serving_model():
            raise RuntimeError(self.state.message or "Model is still loading.")
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_id,
                    "object": "model",
                    "owned_by": "tokenity",
                    "path": self.model,
                    "tokenity": self.state.to_dict(),
                }
            ],
        }

    def complete(self, request: ChatCompletionRequest) -> dict[str, object]:
        result = self._complete_sync(request)
        return {
            "id": f"chatcmpl-tokenity-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": _message_payload(result.text, result.reasoning),
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
                "prompt_tokens_details": {
                    "cached_tokens": result.prompt_cache_tokens or 0,
                },
            },
        }

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        ctx = None
        finish_reason = "stop"
        try:
            ctx, responses = await asyncio.to_thread(self._begin_generation, request)
            self.state.phase = ReadinessPhase.GENERATING
            while True:
                item = await asyncio.to_thread(_next_response, responses)
                if item is _DONE:
                    break
                finish_reason = getattr(item, "finish_reason", None) or finish_reason
                payload = _stream_payload(item, self.model_id, finish_reason=None)
                if payload is not None:
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            final = {
                "id": f"chatcmpl-tokenity-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if ctx is not None:
                ctx.stop()
            self.state.phase = ReadinessPhase.READY

    def _complete_sync(self, request: ChatCompletionRequest) -> _GenerationResult:
        ctx = None
        text = ""
        reasoning = ""
        finish_reason = "stop"
        tokens = 0
        try:
            ctx, responses = self._begin_generation(request)
            self.state.phase = ReadinessPhase.GENERATING
            for item in responses:
                tokens += 1
                finish_reason = item.finish_reason or finish_reason
                if item.state == "reasoning":
                    reasoning += item.text
                elif item.state != "tool":
                    text += item.text
            return _GenerationResult(
                text=text,
                reasoning=reasoning,
                finish_reason=finish_reason,
                prompt_tokens=len(ctx.prompt),
                completion_tokens=tokens,
                prompt_cache_tokens=ctx.prompt_cache_count,
            )
        finally:
            if ctx is not None:
                ctx.stop()
            self.state.phase = ReadinessPhase.READY

    def _begin_generation(self, request: ChatCompletionRequest) -> tuple[Any, Iterator[Any]]:
        if self._symbols is None or self._generator is None:
            raise RuntimeError("Tokenity distributed runtime has not started.")
        if not self._is_serving_model():
            raise RuntimeError(self.state.message or "Model is still loading.")
        if not self.accepts_model(request.model):
            raise ValueError(f"Model is not loaded: {request.model}")
        if request.seed is not None and self.native_mtp.enabled:
            self.native_mtp.record_seeded_fallback()
            logging.info(
                "Native MTP request fallback: native_mtp_reason=seeded_sequential_path seed=%s",
                request.seed,
            )
        self.state.phase = ReadinessPhase.PREFILL_PENDING
        completion_request = self._symbols.CompletionRequest(
            "chat",
            "",
            request.messages,
            request.tools,
            request.role_mapping,
        )
        return self._generator.generate(
            completion_request,
            self._generation_args(request),
        )

    def _generation_args(self, request: ChatCompletionRequest) -> Any:
        assert self._symbols is not None
        max_tokens = request.max_tokens or self.max_tokens
        return self._symbols.GenerationArguments(
            model=self._symbols.ModelDescription(
                model="default_model",
                draft="default_model",
                adapter=None,
            ),
            sampling=self._symbols.SamplingArguments(
                temperature=request.temperature if request.temperature is not None else 0.0,
                top_p=request.top_p if request.top_p is not None else 1.0,
                top_k=request.top_k if request.top_k is not None else 0,
                min_p=request.min_p if request.min_p is not None else 0.0,
                xtc_probability=0.0,
                xtc_threshold=0.0,
            ),
            logits=self._symbols.LogitsProcessorArguments(
                logit_bias=None,
                repetition_penalty=0.0,
                repetition_context_size=20,
                presence_penalty=0.0,
                presence_context_size=20,
                frequency_penalty=0.0,
                frequency_context_size=20,
            ),
            stop_words=_stop_words(request.stop),
            max_tokens=max_tokens,
            num_draft_tokens=0,
            logprobs=False,
            top_logprobs=-1,
            seed=request.seed,
            chat_template_kwargs=request.chat_template_kwargs,
        )

    def _server_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            model=self.model,
            adapter_path=None,
            draft_model=None,
            num_draft_tokens=0,
            trust_remote_code=self.trust_remote_code,
            chat_template="",
            use_default_chat_template=False,
            temp=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            max_tokens=self.max_tokens,
            chat_template_args={},
            decode_concurrency=self.decode_concurrency,
            prompt_concurrency=self.prompt_concurrency,
            prefill_step_size=self.prefill_step_size,
            prompt_cache_size=self.prompt_cache_size,
            prompt_cache_bytes=None,
            pipeline=False,
            allowed_origins=["*"],
        )

    def _map_model_aliases(self, provider: Any) -> None:
        model_map = getattr(provider, "_model_map", None)
        if isinstance(model_map, dict):
            model_map[self.model] = self.model
            model_map[self.model_id] = self.model

    def _is_serving_model(self) -> bool:
        return self.state.phase in {
            ReadinessPhase.READY,
            ReadinessPhase.PREFILL_PENDING,
            ReadinessPhase.GENERATING,
        }

    def _monitor_model_load(self) -> None:
        assert self._generator is not None
        assert self._provider is not None
        generation_thread = getattr(self._generator, "_generation_thread", None)
        while self.state.phase not in {ReadinessPhase.FAILED, ReadinessPhase.STOPPING}:
            if getattr(self._provider, "model", None) is not None:
                if self.state.phase == ReadinessPhase.LOADING_MODEL:
                    self.state.phase = ReadinessPhase.READY
                    self.state.progress = 1.0
                    self.state.progress_current = self.state.progress_total
                    self.state.message = "Runtime is accepting generation requests."
                    _set_active_load_state(None)
                return
            if generation_thread is not None and not generation_thread.is_alive():
                self.state.phase = ReadinessPhase.FAILED
                self.state.message = "MLX-LM generation thread exited before the model finished loading."
                _set_active_load_state(None)
                return
            time.sleep(0.5)


def create_app(
    *,
    model: str,
    require_mlx: bool = False,
    runtime: TokenityDistributedRuntime | None = None,
    request_server_exit: Callable[[], None] | None = None,
) -> FastAPI:
    state = runtime.state if runtime is not None else ReadinessState(model=model)
    app = FastAPI(title="Tokenity Distributed OpenAI Server", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup() -> None:
        if runtime is None:
            await _initialize_skeleton_runtime(state, require_mlx=require_mlx)

    @app.on_event("shutdown")
    def shutdown() -> None:
        if runtime is not None:
            runtime.stop()

    @app.get("/v1/readiness")
    def readiness() -> dict[str, object]:
        return state.to_dict()

    @app.get("/")
    @app.get("/v1/tokenity/info")
    def service_info() -> dict[str, object]:
        return {
            "name": "Tokenity",
            "api": "openai-compatible",
            "base_path": "/v1",
            "model": runtime.model_id if runtime is not None else model,
            "endpoints": ["/v1/models", "/v1/chat/completions"],
            "readiness": state.to_dict(),
        }

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok" if state.phase != ReadinessPhase.FAILED else "failed",
            "phase": state.phase.value,
        }

    @app.post("/v1/tokenity/stop")
    def request_stop() -> dict[str, object]:
        if runtime is not None:
            runtime.request_stop()
        state.phase = ReadinessPhase.STOPPING
        state.message = "Stopping model runtime and releasing MLX memory."
        if request_server_exit is not None:
            # Let the response reach the Node Agent, then ask Uvicorn to run
            # its normal lifespan shutdown instead of requiring SIGTERM.
            timer = threading.Timer(0.05, request_server_exit)
            timer.daemon = True
            timer.start()
        return {"status": "stopping"}

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        if state.phase == ReadinessPhase.FAILED:
            raise HTTPException(status_code=503, detail=state.message or "runtime failed")
        if runtime is not None:
            try:
                return runtime.models_payload()
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "owned_by": "tokenity",
                    "tokenity": state.to_dict(),
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat(request: ChatCompletionRequest):
        if state.phase == ReadinessPhase.FAILED:
            raise HTTPException(status_code=503, detail=state.message or "runtime failed")
        if runtime is None:
            if request.stream:
                return StreamingResponse(_stream_skeleton(state), media_type="text/event-stream")
            return _skeleton_completion(model=request.model)
        if request.stream:
            return StreamingResponse(runtime.stream(request), media_type="text/event-stream")
        try:
            return await asyncio.to_thread(runtime.complete, request)
        except Exception as exc:
            logging.exception("Tokenity generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


async def _initialize_skeleton_runtime(state: ReadinessState, *, require_mlx: bool) -> None:
    state.phase = ReadinessPhase.DISTRIBUTED_INIT
    state.rank = 0
    state.world_size = 1
    state.backend = "skeleton"
    if require_mlx:
        state.phase = ReadinessPhase.FAILED
        state.message = "MLX runtime was required but no runtime was attached."
        return
    state.phase = ReadinessPhase.READY


async def _stream_skeleton(state: ReadinessState) -> AsyncIterator[str]:
    state.phase = ReadinessPhase.GENERATING
    yield (
        "data: "
        '{"choices":[{"index":0,"delta":{"content":"Tokenity skeleton stream online."},"finish_reason":null}]}'
        "\n\n"
    )
    yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    yield "data: [DONE]\n\n"
    state.phase = ReadinessPhase.READY


def _skeleton_completion(*, model: str) -> dict[str, object]:
    return {
        "id": f"chatcmpl-tokenity-skeleton-{int(time.time())}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Tokenity distributed OpenAI server skeleton is running; generation loop is not attached in this test process.",
                },
                "finish_reason": "stop",
            }
        ],
    }


def _message_payload(content: str, reasoning: str) -> dict[str, str]:
    payload = {"role": "assistant", "content": content}
    if reasoning:
        payload["reasoning_content"] = reasoning
    return payload


def _stream_payload(item: Any, model_id: str, finish_reason: str | None) -> dict[str, object] | None:
    if not getattr(item, "text", "") and finish_reason is None:
        return None
    delta: dict[str, object] = {}
    if item.state == "reasoning":
        delta["reasoning_content"] = item.text
    elif item.state != "tool":
        delta["content"] = item.text
    else:
        return None
    return {
        "id": f"chatcmpl-tokenity-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _stop_words(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return value


_DONE = object()


def _next_response(iterator: Iterator[Any]) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return _DONE


def serve(
    *,
    model: str,
    host: str,
    port: int,
    require_mlx: bool = False,
    trust_remote_code: bool = False,
    api_identifier: str | None = None,
    max_tokens: int = 32_768,
    prompt_cache_size: int = 4,
    prefill_step_size: int = 2_048,
    decode_concurrency: int = 1,
    prompt_concurrency: int = 1,
    native_mtp: NativeMTPConfig | dict[str, Any] | None = None,
) -> None:
    import uvicorn

    state = ReadinessState(model=model)
    runtime: TokenityDistributedRuntime | None = None
    try:
        runtime = TokenityDistributedRuntime(
            model=model,
            state=state,
            trust_remote_code=trust_remote_code,
            api_identifier=api_identifier,
            max_tokens=max_tokens,
            prompt_cache_size=prompt_cache_size,
            prefill_step_size=prefill_step_size,
            decode_concurrency=decode_concurrency,
            prompt_concurrency=prompt_concurrency,
            native_mtp=native_mtp,
        )
        runtime.start()
    except Exception as exc:  # pragma: no cover - depends on target MLX runtime
        logging.exception("Tokenity distributed runtime failed during startup")
        state.phase = ReadinessPhase.FAILED
        state.message = f"Tokenity distributed runtime failed: {exc}"
        _set_active_load_state(None)
        if require_mlx or isinstance(exc, NativeMTPStartupError):
            raise

    if runtime is not None and not runtime.is_rank0():
        def request_rank_stop(_signum: int, _frame: Any) -> None:
            if runtime.process_isolated_shutdown:
                os._exit(0)
            runtime.request_stop()

        signal.signal(signal.SIGTERM, request_rank_stop)
        signal.signal(signal.SIGINT, request_rank_stop)
        runtime.join()
        return

    server_holder: dict[str, Any] = {}

    def request_server_exit() -> None:
        if runtime is not None and runtime.process_isolated_shutdown:
            os._exit(0)
        server = server_holder.get("server")
        if server is not None:
            server.should_exit = True

    app = create_app(
        model=model,
        require_mlx=require_mlx,
        runtime=runtime,
        request_server_exit=request_server_exit,
    )
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        timeout_graceful_shutdown=5,
    )
    server = uvicorn.Server(config)
    server_holder["server"] = server
    server.run()
