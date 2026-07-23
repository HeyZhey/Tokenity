from __future__ import annotations

import argparse
import asyncio
import ctypes
import gc
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from types import SimpleNamespace

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
from tokenity.model_inspection import standalone_model_issue

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
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    repetition_penalty: float | None = Field(default=None, ge=0, le=2)
    stop: str | list[str] | None = None
    seed: int | None = None
    tools: list[Any] | None = None
    role_mapping: dict[str, Any] | None = None
    chat_template_kwargs: dict[str, Any] | None = None


class _TimelineStreamingResponse(StreamingResponse):
    def __init__(self, *args: Any, state: ReadinessState, request_id: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._timeline_state = state
        self._timeline_request_id = request_id

    async def stream_response(self, send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        self._timeline_state.record_request_event(self._timeline_request_id, "headers_sent")
        async for chunk in self.body_iterator:
            if not isinstance(chunk, (bytes, memoryview)):
                chunk = chunk.encode(self.charset)
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})


@dataclass
class _GenerationResult:
    text: str
    reasoning: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    prompt_cache_tokens: int | None


class _SingleGenerationContext:
    def __init__(self, prompt: list[int]) -> None:
        self.prompt = prompt
        self.prompt_cache_count = 0
        self._stopped = threading.Event()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def stop(self) -> None:
        self._stopped.set()


class _RepetitionDetector:
    """Detect a generated suffix cycling through the same text several times."""

    def __init__(
        self,
        *,
        window_chars: int = 8_192,
        minimum_unit_chars: int = 24,
        maximum_unit_chars: int = 512,
        repetitions: int = 3,
    ) -> None:
        self.window_chars = window_chars
        self.minimum_unit_chars = minimum_unit_chars
        self.maximum_unit_chars = maximum_unit_chars
        self.repetitions = repetitions
        self._raw = ""
        self._unchecked_chars = 0

    def observe(self, text: str) -> bool:
        if not text:
            return False
        self._raw = (self._raw + text)[-self.window_chars :]
        self._unchecked_chars += len(text)
        if self._unchecked_chars < self.minimum_unit_chars:
            return False
        self._unchecked_chars = 0
        normalized = " ".join(self._raw.lower().split())
        words = normalized.split()
        maximum_words = min(64, len(words) // self.repetitions)
        for width in range(4, maximum_words + 1):
            block = words[-width:]
            if all(
                words[-(width * repetition) : -(width * (repetition - 1))] == block
                for repetition in range(2, self.repetitions + 1)
            ):
                return True
        maximum = min(self.maximum_unit_chars, len(normalized) // self.repetitions)
        for width in range(self.minimum_unit_chars, maximum + 1):
            block = normalized[-width:]
            if len(block.strip()) < self.minimum_unit_chars // 2:
                continue
            if normalized.endswith(block * self.repetitions):
                return True
        return False


@dataclass(frozen=True)
class _LoadEvalPolicy:
    name: str
    max_leaves: int
    target_bytes: int | None
    sleep_seconds: float


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
    state.updated_at = time.time()
    state.publish()


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
    policy = _load_eval_policy()
    log_interval = _positive_int_env("TOKENITY_MLX_LOAD_EVAL_LOG_INTERVAL", 100)
    chunks = _parameter_eval_chunks(flattened_parameters, policy)
    total_chunks = len(chunks)
    total_bytes = sum(_array_nbytes(value) or 0 for _, value in flattened_parameters)
    started_at = time.monotonic()
    logging.warning(
        "Tokenity sharded_load evaluating %s parameter leaves (%s bytes) in %s %s chunks "
        "(max leaves=%s, target bytes=%s, sleep=%.3fs).",
        len(flattened_parameters),
        total_bytes,
        total_chunks,
        policy.name,
        policy.max_leaves,
        policy.target_bytes,
        policy.sleep_seconds,
    )
    _report_load_progress(0, total_chunks)
    for chunk_index, chunk in enumerate(chunks, start=1):
        if _load_was_cancelled():
            raise _ModelLoadCancelled("Tokenity model loading was cancelled.")
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
        if policy.sleep_seconds:
            time.sleep(policy.sleep_seconds)
    elapsed = time.monotonic() - started_at
    gib_per_second = (total_bytes / (1024**3) / elapsed) if total_bytes and elapsed > 0 else 0.0
    logging.warning(
        "Tokenity sharded_load materialized %s parameter leaves in %.3fs (%.3f GiB/s).",
        len(flattened_parameters),
        elapsed,
        gib_per_second,
    )


def _load_eval_policy() -> _LoadEvalPolicy:
    # Pre-upgrade Agents always forward the old numeric defaults (one leaf and
    # a 50 ms sleep). Require an explicit policy marker before honoring those
    # values so deploying the new runtime immediately fixes legacy Agents.
    mode = os.environ.get("TOKENITY_MLX_LOAD_POLICY", "adaptive").strip().lower()
    if mode == "fixed":
        return _LoadEvalPolicy(
            name="fixed",
            max_leaves=_positive_int_env("TOKENITY_MLX_LOAD_EVAL_CHUNK_SIZE", 1),
            target_bytes=None,
            sleep_seconds=_nonnegative_float_env("TOKENITY_MLX_LOAD_EVAL_SLEEP_SECONDS", 0.05),
        )
    if mode not in {"", "adaptive"}:
        logging.warning("Unknown TOKENITY_MLX_LOAD_POLICY=%r; using adaptive loading.", mode)
    return _LoadEvalPolicy(
        name="adaptive",
        max_leaves=_positive_int_env("TOKENITY_MLX_LOAD_ADAPTIVE_MAX_LEAVES", 64),
        target_bytes=_positive_int_env("TOKENITY_MLX_LOAD_ADAPTIVE_TARGET_BYTES", 256 * 1024 * 1024),
        sleep_seconds=0.0,
    )


def _parameter_eval_chunks(
    flattened_parameters: list[tuple[str, Any]],
    policy: _LoadEvalPolicy,
) -> list[list[tuple[str, Any]]]:
    chunks: list[list[tuple[str, Any]]] = []
    chunk: list[tuple[str, Any]] = []
    chunk_bytes = 0
    for parameter in flattened_parameters:
        parameter_bytes = _array_nbytes(parameter[1]) or 0
        exceeds_leaf_limit = len(chunk) >= policy.max_leaves
        exceeds_byte_limit = (
            policy.target_bytes is not None
            and bool(chunk)
            and chunk_bytes + parameter_bytes > policy.target_bytes
        )
        if exceeds_leaf_limit or exceeds_byte_limit:
            chunks.append(chunk)
            chunk = []
            chunk_bytes = 0
        chunk.append(parameter)
        chunk_bytes += parameter_bytes
    if chunk:
        chunks.append(chunk)
    return chunks


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
        execution_mode: str = "distributed",
        warmup_timeout: float = 120.0,
    ) -> None:
        if execution_mode not in {"single", "distributed"}:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")
        self.model = model
        self.model_id = (api_identifier or "").strip() or Path(model).name or model
        self.state = state
        self.max_tokens = max_tokens
        self.trust_remote_code = trust_remote_code
        self.prompt_cache_size = prompt_cache_size
        self.prefill_step_size = prefill_step_size
        self.decode_concurrency = decode_concurrency
        self.prompt_concurrency = prompt_concurrency
        self.execution_mode = execution_mode
        self.warmup_timeout = warmup_timeout
        self.state.execution_mode = execution_mode
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
        self._single_model: Any = None
        self._single_tokenizer: Any = None
        self._single_generation_lock = threading.Lock()
        self._single_contexts: set[_SingleGenerationContext] = set()
        self._single_contexts_lock = threading.Lock()
        self._active_request_count = 0
        self._request_count_lock = threading.Lock()
        self._status_heartbeat_stop = threading.Event()
        self._status_heartbeat: threading.Thread | None = None

    def start(self) -> None:
        self._start_status_heartbeat()
        if issue := standalone_model_issue(self.model):
            self.state.transition(
                ReadinessPhase.FAILED,
                message=issue,
                error={"stage": "model_validation", "message": issue},
            )
            raise ValueError(issue)
        if self.execution_mode == "single":
            self._start_single()
            return

        self.state.transition(
            ReadinessPhase.DISTRIBUTED_INIT,
            message="Initializing the MLX distributed group.",
        )
        self.state.progress = 0.02
        _set_active_load_state(self.state, self._stop_event)
        try:
            import mlx.core as mx  # type: ignore

            if mx.metal.is_available():
                mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
            _delay_rank0_distributed_init()
            self._group = mx.distributed.init()
        except Exception as exc:  # pragma: no cover - depends on target MLX runtime
            self.state.transition(
                ReadinessPhase.FAILED,
                message=f"MLX distributed init failed: {exc}",
                error={"stage": "distributed_initializing", "message": str(exc)},
            )
            _set_active_load_state(None)
            raise

        self.state.rank = int(self._group.rank())
        self.state.world_size = int(self._group.size())
        self.state.backend = "mlx-distributed" if self.state.world_size > 1 else "single"

        self.native_mtp.prepare(mx, self._group)
        self.state.native_mtp = self.native_mtp.telemetry

        self.state.transition(
            ReadinessPhase.LOADING_MODEL,
            message="Loading model metadata and materializing sharded weights.",
        )
        self.state.progress = 0.1
        self._symbols = _MLXServerSymbols.load(self.native_mtp)
        args = self._server_args()
        self._provider = self._symbols.ModelProvider(args)
        self._map_model_aliases(self._provider)
        cache = self._symbols.LRUPromptCache(args.prompt_cache_size)
        self._generator = self._symbols.ResponseGenerator(self._provider, cache)
        self._load_monitor = threading.Thread(target=self._monitor_model_load, daemon=True)
        self._load_monitor.start()

    def _start_single(self) -> None:
        """Load a local model without creating any MLX distributed group."""

        self.state.rank = 0
        self.state.world_size = 1
        self.state.backend = "single"
        self.state.transition(
            ReadinessPhase.LOADING_MODEL,
            message="Loading model metadata on one Mac without distributed initialization.",
        )
        self.state.progress = 0.05
        _set_active_load_state(self.state, self._stop_event)
        try:
            import mlx.core as mx  # type: ignore
            from mlx.utils import tree_flatten  # type: ignore
            from mlx_lm import load  # type: ignore

            if mx.metal.is_available():
                mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
            self._single_model, self._single_tokenizer = load(
                self.model,
                tokenizer_config={"trust_remote_code": True if self.trust_remote_code else None},
                lazy=True,
            )
            self.state.tokenizer_identity = type(self._single_tokenizer).__name__
            self.state.message = "Materializing local model weights."
            self.state.progress = 0.1
            _eval_parameters_in_chunks(mx, tree_flatten(self._single_model.parameters()))
            self.state.transition(
                ReadinessPhase.COMPILING,
                message="Compiling and running an isolated one-token readiness warmup.",
            )
            self.state.progress = 0.96
            self._run_warmup_with_timeout(self._warmup_single)
            evidence = {
                "all_ranks_healthy": True,
                "rank_quorum": "1/1",
                "weights_materialized": True,
                "tokenizer_ready": self._single_tokenizer is not None,
                "generation_engine_ready": self._single_model is not None,
                "one_token_probe": True,
                "warmup_cache_isolated": True,
            }
            self.state.progress = 1.0
            self.state.transition(
                ReadinessPhase.READY,
                message="Runtime is accepting generation requests.",
                evidence=evidence,
            )
        except Exception as exc:
            self._single_model = None
            self._single_tokenizer = None
            self.state.transition(
                ReadinessPhase.FAILED,
                message=f"Single-node runtime failed during startup: {exc}",
                error={"stage": self.state.lifecycle_state, "message": str(exc)},
            )
            raise
        finally:
            _set_active_load_state(None)

    def _run_warmup_with_timeout(self, warmup: Callable[[], None]) -> None:
        result: queue.Queue[BaseException | None] = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                warmup()
            except BaseException as exc:  # propagate the original warmup cause
                result.put(exc)
            else:
                result.put(None)

        thread = threading.Thread(target=run, daemon=True, name="tokenity-readiness-warmup")
        thread.start()
        try:
            failure = result.get(timeout=self.warmup_timeout)
        except queue.Empty as exc:
            self._stop_event.set()
            raise TimeoutError(
                f"One-token readiness warmup exceeded {self.warmup_timeout:.1f} seconds."
            ) from exc
        if failure is not None:
            raise failure

    def _warmup_single(self) -> None:
        request = ChatCompletionRequest(
            model=self.model_id,
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=1,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
        )
        ctx, responses = self._begin_single_generation(request)
        try:
            next(responses, None)
        finally:
            ctx.stop()

    def is_rank0(self) -> bool:
        return self.state.rank == 0

    def join(self) -> None:
        if self.execution_mode == "single":
            return
        if self._generator is not None:
            try:
                self._generator.join()
            finally:
                self._release_memory()

    def request_stop(self) -> None:
        self._stop_event.set()
        self._status_heartbeat_stop.set()
        self.state.transition(
            ReadinessPhase.STOPPING,
            message="Stopping model runtime and releasing MLX memory.",
        )
        with self._single_contexts_lock:
            for context in list(self._single_contexts):
                context.stop()
        if self._generator is not None and not self.process_isolated_shutdown:
            self._generator._stop = True

    def _start_status_heartbeat(self) -> None:
        if not self.state.status_path or self._status_heartbeat is not None:
            return

        def publish() -> None:
            self.memory_metrics()
            while not self._status_heartbeat_stop.wait(2.0):
                self.state.updated_at = time.time()
                self.memory_metrics()

        self._status_heartbeat = threading.Thread(
            target=publish,
            daemon=True,
            name="tokenity-runtime-heartbeat",
        )
        self._status_heartbeat.start()

    def stop(self) -> None:
        self.request_stop()
        if self.execution_mode == "single":
            self._release_memory()
            return
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
        self._single_model = None
        self._single_tokenizer = None
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
        self._acquire_request()
        try:
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
        finally:
            self._release_request()

    async def stream(
        self,
        request: ChatCompletionRequest,
        *,
        request_id: str | None = None,
        accepted_at: float | None = None,
        keepalive_interval: float = 5.0,
    ) -> AsyncIterator[str]:
        request_id = request_id or uuid.uuid4().hex
        self.state.record_request_event(request_id, "accepted", accepted_at)
        self.state.record_request_event(request_id, "prefill_start")
        ctx = None
        finish_reason = "stop"
        tokens = 0
        repetition_detector = _RepetitionDetector()
        self._acquire_request()
        try:
            begin_task = asyncio.create_task(asyncio.to_thread(self._begin_generation, request))
            while not begin_task.done():
                done, _ = await asyncio.wait({begin_task}, timeout=keepalive_interval)
                if done:
                    break
                if self.state.last_request is not None and "first_keepalive" not in self.state.last_request:
                    self.state.record_request_event(request_id, "first_keepalive")
                yield f": keep-alive request_id={request_id}\n\n"
            ctx, responses = await begin_task
            self.state.transition(
                ReadinessPhase.GENERATING,
                message="Generating tokens.",
            )
            while True:
                next_task = asyncio.create_task(asyncio.to_thread(_next_response, responses))
                while not next_task.done():
                    done, _ = await asyncio.wait({next_task}, timeout=keepalive_interval)
                    if done:
                        break
                    if self.state.last_request is not None and "first_keepalive" not in self.state.last_request:
                        self.state.record_request_event(request_id, "first_keepalive")
                    yield f": keep-alive request_id={request_id}\n\n"
                item = await next_task
                if item is _DONE:
                    break
                if repetition_detector.observe(getattr(item, "text", "")):
                    finish_reason = "tokenity_repetition"
                    logging.warning("Tokenity stopped generation after detecting repeated output.")
                    break
                tokens += 1
                finish_reason = getattr(item, "finish_reason", None) or finish_reason
                payload = _stream_payload(item, self.model_id, finish_reason=None)
                if payload is not None:
                    if self.state.last_request is not None and "prefill_end" not in self.state.last_request:
                        self.state.record_request_event(request_id, "prefill_end")
                    if getattr(item, "text", "") and self.state.last_request is not None:
                        if "first_content_token" not in self.state.last_request:
                            self.state.record_request_event(request_id, "first_content_token")
                        self.state.record_request_event(request_id, "last_token")
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
                "usage": {
                    "prompt_tokens": len(getattr(ctx, "prompt", [])),
                    "completion_tokens": tokens,
                    "total_tokens": len(getattr(ctx, "prompt", [])) + tokens,
                    "prompt_tokens_details": {
                        "cached_tokens": getattr(ctx, "prompt_cache_count", 0) or 0,
                    },
                },
            }
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            self.state.record_request_event(request_id, "completed")
        except asyncio.CancelledError:
            self.state.record_request_event(request_id, "cancelled")
            raise
        except Exception as exc:
            self.state.record_request_event(request_id, "failed")
            if self.state.last_request is not None:
                self.state.last_request["error"] = str(exc)
            raise
        finally:
            if ctx is not None:
                ctx.stop()
            self._release_request()

    def _complete_sync(self, request: ChatCompletionRequest) -> _GenerationResult:
        ctx = None
        text = ""
        reasoning = ""
        finish_reason = "stop"
        tokens = 0
        repetition_detector = _RepetitionDetector()
        try:
            ctx, responses = self._begin_generation(request)
            self.state.phase = ReadinessPhase.GENERATING
            for item in responses:
                if repetition_detector.observe(getattr(item, "text", "")):
                    finish_reason = "tokenity_repetition"
                    logging.warning("Tokenity stopped generation after detecting repeated output.")
                    break
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

    def _begin_generation(self, request: ChatCompletionRequest) -> tuple[Any, Iterator[Any]]:
        if self.execution_mode == "single":
            return self._begin_single_generation(request)
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
        self.state.transition(
            ReadinessPhase.PREFILL_PENDING,
            message="Prefilling the request prompt.",
        )
        return self._begin_distributed_generation(request)

    def _begin_distributed_generation(self, request: ChatCompletionRequest) -> tuple[Any, Iterator[Any]]:
        assert self._symbols is not None
        assert self._generator is not None
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

    def _begin_single_generation(self, request: ChatCompletionRequest) -> tuple[Any, Iterator[Any]]:
        if self._single_model is None or self._single_tokenizer is None:
            raise RuntimeError("Tokenity single-node runtime has not started.")
        if not self.accepts_model(request.model):
            raise ValueError(f"Model is not loaded: {request.model}")
        if not self._single_generation_lock.acquire(timeout=600):
            raise TimeoutError("Single-node generation queue did not become available before its deadline.")
        try:
            messages = [dict(message) for message in request.messages]
            tokenizer = self._single_tokenizer
            if getattr(tokenizer, "has_chat_template", False):
                prompt = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **(request.chat_template_kwargs or {}),
                )
            else:
                text = "\n".join(
                    f"{message.get('role', 'user')}: {message.get('content', '')}"
                    for message in messages
                )
                prompt = tokenizer.encode(text)

            from mlx_lm import stream_generate  # type: ignore
            from mlx_lm.sample_utils import make_logits_processors, make_sampler  # type: ignore

            context = _SingleGenerationContext(list(prompt))
            with self._single_contexts_lock:
                self._single_contexts.add(context)

            sampler = make_sampler(
                temp=request.temperature if request.temperature is not None else 0.0,
                top_p=request.top_p if request.top_p is not None else 1.0,
                top_k=request.top_k if request.top_k is not None else 0,
                min_p=request.min_p if request.min_p is not None else 0.0,
            )
            logits_processors = make_logits_processors(
                repetition_penalty=(
                    request.repetition_penalty if request.repetition_penalty not in {None, 0} else None
                ),
                repetition_context_size=20,
                presence_penalty=(
                    request.presence_penalty if request.presence_penalty not in {None, 0} else None
                ),
                presence_context_size=20,
            )

            def progress(_processed: int, _total: int) -> None:
                if context.stopped or self._stop_event.is_set():
                    raise RuntimeError("Generation was cancelled during prefill.")

            generated = stream_generate(
                model=self._single_model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=request.max_tokens or self.max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prefill_step_size=self.prefill_step_size,
                prompt_progress_callback=progress,
            )

            def responses() -> Iterator[Any]:
                try:
                    for item in generated:
                        if context.stopped or self._stop_event.is_set():
                            break
                        yield SimpleNamespace(
                            text=getattr(item, "text", ""),
                            state="content",
                            finish_reason=getattr(item, "finish_reason", None),
                        )
                finally:
                    with self._single_contexts_lock:
                        self._single_contexts.discard(context)
                    self._single_generation_lock.release()

            return context, responses()
        except BaseException:
            self._single_generation_lock.release()
            raise

    def _acquire_request(self) -> None:
        with self._request_count_lock:
            self._active_request_count += 1

    def _release_request(self) -> None:
        with self._request_count_lock:
            self._active_request_count = max(0, self._active_request_count - 1)
            remaining = self._active_request_count
        if remaining == 0 and self.state.phase not in {ReadinessPhase.FAILED, ReadinessPhase.STOPPING}:
            self.state.transition(
                ReadinessPhase.READY,
                message="Runtime is accepting generation requests.",
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
                repetition_penalty=(
                    request.repetition_penalty
                    if request.repetition_penalty is not None
                    else 0.0
                ),
                repetition_context_size=20,
                presence_penalty=(
                    request.presence_penalty
                    if request.presence_penalty is not None
                    else 0.0
                ),
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
                    try:
                        tokenizer = getattr(self._provider, "tokenizer", None)
                        self.state.tokenizer_identity = (
                            type(tokenizer).__name__ if tokenizer is not None else None
                        )
                        if self.is_rank0():
                            self.state.transition(
                                ReadinessPhase.COMPILING,
                                message="Compiling and running an isolated one-token readiness warmup.",
                            )
                            self.state.progress = 0.96
                            self._run_warmup_with_timeout(self._warmup_distributed)
                        self.state.progress = 1.0
                        self.state.progress_current = self.state.progress_total
                        self.state.transition(
                            ReadinessPhase.READY,
                            message="Runtime is accepting generation requests.",
                            evidence={
                                "rank": self.state.rank,
                                "world_size": self.state.world_size,
                                "weights_materialized": True,
                                "tokenizer_ready": tokenizer is not None,
                                "generation_engine_ready": generation_thread is not None
                                and generation_thread.is_alive(),
                                "one_token_probe": self.is_rank0(),
                                "warmup_cache_isolated": self.is_rank0(),
                            },
                        )
                    except Exception as exc:
                        self.state.transition(
                            ReadinessPhase.FAILED,
                            message=f"Readiness warmup failed: {exc}",
                            error={"stage": "compiling_warming", "message": str(exc)},
                        )
                    finally:
                        _set_active_load_state(None)
                return
            if generation_thread is not None and not generation_thread.is_alive():
                message = "MLX-LM generation thread exited before the model finished loading."
                self.state.transition(
                    ReadinessPhase.FAILED,
                    message=message,
                    error={"stage": "materializing_weights", "message": message},
                )
                _set_active_load_state(None)
                return
            time.sleep(0.5)

    def _warmup_distributed(self) -> None:
        request = ChatCompletionRequest(
            model=self.model_id,
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=1,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
        )
        ctx, responses = self._begin_distributed_generation(request)
        try:
            next(responses, None)
        finally:
            ctx.stop()
            if self._generator is not None and self._symbols is not None:
                # The warmup uses the production engine but replaces its cache
                # before READY so no warmup KV/prompt entry is user-visible.
                self._generator.prompt_cache = self._symbols.LRUPromptCache(self.prompt_cache_size)

    def memory_metrics(self) -> dict[str, object]:
        sampled_at = time.time()
        resident = _process_resident_bytes(os.getpid())
        metrics: dict[str, object] = {
            "process_resident_bytes": resident,
            "process_phys_footprint_bytes": _process_phys_footprint_bytes(os.getpid()),
            "mlx_active_bytes": None,
            "mlx_peak_bytes": None,
            "mlx_cache_bytes": None,
            "model_weights_estimated_bytes": _directory_size(self.model),
            "model_resident_observed_bytes": None,
            "kv_cache_bytes": None,
            "prompt_cache_bytes": None,
            "active_request_count": self._active_request_count,
            "sampled_at": sampled_at,
            "stale": False,
        }
        try:
            import mlx.core as mx  # type: ignore

            metrics["mlx_active_bytes"] = int(mx.get_active_memory())
            metrics["mlx_peak_bytes"] = int(mx.get_peak_memory())
            metrics["mlx_cache_bytes"] = int(mx.get_cache_memory())
            metrics["model_resident_observed_bytes"] = metrics["mlx_active_bytes"]
        except Exception:
            pass
        prompt_cache = getattr(self._generator, "prompt_cache", None)
        prompt_cache_bytes = getattr(prompt_cache, "nbytes", None)
        if isinstance(prompt_cache_bytes, int):
            metrics["prompt_cache_bytes"] = prompt_cache_bytes
        self.state.memory = metrics
        self.state.publish()
        return metrics


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
        if runtime is not None:
            runtime.memory_metrics()
        return state.to_dict()

    @app.get("/v1/tokenity/metrics")
    def metrics() -> dict[str, object]:
        if runtime is None:
            return {"memory": None, "readiness": state.to_dict()}
        return {"memory": runtime.memory_metrics(), "readiness": state.to_dict()}

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
        accepted_at = time.time()
        request_id = uuid.uuid4().hex
        state.record_request_event(request_id, "accepted", accepted_at)
        if state.phase == ReadinessPhase.FAILED:
            raise HTTPException(status_code=503, detail=state.message or "runtime failed")
        if runtime is None:
            if request.stream:
                return StreamingResponse(_stream_skeleton(state), media_type="text/event-stream")
            return _skeleton_completion(model=request.model)
        if request.stream:
            return _TimelineStreamingResponse(
                runtime.stream(request, request_id=request_id, accepted_at=accepted_at),
                state=state,
                request_id=request_id,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                    "X-Tokenity-Request-ID": request_id,
                },
            )
        try:
            response = await asyncio.to_thread(runtime.complete, request)
            state.record_request_event(request_id, "completed")
            return response
        except Exception as exc:
            state.record_request_event(request_id, "failed")
            if state.last_request is not None:
                state.last_request["error"] = str(exc)
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


def _process_resident_bytes(pid: int) -> int | None:
    try:
        completed = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return int(completed.stdout.strip()) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _process_phys_footprint_bytes(pid: int) -> int | None:
    if sys.platform != "darwin":
        return None

    class RUsageInfoV0(ctypes.Structure):
        _fields_ = [
            ("uuid", ctypes.c_uint8 * 16),
            ("user_time", ctypes.c_uint64),
            ("system_time", ctypes.c_uint64),
            ("pkg_idle_wakeups", ctypes.c_uint64),
            ("interrupt_wakeups", ctypes.c_uint64),
            ("pageins", ctypes.c_uint64),
            ("wired_size", ctypes.c_uint64),
            ("resident_size", ctypes.c_uint64),
            ("phys_footprint", ctypes.c_uint64),
            ("proc_start_abstime", ctypes.c_uint64),
            ("proc_exit_abstime", ctypes.c_uint64),
        ]

    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
        proc_pid_rusage = libproc.proc_pid_rusage
        proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        proc_pid_rusage.restype = ctypes.c_int
        usage = RUsageInfoV0()
        if proc_pid_rusage(pid, 0, ctypes.byref(usage)) != 0:
            return None
        return int(usage.phys_footprint)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _directory_size(path: str) -> int | None:
    root = Path(path)
    if not root.is_dir():
        return None
    total = 0
    try:
        for item in root.iterdir():
            if item.is_file() and item.suffix in {".safetensors", ".gguf"}:
                total += item.stat().st_size
    except OSError:
        return None
    return total or None


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
    execution_mode: str = "distributed",
    warmup_timeout: float = 120.0,
) -> None:
    import uvicorn

    state = ReadinessState(
        model=model,
        instance_id=os.environ.get("TOKENITY_INSTANCE_ID"),
        operation_id=os.environ.get("TOKENITY_OPERATION_ID"),
        execution_mode=execution_mode,
        cluster_id=os.environ.get("TOKENITY_CLUSTER_ID"),
        connection_mode=os.environ.get("TOKENITY_CONNECTION_MODE"),
        model_revision=os.environ.get("TOKENITY_MODEL_REVISION"),
        status_path=os.environ.get("TOKENITY_STATUS_PATH"),
    )
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
            execution_mode=execution_mode,
            warmup_timeout=warmup_timeout,
        )
        runtime.start()
    except Exception as exc:  # pragma: no cover - depends on target MLX runtime
        logging.exception("Tokenity distributed runtime failed during startup")
        state.transition(
            ReadinessPhase.FAILED,
            message=f"Tokenity runtime failed: {exc}",
            error={"stage": state.lifecycle_state, "message": str(exc)},
        )
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
