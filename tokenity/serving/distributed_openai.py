from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .readiness import ReadinessPhase, ReadinessState


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, object]]
    stream: bool = False


def create_app(*, model: str, require_mlx: bool = False) -> FastAPI:
    state = ReadinessState(model=model)
    app = FastAPI(title="Tokenity Distributed OpenAI Server", version="0.1.0")

    @app.on_event("startup")
    async def startup() -> None:
        await _initialize_runtime(state, require_mlx=require_mlx)

    @app.get("/v1/readiness")
    def readiness() -> dict[str, object]:
        return state.to_dict()

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        if state.phase == ReadinessPhase.FAILED:
            raise HTTPException(status_code=503, detail=state.message or "runtime failed")
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
        if request.stream:
            return StreamingResponse(_stream_skeleton(state, request), media_type="text/event-stream")
        return {
            "id": f"chatcmpl-tokenity-skeleton-{int(time.time())}",
            "object": "chat.completion",
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Tokenity distributed OpenAI server skeleton is running; generation loop is not implemented yet.",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    return app


async def _initialize_runtime(state: ReadinessState, *, require_mlx: bool) -> None:
    state.phase = ReadinessPhase.DISTRIBUTED_INIT
    state.rank = _env_int("RANK", _env_int("OMPI_COMM_WORLD_RANK", 0))
    state.world_size = _env_int("WORLD_SIZE", _env_int("OMPI_COMM_WORLD_SIZE", 1))
    state.backend = "mlx-distributed" if state.world_size > 1 else "single"

    try:
        import mlx.core as mx  # type: ignore

        if state.world_size > 1:
            mx.distributed.init()
    except Exception as exc:  # pragma: no cover - depends on target MLX runtime
        if require_mlx:
            state.phase = ReadinessPhase.FAILED
            state.message = f"MLX distributed init failed: {exc}"
            return
        state.message = f"MLX unavailable in skeleton mode: {exc}"

    state.phase = ReadinessPhase.LOADING_MODEL
    await asyncio.sleep(0)
    state.phase = ReadinessPhase.READY


async def _stream_skeleton(
    state: ReadinessState,
    request: ChatCompletionRequest,
) -> AsyncIterator[str]:
    state.phase = ReadinessPhase.PREFILL_PENDING
    yield ": tokenity prefill keep-alive\n\n"
    await asyncio.sleep(0)
    state.phase = ReadinessPhase.GENERATING
    yield (
        "data: "
        '{"choices":[{"index":0,"delta":{"content":"Tokenity skeleton stream online."},"finish_reason":null}]}'
        "\n\n"
    )
    yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    yield "data: [DONE]\n\n"
    state.phase = ReadinessPhase.READY


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def serve(*, model: str, host: str, port: int, require_mlx: bool = False) -> None:
    import uvicorn

    uvicorn.run(create_app(model=model, require_mlx=require_mlx), host=host, port=port)

