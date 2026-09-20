"""MLX-VLM 0.7.1 adapter for Tokenity's existing single-Mac HTTP lifecycle."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
from types import SimpleNamespace
from typing import Any

from tokenity.mlx.vlm_runtime import validate_packages
from tokenity.model_inspection import read_model_config


def _legacy_glm_weights(weights: dict[str, Any]) -> dict[str, Any]:
    """Lossless naming bridge from the initial #2030 conversion to #2127.

    Keep the already-MLX qkv convolution layout. Upstream sanitize handles
    projection fusion after the forget-gate namespace has been flattened.
    """
    result = {}
    for key, value in weights.items():
        if key.startswith("vision_model."):
            key = "vision_tower." + key[len("vision_model."):]
        key = key.replace(".self_attn.forget_gate.", ".self_attn.")
        key = key.replace(".self_attn.conv1d.weight", ".self_attn.qkv_conv.conv.weight")
        if key in result:
            raise ValueError(f"Duplicate GLM weight after legacy name conversion: {key}")
        result[key] = value
    return result


def load_model(model: str, *, trust_remote_code: bool, native_mtp_mode: str):
    validate_packages()
    if native_mtp_mode == "required":
        raise ValueError("MLX-VLM native MTP is not enabled in Tokenity; select MTP Off or Auto.")
    from mlx_vlm import load

    if read_model_config(model).get("model_type") == "glm5_next":
        from mlx_vlm.models.glm5_next import Model

        original = Model.sanitize
        try:
            Model.sanitize = lambda self, weights: original(self, _legacy_glm_weights(weights))
            return load(model, lazy=True, strict=True, trust_remote_code=trust_remote_code)
        finally:
            Model.sanitize = original
    return load(model, lazy=True, strict=True, trust_remote_code=trust_remote_code)


def validate_request(request: Any) -> None:
    # ponytail: use the existing chat protocol first; tool parsing and per-token
    # logprobs need their own VLM integration before they can be advertised.
    if request.tools or any(m.get("tool_calls") or m.get("role") == "tool" for m in request.messages):
        raise ValueError("Tool calling is not yet supported by Tokenity's MLX-VLM backend.")
    if request.tool_choice is not None or request.parallel_tool_calls is not None or request.response_format is not None:
        raise ValueError("Tool selection and structured response formats are not yet supported by Tokenity's MLX-VLM backend.")
    if request.logprobs or request.top_logprobs:
        raise ValueError("Logprobs are not yet supported by Tokenity's MLX-VLM backend.")
    for message in request.messages:
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("MLX-VLM messages must use system, user, or assistant roles.")
        content = message.get("content", "")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError("Message content must be text or a list of text/image_url parts.")
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("Message parts must be objects.")
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                continue
            if part.get("type") == "image_url":
                image_url = part.get("image_url")
                url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                if isinstance(url, str) and url.startswith("data:image/") and ";base64," in url:
                    continue
                raise ValueError("Send images as image_url.url base64 data URLs.")
            raise ValueError("MLX-VLM currently accepts text and image_url parts only.")


def _images(messages: list[dict[str, Any]]) -> list[Any]:
    from PIL import Image

    images = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_url":
                continue
            try:
                data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1], validate=True)
                with Image.open(BytesIO(data)) as image:
                    images.append(image.convert("RGB"))
            except (ValueError, OSError, binascii.Error) as exc:
                raise ValueError("Invalid base64 image in chat message.") from exc
    return images


class _OutputText:
    """Keep partial stop strings and thinking delimiters across token chunks."""

    def __init__(self, prompt: str, stops: list[str]):
        self.pending = ""
        self.state = "reasoning" if prompt.rfind("<think>") > prompt.rfind("</think>") else "content"
        self.stops = [s for s in stops if s]
        self.stopped = False

    def feed(self, text: str, *, final: bool = False):
        self.pending += text
        markers = [*self.stops, "<think>", "</think>"]
        while self.pending and not self.stopped:
            matches = [(self.pending.find(m), m) for m in markers if m in self.pending]
            if matches:
                index, marker = min(matches, key=lambda item: item[0])
                if index:
                    yield self.state, self.pending[:index]
                self.pending = self.pending[index + len(marker):]
                if marker in self.stops:
                    self.stopped = True
                    self.pending = ""
                else:
                    self.state = "reasoning" if marker == "<think>" else "content"
                continue
            keep = 0 if final else max(
                (n for marker in markers for n in range(1, len(marker)) if self.pending.endswith(marker[:n])),
                default=0,
            )
            end = len(self.pending) - keep
            if end:
                yield self.state, self.pending[:end]
                self.pending = self.pending[end:]
            break


def begin_generation(runtime: Any, request: Any):
    from .distributed_openai import _SingleGenerationContext, _request_max_tokens, _stop_words

    validate_request(request)
    if not runtime._single_generation_lock.acquire(timeout=600):
        raise TimeoutError("MLX-VLM generation queue did not become available before its deadline.")
    context = None
    try:
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        processor = runtime._single_tokenizer
        tokenizer = getattr(processor, "tokenizer", processor)
        messages = [dict(m) for m in request.messages]
        if request.role_mapping:
            for message in messages:
                message["role"] = request.role_mapping.get(message["role"], message["role"])
                if message["role"] not in {"system", "user", "assistant"}:
                    raise ValueError("Mapped MLX-VLM roles must be system, user, or assistant.")
        images = _images(messages)
        template_kwargs = {"enable_thinking": False, **(request.chat_template_kwargs or {})}
        prompt = apply_chat_template(
            processor, runtime._single_model.config, messages,
            num_images=len(images), **template_kwargs,
        )
        context = _SingleGenerationContext(tokenizer.encode(prompt, add_special_tokens=False))
        with runtime._single_contexts_lock:
            runtime._single_contexts.add(context)
        output = _OutputText(prompt, _stop_words(request.stop))

        def responses():
            import mlx.core as mx

            generated = None
            count = 0
            try:
                if context.stopped or runtime._stop_event.is_set():
                    return
                if request.seed is not None:
                    mx.random.seed(request.seed)
                generated = stream_generate(
                    runtime._single_model, processor, prompt, image=images or None,
                    max_tokens=_request_max_tokens(request, runtime.max_tokens),
                    temperature=request.temperature if request.temperature is not None else 0.0,
                    top_p=request.top_p if request.top_p is not None else 1.0,
                    top_k=request.top_k or 0, min_p=request.min_p or 0.0,
                    repetition_penalty=request.repetition_penalty or None,
                    presence_penalty=request.presence_penalty or None,
                    frequency_penalty=request.frequency_penalty or None,
                    prefill_step_size=runtime.prefill_step_size,
                )
                for item in generated:
                    if context.stopped or runtime._stop_event.is_set():
                        break
                    # The final upstream item can flush text without adding a token.
                    added = max(0, item.generation_tokens - count)
                    count = item.generation_tokens
                    context.prompt = range(item.prompt_tokens)
                    parts = list(output.feed(item.text, final=item.finish_reason is not None))
                    reason = "stop" if output.stopped else item.finish_reason
                    for i, (state, text) in enumerate(parts or [(output.state, "")]):
                        yield SimpleNamespace(
                            text=text, state=state, token=item.token,
                            token_count=added if i == 0 else 0,
                            logprob=0.0, top_tokens=(),
                            finish_reason=reason if i == max(0, len(parts) - 1) else None,
                        )
                    if output.stopped:
                        break
            finally:
                try:
                    if generated is not None:
                        generated.close()
                finally:
                    with runtime._single_contexts_lock:
                        runtime._single_contexts.discard(context)
                    runtime._single_generation_lock.release()

        return context, responses()
    except BaseException:
        if context is not None:
            with runtime._single_contexts_lock:
                runtime._single_contexts.discard(context)
        runtime._single_generation_lock.release()
        raise
