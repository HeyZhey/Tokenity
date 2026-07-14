# SPDX-License-Identifier: Apache-2.0
"""Singleton depth-one native MTP dispatch for mlx-lm continuous batching.

Adapted from mlx-lm PR #990 and oMLX's Apache-2.0 batch generator patch.
Tokenity deliberately keeps the narrow MVP: lazy singleton activation,
one draft token, no row-wise path, no adaptive depth, and no draft model.
"""

from __future__ import annotations

import logging
import math
import importlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from . import cache as cache_rollback
from .runtime import PatchTransaction


logger = logging.getLogger(__name__)


@dataclass
class _MTPState:
    uid: int
    telemetry: dict[str, object]
    queue: deque[tuple[int, Any, str, bool]] = field(default_factory=deque)
    mtp_cache: list[Any] = field(default_factory=list)
    next_main: Any = None
    draft_token: Any = None
    draft_id: int = -1
    draft_logprobs: Any = None
    draft_accept_logprobs: Any = None
    last_uncached_token: int | None = None


def install(transaction: PatchTransaction, telemetry: dict[str, object]) -> None:
    import mlx_lm.server as server  # type: ignore

    # ``mlx_lm.__init__`` exports a function named ``generate``; importing
    # through importlib is required to obtain the actual submodule.
    generate = importlib.import_module("mlx_lm.generate")

    generation_batch = generate.GenerationBatch
    batch_generator = generate.BatchGenerator
    if getattr(generation_batch, "_tokenity_native_mtp_generation_patch", False):
        raise RuntimeError("Tokenity Native MTP generation patch is already installed")

    original_make_sampler = server._make_sampler

    def make_sampler(args: Any, tokenizer: Any) -> Any:
        sampler = original_make_sampler(args, tokenizer)
        sampling = args.sampling
        sampler.temp = float(sampling.temperature)
        sampler.top_p = float(sampling.top_p)
        sampler.top_k = int(sampling.top_k)
        sampler.min_p = float(sampling.min_p)
        sampler.min_tokens_to_keep = 1
        sampler.xtc_probability = float(sampling.xtc_probability)
        sampler.xtc_threshold = float(sampling.xtc_threshold)
        sampler._tokenity_native_mtp_sampler = True
        return sampler

    transaction.set(server, "_make_sampler", make_sampler)

    original_next = generation_batch.next
    original_extend = generation_batch.extend
    original_filter = generation_batch.filter

    def next_token(self: Any) -> list[Any]:
        state = getattr(self, "_tokenity_native_mtp_state", None)
        if state is not None and _state_is_valid(self, state):
            return _mtp_next(self, state)
        if state is not None:
            delattr(self, "_tokenity_native_mtp_state")
        if _eligible(self):
            _post_init(self, telemetry)
            state = getattr(self, "_tokenity_native_mtp_state", None)
            if state is not None:
                return _mtp_next(self, state)
        return original_next(self)

    def extend(self: Any, batch: Any) -> Any:
        if getattr(self, "_tokenity_native_mtp_state", None) is not None:
            raise RuntimeError("Native MTP invariant: an active singleton cannot accept a late join")
        if getattr(batch, "_tokenity_native_mtp_state", None) is not None:
            raise RuntimeError("Native MTP invariant: an active singleton cannot be a merge donor")
        return original_extend(self, batch)

    def filter_batch(self: Any, keep: list[int]) -> Any:
        state = getattr(self, "_tokenity_native_mtp_state", None)
        if state is not None and (not keep or keep != [0]):
            _commit_last_uncached(self, state)
            delattr(self, "_tokenity_native_mtp_state")
        return original_filter(self, keep)

    transaction.set(generation_batch, "next", next_token)
    transaction.set(generation_batch, "extend", extend)
    transaction.set(generation_batch, "filter", filter_batch)
    transaction.set(generation_batch, "_tokenity_native_mtp_generation_patch", True)

    original_batch_next = batch_generator._next

    def batch_next(self: Any) -> Any:
        generation = getattr(self, "_generation_batch", None)
        if generation is not None:
            generation._tokenity_native_mtp_activation_safe = _activation_is_safe(self)
        if getattr(generation, "_tokenity_native_mtp_state", None) is None:
            return original_batch_next(self)

        # The existing request owns the only decode slot. Prevent mlx-lm from
        # promoting pending prompt work into its active GenerationBatch.
        old_size = self.completion_batch_size
        self.completion_batch_size = 0
        try:
            return original_batch_next(self)
        finally:
            self.completion_batch_size = old_size

    transaction.set(batch_generator, "_next", batch_next)
    transaction.set(batch_generator, "_tokenity_native_mtp_generation_patch", True)


def _activation_is_safe(generator: Any) -> bool:
    return bool(
        len(getattr(generator, "_generation_batch", ())) == 1
        and len(getattr(generator, "_prompt_batch", ())) == 0
        and not getattr(generator, "_currently_processing", ())
        and not getattr(generator, "_unprocessed_sequences", ())
    )


def _eligible(batch: Any) -> bool:
    if len(getattr(batch, "uids", ())) != 1:
        return False
    if not getattr(batch, "_tokenity_native_mtp_activation_safe", False):
        return False
    model = getattr(batch, "model", None)
    inner = getattr(model, "language_model", model)
    return bool(
        model is not None
        and getattr(inner, "_tokenity_native_mtp_decode_enabled", False)
        and getattr(inner, "mtp", None) is not None
        and callable(getattr(model, "mtp_forward", None))
        and callable(getattr(model, "make_mtp_cache", None))
    )


def _state_is_valid(batch: Any, state: _MTPState) -> bool:
    return len(getattr(batch, "uids", ())) == 1 and batch.uids[0] == state.uid


def _sampler(batch: Any) -> Any:
    if batch.samplers and batch.samplers[0] is not None:
        return batch.samplers[0]
    return batch.fallback_sampler


def _processors(batch: Any) -> list[Any] | None:
    if batch.logits_processors and batch.logits_processors[0]:
        return batch.logits_processors[0]
    return None


def _apply_processors(processors: list[Any] | None, previous: Any, logits: Any) -> Any:
    for processor in processors or ():
        logits = processor(previous, logits)
    return logits


def _logprobs(logits: Any) -> Any:
    import mlx.core as mx  # type: ignore

    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def filtered_logprobs(sampler: Any, logprobs: Any) -> Any:
    """Recreate mlx-lm's filter/temperature distribution for p/q math."""

    import mlx.core as mx  # type: ignore
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p  # type: ignore

    temperature = float(getattr(sampler, "temp", 0.0) or 0.0)
    if temperature == 0.0:
        return logprobs
    if float(getattr(sampler, "xtc_probability", 0.0) or 0.0) != 0.0:
        raise RuntimeError("Native MTP MVP does not support XTC sampling")

    result = logprobs
    top_p = float(getattr(sampler, "top_p", 0.0) or 0.0)
    if 0.0 < top_p < 1.0:
        result = apply_top_p(result, top_p)
    min_p = float(getattr(sampler, "min_p", 0.0) or 0.0)
    if min_p:
        result = apply_min_p(
            result,
            min_p,
            int(getattr(sampler, "min_tokens_to_keep", 1) or 1),
        )
    top_k = int(getattr(sampler, "top_k", 0) or 0)
    if top_k > 0:
        result = apply_top_k(result, top_k)
    result = result / temperature
    return result - mx.logsumexp(result, axis=-1, keepdims=True)


def _post_init(batch: Any, telemetry: dict[str, object]) -> None:
    import mlx.core as mx  # type: ignore

    if batch._next_tokens is None:
        return
    sampler = _sampler(batch)
    processors = _processors(batch)
    main_token = _uint32(batch._next_tokens)
    main_logprobs = batch._next_logprobs[0]
    previous = (
        batch._token_context[0].update_and_fetch(main_token)
        if processors is not None
        else None
    )

    logits, hidden = batch.model(
        main_token[:, None],
        cache=batch.prompt_cache,
        return_hidden=True,
    )
    next_logits = _apply_processors(processors, previous, logits[:, -1, :])
    next_logprobs = _logprobs(next_logits)
    next_main = _uint32(sampler(next_logprobs))

    mtp_cache = batch.model.make_mtp_cache()
    head_logits = batch.model.mtp_forward(
        hidden[:, -1:, :],
        next_main.reshape(1, 1),
        mtp_cache,
    )[:, -1, :]
    if processors is not None:
        head_previous = mx.concatenate([previous, next_main])
        head_logits = _apply_processors(processors, head_previous, head_logits)
    draft_logprobs = _logprobs(head_logits)
    draft_token = _uint32(sampler(draft_logprobs))
    mx.eval(main_token, next_main, draft_token)

    state = _MTPState(uid=batch.uids[0], telemetry=telemetry)
    state.mtp_cache = mtp_cache
    state.next_main = next_main
    state.draft_token = draft_token
    state.draft_id = int(draft_token.item())
    state.draft_logprobs = draft_logprobs.squeeze(0)
    state.draft_accept_logprobs = filtered_logprobs(sampler, draft_logprobs).squeeze(0)
    state.queue.append((int(main_token.item()), main_logprobs, "init", True))
    state.queue.append((int(next_main.item()), next_logprobs.squeeze(0), "init", False))
    batch._tokenity_native_mtp_state = state


def _mtp_next(batch: Any, state: _MTPState) -> list[Any]:
    if not state.queue:
        _verify_cycle(batch, state)
    if not state.queue:
        raise RuntimeError("Native MTP verify cycle produced no output token")
    token, logprobs, source, cached = state.queue.popleft()
    del source
    state.last_uncached_token = None if cached else token
    return _emit(batch, state, token, logprobs, cached)


def _verify_cycle(batch: Any, state: _MTPState) -> None:
    import mlx.core as mx  # type: ignore

    sampler = _sampler(batch)
    processors = _processors(batch)
    inputs = mx.concatenate([state.next_main, state.draft_token])
    previous_main = previous_draft = None
    if processors is not None:
        previous_main = batch._token_context[0].update_and_fetch(state.next_main)
        previous_draft = batch._token_context[0].update_and_fetch(state.draft_token)

    # The previously emitted uncached token is the confirmed first row here.
    state.last_uncached_token = None
    cache_rollback.set_undo_armed(True)
    try:
        logits, hidden = batch.model(
            inputs[None, :],
            cache=batch.prompt_cache,
            return_hidden=True,
            n_confirmed=1,
        )
    finally:
        cache_rollback.set_undo_armed(False)

    verify_logits = _apply_processors(processors, previous_main, logits[:, 0, :])
    bonus_logits = _apply_processors(processors, previous_draft, logits[:, 1, :])
    verify_logprobs = _logprobs(verify_logits)
    bonus_logprobs = _logprobs(bonus_logits)
    bonus_token = _uint32(sampler(bonus_logprobs))
    target_accept = filtered_logprobs(sampler, verify_logprobs)
    draft_accept = state.draft_accept_logprobs

    greedy = float(getattr(sampler, "temp", 0.0) or 0.0) == 0.0
    if greedy:
        target_id = int(mx.argmax(verify_logprobs, axis=-1).item())
        accepted = target_id == state.draft_id
    else:
        probability = acceptance_probability(
            target_accept,
            draft_accept,
            state.draft_id,
        )
        accepted = probability >= 1 or float(mx.random.uniform(shape=()).item()) < probability

    _record_proposal(state.telemetry, accepted)
    hidden_confirmed = hidden[:, 0:1, :]
    hidden_draft = hidden[:, 1:2, :]
    if accepted:
        cache_rollback.clear_rollback(batch.prompt_cache)
        next_draft, next_draft_lp, next_draft_accept = _step_head(
            batch,
            state,
            hidden_draft,
            bonus_token,
            previous_draft,
        )
        state.queue.append((state.draft_id, state.draft_logprobs, "draft", True))
        state.queue.append((int(bonus_token.item()), bonus_logprobs.squeeze(0), "bonus", False))
        state.next_main = bonus_token
    else:
        if not cache_rollback.restore_after_rejection(batch.prompt_cache):
            raise RuntimeError("Native MTP could not roll back every hybrid cache layer")
        if processors is not None:
            buffer = batch._token_context[0]
            buffer._size = max(0, buffer._size - 1)
        if greedy:
            emit_id = int(mx.argmax(verify_logprobs, axis=-1).item())
        else:
            emit_id = _residual_sample(target_accept, draft_accept)
        emit_token = mx.array([emit_id], dtype=mx.uint32)
        next_draft, next_draft_lp, next_draft_accept = _step_head(
            batch,
            state,
            hidden_confirmed,
            emit_token,
            previous_main,
        )
        state.queue.append((emit_id, verify_logprobs.squeeze(0), "verify", False))
        state.next_main = emit_token

    state.draft_token = next_draft
    state.draft_id = int(next_draft.item())
    state.draft_logprobs = next_draft_lp
    state.draft_accept_logprobs = next_draft_accept


def _step_head(
    batch: Any,
    state: _MTPState,
    hidden: Any,
    next_main: Any,
    previous: Any,
) -> tuple[Any, Any, Any]:
    import mlx.core as mx  # type: ignore

    sampler = _sampler(batch)
    processors = _processors(batch)
    logits = batch.model.mtp_forward(
        hidden,
        next_main.reshape(1, 1),
        state.mtp_cache,
    )[:, -1, :]
    if processors is not None:
        logits = _apply_processors(
            processors,
            mx.concatenate([previous, next_main]),
            logits,
        )
    logprobs = _logprobs(logits)
    token = _uint32(sampler(logprobs))
    accept_logprobs = filtered_logprobs(sampler, logprobs)
    mx.eval(token)
    return token, logprobs.squeeze(0), accept_logprobs.squeeze(0)


def _residual_sample(target_logprobs: Any, draft_logprobs: Any) -> int:
    import mlx.core as mx  # type: ignore

    distribution = residual_probabilities(target_logprobs, draft_logprobs)
    return int(mx.random.categorical(mx.log(distribution).reshape(1, -1)).item())


def acceptance_probability(target_logprobs: Any, draft_logprobs: Any, token: int) -> float:
    log_ratio = float(
        target_logprobs.reshape(-1)[token].item()
        - draft_logprobs.reshape(-1)[token].item()
    )
    return min(1.0, math.exp(log_ratio))


def residual_probabilities(target_logprobs: Any, draft_logprobs: Any) -> Any:
    """Return normalized ``max(p-q, 0)`` with the PR #990 fallback."""

    import mlx.core as mx  # type: ignore

    target = mx.exp(target_logprobs.reshape(-1))
    draft = mx.exp(draft_logprobs.reshape(-1))
    residual = mx.maximum(target - draft, 0.0)
    mass = residual.sum(keepdims=True)
    distribution = mx.where(mass > 0, residual, target)
    return distribution / distribution.sum(keepdims=True)


def _emit(
    batch: Any,
    state: _MTPState,
    token: int,
    logprobs: Any,
    cached: bool,
) -> list[Any]:
    response_type = type(batch).Response
    batch.tokens[0].append(token)
    batch._num_tokens[0] += 1
    finish_reason = "length" if batch._num_tokens[0] >= batch.max_tokens[0] else None
    next_state, match, current_state = batch.state_machines[0].match(
        batch._matcher_states[0],
        token,
    )
    batch._matcher_states[0] = next_state
    if match is not None and current_state is None:
        finish_reason = "stop"

    if finish_reason is None:
        return [
            response_type(
                uid=batch.uids[0],
                token=token,
                logprobs=logprobs,
                finish_reason=None,
                current_state=current_state,
                match_sequence=match,
                prompt_cache=None,
                all_tokens=None,
            )
        ]

    if not cached:
        _commit_last_uncached(batch, state)
    response = response_type(
        uid=batch.uids[0],
        token=token,
        logprobs=logprobs,
        finish_reason=finish_reason,
        current_state=current_state,
        match_sequence=match,
        prompt_cache=batch.extract_cache(0),
        all_tokens=batch.tokens[0],
    )
    delattr(batch, "_tokenity_native_mtp_state")
    batch.filter([])
    return [response]


def _commit_last_uncached(batch: Any, state: _MTPState) -> None:
    token = state.last_uncached_token
    if token is None or not batch.prompt_cache:
        return
    import mlx.core as mx  # type: ignore

    batch.model(
        mx.array([[token]], dtype=mx.uint32),
        cache=batch.prompt_cache,
    )
    state.last_uncached_token = None


def _record_proposal(telemetry: dict[str, object], accepted: bool) -> None:
    proposed = int(telemetry.get("proposed_tokens", 0)) + 1
    accepted_count = int(telemetry.get("accepted_tokens", 0)) + int(accepted)
    telemetry["proposed_tokens"] = proposed
    telemetry["accepted_tokens"] = accepted_count
    telemetry["acceptance_rate"] = accepted_count / proposed


def _uint32(value: Any) -> Any:
    import mlx.core as mx  # type: ignore

    return value if value.dtype == mx.uint32 else value.astype(mx.uint32)
