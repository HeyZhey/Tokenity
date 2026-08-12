from __future__ import annotations

import statistics
import time

from tokenity.control.routing import (
    AutoRouter,
    CapabilityRegistry,
    ModelCapabilityProfile,
    ModelRuntimeState,
    RouteContext,
    RoutePolicy,
    RouteReason,
    infer_task_category,
)


def _profile(model_id: str, **overrides) -> ModelCapabilityProfile:
    values = {
        "model_id": model_id,
        "revision": f"{model_id}-revision",
        "context_length": 32_768,
        "max_output_length": 8_192,
        "languages": frozenset({"en", "zh"}),
        "task_quality": {
            "fast-chat": 0.7,
            "general": 0.8,
            "coding": 0.7,
            "reasoning": 0.7,
            "long-context": 0.6,
            "tool-use": 0.5,
        },
        "warm_ttft_p50_ms": 100,
        "warm_ttft_p95_ms": 200,
        "failure_rate": 0.01,
    }
    values.update(overrides)
    return ModelCapabilityProfile(**values)


def _runtime(profile: ModelCapabilityProfile, **overrides) -> ModelRuntimeState:
    values = {
        "model_id": profile.model_id,
        "revision": profile.revision,
        "ready": True,
        "quorum_ready": True,
        "state": "ready",
        "queue_depth": 0,
        "active_request_count": 0,
        "heartbeat_age_seconds": 0.1,
    }
    values.update(overrides)
    return ModelRuntimeState(**values)


def test_hard_constraints_filter_before_scoring():
    fast = _profile("fast", supports_tools=False, supports_json=False)
    tool = _profile(
        "tool",
        supports_tools=True,
        supports_json=True,
        supports_thinking=True,
        modalities=frozenset({"text", "image"}),
    )
    stale = _runtime(tool, heartbeat_age_seconds=60)
    context = RouteContext(
        input_tokens=4_000,
        requested_output_tokens=1_000,
        requires_tools=True,
        requires_json=True,
        requires_thinking=True,
        modality="image",
        allowed_model_ids=frozenset({"fast", "tool"}),
        default_model_id="tool",
    )

    unavailable = AutoRouter().decide(
        [fast, tool],
        [_runtime(fast), stale],
        context,
        RoutePolicy.balanced(),
    )
    assert unavailable.model_id is None
    assert unavailable.reason == RouteReason.NO_ELIGIBLE_MODEL

    decision = AutoRouter().decide(
        [fast, tool],
        [_runtime(fast), _runtime(tool)],
        context,
        RoutePolicy.balanced(),
    )
    assert decision.model_id == "tool"
    assert decision.hard_constraints_satisfied is True
    assert all(candidate.model_id != "fast" or not candidate.eligible for candidate in decision.candidates)


def test_runtime_and_chat_template_parameters_are_hard_constraints():
    generic = _profile(
        "generic",
        supports_thinking=True,
        supported_runtime_parameters=frozenset({"stream", "temperature"}),
    )
    qwen = _profile(
        "qwen",
        supports_thinking=True,
        supported_runtime_parameters=frozenset(
            {"stream", "temperature", "chat_template_kwargs"}
        ),
        supported_chat_template_parameters=frozenset({"enable_thinking"}),
    )
    context = RouteContext(
        requires_thinking=True,
        runtime_parameters=frozenset(
            {"stream", "temperature", "chat_template_kwargs"}
        ),
        chat_template_parameters=frozenset({"enable_thinking"}),
    )

    decision = AutoRouter().decide(
        [generic, qwen],
        [_runtime(generic), _runtime(qwen)],
        context,
        RoutePolicy.balanced(),
    )

    assert decision.model_id == "qwen"
    rejected = next(
        candidate for candidate in decision.candidates if candidate.model_id == "generic"
    )
    assert rejected.eligible is False
    assert "runtime_parameter_unsupported:chat_template_kwargs" in rejected.reasons
    assert "chat_template_parameter_unsupported:enable_thinking" in rejected.reasons


def test_sticky_selection_preserves_the_exact_model_revision():
    old = _profile(
        "shared",
        revision="revision-old",
        task_quality={"general": 0.99},
    )
    new = _profile(
        "shared",
        revision="revision-new",
        task_quality={"general": 0.7},
    )

    decision = AutoRouter().decide(
        [old, new],
        [_runtime(old), _runtime(new)],
        RouteContext(
            sticky_model_id="shared",
            sticky_revision="revision-new",
        ),
        RoutePolicy.quality(),
    )

    assert decision.model_id == "shared"
    assert decision.revision == "revision-new"
    assert decision.reason == RouteReason.SESSION_STICKY


def test_session_sticky_wins_until_a_hard_constraint_or_topic_change():
    first = _profile("first", task_quality={"general": 0.7, "coding": 0.5})
    better = _profile("better", task_quality={"general": 0.95, "coding": 0.95})
    router = AutoRouter()

    sticky = router.decide(
        [first, better],
        [_runtime(first), _runtime(better)],
        RouteContext(sticky_model_id="first", default_model_id="better"),
        RoutePolicy.quality(),
    )
    assert sticky.model_id == "first"
    assert sticky.reason == RouteReason.SESSION_STICKY

    changed = router.decide(
        [first, better],
        [_runtime(first), _runtime(better)],
        RouteContext(
            sticky_model_id="first",
            topic_changed=True,
            contains_code=True,
            default_model_id="better",
        ),
        RoutePolicy.quality(),
    )
    assert changed.model_id == "better"
    assert changed.reason != RouteReason.SESSION_STICKY


def test_fast_balanced_and_quality_policies_change_the_winner():
    quick = _profile(
        "quick",
        task_quality={"general": 0.72},
        warm_ttft_p50_ms=20,
        warm_ttft_p95_ms=30,
    )
    strong = _profile(
        "strong",
        task_quality={"general": 0.98},
        warm_ttft_p50_ms=900,
        warm_ttft_p95_ms=1_400,
    )
    profiles = [quick, strong]
    runtimes = [_runtime(quick), _runtime(strong)]
    context = RouteContext(category="general", default_model_id="quick")
    router = AutoRouter()

    assert router.decide(profiles, runtimes, context, RoutePolicy.fast()).model_id == "quick"
    assert router.decide(profiles, runtimes, context, RoutePolicy.balanced()).model_id == "quick"
    assert router.decide(profiles, runtimes, context, RoutePolicy.quality()).model_id == "strong"


def test_deterministic_feature_categories_never_need_raw_prompt_text():
    assert infer_task_category(RouteContext(contains_code=True)) == "coding"
    assert infer_task_category(RouteContext(has_compiler_error=True)) == "coding"
    assert infer_task_category(RouteContext(has_file_diff=True)) == "coding"
    assert infer_task_category(RouteContext(multi_step_reasoning=True)) == "reasoning"
    assert infer_task_category(RouteContext(long_document=True)) == "long-context"
    assert infer_task_category(RouteContext(requires_tools=True)) == "tool-use"
    assert infer_task_category(RouteContext(is_translation=True)) == "fast-chat"
    assert "prompt" not in RouteContext.__dataclass_fields__


def test_low_confidence_uses_configured_default_and_registry_is_revision_scoped():
    first = _profile("first", task_quality={"general": 0.80})
    second = _profile("second", task_quality={"general": 0.801})
    registry = CapabilityRegistry()
    registry.register(first)
    registry.register(second)
    registry.register(_profile("first", revision="first-revision-v2"))
    assert registry.get("first", "first-revision") == first
    assert registry.get("first", "first-revision-v2").revision == "first-revision-v2"

    decision = AutoRouter().decide(
        registry.profiles(),
        [_runtime(first), _runtime(second)],
        RouteContext(default_model_id="second"),
        RoutePolicy.balanced(),
    )
    assert decision.model_id == "second"
    assert decision.reason == RouteReason.DEFAULT_LOW_CONFIDENCE


def test_rule_router_latency_p95_is_below_ten_milliseconds():
    profiles = [
        _profile(
            f"model-{index}",
            task_quality={"general": 0.6 + index / 100, "coding": 0.5 + index / 100},
            warm_ttft_p95_ms=50 + index * 10,
        )
        for index in range(12)
    ]
    runtimes = [_runtime(profile, queue_depth=index % 3) for index, profile in enumerate(profiles)]
    router = AutoRouter()
    samples_ms = []
    for _ in range(250):
        started = time.perf_counter()
        decision = router.decide(
            profiles,
            runtimes,
            RouteContext(contains_code=True, default_model_id="model-0"),
            RoutePolicy.balanced(),
        )
        samples_ms.append((time.perf_counter() - started) * 1_000)
        assert decision.model_id is not None

    p95 = statistics.quantiles(samples_ms, n=20)[18]
    assert p95 < 10
