from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


ROUTE_CATEGORIES = (
    "fast-chat",
    "general",
    "coding",
    "reasoning",
    "long-context",
    "tool-use",
)


class RouteReason(str, Enum):
    EXPLICIT_MODEL = "explicit_model"
    SESSION_STICKY = "session_sticky"
    DETERMINISTIC_RULE = "deterministic_rule"
    SCORED_BEST = "scored_best"
    DEFAULT_LOW_CONFIDENCE = "default_low_confidence"
    NO_ELIGIBLE_MODEL = "no_eligible_model"


@dataclass(frozen=True)
class RoutePolicy:
    name: str
    quality_weight: float
    latency_weight: float
    queue_weight: float
    failure_weight: float
    affinity_weight: float
    priority_weight: float
    low_confidence_margin: float = 0.04

    @classmethod
    def fast(cls) -> "RoutePolicy":
        return cls("fast", 1.0, 0.0015, 0.22, 1.5, 0.15, 0.04)

    @classmethod
    def balanced(cls) -> "RoutePolicy":
        return cls("balanced", 1.5, 0.0010, 0.16, 2.0, 0.22, 0.06)

    @classmethod
    def quality(cls) -> "RoutePolicy":
        return cls("quality", 3.0, 0.00015, 0.06, 2.5, 0.25, 0.08)

    @classmethod
    def from_name(cls, value: str | None) -> "RoutePolicy":
        normalized = (value or "balanced").strip().lower()
        if normalized == "fast":
            return cls.fast()
        if normalized == "quality":
            return cls.quality()
        return cls.balanced()


@dataclass(frozen=True)
class ModelCapabilityProfile:
    model_id: str
    revision: str
    tokenizer: str | None = None
    context_length: int = 32_768
    max_output_length: int = 8_192
    languages: frozenset[str] = field(default_factory=lambda: frozenset({"en", "zh"}))
    task_tags: frozenset[str] = field(default_factory=lambda: frozenset({"general"}))
    task_quality: Mapping[str, float] = field(default_factory=dict)
    supports_tools: bool = False
    supports_json: bool = False
    supports_thinking: bool = False
    modalities: frozenset[str] = field(default_factory=lambda: frozenset({"text"}))
    allow_auto: bool = True
    user_priority: float = 0.0
    privacy_tier: int = 0
    sampling_defaults: Mapping[str, Any] = field(default_factory=dict)
    chat_template_defaults: Mapping[str, Any] = field(default_factory=dict)
    supported_runtime_parameters: frozenset[str] | None = None
    supported_chat_template_parameters: frozenset[str] = field(default_factory=frozenset)
    warm_ttft_p50_ms: float | None = None
    warm_ttft_p95_ms: float | None = None
    prefill_tokens_per_second: float | None = None
    decode_tokens_per_second: float | None = None
    failure_rate: float = 0.0
    quality_scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.revision.strip():
            raise ValueError("model_id and revision are required")
        if self.context_length <= 0 or self.max_output_length <= 0:
            raise ValueError("context and output lengths must be positive")
        if not 0 <= self.failure_rate <= 1:
            raise ValueError("failure_rate must be between zero and one")

    def quality_for(self, category: str) -> float:
        if category in self.quality_scores:
            return float(self.quality_scores[category])
        if category in self.task_quality:
            return float(self.task_quality[category])
        if category in self.task_tags:
            return 0.82
        if "general" in self.task_tags:
            return 0.68
        return 0.5

    @property
    def predicted_ttft_ms(self) -> float:
        if self.warm_ttft_p95_ms is not None:
            return max(0.0, self.warm_ttft_p95_ms)
        if self.warm_ttft_p50_ms is not None:
            return max(0.0, self.warm_ttft_p50_ms * 1.5)
        return 750.0


@dataclass(frozen=True)
class ModelRuntimeState:
    model_id: str
    revision: str
    ready: bool
    quorum_ready: bool
    state: str
    queue_depth: int = 0
    active_request_count: int = 0
    heartbeat_age_seconds: float = 0.0
    actual_memory_bytes: int | None = None
    recent_failure_rate: float = 0.0
    predicted_queue_wait_ms: float = 0.0

    @property
    def routable(self) -> bool:
        return (
            self.ready
            and self.quorum_ready
            and self.state.lower() in {"ready", "busy"}
            and self.heartbeat_age_seconds <= 15
        )


@dataclass(frozen=True)
class RouteContext:
    category: str | None = None
    input_tokens: int = 0
    requested_output_tokens: int = 0
    language: str | None = None
    requires_tools: bool = False
    requires_json: bool = False
    requires_thinking: bool = False
    modality: str = "text"
    allowed_model_ids: frozenset[str] = field(default_factory=frozenset)
    minimum_privacy_tier: int = 0
    runtime_parameters: frozenset[str] = field(default_factory=frozenset)
    chat_template_parameters: frozenset[str] = field(default_factory=frozenset)
    session_id: str | None = None
    sticky_model_id: str | None = None
    sticky_revision: str | None = None
    explicit_model_id: str | None = None
    explicit_revision: str | None = None
    default_model_id: str | None = None
    topic_changed: bool = False
    contains_code: bool = False
    has_compiler_error: bool = False
    has_file_diff: bool = False
    multi_step_reasoning: bool = False
    long_document: bool = False
    is_translation: bool = False
    is_rewrite: bool = False

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.requested_output_tokens < 0:
            raise ValueError("token counts must be non-negative")


@dataclass(frozen=True)
class CandidateScore:
    model_id: str
    revision: str
    eligible: bool
    score: float | None
    quality: float
    predicted_ttft_ms: float
    queue_penalty: float
    failure_risk: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteDecision:
    model_id: str | None
    revision: str | None
    category: str
    reason: RouteReason
    confidence: float
    routing_latency_ms: float
    candidates: tuple[CandidateScore, ...]
    hard_constraints_satisfied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "category": self.category,
            "reason": self.reason.value,
            "confidence": self.confidence,
            "routing_latency_ms": self.routing_latency_ms,
            "hard_constraints_satisfied": self.hard_constraints_satisfied,
            "candidates": [
                {
                    "model_id": candidate.model_id,
                    "revision": candidate.revision,
                    "eligible": candidate.eligible,
                    "score": candidate.score,
                    "quality": candidate.quality,
                    "predicted_ttft_ms": candidate.predicted_ttft_ms,
                    "queue_penalty": candidate.queue_penalty,
                    "failure_risk": candidate.failure_risk,
                    "reasons": list(candidate.reasons),
                }
                for candidate in self.candidates
            ],
        }


@dataclass
class RouterMetrics:
    decisions: int = 0
    failures: int = 0
    sticky_hits: int = 0
    fallback_count: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    def record(self, decision: RouteDecision) -> None:
        self.decisions += 1
        self.total_latency_ms += decision.routing_latency_ms
        self.max_latency_ms = max(self.max_latency_ms, decision.routing_latency_ms)
        if decision.model_id is None:
            self.failures += 1
        if decision.reason == RouteReason.SESSION_STICKY:
            self.sticky_hits += 1

    def snapshot(self) -> dict[str, float | int]:
        return {
            "decisions": self.decisions,
            "failures": self.failures,
            "sticky_hits": self.sticky_hits,
            "fallback_count": self.fallback_count,
            "mean_latency_ms": (
                self.total_latency_ms / self.decisions if self.decisions else 0.0
            ),
            "max_latency_ms": self.max_latency_ms,
        }


class CapabilityRegistry:
    def __init__(self) -> None:
        self._profiles: dict[tuple[str, str], ModelCapabilityProfile] = {}

    def register(self, profile: ModelCapabilityProfile) -> None:
        self._profiles[(profile.model_id, profile.revision)] = profile

    def get(self, model_id: str, revision: str) -> ModelCapabilityProfile | None:
        return self._profiles.get((model_id, revision))

    def profiles(self) -> list[ModelCapabilityProfile]:
        return [
            self._profiles[key]
            for key in sorted(self._profiles, key=lambda item: (item[0].lower(), item[1]))
        ]

    def remove(self, model_id: str, revision: str) -> None:
        self._profiles.pop((model_id, revision), None)


def infer_task_category(context: RouteContext) -> str:
    if context.category in ROUTE_CATEGORIES:
        return str(context.category)
    if context.requires_tools:
        return "tool-use"
    if context.long_document:
        return "long-context"
    if context.contains_code or context.has_compiler_error or context.has_file_diff:
        return "coding"
    if context.multi_step_reasoning:
        return "reasoning"
    if context.is_translation or context.is_rewrite:
        return "fast-chat"
    return "general"


class AutoRouter:
    def __init__(self, metrics: RouterMetrics | None = None) -> None:
        self.metrics = metrics or RouterMetrics()

    def decide(
        self,
        profiles: Iterable[ModelCapabilityProfile],
        runtimes: Iterable[ModelRuntimeState],
        context: RouteContext,
        policy: RoutePolicy,
    ) -> RouteDecision:
        started = time.perf_counter()
        category = infer_task_category(context)
        runtime_by_key = {
            (runtime.model_id, runtime.revision): runtime for runtime in runtimes
        }
        scored: list[CandidateScore] = []
        eligible_profiles: list[
            tuple[ModelCapabilityProfile, ModelRuntimeState | None, float]
        ] = []

        for profile in profiles:
            runtime = runtime_by_key.get((profile.model_id, profile.revision))
            reasons = self._hard_constraint_reasons(profile, runtime, context)
            quality = profile.quality_for(category)
            predicted_ttft = profile.predicted_ttft_ms
            queue_penalty = 0.0
            failure_risk = profile.failure_rate
            if runtime is not None:
                queue_penalty = (
                    max(0, runtime.queue_depth) * 0.75
                    + max(0, runtime.active_request_count) * 0.35
                    + max(0.0, runtime.predicted_queue_wait_ms) / 1_000
                )
                failure_risk = max(failure_risk, runtime.recent_failure_rate)
            if reasons:
                scored.append(
                    CandidateScore(
                        profile.model_id,
                        profile.revision,
                        False,
                        None,
                        quality,
                        predicted_ttft,
                        queue_penalty,
                        failure_risk,
                        tuple(reasons),
                    )
                )
                continue
            score = (
                policy.quality_weight * quality
                - policy.latency_weight * predicted_ttft
                - policy.queue_weight * queue_penalty
                - policy.failure_weight * failure_risk
                + policy.priority_weight * profile.user_priority
            )
            if context.sticky_model_id == profile.model_id:
                score += policy.affinity_weight
            eligible_profiles.append((profile, runtime, score))
            scored.append(
                CandidateScore(
                    profile.model_id,
                    profile.revision,
                    True,
                    score,
                    quality,
                    predicted_ttft,
                    queue_penalty,
                    failure_risk,
                    (),
                )
            )

        if not eligible_profiles:
            return self._finish(
                started,
                RouteDecision(
                    None,
                    None,
                    category,
                    RouteReason.NO_ELIGIBLE_MODEL,
                    0.0,
                    0.0,
                    tuple(scored),
                    False,
                ),
            )

        eligible_by_key = {
            (profile.model_id, profile.revision): (profile, score)
            for profile, _, score in eligible_profiles
        }

        def best_for_model(model_id: str | None) -> ModelCapabilityProfile | None:
            if not model_id:
                return None
            matches = [
                (profile, score)
                for profile, _, score in eligible_profiles
                if profile.model_id == model_id
            ]
            if not matches:
                return None
            return min(
                matches,
                key=lambda item: (-item[1], item[0].revision),
            )[0]

        selected: ModelCapabilityProfile | None = None
        if context.explicit_model_id and context.explicit_revision:
            explicit = eligible_by_key.get(
                (context.explicit_model_id, context.explicit_revision)
            )
            selected = explicit[0] if explicit is not None else None
        elif context.explicit_model_id:
            selected = best_for_model(context.explicit_model_id)
        if selected is not None:
            return self._selected(
                started, selected, category, RouteReason.EXPLICIT_MODEL, 1.0, scored
            )

        selected = None
        if not context.topic_changed and context.sticky_model_id:
            if context.sticky_revision:
                sticky = eligible_by_key.get(
                    (context.sticky_model_id, context.sticky_revision)
                )
                selected = sticky[0] if sticky is not None else None
            else:
                selected = best_for_model(context.sticky_model_id)
        if selected is not None:
            return self._selected(
                started, selected, category, RouteReason.SESSION_STICKY, 0.98, scored
            )

        eligible_profiles.sort(key=lambda item: (-item[2], item[0].model_id, item[0].revision))
        selected, _, top_score = eligible_profiles[0]
        second_score = eligible_profiles[1][2] if len(eligible_profiles) > 1 else None
        margin = top_score - second_score if second_score is not None else 1.0
        confidence = max(0.0, min(1.0, 0.5 + margin))
        reason = RouteReason.DETERMINISTIC_RULE if category != "general" else RouteReason.SCORED_BEST
        if (
            second_score is not None
            and margin < policy.low_confidence_margin
        ):
            default = best_for_model(context.default_model_id)
            if default is not None:
                selected = default
                reason = RouteReason.DEFAULT_LOW_CONFIDENCE
                confidence = 0.5
        return self._selected(started, selected, category, reason, confidence, scored)

    def hard_constraint_reasons(
        self,
        profile: ModelCapabilityProfile,
        runtime: ModelRuntimeState | None,
        context: RouteContext,
    ) -> tuple[str, ...]:
        return tuple(self._hard_constraint_reasons(profile, runtime, context))

    def _hard_constraint_reasons(
        self,
        profile: ModelCapabilityProfile,
        runtime: ModelRuntimeState | None,
        context: RouteContext,
    ) -> list[str]:
        reasons: list[str] = []
        if not profile.allow_auto and profile.model_id != context.explicit_model_id:
            reasons.append("auto_disabled")
        if runtime is None or not runtime.routable:
            reasons.append("runtime_not_ready")
        if context.allowed_model_ids and profile.model_id not in context.allowed_model_ids:
            reasons.append("not_in_allowed_models")
        if context.input_tokens + context.requested_output_tokens > profile.context_length:
            reasons.append("context_capacity")
        if context.requested_output_tokens > profile.max_output_length:
            reasons.append("output_capacity")
        if context.requires_tools and not profile.supports_tools:
            reasons.append("tools_unsupported")
        if context.requires_json and not profile.supports_json:
            reasons.append("json_unsupported")
        if context.requires_thinking and not profile.supports_thinking:
            reasons.append("thinking_unsupported")
        if context.modality not in profile.modalities:
            reasons.append("modality_unsupported")
        if context.language and profile.languages and context.language not in profile.languages:
            reasons.append("language_unsupported")
        if profile.privacy_tier < context.minimum_privacy_tier:
            reasons.append("privacy_policy")
        if profile.supported_runtime_parameters is not None:
            unsupported_runtime = sorted(
                context.runtime_parameters.difference(
                    profile.supported_runtime_parameters
                )
            )
            reasons.extend(
                f"runtime_parameter_unsupported:{parameter}"
                for parameter in unsupported_runtime
            )
        supported_template_parameters = set(
            profile.supported_chat_template_parameters
        )
        supported_template_parameters.update(
            str(key) for key in profile.chat_template_defaults
        )
        unsupported_template = sorted(
            context.chat_template_parameters.difference(
                supported_template_parameters
            )
        )
        reasons.extend(
            f"chat_template_parameter_unsupported:{parameter}"
            for parameter in unsupported_template
        )
        return reasons

    def _selected(
        self,
        started: float,
        selected: ModelCapabilityProfile,
        category: str,
        reason: RouteReason,
        confidence: float,
        scored: list[CandidateScore],
    ) -> RouteDecision:
        return self._finish(
            started,
            RouteDecision(
                selected.model_id,
                selected.revision,
                category,
                reason,
                confidence,
                0.0,
                tuple(scored),
                True,
            ),
        )

    def _finish(self, started: float, decision: RouteDecision) -> RouteDecision:
        latency_ms = (time.perf_counter() - started) * 1_000
        finished = RouteDecision(
            decision.model_id,
            decision.revision,
            decision.category,
            decision.reason,
            decision.confidence,
            latency_ms,
            decision.candidates,
            decision.hard_constraints_satisfied,
        )
        self.metrics.record(finished)
        return finished
