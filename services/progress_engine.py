"""Deterministic progress and retest comparison engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from schemas import (
    AuthorityEvolution,
    CoachingEvolution,
    DimensionDelta,
    MetricDelta,
    Milestone,
    MomentDelta,
    Progress,
    ProgressComparison,
    ProgressConfidence,
    ProgressState,
    RetestRecommendation,
    TrendSummary,
    WeeklySummary,
)


DIMENSIONS = ("command", "clarity", "composure", "presence", "persuasion", "structure")
METRIC_PATHS = {
    "pace": ("metrics", "raw_acoustic", "words_per_minute"),
    "fillers": ("metrics", "linguistic", "filler_words_per_min"),
    "pause_ownership": ("metrics", "vad", "avg_pause_duration_ms"),
    "terminal_endings": ("metrics", "raw_acoustic", "terminal_rising_ratio"),
    "projection": ("metrics", "derived", "projection_index"),
    "monotony": ("metrics", "derived", "monotony_index"),
    "structure": ("metrics", "linguistic", "structure_score"),
    "certainty_language": ("metrics", "linguistic", "certainty_markers_per_100_words"),
    "rambling": ("metrics", "linguistic", "rambling_score"),
    "dynamic_emphasis": ("metrics", "derived", "dynamic_emphasis_score"),
}
LOWER_IS_BETTER = {"fillers", "terminal_endings", "monotony", "rambling"}
WEAK_MOMENTS = {"weakest_moment", "confidence_drop", "rushing_moment", "hesitation_cluster", "filler_cluster", "weak_ending"}
STRONG_MOMENTS = {"strongest_moment", "decisive_moment", "strong_ending", "best_sentence"}


@dataclass(frozen=True)
class ProgressSnapshot:
    analysis_id: str
    created_at: str
    scenario: str
    authority_score: int
    score_confidence: float
    score_band: str | None
    rarity_label: str | None
    dimension_scores: dict[str, int]
    metrics: dict[str, Any]
    moments: list[dict[str, Any]]
    evidence_ids: list[str]
    authority_type: str | None
    primary_drill_id: str | None
    future_drill_ids: tuple[str, ...]
    audio_usable: bool
    uncertainty_reasons: tuple[str, ...]


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _get_nested(source: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = source
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def snapshot_from_response(response: Any) -> ProgressSnapshot:
    report = getattr(response, "report", None)
    coaching = getattr(response, "coaching_engine", None)
    primary = getattr(getattr(coaching, "selected_interventions", None), "primary_drill", None)
    queue = getattr(coaching, "future_training_queue", []) or []
    evidence = getattr(response, "evidence", []) or []
    return ProgressSnapshot(
        analysis_id=getattr(response, "analysis_id", ""),
        created_at=getattr(response, "created_at", ""),
        scenario=getattr(getattr(response, "request", None), "scenario", "benchmark"),
        authority_score=getattr(getattr(response, "scores", None), "authority_score", 0),
        score_confidence=getattr(getattr(response, "scores", None), "score_confidence", 0.0) or 0.0,
        score_band=getattr(getattr(response, "scores", None), "score_band", None),
        rarity_label=getattr(getattr(response, "scores", None), "score_rarity_label", None),
        dimension_scores=getattr(getattr(response, "scores", None), "dimension_scores", None).model_dump()
        if getattr(getattr(response, "scores", None), "dimension_scores", None)
        else {},
        metrics=getattr(response, "metrics", None).model_dump() if getattr(response, "metrics", None) else {},
        moments=[item.model_dump() if hasattr(item, "model_dump") else dict(item) for item in getattr(response, "moments", [])],
        evidence_ids=[getattr(item, "id", "") for item in evidence],
        authority_type=getattr(getattr(report, "authority_type", None), "label", None),
        primary_drill_id=getattr(primary, "drill_id", None),
        future_drill_ids=tuple(getattr(item, "drill_id", "") for item in queue),
        audio_usable=getattr(getattr(response, "audio_quality", None), "usable", True),
        uncertainty_reasons=tuple(getattr(getattr(response, "uncertainty", None), "reasons", []) or []),
    )


def _parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.min


def _ordered_history(history: list[ProgressSnapshot]) -> list[ProgressSnapshot]:
    unique: dict[str, ProgressSnapshot] = {}
    for item in history:
        if item.analysis_id:
            unique[item.analysis_id] = item
    return sorted(unique.values(), key=lambda item: _parse_time(item.created_at))


def _trend(delta: float | None, threshold: float = 1.0, *, lower_is_better: bool = False) -> str:
    if delta is None:
        return "unavailable"
    adjusted = -delta if lower_is_better else delta
    if adjusted > threshold:
        return "improved"
    if adjusted < -threshold:
        return "declined"
    return "unchanged"


def _dimension_deltas(previous: ProgressSnapshot, current: ProgressSnapshot) -> dict[str, DimensionDelta]:
    deltas = {}
    for dimension in DIMENSIONS:
        old = previous.dimension_scores.get(dimension)
        new = current.dimension_scores.get(dimension)
        delta = None if old is None or new is None else new - old
        relative = None if old in (None, 0) or delta is None else round(delta / old, 3)
        deltas[dimension] = DimensionDelta(
            dimension=dimension,
            old_score=old,
            new_score=new,
            absolute_delta=delta,
            relative_delta=relative,
            trend=_trend(delta),
            confidence=round(min(previous.score_confidence, current.score_confidence), 2),
        )
    return deltas


def _metric_deltas(previous: ProgressSnapshot, current: ProgressSnapshot) -> dict[str, MetricDelta]:
    deltas = {}
    prev_root = {"metrics": previous.metrics}
    cur_root = {"metrics": current.metrics}
    for metric_id, path in METRIC_PATHS.items():
        old_raw = _get_nested(prev_root, path)
        new_raw = _get_nested(cur_root, path)
        if old_raw is None or new_raw is None:
            continue
        old = float(old_raw)
        new = float(new_raw)
        delta = round(new - old, 3)
        relative = None if old == 0 else round(delta / old, 3)
        deltas[metric_id] = MetricDelta(
            metric_id=metric_id,
            old_value=old,
            new_value=new,
            absolute_delta=delta,
            relative_delta=relative,
            trend=_trend(delta, threshold=0.02 if abs(old) <= 1 else 1.0, lower_is_better=metric_id in LOWER_IS_BETTER),
            confidence=round(min(previous.score_confidence, current.score_confidence), 2),
        )
    return deltas


def _moment_counts(snapshot: ProgressSnapshot) -> dict[str, int]:
    counts: dict[str, int] = {}
    for moment in snapshot.moments:
        moment_type = str(moment.get("type", "unknown"))
        counts[moment_type] = counts.get(moment_type, 0) + 1
    return counts


def _moment_comparison(previous: ProgressSnapshot, current: ProgressSnapshot) -> list[MomentDelta]:
    prev = _moment_counts(previous)
    cur = _moment_counts(current)
    results: list[MomentDelta] = []
    for moment_type in sorted(set(prev) | set(cur)):
        old = prev.get(moment_type, 0)
        new = cur.get(moment_type, 0)
        if moment_type in STRONG_MOMENTS and new > old:
            status = "new_strength"
        elif moment_type in WEAK_MOMENTS and old > 0 and new == 0:
            status = "resolved_weakness"
        elif moment_type in WEAK_MOMENTS and old > 0 and new > 0:
            status = "persistent_weakness"
        elif moment_type in WEAK_MOMENTS and old == 0 and new > 0:
            status = "new_weakness"
        else:
            status = "unchanged"
        results.append(MomentDelta(moment_type=moment_type, status=status, previous_count=old, current_count=new, evidence_ids=current.evidence_ids[:3], confidence=round(min(previous.score_confidence, current.score_confidence), 2)))
    return results


def _retest_recommendation(current: ProgressSnapshot) -> RetestRecommendation:
    focus = current.primary_drill_id or (current.future_drill_ids[0] if current.future_drill_ids else "same_prompt_baseline")
    return RetestRecommendation(
        recommended_retest_after_days=3,
        why="Compare the next recording after repeating the current deterministic focus.",
        what_to_compare=["authority_score", "dimension_scores", focus],
        success_definition="Success means the target dimension improves without a confidence drop.",
        comparison_focus=focus,
    )


def _confidence(current: ProgressSnapshot, target: ProgressSnapshot | None, compatible: bool) -> ProgressConfidence:
    confidence = current.score_confidence
    reasons = list(current.uncertainty_reasons)
    if target:
        confidence = min(confidence, target.score_confidence)
    else:
        confidence = min(confidence, 0.45)
        reasons.append("No previous benchmark is available")
    if not compatible:
        confidence = min(confidence, 0.5)
        reasons.append("Scenario mismatch limits comparison certainty")
    elif target and target.scenario != current.scenario:
        confidence = min(confidence, 0.5)
        reasons.append("Scenario mismatch limits comparison certainty")
    if not current.audio_usable or (target and not target.audio_usable):
        confidence = min(confidence, 0.45)
        reasons.append("Audio quality limits progress confidence")
    label = "high" if confidence >= 0.8 else "medium_high" if confidence >= 0.65 else "medium" if confidence >= 0.45 else "low"
    return ProgressConfidence(confidence=round(confidence, 2), confidence_label=label, reasons=list(dict.fromkeys(reasons)))  # type: ignore[arg-type]


def _unavailable_confidence(current: ProgressSnapshot, reasons: list[str]) -> ProgressConfidence:
    confidence = min(current.score_confidence, 0.45)
    label = "medium" if confidence >= 0.45 else "low"
    return ProgressConfidence(confidence=round(confidence, 2), confidence_label=label, reasons=list(dict.fromkeys(reasons)))  # type: ignore[arg-type]


def _authority_evolution(previous: ProgressSnapshot | None, current: ProgressSnapshot, dimension_deltas: dict[str, DimensionDelta]) -> AuthorityEvolution:
    if not previous:
        return AuthorityEvolution(current_type=current.authority_type, status="unavailable", confidence=current.score_confidence)
    if previous.authority_type == current.authority_type:
        improving = sum(1 for delta in dimension_deltas.values() if delta.trend == "improved")
        status = "strengthening" if improving >= 2 else "unchanged"
    else:
        status = "shifting"
    dominant = sorted(current.dimension_scores, key=current.dimension_scores.get, reverse=True)[:2]
    return AuthorityEvolution(previous_type=previous.authority_type, current_type=current.authority_type, status=status, new_dominant_characteristics=dominant, confidence=round(min(previous.score_confidence, current.score_confidence), 2))  # type: ignore[arg-type]


def _coaching_evolution(previous: ProgressSnapshot | None, current: ProgressSnapshot) -> CoachingEvolution:
    if not previous:
        return CoachingEvolution(new_focus=[item for item in (current.primary_drill_id,) if item], future_queue_advancement=list(current.future_drill_ids))
    previous_focus = {item for item in (previous.primary_drill_id,) if item}
    current_focus = {item for item in (current.primary_drill_id,) if item}
    return CoachingEvolution(
        completed_focus=sorted(previous_focus - current_focus),
        continuing_focus=sorted(previous_focus & current_focus),
        new_focus=sorted(current_focus - previous_focus),
        resolved_interventions=sorted(previous_focus - current_focus),
        new_interventions=sorted(current_focus - previous_focus),
        dependency_progression=[item for item in current.future_drill_ids if item not in previous.future_drill_ids],
        future_queue_advancement=list(current.future_drill_ids),
    )


def _stability(history: list[ProgressSnapshot], current: ProgressSnapshot) -> tuple[float, float, float]:
    scores = [item.authority_score for item in history] + [current.authority_score]
    if len(scores) < 2:
        return 1.0, 1.0, 0.0
    avg = sum(scores) / len(scores)
    variance = sum((score - avg) ** 2 for score in scores) / len(scores)
    volatility = min(1.0, (variance ** 0.5) / 25)
    stability = round(1.0 - volatility, 2)
    consistency = round(1.0 - min(1.0, len([1 for a, b in zip(scores, scores[1:]) if abs(b - a) > 8]) / max(1, len(scores) - 1)), 2)
    return consistency, stability, round(volatility, 2)


def build_progress(current: ProgressSnapshot, history: list[ProgressSnapshot] | None = None, *, allow_cross_scenario: bool = False) -> Progress:
    ordered = _ordered_history(history or [])
    baseline = ordered[0] if ordered else None
    same_scenario = [item for item in ordered if item.scenario == current.scenario]
    target = same_scenario[-1] if same_scenario else (ordered[-1] if allow_cross_scenario and ordered else None)
    compatible = target is not None and (target.scenario == current.scenario or allow_cross_scenario)

    if not target:
        if ordered:
            available_scenarios = sorted({item.scenario for item in ordered})
            reason = "scenario_mismatch" if current.scenario not in available_scenarios else "no_valid_comparison_target"
            confidence_reasons = ["Insufficient compatible history"]
            if reason == "scenario_mismatch":
                confidence_reasons.extend([
                    "Previous recordings use a different scenario",
                    "Cross-scenario comparison is disabled",
                ])
            return Progress(
                comparison_available=False,
                baseline_analysis_id=baseline.analysis_id if baseline else None,
                state=ProgressState(
                    state="no_compatible_history",
                    progress_status="no_compatible_history",
                    reason=reason,
                    current_scenario=current.scenario,
                    available_history_scenarios=available_scenarios,
                    cross_scenario_comparison_blocked=not allow_cross_scenario,
                    user_safe_explanation="Previous recordings exist, but none are compatible with this comparison.",
                    baseline_established=True,
                    latest_benchmark_id=current.analysis_id,
                    history_count=len(ordered),
                    progress_preview=["comparison_unavailable"],
                    expected_future_comparisons=["same_scenario_retest", "authority_score", "six_dimensions", "coaching_focus"],
                    next_retest_recommendation=_retest_recommendation(current),
                ),
                retest_recommendation=_retest_recommendation(current),
                confidence=_unavailable_confidence(current, confidence_reasons),
            )
        return Progress(
            comparison_available=False,
            baseline_analysis_id=baseline.analysis_id if baseline else None,
            state=ProgressState(
                state="first_benchmark",
                progress_status="first_benchmark",
                reason="no_previous_benchmark",
                current_scenario=current.scenario,
                available_history_scenarios=[],
                cross_scenario_comparison_blocked=False,
                user_safe_explanation="This is the first benchmark, so progress comparisons will begin after the next compatible recording.",
                baseline_established=True,
                latest_benchmark_id=current.analysis_id,
                history_count=len(ordered),
                progress_preview=["baseline_established"],
                expected_future_comparisons=["authority_score", "six_dimensions", "coaching_focus", "moments"],
                next_retest_recommendation=_retest_recommendation(current),
            ),
            milestones=[Milestone(milestone_id="first_benchmark", label="First benchmark", source_analysis_id=current.analysis_id, confidence=current.score_confidence)],
            retest_recommendation=_retest_recommendation(current),
            confidence=_confidence(current, None, True),
        )

    dimension_deltas = _dimension_deltas(target, current)
    metric_deltas = _metric_deltas(target, current)
    score_delta = current.authority_score - target.authority_score
    trend = _trend(score_delta)
    consistency, stability, volatility = _stability(ordered, current)
    largest_improvement = max(dimension_deltas.values(), key=lambda item: item.absolute_delta if item.absolute_delta is not None else -999)
    largest_regression = min(dimension_deltas.values(), key=lambda item: item.absolute_delta if item.absolute_delta is not None else 999)
    milestones = []
    if len(ordered) == 1:
        milestones.append(Milestone(milestone_id="first_retest", label="First retest", source_analysis_id=current.analysis_id, confidence=current.score_confidence))
    if score_delta > 0 and current.authority_score >= max(item.authority_score for item in ordered):
        milestones.append(Milestone(milestone_id="highest_authority_score", label="Highest authority score", source_analysis_id=current.analysis_id, confidence=current.score_confidence))
    regressions = []
    if score_delta <= -5:
        regressions.append(Milestone(milestone_id="meaningful_decline", label="Meaningful decline", source_analysis_id=current.analysis_id, confidence=min(current.score_confidence, target.score_confidence)))
    if current.score_confidence < target.score_confidence - 0.12:
        regressions.append(Milestone(milestone_id="confidence_reduction", label="Confidence reduction", source_analysis_id=current.analysis_id, confidence=current.score_confidence))

    return Progress(
        comparison_available=compatible,
        baseline_analysis_id=baseline.analysis_id if baseline else target.analysis_id,
        delta_authority_score=score_delta,
        dimension_deltas={key: value.absolute_delta or 0.0 for key, value in dimension_deltas.items()},
        state=ProgressState(
            state="retest",
            progress_status="comparison_available",
            current_scenario=current.scenario,
            available_history_scenarios=sorted({item.scenario for item in ordered}),
            cross_scenario_comparison_blocked=False,
            user_safe_explanation="Progress is compared against the latest compatible recording.",
            baseline_established=True,
            latest_benchmark_id=current.analysis_id,
            history_count=len(ordered),
            progress_preview=[trend],
            expected_future_comparisons=["trend", "stability", "coaching_evolution"],
            next_retest_recommendation=_retest_recommendation(current),
        ),
        comparison=ProgressComparison(
            current_analysis_id=current.analysis_id,
            comparison_target_id=target.analysis_id,
            previous_best_id=max(ordered, key=lambda item: item.authority_score).analysis_id,
            previous_worst_id=min(ordered, key=lambda item: item.authority_score).analysis_id,
            same_scenario=target.scenario == current.scenario,
            compatible=compatible,
            authority_score_delta=score_delta,
            percentage_change=None if target.authority_score == 0 else round(score_delta / target.authority_score * 100, 2),
            relative_improvement=None if target.authority_score == 0 else round(score_delta / target.authority_score, 3),
            band_movement=f"{target.score_band}->{current.score_band}",
            rarity_movement=f"{target.rarity_label}->{current.rarity_label}",
            confidence_movement=round(current.score_confidence - target.score_confidence, 2),
            overall_trend=trend,  # type: ignore[arg-type]
        ),
        dimension_delta_details=dimension_deltas,
        metric_deltas=metric_deltas,
        moment_comparison=_moment_comparison(target, current),
        authority_evolution=_authority_evolution(target, current, dimension_deltas),
        coaching_evolution=_coaching_evolution(target, current),
        trend_summary=TrendSummary(score_trend="improving" if trend == "improved" else "declining" if trend == "declined" else "stable", dimension_trend={k: v.trend for k, v in dimension_deltas.items()}, metric_trend={k: v.trend for k, v in metric_deltas.items()}, coaching_trend="updated" if target.primary_drill_id != current.primary_drill_id else "continuing", authority_type_trend="updated" if target.authority_type != current.authority_type else "unchanged"),
        weekly_summary=WeeklySummary(largest_improvement=largest_improvement.dimension, remaining_limiter=largest_regression.dimension, best_moment=next((item.get("moment_id") for item in current.moments if item.get("type") in STRONG_MOMENTS), None), most_improved_dimension=largest_improvement.dimension, least_improved_dimension=largest_regression.dimension, completed_drills=_coaching_evolution(target, current).completed_focus, recommended_focus=current.primary_drill_id),
        milestones=milestones,
        regressions=regressions,
        consistency_score=consistency,
        stability_score=stability,
        volatility_score=volatility,
        evidence_consistency=round(min(len(set(current.evidence_ids)) / 5, 1.0), 2),
        dimension_stability=stability,
        authority_stability=stability,
        retest_recommendation=_retest_recommendation(current),
        confidence=_confidence(current, target, compatible),
    )
