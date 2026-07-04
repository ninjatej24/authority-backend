"""Deterministic analytics, calibration and feedback preparation layer.

This module observes completed Authority responses. It never feeds analytics
back into scoring, coaching, inference, report generation, or progress.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from schemas import (
    AnalysisAnalytics,
    AnalyticsBundle,
    AnalyticsEvent,
    AnalyticsSummary,
    CalibrationRecord,
    CoachingAnalytics,
    FairnessAuditRecord,
    ProductMetrics,
    ProgressAnalytics,
    RetentionSummary,
    TimelineAnalytics,
)


MOMENT_COUNT_KEYS = {
    "strongest_moment": "strongest_moment_count",
    "weakest_moment": "weakest_moment_count",
    "confidence_drop": "confidence_drop_count",
    "rushing_moment": "rushing_event_count",
    "hesitation_cluster": "hesitation_cluster_count",
    "filler_cluster": "filler_cluster_count",
    "monotone_stretch": "monotone_stretch_count",
    "pause_ownership_moment": "pause_ownership_moment_count",
    "decisive_moment": "decisive_moment_count",
    "most_commanding_moment": "decisive_moment_count",
}


def _dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _compact_nonzero(source: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in source.items():
        if isinstance(value, (int, float)) and value:
            result[key] = round(float(value), 3)
    return result


def _dimension_scores(response: Any) -> dict[str, int]:
    scores = getattr(getattr(response, "scores", None), "dimension_scores", None)
    if not scores:
        return {}
    return {key: int(value) for key, value in scores.model_dump().items()}


def _authority_type(response: Any) -> str | None:
    report_type = getattr(getattr(getattr(response, "report", None), "authority_type", None), "label", None)
    if report_type:
        return report_type
    share_type = getattr(getattr(getattr(response, "report", None), "share_card", None), "authority_type", None)
    return share_type


def _audio_quality_metadata(response: Any) -> dict[str, Any]:
    audio = getattr(response, "audio_quality", None)
    if not audio:
        return {}
    return {
        "usable": getattr(audio, "usable", None),
        "snr_estimate_db": getattr(audio, "snr_estimate_db", None),
        "clipping_detected": getattr(audio, "clipping_detected", None),
        "background_noise_level": getattr(audio, "background_noise_level", None),
        "warning_count": len(getattr(audio, "quality_warnings", []) or []),
    }


def _timeline_counts(response: Any) -> TimelineAnalytics:
    counts = {field: 0 for field in TimelineAnalytics.model_fields}
    moments = list(getattr(getattr(response, "moment_intelligence", None), "moments", []) or [])
    if not moments:
        moments = list(getattr(response, "moments", []) or [])
    for moment in moments:
        moment_type = getattr(moment, "type", None) or _dump(moment).get("type")
        field = MOMENT_COUNT_KEYS.get(str(moment_type))
        if field:
            counts[field] = counts.get(field, 0) + 1
    counts["total_moment_count"] = len(moments)
    return TimelineAnalytics(**counts)


def _moment_count_map(timeline: TimelineAnalytics) -> dict[str, int]:
    return {
        key: value
        for key, value in timeline.model_dump().items()
        if key.endswith("_count") and isinstance(value, int)
    }


def _coaching_analytics(response: Any) -> CoachingAnalytics:
    coaching = getattr(response, "coaching_engine", None)
    primary = getattr(getattr(coaching, "selected_interventions", None), "primary_drill", None)
    expected = getattr(primary, "expected_impact", None)
    report_drill = getattr(getattr(getattr(response, "report", None), "training_prescription", None), "drill_id", None)
    legacy_drill = getattr((getattr(response, "drills", []) or [None])[0], "drill_id", None)
    drill_id = getattr(primary, "drill_id", None) or report_drill or legacy_drill
    dependency_ids = []
    for dependency in getattr(coaching, "dependency_graph", []) or []:
        before = getattr(dependency, "before", "")
        after = getattr(dependency, "after", "")
        if before or after:
            dependency_ids.append(f"{before}->{after}")
    return CoachingAnalytics(
        selected_drill_id=drill_id,
        selected_intervention_id=drill_id,
        highest_leverage_issue=getattr(getattr(response, "recommendations", None), "highest_leverage_issue", None),
        targeted_dimensions=list((getattr(primary, "expected_impact", None) and [
            dimension
            for dimension in ("command", "clarity", "composure", "presence", "persuasion", "structure")
            if getattr(expected, dimension, 0.0)
        ]) or getattr(primary, "target_dimensions", []) or getattr(getattr(getattr(response, "report", None), "highest_leverage_fix", None), "target_dimensions", []) or []),
        expected_improvement=_compact_nonzero(_dump(expected)),
        drill_priority=getattr(primary, "score", None),
        dependency_graph_ids=dependency_ids,
    )


def _progress_analytics(response: Any) -> ProgressAnalytics:
    progress = getattr(response, "progress", None)
    dimension_deltas = getattr(progress, "dimension_deltas", None) or {}
    if not dimension_deltas:
        dimension_deltas = {
            key: getattr(delta, "absolute_delta", None)
            for key, delta in (getattr(progress, "dimension_delta_details", {}) or {}).items()
            if getattr(delta, "absolute_delta", None) is not None
        }
    return ProgressAnalytics(
        score_delta=getattr(progress, "delta_authority_score", None),
        dimension_deltas={key: float(value) for key, value in dimension_deltas.items() if value is not None},
        authority_evolution=_dump(getattr(progress, "authority_evolution", None)),
        milestones_earned=[getattr(item, "milestone_id", "") for item in getattr(progress, "milestones", []) or []],
        regressions=[getattr(item, "milestone_id", "") for item in getattr(progress, "regressions", []) or []],
        coaching_evolution=_dump(getattr(progress, "coaching_evolution", None)),
        retest_recommendation=_dump(getattr(progress, "retest_recommendation", None)),
        trend_summary=_dump(getattr(progress, "trend_summary", None)),
    )


def _analysis_analytics(response: Any) -> AnalysisAnalytics:
    scores = getattr(response, "scores", None)
    components = getattr(scores, "score_components", None)
    pipeline_audit = getattr(getattr(response, "pipeline_validation", None), "audit", None)
    explainability_audit = getattr(getattr(response, "explainability", None), "audit", None)
    return AnalysisAnalytics(
        analysis_id=getattr(response, "analysis_id", None),
        timestamp=getattr(response, "created_at", None),
        scenario=getattr(getattr(response, "request", None), "scenario", "benchmark"),
        duration_ms=getattr(getattr(response, "request", None), "duration_ms", 0),
        authority_score=getattr(scores, "authority_score", None),
        score_band=getattr(scores, "score_band", None),
        confidence=getattr(scores, "score_confidence", None),
        authority_type=_authority_type(response),
        dimension_scores=_dimension_scores(response),
        penalties=_compact_nonzero(_dump(getattr(components, "penalties", None))),
        bonuses=_compact_nonzero(_dump(getattr(components, "bonuses", None))),
        audio_quality=_audio_quality_metadata(response),
        uncertainty_reasons=list(getattr(getattr(response, "uncertainty", None), "reasons", []) or []),
        validation_integrity=getattr(pipeline_audit, "integrity_score", None),
        explainability_integrity=getattr(explainability_audit, "report_integrity_score", None),
    )


def _retention_summary(response: Any) -> RetentionSummary:
    history = getattr(response, "history_summary", None)
    weekly = getattr(response, "weekly_summary", None)
    monthly = getattr(response, "monthly_summary", None)
    retest = getattr(getattr(response, "progress", None), "state", None)
    return RetentionSummary(
        days_since_benchmark=getattr(history, "days_since_benchmark", None),
        days_since_drill=getattr(history, "days_since_drill", None),
        days_since_retest=0 if getattr(retest, "state", "") == "retest" else None,
        weekly_activity=1 if getattr(weekly, "new_milestone", None) else 0,
        monthly_activity=getattr(monthly, "benchmark_count", 0) or 0,
        streak=getattr(history, "practice_streak", 0) or 0,
        drop_off_risk_inputs={
            "days_since_benchmark": getattr(history, "days_since_benchmark", None),
            "days_since_practice": getattr(history, "days_since_practice", None),
            "benchmark_count": getattr(history, "benchmark_count", 0),
        },
    )


def _product_metrics(response: Any) -> ProductMetrics:
    history = getattr(response, "history_summary", None)
    progress = getattr(response, "progress", None)
    dimensions = list(_dimension_scores(response).values())
    confidences = [getattr(getattr(response, "scores", None), "score_confidence", None)]
    usable = getattr(getattr(response, "audio_quality", None), "usable", None)
    quality_values = [1.0 if usable else 0.0] if usable is not None else []
    delta_values = [
        value
        for value in (getattr(progress, "dimension_deltas", None) or {}).values()
        if isinstance(value, (int, float))
    ]
    return ProductMetrics(
        benchmark_completion=bool(getattr(response, "analysis_id", None)),
        premium_unlock_available=bool(getattr(response, "report", None)),
        report_completion=bool(getattr(getattr(response, "report", None), "mirror", None)),
        average_drill_completion=getattr(history, "training_adherence", 0.0) or 0.0,
        retest_frequency=1.0 if getattr(progress, "comparison_available", False) else 0.0,
        share_frequency=1.0 if getattr(getattr(getattr(response, "report", None), "share_card", None), "authority_score", None) is not None else 0.0,
        history_length=getattr(history, "benchmark_count", 0) or 0,
        average_improvement=round(mean(delta_values), 2) if delta_values else None,
        average_confidence=round(mean([value for value in confidences if value is not None]), 2) if any(value is not None for value in confidences) else None,
        average_quality=round(mean(quality_values), 2) if quality_values else None,
    )


def _fairness_flags(response: Any) -> list[str]:
    scores = getattr(response, "scores", None)
    fairness = getattr(scores, "fairness_adjustments", None)
    flags = list(getattr(fairness, "applied_adjustments", []) or [])
    flags.extend(getattr(fairness, "suppressed_features", []) or [])
    audio = getattr(response, "audio_quality", None)
    if audio and not getattr(audio, "usable", True):
        flags.append("quality_gated")
    transcript = getattr(response, "transcript", None)
    if getattr(transcript, "overall_asr_confidence", None) is not None and transcript.overall_asr_confidence < 0.7:
        flags.append("low_asr_confidence")
    return list(dict.fromkeys(flags))


def build_analytics_bundle(response: Any) -> AnalyticsBundle:
    """Collect analytics metadata from an already-completed response."""
    timeline = _timeline_counts(response)
    analysis = _analysis_analytics(response)
    coaching = _coaching_analytics(response)
    progress = _progress_analytics(response)
    fairness_flags = _fairness_flags(response)
    event = AnalyticsEvent(
        event_id=f"benchmark_completed:{getattr(response, 'analysis_id', '')}",
        event_type="benchmark_completed",
        timestamp=getattr(response, "created_at", None),
        analysis_id=getattr(response, "analysis_id", None),
        user_key_present=getattr(getattr(response, "persistence_status", None), "user_key_present", False),
        scenario=analysis.scenario,
        metadata={
            "persisted": getattr(getattr(response, "persistence_status", None), "persisted", False),
            "score_band": analysis.score_band,
            "confidence": analysis.confidence,
        },
    )
    return AnalyticsBundle(
        summary=AnalyticsSummary(
            collected_at=getattr(response, "created_at", None),
            notes=[
                "Analytics observe completed deterministic outputs only.",
                "No transcript, raw audio, report prose, or private moment text is included.",
            ],
        ),
        analysis=analysis,
        coaching=coaching,
        progress=progress,
        timeline=timeline,
        retention=_retention_summary(response),
        product_metrics=_product_metrics(response),
        calibration_record=CalibrationRecord(
            record_id=f"calibration:{getattr(response, 'analysis_id', '')}",
            analysis_id=analysis.analysis_id,
            created_at=analysis.timestamp,
            authority_score=analysis.authority_score,
            score_band=analysis.score_band,
            dimension_scores=analysis.dimension_scores,
            scenario=analysis.scenario,
            confidence=analysis.confidence,
            authority_type=analysis.authority_type,
            audio_quality=analysis.audio_quality,
            validation_integrity=analysis.validation_integrity,
            explainability_integrity=analysis.explainability_integrity,
            moment_counts=_moment_count_map(timeline),
            coaching_id=coaching.selected_drill_id,
            progress_outcome=getattr(getattr(response, "progress", None), "comparison", None)
            and getattr(response.progress.comparison, "overall_trend", None),
        ),
        fairness_audit=FairnessAuditRecord(
            analysis_id=analysis.analysis_id,
            audio_quality=analysis.audio_quality,
            confidence=analysis.confidence,
            scenario=analysis.scenario,
            language=getattr(getattr(response, "request", None), "language", "en"),
            asr_confidence=getattr(getattr(response, "transcript", None), "overall_asr_confidence", None),
            quality_suppression=list(getattr(getattr(response, "uncertainty", None), "reasons", []) or []),
            fairness_flags=fairness_flags,
        ),
        user_behaviour_events=[event],
    )
