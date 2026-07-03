"""Milestone 10 deterministic progress and retest engine tests."""

from __future__ import annotations

from schemas import AuthorityV2Response
from services.progress_engine import ProgressSnapshot, build_progress


def _snapshot(
    analysis_id: str = "a1",
    *,
    score: int = 60,
    scenario: str = "benchmark",
    confidence: float = 0.78,
    dims: dict[str, int] | None = None,
    metrics: dict | None = None,
    moments: list[dict] | None = None,
    authority_type: str = "Thoughtful Strategist",
    drill: str | None = "pause_ownership_v1",
    created_at: str = "2026-07-01T10:00:00Z",
    audio_usable: bool = True,
) -> ProgressSnapshot:
    return ProgressSnapshot(
        analysis_id=analysis_id,
        created_at=created_at,
        scenario=scenario,
        authority_score=score,
        score_confidence=confidence,
        score_band="competent" if score < 67 else "strong",
        rarity_label="middle 30%" if score < 67 else "next 25%",
        dimension_scores=dims
        or {
            "command": score,
            "clarity": score,
            "composure": score,
            "presence": score,
            "persuasion": score,
            "structure": score,
        },
        metrics=metrics
        or {
            "raw_acoustic": {"words_per_minute": 145, "terminal_rising_ratio": 0.3},
            "linguistic": {"filler_words_per_min": 4.0, "structure_score": 0.6, "certainty_markers_per_100_words": 2.0, "rambling_score": 0.4},
            "vad": {"avg_pause_duration_ms": 420},
            "derived": {"projection_index": 0.55, "monotony_index": 0.35, "dynamic_emphasis_score": 0.55},
        },
        moments=moments or [{"moment_id": "m_weak", "type": "weak_ending"}],
        evidence_ids=["ev_1", "ev_2"],
        authority_type=authority_type,
        primary_drill_id=drill,
        future_drill_ids=("drop_the_landing_v1",),
        audio_usable=audio_usable,
        uncertainty_reasons=(),
    )


def test_first_benchmark_establishes_baseline_without_fabricating_progress():
    current = _snapshot()
    progress = build_progress(current, [])

    assert progress.comparison_available is False
    assert progress.state.state == "first_benchmark"
    assert progress.state.baseline_established is True
    assert progress.delta_authority_score is None
    assert progress.milestones[0].milestone_id == "first_benchmark"


def test_second_benchmark_computes_score_dimension_and_metric_deltas():
    previous = _snapshot("a1", score=60)
    current = _snapshot("a2", score=68, dims={"command": 70, "clarity": 68, "composure": 66, "presence": 65, "persuasion": 67, "structure": 69}, created_at="2026-07-04T10:00:00Z")
    progress = build_progress(current, [previous])

    assert progress.comparison_available is True
    assert progress.delta_authority_score == 8
    assert progress.comparison.overall_trend == "improved"
    assert progress.dimension_delta_details["command"].absolute_delta == 10
    assert "pace" in progress.metric_deltas
    assert progress.milestones[0].milestone_id == "first_retest"


def test_third_and_long_history_detect_previous_best_worst_and_stability():
    history = [
        _snapshot("a1", score=55, created_at="2026-07-01T10:00:00Z"),
        _snapshot("a2", score=64, created_at="2026-07-02T10:00:00Z"),
        _snapshot("a3", score=58, created_at="2026-07-03T10:00:00Z"),
        _snapshot("a4", score=70, created_at="2026-07-04T10:00:00Z"),
    ]
    progress = build_progress(_snapshot("a5", score=72, created_at="2026-07-05T10:00:00Z"), history)

    assert progress.comparison.previous_best_id == "a4"
    assert progress.comparison.previous_worst_id == "a1"
    assert progress.consistency_score is not None
    assert progress.volatility_score is not None


def test_regression_and_no_change_are_not_over_optimistic():
    unchanged = build_progress(_snapshot("a2", score=60, created_at="2026-07-02T10:00:00Z"), [_snapshot("a1", score=60)])
    declined = build_progress(_snapshot("a3", score=52, created_at="2026-07-03T10:00:00Z"), [_snapshot("a1", score=60)])

    assert unchanged.comparison.overall_trend == "unchanged"
    assert declined.comparison.overall_trend == "declined"
    assert declined.regressions[0].milestone_id == "meaningful_decline"


def test_moment_comparison_detects_resolved_and_new_weaknesses():
    previous = _snapshot("a1", moments=[{"moment_id": "m1", "type": "weak_ending"}])
    current = _snapshot("a2", moments=[{"moment_id": "m2", "type": "strong_ending"}, {"moment_id": "m3", "type": "rushing_moment"}], created_at="2026-07-02T10:00:00Z")
    progress = build_progress(current, [previous])
    statuses = {item.moment_type: item.status for item in progress.moment_comparison}

    assert statuses["weak_ending"] == "resolved_weakness"
    assert statuses["strong_ending"] == "new_strength"
    assert statuses["rushing_moment"] == "new_weakness"


def test_authority_type_and_coaching_evolution_are_deterministic():
    previous = _snapshot("a1", authority_type="Thoughtful Strategist", drill="pause_ownership_v1")
    current = _snapshot("a2", authority_type="Trusted Expert", drill="drop_the_landing_v1", created_at="2026-07-02T10:00:00Z")
    first = build_progress(current, [previous])
    second = build_progress(current, [previous])

    assert first.model_dump() == second.model_dump()
    assert first.authority_evolution.status == "shifting"
    assert first.coaching_evolution.completed_focus == ["pause_ownership_v1"]
    assert first.coaching_evolution.new_focus == ["drop_the_landing_v1"]


def test_weekly_summary_retest_recommendation_and_schema_validation():
    progress = build_progress(_snapshot("a2", score=66, created_at="2026-07-02T10:00:00Z"), [_snapshot("a1", score=60)])

    assert progress.weekly_summary.most_improved_dimension
    assert progress.retest_recommendation.what_to_compare
    assert progress.confidence.confidence_label in {"low", "medium", "medium_high", "high"}


def test_scenario_mismatch_and_poor_audio_reduce_progress_confidence():
    previous = _snapshot("a1", scenario="interview", confidence=0.8)
    current = _snapshot("a2", scenario="sales", confidence=0.7, audio_usable=False, created_at="2026-07-02T10:00:00Z")
    blocked = build_progress(current, [previous])
    allowed = build_progress(current, [previous], allow_cross_scenario=True)

    assert blocked.comparison_available is False
    assert allowed.comparison.same_scenario is False
    assert allowed.confidence.confidence <= 0.45
    assert any("Scenario mismatch" in reason for reason in allowed.confidence.reasons)


def test_progress_schema_round_trip_inside_authority_response():
    payload = {
        "schema_version": "authority.v2",
        "analysis_id": "00000000-0000-0000-0000-000000000010",
        "created_at": "2026-07-03T10:00:00Z",
        "request": {"scenario": "benchmark", "prompt_id": "p", "language": "en", "duration_ms": 60000},
        "audio_quality": {},
        "transcript": {},
        "scores": {
            "authority_score": 60,
            "score_confidence": 0.7,
            "dimension_scores": {"command": 60, "clarity": 60, "composure": 60, "presence": 60, "persuasion": 60, "structure": 60},
            "derived_axes": {"trust_warmth": 60, "dominance_status": 60, "nervousness": 40, "interview_readiness": 60, "leadership_readiness": 60},
            "score_components": {"weighted_base": 60, "bonuses": {}, "penalties": {}},
        },
        "metrics": {"raw_acoustic": {}, "linguistic": {}, "derived": {}},
        "perception_profile": {"headline": "h", "how_you_currently_come_across": "x", "biggest_strength": {"title": "s", "explanation": "e"}, "biggest_drag": {"title": "d", "explanation": "e"}, "listener_assumptions": [], "reads": {"emotional": "", "professional": "", "social_status": "", "interview": "", "leadership": ""}},
        "evidence": [],
        "moments": [],
        "recommendations": {"highest_leverage_issue": "x", "fastest_improvement_tip": "y", "coaching_summary": "z"},
        "drills": [],
        "progress": build_progress(_snapshot()).model_dump(),
    }
    model = AuthorityV2Response.model_validate(payload)
    assert model.progress.state.state == "first_benchmark"
