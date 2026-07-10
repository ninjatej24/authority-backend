"""Milestone 6 deterministic coaching and intervention tests."""

from __future__ import annotations

from schemas import AudioQuality, Uncertainty
from services.deterministic_coaching import build_deterministic_coaching
from services.response_builder import _deterministic_drills, _deterministic_recommendations
from services.report_builder import apply_coaching_to_report, build_report
from tests.test_diagnostic_reasoning import _diagnostic, _softened_expert_scores
from tests.test_psychological_inference import _infer, _metrics, _scores
from tests.test_report_builder import _evidence, _moments, _test_transcript


def _coaching(**kwargs):
    scores = kwargs.pop("scores", _softened_expert_scores())
    metrics = kwargs.pop("metrics", _metrics())
    audio_quality = kwargs.pop(
        "audio_quality",
        AudioQuality(usable=True, background_noise_level="low"),
    )
    uncertainty = kwargs.pop(
        "uncertainty",
        Uncertainty(overall_confidence_label="medium_high", reasons=[]),
    )
    duration_ms = kwargs.pop("duration_ms", 60000)
    scenario = kwargs.pop("scenario", "benchmark")
    inference = kwargs.pop("inference", _infer(metrics, audio_quality=audio_quality, duration_ms=duration_ms))
    diagnostic = kwargs.pop(
        "diagnostic_reasoning",
        _diagnostic(
            scores=scores,
            metrics=metrics,
            audio_quality=audio_quality,
            uncertainty=uncertainty,
            duration_ms=duration_ms,
            scenario=scenario,
            inference=inference,
        ),
    )
    report = kwargs.pop(
        "report",
        build_report(
            scores=scores,
            metrics=metrics,
            psychological_inference=inference,
            diagnostic_reasoning=diagnostic,
            evidence=_evidence(),
            moments=_moments(),
            uncertainty=uncertainty,
            audio_quality=audio_quality,
            duration_ms=duration_ms,
            scenario=scenario,
        ),
    )
    return build_deterministic_coaching(
        metrics=metrics,
        scores=scores,
        psychological_inference=inference,
        diagnostic_reasoning=diagnostic,
        report=report,
        audio_quality=audio_quality,
        uncertainty=uncertainty,
        duration_ms=duration_ms,
        scenario=scenario,
    )


def test_same_input_always_selects_same_drill():
    metrics = _metrics(
        raw={"terminal_rising_ratio": 0.65},
        linguistic={"closing_strength_score": 0.25, "hedges_per_100_words": 4.0},
    )
    first = _coaching(metrics=metrics)
    second = _coaching(metrics=metrics)

    assert first.drill_library_size >= 20
    assert first.selected_interventions.primary_drill is not None
    assert first.selected_interventions.primary_drill.model_dump() == (
        second.selected_interventions.primary_drill.model_dump()
    )
    assert [item.model_dump() for item in first.intervention_candidates] == [
        item.model_dump() for item in second.intervention_candidates
    ]


def test_changing_evidence_changes_primary_drill():
    command_coaching = _coaching(
        scores=_softened_expert_scores(),
        metrics=_metrics(
            raw={"terminal_rising_ratio": 0.65},
            linguistic={"closing_strength_score": 0.25, "hedges_per_100_words": 4.0},
        ),
    )
    pace_metrics = _metrics(
        raw={"words_per_minute": 195.0},
        rhythm={"speed_up_segments": 2, "burst_speaking_segments": 2, "rhythm_consistency": 0.35},
        derived={"hesitation_cluster_score": 0.75, "composure_index": 0.25},
        linguistic={"filler_words_per_min": 11.0},
    )
    pace_scores = _scores().model_copy(
        update={
            "dimension_scores": _scores().dimension_scores.model_copy(
                update={"composure": 42, "command": 55, "clarity": 66}
            )
        }
    )
    pace_coaching = _coaching(scores=pace_scores, metrics=pace_metrics)

    assert command_coaching.selected_interventions.primary_drill is not None
    assert pace_coaching.selected_interventions.primary_drill is not None
    assert command_coaching.selected_interventions.primary_drill.drill_id != (
        pace_coaching.selected_interventions.primary_drill.drill_id
    )


def test_poor_audio_suppresses_coaching_selection():
    coaching = _coaching(
        audio_quality=AudioQuality(
            usable=False,
            background_noise_level="high",
            quality_warnings=["Very low signal level"],
        ),
        duration_ms=9000,
        uncertainty=Uncertainty(overall_confidence_label="low", reasons=["Poor microphone signal"]),
    )

    assert coaching.selected_interventions.primary_drill is None
    assert coaching.suppressed_interventions
    assert len(coaching.suppressed_interventions) == coaching.drill_library_size
    assert "Poor audio suppresses coaching intervention selection" in coaching.uncertainty.reasons


def test_missing_evidence_reduces_intervention_confidence():
    supported = _coaching(
        metrics=_metrics(
            raw={"terminal_rising_ratio": 0.65},
            linguistic={"closing_strength_score": 0.25, "hedges_per_100_words": 4.0},
        )
    )
    unsupported = _coaching(metrics=_metrics())

    supported_drop = next(
        item for item in supported.intervention_candidates if item.drill_id == "drop_the_landing_v1"
    )
    unsupported_drop = next(
        item for item in unsupported.intervention_candidates if item.drill_id == "drop_the_landing_v1"
    )
    assert supported_drop.confidence > unsupported_drop.confidence
    assert supported_drop.supporting_evidence_ids


def test_dependency_graph_references_valid_drills_and_has_no_self_edges():
    coaching = _coaching()
    drill_ids = {item.drill_id for item in coaching.drill_library}

    assert coaching.dependency_graph
    for edge in coaching.dependency_graph:
        assert edge.before in drill_ids
        assert edge.after in drill_ids
        assert edge.before != edge.after


def test_report_uses_deterministic_coaching_engine_for_drill_selection():
    scores = _softened_expert_scores()
    metrics = _metrics(
        raw={"terminal_rising_ratio": 0.65},
        linguistic={"closing_strength_score": 0.25, "hedges_per_100_words": 4.0},
    )
    audio_quality = AudioQuality(usable=True, background_noise_level="low")
    uncertainty = Uncertainty(overall_confidence_label="medium_high", reasons=[])
    inference = _infer(metrics, audio_quality=audio_quality)
    diagnostic = _diagnostic(
        scores=scores,
        metrics=metrics,
        audio_quality=audio_quality,
        uncertainty=uncertainty,
        inference=inference,
    )
    report = build_report(
        scores=scores,
        metrics=metrics,
        psychological_inference=inference,
        diagnostic_reasoning=diagnostic,
        evidence=_evidence(),
        moments=_moments(),
        uncertainty=uncertainty,
        audio_quality=audio_quality,
        duration_ms=60000,
        scenario="benchmark",
        transcript=_test_transcript(duration_ms=60000),
    )
    coaching = _coaching(
        scores=scores,
        metrics=metrics,
        audio_quality=audio_quality,
        uncertainty=uncertainty,
        inference=inference,
        diagnostic_reasoning=diagnostic,
        report=report,
    )
    updated = apply_coaching_to_report(report, coaching)

    assert coaching.selected_interventions.primary_drill is not None
    assert updated.coaching_engine == coaching
    assert updated.training_prescription.drill_id == coaching.selected_interventions.primary_drill.drill_id
    assert updated.highest_leverage_fix.first_drill_id == coaching.selected_interventions.primary_drill.drill_id


def test_legacy_recommendations_are_human_safe_and_keep_engine_reasoning():
    coaching = _coaching(
        metrics=_metrics(
            raw={"terminal_rising_ratio": 0.65},
            linguistic={"closing_strength_score": 0.25, "hedges_per_100_words": 4.0},
        )
    )
    recommendations = _deterministic_recommendations(coaching)

    unsafe = " ".join(
        [
            recommendations.highest_leverage_issue,
            recommendations.fastest_improvement_tip,
            recommendations.coaching_summary,
        ]
    )
    assert "_v1" not in unsafe
    assert "highest_weighted_intervention_score" not in unsafe
    assert recommendations.highest_leverage_issue != coaching.selected_interventions.primary_drill.drill_id
    assert recommendations.fastest_improvement_tip.endswith(".")
    assert coaching.selected_interventions.primary_drill.drill_id.endswith("_v1")
    assert coaching.selected_interventions.primary_drill.score > 0


def test_legacy_drills_have_safe_titles_instructions_and_metrics():
    coaching = _coaching(
        metrics=_metrics(
            raw={"terminal_rising_ratio": 0.65},
            linguistic={"closing_strength_score": 0.25, "hedges_per_100_words": 4.0},
        )
    )
    drills = _deterministic_drills(coaching)

    assert drills
    for drill in drills:
        visible = " ".join(
            [drill.drill_id, drill.title, drill.goal, *drill.instructions, *drill.target_metrics]
        )
        assert "_v1" not in visible
        assert "highest_weighted" not in visible
        assert "approval seeking" not in visible.lower()
        assert "nervous" not in visible.lower()
        assert drill.title
        assert drill.instructions
        assert drill.duration_min > 0
