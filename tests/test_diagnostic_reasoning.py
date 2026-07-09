"""Milestone 5.1 deterministic diagnostic reasoning tests."""

from __future__ import annotations

import pytest

from schemas import AudioQuality, Uncertainty
from services.diagnostic_reasoning import build_diagnostic_reasoning
from services.report_builder import build_report
from tests.test_psychological_inference import _infer, _metrics, _scores
from tests.test_report_builder import _evidence, _moments


def _diagnostic(**kwargs):
    scores = kwargs.pop("scores", _scores())
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
    return build_diagnostic_reasoning(
        metrics=metrics,
        psychological_inference=inference,
        evidence=kwargs.pop("evidence", _evidence()),
        moments=kwargs.pop("moments", _moments()),
        scores=scores,
        audio_quality=audio_quality,
        uncertainty=uncertainty,
        duration_ms=duration_ms,
        scenario=scenario,
    )


def _softened_expert_scores():
    return _scores().model_copy(
        update={
            "dimension_scores": _scores().dimension_scores.model_copy(
                update={
                    "command": 42,
                    "clarity": 76,
                    "structure": 74,
                    "composure": 63,
                    "presence": 62,
                    "persuasion": 61,
                }
            )
        }
    )


def test_primary_diagnosis_traces_to_evidence_and_moments():
    reasoning = _diagnostic(scores=_softened_expert_scores())

    assert reasoning.primary_diagnosis is not None
    assert reasoning.primary_diagnosis.diagnosis_id == "softened_expert"
    assert reasoning.primary_diagnosis.supporting_evidence_ids
    assert reasoning.primary_diagnosis.supporting_moment_ids
    assert reasoning.primary_diagnosis.affected_dimensions


def test_hidden_cost_always_references_supporting_evidence():
    reasoning = _diagnostic(scores=_softened_expert_scores())

    assert reasoning.hidden_cost_reasoning is not None
    assert reasoning.hidden_cost_reasoning.evidence_ids
    assert reasoning.hidden_cost_reasoning.source_signal
    assert reasoning.hidden_cost_reasoning.interpretation
    assert reasoning.hidden_cost_reasoning.consequence
    assert reasoning.hidden_cost_reasoning.listener_effect


def test_highest_leverage_fix_exposes_weighted_formula():
    reasoning = _diagnostic(scores=_softened_expert_scores())
    fix = reasoning.highest_leverage_reasoning

    assert fix is not None
    expected = (
        fix.severity
        * fix.authority_impact
        * fix.trainability
        * fix.evidence_confidence
        * fix.scenario_relevance
    )
    assert fix.issue_id == "declarative_finality"
    assert fix.recommended_first_drill == "drop_the_landing_v1"
    assert fix.selection_score == pytest.approx(round(expected, 3))
    assert fix.supporting_evidence


def test_contradictions_are_deterministic_and_evidence_backed():
    scores = _softened_expert_scores()
    first = _diagnostic(scores=scores)
    second = _diagnostic(scores=scores)

    assert first.contradictions
    assert [item.model_dump() for item in first.contradictions] == [
        item.model_dump() for item in second.contradictions
    ]
    clarity_command = next(
        item for item in first.contradictions if item.contradiction_id == "clarity_low_command"
    )
    assert clarity_command.strength == "Clarity"
    assert clarity_command.limiter == "Command"
    assert clarity_command.evidence_ids


def test_poor_audio_suppresses_unsupported_reasoning():
    audio_quality = AudioQuality(
        usable=False,
        background_noise_level="high",
        quality_warnings=["Very low signal level"],
    )
    reasoning = _diagnostic(
        scores=_softened_expert_scores(),
        audio_quality=audio_quality,
        duration_ms=9000,
        uncertainty=Uncertainty(overall_confidence_label="low", reasons=["Poor microphone signal"]),
    )

    assert reasoning.suppressed_diagnoses
    assert "Poor audio suppresses unsupported diagnostic reasoning" in reasoning.uncertainty.reasons
    assert "Short recording suppresses low-confidence diagnoses" in reasoning.uncertainty.reasons


def test_suppressed_traits_do_not_support_primary_diagnosis():
    metrics = _metrics(
        linguistic={"filler_words_per_min": 14.0},
        rhythm={"speed_up_segments": 0, "burst_speaking_segments": 0},
        derived={"hesitation_cluster_score": 0.1},
    )
    inference = _infer(metrics)
    nervous = next(trait for trait in inference.traits if trait.trait_id == "nervous")
    assert nervous.suppress_from_report is True

    reasoning = _diagnostic(metrics=metrics, inference=inference)
    if reasoning.primary_diagnosis is not None:
        assert "nervous" not in reasoning.primary_diagnosis.supporting_traits


def test_report_generation_maps_reasoning_without_gpt_decisions():
    scores = _softened_expert_scores()
    metrics = _metrics()
    audio_quality = AudioQuality(usable=True, background_noise_level="low")
    uncertainty = Uncertainty(overall_confidence_label="medium_high", reasons=[])
    inference = _infer(metrics, audio_quality=audio_quality)
    reasoning = _diagnostic(
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
        diagnostic_reasoning=reasoning,
        evidence=_evidence(),
        moments=_moments(),
        uncertainty=uncertainty,
        audio_quality=audio_quality,
        duration_ms=60000,
        scenario="benchmark",
    )

    assert report.primary_diagnosis == reasoning.primary_diagnosis
    assert report.diagnosis.core_behavioural_pattern
    assert report.diagnosis.core_behavioural_pattern != reasoning.primary_diagnosis.diagnosis_id
    pattern = report.diagnosis.core_behavioural_pattern.lower().rstrip(".")
    assert pattern in report.mirror.headline.lower()
    assert pattern in report.highest_leverage_fix.plain_english.lower()
    assert report.hidden_cost_reasoning == reasoning.hidden_cost_reasoning
    assert report.highest_leverage_reasoning == reasoning.highest_leverage_reasoning
