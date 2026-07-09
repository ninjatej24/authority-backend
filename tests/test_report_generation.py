"""Milestone 7 premium report generation tests."""

from __future__ import annotations

from schemas import AudioQuality, Uncertainty
from services.deterministic_coaching import build_deterministic_coaching
from services.report_builder import build_report
from tests.test_diagnostic_reasoning import _diagnostic, _softened_expert_scores
from tests.test_psychological_inference import _infer, _metrics, _scores
from tests.test_report_builder import _evidence, _moments


RAW_MAIN_MARKERS = {
    "raw_acoustic",
    "linguistic.filler_words_per_min",
    "derived.hesitation_cluster_score",
    "rhythm.burst_speaking_segments",
    "rhythm.speed_up_segments",
    "rhythm.rhythm_consistency",
    "filler_words_per_min",
    "burst_speaking_segments",
    "speed_up_segments",
    "hesitation_cluster_score",
    "words_per_minute",
}


def _generated_report(**kwargs):
    scores = kwargs.pop("scores", _softened_expert_scores())
    metrics = kwargs.pop("metrics", _metrics())
    audio_quality = kwargs.pop("audio_quality", AudioQuality(usable=True, background_noise_level="low"))
    uncertainty = kwargs.pop("uncertainty", Uncertainty(overall_confidence_label="medium_high", reasons=[]))
    duration_ms = kwargs.pop("duration_ms", 60000)
    scenario = kwargs.pop("scenario", "benchmark")
    evidence = kwargs.pop("evidence", _evidence())
    moments = kwargs.pop("moments", _moments())
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
            evidence=evidence,
            moments=moments,
        ),
    )
    coaching = build_deterministic_coaching(
        metrics=metrics,
        scores=scores,
        psychological_inference=inference,
        diagnostic_reasoning=diagnostic,
        report=None,
        audio_quality=audio_quality,
        uncertainty=uncertainty,
        duration_ms=duration_ms,
        scenario=scenario,
    )
    return build_report(
        scores=scores,
        metrics=metrics,
        psychological_inference=inference,
        diagnostic_reasoning=diagnostic,
        coaching_engine=coaching,
        evidence=evidence,
        moments=moments,
        uncertainty=uncertainty,
        audio_quality=audio_quality,
        duration_ms=duration_ms,
        scenario=scenario,
    )


def _main_report_text(report) -> str:
    payload = report.model_dump(
        exclude={
            "technical_appendix",
            "diagnostic_reasoning",
            "primary_diagnosis",
            "secondary_diagnosis",
            "hidden_cost_reasoning",
            "dimension_reasoning",
            "trait_reasoning",
            "highest_leverage_reasoning",
            "coaching_engine",
            "validation",
            "explainability",
            "progress",
        }
    )
    return str(payload)


def test_premium_report_contains_all_required_sections_and_validation():
    report = _generated_report()

    assert report.mirror.headline
    assert report.mirror.one_line_identity_read
    assert report.diagnosis.primary_strength_dimension
    assert report.perception_map.first_impression.evidence_ids
    assert report.evidence_chain
    assert report.timeline
    assert set(report.dimension_reports) == {"command", "clarity", "composure", "presence", "persuasion", "structure"}
    assert report.hidden_cost.consequence
    assert report.highest_leverage_fix.first_drill_id
    assert report.training_prescription.drill_id
    assert report.retest_plan.same_prompt_recommended is True
    assert report.technical_appendix.metrics
    assert report.share_card.share_safety == "public_safe"
    assert report.validation.valid is True
    assert report.validation.orphan_links == []


def test_evidence_and_timeline_references_are_valid_and_not_invented():
    report = _generated_report()
    evidence_ids = {item.evidence_id for item in report.evidence_chain}
    moment_ids = {item.moment_id for item in report.timeline}

    assert report.diagnosis.evidence_ids
    assert set(report.diagnosis.evidence_ids).issubset(evidence_ids)
    for item in report.timeline:
        assert item.moment_id in moment_ids
        assert set(item.evidence_ids).issubset(evidence_ids)
        assert item.listener_interpretation
        assert 0.0 <= item.confidence <= 1.0


def test_dimension_reports_are_generated_from_existing_reasoning():
    report = _generated_report()

    for dimension, dimension_report in report.dimension_reports.items():
        assert dimension_report.score == getattr(_softened_expert_scores().dimension_scores, dimension)
        assert dimension_report.meaning
        assert dimension_report.listener_consequence
        assert dimension_report.one_improvement_cue
        assert dimension_report.linked_evidence


def test_authority_type_is_deterministic_from_dimension_profile():
    executive_scores = _scores().model_copy(
        update={
            "authority_score": 91,
            "dimension_scores": _scores().dimension_scores.model_copy(
                update={
                    "command": 86,
                    "clarity": 84,
                    "composure": 85,
                    "presence": 83,
                    "persuasion": 78,
                    "structure": 80,
                }
            ),
        }
    )
    first = _generated_report(scores=executive_scores).authority_type.model_dump()
    second = _generated_report(scores=executive_scores).authority_type.model_dump()

    assert first == second
    assert first["label"] == "Executive Presence"


def test_report_does_not_invent_scores_or_expose_private_share_findings():
    scores = _softened_expert_scores()
    report = _generated_report(scores=scores)

    assert report.share_card.authority_score == scores.authority_score
    assert report.share_card.hidden_private_findings == []
    public_text = " ".join(
        str(value)
        for value in (
            report.share_card.authority_type,
            report.share_card.top_strength,
            report.share_card.growth_area,
            report.share_card.one_line_identity_read,
        )
    ).lower()
    assert "approval seeking" not in public_text
    assert "nervous" not in public_text


def test_report_consumes_v2_scoring_metadata():
    report = _generated_report()
    appendix = report.technical_appendix.score_components

    assert "calibration_metadata" in appendix
    assert "fairness_adjustments" in appendix
    assert "score_rarity_label" in appendix


def test_main_report_copy_does_not_expose_raw_metric_keys():
    report = _generated_report()
    main_text = _main_report_text(report)

    for marker in RAW_MAIN_MARKERS:
        assert marker not in main_text


def test_zero_filler_does_not_produce_high_filler_warning():
    metrics = _metrics(linguistic={"filler_words_per_min": 0.0})
    report = _generated_report(metrics=metrics)

    evidence_ids = {item.id for item in report.evidence_chain}
    evidence_text = str([item.model_dump() for item in report.evidence_chain]).lower()
    assert "filler_burden" not in evidence_ids
    assert "filler burden" not in evidence_text
    assert "interrupt the sense of clean thought control" not in evidence_text


def test_zero_burst_segments_do_not_produce_burst_or_pressure_warning():
    metrics = _metrics(
        raw={"words_per_minute": 190.0},
        rhythm={"speed_up_segments": 0, "burst_speaking_segments": 0, "rhythm_consistency": 0.78},
    )
    report = _generated_report(metrics=metrics)

    evidence_ids = {item.id for item in report.evidence_chain}
    evidence_text = _main_report_text(report).lower()
    assert "pace_pressure" not in evidence_ids
    assert "burst speaking" not in evidence_text
    assert "pace pressure" not in evidence_text


def test_short_recording_suppresses_strong_psychological_claims():
    metrics = _metrics(
        linguistic={"filler_words_per_min": 14.0},
        rhythm={"speed_up_segments": 3, "burst_speaking_segments": 2},
        derived={"hesitation_cluster_score": 0.85},
        raw={"words_per_minute": 195.0},
    )
    report = _generated_report(metrics=metrics, duration_ms=9000)

    assert report.mirror.confidence_label == "low"
    assert "not enough reliable evidence" in report.mirror.headline.lower()
    assert all(item.direction != "negative" for item in report.evidence_chain)


def test_valid_report_evidence_connects_signal_interpretation_consequence_and_fix():
    report = _generated_report()

    assert 3 <= len(report.evidence_chain) <= 5
    for item in report.evidence_chain:
        assert item.signal
        assert item.what_happened
        assert item.listener_interpretation
        assert item.why_it_matters
        assert "Fix:" in item.why_it_matters
        assert item.related_dimension


def test_technical_appendix_keeps_raw_metric_style_values():
    report = _generated_report()
    appendix_metrics = report.technical_appendix.metrics

    assert "words_per_minute" in appendix_metrics
    assert "filler_words_per_min" in appendix_metrics
    assert "pause_frequency_per_minute" in appendix_metrics
    assert "avg_pause_duration_ms" in appendix_metrics
