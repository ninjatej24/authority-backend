"""Milestone 7 premium report generation tests."""

from __future__ import annotations

import re

from schemas import AudioQuality, Moment, Uncertainty
from services.deterministic_coaching import build_deterministic_coaching
from services.report_builder import build_report
from services.report_generation import (
    _behaviour_observations,
    _diagnosis_observations,
    _select_behaviour_diagnosis,
)
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

INTERNAL_COPY_MARKERS = {
    "winning diagnosis",
    "evidence item",
    "hypothesis",
    "observed as",
    "deterministic",
    "selected focus",
    "attached to this issue",
    "raw_acoustic.",
    "linguistic.",
    "rhythm.",
    "derived.",
    "vad.",
    "backend",
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


def _user_facing_copy(report) -> list[str]:
    reads = report.perception_map.model_dump().values()
    return [
        report.mirror.headline,
        report.mirror.identity_read,
        report.diagnosis.core_pattern,
        report.diagnosis.social_consequence,
        *(read["text"] for read in reads if read),
        report.hidden_cost.consequence,
        report.highest_leverage_fix.plain_english,
        report.highest_leverage_fix.why_this_matters,
        report.highest_leverage_fix.action_step,
        report.training_prescription.why_chosen,
        report.training_prescription.success_signal,
        *(report.training_prescription.instructions or []),
        *(item.signal for item in report.evidence_chain),
        *(item.what_happened for item in report.evidence_chain),
        *(item.listener_interpretation for item in report.evidence_chain),
        *(item.why_it_matters for item in report.evidence_chain),
        *(item.summary for item in report.timeline),
        *(item.listener_interpretation for item in report.timeline),
    ]


def _report_user_facing_strings(report) -> list[str]:
    sections = {
        "mirror": report.mirror.model_dump(),
        "diagnosis": report.diagnosis.model_dump(),
        "perception_map": report.perception_map.model_dump(),
        "evidence_chain": [item.model_dump() for item in report.evidence_chain],
        "timeline": [item.model_dump() for item in report.timeline],
        "hidden_cost": report.hidden_cost.model_dump(),
        "highest_leverage_fix": report.highest_leverage_fix.model_dump(),
        "training_prescription": report.training_prescription.model_dump(),
        "retest_plan": report.retest_plan.model_dump(),
    }
    ignored_keys = {
        "id",
        "evidence_id",
        "evidence_ids",
        "supporting_evidence_ids",
        "linked_evidence",
        "moment_id",
        "type",
        "moment_group",
        "severity",
        "word_ids",
        "drill_id",
        "first_drill_id",
        "cost_id",
        "source_metrics",
        "supporting_metrics",
        "start_ms",
        "end_ms",
        "timestamp",
        "confidence",
        "selection_score",
        "duration_min",
        "recommended_retest_after_days",
    }
    strings: list[str] = []

    def walk(value, key: str = ""):
        if key in ignored_keys:
            return
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, child_key)
        elif isinstance(value, list):
            for item in value:
                walk(item, key)

    walk(sections)
    return strings


def test_premium_report_contains_all_required_sections_and_validation():
    report = _generated_report()

    assert report.mirror.headline
    assert report.mirror.one_line_identity_read
    assert report.diagnosis.primary_strength_dimension
    assert report.perception_map.first_impression.evidence_ids
    assert report.evidence_chain
    assert len(report.evidence_chain) <= 3
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


def test_user_facing_report_sections_have_no_raw_or_internal_language():
    report = _generated_report()
    strings = _report_user_facing_strings(report)
    copy = " ".join(strings).lower()

    for marker in INTERNAL_COPY_MARKERS:
        assert marker not in copy
    assert not re.search(r"\b[a-z]+_[a-z0-9_]+\b", copy)


def test_empty_or_no_speech_sample_returns_insufficient_report():
    metrics = _metrics(vad={"speech_ratio": 0.0, "total_speech_duration_ms": 0}, linguistic={"filler_words_per_min": 0.0})
    scores = _scores().model_copy(update={"score_confidence": 0.2})
    audio_quality = AudioQuality(usable=False, background_noise_level="unknown", quality_warnings=["No usable speech detected"])
    report = _generated_report(scores=scores, metrics=metrics, audio_quality=audio_quality, duration_ms=720)

    assert "not enough usable speech" in report.mirror.headline.lower()
    assert report.authority_type.label == "Insufficient Sample"
    assert report.primary_diagnosis is None
    assert report.diagnosis.primary_strength_dimension is None
    assert report.diagnosis.primary_limiting_dimension is None
    assert len(report.evidence_chain) == 1
    assert report.evidence_chain[0].related_dimension == "Sample quality"
    assert report.timeline == []
    assert "30 to 60 second" in report.training_prescription.why_chosen.lower() or "30 to 60 second" in " ".join(report.training_prescription.instructions).lower()
    assert report.share_card.percentile_label is None


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

    assert 1 <= len(report.evidence_chain) <= 3
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


def test_major_sections_trace_to_observable_behaviour():
    report = _generated_report()
    diagnosis = report.diagnosis.core_pattern.lower().rstrip(".")

    major_copy = " ".join(
        value or ""
        for value in [
            report.mirror.headline,
            report.mirror.identity_read,
            report.diagnosis.core_pattern,
            report.perception_map.first_impression.text,
            report.highest_leverage_fix.plain_english,
            report.training_prescription.why_chosen,
        ]
    ).lower()
    assert diagnosis in major_copy


def test_evidence_signals_are_observations_not_template_labels():
    report = _generated_report()
    forbidden_labels = {
        "Pace pressure",
        "Controlled pacing",
        "Filler burden",
        "Low filler control",
        "Hesitation clustering",
        "Pause ownership",
        "Weak closing",
        "Strong opening",
        "Rising endings",
        "Dynamic emphasis",
        "Monotony",
        "Low specificity",
        "Strong specificity",
        "Weak structure",
        "Strong structure",
    }

    assert all(item.signal not in forbidden_labels for item in report.evidence_chain)
    assert all(len(item.signal.split()) >= 5 for item in report.evidence_chain)


def test_user_facing_copy_has_no_internal_ids_or_generic_placeholders():
    report = _generated_report()
    copy = " ".join(value or "" for value in _user_facing_copy(report)).lower()

    forbidden = [
        "weak_finality_reduces_perceived_leadership",
        "winning diagnosis",
        "supported by 1 evidence item",
        "supported by 2 evidence item",
        "supported by 3 evidence item",
        "supported by 4 evidence item",
        "supported by 5 evidence item",
        "contradicted by",
        "selected focus",
        "target area",
        "deterministic",
        "metric itself",
        "dimension label",
    ]
    for phrase in forbidden:
        assert phrase not in copy


def test_report_organizes_around_one_dominant_diagnosis():
    report = _generated_report()
    diagnosis = report.diagnosis.core_pattern.lower().rstrip(".")

    assert diagnosis
    if report.primary_diagnosis is None:
        assert report.mirror.confidence_label in {"low", "medium"}
        assert report.highest_leverage_fix.evidence_ids
        assert len(report.evidence_chain) <= 3
    else:
        assert diagnosis in report.mirror.headline.lower()
        assert diagnosis in report.mirror.identity_read.lower()
        assert diagnosis in report.highest_leverage_fix.plain_english.lower()
        assert diagnosis in report.training_prescription.why_chosen.lower()


def test_selected_diagnosis_is_stable_after_evidence_trimming():
    scores = _softened_expert_scores()
    metrics = _metrics()
    audio_quality = AudioQuality(usable=True, background_noise_level="low")
    uncertainty = Uncertainty(overall_confidence_label="medium_high", reasons=[])
    duration_ms = 60000
    evidence = _evidence()
    moments = _moments()
    inference = _infer(metrics, audio_quality=audio_quality, duration_ms=duration_ms)
    diagnostic = _diagnostic(
        scores=scores,
        metrics=metrics,
        audio_quality=audio_quality,
        uncertainty=uncertainty,
        duration_ms=duration_ms,
        scenario="benchmark",
        inference=inference,
        evidence=evidence,
        moments=moments,
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
        scenario="benchmark",
    )
    confidence = min(inference.overall_inference_confidence, scores.score_confidence or 0.0)
    observations = _behaviour_observations(evidence, inference, diagnostic, coaching, moments, confidence, duration_ms, audio_quality)
    diagnosis = _select_behaviour_diagnosis(observations, confidence, duration_ms, audio_quality, coaching)
    trimmed = _diagnosis_observations(observations, diagnosis)

    if diagnosis is None:
        assert len(trimmed) <= 3
    else:
        assert _select_behaviour_diagnosis(trimmed, confidence, duration_ms, audio_quality, coaching).id == diagnosis.id


def test_evidence_and_timeline_are_filtered_to_winning_story():
    report = _generated_report()
    dimensions = {item.related_dimension for item in report.evidence_chain}

    assert 1 <= len(dimensions) <= 3
    assert len(report.evidence_chain) <= 3
    assert all(item.evidence_ids for item in report.timeline)
    assert all(set(item.evidence_ids).issubset({card.evidence_id for card in report.evidence_chain}) for item in report.timeline)


def test_timeline_suppresses_invalid_and_generic_moments():
    moments = [
        Moment(
            moment_id="bad_time",
            type="strongest_moment",
            start_ms=5000,
            end_ms=4000,
            headline="Invalid timing",
            summary="This should not appear.",
            confidence=0.9,
            severity="medium",
        ),
        Moment(
            moment_id="generic",
            type="generic",
            start_ms=1000,
            end_ms=3000,
            headline="Generic",
            summary="This should not appear.",
            confidence=0.9,
            severity="medium",
        ),
        Moment(
            moment_id="valid",
            type="strongest_moment",
            start_ms=1000,
            end_ms=4000,
            headline="Most controlled stretch",
            summary="The delivery had a clear shape here.",
            confidence=0.8,
            severity="medium",
            timestamp_source="real",
            importance_score=0.8,
            supporting_evidence_ids=["psi_ev_structure_high"],
        ),
    ]
    report = _generated_report(moments=moments)

    assert [item.moment_id for item in report.timeline] == ["valid"]


def test_evidence_cards_support_report_diagnosis_or_clarify_uncertainty():
    report = _generated_report()
    diagnosis_ids = set(report.diagnosis.evidence_ids)

    assert diagnosis_ids
    assert diagnosis_ids.issubset({item.evidence_id for item in report.evidence_chain})
    assert all(item.evidence_id in diagnosis_ids or item.direction != "negative" for item in report.evidence_chain)


def test_diagnostic_reasoning_and_report_diagnosis_do_not_conflict():
    report = _generated_report()

    if report.primary_diagnosis is None:
        assert report.mirror.confidence_label in {"low", "medium"}
    else:
        assert report.primary_diagnosis.diagnosis_name == report.diagnosis.core_pattern
        assert set(report.primary_diagnosis.supporting_evidence_ids).issubset({item.evidence_id for item in report.evidence_chain})


def test_contradictions_reduce_confidence_instead_of_becoming_extra_unrelated_cards():
    metrics = _metrics(
        linguistic={"filler_words_per_min": 0.0, "opening_strength_score": 0.86, "structure_score": 0.82},
        rhythm={"speed_up_segments": 3, "burst_speaking_segments": 2, "rhythm_consistency": 0.35},
        raw={"words_per_minute": 195.0},
    )
    report = _generated_report(metrics=metrics)

    assert report.mirror.confidence_label in {"low", "medium", "medium_high"}
    assert len({item.related_dimension for item in report.evidence_chain}) <= 3
    assert "filler burden" not in str([item.model_dump() for item in report.evidence_chain]).lower()


def test_short_recording_suppresses_timeline_precision():
    report = _generated_report(duration_ms=9000)

    assert report.timeline == []


def test_hidden_cost_fix_and_training_derive_from_selected_diagnosis():
    report = _generated_report()
    pattern = report.diagnosis.core_pattern.lower().rstrip(".")

    if report.primary_diagnosis is not None:
        assert pattern in report.hidden_cost.consequence.lower()
        assert pattern in report.highest_leverage_fix.plain_english.lower()
        assert pattern in report.training_prescription.why_chosen.lower()
    else:
        assert report.highest_leverage_fix.evidence_ids
        assert report.training_prescription.evidence_ids
    assert report.training_prescription.instructions
    assert report.training_prescription.action_step


def test_competing_hypotheses_reduce_certainty_instead_of_disappearing():
    report = _generated_report()

    if any("secondary:" in reason for reason in report.uncertainty.reasons):
        assert report.mirror.confidence_label in {"medium", "medium_high", "low"}
        assert "secondary explanation" in report.mirror.identity_read.lower()
