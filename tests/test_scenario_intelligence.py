"""Milestone 9 deterministic scenario intelligence tests."""

from __future__ import annotations

from schemas import AudioQuality, Uncertainty
from services.deterministic_coaching import build_deterministic_coaching
from services.scenario_profiles import (
    SCENARIO_PROFILES,
    apply_scenario_weights,
    calculate_metric_relevance,
    get_scenario_profile,
    validate_scenario_profile,
)
from services.scoring_engine import DIMENSION_WEIGHTS, compute_authority_score
from tests.test_diagnostic_reasoning import _diagnostic, _softened_expert_scores
from tests.test_psychological_inference import _infer, _metrics
from tests.test_report_builder import _evidence, _moments
from tests.test_report_generation import _generated_report
from tests.test_scoring_engine_v2 import _acoustic, _cognitive, _delivery, _linguistic, _voice


def _scenario_score(scenario: str):
    return compute_authority_score(
        _voice(),
        _cognitive(),
        _delivery(),
        _linguistic(),
        acoustic=_acoustic(),
        asr_confidence=0.88,
        duration_ms=45000,
        scenario=scenario,
    ).scores


def test_scenario_profiles_are_valid_and_benchmark_is_baseline():
    assert set(SCENARIO_PROFILES) >= {
        "benchmark",
        "interview",
        "leadership",
        "sales",
        "founder_pitch",
        "presentation",
        "meeting",
        "podcast",
    }
    for profile in SCENARIO_PROFILES.values():
        validate_scenario_profile(profile)

    assert apply_scenario_weights(DIMENSION_WEIGHTS, "benchmark") == DIMENSION_WEIGHTS
    assert get_scenario_profile("missing").scenario_id == "benchmark"


def test_same_recording_has_scenario_specific_score_composition():
    benchmark = _scenario_score("benchmark")
    interview = _scenario_score("interview")
    sales = _scenario_score("sales")
    leadership = _scenario_score("leadership")

    assert benchmark.dimension_scores == interview.dimension_scores == sales.dimension_scores == leadership.dimension_scores
    assert benchmark.score_components.weighted_base != interview.score_components.weighted_base
    assert sales.score_components.weighted_base != leadership.score_components.weighted_base
    assert interview.scenario_adjustments.major_weight_changes
    assert sales.scenario_used == "sales"


def test_metric_relevance_matches_scenario_examples():
    assert calculate_metric_relevance("rambling", "interview") > calculate_metric_relevance("rambling", "benchmark")
    assert calculate_metric_relevance("dynamic_emphasis", "sales") > calculate_metric_relevance("dynamic_emphasis", "benchmark")
    assert calculate_metric_relevance("terminal_endings", "leadership") > calculate_metric_relevance("terminal_endings", "benchmark")


def test_scenario_weighting_changes_coaching_candidate_relevance_without_changing_metrics():
    metrics = _metrics(
        raw={"terminal_rising_ratio": 0.65},
        linguistic={"closing_strength_score": 0.25, "hedges_per_100_words": 4.0},
    )
    audio_quality = AudioQuality(usable=True, background_noise_level="low")
    uncertainty = Uncertainty(overall_confidence_label="medium_high", reasons=[])
    scores = _softened_expert_scores()
    inference = _infer(metrics, audio_quality=audio_quality, duration_ms=60000)
    evidence = _evidence()
    moments = _moments()
    metrics_before = metrics.model_dump()
    evidence_before = [item.model_dump() for item in evidence]

    diagnostic = _diagnostic(
        scores=scores,
        metrics=metrics,
        audio_quality=audio_quality,
        uncertainty=uncertainty,
        duration_ms=60000,
        scenario="benchmark",
        inference=inference,
        evidence=evidence,
        moments=moments,
    )
    benchmark = build_deterministic_coaching(
        metrics=metrics,
        scores=scores,
        psychological_inference=inference,
        diagnostic_reasoning=diagnostic,
        report=None,
        audio_quality=audio_quality,
        uncertainty=uncertainty,
        duration_ms=60000,
        scenario="benchmark",
    )
    leadership = build_deterministic_coaching(
        metrics=metrics,
        scores=scores,
        psychological_inference=inference,
        diagnostic_reasoning=diagnostic,
        report=None,
        audio_quality=audio_quality,
        uncertainty=uncertainty,
        duration_ms=60000,
        scenario="leadership",
    )

    assert metrics.model_dump() == metrics_before
    assert [item.model_dump() for item in evidence] == evidence_before
    bench_drop = next(item for item in benchmark.intervention_candidates if item.drill_id == "drop_the_landing_v1")
    leader_drop = next(item for item in leadership.intervention_candidates if item.drill_id == "drop_the_landing_v1")
    assert leader_drop.scenario_relevance > bench_drop.scenario_relevance
    assert leader_drop.score > bench_drop.score


def test_report_contains_scenario_summary_and_emphasis():
    report = _generated_report(scenario="interview")

    assert report.scenario_summary is not None
    assert report.scenario_summary.scenario_id == "interview"
    assert report.scenario_summary.why_dimensions_changed
    assert "interview" in (report.perception_map.interview_read.text or "").lower()
    assert "scenario_adjustments" in report.technical_appendix.score_components


def test_founder_presentation_meeting_podcast_profiles_are_distinct():
    founder = _scenario_score("founder_pitch")
    presentation = _scenario_score("presentation")
    meeting = _scenario_score("meeting")
    podcast = _scenario_score("podcast")

    assert founder.score_components.weighted_base != presentation.score_components.weighted_base
    assert meeting.score_components.weighted_base != podcast.score_components.weighted_base
    assert founder.scenario_adjustments.dimension_adjustments != presentation.scenario_adjustments.dimension_adjustments
    assert meeting.scenario_adjustments.dimension_adjustments != podcast.scenario_adjustments.dimension_adjustments
