"""Milestone 4 deterministic psychological inference tests."""

from __future__ import annotations

from schemas import (
    ArticulationMetrics,
    AudioQuality,
    DerivedAxes,
    DerivedMetrics,
    DimensionScores,
    LinguisticMetrics,
    Metrics,
    RawAcousticMetrics,
    RhythmMetrics,
    ScoreBonuses,
    ScoreComponents,
    ScorePenalties,
    Scores,
    VADMetrics,
    Uncertainty,
)
from services.psychological_inference import build_psychological_inference


def _scores() -> Scores:
    return Scores(
        authority_score=64,
        authority_percentile_estimate=None,
        score_confidence=0.79,
        dimension_scores=DimensionScores(
            command=62,
            clarity=64,
            composure=62,
            presence=62,
            persuasion=62,
            structure=62,
        ),
        derived_axes=DerivedAxes(
            trust_warmth=60,
            dominance_status=60,
            nervousness=40,
            interview_readiness=62,
            leadership_readiness=62,
        ),
        score_components=ScoreComponents(
            weighted_base=64,
            bonuses=ScoreBonuses(),
            penalties=ScorePenalties(),
        ),
    )


def _metrics(**overrides) -> Metrics:
    raw = {
        "words_per_minute": 145.0,
        "pause_frequency_per_min": 8.0,
        "avg_pause_ms": 420.0,
        "longest_pause_ms": 900.0,
        "mid_phrase_pause_rate": 0.12,
        "f0_range_semitones": 6.0,
        "f0_variability_semitones": 2.0,
        "loudness_variation_db": 5.5,
        "terminal_rising_ratio": 0.1,
        "terminal_falling_ratio": 0.5,
    }
    linguistic = {
        "filler_words_per_min": 1.0,
        "hedges_per_100_words": 0.5,
        "certainty_markers_per_100_words": 2.5,
        "self_doubt_markers": 0,
        "repetition_rate": 0.1,
        "specificity_score": 0.6,
        "concreteness_score": 0.5,
        "rambling_score": 0.1,
        "opening_strength_score": 0.8,
        "closing_strength_score": 0.8,
        "structure_score": 0.78,
    }
    derived = {
        "monotony_index": 0.1,
        "hesitation_cluster_score": 0.1,
        "dynamic_emphasis_score": 0.7,
        "speech_continuity_score": 0.76,
        "confidence_drop_count": 0,
        "vocal_command_index": 0.72,
        "composure_index": 0.72,
        "rhythm_index": 0.72,
        "projection_index": 0.7,
        "authority_signal_index": 0.74,
    }
    rhythm = {
        "words_per_minute": 145.0,
        "rhythm_consistency": 0.78,
        "hesitation_windows": 0,
        "burst_speaking_segments": 0,
        "speed_up_segments": 0,
        "slow_down_segments": 0,
    }
    articulation = {
        "clarity_proxy": 0.78,
        "articulation_stability": 0.75,
    }
    vad = {
        "speech_ratio": 0.75,
        "total_speech_duration_ms": 45000,
        "total_silence_duration_ms": 15000,
        "avg_pause_duration_ms": 420.0,
        "pause_frequency_per_minute": 9.0,
        "vad_backend": "webrtc",
    }

    for section, values in overrides.items():
        if section == "raw":
            raw.update(values)
        elif section == "linguistic":
            linguistic.update(values)
        elif section == "derived":
            derived.update(values)
        elif section == "rhythm":
            rhythm.update(values)
        elif section == "articulation":
            articulation.update(values)
        elif section == "vad":
            vad.update(values)

    return Metrics(
        raw_acoustic=RawAcousticMetrics(**raw),
        linguistic=LinguisticMetrics(**linguistic),
        derived=DerivedMetrics(**derived),
        rhythm=RhythmMetrics(**rhythm),
        articulation=ArticulationMetrics(**articulation),
        vad=VADMetrics(**vad),
    )


def _infer(
    metrics: Metrics,
    *,
    audio_quality: AudioQuality | None = None,
    duration_ms: int = 60000,
):
    return build_psychological_inference(
        metrics=metrics,
        scores=_scores(),
        audio_quality=audio_quality or AudioQuality(usable=True, background_noise_level="low"),
        uncertainty=Uncertainty(overall_confidence_label="medium_high", reasons=[]),
        duration_ms=duration_ms,
        scenario="benchmark",
        asr_confidence=0.9,
    )


def _behaviour(result, behaviour_id: str):
    return next(item for item in result.micro_behaviours if item.id == behaviour_id)


def _trait(result, trait_id: str):
    return next(item for item in result.traits if item.trait_id == trait_id)


def test_high_fillers_pace_acceleration_hesitation_infers_searching_and_nervous():
    result = _infer(
        _metrics(
            linguistic={"filler_words_per_min": 14.0},
            rhythm={"speed_up_segments": 2, "burst_speaking_segments": 1},
            derived={"hesitation_cluster_score": 0.8, "composure_index": 0.25},
            raw={"words_per_minute": 190.0, "mid_phrase_pause_rate": 0.5},
        )
    )

    assert _behaviour(result, "searching_for_wording").confidence >= 0.7
    nervous = _trait(result, "nervous")
    assert nervous.score > 60
    assert nervous.confidence >= 0.55


def test_owned_pauses_stable_rhythm_low_fillers_infers_composed_and_commanding():
    result = _infer(
        _metrics(
            linguistic={"filler_words_per_min": 0.5},
            rhythm={"rhythm_consistency": 0.85},
            raw={"avg_pause_ms": 500.0, "mid_phrase_pause_rate": 0.05},
            derived={"composure_index": 0.8, "vocal_command_index": 0.75},
        )
    )

    assert _behaviour(result, "pause_ownership").confidence >= 0.7
    assert _trait(result, "composed").score > 60
    assert _trait(result, "commanding").score > 60


def test_low_energy_and_pitch_variation_infers_flat_presence():
    result = _infer(
        _metrics(
            raw={"f0_range_semitones": 2.0, "loudness_variation_db": 2.0},
            derived={"dynamic_emphasis_score": 0.2, "projection_index": 0.25},
        )
    )

    assert _behaviour(result, "flat_delivery").confidence >= 0.65
    assert _trait(result, "flat").score > 60


def test_good_opening_structure_certainty_infers_credible_and_interview_ready():
    result = _infer(
        _metrics(
            linguistic={
                "opening_strength_score": 0.9,
                "structure_score": 0.85,
                "certainty_markers_per_100_words": 3.0,
                "specificity_score": 0.7,
                "concreteness_score": 0.6,
            },
            articulation={"clarity_proxy": 0.85},
        )
    )

    assert _trait(result, "credible").score > 60
    assert _trait(result, "interview_ready").score > 60


def test_poor_audio_reduces_confidence_and_suppresses_fragile_traits():
    result = _infer(
        _metrics(
            raw={"f0_range_semitones": 2.0, "loudness_variation_db": 2.0},
            derived={"dynamic_emphasis_score": 0.2, "projection_index": 0.25},
        ),
        audio_quality=AudioQuality(
            usable=False,
            background_noise_level="high",
            quality_warnings=["Overall recording quality is poor"],
        ),
        duration_ms=10000,
    )

    assert result.overall_inference_confidence < 0.6
    assert "flat" in result.suppressed_traits
    assert result.uncertainty.reasons


def test_single_metric_alone_never_creates_high_confidence_inference():
    result = _infer(
        _metrics(
            linguistic={"filler_words_per_min": 14.0},
            rhythm={"speed_up_segments": 0, "burst_speaking_segments": 0},
            derived={"hesitation_cluster_score": 0.1},
        )
    )

    assert _behaviour(result, "searching_for_wording").confidence <= 0.45
    assert _trait(result, "nervous").confidence < 0.5
    assert _trait(result, "nervous").suppress_from_report is True


def test_evidence_chain_is_populated_and_referenced():
    result = _infer(_metrics())

    assert result.evidence_chain
    confident = _trait(result, "confident")
    assert confident.supporting_evidence_ids
    evidence_ids = {item.evidence_id for item in result.evidence_chain}
    assert set(confident.supporting_evidence_ids).issubset(evidence_ids)
    assert confident.evidence_chain


def test_report_ready_candidates_reference_evidence_ids():
    result = _infer(_metrics())

    assert result.primary_candidates.primary_strength_candidate is not None
    assert result.report_candidates.authority_type_candidates
    assert result.report_candidates.report_priority_order
    candidate = result.primary_candidates.primary_strength_candidate
    assert candidate.evidence_ids
