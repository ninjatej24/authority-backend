"""Milestone 8 deterministic scoring and calibration tests."""

from __future__ import annotations

import pytest

from schemas import DerivedMetrics, RawAcousticMetrics
from services.acoustic_metrics import AcousticAnalysisResult
from services.scoring_engine import DIMENSION_WEIGHTS, compute_authority_score


def _cognitive(score: int = 66) -> dict:
    return {
        "clarity": {"score": score, "reason": "fixture"},
        "persuasion": {"score": score, "reason": "fixture"},
        "coherence": {"score": score, "reason": "fixture"},
        "idea_strength": {"score": score, "reason": "fixture"},
        "conciseness": {"score": score, "reason": "fixture"},
        "failure": False,
    }


def _voice(**overrides) -> dict:
    values = {
        "duration_seconds": 45.0,
        "pitch_variation": 7.0,
        "energy_variation": 8.0,
        "silence_ratio": 0.18,
        "avg_pause_duration": 0.48,
        "pause_frequency": 0.22,
        "terminal_rise_ratio": 0.12,
    }
    values.update(overrides)
    return values


def _delivery(**overrides) -> dict:
    values = {"words_per_minute": 142.0, "filler_density": 0.015}
    values.update(overrides)
    return values


def _linguistic(**overrides) -> dict:
    values = {
        "filler_words_per_min": 2.0,
        "hedges_per_100_words": 1.0,
        "certainty_markers_per_100_words": 3.0,
        "specificity_score": 0.62,
        "concreteness_score": 0.58,
        "rambling_score": 0.22,
        "opening_strength_score": 0.72,
        "closing_strength_score": 0.72,
        "structure_score": 0.66,
    }
    values.update(overrides)
    return values


def _acoustic(**derived_overrides) -> AcousticAnalysisResult:
    derived = {
        "monotony_index": 0.2,
        "hesitation_cluster_score": 0.2,
        "dynamic_emphasis_score": 0.64,
        "speech_continuity_score": 0.72,
        "confidence_drop_count": 0,
        "projection_index": 0.64,
        "rhythm_index": 0.72,
    }
    derived.update(derived_overrides)
    return AcousticAnalysisResult(
        voice_metrics={},
        raw=RawAcousticMetrics(),
        derived=DerivedMetrics(**derived),
        speaking_seconds=45.0,
    )


def _score(**kwargs):
    return compute_authority_score(
        kwargs.pop("voice", _voice()),
        kwargs.pop("cognitive", _cognitive()),
        kwargs.pop("delivery", _delivery()),
        kwargs.pop("linguistic", _linguistic()),
        acoustic=kwargs.pop("acoustic", _acoustic()),
        audio_quality_penalty=kwargs.pop("audio_quality_penalty", 0.0),
        audio_quality_usable=kwargs.pop("audio_quality_usable", True),
        asr_confidence=kwargs.pop("asr_confidence", 0.88),
        duration_ms=kwargs.pop("duration_ms", 45000),
    ).scores


def test_six_dimensions_feed_weighted_latent_score():
    scores = _score()
    dims = scores.dimension_scores.model_dump()
    assert set(dims) == set(DIMENSION_WEIGHTS)
    expected = sum(dims[name] * weight for name, weight in DIMENSION_WEIGHTS.items())
    assert scores.score_components.weighted_base == pytest.approx(expected, abs=0.01)
    assert scores.calibration_metadata.method == "deterministic_v2_pre_human_corpus"


def test_severe_filler_burden_creates_meaningful_penalty_and_cap():
    baseline = _score()
    severe = _score(delivery=_delivery(filler_density=0.11), linguistic=_linguistic(filler_words_per_min=12.0))
    assert severe.score_components.penalties.filler_penalty >= 9.0
    assert severe.authority_score < baseline.authority_score
    assert any(cap.id == "severe_filler_cap" for cap in severe.score_components.caps_applied)


def test_severe_monotony_penalizes_presence():
    baseline = _score()
    monotone = _score(
        voice=_voice(pitch_variation=1.8, energy_variation=2.0),
        acoustic=_acoustic(monotony_index=0.9, dynamic_emphasis_score=0.1, projection_index=0.2),
    )
    assert monotone.dimension_scores.presence < baseline.dimension_scores.presence
    assert monotone.score_components.penalties.monotony_penalty >= 7.0


def test_weak_close_plus_rising_endings_penalizes_command():
    baseline = _score()
    weak_close = _score(
        voice=_voice(terminal_rise_ratio=0.7),
        linguistic=_linguistic(closing_strength_score=0.2, certainty_markers_per_100_words=0.4),
    )
    assert weak_close.dimension_scores.command < baseline.dimension_scores.command
    assert weak_close.score_components.penalties.rising_ending_penalty >= 5.0


def test_poor_audio_reduces_confidence_and_applies_cap():
    poor = _score(audio_quality_penalty=15.0, audio_quality_usable=False, asr_confidence=0.5)
    assert poor.score_confidence < 0.65
    assert any(cap.id == "poor_audio_cap" for cap in poor.score_components.caps_applied)
    assert "audio quality" in " ".join(poor.score_explanation.confidence_reasons).lower()


def test_short_usable_speech_caps_score():
    short_audio = _acoustic()
    short_audio.speaking_seconds = 6.0
    short = _score(acoustic=short_audio, duration_ms=6000)
    assert short.authority_score <= 58
    assert any(cap.id == "very_short_speech_cap" for cap in short.score_components.caps_applied)


def test_elite_scores_are_capped_unless_profile_is_extremely_clean():
    pretty_good = _score(cognitive=_cognitive(82), linguistic=_linguistic(opening_strength_score=0.82, closing_strength_score=0.6))
    excellent = _score(
        voice=_voice(pitch_variation=8.0, energy_variation=10.0, terminal_rise_ratio=0.02),
        cognitive=_cognitive(95),
        delivery=_delivery(words_per_minute=138.0, filler_density=0.0),
        linguistic=_linguistic(
            filler_words_per_min=0.0,
            hedges_per_100_words=0.0,
            certainty_markers_per_100_words=6.0,
            specificity_score=0.92,
            concreteness_score=0.9,
            rambling_score=0.02,
            opening_strength_score=0.95,
            closing_strength_score=0.95,
            structure_score=0.94,
        ),
        acoustic=_acoustic(monotony_index=0.02, dynamic_emphasis_score=0.95, speech_continuity_score=0.94, projection_index=0.9, rhythm_index=0.92),
    )
    assert pretty_good.authority_score < 92
    assert 90 <= excellent.authority_score <= 97


def test_score_spread_across_bad_average_good_excellent_profiles():
    bad = _score(
        voice=_voice(pitch_variation=1.5, energy_variation=1.8, terminal_rise_ratio=0.7),
        cognitive=_cognitive(38),
        delivery=_delivery(words_per_minute=205.0, filler_density=0.12),
        linguistic=_linguistic(filler_words_per_min=14.0, rambling_score=0.9, opening_strength_score=0.2, closing_strength_score=0.2, structure_score=0.25),
        acoustic=_acoustic(monotony_index=0.9, hesitation_cluster_score=0.9, dynamic_emphasis_score=0.1, speech_continuity_score=0.2),
    )
    average = _score()
    good = _score(cognitive=_cognitive(78), linguistic=_linguistic(opening_strength_score=0.82, closing_strength_score=0.82, structure_score=0.8))
    excellent = _score(
        voice=_voice(pitch_variation=8.0, energy_variation=10.0, terminal_rise_ratio=0.02),
        cognitive=_cognitive(94),
        delivery=_delivery(words_per_minute=138.0, filler_density=0.0),
        linguistic=_linguistic(filler_words_per_min=0.0, rambling_score=0.02, opening_strength_score=0.95, closing_strength_score=0.95, structure_score=0.94),
        acoustic=_acoustic(monotony_index=0.02, dynamic_emphasis_score=0.95, speech_continuity_score=0.94, projection_index=0.9, rhythm_index=0.92),
    )
    assert bad.authority_score < average.authority_score < good.authority_score < excellent.authority_score
    assert bad.authority_score < 25
    assert average.authority_score != 64


def test_normal_valid_profile_scores_reasonably_with_sample_limitation_reason():
    scores = _score()

    assert 53 <= scores.authority_score <= 80
    assert scores.score_confidence < 0.8
    assert "single benchmark recording" in " ".join(scores.score_explanation.confidence_reasons).lower()


def test_absolute_pitch_alone_is_suppressed_by_fairness_rules():
    base = _score()
    high_pitch = _score(voice={**_voice(), "pitch_mean": 260.0})
    assert abs(base.authority_score - high_pitch.authority_score) <= 1
    assert "absolute_pitch" in high_pitch.fairness_adjustments.suppressed_features


def test_score_components_are_explainable_and_report_ready():
    scores = _score()
    assert scores.score_components.final_score == scores.authority_score
    assert scores.score_explanation.component_summary
    assert scores.dimension_details["command"].positive_contributors
    assert scores.score_band_label
    assert scores.score_rarity_label
