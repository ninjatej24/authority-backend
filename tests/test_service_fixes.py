"""Unit tests for service-layer bug fixes."""

import numpy as np

from services.acoustic_metrics import estimate_syllables_per_second
from services.audio_preprocessing import _estimate_snr_db
from services.coaching_engine import _weakest_dimension_metric, build_drills
from services.linguistic_metrics import build_linguistic_metrics, compute_delivery_metrics


def test_structure_score_is_deterministic_from_transcript_signals():
    delivery = compute_delivery_metrics(
        "The point is we will deliver results. Therefore the answer is clear.",
        duration_seconds=10.0,
    )
    metrics = build_linguistic_metrics(
        "The point is we will deliver results. Therefore the answer is clear.",
        delivery,
        duration_seconds=10.0,
    )
    assert metrics.structure_score is not None
    assert metrics.structure_score >= 0.5


def test_structure_score_penalizes_hedgy_opening():
    delivery = compute_delivery_metrics(
        "Um maybe I think sort of we could possibly try something.",
        duration_seconds=10.0,
    )
    metrics = build_linguistic_metrics(
        "Um maybe I think sort of we could possibly try something.",
        delivery,
        duration_seconds=10.0,
    )
    assert metrics.opening_strength_score is not None
    assert metrics.opening_strength_score < 0.6


def test_drill_target_metrics_use_metric_identifiers_not_dimension_names():
    dimension_map = {
        "command": 80,
        "clarity": 45,
        "composure": 70,
        "presence": 75,
        "persuasion": 68,
        "structure": 72,
    }
    drills = build_drills(
        {
            "drills": ["Practice drill one", "Practice drill two"],
            "main_issue": "clarity",
            "fixes": ["Fix one"],
        },
        {"filler_density": 0.01, "words_per_minute": 130},
        dimension_map,
    )
    assert drills[0].target_metrics[1] == "filler_words_per_min"
    assert drills[0].target_metrics[1] not in dimension_map


def test_weakest_dimension_metric_mapping():
    assert (
        _weakest_dimension_metric({"command": 90, "clarity": 40, "structure": 80})
        == "filler_words_per_min"
    )


def test_syllable_count_uses_per_word_minimum_not_word_count_fallback():
    rate = estimate_syllables_per_second("interesting authentication", speaking_seconds=2.0)
    assert rate is not None
    assert rate >= 2.5


def test_syllables_per_second_uses_per_word_estimate():
    rate = estimate_syllables_per_second("strengths breeds", speaking_seconds=2.0)
    assert rate is not None
    assert rate >= 1.0


def test_snr_estimate_handles_single_frame_audio():
    sample_rate = 16000
    frame_size = int(sample_rate * 0.02)
    signal = np.sin(np.linspace(0, 8 * np.pi, frame_size)).astype(np.float64)
    noise = np.random.default_rng(0).normal(0, 0.01, frame_size)
    samples = signal + noise

    snr = _estimate_snr_db(samples, sample_rate)
    assert snr is not None
