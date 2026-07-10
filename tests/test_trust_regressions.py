"""Trust-focused regressions for conservative Authority reporting."""

from __future__ import annotations

from schemas import AudioQuality, TranscriptWord, Uncertainty
from services.linguistic_metrics import compute_delivery_metrics, build_linguistic_metrics
from services.moment_intelligence import build_moment_intelligence
from services.transcription import _response_words, _words_from_segments
from tests.test_diagnostic_reasoning import _diagnostic, _softened_expert_scores
from tests.test_moment_intelligence import _windows
from tests.test_psychological_inference import _infer, _metrics
from tests.test_report_builder import _evidence
from tests.test_report_generation import _generated_report, _report_user_facing_strings


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_top_level_and_segment_word_timestamps_keep_provenance():
    response = _Obj(words=[_Obj(word="Clear", start=0.1, end=0.4, confidence=0.9)])
    top_level = _response_words(response)
    assert top_level[0].timestamp_source == "real"

    segment_words, interpolated, source = _words_from_segments(
        [_Obj(text="Clear point", start=0.0, end=1.0, words=[_Obj(word="Clear", start=0.0, end=0.4)])],
        approximate=False,
    )
    assert interpolated is False
    assert source == "segment"
    assert segment_words[0].timestamp_source == "segment"


def test_segment_interpolation_is_marked_and_cannot_be_high_confidence():
    words, interpolated, source = _words_from_segments(
        [_Obj(text="Clear point now", start=0.0, end=3.0, words=[])],
        approximate=False,
    )
    assert interpolated is True
    assert source == "interpolated"
    assert {word.timestamp_source for word in words} == {"interpolated"}

    metrics = _metrics(linguistic={"opening_strength_score": 0.82, "closing_strength_score": 0.42, "structure_score": 0.72})
    bundle = build_moment_intelligence(
        words=words,
        duration_ms=18000,
        windows=_windows(),
        linguistic=metrics.linguistic,
        evidence=_evidence(),
        scores=_softened_expert_scores(),
        audio_quality=AudioQuality(usable=True, background_noise_level="low"),
        uncertainty=Uncertainty(overall_confidence_label="medium_high", reasons=[]),
        scenario="benchmark",
    )
    assert bundle.moments
    assert max(moment.confidence for moment in bundle.moments) <= 0.54


def test_whisper_removed_fillers_still_allows_acoustic_hesitation_copy():
    delivery = compute_delivery_metrics(
        "I think the plan works because the team can execute",
        duration_seconds=30,
        words=[TranscriptWord(text="I", start_ms=0, end_ms=100, timestamp_source="real")],
        speaking_seconds=30,
    )
    linguistic = build_linguistic_metrics(
        "I think the plan works because the team can execute",
        delivery,
        30,
        acoustic_hesitations=2,
        disfluency_confidence=0.74,
    )

    assert linguistic.lexical_fillers == 0
    assert linguistic.acoustic_hesitations == 2
    assert linguistic.confirmed_disfluencies == 2

    metrics = _metrics(
        linguistic={
            "filler_words_per_min": 0.0,
            "lexical_fillers": 0,
            "acoustic_hesitations": 2,
            "confirmed_disfluencies": 2,
            "disfluency_confidence": 0.74,
        },
        derived={"hesitation_cluster_score": 0.62},
        rhythm={"hesitation_windows": 2},
    )
    inference = _infer(metrics)
    diagnostic = _diagnostic(metrics=metrics, inference=inference)
    report = _generated_report(metrics=metrics, inference=inference, diagnostic_reasoning=diagnostic)
    text = " ".join(_report_user_facing_strings(report)).lower()

    assert "paused while searching for your next idea" in text
    assert "zero fillers" not in text


def test_moment_clustering_prevents_duplicate_overlapping_superlatives():
    words = [
        TranscriptWord(text=token, start_ms=i * 600, end_ms=i * 600 + 400, timestamp_source="real")
        for i, token in enumerate("one clear point with proof and finality now".split())
    ]
    metrics = _metrics(linguistic={"opening_strength_score": 0.8, "closing_strength_score": 0.78, "structure_score": 0.76})
    bundle = build_moment_intelligence(
        words=words,
        duration_ms=18000,
        windows=_windows(),
        linguistic=metrics.linguistic,
        evidence=_evidence(),
        scores=_softened_expert_scores(),
        audio_quality=AudioQuality(usable=True, background_noise_level="low"),
        uncertainty=Uncertainty(overall_confidence_label="medium_high", reasons=[]),
        scenario="benchmark",
    )

    assert len(bundle.moments) <= 4
    positive_superlatives = [m for m in bundle.moments if m.type in {"strongest_moment", "high_presence_moment", "most_commanding_moment", "most_persuasive_moment"}]
    assert len(positive_superlatives) <= 1


def test_weak_sample_suppresses_primary_diagnosis():
    metrics = _metrics()
    audio_quality = AudioQuality(usable=True, background_noise_level="low")
    inference = _infer(metrics, duration_ms=18000, audio_quality=audio_quality)
    diagnostic = _diagnostic(metrics=metrics, inference=inference, duration_ms=18000, audio_quality=audio_quality)

    assert diagnostic.primary_diagnosis is None


def test_user_facing_copy_has_no_internal_language_or_raw_metrics():
    report = _generated_report()
    text = " ".join(_report_user_facing_strings(report)).lower()
    forbidden = [
        "winning diagnosis",
        "supported by",
        "contradicted by",
        "evidence items",
        "deterministic",
        "backend",
        "raw_acoustic",
        "linguistic.",
        "derived.",
        "hypothesis",
        "observed as",
    ]
    assert not any(marker in text for marker in forbidden)
