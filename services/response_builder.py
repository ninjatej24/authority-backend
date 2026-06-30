"""Assemble the authority.v2 response from pipeline outputs."""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from schemas import AuthorityV2Response, Metrics, RequestMetadata
from services.acoustic_metrics import AcousticAnalysisResult, extract_acoustic_analysis
from services.audio_preprocessing import preprocess_audio
from services.coaching_engine import build_drills, build_recommendations, generate_feedback
from services.inference_engine import (
    analyze_cognition,
    build_evidence,
    build_perception_profile,
    build_uncertainty,
)
from services.linguistic_metrics import build_linguistic_metrics, compute_delivery_metrics
from services.moments import build_moments
from services.scoring_engine import compute_authority_score
from services.transcription import transcribe_audio


@dataclass
class AnalyzeRequest:
    file_path: str
    original_suffix: str
    context: str = "initial"
    title: str = "Speech Analysis"
    prompt: str = ""
    drill_id: str | None = None
    module_slug: str | None = None
    skill: str | None = None
    device_context: str | None = None
    user_id: str | None = None
    language: str = "en"


def _map_scenario(context: str) -> str:
    normalized = (context or "initial").strip().lower()
    if normalized == "impromptu":
        return "impromptu"
    return "benchmark"


def _prompt_id(request: AnalyzeRequest) -> str:
    if request.drill_id:
        return request.drill_id
    if request.prompt:
        return "custom_prompt"
    return "authority_benchmark_v1"


def _missing_metric_names(acoustic: AcousticAnalysisResult) -> list[str]:
    missing: list[str] = []
    raw = acoustic.raw
    for name, value in raw.model_dump().items():
        if value is None:
            missing.append(name)
    return missing[:6]


def _score_confidence_adjustment(
    base: float,
    audio_usable: bool,
    asr_confidence: float | None,
    duration_ms: int,
    uncertainty_label: str,
) -> float:
    confidence = base
    if not audio_usable:
        confidence -= 0.25
    if asr_confidence is not None and asr_confidence < 0.6:
        confidence -= 0.15
    if duration_ms < 8000:
        confidence -= 0.1
    if uncertainty_label == "low":
        confidence = min(confidence, 0.45)
    elif uncertainty_label == "medium":
        confidence = min(confidence, 0.65)
    return round(max(0.25, min(0.95, confidence)), 2)


def run_analysis(client: OpenAI, request: AnalyzeRequest) -> AuthorityV2Response:
    """Execute the full v2 analysis pipeline."""
    preprocessed = preprocess_audio(request.file_path)
    wav_path = preprocessed.wav_path
    duration_ms = preprocessed.duration_ms
    audio_quality = preprocessed.audio_quality

    transcription = transcribe_audio(
        client,
        wav_path,
        duration_ms=duration_ms,
        language=request.language,
    )
    text = transcription.transcript.full_text

    acoustic = extract_acoustic_analysis(
        wav_path,
        transcription.transcript.words,
        duration_ms=duration_ms,
        audio_usable=audio_quality.usable,
        transcript_text=text,
    )
    voice_metrics = acoustic.voice_metrics

    cognitive = analyze_cognition(text, context=request.context, prompt_text=request.prompt)

    delivery = compute_delivery_metrics(
        text,
        voice_metrics.get("duration_seconds", 1),
        transcription.transcript.words,
        speaking_seconds=acoustic.speaking_seconds,
    )
    delivery_metrics = {
        "words_per_minute": delivery.words_per_minute,
        "filler_density": delivery.filler_density,
    }

    # Refresh derived metrics with transcript-aware filler density.
    acoustic.derived = acoustic.derived.model_copy(
        update={
            "hesitation_cluster_score": round(
                min(
                    1.0,
                    voice_metrics.get("pause_frequency", 0) * 1.5
                    + delivery.filler_density * 8,
                ),
                2,
            )
        }
    )

    linguistic = build_linguistic_metrics(
        text,
        delivery,
        voice_metrics.get("duration_seconds", 1),
        transcription.transcript.words,
        asr_confidence=transcription.transcript.overall_asr_confidence,
        cognitive=cognitive,
    )
    linguistic_dict = linguistic.model_dump()

    audio_penalty = 0.0
    if not audio_quality.usable:
        snr = audio_quality.snr_estimate_db
        audio_penalty = min(
            15.0,
            8.0 + (0 if snr is None else max(0, 12 - snr)),
        )

    scoring = compute_authority_score(
        voice_metrics,
        cognitive,
        delivery_metrics,
        linguistic_dict,
        audio_quality_penalty=audio_penalty,
        acoustic=acoustic,
    )

    feedback = generate_feedback(
        text,
        voice_metrics,
        delivery_metrics,
        cognitive,
        scoring.legacy_authority_score,
        context=request.context,
        prompt_text=request.prompt,
    )

    perception = build_perception_profile(
        scoring.scores.authority_score,
        scoring.dimension_map,
        cognitive,
        delivery_metrics,
        voice_metrics,
    )

    evidence = build_evidence(
        scoring.dimension_map,
        cognitive,
        delivery_metrics,
        voice_metrics,
        linguistic_dict,
    )

    moments = build_moments(
        transcription.transcript.words,
        duration_ms,
        acoustic.windows,
        delivery_metrics,
        linguistic_dict,
    )

    uncertainty = build_uncertainty(
        audio_quality.usable,
        not text.strip(),
        cognitive.get("failure", False),
        duration_ms=duration_ms,
        approximate_timestamps=transcription.approximate_timestamps,
        asr_confidence=transcription.transcript.overall_asr_confidence,
        audio_warnings=audio_quality.quality_warnings,
        missing_metrics=_missing_metric_names(acoustic),
    )

    scores = scoring.scores.model_copy(
        update={
            "score_confidence": _score_confidence_adjustment(
                scoring.scores.score_confidence or 0.79,
                audio_quality.usable,
                transcription.transcript.overall_asr_confidence,
                duration_ms,
                uncertainty.overall_confidence_label,
            )
        }
    )

    recommendations = build_recommendations(feedback, delivery_metrics)
    drills = build_drills(feedback, delivery_metrics, scoring.dimension_map)

    return AuthorityV2Response(
        request=RequestMetadata(
            scenario=_map_scenario(request.context),
            prompt_id=_prompt_id(request),
            language=request.language,
            duration_ms=duration_ms,
            device_context=request.device_context,
            user_id=request.user_id,
        ),
        audio_quality=audio_quality,
        transcript=transcription.transcript,
        scores=scores,
        metrics=Metrics(
            raw_acoustic=acoustic.raw,
            linguistic=linguistic,
            derived=acoustic.derived,
        ),
        perception_profile=perception,
        evidence=evidence,
        moments=moments,
        recommendations=recommendations,
        drills=drills,
        uncertainty=uncertainty,
    )
