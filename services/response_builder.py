"""Assemble the authority.v2 response from pipeline outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import parselmouth
from openai import OpenAI

from schemas import (
    ArticulationMetrics,
    AuthorityV2Response,
    Drill,
    MetricEvidenceBundle,
    Metrics,
    Recommendations,
    RequestMetadata,
    RhythmMetrics,
    VADMetrics,
)
from services.acoustic_metrics import AcousticAnalysisResult, extract_acoustic_analysis
from services.articulation import analyze_articulation
from services.audio_preprocessing import preprocess_audio
from services.derived_indices import calculate_derived_indices
from services.diagnostic_reasoning import build_diagnostic_reasoning
from services.deterministic_coaching import build_deterministic_coaching
from services.evidence import (
    EvidenceCollection,
    add_articulation_evidence,
    add_audio_quality_evidence,
    add_derived_indices_evidence,
    add_energy_contour_evidence,
    add_pitch_contour_evidence,
    add_rhythm_evidence,
    add_vad_evidence,
    add_voice_quality_evidence,
    serialize_evidence_collection,
)
from services.inference_engine import (
    analyze_cognition,
    build_evidence,
    build_perception_profile,
    build_uncertainty,
)
from services.linguistic_metrics import build_linguistic_metrics, compute_delivery_metrics
from services.moments import build_moments
from services.psychological_inference import build_psychological_inference
from services.report_builder import apply_coaching_to_report, build_report
from services.rhythm_analysis import analyze_rhythm
from services.scoring_engine import compute_authority_score
from services.transcription import transcribe_audio
from services.vad import empty_vad_result, prepare_pcm_samples, run_vad


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


def _load_audio_samples(wav_path: str) -> tuple[np.ndarray, int]:
    sound = parselmouth.Sound(wav_path)
    samples = sound.values[0] if sound.values.ndim > 1 else sound.values
    return samples, int(sound.sampling_frequency)


def _should_force_empty_vad(duration_ms: int, audio_warnings: list[str]) -> bool:
    if duration_ms <= 0 or duration_ms < 1000:
        return True

    warning_text = " ".join(audio_warnings).lower()
    return any(
        marker in warning_text
        for marker in (
            "zero duration",
            "could not load audio",
            "near-silent",
            "very low signal",
        )
    )


def _deterministic_recommendations(coaching_engine) -> Recommendations:
    primary = coaching_engine.selected_interventions.primary_drill
    if primary:
        return Recommendations(
            highest_leverage_issue=primary.drill_id,
            fastest_improvement_tip=primary.why_selected or "highest_weighted_intervention_score",
            coaching_summary=coaching_engine.reasoning_chain.reason
            or "deterministic_intervention_selected_from_evidence",
        )
    return Recommendations(
        highest_leverage_issue="insufficient_evidence",
        fastest_improvement_tip="Record a usable sample before selecting a drill.",
        coaching_summary="Coaching was suppressed because evidence was insufficient.",
    )


def _deterministic_drills(coaching_engine) -> list[Drill]:
    selected = [
        item
        for item in (
            coaching_engine.selected_interventions.primary_drill,
            coaching_engine.selected_interventions.secondary_drill,
        )
        if item is not None
    ]
    by_id = {definition.drill_id: definition for definition in coaching_engine.drill_library}
    drills: list[Drill] = []
    for candidate in selected:
        definition = by_id.get(candidate.drill_id)
        if not definition:
            continue
        drills.append(
            Drill(
                drill_id=definition.drill_id,
                title=definition.title,
                goal=definition.category,
                instructions=[definition.description],
                duration_min=definition.estimated_duration_min,
                difficulty=definition.expected_difficulty,
                target_metrics=definition.target_metrics,
            )
        )
    return drills


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

    # Milestone 3: VAD-first segmentation (runs whenever audio exists)
    vad_result = empty_vad_result(duration_ms, backend="none")
    if duration_ms > 0:
        try:
            raw_samples, sample_rate = _load_audio_samples(wav_path)
            pcm_samples, pcm_rate = prepare_pcm_samples(raw_samples, sample_rate)
            vad_result = run_vad(
                pcm_samples,
                pcm_rate,
                transcription.transcript.words or None,
            )
        except Exception:
            vad_result = empty_vad_result(duration_ms, backend="empty_fallback")
    if _should_force_empty_vad(duration_ms, audio_quality.quality_warnings):
        vad_result = empty_vad_result(duration_ms, backend="empty_fallback")

    acoustic = extract_acoustic_analysis(
        wav_path,
        transcription.transcript.words,
        duration_ms=duration_ms,
        audio_usable=audio_quality.usable,
        transcript_text=text,
        vad_result=vad_result,
    )
    voice_metrics = acoustic.voice_metrics

    speech_duration_ms = (
        vad_result.total_speech_duration_ms
        if vad_result and vad_result.total_speech_duration_ms > 0
        else int(acoustic.speaking_seconds * 1000)
    )

    # Milestone 3: rhythm analysis from transcript timestamps
    rhythm_result = None
    if transcription.transcript.words and speech_duration_ms > 0:
        try:
            rhythm_result = analyze_rhythm(
                words=transcription.transcript.words,
                transcript_text=text,
                speech_duration_ms=speech_duration_ms,
                total_duration_ms=duration_ms,
            )
        except Exception:
            rhythm_result = None

    # Milestone 3: articulation analysis
    articulation_result = None
    if transcription.transcript.words and speech_duration_ms > 0:
        try:
            articulation_result = analyze_articulation(
                words=transcription.transcript.words,
                speech_duration_ms=speech_duration_ms,
            )
        except Exception:
            articulation_result = None

    # Milestone 3: derived indices (null when evidence is insufficient)
    derived_indices = None
    try:
        derived_indices = calculate_derived_indices(
            acoustic_result=acoustic,
            vad_result=vad_result,
            rhythm_analysis=rhythm_result,
            articulation_analysis=articulation_result,
            audio_quality_usable=audio_quality.usable,
            duration_ms=duration_ms,
        )
    except Exception:
        derived_indices = None

    # Milestone 3: Collect evidence
    evidence_collection = EvidenceCollection()
    if audio_quality:
        add_audio_quality_evidence(
            evidence_collection,
            audio_quality.snr_estimate_db,
            audio_quality.clipping_detected,
            audio_quality.background_noise_level,
            audio_quality.single_speaker_likelihood,
            audio_quality.usable,
        )
    if acoustic.pitch_contour:
        add_pitch_contour_evidence(evidence_collection, acoustic.pitch_contour)
    if acoustic.energy_contour:
        add_energy_contour_evidence(evidence_collection, acoustic.energy_contour)
    if acoustic.voice_quality:
        add_voice_quality_evidence(evidence_collection, acoustic.voice_quality)
    if rhythm_result:
        add_rhythm_evidence(evidence_collection, rhythm_result.__dict__)
    if articulation_result:
        add_articulation_evidence(evidence_collection, articulation_result.__dict__)
    if vad_result:
        add_vad_evidence(evidence_collection, vad_result.__dict__)
    if derived_indices:
        add_derived_indices_evidence(evidence_collection, derived_indices.__dict__)

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

    # Milestone 3: Map enhanced acoustic metrics to RawAcousticMetrics
    raw_acoustic = acoustic.raw.model_copy()
    if acoustic.pitch_contour:
        pitch_updates = {
            key: acoustic.pitch_contour.get(key)
            for key in (
                "pitch_mean_hz",
                "pitch_std_hz",
                "pitch_slope",
                "pitch_stability",
                "pitch_dynamics",
                "pitch_resets",
                "terminal_slope",
                "terminal_rising",
                "terminal_falling",
                "terminal_rising_ratio",
                "terminal_falling_ratio",
            )
            if key in acoustic.pitch_contour
        }
        if pitch_updates:
            raw_acoustic = raw_acoustic.model_copy(update=pitch_updates)
    if acoustic.energy_contour:
        raw_acoustic = raw_acoustic.model_copy(
            update={
                "energy_mean": acoustic.energy_contour.get("energy_mean"),
                "energy_peak": acoustic.energy_contour.get("energy_peak"),
                "energy_std": acoustic.energy_contour.get("energy_std"),
                "energy_slope": acoustic.energy_contour.get("energy_slope"),
                "dynamic_emphasis": acoustic.energy_contour.get("dynamic_emphasis"),
                "loudness_stability": acoustic.energy_contour.get("loudness_stability"),
                "emphasis_bursts": acoustic.energy_contour.get("emphasis_bursts"),
                "projection_segments": acoustic.energy_contour.get("projection_segments"),
                "energy_cv": acoustic.energy_contour.get("energy_cv"),
            }
        )
    if acoustic.voice_quality:
        raw_acoustic = raw_acoustic.model_copy(
            update={
                "voicing_ratio": acoustic.voice_quality.get("voicing_ratio"),
                "voice_breaks": acoustic.voice_quality.get("voice_breaks"),
                "breathiness_proxy": acoustic.voice_quality.get("breathiness_proxy"),
                "strain_proxy": acoustic.voice_quality.get("strain_proxy"),
                "cpp_proxy": acoustic.voice_quality.get("cpp_proxy"),
            }
        )

    # Milestone 3: Build rhythm metrics
    rhythm_metrics = RhythmMetrics()
    if rhythm_result:
        rhythm_metrics = RhythmMetrics(
            speech_rate=rhythm_result.speech_rate,
            words_per_minute=rhythm_result.words_per_minute,
            pause_cadence=rhythm_result.pause_cadence,
            speech_continuity=rhythm_result.speech_continuity,
            hesitation_windows=rhythm_result.hesitation_windows,
            rhythm_consistency=rhythm_result.rhythm_consistency,
            burst_speaking_segments=rhythm_result.burst_speaking_segments,
            slow_down_segments=rhythm_result.slow_down_segments,
            speed_up_segments=rhythm_result.speed_up_segments,
            articulation_rate=rhythm_result.articulation_rate,
        )

    # Milestone 3: Build articulation metrics
    articulation_metrics = ArticulationMetrics()
    if articulation_result:
        articulation_metrics = ArticulationMetrics(
            articulation_rate=articulation_result.articulation_rate,
            phoneme_timing_consistency=articulation_result.phoneme_timing_consistency,
            speech_precision=articulation_result.speech_precision,
            word_duration_mean_ms=articulation_result.word_duration_mean_ms,
            word_duration_std_ms=articulation_result.word_duration_std_ms,
            word_duration_cv=articulation_result.word_duration_cv,
            clarity_proxy=articulation_result.clarity_proxy,
            articulation_stability=articulation_result.articulation_stability,
        )

    # Milestone 3: Build VAD metrics
    vad_metrics = VADMetrics(
        speech_ratio=vad_result.speech_ratio,
        total_speech_duration_ms=vad_result.total_speech_duration_ms,
        total_silence_duration_ms=vad_result.total_silence_duration_ms,
        pause_durations_ms=vad_result.pause_durations_ms,
        long_pauses_ms=vad_result.long_pauses_ms,
        mid_sentence_pauses_ms=vad_result.mid_sentence_pauses_ms,
        end_of_sentence_pauses_ms=vad_result.end_of_sentence_pauses_ms,
        avg_pause_duration_ms=vad_result.avg_pause_duration_ms,
        pause_frequency_per_minute=vad_result.pause_frequency_per_minute,
        vad_backend=vad_result.vad_backend,
    )

    # Milestone 3: Update derived metrics with derived indices
    if derived_indices:
        acoustic.derived = acoustic.derived.model_copy(
            update={
                "vocal_command_index": derived_indices.vocal_command_index,
                "composure_index": derived_indices.composure_index,
                "rhythm_index": derived_indices.rhythm_index,
                "projection_index": derived_indices.projection_index,
                "authority_signal_index": derived_indices.authority_signal_index,
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
    if vad_result.total_speech_duration_ms <= 0:
        reason = (
            "VAD unavailable for silent audio"
            if vad_result.vad_backend == "none"
            else "No usable speech detected"
        )
        if reason not in uncertainty.reasons:
            uncertainty.reasons.append(reason)

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

    metrics_payload = Metrics(
        raw_acoustic=raw_acoustic,
        linguistic=linguistic,
        derived=acoustic.derived,
        rhythm=rhythm_metrics,
        articulation=articulation_metrics,
        vad=vad_metrics,
    )
    psychological_inference = build_psychological_inference(
        metrics=metrics_payload,
        scores=scores,
        audio_quality=audio_quality,
        uncertainty=uncertainty,
        duration_ms=duration_ms,
        scenario=_map_scenario(request.context),
        asr_confidence=transcription.transcript.overall_asr_confidence,
    )
    diagnostic_reasoning = build_diagnostic_reasoning(
        metrics=metrics_payload,
        psychological_inference=psychological_inference,
        evidence=evidence,
        moments=moments,
        scores=scores,
        audio_quality=audio_quality,
        uncertainty=uncertainty,
        duration_ms=duration_ms,
        scenario=_map_scenario(request.context),
    )
    report = build_report(
        scores=scores,
        metrics=metrics_payload,
        psychological_inference=psychological_inference,
        diagnostic_reasoning=diagnostic_reasoning,
        evidence=evidence,
        moments=moments,
        uncertainty=uncertainty,
        audio_quality=audio_quality,
        duration_ms=duration_ms,
        scenario=_map_scenario(request.context),
    )
    coaching_engine = build_deterministic_coaching(
        metrics=metrics_payload,
        scores=scores,
        psychological_inference=psychological_inference,
        diagnostic_reasoning=diagnostic_reasoning,
        report=report,
        audio_quality=audio_quality,
        uncertainty=uncertainty,
        duration_ms=duration_ms,
        scenario=_map_scenario(request.context),
    )
    report = apply_coaching_to_report(report, coaching_engine)

    recommendations = _deterministic_recommendations(coaching_engine)
    drills = _deterministic_drills(coaching_engine)

    metric_evidence_payload = serialize_evidence_collection(evidence_collection)

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
        metrics=metrics_payload,
        perception_profile=perception,
        evidence=evidence,
        metric_evidence=MetricEvidenceBundle.model_validate(metric_evidence_payload),
        moments=moments,
        recommendations=recommendations,
        drills=drills,
        psychological_inference=psychological_inference,
        report=report,
        coaching_engine=coaching_engine,
        uncertainty=uncertainty,
    )
