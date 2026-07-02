"""Derived indices combining acoustic, rhythm, and articulation metrics."""

from __future__ import annotations

from dataclasses import dataclass

from services.acoustic_metrics import AcousticAnalysisResult, WindowFeature
from services.articulation import ArticulationAnalysis
from services.rhythm_analysis import RhythmAnalysis
from services.vad import VADResult


@dataclass
class DerivedIndices:
    """Composite engineering features for psychological inference and calibration."""

    vocal_command_index: float
    composure_index: float
    rhythm_index: float
    projection_index: float
    authority_signal_index: float
    confidence: float


def _default_rhythm_analysis() -> RhythmAnalysis:
    return RhythmAnalysis(
        speech_rate=0.0,
        words_per_minute=0.0,
        pause_cadence=0.0,
        speech_continuity=0.5,
        hesitation_windows=0,
        rhythm_consistency=0.5,
        burst_speaking_segments=0,
        slow_down_segments=0,
        speed_up_segments=0,
        articulation_rate=0.0,
    )


def _default_articulation_analysis() -> ArticulationAnalysis:
    return ArticulationAnalysis(
        articulation_rate=0.0,
        phoneme_timing_consistency=0.5,
        speech_precision=0.5,
        word_duration_mean_ms=0.0,
        word_duration_std_ms=0.0,
        word_duration_cv=0.0,
        clarity_proxy=0.5,
        articulation_stability=0.5,
    )


def _default_vad_result() -> VADResult:
    return VADResult(
        segments=[],
        speech_segments=[],
        silence_segments=[],
        speech_ratio=0.0,
        total_speech_duration_ms=0,
        total_silence_duration_ms=0,
        pause_durations_ms=[],
        long_pauses_ms=[],
        mid_sentence_pauses_ms=[],
        end_of_sentence_pauses_ms=[],
        avg_pause_duration_ms=0.0,
        pause_frequency_per_minute=0.0,
        vad_backend="missing",
    )


def _calculate_vocal_command_index(
    pitch_contour: dict,
    energy_contour: dict,
    windows: list[WindowFeature],
) -> float:
    """Calculate vocal command index from pace, pitch, and energy control."""
    if not windows:
        return 0.5

    wpms = [w.wpm for w in windows if w.wpm > 0]
    if wpms:
        wpm_std = (max(wpms) - min(wpms)) / (sum(wpms) / len(wpms)) if wpms else 0
        pace_stability = 1.0 / (1.0 + wpm_std)
    else:
        pace_stability = 0.5

    pitch_dynamics = pitch_contour.get("pitch_dynamics", 0.5)
    pitch_stability = pitch_contour.get("pitch_stability", 0.5)
    pitch_score = pitch_dynamics * 0.4 + pitch_stability * 0.6

    dynamic_emphasis = energy_contour.get("dynamic_emphasis", 1.0)
    projection_segments = energy_contour.get("projection_segments", 0)
    energy_score = min(1.0, (dynamic_emphasis - 1.0) * 0.5 + projection_segments * 0.1)

    avg_filler = sum(w.filler_rate for w in windows) / len(windows) if windows else 0
    filler_score = 1.0 - min(1.0, avg_filler * 5)

    command_index = (
        pace_stability * 0.3
        + pitch_score * 0.25
        + energy_score * 0.25
        + filler_score * 0.2
    )

    return round(command_index, 3)


def _calculate_composure_index(
    voice_quality: dict,
    vad_result: VADResult,
    windows: list[WindowFeature],
) -> float:
    """Calculate composure index from stability, voicing, and hesitation patterns."""
    voicing_ratio = voice_quality.get("voicing_ratio", 0.8)
    voicing_score = voicing_ratio

    strain = voice_quality.get("strain_proxy", 0.0)
    breathiness = voice_quality.get("breathiness_proxy", 0.0)
    quality_score = 1.0 - (strain * 0.5 + breathiness * 0.3)

    hesitation_count = sum(1 for w in windows if w.hesitation_cluster)
    hesitation_score = 1.0 - min(1.0, hesitation_count / len(windows) * 2) if windows else 0.5

    pause_freq = vad_result.pause_frequency_per_minute
    if 8 <= pause_freq <= 15:
        pause_score = 1.0
    elif 5 <= pause_freq < 8 or 15 < pause_freq <= 20:
        pause_score = 0.8
    else:
        pause_score = 0.5

    composure_index = (
        voicing_score * 0.3
        + quality_score * 0.3
        + hesitation_score * 0.2
        + pause_score * 0.2
    )

    return round(composure_index, 3)


def _calculate_rhythm_index(
    rhythm_analysis: RhythmAnalysis,
) -> float:
    """Calculate rhythm index from consistency, cadence, and rate stability."""
    consistency = rhythm_analysis.rhythm_consistency
    continuity = rhythm_analysis.speech_continuity

    if 0.7 <= continuity <= 0.85:
        continuity_score = 1.0
    elif 0.6 <= continuity < 0.7 or 0.85 < continuity <= 0.9:
        continuity_score = 0.8
    else:
        continuity_score = 0.5

    total_rate_changes = (
        rhythm_analysis.burst_speaking_segments
        + rhythm_analysis.slow_down_segments
        + rhythm_analysis.speed_up_segments
    )
    rate_stability = 1.0 - min(1.0, total_rate_changes / 5)

    pause_cadence = rhythm_analysis.pause_cadence
    if 0.15 <= pause_cadence <= 0.3:
        cadence_score = 1.0
    elif 0.1 <= pause_cadence < 0.15 or 0.3 < pause_cadence <= 0.4:
        cadence_score = 0.8
    else:
        cadence_score = 0.5

    rhythm_index = (
        consistency * 0.35
        + continuity_score * 0.25
        + rate_stability * 0.2
        + cadence_score * 0.2
    )

    return round(rhythm_index, 3)


def _calculate_projection_index(
    energy_contour: dict,
    voice_quality: dict,
    windows: list[WindowFeature],
) -> float:
    """Calculate projection index from energy, emphasis, and voice quality."""
    projection_segments = energy_contour.get("projection_segments", 0)
    energy_cv = energy_contour.get("energy_cv", 0.5)
    projection_score = min(1.0, projection_segments * 0.15 + (1.0 - energy_cv) * 0.5)

    dynamic_emphasis = energy_contour.get("dynamic_emphasis", 1.0)
    emphasis_score = min(1.0, (dynamic_emphasis - 1.0) * 0.3)

    voicing_ratio = voice_quality.get("voicing_ratio", 0.8)
    strain = voice_quality.get("strain_proxy", 0.0)
    quality_score = voicing_ratio * (1.0 - strain * 0.5)

    loudness_stdevs = [w.loudness_stdev_db for w in windows if w.loudness_stdev_db > 0]
    if loudness_stdevs:
        avg_loudness_stdev = sum(loudness_stdevs) / len(loudness_stdevs)
        stability_score = 1.0 - min(1.0, abs(avg_loudness_stdev - 8) / 10)
    else:
        stability_score = 0.5

    projection_index = (
        projection_score * 0.3
        + emphasis_score * 0.25
        + quality_score * 0.25
        + stability_score * 0.2
    )

    return round(projection_index, 3)


def _calculate_authority_signal_index(
    vocal_command: float,
    composure: float,
    rhythm: float,
    projection: float,
    articulation: ArticulationAnalysis,
) -> float:
    """Calculate overall authority signal index."""
    articulation_score = (
        articulation.phoneme_timing_consistency * 0.4
        + articulation.speech_precision * 0.3
        + articulation.clarity_proxy * 0.3
    )

    authority_signal = (
        vocal_command * 0.3
        + composure * 0.25
        + rhythm * 0.2
        + projection * 0.15
        + articulation_score * 0.1
    )

    return round(authority_signal, 3)


def _calculate_confidence(
    audio_quality_usable: bool,
    voice_quality: dict,
    vad_result: VADResult,
    duration_ms: int,
) -> float:
    """Calculate overall confidence in derived indices."""
    confidence = 1.0

    if not audio_quality_usable:
        confidence -= 0.3

    voicing_ratio = voice_quality.get("voicing_ratio", 0.8)
    if voicing_ratio < 0.6:
        confidence -= 0.2

    if duration_ms < 8000:
        confidence -= 0.15

    if vad_result.speech_ratio < 0.3:
        confidence -= 0.1

    if vad_result.vad_backend in {"missing", "empty", "energy_fallback"}:
        confidence -= 0.05

    return round(max(0.1, confidence), 2)


def calculate_derived_indices(
    acoustic_result: AcousticAnalysisResult,
    vad_result: VADResult | None,
    rhythm_analysis: RhythmAnalysis | None,
    articulation_analysis: ArticulationAnalysis | None,
    audio_quality_usable: bool,
    duration_ms: int,
) -> DerivedIndices | None:
    """
    Calculate derived indices when enough Milestone 3 inputs are available.

    Returns None when the recording lacks usable speech/transcript support.
    """
    rhythm = rhythm_analysis or _default_rhythm_analysis()
    articulation = articulation_analysis or _default_articulation_analysis()
    vad = vad_result or _default_vad_result()

    if not audio_quality_usable and vad.speech_ratio <= 0:
        return None

    if rhythm.words_per_minute <= 0 and articulation.articulation_rate <= 0 and vad.speech_ratio <= 0:
        return None

    pitch_contour = acoustic_result.pitch_contour
    energy_contour = acoustic_result.energy_contour
    voice_quality = acoustic_result.voice_quality
    windows = acoustic_result.windows

    vocal_command = _calculate_vocal_command_index(pitch_contour, energy_contour, windows)
    composure = _calculate_composure_index(voice_quality, vad, windows)
    rhythm_index = _calculate_rhythm_index(rhythm)
    projection = _calculate_projection_index(energy_contour, voice_quality, windows)
    authority_signal = _calculate_authority_signal_index(
        vocal_command,
        composure,
        rhythm_index,
        projection,
        articulation,
    )
    confidence = _calculate_confidence(audio_quality_usable, voice_quality, vad, duration_ms)

    return DerivedIndices(
        vocal_command_index=vocal_command,
        composure_index=composure,
        rhythm_index=rhythm_index,
        projection_index=projection,
        authority_signal_index=authority_signal,
        confidence=confidence,
    )
