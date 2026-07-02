"""Derived indices combining acoustic, rhythm, and articulation metrics."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from services.acoustic_metrics import AcousticAnalysisResult, WindowFeature
from services.articulation import ArticulationAnalysis
from services.rhythm_analysis import RhythmAnalysis
from services.vad import VADResult


@dataclass
class DerivedIndices:
    """Composite engineering features for psychological inference and calibration."""
    vocal_command_index: float  # Combined pace, pitch, energy control
    composure_index: float  # Stability, voicing, hesitation
    rhythm_index: float  # Consistency, cadence, rate stability
    projection_index: float  # Energy, emphasis, voicing
    authority_signal_index: float  # Overall composite of all indices
    confidence: float  # Overall confidence in indices (0-1)


def _calculate_vocal_command_index(
    voice_metrics: dict,
    pitch_contour: dict,
    energy_contour: dict,
    windows: list[WindowFeature],
) -> float:
    """
    Calculate vocal command index from pace, pitch, and energy control.
    
    Formula: Weighted combination of:
    - Pace stability (from windows)
    - Pitch dynamics (controlled variation)
    - Energy projection
    - Filler rate (inverse)
    """
    if not windows:
        return 0.5
    
    # Pace stability: inverse of WPM variance across windows
    wpms = [w.wpm for w in windows if w.wpm > 0]
    if wpms:
        wpm_std = (max(wpms) - min(wpms)) / (sum(wpms) / len(wpms)) if wpms else 0
        pace_stability = 1.0 / (1.0 + wpm_std)
    else:
        pace_stability = 0.5
    
    # Pitch dynamics: moderate variation is good, too little or too much is bad
    pitch_dynamics = pitch_contour.get("pitch_dynamics", 0.5)
    pitch_stability = pitch_contour.get("pitch_stability", 0.5)
    pitch_score = (pitch_dynamics * 0.4 + pitch_stability * 0.6)
    
    # Energy projection
    dynamic_emphasis = energy_contour.get("dynamic_emphasis", 1.0)
    projection_segments = energy_contour.get("projection_segments", 0)
    energy_score = min(1.0, (dynamic_emphasis - 1.0) * 0.5 + projection_segments * 0.1)
    
    # Filler penalty (inverse)
    avg_filler = sum(w.filler_rate for w in windows) / len(windows) if windows else 0
    filler_score = 1.0 - min(1.0, avg_filler * 5)
    
    # Combined index
    command_index = (
        pace_stability * 0.3 +
        pitch_score * 0.25 +
        energy_score * 0.25 +
        filler_score * 0.2
    )
    
    return round(command_index, 3)


def _calculate_composure_index(
    voice_quality: dict,
    vad_result: VADResult,
    windows: list[WindowFeature],
) -> float:
    """
    Calculate composure index from stability, voicing, and hesitation patterns.
    
    Formula: Weighted combination of:
    - Voicing ratio (consistent voicing)
    - Voice quality (low strain, low breathiness)
    - Hesitation clusters (inverse)
    - Pause control (appropriate pause frequency)
    """
    # Voicing ratio
    voicing_ratio = voice_quality.get("voicing_ratio", 0.8)
    voicing_score = voicing_ratio
    
    # Voice quality (inverse of strain and breathiness)
    strain = voice_quality.get("strain_proxy", 0.0)
    breathiness = voice_quality.get("breathiness_proxy", 0.0)
    quality_score = 1.0 - (strain * 0.5 + breathiness * 0.3)
    
    # Hesitation clusters (inverse)
    hesitation_count = sum(1 for w in windows if w.hesitation_cluster)
    hesitation_score = 1.0 - min(1.0, hesitation_count / len(windows) * 2) if windows else 0.5
    
    # Pause control (appropriate frequency, not too many or too few)
    pause_freq = vad_result.pause_frequency_per_minute
    if 8 <= pause_freq <= 15:
        pause_score = 1.0
    elif 5 <= pause_freq < 8 or 15 < pause_freq <= 20:
        pause_score = 0.8
    else:
        pause_score = 0.5
    
    # Combined index
    composure_index = (
        voicing_score * 0.3 +
        quality_score * 0.3 +
        hesitation_score * 0.2 +
        pause_score * 0.2
    )
    
    return round(composure_index, 3)


def _calculate_rhythm_index(
    rhythm_analysis: RhythmAnalysis,
    voice_metrics: dict,
) -> float:
    """
    Calculate rhythm index from consistency, cadence, and rate stability.
    
    Formula: Weighted combination of:
    - Rhythm consistency (inverse of rate variability)
    - Speech continuity (appropriate speech ratio)
    - Rate change stability (few bursts/slow-downs)
    - Pause cadence
    """
    # Rhythm consistency
    consistency = rhythm_analysis.rhythm_consistency
    
    # Speech continuity
    continuity = rhythm_analysis.speech_continuity
    # Optimal is around 0.7-0.85
    if 0.7 <= continuity <= 0.85:
        continuity_score = 1.0
    elif 0.6 <= continuity < 0.7 or 0.85 < continuity <= 0.9:
        continuity_score = 0.8
    else:
        continuity_score = 0.5
    
    # Rate change stability (inverse of extreme rate changes)
    total_rate_changes = (
        rhythm_analysis.burst_speaking_segments +
        rhythm_analysis.slow_down_segments +
        rhythm_analysis.speed_up_segments
    )
    rate_stability = 1.0 - min(1.0, total_rate_changes / 5)
    
    # Pause cadence (appropriate frequency)
    pause_cadence = rhythm_analysis.pause_cadence
    if 0.15 <= pause_cadence <= 0.3:
        cadence_score = 1.0
    elif 0.1 <= pause_cadence < 0.15 or 0.3 < pause_cadence <= 0.4:
        cadence_score = 0.8
    else:
        cadence_score = 0.5
    
    # Combined index
    rhythm_index = (
        consistency * 0.35 +
        continuity_score * 0.25 +
        rate_stability * 0.2 +
        cadence_score * 0.2
    )
    
    return round(rhythm_index, 3)


def _calculate_projection_index(
    energy_contour: dict,
    voice_quality: dict,
    windows: list[WindowFeature],
) -> float:
    """
    Calculate projection index from energy, emphasis, and voice quality.
    
    Formula: Weighted combination of:
    - Energy projection (sustained high energy)
    - Dynamic emphasis (local peaks)
    - Voice quality (good voicing, low strain)
    - Loudness stability (consistent projection)
    """
    # Energy projection
    projection_segments = energy_contour.get("projection_segments", 0)
    energy_cv = energy_contour.get("energy_cv", 0.5)
    # More projection segments with moderate CV is good
    projection_score = min(1.0, projection_segments * 0.15 + (1.0 - energy_cv) * 0.5)
    
    # Dynamic emphasis
    dynamic_emphasis = energy_contour.get("dynamic_emphasis", 1.0)
    emphasis_score = min(1.0, (dynamic_emphasis - 1.0) * 0.3)
    
    # Voice quality (voicing and strain)
    voicing_ratio = voice_quality.get("voicing_ratio", 0.8)
    strain = voice_quality.get("strain_proxy", 0.0)
    quality_score = voicing_ratio * (1.0 - strain * 0.5)
    
    # Loudness stability (from windows)
    loudness_stdevs = [w.loudness_stdev_db for w in windows if w.loudness_stdev_db > 0]
    if loudness_stdevs:
        avg_loudness_stdev = sum(loudness_stdevs) / len(loudness_stdevs)
        # Moderate variation is good for projection
        stability_score = 1.0 - min(1.0, abs(avg_loudness_stdev - 8) / 10)
    else:
        stability_score = 0.5
    
    # Combined index
    projection_index = (
        projection_score * 0.3 +
        emphasis_score * 0.25 +
        quality_score * 0.25 +
        stability_score * 0.2
    )
    
    return round(projection_index, 3)


def _calculate_authority_signal_index(
    vocal_command: float,
    composure: float,
    rhythm: float,
    projection: float,
    articulation: ArticulationAnalysis,
) -> float:
    """
    Calculate overall authority signal index.
    
    This is a composite of all derived indices plus articulation quality.
    It serves as an intermediate engineering feature for calibration.
    """
    # Articulation quality
    articulation_score = (
        articulation.phoneme_timing_consistency * 0.4 +
        articulation.speech_precision * 0.3 +
        articulation.clarity_proxy * 0.3
    )
    
    # Combined authority signal
    authority_signal = (
        vocal_command * 0.3 +
        composure * 0.25 +
        rhythm * 0.2 +
        projection * 0.15 +
        articulation_score * 0.1
    )
    
    return round(authority_signal, 3)


def _calculate_confidence(
    audio_quality_usable: bool,
    voice_quality: dict,
    vad_result: VADResult,
    duration_ms: int,
) -> float:
    """
    Calculate overall confidence in derived indices.
    
    Lower confidence for:
    - Poor audio quality
    - Low voicing ratio
    - Very short recordings
    - High uncertainty in VAD
    """
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
    
    return round(max(0.1, confidence), 2)


def calculate_derived_indices(
    acoustic_result: AcousticAnalysisResult,
    vad_result: VADResult,
    rhythm_analysis: RhythmAnalysis,
    articulation_analysis: ArticulationAnalysis,
    audio_quality_usable: bool,
    duration_ms: int,
) -> DerivedIndices:
    """
    Calculate all derived indices from analysis results.
    
    Args:
        acoustic_result: Complete acoustic analysis result
        vad_result: Voice Activity Detection result
        rhythm_analysis: Rhythm analysis result
        articulation_analysis: Articulation analysis result
        audio_quality_usable: Whether audio quality is usable
        duration_ms: Recording duration in milliseconds
    
    Returns:
        DerivedIndices with all composite engineering features
    """
    voice_metrics = acoustic_result.voice_metrics
    pitch_contour = acoustic_result.pitch_contour
    energy_contour = acoustic_result.energy_contour
    voice_quality = acoustic_result.voice_quality
    windows = acoustic_result.windows
    
    # Calculate individual indices
    vocal_command = _calculate_vocal_command_index(
        voice_metrics, pitch_contour, energy_contour, windows
    )
    
    composure = _calculate_composure_index(
        voice_quality, vad_result, windows
    )
    
    rhythm = _calculate_rhythm_index(rhythm_analysis, voice_metrics)
    
    projection = _calculate_projection_index(
        energy_contour, voice_quality, windows
    )
    
    authority_signal = _calculate_authority_signal_index(
        vocal_command, composure, rhythm, projection, articulation_analysis
    )
    
    confidence = _calculate_confidence(
        audio_quality_usable, voice_quality, vad_result, duration_ms
    )
    
    return DerivedIndices(
        vocal_command_index=vocal_command,
        composure_index=composure,
        rhythm_index=rhythm,
        projection_index=projection,
        authority_signal_index=authority_signal,
        confidence=confidence,
    )
