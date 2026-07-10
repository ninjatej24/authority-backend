"""Deterministic acoustic and prosodic feature extraction."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import parselmouth

from schemas import DerivedMetrics, RawAcousticMetrics, TranscriptWord
from services.vad import VADResult, build_silence_frame_mask, pauses_from_vad

VoiceMetrics = dict[str, float]

PAUSE_MIN_MS = 200
WINDOW_MS = 3000
HOP_MS = 1000
TERMINAL_WINDOW_MS = 400


@dataclass
class WindowFeature:
    start_ms: int
    end_ms: int
    command_score: float
    clarity_score: float
    composure_score: float
    presence_score: float
    filler_rate: float
    wpm: float
    pitch_stdev_semitones: float
    loudness_stdev_db: float
    pause_ms: float
    monotone: bool
    rushing: bool
    # Enhanced features for Milestone 3
    pitch_dynamics: float = 0.0
    energy_cv: float = 0.0
    dynamic_emphasis: float = 0.0
    voicing_ratio: float = 0.0
    hesitation_cluster: bool = False
    articulation_consistency: float = 0.5


@dataclass
class AcousticAnalysisResult:
    voice_metrics: VoiceMetrics
    raw: RawAcousticMetrics
    derived: DerivedMetrics
    windows: list[WindowFeature] = field(default_factory=list)
    speaking_seconds: float = 0.0
    pitch_contour: dict[str, float] = field(default_factory=dict)
    energy_contour: dict[str, float] = field(default_factory=dict)
    voice_quality: dict[str, float] = field(default_factory=dict)


def _hz_to_semitones(hz: float, reference_hz: float) -> float:
    if hz <= 0 or reference_hz <= 0:
        return 0.0
    return 12 * math.log2(hz / reference_hz)


def _empty_voice_metrics() -> VoiceMetrics:
    return {
        "duration_seconds": 0.0,
        "pitch_mean": 0.0,
        "pitch_median": 0.0,
        "pitch_variation": 0.0,
        "energy_mean": 0.0,
        "energy_variation": 0.0,
        "silence_ratio": 0.0,
        "avg_pause_duration": 0.0,
        "pause_frequency": 0.0,
        "speech_density": 0.0,
        "longest_pause_seconds": 0.0,
        "pause_count": 0.0,
        "mid_phrase_pause_rate": 0.0,
        "terminal_rise_ratio": 0.0,
        "f0_range_semitones": 0.0,
        "f0_variability_semitones": 0.0,
    }


def extract_voice_metrics(audio_path: str) -> VoiceMetrics:
    """Backward-compatible voice metrics dict."""
    return extract_acoustic_analysis(audio_path, [], duration_ms=0).voice_metrics


def extract_acoustic_analysis(
    audio_path: str,
    words: list[TranscriptWord],
    duration_ms: int,
    *,
    audio_usable: bool = True,
    transcript_text: str = "",
    vad_result: VADResult | None = None,
) -> AcousticAnalysisResult:
    """Full deterministic acoustic pipeline."""
    try:
        snd = parselmouth.Sound(str(audio_path))
    except Exception:
        empty = _empty_voice_metrics()
        return AcousticAnalysisResult(
            voice_metrics=empty,
            raw=_build_raw_from_voice(empty, 0.0, None),
            derived=_build_derived(empty, {}, 0),
            windows=[],
            speaking_seconds=0.0,
        )

    duration = float(snd.get_total_duration())
    if duration <= 0:
        empty = _empty_voice_metrics()
        return AcousticAnalysisResult(
            voice_metrics=empty,
            raw=_build_raw_from_voice(empty, 0.0, None),
            derived=_build_derived(empty, {}, 0),
            windows=[],
            speaking_seconds=0.0,
        )

    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    voiced_f0 = pitch_values[pitch_values > 0]

    intensity = snd.to_intensity()
    intensity_values = intensity.values[0]
    frame_duration = float(intensity.dx) if len(intensity_values) else 0.01

    if len(intensity_values) == 0:
        voice = _empty_voice_metrics()
        voice["duration_seconds"] = duration
        return AcousticAnalysisResult(
            voice_metrics=voice,
            raw=_build_raw_from_voice(voice, 0.0, None),
            derived=_build_derived(voice, {}, 0),
            windows=[],
            speaking_seconds=0.0,
        )

    energy_mean = float(np.mean(intensity_values))
    energy_std = float(np.std(intensity_values))

    threshold = energy_mean * 0.55
    threshold = max(threshold, float(np.percentile(intensity_values, 12)))
    threshold = min(threshold, float(np.percentile(intensity_values, 38)))
    intensity_silence_frames = intensity_values < threshold

    if vad_result is not None and vad_result.segments:
        silence_frames = build_silence_frame_mask(
            vad_result,
            frame_duration,
            len(intensity_values),
        )
        pauses = pauses_from_vad(vad_result)
        speech_duration = max(vad_result.total_speech_duration_ms / 1000.0, 0.01)
        silence_duration = vad_result.total_silence_duration_ms / 1000.0
    else:
        silence_frames = intensity_silence_frames
        pauses = _detect_pauses(silence_frames, frame_duration)
        silence_duration = float(np.sum(silence_frames) * frame_duration)
        speech_duration = max(duration - silence_duration, 0.01)

    silence_ratio = silence_duration / duration if duration > 0 else 0.0
    pause_count = len(pauses)
    avg_pause = float(np.mean(pauses)) if pauses else 0.0
    longest_pause = float(max(pauses)) if pauses else 0.0
    pause_frequency = pause_count / duration if duration > 0 else 0.0
    speech_density = speech_duration / duration if duration > 0 else 0.0

    pitch_median = float(np.median(voiced_f0)) if len(voiced_f0) else 0.0
    pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) else 0.0
    pitch_std_hz = float(np.std(voiced_f0)) if len(voiced_f0) else 0.0

    f0_range_semitones, f0_variability_semitones = _f0_shape_metrics(voiced_f0, pitch_median)
    terminal_rise_ratio = _estimate_terminal_rise_ratio(
        pitch, intensity, silence_frames, frame_duration, pitch_median
    )
    mid_phrase_pause_rate = _mid_phrase_pause_rate(words, pauses, frame_duration, silence_frames)
    if mid_phrase_pause_rate is None:
        mid_phrase_pause_rate = 0.0

    # Enhanced pitch contour analysis (VAD-aligned phrase boundaries when available)
    pitch_contour = _analyze_pitch_contour(
        pitch_values,
        pitch_median,
        frame_duration,
        silence_frames=silence_frames,
        vad_result=vad_result,
    )
    
    # Enhanced energy contour analysis
    energy_contour = _analyze_energy_contour(intensity_values, frame_duration)
    
    # Enhanced voice quality analysis
    voice_quality = _analyze_voice_quality(snd, pitch, intensity, audio_usable, duration)

    loudness_mean_db = _relative_db(energy_mean)
    loudness_variation_db = float(energy_std)

    word_count = len(words) if words else len(transcript_text.split())
    speaking_seconds = speech_duration
    wpm = (word_count / speaking_seconds) * 60 if speaking_seconds > 0 else 0.0
    syllables_per_second = _syllables_per_second(transcript_text, speaking_seconds)

    voice_metrics: VoiceMetrics = {
        "duration_seconds": duration,
        "pitch_mean": pitch_mean,
        "pitch_median": pitch_median,
        "pitch_variation": pitch_std_hz,
        "energy_mean": energy_mean,
        "energy_variation": energy_std,
        "silence_ratio": float(silence_ratio),
        "avg_pause_duration": avg_pause,
        "pause_frequency": float(pause_frequency),
        "speech_density": float(speech_density),
        "longest_pause_seconds": longest_pause,
        "pause_count": float(pause_count),
        "mid_phrase_pause_rate": float(mid_phrase_pause_rate),
        "terminal_rise_ratio": float(terminal_rise_ratio)
        if terminal_rise_ratio is not None
        else 0.0,
        "f0_range_semitones": float(f0_range_semitones),
        "f0_variability_semitones": float(f0_variability_semitones),
    }

    raw = _build_raw_from_voice(voice_metrics, wpm, syllables_per_second)
    
    # Extract HNR, jitter, shimmer from voice_quality for backward compatibility
    hnr = voice_quality.get("hnr") if voice_quality else None
    jitter = voice_quality.get("jitter") if voice_quality else None
    shimmer = voice_quality.get("shimmer") if voice_quality else None
    
    raw = RawAcousticMetrics(
        words_per_minute=raw.words_per_minute,
        syllables_per_second=raw.syllables_per_second,
        pause_frequency_per_min=raw.pause_frequency_per_min,
        avg_pause_ms=raw.avg_pause_ms,
        longest_pause_ms=raw.longest_pause_ms,
        mid_phrase_pause_rate=round(mid_phrase_pause_rate, 2) if mid_phrase_pause_rate else None,
        f0_median_hz=round(pitch_median, 1) if pitch_median > 0 else None,
        f0_range_semitones=round(f0_range_semitones, 1) if f0_range_semitones else None,
        f0_variability_semitones=round(f0_variability_semitones, 1)
        if f0_variability_semitones
        else None,
        terminal_rise_ratio=round(terminal_rise_ratio, 2) if terminal_rise_ratio is not None else None,
        loudness_mean_db_relative=round(loudness_mean_db, 1)
        if loudness_mean_db is not None
        else (round(energy_mean, 1) if energy_mean > 0 else None),
        loudness_variation_db=round(loudness_variation_db, 1) if loudness_variation_db else None,
        hnr=round(hnr, 1) if hnr is not None else None,
        jitter_local=round(jitter, 5) if jitter is not None else None,
        shimmer_local=round(shimmer, 5) if shimmer is not None else None,
    )

    windows = _sliding_window_features(
        words,
        duration_ms or int(duration * 1000),
        pitch,
        intensity,
        silence_frames,
        frame_duration,
        pitch_median,
    )
    confidence_drop_count = _count_confidence_drops(windows)
    derived = _build_derived(voice_metrics, {"filler_density": _window_mean_fillers(windows)}, confidence_drop_count)

    return AcousticAnalysisResult(
        voice_metrics=voice_metrics,
        raw=raw,
        derived=derived,
        windows=windows,
        speaking_seconds=speaking_seconds,
        pitch_contour=pitch_contour,
        energy_contour=energy_contour,
        voice_quality=voice_quality,
    )


def build_raw_acoustic_metrics(
    voice_metrics: VoiceMetrics,
    words_per_minute: float,
    syllables_per_second: float | None = None,
) -> RawAcousticMetrics:
    return _build_raw_from_voice(voice_metrics, words_per_minute, syllables_per_second)


def estimate_syllables_per_second(text: str, speaking_seconds: float) -> float | None:
    return _syllables_per_second(text, speaking_seconds)


def _syllables_per_second(text: str, speaking_seconds: float) -> float | None:
    if speaking_seconds <= 0 or not text.strip():
        return None
    syllables = sum(
        max(1, len(re.findall(r"[aeiouyAEIOUY]+", word))) for word in text.split()
    )
    if syllables <= 0:
        return None
    return round(syllables / speaking_seconds, 1)


def _build_raw_from_voice(
    voice_metrics: VoiceMetrics,
    words_per_minute: float,
    syllables_per_second: float | None,
) -> RawAcousticMetrics:
    return RawAcousticMetrics(
        words_per_minute=round(words_per_minute, 1) if words_per_minute else None,
        syllables_per_second=syllables_per_second,
        pause_frequency_per_min=round(voice_metrics.get("pause_frequency", 0) * 60, 1),
        avg_pause_ms=round(voice_metrics.get("avg_pause_duration", 0) * 1000, 1),
        longest_pause_ms=round(voice_metrics.get("longest_pause_seconds", 0) * 1000, 1),
        mid_phrase_pause_rate=voice_metrics.get("mid_phrase_pause_rate"),
        f0_median_hz=round(voice_metrics.get("pitch_median", 0), 1)
        if voice_metrics.get("pitch_median", 0) > 0
        else None,
        f0_range_semitones=voice_metrics.get("f0_range_semitones"),
        f0_variability_semitones=voice_metrics.get("f0_variability_semitones"),
        terminal_rise_ratio=voice_metrics.get("terminal_rise_ratio"),
        loudness_mean_db_relative=_relative_db(voice_metrics.get("energy_mean", 0)),
        loudness_variation_db=round(voice_metrics.get("energy_variation", 0), 1)
        if voice_metrics.get("energy_variation", 0)
        else None,
        hnr=None,
        jitter_local=None,
        shimmer_local=None,
    )


def _relative_db(value: float) -> float | None:
    if value <= 0:
        return None
    return round(20 * math.log10(value), 1)


def _detect_pauses(silence_frames: np.ndarray, frame_duration: float) -> list[float]:
    pauses: list[float] = []
    current_pause = 0.0
    min_pause = PAUSE_MIN_MS / 1000.0
    for is_silent in silence_frames:
        if is_silent:
            current_pause += frame_duration
        else:
            if current_pause >= min_pause:
                pauses.append(current_pause)
            current_pause = 0.0
    if current_pause >= min_pause:
        pauses.append(current_pause)
    return pauses


def _f0_shape_metrics(voiced_f0: np.ndarray, pitch_median: float) -> tuple[float, float]:
    if len(voiced_f0) == 0 or pitch_median <= 0:
        return 0.0, 0.0
    semitones = np.array([_hz_to_semitones(hz, pitch_median) for hz in voiced_f0])
    p10 = float(np.percentile(semitones, 10))
    p90 = float(np.percentile(semitones, 90))
    return round(p90 - p10, 2), round(float(np.std(semitones)), 2)


def _analyze_pitch_contour(
    pitch_values: np.ndarray,
    pitch_median: float,
    frame_duration: float,
    *,
    silence_frames: np.ndarray | None = None,
    vad_result: VADResult | None = None,
) -> dict[str, float]:
    """Enhanced pitch contour analysis including dynamics, resets, and terminal movement."""
    if len(pitch_values) == 0 or pitch_median <= 0:
        return {}

    voiced_f0 = pitch_values[pitch_values > 0]
    if len(voiced_f0) < 10:
        return {}

    semitones = np.array([_hz_to_semitones(hz, pitch_median) for hz in voiced_f0])

    pitch_mean_hz = float(np.mean(voiced_f0))
    pitch_std_hz = float(np.std(voiced_f0))
    pitch_slope = float(np.polyfit(range(len(semitones)), semitones, 1)[0])
    pitch_stability = 1.0 / (1.0 + float(np.std(semitones)))
    pitch_diffs = np.diff(semitones)
    pitch_dynamics = float(np.mean(np.abs(pitch_diffs)))
    reset_threshold = 3.0
    pitch_resets = int(np.sum(pitch_diffs < -reset_threshold))

    terminal_ratios = _compute_terminal_intonation_ratios(
        pitch_values,
        pitch_median,
        frame_duration,
        silence_frames=silence_frames,
        vad_result=vad_result,
    )
    terminal_slope = terminal_ratios.get("terminal_slope", 0.0)
    terminal_rising = terminal_ratios.get("terminal_rising", 0.0)
    terminal_falling = terminal_ratios.get("terminal_falling", 0.0)
    terminal_rising_ratio = terminal_ratios.get("terminal_rising_ratio")
    terminal_falling_ratio = terminal_ratios.get("terminal_falling_ratio")

    result: dict[str, float] = {
        "pitch_mean_hz": round(pitch_mean_hz, 1),
        "pitch_median_hz": round(pitch_median, 1),
        "pitch_std_hz": round(pitch_std_hz, 1),
        "pitch_slope": round(pitch_slope, 3),
        "pitch_stability": round(pitch_stability, 3),
        "pitch_dynamics": round(pitch_dynamics, 3),
        "pitch_resets": pitch_resets,
        "terminal_slope": round(terminal_slope, 3),
        "terminal_rising": terminal_rising,
        "terminal_falling": terminal_falling,
    }
    if terminal_rising_ratio is not None:
        result["terminal_rising_ratio"] = round(terminal_rising_ratio, 2)
    if terminal_falling_ratio is not None:
        result["terminal_falling_ratio"] = round(terminal_falling_ratio, 2)

    return result


def _compute_terminal_intonation_ratios(
    pitch_values: np.ndarray,
    pitch_median: float,
    frame_duration: float,
    *,
    silence_frames: np.ndarray | None = None,
    vad_result: VADResult | None = None,
) -> dict[str, float]:
    """
    Compute rising/falling ratios across phrase endings using VAD or silence frames.

    Returns null-equivalent omission for ratios when fewer than 2 reliable endings.
    """
    if pitch_median <= 0 or len(pitch_values) == 0:
        return {}

    terminal_frames = max(int(TERMINAL_WINDOW_MS / 1000 / frame_duration), 3)
    phrase_end_frames: list[int] = []

    if vad_result is not None and vad_result.speech_segments:
        for segment in vad_result.speech_segments:
            end_frame = int((segment.end_ms / 1000.0) / frame_duration)
            if 0 < end_frame < len(pitch_values):
                phrase_end_frames.append(end_frame)
    elif silence_frames is not None and len(silence_frames) > 0:
        i = 0
        while i < len(silence_frames):
            if silence_frames[i]:
                i += 1
                continue
            start = i
            while i < len(silence_frames) and not silence_frames[i]:
                i += 1
            end = i
            if end - start >= terminal_frames * 2:
                phrase_end_frames.append(end)

    rises = 0
    falls = 0
    measured = 0
    last_slope = 0.0

    for end_frame in phrase_end_frames:
        start_frame = max(0, end_frame - terminal_frames * 2)
        segment_f0 = pitch_values[start_frame:end_frame]
        voiced = segment_f0[segment_f0 > 0]
        if len(voiced) < terminal_frames:
            continue

        tail = voiced[-terminal_frames:]
        semitones = np.array([_hz_to_semitones(hz, pitch_median) for hz in tail])
        if len(semitones) < 2:
            continue

        slope = float(np.polyfit(range(len(semitones)), semitones, 1)[0])
        last_slope = slope
        measured += 1
        if slope > 0.5:
            rises += 1
        elif slope < -0.5:
            falls += 1

    result: dict[str, float] = {
        "terminal_slope": last_slope,
        "terminal_rising": 1.0 if last_slope > 0.5 else 0.0,
        "terminal_falling": 1.0 if last_slope < -0.5 else 0.0,
    }
    if measured >= 2:
        result["terminal_rising_ratio"] = rises / measured
        result["terminal_falling_ratio"] = falls / measured

    return result


def _analyze_energy_contour(
    intensity_values: np.ndarray,
    frame_duration: float,
) -> dict[str, float]:
    """Enhanced energy contour analysis including emphasis bursts and projection proxy."""
    if len(intensity_values) == 0:
        return {}
    
    # Basic energy statistics
    energy_mean = float(np.mean(intensity_values))
    energy_peak = float(np.max(intensity_values))
    energy_std = float(np.std(intensity_values))
    
    # Energy contour (rate of change)
    energy_diffs = np.diff(intensity_values)
    energy_slope = float(np.polyfit(range(len(intensity_values)), intensity_values, 1)[0])
    
    # Dynamic emphasis (local peaks relative to surrounding)
    window_size = max(5, len(intensity_values) // 20)
    local_peaks = []
    for i in range(window_size, len(intensity_values) - window_size):
        local_window = intensity_values[i - window_size : i + window_size + 1]
        if intensity_values[i] == np.max(local_window):
            local_peaks.append(intensity_values[i])
    
    dynamic_emphasis = float(np.mean(local_peaks)) / energy_mean if local_peaks and energy_mean > 0 else 0.0
    
    # Loudness stability (inverse of variation)
    loudness_stability = 1.0 / (1.0 + energy_std / energy_mean) if energy_mean > 0 else 0.0
    
    # Emphasis bursts (sudden energy increases)
    emphasis_threshold = energy_mean + 2 * energy_std
    emphasis_bursts = int(np.sum(intensity_values > emphasis_threshold))
    
    # Projection proxy (sustained high energy segments)
    projection_threshold = energy_mean + energy_std
    projection_segments = 0
    in_projection = False
    for val in intensity_values:
        if val > projection_threshold:
            if not in_projection:
                projection_segments += 1
                in_projection = True
        else:
            in_projection = False
    
    # Energy variation coefficient (normalized by mean)
    energy_cv = energy_std / energy_mean if energy_mean > 0 else 0.0
    
    return {
        "energy_mean": round(energy_mean, 2),
        "energy_peak": round(energy_peak, 2),
        "energy_std": round(energy_std, 2),
        "energy_slope": round(energy_slope, 4),
        "dynamic_emphasis": round(dynamic_emphasis, 3),
        "loudness_stability": round(loudness_stability, 3),
        "emphasis_bursts": emphasis_bursts,
        "projection_segments": projection_segments,
        "energy_cv": round(energy_cv, 3),
    }


def _estimate_terminal_rise_ratio(
    pitch,
    intensity,
    silence_frames: np.ndarray,
    frame_duration: float,
    pitch_median: float,
) -> float | None:
    if pitch_median <= 0:
        return None

    pitch_values = pitch.selected_array["frequency"]
    rises = 0
    endings = 0
    terminal_frames = max(int(TERMINAL_WINDOW_MS / 1000 / frame_duration), 3)

    i = 0
    while i < len(silence_frames):
        if silence_frames[i]:
            i += 1
            continue

        start = i
        while i < len(silence_frames) and not silence_frames[i]:
            i += 1
        end = i
        if end - start < terminal_frames * 2:
            continue

        endings += 1
        segment_f0 = pitch_values[start:end]
        voiced = segment_f0[segment_f0 > 0]
        if len(voiced) < terminal_frames:
            continue
        tail = voiced[-terminal_frames:]
        slope = float(tail[-1] - tail[0])
        if slope > 0:
            rises += 1

    if endings == 0:
        return None
    return round(rises / endings, 2)


def _mid_phrase_pause_rate(
    words: list[TranscriptWord],
    pauses: list[float],
    frame_duration: float,
    silence_frames: np.ndarray,
) -> float | None:
    if not words or len(words) < 2:
        return None

    gaps: list[tuple[TranscriptWord, float]] = []
    for left, right in zip(words, words[1:]):
        gap_ms = right.start_ms - left.end_ms
        if gap_ms >= PAUSE_MIN_MS:
            gaps.append((left, gap_ms / 1000.0))

    if not gaps:
        return 0.0

    mid_phrase = sum(
        1 for left, _ in gaps if not re.search(r"[.!?]$", left.text.strip())
    )
    return round(mid_phrase / len(gaps), 2)


def _extract_hnr(snd: parselmouth.Sound) -> float | None:
    try:
        harmonicity = snd.to_harmonicity()
        values = harmonicity.values[harmonicity.values != -200]
        if len(values) == 0:
            return None
        return float(np.mean(values))
    except Exception:
        return None


def _extract_jitter_shimmer(snd: parselmouth.Sound, pitch) -> tuple[float | None, float | None]:
    try:
        point_process = parselmouth.praat.call(pitch, "To PointProcess")
        jitter = parselmouth.praat.call(
            point_process,
            "Get jitter (local)",
            0,
            0,
            0.0001,
            0.02,
            1.3,
        )
        shimmer = parselmouth.praat.call(
            [snd, point_process],
            "Get shimmer (local)",
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6,
        )
        return float(jitter), float(shimmer)
    except Exception:
        return None, None


def _analyze_voice_quality(
    snd: parselmouth.Sound,
    pitch,
    intensity,
    audio_usable: bool,
    duration: float,
) -> dict[str, float]:
    """Enhanced voice quality analysis including voicing ratio, breathiness, and strain proxies."""
    if not audio_usable or duration < 1.0:
        return {}
    
    try:
        # Voicing ratio (proportion of voiced frames)
        pitch_values = pitch.selected_array["frequency"]
        voiced_frames = np.sum(pitch_values > 0)
        total_frames = len(pitch_values)
        voicing_ratio = voiced_frames / total_frames if total_frames > 0 else 0.0
        
        # Voice breaks (unvoiced segments within speech)
        voice_breaks = 0
        in_voiced = False
        for val in pitch_values:
            if val > 0:
                if not in_voiced:
                    in_voiced = True
            else:
                if in_voiced:
                    voice_breaks += 1
                    in_voiced = False
        
        # Harmonicity (HNR) - already extracted, but include in quality dict
        hnr = _extract_hnr(snd)
        
        # Jitter and shimmer
        jitter, shimmer = _extract_jitter_shimmer(snd, pitch)
        
        # Breathiness proxy (ratio of high-frequency energy to total energy)
        try:
            # Spectral tilt as breathiness indicator
            spectrum = snd.to_spectrum()
            spectral_values = spectrum.values[0] if len(spectrum.values) > 0 else np.array([])
            if len(spectral_values) > 10:
                low_freq_energy = float(np.mean(spectral_values[:len(spectral_values)//4]))
                high_freq_energy = float(np.mean(spectral_values[-len(spectral_values)//4:]))
                spectral_tilt = high_freq_energy / low_freq_energy if low_freq_energy > 0 else 0.0
                breathiness_proxy = spectral_tilt
            else:
                breathiness_proxy = 0.0
        except Exception:
            breathiness_proxy = 0.0
        
        # Strain proxy (combination of high jitter, shimmer, and reduced voicing)
        strain_components = []
        if jitter is not None and jitter > 0.02:
            strain_components.append(1.0)
        if shimmer is not None and shimmer > 0.05:
            strain_components.append(1.0)
        if voicing_ratio < 0.7:
            strain_components.append(1.0)
        strain_proxy = sum(strain_components) / len(strain_components) if strain_components else 0.0
        
        # CPP (Cepstral Peak Prominence) - simplified proxy using spectral peakiness
        try:
            spectrum = snd.to_spectrum()
            spectral_values = spectrum.values[0] if len(spectrum.values) > 0 else np.array([])
            if len(spectral_values) > 0:
                spectral_mean = float(np.mean(spectral_values))
                spectral_peak = float(np.max(spectral_values))
                cpp_proxy = (spectral_peak - spectral_mean) / spectral_mean if spectral_mean > 0 else 0.0
            else:
                cpp_proxy = 0.0
        except Exception:
            cpp_proxy = 0.0
        
        quality_dict = {
            "voicing_ratio": round(voicing_ratio, 3),
            "voice_breaks": voice_breaks,
            "breathiness_proxy": round(breathiness_proxy, 3),
            "strain_proxy": round(strain_proxy, 3),
            "cpp_proxy": round(cpp_proxy, 3),
        }
        
        if hnr is not None:
            quality_dict["hnr"] = round(hnr, 2)
        if jitter is not None:
            quality_dict["jitter"] = round(jitter, 5)
        if shimmer is not None:
            quality_dict["shimmer"] = round(shimmer, 5)
        
        return quality_dict
        
    except Exception:
        return {}


def _sliding_window_features(
    words: list[TranscriptWord],
    duration_ms: int,
    pitch,
    intensity,
    silence_frames: np.ndarray,
    frame_duration: float,
    pitch_median: float,
) -> list[WindowFeature]:
    if duration_ms < WINDOW_MS:
        return []

    pitch_values = pitch.selected_array["frequency"]
    intensity_values = intensity.values[0]
    windows: list[WindowFeature] = []

    for start_ms in range(0, max(duration_ms - WINDOW_MS, 0) + 1, HOP_MS):
        end_ms = start_ms + WINDOW_MS
        start_s = start_ms / 1000.0
        end_s = end_ms / 1000.0

        start_frame = int(start_s / frame_duration)
        end_frame = min(int(end_s / frame_duration), len(silence_frames))

        window_words = [w for w in words if w.start_ms >= start_ms and w.end_ms <= end_ms]
        word_count = len(window_words)
        filler_count = sum(1 for w in window_words if w.is_filler)
        filler_rate = filler_count / max(word_count, 1)
        window_seconds = max((end_ms - start_ms) / 1000.0, 0.1)
        wpm = (word_count / window_seconds) * 60

        segment_f0 = pitch_values[start_frame:end_frame]
        voiced = segment_f0[segment_f0 > 0]
        if pitch_median > 0 and len(voiced) > 1:
            semitones = np.array([_hz_to_semitones(hz, pitch_median) for hz in voiced])
            pitch_stdev = float(np.std(semitones))
            # Enhanced: pitch dynamics (rate of change)
            pitch_diffs = np.diff(semitones)
            pitch_dynamics = float(np.mean(np.abs(pitch_diffs))) if len(pitch_diffs) > 0 else 0.0
        else:
            pitch_stdev = 0.0
            pitch_dynamics = 0.0

        loudness_slice = intensity_values[start_frame:end_frame]
        loudness_stdev = float(np.std(loudness_slice)) if len(loudness_slice) else 0.0
        loudness_mean = float(np.mean(loudness_slice)) if len(loudness_slice) else 0.0
        # Enhanced: energy coefficient of variation
        energy_cv = (loudness_stdev / loudness_mean) if loudness_mean > 0 else 0.0
        # Enhanced: dynamic emphasis (local peaks)
        dynamic_emphasis = 0.0
        if len(loudness_slice) > 10:
            local_max = float(np.max(loudness_slice))
            dynamic_emphasis = (local_max / loudness_mean) if loudness_mean > 0 else 0.0
        
        pause_ms = float(np.sum(silence_frames[start_frame:end_frame]) * frame_duration * 1000)
        
        # Enhanced: voicing ratio
        voicing_ratio = len(voiced) / len(segment_f0) if len(segment_f0) > 0 else 0.0
        
        # Enhanced: articulation consistency (word duration variability)
        articulation_consistency = 0.5
        if len(window_words) > 2:
            word_durs = [w.end_ms - w.start_ms for w in window_words if w.end_ms > w.start_ms]
            if word_durs:
                dur_std = float(np.std(word_durs))
                dur_mean = float(np.mean(word_durs))
                if dur_mean > 0:
                    articulation_consistency = 1.0 / (1.0 + dur_std / dur_mean)

        monotone = pitch_stdev < 1.2 and loudness_stdev < 4
        rushing = wpm > 175

        # Enhanced: hesitation cluster detection. This is deliberately separate
        # from lexical fillers: it captures clustered timing disruption, not
        # every pause and not every silence.
        lexical_disruption = filler_rate > 0.08 and pause_ms > 250
        acoustic_search = (
            250 <= pause_ms <= 1300
            and word_count >= 2
            and (voicing_ratio < 0.72 or articulation_consistency < 0.62 or rushing)
        )
        hesitation_cluster = lexical_disruption or acoustic_search

        hesitation_penalty = 0.08 if acoustic_search else 0.0
        command = _clamp01(0.7 - filler_rate * 2 - (0.15 if rushing else 0) - hesitation_penalty)
        clarity = _clamp01(0.75 - filler_rate * 2.5 - hesitation_penalty * 0.5)
        composure = _clamp01(0.8 - pause_ms / 1200 - filler_rate - hesitation_penalty)
        presence = _clamp01((pitch_stdev / 4 + loudness_stdev / 20) / 2)

        windows.append(
            WindowFeature(
                start_ms=start_ms,
                end_ms=end_ms,
                command_score=round(command, 2),
                clarity_score=round(clarity, 2),
                composure_score=round(composure, 2),
                presence_score=round(presence, 2),
                filler_rate=round(filler_rate, 3),
                wpm=round(wpm, 1),
                pitch_stdev_semitones=round(pitch_stdev, 2),
                loudness_stdev_db=round(loudness_stdev, 2),
                pause_ms=round(pause_ms, 1),
                monotone=monotone,
                rushing=rushing,
                pitch_dynamics=round(pitch_dynamics, 3),
                energy_cv=round(energy_cv, 3),
                dynamic_emphasis=round(dynamic_emphasis, 3),
                voicing_ratio=round(voicing_ratio, 3),
                hesitation_cluster=hesitation_cluster,
                articulation_consistency=round(articulation_consistency, 3),
            )
        )

    return windows


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _window_mean_fillers(windows: list[WindowFeature]) -> float:
    if not windows:
        return 0.0
    return float(np.mean([w.filler_rate for w in windows]))


def _count_confidence_drops(windows: list[WindowFeature]) -> int:
    if len(windows) < 3:
        return 0
    drops = 0
    for index in range(1, len(windows) - 1):
        prev_score = windows[index - 1].composure_score
        cur_score = windows[index].composure_score
        next_score = windows[index + 1].composure_score
        if cur_score < prev_score - 0.15 and cur_score < next_score - 0.1:
            drops += 1
    return drops


def _build_derived(
    voice_metrics: VoiceMetrics,
    delivery: dict,
    confidence_drop_count: int,
) -> DerivedMetrics:
    pitch_range = voice_metrics.get("f0_range_semitones", 0)
    energy_variation = voice_metrics.get("energy_variation", 0)
    pause_frequency = voice_metrics.get("pause_frequency", 0)
    filler_density = delivery.get("filler_density", 0)

    monotony_index = None
    if pitch_range or energy_variation:
        flat_pitch = max(0.0, 1.0 - min(pitch_range / 8, 1.0))
        flat_energy = max(0.0, 1.0 - min(energy_variation / 15, 1.0))
        monotony_index = round((flat_pitch + flat_energy) / 2, 2)

    hesitation_cluster = round(min(1.0, pause_frequency * 1.5 + filler_density * 8), 2)

    dynamic_emphasis = None
    if pitch_range or energy_variation:
        dynamic_emphasis = round(
            min(1.0, (pitch_range / 10 + energy_variation / 20) / 2),
            2,
        )

    speech_continuity = round(max(0.0, min(1.0, voice_metrics.get("speech_density", 0.8))), 2)

    return DerivedMetrics(
        monotony_index=monotony_index,
        hesitation_cluster_score=hesitation_cluster,
        dynamic_emphasis_score=dynamic_emphasis,
        speech_continuity_score=speech_continuity,
        confidence_drop_count=confidence_drop_count,
    )
