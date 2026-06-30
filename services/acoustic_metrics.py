"""Deterministic acoustic and prosodic feature extraction."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import parselmouth

from schemas import DerivedMetrics, RawAcousticMetrics, TranscriptWord

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


@dataclass
class AcousticAnalysisResult:
    voice_metrics: VoiceMetrics
    raw: RawAcousticMetrics
    derived: DerivedMetrics
    windows: list[WindowFeature] = field(default_factory=list)
    speaking_seconds: float = 0.0


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
    silence_frames = intensity_values < threshold

    silence_duration = float(np.sum(silence_frames) * frame_duration)
    silence_ratio = silence_duration / duration

    pauses = _detect_pauses(silence_frames, frame_duration)
    pause_count = len(pauses)
    avg_pause = float(np.mean(pauses)) if pauses else 0.0
    longest_pause = float(max(pauses)) if pauses else 0.0
    pause_frequency = pause_count / duration
    speech_duration = max(duration - silence_duration, 0.01)
    speech_density = speech_duration / duration

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

    hnr = jitter = shimmer = None
    if audio_usable and duration >= 1.0 and len(voiced_f0) > 20:
        hnr = _extract_hnr(snd)
        jitter, shimmer = _extract_jitter_shimmer(snd, pitch)

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
        else:
            pitch_stdev = 0.0

        loudness_slice = intensity_values[start_frame:end_frame]
        loudness_stdev = float(np.std(loudness_slice)) if len(loudness_slice) else 0.0
        pause_ms = float(np.sum(silence_frames[start_frame:end_frame]) * frame_duration * 1000)

        monotone = pitch_stdev < 1.2 and loudness_stdev < 4
        rushing = wpm > 175

        command = _clamp01(0.7 - filler_rate * 2 - (0.15 if rushing else 0))
        clarity = _clamp01(0.75 - filler_rate * 2.5)
        composure = _clamp01(0.8 - pause_ms / 1200 - filler_rate)
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
