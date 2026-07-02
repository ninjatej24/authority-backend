"""Voice Activity Detection using WebRTC VAD for precise speech segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import webrtcvad

    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False

VAD_AGGRESSIVENESS = 2  # 0-3, 2 is balanced for speech analysis
FRAME_DURATION_MS = 30  # WebRTC VAD supports 10, 20, 30 ms
TARGET_SAMPLE_RATE = 16000
WEBRTC_SAMPLE_RATES = (8000, 16000, 32000, 48000)
MIN_SPEECH_DURATION_MS = 250  # Minimum duration to consider as speech
MIN_PAUSE_DURATION_MS = 200  # Minimum duration to consider as pause


class SegmentType(Enum):
    SPEECH = "speech"
    SILENCE = "silence"


@dataclass
class SpeechSegment:
    """Represents a contiguous speech or silence segment."""

    start_ms: int
    end_ms: int
    duration_ms: int
    segment_type: SegmentType
    confidence: float = 1.0


@dataclass
class VADResult:
    """Complete VAD analysis results."""

    segments: list[SpeechSegment]
    speech_segments: list[SpeechSegment]
    silence_segments: list[SpeechSegment]
    speech_ratio: float
    total_speech_duration_ms: int
    total_silence_duration_ms: int
    pause_durations_ms: list[float]
    long_pauses_ms: list[float]
    mid_sentence_pauses_ms: list[float]
    end_of_sentence_pauses_ms: list[float]
    avg_pause_duration_ms: float
    pause_frequency_per_minute: float
    vad_backend: str = "unknown"


def empty_vad_result(total_duration_ms: int = 0, backend: str = "empty_fallback") -> VADResult:
    total_duration_ms = max(int(total_duration_ms), 0)
    return VADResult(
        segments=[],
        speech_segments=[],
        silence_segments=[],
        speech_ratio=0.0,
        total_speech_duration_ms=0,
        total_silence_duration_ms=total_duration_ms,
        pause_durations_ms=[],
        long_pauses_ms=[],
        mid_sentence_pauses_ms=[],
        end_of_sentence_pauses_ms=[],
        avg_pause_duration_ms=0.0,
        pause_frequency_per_minute=0.0,
        vad_backend=backend,
    )


def _empty_vad_result(total_duration_ms: int = 0, backend: str = "empty_fallback") -> VADResult:
    return empty_vad_result(total_duration_ms, backend)


def _resample_int16(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear resample mono int16 PCM to target_rate."""
    if source_rate == target_rate or len(samples) == 0:
        return samples

    duration_s = len(samples) / source_rate
    target_length = max(int(round(duration_s * target_rate)), 1)
    source_times = np.linspace(0, duration_s, num=len(samples), endpoint=False)
    target_times = np.linspace(0, duration_s, num=target_length, endpoint=False)
    resampled = np.interp(target_times, source_times, samples.astype(np.float64))
    return np.clip(np.round(resampled), -32768, 32767).astype(np.int16)


def prepare_pcm_samples(
    samples: np.ndarray,
    sample_rate: int,
    *,
    target_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """
    Convert Parselmouth float or mixed PCM to mono int16 at a WebRTC-compatible rate.

    WebRTC VAD accepts 8/16/32/48 kHz 16-bit mono PCM frames.
    """
    if samples.ndim > 1:
        samples = samples[0] if samples.shape[0] == 1 else np.mean(samples, axis=0)

    if samples.dtype in (np.float32, np.float64):
        clipped = np.clip(samples, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)
    elif samples.dtype != np.int16:
        pcm = samples.astype(np.int16)
    else:
        pcm = samples

    if sample_rate not in WEBRTC_SAMPLE_RATES:
        pcm = _resample_int16(pcm, sample_rate, target_rate)
        sample_rate = target_rate

    return pcm, sample_rate


def _frame_generator(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Generate audio frames for VAD processing."""
    frame_size = int(sample_rate * FRAME_DURATION_MS / 1000)
    offset = 0
    while offset + frame_size <= len(samples):
        yield samples[offset : offset + frame_size]
        offset += frame_size


def _collect_segments_webrtc(
    vad: webrtcvad.Vad,
    samples: np.ndarray,
    sample_rate: int,
) -> list[tuple[int, int, bool]]:
    """Run WebRTC VAD and collect speech segments as (start_ms, end_ms, is_speech)."""
    frames = list(_frame_generator(samples, sample_rate))
    if not frames:
        return []

    triggered = False
    num_voiced = 0
    segments: list[tuple[int, int, bool]] = []

    for i, frame in enumerate(frames):
        is_speech = vad.is_speech(frame.tobytes(), sample_rate)

        if not triggered:
            if is_speech:
                triggered = True
                num_voiced = 1
            else:
                num_voiced = 0
        else:
            if is_speech:
                num_voiced += 1
            else:
                if num_voiced < 3:
                    triggered = False
                else:
                    triggered = False
                    start_ms = max(0, (i - num_voiced) * FRAME_DURATION_MS)
                    end_ms = (i + 1) * FRAME_DURATION_MS
                    segments.append((start_ms, end_ms, True))
                    num_voiced = 0

    if triggered and num_voiced >= 3:
        start_ms = max(0, (len(frames) - num_voiced) * FRAME_DURATION_MS)
        end_ms = len(frames) * FRAME_DURATION_MS
        segments.append((start_ms, end_ms, True))

    return segments


def _collect_segments_energy(
    samples: np.ndarray,
    sample_rate: int,
) -> list[tuple[int, int, bool]]:
    """Energy-based VAD fallback with frame classification and hangover."""
    frame_size = max(int(sample_rate * FRAME_DURATION_MS / 1000), 1)
    frame_ms = int(round(frame_size / sample_rate * 1000))
    energies: list[float] = []

    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if len(frame) == 0:
            continue
        energies.append(float(np.mean(frame.astype(np.float64) ** 2)))

    if not energies:
        return []

    threshold = float(np.median(energies)) * 0.5
    if threshold <= 0:
        threshold = float(np.mean(energies)) * 0.5

    voiced = [energy > threshold for energy in energies]
    segments: list[tuple[int, int, bool]] = []
    in_speech = False
    speech_start = 0
    hangover = 0

    for index, is_voiced in enumerate(voiced):
        if is_voiced:
            if not in_speech:
                in_speech = True
                speech_start = index
            hangover = 0
        elif in_speech:
            hangover += 1
            if hangover >= 3:
                end_index = index - hangover + 1
                start_ms = speech_start * frame_ms
                end_ms = end_index * frame_ms
                if end_ms - start_ms >= MIN_SPEECH_DURATION_MS:
                    segments.append((start_ms, end_ms, True))
                in_speech = False
                hangover = 0

    if in_speech:
        start_ms = speech_start * frame_ms
        end_ms = len(voiced) * frame_ms
        if end_ms - start_ms >= MIN_SPEECH_DURATION_MS:
            segments.append((start_ms, end_ms, True))

    return segments


def _merge_adjacent_segments(
    segments: list[tuple[int, int, bool]],
    max_gap_ms: int = 150,
) -> list[tuple[int, int, bool]]:
    """Merge segments that are close together."""
    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        last_start, last_end, last_is_speech = merged[-1]
        curr_start, curr_end, curr_is_speech = seg

        if curr_is_speech == last_is_speech and (curr_start - last_end) <= max_gap_ms:
            merged[-1] = (last_start, max(last_end, curr_end), last_is_speech)
        else:
            merged.append(seg)

    return merged


def _filter_short_segments(
    segments: list[tuple[int, int, bool]],
    min_duration_ms: int,
) -> list[tuple[int, int, bool]]:
    """Remove segments shorter than minimum duration."""
    return [
        (start, end, is_speech)
        for start, end, is_speech in segments
        if (end - start) >= min_duration_ms
    ]


def _build_complete_timeline(
    speech_segments: list[tuple[int, int, bool]],
    total_duration_ms: int,
) -> list[tuple[int, int, bool]]:
    """Build complete timeline with both speech and silence segments."""
    if not speech_segments:
        return [(0, total_duration_ms, False)]

    timeline: list[tuple[int, int, bool]] = []
    last_end = 0

    for start, end, is_speech in sorted(speech_segments, key=lambda x: x[0]):
        start = max(0, min(start, total_duration_ms))
        end = max(start, min(end, total_duration_ms))
        if end <= start:
            continue
        if start > last_end:
            timeline.append((last_end, start, False))
        timeline.append((start, end, True))
        last_end = end

    if last_end < total_duration_ms:
        timeline.append((last_end, total_duration_ms, False))

    return timeline


def _classify_pauses(
    silence_segments: list[tuple[int, int, bool]],
    transcript_words: list | None = None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Classify pauses by type: all, long, mid-sentence, end-of-sentence."""
    all_pauses = [end - start for start, end, _ in silence_segments if (end - start) >= MIN_PAUSE_DURATION_MS]
    long_pauses = [duration for duration in all_pauses if duration >= 500]

    mid_sentence_pauses: list[float] = []
    end_of_sentence_pauses: list[float] = []

    if transcript_words:
        for pause_start, pause_end, _ in silence_segments:
            pause_duration = pause_end - pause_start
            if pause_duration < MIN_PAUSE_DURATION_MS:
                continue

            pause_mid = (pause_start + pause_end) / 2
            before_words = [w for w in transcript_words if w.end_ms <= pause_mid]
            after_words = [w for w in transcript_words if w.start_ms >= pause_mid]

            if before_words and after_words:
                last_word_before = before_words[-1]
                is_sentence_ending = any(
                    last_word_before.text.strip().endswith(punct)
                    for punct in [".", "!", "?"]
                )
                if is_sentence_ending or pause_duration >= 600:
                    end_of_sentence_pauses.append(pause_duration)
                else:
                    mid_sentence_pauses.append(pause_duration)
            else:
                mid_sentence_pauses.append(pause_duration)
    else:
        for pause_duration in all_pauses:
            if pause_duration >= 600:
                end_of_sentence_pauses.append(pause_duration)
            else:
                mid_sentence_pauses.append(pause_duration)

    return all_pauses, long_pauses, mid_sentence_pauses, end_of_sentence_pauses


def _finalize_vad_result(
    timeline: list[tuple[int, int, bool]],
    total_duration_ms: int,
    transcript_words: list | None,
    backend: str,
) -> VADResult:
    """Build VADResult from a speech/silence timeline."""
    speech_segments_list = [
        SpeechSegment(start, end, end - start, SegmentType.SPEECH)
        for start, end, is_speech in timeline
        if is_speech
    ]
    silence_segments_list = [
        SpeechSegment(start, end, end - start, SegmentType.SILENCE)
        for start, end, is_speech in timeline
        if not is_speech
    ]
    segments = [
        SpeechSegment(
            start,
            end,
            end - start,
            SegmentType.SPEECH if is_speech else SegmentType.SILENCE,
        )
        for start, end, is_speech in timeline
    ]

    total_speech_ms = sum(seg.duration_ms for seg in speech_segments_list)
    total_silence_ms = sum(seg.duration_ms for seg in silence_segments_list)
    speech_ratio = total_speech_ms / total_duration_ms if total_duration_ms > 0 else 0.0

    silence_tuples = [(seg.start_ms, seg.end_ms, False) for seg in silence_segments_list]
    all_pauses, long_pauses, mid_sentence_pauses, end_of_sentence_pauses = _classify_pauses(
        silence_tuples,
        transcript_words,
    )
    avg_pause = sum(all_pauses) / len(all_pauses) if all_pauses else 0.0
    speech_minutes = total_speech_ms / 60000
    pause_frequency = len(all_pauses) / speech_minutes if speech_minutes > 0 else 0.0

    return VADResult(
        segments=segments,
        speech_segments=speech_segments_list,
        silence_segments=silence_segments_list,
        speech_ratio=speech_ratio,
        total_speech_duration_ms=int(total_speech_ms),
        total_silence_duration_ms=int(total_silence_ms),
        pause_durations_ms=all_pauses,
        long_pauses_ms=long_pauses,
        mid_sentence_pauses_ms=mid_sentence_pauses,
        end_of_sentence_pauses_ms=end_of_sentence_pauses,
        avg_pause_duration_ms=avg_pause,
        pause_frequency_per_minute=pause_frequency,
        vad_backend=backend,
    )


def run_vad(
    samples: np.ndarray,
    sample_rate: int,
    transcript_words: list | None = None,
) -> VADResult:
    """
    Run Voice Activity Detection and return comprehensive segmentation.

    Args:
        samples: Audio samples (float [-1,1] or int16 PCM)
        sample_rate: Sample rate in Hz
        transcript_words: Optional transcript words for pause classification

    Returns:
        VADResult with complete segmentation and pause analysis
    """
    if len(samples) == 0 or sample_rate <= 0:
        return _empty_vad_result(backend="none")

    pcm, sample_rate = prepare_pcm_samples(samples, sample_rate)
    total_duration_ms = int(len(pcm) / sample_rate * 1000)
    if total_duration_ms <= 0:
        return _empty_vad_result(total_duration_ms, backend="none")

    if WEBRTC_VAD_AVAILABLE:
        try:
            vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
            raw_speech_segments = _collect_segments_webrtc(vad, pcm, sample_rate)
            backend = "webrtc"
        except Exception:
            raw_speech_segments = _collect_segments_energy(pcm, sample_rate)
            backend = "energy_fallback"
    else:
        raw_speech_segments = _collect_segments_energy(pcm, sample_rate)
        backend = "energy_fallback"

    merged_speech = _merge_adjacent_segments(raw_speech_segments)
    speech_segments = _filter_short_segments(merged_speech, MIN_SPEECH_DURATION_MS)
    if not speech_segments:
        return _empty_vad_result(total_duration_ms, backend="empty_fallback")

    timeline = _build_complete_timeline(speech_segments, total_duration_ms)

    return _finalize_vad_result(timeline, total_duration_ms, transcript_words, backend)


def build_silence_frame_mask(
    vad_result: VADResult,
    frame_duration: float,
    num_frames: int,
) -> np.ndarray:
    """Map VAD silence segments onto an intensity-frame boolean mask."""
    silence_frames = np.zeros(num_frames, dtype=bool)
    if frame_duration <= 0 or num_frames <= 0:
        return silence_frames

    for segment in vad_result.silence_segments:
        start_frame = int((segment.start_ms / 1000.0) / frame_duration)
        end_frame = int((segment.end_ms / 1000.0) / frame_duration)
        start_frame = max(0, min(start_frame, num_frames))
        end_frame = max(start_frame, min(end_frame, num_frames))
        silence_frames[start_frame:end_frame] = True

    return silence_frames


def pauses_from_vad(vad_result: VADResult) -> list[float]:
    """Return pause durations in seconds from VAD silence segments."""
    return [duration / 1000.0 for duration in vad_result.pause_durations_ms]
