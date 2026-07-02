"""Voice Activity Detection using WebRTC VAD for precise speech segmentation."""

from __future__ import annotations

import collections
import dataclasses
import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False

VAD_AGGRESSIVENESS = 2  # 0-3, 2 is balanced for speech analysis
FRAME_DURATION_MS = 30  # WebRTC VAD supports 10, 20, 30 ms
SAMPLE_RATE = 16000
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


def _frame_generator(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Generate audio frames for VAD processing."""
    frame_size = int(sample_rate * FRAME_DURATION_MS / 1000)
    offset = 0
    while offset + frame_size <= len(samples):
        yield samples[offset:offset + frame_size]
        offset += frame_size


def _collect_segments(
    vad: webrtcvad.Vad,
    samples: np.ndarray,
    sample_rate: int,
) -> list[tuple[int, int, bool]]:
    """Run VAD and collect speech/silence segments as (start_ms, end_ms, is_speech)."""
    frames = list(_frame_generator(samples, sample_rate))
    if not frames:
        return []
    
    num_voiced = 0
    triggered = False
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
                if num_voiced < 3:  # Short speech burst, treat as noise
                    triggered = False
                else:
                    triggered = False
                    start_ms = max(0, (i - num_voiced - 1) * FRAME_DURATION_MS)
                    end_ms = (i + 1) * FRAME_DURATION_MS
                    segments.append((start_ms, end_ms, True))
                    num_voiced = 0
    
    # Handle case where speech continues to end
    if triggered:
        start_ms = max(0, (len(frames) - num_voiced - 1) * FRAME_DURATION_MS)
        end_ms = len(frames) * FRAME_DURATION_MS
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
        if start > last_end:
            timeline.append((last_end, start, False))
        timeline.append((start, end, True))
        last_end = end
    
    if last_end < total_duration_ms:
        timeline.append((last_end, total_duration_ms, False))
    
    return timeline


def _classify_pauses(
    silence_segments: list[tuple[int, int, bool]],
    speech_segments: list[tuple[int, int, bool]],
    transcript_words: list | None = None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Classify pauses by type: all, long, mid-sentence, end-of-sentence."""
    all_pauses = [end - start for start, end, _ in silence_segments]
    long_pauses = [d for d in all_pauses if d >= 500]
    
    # Mid-sentence vs end-of-sentence classification
    # This is heuristic-based without punctuation from transcript
    mid_sentence_pauses: list[float] = []
    end_of_sentence_pauses: list[float] = []
    
    if transcript_words:
        # Use word timestamps to classify pauses
        for i, (pause_start, pause_end, _) in enumerate(silence_segments):
            pause_mid = (pause_start + pause_end) / 2
            
            # Find words before and after pause
            before_words = [w for w in transcript_words if w.end_ms <= pause_mid]
            after_words = [w for w in transcript_words if w.start_ms >= pause_mid]
            
            if before_words and after_words:
                last_word_before = before_words[-1]
                first_word_after = after_words[0]
                
                # Check if pause is between sentences (longer or follows sentence-ending word)
                is_sentence_ending = any(
                    last_word_before.text.strip().endswith(punct)
                    for punct in [".", "!", "?"]
                )
                
                if is_sentence_ending or (pause_end - pause_start) >= 600:
                    end_of_sentence_pauses.append(pause_end - pause_start)
                else:
                    mid_sentence_pauses.append(pause_end - pause_start)
            else:
                mid_sentence_pauses.append(pause_end - pause_start)
    else:
        # Fallback: classify by duration
        for pause_duration in all_pauses:
            if pause_duration >= 600:
                end_of_sentence_pauses.append(pause_duration)
            else:
                mid_sentence_pauses.append(pause_duration)
    
    return all_pauses, long_pauses, mid_sentence_pauses, end_of_sentence_pauses


def run_vad(
    samples: np.ndarray,
    sample_rate: int,
    transcript_words: list | None = None,
) -> VADResult:
    """
    Run Voice Activity Detection and return comprehensive segmentation.
    
    Args:
        samples: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        transcript_words: Optional list of transcript words with timestamps for pause classification
    
    Returns:
        VADResult with complete segmentation and pause analysis
    """
    if len(samples) == 0:
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
        )
    
    total_duration_ms = int(len(samples) / sample_rate * 1000)
    
    # Fallback if webrtcvad is not available
    if not WEBRTC_VAD_AVAILABLE:
        # Simple energy-based fallback
        frame_size = max(int(sample_rate * 0.02), 1)
        energies = []
        for start in range(0, len(samples), frame_size):
            frame = samples[start : start + frame_size]
            if len(frame) == 0:
                continue
            energies.append(float(np.mean(frame**2)))
        
        if not energies:
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
            )
        
        threshold = float(np.median(energies)) * 0.5
        speech_frames = [e for e in energies if e > threshold]
        speech_ratio = len(speech_frames) / len(energies) if energies else 0.0
        
        return VADResult(
            segments=[],
            speech_segments=[],
            silence_segments=[],
            speech_ratio=speech_ratio,
            total_speech_duration_ms=int(total_duration_ms * speech_ratio),
            total_silence_duration_ms=int(total_duration_ms * (1 - speech_ratio)),
            pause_durations_ms=[],
            long_pauses_ms=[],
            mid_sentence_pauses_ms=[],
            end_of_sentence_pauses_ms=[],
            avg_pause_duration_ms=0.0,
            pause_frequency_per_minute=0.0,
        )
    
    # Initialize VAD
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    
    # Collect speech segments
    raw_speech_segments = _collect_segments(vad, samples, sample_rate)
    
    # Merge adjacent segments
    merged_speech = _merge_adjacent_segments(raw_speech_segments)
    
    # Filter short segments
    speech_segments = _filter_short_segments(merged_speech, MIN_SPEECH_DURATION_MS)
    
    # Build complete timeline
    timeline = _build_complete_timeline(speech_segments, total_duration_ms)
    
    # Separate speech and silence
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
    
    # Convert to SpeechSegment objects for timeline
    segments = [
        SpeechSegment(start, end, end - start, SegmentType.SPEECH if is_speech else SegmentType.SILENCE)
        for start, end, is_speech in timeline
    ]
    
    # Calculate statistics
    total_speech_ms = sum(seg.duration_ms for seg in speech_segments_list)
    total_silence_ms = sum(seg.duration_ms for seg in silence_segments_list)
    speech_ratio = total_speech_ms / total_duration_ms if total_duration_ms > 0 else 0.0
    
    # Classify pauses
    silence_tuples = [(seg.start_ms, seg.end_ms, False) for seg in silence_segments_list]
    speech_tuples = [(seg.start_ms, seg.end_ms, True) for seg in speech_segments_list]
    
    (
        all_pauses,
        long_pauses,
        mid_sentence_pauses,
        end_of_sentence_pauses,
    ) = _classify_pauses(silence_tuples, speech_tuples, transcript_words)
    
    avg_pause = sum(all_pauses) / len(all_pauses) if all_pauses else 0.0
    
    # Calculate pause frequency (pauses per minute of speech)
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
    )
