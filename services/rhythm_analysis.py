"""Rhythm analysis including cadence, hesitation windows, and rate changes."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

import numpy as np

from schemas import TranscriptWord


@dataclass
class RhythmAnalysis:
    """Comprehensive rhythm analysis results."""
    speech_rate: float  # syllables per second
    words_per_minute: float
    pause_cadence: float  # pauses per second of speech
    speech_continuity: float  # proportion of time spent speaking
    hesitation_windows: int
    rhythm_consistency: float  # inverse of rate variability
    burst_speaking_segments: int
    slow_down_segments: int
    speed_up_segments: int
    articulation_rate: float  # phonemes per second (proxy via syllables)


@dataclass
class RateChangeSegment:
    """Represents a segment with significant rate change."""
    start_ms: int
    end_ms: int
    duration_ms: int
    rate_type: Literal["burst", "slow_down", "speed_up"]
    rate_value: float
    baseline_rate: float


def _estimate_syllables(text: str) -> int:
    """Estimate syllable count from text (simplified heuristic)."""
    if not text:
        return 0
    
    text = text.lower()
    # Count vowel groups as syllable approximation
    vowels = "aeiouy"
    syllable_count = 0
    prev_char_was_vowel = False
    
    for char in text:
        if char in vowels:
            if not prev_char_was_vowel:
                syllable_count += 1
            prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False
    
    # Adjust for silent 'e' at end
    if text.endswith("e") and syllable_count > 1:
        syllable_count -= 1
    
    return max(1, syllable_count) if syllable_count > 0 else 0


def _analyze_rate_changes(
    words: list[TranscriptWord],
    window_ms: int = 3000,
    hop_ms: int = 1000,
) -> list[RateChangeSegment]:
    """Detect burst speaking, slow-down, and speed-up segments."""
    if not words or len(words) < 3:
        return []
    
    segments: list[RateChangeSegment] = []
    
    # Calculate baseline rate
    total_duration_ms = words[-1].end_ms - words[0].start_ms
    if total_duration_ms <= 0:
        return []
    
    baseline_wpm = (len(words) / total_duration_ms) * 60000
    
    # Analyze sliding windows
    for i in range(0, len(words) - 2):
        window_words = []
        window_start = words[i].start_ms
        
        for j in range(i, len(words)):
            if words[j].end_ms - window_start <= window_ms:
                window_words.append(words[j])
            else:
                break
        
        if len(window_words) < 2:
            continue
        
        window_duration = window_words[-1].end_ms - window_words[0].start_ms
        if window_duration <= 0:
            continue
        
        window_wpm = (len(window_words) / window_duration) * 60000
        
        # Classify rate change
        rate_ratio = window_wpm / baseline_wpm if baseline_wpm > 0 else 1.0
        
        if rate_ratio >= 1.4 and window_wpm >= 170:
            segments.append(
                RateChangeSegment(
                    start_ms=window_words[0].start_ms,
                    end_ms=window_words[-1].end_ms,
                    duration_ms=window_duration,
                    rate_type="burst",
                    rate_value=window_wpm,
                    baseline_rate=baseline_wpm,
                )
            )
        elif rate_ratio <= 0.6 and window_wpm <= 100:
            segments.append(
                RateChangeSegment(
                    start_ms=window_words[0].start_ms,
                    end_ms=window_words[-1].end_ms,
                    duration_ms=window_duration,
                    rate_type="slow_down",
                    rate_value=window_wpm,
                    baseline_rate=baseline_wpm,
                )
            )
        elif rate_ratio >= 1.25 and window_wpm >= 150:
            segments.append(
                RateChangeSegment(
                    start_ms=window_words[0].start_ms,
                    end_ms=window_words[-1].end_ms,
                    duration_ms=window_duration,
                    rate_type="speed_up",
                    rate_value=window_wpm,
                    baseline_rate=baseline_wpm,
                )
            )
    
    # Merge overlapping segments
    merged: list[RateChangeSegment] = []
    for seg in sorted(segments, key=lambda x: x.start_ms):
        if not merged:
            merged.append(seg)
        else:
            last = merged[-1]
            if seg.start_ms <= last.end_ms + 500:  # 500ms gap tolerance
                merged[-1] = RateChangeSegment(
                    start_ms=min(last.start_ms, seg.start_ms),
                    end_ms=max(last.end_ms, seg.end_ms),
                    duration_ms=max(last.end_ms, seg.start_ms) - min(last.start_ms, seg.start_ms),
                    rate_type=last.rate_type,  # Keep first type
                    rate_value=max(last.rate_value, seg.rate_value),
                    baseline_rate=last.baseline_rate,
                )
            else:
                merged.append(seg)
    
    return merged


def _detect_hesitation_windows(
    words: list[TranscriptWord],
    pause_threshold_ms: int = 500,
    window_ms: int = 2000,
) -> int:
    """Count windows with hesitation patterns (multiple short pauses)."""
    if not words or len(words) < 2:
        return 0
    
    hesitation_count = 0
    
    for i in range(len(words) - 1):
        gap = words[i + 1].start_ms - words[i].end_ms
        if gap >= pause_threshold_ms:
            # Check if this pause is followed by another short pause
            if i + 2 < len(words):
                next_gap = words[i + 2].start_ms - words[i + 1].end_ms
                if next_gap >= pause_threshold_ms:
                    hesitation_count += 1
    
    return hesitation_count


def _calculate_rhythm_consistency(
    words: list[TranscriptWord],
) -> float:
    """Calculate rhythm consistency based on inter-word interval variability."""
    if not words or len(words) < 3:
        return 0.5
    
    intervals = []
    for i in range(len(words) - 1):
        gap = words[i + 1].start_ms - words[i].end_ms
        if gap > 0:
            intervals.append(gap)
    
    if not intervals:
        return 0.5
    
    interval_std = float(np.std(intervals))
    interval_mean = float(np.mean(intervals))
    
    if interval_mean <= 0:
        return 0.5
    
    # Coefficient of variation - lower is more consistent
    cv = interval_std / interval_mean
    
    # Convert to consistency score (inverse of CV)
    consistency = 1.0 / (1.0 + cv)
    
    return round(consistency, 3)


def analyze_rhythm(
    words: list[TranscriptWord],
    transcript_text: str,
    speech_duration_ms: int,
    total_duration_ms: int,
) -> RhythmAnalysis:
    """
    Perform comprehensive rhythm analysis.
    
    Args:
        words: List of transcript words with timestamps
        transcript_text: Full transcript text
        speech_duration_ms: Duration of speech (excluding silence)
        total_duration_ms: Total recording duration
    
    Returns:
        RhythmAnalysis with comprehensive rhythm metrics
    """
    if not words:
        return RhythmAnalysis(
            speech_rate=0.0,
            words_per_minute=0.0,
            pause_cadence=0.0,
            speech_continuity=0.0,
            hesitation_windows=0,
            rhythm_consistency=0.5,
            burst_speaking_segments=0,
            slow_down_segments=0,
            speed_up_segments=0,
            articulation_rate=0.0,
        )
    
    # Basic rate metrics
    speech_duration_s = speech_duration_ms / 1000
    total_duration_s = total_duration_ms / 1000
    
    words_per_minute = (len(words) / speech_duration_s) * 60 if speech_duration_s > 0 else 0.0
    
    syllable_count = _estimate_syllables(transcript_text)
    speech_rate = syllable_count / speech_duration_s if speech_duration_s > 0 else 0.0
    articulation_rate = speech_rate  # Proxy: syllables/sec approximates phonemes/sec
    
    # Pause cadence (pauses per second of speech)
    pause_count = 0
    for i in range(len(words) - 1):
        gap = words[i + 1].start_ms - words[i].end_ms
        if gap >= 200:  # Minimum pause threshold
            pause_count += 1
    
    pause_cadence = pause_count / speech_duration_s if speech_duration_s > 0 else 0.0
    
    # Speech continuity
    speech_continuity = speech_duration_s / total_duration_s if total_duration_s > 0 else 0.0
    
    # Advanced rhythm features
    hesitation_windows = _detect_hesitation_windows(words)
    rhythm_consistency = _calculate_rhythm_consistency(words)
    
    # Rate change detection
    rate_changes = _analyze_rate_changes(words)
    burst_segments = sum(1 for seg in rate_changes if seg.rate_type == "burst")
    slow_down_segments = sum(1 for seg in rate_changes if seg.rate_type == "slow_down")
    speed_up_segments = sum(1 for seg in rate_changes if seg.rate_type == "speed_up")
    
    return RhythmAnalysis(
        speech_rate=round(speech_rate, 2),
        words_per_minute=round(words_per_minute, 1),
        pause_cadence=round(pause_cadence, 2),
        speech_continuity=round(speech_continuity, 3),
        hesitation_windows=hesitation_windows,
        rhythm_consistency=rhythm_consistency,
        burst_speaking_segments=burst_segments,
        slow_down_segments=slow_down_segments,
        speed_up_segments=speed_up_segments,
        articulation_rate=round(articulation_rate, 2),
    )
