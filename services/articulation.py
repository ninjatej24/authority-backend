"""Articulation proxies using word timing consistency and speech precision metrics."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from schemas import TranscriptWord


@dataclass
class ArticulationAnalysis:
    """Articulation quality analysis results."""
    articulation_rate: float  # words per second of speech
    phoneme_timing_consistency: float  # proxy via word duration variability
    speech_precision: float  # inverse of word duration outliers
    word_duration_mean_ms: float
    word_duration_std_ms: float
    word_duration_cv: float  # coefficient of variation
    clarity_proxy: float  # based on consistent word durations
    articulation_stability: float  # consistency across recording


def _calculate_word_durations(words: list[TranscriptWord]) -> list[float]:
    """Calculate duration of each word in milliseconds."""
    return [word.end_ms - word.start_ms for word in words if word.end_ms > word.start_ms]


def _analyze_word_timing_consistency(word_durations: list[float]) -> dict[str, float]:
    """Analyze consistency of word timing as proxy for phoneme timing consistency."""
    if not word_durations:
        return {
            "mean": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "consistency": 0.5,
        }
    
    durations_array = np.array(word_durations)
    mean_dur = float(np.mean(durations_array))
    std_dur = float(np.std(durations_array))
    
    if mean_dur <= 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "consistency": 0.5,
        }
    
    cv = std_dur / mean_dur  # Coefficient of variation
    
    # Consistency is inverse of CV (lower CV = more consistent)
    consistency = 1.0 / (1.0 + cv)
    
    return {
        "mean": round(mean_dur, 1),
        "std": round(std_dur, 1),
        "cv": round(cv, 3),
        "consistency": round(consistency, 3),
    }


def _calculate_speech_precision(
    word_durations: list[float],
    expected_mean_ms: float = 250.0,
) -> float:
    """
    Calculate speech precision as inverse of outlier word durations.
    
    Words that are too short or too long suggest imprecise articulation.
    """
    if not word_durations:
        return 0.5
    
    durations_array = np.array(word_durations)
    
    # Count outliers (more than 2 std from expected mean)
    if len(durations_array) < 3:
        return 0.5
    
    actual_mean = float(np.mean(durations_array))
    actual_std = float(np.std(durations_array))
    
    if actual_std <= 0:
        return 0.5
    
    outliers = 0
    for dur in durations_array:
        if abs(dur - actual_mean) > 2 * actual_std:
            outliers += 1
    
    outlier_ratio = outliers / len(durations_array)
    precision = 1.0 - outlier_ratio
    
    return round(precision, 3)


def _calculate_clarity_proxy(
    word_durations: list[float],
    inter_word_gaps: list[float],
) -> float:
    """
    Calculate clarity proxy based on consistent word durations and appropriate gaps.
    
    Clear articulation typically shows consistent word timing without excessive gaps.
    """
    if not word_durations or not inter_word_gaps:
        return 0.5
    
    timing_stats = _analyze_word_timing_consistency(word_durations)
    timing_consistency = timing_stats["consistency"]
    
    # Penalize excessive gaps
    if inter_word_gaps:
        mean_gap = float(np.mean(inter_word_gaps))
        # Optimal gap is around 100-300ms
        if mean_gap < 100:
            gap_score = 0.7  # Too rushed
        elif mean_gap > 500:
            gap_score = 0.5  # Too slow
        else:
            gap_score = 1.0  # Good
    else:
        gap_score = 0.5
    
    clarity = (timing_consistency * 0.7 + gap_score * 0.3)
    
    return round(clarity, 3)


def _calculate_articulation_stability(
    word_durations: list[float],
    window_size: int = 5,
) -> float:
    """
    Calculate articulation stability across the recording.
    
    Measures whether articulation quality is consistent from beginning to end.
    """
    if len(word_durations) < window_size * 2:
        return 0.5
    
    # Split into windows and calculate consistency in each
    windows = []
    for i in range(0, len(word_durations) - window_size + 1, window_size):
        window = word_durations[i:i + window_size]
        if len(window) >= 3:
            stats = _analyze_word_timing_consistency(window)
            windows.append(stats["consistency"])
    
    if not windows:
        return 0.5
    
    # Stability is inverse of variation in consistency across windows
    consistency_array = np.array(windows)
    if len(consistency_array) < 2:
        return 0.5
    
    std_consistency = float(np.std(consistency_array))
    stability = 1.0 / (1.0 + std_consistency)
    
    return round(stability, 3)


def analyze_articulation(
    words: list[TranscriptWord],
    speech_duration_ms: int,
) -> ArticulationAnalysis:
    """
    Perform comprehensive articulation analysis.
    
    Args:
        words: List of transcript words with timestamps
        speech_duration_ms: Duration of speech (excluding silence)
    
    Returns:
        ArticulationAnalysis with articulation quality metrics
    """
    if not words:
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
    
    # Calculate word durations
    word_durations = _calculate_word_durations(words)
    
    if not word_durations:
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
    
    # Calculate inter-word gaps
    inter_word_gaps = []
    for i in range(len(words) - 1):
        gap = words[i + 1].start_ms - words[i].end_ms
        if gap > 0:
            inter_word_gaps.append(gap)
    
    # Articulation rate (words per second of speech)
    speech_duration_s = speech_duration_ms / 1000
    articulation_rate = len(words) / speech_duration_s if speech_duration_s > 0 else 0.0
    
    # Word timing statistics
    timing_stats = _analyze_word_timing_consistency(word_durations)
    phoneme_timing_consistency = timing_stats["consistency"]
    
    # Speech precision
    speech_precision = _calculate_speech_precision(word_durations)
    
    # Clarity proxy
    clarity_proxy = _calculate_clarity_proxy(word_durations, inter_word_gaps)
    
    # Articulation stability
    articulation_stability = _calculate_articulation_stability(word_durations)
    
    return ArticulationAnalysis(
        articulation_rate=round(articulation_rate, 2),
        phoneme_timing_consistency=phoneme_timing_consistency,
        speech_precision=speech_precision,
        word_duration_mean_ms=timing_stats["mean"],
        word_duration_std_ms=timing_stats["std"],
        word_duration_cv=timing_stats["cv"],
        clarity_proxy=clarity_proxy,
        articulation_stability=articulation_stability,
    )
