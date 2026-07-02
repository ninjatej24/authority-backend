"""Tests for rhythm analysis module."""

import pytest

from schemas import TranscriptWord
from services.rhythm_analysis import RhythmAnalysis, analyze_rhythm, _estimate_syllables


def test_estimate_syllables():
    """Test syllable estimation."""
    assert _estimate_syllables("hello") == 2
    assert _estimate_syllables("world") == 1
    assert _estimate_syllables("testing") == 2
    assert _estimate_syllables("") == 0  # Empty string returns 0
    assert _estimate_syllables("the quick brown fox") > 0


def test_analyze_rhythm_with_words():
    """Test rhythm analysis with transcript words."""
    words = [
        TranscriptWord(text="hello", start_ms=0, end_ms=200),
        TranscriptWord(text="world", start_ms=300, end_ms=500),
        TranscriptWord(text="test", start_ms=600, end_ms=800),
    ]
    
    result = analyze_rhythm(
        words=words,
        transcript_text="hello world test",
        speech_duration_ms=800,
        total_duration_ms=1000,
    )
    
    assert isinstance(result, RhythmAnalysis)
    assert result.words_per_minute > 0
    assert result.speech_rate > 0
    assert result.pause_cadence >= 0
    assert result.speech_continuity > 0
    assert result.hesitation_windows >= 0
    assert result.rhythm_consistency >= 0
    assert result.burst_speaking_segments >= 0
    assert result.slow_down_segments >= 0
    assert result.speed_up_segments >= 0
    assert result.articulation_rate > 0


def test_analyze_rhythm_empty_words():
    """Test rhythm analysis with no words."""
    result = analyze_rhythm(
        words=[],
        transcript_text="",
        speech_duration_ms=1000,
        total_duration_ms=1000,
    )
    
    assert isinstance(result, RhythmAnalysis)
    assert result.words_per_minute == 0.0
    assert result.speech_rate == 0.0
    assert result.pause_cadence == 0.0
    assert result.hesitation_windows == 0


def test_analyze_rhythm_with_fillers():
    """Test rhythm analysis with filler words causing hesitation."""
    words = [
        TranscriptWord(text="um", start_ms=0, end_ms=100, is_filler=True),
        TranscriptWord(text="hello", start_ms=200, end_ms=400),
        TranscriptWord(text="uh", start_ms=500, end_ms=600, is_filler=True),
        TranscriptWord(text="world", start_ms=700, end_ms=900),
    ]
    
    result = analyze_rhythm(
        words=words,
        transcript_text="um hello uh world",
        speech_duration_ms=900,
        total_duration_ms=1000,
    )
    
    assert isinstance(result, RhythmAnalysis)
    assert result.hesitation_windows >= 0


def test_rhythm_analysis_structure():
    """Test RhythmAnalysis dataclass structure."""
    result = RhythmAnalysis(
        speech_rate=4.5,
        words_per_minute=150.0,
        pause_cadence=0.25,
        speech_continuity=0.8,
        hesitation_windows=2,
        rhythm_consistency=0.75,
        burst_speaking_segments=1,
        slow_down_segments=0,
        speed_up_segments=1,
        articulation_rate=4.5,
    )
    
    assert result.speech_rate == 4.5
    assert result.words_per_minute == 150.0
    assert result.pause_cadence == 0.25
    assert result.speech_continuity == 0.8
    assert result.hesitation_windows == 2
    assert result.rhythm_consistency == 0.75
    assert result.burst_speaking_segments == 1
    assert result.slow_down_segments == 0
    assert result.speed_up_segments == 1
