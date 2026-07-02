"""Tests for articulation analysis module."""

import pytest

from schemas import TranscriptWord
from services.articulation import ArticulationAnalysis, analyze_articulation


def test_analyze_articulation_with_words():
    """Test articulation analysis with transcript words."""
    words = [
        TranscriptWord(text="hello", start_ms=0, end_ms=200),
        TranscriptWord(text="world", start_ms=300, end_ms=500),
        TranscriptWord(text="test", start_ms=600, end_ms=800),
    ]
    
    result = analyze_articulation(words, speech_duration_ms=800)
    
    assert isinstance(result, ArticulationAnalysis)
    assert result.articulation_rate > 0
    assert result.phoneme_timing_consistency >= 0
    assert result.speech_precision >= 0
    assert result.word_duration_mean_ms > 0
    assert result.word_duration_std_ms >= 0
    assert result.word_duration_cv >= 0
    assert result.clarity_proxy >= 0
    assert result.articulation_stability >= 0


def test_analyze_articulation_empty_words():
    """Test articulation analysis with no words."""
    result = analyze_articulation([], speech_duration_ms=1000)
    
    assert isinstance(result, ArticulationAnalysis)
    assert result.articulation_rate == 0.0
    assert result.phoneme_timing_consistency == 0.5
    assert result.speech_precision == 0.5
    assert result.word_duration_mean_ms == 0.0


def test_analyze_articulation_consistent_timing():
    """Test articulation analysis with consistent word durations."""
    words = [
        TranscriptWord(text="one", start_ms=0, end_ms=200),
        TranscriptWord(text="two", start_ms=300, end_ms=500),
        TranscriptWord(text="three", start_ms=600, end_ms=800),
        TranscriptWord(text="four", start_ms=900, end_ms=1100),
    ]
    
    result = analyze_articulation(words, speech_duration_ms=1100)
    
    assert isinstance(result, ArticulationAnalysis)
    # Consistent timing should yield high consistency
    assert result.phoneme_timing_consistency >= 0.5
    assert result.articulation_stability >= 0.5


def test_analyze_articulation_variable_timing():
    """Test articulation analysis with variable word durations."""
    words = [
        TranscriptWord(text="short", start_ms=0, end_ms=100),
        TranscriptWord(text="medium", start_ms=200, end_ms=500),
        TranscriptWord(text="verylongword", start_ms=600, end_ms=1200),
    ]
    
    result = analyze_articulation(words, speech_duration_ms=1200)
    
    assert isinstance(result, ArticulationAnalysis)
    # Variable timing should yield lower consistency
    assert result.word_duration_std_ms > 0
    assert result.word_duration_cv > 0


def test_articulation_analysis_structure():
    """Test ArticulationAnalysis dataclass structure."""
    result = ArticulationAnalysis(
        articulation_rate=4.5,
        phoneme_timing_consistency=0.8,
        speech_precision=0.75,
        word_duration_mean_ms=250.0,
        word_duration_std_ms=50.0,
        word_duration_cv=0.2,
        clarity_proxy=0.85,
        articulation_stability=0.9,
    )
    
    assert result.articulation_rate == 4.5
    assert result.phoneme_timing_consistency == 0.8
    assert result.speech_precision == 0.75
    assert result.word_duration_mean_ms == 250.0
    assert result.word_duration_std_ms == 50.0
    assert result.word_duration_cv == 0.2
    assert result.clarity_proxy == 0.85
    assert result.articulation_stability == 0.9
