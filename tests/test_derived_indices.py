"""Tests for derived indices module."""

import pytest

from services.acoustic_metrics import AcousticAnalysisResult, WindowFeature, VoiceMetrics
from services.articulation import ArticulationAnalysis
from services.derived_indices import DerivedIndices, calculate_derived_indices
from services.rhythm_analysis import RhythmAnalysis
from services.vad import VADResult


def test_calculate_derived_indices():
    """Test calculation of all derived indices."""
    # Create mock acoustic result
    voice_metrics: VoiceMetrics = {
        "duration_seconds": 30.0,
        "pitch_mean": 150.0,
        "pitch_median": 145.0,
        "pitch_variation": 25.0,
        "energy_mean": 50.0,
        "energy_variation": 10.0,
        "silence_ratio": 0.2,
        "avg_pause_duration": 0.4,
        "pause_frequency": 0.3,
        "speech_density": 0.8,
        "longest_pause_seconds": 1.5,
        "pause_count": 9.0,
        "mid_phrase_pause_rate": 0.15,
        "terminal_rise_ratio": 0.25,
        "f0_range_semitones": 8.0,
        "f0_variability_semitones": 2.0,
    }
    
    acoustic_result = AcousticAnalysisResult(
        voice_metrics=voice_metrics,
        raw=None,  # type: ignore
        derived=None,  # type: ignore
        windows=[
            WindowFeature(
                start_ms=0,
                end_ms=3000,
                command_score=0.7,
                clarity_score=0.75,
                composure_score=0.8,
                presence_score=0.7,
                filler_rate=0.05,
                wpm=150.0,
                pitch_stdev_semitones=2.0,
                loudness_stdev_db=5.0,
                pause_ms=300.0,
                monotone=False,
                rushing=False,
            )
        ],
        speaking_seconds=24.0,
        pitch_contour={
            "pitch_mean_hz": 150.0,
            "pitch_median_hz": 145.0,
            "pitch_std_hz": 25.0,
            "pitch_slope": 0.01,
            "pitch_stability": 0.8,
            "pitch_dynamics": 0.5,
            "pitch_resets": 2,
            "terminal_slope": -0.1,
            "terminal_rising": 0.0,
            "terminal_falling": 1.0,
            "terminal_rising_ratio": 0.1,
            "terminal_falling_ratio": 0.5,
        },
        energy_contour={
            "energy_mean": 50.0,
            "energy_peak": 70.0,
            "energy_std": 10.0,
            "energy_slope": 0.001,
            "dynamic_emphasis": 1.4,
            "loudness_stability": 0.8,
            "emphasis_bursts": 3,
            "projection_segments": 2,
            "energy_cv": 0.2,
        },
        voice_quality={
            "voicing_ratio": 0.85,
            "voice_breaks": 1,
            "breathiness_proxy": 0.3,
            "strain_proxy": 0.1,
            "cpp_proxy": 0.8,
        },
    )
    
    # Create mock VAD result
    vad_result = VADResult(
        segments=[],
        speech_segments=[],
        silence_segments=[],
        speech_ratio=0.8,
        total_speech_duration_ms=24000,
        total_silence_duration_ms=6000,
        pause_durations_ms=[200, 300, 400],
        long_pauses_ms=[400],
        mid_sentence_pauses_ms=[200],
        end_of_sentence_pauses_ms=[400],
        avg_pause_duration_ms=300.0,
        pause_frequency_per_minute=12.0,
    )
    
    # Create mock rhythm analysis
    rhythm_analysis = RhythmAnalysis(
        speech_rate=4.5,
        words_per_minute=150.0,
        pause_cadence=0.25,
        speech_continuity=0.8,
        hesitation_windows=1,
        rhythm_consistency=0.75,
        burst_speaking_segments=0,
        slow_down_segments=0,
        speed_up_segments=0,
        articulation_rate=4.5,
    )
    
    # Create mock articulation analysis
    articulation_analysis = ArticulationAnalysis(
        articulation_rate=4.5,
        phoneme_timing_consistency=0.8,
        speech_precision=0.75,
        word_duration_mean_ms=250.0,
        word_duration_std_ms=50.0,
        word_duration_cv=0.2,
        clarity_proxy=0.85,
        articulation_stability=0.9,
    )
    
    result = calculate_derived_indices(
        acoustic_result=acoustic_result,
        vad_result=vad_result,
        rhythm_analysis=rhythm_analysis,
        articulation_analysis=articulation_analysis,
        audio_quality_usable=True,
        duration_ms=30000,
    )
    
    assert isinstance(result, DerivedIndices)
    assert 0.0 <= result.vocal_command_index <= 1.0
    assert 0.0 <= result.composure_index <= 1.0
    assert 0.0 <= result.rhythm_index <= 1.0
    assert 0.0 <= result.projection_index <= 1.0
    assert 0.0 <= result.authority_signal_index <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_derived_indices_with_poor_audio():
    """Test derived indices with poor audio quality."""
    # Minimal mock data
    voice_metrics: VoiceMetrics = {
        "duration_seconds": 10.0,
        "pitch_mean": 0.0,
        "pitch_median": 0.0,
        "pitch_variation": 0.0,
        "energy_mean": 0.0,
        "energy_variation": 0.0,
        "silence_ratio": 0.5,
        "avg_pause_duration": 0.5,
        "pause_frequency": 0.2,
        "speech_density": 0.5,
        "longest_pause_seconds": 2.0,
        "pause_count": 2.0,
        "mid_phrase_pause_rate": 0.0,
        "terminal_rise_ratio": 0.0,
        "f0_range_semitones": 0.0,
        "f0_variability_semitones": 0.0,
    }
    
    acoustic_result = AcousticAnalysisResult(
        voice_metrics=voice_metrics,
        raw=None,  # type: ignore
        derived=None,  # type: ignore
        windows=[],
        speaking_seconds=5.0,
        pitch_contour={},
        energy_contour={},
        voice_quality={"voicing_ratio": 0.5},
    )
    
    vad_result = VADResult(
        segments=[],
        speech_segments=[],
        silence_segments=[],
        speech_ratio=0.5,
        total_speech_duration_ms=5000,
        total_silence_duration_ms=5000,
        pause_durations_ms=[],
        long_pauses_ms=[],
        mid_sentence_pauses_ms=[],
        end_of_sentence_pauses_ms=[],
        avg_pause_duration_ms=0.0,
        pause_frequency_per_minute=0.0,
    )
    
    rhythm_analysis = RhythmAnalysis(
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
    
    articulation_analysis = ArticulationAnalysis(
        articulation_rate=0.0,
        phoneme_timing_consistency=0.5,
        speech_precision=0.5,
        word_duration_mean_ms=0.0,
        word_duration_std_ms=0.0,
        word_duration_cv=0.0,
        clarity_proxy=0.5,
        articulation_stability=0.5,
    )
    
    result = calculate_derived_indices(
        acoustic_result=acoustic_result,
        vad_result=vad_result,
        rhythm_analysis=rhythm_analysis,
        articulation_analysis=articulation_analysis,
        audio_quality_usable=False,
        duration_ms=10000,
    )
    
    assert isinstance(result, DerivedIndices)
    # Poor audio should reduce confidence
    assert result.confidence < 1.0


def test_derived_indices_structure():
    """Test DerivedIndices dataclass structure."""
    result = DerivedIndices(
        vocal_command_index=0.75,
        composure_index=0.8,
        rhythm_index=0.7,
        projection_index=0.85,
        authority_signal_index=0.78,
        confidence=0.85,
    )
    
    assert result.vocal_command_index == 0.75
    assert result.composure_index == 0.8
    assert result.rhythm_index == 0.7
    assert result.projection_index == 0.85
    assert result.authority_signal_index == 0.78
    assert result.confidence == 0.85
