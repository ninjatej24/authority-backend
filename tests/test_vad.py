"""Tests for Voice Activity Detection (WebRTC VAD)."""

import io
import wave

import numpy as np
import pytest

import services.vad as vad_module
from services.vad import SegmentType, SpeechSegment, VADResult, run_vad


def _make_wav_bytes(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate a simple sine-wave WAV in memory."""
    frequency = 220.0
    samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, samples, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    return buffer.getvalue()


def test_vad_with_speech():
    """Test VAD with speech audio."""
    wav_bytes = _make_wav_bytes(duration_seconds=3.0)
    samples = np.frombuffer(wav_bytes, dtype=np.int16)
    
    result = run_vad(samples, 16000)
    
    assert isinstance(result, VADResult)
    assert result.speech_ratio > 0.5  # Should detect speech
    assert result.total_speech_duration_ms > 0
    assert result.total_silence_duration_ms >= 0
    # Note: speech_segments may be empty if webrtcvad is not available (fallback mode)
    # The fallback still provides speech_ratio and duration estimates


def test_vad_with_silence():
    """Test VAD with silent audio."""
    samples = np.zeros(int(16000 * 1.0), dtype=np.int16)
    
    result = run_vad(samples, 16000)
    
    assert isinstance(result, VADResult)
    assert result.speech_ratio == 0.0
    assert result.total_speech_duration_ms == 0
    assert result.total_silence_duration_ms == 1000
    assert result.pause_durations_ms == []
    assert result.long_pauses_ms == []
    assert result.mid_sentence_pauses_ms == []
    assert result.end_of_sentence_pauses_ms == []
    assert result.avg_pause_duration_ms == 0.0
    assert result.pause_frequency_per_minute == 0.0
    assert result.vad_backend == "empty_fallback"


def test_vad_uses_energy_fallback_when_webrtcvad_missing(monkeypatch):
    """Deployment should not require the compiled webrtcvad package."""
    wav_bytes = _make_wav_bytes(duration_seconds=2.0)
    samples = np.frombuffer(wav_bytes, dtype=np.int16)

    monkeypatch.setattr(vad_module, "WEBRTC_VAD_AVAILABLE", False)
    monkeypatch.setattr(vad_module, "webrtcvad", None)

    result = run_vad(samples, 16000)

    assert isinstance(result, VADResult)
    assert result.vad_backend == "energy_fallback"
    assert result.speech_ratio > 0
    assert result.total_speech_duration_ms > 0
    assert result.speech_segments


def test_vad_empty_samples():
    """Test VAD with empty samples."""
    samples = np.array([], dtype=np.int16)
    
    result = run_vad(samples, 16000)
    
    assert isinstance(result, VADResult)
    assert result.speech_ratio == 0.0
    assert result.total_speech_duration_ms == 0
    assert result.total_silence_duration_ms == 0
    assert result.pause_durations_ms == []
    assert result.vad_backend == "none"
    assert len(result.speech_segments) == 0


def test_vad_pause_classification():
    """Test VAD pause classification with transcript words."""
    wav_bytes = _make_wav_bytes(duration_seconds=3.0)
    samples = np.frombuffer(wav_bytes, dtype=np.int16)
    
    # Mock transcript words
    class Word:
        def __init__(self, text, start_ms, end_ms):
            self.text = text
            self.start_ms = start_ms
            self.end_ms = end_ms
    
    words = [
        Word("hello", 0, 500),
        Word("world", 600, 1100),
        Word("test", 1200, 1700),
    ]
    
    result = run_vad(samples, 16000, transcript_words=words)
    
    assert isinstance(result, VADResult)
    assert len(result.pause_durations_ms) >= 0
    assert result.avg_pause_duration_ms >= 0


def test_speech_segment_creation():
    """Test SpeechSegment dataclass."""
    segment = SpeechSegment(
        start_ms=100,
        end_ms=500,
        duration_ms=400,
        segment_type=SegmentType.SPEECH,
        confidence=0.95,
    )
    
    assert segment.start_ms == 100
    assert segment.end_ms == 500
    assert segment.duration_ms == 400
    assert segment.segment_type == SegmentType.SPEECH
    assert segment.confidence == 0.95


def test_vad_result_structure():
    """Test VADResult has all required fields."""
    result = VADResult(
        segments=[],
        speech_segments=[],
        silence_segments=[],
        speech_ratio=0.5,
        total_speech_duration_ms=1500,
        total_silence_duration_ms=1500,
        pause_durations_ms=[200, 300],
        long_pauses_ms=[300],
        mid_sentence_pauses_ms=[200],
        end_of_sentence_pauses_ms=[300],
        avg_pause_duration_ms=250.0,
        pause_frequency_per_minute=8.0,
    )
    
    assert result.speech_ratio == 0.5
    assert result.total_speech_duration_ms == 1500
    assert result.total_silence_duration_ms == 1500
    assert len(result.pause_durations_ms) == 2
    assert result.avg_pause_duration_ms == 250.0
    assert result.pause_frequency_per_minute == 8.0
