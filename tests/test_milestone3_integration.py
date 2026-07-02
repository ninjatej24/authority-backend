"""Additional Milestone 3 integration and VAD tests."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from services.acoustic_metrics import extract_acoustic_analysis
from services.vad import prepare_pcm_samples, run_vad


def _make_wav_file(path, duration_seconds=3.0, sample_rate=16000, amplitude=0.3):
    samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, samples, endpoint=False)
    audio = (amplitude * np.sin(2 * np.pi * 220.0 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())


def test_prepare_pcm_samples_converts_float_to_int16():
    float_samples = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float64)
    pcm, rate = prepare_pcm_samples(float_samples, 44100)
    assert pcm.dtype == np.int16
    assert rate == 16000
    assert len(pcm) > 0


def test_energy_vad_produces_pause_segments(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _make_wav_file(wav_path, duration_seconds=3.0)
    samples = np.frombuffer(wave.open(str(wav_path), "rb").readframes(999999), dtype=np.int16)
    with wave.open(str(wav_path), "rb") as wav_file:
        samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
        sample_rate = wav_file.getframerate()

    result = run_vad(samples, sample_rate)
    assert result.speech_ratio > 0
    assert result.total_speech_duration_ms > 0
    assert result.vad_backend in {"energy_fallback", "webrtc"}
    assert len(result.speech_segments) >= 1


def test_acoustic_metrics_use_vad_pauses(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _make_wav_file(wav_path, duration_seconds=3.0)
    with wave.open(str(wav_path), "rb") as wav_file:
        samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
        sample_rate = wav_file.getframerate()

    vad_result = run_vad(samples, sample_rate)
    analysis = extract_acoustic_analysis(
        str(wav_path),
        words=[],
        duration_ms=3000,
        audio_usable=True,
        transcript_text="hello world testing speech",
        vad_result=vad_result,
    )

    assert analysis.voice_metrics["pause_count"] == len(vad_result.pause_durations_ms)
    assert "terminal_rising_ratio" not in analysis.pitch_contour or analysis.pitch_contour.get(
        "terminal_rising_ratio"
    ) is None or 0.0 <= analysis.pitch_contour["terminal_rising_ratio"] <= 1.0


def test_terminal_ratios_omitted_without_enough_phrase_endings(tmp_path):
    wav_path = tmp_path / "short.wav"
    _make_wav_file(wav_path, duration_seconds=1.0)
    analysis = extract_acoustic_analysis(
        str(wav_path),
        words=[],
        duration_ms=1000,
        audio_usable=True,
        transcript_text="hi",
    )
    contour = analysis.pitch_contour
    if "terminal_rising_ratio" in contour:
        assert 0.0 <= contour["terminal_rising_ratio"] <= 1.0
    if "terminal_falling_ratio" in contour:
        assert 0.0 <= contour["terminal_falling_ratio"] <= 1.0
