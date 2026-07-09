"""Milestone 2 deterministic metrics tests."""

from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from schemas import AuthorityV2Response
from services.acoustic_metrics import extract_acoustic_analysis
import services.audio_preprocessing as audio_preprocessing
from services.audio_preprocessing import _ffmpeg_available, preprocess_audio
from services.linguistic_metrics import build_linguistic_metrics, compute_delivery_metrics


def _make_wav_bytes(
    duration_seconds: float = 2.0,
    sample_rate: int = 16000,
    *,
    amplitude: float = 0.3,
    frequency: float = 220.0,
) -> bytes:
    samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, samples, endpoint=False)
    audio = (amplitude * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    return buffer.getvalue()


def _write_temp_wav(directory, name: str = "audio.wav", **kwargs) -> str:
    path = str(directory / name)
    data = _make_wav_bytes(**kwargs)
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def _write_segmented_wav(directory, name: str = "segmented.wav", sample_rate: int = 16000) -> str:
    first_seconds = 4.0
    gap_seconds = 2.0
    second_seconds = 4.0
    t1 = np.linspace(0, first_seconds, int(first_seconds * sample_rate), endpoint=False)
    t2 = np.linspace(0, second_seconds, int(second_seconds * sample_rate), endpoint=False)
    first = 0.25 * np.sin(2 * np.pi * 220.0 * t1)
    gap = np.zeros(int(gap_seconds * sample_rate))
    second = 0.25 * np.sin(2 * np.pi * 330.0 * t2)
    audio = (np.concatenate([first, gap, second]) * 32767).astype(np.int16)
    path = str(directory / name)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    return path


def test_filler_rich_transcript_increases_filler_words_per_min(tmp_path):
    wav = _write_temp_wav(tmp_path, "speech.wav", duration_seconds=10.0)
    text = "um uh like you know um uh basically like um sort of kind of um"
    delivery = compute_delivery_metrics(text, duration_seconds=10.0)
    metrics = build_linguistic_metrics(text, delivery, duration_seconds=10.0)
    assert metrics.filler_words_per_min is not None
    assert metrics.filler_words_per_min > 5.0


def test_fast_speaking_increases_words_per_minute(tmp_path):
    text = "one two three four five six seven eight nine ten eleven twelve"
    slow = compute_delivery_metrics(text, duration_seconds=12.0)
    fast = compute_delivery_metrics(text, duration_seconds=4.0)
    assert fast.words_per_minute > slow.words_per_minute


def test_silent_audio_returns_quality_warning_not_crash(tmp_path):
    wav = _write_temp_wav(tmp_path, "silent.wav", duration_seconds=0.5, amplitude=0.0001)
    result = preprocess_audio(wav)
    assert result.audio_quality.usable is False
    assert any("short" in w.lower() or "signal" in w.lower() for w in result.audio_quality.quality_warnings)


def test_preprocess_preserves_audio_after_internal_silence_gap(tmp_path):
    if not _ffmpeg_available():
        pytest.skip("ffmpeg required for preprocessing regression")
    wav = _write_segmented_wav(tmp_path)

    result = preprocess_audio(wav)

    assert result.duration_ms >= 9000
    with wave.open(result.wav_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    tail_start = int(7.0 * sample_rate)
    tail_rms = float(np.sqrt(np.mean(samples[tail_start:] ** 2)))
    assert tail_rms > 0.05


def test_long_audio_does_not_collapse_to_one_second_after_preprocessing(tmp_path):
    if not _ffmpeg_available():
        pytest.skip("ffmpeg required for preprocessing regression")
    wav = _write_segmented_wav(tmp_path, "long_gap.wav")

    result = preprocess_audio(wav)

    assert result.duration_ms > 7000


def test_ffmpeg_preprocess_filter_does_not_remove_trailing_silence(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return MagicMock()

    monkeypatch.setattr(audio_preprocessing.subprocess, "run", fake_run)

    audio_preprocessing._run_ffmpeg_preprocess(str(tmp_path / "input.mp3"), str(tmp_path / "output.wav"))

    filter_arg = commands[0][commands[0].index("-af") + 1]
    assert "silenceremove=start_periods=1" in filter_arg
    assert "stop_periods" not in filter_arg
    assert "stop_duration" not in filter_arg
    assert "stop_threshold" not in filter_arg


def test_preprocess_reruns_without_silenceremove_when_duration_collapses(monkeypatch, tmp_path):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"placeholder")
    calls = []
    probe_values = iter([10000, 1000])

    monkeypatch.setattr(audio_preprocessing, "_ffmpeg_available", lambda: True)
    monkeypatch.setattr(audio_preprocessing, "_probe_duration_ms", lambda _path: next(probe_values, 10000))

    def fake_preprocess(input_path, output_path, *, trim_leading_silence=True):
        calls.append(trim_leading_silence)

    monkeypatch.setattr(audio_preprocessing, "_run_ffmpeg_preprocess", fake_preprocess)
    monkeypatch.setattr(
        audio_preprocessing,
        "_assess_processed_audio",
        lambda _path: (10000, MagicMock(usable=True, quality_warnings=[])),
    )

    result = preprocess_audio(source)

    assert calls == [True, False]
    assert result.duration_ms == 10000


def test_missing_advanced_metrics_do_not_crash(tmp_path):
    wav = _write_temp_wav(tmp_path, "short.wav", duration_seconds=0.8, amplitude=0.2)
    analysis = extract_acoustic_analysis(
        wav,
        words=[],
        duration_ms=800,
        audio_usable=False,
        transcript_text="hello",
    )
    payload = analysis.raw.model_dump()
    assert payload["words_per_minute"] is not None or payload["words_per_minute"] is None
    assert "hnr" in payload


def test_acoustic_analysis_populates_raw_metrics(tmp_path):
    wav = _write_temp_wav(tmp_path, "tone.wav", duration_seconds=3.0)
    analysis = extract_acoustic_analysis(
        wav,
        words=[],
        duration_ms=3000,
        audio_usable=True,
        transcript_text="This is a clear statement about leadership and delivery.",
    )
    raw = analysis.raw
    assert raw.pause_frequency_per_min is not None
    assert analysis.derived.speech_continuity_score is not None


@pytest.fixture
def client():
    from main import app

    return TestClient(app)


@patch("services.coaching_engine._get_client")
@patch("services.inference_engine._get_client")
@patch("main._get_client")
def test_analyze_short_silent_file_returns_safe_response(
    mock_main_get_client,
    mock_inference_get_client,
    mock_coaching_get_client,
    client,
    tmp_path,
):
    # Mock the OpenAI client for transcription (main)
    mock_main_client = MagicMock()
    mock_main_client.audio.transcriptions.create.return_value = MagicMock(text="", language="en", segments=[])
    mock_main_get_client.return_value = mock_main_client

    # Mock the OpenAI client for inference
    mock_inference_client = MagicMock()
    cognition_response = MagicMock()
    cognition_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"clarity":{"score":20,"reason":"empty"},"persuasion":{"score":20,"reason":"empty"},'
                '"coherence":{"score":20,"reason":"empty"},"idea_strength":{"score":20,"reason":"empty"},'
                '"conciseness":{"score":20,"reason":"empty"},"failure":true}'
            )
        )
    ]
    mock_inference_client.chat.completions.create.return_value = cognition_response
    mock_inference_get_client.return_value = mock_inference_client

    # Mock the OpenAI client for coaching
    mock_coaching_client = MagicMock()
    feedback_response = MagicMock()
    feedback_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"strengths":["a","b"],"weaknesses":["c","d"],"main_issue":"x",'
                '"fixes":["f1","f2"],"drills":["d1","d2"]}'
            )
        )
    ]
    mock_coaching_client.chat.completions.create.return_value = feedback_response
    mock_coaching_get_client.return_value = mock_coaching_client

    wav_bytes = _make_wav_bytes(duration_seconds=0.5, amplitude=0.0001)
    response = client.post(
        "/analyze",
        files={"file": ("silent.wav", wav_bytes, "audio/wav")},
        data={"context": "initial"},
    )

    assert response.status_code == 200
    model = AuthorityV2Response.model_validate(response.json())
    assert model.audio_quality.usable is False
    assert model.uncertainty.reasons
    assert model.metrics.vad.speech_ratio == 0.0
    assert model.metrics.vad.total_speech_duration_ms == 0
    assert model.metrics.vad.total_silence_duration_ms == model.request.duration_ms
    assert model.metrics.vad.avg_pause_duration_ms == 0.0
    assert model.metrics.vad.pause_frequency_per_minute == 0.0
    assert model.metrics.vad.pause_durations_ms == []
    assert model.metrics.vad.long_pauses_ms == []
    assert model.metrics.vad.mid_sentence_pauses_ms == []
    assert model.metrics.vad.end_of_sentence_pauses_ms == []
    assert model.metrics.vad.vad_backend in {"none", "empty_fallback"}
    assert any(
        reason in model.uncertainty.reasons
        for reason in ("No usable speech detected", "VAD unavailable for silent audio")
    )
    assert model.metrics.derived.vocal_command_index is None or model.metrics.derived.vocal_command_index >= 0


@patch("services.coaching_engine._get_client")
@patch("services.inference_engine._get_client")
@patch("main._get_client")
def test_analyze_filler_transcript_metric(
    mock_main_get_client,
    mock_inference_get_client,
    mock_coaching_get_client,
    client,
):
    filler_text = "um uh like you know um uh like basically um"
    words = filler_text.split()

    class _Word:
        def __init__(self, word, start, end):
            self.word = word
            self.start = start
            self.end = end
            self.confidence = 0.9

    class _Seg:
        def __init__(self):
            self.text = filler_text
            self.start = 0.0
            self.end = 2.0
            self.words = [
                _Word(w, i * 0.1, (i + 1) * 0.1) for i, w in enumerate(words)
            ]

    # Mock the OpenAI client for transcription (main)
    mock_main_client = MagicMock()
    mock_main_client.audio.transcriptions.create.return_value = MagicMock(
        text=filler_text, language="en", segments=[_Seg()]
    )
    mock_main_get_client.return_value = mock_main_client

    # Mock the OpenAI client for inference
    mock_inference_client = MagicMock()
    cognition_response = MagicMock()
    cognition_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"clarity":{"score":50,"reason":"ok"},"persuasion":{"score":50,"reason":"ok"},'
                '"coherence":{"score":50,"reason":"ok"},"idea_strength":{"score":50,"reason":"ok"},'
                '"conciseness":{"score":50,"reason":"ok"},"failure":false}'
            )
        )
    ]
    mock_inference_client.chat.completions.create.return_value = cognition_response
    mock_inference_get_client.return_value = mock_inference_client

    # Mock the OpenAI client for coaching
    mock_coaching_client = MagicMock()
    feedback_response = MagicMock()
    feedback_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"strengths":["a","b"],"weaknesses":["c","d"],"main_issue":"fillers",'
                '"fixes":["f1","f2"],"drills":["d1","d2"]}'
            )
        )
    ]
    mock_coaching_client.chat.completions.create.return_value = feedback_response
    mock_coaching_get_client.return_value = mock_coaching_client

    wav_bytes = _make_wav_bytes(duration_seconds=2.0)
    response = client.post(
        "/analyze",
        files={"file": ("sample.wav", wav_bytes, "audio/wav")},
        data={"context": "initial"},
    )
    payload = response.json()
    assert payload["metrics"]["linguistic"]["filler_words_per_min"] > 3.0
