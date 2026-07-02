"""Integration-style tests for /analyze returning authority.v2 JSON."""

from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from schemas import AuthorityV2Response


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


class _FakeWord:
    def __init__(self, word: str, start: float, end: float):
        self.word = word
        self.start = start
        self.end = end
        self.confidence = 0.95


class _FakeSegment:
    def __init__(self, text: str, start: float, end: float, words: list[_FakeWord]):
        self.text = text
        self.start = start
        self.end = end
        self.words = words


class _FakeTranscription:
    def __init__(self):
        self.text = "I believe we should move forward with clarity and purpose today."
        self.language = "en"
        self.segments = [
            _FakeSegment(
                self.text,
                0.0,
                2.0,
                [
                    _FakeWord("I", 0.0, 0.2),
                    _FakeWord("believe", 0.2, 0.5),
                    _FakeWord("we", 0.5, 0.7),
                    _FakeWord("should", 0.7, 1.0),
                    _FakeWord("move", 1.0, 1.3),
                    _FakeWord("forward", 1.3, 2.0),
                ],
            )
        ]


def _fake_gpt_json() -> str:
    return """
    {
      "clarity": {"score": 68, "reason": "Clear meaning."},
      "persuasion": {"score": 62, "reason": "Moderately convincing."},
      "coherence": {"score": 70, "reason": "Logical flow."},
      "idea_strength": {"score": 65, "reason": "Solid point."},
      "conciseness": {"score": 60, "reason": "Mostly efficient."},
      "failure": false
    }
    """


def _fake_feedback_json() -> str:
    return """
    {
      "strengths": ["Clear central idea.", "Steady logical direction."],
      "weaknesses": ["Pacing could be more deliberate.", "Ending lacks finality."],
      "main_issue": "Declarative finality",
      "fixes": ["Pause after key statements.", "End sentences with downward energy."],
      "drills": ["Read 8 decisive statements with a half-beat pause.", "Record a 30-second point-support-close answer."]
    }
    """


@pytest.fixture
def client():
    from main import app

    return TestClient(app)


@patch("services.coaching_engine._get_client")
@patch("services.inference_engine._get_client")
@patch("main._get_client")
def test_analyze_returns_valid_authority_v2(
    mock_main_get_client,
    mock_inference_get_client,
    mock_coaching_get_client,
    client,
):
    # Mock the OpenAI client for transcription (main)
    mock_main_client = MagicMock()
    mock_main_client.audio.transcriptions.create.return_value = _FakeTranscription()
    mock_main_get_client.return_value = mock_main_client

    # Mock the OpenAI client for inference
    mock_inference_client = MagicMock()
    cognition_response = MagicMock()
    cognition_response.choices = [MagicMock(message=MagicMock(content=_fake_gpt_json()))]
    mock_inference_client.chat.completions.create.return_value = cognition_response
    mock_inference_get_client.return_value = mock_inference_client

    # Mock the OpenAI client for coaching
    mock_coaching_client = MagicMock()
    feedback_response = MagicMock()
    feedback_response.choices = [MagicMock(message=MagicMock(content=_fake_feedback_json()))]
    mock_coaching_client.chat.completions.create.return_value = feedback_response
    mock_coaching_get_client.return_value = mock_coaching_client

    wav_bytes = _make_wav_bytes()
    response = client.post(
        "/analyze",
        files={"file": ("sample.wav", wav_bytes, "audio/wav")},
        data={"context": "initial", "title": "Test", "prompt": "Tell me about leadership"},
    )

    assert response.status_code == 200
    payload = response.json()

    model = AuthorityV2Response.model_validate(payload)
    assert model.schema_version == "authority.v2"
    assert model.request.scenario == "benchmark"
    assert model.transcript.full_text
    assert 25 <= model.scores.authority_score <= 95
    assert model.scores.dimension_scores.command >= 20
    assert model.recommendations.fastest_improvement_tip
    assert model.paywall.locked_modules
    assert model.safety.responsible_framing


@patch("services.coaching_engine._get_client")
@patch("services.inference_engine._get_client")
@patch("main._get_client")
def test_analyze_impromptu_scenario_mapping(
    mock_main_get_client,
    mock_inference_get_client,
    mock_coaching_get_client,
    client,
):
    # Mock the OpenAI client for transcription (main)
    mock_main_client = MagicMock()
    mock_main_client.audio.transcriptions.create.return_value = _FakeTranscription()
    mock_main_get_client.return_value = mock_main_client
    
    # Mock the OpenAI client for inference
    mock_inference_client = MagicMock()
    cognition_response = MagicMock()
    cognition_response.choices = [MagicMock(message=MagicMock(content=_fake_gpt_json()))]
    mock_inference_client.chat.completions.create.return_value = cognition_response
    mock_inference_get_client.return_value = mock_inference_client
    
    # Mock the OpenAI client for coaching
    mock_coaching_client = MagicMock()
    feedback_response = MagicMock()
    feedback_response.choices = [MagicMock(message=MagicMock(content=_fake_feedback_json()))]
    mock_coaching_client.chat.completions.create.return_value = feedback_response
    mock_coaching_get_client.return_value = mock_coaching_client

    wav_bytes = _make_wav_bytes()
    response = client.post(
        "/analyze",
        files={"file": ("sample.wav", wav_bytes, "audio/wav")},
        data={"context": "impromptu", "prompt": "What is your biggest weakness?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request"]["scenario"] == "impromptu"


@patch("services.coaching_engine._get_client")
@patch("services.inference_engine._get_client")
@patch("main._get_client")
def test_analyze_returns_milestone3_metrics(
    mock_main_get_client,
    mock_inference_get_client,
    mock_coaching_get_client,
    client,
):
    """Test that /analyze returns populated Milestone 3 metrics."""
    # Mock the OpenAI client for transcription (main)
    mock_main_client = MagicMock()
    mock_main_client.audio.transcriptions.create.return_value = _FakeTranscription()
    mock_main_get_client.return_value = mock_main_client

    # Mock the OpenAI client for inference
    mock_inference_client = MagicMock()
    cognition_response = MagicMock()
    cognition_response.choices = [MagicMock(message=MagicMock(content=_fake_gpt_json()))]
    mock_inference_client.chat.completions.create.return_value = cognition_response
    mock_inference_get_client.return_value = mock_inference_client

    # Mock the OpenAI client for coaching
    mock_coaching_client = MagicMock()
    feedback_response = MagicMock()
    feedback_response.choices = [MagicMock(message=MagicMock(content=_fake_feedback_json()))]
    mock_coaching_client.chat.completions.create.return_value = feedback_response
    mock_coaching_get_client.return_value = mock_coaching_client

    wav_bytes = _make_wav_bytes(duration_seconds=5.0)  # Longer audio for better analysis
    response = client.post(
        "/analyze",
        files={"file": ("sample.wav", wav_bytes, "audio/wav")},
        data={"context": "initial", "title": "Test", "prompt": "Tell me about leadership"},
    )

    assert response.status_code == 200
    payload = response.json()

    # Check that Milestone 3 metric categories exist
    assert "metrics" in payload
    assert "rhythm" in payload["metrics"]
    assert "articulation" in payload["metrics"]
    assert "vad" in payload["metrics"]

    # Check that raw_acoustic has Milestone 3 enhanced fields
    raw_acoustic = payload["metrics"]["raw_acoustic"]
    # Some enhanced fields may be null due to audio quality, but they should exist in the schema
    assert "pitch_mean_hz" in raw_acoustic
    assert "energy_mean" in raw_acoustic
    assert "voicing_ratio" in raw_acoustic

    # Check that derived metrics has Milestone 3 indices
    derived = payload["metrics"]["derived"]
    assert "vocal_command_index" in derived
    assert "composure_index" in derived
    assert "rhythm_index" in derived
    assert "projection_index" in derived
    assert "authority_signal_index" in derived
