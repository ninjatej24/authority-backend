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
    assert 20 <= model.scores.authority_score <= 97
    assert model.scores.dimension_scores.command >= 20
    assert model.progress.state.progress_status == "first_benchmark"
    assert model.progress.comparison_available is False
    assert model.pipeline_validation.pipeline_version == "authority.v2.milestone12"
    assert model.pipeline_validation.audit.completed_stages
    assert model.moment_intelligence.engine_version == "moment_intelligence_v1"
    assert model.polished_report.engine_version == "llm_polish_v1"
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

    wav_bytes = _make_wav_bytes(duration_seconds=10.0)  # Long enough to avoid insufficient-sample gating
    response = client.post(
        "/analyze",
        files={"file": ("sample.wav", wav_bytes, "audio/wav")},
        data={"context": "initial", "title": "Test", "prompt": "Tell me about leadership"},
    )

    assert response.status_code == 200
    payload = response.json()

    metrics = payload["metrics"]
    assert metrics["rhythm"]["words_per_minute"] is not None
    assert metrics["rhythm"]["words_per_minute"] > 0
    assert metrics["rhythm"]["rhythm_consistency"] is not None
    assert 0.0 <= metrics["rhythm"]["rhythm_consistency"] <= 1.0

    assert metrics["articulation"]["articulation_rate"] is not None
    assert metrics["articulation"]["articulation_rate"] > 0
    assert metrics["articulation"]["clarity_proxy"] is not None
    assert 0.0 <= metrics["articulation"]["clarity_proxy"] <= 1.0

    assert metrics["vad"]["speech_ratio"] is not None
    assert metrics["vad"]["speech_ratio"] > 0
    assert metrics["vad"]["total_speech_duration_ms"] is not None
    assert metrics["vad"]["total_speech_duration_ms"] > 0

    derived = metrics["derived"]
    for key in (
        "vocal_command_index",
        "composure_index",
        "rhythm_index",
        "projection_index",
        "authority_signal_index",
    ):
        assert derived[key] is not None, f"{key} should be populated for valid speech input"
        assert 0.0 <= derived[key] <= 1.0

    raw_acoustic = metrics["raw_acoustic"]
    assert raw_acoustic.get("pitch_mean_hz") is not None
    assert raw_acoustic.get("energy_mean") is not None

    assert "metric_evidence" in payload
    assert len(payload["metric_evidence"]["vad"]) > 0
    assert len(payload["metric_evidence"]["rhythm"]) > 0
    assert "psychological_inference" in payload
    assert len(payload["psychological_inference"]["micro_behaviours"]) >= 25
    assert len(payload["psychological_inference"]["traits"]) >= 20
    assert payload["psychological_inference"]["evidence_chain"]

    report = payload["report"]
    assert report["mirror"]["evidence_ids"]
    assert report["diagnosis"]["supporting_evidence_ids"]
    assert report["perception_map"]["first_impression"]["evidence_ids"]
    if report["authority_type"]["label"] == "Insufficient Sample":
        assert "30 to 60 second" in report["highest_leverage_fix"]["plain_english"]
        assert report["training_prescription"]["instructions"]
    else:
        assert report["highest_leverage_fix"]["first_drill_id"]
        assert report["training_prescription"]["drill_id"]
    assert report["authority_type"]["label"]
    assert report["share_card"]["share_safety"] == "public_safe"
    assert report["technical_appendix"]["metrics"]
    if report["authority_type"]["label"] == "Insufficient Sample":
        assert report["primary_diagnosis"] is None
    else:
        assert report["diagnostic_reasoning"]["dimension_reasoning"]
        assert report["primary_diagnosis"]["supporting_evidence_ids"]
        assert report["highest_leverage_reasoning"]["selection_score"] > 0

    coaching = payload["coaching_engine"]
    assert coaching["drill_library_size"] >= 20
    assert coaching["intervention_candidates"]
    assert coaching["dependency_graph"]
    assert "root_causes" in coaching
