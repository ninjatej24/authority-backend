"""Milestone 15 dashboard state tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from main import app
from schemas import AuthorityV2Response
from services.dashboard_state import build_dashboard_state
from services.history_engine import build_history
from services.persistence import DEFAULT_REPOSITORY, InMemoryAuthorityRepository, persist_analysis
from tests.test_analyze_endpoint import _FakeTranscription, _fake_gpt_json, _make_wav_bytes
from tests.test_history_engine import _stored_response


def test_dashboard_state_for_first_benchmark():
    repo = InMemoryAuthorityRepository()
    response = _stored_response("a1", 60, created_at="2026-07-01T10:00:00Z")
    persist_analysis(response, repository=repo)
    history = build_history(repo.list_benchmarks("u1"), user_id="u1")
    dashboard = build_dashboard_state(history)

    assert dashboard.momentum == "new_baseline"
    assert dashboard.authority_snapshot.analysis_id == "a1"
    assert dashboard.today_mission
    assert dashboard.next_retest["baseline_benchmark_id"] == "a1"


def test_dashboard_state_for_improving_history():
    repo = InMemoryAuthorityRepository()
    persist_analysis(_stored_response("a1", 55, created_at="2026-07-01T10:00:00Z"), repository=repo)
    persist_analysis(_stored_response("a2", 66, created_at="2026-07-02T10:00:00Z"), repository=repo)
    history = build_history(repo.list_benchmarks("u1"), user_id="u1")
    dashboard = build_dashboard_state(history)

    assert dashboard.momentum == "improving"
    assert dashboard.growth_signal
    assert dashboard.weekly_summary.weekly_improvement == 11
    assert dashboard.identity_summary


def test_dashboard_state_for_declining_history():
    repo = InMemoryAuthorityRepository()
    persist_analysis(_stored_response("a1", 70, created_at="2026-07-01T10:00:00Z"), repository=repo)
    persist_analysis(_stored_response("a2", 60, created_at="2026-07-02T10:00:00Z"), repository=repo)
    history = build_history(repo.list_benchmarks("u1"), user_id="u1")
    dashboard = build_dashboard_state(history)

    assert dashboard.momentum == "declining"
    assert dashboard.active_training_stage


@patch("services.coaching_engine._get_client")
@patch("services.inference_engine._get_client")
@patch("main._get_client")
def test_endpoint_attaches_history_and_dashboard_outputs(
    mock_main_get_client,
    mock_inference_get_client,
    mock_coaching_get_client,
):
    DEFAULT_REPOSITORY.clear()
    mock_main_client = MagicMock()
    mock_main_client.audio.transcriptions.create.return_value = _FakeTranscription()
    mock_main_get_client.return_value = mock_main_client

    mock_inference_client = MagicMock()
    cognition_response = MagicMock()
    cognition_response.choices = [MagicMock(message=MagicMock(content=_fake_gpt_json()))]
    mock_inference_client.chat.completions.create.return_value = cognition_response
    mock_inference_get_client.return_value = mock_inference_client

    mock_coaching_client = MagicMock()
    feedback_response = MagicMock()
    feedback_response.choices = [MagicMock(message=MagicMock(content="{}"))]
    mock_coaching_client.chat.completions.create.return_value = feedback_response
    mock_coaching_get_client.return_value = mock_coaching_client

    response = TestClient(app).post(
        "/analyze",
        files={"file": ("sample.wav", _make_wav_bytes(), "audio/wav")},
        data={"context": "benchmark", "title": "Test", "prompt": "Tell me about leadership", "installation_id": "install-dashboard"},
    )

    assert response.status_code == 200
    model = AuthorityV2Response.model_validate(response.json())
    assert model.history_summary.benchmark_count >= 1
    assert model.dashboard_state.authority_snapshot.analysis_id == model.analysis_id
    assert model.authority_journey.stage
    assert model.weekly_summary.drill_completion_summary is not None
    assert model.user_snapshot.benchmark_count >= 1
