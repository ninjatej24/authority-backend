"""Milestone 16 production hardening and endpoint tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from main import app
from schemas import AuthorityV2Response
from services.persistence import DEFAULT_REPOSITORY
from tests.test_analyze_endpoint import _FakeTranscription, _fake_gpt_json, _make_wav_bytes


def _client_with_mocks(mock_main_get_client, mock_inference_get_client, mock_coaching_get_client):
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
    return TestClient(app)


@patch("services.coaching_engine._get_client")
@patch("services.inference_engine._get_client")
@patch("main._get_client")
def test_analyze_without_stable_user_key_succeeds_but_does_not_persist(
    mock_main_get_client,
    mock_inference_get_client,
    mock_coaching_get_client,
):
    DEFAULT_REPOSITORY.clear()
    client = _client_with_mocks(mock_main_get_client, mock_inference_get_client, mock_coaching_get_client)
    response = client.post(
        "/analyze",
        files={"file": ("sample.wav", _make_wav_bytes(), "audio/wav")},
        data={"context": "benchmark", "title": "Test"},
    )

    assert response.status_code == 200
    model = AuthorityV2Response.model_validate(response.json())
    assert model.persistence_status.persisted is False
    assert model.persistence_status.user_key_present is False
    assert "missing_user_key" in model.persistence_status.audit_events
    assert model.history_summary.benchmark_count == 0
    assert DEFAULT_REPOSITORY.list_benchmarks("anonymous") == []


@patch("services.response_builder.persist_analysis", side_effect=RuntimeError("db down"))
@patch("services.coaching_engine._get_client")
@patch("services.inference_engine._get_client")
@patch("main._get_client")
def test_analyze_succeeds_when_database_write_fails(
    mock_main_get_client,
    mock_inference_get_client,
    mock_coaching_get_client,
    _mock_persist,
):
    client = _client_with_mocks(mock_main_get_client, mock_inference_get_client, mock_coaching_get_client)
    response = client.post(
        "/analyze",
        files={"file": ("sample.wav", _make_wav_bytes(), "audio/wav")},
        data={"context": "benchmark", "installation_id": "install-db-fail"},
    )

    assert response.status_code == 200
    model = AuthorityV2Response.model_validate(response.json())
    assert model.persistence_status.persisted is False
    assert "database_write_failure" in model.persistence_status.audit_events


@patch("services.coaching_engine._get_client")
@patch("services.inference_engine._get_client")
@patch("main._get_client")
def test_history_dashboard_progress_and_analysis_endpoints(
    mock_main_get_client,
    mock_inference_get_client,
    mock_coaching_get_client,
):
    DEFAULT_REPOSITORY.clear()
    client = _client_with_mocks(mock_main_get_client, mock_inference_get_client, mock_coaching_get_client)
    response = client.post(
        "/analyze",
        files={"file": ("sample.wav", _make_wav_bytes(), "audio/wav")},
        data={"context": "benchmark", "installation_id": "install-endpoints"},
    )
    model = AuthorityV2Response.model_validate(response.json())

    history = client.get("/history", params={"installation_id": "install-endpoints"})
    dashboard = client.get("/dashboard-state", params={"installation_id": "install-endpoints"})
    progress = client.get("/progress-history", params={"installation_id": "install-endpoints"})
    owned = client.get(f"/analysis/{model.analysis_id}", params={"installation_id": "install-endpoints"})
    forbidden = client.get(f"/analysis/{model.analysis_id}", params={"installation_id": "other-install"})

    assert history.status_code == 200
    assert history.json()["history"]["history_summary"]["benchmark_count"] >= 1
    assert dashboard.status_code == 200
    assert dashboard.json()["dashboard_state"]["authority_snapshot"]["analysis_id"] == model.analysis_id
    assert progress.status_code == 200
    assert owned.status_code == 200
    assert owned.json()["analysis_id"] == model.analysis_id
    assert forbidden.status_code == 404


def test_history_endpoint_missing_user_key_degrades_gracefully():
    response = TestClient(app).get("/history")

    assert response.status_code == 200
    assert response.json()["warnings"]
    assert response.json()["history"]["history_summary"]["benchmark_count"] == 0


def test_drill_completion_endpoint_persists_minimal_completion():
    DEFAULT_REPOSITORY.clear()
    response = TestClient(app).post(
        "/drills/complete",
        params={"installation_id": "install-drill"},
        json={
            "drill_id": "pause_ownership_v1",
            "analysis_id": "a1",
            "scenario": "benchmark",
            "duration_seconds": 180,
            "target_dimensions": ["command"],
            "linked_moment_ids": ["m1"],
            "confidence": 0.7,
        },
    )

    assert response.status_code == 200
    assert response.json()["stored"] is True
    assert response.json()["drill_id"] == "pause_ownership_v1"
