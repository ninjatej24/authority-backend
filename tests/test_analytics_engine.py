"""Milestone 17 analytics, calibration and feedback preparation tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from main import app
from schemas import (
    AnalyticsBundle,
    DrillFeedback,
    GeneralFeedback,
    ReportFeedback,
    RetestFeedback,
    SubjectiveAccuracyFeedback,
)
from services.analytics_engine import build_analytics_bundle
from services.dashboard_state import build_dashboard_state
from services.history_engine import build_history
from services.persistence import DEFAULT_REPOSITORY, InMemoryAuthorityRepository, persist_analysis
from tests.test_analyze_endpoint import _FakeTranscription, _fake_gpt_json, _make_wav_bytes
from tests.test_history_engine import _stored_response
from tests.test_pipeline_validator import _response


def _response_with_history():
    repo = InMemoryAuthorityRepository()
    persist_analysis(_stored_response("a1", 58, created_at="2026-07-01T10:00:00Z"), repository=repo)
    current = _stored_response("a2", 67, created_at="2026-07-04T10:00:00Z")
    persist_analysis(current, repository=repo)
    history = build_history(repo.list_benchmarks("u1"), user_id="u1")
    dashboard = build_dashboard_state(history)
    return current.model_copy(
        update={
            "history_summary": history.history_summary,
            "authority_journey": history.authority_journey,
            "weekly_summary": history.weekly_summary,
            "monthly_summary": history.monthly_summary,
            "training_history": history.training_history,
            "scenario_history": history.scenario_history,
            "dashboard_state": dashboard,
            "user_snapshot": history.user_profile,
        }
    )


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_analytics_bundle_is_deterministic_and_schema_validates():
    response = _response_with_history()
    first = build_analytics_bundle(response)
    second = build_analytics_bundle(response)

    assert first.model_dump() == second.model_dump()
    assert AnalyticsBundle.model_validate(first.model_dump()).summary.deterministic is True
    assert first.summary.privacy_mode == "metadata_only"
    assert first.analysis.analysis_id == "a2"


def test_analysis_calibration_and_fairness_records_are_metadata_only():
    response = _response_with_history()
    analytics = build_analytics_bundle(response)

    assert analytics.calibration_record.immutable_snapshot is True
    assert analytics.calibration_record.authority_score == response.scores.authority_score
    assert analytics.calibration_record.dimension_scores["command"] == response.scores.dimension_scores.command
    assert analytics.fairness_audit.demographic_inference_included is False
    assert analytics.fairness_audit.protected_characteristics_included is False
    assert analytics.fairness_audit.language == "en"

    payload_strings = list(_strings(analytics.model_dump()))
    assert response.transcript.full_text not in payload_strings
    assert response.perception_profile.headline not in payload_strings
    assert all("Listeners are likely" not in item for item in payload_strings)


def test_timeline_coaching_and_progress_analytics_are_collected_as_counts_and_ids():
    response = _response()
    analytics = build_analytics_bundle(response)

    assert analytics.timeline.total_moment_count >= len(response.moments)
    assert analytics.timeline.strongest_moment_count >= 1
    assert analytics.coaching.selected_drill_id
    assert analytics.coaching.dependency_graph_ids
    assert analytics.progress.trend_summary["score_trend"] in {
        "improving",
        "stable",
        "declining",
        "insufficient_history",
    }


def test_retention_and_product_metrics_use_observable_history_only():
    response = _response_with_history()
    analytics = build_analytics_bundle(response)

    assert analytics.retention.days_since_benchmark is not None
    assert analytics.retention.drop_off_risk_inputs["benchmark_count"] == 2
    assert analytics.product_metrics.benchmark_completion is True
    assert analytics.product_metrics.history_length == 2
    assert analytics.product_metrics.average_confidence is not None


def test_feedback_storage_schemas_are_backward_compatible():
    analysis_id = "a1"
    feedback = SubjectiveAccuracyFeedback(rating=4, free_text="Felt specific.", analysis_id=analysis_id)
    report = ReportFeedback(rating=5, analysis_id=analysis_id, section_id="mirror")
    drill = DrillFeedback(rating=3, analysis_id=analysis_id, drill_id="pause_ownership_v1")
    retest = RetestFeedback(rating=4, analysis_id="a2", baseline_analysis_id="a1", retest_analysis_id="a2")
    general = GeneralFeedback(rating=5, analysis_id=analysis_id)

    assert feedback.schema_version == "authority.v2.analytics"
    assert report.section_id == "mirror"
    assert drill.drill_id == "pause_ownership_v1"
    assert retest.baseline_analysis_id == "a1"
    assert general.feedback_type == "general"


@patch("services.coaching_engine._get_client")
@patch("services.inference_engine._get_client")
@patch("main._get_client")
def test_endpoint_returns_analytics_without_changing_core_outputs(
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
        data={"context": "benchmark", "installation_id": "install-analytics"},
    )

    assert response.status_code == 200
    payload = response.json()
    model = AnalyticsBundle.model_validate(payload["analytics"])
    assert model.analysis.analysis_id == payload["analysis_id"]
    assert model.calibration_record.authority_score == payload["scores"]["authority_score"]
    assert model.user_behaviour_events[0].event_type == "benchmark_completed"
