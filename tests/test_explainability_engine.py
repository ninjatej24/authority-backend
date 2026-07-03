"""Milestone 11 deterministic explainability and safety tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from main import app
from schemas import AudioQuality, AuthorityV2Response, Uncertainty
from services.deterministic_coaching import build_deterministic_coaching
from services.explainability_engine import build_explainability
from services.progress_engine import build_progress
from tests.test_analyze_endpoint import _FakeTranscription, _fake_gpt_json, _make_wav_bytes
from tests.test_diagnostic_reasoning import _diagnostic, _softened_expert_scores
from tests.test_progress_engine import _snapshot
from tests.test_psychological_inference import _infer, _metrics
from tests.test_report_builder import _evidence, _moments
from tests.test_report_generation import _generated_report


def _bundle(**kwargs):
    metrics = kwargs.pop("metrics", _metrics())
    scores = kwargs.pop("scores", _softened_expert_scores())
    audio_quality = kwargs.pop("audio_quality", AudioQuality(usable=True, background_noise_level="low"))
    uncertainty = kwargs.pop("uncertainty", Uncertainty(overall_confidence_label="medium_high", reasons=[]))
    evidence = kwargs.pop("evidence", _evidence())
    moments = kwargs.pop("moments", _moments())
    inference = kwargs.pop("inference", _infer(metrics, audio_quality=audio_quality, duration_ms=60000))
    diagnostic = kwargs.pop(
        "diagnostic_reasoning",
        _diagnostic(
            scores=scores,
            metrics=metrics,
            audio_quality=audio_quality,
            uncertainty=uncertainty,
            duration_ms=60000,
            scenario="benchmark",
            inference=inference,
            evidence=evidence,
            moments=moments,
        ),
    )
    coaching = kwargs.pop(
        "coaching",
        build_deterministic_coaching(
            metrics=metrics,
            scores=scores,
            psychological_inference=inference,
            diagnostic_reasoning=diagnostic,
            report=None,
            audio_quality=audio_quality,
            uncertainty=uncertainty,
            duration_ms=60000,
            scenario="benchmark",
        ),
    )
    report = kwargs.pop(
        "report",
        _generated_report(
            scores=scores,
            metrics=metrics,
            audio_quality=audio_quality,
            uncertainty=uncertainty,
            evidence=evidence,
            moments=moments,
            inference=inference,
            diagnostic_reasoning=diagnostic,
        ),
    )
    progress = kwargs.pop("progress", build_progress(_snapshot()))
    return build_explainability(
        metrics=metrics,
        evidence=evidence,
        psychological_inference=inference,
        diagnostic_reasoning=diagnostic,
        scores=scores,
        scenario="benchmark",
        coaching_engine=coaching,
        report=report,
        progress=progress,
        moments=moments,
        audio_quality=audio_quality,
        uncertainty=uncertainty,
    )


def _claim(bundle, claim_id: str):
    return next(item for item in bundle.claims if item.claim_id == claim_id)


def test_same_input_produces_identical_explainability_and_claim_coverage():
    first = _bundle()
    second = _bundle()

    assert first.model_dump() == second.model_dump()
    claim_ids = {claim.claim_id for claim in first.claims}
    assert {
        "mirror_headline",
        "authority_type",
        "primary_diagnosis",
        "hidden_cost",
        "highest_leverage_fix",
        "training_prescription",
        "authority_score",
        "dimension_scores",
        "top_strength",
        "primary_limiter",
        "authority_evolution",
        "scenario_adjustments",
        "share_card",
        "progress_interpretation",
        "coaching_selection",
        "retest_recommendation",
    }.issubset(claim_ids)


def test_unsupported_and_low_confidence_claims_are_suppressed():
    bundle = _bundle(audio_quality=AudioQuality(usable=False, background_noise_level="high"))

    assert bundle.audit.suppressed_claim_count > 0
    assert any(claim.suppressed for claim in bundle.claims)
    assert any("poor_audio" in claim.validation.failed_checks for claim in bundle.claims)


def test_low_confidence_propagates_to_claim_reasons_and_alternatives():
    uncertainty = Uncertainty(overall_confidence_label="low", reasons=["Short recording limits confidence"])
    bundle = _bundle(uncertainty=uncertainty)
    diagnosis = _claim(bundle, "primary_diagnosis")

    assert "Short recording limits confidence" in diagnosis.uncertainty_reasons
    assert diagnosis.alternative_interpretations


def test_contradictions_detected_from_dimension_pattern():
    scores = _softened_expert_scores().model_copy(
        update={
            "authority_score": 84,
            "score_confidence": 0.5,
            "dimension_scores": _softened_expert_scores().dimension_scores.model_copy(
                update={"command": 78, "composure": 50, "persuasion": 76, "clarity": 54}
            ),
        }
    )
    bundle = _bundle(scores=scores)
    ids = {item.contradiction_id for item in bundle.contradictions}

    assert "high_command_low_composure" in ids
    assert "high_persuasion_low_clarity" in ids
    assert "high_score_low_confidence" in ids


def test_validation_catches_broken_references_and_integrity_drops():
    report = _generated_report()
    report = report.model_copy(
        update={
            "mirror": report.mirror.model_copy(update={"evidence_ids": ["missing_ev"]}),
        }
    )
    bundle = _bundle(report=report)

    assert "missing_ev" in bundle.validation_summary.orphan_evidence_ids
    assert bundle.audit.orphan_evidence_count == 1
    assert bundle.audit.report_integrity_score < 1.0


def test_audit_metadata_is_deterministic_and_counts_claims():
    bundle = _bundle()

    assert bundle.audit.engine_version == "explainability_v1"
    assert "explainability" in bundle.audit.analysis_steps_completed
    assert bundle.audit.validated_claim_count + bundle.audit.failed_validation_count == len(bundle.claims)
    assert 0.0 <= bundle.audit.report_integrity_score <= 1.0


@patch("services.coaching_engine._get_client")
@patch("services.inference_engine._get_client")
@patch("main._get_client")
def test_endpoint_returns_explainability(mock_main_get_client, mock_inference_get_client, mock_coaching_get_client):
    mock_main_client = MagicMock()
    mock_main_client.audio.transcriptions.create.return_value = _FakeTranscription()
    mock_main_get_client.return_value = mock_main_client

    mock_inference_client = MagicMock()
    cognition_response = MagicMock()
    cognition_response.choices = [MagicMock(message=MagicMock(content=_fake_gpt_json()))]
    mock_inference_client.chat.completions.create.return_value = cognition_response
    mock_inference_get_client.return_value = mock_inference_client
    mock_coaching_get_client.return_value = MagicMock()

    response = TestClient(app).post(
        "/analyze",
        files={"file": ("sample.wav", _make_wav_bytes(), "audio/wav")},
        data={"context": "initial", "title": "Test"},
    )

    assert response.status_code == 200
    model = AuthorityV2Response.model_validate(response.json())
    assert model.explainability.claims
    assert model.report.explainability is not None
