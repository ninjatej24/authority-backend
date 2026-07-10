"""Milestone 12 deterministic pipeline validation tests."""

from __future__ import annotations

from schemas import (
    AudioQuality,
    AuthorityV2Response,
    PerceptionHighlight,
    PerceptionProfile,
    PerceptionReads,
    Recommendations,
    Uncertainty,
)
from services.deterministic_coaching import build_deterministic_coaching
from services.explainability_engine import build_explainability
from services.pipeline_validator import build_pipeline_validation
from services.progress_engine import build_progress
from tests.test_diagnostic_reasoning import _diagnostic, _softened_expert_scores
from tests.test_progress_engine import _snapshot
from tests.test_psychological_inference import _infer, _metrics
from tests.test_report_builder import _evidence, _moments
from tests.test_report_generation import _generated_report


def _perception() -> PerceptionProfile:
    return PerceptionProfile(
        headline="This recording suggests controlled authority.",
        how_you_currently_come_across="Listeners are likely to hear a clear point.",
        biggest_strength=PerceptionHighlight(title="Command", explanation="The point lands cleanly."),
        biggest_drag=PerceptionHighlight(title="Structure", explanation="The path could be tighter."),
        listener_assumptions=["The speaker is prepared."],
        reads=PerceptionReads(
            emotional="steady",
            professional="credible",
            social_status="clear",
            interview="ready",
            leadership="promising",
        ),
    )


def _response(**kwargs) -> AuthorityV2Response:
    metrics = kwargs.pop(
        "metrics",
        _metrics(
            linguistic={
                "specificity_score": 0.12,
                "concreteness_score": 0.08,
                "opening_strength_score": 0.82,
                "closing_strength_score": 0.72,
                "structure_score": 0.74,
            }
        ),
    )
    scores = kwargs.pop("scores", _softened_expert_scores())
    audio_quality = kwargs.pop("audio_quality", AudioQuality(usable=True, background_noise_level="low"))
    uncertainty = kwargs.pop("uncertainty", Uncertainty(overall_confidence_label="medium_high", reasons=[]))
    evidence = kwargs.pop("evidence", _evidence())
    moments = kwargs.pop("moments", _moments())
    scenario = kwargs.pop("scenario", "benchmark")
    inference = kwargs.pop("inference", _infer(metrics, audio_quality=audio_quality, duration_ms=60000))
    diagnostic = kwargs.pop(
        "diagnostic_reasoning",
        _diagnostic(
            scores=scores,
            metrics=metrics,
            audio_quality=audio_quality,
            uncertainty=uncertainty,
            duration_ms=60000,
            scenario=scenario,
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
            scenario=scenario,
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
            scenario=scenario,
        ),
    )
    progress = kwargs.pop("progress", build_progress(_snapshot(scenario=scenario)))
    explainability = kwargs.pop(
        "explainability",
        build_explainability(
            metrics=metrics,
            evidence=evidence,
            psychological_inference=inference,
            diagnostic_reasoning=diagnostic,
            scores=scores,
            scenario=scenario,
            coaching_engine=coaching,
            report=report,
            progress=progress,
            moments=moments,
            audio_quality=audio_quality,
            uncertainty=uncertainty,
        ),
    )
    response = AuthorityV2Response(
        request={"scenario": scenario, "prompt_id": "authority_benchmark_v1", "language": "en", "duration_ms": 60000},
        audio_quality=audio_quality,
        transcript={"full_text": "I believe we should move forward with clarity.", "words": []},
        scores=scores,
        metrics=metrics,
        perception_profile=_perception(),
        evidence=evidence,
        moments=moments,
        recommendations=Recommendations(
            highest_leverage_issue="Pause ownership",
            fastest_improvement_tip="Use a deliberate pause before key claims.",
            coaching_summary="The selected drill is supported by the strongest evidence cluster.",
        ),
        drills=[],
        psychological_inference=inference,
        report=report.model_copy(update={"progress": progress, "explainability": explainability}),
        coaching_engine=coaching,
        progress=progress,
        explainability=explainability,
        uncertainty=uncertainty,
    )
    return response.model_copy(update={"pipeline_validation": build_pipeline_validation(response)})


def test_valid_pipeline_returns_canonical_stage_order_and_contract():
    response = _response()
    validation = response.pipeline_validation

    assert validation.schema_version == "authority.v2"
    assert validation.valid is True
    assert [stage.stage_id for stage in validation.stages] == [
        "audio_quality",
        "transcription",
        "vad",
        "metrics",
        "psychological_inference",
        "diagnostic_reasoning",
        "scoring",
        "scenario",
        "coaching",
        "report_generation",
        "progress",
        "explainability",
        "final_response",
    ]
    assert validation.audit.overall_pipeline_health in {"healthy", "degraded"}
    AuthorityV2Response.model_validate(response.model_dump())


def test_report_only_evidence_and_moments_do_not_validate_as_upstream():
    response = _response()
    report = response.report.model_copy(
        update={
            "mirror": response.report.mirror.model_copy(update={"evidence_ids": ["report_only_ev"]}),
            "evidence_chain": [
                *response.report.evidence_chain,
                response.report.evidence_chain[0].model_copy(update={"evidence_id": "report_only_ev"}),
            ],
            "timeline": [
                *response.report.timeline,
                response.report.timeline[0].model_copy(update={"moment_id": "report_only_moment"}),
            ],
        }
    )
    validation = build_pipeline_validation(response.model_copy(update={"report": report}))

    assert validation.valid is False
    assert "report_only_ev" in validation.missing_references["evidence"]
    assert "report_only_moment" in validation.missing_references["moments"]


def test_duplicate_ids_and_invalid_timestamps_are_reported():
    response = _response()
    duplicate_evidence = response.evidence + [response.evidence[0]]
    bad_moment = response.moments[0].model_copy(update={"start_ms": 5000, "end_ms": 1000})
    validation = build_pipeline_validation(
        response.model_copy(update={"evidence": duplicate_evidence, "moments": [bad_moment]})
    )

    assert response.evidence[0].id in validation.duplicate_ids["evidence"]
    assert any(issue.issue_id == "invalid_moment_timestamp" for issue in validation.integrity_issues)
    assert validation.valid is False


def test_missing_stage_outputs_do_not_crash_and_are_reported():
    response = _response().model_copy(update={"psychological_inference": None, "coaching_engine": None})
    validation = build_pipeline_validation(response)

    assert "psychological_inference" in validation.null_outputs
    assert "coaching" in validation.null_outputs
    assert validation.valid is False


def test_training_drill_share_card_and_progress_consistency_are_checked():
    response = _response()
    bad_report = response.report.model_copy(
        update={
            "training_prescription": response.report.training_prescription.model_copy(update={"drill_id": "different_drill_v1"}),
            "share_card": response.report.share_card.model_copy(update={"authority_type": "Different Type"}),
        }
    )
    bad_progress = response.progress.model_copy(update={"delta_authority_score": 99})
    validation = build_pipeline_validation(response.model_copy(update={"report": bad_report, "progress": bad_progress}))
    issue_ids = {issue.issue_id for issue in validation.dependency_issues + validation.consistency_issues + validation.integrity_issues}

    assert "training_drill_not_from_coaching" in issue_ids
    assert "share_card_authority_type_mismatch" in issue_ids
    assert "missing_drill_references" in issue_ids


def test_pipeline_validation_is_deterministic():
    response = _response()
    first = build_pipeline_validation(response)
    second = build_pipeline_validation(response)

    assert first.model_dump() == second.model_dump()
