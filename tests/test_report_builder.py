"""Milestone 5 deterministic report builder tests."""

from __future__ import annotations

from schemas import AudioQuality, EvidenceItem, Moment, Uncertainty
from services.report_builder import build_report
from tests.test_psychological_inference import _infer, _metrics, _scores


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id="ev_command_1",
            trait="command",
            direction="positive",
            headline="Falling endings supported command",
            why_it_matters="Finality helps the listener feel led.",
            signals=["terminal_falling_ratio"],
        ),
        EvidenceItem(
            id="ev_structure_1",
            trait="structure",
            direction="positive",
            headline="Clear structure supported credibility",
            why_it_matters="Structure reduces listener effort.",
            signals=["structure_score"],
        ),
        EvidenceItem(
            id="ev_presence_1",
            trait="presence",
            direction="positive",
            headline="Dynamic emphasis supported presence",
            why_it_matters="Contrast makes the point more memorable.",
            signals=["dynamic_emphasis_score"],
        ),
    ]


def _moments() -> list[Moment]:
    return [
        Moment(
            moment_id="m1",
            type="strongest_moment",
            start_ms=1000,
            end_ms=4000,
            severity="highlight",
            headline="Most authoritative section",
            summary="Pace and ending aligned here.",
            dimension_impact={"command": 0.2},
            preview_visible_free=True,
        )
    ]


def _report(**kwargs):
    scores = kwargs.pop("scores", _scores())
    metrics = kwargs.pop("metrics", _metrics())
    audio_quality = kwargs.pop(
        "audio_quality",
        AudioQuality(usable=True, background_noise_level="low"),
    )
    uncertainty = kwargs.pop(
        "uncertainty",
        Uncertainty(overall_confidence_label="medium_high", reasons=[]),
    )
    inference = kwargs.pop("inference", _infer(metrics, audio_quality=audio_quality))
    return build_report(
        scores=scores,
        metrics=metrics,
        psychological_inference=inference,
        evidence=kwargs.pop("evidence", _evidence()),
        moments=kwargs.pop("moments", _moments()),
        uncertainty=uncertainty,
        audio_quality=audio_quality,
        duration_ms=kwargs.pop("duration_ms", 60000),
        scenario=kwargs.pop("scenario", "benchmark"),
    )


def test_report_builder_populates_all_major_sections_with_evidence_ids():
    report = _report()

    assert report.mirror is not None
    assert report.diagnosis is not None
    assert report.perception_map is not None
    assert report.hidden_cost is not None
    assert report.highest_leverage_fix is not None
    assert report.training_prescription is not None
    assert report.retest_plan is not None
    assert report.authority_type is not None
    assert report.share_card is not None
    assert report.technical_appendix is not None

    assert report.mirror.evidence_ids
    assert report.diagnosis.supporting_evidence_ids
    assert report.hidden_cost.evidence_ids
    assert report.highest_leverage_fix.evidence_ids
    assert report.training_prescription.evidence_ids
    assert report.retest_plan.evidence_ids
    assert report.authority_type.evidence_ids
    assert report.technical_appendix.evidence_ids
    assert report.perception_map.first_impression.evidence_ids


def test_highest_leverage_fix_and_training_are_deterministic_from_limiter():
    scores = _scores().model_copy(
        update={
            "dimension_scores": _scores().dimension_scores.model_copy(
                update={"command": 42, "clarity": 76, "composure": 64}
            )
        }
    )
    report = _report(scores=scores)

    assert report.diagnosis.limiting_dimension == "Command"
    assert report.highest_leverage_fix.issue == "declarative finality"
    assert report.highest_leverage_fix.first_drill_id == "drop_the_landing_v1"
    assert "command" in report.highest_leverage_fix.target_dimensions
    assert report.training_prescription.drill_id == "drop_the_landing_v1"
    assert report.training_prescription.instructions
    assert report.retest_plan.focus_metric == "terminal_rising_ratio"


def test_authority_type_mapping_supports_required_type_outputs():
    executive_scores = _scores().model_copy(
        update={
            "authority_score": 91,
            "dimension_scores": _scores().dimension_scores.model_copy(
                update={
                    "command": 86,
                    "clarity": 84,
                    "composure": 85,
                    "presence": 83,
                    "persuasion": 78,
                    "structure": 80,
                }
            ),
        }
    )
    quiet_scores = _scores().model_copy(
        update={
            "dimension_scores": _scores().dimension_scores.model_copy(
                update={"clarity": 74, "presence": 50, "command": 58, "structure": 66}
            )
        }
    )
    rushed_scores = _scores().model_copy(
        update={
            "derived_axes": _scores().derived_axes.model_copy(update={"nervousness": 72}),
            "dimension_scores": _scores().dimension_scores.model_copy(
                update={"composure": 52, "command": 56}
            ),
        }
    )

    assert _report(scores=executive_scores).authority_type.label == "Executive Presence"
    assert _report(scores=quiet_scores).authority_type.label == "Quiet Analyst"
    assert _report(scores=rushed_scores).authority_type.label == "Rushed Achiever"


def test_share_card_is_public_safe_and_omits_private_findings():
    metrics = _metrics(
        linguistic={"filler_words_per_min": 14.0},
        rhythm={"speed_up_segments": 2},
        derived={"hesitation_cluster_score": 0.8},
    )
    report = _report(metrics=metrics)
    share = report.share_card

    assert share.share_safety == "public_safe"
    assert share.hidden_private_findings == []
    public_text = " ".join(
        str(value)
        for value in (
            share.authority_type,
            share.top_strength,
            share.growth_area,
            share.one_line_identity_read,
            share.percentile_label,
        )
    ).lower()
    assert "approval" not in public_text
    assert "nervous" not in public_text


def test_poor_audio_propagates_uncertainty_without_fabricating_confidence():
    audio_quality = AudioQuality(
        usable=False,
        background_noise_level="high",
        quality_warnings=["Very low signal level"],
    )
    uncertainty = Uncertainty(
        overall_confidence_label="low",
        reasons=["Poor microphone signal"],
    )
    report = _report(audio_quality=audio_quality, uncertainty=uncertainty, duration_ms=9000)

    assert report.uncertainty.reasons
    assert "Poor microphone signal" in report.uncertainty.reasons
    assert "Short recording limits full report confidence" in report.uncertainty.reasons
    assert report.technical_appendix.audio_quality_warnings == ["Very low signal level"]
    assert report.share_card.share_safety == "public_safe"
