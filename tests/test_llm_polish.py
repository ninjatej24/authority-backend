"""Milestone 14 LLM polish layer tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

from schemas import AuthorityV2Response, CoachingEngine, ExplainabilityBundle, PipelineValidation
from services.llm_polish import polish_authority_report
from services.progress_engine import build_progress
from tests.test_progress_engine import _snapshot
from tests.test_report_generation import _generated_report


class _FakeClient:
    def __init__(self, payload: dict | str | Exception):
        self.payload = payload
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        if isinstance(self.payload, Exception):
            raise self.payload
        content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _context():
    report = _generated_report()
    return {
        "report": report,
        "explainability": ExplainabilityBundle(),
        "pipeline_validation": PipelineValidation(valid=True),
        "progress": build_progress(_snapshot()),
        "moment_intelligence": report.moment_intelligence,
        "coaching": report.coaching_engine or CoachingEngine(),
    }


def _polish_payload(report):
    return {
        "mirror": {"polished_text": "This recording suggests capable authority with a sharper edge."},
        "diagnosis": {"polished_text": "The pattern is clear: the strength lands, while the limiter sets the ceiling."},
        "perception_map": {
            "first_impression": {"polished_text": "Listeners are likely to hear a capable speaker with room for more command."}
        },
        "hidden_cost": {"polished_text": "The hidden cost is that the point may be understood before it is fully felt."},
        "highest_leverage_fix": {"polished_text": "The highest-leverage fix is to make the selected drill land with cleaner control."},
        "training_prescription": {"polished_text": "Use the selected drill exactly as prescribed, with calmer execution."},
        "retest_plan": {"polished_text": "Retest on the same prompt and compare the same target signals."},
        "timeline": [
            {"item_id": report.timeline[0].moment_id, "polished_text": "This moment shows the clearest local authority signal."},
            {"item_id": "invented_moment", "polished_text": "This should be ignored."},
        ],
        "evidence": [
            {"item_id": report.evidence_chain[0].evidence_id, "polished_text": "This evidence explains why the read is grounded."},
            {"item_id": "invented_evidence", "polished_text": "This should be ignored."},
        ],
        "share_card": {"polished_text": "A concise public-safe read of the same result."},
        "weekly_summary": {"polished_text": "The next summary should stay focused on the same deterministic target."},
        "progress_summary": {"polished_text": "Progress is not fabricated when no comparison exists."},
        "invented_section": {"polished_text": "This should be ignored."},
    }


def test_polish_returns_structured_sidecar_without_changing_report():
    ctx = _context()
    original_report = ctx["report"].model_dump()
    polished = polish_authority_report(client=_FakeClient(_polish_payload(ctx["report"])), **ctx)

    assert polished.status in {"polished", "partial"}
    assert polished.mirror.polished_text != polished.mirror.original_text
    assert ctx["report"].model_dump() == original_report
    assert polished.preserved_ids["evidence_ids"] == [item.evidence_id for item in ctx["report"].evidence_chain]


def test_llm_cannot_invent_moments_evidence_or_sections():
    ctx = _context()
    polished = polish_authority_report(client=_FakeClient(_polish_payload(ctx["report"])), **ctx)

    assert "invented_section" in polished.warnings[0]
    assert "invented_moment" not in [item.item_id for item in polished.timeline]
    assert "invented_evidence" not in [item.item_id for item in polished.evidence]
    assert [item.item_id for item in polished.timeline] == [item.moment_id for item in ctx["report"].timeline]


def test_uncertainty_removal_is_rejected():
    ctx = _context()
    report = ctx["report"].model_copy(
        update={"mirror": ctx["report"].mirror.model_copy(update={"headline": "This recording suggests you may sound capable."})}
    )
    payload = _polish_payload(report)
    payload["mirror"] = {"polished_text": "You sound capable."}
    ctx["report"] = report
    polished = polish_authority_report(client=_FakeClient(payload), **ctx)

    assert polished.mirror.status == "rejected"
    assert polished.mirror.polished_text == "This recording suggests you may sound capable."


def test_fallback_activates_on_llm_failure_and_invalid_json():
    ctx = _context()
    failed = polish_authority_report(client=_FakeClient(RuntimeError("rate_limit")), **ctx)
    invalid = polish_authority_report(client=_FakeClient("not json"), **ctx)

    assert failed.status == "fallback"
    assert invalid.status == "fallback"
    assert failed.mirror.polished_text == ctx["report"].mirror.headline
    assert invalid.training_prescription.polished_text


def test_endpoint_schema_accepts_polished_report_sidecar():
    ctx = _context()
    polished = polish_authority_report(client=_FakeClient(_polish_payload(ctx["report"])), **ctx)
    payload = {
        "schema_version": "authority.v2",
        "request": {"scenario": "benchmark", "prompt_id": "p", "language": "en", "duration_ms": 60000},
        "audio_quality": {},
        "transcript": {},
        "scores": {
            "authority_score": 60,
            "score_confidence": 0.7,
            "dimension_scores": {"command": 60, "clarity": 60, "composure": 60, "presence": 60, "persuasion": 60, "structure": 60},
            "derived_axes": {"trust_warmth": 60, "dominance_status": 60, "nervousness": 40, "interview_readiness": 60, "leadership_readiness": 60},
            "score_components": {"weighted_base": 60, "bonuses": {}, "penalties": {}},
        },
        "metrics": {"raw_acoustic": {}, "linguistic": {}, "derived": {}},
        "perception_profile": {"headline": "h", "how_you_currently_come_across": "x", "biggest_strength": {"title": "s", "explanation": "e"}, "biggest_drag": {"title": "d", "explanation": "e"}, "listener_assumptions": [], "reads": {"emotional": "", "professional": "", "social_status": "", "interview": "", "leadership": ""}},
        "evidence": [],
        "moments": [],
        "recommendations": {"highest_leverage_issue": "x", "fastest_improvement_tip": "y", "coaching_summary": "z"},
        "drills": [],
        "polished_report": polished.model_dump(),
    }

    model = AuthorityV2Response.model_validate(payload)
    assert model.polished_report.engine_version == "llm_polish_v1"
