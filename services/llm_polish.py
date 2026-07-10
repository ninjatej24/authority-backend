"""Milestone 14 LLM polish layer.

This module is presentation-only. It never changes deterministic conclusions,
references, scores, diagnoses, coaching selections, or report structure.
"""

from __future__ import annotations

import json
from typing import Any

from schemas import (
    AuthorityReport,
    CoachingEngine,
    ExplainabilityBundle,
    MomentIntelligence,
    PipelineValidation,
    PolishedAuthorityReport,
    PolishedListItem,
    PolishedTextSection,
    Progress,
)


ENGINE_VERSION = "llm_polish_v1"
DEFAULT_MODEL = "gpt-4o-mini"
SECTION_KEYS = {
    "mirror",
    "diagnosis",
    "perception_map",
    "hidden_cost",
    "highest_leverage_fix",
    "training_prescription",
    "retest_plan",
    "timeline",
    "evidence",
    "share_card",
    "weekly_summary",
    "progress_summary",
}
UNCERTAINTY_MARKERS = ("likely", "may", "suggest", "suggests", "can", "could")


SYSTEM_PROMPT = """You are rewriting language only.

You must preserve every factual statement.
You may not alter numerical values.
You may not invent evidence.
You may not increase certainty.
You may not remove uncertainty wording.
You may not change the coaching recommendation.
You may only improve wording.

Write in the Authority style: sharp, specific, premium, calm, psychologically literate, direct, evidence-backed, and emotionally revealing.
Never use therapy language, startup cliches, motivational fluff, generic self-help, fake certainty, toxic language, or humiliating phrasing.
Return only valid JSON matching the requested structure."""


def _clean_json(content: str) -> str:
    content = (content or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3:
            content = "\n".join(lines[1:-1]).strip()
    return content


def _text_section(text: str | None, status: str = "fallback", warnings: list[str] | None = None) -> PolishedTextSection:
    return PolishedTextSection(
        original_text=text,
        polished_text=text,
        status=status,  # type: ignore[arg-type]
        warnings=warnings or [],
    )


def _list_item(item_id: str, text: str | None, status: str = "fallback", warnings: list[str] | None = None) -> PolishedListItem:
    return PolishedListItem(
        item_id=item_id,
        original_text=text,
        polished_text=text,
        status=status,  # type: ignore[arg-type]
        warnings=warnings or [],
    )


def _diagnosis_text(report: AuthorityReport) -> str | None:
    if not report.diagnosis:
        return None
    pieces = [
        report.diagnosis.core_pattern or report.diagnosis.core_behavioural_pattern,
        report.diagnosis.social_consequence,
    ]
    return " ".join(piece for piece in pieces if piece) or None


def _fix_text(report: AuthorityReport) -> str | None:
    fix = report.highest_leverage_fix
    if not fix:
        return None
    pieces = [fix.issue, fix.plain_english, fix.why_this_matters]
    return " ".join(piece for piece in pieces if piece) or None


def _training_text(report: AuthorityReport) -> str | None:
    training = report.training_prescription
    if not training:
        return None
    pieces = [training.title, training.why_chosen, " ".join(training.instructions), training.success_signal]
    return " ".join(piece for piece in pieces if piece) or None


def _retest_text(report: AuthorityReport) -> str | None:
    retest = report.retest_plan
    if not retest:
        return None
    pieces = [retest.focus_metric, retest.success_definition]
    if retest.recommended_retest_after_days is not None:
        pieces.insert(0, f"Retest after {retest.recommended_retest_after_days} days")
    return " ".join(piece for piece in pieces if piece) or None


def _progress_text(progress: Progress) -> str | None:
    if progress.comparison_available and progress.comparison:
        return f"Progress status: {progress.comparison.overall_trend}. Score delta: {progress.comparison.authority_score_delta}."
    return progress.state.user_safe_explanation or progress.state.progress_status


def _weekly_text(progress: Progress) -> str | None:
    summary = progress.weekly_summary
    pieces = [
        summary.largest_improvement,
        summary.remaining_limiter,
        summary.recommended_focus,
    ]
    return " ".join(piece for piece in pieces if piece) or None


def _fallback_report(
    report: AuthorityReport,
    progress: Progress,
    *,
    warning: str,
    status: str = "fallback",
    model: str | None = None,
) -> PolishedAuthorityReport:
    perception = {}
    if report.perception_map:
        for key, value in report.perception_map.model_dump().items():
            if value:
                perception[key] = _text_section(value.get("text"), status, [warning])
    return PolishedAuthorityReport(
        engine_version=ENGINE_VERSION,
        status=status,  # type: ignore[arg-type]
        model=model,
        warnings=[warning],
        mirror=_text_section(report.mirror.headline if report.mirror else None, status, [warning]),
        diagnosis=_text_section(_diagnosis_text(report), status, [warning]),
        perception_map=perception,
        hidden_cost=_text_section(report.hidden_cost.consequence if report.hidden_cost else None, status, [warning]),
        highest_leverage_fix=_text_section(_fix_text(report), status, [warning]),
        training_prescription=_text_section(_training_text(report), status, [warning]),
        retest_plan=_text_section(_retest_text(report), status, [warning]),
        timeline=[_list_item(item.moment_id, item.summary, status, [warning]) for item in report.timeline],
        evidence=[_list_item(item.evidence_id, item.why_it_matters, status, [warning]) for item in report.evidence_chain],
        share_card=_text_section(report.share_card.one_line_identity_read if report.share_card else None, status, [warning]),
        weekly_summary=_text_section(_weekly_text(progress), status, [warning]),
        progress_summary=_text_section(_progress_text(progress), status, [warning]),
        preserved_ids=_preserved_ids(report),
    )


def _preserved_ids(report: AuthorityReport) -> dict[str, list[str]]:
    return {
        "evidence_ids": [item.evidence_id for item in report.evidence_chain],
        "moment_ids": [item.moment_id for item in report.timeline],
        "drill_ids": [report.training_prescription.drill_id] if report.training_prescription and report.training_prescription.drill_id else [],
        "diagnosis_ids": [report.primary_diagnosis.diagnosis_id] if report.primary_diagnosis else [],
    }


def _prompt_payload(
    report: AuthorityReport,
    explainability: ExplainabilityBundle,
    pipeline_validation: PipelineValidation,
    progress: Progress,
    moment_intelligence: MomentIntelligence,
    coaching: CoachingEngine,
) -> dict[str, Any]:
    perception = {}
    if report.perception_map:
        for key, value in report.perception_map.model_dump().items():
            if value and value.get("text"):
                perception[key] = {
                    "original_text": value.get("text"),
                    "evidence_ids": value.get("evidence_ids", []),
                    "confidence": value.get("confidence"),
                }
    primary = coaching.selected_interventions.primary_drill
    return {
        "style": {
            "tone": ["sharp", "specific", "premium", "calm", "direct", "evidence-backed"],
            "forbidden": ["new evidence", "new scores", "new moments", "new drills", "increased certainty", "humiliation"],
        },
        "ids_to_preserve": _preserved_ids(report),
        "validation": {
            "pipeline_valid": pipeline_validation.valid,
            "pipeline_health": pipeline_validation.audit.overall_pipeline_health,
            "explainability_integrity": explainability.audit.report_integrity_score,
        },
        "sections": {
            "mirror": {"original_text": report.mirror.headline if report.mirror else None},
            "diagnosis": {"original_text": _diagnosis_text(report), "diagnosis_id": report.primary_diagnosis.diagnosis_id if report.primary_diagnosis else None},
            "perception_map": perception,
            "hidden_cost": {"original_text": report.hidden_cost.consequence if report.hidden_cost else None, "evidence_ids": report.hidden_cost.evidence_ids if report.hidden_cost else []},
            "highest_leverage_fix": {"original_text": _fix_text(report), "drill_id": report.highest_leverage_fix.first_drill_id if report.highest_leverage_fix else None},
            "training_prescription": {"original_text": _training_text(report), "drill_id": report.training_prescription.drill_id if report.training_prescription else None},
            "retest_plan": {"original_text": _retest_text(report)},
            "timeline": [{"item_id": item.moment_id, "original_text": item.summary, "evidence_ids": item.evidence_ids} for item in report.timeline[:8]],
            "evidence": [{"item_id": item.evidence_id, "original_text": item.why_it_matters} for item in report.evidence_chain[:5]],
            "share_card": {"original_text": report.share_card.one_line_identity_read if report.share_card else None, "share_safety": report.share_card.share_safety if report.share_card else "public_safe"},
            "weekly_summary": {"original_text": _weekly_text(progress)},
            "progress_summary": {"original_text": _progress_text(progress)},
        },
        "moment_summary": {
            "top_free_moment": moment_intelligence.top_free_moment,
            "top_premium_moments": moment_intelligence.top_premium_moments,
        },
        "coaching_summary": {
            "primary_drill_id": primary.drill_id if primary else None,
            "primary_drill_title": primary.title if primary else None,
        },
    }


def _user_prompt(payload: dict[str, Any]) -> str:
    return (
        "Rewrite only the original_text fields into premium Authority wording. "
        "Return JSON with this shape: "
        '{"mirror":{"polished_text":"..."},"diagnosis":{"polished_text":"..."},'
        '"perception_map":{"first_impression":{"polished_text":"..."}},'
        '"hidden_cost":{"polished_text":"..."},"highest_leverage_fix":{"polished_text":"..."},'
        '"training_prescription":{"polished_text":"..."},"retest_plan":{"polished_text":"..."},'
        '"timeline":[{"item_id":"...","polished_text":"..."}],'
        '"evidence":[{"item_id":"...","polished_text":"..."}],'
        '"share_card":{"polished_text":"..."},"weekly_summary":{"polished_text":"..."},'
        '"progress_summary":{"polished_text":"..."}}.\n\n'
        f"Validated deterministic input:\n{json.dumps(payload, ensure_ascii=True)}"
    )


def _call_llm(client: Any, payload: dict[str, Any], model: str, timeout_seconds: float) -> dict[str, Any]:
    del timeout_seconds
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(payload)},
        ],
    )
    content = response.choices[0].message.content or ""
    parsed = json.loads(_clean_json(content))
    if not isinstance(parsed, dict):
        raise ValueError("polish_response_not_object")
    return parsed


def _candidate_text(raw: Any) -> str | None:
    if isinstance(raw, dict):
        value = raw.get("polished_text")
        return value if isinstance(value, str) else None
    return None


def _certainty_removed(original: str | None, polished: str | None) -> bool:
    if not original or not polished:
        return False
    original_lower = original.lower()
    polished_lower = polished.lower()
    original_markers = [marker for marker in UNCERTAINTY_MARKERS if marker in original_lower]
    if not original_markers:
        return False
    return not any(marker in polished_lower for marker in original_markers)


def _polish_section(original: str | None, raw: Any) -> PolishedTextSection:
    polished = _candidate_text(raw)
    if not original:
        return PolishedTextSection(original_text=None, polished_text=None, status="omitted")
    if not polished:
        return _text_section(original, "fallback", ["LLM omitted this section"])
    if _certainty_removed(original, polished):
        return _text_section(original, "rejected", ["LLM removed required uncertainty wording"])
    return PolishedTextSection(original_text=original, polished_text=polished, status="polished")


def _polish_item(item_id: str, original: str | None, raw: dict[str, Any] | None) -> PolishedListItem:
    if not original:
        return PolishedListItem(item_id=item_id, original_text=None, polished_text=None, status="omitted")
    polished = _candidate_text(raw)
    if not polished:
        return _list_item(item_id, original, "fallback", ["LLM omitted this item"])
    if _certainty_removed(original, polished):
        return _list_item(item_id, original, "rejected", ["LLM removed required uncertainty wording"])
    return PolishedListItem(item_id=item_id, original_text=original, polished_text=polished, status="polished")


def _assemble_polished(report: AuthorityReport, progress: Progress, parsed: dict[str, Any], model: str) -> PolishedAuthorityReport:
    warnings = []
    unknown = sorted(set(parsed) - SECTION_KEYS)
    if unknown:
        warnings.append(f"Ignored unknown polish sections: {', '.join(unknown)}")
    perception: dict[str, PolishedTextSection] = {}
    if report.perception_map:
        raw_perception = parsed.get("perception_map") if isinstance(parsed.get("perception_map"), dict) else {}
        for key, value in report.perception_map.model_dump().items():
            if value:
                perception[key] = _polish_section(value.get("text"), raw_perception.get(key))
    raw_timeline = {
        item.get("item_id"): item
        for item in parsed.get("timeline", [])
        if isinstance(item, dict) and isinstance(item.get("item_id"), str)
    }
    raw_evidence = {
        item.get("item_id"): item
        for item in parsed.get("evidence", [])
        if isinstance(item, dict) and isinstance(item.get("item_id"), str)
    }
    polished = PolishedAuthorityReport(
        engine_version=ENGINE_VERSION,
        status="polished",
        model=model,
        warnings=warnings,
        mirror=_polish_section(report.mirror.headline if report.mirror else None, parsed.get("mirror")),
        diagnosis=_polish_section(_diagnosis_text(report), parsed.get("diagnosis")),
        perception_map=perception,
        hidden_cost=_polish_section(report.hidden_cost.consequence if report.hidden_cost else None, parsed.get("hidden_cost")),
        highest_leverage_fix=_polish_section(_fix_text(report), parsed.get("highest_leverage_fix")),
        training_prescription=_polish_section(_training_text(report), parsed.get("training_prescription")),
        retest_plan=_polish_section(_retest_text(report), parsed.get("retest_plan")),
        timeline=[_polish_item(item.moment_id, item.summary, raw_timeline.get(item.moment_id)) for item in report.timeline],
        evidence=[_polish_item(item.evidence_id, item.why_it_matters, raw_evidence.get(item.evidence_id)) for item in report.evidence_chain],
        share_card=_polish_section(report.share_card.one_line_identity_read if report.share_card else None, parsed.get("share_card")),
        weekly_summary=_polish_section(_weekly_text(progress), parsed.get("weekly_summary")),
        progress_summary=_polish_section(_progress_text(progress), parsed.get("progress_summary")),
        preserved_ids=_preserved_ids(report),
    )
    statuses = []
    for value in [
        polished.mirror,
        polished.diagnosis,
        polished.hidden_cost,
        polished.highest_leverage_fix,
        polished.training_prescription,
        polished.retest_plan,
        polished.share_card,
        polished.weekly_summary,
        polished.progress_summary,
        *polished.perception_map.values(),
        *polished.timeline,
        *polished.evidence,
    ]:
        statuses.append(value.status)
    if any(status in {"fallback", "rejected"} for status in statuses):
        polished = polished.model_copy(update={"status": "partial"})
    return polished


def polish_authority_report(
    *,
    report: AuthorityReport,
    explainability: ExplainabilityBundle,
    pipeline_validation: PipelineValidation,
    progress: Progress,
    moment_intelligence: MomentIntelligence,
    coaching: CoachingEngine,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    timeout_seconds: float = 8.0,
) -> PolishedAuthorityReport:
    """Polish deterministic wording without changing any conclusions."""
    if report.report_mode == "insufficient":
        return _fallback_report(report, progress, warning="Report insufficient for polish", status="fallback", model=model)
    if client is None:
        return _fallback_report(report, progress, warning="LLM polish unavailable", status="fallback", model=model)
    payload = _prompt_payload(
        report=report,
        explainability=explainability,
        pipeline_validation=pipeline_validation,
        progress=progress,
        moment_intelligence=moment_intelligence,
        coaching=coaching,
    )
    try:
        parsed = _call_llm(client, payload, model, timeout_seconds)
        return _assemble_polished(report, progress, parsed, model)
    except Exception as exc:
        return _fallback_report(report, progress, warning=f"LLM polish fallback: {type(exc).__name__}", status="fallback", model=model)
