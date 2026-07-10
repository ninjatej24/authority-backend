"""Milestone 12 deterministic pipeline and response integrity validation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Iterable

from schemas import (
    AuthorityV2Response,
    PipelineAudit,
    PipelineStageStatus,
    PipelineValidation,
    PipelineValidationIssue,
)


PIPELINE_VERSION = "authority.v2.milestone12"
DIMENSIONS = {"command", "clarity", "composure", "presence", "persuasion", "structure"}


def _dedupe(values: Iterable[str | None]) -> list[str]:
    return [item for item in dict.fromkeys(value for value in values if value).keys()]


def _issue(
    issue_id: str,
    stage: str,
    message: str,
    *,
    severity: str = "warning",
    references: Iterable[str | None] = (),
) -> PipelineValidationIssue:
    return PipelineValidationIssue(
        issue_id=issue_id,
        stage=stage,
        severity=severity,  # type: ignore[arg-type]
        message=message,
        references=_dedupe(references),
    )


def _duplicates(values: Iterable[str | None]) -> list[str]:
    counts = Counter(value for value in values if value)
    return sorted(value for value, count in counts.items() if count > 1)


def _valid_iso_timestamp(value: str | None) -> bool:
    if not value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _stage(stage_id: str, completed: bool, dependencies: list[str], warnings: list[str] | None = None) -> PipelineStageStatus:
    stage_warnings = list(warnings or [])
    status = "completed" if completed and not stage_warnings else "warning" if completed else "failed"
    return PipelineStageStatus(
        stage_id=stage_id,
        completed=completed,
        status=status,  # type: ignore[arg-type]
        dependencies=dependencies,
        warnings=stage_warnings,
    )


def _upstream_evidence_ids(response: AuthorityV2Response) -> set[str]:
    evidence_ids = {item.id for item in getattr(response, "evidence", []) or []}
    report = getattr(response, "report", None)
    if report:
        evidence_ids.update(report.validation.evidence_ids_checked)
    inference = getattr(response, "psychological_inference", None)
    if inference:
        evidence_ids.update(item.evidence_id for item in inference.evidence_chain)
        for trait in inference.traits:
            evidence_ids.update(trait.supporting_evidence_ids)
        for behaviour in inference.micro_behaviours:
            evidence_ids.update(behaviour.supporting_evidence_ids)
    diagnostic = getattr(getattr(response, "report", None), "diagnostic_reasoning", None)
    if diagnostic:
        for diagnosis in [diagnostic.primary_diagnosis, diagnostic.secondary_diagnosis, *diagnostic.suppressed_diagnoses]:
            if diagnosis:
                evidence_ids.update(diagnosis.supporting_evidence_ids)
        for contradiction in diagnostic.contradictions:
            evidence_ids.update(contradiction.evidence_ids)
        if diagnostic.hidden_cost_reasoning:
            evidence_ids.update(diagnostic.hidden_cost_reasoning.evidence_ids)
        if diagnostic.highest_leverage_reasoning:
            evidence_ids.update(diagnostic.highest_leverage_reasoning.supporting_evidence)
        for reasoning in diagnostic.dimension_reasoning.values():
            evidence_ids.update(reasoning.supporting_evidence_ids)
    coaching = getattr(response, "coaching_engine", None)
    if coaching:
        for root in coaching.root_causes:
            evidence_ids.update(root.evidence_ids)
        for candidate in coaching.intervention_candidates + coaching.suppressed_interventions + coaching.future_training_queue:
            evidence_ids.update(candidate.supporting_evidence_ids)
        evidence_ids.update(coaching.reasoning_chain.evidence_ids)
    progress = getattr(response, "progress", None)
    if progress:
        for moment in progress.moment_comparison:
            evidence_ids.update(moment.evidence_ids)
    return {item for item in evidence_ids if item}


def _upstream_moment_ids(response: AuthorityV2Response) -> set[str]:
    moment_ids = {item.moment_id for item in getattr(response, "moments", []) or []}
    diagnostic = getattr(getattr(response, "report", None), "diagnostic_reasoning", None)
    if diagnostic:
        for diagnosis in [diagnostic.primary_diagnosis, diagnostic.secondary_diagnosis]:
            if diagnosis:
                moment_ids.update(diagnosis.supporting_moment_ids)
        if diagnostic.hidden_cost_reasoning:
            moment_ids.update(diagnostic.hidden_cost_reasoning.moment_ids)
    return {item for item in moment_ids if item}


def _drill_library_ids(response: AuthorityV2Response) -> set[str]:
    coaching = getattr(response, "coaching_engine", None)
    return {item.drill_id for item in coaching.drill_library} if coaching else set()


def _referenced_report_evidence(response: AuthorityV2Response) -> set[str]:
    report = getattr(response, "report", None)
    if not report:
        return set()
    referenced: set[str] = set()
    for section in [
        report.mirror,
        report.hidden_cost,
        report.highest_leverage_fix,
        report.training_prescription,
        report.retest_plan,
        report.authority_type,
        report.technical_appendix,
    ]:
        if section and hasattr(section, "evidence_ids"):
            referenced.update(section.evidence_ids)
    if report.diagnosis:
        referenced.update(report.diagnosis.supporting_evidence_ids)
        referenced.update(report.diagnosis.evidence_ids)
    if report.perception_map:
        for read in report.perception_map.model_dump().values():
            if isinstance(read, dict):
                referenced.update(read.get("evidence_ids", []))
    for item in report.evidence_chain:
        referenced.add(item.evidence_id)
    for item in report.timeline:
        referenced.update(item.evidence_ids)
    for dimension in report.dimension_reports.values():
        referenced.update(dimension.linked_evidence)
    return {item for item in referenced if item}


def _referenced_report_moments(response: AuthorityV2Response) -> set[str]:
    report = getattr(response, "report", None)
    return {item.moment_id for item in report.timeline} if report else set()


def _referenced_drills(response: AuthorityV2Response) -> set[str]:
    report = getattr(response, "report", None)
    coaching = getattr(response, "coaching_engine", None)
    referenced: set[str] = set()
    if report:
        if report.training_prescription and report.training_prescription.drill_id:
            referenced.add(report.training_prescription.drill_id)
        if report.highest_leverage_fix and report.highest_leverage_fix.first_drill_id:
            referenced.add(report.highest_leverage_fix.first_drill_id)
    if coaching:
        for candidate in coaching.intervention_candidates + coaching.suppressed_interventions + coaching.future_training_queue:
            referenced.add(candidate.drill_id)
        if coaching.selected_interventions.primary_drill:
            referenced.add(coaching.selected_interventions.primary_drill.drill_id)
        if coaching.selected_interventions.secondary_drill:
            referenced.add(coaching.selected_interventions.secondary_drill.drill_id)
        for dependency in coaching.dependency_graph:
            referenced.update([dependency.before, dependency.after])
    return {item for item in referenced if item}


def _collect_integrity(response: AuthorityV2Response) -> tuple[list[PipelineValidationIssue], dict[str, list[str]], dict[str, list[str]]]:
    issues: list[PipelineValidationIssue] = []
    duplicate_ids: dict[str, list[str]] = {}
    missing_refs: dict[str, list[str]] = {}

    evidence_dupes = _duplicates(item.id for item in getattr(response, "evidence", []) or [])
    moment_dupes = _duplicates(item.moment_id for item in getattr(response, "moments", []) or [])
    report = getattr(response, "report", None)
    report_evidence_dupes = _duplicates(item.evidence_id for item in report.evidence_chain) if report else []
    report_moment_dupes = _duplicates(item.moment_id for item in report.timeline) if report else []
    drill_dupes = _duplicates(item.drill_id for item in getattr(getattr(response, "coaching_engine", None), "drill_library", []) or [])

    for key, values in {
        "evidence": evidence_dupes,
        "moments": moment_dupes,
        "report_evidence": report_evidence_dupes,
        "report_moments": report_moment_dupes,
        "drills": drill_dupes,
    }.items():
        if values:
            duplicate_ids[key] = values
            issues.append(_issue(f"duplicate_{key}", "response_integrity", f"Duplicate {key} ids detected.", severity="error", references=values))

    upstream_evidence = _upstream_evidence_ids(response)
    upstream_moments = _upstream_moment_ids(response)
    drill_ids = _drill_library_ids(response)
    missing_evidence = sorted(_referenced_report_evidence(response) - upstream_evidence)
    missing_moments = sorted(_referenced_report_moments(response) - upstream_moments)
    missing_drills = sorted(_referenced_drills(response) - drill_ids)
    if missing_evidence:
        missing_refs["evidence"] = missing_evidence
        issues.append(_issue("missing_evidence_references", "response_integrity", "Report references evidence that is not present upstream.", severity="error", references=missing_evidence))
    if missing_moments:
        missing_refs["moments"] = missing_moments
        issues.append(_issue("missing_moment_references", "response_integrity", "Report timeline references moments that are not present upstream.", severity="error", references=missing_moments))
    if missing_drills:
        missing_refs["drills"] = missing_drills
        issues.append(_issue("missing_drill_references", "response_integrity", "Report or coaching references drills outside the deterministic library.", severity="error", references=missing_drills))

    if not _valid_iso_timestamp(getattr(response, "created_at", None)):
        issues.append(_issue("invalid_created_at", "response_integrity", "Response created_at is not a valid ISO timestamp.", severity="error", references=[getattr(response, "created_at", None)]))
    for item in getattr(response, "moments", []) or []:
        if item.start_ms < 0 or item.end_ms < item.start_ms:
            issues.append(_issue("invalid_moment_timestamp", "moments", "Moment timestamps are invalid.", severity="error", references=[item.moment_id]))
    if report:
        for item in report.timeline:
            if item.start_ms < 0 or item.end_ms < item.start_ms:
                issues.append(_issue("invalid_timeline_timestamp", "report_generation", "Report timeline timestamps are invalid.", severity="error", references=[item.moment_id]))

    return issues, duplicate_ids, missing_refs


def _collect_dependency_issues(response: AuthorityV2Response) -> list[PipelineValidationIssue]:
    issues: list[PipelineValidationIssue] = []
    report = getattr(response, "report", None)
    coaching = getattr(response, "coaching_engine", None)
    progress = getattr(response, "progress", None)
    explainability = getattr(response, "explainability", None)

    if report:
        if not report.diagnostic_reasoning:
            issues.append(_issue("report_missing_diagnostic_reasoning", "report_generation", "Report must consume DiagnosticReasoning.", severity="error"))
        if coaching and not report.coaching_engine:
            issues.append(_issue("report_missing_coaching_engine", "report_generation", "Report should carry the coaching engine it consumed.", severity="warning"))
        if report.primary_diagnosis and report.diagnostic_reasoning and report.diagnostic_reasoning.primary_diagnosis:
            if report.primary_diagnosis.diagnosis_id != report.diagnostic_reasoning.primary_diagnosis.diagnosis_id:
                issues.append(_issue("report_diagnosis_mismatch", "report_generation", "Report primary diagnosis differs from DiagnosticReasoning.", severity="error", references=[report.primary_diagnosis.diagnosis_id, report.diagnostic_reasoning.primary_diagnosis.diagnosis_id]))
        if report.hidden_cost_reasoning and report.hidden_cost:
            if report.hidden_cost.cost_id and report.hidden_cost_reasoning.cost_id and report.hidden_cost.cost_id != report.hidden_cost_reasoning.cost_id:
                issues.append(_issue("hidden_cost_reasoning_mismatch", "report_generation", "Hidden cost does not match diagnostic hidden-cost reasoning.", severity="error", references=[report.hidden_cost.cost_id, report.hidden_cost_reasoning.cost_id]))

    if coaching:
        coaching_drill_ids = {
            candidate.drill_id
            for candidate in coaching.intervention_candidates + coaching.suppressed_interventions + coaching.future_training_queue
        }
        if coaching.selected_interventions.primary_drill:
            coaching_drill_ids.add(coaching.selected_interventions.primary_drill.drill_id)
        if coaching.selected_interventions.secondary_drill:
            coaching_drill_ids.add(coaching.selected_interventions.secondary_drill.drill_id)
        if coaching.selected_interventions.primary_drill and not coaching.reasoning_chain.evidence_ids:
            issues.append(_issue("coaching_missing_reasoning_evidence", "coaching", "Selected coaching intervention lacks a reasoning evidence chain.", severity="warning", references=[coaching.selected_interventions.primary_drill.drill_id]))
        if report and report.training_prescription and report.training_prescription.drill_id:
            if coaching_drill_ids and report.training_prescription.drill_id not in coaching_drill_ids:
                issues.append(_issue("training_drill_not_from_coaching", "report_generation", "Training prescription does not come from the deterministic coaching engine.", severity="error", references=[report.training_prescription.drill_id]))
        if report and report.training_prescription and coaching.selected_interventions.primary_drill:
            if report.training_prescription.drill_id != coaching.selected_interventions.primary_drill.drill_id:
                issues.append(_issue("training_drill_not_from_coaching", "report_generation", "Training prescription does not match the deterministic coaching primary drill.", severity="error", references=[report.training_prescription.drill_id, coaching.selected_interventions.primary_drill.drill_id]))
        if report and report.highest_leverage_fix and coaching.selected_interventions.primary_drill:
            if report.highest_leverage_fix.first_drill_id != coaching.selected_interventions.primary_drill.drill_id:
                issues.append(_issue("leverage_fix_not_from_coaching", "report_generation", "Highest leverage fix does not reference the selected coaching drill.", severity="warning", references=[report.highest_leverage_fix.first_drill_id, coaching.selected_interventions.primary_drill.drill_id]))

    if progress and progress.comparison_available and not progress.comparison:
        issues.append(_issue("progress_missing_comparison", "progress", "Progress says comparison is available but no comparison object exists.", severity="error"))
    if explainability:
        if explainability.validation_summary.orphan_evidence_ids:
            issues.append(_issue("explainability_orphan_evidence", "explainability", "Explainability found orphan evidence references.", severity="error", references=explainability.validation_summary.orphan_evidence_ids))
        if explainability.validation_summary.orphan_moment_ids:
            issues.append(_issue("explainability_orphan_moments", "explainability", "Explainability found orphan moment references.", severity="error", references=explainability.validation_summary.orphan_moment_ids))
        if explainability.validation_summary.orphan_drill_ids:
            issues.append(_issue("explainability_orphan_drills", "explainability", "Explainability found drill references outside the deterministic library.", severity="error", references=explainability.validation_summary.orphan_drill_ids))
    return issues


def _collect_consistency_issues(response: AuthorityV2Response) -> list[PipelineValidationIssue]:
    issues: list[PipelineValidationIssue] = []
    report = getattr(response, "report", None)
    scores = getattr(response, "scores", None)
    progress = getattr(response, "progress", None)
    if not report or not scores:
        return issues

    dims = scores.dimension_scores.model_dump()
    if report.authority_type:
        top_dimensions = [dim.lower() for dim in report.authority_type.top_dimensions]
        growth_dimensions = [dim.lower() for dim in report.authority_type.growth_dimensions]
        invalid_dims = sorted(set(top_dimensions + growth_dimensions) - DIMENSIONS)
        if invalid_dims:
            issues.append(_issue("authority_type_invalid_dimensions", "report_generation", "Authority Type references dimensions outside the canonical six.", severity="error", references=invalid_dims))
        valid_top_scores = [dims.get(dim, 0) for dim in top_dimensions if dim in dims]
        if valid_top_scores:
            lowest_top = min(valid_top_scores)
            if lowest_top < 45 and scores.authority_score >= 70:
                issues.append(_issue("authority_type_dimension_inconsistency", "report_generation", "Authority Type top dimensions are inconsistent with the dimension profile.", severity="warning", references=report.authority_type.top_dimensions))

    if report.share_card and report.authority_type and report.share_card.authority_type != report.authority_type.label:
        issues.append(_issue("share_card_authority_type_mismatch", "report_generation", "Share card authority type differs from report authority type.", severity="error", references=[report.share_card.authority_type, report.authority_type.label]))

    if report.highest_leverage_fix and report.primary_diagnosis:
        limiter_dims = set(report.primary_diagnosis.affected_dimensions)
        fix_dims = set(report.highest_leverage_fix.target_dimensions)
        if limiter_dims and fix_dims and not limiter_dims.intersection(fix_dims):
            issues.append(_issue("leverage_fix_diagnosis_mismatch", "report_generation", "Highest leverage fix target dimensions do not overlap the primary diagnosis.", severity="warning", references=list(fix_dims | limiter_dims)))

    if report.mirror and report.diagnosis:
        diagnosis_evidence = set(report.diagnosis.evidence_ids + report.diagnosis.supporting_evidence_ids)
        if report.mirror.evidence_ids and diagnosis_evidence and not set(report.mirror.evidence_ids).intersection(diagnosis_evidence):
            issues.append(_issue("mirror_diagnosis_mismatch", "report_generation", "Mirror evidence does not overlap diagnosis evidence.", severity="warning", references=report.mirror.evidence_ids))

    if progress and progress.comparison_available and progress.comparison:
        if progress.delta_authority_score != progress.comparison.authority_score_delta:
            issues.append(_issue("progress_delta_mismatch", "progress", "Legacy progress delta differs from structured progress comparison.", severity="error", references=[str(progress.delta_authority_score), str(progress.comparison.authority_score_delta)]))

    if scores.scenario_used != response.request.scenario:
        issues.append(_issue("score_scenario_mismatch", "scoring", "Scores scenario does not match request scenario.", severity="warning", references=[scores.scenario_used, response.request.scenario]))

    return issues


def _stage_statuses(response: AuthorityV2Response, issues: list[PipelineValidationIssue]) -> list[PipelineStageStatus]:
    issue_by_stage: dict[str, list[str]] = {}
    for issue in issues:
        issue_by_stage.setdefault(issue.stage, []).append(issue.message)
    report = getattr(response, "report", None)
    metrics = getattr(response, "metrics", None)
    stages = [
        ("audio_quality", bool(getattr(response, "audio_quality", None)), []),
        ("transcription", bool(getattr(response, "transcript", None)), ["audio_quality"]),
        ("vad", bool(metrics and metrics.vad), ["audio_quality"]),
        ("metrics", bool(metrics), ["vad", "transcription"]),
        ("psychological_inference", bool(getattr(response, "psychological_inference", None)), ["metrics", "scoring"]),
        ("diagnostic_reasoning", bool(report and report.diagnostic_reasoning), ["psychological_inference", "metrics", "evidence"]),
        ("scoring", bool(getattr(response, "scores", None)), ["metrics", "scenario"]),
        ("scenario", bool(getattr(response, "request", None) and response.request.scenario), ["request"]),
        ("coaching", bool(getattr(response, "coaching_engine", None)), ["diagnostic_reasoning", "scoring"]),
        ("report_generation", bool(report), ["diagnostic_reasoning", "coaching", "scoring"]),
        ("progress", bool(getattr(response, "progress", None)), ["scoring", "report_generation"]),
        ("explainability", bool(getattr(response, "explainability", None)), ["report_generation", "progress", "upstream_evidence"]),
        ("final_response", response.schema_version == "authority.v2", ["explainability"]),
    ]
    return [_stage(stage_id, completed, deps, issue_by_stage.get(stage_id)) for stage_id, completed, deps in stages]


def _score(total: int, failed: int) -> float:
    return round(max(0.0, 1.0 - failed / max(total, 1)), 2)


def build_pipeline_validation(response: AuthorityV2Response) -> PipelineValidation:
    """Validate the assembled Authority v2 response without recomputing analysis."""
    dependency_issues = _collect_dependency_issues(response)
    consistency_issues = _collect_consistency_issues(response)
    integrity_issues, duplicate_ids, missing_references = _collect_integrity(response)
    all_issues = dependency_issues + consistency_issues + integrity_issues
    stages = _stage_statuses(response, all_issues)
    failed_stages = [stage.stage_id for stage in stages if stage.status == "failed"]
    completed_stages = [stage.stage_id for stage in stages if stage.completed]
    null_outputs = [stage.stage_id for stage in stages if not stage.completed]
    error_count = sum(1 for issue in all_issues if issue.severity == "error")
    warning_text = _dedupe(issue.message for issue in all_issues if issue.severity != "error")

    stage_score = _score(len(stages), len(failed_stages))
    dependency_score = _score(max(len(dependency_issues), 1), sum(1 for item in dependency_issues if item.severity == "error"))
    consistency_score = _score(max(len(consistency_issues), 1), sum(1 for item in consistency_issues if item.severity == "error"))
    integrity_score = _score(max(len(integrity_issues), 1), sum(1 for item in integrity_issues if item.severity == "error"))
    validation_score = round((stage_score + dependency_score + consistency_score + integrity_score) / 4, 2)
    health = "healthy" if validation_score >= 0.95 and error_count == 0 else "degraded" if validation_score >= 0.7 else "invalid"
    audit = PipelineAudit(
        pipeline_version=PIPELINE_VERSION,
        schema_version=response.schema_version,
        completed_stages=completed_stages,
        failed_stages=failed_stages,
        warnings=warning_text,
        validation_score=validation_score,
        dependency_score=dependency_score,
        consistency_score=consistency_score,
        integrity_score=integrity_score,
        overall_pipeline_health=health,  # type: ignore[arg-type]
    )
    return PipelineValidation(
        pipeline_version=PIPELINE_VERSION,
        schema_version=response.schema_version,
        valid=error_count == 0 and not failed_stages,
        stages=stages,
        dependency_issues=dependency_issues,
        consistency_issues=consistency_issues,
        integrity_issues=integrity_issues,
        warnings=warning_text,
        duplicate_ids=duplicate_ids,
        missing_references=missing_references,
        null_outputs=null_outputs,
        audit=audit,
    )
