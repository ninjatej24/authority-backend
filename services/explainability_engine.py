"""Deterministic explainability, safety validation, and audit trail."""

from __future__ import annotations

from typing import Iterable

from schemas import (
    AlternativeInterpretation,
    AudioQuality,
    ClaimValidation,
    CoachingEngine,
    DiagnosticContradiction,
    DiagnosticReasoning,
    EvidenceItem,
    ExplainabilityBundle,
    ExplainabilityClaim,
    Metrics,
    Moment,
    Progress,
    ReportAudit,
    Scores,
    Uncertainty,
    ValidationSummary,
    AuthorityReport,
    PsychologicalInference,
)


ENGINE_VERSION = "explainability_v1"
PIPELINE_VERSION = "authority.v2.milestone11"
MIN_CLAIM_CONFIDENCE = 0.35


def _label(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium_high"
    if confidence >= 0.4:
        return "medium"
    return "low"


def _dedupe(values: Iterable[str | None]) -> list[str]:
    return [item for item in dict.fromkeys(value for value in values if value).keys()]


def _metric_paths(metrics: list[str] | None = None) -> list[str]:
    return list(metrics or [])


def _metric_registry(metrics: Metrics, scores: Scores, progress: Progress) -> set[str]:
    registry = {"score_components.weighted_base", "score_components.penalties", "score_components.bonuses", "scores.authority_score", "scores.scenario_adjustments"}
    metric_dump = metrics.model_dump()
    for section, values in metric_dump.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if value is not None:
                    registry.add(f"{section}.{key}")
                    registry.add(key)
    for key, detail in scores.dimension_details.items():
        registry.add(key)
        for metric in detail.positive_contributors + detail.negative_contributors:
            registry.add(metric)
    if progress.comparison:
        registry.update({"progress.comparison", "progress.trend_summary", "progress.dimension_delta_details"})
    for metric_id in progress.metric_deltas:
        registry.add(metric_id)
    return registry


def _alternatives(claim_type: str, confidence: float) -> list[AlternativeInterpretation]:
    if confidence >= 0.6:
        return []
    mapping = {
        "primary_diagnosis": ("alt_pressure_as_energy", "This could also reflect excitement rather than pressure.", "Medium-confidence delivery signals can have more than one listener interpretation."),
        "hidden_cost": ("alt_context_cost", "This cost may be less visible in low-stakes contexts.", "Scenario and listener context can reduce the social consequence."),
        "authority_type": ("alt_type_adjacent", "An adjacent authority type may also fit if the next recording shifts the dimension profile.", "Authority type depends on the relative dimension pattern."),
        "progress_interpretation": ("alt_temporary_variation", "This may reflect normal recording-to-recording variation.", "Progress comparisons are less certain with short histories."),
        "coaching_selection": ("alt_parallel_drill", "A secondary drill may be similarly useful if it targets the same evidence cluster.", "Multiple deterministic interventions can share evidence."),
    }
    if claim_type not in mapping:
        return []
    alt_id, text, reason = mapping[claim_type]
    return [AlternativeInterpretation(alternative_id=alt_id, text=text, reason=reason, confidence=round(1 - confidence, 2))]


def _validation(
    *,
    claim_id: str,
    evidence_ids: list[str],
    moment_ids: list[str],
    valid_evidence_ids: set[str],
    valid_moment_ids: set[str],
    confidence: float,
    audio_quality: AudioQuality,
    uncertainty_reasons: list[str],
    metric_ids: list[str],
    valid_metric_ids: set[str],
    required_dependencies: list[str],
) -> ClaimValidation:
    failed = []
    missing_dependencies = []
    if evidence_ids and not set(evidence_ids).issubset(valid_evidence_ids):
        failed.append("orphan_evidence")
    if moment_ids and not set(moment_ids).issubset(valid_moment_ids):
        failed.append("orphan_moment")
    if metric_ids and not set(metric_ids).intersection(valid_metric_ids):
        failed.append("unsupported_metric")
        missing_dependencies.extend([metric for metric in metric_ids if metric not in valid_metric_ids])
    if required_dependencies:
        if "evidence" in required_dependencies and not evidence_ids:
            failed.append("missing_upstream_dependency")
            missing_dependencies.append("evidence")
        if "metrics" in required_dependencies and not metric_ids:
            failed.append("missing_upstream_dependency")
            missing_dependencies.append("metrics")
        if "moments" in required_dependencies and not moment_ids:
            failed.append("missing_upstream_dependency")
            missing_dependencies.append("moments")
        if "traits" in required_dependencies:
            missing_dependencies.append("traits")
    if not evidence_ids and not moment_ids and not metric_ids and claim_id not in {"scenario_adjustments"}:
        failed.append("insufficient_evidence")
    if confidence < MIN_CLAIM_CONFIDENCE:
        failed.append("low_confidence")
    if not audio_quality.usable:
        failed.append("poor_audio")
    if any("short" in reason.lower() for reason in uncertainty_reasons):
        failed.append("short_recording")
    suppressed = bool(failed)
    return ClaimValidation(
        valid=not failed,
        suppressed=suppressed,
        suppression_reason=failed[0] if failed else None,
        failed_checks=failed,
        missing_dependencies=_dedupe(missing_dependencies),
        confidence_before_suppression=round(confidence, 2),
        minimum_required_confidence=MIN_CLAIM_CONFIDENCE,
    )


def _claim(
    *,
    claim_id: str,
    claim: str | None,
    claim_type: str,
    evidence_ids: list[str],
    metrics: list[str],
    traits: list[str],
    moments: list[str],
    dimensions: list[str],
    positive: list[str],
    negative: list[str],
    confidence: float,
    confidence_reasons: list[str],
    uncertainty_reasons: list[str],
    valid_evidence_ids: set[str],
    valid_moment_ids: set[str],
    valid_metric_ids: set[str],
    audio_quality: AudioQuality,
    required_dependencies: list[str] | None = None,
) -> ExplainabilityClaim:
    adjusted_confidence = confidence
    uncertainty_text = " ".join(uncertainty_reasons + confidence_reasons).lower()
    if "short" in uncertainty_text:
        adjusted_confidence = min(adjusted_confidence, 0.55)
    if "low confidence" in uncertainty_text or "asr" in uncertainty_text:
        adjusted_confidence = min(adjusted_confidence, 0.5)
    if not audio_quality.usable:
        adjusted_confidence = min(adjusted_confidence, 0.4)
    validation = _validation(
        claim_id=claim_id,
        evidence_ids=evidence_ids,
        moment_ids=moments,
        valid_evidence_ids=valid_evidence_ids,
        valid_moment_ids=valid_moment_ids,
        confidence=adjusted_confidence,
        audio_quality=audio_quality,
        uncertainty_reasons=uncertainty_reasons,
        metric_ids=metrics,
        valid_metric_ids=valid_metric_ids,
        required_dependencies=required_dependencies or [],
    )
    return ExplainabilityClaim(
        claim_id=claim_id,
        claim=claim,
        claim_type=claim_type,
        supporting_evidence_ids=_dedupe(evidence_ids),
        supporting_metrics=_dedupe(metrics),
        supporting_traits=_dedupe(traits),
        supporting_moments=_dedupe(moments),
        supporting_dimensions=_dedupe(dimensions),
        positive_evidence=_dedupe(positive),
        negative_evidence=_dedupe(negative),
        confidence=round(adjusted_confidence, 2),
        confidence_label=_label(adjusted_confidence),  # type: ignore[arg-type]
        confidence_reasons=_dedupe(confidence_reasons),
        uncertainty_reasons=_dedupe(uncertainty_reasons),
        suppressed=validation.suppressed,
        suppression_reason=validation.suppression_reason,
        alternative_interpretations=_alternatives(claim_type, adjusted_confidence),
        validation=validation,
    )


def _detect_contradictions(scores: Scores, progress: Progress, diagnostic: DiagnosticReasoning) -> list[DiagnosticContradiction]:
    contradictions = list(diagnostic.contradictions)
    dims = scores.dimension_scores.model_dump()

    def add(cid: str, strength: str, limiter: str, effect: str, confidence: float) -> None:
        contradictions.append(
            DiagnosticContradiction(
                contradiction_id=cid,
                strength=strength,
                limiter=limiter,
                why_it_happens=[f"{strength} is high while {limiter} is low"],
                listener_effect=effect,
                evidence_ids=[],
                confidence=confidence,
            )
        )

    if dims.get("command", 0) >= 72 and dims.get("composure", 100) <= 55:
        add("high_command_low_composure", "command", "composure", "The recording may sound decisive but pressured.", 0.72)
    if dims.get("persuasion", 0) >= 72 and dims.get("clarity", 100) <= 58:
        add("high_persuasion_low_clarity", "persuasion", "clarity", "The recording may sound compelling but harder to follow.", 0.7)
    if dims.get("presence", 0) >= 72 and dims.get("structure", 100) <= 58:
        add("high_presence_weak_structure", "presence", "structure", "The recording may hold attention without a fully controlled path.", 0.7)
    if scores.authority_score >= 80 and (scores.score_confidence or 0.0) < 0.55:
        add("high_score_low_confidence", "authority_score", "score_confidence", "The score is strong but should be treated cautiously.", 0.68)
    if progress.moment_comparison:
        weak = [item for item in progress.moment_comparison if item.status in {"new_weakness", "persistent_weakness"}]
        strong = [item for item in progress.moment_comparison if item.status == "new_strength"]
        if weak and strong:
            add("strong_evidence_contradictory_moments", "strong_moments", "weak_moments", "The progress signal is mixed across moments.", 0.62)
    return contradictions


def _drill_validation(report: AuthorityReport, coaching: CoachingEngine) -> tuple[list[str], list[str], list[str], float]:
    library_ids = {item.drill_id for item in coaching.drill_library}
    referenced: set[str] = set()
    dependency_refs: set[str] = set()
    if report.training_prescription and report.training_prescription.drill_id:
        referenced.add(report.training_prescription.drill_id)
    if report.highest_leverage_fix and report.highest_leverage_fix.first_drill_id:
        referenced.add(report.highest_leverage_fix.first_drill_id)
    for candidate in coaching.intervention_candidates + coaching.suppressed_interventions + coaching.future_training_queue:
        referenced.add(candidate.drill_id)
    selected = coaching.selected_interventions
    if selected.primary_drill:
        referenced.add(selected.primary_drill.drill_id)
    if selected.secondary_drill:
        referenced.add(selected.secondary_drill.drill_id)
    for dependency in coaching.dependency_graph:
        dependency_refs.update([dependency.before, dependency.after])

    orphan_drills = sorted(item for item in referenced if item and item not in library_ids)
    invalid_dependencies = sorted(item for item in dependency_refs if item and item not in library_ids)
    missing = sorted(orphan_drills + [f"dependency:{item}" for item in invalid_dependencies])
    total = max(len(referenced) + len(dependency_refs), 1)
    score = round(max(0.0, 1.0 - (len(orphan_drills) + len(invalid_dependencies)) / total), 2)
    return orphan_drills, missing, invalid_dependencies, score


def _grade(score: float) -> str:
    if score >= 0.95:
        return "A"
    if score >= 0.85:
        return "B"
    if score >= 0.7:
        return "C"
    if score >= 0.55:
        return "D"
    return "F"


def build_explainability(
    *,
    metrics: Metrics,
    evidence: list[EvidenceItem],
    psychological_inference: PsychologicalInference,
    diagnostic_reasoning: DiagnosticReasoning,
    scores: Scores,
    scenario: str,
    coaching_engine: CoachingEngine,
    report: AuthorityReport,
    progress: Progress,
    moments: list[Moment],
    audio_quality: AudioQuality,
    uncertainty: Uncertainty,
) -> ExplainabilityBundle:
    """Build deterministic claim explanations and audit metadata from existing outputs."""
    valid_evidence_ids = {item.id for item in evidence} | {item.evidence_id for item in psychological_inference.evidence_chain}
    valid_evidence_ids.update(
        evidence_id
        for diagnosis in [diagnostic_reasoning.primary_diagnosis, diagnostic_reasoning.secondary_diagnosis]
        if diagnosis
        for evidence_id in diagnosis.supporting_evidence_ids
    )
    if diagnostic_reasoning.hidden_cost_reasoning:
        valid_evidence_ids.update(diagnostic_reasoning.hidden_cost_reasoning.evidence_ids)
    if diagnostic_reasoning.highest_leverage_reasoning:
        valid_evidence_ids.update(diagnostic_reasoning.highest_leverage_reasoning.supporting_evidence)
    for candidate in coaching_engine.intervention_candidates:
        valid_evidence_ids.update(candidate.supporting_evidence_ids)
    valid_moment_ids = {item.moment_id for item in moments}
    valid_metric_ids = _metric_registry(metrics, scores, progress)
    base_reasons = list(dict.fromkeys(uncertainty.reasons + scores.score_explanation.confidence_reasons))
    claims: list[ExplainabilityClaim] = []
    primary = diagnostic_reasoning.primary_diagnosis
    secondary = diagnostic_reasoning.secondary_diagnosis
    highest = diagnostic_reasoning.highest_leverage_reasoning
    selected = coaching_engine.selected_interventions.primary_drill

    def add(**kwargs) -> None:
        claims.append(_claim(valid_evidence_ids=valid_evidence_ids, valid_moment_ids=valid_moment_ids, valid_metric_ids=valid_metric_ids, audio_quality=audio_quality, **kwargs))

    mirror = report.mirror
    add(claim_id="mirror_headline", claim=mirror.headline if mirror else None, claim_type="mirror_headline", evidence_ids=(mirror.evidence_ids if mirror else []), metrics=["scores.authority_score"], traits=[], moments=[], dimensions=(report.diagnosis.evidence_ids if report.diagnosis else []), positive=(mirror.evidence_ids if mirror else []), negative=[], confidence=scores.score_confidence or 0.0, confidence_reasons=base_reasons, uncertainty_reasons=uncertainty.reasons, required_dependencies=["evidence", "metrics"])
    authority_type = report.authority_type
    add(claim_id="authority_type", claim=authority_type.label if authority_type else None, claim_type="authority_type", evidence_ids=(authority_type.evidence_ids if authority_type else []), metrics=[], traits=(psychological_inference.report_candidates.report_priority_order if psychological_inference.report_candidates else []), moments=[], dimensions=(authority_type.top_dimensions if authority_type else []), positive=(authority_type.evidence_ids if authority_type else []), negative=[], confidence=(authority_type.confidence if authority_type else 0.0), confidence_reasons=base_reasons, uncertainty_reasons=uncertainty.reasons, required_dependencies=["evidence"])
    if primary:
        add(claim_id="primary_diagnosis", claim=primary.diagnosis_name, claim_type="primary_diagnosis", evidence_ids=primary.supporting_evidence_ids, metrics=[], traits=primary.supporting_traits, moments=primary.supporting_moment_ids, dimensions=primary.affected_dimensions, positive=primary.supporting_evidence_ids, negative=primary.contradicting_traits, confidence=primary.confidence, confidence_reasons=base_reasons, uncertainty_reasons=diagnostic_reasoning.uncertainty.reasons, required_dependencies=["evidence"])
    if secondary:
        add(claim_id="secondary_diagnosis", claim=secondary.diagnosis_name, claim_type="secondary_diagnosis", evidence_ids=secondary.supporting_evidence_ids, metrics=[], traits=secondary.supporting_traits, moments=secondary.supporting_moment_ids, dimensions=secondary.affected_dimensions, positive=secondary.supporting_evidence_ids, negative=secondary.contradicting_traits, confidence=secondary.confidence, confidence_reasons=base_reasons, uncertainty_reasons=diagnostic_reasoning.uncertainty.reasons, required_dependencies=["evidence"])
    hidden = diagnostic_reasoning.hidden_cost_reasoning
    add(claim_id="hidden_cost", claim=report.hidden_cost.consequence if report.hidden_cost else None, claim_type="hidden_cost", evidence_ids=(hidden.evidence_ids if hidden else (report.hidden_cost.evidence_ids if report.hidden_cost else [])), metrics=[], traits=[], moments=(hidden.moment_ids if hidden else []), dimensions=(hidden.affected_dimensions if hidden else []), positive=(hidden.evidence_ids if hidden else []), negative=[], confidence=(hidden.confidence if hidden else 0.0), confidence_reasons=base_reasons, uncertainty_reasons=uncertainty.reasons, required_dependencies=["evidence"])
    add(claim_id="highest_leverage_fix", claim=report.highest_leverage_fix.issue if report.highest_leverage_fix else None, claim_type="highest_leverage_fix", evidence_ids=(highest.supporting_evidence if highest else []), metrics=[], traits=[], moments=[], dimensions=(highest.affected_dimensions if highest else []), positive=(highest.supporting_evidence if highest else []), negative=[], confidence=(highest.confidence if highest else 0.0), confidence_reasons=base_reasons, uncertainty_reasons=uncertainty.reasons, required_dependencies=["evidence"])
    add(claim_id="training_prescription", claim=report.training_prescription.title if report.training_prescription else None, claim_type="training_prescription", evidence_ids=(report.training_prescription.evidence_ids if report.training_prescription else []), metrics=(report.training_prescription.target_metrics if report.training_prescription else []), traits=[], moments=[], dimensions=(selected.required_evidence if selected else []), positive=(selected.supporting_evidence_ids if selected else []), negative=[], confidence=(selected.confidence if selected else 0.0), confidence_reasons=base_reasons, uncertainty_reasons=coaching_engine.uncertainty.reasons, required_dependencies=["evidence", "metrics"])
    add(claim_id="authority_score", claim=str(scores.authority_score), claim_type="authority_score", evidence_ids=[], metrics=["score_components.weighted_base", "score_components.penalties", "score_components.bonuses"], traits=[], moments=[], dimensions=list(scores.dimension_scores.model_dump()), positive=[item.id for item in scores.score_components.bonus_items], negative=[item.id for item in scores.score_components.penalty_items], confidence=scores.score_confidence or 0.0, confidence_reasons=scores.score_explanation.confidence_reasons, uncertainty_reasons=uncertainty.reasons)
    add(claim_id="dimension_scores", claim="six_dimension_scores", claim_type="dimension_scores", evidence_ids=[], metrics=list(scores.dimension_details), traits=[], moments=[], dimensions=list(scores.dimension_scores.model_dump()), positive=[], negative=[], confidence=scores.score_confidence or 0.0, confidence_reasons=scores.score_explanation.confidence_reasons, uncertainty_reasons=uncertainty.reasons)
    add(claim_id="top_strength", claim=report.diagnosis.strongest_dimension if report.diagnosis else None, claim_type="top_strength", evidence_ids=(report.diagnosis.evidence_ids if report.diagnosis else []), metrics=[], traits=[], moments=[], dimensions=[report.diagnosis.strongest_dimension] if report.diagnosis and report.diagnosis.strongest_dimension else [], positive=(report.diagnosis.evidence_ids if report.diagnosis else []), negative=[], confidence=scores.score_confidence or 0.0, confidence_reasons=base_reasons, uncertainty_reasons=uncertainty.reasons)
    add(claim_id="primary_limiter", claim=report.diagnosis.limiting_dimension if report.diagnosis else None, claim_type="primary_limiter", evidence_ids=(report.diagnosis.evidence_ids if report.diagnosis else []), metrics=[], traits=[], moments=[], dimensions=[report.diagnosis.limiting_dimension] if report.diagnosis and report.diagnosis.limiting_dimension else [], positive=[], negative=(report.diagnosis.evidence_ids if report.diagnosis else []), confidence=scores.score_confidence or 0.0, confidence_reasons=base_reasons, uncertainty_reasons=uncertainty.reasons)
    add(claim_id="authority_evolution", claim=progress.authority_evolution.status, claim_type="authority_evolution", evidence_ids=[], metrics=["progress.dimension_delta_details"], traits=[], moments=[], dimensions=progress.authority_evolution.new_dominant_characteristics, positive=[], negative=[], confidence=progress.authority_evolution.confidence, confidence_reasons=progress.confidence.reasons, uncertainty_reasons=progress.confidence.reasons)
    add(claim_id="scenario_adjustments", claim=scenario, claim_type="scenario_adjustments", evidence_ids=[], metrics=["scores.scenario_adjustments"], traits=[], moments=[], dimensions=list(scores.scenario_adjustments.dimension_adjustments), positive=scores.scenario_adjustments.major_weight_changes, negative=[], confidence=scores.score_confidence or 0.0, confidence_reasons=base_reasons, uncertainty_reasons=uncertainty.reasons)
    share_dimensions = _dedupe([report.share_card.top_strength, report.share_card.growth_area] if report.share_card else [])
    add(claim_id="share_card", claim=report.share_card.one_line_identity_read if report.share_card else None, claim_type="share_card", evidence_ids=(report.mirror.evidence_ids if report.mirror else []), metrics=["scores.authority_score"], traits=[], moments=[], dimensions=share_dimensions, positive=(report.mirror.evidence_ids if report.mirror else []), negative=[], confidence=scores.score_confidence or 0.0, confidence_reasons=base_reasons, uncertainty_reasons=uncertainty.reasons)
    add(claim_id="progress_interpretation", claim=progress.state.progress_status, claim_type="progress_interpretation", evidence_ids=[], metrics=["progress.comparison", "progress.trend_summary"], traits=[], moments=[], dimensions=list(progress.dimension_delta_details), positive=[], negative=[item.milestone_id for item in progress.regressions], confidence=progress.confidence.confidence, confidence_reasons=progress.confidence.reasons, uncertainty_reasons=progress.confidence.reasons)
    add(claim_id="coaching_selection", claim=selected.title if selected else None, claim_type="coaching_selection", evidence_ids=(selected.supporting_evidence_ids if selected else []), metrics=(selected.required_evidence if selected else []), traits=[], moments=[], dimensions=(selected.required_evidence if selected else []), positive=(selected.supporting_evidence_ids if selected else []), negative=[], confidence=(selected.confidence if selected else 0.0), confidence_reasons=coaching_engine.uncertainty.reasons, uncertainty_reasons=coaching_engine.uncertainty.reasons, required_dependencies=["evidence", "metrics"])
    add(claim_id="retest_recommendation", claim=progress.retest_recommendation.comparison_focus, claim_type="retest_recommendation", evidence_ids=[], metrics=progress.retest_recommendation.what_to_compare, traits=[], moments=[], dimensions=[], positive=[], negative=[], confidence=progress.confidence.confidence, confidence_reasons=progress.confidence.reasons, uncertainty_reasons=progress.confidence.reasons)

    contradictions = _detect_contradictions(scores, progress, diagnostic_reasoning)
    suppressed = [claim for claim in claims if claim.suppressed]
    failed = [claim for claim in claims if not claim.validation.valid]
    report_evidence_ids = {item.evidence_id for item in report.evidence_chain}
    report_moment_ids = {item.moment_id for item in report.timeline}
    orphan_evidence = sorted(
        {e for claim in claims for e in claim.supporting_evidence_ids if e not in valid_evidence_ids}
        | {item for item in report_evidence_ids if item not in valid_evidence_ids}
    )
    orphan_moments = sorted(
        {m for claim in claims for m in claim.supporting_moments if m not in valid_moment_ids}
        | {item for item in report_moment_ids if item not in valid_moment_ids}
    )
    orphan_drills, missing_drill_references, invalid_dependency_references, drill_score = _drill_validation(report, coaching_engine)
    valid_count = len(claims) - len(failed)
    total = max(len(claims), 1)
    evidence_ref_total = sum(len(claim.supporting_evidence_ids) for claim in claims)
    moment_ref_total = sum(len(claim.supporting_moments) for claim in claims)
    reference_total = max(evidence_ref_total + moment_ref_total + len(missing_drill_references), 1)
    invalid_reference_total = len(orphan_evidence) + len(orphan_moments) + len(missing_drill_references)
    evidence_score = round(1.0 - len(orphan_evidence) / max(evidence_ref_total, 1), 2)
    moment_score = round(1.0 - len(orphan_moments) / max(moment_ref_total, 1), 2)
    reference_percentage = round(1.0 - invalid_reference_total / reference_total, 2)
    claim_percentage = round(valid_count / total, 2)
    integrity = round(max(0.0, (claim_percentage + reference_percentage + drill_score + evidence_score + moment_score) / 5), 2)
    summary = ValidationSummary(
        valid=not failed and not orphan_evidence and not orphan_moments,
        checked_claims=len(claims),
        suppressed_claims=len(suppressed),
        failed_claims=len(failed),
        orphan_evidence_ids=orphan_evidence,
        orphan_moment_ids=orphan_moments,
        orphan_drill_ids=orphan_drills,
        missing_drill_references=missing_drill_references,
        invalid_dependency_references=invalid_dependency_references,
        missing_dependency_claims=[claim.claim_id for claim in failed if "missing_upstream_dependency" in claim.validation.failed_checks],
    )
    audit = ReportAudit(
        engine_version=ENGINE_VERSION,
        pipeline_version=PIPELINE_VERSION,
        analysis_steps_completed=[
            "measurement",
            "inference",
            "diagnostic_reasoning",
            "scoring",
            "coaching",
            "report_generation",
            "progress",
            "explainability",
        ],
        suppressed_claim_count=len(suppressed),
        validated_claim_count=valid_count,
        failed_validation_count=len(failed),
        orphan_evidence_count=len(orphan_evidence),
        orphan_moment_count=len(orphan_moments),
        consistency_score=round(progress.consistency_score or scores.score_confidence or 0.0, 2),
        report_integrity_score=integrity,
        validated_claim_percentage=claim_percentage,
        validated_reference_percentage=reference_percentage,
        drill_validation_score=drill_score,
        evidence_validation_score=evidence_score,
        moment_validation_score=moment_score,
        overall_integrity_grade=_grade(integrity),  # type: ignore[arg-type]
    )
    return ExplainabilityBundle(claims=claims, contradictions=contradictions, validation_summary=summary, audit=audit)
