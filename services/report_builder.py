"""Deterministic Authority report assembly from measured backend facts."""

from __future__ import annotations

from schemas import (
    AudioQuality,
    AuthorityReport,
    CoachingEngine,
    DiagnosticReasoning,
    EvidenceItem,
    Metrics,
    Moment,
    MomentIntelligence,
    PsychologicalInference,
    Scores,
    Uncertainty,
)
from services.report_generation import build_generated_report


def apply_coaching_to_report(report: AuthorityReport, coaching: CoachingEngine) -> AuthorityReport:
    """Attach deterministic coaching output and use it for report drill selection."""
    primary = coaching.selected_interventions.primary_drill
    if not primary:
        return report.model_copy(update={"coaching_engine": coaching})

    drill = next(
        (item for item in coaching.drill_library if item.drill_id == primary.drill_id),
        None,
    )
    fix = report.highest_leverage_fix
    if fix:
        fix = fix.model_copy(
            update={
                "first_drill_id": primary.drill_id,
                "selection_score": primary.score,
                "evidence_ids": primary.supporting_evidence_ids,
                "target_dimensions": drill.target_dimensions if drill else fix.target_dimensions,
            }
        )

    training = report.training_prescription
    if training and drill:
        training = training.model_copy(
            update={
                "drill_id": drill.drill_id,
                "title": drill.title,
                "why_chosen": primary.why_selected,
                "target_metrics": drill.target_metrics,
                "success_signal": "expected_improvement_model_attached",
                "evidence_ids": primary.supporting_evidence_ids,
            }
        )

    return report.model_copy(
        update={
            "highest_leverage_fix": fix,
            "training_prescription": training,
            "coaching_engine": coaching,
        }
    )


def build_report(
    *,
    scores: Scores,
    metrics: Metrics,
    psychological_inference: PsychologicalInference,
    diagnostic_reasoning: DiagnosticReasoning,
    evidence: list[EvidenceItem],
    moments: list[Moment],
    uncertainty: Uncertainty,
    audio_quality: AudioQuality,
    duration_ms: int,
    scenario: str,
    coaching_engine: CoachingEngine | None = None,
    moment_intelligence: MomentIntelligence | None = None,
) -> AuthorityReport:
    """Build the deterministic report via the Milestone 7 report generator."""
    return build_generated_report(
        scores=scores,
        metrics=metrics,
        psychological_inference=psychological_inference,
        diagnostic_reasoning=diagnostic_reasoning,
        coaching_engine=coaching_engine,
        evidence=evidence,
        moments=moments,
        uncertainty=uncertainty,
        audio_quality=audio_quality,
        duration_ms=duration_ms,
        scenario=scenario,
        moment_intelligence=moment_intelligence,
    )
