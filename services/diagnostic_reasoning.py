"""Deterministic diagnostic reasoning between inference and report assembly."""

from __future__ import annotations

from dataclasses import dataclass

from schemas import (
    AudioQuality,
    DiagnosticContradiction,
    DiagnosticDiagnosis,
    DiagnosticReasoning,
    DimensionReasoning,
    EvidenceItem,
    HiddenCostReasoning,
    HighestLeverageReasoning,
    Metrics,
    Moment,
    PsychologicalInference,
    PsychologicalTrait,
    Scores,
    TraitReasoning,
    Uncertainty,
)
from services.scenario_profiles import calculate_dimension_relevance


DIMENSIONS = ("command", "clarity", "composure", "presence", "persuasion", "structure")

DIMENSION_LABELS = {
    "command": "Command",
    "clarity": "Clarity",
    "composure": "Composure",
    "presence": "Presence",
    "persuasion": "Persuasion",
    "structure": "Structure",
}

DIMENSION_SIGNALS = {
    "command": {
        "metrics": ("raw_acoustic.terminal_falling_ratio", "raw_acoustic.terminal_rising_ratio"),
        "linguistic": ("linguistic.closing_strength_score", "linguistic.certainty_markers_per_100_words"),
        "behaviours": ("executive_finality", "pause_ownership", "speaking_with_conviction", "approval_seeking_cues"),
    },
    "clarity": {
        "metrics": ("articulation.clarity_proxy", "linguistic.filler_words_per_min"),
        "linguistic": ("linguistic.structure_score", "linguistic.rambling_score", "linguistic.specificity_score"),
        "behaviours": ("articulation_clarity", "structured_thinking", "clear_opening_control"),
    },
    "composure": {
        "metrics": ("rhythm.rhythm_consistency", "derived.hesitation_cluster_score"),
        "linguistic": ("linguistic.self_doubt_markers", "linguistic.filler_words_per_min"),
        "behaviours": ("composure_under_pressure", "deliberate_pacing", "pressure_leakage", "pace_pressure"),
    },
    "presence": {
        "metrics": ("derived.dynamic_emphasis_score", "raw_acoustic.f0_range_semitones"),
        "linguistic": ("linguistic.opening_strength_score",),
        "behaviours": ("vocal_variety", "projection_control", "flat_delivery", "under_projected"),
    },
    "persuasion": {
        "metrics": ("derived.dynamic_emphasis_score", "raw_acoustic.terminal_falling_ratio"),
        "linguistic": ("linguistic.certainty_markers_per_100_words", "linguistic.specificity_score"),
        "behaviours": ("persuasive_momentum", "speaking_with_conviction", "explanation_without_pull"),
    },
    "structure": {
        "metrics": ("linguistic.structure_score", "linguistic.repetition_rate"),
        "linguistic": ("linguistic.opening_strength_score", "linguistic.closing_strength_score", "linguistic.rambling_score"),
        "behaviours": ("structured_thinking", "clear_opening_control", "loose_structure"),
    },
}


@dataclass(frozen=True)
class DiagnosisRule:
    diagnosis_id: str
    diagnosis_name: str
    strength: str
    limiter: str
    supporting_traits: tuple[str, ...]
    contradicting_traits: tuple[str, ...]
    affected_dimensions: tuple[str, ...]
    authority_impact: float
    trainability: float


@dataclass(frozen=True)
class FixRule:
    issue_id: str
    plain_reason: str
    dimensions: tuple[str, ...]
    drill_id: str
    authority_impact: float
    trainability: float


DIAGNOSES = (
    DiagnosisRule(
        "softened_expert",
        "The Softened Expert",
        "clarity",
        "command",
        ("credible", "structured_thinker", "clear_communicator", "approval_seeking", "hesitant"),
        ("commanding", "executive_presence"),
        ("clarity", "structure", "command"),
        0.95,
        0.9,
    ),
    DiagnosisRule(
        "rushed_achiever",
        "The Rushed Achiever",
        "clarity",
        "composure",
        ("rushed", "nervous", "clear_communicator", "energetic"),
        ("calm", "composed"),
        ("clarity", "composure", "command"),
        0.9,
        0.9,
    ),
    DiagnosisRule(
        "flat_specialist",
        "The Flat Specialist",
        "clarity",
        "presence",
        ("credible", "flat", "monotone", "structured_thinker"),
        ("energetic", "persuasive"),
        ("clarity", "presence", "persuasion"),
        0.75,
        0.8,
    ),
    DiagnosisRule(
        "polished_persuader",
        "The Polished Persuader",
        "persuasion",
        "structure",
        ("persuasive", "energetic", "loose_structure"),
        ("structured_thinker", "clear_communicator"),
        ("persuasion", "presence", "structure"),
        0.75,
        0.85,
    ),
    DiagnosisRule(
        "controlled_leader",
        "The Controlled Leader",
        "command",
        "presence",
        ("commanding", "composed", "leadership_ready", "executive_presence"),
        ("rushed", "approval_seeking"),
        ("command", "clarity", "composure"),
        0.65,
        0.55,
    ),
    DiagnosisRule(
        "unclear_path",
        "The Unclear Path",
        "presence",
        "structure",
        ("loose_structure", "hesitant", "rushed"),
        ("structured_thinker", "clear_communicator"),
        ("structure", "clarity", "persuasion"),
        0.8,
        0.9,
    ),
)

FIX_RULES = {
    "command": FixRule(
        "declarative_finality",
        "weak_finality_reduces_perceived_leadership",
        ("command", "composure"),
        "drop_the_landing_v1",
        0.95,
        0.9,
    ),
    "clarity": FixRule(
        "clarity_compression",
        "listener_effort_reduces_credibility",
        ("clarity", "structure"),
        "one_point_one_proof_v1",
        0.85,
        0.85,
    ),
    "composure": FixRule(
        "pace_control",
        "pressure_leakage_reduces_control",
        ("composure", "command"),
        "pace_anchor_v1",
        0.9,
        0.9,
    ),
    "presence": FixRule(
        "dynamic_emphasis",
        "low_contrast_reduces_memorability",
        ("presence", "persuasion"),
        "emphasis_ladder_v1",
        0.75,
        0.8,
    ),
    "persuasion": FixRule(
        "conviction_framing",
        "explanation_without_pull_reduces_movement",
        ("persuasion", "structure"),
        "claim_stakes_action_v1",
        0.75,
        0.75,
    ),
    "structure": FixRule(
        "structure_compression",
        "unclear_answer_path_reduces_trust_in_control",
        ("structure", "clarity"),
        "point_proof_close_v1",
        0.8,
        0.9,
    ),
}

HIDDEN_COSTS = {
    "command": ("weak_declarative_endings", "reduced_perceived_finality", "point_understood", "listener_not_fully_led"),
    "clarity": ("high_listener_effort", "reduced_processing_capacity", "point_less_easy_to_follow", "less_energy_left_for_persuasion"),
    "composure": ("pressure_leakage", "delivery_feels_reactive", "words_may_be_correct", "listener_feels_the_pressure"),
    "presence": ("low_vocal_contrast", "reduced_memorability", "point_may_be_accepted", "point_less_likely_to_stick"),
    "persuasion": ("weak_conviction_path", "explanation_without_movement", "point_is_understood", "listener_less_pulled_to_action"),
    "structure": ("unclear_answer_path", "authority_drift", "content_has_value", "listener_trusts_control_less"),
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium_high"
    if confidence >= 0.4:
        return "medium"
    return "low"


def _severity_from_score(score: int) -> str:
    if score < 45:
        return "high"
    if score < 60:
        return "medium"
    return "low"


def _severity_value(severity: str) -> float:
    return {"high": 1.0, "medium": 0.65, "low": 0.35}[severity]


def _dimension_scores(scores: Scores) -> dict[str, int]:
    return scores.dimension_scores.model_dump()


def _traits(inference: PsychologicalInference) -> dict[str, PsychologicalTrait]:
    return {trait.trait_id: trait for trait in inference.traits}


def _visible_traits(inference: PsychologicalInference) -> dict[str, PsychologicalTrait]:
    return {
        trait.trait_id: trait
        for trait in inference.traits
        if not trait.suppress_from_report and trait.confidence >= 0.4
    }


def _fallback_evidence_ids(evidence: list[EvidenceItem], dimension: str) -> list[str]:
    matches = [item.id for item in evidence if item.trait == dimension]
    if matches:
        return matches[:3]
    return [item.id for item in evidence[:3]]


def _trait_evidence_ids(traits: list[PsychologicalTrait]) -> list[str]:
    ids: list[str] = []
    for trait in traits:
        ids.extend(trait.supporting_evidence_ids)
    return list(dict.fromkeys(ids))


def _moment_ids(moments: list[Moment], dimensions: tuple[str, ...] | list[str]) -> list[str]:
    dim_set = set(dimensions)
    ids = [
        moment.moment_id
        for moment in moments
        if dim_set.intersection(moment.dimension_impact.keys())
    ]
    return ids[:4]


def _rule_score(rule: DiagnosisRule, dims: dict[str, int], traits: dict[str, PsychologicalTrait]) -> tuple[float, list[str], list[str]]:
    strength_score = dims.get(rule.strength, 50)
    limiter_score = dims.get(rule.limiter, 50)
    spread = max(0, strength_score - limiter_score) / 45
    weakness = max(0, 65 - limiter_score) / 45

    supporting = [trait_id for trait_id in rule.supporting_traits if trait_id in traits]
    contradicting = [trait_id for trait_id in rule.contradicting_traits if trait_id in traits]
    support_conf = sum(traits[trait_id].confidence for trait_id in supporting) / max(len(supporting), 1)
    contradiction_penalty = 0.12 * len(contradicting)

    confidence = _clamp(0.22 + spread * 0.35 + weakness * 0.25 + support_conf * 0.35 - contradiction_penalty)
    return confidence, supporting, contradicting


def _timestamp_quality(moments: list[Moment]) -> float:
    if not moments:
        return 0.55
    weights = {"real": 1.0, "segment": 0.82, "interpolated": 0.52, "estimated": 0.38}
    return sum(weights.get(getattr(moment, "timestamp_source", "estimated"), 0.38) for moment in moments) / len(moments)


def _evidence_independence(evidence_ids: list[str], traits: dict[str, PsychologicalTrait]) -> int:
    families: set[str] = set()
    families.update(evidence_id.removeprefix("psi_ev_").split("_", 1)[0] for evidence_id in evidence_ids)
    for trait in traits.values():
        if set(trait.supporting_evidence_ids).intersection(evidence_ids):
            for metric in trait.supporting_metrics:
                parts = metric.split(".", 1)
                families.add(parts[1].split("_", 1)[0] if len(parts) > 1 else parts[0])
            for behaviour in trait.supporting_behaviours:
                families.add(behaviour.split("_", 1)[0])
    return len(families)


def _diagnosis_from_rule(
    rule: DiagnosisRule,
    confidence: float,
    supporting: list[str],
    contradicting: list[str],
    traits: dict[str, PsychologicalTrait],
    evidence: list[EvidenceItem],
    moments: list[Moment],
    dims: dict[str, int],
) -> DiagnosticDiagnosis:
    evidence_ids = _trait_evidence_ids([traits[trait_id] for trait_id in supporting if trait_id in traits])
    if not evidence_ids:
        evidence_ids = _fallback_evidence_ids(evidence, rule.limiter)
    severity = _severity_from_score(dims.get(rule.limiter, 50))
    return DiagnosticDiagnosis(
        diagnosis_id=rule.diagnosis_id,
        diagnosis_name=rule.diagnosis_name,
        confidence=round(confidence, 2),
        severity=severity,  # type: ignore[arg-type]
        supporting_traits=supporting,
        contradicting_traits=contradicting,
        supporting_evidence_ids=evidence_ids,
        supporting_moment_ids=_moment_ids(moments, rule.affected_dimensions),
        affected_dimensions=[DIMENSION_LABELS[dimension] for dimension in rule.affected_dimensions],
    )


def _select_diagnoses(
    scores: Scores,
    inference: PsychologicalInference,
    evidence: list[EvidenceItem],
    moments: list[Moment],
    audio_quality: AudioQuality,
    duration_ms: int,
) -> tuple[DiagnosticDiagnosis | None, DiagnosticDiagnosis | None, list[DiagnosticDiagnosis]]:
    dims = _dimension_scores(scores)
    traits = _visible_traits(inference)
    threshold = 0.62
    if not audio_quality.usable:
        threshold += 0.1
    if duration_ms and duration_ms < 45000:
        threshold += 0.08
    if duration_ms and duration_ms < 25000:
        threshold += 0.12
    ts_quality = _timestamp_quality(moments)
    if ts_quality < 0.7:
        threshold += 0.04

    candidates: list[DiagnosticDiagnosis] = []
    suppressed: list[DiagnosticDiagnosis] = []
    for rule in DIAGNOSES:
        confidence, supporting, contradicting = _rule_score(rule, dims, traits)
        support_traits = [traits[trait_id] for trait_id in supporting if trait_id in traits]
        evidence_ids = _trait_evidence_ids(support_traits)
        independent_sources = _evidence_independence(evidence_ids, traits)
        confidence *= 0.92 if duration_ms < 45000 else 1.0
        confidence *= 0.82 if not audio_quality.usable else 1.0
        confidence *= 0.95 if ts_quality < 0.7 else 1.0
        confidence *= 0.8 if independent_sources < 2 else 1.0
        diagnosis = _diagnosis_from_rule(rule, confidence, supporting, contradicting, traits, evidence, moments, dims)
        if confidence >= threshold and diagnosis.supporting_evidence_ids and independent_sources >= 2:
            candidates.append(diagnosis)
        else:
            suppressed.append(diagnosis)

    candidates.sort(key=lambda item: (item.confidence, _severity_value(item.severity)), reverse=True)
    primary = candidates[0] if candidates else None
    secondary = None
    if primary:
        primary_dims = set(primary.affected_dimensions)
        for diagnosis in candidates[1:]:
            overlap = len(primary_dims.intersection(diagnosis.affected_dimensions))
            if overlap / max(len(primary_dims), 1) < 0.5:
                secondary = diagnosis
                break
    return primary, secondary, suppressed


def _contradictions(
    scores: Scores,
    inference: PsychologicalInference,
    evidence: list[EvidenceItem],
    audio_quality: AudioQuality,
) -> list[DiagnosticContradiction]:
    dims = _dimension_scores(scores)
    traits = _visible_traits(inference)
    patterns = (
        ("clarity_low_command", "Clarity", "Command", dims["clarity"] >= 68 and dims["command"] <= 60, ("clear_communicator", "credible", "approval_seeking"), "clear_content_softened_by_low_finality"),
        ("structure_low_presence", "Structure", "Presence", dims["structure"] >= 68 and dims["presence"] <= 58, ("structured_thinker", "flat"), "organised_content_under_signals_importance"),
        ("presence_low_composure", "Presence", "Composure", dims["presence"] >= 68 and dims["composure"] <= 58, ("energetic", "rushed"), "attention_energy_arrives_with_pressure"),
        ("persuasion_weak_closing", "Persuasion", "Closing", dims["persuasion"] >= 65 and (inference and any(t.trait_id == "approval_seeking" and not t.suppress_from_report for t in inference.traits)), ("persuasive", "approval_seeking"), "persuasive_pull_loses_finality_at_close"),
        ("command_low_warmth", "Command", "Warmth", dims["command"] >= 70 and scores.derived_axes.trust_warmth <= 50, ("commanding",), "authority_signal_may_feel_less_relational"),
        ("confidence_high_rambling", "Confidence", "Structure", dims["command"] >= 68 and (inference and any(t.trait_id == "structured_thinker" and t.suppress_from_report for t in inference.traits)), ("commanding",), "confidence_signal_outpaces_answer_path"),
    )
    results: list[DiagnosticContradiction] = []
    quality_factor = 0.75 if not audio_quality.usable else 1.0
    for contradiction_id, strength, limiter, active, trait_ids, effect in patterns:
        if not active:
            continue
        source_traits = [traits[trait_id] for trait_id in trait_ids if trait_id in traits]
        evidence_ids = _trait_evidence_ids(source_traits) or _fallback_evidence_ids(evidence, limiter.lower())
        if not evidence_ids:
            continue
        trait_conf = sum(trait.confidence for trait in source_traits) / max(len(source_traits), 1)
        confidence = _clamp((0.55 + trait_conf * 0.35) * quality_factor)
        if confidence < 0.45:
            continue
        results.append(
            DiagnosticContradiction(
                contradiction_id=contradiction_id,
                strength=strength,
                limiter=limiter,
                why_it_happens=[trait.trait_id for trait in source_traits] or [effect],
                listener_effect=effect,
                evidence_ids=evidence_ids,
                confidence=round(confidence, 2),
            )
        )
    return sorted(results, key=lambda item: item.confidence, reverse=True)


def _hidden_cost(
    primary: DiagnosticDiagnosis | None,
    contradictions: list[DiagnosticContradiction],
) -> HiddenCostReasoning | None:
    if not primary:
        return None
    limiter = next((dimension.lower() for dimension in primary.affected_dimensions if dimension.lower() in HIDDEN_COSTS), "command")
    if len(primary.affected_dimensions) >= 2:
        limiter = primary.affected_dimensions[-1].lower()
    if limiter not in HIDDEN_COSTS:
        limiter = "command"
    source_signal, interpretation, consequence, listener_effect = HIDDEN_COSTS[limiter]
    evidence_ids = list(primary.supporting_evidence_ids)
    if contradictions:
        evidence_ids = list(dict.fromkeys(evidence_ids + contradictions[0].evidence_ids))
    return HiddenCostReasoning(
        cost_id=f"hidden_cost_{limiter}",
        source_signal=source_signal,
        interpretation=interpretation,
        consequence=consequence,
        listener_effect=listener_effect,
        affected_dimensions=primary.affected_dimensions,
        evidence_ids=evidence_ids,
        moment_ids=primary.supporting_moment_ids,
        confidence=primary.confidence,
    )


def _highest_leverage(
    primary: DiagnosticDiagnosis | None,
    scores: Scores,
    inference: PsychologicalInference,
    scenario: str,
) -> HighestLeverageReasoning | None:
    if not primary:
        return None
    dimension = next((item.lower() for item in reversed(primary.affected_dimensions) if item.lower() in FIX_RULES), "command")
    rule = FIX_RULES[dimension]
    severity = _severity_value(primary.severity)
    evidence_confidence = _clamp(min(primary.confidence, inference.overall_inference_confidence or primary.confidence))
    scenario_relevance = max(calculate_dimension_relevance(dimension, scenario), 0.85)
    selection_score = severity * rule.authority_impact * rule.trainability * evidence_confidence * scenario_relevance
    expected_lift = "high" if selection_score >= 0.55 else "medium" if selection_score >= 0.28 else "low"
    return HighestLeverageReasoning(
        issue_id=rule.issue_id,
        plain_reason=rule.plain_reason,
        affected_dimensions=list(rule.dimensions),
        supporting_evidence=primary.supporting_evidence_ids,
        expected_score_lift=expected_lift,  # type: ignore[arg-type]
        recommended_first_drill=rule.drill_id,
        confidence=round(evidence_confidence, 2),
        severity=severity,
        authority_impact=rule.authority_impact,
        trainability=rule.trainability,
        evidence_confidence=round(evidence_confidence, 2),
        scenario_relevance=scenario_relevance,
        selection_score=round(selection_score, 3),
    )


def _metric_value(metrics: Metrics, metric_name: str) -> float | int | str | bool | None:
    section, field = metric_name.split(".", 1)
    source = {
        "raw_acoustic": metrics.raw_acoustic,
        "linguistic": metrics.linguistic,
        "derived": metrics.derived,
        "rhythm": metrics.rhythm,
        "articulation": metrics.articulation,
        "vad": metrics.vad,
    }.get(section)
    return getattr(source, field, None) if source else None


def _dimension_reasoning(
    scores: Scores,
    metrics: Metrics,
    inference: PsychologicalInference,
    evidence: list[EvidenceItem],
) -> dict[str, DimensionReasoning]:
    dims = _dimension_scores(scores)
    behaviours = {
        behaviour.id: behaviour
        for behaviour in inference.micro_behaviours
        if behaviour.confidence >= 0.55
    }
    result: dict[str, DimensionReasoning] = {}
    for dimension in DIMENSIONS:
        score = dims[dimension]
        signals = DIMENSION_SIGNALS[dimension]
        active_behaviours = [item for item in signals["behaviours"] if item in behaviours]
        evidence_ids: list[str] = []
        for behaviour_id in active_behaviours:
            evidence_ids.extend(behaviours[behaviour_id].supporting_evidence_ids)
        evidence_ids = list(dict.fromkeys(evidence_ids)) or _fallback_evidence_ids(evidence, dimension)

        metric_values = [
            (name, _metric_value(metrics, name))
            for name in signals["metrics"]
            if _metric_value(metrics, name) is not None
        ]
        linguistic_values = [
            (name, _metric_value(metrics, name))
            for name in signals["linguistic"]
            if _metric_value(metrics, name) is not None
        ]
        why_high = []
        why_low = []
        if score >= 68:
            why_high.append(f"{dimension}_score_above_report_threshold")
        if score <= 58:
            why_low.append(f"{dimension}_score_below_report_threshold")
        why_high.extend([f"behaviour:{item}" for item in active_behaviours[:2] if not item.startswith(("flat", "under", "loose", "pressure"))])
        why_low.extend([f"behaviour:{item}" for item in active_behaviours[:2] if item.startswith(("flat", "under", "loose", "pressure"))])

        result[dimension] = DimensionReasoning(
            dimension=DIMENSION_LABELS[dimension],
            score=score,
            why_score_is_high=why_high,
            why_score_is_low=why_low,
            largest_positive_signal=why_high[0] if why_high else None,
            largest_negative_signal=why_low[0] if why_low else None,
            biggest_metric_contributor=metric_values[0][0] if metric_values else None,
            biggest_linguistic_contributor=linguistic_values[0][0] if linguistic_values else None,
            biggest_behavioural_contributor=active_behaviours[0] if active_behaviours else None,
            confidence=round(min(scores.score_confidence or 0.5, inference.overall_inference_confidence or 0.5), 2),
            supporting_evidence_ids=evidence_ids,
        )
    return result


def _trait_reasoning(
    inference: PsychologicalInference,
    moments: list[Moment],
) -> dict[str, TraitReasoning]:
    by_dimension_moments = {
        dimension: _moment_ids(moments, [dimension])
        for dimension in DIMENSIONS
    }
    result: dict[str, TraitReasoning] = {}
    for trait in inference.traits:
        if trait.suppress_from_report and not trait.supporting_evidence_ids:
            continue
        dimensions = [
            dimension
            for dimension in DIMENSIONS
            if any(dimension in metric for metric in trait.supporting_metrics)
            or any(dimension in behaviour for behaviour in trait.supporting_behaviours)
        ]
        supporting_moments: list[str] = []
        for dimension in dimensions:
            supporting_moments.extend(by_dimension_moments.get(dimension, []))
        result[trait.trait_id] = TraitReasoning(
            trait_id=trait.trait_id,
            label=trait.label,
            positive_evidence=trait.supporting_evidence_ids,
            negative_evidence=trait.contradicting_behaviours,
            confidence=trait.confidence,
            suppression_reason=trait.uncertainty_reason if trait.suppress_from_report else None,
            supporting_metrics=trait.supporting_metrics,
            supporting_moments=list(dict.fromkeys(supporting_moments)),
        )
    return result


def build_diagnostic_reasoning(
    *,
    metrics: Metrics,
    psychological_inference: PsychologicalInference,
    evidence: list[EvidenceItem],
    moments: list[Moment],
    scores: Scores,
    audio_quality: AudioQuality,
    uncertainty: Uncertainty,
    duration_ms: int,
    scenario: str,
) -> DiagnosticReasoning:
    """Build deterministic report reasoning from measured and inferred facts."""
    primary, secondary, suppressed = _select_diagnoses(
        scores,
        psychological_inference,
        evidence,
        moments,
        audio_quality,
        duration_ms,
    )
    contradictions = _contradictions(scores, psychological_inference, evidence, audio_quality)
    hidden_cost = _hidden_cost(primary, contradictions)
    leverage = _highest_leverage(primary, scores, psychological_inference, scenario)
    dimension_reasoning = _dimension_reasoning(scores, metrics, psychological_inference, evidence)
    trait_reasoning = _trait_reasoning(psychological_inference, moments)

    reasons = list(dict.fromkeys(uncertainty.reasons + psychological_inference.uncertainty.reasons))
    if not audio_quality.usable:
        reasons.append("Poor audio suppresses unsupported diagnostic reasoning")
    if duration_ms and duration_ms < 25000:
        reasons.append("Short recording suppresses low-confidence diagnoses")

    visible_confidences = [
        item.confidence
        for item in (primary, secondary)
        if item is not None
    ]
    overall = sum(visible_confidences) / len(visible_confidences) if visible_confidences else 0.0
    return DiagnosticReasoning(
        primary_diagnosis=primary,
        secondary_diagnosis=secondary,
        suppressed_diagnoses=suppressed,
        contradictions=contradictions,
        hidden_cost_reasoning=hidden_cost,
        dimension_reasoning=dimension_reasoning,
        trait_reasoning=trait_reasoning,
        highest_leverage_reasoning=leverage,
        uncertainty=Uncertainty(
            overall_confidence_label=_confidence_label(overall),  # type: ignore[arg-type]
            suppressed_traits=psychological_inference.suppressed_traits,
            reasons=list(dict.fromkeys(reasons)),
        ),
    )
