"""Deterministic coaching and intervention selection for Milestone 6."""

from __future__ import annotations

from dataclasses import dataclass

from schemas import (
    AudioQuality,
    CoachingDrillDefinition,
    CoachingEngine,
    CoachingReasoningChain,
    CoachingRootCause,
    DiagnosticReasoning,
    DrillDependency,
    ExpectedImprovement,
    InterventionCandidate,
    Metrics,
    PsychologicalInference,
    Scores,
    SelectedInterventions,
    Uncertainty,
)


DIMENSIONS = ("command", "clarity", "composure", "presence", "persuasion", "structure")


@dataclass(frozen=True)
class RootCauseRule:
    root_cause_id: str
    label: str
    required_signals: tuple[str, ...]
    optional_signals: tuple[str, ...]
    affected_dimensions: tuple[str, ...]


ROOT_CAUSE_RULES = (
    RootCauseRule(
        "cognitive_overload",
        "cognitive overload",
        ("high_fillers", "hesitation_high"),
        ("mid_phrase_pauses", "pace_acceleration", "rambling_high"),
        ("clarity", "composure", "structure"),
    ),
    RootCauseRule(
        "low_declarative_ownership",
        "low declarative ownership",
        ("closing_weak",),
        ("rising_endings", "hedges_high", "certainty_low"),
        ("command", "persuasion"),
    ),
    RootCauseRule(
        "pressure_leakage",
        "pressure leakage",
        ("pace_acceleration",),
        ("pace_fast", "burst_speaking", "hesitation_high", "unstable_rhythm"),
        ("composure", "command"),
    ),
    RootCauseRule(
        "low_vocal_contrast",
        "low vocal contrast",
        ("dynamic_emphasis_low",),
        ("pitch_variation_low", "energy_variation_low", "projection_low"),
        ("presence", "persuasion"),
    ),
    RootCauseRule(
        "unclear_answer_path",
        "unclear answer path",
        ("structure_low",),
        ("rambling_high", "repetition_high", "opening_weak", "closing_weak"),
        ("structure", "clarity"),
    ),
    RootCauseRule(
        "thin_proof",
        "thin proof",
        ("specificity_low",),
        ("concreteness_low", "certainty_low"),
        ("clarity", "persuasion"),
    ),
    RootCauseRule(
        "soft_commitment_language",
        "soft commitment language",
        ("hedges_high",),
        ("self_doubt", "certainty_low", "rising_endings"),
        ("command", "persuasion"),
    ),
    RootCauseRule(
        "rhythm_instability",
        "rhythm instability",
        ("unstable_rhythm",),
        ("mid_phrase_pauses", "hesitation_windows", "pace_acceleration"),
        ("composure", "clarity"),
    ),
    RootCauseRule(
        "articulation_load",
        "articulation load",
        ("articulation_weak",),
        ("pace_fast", "little_voiced_speech"),
        ("clarity",),
    ),
)


def drill_library() -> list[CoachingDrillDefinition]:
    """Return the deterministic drill library used by the selector."""
    return [
        _drill("pause_ownership_v1", "Pause Ownership", "pause_ownership", "Use intentional silence before key claims.", ("pause_ownership", "deliberate_pacing"), ("raw_acoustic.avg_pause_ms", "raw_acoustic.mid_phrase_pause_rate"), ("command", "composure"), 0.9, "beginner", 4, 0.9, (), (), ("mid_phrase_pauses", "hesitation_high", "pace_acceleration")),
        _drill("drop_the_landing_v1", "Drop the Landing", "declarative_finality", "Make important sentence endings land cleanly.", ("executive_finality", "speaking_with_conviction"), ("raw_acoustic.terminal_rising_ratio", "raw_acoustic.terminal_falling_ratio", "linguistic.closing_strength_score"), ("command", "composure"), 0.95, "beginner", 4, 0.9, ("pause_ownership_v1",), (), ("closing_weak", "rising_endings", "hedges_high")),
        _drill("filler_cut_v1", "Filler Cut", "filler_reduction", "Replace filler bursts with silence and restart control.", ("searching_for_wording", "pause_ownership"), ("linguistic.filler_words_per_min", "derived.hesitation_cluster_score"), ("clarity", "composure"), 0.82, "beginner", 5, 0.85, ("pause_ownership_v1",), (), ("high_fillers", "very_high_fillers", "hesitation_high")),
        _drill("pace_anchor_v1", "Pace Anchor", "pace_regulation", "Hold pace steady when the point becomes important.", ("deliberate_pacing", "composure_under_pressure"), ("raw_acoustic.words_per_minute", "rhythm.rhythm_consistency", "rhythm.speed_up_segments"), ("composure", "command"), 0.9, "beginner", 4, 0.9, (), (), ("pace_fast", "pace_acceleration", "burst_speaking")),
        _drill("emphasis_ladder_v1", "Emphasis Ladder", "dynamic_emphasis", "Give key words controlled contrast.", ("vocal_variety", "projection_control"), ("derived.dynamic_emphasis_score", "raw_acoustic.loudness_variation_db", "raw_acoustic.f0_range_semitones"), ("presence", "persuasion"), 0.75, "intermediate", 5, 0.8, ("projection_baseline_v1",), (), ("dynamic_emphasis_low", "pitch_variation_low", "energy_variation_low")),
        _drill("projection_baseline_v1", "Projection Baseline", "projection", "Build stable energy before adding emphasis.", ("projection_control",), ("derived.projection_index", "raw_acoustic.loudness_variation_db"), ("presence", "command"), 0.72, "beginner", 4, 0.82, (), ("audio_quality_poor",), ("projection_low", "energy_variation_low")),
        _drill("command_claim_v1", "Command Claim", "command", "Open claims with direct ownership.", ("speaking_with_conviction", "clear_opening_control"), ("linguistic.certainty_markers_per_100_words", "linguistic.opening_strength_score"), ("command", "structure"), 0.88, "intermediate", 5, 0.78, ("drop_the_landing_v1",), (), ("certainty_low", "opening_weak", "hedges_high")),
        _drill("presence_contrast_v1", "Presence Contrast", "presence", "Make important and unimportant words sound different.", ("vocal_variety", "projection_control"), ("derived.dynamic_emphasis_score", "raw_acoustic.f0_range_semitones"), ("presence",), 0.72, "intermediate", 5, 0.75, ("projection_baseline_v1",), (), ("dynamic_emphasis_low", "pitch_variation_low")),
        _drill("pressure_reset_v1", "Pressure Reset", "composure", "Reset pace and breath before the next claim.", ("composure_under_pressure", "deliberate_pacing"), ("rhythm.rhythm_consistency", "derived.hesitation_cluster_score"), ("composure", "clarity"), 0.86, "beginner", 4, 0.88, ("pause_ownership_v1",), (), ("pressure_leakage", "hesitation_high", "unstable_rhythm")),
        _drill("answer_first_v1", "Answer First", "opening_strength", "Put the main answer in the first sentence.", ("clear_opening_control", "structured_thinking"), ("linguistic.opening_strength_score", "linguistic.structure_score"), ("structure", "command"), 0.8, "beginner", 4, 0.88, (), (), ("opening_weak", "structure_low", "certainty_low")),
        _drill("clean_close_v1", "Clean Close", "closing_strength", "End with a full-stop takeaway.", ("clean_closing", "executive_finality"), ("linguistic.closing_strength_score", "raw_acoustic.terminal_falling_ratio"), ("command", "structure"), 0.85, "beginner", 4, 0.86, ("drop_the_landing_v1",), (), ("closing_weak", "rising_endings")),
        _drill("point_proof_close_v1", "Point, Proof, Close", "structure_compression", "Compress the answer into claim, proof, close.", ("structured_thinking", "clean_closing"), ("linguistic.structure_score", "linguistic.rambling_score", "linguistic.closing_strength_score"), ("structure", "clarity"), 0.8, "beginner", 5, 0.9, (), (), ("structure_low", "rambling_high", "closing_weak")),
        _drill("rambling_gate_v1", "Rambling Gate", "rambling_reduction", "Stop each answer once the point has advanced.", ("structured_thinking",), ("linguistic.rambling_score", "linguistic.repetition_rate"), ("structure", "clarity"), 0.78, "intermediate", 5, 0.82, ("point_proof_close_v1",), (), ("rambling_high", "repetition_high")),
        _drill("one_point_one_proof_v1", "One Point, One Proof", "specificity", "Add one concrete proof point to one claim.", ("grounded_specificity", "structured_thinking"), ("linguistic.specificity_score", "linguistic.concreteness_score"), ("clarity", "persuasion"), 0.78, "beginner", 5, 0.85, (), (), ("specificity_low", "concreteness_low")),
        _drill("certainty_replace_v1", "Certainty Replace", "certainty_language", "Replace soft qualifiers with clean commitments.", ("speaking_with_conviction", "low_approval_seeking"), ("linguistic.certainty_markers_per_100_words", "linguistic.hedges_per_100_words"), ("command", "persuasion"), 0.82, "beginner", 4, 0.8, (), (), ("certainty_low", "hedges_high", "self_doubt")),
        _drill("hedge_trim_v1", "Hedge Trim", "hedging_reduction", "Keep necessary nuance while removing reflexive hedges.", ("low_approval_seeking", "speaking_with_conviction"), ("linguistic.hedges_per_100_words", "linguistic.self_doubt_markers"), ("command", "clarity"), 0.76, "beginner", 4, 0.82, ("certainty_replace_v1",), (), ("hedges_high", "self_doubt")),
        _drill("rhythm_grid_v1", "Rhythm Grid", "rhythm_consistency", "Speak in even phrase groups without rushing the middle.", ("deliberate_pacing", "pause_ownership"), ("rhythm.rhythm_consistency", "rhythm.speed_up_segments"), ("composure", "clarity"), 0.8, "intermediate", 6, 0.78, ("pace_anchor_v1",), (), ("unstable_rhythm", "pace_acceleration", "mid_phrase_pauses")),
        _drill("articulation_edges_v1", "Articulation Edges", "articulation", "Clarify consonant edges without over-performing.", ("articulation_clarity",), ("articulation.clarity_proxy", "articulation.articulation_stability"), ("clarity",), 0.65, "beginner", 4, 0.74, (), ("asr_low_confidence",), ("articulation_weak",)),
        _drill("breath_mark_v1", "Breath Mark", "breath_control", "Mark the breath before high-value phrases.", ("composure_under_pressure", "projection_control"), ("rhythm.rhythm_consistency", "derived.projection_index"), ("composure", "presence"), 0.74, "beginner", 4, 0.82, ("pause_ownership_v1",), (), ("pressure_leakage", "projection_low", "pace_acceleration")),
        _drill("pressure_claim_v1", "Pressure Claim", "confidence_under_pressure", "Deliver the key claim after a pause without speeding up.", ("composure_under_pressure", "speaking_with_conviction"), ("derived.composure_index", "derived.vocal_command_index", "rhythm.rhythm_consistency"), ("command", "composure"), 0.92, "advanced", 6, 0.76, ("pause_ownership_v1", "drop_the_landing_v1"), (), ("pressure_leakage", "rising_endings", "pace_acceleration")),
    ]


def _drill(
    drill_id: str,
    title: str,
    category: str,
    description: str,
    target_behaviours: tuple[str, ...],
    target_metrics: tuple[str, ...],
    target_dimensions: tuple[str, ...],
    impact: float,
    difficulty: str,
    duration: int,
    trainability: float,
    prerequisites: tuple[str, ...],
    contraindications: tuple[str, ...],
    evidence_requirements: tuple[str, ...],
) -> CoachingDrillDefinition:
    return CoachingDrillDefinition(
        drill_id=drill_id,
        title=title,
        category=category,
        description=description,
        target_behaviours=list(target_behaviours),
        target_metrics=list(target_metrics),
        target_dimensions=list(target_dimensions),
        expected_authority_impact=impact,
        expected_difficulty=difficulty,  # type: ignore[arg-type]
        estimated_duration_min=duration,
        trainability_score=trainability,
        prerequisites=list(prerequisites),
        contraindications=list(contraindications),
        evidence_requirements=list(evidence_requirements),
    )


def dependency_graph() -> list[DrillDependency]:
    return [
        DrillDependency(before="pause_ownership_v1", after="drop_the_landing_v1", reason="owned silence makes finality easier to hear"),
        DrillDependency(before="pause_ownership_v1", after="filler_cut_v1", reason="silence must replace filler before fillers can drop"),
        DrillDependency(before="pause_ownership_v1", after="pressure_reset_v1", reason="resetting pressure starts with owning the pause"),
        DrillDependency(before="pace_anchor_v1", after="rhythm_grid_v1", reason="rhythm consistency depends on baseline pace control"),
        DrillDependency(before="projection_baseline_v1", after="emphasis_ladder_v1", reason="dynamic emphasis requires stable projection first"),
        DrillDependency(before="projection_baseline_v1", after="presence_contrast_v1", reason="presence contrast should not be built on unstable energy"),
        DrillDependency(before="drop_the_landing_v1", after="command_claim_v1", reason="command claims need clean endings to land"),
        DrillDependency(before="drop_the_landing_v1", after="clean_close_v1", reason="closing strength depends on finality"),
        DrillDependency(before="point_proof_close_v1", after="rambling_gate_v1", reason="rambling reduction needs a compact answer shape"),
        DrillDependency(before="certainty_replace_v1", after="hedge_trim_v1", reason="hedge removal works best after a replacement pattern exists"),
        DrillDependency(before="pause_ownership_v1", after="breath_mark_v1", reason="breath control starts with deliberate phrase spacing"),
        DrillDependency(before="drop_the_landing_v1", after="pressure_claim_v1", reason="pressure claims require finality under load"),
    ]


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


def _active_signal_evidence(inference: PsychologicalInference) -> dict[str, list[str]]:
    signals: dict[str, list[str]] = {}
    for behaviour in inference.micro_behaviours:
        for evidence_id in behaviour.supporting_evidence_ids:
            signal = evidence_id.removeprefix("psi_ev_")
            signals.setdefault(signal, []).append(evidence_id)
    if inference.report_candidates.highest_leverage_candidates:
        for candidate in inference.report_candidates.highest_leverage_candidates:
            for evidence_id in candidate.evidence_ids:
                signal = evidence_id.removeprefix("psi_ev_")
                signals.setdefault(signal, []).append(evidence_id)
    return {key: list(dict.fromkeys(value)) for key, value in signals.items()}


def _build_root_causes(
    inference: PsychologicalInference,
    diagnostic_reasoning: DiagnosticReasoning,
) -> list[CoachingRootCause]:
    active = _active_signal_evidence(inference)
    causes: list[CoachingRootCause] = []
    for rule in ROOT_CAUSE_RULES:
        required_hits = [signal for signal in rule.required_signals if signal in active]
        optional_hits = [signal for signal in rule.optional_signals if signal in active]
        if not required_hits:
            continue
        evidence_ids = []
        for signal in required_hits + optional_hits:
            evidence_ids.extend(active.get(signal, []))
        confidence = _clamp(0.45 + 0.25 * len(required_hits) + 0.1 * len(optional_hits))
        if diagnostic_reasoning.primary_diagnosis and any(
            dimension.title() in diagnostic_reasoning.primary_diagnosis.affected_dimensions
            for dimension in rule.affected_dimensions
        ):
            confidence = _clamp(confidence + 0.08)
        causes.append(
            CoachingRootCause(
                root_cause_id=rule.root_cause_id,
                label=rule.label,
                contributing_signals=required_hits + optional_hits,
                evidence_ids=list(dict.fromkeys(evidence_ids)),
                confidence=round(confidence, 2),
                affected_dimensions=list(rule.affected_dimensions),
            )
        )
    return sorted(causes, key=lambda item: item.confidence, reverse=True)


def _dimension_scores(scores: Scores) -> dict[str, int]:
    return scores.dimension_scores.model_dump()


def _severity(drill: CoachingDrillDefinition, scores: Scores, causes: list[CoachingRootCause]) -> float:
    dims = _dimension_scores(scores)
    dimension_severity = max(
        (70 - dims.get(dimension, 70)) / 40
        for dimension in drill.target_dimensions
        if dimension in dims
    )
    cause_bonus = max(
        (
            cause.confidence
            for cause in causes
            if set(cause.affected_dimensions).intersection(drill.target_dimensions)
        ),
        default=0.0,
    )
    return round(_clamp(max(0.2, dimension_severity) + cause_bonus * 0.2), 2)


def _candidate_evidence(
    drill: CoachingDrillDefinition,
    active: dict[str, list[str]],
    diagnostic_reasoning: DiagnosticReasoning,
) -> list[str]:
    ids: list[str] = []
    for requirement in drill.evidence_requirements:
        ids.extend(active.get(requirement, []))
    if diagnostic_reasoning.highest_leverage_reasoning:
        if set(drill.target_dimensions).intersection(diagnostic_reasoning.highest_leverage_reasoning.affected_dimensions):
            ids.extend(diagnostic_reasoning.highest_leverage_reasoning.supporting_evidence)
    return list(dict.fromkeys(ids))


def _scenario_relevance(drill: CoachingDrillDefinition, scenario: str) -> float:
    if scenario == "benchmark":
        return 1.0
    if scenario == "impromptu" and drill.category in {"pace_regulation", "pause_ownership", "composure", "structure_compression"}:
        return 1.05
    return 1.0


def _expected_improvement(
    drill: CoachingDrillDefinition,
    score: float,
    confidence: float,
) -> ExpectedImprovement:
    authority = round(score * 8, 2)
    values = {dimension: 0.0 for dimension in DIMENSIONS}
    for dimension in drill.target_dimensions:
        if dimension in values:
            values[dimension] = round(authority * (0.7 if len(drill.target_dimensions) > 1 else 1.0), 2)
    return ExpectedImprovement(
        drill_id=drill.drill_id,
        authority_score=authority,
        command=values["command"],
        clarity=values["clarity"],
        composure=values["composure"],
        presence=values["presence"],
        persuasion=values["persuasion"],
        structure=values["structure"],
        confidence=round(confidence, 2),
    )


def _score_candidate(
    drill: CoachingDrillDefinition,
    scores: Scores,
    causes: list[CoachingRootCause],
    active: dict[str, list[str]],
    diagnostic_reasoning: DiagnosticReasoning,
    audio_quality: AudioQuality,
    duration_ms: int,
    scenario: str,
) -> InterventionCandidate:
    evidence_ids = _candidate_evidence(drill, active, diagnostic_reasoning)
    severity = _severity(drill, scores, causes)
    evidence_ratio = len(set(drill.evidence_requirements).intersection(active.keys())) / max(len(drill.evidence_requirements), 1)
    confidence = _clamp(0.25 + evidence_ratio * 0.55)
    if evidence_ids:
        confidence = _clamp(confidence + 0.1)
    if not audio_quality.usable:
        confidence *= 0.45
    if duration_ms and duration_ms < 25000:
        confidence *= 0.7
    scenario_relevance = _scenario_relevance(drill, scenario)
    score = severity * drill.expected_authority_impact * drill.trainability_score * confidence * scenario_relevance
    why_not = None
    contraindications = set(drill.contraindications)
    if not audio_quality.usable and "audio_quality_poor" in contraindications:
        why_not = "suppressed_by_audio_quality"
        score *= 0.25
    elif confidence < 0.4:
        why_not = "insufficient_supporting_evidence"
    candidate = InterventionCandidate(
        drill_id=drill.drill_id,
        title=drill.title,
        score=round(score, 3),
        severity=severity,
        authority_impact=drill.expected_authority_impact,
        trainability=drill.trainability_score,
        confidence=round(confidence, 2),
        scenario_relevance=scenario_relevance,
        required_evidence=list(drill.evidence_requirements),
        supporting_evidence_ids=evidence_ids,
        expected_impact=_expected_improvement(drill, score, confidence),
        why_selected=None,
        why_not_selected=why_not,
    )
    return candidate


def _select_candidates(
    candidates: list[InterventionCandidate],
    audio_quality: AudioQuality,
    duration_ms: int,
) -> tuple[SelectedInterventions, list[InterventionCandidate], list[InterventionCandidate]]:
    selectable = [
        candidate
        for candidate in candidates
        if candidate.why_not_selected is None and candidate.supporting_evidence_ids and candidate.confidence >= 0.4
    ]
    if not audio_quality.usable or (duration_ms and duration_ms < 8000):
        selectable = []

    selectable.sort(key=lambda item: (item.score, item.confidence, item.drill_id), reverse=True)
    primary = selectable[0].model_copy(update={"why_selected": "highest_weighted_intervention_score"}) if selectable else None
    secondary = None
    if primary:
        for candidate in selectable[1:]:
            if candidate.drill_id != primary.drill_id:
                secondary = candidate.model_copy(update={"why_selected": "next_highest_distinct_intervention_score"})
                break

    selected_ids = {item.drill_id for item in (primary, secondary) if item}
    queue = [candidate for candidate in selectable if candidate.drill_id not in selected_ids][:5]
    suppressed = [
        candidate
        for candidate in candidates
        if candidate.drill_id not in {item.drill_id for item in selectable}
    ]
    if not audio_quality.usable or (duration_ms and duration_ms < 8000):
        suppressed = [
            candidate.model_copy(update={"why_not_selected": "suppressed_by_recording_quality"})
            for candidate in candidates
        ]
    return SelectedInterventions(primary_drill=primary, secondary_drill=secondary), queue, suppressed


def _reasoning_chain(
    root_causes: list[CoachingRootCause],
    selected: SelectedInterventions,
) -> CoachingReasoningChain:
    primary_cause = root_causes[0] if root_causes else None
    primary = selected.primary_drill
    evidence_ids = []
    if primary_cause:
        evidence_ids.extend(primary_cause.evidence_ids)
    if primary:
        evidence_ids.extend(primary.supporting_evidence_ids)
    return CoachingReasoningChain(
        detected=[cause.label for cause in root_causes[:3]],
        contributing_factors=primary_cause.contributing_signals if primary_cause else [],
        root_issue=primary_cause.root_cause_id if primary_cause else None,
        highest_leverage_intervention=primary.drill_id if primary else None,
        reason="addresses_multiple_downstream_authority_dimensions" if primary else None,
        evidence_ids=list(dict.fromkeys(evidence_ids)),
    )


def build_deterministic_coaching(
    *,
    metrics: Metrics,
    scores: Scores,
    psychological_inference: PsychologicalInference,
    diagnostic_reasoning: DiagnosticReasoning,
    report,
    audio_quality: AudioQuality,
    uncertainty: Uncertainty,
    duration_ms: int,
    scenario: str,
) -> CoachingEngine:
    """Select deterministic interventions from report-ready evidence."""
    del metrics, report
    library = drill_library()
    active = _active_signal_evidence(psychological_inference)
    root_causes = _build_root_causes(psychological_inference, diagnostic_reasoning)
    candidates = [
        _score_candidate(
            drill,
            scores,
            root_causes,
            active,
            diagnostic_reasoning,
            audio_quality,
            duration_ms,
            scenario,
        )
        for drill in library
    ]
    candidates.sort(key=lambda item: (item.score, item.confidence, item.drill_id), reverse=True)
    selected, queue, suppressed = _select_candidates(candidates, audio_quality, duration_ms)
    expected = {
        candidate.drill_id: candidate.expected_impact
        for candidate in candidates
    }
    reasons = list(dict.fromkeys(uncertainty.reasons + diagnostic_reasoning.uncertainty.reasons))
    if not selected.primary_drill:
        reasons.append("Coaching suppressed because intervention evidence is insufficient")
    if not audio_quality.usable:
        reasons.append("Poor audio suppresses coaching intervention selection")

    confidences = [candidate.confidence for candidate in candidates if candidate.supporting_evidence_ids]
    overall = sum(confidences) / len(confidences) if confidences else 0.0
    return CoachingEngine(
        drill_library=library,
        drill_library_size=len(library),
        root_causes=root_causes,
        intervention_candidates=candidates,
        selected_interventions=selected,
        suppressed_interventions=suppressed,
        reasoning_chain=_reasoning_chain(root_causes, selected),
        expected_improvements=expected,
        dependency_graph=dependency_graph(),
        future_training_queue=queue,
        uncertainty=Uncertainty(
            overall_confidence_label=_confidence_label(overall),  # type: ignore[arg-type]
            suppressed_traits=diagnostic_reasoning.uncertainty.suppressed_traits,
            reasons=reasons,
        ),
    )
