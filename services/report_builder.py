"""Deterministic Authority report assembly from measured backend facts."""

from __future__ import annotations

from dataclasses import dataclass

from schemas import (
    AudioQuality,
    AuthorityReport,
    DiagnosticDiagnosis,
    DiagnosticReasoning,
    EvidenceItem,
    HiddenCostReasoning,
    HighestLeverageReasoning,
    Metrics,
    Moment,
    PsychologicalInference,
    PsychologicalTrait,
    ReportAuthorityType,
    ReportDiagnosis,
    ReportHiddenCost,
    ReportHighestLeverageFix,
    ReportMirror,
    ReportPerceptionMap,
    ReportPerceptionRead,
    ReportRetestPlan,
    ReportShareCard,
    ReportTechnicalAppendix,
    ReportTrainingPrescription,
    Scores,
    Uncertainty,
)


DIMENSION_LABELS = {
    "command": "Command",
    "clarity": "Clarity",
    "composure": "Composure",
    "presence": "Presence",
    "persuasion": "Persuasion",
    "structure": "Structure",
}

POSITIVE_TRAITS = {
    "confident",
    "composed",
    "credible",
    "trustworthy",
    "warm",
    "commanding",
    "high_status",
    "persuasive",
    "energetic",
    "calm",
    "leadership_ready",
    "interview_ready",
    "executive_presence",
    "structured_thinker",
    "clear_communicator",
}

PRIVATE_TRAITS = {"nervous", "approval_seeking", "hesitant"}


@dataclass(frozen=True)
class FixRule:
    issue: str
    plain_english: str
    why_this_matters: str
    target_dimensions: tuple[str, ...]
    drill_id: str
    compare_metrics: tuple[str, ...]
    trainability: float
    authority_impact: float


FIX_RULES: dict[str, FixRule] = {
    "command": FixRule(
        issue="declarative finality",
        plain_english="Your key lines need to land more cleanly.",
        why_this_matters="Cleaner endings make ideas sound more final and less permission-seeking.",
        target_dimensions=("command", "composure"),
        drill_id="drop_the_landing_v1",
        compare_metrics=("terminal_rising_ratio", "terminal_falling_ratio", "closing_strength_score"),
        trainability=0.9,
        authority_impact=0.95,
    ),
    "clarity": FixRule(
        issue="clarity compression",
        plain_english="Your answer needs to become easier to follow.",
        why_this_matters="Listeners trust the point faster when wording and structure reduce effort.",
        target_dimensions=("clarity", "structure"),
        drill_id="one_point_one_proof_v1",
        compare_metrics=("structure_score", "filler_words_per_min", "rambling_score"),
        trainability=0.85,
        authority_impact=0.85,
    ),
    "composure": FixRule(
        issue="pace control",
        plain_english="Your delivery needs to sound less reactive under pressure.",
        why_this_matters="A steadier rhythm makes listeners feel you are leading the moment.",
        target_dimensions=("composure", "command"),
        drill_id="pace_anchor_v1",
        compare_metrics=("words_per_minute", "rhythm_consistency", "hesitation_cluster_score"),
        trainability=0.9,
        authority_impact=0.9,
    ),
    "presence": FixRule(
        issue="dynamic emphasis",
        plain_english="Your important words need more contrast.",
        why_this_matters="Selective emphasis helps the listener feel what matters and remember it.",
        target_dimensions=("presence", "persuasion"),
        drill_id="emphasis_ladder_v1",
        compare_metrics=("dynamic_emphasis_score", "loudness_variation_db", "f0_range_semitones"),
        trainability=0.8,
        authority_impact=0.75,
    ),
    "persuasion": FixRule(
        issue="conviction framing",
        plain_english="Your argument needs a clearer pull toward a conclusion.",
        why_this_matters="Persuasion comes from guiding attention, not just explaining clearly.",
        target_dimensions=("persuasion", "structure"),
        drill_id="claim_stakes_action_v1",
        compare_metrics=("certainty_markers_per_100_words", "specificity_score", "dynamic_emphasis_score"),
        trainability=0.75,
        authority_impact=0.75,
    ),
    "structure": FixRule(
        issue="structure compression",
        plain_english="Your answer needs a cleaner path from point to proof to close.",
        why_this_matters="A clear path makes listeners feel you know where the answer is going.",
        target_dimensions=("structure", "clarity"),
        drill_id="point_proof_close_v1",
        compare_metrics=("opening_strength_score", "structure_score", "closing_strength_score"),
        trainability=0.9,
        authority_impact=0.8,
    ),
}


DRILLS = {
    "drop_the_landing_v1": {
        "title": "Drop the Landing",
        "instructions": [
            "Read 8 short statements aloud.",
            "Let the final stressed word fall slightly.",
            "Hold silence for half a second after each line.",
            "Repeat once with more calm than force.",
        ],
        "target_metrics": ["terminal_rising_ratio", "terminal_falling_ratio", "closing_strength_score"],
        "success_signal": "Endings should sound cleaner, slower, and less like questions.",
    },
    "one_point_one_proof_v1": {
        "title": "One Point, One Proof",
        "instructions": [
            "State one claim in the first sentence.",
            "Give one concrete proof point.",
            "Cut any restart or filler before the close.",
        ],
        "target_metrics": ["structure_score", "filler_words_per_min", "rambling_score"],
        "success_signal": "The answer should feel easier to follow without extra explanation.",
    },
    "pace_anchor_v1": {
        "title": "Pace Anchor",
        "instructions": [
            "Speak one sentence at a controlled pace.",
            "Pause before the key claim.",
            "Repeat the claim without speeding up.",
        ],
        "target_metrics": ["words_per_minute", "rhythm_consistency", "hesitation_cluster_score"],
        "success_signal": "Pace should stay steady when the point becomes important.",
    },
    "emphasis_ladder_v1": {
        "title": "Emphasis Ladder",
        "instructions": [
            "Choose three important words in a short answer.",
            "Give each word slightly more energy than the surrounding phrase.",
            "Keep the rest of the sentence calm.",
        ],
        "target_metrics": ["dynamic_emphasis_score", "loudness_variation_db", "f0_range_semitones"],
        "success_signal": "Important words should stand out without sounding forced.",
    },
    "claim_stakes_action_v1": {
        "title": "Claim, Stakes, Action",
        "instructions": [
            "Open with the claim.",
            "Name why it matters.",
            "End with the action or conclusion you want remembered.",
        ],
        "target_metrics": ["certainty_markers_per_100_words", "specificity_score", "dynamic_emphasis_score"],
        "success_signal": "The listener should feel guided toward one conclusion.",
    },
    "point_proof_close_v1": {
        "title": "Point, Proof, Close",
        "instructions": [
            "Make the point in one sentence.",
            "Add one proof point.",
            "Close with a clean takeaway.",
        ],
        "target_metrics": ["opening_strength_score", "structure_score", "closing_strength_score"],
        "success_signal": "The answer should have a visible beginning, middle, and end.",
    },
}


def _dimension_scores(scores: Scores) -> dict[str, int]:
    return scores.dimension_scores.model_dump()


def _ordered_dimensions(scores: Scores) -> list[tuple[str, int]]:
    return sorted(_dimension_scores(scores).items(), key=lambda item: item[1], reverse=True)


def _traits(psychological_inference: PsychologicalInference) -> dict[str, PsychologicalTrait]:
    return {
        trait.trait_id: trait
        for trait in psychological_inference.traits
        if not trait.suppress_from_report
    }


def _trait_evidence_ids(traits: list[PsychologicalTrait]) -> list[str]:
    ids: list[str] = []
    for trait in traits:
        ids.extend(trait.supporting_evidence_ids)
    return list(dict.fromkeys(ids))


def _fallback_evidence_ids(evidence: list[EvidenceItem]) -> list[str]:
    return [item.id for item in evidence[:3]]


def _primary_evidence_ids(
    psychological_inference: PsychologicalInference,
    evidence: list[EvidenceItem],
) -> list[str]:
    candidate = psychological_inference.primary_candidates.primary_strength_candidate
    ids = list(candidate.evidence_ids) if candidate else []
    if not ids:
        ids = _fallback_evidence_ids(evidence)
    return ids


def _confidence_label(confidence: float | None) -> str:
    value = confidence or 0.0
    if value >= 0.8:
        return "high"
    if value >= 0.6:
        return "medium_high"
    if value >= 0.4:
        return "medium"
    return "low"


def _severity(score: int) -> str:
    if score < 45:
        return "high"
    if score < 60:
        return "medium"
    return "low"


def _severity_value(severity: str | None) -> float:
    return {"high": 1.0, "medium": 0.65, "low": 0.35}.get(severity or "low", 0.35)


def _authority_type(
    scores: Scores,
    psychological_inference: PsychologicalInference,
    evidence_ids: list[str],
) -> ReportAuthorityType:
    dims = _dimension_scores(scores)
    top = [name for name, _ in _ordered_dimensions(scores)[:2]]
    low = [name for name, _ in sorted(dims.items(), key=lambda item: item[1])[:2]]
    axes = scores.derived_axes

    def high(*names: str, threshold: int = 70) -> bool:
        return all(dims.get(name, 0) >= threshold for name in names)

    def low_dim(name: str, threshold: int = 58) -> bool:
        return dims.get(name, 100) < threshold

    trait_ids = {
        trait.trait_id
        for trait in psychological_inference.traits
        if not trait.suppress_from_report and trait.score >= 60
    }

    type_id = "developing_voice"
    label = "Developing Voice"
    description = "Your communication has a foundation, but no single authority signal dominates yet."

    if high("command", "clarity", "composure", "presence", threshold=82):
        type_id, label = "executive_presence", "Executive Presence"
        description = "You sound clear, intentional, and fully self-possessed."
    elif high("command", "presence", "composure", threshold=72):
        type_id, label = "natural_leader", "Natural Leader"
        description = "You sound decisive, settled, and easy to trust with the floor."
    elif axes.nervousness >= 75 or (low_dim("composure", 50) and low_dim("command", 55)):
        type_id, label = "unsettled_speaker", "Unsettled Speaker"
        description = "Your ideas may be stronger than your delivery currently allows people to feel."
    elif "rushed" in trait_ids or axes.nervousness >= 65:
        type_id, label = "rushed_achiever", "Rushed Achiever"
        description = "You have useful ideas, but delivery can make them sound pressured."
    elif high("clarity", "structure", threshold=70) and dims["presence"] >= 55:
        type_id, label = "trusted_expert", "Trusted Expert"
        description = "You sound knowledgeable and reliable, with credibility as the strongest signal."
    elif high("composure", threshold=70) and low_dim("presence", 65):
        type_id, label = "calm_professional", "Calm Professional"
        description = "You sound steady and controlled, with an opportunity to become more memorable."
    elif high("clarity", threshold=66) and low_dim("presence", 58):
        type_id, label = "quiet_analyst", "Quiet Analyst"
        description = "You sound thoughtful and precise, but may under-signal importance."
    elif high("structure", "clarity", threshold=66) and dims["command"] < 70:
        type_id, label = "thoughtful_strategist", "Thoughtful Strategist"
        description = "You sound intelligent and measured, with room to sound more commanding."
    elif high("persuasion", "presence", threshold=68):
        type_id, label = "persuasive_operator", "Persuasive Operator"
        description = "You sound engaging and influential, with structure as the main stabiliser."

    confidence = max(
        psychological_inference.overall_inference_confidence,
        scores.score_confidence or 0.0,
    )
    return ReportAuthorityType(
        type_id=type_id,
        label=label,
        description=description,
        top_dimensions=[DIMENSION_LABELS[name] for name in top],
        growth_dimensions=[DIMENSION_LABELS[name] for name in low],
        evidence_ids=evidence_ids,
        confidence=round(min(confidence, 0.95), 2),
    )


def _mirror(
    scores: Scores,
    authority_type: ReportAuthorityType,
    strongest: str,
    limiter: str,
    confidence_label: str,
    evidence_ids: list[str],
) -> ReportMirror:
    score = scores.authority_score
    if score >= 90:
        headline = "You sound like someone people naturally defer to: clear, intentional, and fully in control."
    elif score >= 80:
        headline = "You sound clear, composed, and easy to trust with the floor."
    elif score >= 67:
        headline = "You sound composed, intelligent, and increasingly authoritative."
    elif score >= 53:
        headline = f"You sound capable and {strongest.lower()}, but not yet fully {limiter.lower()}."
    elif score >= 39:
        headline = "You sound thoughtful, but not yet consistently settled."
    else:
        headline = "This recording suggests your delivery may be under-signalling your point."

    identity_read = (
        f"Listeners are likely to feel your {strongest.lower()} first, while "
        f"{limiter.lower()} is the main growth area."
    )
    return ReportMirror(
        headline=headline,
        identity_read=identity_read,
        core_tension=f"{strongest} constrained by {limiter}",
        emotional_tone=_emotional_tone(scores),
        authority_type=authority_type.label,
        confidence_label=confidence_label,  # type: ignore[arg-type]
        evidence_ids=evidence_ids,
    )


def _emotional_tone(scores: Scores) -> str:
    dims = _dimension_scores(scores)
    if dims["composure"] >= 72 and dims["presence"] >= 65:
        return "settled and engaged"
    if dims["composure"] >= 68:
        return "calm and measured"
    if dims["presence"] < 55:
        return "restrained and low-contrast"
    if dims["composure"] < 55:
        return "pressured and reactive"
    return "competent with some unevenness"


def _diagnosis(
    scores: Scores,
    psychological_inference: PsychologicalInference,
    evidence_ids: list[str],
    diagnostic: DiagnosticDiagnosis | None = None,
) -> ReportDiagnosis:
    if diagnostic:
        strongest = diagnostic.affected_dimensions[0] if diagnostic.affected_dimensions else None
        limiter = diagnostic.affected_dimensions[-1] if diagnostic.affected_dimensions else None
        return ReportDiagnosis(
            strongest_dimension=strongest,
            limiting_dimension=limiter,
            core_behavioural_pattern=diagnostic.diagnosis_id,
            social_consequence=diagnostic.diagnosis_name,
            supporting_evidence_ids=diagnostic.supporting_evidence_ids,
            severity=diagnostic.severity,
        )

    ordered = _ordered_dimensions(scores)
    strongest_key, strongest_score = ordered[0]
    limiting_key, limiting_score = sorted(_dimension_scores(scores).items(), key=lambda item: item[1])[0]
    traits = _traits(psychological_inference)

    pattern = f"{DIMENSION_LABELS[strongest_key]} limited by {DIMENSION_LABELS[limiting_key].lower()}"
    if "rushed" in traits:
        pattern = "strong ideas pressured by accelerating delivery"
    elif "flat" in traits:
        pattern = "clear content undercut by low vocal contrast"
    elif "approval_seeking" in traits:
        pattern = "strong ideas softened by hesitant delivery"
    elif limiting_key == "structure":
        pattern = "useful content weakened by an unclear answer path"

    consequence = {
        "command": "listeners may understand the point without fully feeling led by it",
        "clarity": "listeners may spend more effort following the answer than weighing the idea",
        "composure": "listeners may hear pressure even when the words are correct",
        "presence": "listeners may agree in the moment but remember less afterwards",
        "persuasion": "listeners may understand the explanation without feeling pulled toward action",
        "structure": "listeners may trust the content less when the path feels unclear",
    }[limiting_key]

    return ReportDiagnosis(
        strongest_dimension=DIMENSION_LABELS[strongest_key],
        limiting_dimension=DIMENSION_LABELS[limiting_key],
        core_behavioural_pattern=pattern,
        social_consequence=consequence,
        supporting_evidence_ids=evidence_ids,
        severity=_severity(limiting_score),  # type: ignore[arg-type]
    )


def _read(label: str, text: str, evidence_ids: list[str], confidence: float) -> ReportPerceptionRead:
    return ReportPerceptionRead(
        label=label,
        text=text,
        evidence_ids=evidence_ids,
        confidence=round(confidence, 2),
    )


def _perception_map(
    scores: Scores,
    diagnosis: ReportDiagnosis,
    authority_type: ReportAuthorityType,
    evidence_ids: list[str],
    confidence: float,
) -> ReportPerceptionMap:
    limiter = (diagnosis.limiting_dimension or "Command").lower()
    strength = (diagnosis.strongest_dimension or "Clarity").lower()
    return ReportPerceptionMap(
        first_impression=_read(
            authority_type.label or "Developing Voice",
            f"The first impression is {authority_type.description or 'mixed but interpretable'}.",
            evidence_ids,
            confidence,
        ),
        professional_read=_read(
            f"{strength.title()} led",
            f"Professionally, the recording is most likely to land through {strength}, with {limiter} shaping the ceiling.",
            evidence_ids,
            confidence,
        ),
        leadership_read=_read(
            "Leadership signal",
            "Leadership read is strongest when command, composure, and structure align.",
            evidence_ids,
            confidence,
        ),
        interview_read=_read(
            "Interview signal",
            "Interview readiness depends on clear openings, controlled delivery, and clean closing.",
            evidence_ids,
            confidence,
        ),
        social_status_read=_read(
            "Status signal",
            f"Status read is currently limited most by {limiter}.",
            evidence_ids,
            confidence,
        ),
        emotional_read=_read(
            _emotional_tone(scores).title(),
            f"The emotional read is {_emotional_tone(scores)} based on composure and presence signals.",
            evidence_ids,
            confidence,
        ),
        trust_read=_read(
            "Trust signal",
            "Trust is supported when clarity, structure, and composure are present together.",
            evidence_ids,
            confidence,
        ),
        persuasion_read=_read(
            "Persuasion signal",
            "Persuasion strengthens when conviction, structure, and emphasis are aligned.",
            evidence_ids,
            confidence,
        ),
    )


def _hidden_cost(
    diagnosis: ReportDiagnosis,
    evidence_ids: list[str],
    confidence: float,
    reasoning: HiddenCostReasoning | None = None,
) -> ReportHiddenCost:
    if reasoning:
        return ReportHiddenCost(
            dimension=diagnosis.limiting_dimension,
            cost_id=reasoning.cost_id,
            consequence=reasoning.listener_effect,
            evidence_ids=reasoning.evidence_ids,
            confidence=round(reasoning.confidence, 2),
        )

    dimension = (diagnosis.limiting_dimension or "Command").lower()
    consequence = {
        "command": "The hidden cost is that listeners may understand you and still not feel fully led by you.",
        "clarity": "The hidden cost is cognitive effort: listeners have less energy left to be persuaded.",
        "composure": "The hidden cost is pressure leakage: the listener can feel the pressure in real time.",
        "presence": "The hidden cost is memorability: the point may not stay with the listener.",
        "persuasion": "The hidden cost is movement: explanation may not turn into action.",
        "structure": "The hidden cost is authority drift: an unclear path can reduce trust in your control.",
    }.get(dimension, "The hidden cost is reduced authority signal.")
    return ReportHiddenCost(
        dimension=diagnosis.limiting_dimension,
        cost_id=f"hidden_cost_{dimension}",
        consequence=consequence,
        evidence_ids=evidence_ids,
        confidence=round(confidence, 2),
    )


def _fix(
    diagnosis: ReportDiagnosis,
    psychological_inference: PsychologicalInference,
    evidence_ids: list[str],
    scenario: str,
    reasoning: HighestLeverageReasoning | None = None,
) -> ReportHighestLeverageFix:
    if reasoning:
        return ReportHighestLeverageFix(
            issue=(reasoning.issue_id or "").replace("_", " ") or None,
            plain_english=reasoning.plain_reason,
            why_this_matters=reasoning.plain_reason,
            expected_score_lift=reasoning.expected_score_lift,
            target_dimensions=list(reasoning.affected_dimensions),
            first_drill_id=reasoning.recommended_first_drill,
            selection_score=reasoning.selection_score,
            evidence_ids=reasoning.supporting_evidence,
        )

    dimension_key = (diagnosis.limiting_dimension or "Command").lower()
    rule = FIX_RULES.get(dimension_key, FIX_RULES["command"])
    severity = _severity_value(diagnosis.severity)
    confidence = max(psychological_inference.overall_inference_confidence, 0.35)
    scenario_relevance = 1.0
    score = severity * rule.authority_impact * rule.trainability * confidence * scenario_relevance
    expected_lift = "high" if score >= 0.55 else "medium" if score >= 0.28 else "low"
    return ReportHighestLeverageFix(
        issue=rule.issue,
        plain_english=rule.plain_english,
        why_this_matters=rule.why_this_matters,
        expected_score_lift=expected_lift,  # type: ignore[arg-type]
        target_dimensions=list(rule.target_dimensions),
        first_drill_id=rule.drill_id,
        selection_score=round(score, 3),
        evidence_ids=evidence_ids,
    )


def _training(fix: ReportHighestLeverageFix) -> ReportTrainingPrescription:
    drill = DRILLS.get(fix.first_drill_id or "", DRILLS["drop_the_landing_v1"])
    return ReportTrainingPrescription(
        drill_id=fix.first_drill_id,
        title=drill["title"],
        why_chosen=f"Chosen because the dominant deterministic fix is {fix.issue}.",
        instructions=list(drill["instructions"]),
        target_metrics=list(drill["target_metrics"]),
        success_signal=drill["success_signal"],
        evidence_ids=fix.evidence_ids,
    )


def _retest(fix: ReportHighestLeverageFix, duration_ms: int) -> ReportRetestPlan:
    rule = next((item for item in FIX_RULES.values() if item.drill_id == fix.first_drill_id), FIX_RULES["command"])
    days = 3 if duration_ms >= 25000 else 1
    return ReportRetestPlan(
        recommended_retest_after_days=days,
        focus_metric=rule.compare_metrics[0],
        compare_metrics=list(rule.compare_metrics),
        same_prompt_recommended=True,
        success_definition=f"Improvement means cleaner {fix.issue} signals across the same prompt.",
        evidence_ids=fix.evidence_ids,
    )


def _share_card(
    scores: Scores,
    authority_type: ReportAuthorityType,
    mirror: ReportMirror,
    diagnosis: ReportDiagnosis,
) -> ReportShareCard:
    percentile_label = None
    if scores.score_confidence is not None and scores.score_confidence >= 0.6:
        if scores.authority_percentile_estimate is not None:
            percentile = int(round(scores.authority_percentile_estimate * 100))
            percentile_label = f"Top {max(1, 100 - percentile)}% for vocal authority"

    top_strength = diagnosis.strongest_dimension
    growth_area = diagnosis.limiting_dimension
    return ReportShareCard(
        authority_score=scores.authority_score,
        authority_type=authority_type.label,
        top_strength=top_strength,
        growth_area=growth_area,
        one_line_identity_read=mirror.identity_read,
        percentile_label=percentile_label,
        share_safety="public_safe",
        hidden_private_findings=[],
    )


def _appendix(
    metrics: Metrics,
    scores: Scores,
    audio_quality: AudioQuality,
    evidence_ids: list[str],
) -> ReportTechnicalAppendix:
    return ReportTechnicalAppendix(
        metrics=metrics.model_dump(),
        audio_quality_warnings=audio_quality.quality_warnings,
        score_components=scores.score_components.model_dump(),
        evidence_ids=evidence_ids,
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
) -> AuthorityReport:
    """Assemble the deterministic report object from existing backend facts."""
    report_confidence = min(
        max(psychological_inference.overall_inference_confidence, scores.score_confidence or 0.0),
        0.95,
    )
    primary_diagnosis = diagnostic_reasoning.primary_diagnosis
    evidence_ids = (
        list(primary_diagnosis.supporting_evidence_ids)
        if primary_diagnosis and primary_diagnosis.supporting_evidence_ids
        else _primary_evidence_ids(psychological_inference, evidence)
    )
    if not audio_quality.usable and report_confidence < 0.4:
        evidence_ids = evidence_ids[:2]

    dims = _ordered_dimensions(scores)
    strongest = DIMENSION_LABELS[dims[0][0]]
    limiter = DIMENSION_LABELS[sorted(_dimension_scores(scores).items(), key=lambda item: item[1])[0][0]]
    authority_type = _authority_type(scores, psychological_inference, evidence_ids)
    confidence_label = _confidence_label(report_confidence)
    mirror = _mirror(scores, authority_type, strongest, limiter, confidence_label, evidence_ids)
    diagnosis = _diagnosis(
        scores,
        psychological_inference,
        evidence_ids,
        diagnostic=primary_diagnosis,
    )
    perception_map = _perception_map(scores, diagnosis, authority_type, evidence_ids, report_confidence)
    hidden_cost = _hidden_cost(
        diagnosis,
        evidence_ids,
        report_confidence,
        reasoning=diagnostic_reasoning.hidden_cost_reasoning,
    )
    fix = _fix(
        diagnosis,
        psychological_inference,
        evidence_ids,
        scenario,
        reasoning=diagnostic_reasoning.highest_leverage_reasoning,
    )
    training = _training(fix)
    retest = _retest(fix, duration_ms)
    share_card = _share_card(scores, authority_type, mirror, diagnosis)
    appendix = _appendix(metrics, scores, audio_quality, evidence_ids)

    report_uncertainty = Uncertainty(
        overall_confidence_label=confidence_label,  # type: ignore[arg-type]
        suppressed_traits=psychological_inference.suppressed_traits,
            reasons=list(
                dict.fromkeys(
                    uncertainty.reasons
                    + psychological_inference.uncertainty.reasons
                    + diagnostic_reasoning.uncertainty.reasons
                )
            ),
    )

    if duration_ms and duration_ms < 25000:
        report_uncertainty.reasons.append("Short recording limits full report confidence")

    return AuthorityReport(
        mirror=mirror,
        diagnosis=diagnosis,
        perception_map=perception_map,
        evidence_chain=psychological_inference.evidence_chain,
        timeline=moments,
        hidden_cost=hidden_cost,
        highest_leverage_fix=fix,
        training_prescription=training,
        retest_plan=retest,
        authority_type=authority_type,
        share_card=share_card,
        technical_appendix=appendix,
        diagnostic_reasoning=diagnostic_reasoning,
        primary_diagnosis=diagnostic_reasoning.primary_diagnosis,
        secondary_diagnosis=diagnostic_reasoning.secondary_diagnosis,
        contradictions=diagnostic_reasoning.contradictions,
        hidden_cost_reasoning=diagnostic_reasoning.hidden_cost_reasoning,
        dimension_reasoning=diagnostic_reasoning.dimension_reasoning,
        trait_reasoning=diagnostic_reasoning.trait_reasoning,
        highest_leverage_reasoning=diagnostic_reasoning.highest_leverage_reasoning,
        uncertainty=report_uncertainty,
    )
