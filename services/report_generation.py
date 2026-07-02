"""Milestone 7 deterministic premium report generation."""

from __future__ import annotations

from schemas import (
    AudioQuality,
    AuthorityReport,
    CoachingEngine,
    DiagnosticReasoning,
    EvidenceItem,
    Metrics,
    Moment,
    PsychologicalEvidenceSignal,
    PsychologicalInference,
    ReportAuthorityType,
    ReportDiagnosis,
    ReportDimensionReport,
    ReportEvidenceCard,
    ReportHiddenCost,
    ReportHighestLeverageFix,
    ReportMirror,
    ReportPerceptionMap,
    ReportPerceptionRead,
    ReportRetestPlan,
    ReportShareCard,
    ReportTechnicalAppendix,
    ReportTimelineItem,
    ReportTrainingPrescription,
    ReportValidation,
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

DIMENSION_MEANING = {
    "command": "Measures decisiveness, ownership of pauses, clean endings, directness, and status signalling.",
    "clarity": "Measures intelligibility, verbal directness, filler burden, articulation, and ease of following.",
    "composure": "Measures steadiness under pressure, low disruption, vocal stability, and controlled rhythm.",
    "presence": "Measures attention-holding energy, vocal variation, projection, emphasis, and memorability.",
    "persuasion": "Measures conviction, listener pull, framing, stakes, and vocal influence.",
    "structure": "Measures opening, sequencing, idea control, concision, and closing.",
}

DIMENSION_CONSEQUENCE = {
    "command": "Listeners are likely to feel more led when this dimension is stronger.",
    "clarity": "Listeners are likely to spend less effort following the point when this dimension is stronger.",
    "composure": "Listeners are likely to feel less pressure in the delivery when this dimension is stronger.",
    "presence": "Listeners are likely to remember the point more easily when this dimension is stronger.",
    "persuasion": "Listeners are likely to feel more guided toward a conclusion when this dimension is stronger.",
    "structure": "Listeners are likely to trust the speaker's control of the answer path when this dimension is stronger.",
}

DIMENSION_CUE = {
    "command": "Make the key sentence end cleanly, then hold a short pause.",
    "clarity": "Compress the answer into one point and one proof.",
    "composure": "Pause before the important claim instead of speeding through it.",
    "presence": "Give the most important words more contrast than the surrounding phrase.",
    "persuasion": "Name the claim, the stakes, and the action you want remembered.",
    "structure": "Use a point, proof, close shape.",
}

TECHNICAL_APPENDIX_METRICS = {
    "words_per_minute": ("raw_acoustic", "words_per_minute"),
    "filler_words_per_min": ("linguistic", "filler_words_per_min"),
    "pause_frequency_per_minute": ("vad", "pause_frequency_per_minute"),
    "avg_pause_duration_ms": ("vad", "avg_pause_duration_ms"),
    "longest_pause_ms": ("raw_acoustic", "longest_pause_ms"),
    "pitch_range_semitones": ("raw_acoustic", "f0_range_semitones"),
    "terminal_rising_ratio": ("raw_acoustic", "terminal_rising_ratio"),
    "loudness_variation_db": ("raw_acoustic", "loudness_variation_db"),
    "monotony_index": ("derived", "monotony_index"),
    "structure_score": ("linguistic", "structure_score"),
}


def _dimension_scores(scores: Scores) -> dict[str, int]:
    return scores.dimension_scores.model_dump()


def _ordered_dimensions(scores: Scores) -> list[tuple[str, int]]:
    return sorted(_dimension_scores(scores).items(), key=lambda item: item[1], reverse=True)


def _confidence_label(confidence: float | None) -> str:
    value = confidence or 0.0
    if value >= 0.8:
        return "high"
    if value >= 0.6:
        return "medium_high"
    if value >= 0.4:
        return "medium"
    return "low"


def _confidence_phrase(confidence: float) -> str:
    if confidence >= 0.8:
        return "strongly suggests"
    if confidence >= 0.6:
        return "often suggests"
    return "may suggest"


def _severity(score: int) -> str:
    if score < 45:
        return "high"
    if score < 60:
        return "medium"
    return "low"


def _authority_type(scores: Scores, evidence_ids: list[str], confidence: float) -> ReportAuthorityType:
    dims = _dimension_scores(scores)
    top = [name for name, _ in _ordered_dimensions(scores)[:2]]
    low = [name for name, _ in sorted(dims.items(), key=lambda item: item[1])[:2]]
    axes = scores.derived_axes

    def high(*names: str, threshold: int) -> bool:
        return all(dims[name] >= threshold for name in names)

    type_id = "developing_voice"
    label = "Developing Voice"
    description = "This recording suggests a foundation is present, but no single authority signal dominates yet."

    if high("command", "clarity", "composure", "presence", threshold=82) and scores.authority_score >= 88:
        type_id, label = "executive_presence", "Executive Presence"
        description = "This recording suggests clear, intentional, and self-possessed authority."
    elif high("command", "presence", "composure", threshold=72):
        type_id, label = "natural_leader", "Natural Leader"
        description = "Listeners are likely to hear decisiveness, steadiness, and command of the floor."
    elif dims["clarity"] >= 70 and dims["structure"] >= 70 and dims["presence"] >= 55:
        type_id, label = "trusted_expert", "Trusted Expert"
        description = "Listeners are likely to hear knowledge and reliability first."
    elif axes.nervousness >= 65 or (dims["composure"] < 58 and dims["clarity"] >= 60):
        type_id, label = "rushed_achiever", "Rushed Achiever"
        description = "This recording suggests useful ideas that can sound pressured when delivery accelerates."
    elif dims["clarity"] >= 66 and dims["presence"] < 58:
        type_id, label = "quiet_analyst", "Quiet Analyst"
        description = "Listeners are likely to hear thoughtfulness, with presence and contrast as the growth edge."
    elif dims["structure"] >= 66 and dims["clarity"] >= 66 and dims["command"] < 70:
        type_id, label = "thoughtful_strategist", "Thoughtful Strategist"
        description = "Listeners are likely to hear intelligence and measured thinking, with command as the growth edge."
    elif dims["persuasion"] >= 68 and dims["presence"] >= 68:
        type_id, label = "persuasive_operator", "Persuasive Operator"
        description = "Listeners are likely to hear engagement and influence, with structure stabilising the message."
    elif dims["composure"] >= 70 and dims["presence"] < 65:
        type_id, label = "calm_professional", "Calm Professional"
        description = "Listeners are likely to hear steadiness, with memorability as the opportunity."
    elif dims["composure"] < 50 and dims["command"] < 55:
        type_id, label = "unsettled_speaker", "Unsettled Speaker"
        description = "This recording suggests the ideas may be stronger than the current delivery allows listeners to feel."

    return ReportAuthorityType(
        type_id=type_id,
        label=label,
        description=description,
        top_dimensions=[DIMENSION_LABELS[name] for name in top],
        growth_dimensions=[DIMENSION_LABELS[name] for name in low],
        evidence_ids=evidence_ids,
        confidence=round(confidence, 2),
    )


def _mirror(scores: Scores, authority_type: ReportAuthorityType, strongest: str, limiter: str, confidence_label: str, evidence_ids: list[str]) -> ReportMirror:
    score = scores.authority_score
    prefix = "This recording suggests" if confidence_label in {"low", "medium"} else "You sound"
    if score >= 91:
        headline = "You sound like someone people naturally defer to: clear, intentional, and fully in control."
    elif score >= 81:
        headline = "You sound clear, composed, and easy to trust with the floor."
    elif score >= 67:
        headline = "You sound composed, intelligent, and increasingly authoritative."
    elif score >= 53:
        headline = f"{prefix} capable and {strongest.lower()}, but not yet fully {limiter.lower()}."
    elif score >= 39:
        headline = "This recording suggests you sound thoughtful, but not yet consistently settled."
    else:
        headline = "This recording suggests your delivery may be under-signalling your point."

    identity = f"Listeners are likely to notice your {strongest.lower()}, while {limiter.lower()} shapes the current growth edge."
    return ReportMirror(
        headline=headline,
        identity_read=identity,
        one_line_identity_read=identity,
        core_tension=f"{strongest} constrained by {limiter}",
        emotional_tone=_emotional_tone(scores),
        authority_type=authority_type.label,
        confidence_label=confidence_label,  # type: ignore[arg-type]
        confidence_level=confidence_label,  # type: ignore[arg-type]
        evidence_ids=evidence_ids,
    )


def _emotional_tone(scores: Scores) -> str:
    dims = _dimension_scores(scores)
    if dims["composure"] >= 72 and dims["presence"] >= 65:
        return "calm, engaged, and settled"
    if dims["composure"] >= 68:
        return "calm and measured"
    if dims["presence"] < 55:
        return "thoughtful and restrained"
    if dims["composure"] < 55:
        return "pressured and reactive"
    return "competent with some unevenness"


def _diagnosis(scores: Scores, diagnostic: DiagnosticReasoning, evidence_ids: list[str]) -> ReportDiagnosis:
    dims = _dimension_scores(scores)
    primary = diagnostic.primary_diagnosis
    if primary:
        strength = primary.affected_dimensions[0] if primary.affected_dimensions else DIMENSION_LABELS[_ordered_dimensions(scores)[0][0]]
        limiter = primary.affected_dimensions[-1] if primary.affected_dimensions else DIMENSION_LABELS[sorted(dims.items(), key=lambda item: item[1])[0][0]]
        return ReportDiagnosis(
            strongest_dimension=strength,
            limiting_dimension=limiter,
            primary_strength_dimension=strength,
            primary_limiting_dimension=limiter,
            core_behavioural_pattern=primary.diagnosis_id,
            core_pattern=primary.diagnosis_id,
            social_consequence=_diagnosis_consequence(limiter),
            supporting_evidence_ids=primary.supporting_evidence_ids,
            evidence_ids=primary.supporting_evidence_ids,
            severity=primary.severity,
        )

    strongest = DIMENSION_LABELS[_ordered_dimensions(scores)[0][0]]
    limiter_key = sorted(dims.items(), key=lambda item: item[1])[0][0]
    limiter = DIMENSION_LABELS[limiter_key]
    return ReportDiagnosis(
        strongest_dimension=strongest,
        limiting_dimension=limiter,
        primary_strength_dimension=strongest,
        primary_limiting_dimension=limiter,
        core_behavioural_pattern=f"{strongest.lower()} constrained by {limiter.lower()}",
        core_pattern=f"{strongest.lower()} constrained by {limiter.lower()}",
        social_consequence=_diagnosis_consequence(limiter),
        supporting_evidence_ids=evidence_ids,
        evidence_ids=evidence_ids,
        severity=_severity(dims[limiter_key]),  # type: ignore[arg-type]
    )


def _diagnosis_consequence(limiter: str | None) -> str:
    key = (limiter or "Command").lower()
    return {
        "command": "Listeners may understand the point without fully feeling led by it.",
        "clarity": "Listeners may spend more effort following the answer than weighing the idea.",
        "composure": "Listeners may hear pressure even when the words are correct.",
        "presence": "Listeners may agree in the moment but remember less afterwards.",
        "persuasion": "Listeners may understand the explanation without feeling pulled toward action.",
        "structure": "Listeners may trust the content less when the path feels unclear.",
    }.get(key, "Listeners may need more evidence before the strongest impression lands.")


def _read(label: str, text: str, evidence_ids: list[str], confidence: float) -> ReportPerceptionRead:
    return ReportPerceptionRead(label=label, text=text, evidence_ids=evidence_ids, confidence=round(confidence, 2))


def _perception_map(diagnosis: ReportDiagnosis, authority_type: ReportAuthorityType, confidence: float, evidence_ids: list[str]) -> ReportPerceptionMap:
    limiter = (diagnosis.limiting_dimension or "Command").lower()
    strength = (diagnosis.strongest_dimension or "Clarity").lower()
    phrase = _confidence_phrase(confidence)
    return ReportPerceptionMap(
        first_impression=_read(authority_type.label or "Developing Voice", f"The first impression {phrase} {authority_type.description or 'a mixed but interpretable authority signal'}.", evidence_ids, confidence),
        professional_read=_read(f"{strength.title()} led", f"Professionally, listeners are likely to read the recording through {strength}, with {limiter} shaping the ceiling.", evidence_ids, confidence),
        social_status_read=_read("Status signal", f"This may come across as respectable, with {limiter} limiting automatic deference.", evidence_ids, confidence),
        emotional_read=_read("Emotional signal", f"This recording {phrase} a delivery shaped by {limiter} and supported by {strength}.", evidence_ids, confidence),
        interview_read=_read("Interview signal", "Listeners are likely to reward clear openings, controlled pacing, and clean closing in this recording.", evidence_ids, confidence),
        leadership_read=_read("Leadership signal", "Leadership read is strongest when command, composure, and structure align in the evidence.", evidence_ids, confidence),
        trust_read=_read("Trust signal", "Trust is supported when clarity, structure, and composure point in the same direction.", evidence_ids, confidence),
        persuasion_read=_read("Persuasion signal", "Persuasion strengthens when conviction, structure, and emphasis are aligned.", evidence_ids, confidence),
    )


def _evidence_cards(evidence: list[EvidenceItem], psychological: PsychologicalInference, moments: list[Moment]) -> list[ReportEvidenceCard]:
    cards: list[ReportEvidenceCard] = []
    moment = moments[0] if moments else None
    for item in evidence[:5]:
        cards.append(
            ReportEvidenceCard(
                evidence_id=item.id,
                signal=item.headline,
                what_happened=", ".join(item.signals) if item.signals else item.headline,
                why_it_matters=item.why_it_matters,
                listener_interpretation=f"This may shape perceived {item.trait}.",
                related_dimension=item.trait,
                confidence=0.72 if item.direction == "positive" else 0.68,
                timestamp=[moment.start_ms, moment.end_ms] if moment else None,
            )
        )
    existing = {card.evidence_id for card in cards}
    for signal in psychological.evidence_chain:
        if signal.evidence_id in existing:
            continue
        cards.append(_psychological_evidence_card(signal))
    return cards


def _psychological_evidence_card(signal: PsychologicalEvidenceSignal) -> ReportEvidenceCard:
    related = signal.metric.split(".")[0]
    return ReportEvidenceCard(
        evidence_id=signal.evidence_id,
        signal=signal.metric,
        what_happened=f"{signal.metric} observed as {signal.observed_value}",
        why_it_matters=signal.why_it_matters_psychologically,
        listener_interpretation="This signal contributes to the listener-perception read when supported by other evidence.",
        related_dimension=related,
        confidence=round(min(0.95, max(0.35, signal.weight)), 2),
        timestamp=None,
    )


def _timeline(moments: list[Moment], evidence_ids: list[str]) -> list[ReportTimelineItem]:
    items: list[ReportTimelineItem] = []
    for moment in moments:
        impact_values = list(moment.dimension_impact.values())
        confidence = min(0.9, max(0.45, 0.62 + sum(abs(value) for value in impact_values[:3]) * 0.4))
        items.append(
            ReportTimelineItem(
                moment_id=moment.moment_id,
                type=moment.type,
                headline=moment.headline,
                summary=moment.summary,
                listener_interpretation=_moment_interpretation(moment),
                dimension_impact=moment.dimension_impact,
                confidence=round(confidence, 2),
                start_ms=moment.start_ms,
                end_ms=moment.end_ms,
                evidence_ids=evidence_ids[:3],
                severity=moment.severity,
                preview_visible_free=moment.preview_visible_free,
            )
        )
    return items


def _moment_interpretation(moment: Moment) -> str:
    if moment.type in {"strongest_moment", "decisive_moment", "strong_ending"}:
        return "Listeners are likely to hear this as one of the more authoritative parts of the recording."
    if moment.type in {"confidence_drop", "rushing_moment", "hesitation_cluster", "filler_cluster", "weak_ending"}:
        return "This may come across as a moment where control is less fully signalled."
    if moment.type == "monotone_stretch":
        return "This may come across as lower contrast and less memorable."
    return "This moment is included because existing analysis marked it as report-relevant."


def _dimension_reports(scores: Scores, diagnostic: DiagnosticReasoning, evidence_ids: list[str], confidence: float) -> dict[str, ReportDimensionReport]:
    dims = _dimension_scores(scores)
    reports = {}
    for dimension, score in dims.items():
        reasoning = diagnostic.dimension_reasoning.get(dimension)
        linked = reasoning.supporting_evidence_ids if reasoning and reasoning.supporting_evidence_ids else evidence_ids[:3]
        why = []
        if reasoning:
            why.extend(reasoning.why_score_is_high)
            why.extend(reasoning.why_score_is_low)
            if reasoning.biggest_metric_contributor:
                why.append(reasoning.biggest_metric_contributor)
            if reasoning.biggest_linguistic_contributor:
                why.append(reasoning.biggest_linguistic_contributor)
            if reasoning.biggest_behavioural_contributor:
                why.append(reasoning.biggest_behavioural_contributor)
        detail = scores.dimension_details.get(dimension)
        if detail:
            contributor_notes = detail.positive_contributors + detail.negative_contributors
            if contributor_notes:
                why = contributor_notes[:4]
        label = "strong" if score >= 75 else "developing" if score >= 58 else "limited"
        reports[dimension] = ReportDimensionReport(
            dimension=DIMENSION_LABELS[dimension],
            score=score,
            label=label,
            meaning=DIMENSION_MEANING[dimension],
            why=list(dict.fromkeys(why))[:5],
            listener_consequence=DIMENSION_CONSEQUENCE[dimension],
            one_improvement_cue=DIMENSION_CUE[dimension],
            linked_evidence=linked,
            confidence=round(reasoning.confidence if reasoning else confidence, 2),
        )
    return reports


def _hidden_cost(diagnosis: ReportDiagnosis, diagnostic: DiagnosticReasoning, evidence_ids: list[str], confidence: float) -> ReportHiddenCost:
    reasoning = diagnostic.hidden_cost_reasoning
    if reasoning:
        return ReportHiddenCost(
            dimension=diagnosis.limiting_dimension,
            cost_id=reasoning.cost_id,
            consequence=_hidden_cost_sentence(reasoning.listener_effect),
            evidence_ids=reasoning.evidence_ids,
            confidence=reasoning.confidence,
        )
    return ReportHiddenCost(dimension=diagnosis.limiting_dimension, cost_id="hidden_cost", consequence=_diagnosis_consequence(diagnosis.limiting_dimension), evidence_ids=evidence_ids, confidence=confidence)


def _hidden_cost_sentence(effect: str | None) -> str:
    return {
        "listener_not_fully_led": "The hidden cost is that listeners may understand you and still not feel fully led by you.",
        "less_energy_left_for_persuasion": "The hidden cost is cognitive effort: listeners have less energy left to be persuaded.",
        "listener_feels_the_pressure": "The hidden cost is pressure leakage: the listener can feel pressure even when the words are correct.",
        "point_less_likely_to_stick": "The hidden cost is memorability: the point may not stay with the listener.",
        "listener_less_pulled_to_action": "The hidden cost is movement: explanation may not turn into action.",
        "listener_trusts_control_less": "The hidden cost is authority drift: an unclear path can reduce trust in your control.",
    }.get(effect or "", "The hidden cost is a reduced authority signal in this recording.")


def _expected_score_lift_label(primary, reasoning) -> str | None:
    if reasoning and reasoning.expected_score_lift:
        return reasoning.expected_score_lift
    if not primary:
        return None

    lift = primary.expected_impact.authority_score
    if lift >= 4.0:
        return "high"
    if lift >= 2.0:
        return "medium"
    return "low"


def _highest_leverage_fix(coaching: CoachingEngine | None, diagnostic: DiagnosticReasoning, evidence_ids: list[str]) -> ReportHighestLeverageFix:
    primary = coaching.selected_interventions.primary_drill if coaching else None
    drill = None
    if primary and coaching:
        drill = next((item for item in coaching.drill_library if item.drill_id == primary.drill_id), None)
    reasoning = diagnostic.highest_leverage_reasoning
    return ReportHighestLeverageFix(
        issue=drill.title if drill else (reasoning.issue_id.replace("_", " ") if reasoning and reasoning.issue_id else None),
        plain_english=drill.description if drill else (reasoning.plain_reason if reasoning else None),
        why_this_matters=drill.description if drill else (reasoning.plain_reason if reasoning else None),
        expected_score_lift=_expected_score_lift_label(primary, reasoning),
        target_dimensions=drill.target_dimensions if drill else (reasoning.affected_dimensions if reasoning else []),
        first_drill_id=drill.drill_id if drill else (reasoning.recommended_first_drill if reasoning else None),
        selection_score=primary.score if primary else (reasoning.selection_score if reasoning else 0.0),
        evidence_ids=primary.supporting_evidence_ids if primary else (reasoning.supporting_evidence if reasoning else evidence_ids),
    )


def _training(coaching: CoachingEngine | None, fix: ReportHighestLeverageFix) -> ReportTrainingPrescription:
    primary = coaching.selected_interventions.primary_drill if coaching else None
    drill = None
    if primary and coaching:
        drill = next((item for item in coaching.drill_library if item.drill_id == primary.drill_id), None)
    if drill:
        return ReportTrainingPrescription(
            drill_id=drill.drill_id,
            title=drill.title,
            why_chosen=f"Chosen because {drill.title.lower()} is the highest-supported deterministic intervention for this recording.",
            instructions=[drill.description],
            target_metrics=drill.target_metrics,
            success_signal="Compare the expected improvement model against the next recording.",
            evidence_ids=primary.supporting_evidence_ids if primary else fix.evidence_ids,
        )
    return ReportTrainingPrescription(
        drill_id=fix.first_drill_id,
        title=fix.issue,
        why_chosen=fix.why_this_matters,
        instructions=["Practise the selected focus using the deterministic drill attached to this issue."],
        target_metrics=fix.target_dimensions,
        success_signal="The next recording should show stronger evidence on the target dimensions.",
        evidence_ids=fix.evidence_ids,
    )


def _retest(fix: ReportHighestLeverageFix, duration_ms: int) -> ReportRetestPlan:
    days = 3 if duration_ms >= 25000 else 1
    metrics = fix.target_dimensions[:]
    if fix.issue:
        metrics.append(fix.issue.lower())
    focus_metric = {
        "declarative finality": "terminal_rising_ratio",
        "Drop the Landing": "terminal_rising_ratio",
        "Pace Anchor": "words_per_minute",
        "Emphasis Ladder": "dynamic_emphasis_score",
        "Point, Proof, Close": "structure_score",
    }.get(fix.issue or "", fix.issue)
    return ReportRetestPlan(
        recommended_retest_after_days=days,
        focus_metric=focus_metric,
        compare_metrics=metrics,
        same_prompt_recommended=True,
        success_definition=f"Improvement means stronger {fix.issue or 'target'} evidence on the same prompt.",
        evidence_ids=fix.evidence_ids,
    )


def _technical_appendix(metrics: Metrics, scores: Scores, audio_quality: AudioQuality, evidence_ids: list[str]) -> ReportTechnicalAppendix:
    metric_dump = metrics.model_dump()
    selected = {}
    for label, (section, field) in TECHNICAL_APPENDIX_METRICS.items():
        selected[label] = metric_dump.get(section, {}).get(field)
    score_components = scores.score_components.model_dump()
    score_components["calibration_metadata"] = scores.calibration_metadata.model_dump()
    score_components["fairness_adjustments"] = scores.fairness_adjustments.model_dump()
    score_components["score_band"] = scores.score_band
    score_components["score_rarity_label"] = scores.score_rarity_label
    return ReportTechnicalAppendix(metrics=selected, audio_quality_warnings=audio_quality.quality_warnings, score_components=score_components, evidence_ids=evidence_ids)


def _share_card(scores: Scores, authority_type: ReportAuthorityType, mirror: ReportMirror, diagnosis: ReportDiagnosis) -> ReportShareCard:
    percentile_label = None
    if scores.score_confidence is not None and scores.score_confidence >= 0.6 and scores.authority_percentile_estimate is not None:
        percentile_label = scores.score_rarity_label
        if not percentile_label:
            percentile = int(round(scores.authority_percentile_estimate * 100))
            percentile_label = f"Top {max(1, 100 - percentile)}% for vocal authority"
    return ReportShareCard(
        authority_score=scores.authority_score,
        authority_type=authority_type.label,
        top_strength=diagnosis.strongest_dimension,
        growth_area=diagnosis.limiting_dimension,
        one_line_identity_read=mirror.one_line_identity_read,
        percentile_label=percentile_label,
        share_safety="public_safe",
        hidden_private_findings=[],
    )


def _validate_report(report: AuthorityReport, coaching: CoachingEngine | None) -> ReportValidation:
    evidence_ids = {item.evidence_id for item in report.evidence_chain}
    moment_ids = {item.moment_id for item in report.timeline}
    drill_ids = {item.drill_id for item in coaching.drill_library} if coaching else set()
    referenced_evidence = set()
    for section in (report.mirror, report.hidden_cost, report.highest_leverage_fix, report.training_prescription, report.retest_plan, report.authority_type, report.technical_appendix):
        if section and hasattr(section, "evidence_ids"):
            referenced_evidence.update(section.evidence_ids)
    if report.diagnosis:
        referenced_evidence.update(report.diagnosis.supporting_evidence_ids)
        referenced_evidence.update(report.diagnosis.evidence_ids)
    if report.perception_map:
        for read in report.perception_map.model_dump().values():
            if read:
                referenced_evidence.update(read.get("evidence_ids", []))
    for dimension in report.dimension_reports.values():
        referenced_evidence.update(dimension.linked_evidence)
    for item in report.timeline:
        referenced_evidence.update(item.evidence_ids)
    orphan_links = [item for item in sorted(referenced_evidence) if item not in evidence_ids]
    if report.training_prescription and report.training_prescription.drill_id:
        if drill_ids and report.training_prescription.drill_id not in drill_ids:
            orphan_links.append(report.training_prescription.drill_id)
    return ReportValidation(
        valid=not orphan_links,
        evidence_ids_checked=sorted(evidence_ids),
        moment_ids_checked=sorted(moment_ids),
        drill_ids_checked=sorted(drill_ids),
        orphan_links=orphan_links,
        duplicate_sections=[],
    )


def build_generated_report(
    *,
    scores: Scores,
    metrics: Metrics,
    psychological_inference: PsychologicalInference,
    diagnostic_reasoning: DiagnosticReasoning,
    coaching_engine: CoachingEngine | None,
    evidence: list[EvidenceItem],
    moments: list[Moment],
    uncertainty: Uncertainty,
    audio_quality: AudioQuality,
    duration_ms: int,
    scenario: str,
) -> AuthorityReport:
    del scenario
    confidence = min(max(psychological_inference.overall_inference_confidence, scores.score_confidence or 0.0), 0.95)
    confidence_label = _confidence_label(confidence)
    evidence_cards = _evidence_cards(evidence, psychological_inference, moments)
    evidence_ids = [item.evidence_id for item in evidence_cards[:5]]
    if diagnostic_reasoning.primary_diagnosis and diagnostic_reasoning.primary_diagnosis.supporting_evidence_ids:
        evidence_ids = diagnostic_reasoning.primary_diagnosis.supporting_evidence_ids
    dims = _ordered_dimensions(scores)
    strongest = DIMENSION_LABELS[dims[0][0]]
    limiter = DIMENSION_LABELS[sorted(_dimension_scores(scores).items(), key=lambda item: item[1])[0][0]]
    authority_type = _authority_type(scores, evidence_ids, confidence)
    mirror = _mirror(scores, authority_type, strongest, limiter, confidence_label, evidence_ids)
    diagnosis = _diagnosis(scores, diagnostic_reasoning, evidence_ids)
    perception_map = _perception_map(diagnosis, authority_type, confidence, evidence_ids)
    timeline = _timeline(moments, evidence_ids)
    dimension_reports = _dimension_reports(scores, diagnostic_reasoning, evidence_ids, confidence)
    hidden_cost = _hidden_cost(diagnosis, diagnostic_reasoning, evidence_ids, confidence)
    fix = _highest_leverage_fix(coaching_engine, diagnostic_reasoning, evidence_ids)
    training = _training(coaching_engine, fix)
    retest = _retest(fix, duration_ms)
    appendix = _technical_appendix(metrics, scores, audio_quality, evidence_ids)
    share_card = _share_card(scores, authority_type, mirror, diagnosis)
    report_uncertainty = Uncertainty(
        overall_confidence_label=confidence_label,  # type: ignore[arg-type]
        suppressed_traits=psychological_inference.suppressed_traits,
        reasons=list(dict.fromkeys(uncertainty.reasons + psychological_inference.uncertainty.reasons + diagnostic_reasoning.uncertainty.reasons)),
    )
    if duration_ms and duration_ms < 25000:
        report_uncertainty.reasons.append("Short recording limits full report confidence")
    report = AuthorityReport(
        mirror=mirror,
        diagnosis=diagnosis,
        perception_map=perception_map,
        evidence_chain=evidence_cards,
        timeline=timeline,
        dimension_reports=dimension_reports,
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
        coaching_engine=coaching_engine,
        uncertainty=report_uncertainty,
    )
    return report.model_copy(update={"validation": _validate_report(report, coaching_engine)})
