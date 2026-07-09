"""Milestone 7 deterministic premium report generation."""

from __future__ import annotations

from dataclasses import dataclass

from schemas import (
    AudioQuality,
    AuthorityReport,
    CoachingEngine,
    DiagnosticReasoning,
    EvidenceItem,
    Metrics,
    Moment,
    MomentIntelligence,
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
    ReportScenarioSummary,
    ReportShareCard,
    ReportTechnicalAppendix,
    ReportTimelineItem,
    ReportTrainingPrescription,
    ReportValidation,
    Scores,
    Uncertainty,
)
from services.scenario_profiles import get_scenario_profile, major_weight_changes


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

MAIN_REPORT_RAW_MARKERS = (
    "raw_acoustic.",
    "linguistic.",
    "derived.",
    "rhythm.",
    "vad.",
    "articulation.",
    "words_per_minute",
    "filler_words_per_min",
    "burst_speaking_segments",
    "speed_up_segments",
    "hesitation_cluster_score",
    "rhythm_consistency",
    "terminal_rising_ratio",
    "dynamic_emphasis_score",
    "structure_score",
)


@dataclass(frozen=True)
class EvidenceTemplate:
    id: str
    trait: str
    dimension: str
    direction: str
    signal: str
    what_happened: str
    why_it_matters: str
    listener_interpretation: str
    fix: str
    source_signals: tuple[str, ...]
    rank: float
    min_duration_ms: int = 25000


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


def _soften(text: str, confidence: float, weak_sample: bool) -> str:
    if confidence >= 0.6 and not weak_sample:
        return text
    lowered = text[:1].lower() + text[1:] if text else text
    return f"In this sample, this may suggest {lowered}"


def _plain_metric_label(metric: str) -> str:
    return {
        "raw_acoustic.words_per_minute": "speaking pace",
        "linguistic.filler_words_per_min": "filler load",
        "derived.hesitation_cluster_score": "hesitation clustering",
        "raw_acoustic.avg_pause_ms": "pause length",
        "raw_acoustic.mid_phrase_pause_rate": "mid-phrase pausing",
        "linguistic.closing_strength_score": "closing strength",
        "linguistic.opening_strength_score": "opening strength",
        "raw_acoustic.terminal_rising_ratio": "rising endings",
        "derived.dynamic_emphasis_score": "dynamic emphasis",
        "raw_acoustic.f0_range_semitones": "pitch contrast",
        "raw_acoustic.loudness_variation_db": "energy contrast",
        "derived.monotony_index": "vocal monotony",
        "linguistic.specificity_score": "specificity",
        "linguistic.structure_score": "structure",
        "rhythm.rhythm_consistency": "rhythm stability",
        "rhythm.speed_up_segments": "pace acceleration",
        "rhythm.burst_speaking_segments": "burst speaking",
    }.get(metric, metric.split(".")[-1].replace("_", " "))


def _clean_report_text(text: str) -> str:
    cleaned = text
    metric_labels = {
        "raw_acoustic.words_per_minute": "speaking pace",
        "linguistic.filler_words_per_min": "filler load",
        "derived.hesitation_cluster_score": "hesitation clustering",
        "raw_acoustic.terminal_rising_ratio": "rising endings",
        "derived.dynamic_emphasis_score": "dynamic emphasis",
        "linguistic.structure_score": "answer structure",
        "rhythm.rhythm_consistency": "rhythm stability",
        "rhythm.speed_up_segments": "pace acceleration",
        "rhythm.burst_speaking_segments": "burst speaking",
        "articulation.clarity_proxy": "articulation clarity",
        "linguistic.self_doubt_markers": "self-doubt markers",
    }
    for marker, label in sorted(metric_labels.items(), key=lambda item: len(item[0]), reverse=True):
        cleaned = cleaned.replace(marker, label)
    for prefix in ("raw_acoustic.", "linguistic.", "derived.", "rhythm.", "vad.", "articulation."):
        cleaned = cleaned.replace(prefix, "")
    cleaned = cleaned.replace("behaviour:", "")
    return cleaned.replace("_", " ")


def _num(value) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _severity(score: int) -> str:
    if score < 45:
        return "high"
    if score < 60:
        return "medium"
    return "low"


def _visible_evidence_ids(candidate_ids: list[str], cards: list[ReportEvidenceCard]) -> list[str]:
    visible = {card.evidence_id for card in cards}
    filtered = [evidence_id for evidence_id in candidate_ids if evidence_id in visible]
    return filtered or [card.evidence_id for card in cards[:3]]


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
        linked = [item for item in primary.supporting_evidence_ids if item in evidence_ids] or evidence_ids
        return ReportDiagnosis(
            strongest_dimension=strength,
            limiting_dimension=limiter,
            primary_strength_dimension=strength,
            primary_limiting_dimension=limiter,
            core_behavioural_pattern=primary.diagnosis_id,
            core_pattern=primary.diagnosis_id,
            social_consequence=_diagnosis_consequence(limiter),
            supporting_evidence_ids=linked,
            evidence_ids=linked,
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


def _scenario_read_text(read: ReportPerceptionRead | None, scenario_id: str, emphasis: str) -> ReportPerceptionRead | None:
    if read is None or scenario_id == "benchmark":
        return read
    return read.model_copy(update={"text": f"{read.text} This matters more in {scenario_id.replace('_', ' ')} because {emphasis}."})


def _apply_scenario_perception(perception_map: ReportPerceptionMap, scenario: str) -> ReportPerceptionMap:
    profile = get_scenario_profile(scenario)
    if profile.scenario_id == "benchmark":
        return perception_map
    emphasis = ", ".join(profile.expected_speaking_style[:2])
    updates = {}
    if "interview_read" in profile.report_emphasis:
        updates["interview_read"] = _scenario_read_text(perception_map.interview_read, profile.scenario_id, f"answers are expected to be {emphasis}")
    if "leadership_read" in profile.report_emphasis:
        updates["leadership_read"] = _scenario_read_text(perception_map.leadership_read, profile.scenario_id, f"listeners look for {emphasis} control")
    if "persuasion_read" in profile.report_emphasis:
        updates["persuasion_read"] = _scenario_read_text(perception_map.persuasion_read, profile.scenario_id, "listener pull and trust signals carry extra weight")
    if "trust_read" in profile.report_emphasis:
        updates["trust_read"] = _scenario_read_text(perception_map.trust_read, profile.scenario_id, "trust and ease of following shape the read")
    return perception_map.model_copy(update=updates) if updates else perception_map


def _scenario_summary(scores: Scores, fix: ReportHighestLeverageFix, coaching: CoachingEngine | None, scenario: str) -> ReportScenarioSummary:
    profile = get_scenario_profile(scenario)
    dims = _dimension_scores(scores)
    primary = sorted(profile.primary_dimensions, key=lambda dimension: dims.get(dimension, 0), reverse=True)
    weak = sorted(profile.primary_dimensions + profile.secondary_dimensions, key=lambda dimension: dims.get(dimension, 100))
    coaching_reason = None
    if coaching and coaching.selected_interventions.primary_drill:
        primary_drill = coaching.selected_interventions.primary_drill
        coaching_reason = (
            f"{primary_drill.title} is weighted for {profile.scenario_id.replace('_', ' ')} because it targets "
            f"{', '.join(primary_drill.required_evidence[:2]) or 'scenario-relevant evidence'}."
        )
    return ReportScenarioSummary(
        scenario_id=profile.scenario_id,
        description=profile.description,
        why_dimensions_changed=[
            f"{parts[0]} is {parts[1]} for {profile.scenario_id.replace('_', ' ')}"
            for change in major_weight_changes(profile.scenario_id)
            for parts in [change.split(":")]
        ],
        scenario_expectations=list(profile.expected_speaking_style),
        adjusted_strengths=[DIMENSION_LABELS[dimension] for dimension in primary[:2]],
        adjusted_weaknesses=[DIMENSION_LABELS[dimension] for dimension in weak[:2]],
        highest_leverage_fix=fix.issue,
        coaching_explanation=coaching_reason,
        perception_emphasis=list(profile.report_emphasis),
    )


def _evidence_templates() -> dict[str, EvidenceTemplate]:
    templates = [
        EvidenceTemplate(
            "pace_pressure",
            "composure",
            "Composure",
            "negative",
            "Pace pressure",
            "The delivery compressed important ideas instead of giving them room to land.",
            "When pace feels pressured, listeners can read urgency as loss of control.",
            "Capable, but pushing the point faster than the listener can comfortably absorb it.",
            "Use a one-beat pause before the key claim, then deliver the sentence at an even pace.",
            ("pace_fast", "pace_acceleration", "burst_speaking"),
            0.92,
        ),
        EvidenceTemplate(
            "controlled_pacing",
            "composure",
            "Composure",
            "positive",
            "Controlled pacing",
            "The pace sat in a controlled range and the rhythm stayed easy to follow.",
            "Measured pacing lowers listener effort and makes the speaker sound more settled.",
            "Composed enough for the listener to stay with the idea rather than track the delivery.",
            "Keep the same pace, then add slightly longer pauses before the most important lines.",
            ("pace_controlled", "stable_rhythm"),
            0.78,
        ),
        EvidenceTemplate(
            "filler_burden",
            "clarity",
            "Clarity",
            "negative",
            "Filler burden",
            "Fillers appeared often enough to interrupt the sense of clean thought control.",
            "Repeated fillers make the listener spend attention on searching rather than substance.",
            "Knowledgeable, but momentarily less certain about the wording.",
            "Replace the first filler impulse with silence, then restart the phrase cleanly.",
            ("high_fillers", "very_high_fillers"),
            0.9,
        ),
        EvidenceTemplate(
            "low_filler_control",
            "clarity",
            "Clarity",
            "positive",
            "Low filler control",
            "The recording stayed largely free of filler clutter.",
            "Low filler load helps the listener hear verbal control and confidence.",
            "Clearer and more prepared because the phrasing does not keep asking for repair.",
            "Preserve this by pausing before hard words instead of filling the gap.",
            ("low_fillers",),
            0.72,
        ),
        EvidenceTemplate(
            "hesitation_clustering",
            "composure",
            "Composure",
            "negative",
            "Hesitation clustering",
            "Disruptions concentrated into clusters rather than staying isolated.",
            "Clustered hesitation is more noticeable than the same amount spread across a recording.",
            "The thought process may sound briefly overloaded, even when the content is useful.",
            "Stop after the first disruption, breathe, and restart with the next complete clause.",
            ("hesitation_high", "hesitation_windows"),
            0.88,
        ),
        EvidenceTemplate(
            "pause_ownership",
            "command",
            "Command",
            "positive",
            "Pause ownership",
            "Pauses landed as intentional space rather than loss of wording.",
            "Owned silence gives claims more status and makes the speaker sound less rushed.",
            "Comfortable holding the floor without filling every gap.",
            "Keep pauses at clause endings and avoid breaking the middle of key phrases.",
            ("owned_pauses", "low_mid_phrase_pauses"),
            0.82,
        ),
        EvidenceTemplate(
            "weak_closing",
            "structure",
            "Structure",
            "negative",
            "Weak closing",
            "The ending did not fully preserve the force of the answer.",
            "A weak close can make the final impression feel less decisive than the content deserves.",
            "The listener may understand the point but feel less finality at the end.",
            "End with one short takeaway sentence and let the voice fall to a full stop.",
            ("closing_weak",),
            0.84,
        ),
        EvidenceTemplate(
            "strong_opening",
            "structure",
            "Structure",
            "positive",
            "Strong opening",
            "The opening established the point quickly.",
            "Strong openings frame the listener's first impression before doubts can form.",
            "Prepared and easy to follow from the start.",
            "Keep leading with the answer first, then add proof.",
            ("opening_strong",),
            0.8,
        ),
        EvidenceTemplate(
            "rising_endings",
            "command",
            "Command",
            "negative",
            "Rising endings",
            "Some declarative lines lifted at the end instead of landing as finished claims.",
            "Rising endings can make statements sound like they are asking for permission.",
            "Less fully led, especially when the sentence is meant to be a conclusion.",
            "Drop the final word slightly and hold silence after important statements.",
            ("rising_endings",),
            0.82,
        ),
        EvidenceTemplate(
            "dynamic_emphasis",
            "presence",
            "Presence",
            "positive",
            "Dynamic emphasis",
            "Important words carried more vocal contrast than surrounding material.",
            "Contrast makes the listener remember which ideas matter most.",
            "More engaged and easier to keep listening to.",
            "Use the same contrast on only one or two words per sentence.",
            ("dynamic_emphasis_high", "pitch_variation_healthy", "energy_variation_healthy"),
            0.77,
        ),
        EvidenceTemplate(
            "monotony",
            "presence",
            "Presence",
            "negative",
            "Monotony",
            "The delivery gave too little contrast to the ideas that should stand out.",
            "Flat emphasis weakens memorability even when the words are clear.",
            "Competent, but less memorable than the content could be.",
            "Choose the one word that carries the sentence and give it more pitch or energy contrast.",
            ("dynamic_emphasis_low", "pitch_variation_low", "energy_variation_low"),
            0.8,
        ),
        EvidenceTemplate(
            "low_specificity",
            "persuasion",
            "Persuasion",
            "negative",
            "Low specificity",
            "The answer leaned more on general claims than concrete proof.",
            "Specific evidence makes confidence feel earned rather than merely asserted.",
            "Plausible, but not yet grounded enough to create strong belief.",
            "Add one named example, number, or observable detail after the main claim.",
            ("specificity_low", "concreteness_low"),
            0.74,
        ),
        EvidenceTemplate(
            "strong_specificity",
            "persuasion",
            "Persuasion",
            "positive",
            "Strong specificity",
            "The answer gave the listener concrete details to hold onto.",
            "Specificity turns a claim into something that feels more credible.",
            "Grounded and easier to trust.",
            "Keep pairing each main point with one concrete proof point.",
            ("specificity_high", "concreteness_high"),
            0.72,
        ),
        EvidenceTemplate(
            "weak_structure",
            "structure",
            "Structure",
            "negative",
            "Weak structure",
            "The answer path did not feel fully controlled.",
            "Loose structure creates authority drift because listeners have to infer the route themselves.",
            "Thoughtful, but harder to follow than it needs to be.",
            "Use point, proof, close: one claim, one example, one final sentence.",
            ("structure_low", "rambling_high", "repetition_high"),
            0.86,
        ),
        EvidenceTemplate(
            "strong_structure",
            "structure",
            "Structure",
            "positive",
            "Strong structure",
            "The answer gave the listener a clear path through the idea.",
            "Structure reduces cognitive effort and increases trust in the speaker's control.",
            "Organised, prepared, and easier to believe.",
            "Keep the sequence visible: answer first, proof second, close cleanly.",
            ("structure_high", "rambling_low"),
            0.76,
        ),
    ]
    return {template.id: template for template in templates}


def _signal_is_active(signal: PsychologicalEvidenceSignal) -> bool:
    value = signal.observed_value
    if value is None:
        return False
    signal_id = signal.evidence_id.removeprefix("psi_ev_")
    numeric = _num(value)
    return {
        "high_fillers": numeric >= 8,
        "very_high_fillers": numeric >= 12,
        "low_fillers": numeric <= 3,
        "pace_fast": numeric >= 175,
        "pace_controlled": 115 <= numeric <= 165,
        "pace_slow": 0 < numeric <= 95,
        "pace_acceleration": numeric >= 1,
        "burst_speaking": numeric >= 1,
        "stable_rhythm": numeric >= 0.70,
        "unstable_rhythm": numeric <= 0.45,
        "hesitation_high": numeric >= 0.55,
        "hesitation_low": numeric <= 0.25,
        "hesitation_windows": numeric >= 1,
        "owned_pauses": 250 <= numeric <= 800,
        "mid_phrase_pauses": numeric >= 0.35,
        "low_mid_phrase_pauses": numeric <= 0.25,
        "falling_endings": numeric >= 0.35,
        "rising_endings": numeric >= 0.45,
        "dynamic_emphasis_high": numeric >= 0.60,
        "dynamic_emphasis_low": numeric <= 0.30,
        "pitch_variation_low": 0 < numeric <= 3.5,
        "pitch_variation_healthy": numeric >= 5,
        "energy_variation_low": 0 <= numeric <= 3.5,
        "energy_variation_healthy": numeric >= 4.5,
        "opening_strong": numeric >= 0.70,
        "opening_weak": numeric <= 0.45,
        "closing_strong": numeric >= 0.70,
        "closing_weak": numeric <= 0.50,
        "structure_high": numeric >= 0.70,
        "structure_low": numeric <= 0.45,
        "specificity_high": numeric >= 0.55,
        "specificity_low": numeric <= 0.30,
        "concreteness_high": numeric >= 0.45,
        "concreteness_low": numeric <= 0.25,
        "rambling_high": numeric >= 0.45,
        "rambling_low": numeric <= 0.25,
        "repetition_high": numeric >= 0.45,
    }.get(signal_id, False)


def _active_signal_map(psychological: PsychologicalInference) -> dict[str, PsychologicalEvidenceSignal]:
    active = {}
    for signal in psychological.evidence_chain:
        signal_id = signal.evidence_id.removeprefix("psi_ev_")
        if _signal_is_active(signal):
            active[signal_id] = signal
    if "pace_controlled" in active:
        active.pop("pace_fast", None)
        active.pop("pace_slow", None)
    if "stable_rhythm" in active:
        active.pop("unstable_rhythm", None)
    if "low_fillers" in active:
        active.pop("high_fillers", None)
        active.pop("very_high_fillers", None)
    if "hesitation_low" in active:
        active.pop("hesitation_high", None)
        active.pop("hesitation_windows", None)
    if "dynamic_emphasis_high" in active:
        active.pop("dynamic_emphasis_low", None)
    if "structure_high" in active:
        active.pop("structure_low", None)
    if "specificity_high" in active:
        active.pop("specificity_low", None)
    return active


def _template_supported(template: EvidenceTemplate, active: dict[str, PsychologicalEvidenceSignal], duration_ms: int, confidence: float, audio_quality: AudioQuality) -> bool:
    if template.direction == "negative" and duration_ms and duration_ms < template.min_duration_ms:
        return False
    if template.direction == "negative" and confidence < 0.45:
        return False
    if template.direction == "negative" and not audio_quality.usable and template.dimension in {"Composure", "Presence", "Command"}:
        return False
    hits = [signal for signal in template.source_signals if signal in active]
    if not hits:
        return False
    if template.id in {"pace_pressure", "hesitation_clustering", "monotony", "weak_structure", "strong_structure", "strong_specificity", "dynamic_emphasis"}:
        return len(hits) >= 2
    return True


def _card_from_template(template: EvidenceTemplate, active: dict[str, PsychologicalEvidenceSignal], confidence: float, weak_sample: bool, moment: Moment | None) -> ReportEvidenceCard:
    source = [active[signal] for signal in template.source_signals if signal in active]
    evidence_id = source[0].evidence_id if source else f"report_ev_{template.id}"
    card_confidence = round(min(0.92, max(0.42, confidence * 0.75 + template.rank * 0.25)), 2)
    timestamp = None
    start_ms = end_ms = None
    if moment and moment.start_ms is not None and moment.end_ms is not None and moment.end_ms > moment.start_ms:
        start_ms = moment.start_ms
        end_ms = moment.end_ms
        timestamp = [start_ms, end_ms]
    return ReportEvidenceCard(
        evidence_id=evidence_id,
        id=template.id,
        trait=template.trait,
        dimension=template.dimension,
        direction=template.direction,  # type: ignore[arg-type]
        signal=template.signal,
        what_happened=_soften(template.what_happened, confidence, weak_sample),
        why_it_matters=f"{template.why_it_matters} Fix: {template.fix}",
        listener_interpretation=template.listener_interpretation,
        related_dimension=template.dimension,
        confidence=card_confidence,
        source_metrics=[_plain_metric_label(signal.metric) for signal in source],
        start_ms=start_ms,
        end_ms=end_ms,
        timestamp=timestamp,
    )


def _rank_evidence_cards(cards: list[ReportEvidenceCard], diagnosis: DiagnosticReasoning, coaching: CoachingEngine | None) -> list[ReportEvidenceCard]:
    if not cards:
        return []
    limiter_dims = set()
    if diagnosis.primary_diagnosis:
        limiter_dims.update(diagnosis.primary_diagnosis.affected_dimensions)
    if diagnosis.highest_leverage_reasoning:
        limiter_dims.update(diagnosis.highest_leverage_reasoning.affected_dimensions)
    drill_evidence = set()
    if coaching and coaching.selected_interventions.primary_drill:
        drill_evidence.update(coaching.selected_interventions.primary_drill.supporting_evidence_ids)

    def score(card: ReportEvidenceCard) -> tuple[float, float]:
        value = card.confidence
        if card.direction == "negative":
            value += 0.12
        if card.related_dimension in limiter_dims or (card.trait or "") in limiter_dims:
            value += 0.1
        if card.evidence_id in drill_evidence:
            value += 0.12
        if card.id in {"controlled_pacing", "low_filler_control", "strong_opening", "dynamic_emphasis", "strong_specificity", "strong_structure", "pause_ownership"}:
            value += 0.04
        return value, card.confidence

    ordered = sorted(cards, key=score, reverse=True)
    selected: list[ReportEvidenceCard] = []
    families: set[str] = set()
    dimensions: set[str] = set()
    for card in ordered:
        family = {
            "pace_pressure": "pace",
            "controlled_pacing": "pace",
            "filler_burden": "filler",
            "low_filler_control": "filler",
            "hesitation_clustering": "hesitation",
            "pause_ownership": "pause",
            "weak_closing": "closing",
            "strong_opening": "opening",
            "rising_endings": "ending",
            "dynamic_emphasis": "emphasis",
            "monotony": "emphasis",
            "low_specificity": "specificity",
            "strong_specificity": "specificity",
            "weak_structure": "structure",
            "strong_structure": "structure",
        }.get(card.id or card.signal, card.signal)
        if family in families:
            continue
        if len(selected) >= 3 and card.related_dimension in dimensions:
            continue
        selected.append(card)
        families.add(family)
        dimensions.add(card.related_dimension)
        if len(selected) == 5:
            break
    if len(selected) < 3:
        for card in ordered:
            if card not in selected:
                selected.append(card)
            if len(selected) == min(3, len(ordered)):
                break
    has_positive = any(card.direction == "positive" for card in selected)
    has_negative = any(card.direction == "negative" for card in selected)
    if not has_positive:
        positive = next((card for card in ordered if card.direction == "positive" and card not in selected), None)
        if positive:
            selected[-1:] = [positive]
    if not has_negative:
        negative = next((card for card in ordered if card.direction == "negative" and card not in selected), None)
        if negative and len(selected) >= 3:
            selected[-1:] = [negative]
    return selected[:5]


def _fallback_evidence_cards(evidence: list[EvidenceItem], confidence: float, weak_sample: bool, moment: Moment | None) -> list[ReportEvidenceCard]:
    cards = []
    for item in evidence[:3]:
        timestamp = [moment.start_ms, moment.end_ms] if moment and moment.end_ms > moment.start_ms else None
        cards.append(
            ReportEvidenceCard(
                evidence_id=item.id,
                id=item.id,
                trait=item.trait,
                dimension=DIMENSION_LABELS.get(item.trait, item.trait.title()),
                direction=item.direction if item.direction in {"positive", "negative"} else "neutral",  # type: ignore[arg-type]
                signal=item.headline,
                what_happened=_soften(item.headline, confidence, weak_sample),
                why_it_matters=f"{item.why_it_matters} Fix: {DIMENSION_CUE.get(item.trait, 'Retest with one clear behaviour change.')}",
                listener_interpretation=f"This may shape perceived {item.trait}.",
                related_dimension=DIMENSION_LABELS.get(item.trait, item.trait.title()),
                confidence=round(0.6 if item.direction == "positive" else 0.55, 2),
                source_metrics=[],
                start_ms=timestamp[0] if timestamp else None,
                end_ms=timestamp[1] if timestamp else None,
                timestamp=timestamp,
            )
        )
    return cards


def _evidence_cards(
    evidence: list[EvidenceItem],
    psychological: PsychologicalInference,
    diagnostic: DiagnosticReasoning,
    coaching: CoachingEngine | None,
    moments: list[Moment],
    confidence: float,
    duration_ms: int,
    audio_quality: AudioQuality,
) -> list[ReportEvidenceCard]:
    weak_sample = bool(duration_ms and duration_ms < 25000) or confidence < 0.45 or not audio_quality.usable
    active = _active_signal_map(psychological)
    moment = moments[0] if moments else None
    cards = [
        _card_from_template(template, active, confidence, weak_sample, moment)
        for template in _evidence_templates().values()
        if _template_supported(template, active, duration_ms, confidence, audio_quality)
    ]
    ranked = _rank_evidence_cards(cards, diagnostic, coaching)
    if ranked:
        return ranked
    return _fallback_evidence_cards(evidence, confidence, weak_sample, moment)


def _timeline(moments: list[Moment], evidence_ids: list[str]) -> list[ReportTimelineItem]:
    items: list[ReportTimelineItem] = []
    for moment in moments:
        impact_values = list(moment.dimension_impact.values())
        confidence = moment.confidence or min(0.9, max(0.45, 0.62 + sum(abs(value) for value in impact_values[:3]) * 0.4))
        moment_evidence = [item for item in moment.supporting_evidence_ids if item in evidence_ids] or evidence_ids[:3]
        items.append(
            ReportTimelineItem(
                moment_id=moment.moment_id,
                type=moment.type,
                priority=moment.priority,
                headline=moment.headline,
                summary=moment.summary,
                listener_interpretation=moment.listener_interpretation or _moment_interpretation(moment),
                why_it_matters=moment.why_it_matters,
                dimension_impact=moment.dimension_impact,
                confidence=round(confidence, 2),
                start_ms=moment.start_ms,
                end_ms=moment.end_ms,
                evidence_ids=moment_evidence,
                supporting_metrics=moment.supporting_metrics,
                transcript_span=moment.transcript_span,
                word_ids=moment.word_ids,
                scenario_relevance=moment.scenario_relevance,
                coaching_relevance=moment.coaching_relevance,
                importance_score=moment.importance_score,
                moment_group=_moment_group(moment.type),
                severity=moment.severity,
                preview_visible_free=moment.preview_visible_free,
            )
        )
    return items


def _moment_group(moment_type: str) -> str:
    if moment_type in {"strongest_moment", "high_presence_moment", "pause_ownership_moment", "strong_opening", "strong_closing", "best_sentence", "most_commanding_moment", "most_composed_moment", "most_persuasive_moment"}:
        return "authority_peak"
    if moment_type in {"confidence_drop", "weakest_moment", "rushing_moment", "filler_cluster", "hesitation_cluster", "monotone_stretch", "weak_opening", "weak_closing", "most_costly_sentence", "most_unstable_section"}:
        return "attention_leak"
    if moment_type in {"confidence_recovery", "most_improved_section"}:
        return "recovery"
    return "timeline_evidence"


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
        linked = [item for item in linked if item in evidence_ids] or evidence_ids[:3]
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
        why = [_clean_report_text(item) for item in why]
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
        linked = [item for item in reasoning.evidence_ids if item in evidence_ids] or evidence_ids
        return ReportHiddenCost(
            dimension=diagnosis.limiting_dimension,
            cost_id=reasoning.cost_id,
            consequence=_hidden_cost_sentence(reasoning.listener_effect),
            evidence_ids=linked,
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
    linked = primary.supporting_evidence_ids if primary else (reasoning.supporting_evidence if reasoning else evidence_ids)
    linked = [item for item in linked if item in evidence_ids] or evidence_ids
    issue = drill.title if drill else (reasoning.issue_id.replace("_", " ") if reasoning and reasoning.issue_id else "Practice focus")
    plain = drill.description if drill else (reasoning.plain_reason if reasoning else "Use the clearest supported practice focus from this recording.")
    duration = drill.estimated_duration_min if drill else None
    return ReportHighestLeverageFix(
        issue=issue,
        plain_english=plain,
        why_this_matters=f"This is the fastest useful fix because it changes how the listener reads {', '.join(drill.target_dimensions if drill else (reasoning.affected_dimensions if reasoning else [])) or 'the limiting signal'}.",
        expected_score_lift=_expected_score_lift_label(primary, reasoning),
        target_dimensions=drill.target_dimensions if drill else (reasoning.affected_dimensions if reasoning else []),
        first_drill_id=drill.drill_id if drill else (reasoning.recommended_first_drill if reasoning else None),
        action_step=drill.description if drill else plain,
        success_signal=f"The next recording should sound more controlled in {', '.join(drill.target_dimensions if drill else (reasoning.affected_dimensions if reasoning else [])) or 'the target area'}.",
        duration_min=duration,
        selection_score=primary.score if primary else (reasoning.selection_score if reasoning else 0.0),
        evidence_ids=linked,
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
            why_chosen=f"Chosen because this recording's strongest coachable listener-cost points to {', '.join(drill.target_dimensions)}.",
            instructions=[drill.description],
            target_metrics=[_plain_metric_label(metric) for metric in drill.target_metrics],
            target_dimensions=drill.target_dimensions,
            action_step=drill.description,
            expected_score_lift=fix.expected_score_lift,
            duration_min=drill.estimated_duration_min,
            success_signal=fix.success_signal or "The next recording should make the target behaviour easier for a listener to hear.",
            evidence_ids=[item for item in (primary.supporting_evidence_ids if primary else fix.evidence_ids) if item in fix.evidence_ids] or fix.evidence_ids,
        )
    return ReportTrainingPrescription(
        drill_id=fix.first_drill_id,
        title=fix.issue,
        why_chosen=fix.why_this_matters,
        instructions=[fix.action_step or "Practice the target behaviour once, then retest on the same prompt."],
        target_metrics=fix.target_dimensions,
        target_dimensions=fix.target_dimensions,
        action_step=fix.action_step,
        expected_score_lift=fix.expected_score_lift,
        duration_min=fix.duration_min,
        success_signal=fix.success_signal or "The next recording should make the target behaviour easier for a listener to hear.",
        evidence_ids=fix.evidence_ids,
    )


def _retest(fix: ReportHighestLeverageFix, duration_ms: int) -> ReportRetestPlan:
    days = 3 if duration_ms >= 25000 else 1
    metrics = fix.target_dimensions[:]
    if fix.issue:
        metrics.append(fix.issue.lower())
    focus_metric = {
        "declarative finality": "cleaner final endings",
        "Drop the Landing": "cleaner final endings",
        "Pace Anchor": "steadier speaking pace",
        "Emphasis Ladder": "stronger dynamic emphasis",
        "Point, Proof, Close": "clearer answer structure",
    }.get(fix.issue or "", fix.issue)
    return ReportRetestPlan(
        recommended_retest_after_days=days,
        focus_metric=focus_metric,
        compare_metrics=[metric.replace("_", " ") for metric in metrics],
        same_prompt_recommended=True,
        success_definition=fix.success_signal or f"Improvement means the same prompt lands with stronger {fix.issue or 'target'} evidence.",
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
    score_components["scenario_adjustments"] = scores.scenario_adjustments.model_dump()
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
    moment_intelligence: MomentIntelligence | None = None,
) -> AuthorityReport:
    profile = get_scenario_profile(scenario)
    confidence = min(max(psychological_inference.overall_inference_confidence, scores.score_confidence or 0.0), 0.95)
    confidence_label = _confidence_label(confidence)
    evidence_cards = _evidence_cards(
        evidence,
        psychological_inference,
        diagnostic_reasoning,
        coaching_engine,
        moments,
        confidence,
        duration_ms,
        audio_quality,
    )
    evidence_ids = [item.evidence_id for item in evidence_cards[:5]]
    if diagnostic_reasoning.primary_diagnosis and diagnostic_reasoning.primary_diagnosis.supporting_evidence_ids:
        evidence_ids = _visible_evidence_ids(diagnostic_reasoning.primary_diagnosis.supporting_evidence_ids, evidence_cards)
    dims = _ordered_dimensions(scores)
    strongest = DIMENSION_LABELS[dims[0][0]]
    limiter = DIMENSION_LABELS[sorted(_dimension_scores(scores).items(), key=lambda item: item[1])[0][0]]
    authority_type = _authority_type(scores, evidence_ids, confidence)
    mirror = _mirror(scores, authority_type, strongest, limiter, confidence_label, evidence_ids)
    diagnosis = _diagnosis(scores, diagnostic_reasoning, evidence_ids)
    perception_map = _apply_scenario_perception(_perception_map(diagnosis, authority_type, confidence, evidence_ids), profile.scenario_id)
    weak_sample = bool(duration_ms and duration_ms < 25000) or confidence < 0.45 or not audio_quality.usable
    if weak_sample:
        limited_text = "This sample does not contain enough reliable evidence for a strong psychological read. Treat the result as a light directional signal and retest with a longer, clearer recording."
        mirror = mirror.model_copy(
            update={
                "headline": "There is not enough reliable evidence for a full authority diagnosis yet.",
                "identity_read": limited_text,
                "one_line_identity_read": limited_text,
                "confidence_label": "low",
                "confidence_level": "low",
            }
        )
        if perception_map.first_impression:
            perception_map = perception_map.model_copy(
                update={
                    "first_impression": perception_map.first_impression.model_copy(
                        update={"text": limited_text, "confidence": min(perception_map.first_impression.confidence, 0.4)}
                    )
                }
            )
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
        moment_intelligence=moment_intelligence or MomentIntelligence(moments=moments),
        dimension_reports=dimension_reports,
        hidden_cost=hidden_cost,
        highest_leverage_fix=fix,
        training_prescription=training,
        retest_plan=retest,
        authority_type=authority_type,
        share_card=share_card,
        technical_appendix=appendix,
        scenario_summary=_scenario_summary(scores, fix, coaching_engine, profile.scenario_id),
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
