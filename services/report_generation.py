"""Milestone 7 deterministic premium report generation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from schemas import (
    AudioQuality,
    AuthorityReport,
    CoachingEngine,
    DiagnosticReasoning,
    DiagnosticDiagnosis,
    HiddenCostReasoning,
    HighestLeverageReasoning,
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
    Transcript,
    TranscriptSegment,
    TranscriptWord,
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


@dataclass(frozen=True)
class BehaviourObservation:
    id: str
    dimension: str
    direction: str
    observed_cue: str
    behaviour: str
    listener_interpretation: str
    consequence: str
    fix: str
    evidence_id: str
    confidence: float
    source_metrics: tuple[str, ...] = ()
    start_ms: int | None = None
    end_ms: int | None = None
    impact_weight: float = 0.0
    expected_leverage: float = 0.0


@dataclass(frozen=True)
class BehaviourDiagnosis:
    id: str
    label_internal: str
    user_facing_title: str
    one_sentence_pattern: str
    observed_behaviour: str
    listener_interpretation: str
    social_consequence: str
    primary_dimension: str
    secondary_dimensions: tuple[str, ...]
    confidence: float
    supporting_observations: tuple[BehaviourObservation, ...]
    contradicting_observations: tuple[BehaviourObservation, ...]
    evidence_ids: tuple[str, ...]
    moment_ids: tuple[str, ...]
    fix_category: str
    drill_id: str | None
    uncertainty_note: str | None = None


@dataclass(frozen=True)
class RecordingFact:
    fact_id: str
    fact_type: str
    source: str
    start_ms: int | None = None
    end_ms: int | None = None
    transcript_text: str | None = None
    observed_behavior: str = ""
    measurement_summary: str | None = None
    related_dimensions: tuple[str, ...] = ()
    confidence: float = 0.0
    timestamp_source: str = "estimated"
    supporting_metric_ids: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    user_safe: bool = True


@dataclass(frozen=True)
class FactObservation:
    observation_id: str
    title: str
    observed_pattern: str
    supporting_fact_ids: tuple[str, ...]
    listener_effect: str
    related_dimensions: tuple[str, ...]
    confidence: float
    importance: float
    trainability: float
    distinctiveness: float
    contradiction_penalty: float
    target_behavior: str
    recommended_drill_categories: tuple[str, ...]


@dataclass(frozen=True)
class FactDiagnosis:
    diagnosis_id: str
    title: str
    one_sentence_pattern: str
    mechanism: str
    listener_consequence: str
    primary_observation_ids: tuple[str, ...]
    secondary_observation_ids: tuple[str, ...]
    related_dimensions: tuple[str, ...]
    confidence: float
    uncertainty_note: str | None
    target_behavior: str
    recommended_drill_id: str | None
    fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class ListenerState:
    state_id: str
    observation_id: str
    source_fact_ids: tuple[str, ...]
    start_ms: int | None
    current_expectation: str
    current_confidence: str
    processing_load: str
    credibility: str
    certainty: str
    engagement: str
    attention: str
    authority_signal: str
    trust_signal: str
    momentary_confusion: str
    emotional_tension: str
    predicted_next_reaction: str
    perception_shift: str
    confidence: float


@dataclass(frozen=True)
class ListenerPerceptionReconstruction:
    states: tuple[ListenerState, ...]
    primary_state: ListenerState | None
    observation_confidence: float
    diagnosis_confidence: float
    perception_confidence: float
    report_confidence: float
    timeline_confidence: float


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


def _sentence(text: str | None) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if not cleaned:
        return ""
    return cleaned if cleaned.endswith((".", "!", "?")) else f"{cleaned}."


def _lower_first(text: str | None) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    return cleaned[:1].lower() + cleaned[1:]


def _upper_first(text: str | None) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    return cleaned[:1].upper() + cleaned[1:]


def _window_label(start_ms: int | None, end_ms: int | None, duration_ms: int | None = None) -> str:
    if start_ms is None or end_ms is None:
        return "in the recording"
    midpoint = (start_ms + end_ms) / 2
    if duration_ms and duration_ms > 0:
        position = midpoint / duration_ms
        if position < 0.25:
            return "during the opening"
        if position > 0.75:
            return "near the end"
        return "during the middle of the answer"
    return "at the highlighted moment"


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
    replacements = {
        "supported by": "visible in",
        "contradicted by": "complicated by",
        "evidence items": "signs",
        "deterministic": "rule-based",
        "backend": "analysis",
        "metric": "signal",
        "hypothesis": "read",
        "observed as": "heard as",
        "winning diagnosis": "main read",
    }
    for marker, replacement in replacements.items():
        cleaned = cleaned.replace(marker, replacement).replace(marker.title(), replacement.title())
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


def _primary_positive(cards: list[ReportEvidenceCard]) -> ReportEvidenceCard | None:
    return next((card for card in cards if card.direction == "positive"), cards[0] if cards else None)


def _primary_negative(cards: list[ReportEvidenceCard]) -> ReportEvidenceCard | None:
    return next((card for card in cards if card.direction == "negative"), None)


def _card_behaviour(card: ReportEvidenceCard | None) -> str:
    return _lower_first((card.what_happened if card else "") or (card.signal if card else "")).rstrip(".")


def _card_observation(card: ReportEvidenceCard | None) -> str:
    return _lower_first((card.signal if card else "") or (card.what_happened if card else "")).rstrip(".")


def _without_period(text: str | None) -> str:
    return (text or "").strip().rstrip(".")


def _diagnosis_seed(observation: BehaviourObservation) -> tuple[str, str, str, str]:
    observation_id = observation.id or ""
    behaviour = _lower_first(observation.behaviour).rstrip(".")
    dimension = observation.dimension or "Authority"
    if observation_id in {"pace_pressure", "hesitation_clustering"}:
        return (
            "thinks_faster_than_structure",
            "You think faster than the answer can organise itself.",
            f"{behaviour}, so useful ideas can arrive before the listener has a stable path.",
            "Slow the idea down at the moment it starts to compress: pause, name the point, then continue.",
        )
    if observation_id in {"weak_closing", "rising_endings"}:
        return (
            "explains_without_landing",
            "You explain the idea without always landing it.",
            f"{behaviour}, so the listener can understand the point without feeling its finality.",
            "Make the last sentence a takeaway, then stop before adding more.",
        )
    if observation_id in {"low_specificity", "weak_structure"}:
        return (
            "clear_idea_weak_proof",
            "You give the listener the idea before giving them enough proof.",
            f"{behaviour}, so the message can sound plausible before it feels fully evidenced.",
            "Attach one concrete example to the main claim before moving to the next idea.",
        )
    if observation_id == "monotony":
        return (
            "polished_but_flat",
            "You sound controlled, but the emotional contrast is too low.",
            f"{behaviour}, so clear content can become less memorable.",
            "Mark the sentence's most important word with controlled contrast.",
        )
    if observation_id == "filler_burden":
        return (
            "searches_while_speaking",
            "You keep speaking while the wording is still being found.",
            f"{behaviour}, so listeners may hear the search rather than just the point.",
            "Replace the first filler impulse with silence and restart on a complete clause.",
        )
    if observation_id in {"pause_ownership", "controlled_pacing"}:
        return (
            "controlled_when_you_create_space",
            "You sound strongest when you give the thought space.",
            f"{behaviour}, so authority rises when you do less and let the point breathe.",
            "Repeat that controlled space before the most important claim.",
        )
    if observation_id in {"strong_opening", "strong_structure"}:
        return (
            "clear_when_framed_first",
            "You sound clearest when the frame comes before the detail.",
            f"{behaviour}, so the listener trusts you when the path is visible early.",
            "Open with the answer, then give one proof point.",
        )
    if observation_id in {"dynamic_emphasis", "strong_specificity"}:
        return (
            "convincing_when_you_anchor_attention",
            "You become more convincing when you give the listener an anchor.",
            f"{behaviour}, so belief increases when attention has something specific to hold.",
            "Keep one anchor per main claim: a contrast word, example, number, or named detail.",
        )
    return (
        f"behaviour_{(observation_id or dimension).lower()}",
        f"Your main pattern is visible in {dimension.lower()}.",
        f"{behaviour}.",
        "Repeat the strongest observable behaviour and reduce the behaviour that weakens listener trust.",
    )


def _family_for_id(observation_id: str, dimension: str) -> str:
    return {
        "pace_pressure": "pace_control",
        "controlled_pacing": "pace_control",
        "hesitation_clustering": "pace_control",
        "pause_ownership": "pace_control",
        "filler_burden": "verbal_repair",
        "low_filler_control": "verbal_repair",
        "weak_closing": "finality",
        "rising_endings": "finality",
        "strong_opening": "framing",
        "weak_structure": "framing",
        "strong_structure": "framing",
        "low_specificity": "proof",
        "strong_specificity": "proof",
        "monotony": "attention",
        "dynamic_emphasis": "attention",
    }.get(observation_id or "", dimension.lower())


def _card_family(card: ReportEvidenceCard) -> str:
    return _family_for_id(card.id or "", card.related_dimension)


def _observation_family(observation: BehaviourObservation) -> str:
    return _family_for_id(observation.id, observation.dimension)


OBSERVATION_IMPACT_WEIGHTS = {
    "low_filler_control": 0.18,
    "controlled_pacing": 0.38,
    "strong_opening": 0.45,
    "strong_structure": 0.52,
    "dynamic_emphasis": 0.55,
    "strong_specificity": 0.58,
    "filler_burden": 0.48,
    "rising_endings": 0.58,
    "weak_closing": 0.64,
    "monotony": 0.7,
    "pace_pressure": 0.78,
    "weak_structure": 0.84,
    "hesitation_clustering": 0.94,
    "low_specificity": 1.0,
}


OBSERVATION_TRAINABILITY = {
    "low_specificity": 0.92,
    "weak_structure": 0.9,
    "weak_closing": 0.88,
    "pace_pressure": 0.86,
    "hesitation_clustering": 0.84,
    "rising_endings": 0.82,
    "filler_burden": 0.78,
    "monotony": 0.72,
}


def _observation_impact(observation_id: str, confidence: float, source_count: int, has_timestamp: bool) -> float:
    base = OBSERVATION_IMPACT_WEIGHTS.get(observation_id, 0.5)
    evidence_bonus = min(0.08, max(0, source_count - 1) * 0.04)
    timestamp_bonus = 0.04 if has_timestamp else 0.0
    return round(min(1.0, base * (0.72 + confidence * 0.28) + evidence_bonus + timestamp_bonus), 3)


def _observation_leverage(observation_id: str, direction: str, confidence: float, impact: float) -> float:
    if direction != "negative":
        return round(impact * confidence * 0.35, 3)
    trainability = OBSERVATION_TRAINABILITY.get(observation_id, 0.7)
    return round(impact * confidence * trainability, 3)


def _recording_quality_factor(duration_ms: int, audio_quality: AudioQuality, base_confidence: float) -> float:
    factor = max(0.35, min(1.0, base_confidence))
    if duration_ms and duration_ms < 25000:
        factor -= 0.22
    if duration_ms and duration_ms < 12000:
        factor -= 0.12
    if not audio_quality.usable:
        factor -= 0.2
    warning_text = " ".join(audio_quality.quality_warnings).lower()
    if "poor" in warning_text or "low signal" in warning_text:
        factor -= 0.08
    return max(0.25, min(1.0, factor))


def _drill_for_diagnosis(diagnosis_id: str, coaching: CoachingEngine | None) -> str | None:
    if not coaching:
        return None
    category_map = {
        "thinks_faster_than_structure": {"pace_regulation", "pause_ownership", "composure"},
        "explains_without_landing": {"declarative_finality", "closing_strength"},
        "clear_idea_weak_proof": {"specificity", "structure_compression"},
        "polished_but_flat": {"dynamic_emphasis", "presence"},
        "searches_while_speaking": {"filler_reduction", "clarity"},
        "controlled_when_you_create_space": {"pause_ownership", "command"},
        "clear_when_framed_first": {"opening_strength", "structure_compression"},
        "convincing_when_you_anchor_attention": {"dynamic_emphasis", "specificity", "persuasion"},
    }
    preferred = category_map.get(diagnosis_id, set())
    primary = coaching.selected_interventions.primary_drill
    candidates = [primary] if primary else []
    secondary = coaching.selected_interventions.secondary_drill
    if secondary:
        candidates.append(secondary)
    by_id = {item.drill_id: item for item in coaching.drill_library}
    for candidate in candidates:
        drill = by_id.get(candidate.drill_id)
        if drill and drill.category in preferred:
            return drill.drill_id
    for drill in coaching.drill_library:
        if drill.category in preferred:
            return drill.drill_id
    for candidate in candidates:
        drill = by_id.get(candidate.drill_id)
        if drill and set(drill.target_dimensions).intersection(preferred):
            return drill.drill_id
    for drill in coaching.drill_library:
        if set(drill.target_dimensions).intersection(preferred):
            return drill.drill_id
    return primary.drill_id if primary else None


def _fallback_drill_for_observation(observation_id: str | None, coaching: CoachingEngine | None) -> str | None:
    preferred = {
        "strong_structure": "point_proof_close_v1",
        "strong_specificity": "one_point_one_proof_v1",
        "pause_ownership": "pause_ownership_v1",
        "dynamic_emphasis": "emphasis_ladder_v1",
        "controlled_pacing": "pace_anchor_v1",
        "weak_structure": "point_proof_close_v1",
        "low_specificity": "one_point_one_proof_v1",
        "hesitation_clustering": "pressure_reset_v1",
        "pace_pressure": "pace_anchor_v1",
        "weak_closing": "drop_the_landing_v1",
        "rising_endings": "drop_the_landing_v1",
    }.get(observation_id or "")
    if not preferred or not coaching:
        return preferred
    drill_ids = {drill.drill_id for drill in coaching.drill_library}
    return preferred if preferred in drill_ids else None


def _competing_diagnoses(
    observations: list[BehaviourObservation],
    base_confidence: float,
    duration_ms: int,
    audio_quality: AudioQuality,
    coaching: CoachingEngine | None,
) -> list[BehaviourDiagnosis]:
    if not observations:
        return []
    groups: dict[str, list[BehaviourObservation]] = {}
    for observation in observations:
        seed_id, *_ = _diagnosis_seed(observation)
        groups.setdefault(seed_id, []).append(observation)

    diagnoses: list[BehaviourDiagnosis] = []
    all_observations = tuple(observations)
    quality_factor = _recording_quality_factor(duration_ms, audio_quality, base_confidence)
    for seed_id, group in groups.items():
        seed = next((item for item in group if item.direction == "negative"), group[0])
        _, summary, behaviour, coaching_target = _diagnosis_seed(seed)
        families = {_observation_family(item) for item in group}
        dimensions = {item.dimension for item in group}
        support = tuple(
            item for item in all_observations
            if item in group
            or (_observation_family(item) in families and item.direction == seed.direction)
            or (seed.direction == "positive" and item.direction == "positive" and item.dimension in dimensions)
        )
        contradict = tuple(
            item for item in all_observations
            if item not in support
            and item.direction != seed.direction
            and (
                item.dimension in dimensions
                or _observation_family(item) in families
            )
        )
        listener_impact = sum(item.impact_weight * item.confidence for item in support)
        leverage = sum(item.expected_leverage for item in support)
        contradiction_weight = sum(item.impact_weight * item.confidence * 0.9 for item in contradict)
        breadth = min(0.16, (len({item.dimension for item in support}) + len({_observation_family(item) for item in support})) * 0.035)
        first_mention = max((item.impact_weight * item.confidence for item in support), default=0.0)
        raw_score = max(0.0, listener_impact * 0.55 + leverage * 0.25 + first_mention * 0.25 + breadth - contradiction_weight)
        denominator = max(sum(item.impact_weight for item in support) + sum(item.impact_weight for item in contradict) * 0.75, 1.0)
        confidence = round(min(0.94, max(0.18, (raw_score / denominator) * quality_factor + 0.16)), 2)
        secondary_dimensions = tuple(
            item for item in dict.fromkeys(obs.dimension for obs in support if obs.dimension != seed.dimension)
        )
        moment_ids = tuple(
            f"{obs.start_ms}-{obs.end_ms}"
            for obs in support
            if obs.start_ms is not None and obs.end_ms is not None and obs.end_ms > obs.start_ms
        )
        diagnoses.append(
            BehaviourDiagnosis(
                id=seed_id,
                label_internal=seed_id,
                user_facing_title=summary,
                one_sentence_pattern=summary,
                observed_behaviour=behaviour,
                listener_interpretation=_sentence(seed.listener_interpretation),
                social_consequence=_sentence(seed.consequence),
                primary_dimension=seed.dimension,
                secondary_dimensions=secondary_dimensions,
                confidence=confidence,
                supporting_observations=support,
                contradicting_observations=contradict,
                evidence_ids=tuple(item.evidence_id for item in support),
                moment_ids=moment_ids,
                fix_category=coaching_target,
                drill_id=_drill_for_diagnosis(seed_id, coaching),
            )
        )

    ordered = sorted(
        diagnoses,
        key=lambda item: (
            max((obs.impact_weight * obs.confidence for obs in item.supporting_observations), default=0.0),
            sum(obs.expected_leverage for obs in item.supporting_observations),
            item.confidence,
            -sum(obs.impact_weight for obs in item.contradicting_observations),
        ),
        reverse=True,
    )
    if not ordered:
        return []
    top = ordered[0]
    second = ordered[1] if len(ordered) > 1 else None
    margin = round(top.confidence - (second.confidence if second else 0.0), 2)
    uncertainty_note = None
    if second and margin < 0.18:
        uncertainty_note = f"A second plausible read is {_without_period(_lower_first(second.one_sentence_pattern))}."
    adjusted = [
        diagnosis if index else BehaviourDiagnosis(
            id=diagnosis.id,
            label_internal=diagnosis.label_internal,
            user_facing_title=diagnosis.user_facing_title,
            one_sentence_pattern=diagnosis.one_sentence_pattern,
            observed_behaviour=diagnosis.observed_behaviour,
            listener_interpretation=diagnosis.listener_interpretation,
            social_consequence=diagnosis.social_consequence,
            primary_dimension=diagnosis.primary_dimension,
            secondary_dimensions=diagnosis.secondary_dimensions,
            confidence=round(max(0.0, diagnosis.confidence - max(0.0, 0.18 - margin)), 2),
            supporting_observations=diagnosis.supporting_observations,
            contradicting_observations=diagnosis.contradicting_observations,
            evidence_ids=diagnosis.evidence_ids,
            moment_ids=diagnosis.moment_ids,
            fix_category=diagnosis.fix_category,
            drill_id=diagnosis.drill_id,
            uncertainty_note=uncertainty_note,
        )
        for index, diagnosis in enumerate(ordered)
    ]
    return adjusted


def _select_behaviour_diagnosis(
    observations: list[BehaviourObservation],
    confidence: float,
    duration_ms: int,
    audio_quality: AudioQuality,
    coaching: CoachingEngine | None,
) -> BehaviourDiagnosis | None:
    diagnoses = _competing_diagnoses(observations, confidence, duration_ms, audio_quality, coaching)
    if not diagnoses:
        return None
    viable = [
        diagnosis for diagnosis in diagnoses
        if len(diagnosis.supporting_observations) >= 2
        or len({_observation_family(item) for item in diagnosis.supporting_observations}) >= 2
        or len({item.dimension for item in diagnosis.supporting_observations}) >= 2
    ]
    selected = max(
        viable or diagnoses,
        key=lambda item: (
            max((obs.impact_weight * obs.confidence for obs in item.supporting_observations), default=0.0),
            sum(obs.expected_leverage for obs in item.supporting_observations),
            item.confidence,
        ),
    )
    independent_families = {_observation_family(item) for item in selected.supporting_observations}
    independent_dimensions = {item.dimension for item in selected.supporting_observations}
    negative_support = [item for item in selected.supporting_observations if item.direction == "negative"]
    timestamped_support = [
        item for item in selected.supporting_observations
        if item.start_ms is not None and item.end_ms is not None and item.end_ms > item.start_ms
    ]
    reliability_penalty = 0.0 if timestamped_support else 0.06
    contradiction_penalty = min(0.28, sum(item.impact_weight for item in selected.contradicting_observations) * 0.12)
    adjusted_confidence = round(max(0.0, selected.confidence - reliability_penalty - contradiction_penalty), 2)
    has_independent_support = len(selected.supporting_observations) >= 2 and (len(independent_families) >= 2 or len(independent_dimensions) >= 2)
    has_high_value_focus = any(item.impact_weight >= 0.82 and item.confidence >= 0.68 for item in selected.supporting_observations)
    contradiction_load = sum(item.impact_weight for item in selected.contradicting_observations)
    if (
        adjusted_confidence < 0.62
        or not has_independent_support
        or not has_high_value_focus
        or not negative_support
        or contradiction_load >= sum(item.impact_weight for item in selected.supporting_observations) * 0.45
        or (duration_ms and duration_ms < 25000)
        or not audio_quality.usable
    ):
        return None
    return selected if adjusted_confidence == selected.confidence else BehaviourDiagnosis(
        id=selected.id,
        label_internal=selected.label_internal,
        user_facing_title=selected.user_facing_title,
        one_sentence_pattern=selected.one_sentence_pattern,
        observed_behaviour=selected.observed_behaviour,
        listener_interpretation=selected.listener_interpretation,
        social_consequence=selected.social_consequence,
        primary_dimension=selected.primary_dimension,
        secondary_dimensions=selected.secondary_dimensions,
        confidence=adjusted_confidence,
        supporting_observations=selected.supporting_observations,
        contradicting_observations=selected.contradicting_observations,
        evidence_ids=selected.evidence_ids,
        moment_ids=selected.moment_ids,
        fix_category=selected.fix_category,
        drill_id=selected.drill_id,
        uncertainty_note=selected.uncertainty_note,
    )


def _diagnosis_observations(observations: list[BehaviourObservation], diagnosis: BehaviourDiagnosis | None) -> list[BehaviourObservation]:
    if diagnosis is None:
        return observations[:3]
    selected: list[ReportEvidenceCard] = []
    selected_observations: list[BehaviourObservation] = []
    for observation in diagnosis.supporting_observations:
        if observation not in selected_observations:
            selected_observations.append(observation)
    if diagnosis.contradicting_observations and len(selected_observations) < 3:
        counter = max(diagnosis.contradicting_observations, key=lambda item: item.impact_weight)
        if counter.impact_weight >= 0.7:
            selected_observations.append(counter)
    if len(selected_observations) < 3:
        for observation in observations:
            if observation not in selected_observations:
                selected_observations.append(observation)
            if len(selected_observations) >= min(3, len(observations)):
                break
    return sorted(selected_observations, key=lambda item: (item.impact_weight, item.expected_leverage, item.confidence), reverse=True)[:3]


def _diagnosis_with_evidence(diagnosis: BehaviourDiagnosis | None, evidence_cards: list[ReportEvidenceCard]) -> BehaviourDiagnosis | None:
    if diagnosis is None:
        return None
    visible = {card.id for card in evidence_cards}
    support = tuple(obs for obs in diagnosis.supporting_observations if obs.id in visible)
    contradict = tuple(obs for obs in diagnosis.contradicting_observations if obs.id in visible)
    if not support:
        return None
    return BehaviourDiagnosis(
        id=diagnosis.id,
        label_internal=diagnosis.label_internal,
        user_facing_title=diagnosis.user_facing_title,
        one_sentence_pattern=diagnosis.one_sentence_pattern,
        observed_behaviour=diagnosis.observed_behaviour,
        listener_interpretation=diagnosis.listener_interpretation,
        social_consequence=diagnosis.social_consequence,
        primary_dimension=diagnosis.primary_dimension,
        secondary_dimensions=diagnosis.secondary_dimensions,
        confidence=diagnosis.confidence,
        supporting_observations=support,
        contradicting_observations=contradict,
        evidence_ids=tuple(card.evidence_id for card in evidence_cards if card.id in {obs.id for obs in support} or not support),
        moment_ids=diagnosis.moment_ids,
        fix_category=diagnosis.fix_category,
        drill_id=diagnosis.drill_id,
        uncertainty_note=diagnosis.uncertainty_note,
    )
    if len(selected) < 3:
        for card in cards:
            if card not in selected:
                selected.append(card)
            if len(selected) >= min(3, len(cards)):
                break
    return selected[:5]


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


def _mirror(scores: Scores, authority_type: ReportAuthorityType, strongest: str, limiter: str, confidence_label: str, evidence_ids: list[str], evidence_cards: list[ReportEvidenceCard], diagnosis_model: BehaviourDiagnosis | None) -> ReportMirror:
    score = scores.authority_score
    positive = _primary_positive(evidence_cards)
    negative = _primary_negative(evidence_cards)
    pos_behaviour = _card_behaviour(positive)
    neg_observation = _card_observation(negative)
    prefix = "This recording suggests you " if confidence_label in {"low", "medium"} else "You "
    if diagnosis_model:
        diagnosis_prefix = "This recording suggests " if confidence_label in {"low", "medium"} else ""
        headline = f"{diagnosis_prefix}{_without_period(diagnosis_model.one_sentence_pattern)}: {_without_period(_lower_first(diagnosis_model.observed_behaviour))}."
    elif positive and negative:
        headline = f"{prefix}land best when {pos_behaviour}; the drag is that {neg_observation}."
    elif positive:
        headline = f"{prefix}sound most authoritative when {pos_behaviour}."
    elif negative:
        headline = f"{prefix}are currently held back by this behaviour: {neg_observation}."
    elif score >= 81:
        headline = "You sound clear, composed, and easy to trust with the floor."
    elif score >= 53:
        headline = f"{prefix}sound capable, but this sample needs more behavioural evidence to name the exact limiter."
    else:
        headline = "This recording suggests your delivery may be under-signalling your point."

    if diagnosis_model:
        uncertainty_tail = f" {diagnosis_model.uncertainty_note}" if diagnosis_model.uncertainty_note else ""
        identity = f"Listeners are likely to read the recording through one pattern: {_without_period(_lower_first(diagnosis_model.one_sentence_pattern))}. {_without_period(_lower_first(diagnosis_model.observed_behaviour))}.{uncertainty_tail}"
        tension = diagnosis_model.one_sentence_pattern
    elif positive and negative:
        identity = f"Listeners are likely to trust the moments where {pos_behaviour}, while {neg_observation} makes the message feel less fully led."
        tension = f"{positive.related_dimension} shows control; {negative.related_dimension} is the limiting behaviour"
    else:
        identity = f"Listeners are likely to notice your {strongest.lower()}, while {limiter.lower()} shapes the current growth edge."
        tension = f"{strongest} constrained by {limiter}"
    return ReportMirror(
        headline=headline,
        identity_read=identity,
        one_line_identity_read=identity,
        core_tension=tension,
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


def _diagnosis(scores: Scores, diagnostic: DiagnosticReasoning, evidence_ids: list[str], evidence_cards: list[ReportEvidenceCard], diagnosis_model: BehaviourDiagnosis | None) -> ReportDiagnosis:
    dims = _dimension_scores(scores)
    primary = diagnostic.primary_diagnosis
    positive = _primary_positive(evidence_cards)
    negative = _primary_negative(evidence_cards)
    if primary:
        strength = primary.affected_dimensions[0] if primary.affected_dimensions else DIMENSION_LABELS[_ordered_dimensions(scores)[0][0]]
        limiter = primary.affected_dimensions[-1] if primary.affected_dimensions else DIMENSION_LABELS[sorted(dims.items(), key=lambda item: item[1])[0][0]]
        linked = [item for item in primary.supporting_evidence_ids if item in evidence_ids] or evidence_ids
        if diagnosis_model:
            pattern = diagnosis_model.one_sentence_pattern
            consequence = diagnosis_model.social_consequence
            limiter = diagnosis_model.primary_dimension
            linked = [item for item in diagnosis_model.evidence_ids if item in evidence_ids] or linked
        elif positive and negative:
            pattern = f"{_card_observation(positive)}; but {_card_observation(negative)}"
            consequence = f"Listeners may believe the stronger moment, then discount some authority when {_card_behaviour(negative)}"
        elif negative:
            pattern = _card_observation(negative)
            consequence = _diagnosis_consequence(negative.related_dimension)
        elif positive:
            pattern = _card_observation(positive)
            consequence = f"Listeners are likely to trust the recording most when {_card_behaviour(positive)}."
        else:
            pattern = primary.diagnosis_id.replace("_", " ")
            consequence = _diagnosis_consequence(limiter)
        return ReportDiagnosis(
            strongest_dimension=strength,
            limiting_dimension=limiter,
            primary_strength_dimension=strength,
            primary_limiting_dimension=limiter,
            core_behavioural_pattern=pattern,
            core_pattern=pattern,
            social_consequence=consequence,
            supporting_evidence_ids=linked,
            evidence_ids=linked,
            severity=primary.severity,
        )

    strongest = DIMENSION_LABELS[_ordered_dimensions(scores)[0][0]]
    limiter_key = sorted(dims.items(), key=lambda item: item[1])[0][0]
    limiter = DIMENSION_LABELS[limiter_key]
    if diagnosis_model:
        pattern = diagnosis_model.one_sentence_pattern
        consequence = diagnosis_model.social_consequence
        limiter = diagnosis_model.primary_dimension
    elif positive and negative:
        pattern = f"{_card_observation(positive)}; but {_card_observation(negative)}"
        consequence = f"Listeners may trust the controlled parts, then feel less led when {_card_behaviour(negative)}"
    elif negative:
        pattern = _card_observation(negative)
        consequence = _diagnosis_consequence(negative.related_dimension)
    else:
        pattern = f"{strongest.lower()} constrained by {limiter.lower()}"
        consequence = _diagnosis_consequence(limiter)
    return ReportDiagnosis(
        strongest_dimension=strongest,
        limiting_dimension=limiter,
        primary_strength_dimension=strongest,
        primary_limiting_dimension=limiter,
        core_behavioural_pattern=pattern,
        core_pattern=pattern,
        social_consequence=consequence,
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


def _perception_map(diagnosis: ReportDiagnosis, authority_type: ReportAuthorityType, confidence: float, evidence_ids: list[str], evidence_cards: list[ReportEvidenceCard], diagnosis_model: BehaviourDiagnosis | None) -> ReportPerceptionMap:
    positive = _primary_positive(evidence_cards)
    negative = _primary_negative(evidence_cards)
    pos = _card_behaviour(positive) or "the answer gives the listener a clear route"
    neg = _card_observation(negative)
    neg_behaviour = _card_behaviour(negative)
    phrase = _confidence_phrase(confidence)
    first_label = "Behaviour-led read" if not authority_type.label else authority_type.label
    if diagnosis_model:
        uncertainty_tail = f" {diagnosis_model.uncertainty_note}" if diagnosis_model.uncertainty_note else ""
        first_text = f"The first impression {phrase} one dominant behavioural pattern: {_lower_first(diagnosis_model.observed_behaviour)}{uncertainty_tail}"
        professional_text = f"Professionally, listeners are likely to judge the recording through that pattern: {_lower_first(diagnosis_model.one_sentence_pattern)}"
        status_text = f"The status signal depends on whether the listener feels the behaviour as control or leakage: {_lower_first(diagnosis_model.listener_interpretation)}"
        emotional_text = f"The emotional read stays tied to the observable behaviour: {_lower_first(diagnosis_model.observed_behaviour)}"
        interview_text = f"In an interview, this would likely matter because {_lower_first(diagnosis_model.social_consequence)}"
        leadership_text = f"The leadership read follows from the same behaviour: {_lower_first(diagnosis_model.listener_interpretation)}"
        persuasion_text = f"Persuasion improves by changing the behaviour directly: {_clean_report_text(_lower_first(diagnosis_model.fix_category))}"
    elif not negative:
        first_text = f"The first impression {phrase} that you are easiest to trust when {pos}. The report does not have a strong enough negative behaviour to make a sharper limiting claim."
        professional_text = f"Professionally, the clearest read is behavioural: {pos}."
        status_text = "The status signal comes from the controlled behaviour in the evidence, rather than from a strong negative drag."
        emotional_text = f"The emotional read is not fixed personality. In this recording, the listener likely feels steadier when {pos}."
        interview_text = f"In an interview, this would likely be easiest to trust where {pos}."
        leadership_text = f"The leadership signal comes from the moments where {pos}; a sharper limiter needs more negative evidence."
        persuasion_text = "Persuasion should build from the strongest observed behaviour before adding new proof or contrast."
    else:
        first_text = f"The first impression {phrase} that you are easiest to trust when {pos}, but less commanding when {neg}."
        professional_text = f"Professionally, the useful signal is behavioural: {pos}. The ceiling is the moment where {neg_behaviour}."
        status_text = f"The status signal rises when the delivery holds its shape; it dips when {neg_behaviour}, because the listener has to do more interpretive work."
        emotional_text = f"The emotional read is not fixed personality. In this recording, the listener likely feels steadier during the controlled behaviour and more tension around the drag: {neg}."
        interview_text = f"In an interview, this would likely be understood, but the answer would land stronger if the main claim were supported or closed before {neg_behaviour}."
        leadership_text = f"The leadership signal comes from the moments where {pos}; the risk is that {neg_behaviour} makes the listener feel less fully led."
        persuasion_text = f"Persuasion depends on giving belief something to attach to; the next gain is to fix the behaviour where {neg_behaviour}."
    return ReportPerceptionMap(
        first_impression=_read(first_label, first_text, evidence_ids, confidence),
        professional_read=_read("Professional read", professional_text, evidence_ids, confidence),
        social_status_read=_read("Status read", status_text, evidence_ids, confidence),
        emotional_read=_read("Emotional read", emotional_text, evidence_ids, confidence),
        interview_read=_read("Interview read", interview_text, evidence_ids, confidence),
        leadership_read=_read("Leadership read", leadership_text, evidence_ids, confidence),
        trust_read=_read("Trust read", f"Trust comes from the observable control in this sample, especially when {pos}.", evidence_ids, confidence),
        persuasion_read=_read("Persuasion read", persuasion_text, evidence_ids, confidence),
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
            "The recording contains moments of vocal searching that are not necessarily visible in the transcript.",
            "Listeners hear hesitation through timing and restart patterns, not only through written filler words.",
            "Thoughtful, but occasionally pausing while searching for the next idea.",
            "Stop after the first disruption, breathe, and restart with the next complete clause.",
            ("hesitation_high", "hesitation_windows", "acoustic_hesitations"),
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
        "acoustic_hesitations": numeric >= 1,
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
    if "acoustic_hesitations" in active:
        active.pop("low_fillers", None)
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


def _best_moment_for_template(template_id: str, moments: list[Moment]) -> Moment | None:
    type_map = {
        "pace_pressure": {"rushing_moment", "confidence_drop", "most_unstable_section"},
        "controlled_pacing": {"most_composed_moment", "pause_ownership_moment", "strongest_moment"},
        "filler_burden": {"filler_cluster", "confidence_drop", "most_costly_sentence"},
        "low_filler_control": {"strongest_moment", "best_sentence", "most_composed_moment"},
        "hesitation_clustering": {"hesitation_cluster", "confidence_drop", "most_costly_sentence"},
        "pause_ownership": {"pause_ownership_moment", "most_commanding_moment", "most_composed_moment"},
        "weak_closing": {"weak_closing", "weakest_moment"},
        "strong_opening": {"strong_opening", "best_sentence", "strongest_moment"},
        "rising_endings": {"weak_closing", "most_costly_sentence"},
        "dynamic_emphasis": {"high_presence_moment", "most_persuasive_moment", "strongest_moment"},
        "monotony": {"monotone_stretch", "weakest_moment"},
        "low_specificity": {"most_costly_sentence", "weakest_moment"},
        "strong_specificity": {"best_sentence", "most_persuasive_moment", "strongest_moment"},
        "weak_structure": {"weak_opening", "weak_closing", "most_unstable_section"},
        "strong_structure": {"strong_opening", "strong_closing", "best_sentence"},
    }
    preferred = type_map.get(template_id, set())
    return next((moment for moment in moments if moment.type in preferred), moments[0] if moments else None)


def _transcript_reference(moment: Moment | None, confidence: float) -> str:
    if not moment or not moment.transcript_span or confidence < 0.78:
        return ""
    words = moment.transcript_span.split()
    if len(words) < 3 or len(words) > 18:
        return ""
    return " in the highlighted phrase"


def _observation_from_template(
    template: EvidenceTemplate,
    active: dict[str, PsychologicalEvidenceSignal],
    confidence: float,
    weak_sample: bool,
    moment: Moment | None,
    duration_ms: int,
) -> BehaviourObservation:
    source = [active[signal] for signal in template.source_signals if signal in active]
    evidence_id = source[0].evidence_id if source else f"report_ev_{template.id}"
    card_confidence = round(min(0.92, max(0.42, confidence * 0.75 + template.rank * 0.25)), 2)
    start_ms = end_ms = None
    if moment and moment.start_ms is not None and moment.end_ms is not None and moment.end_ms > moment.start_ms:
        start_ms = moment.start_ms
        end_ms = moment.end_ms
    place = _window_label(start_ms, end_ms, duration_ms)
    transcript_ref = _transcript_reference(moment, confidence)

    observed = template.what_happened
    behaviour = template.what_happened
    listener = template.listener_interpretation
    consequence = template.why_it_matters
    fix = template.fix

    if template.id == "pace_pressure":
        observed = f"Your delivery sped up {place}{transcript_ref} instead of holding the same measured pace."
        behaviour = "The main point started to feel compressed just as it needed more space."
        listener = "Listeners may hear that as you trying to keep up with your own thought rather than leading them through it."
        consequence = "The social cost is pressure leakage: the idea can sound less settled than it is."
        fix = "Pause before the key claim, then deliver the next sentence without accelerating."
    elif template.id == "controlled_pacing":
        observed = f"Your pace stayed even {place}, and the phrasing gave the listener enough room to follow."
        behaviour = "The recording shows you can hold the floor without rushing to fill it."
        listener = "Listeners are likely to experience this as calm control rather than effort."
        consequence = "That steadiness helps the listener feel that the moment is not pushing you off balance."
        fix = "Keep this pace and add a slightly longer pause before the sentence you most want remembered."
    elif template.id == "filler_burden":
        observed = f"Fillers clustered {place}, so the wording sounded repaired while the thought was still moving."
        behaviour = "The answer briefly shifted from delivering the point to searching for the next phrase."
        listener = "Listeners may start tracking the search instead of the substance."
        consequence = "The cost is clarity: the idea asks for more effort than it needs."
        fix = "Replace the first filler impulse with silence, then restart on the next complete clause."
    elif template.id == "low_filler_control":
        observed = f"The recording stayed mostly free of filler clutter {place}."
        behaviour = "You left fewer verbal repair marks in the path of the answer."
        listener = "Listeners are likely to read the phrasing as prepared and under control."
        consequence = "That gives clarity more room to land because the listener does not have to filter around repairs."
        fix = "Protect this strength by pausing before difficult wording instead of filling the gap."
    elif template.id == "hesitation_clustering":
        observed = f"You occasionally paused while searching for your next idea {place}."
        behaviour = "The hesitation was audible in timing and restart patterns, even where the transcript may not show a filler word."
        listener = "Listeners may hear a brief search for the next point rather than a fully owned pause."
        consequence = "The cost is composure: the recording can sound more pressured than the content requires."
        fix = "After the first disruption, stop, breathe, and restart with one clean clause."
    elif template.id == "pause_ownership":
        observed = f"A pause landed cleanly {place}, giving the previous idea space instead of breaking it."
        behaviour = "The silence sounded intentional rather than like missing wording."
        listener = "Listeners are likely to feel you are comfortable holding the floor."
        consequence = "That creates command because the listener is not rushed into the next point."
        fix = "Keep placing silence after complete thoughts, not in the middle of the important phrase."
    elif template.id == "weak_closing":
        observed = f"The answer ended {place} without fully reinforcing the message."
        behaviour = "The final point arrived, then the recording moved on before it had a clean landing."
        listener = "Listeners may understand the answer but feel less finality at the end."
        consequence = "The hidden cost is a weaker last impression, which can make the whole answer feel less led."
        fix = "End with one short takeaway sentence, then hold silence after it."
    elif template.id == "strong_opening":
        observed = f"The opening established the point quickly {place}."
        behaviour = "You gave the listener a frame before asking them to process detail."
        listener = "Listeners are likely to read the start as prepared and easy to follow."
        consequence = "That strengthens structure because the answer has a visible route from the first few seconds."
        fix = "Keep leading with the answer first, then add proof."
    elif template.id == "rising_endings":
        observed = f"Some statements lifted at the end {place} instead of landing as finished claims."
        behaviour = "The sentence endings softened the authority of the words."
        listener = "Listeners may hear the line as checking for agreement rather than delivering a position."
        consequence = "The cost is command: the point can sound less final than the content deserves."
        fix = "Drop the final stressed word slightly and hold a half-beat of silence."
    elif template.id == "dynamic_emphasis":
        observed = f"Your voice gave the important words more contrast {place}."
        behaviour = "The delivery marked what mattered instead of treating every phrase equally."
        listener = "Listeners are more likely to remember the point because the emphasis guides their attention."
        consequence = "That supports presence and persuasion by making the message easier to keep listening to."
        fix = "Use this contrast on one or two words per sentence, not every word."
    elif template.id == "monotony":
        observed = f"The delivery flattened {place}, with too little contrast around the ideas that needed emphasis."
        behaviour = "Clear content was delivered with less vocal shape than the point required."
        listener = "Listeners may understand the words but remember less of what mattered."
        consequence = "The cost is presence: the answer can feel less consequential than it is."
        fix = "Choose one word in the next key sentence and give it more pitch or energy contrast."
    elif template.id == "low_specificity":
        observed = "The answer leaned on broad claims without giving the listener many concrete anchors."
        behaviour = "The message explained the idea more than it proved it."
        listener = "Listeners may find the point plausible but not fully evidenced."
        consequence = "The cost is persuasion: belief has fewer details to attach to."
        fix = "Support the main claim with one example, number, named situation, or observable detail before moving on."
    elif template.id == "strong_specificity":
        observed = "The answer gave the listener concrete details to hold onto."
        behaviour = "The claim was supported rather than left as a general assertion."
        listener = "Listeners are likely to experience the message as more grounded."
        consequence = "That strengthens persuasion because confidence feels earned."
        fix = "Keep pairing each main point with one concrete proof point."
    elif template.id == "weak_structure":
        observed = f"The answer path became less controlled {place}; the listener had to infer the route."
        behaviour = "The recording sounded more like thinking through the point than guiding the listener through it."
        listener = "Listeners may trust the content but feel less certain where the answer is going."
        consequence = "The cost is authority drift: unclear sequencing weakens confidence in your control."
        fix = "Use point, proof, close: one claim, one example, one final sentence."
    elif template.id == "strong_structure":
        observed = f"The answer gave the listener a clear route {place}."
        behaviour = "The sequence made the idea easier to follow without extra listener work."
        listener = "Listeners are likely to read this as organised and prepared."
        consequence = "That supports structure because the path is visible, not implied."
        fix = "Keep the sequence visible: answer first, proof second, close cleanly."

    if weak_sample:
        observed = _soften(observed, confidence, weak_sample)

    has_timestamp = start_ms is not None and end_ms is not None and end_ms > start_ms
    impact = _observation_impact(template.id, card_confidence, len(source), has_timestamp)
    leverage = _observation_leverage(template.id, template.direction, card_confidence, impact)

    return BehaviourObservation(
        id=template.id,
        dimension=template.dimension,
        direction=template.direction,
        observed_cue=_sentence(observed),
        behaviour=_sentence(behaviour),
        listener_interpretation=_sentence(listener),
        consequence=_sentence(consequence),
        fix=_sentence(fix),
        evidence_id=evidence_id,
        confidence=card_confidence,
        source_metrics=tuple(_plain_metric_label(signal.metric) for signal in source),
        start_ms=start_ms,
        end_ms=end_ms,
        impact_weight=impact,
        expected_leverage=leverage,
    )


def _card_from_observation(observation: BehaviourObservation) -> ReportEvidenceCard:
    timestamp = None
    if observation.start_ms is not None and observation.end_ms is not None and observation.end_ms > observation.start_ms:
        timestamp = [observation.start_ms, observation.end_ms]
    return ReportEvidenceCard(
        evidence_id=observation.evidence_id,
        id=observation.id,
        trait=observation.dimension.lower(),
        dimension=observation.dimension,
        direction=observation.direction,  # type: ignore[arg-type]
        signal=observation.observed_cue,
        what_happened=observation.behaviour,
        why_it_matters=f"{observation.consequence} Fix: {observation.fix}",
        listener_interpretation=observation.listener_interpretation,
        related_dimension=observation.dimension,
        confidence=observation.confidence,
        source_metrics=list(observation.source_metrics),
        start_ms=observation.start_ms,
        end_ms=observation.end_ms,
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
        impact = OBSERVATION_IMPACT_WEIGHTS.get(card.id or "", 0.5)
        value = card.confidence * 0.35 + impact * 0.65
        if card.direction == "negative":
            value += 0.12
        if card.related_dimension in limiter_dims or (card.trait or "") in limiter_dims:
            value += 0.1
        if card.evidence_id in drill_evidence:
            value += 0.12
        if card.id == "low_specificity":
            value += 0.16
        if card.id in {"controlled_pacing", "low_filler_control", "strong_opening", "dynamic_emphasis", "strong_specificity", "strong_structure", "pause_ownership"}:
            value += 0.04
        return value, impact

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
        if len(selected) == 3:
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
    return selected[:3]


def _rank_observations(observations: list[BehaviourObservation], diagnostic: DiagnosticReasoning, coaching: CoachingEngine | None) -> list[BehaviourObservation]:
    if not observations:
        return []
    merged_by_family: dict[str, BehaviourObservation] = {}
    for observation in observations:
        family = _observation_family(observation)
        existing = merged_by_family.get(family)
        if existing is None or (
            observation.impact_weight,
            observation.expected_leverage,
            observation.confidence,
        ) > (
            existing.impact_weight,
            existing.expected_leverage,
            existing.confidence,
        ):
            merged_by_family[family] = observation
    observations = list(merged_by_family.values())
    cards = [_card_from_observation(observation) for observation in observations]
    ranked_cards = _rank_evidence_cards(cards, diagnostic, coaching)
    by_key = {(observation.id, observation.evidence_id): observation for observation in observations}
    ranked = [
        by_key[(card.id, card.evidence_id)]
        for card in ranked_cards
        if (card.id, card.evidence_id) in by_key
    ]
    for observation in sorted(observations, key=lambda item: (item.impact_weight, item.expected_leverage, item.confidence), reverse=True):
        if observation not in ranked:
            ranked.append(observation)
    return ranked


def _behaviour_observations(
    evidence: list[EvidenceItem],
    psychological: PsychologicalInference,
    diagnostic: DiagnosticReasoning,
    coaching: CoachingEngine | None,
    moments: list[Moment],
    confidence: float,
    duration_ms: int,
    audio_quality: AudioQuality,
) -> list[BehaviourObservation]:
    weak_sample = bool(duration_ms and duration_ms < 25000) or confidence < 0.45 or not audio_quality.usable
    active = _active_signal_map(psychological)
    observations = [
        _observation_from_template(
            template,
            active,
            confidence,
            weak_sample,
            _best_moment_for_template(template.id, moments),
            duration_ms,
        )
        for template in _evidence_templates().values()
        if _template_supported(template, active, duration_ms, confidence, audio_quality)
    ]
    ranked = _rank_observations(observations, diagnostic, coaching)
    if ranked:
        return ranked
    moment = moments[0] if moments else None
    return [
        BehaviourObservation(
            id=item.id,
            dimension=DIMENSION_LABELS.get(item.trait, item.trait.title()),
            direction=item.direction if item.direction in {"positive", "negative"} else "neutral",
            observed_cue=_soften(item.headline, confidence, weak_sample),
            behaviour=_soften(item.headline, confidence, weak_sample),
            listener_interpretation=f"This may shape perceived {item.trait}.",
            consequence=item.why_it_matters,
            fix=DIMENSION_CUE.get(item.trait, "Retest with one clear behaviour change."),
            evidence_id=item.id,
            confidence=round(0.6 if item.direction == "positive" else 0.55, 2),
            start_ms=moment.start_ms if moment and moment.end_ms > moment.start_ms else None,
            end_ms=moment.end_ms if moment and moment.end_ms > moment.start_ms else None,
            impact_weight=0.35,
            expected_leverage=0.12,
        )
        for item in evidence[:3]
    ]


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
    cards = [_card_from_observation(observation) for observation in _behaviour_observations(evidence, psychological, diagnostic, coaching, moments, confidence, duration_ms, audio_quality)]
    ranked = _rank_evidence_cards(cards, diagnostic, coaching)
    if ranked:
        return ranked
    moment = moments[0] if moments else None
    return _fallback_evidence_cards(evidence, confidence, weak_sample, moment)


def _moment_copy(moment: Moment, duration_ms: int) -> tuple[str, str, str, str]:
    place = _window_label(moment.start_ms, moment.end_ms, duration_ms)
    if moment.type == "rushing_moment":
        return (
            "The answer sped up here",
            f"{place}, the delivery compressed the point instead of giving it space.",
            "Listeners may hear this as pressure entering the answer.",
            "This matters because pace changes are easiest to feel when the idea becomes important.",
        )
    if moment.type == "filler_cluster":
        return (
            "The wording needed repair here",
            f"{place}, fillers clustered enough to pull attention away from the point.",
            "Listeners may hear this as searching for wording rather than delivering the idea.",
            "This matters because clustered repairs are more noticeable than isolated fillers.",
        )
    if moment.type == "hesitation_cluster":
        return (
            "The thought briefly lost flow",
            f"{place}, pauses grouped together and made the answer path feel interrupted.",
            "Listeners may experience this as a short loss of control over the point.",
            "This matters because hesitation clusters can make solid content feel less settled.",
        )
    if moment.type in {"weak_closing", "weak_ending"}:
        return (
            "The ending did not fully land",
            "The answer reached its final point without reinforcing the message as a conclusion.",
            "Listeners may understand the point but feel less finality from it.",
            "This matters because endings disproportionately shape what the listener remembers.",
        )
    if moment.type == "monotone_stretch":
        return (
            "The delivery flattened here",
            f"{place}, the voice gave less contrast to the material than the point needed.",
            "Listeners may understand the words but remember less of the emphasis.",
            "This matters because low contrast reduces presence and persuasive pull.",
        )
    if moment.type in {"strong_opening", "best_sentence"}:
        return (
            "The point was clearest here",
            f"{place}, the answer gave the listener a cleaner frame for the idea.",
            "Listeners are likely to hear this as prepared and easy to follow.",
            "This matters because strong local phrasing shows what the speaker can repeat deliberately.",
        )
    if moment.type in {"pause_ownership_moment", "most_commanding_moment"}:
        return (
            "You held the floor here",
            f"{place}, the delivery sounded controlled enough to let the point breathe.",
            "Listeners are likely to read the silence or pacing as intentional.",
            "This matters because owned space creates command without forcing volume.",
        )
    if moment.type in {"high_presence_moment", "most_persuasive_moment"}:
        return (
            "The message had more pull here",
            f"{place}, emphasis and energy made the point easier to notice.",
            "Listeners are more likely to keep attention on this part.",
            "This matters because persuasion needs guided attention, not only clear words.",
        )
    if moment.type in {"strongest_moment", "most_composed_moment"}:
        return (
            "This was the most controlled stretch",
            f"{place}, the delivery showed the cleanest version of the recording's authority signal.",
            "Listeners are likely to hear this as one of the more settled parts of the answer.",
            "This matters because it gives the training plan a concrete behavioural target.",
        )
    if moment.type in {"confidence_drop", "weakest_moment", "most_costly_sentence", "most_unstable_section"}:
        return (
            "Control dipped here",
            f"{place}, the delivery became less settled than the surrounding material.",
            "Listeners may feel the answer working harder than it needs to.",
            "This matters because local drops explain why the overall impression is not just the average score.",
        )
    return (
        moment.headline,
        moment.summary,
        moment.listener_interpretation or _moment_interpretation(moment),
        moment.why_it_matters or "This moment is included because it carries interpretable evidence.",
    )


def _moment_supports_diagnosis(moment: Moment, diagnosis_model: BehaviourDiagnosis | None, evidence_ids: list[str]) -> bool:
    if diagnosis_model is None:
        return True
    if set(moment.supporting_evidence_ids).intersection(evidence_ids):
        return True
    primary = diagnosis_model.id
    type_groups = {
        "thinks_faster_than_structure": {"rushing_moment", "confidence_drop", "hesitation_cluster", "most_unstable_section", "most_composed_moment", "pause_ownership_moment"},
        "explains_without_landing": {"weak_closing", "strong_closing", "most_commanding_moment", "confidence_drop"},
        "clear_idea_weak_proof": {"most_costly_sentence", "best_sentence", "weak_opening", "strong_opening", "most_unstable_section"},
        "polished_but_flat": {"monotone_stretch", "high_presence_moment", "most_persuasive_moment"},
        "searches_while_speaking": {"filler_cluster", "hesitation_cluster", "best_sentence"},
        "controlled_when_you_create_space": {"pause_ownership_moment", "most_composed_moment", "strongest_moment", "rushing_moment"},
        "clear_when_framed_first": {"strong_opening", "best_sentence", "strongest_moment", "weak_closing"},
        "convincing_when_you_anchor_attention": {"most_persuasive_moment", "high_presence_moment", "best_sentence", "strongest_moment"},
    }
    return moment.type in type_groups.get(primary, set())


def _timeline(moments: list[Moment], evidence_ids: list[str], duration_ms: int, diagnosis_model: BehaviourDiagnosis | None) -> list[ReportTimelineItem]:
    items: list[ReportTimelineItem] = []
    if duration_ms and duration_ms < 25000:
        return items
    for moment in moments:
        if not _moment_supports_diagnosis(moment, diagnosis_model, evidence_ids):
            continue
        if moment.start_ms is None or moment.end_ms <= moment.start_ms or (moment.confidence is not None and 0 < moment.confidence < 0.35):
            continue
        if not (moment.headline or "").strip() or not (moment.summary or "").strip():
            continue
        if moment.type in {"generic", "timeline_evidence", "other"}:
            continue
        impact_values = list(moment.dimension_impact.values())
        confidence = moment.confidence or min(0.9, max(0.45, 0.62 + sum(abs(value) for value in impact_values[:3]) * 0.4))
        if moment.timestamp_source in {"interpolated", "estimated"}:
            confidence = min(confidence, 0.54 if moment.timestamp_source == "interpolated" else 0.44)
        if confidence < 0.4:
            continue
        moment_evidence = [item for item in moment.supporting_evidence_ids if item in evidence_ids] or evidence_ids[:3]
        if not moment_evidence:
            continue
        if moment.importance_score < 0.5 and confidence < 0.65:
            continue
        headline, summary, interpretation, why = _moment_copy(moment, duration_ms)
        if not all((headline.strip(), summary.strip(), interpretation.strip(), (why or "").strip())):
            continue
        items.append(
            ReportTimelineItem(
                moment_id=moment.moment_id,
                type=moment.type,
                priority=moment.priority,
                headline=headline,
                summary=summary,
                listener_interpretation=interpretation,
                why_it_matters=why,
                dimension_impact=moment.dimension_impact,
                confidence=round(confidence, 2),
                start_ms=moment.start_ms,
                end_ms=moment.end_ms,
                timestamp_source=moment.timestamp_source,
                evidence_ids=moment_evidence,
                supporting_metrics=[_plain_metric_label(metric) for metric in moment.supporting_metrics],
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
    return sorted(items, key=lambda item: (item.importance_score, item.confidence), reverse=True)[:3]


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


def _dimension_reports(scores: Scores, diagnostic: DiagnosticReasoning, evidence_ids: list[str], confidence: float, evidence_cards: list[ReportEvidenceCard]) -> dict[str, ReportDimensionReport]:
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
        observed = [
            f"{card.signal} This affected {card.related_dimension.lower()} because {card.listener_interpretation}"
            for card in evidence_cards
            if card.related_dimension.lower() == DIMENSION_LABELS[dimension].lower()
        ]
        if observed:
            why = observed + why
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


def _hidden_cost(diagnosis: ReportDiagnosis, diagnostic: DiagnosticReasoning, evidence_ids: list[str], confidence: float, evidence_cards: list[ReportEvidenceCard], diagnosis_model: BehaviourDiagnosis | None) -> ReportHiddenCost:
    reasoning = diagnostic.hidden_cost_reasoning
    negative = _primary_negative(evidence_cards)
    behaviour = _card_behaviour(negative)
    if diagnosis_model:
        return ReportHiddenCost(
            dimension=diagnosis_model.primary_dimension,
            cost_id=reasoning.cost_id if reasoning and reasoning.cost_id else diagnosis_model.id,
            consequence=f"The hidden cost follows from the same pattern: {_without_period(_lower_first(diagnosis_model.one_sentence_pattern))}. {_without_period(_upper_first(diagnosis_model.observed_behaviour))}. {diagnosis_model.social_consequence}",
            evidence_ids=[item for item in diagnosis_model.evidence_ids if item in evidence_ids] or evidence_ids,
            confidence=diagnosis_model.confidence,
        )
    if reasoning:
        linked = [item for item in reasoning.evidence_ids if item in evidence_ids] or evidence_ids
        consequence = _hidden_cost_sentence(reasoning.listener_effect)
        if behaviour:
            consequence = f"The hidden cost is that when {behaviour}, {consequence[0].lower() + consequence[1:]}"
        return ReportHiddenCost(
            dimension=diagnosis.limiting_dimension,
            cost_id=reasoning.cost_id,
            consequence=consequence,
            evidence_ids=linked,
            confidence=reasoning.confidence,
        )
    consequence = _diagnosis_consequence(diagnosis.limiting_dimension)
    if behaviour:
        consequence = f"The hidden cost is that when {behaviour}, {consequence[0].lower() + consequence[1:]}"
    return ReportHiddenCost(dimension=diagnosis.limiting_dimension, cost_id="hidden_cost", consequence=consequence, evidence_ids=evidence_ids, confidence=confidence)


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


def _highest_leverage_fix(coaching: CoachingEngine | None, diagnostic: DiagnosticReasoning, evidence_ids: list[str], evidence_cards: list[ReportEvidenceCard], diagnosis_model: BehaviourDiagnosis | None) -> ReportHighestLeverageFix:
    primary = coaching.selected_interventions.primary_drill if coaching else None
    drill = None
    if diagnosis_model and diagnosis_model.drill_id and coaching:
        drill = next((item for item in coaching.drill_library if item.drill_id == diagnosis_model.drill_id), None)
    if primary and coaching:
        drill = drill or next((item for item in coaching.drill_library if item.drill_id == primary.drill_id), None)
    reasoning = diagnostic.highest_leverage_reasoning
    fallback_drill_id = None
    linked = primary.supporting_evidence_ids if primary else (reasoning.supporting_evidence if reasoning else evidence_ids)
    linked = [item for item in linked if item in evidence_ids] or evidence_ids
    issue = drill.title if drill else (reasoning.issue_id.replace("_", " ") if reasoning and reasoning.issue_id else "Practice focus")
    plain = drill.description if drill else (reasoning.plain_reason if reasoning else "Use the clearest supported practice focus from this recording.")
    duration = drill.estimated_duration_min if drill else None
    focus_card = next((card for card in evidence_cards if diagnosis_model and card.evidence_id in diagnosis_model.evidence_ids), None) or _primary_negative(evidence_cards) or _primary_positive(evidence_cards)
    behaviour = _card_behaviour(focus_card)
    if diagnosis_model:
        linked = [item for item in diagnosis_model.evidence_ids if item in evidence_ids] or linked
        focus_observation = max(diagnosis_model.supporting_observations, key=lambda item: item.expected_leverage)
        plain = f"The fastest improvement is to change the behaviour behind the report: {_lower_first(diagnosis_model.one_sentence_pattern)}"
        action_step = focus_observation.fix
        why = f"This matters because {_lower_first(focus_observation.listener_interpretation)} {_without_period(focus_observation.consequence)}."
        success = f"The next recording should make this behaviour easier to hear: {_lower_first(focus_observation.behaviour)}"
    elif focus_card and behaviour:
        fallback_drill_id = _fallback_drill_for_observation(focus_card.id, coaching)
        plain = (
            f"The fastest improvement is to change the behaviour where {behaviour}."
            if focus_card.direction == "negative"
            else f"The fastest improvement is to repeat the behaviour deliberately where {behaviour}."
        )
        action_step = _sentence((focus_card.why_it_matters.split("Fix:", 1)[1] if "Fix:" in focus_card.why_it_matters else focus_card.what_happened).strip())
        why = f"This matters because listeners are likely to experience that behaviour as: {focus_card.listener_interpretation}"
        success = f"The next recording should make this target behaviour clearer: {focus_card.signal}"
    else:
        plain = _clean_report_text(plain)
        action_step = _clean_report_text(drill.description if drill else plain)
        why = f"This is the fastest useful fix because it changes how the listener reads {', '.join(drill.target_dimensions if drill else (reasoning.affected_dimensions if reasoning else [])) or 'the limiting signal'}."
        success = f"The next recording should sound more controlled in {', '.join(drill.target_dimensions if drill else (reasoning.affected_dimensions if reasoning else [])) or 'the target area'}."
    return ReportHighestLeverageFix(
        issue=issue,
        plain_english=plain,
        why_this_matters=why,
        expected_score_lift=_expected_score_lift_label(primary, reasoning),
        target_dimensions=drill.target_dimensions if drill else (reasoning.affected_dimensions if reasoning else []),
        first_drill_id=drill.drill_id if drill else (diagnosis_model.drill_id if diagnosis_model else (fallback_drill_id or (reasoning.recommended_first_drill if reasoning else None))),
        action_step=action_step,
        success_signal=success,
        duration_min=duration,
        selection_score=primary.score if primary else (reasoning.selection_score if reasoning else 0.0),
        evidence_ids=linked,
    )


def _training(coaching: CoachingEngine | None, fix: ReportHighestLeverageFix, evidence_cards: list[ReportEvidenceCard], diagnosis_model: BehaviourDiagnosis | None) -> ReportTrainingPrescription:
    primary = coaching.selected_interventions.primary_drill if coaching else None
    drill = None
    if fix.first_drill_id and coaching:
        drill = next((item for item in coaching.drill_library if item.drill_id == fix.first_drill_id), None)
    if primary and coaching:
        drill = drill or next((item for item in coaching.drill_library if item.drill_id == primary.drill_id), None)
    if drill:
        focus_card = _primary_negative(evidence_cards) or _primary_positive(evidence_cards)
        why_chosen = (
            f"Chosen because it targets the behaviour the report is built around: {_without_period(diagnosis_model.one_sentence_pattern)}."
            if diagnosis_model
            else (
            f"Chosen because the report's clearest trainable behaviour is: {_card_observation(focus_card)}."
            if focus_card
            else f"Chosen because this recording's strongest coachable listener-cost points to {', '.join(drill.target_dimensions)}."
            )
        )
        return ReportTrainingPrescription(
            drill_id=drill.drill_id,
            title=drill.title,
            why_chosen=why_chosen,
            instructions=[fix.action_step or drill.description],
            target_metrics=[_plain_metric_label(metric) for metric in drill.target_metrics],
            target_dimensions=drill.target_dimensions,
            action_step=fix.action_step or drill.description,
            expected_score_lift=fix.expected_score_lift,
            duration_min=drill.estimated_duration_min,
            success_signal=fix.success_signal or "The next recording should make the target behaviour easier for a listener to hear.",
            evidence_ids=[item for item in (primary.supporting_evidence_ids if primary else fix.evidence_ids) if item in fix.evidence_ids] or fix.evidence_ids,
        )
    return ReportTrainingPrescription(
        drill_id=fix.first_drill_id,
        title=fix.issue,
        why_chosen=(
            f"Chosen because it targets the behaviour the report is built around: {_without_period(diagnosis_model.one_sentence_pattern)}."
            if diagnosis_model
            else f"Chosen because the clearest trainable behaviour is: {fix.plain_english}"
        ),
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
    if scores.score_confidence is not None and scores.score_confidence >= 0.8 and scores.score_rarity_label:
        percentile_label = scores.score_rarity_label
    elif scores.score_confidence is not None and scores.score_confidence >= 0.6 and scores.score_rarity_label:
        percentile_label = f"Directional: {scores.score_rarity_label}"
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


def _voiced_speech_ms(metrics: Metrics) -> int:
    return int(metrics.vad.total_speech_duration_ms or 0)


def _is_insufficient_sample(metrics: Metrics, audio_quality: AudioQuality, duration_ms: int, confidence: float) -> bool:
    voiced_ms = _voiced_speech_ms(metrics)
    speech_ratio = _num(metrics.vad.speech_ratio)
    return (
        not audio_quality.usable
        or (duration_ms > 0 and duration_ms < 8000)
        or voiced_ms <= 750
        or (duration_ms > 0 and speech_ratio < 0.03)
        or confidence < 0.25
    )


def _insufficient_evidence_card(confidence: float, audio_quality: AudioQuality) -> ReportEvidenceCard:
    warning = audio_quality.quality_warnings[0] if audio_quality.quality_warnings else "The recording did not contain enough usable speech for a full read."
    return ReportEvidenceCard(
        evidence_id="sample_quality_insufficient",
        id="sample_quality",
        trait="sample quality",
        dimension="Sample quality",
        direction="neutral",
        signal="The recording does not contain enough usable speech for a full authority read.",
        what_happened="There was too little reliable speech to separate delivery behaviour from recording quality.",
        why_it_matters=f"Authority should only make listener-perception claims when the sample can support them. Fix: Record a clear 30 to 60 second answer in a quiet place.",
        listener_interpretation="No reliable listener interpretation is made from this sample.",
        related_dimension="Sample quality",
        confidence=round(min(confidence, 0.35), 2),
        source_metrics=[],
    )


def _insufficient_report(
    *,
    scores: Scores,
    metrics: Metrics,
    audio_quality: AudioQuality,
    duration_ms: int,
    scenario: str,
    uncertainty: Uncertainty,
    psychological_inference: PsychologicalInference,
    diagnostic_reasoning: DiagnosticReasoning,
    coaching_engine: CoachingEngine | None,
    moment_intelligence: MomentIntelligence | None,
) -> AuthorityReport:
    evidence_card = _insufficient_evidence_card(scores.score_confidence or 0.0, audio_quality)
    evidence_ids = [evidence_card.evidence_id]
    authority_type = ReportAuthorityType(
        type_id="insufficient_sample",
        label="Insufficient Sample",
        description="There is not enough usable speech to assign an authority type.",
        top_dimensions=[],
        growth_dimensions=[],
        evidence_ids=evidence_ids,
        confidence=round(min(scores.score_confidence or 0.0, 0.35), 2),
    )
    limited_text = "There is not enough usable speech for a trustworthy authority diagnosis. Record a clear 30 to 60 second answer and try again."
    mirror = ReportMirror(
        headline="There is not enough usable speech for a full authority read.",
        identity_read=limited_text,
        one_line_identity_read=limited_text,
        core_tension="Insufficient sample quality",
        emotional_tone="not enough reliable speech",
        authority_type=authority_type.label,
        confidence_label="low",
        confidence_level="low",
        evidence_ids=evidence_ids,
    )
    diagnosis = ReportDiagnosis(
        strongest_dimension=None,
        limiting_dimension=None,
        primary_strength_dimension=None,
        primary_limiting_dimension=None,
        core_behavioural_pattern="Insufficient usable speech.",
        core_pattern="Insufficient usable speech.",
        social_consequence="No social consequence is inferred from this sample.",
        supporting_evidence_ids=evidence_ids,
        evidence_ids=evidence_ids,
        severity="low",
    )
    perception = ReportPerceptionMap(
        first_impression=ReportPerceptionRead(
            label="Sample quality",
            text="The recording is too limited for a reliable listener-perception read.",
            evidence_ids=evidence_ids,
            confidence=round(min(scores.score_confidence or 0.0, 0.35), 2),
        )
    )
    hidden_cost = ReportHiddenCost(
        dimension="Sample quality",
        cost_id="insufficient_sample",
        consequence="The hidden cost is measurement uncertainty: the system cannot tell whether the issue is delivery or the recording itself.",
        evidence_ids=evidence_ids,
        confidence=round(min(scores.score_confidence or 0.0, 0.35), 2),
    )
    fix = ReportHighestLeverageFix(
        issue="Record a clearer sample",
        plain_english="Record a clear 30 to 60 second answer before treating the report as diagnostic.",
        why_this_matters="This matters because Authority should only describe listener perception when there is enough speech to support the read.",
        expected_score_lift=None,
        target_dimensions=[],
        first_drill_id=None,
        action_step="Record in a quiet place, speak for 30 to 60 seconds, and avoid stopping after only a few words.",
        success_signal="The next upload should contain enough voiced speech for evidence, moments, and a diagnosis.",
        duration_min=1,
        selection_score=0.0,
        evidence_ids=evidence_ids,
    )
    training = ReportTrainingPrescription(
        drill_id=None,
        title="Clear sample retest",
        why_chosen="Chosen because the current recording is not strong enough to support a normal diagnosis.",
        instructions=["Record a clear 30 to 60 second answer in a quiet place, then rerun the benchmark."],
        target_metrics=[],
        target_dimensions=[],
        action_step=fix.action_step,
        expected_score_lift=None,
        duration_min=1,
        success_signal=fix.success_signal,
        evidence_ids=evidence_ids,
    )
    retest = ReportRetestPlan(
        recommended_retest_after_days=1,
        focus_metric="clearer sample quality",
        compare_metrics=["usable speech", "recording clarity"],
        same_prompt_recommended=True,
        success_definition=fix.success_signal,
        evidence_ids=evidence_ids,
    )
    appendix = _technical_appendix(metrics, scores, audio_quality, evidence_ids)
    share_card = ReportShareCard(
        authority_score=scores.authority_score,
        authority_type=authority_type.label,
        top_strength=None,
        growth_area=None,
        one_line_identity_read=mirror.one_line_identity_read,
        percentile_label=None,
        share_safety="public_safe",
        hidden_private_findings=[],
    )
    reasons = list(dict.fromkeys(uncertainty.reasons + psychological_inference.uncertainty.reasons + ["Insufficient usable speech for full report"]))
    if duration_ms and duration_ms < 25000:
        reasons.append("Short recording limits full report confidence")
    report_uncertainty = Uncertainty(
        overall_confidence_label="low",
        suppressed_traits=list(dict.fromkeys(psychological_inference.suppressed_traits + ["insufficient_sample"])),
        reasons=list(dict.fromkeys(reasons)),
    )
    report = AuthorityReport(
        mirror=mirror,
        diagnosis=diagnosis,
        perception_map=perception,
        evidence_chain=[evidence_card],
        timeline=[],
        moment_intelligence=moment_intelligence or MomentIntelligence(moments=[]),
        dimension_reports={},
        hidden_cost=hidden_cost,
        highest_leverage_fix=fix,
        training_prescription=training,
        retest_plan=retest,
        authority_type=authority_type,
        share_card=share_card,
        technical_appendix=appendix,
        scenario_summary=ReportScenarioSummary(
            scenario_id=get_scenario_profile(scenario).scenario_id,
            highest_leverage_fix=fix.issue,
            coaching_explanation="A clearer recording is needed before selecting a behavioural drill.",
        ),
        diagnostic_reasoning=diagnostic_reasoning,
        primary_diagnosis=None,
        secondary_diagnosis=None,
        contradictions=diagnostic_reasoning.contradictions,
        hidden_cost_reasoning=None,
        dimension_reasoning={},
        trait_reasoning=diagnostic_reasoning.trait_reasoning,
        highest_leverage_reasoning=None,
        coaching_engine=coaching_engine,
        uncertainty=report_uncertainty,
    )
    return report.model_copy(update={"validation": _validate_report(report, coaching_engine)})


def _reconciled_diagnostic_reasoning(diagnostic: DiagnosticReasoning, diagnosis_model: BehaviourDiagnosis | None, evidence_ids: list[str]) -> DiagnosticReasoning:
    if diagnosis_model is None:
        return diagnostic
    primary = diagnostic.primary_diagnosis
    severity = primary.severity if primary else "medium"
    supporting = [item for item in diagnosis_model.evidence_ids if item in evidence_ids] or evidence_ids
    moments = list(diagnosis_model.moment_ids)
    reconciled_primary = DiagnosticDiagnosis(
        diagnosis_id=diagnosis_model.id,
        diagnosis_name=diagnosis_model.user_facing_title,
        confidence=diagnosis_model.confidence,
        severity=severity,
        supporting_traits=[diagnosis_model.primary_dimension],
        contradicting_traits=[item.dimension for item in diagnosis_model.contradicting_observations],
        supporting_evidence_ids=supporting,
        supporting_moment_ids=moments,
        affected_dimensions=[diagnosis_model.primary_dimension, *diagnosis_model.secondary_dimensions],
    )
    hidden = diagnostic.hidden_cost_reasoning or HiddenCostReasoning(
        cost_id=f"hidden_cost_{diagnosis_model.primary_dimension.lower()}",
        source_signal=diagnosis_model.observed_behaviour,
        interpretation=diagnosis_model.listener_interpretation,
        consequence=diagnosis_model.social_consequence,
        listener_effect=diagnosis_model.social_consequence,
        affected_dimensions=[diagnosis_model.primary_dimension, *diagnosis_model.secondary_dimensions],
        evidence_ids=supporting,
        moment_ids=moments,
        confidence=diagnosis_model.confidence,
    )
    leverage = diagnostic.highest_leverage_reasoning or HighestLeverageReasoning(
        issue_id=diagnosis_model.fix_category,
        plain_reason=diagnosis_model.fix_category,
        affected_dimensions=[diagnosis_model.primary_dimension, *diagnosis_model.secondary_dimensions],
        supporting_evidence=supporting,
        expected_score_lift="medium",
        recommended_first_drill=diagnosis_model.drill_id,
        confidence=diagnosis_model.confidence,
        severity=0.65,
        authority_impact=0.75,
        trainability=0.75,
        evidence_confidence=diagnosis_model.confidence,
        scenario_relevance=1.0,
        selection_score=round(0.65 * 0.75 * 0.75 * diagnosis_model.confidence, 3),
    )
    return diagnostic.model_copy(
        update={
            "primary_diagnosis": reconciled_primary,
            "hidden_cost_reasoning": hidden,
            "highest_leverage_reasoning": leverage,
        }
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _words_confidence(words: list[TranscriptWord]) -> float | None:
    values = [word.confidence for word in words if word.confidence is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _transcript_safe(transcript: Transcript | None) -> bool:
    if not transcript or not transcript.full_text.strip():
        return False
    if transcript.overall_asr_confidence is not None and transcript.overall_asr_confidence < 0.65:
        return False
    word_conf = _words_confidence(transcript.words)
    return word_conf is None or word_conf >= 0.65


def _timestamps_safe(start_ms: int | None, end_ms: int | None, timestamp_source: str | None) -> bool:
    return (
        start_ms is not None
        and end_ms is not None
        and end_ms > start_ms
        and (timestamp_source or "estimated") in {"real", "segment"}
    )


def _transcript_window(words: list[TranscriptWord], start_ms: int | None, end_ms: int | None) -> str | None:
    if start_ms is None or end_ms is None:
        return None
    selected = [word.text for word in words if word.end_ms >= start_ms and word.start_ms <= end_ms]
    text = " ".join(selected).strip()
    return text or None


def _segment_by_role(transcript: Transcript | None, role: str) -> TranscriptSegment | None:
    if not transcript:
        return None
    return next((segment for segment in transcript.segments if segment.role == role), None)


def _opening_text(transcript: Transcript | None) -> str:
    segment = _segment_by_role(transcript, "opening")
    if segment and segment.text.strip():
        return segment.text.strip()
    return " ".join((transcript.full_text if transcript else "").split()[:22]).strip()


def _closing_text(transcript: Transcript | None) -> str:
    segment = _segment_by_role(transcript, "closing")
    if segment and segment.text.strip():
        return segment.text.strip()
    return " ".join((transcript.full_text if transcript else "").split()[-22:]).strip()


def _segment_timing(segment: TranscriptSegment | None) -> tuple[int | None, int | None, str]:
    if not segment:
        return None, None, "estimated"
    return segment.start_ms, segment.end_ms, segment.timestamp_source


def _short_clause(text: str, max_words: int = 18) -> str:
    cleaned = " ".join((text or "").replace("\n", " ").split()).strip(" ,;:")
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+|\b(?:but|because|so|and then)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)
    candidate = parts[0].strip(" ,;:")
    words = candidate.split()
    if len(words) > max_words:
        candidate = " ".join(words[:max_words])
    return candidate.strip(" ,;:.")


_UNSAFE_CLAUSE_ENDINGS = {
    "a",
    "an",
    "and",
    "as",
    "because",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "of",
    "or",
    "so",
    "that",
    "the",
    "to",
    "with",
}


def _safe_transcript_clause(text: str | None, max_words: int = 18) -> str | None:
    candidate = _short_clause(text or "", max_words=max_words)
    if not candidate:
        return None
    words = candidate.split()
    if len(words) < 3:
        return None
    last = re.sub(r"[^a-z0-9']", "", words[-1].lower())
    first = re.sub(r"[^a-z0-9']", "", words[0].lower())
    if last in _UNSAFE_CLAUSE_ENDINGS or first in {"and", "but", "so", "because"}:
        return None
    if candidate.count("(") != candidate.count(")") or candidate.count('"') % 2:
        return None
    has_action_word = any(
        re.search(r"(ed|ing|es|s)$", re.sub(r"[^a-z']", "", token.lower()))
        or token.lower().strip(".,!?") in {"am", "are", "be", "being", "been", "can", "could", "did", "do", "does", "had", "has", "have", "is", "keeps", "make", "makes", "matter", "matters", "need", "needs", "should", "was", "were", "will", "would"}
        for token in words
    )
    if not has_action_word:
        return None
    return candidate


def _fact_phrase(fact: RecordingFact) -> str:
    if fact.user_safe and fact.transcript_text:
        return _safe_transcript_clause(fact.transcript_text, 16) or ""
    return fact.observed_behavior.rstrip(".")


def _content_reference(fact: RecordingFact) -> str:
    phrase = _fact_phrase(fact)
    if fact.user_safe and fact.transcript_text and phrase:
        return f"you said {phrase}"
    return _lower_first(phrase or fact.observed_behavior.rstrip("."))


def _metric_summary(label: str, value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return label
    if isinstance(value, float):
        if abs(value) >= 10:
            rendered = f"{value:.0f}"
        else:
            rendered = f"{value:.2f}".rstrip("0").rstrip(".")
    else:
        rendered = str(value)
    return f"{label}: {rendered}{suffix}"


def _add_fact(
    facts: list[RecordingFact],
    counters: dict[str, int],
    *,
    fact_type: str,
    source: str,
    observed_behavior: str,
    related_dimensions: tuple[str, ...],
    confidence: float,
    start_ms: int | None = None,
    end_ms: int | None = None,
    transcript_text: str | None = None,
    measurement_summary: str | None = None,
    timestamp_source: str = "estimated",
    supporting_metric_ids: tuple[str, ...] = (),
    contradictions: tuple[str, ...] = (),
    user_safe: bool = True,
) -> None:
    counters[fact_type] = counters.get(fact_type, 0) + 1
    facts.append(
        RecordingFact(
            fact_id=f"{fact_type}_{counters[fact_type]}",
            fact_type=fact_type,
            source=source,
            start_ms=start_ms,
            end_ms=end_ms,
            transcript_text=transcript_text.strip() if transcript_text else None,
            observed_behavior=_sentence(observed_behavior),
            measurement_summary=measurement_summary,
            related_dimensions=related_dimensions,
            confidence=round(_clamp01(confidence), 2),
            timestamp_source=timestamp_source,
            supporting_metric_ids=supporting_metric_ids,
            contradictions=contradictions,
            user_safe=user_safe,
        )
    )


def _repeated_phrase(text: str) -> str | None:
    tokens = [re.sub(r"[^a-z0-9']", "", token.lower()) for token in text.split()]
    tokens = [token for token in tokens if token and len(token) > 2]
    if len(tokens) < 8:
        return None
    for size in (4, 3, 2):
        counts: dict[tuple[str, ...], int] = {}
        for index in range(0, len(tokens) - size + 1):
            gram = tuple(tokens[index:index + size])
            if any(token in {"the", "and", "that", "this", "with", "from"} for token in gram):
                continue
            counts[gram] = counts.get(gram, 0) + 1
        repeated = [gram for gram, count in counts.items() if count >= 2]
        if repeated:
            return " ".join(repeated[0])
    return None


def _build_recording_fact_ledger(
    *,
    transcript: Transcript | None,
    metrics: Metrics,
    psychological_inference: PsychologicalInference,
    moments: list[Moment],
    audio_quality: AudioQuality,
    duration_ms: int,
    base_confidence: float,
) -> list[RecordingFact]:
    del psychological_inference
    facts: list[RecordingFact] = []
    counters: dict[str, int] = {}
    text_safe = _transcript_safe(transcript)
    text = (transcript.full_text if transcript else "").strip()
    opening = _opening_text(transcript)
    closing = _closing_text(transcript)
    opening_segment = _segment_by_role(transcript, "opening")
    closing_segment = _segment_by_role(transcript, "closing")
    opening_start, opening_end, opening_ts = _segment_timing(opening_segment)
    closing_start, closing_end, closing_ts = _segment_timing(closing_segment)
    metric_conf = min(base_confidence or 0.65, 0.88)

    if text_safe and opening and (metrics.linguistic.opening_strength_score or 0) >= 0.7:
        claim = _short_clause(opening)
        _add_fact(
            facts,
            counters,
            fact_type="clear_opening_claim",
            source="transcript",
            start_ms=opening_start,
            end_ms=opening_end,
            transcript_text=claim,
            observed_behavior=f"The answer established its direction early with the idea: {claim}.",
            related_dimensions=("Structure", "Command"),
            confidence=min(metric_conf + 0.04, 0.92),
            timestamp_source=opening_ts,
            supporting_metric_ids=("linguistic.opening_strength_score",),
            user_safe=True,
        )
    elif text_safe and opening and (metrics.linguistic.opening_strength_score or 1) <= 0.45:
        _add_fact(
            facts,
            counters,
            fact_type="delayed_main_point",
            source="transcript",
            start_ms=opening_start,
            end_ms=opening_end,
            transcript_text=opening,
            observed_behavior="The opening used runway before giving the listener a clear main point.",
            related_dimensions=("Structure", "Clarity"),
            confidence=metric_conf,
            timestamp_source=opening_ts,
            supporting_metric_ids=("linguistic.opening_strength_score",),
            user_safe=True,
        )

    has_example = bool(re.search(r"\b(for example|such as|like when|when I|when we|for instance)\b|\b\d+\b", text, re.IGNORECASE))
    if text_safe and text and not has_example and (
        (metrics.linguistic.specificity_score is not None and metrics.linguistic.specificity_score <= 0.34)
        or (metrics.linguistic.concreteness_score is not None and metrics.linguistic.concreteness_score <= 0.32)
    ):
        claim = _short_clause(opening or text, 22)
        _add_fact(
            facts,
            counters,
            fact_type="claim_without_proof",
            source="transcript",
            start_ms=opening_start,
            end_ms=opening_end,
            transcript_text=claim,
            observed_behavior=f"The answer states the idea {claim} without giving a concrete situation, example, number, or named detail to prove it.",
            related_dimensions=("Persuasion", "Clarity", "Structure"),
            confidence=min(metric_conf + 0.08, 0.9),
            timestamp_source=opening_ts,
            supporting_metric_ids=("linguistic.specificity_score", "linguistic.concreteness_score"),
            user_safe=True,
        )
    if text_safe and text and has_example:
        match = re.search(r"\b(for example|such as|like when|when I|when we|for instance)\b[^.!?]*", text, re.IGNORECASE)
        example = _short_clause(match.group(0) if match else text, 20)
        _add_fact(
            facts,
            counters,
            fact_type="concrete_example",
            source="transcript",
            transcript_text=example,
            observed_behavior=f"The answer gave a concrete support point: {example}.",
            related_dimensions=("Persuasion", "Clarity"),
            confidence=min(metric_conf + 0.05, 0.9),
            timestamp_source="estimated",
            supporting_metric_ids=("linguistic.specificity_score", "linguistic.concreteness_score"),
            user_safe=True,
        )

    repeated = _repeated_phrase(text) if text_safe else None
    if repeated and (metrics.linguistic.repetition_rate or 0) >= 0.35:
        _add_fact(
            facts,
            counters,
            fact_type="repeated_phrase",
            source="transcript",
            transcript_text=repeated,
            observed_behavior=f"The phrase {repeated} repeated instead of advancing the answer.",
            related_dimensions=("Structure", "Clarity"),
            confidence=min(metric_conf, 0.82),
            supporting_metric_ids=("linguistic.repetition_rate",),
            user_safe=True,
        )

    if (metrics.linguistic.structure_score is not None and metrics.linguistic.structure_score <= 0.45) or (metrics.linguistic.rambling_score or 0) >= 0.45:
        _add_fact(
            facts,
            counters,
            fact_type="weak_local_structure",
            source="transcript" if text_safe else "metrics",
            transcript_text=_short_clause(text, 24) if text_safe else None,
            observed_behavior="The answer path did not consistently separate point, support, and close.",
            related_dimensions=("Structure", "Clarity"),
            confidence=metric_conf,
            supporting_metric_ids=("linguistic.structure_score", "linguistic.rambling_score"),
            user_safe=text_safe,
        )
    elif metrics.linguistic.structure_score is not None and metrics.linguistic.structure_score >= 0.72:
        _add_fact(
            facts,
            counters,
            fact_type="strong_local_structure",
            source="transcript" if text_safe else "metrics",
            transcript_text=_short_clause(opening or text, 22) if text_safe else None,
            observed_behavior="The answer kept a visible local structure.",
            related_dimensions=("Structure", "Clarity"),
            confidence=min(metric_conf + 0.03, 0.9),
            supporting_metric_ids=("linguistic.structure_score",),
            user_safe=text_safe,
        )

    if text_safe and closing and ((metrics.linguistic.closing_strength_score or 1) <= 0.5 or re.search(r"\b(and|but|so|um|uh)$", closing.strip(), re.IGNORECASE)):
        _add_fact(
            facts,
            counters,
            fact_type="abrupt_ending",
            source="transcript",
            start_ms=closing_start,
            end_ms=closing_end,
            transcript_text=closing,
            observed_behavior="The closing did not turn the final idea into a clean takeaway.",
            related_dimensions=("Command", "Structure"),
            confidence=metric_conf,
            timestamp_source=closing_ts,
            supporting_metric_ids=("linguistic.closing_strength_score",),
            user_safe=True,
        )
    elif (metrics.linguistic.closing_strength_score or 0) >= 0.72:
        _add_fact(
            facts,
            counters,
            fact_type="reinforced_ending",
            source="transcript" if text_safe else "metrics",
            start_ms=closing_start,
            end_ms=closing_end,
            transcript_text=closing if text_safe else None,
            observed_behavior="The closing preserved the final impression instead of drifting away.",
            related_dimensions=("Command", "Structure"),
            confidence=min(metric_conf + 0.04, 0.9),
            timestamp_source=closing_ts,
            supporting_metric_ids=("linguistic.closing_strength_score",),
            user_safe=text_safe,
        )

    if (metrics.linguistic.lexical_fillers or 0) > 0 or (metrics.linguistic.filler_words_per_min or 0) >= 6:
        filler_words = []
        if transcript and text_safe:
            filler_words = [word.text for word in transcript.words if word.is_filler][:4]
        summary = f"{len(filler_words)} visible filler word{'s' if len(filler_words) != 1 else ''}" if filler_words else _metric_summary("filler words per minute", metrics.linguistic.filler_words_per_min)
        _add_fact(
            facts,
            counters,
            fact_type="lexical_filler",
            source="transcript" if text_safe else "metrics",
            transcript_text=", ".join(filler_words) if filler_words else None,
            observed_behavior="Lexical fillers appeared in the wording.",
            measurement_summary=summary,
            related_dimensions=("Clarity", "Composure"),
            confidence=min(metric_conf + 0.02, 0.88),
            supporting_metric_ids=("linguistic.lexical_fillers", "linguistic.filler_words_per_min"),
            user_safe=bool(filler_words),
        )

    if (metrics.linguistic.acoustic_hesitations or 0) > 0 or (metrics.derived.hesitation_cluster_score or 0) >= 0.55:
        count = metrics.linguistic.acoustic_hesitations or metrics.rhythm.hesitation_windows or 1
        matching = next((moment for moment in moments if moment.type in {"hesitation_cluster", "confidence_drop", "most_unstable_section"}), None)
        span = _transcript_window(transcript.words, matching.start_ms, matching.end_ms) if transcript and matching and text_safe else None
        _add_fact(
            facts,
            counters,
            fact_type="acoustic_hesitation",
            source="acoustic",
            start_ms=matching.start_ms if matching else None,
            end_ms=matching.end_ms if matching else None,
            transcript_text=span,
            observed_behavior=f"{int(count)} hesitation event{'s' if int(count) != 1 else ''} appeared in the delivery timing.",
            measurement_summary=f"{int(count)} hesitation event{'s' if int(count) != 1 else ''}",
            related_dimensions=("Composure", "Command"),
            confidence=min(metric_conf + 0.04, 0.88),
            timestamp_source=matching.timestamp_source if matching else "estimated",
            supporting_metric_ids=("linguistic.acoustic_hesitations", "derived.hesitation_cluster_score", "rhythm.hesitation_windows"),
            user_safe=True,
        )

    if (metrics.rhythm.speed_up_segments or 0) > 0 or (metrics.rhythm.burst_speaking_segments or 0) > 0:
        matching = next((moment for moment in moments if moment.type in {"rushing_moment", "confidence_drop", "most_unstable_section"}), None)
        span = _transcript_window(transcript.words, matching.start_ms, matching.end_ms) if transcript and matching and text_safe else None
        _add_fact(
            facts,
            counters,
            fact_type="pace_acceleration",
            source="acoustic",
            start_ms=matching.start_ms if matching else None,
            end_ms=matching.end_ms if matching else None,
            transcript_text=span,
            observed_behavior="Pace increased during part of the answer instead of staying even.",
            measurement_summary=_metric_summary("speed-up sections", metrics.rhythm.speed_up_segments),
            related_dimensions=("Composure", "Clarity", "Command"),
            confidence=min(metric_conf + 0.03, 0.88),
            timestamp_source=matching.timestamp_source if matching else "estimated",
            supporting_metric_ids=("rhythm.speed_up_segments", "rhythm.burst_speaking_segments"),
            user_safe=True,
        )
    elif (metrics.rhythm.rhythm_consistency or 0) >= 0.75 and 115 <= (metrics.raw_acoustic.words_per_minute or metrics.rhythm.words_per_minute or 0) <= 165:
        _add_fact(
            facts,
            counters,
            fact_type="pace_stabilization",
            source="acoustic",
            observed_behavior="Pace stayed in a controlled range across the answer.",
            measurement_summary="controlled pace range",
            related_dimensions=("Composure", "Clarity"),
            confidence=min(metric_conf + 0.03, 0.9),
            supporting_metric_ids=("rhythm.rhythm_consistency", "raw_acoustic.words_per_minute"),
            user_safe=True,
        )

    if (metrics.raw_acoustic.mid_phrase_pause_rate or 0) >= 0.35 or (metrics.vad.mid_sentence_pauses_ms and len(metrics.vad.mid_sentence_pauses_ms) >= 2):
        _add_fact(
            facts,
            counters,
            fact_type="pause_cluster",
            source="acoustic",
            observed_behavior="Pauses clustered inside the thought rather than only between complete phrases.",
            measurement_summary="clustered mid-thought pauses",
            related_dimensions=("Composure", "Clarity"),
            confidence=metric_conf,
            supporting_metric_ids=("raw_acoustic.mid_phrase_pause_rate", "vad.mid_sentence_pauses_ms"),
            user_safe=True,
        )
    elif 250 <= (metrics.raw_acoustic.avg_pause_ms or 0) <= 800 and (metrics.raw_acoustic.mid_phrase_pause_rate or 1) <= 0.25:
        _add_fact(
            facts,
            counters,
            fact_type="owned_pause",
            source="acoustic",
            observed_behavior="Pauses were more likely to land between thoughts than interrupt them.",
            measurement_summary="owned phrase spacing",
            related_dimensions=("Command", "Composure"),
            confidence=min(metric_conf + 0.04, 0.9),
            supporting_metric_ids=("raw_acoustic.avg_pause_ms", "raw_acoustic.mid_phrase_pause_rate"),
            user_safe=True,
        )

    if (metrics.derived.dynamic_emphasis_score is not None and metrics.derived.dynamic_emphasis_score <= 0.3) or (metrics.derived.monotony_index or 0) >= 0.45:
        _add_fact(
            facts,
            counters,
            fact_type="flat_emphasis",
            source="acoustic",
            observed_behavior="Important words did not receive much contrast from the surrounding phrase.",
            measurement_summary="low vocal contrast",
            related_dimensions=("Presence", "Persuasion"),
            confidence=metric_conf,
            supporting_metric_ids=("derived.dynamic_emphasis_score", "derived.monotony_index"),
            user_safe=True,
        )
    elif (metrics.derived.dynamic_emphasis_score or 0) >= 0.62:
        _add_fact(
            facts,
            counters,
            fact_type="strong_emphasis",
            source="acoustic",
            observed_behavior="Key moments received more vocal contrast than the surrounding delivery.",
            measurement_summary="strong local emphasis",
            related_dimensions=("Presence", "Persuasion"),
            confidence=min(metric_conf + 0.03, 0.9),
            supporting_metric_ids=("derived.dynamic_emphasis_score",),
            user_safe=True,
        )

    if (metrics.raw_acoustic.terminal_rising_ratio or 0) >= 0.45:
        _add_fact(
            facts,
            counters,
            fact_type="rising_ending",
            source="acoustic",
            start_ms=closing_start,
            end_ms=closing_end,
            transcript_text=closing if text_safe else None,
            observed_behavior="Some declarative endings rose instead of landing with a clear finish.",
            measurement_summary="rising declarative endings",
            related_dimensions=("Command", "Composure"),
            confidence=metric_conf,
            timestamp_source=closing_ts,
            supporting_metric_ids=("raw_acoustic.terminal_rising_ratio",),
            user_safe=text_safe or not closing,
        )
    elif (metrics.raw_acoustic.terminal_falling_ratio or 0) >= 0.35 and (metrics.linguistic.closing_strength_score or 0) >= 0.65:
        _add_fact(
            facts,
            counters,
            fact_type="decisive_ending",
            source="acoustic",
            start_ms=closing_start,
            end_ms=closing_end,
            transcript_text=closing if text_safe else None,
            observed_behavior="The final delivery pattern supported a clear full-stop ending.",
            measurement_summary="falling ending pattern",
            related_dimensions=("Command", "Structure"),
            confidence=min(metric_conf + 0.03, 0.9),
            timestamp_source=closing_ts,
            supporting_metric_ids=("raw_acoustic.terminal_falling_ratio", "linguistic.closing_strength_score"),
            user_safe=text_safe or not closing,
        )

    if not audio_quality.usable:
        for index, fact in enumerate(facts):
            if fact.source == "acoustic":
                facts[index] = RecordingFact(**{**fact.__dict__, "confidence": round(min(fact.confidence, 0.55), 2)})
    if duration_ms and duration_ms < 25000:
        return [fact for fact in facts if fact.confidence >= 0.62 and fact.fact_type in {"clear_opening_claim", "reinforced_ending", "pace_stabilization", "owned_pause"}]
    return facts


def _facts_by_type(facts: list[RecordingFact], *types: str) -> list[RecordingFact]:
    wanted = set(types)
    return [fact for fact in facts if fact.fact_type in wanted]


def _fact_map(facts: list[RecordingFact]) -> dict[str, RecordingFact]:
    return {fact.fact_id: fact for fact in facts}


def _observation_from_facts(
    observation_id: str,
    title: str,
    pattern: str,
    effect: str,
    facts: list[RecordingFact],
    dimensions: tuple[str, ...],
    impact: float,
    trainability: float,
    target_behavior: str,
    drill_categories: tuple[str, ...],
    contradictions: list[RecordingFact] | None = None,
) -> FactObservation:
    contradictions = contradictions or []
    confidences = [fact.confidence for fact in facts]
    confidence = sum(confidences) / max(len(confidences), 1)
    independence = min(1.0, len({fact.source for fact in facts}) * 0.22 + len({fact.fact_type for fact in facts}) * 0.18)
    contradiction_penalty = min(0.35, sum(fact.confidence for fact in contradictions) * 0.12)
    importance = _clamp01(impact * (0.55 + confidence * 0.35 + independence * 0.1) - contradiction_penalty)
    distinctiveness = _clamp01(0.45 + len({fact.fact_type for fact in facts}) * 0.16 + len({fact.source for fact in facts}) * 0.08)
    return FactObservation(
        observation_id=observation_id,
        title=title,
        observed_pattern=_sentence(pattern),
        supporting_fact_ids=tuple(fact.fact_id for fact in facts),
        listener_effect=_sentence(effect),
        related_dimensions=dimensions,
        confidence=round(confidence, 2),
        importance=round(importance, 3),
        trainability=trainability,
        distinctiveness=round(distinctiveness, 2),
        contradiction_penalty=round(contradiction_penalty, 2),
        target_behavior=target_behavior,
        recommended_drill_categories=drill_categories,
    )


def _strength_title_and_pattern(facts: list[RecordingFact]) -> tuple[str, str, str]:
    first_type = next((fact.fact_type for fact in facts), "")
    fact_types = [fact.fact_type for fact in facts]
    selected = first_type if first_type in {
        "clear_opening_claim",
        "pace_stabilization",
        "concrete_example",
        "reinforced_ending",
        "decisive_ending",
        "owned_pause",
        "strong_emphasis",
    } else next((fact_type for fact_type in fact_types if fact_type), "")
    if selected == "clear_opening_claim":
        return (
            "Opening gives the listener the frame",
            "The listener does not have to wonder what the answer is about.",
            "That early direction should stay in place while the main limiter is trained.",
        )
    if selected == "pace_stabilization":
        return (
            "Pace gives the listener room",
            "The pace settles enough for the listener to absorb the point.",
            "That steadiness should be preserved while the main limiter is trained.",
        )
    if selected == "concrete_example":
        return (
            "Example makes the idea visible",
            "The listener no longer has to imagine what the speaker means once the example appears.",
            "That proof habit should remain attached to the main claim.",
        )
    if selected in {"reinforced_ending", "decisive_ending"}:
        return (
            "Ending leaves a final idea",
            "The final stretch gives the listener a clear last thought to carry away.",
            "That landing should be preserved while the earlier limiter is trained.",
        )
    if selected == "owned_pause":
        return (
            "Pauses read as control",
            "The listener hears silence between thoughts rather than searching inside a thought.",
            "That controlled spacing should stay intact during the next recording.",
        )
    if selected == "strong_emphasis":
        return (
            "Emphasis marks what matters",
            "Key words receive enough contrast for the listener to notice the important idea.",
            "That contrast should stay focused rather than becoming general intensity.",
        )
    return (
        "One behaviour already helps the listener",
        facts[0].observed_behavior if facts else "One supported behaviour gives the listener something stable.",
        "Keep that supported behaviour while training the main limiter.",
    )


def _build_fact_observations(facts: list[RecordingFact]) -> list[FactObservation]:
    observations: list[FactObservation] = []
    clear = _facts_by_type(facts, "clear_opening_claim")
    unsupported = _facts_by_type(facts, "claim_without_proof")
    examples = _facts_by_type(facts, "concrete_example")
    closing_weak = _facts_by_type(facts, "abrupt_ending")
    if unsupported:
        support = (clear[:1] + unsupported[:1] + closing_weak[:1]) or unsupported[:1]
        claim_text = _fact_phrase(unsupported[0])
        claim = f"the claim about {_lower_first(claim_text)}" if claim_text else "the main claim"
        observations.append(
            _observation_from_facts(
                "thin_proof",
                "Clear claim, thin proof",
                f"The answer gave the listener a direction, but {claim} did not get a concrete example or proof point behind it.",
                "The listener can understand the claim before they have enough evidence to believe it.",
                support,
                ("Persuasion", "Clarity", "Structure"),
                0.94,
                0.92,
                "grounded_specificity",
                ("specificity", "structure_compression"),
                examples[:1],
            )
        )

    hesitation = _facts_by_type(facts, "acoustic_hesitation", "pause_cluster", "lexical_filler", "pace_acceleration")
    if hesitation:
        acoustic = _facts_by_type(hesitation, "acoustic_hesitation")
        lexical = _facts_by_type(hesitation, "lexical_filler")
        pieces = []
        if acoustic:
            pieces.append(acoustic[0].measurement_summary or "hesitation events")
        if lexical:
            pieces.append(lexical[0].measurement_summary or "lexical fillers")
        if _facts_by_type(hesitation, "pace_acceleration"):
            pieces.append("a local pace increase")
        summary = ", ".join(pieces) if pieces else "delivery interruptions"
        pattern = (
            f"{summary} appeared; you paused while searching for your next idea instead of restarting cleanly."
            if acoustic and not lexical
            else f"{summary} appeared while the answer was developing."
        )
        observations.append(
            _observation_from_facts(
                "hesitation_control",
                "Delivery searched while the idea was moving",
                pattern,
                "The listener may hear the speaker finding the wording in real time instead of leading the thought cleanly.",
                hesitation[:3],
                ("Composure", "Command", "Clarity"),
                0.9,
                0.86,
                "pause_ownership",
                ("pause_ownership", "composure", "pace_regulation", "filler_reduction"),
            )
        )

    unclear = _facts_by_type(facts, "delayed_main_point", "weak_local_structure", "repeated_phrase", "weak_transition", "topic_shift", "unclear_sentence_path")
    if unclear:
        observations.append(
            _observation_from_facts(
                "unclear_path",
                "The answer path did not stay visible",
                "The answer made the listener work to separate the main point, the support, and the close.",
                "That can reduce trust in the speaker's control even when individual ideas are understandable.",
                unclear[:3],
                ("Structure", "Clarity", "Persuasion"),
                0.86,
                0.9,
                "structured_thinking",
                ("opening_strength", "structure_compression"),
            )
        )

    weak_close = _facts_by_type(facts, "abrupt_ending", "rising_ending")
    decisive = _facts_by_type(facts, "decisive_ending", "reinforced_ending")
    if weak_close:
        close_fact = weak_close[0]
        safe_close = _safe_transcript_clause(close_fact.transcript_text, 16) if close_fact.user_safe else None
        close_pattern = (
            f"Near the end, you said {safe_close} without turning it into a clean final takeaway."
            if safe_close
            else "Near the end, the answer stopped before becoming a clean final takeaway."
        )
        observations.append(
            _observation_from_facts(
                "weak_close",
                "The ending did not fully land",
                close_pattern,
                "The final impression can feel less settled than the answer's earlier direction.",
                weak_close[:2],
                ("Command", "Structure"),
                0.82,
                0.88,
                "clean_closing",
                ("closing_strength", "declarative_finality"),
                decisive[:1],
            )
        )

    flat = _facts_by_type(facts, "flat_emphasis")
    if flat:
        observations.append(
            _observation_from_facts(
                "flat_presence",
                "Important words did not get enough contrast",
                flat[0].observed_behavior,
                "The point may be clear but less memorable because the delivery does not mark what matters most.",
                flat[:1],
                ("Presence", "Persuasion"),
                0.72,
                0.74,
                "vocal_variety",
                ("dynamic_emphasis", "presence"),
            )
        )

    strengths = _facts_by_type(facts, "strong_local_structure", "pace_stabilization", "owned_pause", "strong_emphasis", "clear_opening_claim", "reinforced_ending", "decisive_ending")
    if strengths:
        chosen = strengths[:3]
        strength_title, strength_pattern, strength_effect = _strength_title_and_pattern(chosen)
        observations.append(
            _observation_from_facts(
                "usable_strength",
                strength_title,
                strength_pattern,
                strength_effect,
                chosen,
                tuple(dict.fromkeys(dimension for fact in chosen for dimension in fact.related_dimensions)) or ("Clarity",),
                0.42,
                0.55,
                "preserve_strength",
                (),
            )
        )

    merged: dict[str, FactObservation] = {}
    for observation in observations:
        existing = merged.get(observation.observation_id)
        if not existing or observation.importance > existing.importance:
            merged[observation.observation_id] = observation
    return sorted(merged.values(), key=lambda item: (item.importance, item.confidence, item.distinctiveness), reverse=True)


def _select_fact_diagnosis(
    observations: list[FactObservation],
    facts: list[RecordingFact],
    coaching: CoachingEngine | None,
    confidence: float,
    audio_quality: AudioQuality,
) -> FactDiagnosis | None:
    negative = [item for item in observations if item.observation_id != "usable_strength"]
    if not negative:
        return None
    selected = max(
        negative,
        key=lambda item: (
            item.importance * 0.38
            + item.confidence * 0.22
            + min(1.0, len(item.supporting_fact_ids) / 3) * 0.14
            + item.trainability * 0.16
            + item.distinctiveness * 0.1
            - item.contradiction_penalty
        ),
    )
    fact_lookup = _fact_map(facts)
    supporting = [fact_lookup[fact_id] for fact_id in selected.supporting_fact_ids if fact_id in fact_lookup]
    fact_ids = tuple(fact.fact_id for fact in supporting)
    support_sources = len({fact.source for fact in supporting})
    quality = 0.82 if audio_quality.usable else 0.58
    diagnosis_confidence = _clamp01(min(confidence or selected.confidence, selected.confidence) * 0.64 + quality * 0.16 + min(1.0, support_sources / 2) * 0.12 + min(1.0, len(fact_ids) / 3) * 0.08 - selected.contradiction_penalty)
    uncertainty_note = None
    if selected.contradiction_penalty:
        uncertainty_note = "A competing signal reduces certainty, so the report focuses only on the strongest supported pattern."
    drill_id = _valid_drill_for_target(selected.target_behavior, selected.recommended_drill_categories, coaching)
    title_map = {
        "thin_proof": "Clear Direction, Thin Proof",
        "hesitation_control": "Interrupted Control",
        "unclear_path": "Unclear Answer Path",
        "weak_close": "Unfinished Landing",
        "flat_presence": "Low-Contrast Delivery",
    }
    mechanism_map = {
        "thin_proof": "A clear claim arrived, but the answer did not attach a concrete situation, number, or example to make the claim feel proven.",
        "hesitation_control": "The answer developed while timing interruptions were audible, so the delivery exposed the search process.",
        "unclear_path": "The answer asked the listener to infer the structure instead of hearing a clean point, support, and close.",
        "weak_close": "The recording gave away finality near the end, so the answer stopped before it fully landed.",
        "flat_presence": "The important words were not separated enough from the surrounding delivery, so the message had less pull.",
    }
    return FactDiagnosis(
        diagnosis_id=selected.observation_id,
        title=title_map.get(selected.observation_id, selected.title),
        one_sentence_pattern=selected.observed_pattern,
        mechanism=mechanism_map.get(selected.observation_id, selected.observed_pattern),
        listener_consequence=selected.listener_effect,
        primary_observation_ids=(selected.observation_id,),
        secondary_observation_ids=tuple(item.observation_id for item in observations if item.observation_id != selected.observation_id)[:2],
        related_dimensions=selected.related_dimensions,
        confidence=round(diagnosis_confidence, 2),
        uncertainty_note=uncertainty_note,
        target_behavior=selected.target_behavior,
        recommended_drill_id=drill_id,
        fact_ids=fact_ids,
    )


def _earliest_fact_start(observation: FactObservation, fact_lookup: dict[str, RecordingFact]) -> int | None:
    starts = [
        fact_lookup[fact_id].start_ms
        for fact_id in observation.supporting_fact_ids
        if fact_id in fact_lookup and fact_lookup[fact_id].start_ms is not None
    ]
    return min(starts) if starts else None


def _listener_state_for_observation(
    observation: FactObservation,
    fact_lookup: dict[str, RecordingFact],
    index: int,
) -> ListenerState:
    start_ms = _earliest_fact_start(observation, fact_lookup)
    fact_ids = observation.supporting_fact_ids
    if observation.observation_id == "thin_proof":
        expectation = "The listener understands the claim and waits for proof."
        confidence = "The point is understandable, but belief has not fully arrived."
        processing = "Moderate: the listener is holding the claim open while searching for evidence."
        credibility = "Credibility stalls until an example or concrete result appears."
        certainty = "Certainty remains pending."
        engagement = "Engagement depends on whether proof arrives quickly."
        attention = "Attention stays on the gap between the claim and its demonstration."
        authority = "Authority reads as organized but not yet proven."
        trust = "Trust is being requested before it has been earned by evidence."
        confusion = "Low confusion; the issue is proof, not basic comprehension."
        tension = "Mild tension: the listener is waiting for the reason to believe."
        reaction = "They wait for a specific situation, number, or named example."
        shift = "Understanding arrives before conviction."
    elif observation.observation_id == "hesitation_control":
        expectation = "The listener expects the thought to keep moving cleanly."
        confidence = "Confidence in the content can remain, but confidence in control dips."
        processing = "Higher: attention splits between the idea and the search for wording."
        credibility = "Credibility becomes less settled during the interruption."
        certainty = "Certainty wavers while the delivery searches."
        engagement = "Engagement is interrupted because delivery becomes the object of attention."
        attention = "Attention shifts from what is being said to how it is being found."
        authority = "Authority signal weakens at the exact moment control should stay quiet."
        trust = "Trust in preparation becomes less automatic."
        confusion = "Brief confusion appears around the interrupted stretch."
        tension = "Tension rises because the listener can hear effort."
        reaction = "They wait for the speaker to regain the sentence."
        shift = "The listener briefly hears the work behind the answer."
    elif observation.observation_id == "unclear_path":
        expectation = "The listener expects a route through point, support, and close."
        confidence = "Confidence is delayed because the path has to be assembled."
        processing = "High: the listener spends effort sorting the answer."
        credibility = "Credibility depends on whether the structure becomes visible soon."
        certainty = "Certainty stays low until the main point is separated from support."
        engagement = "Engagement becomes more fragile as the listener works."
        attention = "Attention moves to organizing the answer instead of weighing the point."
        authority = "Authority signal weakens because the speaker is not visibly leading the path."
        trust = "Trust in the speaker's control drops before the content can be judged."
        confusion = "Confusion rises around the answer route."
        tension = "Tension is cognitive rather than emotional: the listener has to do extra sorting."
        reaction = "They look for the sentence that tells them where the answer is going."
        shift = "The listener has to build the structure the speaker did not supply."
    elif observation.observation_id == "weak_close":
        expectation = "The listener expects the final sentence to tell them what to keep."
        confidence = "Confidence created earlier loses some finality."
        processing = "Moderate: the listener has to decide whether the answer is finished."
        credibility = "Credibility is not erased, but the last impression is softer."
        certainty = "Certainty drops at the point where it should peak."
        engagement = "Engagement ends without a clean final cue."
        attention = "Attention moves to the unfinished edge of the answer."
        authority = "Authority signal weakens because the ending does not close the loop."
        trust = "Trust in the recommendation is less easy to carry away."
        confusion = "Low to moderate: the listener understands the answer but not the final landing."
        tension = "A small unresolved tension remains at the end."
        reaction = "They mentally complete the ending instead of receiving it."
        shift = "The final impression becomes less certain than the earlier answer."
    elif observation.observation_id == "flat_presence":
        expectation = "The listener expects the important idea to be marked."
        confidence = "Confidence in meaning may remain, but memorability drops."
        processing = "Moderate: the listener has to infer which words matter most."
        credibility = "Credibility is steady but low in force."
        certainty = "Certainty about priority is weaker."
        engagement = "Engagement fades because the delivery does not create contrast."
        attention = "Attention has fewer cues for what to hold onto."
        authority = "Authority signal becomes quieter than the content needs."
        trust = "Trust is not damaged; urgency is."
        confusion = "Low confusion, but low salience."
        tension = "Little emotional tension; the cost is attention."
        reaction = "They understand the point without feeling a clear reason to remember it."
        shift = "Meaning lands, but priority does not."
    else:
        expectation = "The listener expects the speaker to keep the useful behaviour available."
        confidence = observation.observed_pattern
        processing = "Lower: this behaviour makes the answer easier to receive."
        credibility = "Credibility rises where the behaviour appears."
        certainty = "Certainty improves around this supported behaviour."
        engagement = "Engagement becomes easier to maintain."
        attention = "Attention can stay on the idea."
        authority = "Authority signal improves locally."
        trust = "Trust becomes easier because the behaviour is visible."
        confusion = "Confusion drops."
        tension = "Tension reduces."
        reaction = "They follow the next idea with less effort."
        shift = observation.listener_effect
    return ListenerState(
        state_id=f"listener_state_{index + 1}",
        observation_id=observation.observation_id,
        source_fact_ids=fact_ids,
        start_ms=start_ms,
        current_expectation=expectation,
        current_confidence=confidence,
        processing_load=processing,
        credibility=credibility,
        certainty=certainty,
        engagement=engagement,
        attention=attention,
        authority_signal=authority,
        trust_signal=trust,
        momentary_confusion=confusion,
        emotional_tension=tension,
        predicted_next_reaction=reaction,
        perception_shift=shift,
        confidence=round(observation.confidence * 0.72 + observation.distinctiveness * 0.18 + min(1.0, len(fact_ids) / 3) * 0.1, 2),
    )


def _reconstruct_listener_perception(
    observations: list[FactObservation],
    facts: list[RecordingFact],
    diagnosis: FactDiagnosis | None,
    report_confidence: float,
) -> ListenerPerceptionReconstruction:
    fact_lookup = _fact_map(facts)
    relevant = [
        observation
        for observation in observations
        if observation.supporting_fact_ids
    ]
    relevant.sort(
        key=lambda observation: (
            _earliest_fact_start(observation, fact_lookup) is None,
            _earliest_fact_start(observation, fact_lookup) or 10**9,
            -observation.importance,
        )
    )
    states = tuple(_listener_state_for_observation(observation, fact_lookup, index) for index, observation in enumerate(relevant))
    primary_state = None
    if diagnosis:
        primary_state = next((state for state in states if state.observation_id == diagnosis.diagnosis_id), None)
    if primary_state is None and states:
        primary_state = states[0]
    observation_confidence = round(sum(item.confidence for item in observations) / max(len(observations), 1), 2)
    diagnosis_confidence = diagnosis.confidence if diagnosis else min(report_confidence, 0.58)
    perception_confidence = round(
        sum(state.confidence for state in states) / max(len(states), 1) * 0.72 + diagnosis_confidence * 0.28,
        2,
    )
    timestamped = [fact for fact in facts if _timestamps_safe(fact.start_ms, fact.end_ms, fact.timestamp_source)]
    timeline_confidence = round(sum(fact.confidence for fact in timestamped) / max(len(timestamped), 1), 2) if timestamped else 0.0
    return ListenerPerceptionReconstruction(
        states=states,
        primary_state=primary_state,
        observation_confidence=observation_confidence,
        diagnosis_confidence=round(diagnosis_confidence, 2),
        perception_confidence=perception_confidence,
        report_confidence=round(report_confidence, 2),
        timeline_confidence=timeline_confidence,
    )


def _state_for(reconstruction: ListenerPerceptionReconstruction | None, observation_id: str) -> ListenerState | None:
    if not reconstruction:
        return None
    return next((state for state in reconstruction.states if state.observation_id == observation_id), None)


def _valid_drill_for_target(target_behavior: str, categories: tuple[str, ...], coaching: CoachingEngine | None) -> str | None:
    if not coaching:
        return None
    library = {drill.drill_id: drill for drill in coaching.drill_library}
    ordered_ids = []
    for candidate in (coaching.selected_interventions.primary_drill, coaching.selected_interventions.secondary_drill):
        if candidate:
            ordered_ids.append(candidate.drill_id)
    ordered_ids.extend(candidate.drill_id for candidate in coaching.future_training_queue)
    ordered_ids.extend(drill.drill_id for drill in coaching.drill_library)
    for drill_id in dict.fromkeys(ordered_ids):
        drill = library.get(drill_id)
        if not drill:
            continue
        if target_behavior in drill.target_behaviours or drill.category in categories:
            return drill.drill_id
    return None


def _drill_definition(drill_id: str | None, coaching: CoachingEngine | None):
    if not drill_id or not coaching:
        return None
    return next((item for item in coaching.drill_library if item.drill_id == drill_id), None)


def _evidence_card_from_fact_observation(
    observation: FactObservation,
    facts: dict[str, RecordingFact],
    *,
    diagnosis: FactDiagnosis | None,
    reconstruction: ListenerPerceptionReconstruction | None = None,
) -> ReportEvidenceCard:
    source_facts = [facts[fact_id] for fact_id in observation.supporting_fact_ids if fact_id in facts]
    first = source_facts[0] if source_facts else None
    timestamp = [first.start_ms, first.end_ms] if first and _timestamps_safe(first.start_ms, first.end_ms, first.timestamp_source) else None
    start_ms = timestamp[0] if timestamp else None
    end_ms = timestamp[1] if timestamp else None
    state = _state_for(reconstruction, observation.observation_id)
    first_fact = first.observed_behavior if first else observation.observed_pattern
    link = (
        "This is the exhibit that proves the main diagnosis."
        if diagnosis and observation.observation_id in diagnosis.primary_observation_ids
        else "This exhibit shows a supporting behaviour that changes the listener's read in this recording."
    )
    if observation.observation_id == "thin_proof":
        what_happened = first_fact
        why = "Because the listener receives the claim before receiving proof, belief has to rest on trust rather than demonstration."
    elif observation.observation_id == "hesitation_control":
        what_happened = first_fact
        why = "Because the timing interruption is audible, attention briefly moves away from the idea and toward the delivery."
    elif observation.observation_id == "unclear_path":
        what_happened = first_fact
        why = "Because the path is not separated cleanly, the listener spends effort organizing the answer instead of judging the point."
    elif observation.observation_id == "weak_close":
        what_happened = first_fact
        why = "Because the final sentence does not close the loop, the listener has to supply the ending themselves."
    elif observation.observation_id == "flat_presence":
        what_happened = first_fact
        why = "Because the important words are not marked, the listener gets meaning without a strong cue for what to remember."
    else:
        what_happened = observation.observed_pattern
        why = state.perception_shift if state else observation.listener_effect
    why = f"{why} {link}"
    listener_interpretation = state.perception_shift if state else observation.listener_effect
    return ReportEvidenceCard(
        evidence_id=f"fact_ev_{observation.observation_id}",
        id=observation.observation_id,
        trait=observation.related_dimensions[0] if observation.related_dimensions else None,
        dimension=observation.related_dimensions[0] if observation.related_dimensions else None,
        direction="positive" if observation.observation_id == "usable_strength" else "negative",
        signal=observation.title,
        what_happened=_clean_report_text(what_happened),
        why_it_matters=_clean_report_text(why),
        listener_interpretation=_clean_report_text(listener_interpretation),
        related_dimension=observation.related_dimensions[0] if observation.related_dimensions else "Authority",
        confidence=observation.confidence,
        source_metrics=[],
        recording_fact_ids=list(observation.supporting_fact_ids),
        start_ms=start_ms,
        end_ms=end_ms,
        timestamp=timestamp,
    )


def _fact_evidence_cards(
    observations: list[FactObservation],
    facts: list[RecordingFact],
    diagnosis: FactDiagnosis | None,
    reconstruction: ListenerPerceptionReconstruction | None = None,
) -> list[ReportEvidenceCard]:
    fact_lookup = _fact_map(facts)
    primary_ids = set(diagnosis.primary_observation_ids if diagnosis else ())
    ordered = sorted(
        observations,
        key=lambda item: (
            item.observation_id in primary_ids,
            item.importance,
            item.confidence,
            item.distinctiveness,
        ),
        reverse=True,
    )
    cards: list[ReportEvidenceCard] = []
    seen_text: set[str] = set()
    for observation in ordered:
        if len(cards) >= 3:
            break
        card = _evidence_card_from_fact_observation(observation, fact_lookup, diagnosis=diagnosis, reconstruction=reconstruction)
        normalized = _normalize_copy(" ".join([card.signal, card.what_happened, card.listener_interpretation]))
        if normalized in seen_text:
            continue
        seen_text.add(normalized)
        cards.append(card)
    return cards


def _normalize_copy(text: str | None) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    tokens = [
        token
        for token in cleaned.split()
        if token not in {"the", "a", "an", "and", "or", "to", "of", "in", "for", "with", "that", "this", "it", "is", "are", "was", "were"}
    ]
    return " ".join(tokens)


def _copy_similarity(a: str | None, b: str | None) -> float:
    a_tokens = set(_normalize_copy(a).split())
    b_tokens = set(_normalize_copy(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


_REPORT_FORBIDDEN_PHRASES = (
    "concrete anchors",
    "process detail",
    "the message explained the idea more than it proved it",
    "change the behaviour",
    "useful signal is behavioural",
    "the ceiling is",
    "selected focus",
    "deterministic drill",
    "winning diagnosis",
    "supported by x evidence items",
    "observed as",
    "hypothesis",
    "backend",
    "good communication",
    "effective communication",
    "strong communication",
    "clear communication",
    "powerful communication",
    "better communication",
    "repeatable strength",
    "local change in control",
)


def _meaning_signature(text: str | None) -> str:
    normalized = _normalize_copy(text)
    tokens = set(normalized.split())
    if {"proof", "claim"} & tokens and {"believe", "belief", "evidence", "example", "demonstration", "picture"} & tokens:
        return "waiting_for_proof"
    if {"hesitation", "search", "delivery", "wording", "interruption", "interrupted"} & tokens:
        return "delivery_search_visible"
    if {"route", "path", "assemble", "organize", "structure", "sorting"} & tokens:
        return "listener_builds_route"
    if {"ending", "final", "close", "takeaway", "recall", "carry"} & tokens:
        return "unfinished_landing"
    if {"emphasis", "contrast", "priority", "remember", "marked"} & tokens:
        return "low_priority_signal"
    if {"opening", "direction", "frame"} & tokens:
        return "opening_orientation"
    if {"pace", "settled", "timing", "pause", "silence"} & tokens:
        return "timing_control"
    return normalized[:80]


def _semantic_similarity(a: str | None, b: str | None) -> float:
    left = _meaning_signature(a)
    right = _meaning_signature(b)
    if left and right and left == right and left not in {_normalize_copy(a)[:80], _normalize_copy(b)[:80]}:
        return 1.0
    return _copy_similarity(a, b)


def _contains_forbidden_report_phrase(text: str | None) -> bool:
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in _REPORT_FORBIDDEN_PHRASES)


def _fact_evidence_ids(cards: list[ReportEvidenceCard]) -> list[str]:
    return [card.evidence_id for card in cards[:3]]


def _fact_mirror(
    scores: Scores,
    authority_type: ReportAuthorityType,
    diagnosis: FactDiagnosis | None,
    cards: list[ReportEvidenceCard],
    confidence_label: str,
    reconstruction: ListenerPerceptionReconstruction | None = None,
) -> ReportMirror:
    evidence_ids = _fact_evidence_ids(cards)
    strength = next((card for card in cards if card.direction == "positive"), None)
    if diagnosis:
        state = reconstruction.primary_state if reconstruction else None
        strength_phrase = strength.listener_interpretation if strength else "One supported behaviour makes the answer easier to receive."
        if diagnosis.diagnosis_id == "thin_proof":
            headline = "Understood Before Believed"
            identity = "The listener gets the point early, then waits for proof that would make the claim believable."
            tension = "What landed was direction; what held the answer back was the missing example after the claim."
        elif diagnosis.diagnosis_id == "hesitation_control":
            headline = "The Idea Survived The Search"
            identity = "The listener can follow the content, but the delivery exposes the moment where the next words are being found."
            tension = "What landed was the answer; what held it back was attention shifting from the idea to the search process."
        elif diagnosis.diagnosis_id == "unclear_path":
            headline = "The Listener Had To Build The Route"
            identity = "The listener hears pieces of an answer, but has to assemble the point, support, and close themselves."
            tension = "What landed was individual meaning; what held the answer back was the missing route through it."
        elif diagnosis.diagnosis_id == "weak_close":
            headline = "The Ending Softened The Answer"
            identity = "The listener receives a usable answer, then the final moment gives them less certainty than the earlier structure created."
            tension = "What landed was the plan; what held it back was the absence of a clean final takeaway."
        elif diagnosis.diagnosis_id == "flat_presence":
            headline = "Meaning Landed Without Priority"
            identity = "The listener understands the words, but the delivery does not clearly mark which idea should stay with them."
            tension = "What landed was comprehension; what held it back was low contrast on the important words."
        else:
            headline = diagnosis.title
            identity = state.perception_shift if state else diagnosis.listener_consequence
            tension = strength_phrase
    else:
        headline = "This recording shows a mostly controlled authority signal."
        identity = "The report found more repeatable strengths than limiting behaviours in this sample."
        tension = "The next improvement is to preserve the strongest behaviour under a harder prompt."
    return ReportMirror(
        headline=_clean_report_text(headline),
        identity_read=_clean_report_text(identity),
        one_line_identity_read=_clean_report_text(identity),
        core_tension=_clean_report_text(tension),
        emotional_tone=_emotional_tone(scores),
        authority_type=authority_type.label,
        confidence_label=confidence_label,  # type: ignore[arg-type]
        confidence_level=confidence_label,  # type: ignore[arg-type]
        evidence_ids=evidence_ids,
    )


def _fact_report_diagnosis(
    scores: Scores,
    diagnosis: FactDiagnosis | None,
    cards: list[ReportEvidenceCard],
    reconstruction: ListenerPerceptionReconstruction | None = None,
) -> ReportDiagnosis:
    dims = _ordered_dimensions(scores)
    strongest = DIMENSION_LABELS[dims[0][0]]
    limiter = diagnosis.related_dimensions[0] if diagnosis and diagnosis.related_dimensions else DIMENSION_LABELS[sorted(_dimension_scores(scores).items(), key=lambda item: item[1])[0][0]]
    evidence_ids = _fact_evidence_ids(cards)
    if diagnosis:
        state = reconstruction.primary_state if reconstruction else None
        if diagnosis.diagnosis_id == "thin_proof":
            core = "The claim arrives before demonstration, so the listener has to believe the speaker before they can see the evidence."
            consequence = "The listener understands the message, but conviction stays pending until a concrete example appears."
        elif diagnosis.diagnosis_id == "hesitation_control":
            core = "The sentence flow exposes the act of finding the next words, so delivery becomes part of the listener's task."
            consequence = "The listener briefly evaluates control instead of staying fully inside the idea."
        elif diagnosis.diagnosis_id == "unclear_path":
            core = "The answer does not make its route visible early enough, so the listener has to sort the structure while listening."
            consequence = "That extra sorting reduces the speaker's sense of control before the content gets a fair hearing."
        elif diagnosis.diagnosis_id == "weak_close":
            core = "The ending fails to convert the last idea into a final takeaway, so the listener is left to infer the landing."
            consequence = "The final impression carries less certainty than the answer had earned earlier."
        elif diagnosis.diagnosis_id == "flat_presence":
            core = "The delivery gives similar weight to too many words, so the listener receives meaning without priority."
            consequence = "The idea becomes easier to understand than to remember."
        else:
            core = diagnosis.mechanism
            consequence = state.perception_shift if state else diagnosis.listener_consequence
    else:
        core = "No single limiter was strong enough to become the report's main diagnosis."
        consequence = "Treat this as a baseline and retest with a harder prompt to expose the next trainable edge."
    return ReportDiagnosis(
        strongest_dimension=strongest,
        limiting_dimension=limiter,
        primary_strength_dimension=strongest,
        primary_limiting_dimension=limiter,
        core_behavioural_pattern=_clean_report_text(core),
        core_pattern=_clean_report_text(core),
        social_consequence=_clean_report_text(consequence),
        supporting_evidence_ids=evidence_ids,
        evidence_ids=evidence_ids,
        severity=_severity(_dimension_scores(scores).get(limiter.lower(), 60) if limiter else 60),  # type: ignore[arg-type]
    )


def _fact_perception_map(
    diagnosis: FactDiagnosis | None,
    cards: list[ReportEvidenceCard],
    confidence: float,
    scenario: str,
    reconstruction: ListenerPerceptionReconstruction | None = None,
) -> ReportPerceptionMap:
    evidence_ids = _fact_evidence_ids(cards)
    if not diagnosis:
        reads = {
            "first_impression": _read("Controlled baseline", "The first impression is that this recording gives the listener enough control to follow the speaker.", evidence_ids, confidence)
        }
        if scenario == "interview":
            reads["interview_read"] = _read("Interview baseline", "In an interview, this would read as a usable baseline, with the next pass needing one sharper answer target.", evidence_ids, confidence)
        return ReportPerceptionMap(**reads)
    reads: dict[str, ReportPerceptionRead] = {}
    if diagnosis.diagnosis_id == "thin_proof":
        reads["first_impression"] = _read("Opening assumption", "The listener assumes the answer has direction because the main idea appears early.", evidence_ids, confidence)
        reads["professional_read"] = _read("Colleague read", "A colleague can understand the recommendation, but would pause before acting because the evidence has not arrived yet.", evidence_ids, confidence)
        reads["interview_read"] = _read("Hiring read", "An interviewer hears self-awareness, then waits for the example that would prove the claim under pressure.", evidence_ids, confidence)
    elif diagnosis.diagnosis_id == "hesitation_control":
        reads["first_impression"] = _read("Opening assumption", "The listener expects a controlled answer, then notices the delivery working to catch the next phrase.", evidence_ids, confidence)
        reads["professional_read"] = _read("Colleague read", "A colleague may still trust the substance, but the interrupted timing makes the speaker sound less prepared in the moment.", evidence_ids, confidence)
        reads["leadership_read"] = _read("Leadership signal", "The leadership signal weakens when silence is not owned before the next claim.", evidence_ids, confidence)
        reads["emotional_read"] = _read("Emotional experience", "Emotionally, the listener feels a small rise in pressure because the effort becomes audible.", evidence_ids, confidence)
    elif diagnosis.diagnosis_id == "unclear_path":
        reads["first_impression"] = _read("Opening assumption", "The listener waits for the route through the answer and does not receive it soon enough.", evidence_ids, confidence)
        reads["professional_read"] = _read("Colleague read", "A colleague has to organize the point while listening, which slows judgment of the idea itself.", evidence_ids, confidence)
        reads["leadership_read"] = _read("Leadership signal", "The leadership signal becomes weaker because the next step is not made obvious before more detail arrives.", evidence_ids, confidence)
    elif diagnosis.diagnosis_id == "weak_close":
        reads["first_impression"] = _read("Opening assumption", "The listener initially assumes the answer has a workable plan because the route is easy to follow.", evidence_ids, confidence)
        reads["professional_read"] = _read("Colleague read", "A colleague hears a usable plan, but the final edge makes the next action feel less sealed.", evidence_ids, confidence)
        reads["leadership_read"] = _read("Leadership signal", "The leadership signal drops at the end because finality is where command should peak.", evidence_ids, confidence)
        reads["interview_read"] = _read("Hiring read", "An interviewer would remember the answer more easily if the last sentence became a completed takeaway.", evidence_ids, confidence)
        reads["social_status_read"] = _read("Social read", "Socially, the speaker seems thoughtful but slightly less decisive because the listener has to finish the final beat.", evidence_ids, confidence)
    elif diagnosis.diagnosis_id == "flat_presence":
        reads["first_impression"] = _read("Opening assumption", "The listener assumes the speaker is controlled, then receives too few cues about which idea matters most.", evidence_ids, confidence)
        reads["emotional_read"] = _read("Emotional experience", "Emotionally, the delivery feels low pressure, but also gives the listener little reason to lean in.", evidence_ids, confidence)
        reads["persuasion_read"] = _read("Persuasion read", "For persuasion, the point needs contrast so the listener can feel the hierarchy of the idea.", evidence_ids, confidence)

    if scenario == "interview" and "interview_read" not in reads:
        reads["interview_read"] = _read("Hiring read", "In an interview, the listener looks for a complete answer shape: point, proof, and clean close.", evidence_ids, confidence)
    ordered = list(reads.items())[:5]
    return ReportPerceptionMap(**{key: value for key, value in ordered})


def _timeline_story(moment_type: str, reconstruction: ListenerPerceptionReconstruction | None) -> tuple[str, str, str, str, str]:
    primary = reconstruction.primary_state if reconstruction else None
    expectation = primary.current_expectation if primary else "The listener is tracking whether the answer stays easy to follow."
    if moment_type in {"strong_opening", "weak_opening"}:
        return (
            expectation,
            "The opening either gives or delays the route into the answer.",
            "The listener decides how much structure they will have to supply.",
            "Authority rises when the route is visible and drops when the listener has to search for it.",
            "That first read shapes how generously the rest of the answer is received.",
        )
    if moment_type in {"hesitation_cluster", "filler_cluster", "confidence_drop", "rushing_moment"}:
        return (
            expectation,
            "Delivery control becomes less steady in this stretch.",
            "The listener briefly notices the delivery instead of only the idea.",
            "Authority drops because control is being evaluated in real time.",
            "The next sentence has to re-earn ease.",
        )
    if moment_type in {"confidence_recovery", "most_composed_moment", "pause_ownership_moment"}:
        return (
            expectation,
            "Timing becomes more settled here.",
            "The listener can return attention to the point.",
            "Authority rises because the delivery stops asking for attention.",
            "The following idea becomes easier to trust.",
        )
    if moment_type in {"weak_closing", "strong_closing"}:
        return (
            "The listener expects the ending to tell them what to carry away.",
            "The final stretch either lands the takeaway or leaves the edge unfinished.",
            "The listener decides whether the answer feels complete.",
            "Authority depends on whether certainty peaks at the end.",
            "That final impression is what the listener carries forward.",
        )
    if moment_type in {"monotone_stretch", "high_presence_moment", "most_persuasive_moment"}:
        return (
            expectation,
            "Emphasis changes how clearly the important words stand out.",
            "The listener either gets a priority cue or has to infer it.",
            "Authority rises when vocal contrast marks the point.",
            "The marked idea becomes easier to remember.",
        )
    return (
        expectation,
        "A local behaviour changes how the answer is received.",
        "The listener updates their read of control in this stretch.",
        "Authority changes because the delivery pattern changes.",
        "That local read carries into the next part of the answer.",
    )


def _fact_timeline(
    moments: list[Moment],
    cards: list[ReportEvidenceCard],
    duration_ms: int,
    scenario: str,
    reconstruction: ListenerPerceptionReconstruction | None = None,
) -> list[ReportTimelineItem]:
    if duration_ms and duration_ms < 25000:
        return []
    if scenario == "benchmark":
        moments = [moment for moment in moments if moment.type != "most_improved_section"]
    evidence_ids = _fact_evidence_ids(cards)
    if not evidence_ids:
        return []
    label_map = {
        "rushing_moment": ("Pace increased", "Pace became faster in this local stretch."),
        "hesitation_cluster": ("Hesitation increased", "Hesitation events became more concentrated here."),
        "filler_cluster": ("Lexical fillers clustered", "Filler words appeared close together in this stretch."),
        "confidence_drop": ("Composure dropped", "The delivery became less settled than the surrounding section."),
        "confidence_recovery": ("Composure recovered", "The delivery became more settled after the previous stretch."),
        "weak_closing": ("Closing lost finality", "The ending carried less finality than the answer needed."),
        "strong_closing": ("Close became decisive", "The final stretch supported a cleaner landing."),
        "strong_opening": ("Opening set direction", "The opening made the answer path visible early."),
        "weak_opening": ("Opening delayed the point", "The answer took longer to give the listener a clear direction."),
        "monotone_stretch": ("Emphasis flattened", "Pitch and energy contrast became lower here."),
        "pause_ownership_moment": ("Pause sounded owned", "The silence landed between thoughts rather than interrupting one."),
        "high_presence_moment": ("Emphasis increased", "Energy and emphasis made this stretch easier to notice."),
        "most_commanding_moment": ("Command strengthened", "Command cues were locally stronger here."),
        "most_composed_moment": ("Pace and timing settled", "Composure cues were locally stronger here."),
        "most_persuasive_moment": ("Persuasive contrast increased", "Presence and command aligned more strongly here."),
        "strongest_moment": ("Structure and delivery aligned", "This stretch showed the strongest local combination of control and clarity."),
        "most_costly_sentence": ("Local cost increased", "This span carried the clearest local cost to clarity or composure."),
    }
    items: list[ReportTimelineItem] = []
    for moment in sorted(moments, key=lambda item: (item.importance_score, item.confidence), reverse=True):
        if len(items) >= 3:
            break
        if moment.type not in label_map:
            continue
        if moment.end_ms <= moment.start_ms or moment.confidence < 0.45:
            continue
        if moment.timestamp_source not in {"real", "segment"}:
            continue
        headline, summary = label_map[moment.type]
        expectation, behaviour, interpretation, authority_impact, carry_forward = _timeline_story(moment.type, reconstruction)
        linked = [evidence_id for evidence_id in evidence_ids if evidence_id in set(moment.supporting_evidence_ids)] or evidence_ids[:1]
        items.append(
            ReportTimelineItem(
                moment_id=moment.moment_id,
                type=moment.type,
                priority=moment.priority,
                headline=headline,
                summary=_clean_report_text(f"Expectation: {expectation} Behaviour: {summary if summary else behaviour}"),
                listener_interpretation=_clean_report_text(f"Interpretation: {interpretation}"),
                why_it_matters=_clean_report_text(f"Authority impact: {authority_impact} Carry-forward: {carry_forward}"),
                dimension_impact=moment.dimension_impact,
                confidence=moment.confidence,
                start_ms=moment.start_ms,
                end_ms=moment.end_ms,
                timestamp_source=moment.timestamp_source,
                evidence_ids=linked,
                supporting_metrics=[_plain_metric_label(metric) for metric in moment.supporting_metrics],
                transcript_span=moment.transcript_span if moment.timestamp_source in {"real", "segment"} else None,
                word_ids=moment.word_ids if moment.timestamp_source in {"real", "segment"} else [],
                scenario_relevance=moment.scenario_relevance,
                coaching_relevance=moment.coaching_relevance,
                importance_score=moment.importance_score,
                moment_group=_moment_group(moment.type),
                severity=moment.severity,
                preview_visible_free=moment.preview_visible_free,
            )
        )
    return items


def _fact_dimension_reports(scores: Scores, facts: list[RecordingFact], cards: list[ReportEvidenceCard], confidence: float) -> dict[str, ReportDimensionReport]:
    dims = _dimension_scores(scores)
    card_by_fact = {
        fact_id: card.evidence_id
        for card in cards
        for fact_id in card.recording_fact_ids
    }
    reports: dict[str, ReportDimensionReport] = {}
    for dimension, score in dims.items():
        label = DIMENSION_LABELS[dimension]
        related = [fact for fact in facts if label in fact.related_dimensions]
        positive = [fact for fact in related if fact.fact_type in {"clear_opening_claim", "concrete_example", "reinforced_ending", "decisive_ending", "pace_stabilization", "owned_pause", "strong_emphasis", "strong_local_structure"}]
        limiting = [fact for fact in related if fact.fact_type not in {fact.fact_type for fact in positive}]
        why: list[str] = []
        if positive:
            why.append(f"Strongest supporting fact: {positive[0].observed_behavior}")
        if limiting:
            why.append(f"Strongest limiting fact: {limiting[0].observed_behavior}")
        if not why:
            why.append("There is limited dimension-specific evidence in this recording, so this score should be read as a broad baseline.")
        linked = list(dict.fromkeys(card_by_fact[fact.fact_id] for fact in related if fact.fact_id in card_by_fact))
        if not linked:
            linked = _fact_evidence_ids(cards)[:1]
        reports[dimension] = ReportDimensionReport(
            dimension=dimension,
            score=score,
            label=label,
            meaning=DIMENSION_MEANING[dimension],
            why=[_clean_report_text(item) for item in why[:2]],
            listener_consequence=DIMENSION_CONSEQUENCE[dimension],
            one_improvement_cue=_dimension_improvement_cue(dimension, limiting[:1]),
            linked_evidence=linked,
            confidence=round(confidence, 2),
        )
    return reports


def _dimension_improvement_cue(dimension: str, limiting: list[RecordingFact]) -> str:
    if limiting:
        fact = limiting[0]
        if fact.fact_type == "claim_without_proof":
            return "Attach one specific example to the main claim."
        if fact.fact_type in {"acoustic_hesitation", "pause_cluster", "lexical_filler"}:
            return "Pause, then restart on a complete clause."
        if fact.fact_type in {"abrupt_ending", "rising_ending"}:
            return "Make the final sentence a takeaway and stop cleanly."
        if fact.fact_type == "flat_emphasis":
            return "Give the most important word a controlled contrast."
    return DIMENSION_CUE[dimension]


def _fact_hidden_cost(diagnosis: FactDiagnosis | None, cards: list[ReportEvidenceCard], confidence: float) -> ReportHiddenCost:
    evidence_ids = _fact_evidence_ids(cards)
    if not diagnosis:
        consequence = "The downstream risk is plateau: without a sharper target, the next recording may repeat the same baseline instead of testing a specific behaviour."
        dimension = "Authority"
        cost_id = "baseline_plateau"
    elif diagnosis.diagnosis_id == "thin_proof":
        consequence = "The quiet loss is commitment: the listener can agree with the topic and still withhold belief because no proof lets them picture it."
        dimension = "Persuasion"
        cost_id = "credibility_leakage"
    elif diagnosis.diagnosis_id == "hesitation_control":
        consequence = "The quiet loss is ease: the listener spends attention on the search for words that should have stayed invisible."
        dimension = "Composure"
        cost_id = "pressure_leakage"
    elif diagnosis.diagnosis_id == "unclear_path":
        consequence = "The quiet loss is judgment speed: the listener cannot evaluate the point until they have built the route through it."
        dimension = "Structure"
        cost_id = "listener_effort"
    elif diagnosis.diagnosis_id == "weak_close":
        consequence = "The quiet loss is recall: the listener leaves with a softer final idea than the answer had earned."
        dimension = "Command"
        cost_id = "recency_loss"
    else:
        consequence = "The quiet loss is priority: the listener can follow the words without knowing which one deserves to stay."
        dimension = "Presence"
        cost_id = "memorability_loss"
    return ReportHiddenCost(dimension=dimension, cost_id=cost_id, consequence=consequence, evidence_ids=evidence_ids, confidence=confidence)


def _fact_highest_fix(diagnosis: FactDiagnosis | None, coaching: CoachingEngine | None, cards: list[ReportEvidenceCard]) -> ReportHighestLeverageFix:
    evidence_ids = _fact_evidence_ids(cards)
    fallback_drill_id = coaching.selected_interventions.primary_drill.drill_id if coaching and coaching.selected_interventions.primary_drill else None
    if not diagnosis and not fallback_drill_id and coaching and coaching.drill_library:
        fallback_drill_id = next((drill.drill_id for drill in coaching.drill_library if drill.drill_id == "answer_first_v1"), coaching.drill_library[0].drill_id)
    if not diagnosis and not fallback_drill_id:
        fallback_drill_id = "answer_first_v1"
    drill = _drill_definition(diagnosis.recommended_drill_id if diagnosis else fallback_drill_id, coaching)
    if diagnosis and diagnosis.diagnosis_id == "thin_proof":
        issue = "Add proof to the main claim"
        plain = "Stop moving from claim to explanation without a proof point; instead, make one claim and immediately attach one concrete example."
        why = "That change gives the listener something specific to believe, not just a reason to understand."
        action = "Say the claim in one sentence, then begin the next sentence with a concrete situation, number, or named example."
        success = "The next recording should make the main claim easier to picture."
    elif diagnosis and diagnosis.diagnosis_id == "hesitation_control":
        issue = "Own the pause before the thought"
        plain = "Stop filling the search moment with sound; instead, pause before the next claim and restart on a complete sentence."
        why = "That change hides the search process and lets the listener hear control before content resumes."
        action = "When you feel a repair coming, hold silence for half a beat, then restart the sentence cleanly."
        success = "The next recording should contain fewer audible search moments before important claims."
    elif diagnosis and diagnosis.diagnosis_id == "unclear_path":
        issue = "Put the answer path first"
        plain = "Stop adding support before the listener knows the route; instead, open with the answer, then give one proof, then close."
        why = "That change makes the listener feel led through the answer rather than asked to assemble it."
        action = "Use three sentences only: answer, proof, close."
        success = "The next recording should make the point, support, and close easy to separate."
    elif diagnosis and diagnosis.diagnosis_id == "weak_close":
        issue = "Turn the final line into a takeaway"
        plain = "Stop letting the ending trail out; instead, make the last sentence a complete takeaway and leave silence after it."
        why = "That change protects the final impression and makes the answer sound more decided."
        action = "Write one closing sentence before recording, say it, then stop."
        success = "The next recording should end with a sentence that sounds finished."
    elif diagnosis and diagnosis.diagnosis_id == "flat_presence":
        issue = "Mark the important word"
        plain = "Stop giving every word the same weight; instead, choose one word in the key sentence and give it controlled contrast."
        why = "That change tells the listener which idea deserves attention."
        action = "Repeat the key sentence three times, moving the emphasis to the one word that carries the point."
        success = "The next recording should make the key idea easier to remember."
    else:
        issue = "Retest with one sharper target"
        plain = "Keep the strongest behaviour and use the next recording to test one harder prompt."
        why = "That change turns a broad baseline into a trainable comparison."
        action = "Record the same prompt again and deliberately exaggerate one improvement target."
        success = "The next recording should reveal a clearer limiter or a measurable improvement."
    return ReportHighestLeverageFix(
        issue=issue,
        plain_english=plain,
        why_this_matters=why,
        expected_score_lift="medium" if diagnosis else "low",
        target_dimensions=drill.target_dimensions if drill else list(diagnosis.related_dimensions if diagnosis else ["Structure"]),
        first_drill_id=drill.drill_id if drill else fallback_drill_id,
        action_step=action,
        success_signal=success,
        duration_min=drill.estimated_duration_min if drill else 4,
        selection_score=round(diagnosis.confidence if diagnosis else 0.3, 3),
        evidence_ids=evidence_ids,
    )


def _instructions_for_drill(drill, diagnosis: FactDiagnosis | None) -> list[str]:
    if not drill:
        return ["Record the same prompt again for 45 to 60 seconds with one deliberate improvement target."]
    category = drill.category
    if category == "specificity":
        return [
            "Write one claim from the recording.",
            "Add one concrete proof point: a situation, number, named example, or visible result.",
            "Say only two sentences: claim, then proof.",
            "Repeat 6 reps, changing the proof point each time.",
        ]
    if category in {"pause_ownership", "composure", "pace_regulation", "filler_reduction"}:
        return [
            "Choose one sentence from the recording.",
            "Say the first half, pause silently for one beat, then finish the sentence.",
            "If a filler starts, stop, hold silence, and restart the clause.",
            "Complete 8 clean reps.",
        ]
    if category in {"opening_strength", "structure_compression"}:
        return [
            "Answer the prompt in exactly three sentences.",
            "Sentence 1 gives the answer.",
            "Sentence 2 gives one proof point.",
            "Sentence 3 closes with the takeaway. Complete 5 reps.",
        ]
    if category in {"closing_strength", "declarative_finality"}:
        return [
            "Write one final takeaway sentence.",
            "Say it with a slight downward finish.",
            "Hold silence for half a second after the last word.",
            "Repeat 8 reps without adding an extra sentence.",
        ]
    if category in {"dynamic_emphasis", "presence"}:
        return [
            "Pick the key sentence from the recording.",
            "Choose the single word that carries the point.",
            "Repeat the sentence 6 times, giving that word more contrast without speeding up.",
            "Record the final 2 reps and keep the one where the point is easiest to remember.",
        ]
    target = diagnosis.target_behavior.replace("_", " ") if diagnosis else "the target behaviour"
    return [
        f"Practise {target} for {drill.estimated_duration_min} minutes.",
        "Use short reps, not a full speech.",
        "Keep the best take and compare it with the original recording.",
    ]


def _fact_training(coaching: CoachingEngine | None, fix: ReportHighestLeverageFix, diagnosis: FactDiagnosis | None, cards: list[ReportEvidenceCard]) -> ReportTrainingPrescription:
    drill = _drill_definition(fix.first_drill_id, coaching)
    title = drill.title if drill else ("Answer First" if fix.first_drill_id == "answer_first_v1" else "Focused Retest")
    why = (
        f"Chosen because it trains {diagnosis.target_behavior.replace('_', ' ')}, which is the behaviour behind the main diagnosis."
        if diagnosis
        else "Chosen because the current recording is better used as a baseline than as a full diagnosis."
    )
    return ReportTrainingPrescription(
        drill_id=drill.drill_id if drill else fix.first_drill_id,
        title=title,
        why_chosen=why,
        instructions=_instructions_for_drill(drill, diagnosis),
        target_metrics=[],
        target_dimensions=drill.target_dimensions if drill else fix.target_dimensions,
        action_step=fix.action_step,
        expected_score_lift=fix.expected_score_lift,
        duration_min=drill.estimated_duration_min if drill else fix.duration_min,
        success_signal=f"{fix.success_signal} Retest on the same prompt immediately after the reps.",
        evidence_ids=_fact_evidence_ids(cards),
    )


def _fact_retest(fix: ReportHighestLeverageFix) -> ReportRetestPlan:
    focus = fix.issue or "same prompt retest"
    compare = fix.target_dimensions or ["authority"]
    return ReportRetestPlan(
        recommended_retest_after_days=3,
        focus_metric=focus,
        compare_metrics=compare,
        same_prompt_recommended=True,
        success_definition=fix.success_signal or "The target behaviour should be easier to hear.",
        evidence_ids=fix.evidence_ids,
    )


def _fact_primary_diagnostic(diagnosis: FactDiagnosis | None, cards: list[ReportEvidenceCard]) -> DiagnosticDiagnosis | None:
    if not diagnosis:
        return None
    return DiagnosticDiagnosis(
        diagnosis_id=diagnosis.diagnosis_id,
        diagnosis_name=diagnosis.mechanism,
        confidence=diagnosis.confidence,
        severity="high" if diagnosis.confidence >= 0.82 else "medium" if diagnosis.confidence >= 0.58 else "low",
        supporting_traits=[diagnosis.target_behavior],
        contradicting_traits=[],
        supporting_evidence_ids=_fact_evidence_ids(cards),
        supporting_moment_ids=[],
        affected_dimensions=list(diagnosis.related_dimensions),
    )


def _apply_report_repetition_guard(report: AuthorityReport) -> AuthorityReport:
    if report.perception_map:
        kept: dict[str, ReportPerceptionRead | None] = {}
        prior: list[str] = []
        for key, value in report.perception_map.model_dump().items():
            if not value or not value.get("text"):
                kept[key] = None
                continue
            text = value["text"]
            if _contains_forbidden_report_phrase(text) or any(_semantic_similarity(text, existing) >= 0.72 for existing in prior):
                kept[key] = None
                continue
            prior.append(text)
            kept[key] = ReportPerceptionRead(**value)
        report = report.model_copy(update={"perception_map": ReportPerceptionMap(**kept)})
    cards = []
    seen: set[str] = set()
    for card in report.evidence_chain:
        merged = _normalize_copy(" ".join([card.signal, card.what_happened, card.why_it_matters]))
        if merged in seen:
            continue
        if any(_contains_forbidden_report_phrase(text) for text in [card.signal, card.what_happened, card.why_it_matters, card.listener_interpretation]):
            continue
        seen.add(merged)
        cards.append(card)
    return report.model_copy(update={"evidence_chain": cards[:3], "timeline": report.timeline[:3]})


def _fact_led_report(
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
    moment_intelligence: MomentIntelligence | None,
    transcript: Transcript | None,
) -> AuthorityReport:
    del evidence
    confidence = min(max(psychological_inference.overall_inference_confidence, scores.score_confidence or 0.0), 0.95)
    if _is_insufficient_sample(metrics, audio_quality, duration_ms, confidence):
        return _insufficient_report(
            scores=scores,
            metrics=metrics,
            audio_quality=audio_quality,
            duration_ms=duration_ms,
            scenario=scenario,
            uncertainty=uncertainty,
            psychological_inference=psychological_inference,
            diagnostic_reasoning=diagnostic_reasoning,
            coaching_engine=coaching_engine,
            moment_intelligence=moment_intelligence,
        )
    facts = _build_recording_fact_ledger(
        transcript=transcript,
        metrics=metrics,
        psychological_inference=psychological_inference,
        moments=moments,
        audio_quality=audio_quality,
        duration_ms=duration_ms,
        base_confidence=confidence,
    )
    observations = _build_fact_observations(facts)
    diagnosis_model = _select_fact_diagnosis(observations, facts, coaching_engine, confidence, audio_quality)
    reconstruction = _reconstruct_listener_perception(observations, facts, diagnosis_model, confidence)
    cards = _fact_evidence_cards(observations, facts, diagnosis_model, reconstruction)
    if not cards:
        cards = [_insufficient_evidence_card(confidence, audio_quality)]
    evidence_ids = _fact_evidence_ids(cards)
    authority_type = _authority_type(scores, evidence_ids, confidence)
    diagnosis_confidence = diagnosis_model.confidence if diagnosis_model else min(confidence, 0.58)
    confidence_label = _confidence_label(diagnosis_confidence)
    mirror = _fact_mirror(scores, authority_type, diagnosis_model, cards, confidence_label, reconstruction)
    diagnosis = _fact_report_diagnosis(scores, diagnosis_model, cards, reconstruction)
    perception = _fact_perception_map(diagnosis_model, cards, diagnosis_confidence, scenario, reconstruction)
    timeline = _fact_timeline(moments, cards, duration_ms, scenario, reconstruction)
    dimension_reports = _fact_dimension_reports(scores, facts, cards, confidence)
    hidden_cost = _fact_hidden_cost(diagnosis_model, cards, diagnosis_confidence)
    fix = _fact_highest_fix(diagnosis_model, coaching_engine, cards)
    training = _fact_training(coaching_engine, fix, diagnosis_model, cards)
    retest = _fact_retest(fix)
    appendix = _technical_appendix(metrics, scores, audio_quality, evidence_ids)
    share_card = _share_card(scores, authority_type, mirror, diagnosis)
    primary = _fact_primary_diagnostic(diagnosis_model, cards)
    report_uncertainty = Uncertainty(
        overall_confidence_label=confidence_label,  # type: ignore[arg-type]
        suppressed_traits=psychological_inference.suppressed_traits,
        reasons=list(dict.fromkeys(uncertainty.reasons + psychological_inference.uncertainty.reasons + ([diagnosis_model.uncertainty_note] if diagnosis_model and diagnosis_model.uncertainty_note else []))),
    )
    if duration_ms and duration_ms < 25000:
        limited_text = "This sample does not contain enough reliable evidence for a strong authority diagnosis. Treat it as a light baseline and retest with a longer recording."
        mirror = mirror.model_copy(
            update={
                "headline": "There is not enough reliable evidence for a full authority diagnosis yet.",
                "identity_read": limited_text,
                "one_line_identity_read": limited_text,
                "confidence_label": "low",
                "confidence_level": "low",
            }
        )
        confidence_label = "low"
        report_uncertainty = report_uncertainty.model_copy(update={"overall_confidence_label": "low"})
        report_uncertainty.reasons.append("Short recording limits full report confidence")
    if not _transcript_safe(transcript):
        report_uncertainty.reasons.append("Transcript-specific wording was suppressed because transcript confidence was limited")
    report = AuthorityReport(
        mirror=mirror,
        diagnosis=diagnosis,
        perception_map=perception,
        evidence_chain=cards,
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
        scenario_summary=_scenario_summary(scores, fix, coaching_engine, get_scenario_profile(scenario).scenario_id),
        diagnostic_reasoning=diagnostic_reasoning,
        primary_diagnosis=primary,
        secondary_diagnosis=None,
        contradictions=diagnostic_reasoning.contradictions,
        hidden_cost_reasoning=diagnostic_reasoning.hidden_cost_reasoning if diagnosis_model else None,
        dimension_reasoning=diagnostic_reasoning.dimension_reasoning,
        trait_reasoning=diagnostic_reasoning.trait_reasoning,
        highest_leverage_reasoning=diagnostic_reasoning.highest_leverage_reasoning if diagnosis_model else None,
        coaching_engine=coaching_engine,
        uncertainty=report_uncertainty,
    )
    report = _apply_report_repetition_guard(report)
    return report.model_copy(update={"validation": _validate_report(report, coaching_engine)})


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
    duplicate_sections: list[str] = []
    user_strings: list[tuple[str, str]] = []
    if report.mirror:
        user_strings.extend(("mirror", value) for value in [report.mirror.headline, report.mirror.identity_read] if value)
    if report.diagnosis:
        user_strings.extend(("diagnosis", value) for value in [report.diagnosis.core_pattern, report.diagnosis.social_consequence] if value)
    if report.perception_map:
        for key, value in report.perception_map.model_dump().items():
            if value and value.get("text"):
                user_strings.append((key, value["text"]))
    if report.hidden_cost and report.hidden_cost.consequence:
        user_strings.append(("hidden_cost", report.hidden_cost.consequence))
    if report.highest_leverage_fix:
        user_strings.extend(("highest_leverage_fix", value) for value in [report.highest_leverage_fix.plain_english, report.highest_leverage_fix.why_this_matters, report.highest_leverage_fix.action_step] if value)
    if report.training_prescription:
        user_strings.extend(("training_prescription", value) for value in [report.training_prescription.why_chosen, report.training_prescription.success_signal, *report.training_prescription.instructions] if value)
    seen_norm: dict[str, str] = {}
    for section, text in user_strings:
        norm = _normalize_copy(text)
        if len(norm.split()) < 5:
            continue
        if norm in seen_norm:
            duplicate_sections.append(f"{seen_norm[norm]}::{section}")
            continue
        for prior_section, prior_text in user_strings:
            if prior_section == section or prior_text == text:
                continue
            if _copy_similarity(text, prior_text) >= 0.86:
                duplicate_sections.append(f"{prior_section}::{section}")
                break
        seen_norm[norm] = section
    return ReportValidation(
        valid=not orphan_links and not duplicate_sections,
        evidence_ids_checked=sorted(evidence_ids),
        moment_ids_checked=sorted(moment_ids),
        drill_ids_checked=sorted(drill_ids),
        orphan_links=orphan_links,
        duplicate_sections=list(dict.fromkeys(duplicate_sections)),
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
    transcript: Transcript | None = None,
) -> AuthorityReport:
    return _fact_led_report(
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
        transcript=transcript,
    )

    profile = get_scenario_profile(scenario)
    confidence = min(max(psychological_inference.overall_inference_confidence, scores.score_confidence or 0.0), 0.95)
    if _is_insufficient_sample(metrics, audio_quality, duration_ms, confidence):
        return _insufficient_report(
            scores=scores,
            metrics=metrics,
            audio_quality=audio_quality,
            duration_ms=duration_ms,
            scenario=scenario,
            uncertainty=uncertainty,
            psychological_inference=psychological_inference,
            diagnostic_reasoning=diagnostic_reasoning,
            coaching_engine=coaching_engine,
            moment_intelligence=moment_intelligence,
        )
    confidence_label = _confidence_label(confidence)
    observations = _behaviour_observations(
        evidence,
        psychological_inference,
        diagnostic_reasoning,
        coaching_engine,
        moments,
        confidence,
        duration_ms,
        audio_quality,
    )
    diagnosis_model = _select_behaviour_diagnosis(observations, confidence, duration_ms, audio_quality, coaching_engine)
    selected_observations = _diagnosis_observations(observations, diagnosis_model)
    evidence_cards = _rank_evidence_cards([_card_from_observation(item) for item in selected_observations], diagnostic_reasoning, coaching_engine)[:5]
    evidence_ids = [item.evidence_id for item in evidence_cards[:3]]
    diagnosis_model = _diagnosis_with_evidence(diagnosis_model, evidence_cards)
    diagnostic_reasoning = _reconciled_diagnostic_reasoning(diagnostic_reasoning, diagnosis_model, evidence_ids)
    if diagnosis_model is None:
        diagnostic_reasoning = diagnostic_reasoning.model_copy(
            update={
                "primary_diagnosis": None,
                "secondary_diagnosis": None,
                "hidden_cost_reasoning": None,
                "highest_leverage_reasoning": None,
            }
        )
    if not diagnosis_model and diagnostic_reasoning.primary_diagnosis and diagnostic_reasoning.primary_diagnosis.supporting_evidence_ids:
        evidence_ids = _visible_evidence_ids(diagnostic_reasoning.primary_diagnosis.supporting_evidence_ids, evidence_cards)
    dims = _ordered_dimensions(scores)
    strongest = DIMENSION_LABELS[dims[0][0]]
    limiter = DIMENSION_LABELS[sorted(_dimension_scores(scores).items(), key=lambda item: item[1])[0][0]]
    authority_type = _authority_type(scores, evidence_ids, confidence)
    diagnosis_confidence = min(confidence, diagnosis_model.confidence if diagnosis_model else confidence)
    if diagnosis_model is None:
        diagnosis_confidence = min(diagnosis_confidence, 0.59)
    visible_contradictions = [
        card for card in evidence_cards
        if diagnosis_model and card.direction == "negative" and card.evidence_id not in diagnosis_model.evidence_ids
    ]
    if visible_contradictions:
        diagnosis_confidence = min(0.79, max(0.35, diagnosis_confidence - min(0.18, len(visible_contradictions) * 0.08)))
    confidence_label = _confidence_label(diagnosis_confidence)
    mirror = _mirror(scores, authority_type, strongest, limiter, confidence_label, evidence_ids, evidence_cards, diagnosis_model)
    diagnosis = _diagnosis(scores, diagnostic_reasoning, evidence_ids, evidence_cards, diagnosis_model)
    perception_map = _apply_scenario_perception(_perception_map(diagnosis, authority_type, diagnosis_confidence, evidence_ids, evidence_cards, diagnosis_model), profile.scenario_id)
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
    timeline = _timeline(moments, evidence_ids, duration_ms, diagnosis_model)
    dimension_reports = _dimension_reports(scores, diagnostic_reasoning, evidence_ids, confidence, evidence_cards)
    hidden_cost = _hidden_cost(diagnosis, diagnostic_reasoning, evidence_ids, diagnosis_confidence, evidence_cards, diagnosis_model)
    fix = _highest_leverage_fix(coaching_engine, diagnostic_reasoning, evidence_ids, evidence_cards, diagnosis_model)
    training = _training(coaching_engine, fix, evidence_cards, diagnosis_model)
    retest = _retest(fix, duration_ms)
    appendix = _technical_appendix(metrics, scores, audio_quality, evidence_ids)
    share_card = _share_card(scores, authority_type, mirror, diagnosis)
    report_uncertainty = Uncertainty(
        overall_confidence_label=confidence_label,  # type: ignore[arg-type]
        suppressed_traits=psychological_inference.suppressed_traits,
        reasons=list(dict.fromkeys(uncertainty.reasons + psychological_inference.uncertainty.reasons + diagnostic_reasoning.uncertainty.reasons)),
    )
    if diagnosis_model and diagnosis_model.uncertainty_note:
        report_uncertainty.reasons.append("Multiple behavioural reads remain plausible; report confidence was reduced accordingly")
    if visible_contradictions:
        report_uncertainty.reasons.append("Competing behavioural signals lowered report confidence")
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
