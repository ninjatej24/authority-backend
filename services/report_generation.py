"""Milestone 7 deterministic premium report generation."""

from __future__ import annotations

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
        hidden_cost_reasoning=diagnostic_reasoning.hidden_cost_reasoning,
        dimension_reasoning={},
        trait_reasoning=diagnostic_reasoning.trait_reasoning,
        highest_leverage_reasoning=diagnostic_reasoning.highest_leverage_reasoning,
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
