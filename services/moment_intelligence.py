"""Deterministic Moment Intelligence Engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from schemas import (
    AudioQuality,
    AuthorityArc,
    CoachingEngine,
    EvidenceItem,
    LinguisticMetrics,
    Moment,
    MomentDimensionSnapshot,
    MomentIntelligence,
    Scores,
    TranscriptWord,
    Uncertainty,
)
from services.acoustic_metrics import WindowFeature
from services.scenario_profiles import get_scenario_profile


ENGINE_VERSION = "moment_intelligence_v1"
DIMENSIONS = ("command", "clarity", "composure", "presence", "persuasion", "structure")
NEGATIVE_TYPES = {
    "weakest_moment",
    "confidence_drop",
    "rushing_moment",
    "filler_cluster",
    "hesitation_cluster",
    "monotone_stretch",
    "weak_opening",
    "weak_closing",
    "most_costly_sentence",
    "most_unstable_section",
}
POSITIVE_TYPES = {
    "strongest_moment",
    "confidence_recovery",
    "pause_ownership_moment",
    "high_presence_moment",
    "strong_opening",
    "strong_closing",
    "best_sentence",
    "most_persuasive_moment",
    "most_commanding_moment",
    "most_composed_moment",
    "most_improved_section",
}
TYPE_METRICS = {
    "strongest_moment": ("window.command_score", "window.presence_score"),
    "weakest_moment": ("window.composure_score", "window.clarity_score"),
    "confidence_drop": ("window.composure_score_delta",),
    "confidence_recovery": ("window.composure_score_delta",),
    "rushing_moment": ("window.wpm",),
    "pause_ownership_moment": ("window.pause_ms", "window.command_score"),
    "filler_cluster": ("window.filler_rate",),
    "hesitation_cluster": ("window.pause_ms", "window.hesitation_cluster"),
    "monotone_stretch": ("window.pitch_stdev_semitones", "window.loudness_stdev_db"),
    "high_presence_moment": ("window.presence_score", "window.dynamic_emphasis"),
    "strong_opening": ("linguistic.opening_strength_score",),
    "weak_opening": ("linguistic.opening_strength_score",),
    "strong_closing": ("linguistic.closing_strength_score",),
    "weak_closing": ("linguistic.closing_strength_score",),
    "best_sentence": ("window.command_score", "transcript.words"),
    "most_costly_sentence": ("window.composure_score", "transcript.words"),
    "most_persuasive_moment": ("window.presence_score", "window.dynamic_emphasis"),
    "most_commanding_moment": ("window.command_score",),
    "most_composed_moment": ("window.composure_score",),
    "most_improved_section": ("window.local_delta",),
    "most_unstable_section": ("window.local_variance",),
}


@dataclass(frozen=True)
class _Candidate:
    moment_type: str
    window: WindowFeature
    dimensions: dict[str, float]
    confidence: float
    severity: str
    priority: int
    headline: str
    summary: str
    listener_interpretation: str
    why_it_matters: str


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _dedupe(values: Iterable[str | None]) -> list[str]:
    return [item for item in dict.fromkeys(value for value in values if value).keys()]


def _quality_weight(audio_quality: AudioQuality, duration_ms: int, uncertainty: Uncertainty) -> float:
    weight = 1.0
    if not audio_quality.usable:
        weight -= 0.45
    if duration_ms < 12000:
        weight -= 0.25
    if any("timestamp" in reason.lower() or "short" in reason.lower() for reason in uncertainty.reasons):
        weight -= 0.15
    return round(_clamp(weight, 0.0, 1.0), 2)


def _window_dimensions(window: WindowFeature, linguistic: LinguisticMetrics, position: float) -> dict[str, float]:
    structure = 0.5
    if position <= 0.2 and linguistic.opening_strength_score is not None:
        structure = linguistic.opening_strength_score
    elif position >= 0.8 and linguistic.closing_strength_score is not None:
        structure = linguistic.closing_strength_score
    elif linguistic.structure_score is not None:
        structure = linguistic.structure_score
    persuasion = _clamp((window.presence_score + window.command_score + window.dynamic_emphasis) / 3)
    return {
        "command": _clamp(window.command_score),
        "clarity": _clamp(window.clarity_score),
        "composure": _clamp(window.composure_score),
        "presence": _clamp(window.presence_score),
        "persuasion": persuasion,
        "structure": _clamp(structure),
    }


def _dimension_evolution(
    windows: list[WindowFeature],
    linguistic: LinguisticMetrics,
    duration_ms: int,
    evidence_ids: list[str],
    quality_weight: float,
) -> list[MomentDimensionSnapshot]:
    snapshots: list[MomentDimensionSnapshot] = []
    for window in windows:
        position = (window.start_ms + window.end_ms) / 2 / max(duration_ms, 1)
        dims = _window_dimensions(window, linguistic, position)
        evidence_density = min(1.0, len(evidence_ids) / 5)
        confidence = round(_clamp((sum(dims.values()) / len(dims)) * 0.65 + evidence_density * 0.2 + quality_weight * 0.15), 2)
        snapshots.append(
            MomentDimensionSnapshot(
                start_ms=window.start_ms,
                end_ms=window.end_ms,
                command=round(dims["command"], 2),
                clarity=round(dims["clarity"], 2),
                composure=round(dims["composure"], 2),
                presence=round(dims["presence"], 2),
                persuasion=round(dims["persuasion"], 2),
                structure=round(dims["structure"], 2),
                evidence_density=round(evidence_density, 2),
                confidence=confidence,
                quality_weighting=quality_weight,
            )
        )
    return snapshots


def _evidence_for_dimensions(evidence: list[EvidenceItem], dimensions: Iterable[str]) -> list[str]:
    targets = {dimension.lower() for dimension in dimensions}
    matched = [
        item.id
        for item in evidence
        if item.trait.lower() in targets or any(signal.lower().split(".")[0] in targets for signal in item.signals)
    ]
    return matched[:4] or [item.id for item in evidence[:3]]


def _words_in_window(words: list[TranscriptWord], start_ms: int, end_ms: int) -> list[tuple[int, TranscriptWord]]:
    return [
        (index, word)
        for index, word in enumerate(words)
        if word.end_ms >= start_ms and word.start_ms <= end_ms
    ]


def _transcript_span(words: list[TranscriptWord], start_ms: int, end_ms: int) -> tuple[str | None, list[str]]:
    selected = _words_in_window(words, start_ms, end_ms)
    if not selected:
        return None, []
    text = " ".join(word.text for _, word in selected).strip()
    return text or None, [f"w{index}" for index, _ in selected]


def _scenario_relevance(moment_type: str, scenario: str) -> float:
    profile = get_scenario_profile(scenario)
    priorities = " ".join(profile.coaching_priorities + profile.report_emphasis).lower()
    mapping = {
        "interview": {"weak_opening", "strong_opening", "best_sentence", "most_costly_sentence", "hesitation_cluster"},
        "sales": {"most_persuasive_moment", "high_presence_moment", "confidence_drop", "confidence_recovery"},
        "leadership": {"most_commanding_moment", "pause_ownership_moment", "strong_closing", "weak_closing"},
        "founder_pitch": {"strong_opening", "strong_closing", "most_persuasive_moment", "most_unstable_section"},
        "presentation": {"high_presence_moment", "monotone_stretch", "rushing_moment"},
        "meeting": {"pause_ownership_moment", "filler_cluster", "hesitation_cluster"},
        "podcast": {"high_presence_moment", "monotone_stretch", "most_composed_moment"},
    }
    score = 1.0
    if moment_type in mapping.get(profile.scenario_id, set()):
        score += 0.25
    if any(token in priorities for token in moment_type.split("_")):
        score += 0.1
    return round(min(score, 1.4), 2)


def _confidence(window: WindowFeature, evidence_strength: float, quality_weight: float, *signals: float) -> float:
    signal_strength = max([abs(value) for value in signals] or [0.0])
    base = 0.45 + signal_strength * 0.25 + evidence_strength * 0.2 + quality_weight * 0.1
    if window.hesitation_cluster:
        base += 0.04
    return round(_clamp(base, 0.0, 0.94), 2)


def _impact_score(dimensions: dict[str, float]) -> float:
    return sum(abs(value) for value in dimensions.values()) / max(len(dimensions), 1)


def _window_candidates(
    windows: list[WindowFeature],
    linguistic: LinguisticMetrics,
    duration_ms: int,
    evidence_strength: float,
    quality_weight: float,
) -> list[_Candidate]:
    if not windows:
        return []
    candidates: list[_Candidate] = []
    strongest = max(windows, key=lambda w: w.command_score + w.presence_score + w.dynamic_emphasis)
    weakest = min(windows, key=lambda w: w.composure_score + w.clarity_score - w.filler_rate)
    most_presence = max(windows, key=lambda w: w.presence_score + w.dynamic_emphasis)
    most_command = max(windows, key=lambda w: w.command_score + max(0.0, w.pause_ms / 1000) * 0.08)
    most_composed = max(windows, key=lambda w: w.composure_score - w.filler_rate - (0.12 if w.rushing else 0.0))
    most_persuasive = max(windows, key=lambda w: w.presence_score + w.command_score + w.dynamic_emphasis)
    unstable = max(windows, key=lambda w: abs(w.command_score - w.composure_score) + abs(w.presence_score - w.clarity_score))

    def add(moment_type: str, window: WindowFeature, dimensions: dict[str, float], severity: str, priority: int, headline: str, summary: str, interpretation: str, why: str, *signals: float) -> None:
        candidates.append(
            _Candidate(
                moment_type=moment_type,
                window=window,
                dimensions=dimensions,
                confidence=_confidence(window, evidence_strength, quality_weight, *signals),
                severity=severity,
                priority=priority,
                headline=headline,
                summary=summary,
                listener_interpretation=interpretation,
                why_it_matters=why,
            )
        )

    add(
        "strongest_moment",
        strongest,
        {"command": 0.16, "presence": 0.14},
        "highlight",
        1,
        "Strongest authority moment",
        "Command and presence were strongest in this window.",
        "Listeners are likely to hear this as one of the most authoritative parts of the recording.",
        "Authority peaks matter because they show what the speaker can reliably train toward.",
        strongest.command_score + strongest.presence_score,
    )
    add(
        "weakest_moment",
        weakest,
        {"composure": -0.15, "clarity": -0.12},
        "medium",
        2,
        "Lowest-control moment",
        "Composure and clarity were weakest in this window.",
        "Listeners may hear this as the section where control is least fully signalled.",
        "Low-control moments often explain why a capable answer can feel less settled.",
        1 - weakest.composure_score,
    )
    rushing = next((window for window in windows if window.rushing or window.wpm >= 185), None)
    if rushing:
        add("rushing_moment", rushing, {"composure": -0.14, "clarity": -0.08}, "medium", 5, "Pace accelerated here", f"Pace reached about {rushing.wpm:.0f} WPM in this window.", "This may sound like the speaker is trying to keep up with the thought instead of leading it.", "Pace jumps during important points can read as pressure rather than conviction.", rushing.wpm / 220)
    filler = next((window for window in windows if window.filler_rate >= 0.1), None)
    if filler:
        add("filler_cluster", filler, {"clarity": -0.13, "command": -0.08}, "medium", 6, "Filler cluster", "Filler density rose enough to affect the local authority signal.", "This may sound like searching for wording rather than delivering the point.", "Filler clusters matter because they interrupt the listener's sense of finality.", filler.filler_rate)
    hesitation = next((window for window in windows if window.hesitation_cluster or window.pause_ms >= 700), None)
    if hesitation:
        add("hesitation_cluster", hesitation, {"composure": -0.13, "command": -0.08}, "medium", 7, "Hesitation cluster", "Pauses concentrated in a way that can sound like mid-thought searching.", "Listeners may hear this as a temporary loss of command over the answer path.", "Hesitation clusters are more noticeable when they happen inside a point rather than between points.", hesitation.pause_ms / 1000)
    monotone = next((window for window in windows if window.monotone or (window.pitch_stdev_semitones <= 1.2 and window.loudness_stdev_db <= 2.0)), None)
    if monotone:
        add("monotone_stretch", monotone, {"presence": -0.14, "persuasion": -0.08}, "low", 8, "Lower-contrast stretch", "Pitch and loudness variation were limited here.", "This may sound less memorable even if the wording is clear.", "Low contrast can reduce listener attention and persuasive pull.", 1 - min(monotone.pitch_stdev_semitones / 4, 1))
    pause_owned = next((window for window in windows if 350 <= window.pause_ms <= 900 and window.command_score >= 0.62 and not window.rushing), None)
    if pause_owned:
        add("pause_ownership_moment", pause_owned, {"command": 0.12, "composure": 0.1}, "highlight", 4, "Owned pause", "A pause landed with enough control to support the point.", "Listeners are likely to hear the silence as intentional rather than uncertain.", "Owned pauses create status because they show the speaker does not need to rush.", pause_owned.command_score)
    add("high_presence_moment", most_presence, {"presence": 0.16, "persuasion": 0.1}, "highlight", 4, "Highest presence moment", "Energy and emphasis were strongest in this window.", "This is where the recording is most likely to hold attention.", "Presence moments show where the message becomes easiest to remember.", most_presence.presence_score + most_presence.dynamic_emphasis)
    add("most_commanding_moment", most_command, {"command": 0.17}, "highlight", 4, "Most commanding moment", "Command cues were strongest in this window.", "Listeners are most likely to feel led in this section.", "Commanding moments reveal the strongest local status signal.", most_command.command_score)
    add("most_composed_moment", most_composed, {"composure": 0.16}, "highlight", 4, "Most composed moment", "Composure cues were strongest in this window.", "This section is likely to feel comparatively settled.", "Composed moments show where pressure is least audible.", most_composed.composure_score)
    add("most_persuasive_moment", most_persuasive, {"persuasion": 0.16, "presence": 0.08}, "highlight", 4, "Most persuasive moment", "Presence, emphasis, and command aligned most strongly here.", "Listeners are most likely to feel pulled toward the point in this section.", "Persuasion improves when conviction and contrast arrive together.", most_persuasive.presence_score + most_persuasive.command_score)
    add("most_unstable_section", unstable, {"composure": -0.12, "structure": -0.08}, "medium", 9, "Most unstable section", "Local dimension scores diverged most strongly in this window.", "This may sound uneven: some cues work while others leak control.", "Instability matters because mixed signals make listener interpretation less clean.", abs(unstable.command_score - unstable.composure_score))

    if len(windows) >= 3:
        for index in range(1, len(windows)):
            prev = windows[index - 1]
            cur = windows[index]
            delta = cur.composure_score - prev.composure_score
            if delta <= -0.14:
                add("confidence_drop", cur, {"composure": -0.16, "command": -0.09}, "medium", 3, "Confidence dropped here", "Composure fell compared with the previous window.", "Listeners may hear pressure enter the delivery here.", "A local confidence drop explains exactly where perception deteriorated.", abs(delta))
                break
        for index in range(1, len(windows)):
            prev = windows[index - 1]
            cur = windows[index]
            delta = cur.composure_score - prev.composure_score
            if delta >= 0.14:
                add("confidence_recovery", cur, {"composure": 0.15, "command": 0.08}, "highlight", 4, "Confidence recovered here", "Composure improved compared with the previous window.", "Listeners may feel the speaker regain control of the point.", "Recovery moments matter because they show the speaker can self-correct inside the recording.", delta)
                break
        best_delta = max((windows[i].command_score + windows[i].presence_score) - (windows[i - 1].command_score + windows[i - 1].presence_score) for i in range(1, len(windows)))
        improved = max(range(1, len(windows)), key=lambda i: (windows[i].command_score + windows[i].presence_score) - (windows[i - 1].command_score + windows[i - 1].presence_score))
        if best_delta >= 0.16:
            add("most_improved_section", windows[improved], {"command": 0.1, "presence": 0.1}, "highlight", 5, "Most improved section", "Authority cues improved most sharply here.", "Listeners may feel the recording becoming more controlled at this point.", "Improvement moments show the shape of the speaker's strongest adjustment.", best_delta)

    return candidates


def _boundary_candidates(
    windows: list[WindowFeature],
    linguistic: LinguisticMetrics,
    words: list[TranscriptWord],
    duration_ms: int,
    evidence_strength: float,
    quality_weight: float,
) -> list[_Candidate]:
    if not words:
        return []
    candidates: list[_Candidate] = []
    first = windows[0] if windows else WindowFeature(0, min(3000, duration_ms), 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)
    last = windows[-1] if windows else WindowFeature(max(0, duration_ms - 3000), duration_ms, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)

    def add(moment_type: str, window: WindowFeature, score: float | None, positive: bool, priority: int) -> None:
        if score is None:
            return
        is_strong = score >= 0.7
        if positive != is_strong:
            return
        label = "opening" if "opening" in moment_type else "closing"
        dimensions = {"structure": 0.14 if positive else -0.14, "command": 0.08 if positive else -0.08}
        candidates.append(
            _Candidate(
                moment_type=moment_type,
                window=window,
                dimensions=dimensions,
                confidence=_confidence(window, evidence_strength, quality_weight, score),
                severity="highlight" if positive else "medium",
                priority=priority,
                headline=("Strong " if positive else "Weak ") + label,
                summary=f"The {label} score {'supported' if positive else 'limited'} local authority in this recording.",
                listener_interpretation=f"Listeners are likely to hear the {label} as {'clearer and more controlled' if positive else 'less controlled than the content needs'}.",
                why_it_matters=f"The {label} shapes first and final impressions, so it carries extra listener weight.",
            )
        )

    add("strong_opening", first, linguistic.opening_strength_score, True, 4)
    add("weak_opening", first, linguistic.opening_strength_score, False, 5)
    add("strong_closing", last, linguistic.closing_strength_score, True, 4)
    add("weak_closing", last, linguistic.closing_strength_score, False, 5)
    return candidates


def _sentence_candidates(windows: list[WindowFeature], words: list[TranscriptWord], evidence_strength: float, quality_weight: float) -> list[_Candidate]:
    if not words or not windows:
        return []
    best = max(windows, key=lambda w: w.command_score + w.clarity_score + w.presence_score)
    costly = min(windows, key=lambda w: w.composure_score + w.clarity_score - w.filler_rate - (0.2 if w.rushing else 0.0))
    return [
        _Candidate("best_sentence", best, {"clarity": 0.12, "command": 0.1}, _confidence(best, evidence_strength, quality_weight, best.command_score), "highlight", 4, "Best sentence", "This sentence-level span carried the strongest combination of clarity and command.", "Listeners are likely to hear this as the cleanest local expression of the point.", "Best sentences show the phrasing and delivery pattern worth repeating."),
        _Candidate("most_costly_sentence", costly, {"clarity": -0.13, "composure": -0.1}, _confidence(costly, evidence_strength, quality_weight, 1 - costly.composure_score), "medium", 5, "Most costly sentence", "This sentence-level span carried the highest local cost to clarity or composure.", "Listeners may notice this as the point where the answer works harder than it needs to.", "Costly sentences show the exact local pattern a drill should target."),
    ]


def _candidate_to_moment(
    candidate: _Candidate,
    index: int,
    evidence: list[EvidenceItem],
    words: list[TranscriptWord],
    scenario: str,
    quality_weight: float,
) -> Moment:
    dims = {dimension: round(value, 2) for dimension, value in candidate.dimensions.items()}
    evidence_ids = _evidence_for_dimensions(evidence, dims)
    transcript, word_ids = _transcript_span(words, candidate.window.start_ms, candidate.window.end_ms)
    scenario_relevance = _scenario_relevance(candidate.moment_type, scenario)
    training_value = 0.9 if candidate.moment_type in NEGATIVE_TYPES else 0.65
    importance = round(
        _clamp(
            _impact_score(dims) * 1.5
            + min(candidate.confidence, 1.0) * 0.35
            + scenario_relevance * 0.15
            + training_value * 0.15
        ),
        2,
    )
    return Moment(
        moment_id=f"m_{candidate.moment_type}_{index}",
        type=candidate.moment_type,
        priority=candidate.priority,
        start_ms=max(0, candidate.window.start_ms),
        end_ms=max(candidate.window.start_ms, candidate.window.end_ms),
        severity=candidate.severity,  # type: ignore[arg-type]
        headline=candidate.headline,
        summary=candidate.summary,
        listener_interpretation=candidate.listener_interpretation,
        why_it_matters=candidate.why_it_matters,
        confidence=round(candidate.confidence * quality_weight, 2),
        supporting_metrics=list(TYPE_METRICS.get(candidate.moment_type, ())),
        supporting_evidence_ids=evidence_ids,
        supporting_dimension_scores=dims,
        transcript_span=transcript,
        word_ids=word_ids,
        scenario_relevance=scenario_relevance,
        coaching_relevance=[],
        playback_available=bool(transcript or candidate.window.end_ms > candidate.window.start_ms),
        importance_score=importance,
        dimension_impact=dims,
        preview_visible_free=False,
    )


def _prioritise(moments: list[Moment]) -> tuple[list[Moment], list[str], str | None, list[str]]:
    ordered = sorted(moments, key=lambda item: (item.importance_score, -item.priority, item.confidence), reverse=True)
    top_ids = [item.moment_id for item in ordered[:8]]
    free = next((item.moment_id for item in ordered if item.type in POSITIVE_TYPES), ordered[0].moment_id if ordered else None)
    hidden = [item.moment_id for item in ordered[8:]]
    free_set = {free} if free else set()
    result = [
        item.model_copy(update={"preview_visible_free": item.moment_id in free_set})
        for item in ordered
    ]
    return result, top_ids, free, hidden


def _authority_arc(moments: list[Moment], evolution: list[MomentDimensionSnapshot]) -> AuthorityArc:
    if not evolution:
        return AuthorityArc(authority_arc=None, arc_confidence=0.0, major_turning_points=[])
    first = (evolution[0].command + evolution[0].composure + evolution[0].presence) / 3
    middle = (evolution[len(evolution) // 2].command + evolution[len(evolution) // 2].composure + evolution[len(evolution) // 2].presence) / 3
    last = (evolution[-1].command + evolution[-1].composure + evolution[-1].presence) / 3
    turn_ids = [item.moment_id for item in moments if item.type in {"confidence_drop", "confidence_recovery", "strong_opening", "weak_closing", "strong_closing"}][:4]
    if first >= 0.62 and middle <= first - 0.12 and last >= middle + 0.1:
        arc = "strong_start_pressure_middle_recovered_ending"
    elif first < 0.55 and last >= first + 0.14:
        arc = "weak_opening_growing_confidence"
    elif last <= first - 0.14:
        arc = "late_collapse"
    elif max(first, middle, last) - min(first, middle, last) <= 0.1:
        arc = "consistent_authority"
    elif middle < first and last >= first:
        arc = "early_hesitation_stable_recovery"
    else:
        arc = "mixed_authority_arc"
    confidence = round(_clamp(sum(item.confidence for item in evolution) / len(evolution)), 2)
    return AuthorityArc(authority_arc=arc, arc_confidence=confidence, major_turning_points=turn_ids)


def build_moment_intelligence(
    *,
    words: list[TranscriptWord],
    duration_ms: int,
    windows: list[WindowFeature],
    linguistic: LinguisticMetrics,
    evidence: list[EvidenceItem],
    scores: Scores,
    audio_quality: AudioQuality,
    uncertainty: Uncertainty,
    scenario: str,
) -> MomentIntelligence:
    """Build deterministic, evidence-backed moment intelligence from existing outputs."""
    del scores
    quality_weight = _quality_weight(audio_quality, duration_ms, uncertainty)
    suppressed = []
    if duration_ms <= 0:
        return MomentIntelligence(engine_version=ENGINE_VERSION, suppressed_moments=["zero_duration"])
    if not audio_quality.usable or quality_weight < 0.45:
        return MomentIntelligence(engine_version=ENGINE_VERSION, suppressed_moments=["poor_audio_or_unreliable_windows"])
    if duration_ms < 8000 or not windows:
        return MomentIntelligence(engine_version=ENGINE_VERSION, suppressed_moments=["insufficient_window_evidence"])
    if not evidence:
        return MomentIntelligence(engine_version=ENGINE_VERSION, suppressed_moments=["insufficient_upstream_evidence"])

    evidence_ids = [item.id for item in evidence]
    evidence_strength = min(1.0, len(evidence_ids) / 5)
    evolution = _dimension_evolution(windows, linguistic, duration_ms, evidence_ids, quality_weight)
    candidates = (
        _window_candidates(windows, linguistic, duration_ms, evidence_strength, quality_weight)
        + _boundary_candidates(windows, linguistic, words, duration_ms, evidence_strength, quality_weight)
        + _sentence_candidates(windows, words, evidence_strength, quality_weight)
    )
    unique: dict[str, _Candidate] = {}
    for candidate in candidates:
        existing = unique.get(candidate.moment_type)
        if not existing or candidate.confidence > existing.confidence:
            unique[candidate.moment_type] = candidate
    moments = [
        _candidate_to_moment(candidate, index + 1, evidence, words, scenario, quality_weight)
        for index, candidate in enumerate(unique.values())
        if candidate.confidence * quality_weight >= 0.35
    ]
    if not moments:
        suppressed.append("moment_confidence_below_threshold")
    ordered, top_ids, free_id, hidden_ids = _prioritise(moments)
    return MomentIntelligence(
        engine_version=ENGINE_VERSION,
        moments=ordered,
        dimension_evolution=evolution,
        authority_arc=_authority_arc(ordered, evolution),
        top_premium_moments=top_ids,
        top_free_moment=free_id,
        hidden_moments=hidden_ids,
        suppressed_moments=suppressed,
    )


def attach_coaching_relevance(
    moment_intelligence: MomentIntelligence,
    coaching_engine: CoachingEngine,
) -> MomentIntelligence:
    """Attach deterministic coaching links to already-detected moments."""
    primary = coaching_engine.selected_interventions.primary_drill
    secondary = coaching_engine.selected_interventions.secondary_drill
    selected = [item for item in (primary, secondary) if item is not None]
    if not selected and not coaching_engine.root_causes:
        return moment_intelligence
    root_dimension_links = {
        root.root_cause_id: {dimension.lower() for dimension in root.affected_dimensions}
        for root in coaching_engine.root_causes
    }
    updated: list[Moment] = []
    for moment in moment_intelligence.moments:
        links = []
        metrics = set(moment.supporting_metrics)
        dims = {dimension.lower() for dimension in moment.supporting_dimension_scores}
        for root_id, root_dims in root_dimension_links.items():
            if dims.intersection(root_dims):
                links.append(root_id)
        for candidate in selected:
            required = set(candidate.required_evidence)
            target_dims = set()
            for definition in coaching_engine.drill_library:
                if definition.drill_id == candidate.drill_id:
                    target_dims.update(dimension.lower() for dimension in definition.target_dimensions)
                    required.update(definition.target_metrics)
            if metrics.intersection(required) or dims.intersection(target_dims) or set(moment.supporting_evidence_ids).intersection(candidate.supporting_evidence_ids):
                links.append(candidate.drill_id)
        updated.append(moment.model_copy(update={"coaching_relevance": _dedupe(links)}))
    return moment_intelligence.model_copy(update={"moments": updated})
