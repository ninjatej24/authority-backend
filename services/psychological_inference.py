"""Deterministic psychological inference from measured speech signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from schemas import (
    AudioQuality,
    InferenceCandidate,
    Metrics,
    MicroBehaviour,
    PsychologicalEvidenceSignal,
    PsychologicalInference,
    PsychologicalPrimaryCandidates,
    PsychologicalReportCandidates,
    PsychologicalTrait,
    Scores,
    Uncertainty,
)
from services.scenario_profiles import calculate_trait_relevance


@dataclass(frozen=True)
class SignalRule:
    id: str
    metric: str
    expected_range: str
    direction: str
    weight: float
    why: str
    active: Callable[[float | int | str | bool | None], bool]


@dataclass(frozen=True)
class BehaviourRule:
    id: str
    label: str
    supports: tuple[str, ...]
    contradicts: tuple[str, ...] = ()
    min_supports: int = 2


@dataclass(frozen=True)
class TraitRule:
    id: str
    label: str
    supports: tuple[str, ...]
    contradicts: tuple[str, ...] = ()
    fragile: bool = False
    scenario_adjustments: dict[str, float] | None = None


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


def _metric_values(
    metrics: Metrics,
    scores: Scores,
    audio_quality: AudioQuality,
    duration_ms: int,
) -> dict[str, float | int | str | bool | None]:
    raw = metrics.raw_acoustic
    linguistic = metrics.linguistic
    derived = metrics.derived
    rhythm = metrics.rhythm
    articulation = metrics.articulation
    vad = metrics.vad

    return {
        "audio_quality.usable": audio_quality.usable,
        "audio_quality.background_noise_level": audio_quality.background_noise_level,
        "request.duration_ms": duration_ms,
        "transcript.overall_asr_confidence": None,
        "raw_acoustic.words_per_minute": raw.words_per_minute or rhythm.words_per_minute,
        "raw_acoustic.pause_frequency_per_min": raw.pause_frequency_per_min,
        "raw_acoustic.avg_pause_ms": raw.avg_pause_ms,
        "raw_acoustic.longest_pause_ms": raw.longest_pause_ms,
        "raw_acoustic.mid_phrase_pause_rate": raw.mid_phrase_pause_rate,
        "raw_acoustic.f0_range_semitones": raw.f0_range_semitones,
        "raw_acoustic.f0_variability_semitones": raw.f0_variability_semitones,
        "raw_acoustic.loudness_variation_db": raw.loudness_variation_db,
        "raw_acoustic.terminal_rising_ratio": raw.terminal_rising_ratio,
        "raw_acoustic.terminal_falling_ratio": raw.terminal_falling_ratio,
        "raw_acoustic.dynamic_emphasis": raw.dynamic_emphasis,
        "raw_acoustic.projection_segments": raw.projection_segments,
        "raw_acoustic.energy_cv": raw.energy_cv,
        "linguistic.filler_words_per_min": linguistic.filler_words_per_min,
        "linguistic.lexical_fillers": linguistic.lexical_fillers,
        "linguistic.acoustic_hesitations": linguistic.acoustic_hesitations,
        "linguistic.confirmed_disfluencies": linguistic.confirmed_disfluencies,
        "linguistic.disfluency_confidence": linguistic.disfluency_confidence,
        "linguistic.hedges_per_100_words": linguistic.hedges_per_100_words,
        "linguistic.certainty_markers_per_100_words": linguistic.certainty_markers_per_100_words,
        "linguistic.self_doubt_markers": linguistic.self_doubt_markers,
        "linguistic.repetition_rate": linguistic.repetition_rate,
        "linguistic.specificity_score": linguistic.specificity_score,
        "linguistic.concreteness_score": linguistic.concreteness_score,
        "linguistic.rambling_score": linguistic.rambling_score,
        "linguistic.opening_strength_score": linguistic.opening_strength_score,
        "linguistic.closing_strength_score": linguistic.closing_strength_score,
        "linguistic.structure_score": linguistic.structure_score,
        "rhythm.rhythm_consistency": rhythm.rhythm_consistency,
        "rhythm.hesitation_windows": rhythm.hesitation_windows,
        "rhythm.burst_speaking_segments": rhythm.burst_speaking_segments,
        "rhythm.speed_up_segments": rhythm.speed_up_segments,
        "rhythm.slow_down_segments": rhythm.slow_down_segments,
        "articulation.clarity_proxy": articulation.clarity_proxy,
        "articulation.articulation_stability": articulation.articulation_stability,
        "vad.speech_ratio": vad.speech_ratio,
        "vad.total_speech_duration_ms": vad.total_speech_duration_ms,
        "vad.pause_frequency_per_minute": vad.pause_frequency_per_minute,
        "derived.monotony_index": derived.monotony_index,
        "derived.hesitation_cluster_score": derived.hesitation_cluster_score,
        "derived.dynamic_emphasis_score": derived.dynamic_emphasis_score,
        "derived.speech_continuity_score": derived.speech_continuity_score,
        "derived.vocal_command_index": derived.vocal_command_index,
        "derived.composure_index": derived.composure_index,
        "derived.rhythm_index": derived.rhythm_index,
        "derived.projection_index": derived.projection_index,
        "derived.authority_signal_index": derived.authority_signal_index,
        "scores.command": scores.dimension_scores.command,
        "scores.clarity": scores.dimension_scores.clarity,
        "scores.composure": scores.dimension_scores.composure,
        "scores.presence": scores.dimension_scores.presence,
        "scores.persuasion": scores.dimension_scores.persuasion,
        "scores.structure": scores.dimension_scores.structure,
    }


def _signals() -> list[SignalRule]:
    return [
        SignalRule("high_fillers", "linguistic.filler_words_per_min", "<= 3 preferred; >= 8 disruptive", "supporting", 0.9, "High filler load makes speech sound less fluent and can weaken perceived control.", lambda v: _num(v) >= 8),
        SignalRule("very_high_fillers", "linguistic.filler_words_per_min", "< 8", "supporting", 1.0, "Very high filler load is a stronger disfluency signal than occasional conversational fillers.", lambda v: _num(v) >= 12),
        SignalRule("low_fillers", "linguistic.filler_words_per_min", "<= 3", "supporting", 0.75, "Low filler load supports impressions of verbal control.", lambda v: v is not None and _num(v) <= 3),
        SignalRule("acoustic_hesitations", "linguistic.acoustic_hesitations", "0", "supporting", 0.35, "Audio timing suggests searching even when the transcript does not preserve filled pauses.", lambda v: _num(v) >= 1),
        SignalRule("pace_fast", "raw_acoustic.words_per_minute", "115-165 typical controlled range", "supporting", 0.75, "Fast pace can read as urgency or pressure when paired with disruption.", lambda v: _num(v) >= 175),
        SignalRule("pace_controlled", "raw_acoustic.words_per_minute", "115-165", "supporting", 0.65, "Controlled pace makes the listener work less and supports composure.", lambda v: 115 <= _num(v) <= 165),
        SignalRule("pace_slow", "raw_acoustic.words_per_minute", ">= 95", "supporting", 0.5, "Very slow pace can reduce energy unless balanced by strong structure.", lambda v: 0 < _num(v) <= 95),
        SignalRule("pace_acceleration", "rhythm.speed_up_segments", "0", "supporting", 0.75, "Acceleration during a recording can make important content feel less settled.", lambda v: _num(v) >= 1),
        SignalRule("burst_speaking", "rhythm.burst_speaking_segments", "0", "supporting", 0.75, "Burst speaking often sounds like the speaker is chasing the point.", lambda v: _num(v) >= 1),
        SignalRule("stable_rhythm", "rhythm.rhythm_consistency", ">= 0.70", "supporting", 0.75, "Stable rhythm supports impressions of steadiness under pressure.", lambda v: _num(v) >= 0.70),
        SignalRule("unstable_rhythm", "rhythm.rhythm_consistency", ">= 0.55", "supporting", 0.75, "Unstable rhythm can make delivery feel reactive rather than led.", lambda v: v is not None and _num(v) <= 0.45),
        SignalRule("hesitation_high", "derived.hesitation_cluster_score", "< 0.45", "supporting", 0.9, "Clustered hesitation is more perceptually costly than isolated disfluency.", lambda v: _num(v) >= 0.55),
        SignalRule("hesitation_low", "derived.hesitation_cluster_score", "< 0.25", "supporting", 0.75, "Low hesitation clustering supports fluent thought control.", lambda v: v is not None and _num(v) <= 0.25),
        SignalRule("hesitation_windows", "rhythm.hesitation_windows", "0", "supporting", 0.75, "Repeated hesitation windows suggest local disruption under pressure.", lambda v: _num(v) >= 1),
        SignalRule("owned_pauses", "raw_acoustic.avg_pause_ms", "250-800 ms", "supporting", 0.7, "Moderate pauses can feel intentional and give key points room to land.", lambda v: 250 <= _num(v) <= 800),
        SignalRule("mid_phrase_pauses", "raw_acoustic.mid_phrase_pause_rate", "<= 0.25", "supporting", 0.75, "Mid-phrase pauses are more likely to sound like searching than owned silence.", lambda v: _num(v) >= 0.35),
        SignalRule("low_mid_phrase_pauses", "raw_acoustic.mid_phrase_pause_rate", "<= 0.25", "supporting", 0.65, "Low mid-phrase pause load supports clean phrase control.", lambda v: v is not None and _num(v) <= 0.25),
        SignalRule("falling_endings", "raw_acoustic.terminal_falling_ratio", ">= 0.35", "supporting", 0.8, "Falling declarative endings help key statements sound final.", lambda v: _num(v) >= 0.35),
        SignalRule("rising_endings", "raw_acoustic.terminal_rising_ratio", "<= 0.25", "supporting", 0.8, "Frequent rising declarative endings can sound permission-seeking in high-stakes speech.", lambda v: _num(v) >= 0.45),
        SignalRule("dynamic_emphasis_high", "derived.dynamic_emphasis_score", ">= 0.60", "supporting", 0.75, "Dynamic emphasis makes important lines easier to remember and feel.", lambda v: _num(v) >= 0.60),
        SignalRule("dynamic_emphasis_low", "derived.dynamic_emphasis_score", ">= 0.40", "supporting", 0.75, "Low emphasis can make even clear content feel less consequential.", lambda v: v is not None and _num(v) <= 0.30),
        SignalRule("pitch_variation_low", "raw_acoustic.f0_range_semitones", ">= 4 semitones", "supporting", 0.65, "Low within-speaker pitch movement can reduce perceived presence.", lambda v: v is not None and 0 < _num(v) <= 3.5),
        SignalRule("pitch_variation_healthy", "raw_acoustic.f0_range_semitones", ">= 5 semitones", "supporting", 0.55, "Within-speaker pitch movement supports vocal colour without judging natural pitch.", lambda v: _num(v) >= 5),
        SignalRule("energy_variation_low", "raw_acoustic.loudness_variation_db", ">= 4 dB relative spread", "supporting", 0.65, "Low within-recording energy variation can make delivery feel flat.", lambda v: v is not None and 0 <= _num(v) <= 3.5),
        SignalRule("energy_variation_healthy", "raw_acoustic.loudness_variation_db", ">= 4.5 dB relative spread", "supporting", 0.55, "Relative energy variation supports attention without relying on microphone loudness.", lambda v: _num(v) >= 4.5),
        SignalRule("opening_strong", "linguistic.opening_strength_score", ">= 0.70", "supporting", 0.8, "Strong openings frame the listener's first impression of control.", lambda v: _num(v) >= 0.70),
        SignalRule("opening_weak", "linguistic.opening_strength_score", ">= 0.55", "supporting", 0.7, "Weak openings delay the point and can reduce early authority.", lambda v: v is not None and _num(v) <= 0.45),
        SignalRule("closing_strong", "linguistic.closing_strength_score", ">= 0.70", "supporting", 0.8, "Strong closes preserve the final impression of certainty.", lambda v: _num(v) >= 0.70),
        SignalRule("closing_weak", "linguistic.closing_strength_score", ">= 0.55", "supporting", 0.8, "Weak closes can make a strong answer fade before it lands.", lambda v: v is not None and _num(v) <= 0.50),
        SignalRule("hedges_high", "linguistic.hedges_per_100_words", "<= 2", "supporting", 0.75, "Repeated hedging can soften commitment in high-stakes speech.", lambda v: _num(v) >= 3.0),
        SignalRule("hedges_low", "linguistic.hedges_per_100_words", "<= 1.5", "supporting", 0.55, "Low hedge load supports clean commitment language.", lambda v: v is not None and _num(v) <= 1.5),
        SignalRule("certainty_high", "linguistic.certainty_markers_per_100_words", ">= 2", "supporting", 0.7, "Clear commitment language supports perceived confidence when delivery is controlled.", lambda v: _num(v) >= 2.0),
        SignalRule("certainty_low", "linguistic.certainty_markers_per_100_words", ">= 1", "supporting", 0.55, "Low certainty language can make the listener work harder to find the stance.", lambda v: v is not None and _num(v) <= 0.5),
        SignalRule("self_doubt", "linguistic.self_doubt_markers", "0", "supporting", 0.8, "Self-doubt markers directly weaken the force of otherwise useful content.", lambda v: _num(v) >= 1),
        SignalRule("structure_high", "linguistic.structure_score", ">= 0.70", "supporting", 0.8, "Structure makes the listener feel the speaker knows where the answer is going.", lambda v: _num(v) >= 0.70),
        SignalRule("structure_low", "linguistic.structure_score", ">= 0.55", "supporting", 0.8, "Loose structure can make competent content feel less controlled.", lambda v: v is not None and _num(v) <= 0.45),
        SignalRule("specificity_high", "linguistic.specificity_score", ">= 0.55", "supporting", 0.65, "Specific details make claims feel grounded and credible.", lambda v: _num(v) >= 0.55),
        SignalRule("specificity_low", "linguistic.specificity_score", ">= 0.35", "supporting", 0.55, "Low specificity makes conclusions feel less proven.", lambda v: v is not None and _num(v) <= 0.30),
        SignalRule("concreteness_high", "linguistic.concreteness_score", ">= 0.45", "supporting", 0.55, "Concrete language gives the listener evidence to hold onto.", lambda v: _num(v) >= 0.45),
        SignalRule("concreteness_low", "linguistic.concreteness_score", ">= 0.30", "supporting", 0.45, "Low concreteness can make speech feel generic.", lambda v: v is not None and _num(v) <= 0.25),
        SignalRule("rambling_high", "linguistic.rambling_score", "<= 0.25", "supporting", 0.8, "Rambling weakens perceived executive control.", lambda v: _num(v) >= 0.45),
        SignalRule("rambling_low", "linguistic.rambling_score", "<= 0.25", "supporting", 0.6, "Low rambling supports concise thought control.", lambda v: v is not None and _num(v) <= 0.25),
        SignalRule("repetition_high", "linguistic.repetition_rate", "<= 0.25", "supporting", 0.55, "Uncontrolled repetition can sound like searching rather than emphasis.", lambda v: _num(v) >= 0.45),
        SignalRule("articulation_clear", "articulation.clarity_proxy", ">= 0.70", "supporting", 0.65, "Clear articulation reduces listener effort, with ASR fairness safeguards.", lambda v: _num(v) >= 0.70),
        SignalRule("articulation_weak", "articulation.clarity_proxy", ">= 0.55", "supporting", 0.45, "Weak articulation proxy can raise listener effort, but should be downweighted when ASR is weak.", lambda v: v is not None and _num(v) <= 0.45),
        SignalRule("projection_high", "derived.projection_index", ">= 0.65", "supporting", 0.65, "Projection based on relative dynamics supports presence without judging microphone loudness.", lambda v: _num(v) >= 0.65),
        SignalRule("projection_low", "derived.projection_index", ">= 0.45", "supporting", 0.55, "Low projection can under-signal the importance of the message.", lambda v: v is not None and _num(v) <= 0.40),
        SignalRule("command_high", "derived.vocal_command_index", ">= 0.65", "supporting", 0.75, "Composite command signals reflect stable pace, dynamic control, and low disruption.", lambda v: _num(v) >= 0.65),
        SignalRule("composure_high", "derived.composure_index", ">= 0.65", "supporting", 0.75, "Composite composure signals suggest the moment is not knocking the speaker off balance.", lambda v: _num(v) >= 0.65),
        SignalRule("composure_low", "derived.composure_index", ">= 0.45", "supporting", 0.75, "Low composure signals suggest pressure may be audible in the recording.", lambda v: v is not None and _num(v) <= 0.40),
        SignalRule("authority_signal_high", "derived.authority_signal_index", ">= 0.65", "supporting", 0.75, "A broad authority composite supports stronger perception claims when individual signals agree.", lambda v: _num(v) >= 0.65),
        SignalRule("little_voiced_speech", "vad.speech_ratio", ">= 0.30", "uncertainty", 1.0, "Little voiced speech limits the evidence available for listener-perception inference.", lambda v: v is not None and _num(v) < 0.30),
        SignalRule("poor_audio", "audio_quality.usable", "true", "uncertainty", 1.0, "Poor audio quality reduces confidence in fragile acoustic claims.", lambda v: v is False),
        SignalRule("short_recording", "request.duration_ms", ">= 25000 ms for deep inference", "uncertainty", 0.9, "Short recordings should produce limited and lower-confidence inference.", lambda v: _num(v) > 0 and _num(v) < 25000),
    ]


def _num(value: float | int | str | bool | None) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _behaviours() -> list[BehaviourRule]:
    return [
        BehaviourRule("searching_for_wording", "Searching for wording under pressure", ("high_fillers", "acoustic_hesitations", "hesitation_high", "pace_acceleration"), ("low_fillers", "stable_rhythm")),
        BehaviourRule("comfort_holding_floor", "Comfort holding the conversational floor", ("stable_rhythm", "owned_pauses", "low_fillers"), ("pace_fast", "hesitation_high")),
        BehaviourRule("speaking_with_conviction", "Speaking with conviction", ("falling_endings", "dynamic_emphasis_high", "certainty_high"), ("rising_endings", "hedges_high")),
        BehaviourRule("reluctance_to_commit", "Reluctance to commit strongly", ("closing_weak", "hedges_high", "rising_endings"), ("certainty_high", "falling_endings")),
        BehaviourRule("pace_pressure", "Pace pressure", ("pace_fast", "pace_acceleration", "burst_speaking"), ("pace_controlled", "stable_rhythm")),
        BehaviourRule("deliberate_pacing", "Deliberate pacing", ("pace_controlled", "owned_pauses", "stable_rhythm"), ("pace_fast", "pace_acceleration")),
        BehaviourRule("flat_delivery", "Flat delivery", ("pitch_variation_low", "energy_variation_low", "dynamic_emphasis_low"), ("dynamic_emphasis_high", "pitch_variation_healthy")),
        BehaviourRule("vocal_variety", "Vocal variety", ("pitch_variation_healthy", "energy_variation_healthy", "dynamic_emphasis_high"), ("pitch_variation_low", "energy_variation_low")),
        BehaviourRule("clear_opening_control", "Clear opening control", ("opening_strong", "certainty_high", "low_fillers"), ("opening_weak", "hedges_high")),
        BehaviourRule("weak_opening", "Weak opening control", ("opening_weak", "high_fillers", "hedges_high"), ("opening_strong", "certainty_high")),
        BehaviourRule("clean_closing", "Clean closing", ("closing_strong", "falling_endings", "low_fillers"), ("closing_weak", "rising_endings")),
        BehaviourRule("weak_closing", "Weak closing", ("closing_weak", "rising_endings", "hedges_high"), ("closing_strong", "falling_endings")),
        BehaviourRule("structured_thinking", "Structured thinking", ("structure_high", "specificity_high", "rambling_low"), ("structure_low", "rambling_high")),
        BehaviourRule("loose_structure", "Loose structure", ("structure_low", "rambling_high", "repetition_high"), ("structure_high", "rambling_low")),
        BehaviourRule("grounded_specificity", "Grounded specificity", ("specificity_high", "concreteness_high", "certainty_high"), ("specificity_low", "concreteness_low")),
        BehaviourRule("vague_generalising", "Vague generalising", ("specificity_low", "concreteness_low", "hedges_high"), ("specificity_high", "certainty_high")),
        BehaviourRule("pause_ownership", "Pause ownership", ("owned_pauses", "low_mid_phrase_pauses", "stable_rhythm"), ("mid_phrase_pauses", "hesitation_high")),
        BehaviourRule("pause_disruption", "Pause disruption", ("mid_phrase_pauses", "hesitation_high", "unstable_rhythm"), ("owned_pauses", "low_mid_phrase_pauses")),
        BehaviourRule("projection_control", "Projection control", ("projection_high", "dynamic_emphasis_high", "energy_variation_healthy"), ("projection_low", "energy_variation_low")),
        BehaviourRule("under_projected", "Under-projected delivery", ("projection_low", "energy_variation_low", "dynamic_emphasis_low"), ("projection_high", "dynamic_emphasis_high")),
        BehaviourRule("articulation_clarity", "Articulation clarity", ("articulation_clear", "pace_controlled", "structure_high"), ("articulation_weak", "pace_fast")),
        BehaviourRule("clarity_fragile", "Fragile clarity", ("articulation_weak", "pace_fast", "rambling_high"), ("articulation_clear", "structure_high")),
        BehaviourRule("composure_under_pressure", "Composure under pressure", ("composure_high", "stable_rhythm", "hesitation_low"), ("composure_low", "hesitation_high")),
        BehaviourRule("pressure_leakage", "Pressure leakage", ("composure_low", "pace_acceleration", "hesitation_high"), ("composure_high", "stable_rhythm")),
        BehaviourRule("low_approval_seeking", "Low approval-seeking delivery", ("falling_endings", "certainty_high", "hedges_low"), ("rising_endings", "self_doubt")),
        BehaviourRule("approval_seeking_cues", "Approval-seeking cues", ("rising_endings", "hedges_high", "self_doubt"), ("falling_endings", "certainty_high")),
        BehaviourRule("persuasive_momentum", "Persuasive momentum", ("dynamic_emphasis_high", "specificity_high", "certainty_high"), ("dynamic_emphasis_low", "specificity_low")),
        BehaviourRule("explanation_without_pull", "Explanation without pull", ("dynamic_emphasis_low", "certainty_low", "specificity_low"), ("dynamic_emphasis_high", "certainty_high")),
        BehaviourRule("warm_steady_presence", "Warm steady presence", ("pace_controlled", "owned_pauses", "energy_variation_healthy"), ("pace_fast", "hesitation_high")),
        BehaviourRule("executive_finality", "Executive finality", ("command_high", "falling_endings", "closing_strong"), ("rising_endings", "closing_weak")),
        BehaviourRule("interview_readiness_pattern", "Interview readiness pattern", ("opening_strong", "structure_high", "articulation_clear", "certainty_high"), ("rambling_high", "weak_closing")),
        BehaviourRule("leadership_readiness_pattern", "Leadership readiness pattern", ("command_high", "composure_high", "authority_signal_high"), ("approval_seeking_cues", "pressure_leakage")),
        BehaviourRule("limited_inference_evidence", "Limited inference evidence", ("poor_audio", "short_recording", "little_voiced_speech"), (), min_supports=1),
    ]


def _trait_rules() -> list[TraitRule]:
    return [
        TraitRule("confident", "Confident", ("speaking_with_conviction", "deliberate_pacing", "executive_finality"), ("reluctance_to_commit", "pressure_leakage")),
        TraitRule("composed", "Composed", ("composure_under_pressure", "comfort_holding_floor", "pause_ownership"), ("pressure_leakage", "pause_disruption")),
        TraitRule("credible", "Credible", ("structured_thinking", "grounded_specificity", "articulation_clarity"), ("vague_generalising", "loose_structure")),
        TraitRule("trustworthy", "Trustworthy", ("warm_steady_presence", "structured_thinking", "deliberate_pacing"), ("pressure_leakage", "approval_seeking_cues"), fragile=True),
        TraitRule("warm", "Warm", ("warm_steady_presence", "comfort_holding_floor", "deliberate_pacing"), ("pace_pressure", "under_projected"), fragile=True),
        TraitRule("commanding", "Commanding", ("executive_finality", "pause_ownership", "speaking_with_conviction"), ("approval_seeking_cues", "weak_closing")),
        TraitRule("high_status", "High Status", ("executive_finality", "comfort_holding_floor", "leadership_readiness_pattern"), ("approval_seeking_cues", "reluctance_to_commit"), fragile=True),
        TraitRule("persuasive", "Persuasive", ("persuasive_momentum", "speaking_with_conviction", "vocal_variety"), ("explanation_without_pull", "flat_delivery")),
        TraitRule("energetic", "Energetic", ("projection_control", "vocal_variety", "persuasive_momentum"), ("under_projected", "flat_delivery"), fragile=True),
        TraitRule("calm", "Calm", ("deliberate_pacing", "comfort_holding_floor", "composure_under_pressure"), ("pace_pressure", "pressure_leakage")),
        TraitRule("rushed", "Rushed", ("pace_pressure", "pressure_leakage", "searching_for_wording"), ("deliberate_pacing", "pause_ownership")),
        TraitRule("nervous", "Nervous", ("searching_for_wording", "pressure_leakage", "pause_disruption"), ("composure_under_pressure", "deliberate_pacing"), fragile=True),
        TraitRule("flat", "Flat", ("flat_delivery", "under_projected", "explanation_without_pull"), ("vocal_variety", "projection_control"), fragile=True),
        TraitRule("monotone", "Monotone", ("flat_delivery", "explanation_without_pull"), ("vocal_variety",), fragile=True),
        TraitRule("hesitant", "Hesitant", ("searching_for_wording", "pause_disruption", "reluctance_to_commit"), ("speaking_with_conviction", "pause_ownership")),
        TraitRule("approval_seeking", "Approval Seeking", ("approval_seeking_cues", "reluctance_to_commit", "weak_closing"), ("low_approval_seeking", "executive_finality")),
        TraitRule("leadership_ready", "Leadership Ready", ("leadership_readiness_pattern", "executive_finality", "structured_thinking"), ("pressure_leakage", "approval_seeking_cues"), scenario_adjustments={"benchmark": 1.0, "impromptu": 0.95}),
        TraitRule("interview_ready", "Interview Ready", ("interview_readiness_pattern", "structured_thinking", "articulation_clarity"), ("loose_structure", "weak_closing"), scenario_adjustments={"benchmark": 1.0, "impromptu": 1.05}),
        TraitRule("executive_presence", "Executive Presence", ("executive_finality", "leadership_readiness_pattern", "projection_control", "composure_under_pressure"), ("flat_delivery", "approval_seeking_cues"), fragile=True),
        TraitRule("structured_thinker", "Structured Thinker", ("structured_thinking", "clear_opening_control", "grounded_specificity"), ("loose_structure", "vague_generalising")),
        TraitRule("clear_communicator", "Clear Communicator", ("articulation_clarity", "structured_thinking", "clear_opening_control"), ("clarity_fragile", "loose_structure")),
    ]


def _evaluate_signals(
    values: dict[str, float | int | str | bool | None],
) -> tuple[dict[str, PsychologicalEvidenceSignal], set[str]]:
    evidence: dict[str, PsychologicalEvidenceSignal] = {}
    active: set[str] = set()

    for rule in _signals():
        value = values.get(rule.metric)
        is_active = value is not None and rule.active(value)
        if is_active:
            active.add(rule.id)
        evidence_id = f"psi_ev_{rule.id}"
        evidence[rule.id] = PsychologicalEvidenceSignal(
            evidence_id=evidence_id,
            metric=rule.metric,
            observed_value=value,
            expected_range=rule.expected_range,
            direction=rule.direction,  # type: ignore[arg-type]
            weight=rule.weight,
            why_it_matters_psychologically=rule.why,
        )

    return evidence, active


def _uncertainty_factor(
    active_signals: set[str],
    audio_quality: AudioQuality,
    duration_ms: int,
) -> tuple[float, list[str]]:
    factor = 1.0
    reasons: list[str] = []

    if "poor_audio" in active_signals or not audio_quality.usable:
        factor -= 0.25
        reasons.append("Audio quality reduces confidence in fragile acoustic inference")
    if "short_recording" in active_signals or (duration_ms and duration_ms < 25000):
        factor -= 0.15
        reasons.append("Short recording limits deep perception inference")
    if "little_voiced_speech" in active_signals:
        factor -= 0.2
        reasons.append("Little voiced speech limits listener-perception evidence")

    return _clamp(factor, 0.25, 1.0), reasons


def _build_micro_behaviours(
    evidence_by_signal: dict[str, PsychologicalEvidenceSignal],
    active_signals: set[str],
    uncertainty_factor: float,
) -> list[MicroBehaviour]:
    behaviours: list[MicroBehaviour] = []
    rules_by_signal = {rule.id: rule for rule in _signals()}

    for rule in _behaviours():
        support_hits = [signal for signal in rule.supports if signal in active_signals]
        contradict_hits = [signal for signal in rule.contradicts if signal in active_signals]
        support_weight = sum(rules_by_signal[signal].weight for signal in support_hits)
        possible_weight = sum(rules_by_signal[signal].weight for signal in rule.supports)
        contradict_weight = sum(rules_by_signal[signal].weight for signal in contradict_hits)
        raw_confidence = (support_weight - contradict_weight * 0.6) / max(possible_weight, 0.1)
        confidence = _clamp(raw_confidence * uncertainty_factor)
        if len(support_hits) < rule.min_supports:
            confidence = min(confidence, 0.45)
        if contradict_hits:
            confidence = min(confidence, 0.7)

        uncertainty_reason = None
        if len(support_hits) < rule.min_supports:
            uncertainty_reason = "Insufficient independent signals for a strong behaviour inference"
        elif uncertainty_factor < 0.75:
            uncertainty_reason = "Recording quality or duration reduces confidence"

        behaviours.append(
            MicroBehaviour(
                id=rule.id,
                label=rule.label,
                confidence=round(confidence, 2),
                supporting_metrics=[evidence_by_signal[s].metric for s in support_hits],
                contradicting_metrics=[evidence_by_signal[s].metric for s in contradict_hits],
                supporting_evidence_ids=[evidence_by_signal[s].evidence_id for s in support_hits],
                uncertainty_reason=uncertainty_reason,
            )
        )

    return behaviours


def _build_traits(
    behaviours: list[MicroBehaviour],
    evidence_by_id: dict[str, PsychologicalEvidenceSignal],
    uncertainty_factor: float,
    scenario: str,
) -> list[PsychologicalTrait]:
    behaviour_by_id = {behaviour.id: behaviour for behaviour in behaviours}
    traits: list[PsychologicalTrait] = []

    for rule in _trait_rules():
        support_behaviours = [
            behaviour_by_id[behaviour_id]
            for behaviour_id in rule.supports
            if behaviour_id in behaviour_by_id and behaviour_by_id[behaviour_id].confidence >= 0.55
        ]
        contradict_behaviours = [
            behaviour_by_id[behaviour_id]
            for behaviour_id in rule.contradicts
            if behaviour_id in behaviour_by_id and behaviour_by_id[behaviour_id].confidence >= 0.55
        ]

        support_strength = sum(behaviour.confidence for behaviour in support_behaviours)
        contradict_strength = sum(behaviour.confidence for behaviour in contradict_behaviours)
        score = int(round(_clamp(0.5 + support_strength * 0.18 - contradict_strength * 0.14) * 100))

        independent_supports = len(support_behaviours)
        confidence = _clamp(
            (support_strength / max(independent_supports, 1)) * uncertainty_factor
        )
        if independent_supports < 2:
            confidence = min(confidence, 0.45)
        if contradict_behaviours:
            confidence = min(confidence, 0.72)

        evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for behaviour in support_behaviours
                for evidence_id in behaviour.supporting_evidence_ids
            )
        )
        metric_names = list(
            dict.fromkeys(
                metric
                for behaviour in support_behaviours
                for metric in behaviour.supporting_metrics
            )
        )
        evidence_chain = [
            evidence_by_id[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in evidence_by_id
        ]

        suppress = confidence < 0.4
        uncertainty_reason = None
        if independent_supports < 2:
            uncertainty_reason = "Trait requires multiple independent supporting behaviours"
            suppress = True
        elif uncertainty_factor < 0.75 and rule.fragile:
            uncertainty_reason = "Fragile acoustic trait suppressed or softened by recording uncertainty"
            suppress = True
        elif uncertainty_factor < 0.75:
            uncertainty_reason = "Recording uncertainty lowered confidence"

        scenario_adjustments = dict(rule.scenario_adjustments or {})
        relevance = calculate_trait_relevance(rule.id, scenario)
        scenario_adjustments[scenario or "benchmark"] = relevance
        if relevance != 1.0:
            confidence = _clamp(confidence * (0.92 + relevance * 0.08))
            if confidence < 0.4:
                suppress = True

        traits.append(
            PsychologicalTrait(
                trait_id=rule.id,
                label=rule.label,
                score=max(0, min(100, score)),
                confidence=round(confidence, 2),
                confidence_label=_confidence_label(confidence),  # type: ignore[arg-type]
                supporting_behaviours=[behaviour.id for behaviour in support_behaviours],
                contradicting_behaviours=[behaviour.id for behaviour in contradict_behaviours],
                supporting_metrics=metric_names,
                supporting_evidence_ids=evidence_ids,
                evidence_chain=evidence_chain,
                uncertainty_reason=uncertainty_reason,
                suppress_from_report=suppress,
                scenario_weight_adjustments=scenario_adjustments,
            )
        )

    return traits


def _candidate(
    candidate_id: str,
    label: str,
    traits: list[PsychologicalTrait],
    behaviours: list[MicroBehaviour],
    confidence: float,
) -> InferenceCandidate:
    evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for trait in traits
            for evidence_id in trait.supporting_evidence_ids
        )
    )
    return InferenceCandidate(
        candidate_id=candidate_id,
        label=label,
        trait_ids=[trait.trait_id for trait in traits],
        behaviour_ids=list(
            dict.fromkeys(
                behaviour.id
                for behaviour in behaviours
                if behaviour.confidence >= 0.55
            )
        ),
        evidence_ids=evidence_ids,
        confidence=round(confidence, 2),
    )


def _build_candidates(
    traits: list[PsychologicalTrait],
    behaviours: list[MicroBehaviour],
) -> tuple[PsychologicalPrimaryCandidates, PsychologicalReportCandidates]:
    visible = [trait for trait in traits if not trait.suppress_from_report]
    positive_ids = {
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
    limiter_ids = {"rushed", "nervous", "flat", "monotone", "hesitant", "approval_seeking"}

    positives = [trait for trait in visible if trait.trait_id in positive_ids]
    limiters = [trait for trait in visible if trait.trait_id in limiter_ids]
    strength = max(positives, key=lambda t: (t.score, t.confidence), default=None)
    limiter = max(limiters, key=lambda t: (t.score, t.confidence), default=None)
    weakest_positive = min(positives, key=lambda t: (t.score, -t.confidence), default=None)

    primary_strength = (
        _candidate("primary_strength", strength.label, [strength], behaviours, strength.confidence)
        if strength
        else None
    )
    limiter_trait = limiter or weakest_positive
    primary_limiter = (
        _candidate("primary_limiter", limiter_trait.label, [limiter_trait], behaviours, limiter_trait.confidence)
        if limiter_trait
        else None
    )
    core_tension = None
    if strength and limiter_trait:
        core_tension = _candidate(
            "core_tension",
            f"{strength.label} constrained by {limiter_trait.label}",
            [strength, limiter_trait],
            behaviours,
            min(strength.confidence, limiter_trait.confidence),
        )

    type_candidates = _authority_type_candidates(visible, behaviours)
    ordered_traits = sorted(visible, key=lambda t: (t.confidence, abs(t.score - 50)), reverse=True)

    hidden_cost_candidates = []
    if limiter_trait:
        hidden_cost_candidates.append(
            _candidate(
                f"hidden_cost_{limiter_trait.trait_id}",
                f"Hidden cost candidate: {limiter_trait.label}",
                [limiter_trait],
                behaviours,
                limiter_trait.confidence,
            )
        )

    leverage_candidates = []
    for trait in sorted(limiters, key=lambda t: (t.score, t.confidence), reverse=True)[:3]:
        leverage_candidates.append(
            _candidate(
                f"highest_leverage_{trait.trait_id}",
                f"Highest leverage candidate: reduce {trait.label}",
                [trait],
                behaviours,
                trait.confidence,
            )
        )

    return (
        PsychologicalPrimaryCandidates(
            primary_strength_candidate=primary_strength,
            primary_limiter_candidate=primary_limiter,
            core_tension_candidate=core_tension,
        ),
        PsychologicalReportCandidates(
            authority_type_candidates=type_candidates,
            strongest_positive_impression=primary_strength,
            strongest_negative_impression=primary_limiter if limiter else None,
            highest_confidence_trait=ordered_traits[0].trait_id if ordered_traits else None,
            lowest_confidence_trait=ordered_traits[-1].trait_id if ordered_traits else None,
            report_priority_order=[trait.trait_id for trait in ordered_traits[:8]],
            hidden_cost_candidates=hidden_cost_candidates,
            highest_leverage_candidates=leverage_candidates,
        ),
    )


def _authority_type_candidates(
    traits: list[PsychologicalTrait],
    behaviours: list[MicroBehaviour],
) -> list[InferenceCandidate]:
    by_id = {trait.trait_id: trait for trait in traits}

    def has(trait_id: str, score: int = 60) -> bool:
        trait = by_id.get(trait_id)
        return bool(trait and trait.score >= score and trait.confidence >= 0.45)

    candidates: list[InferenceCandidate] = []
    if has("executive_presence", 70) and has("commanding", 65):
        source = [by_id["executive_presence"], by_id["commanding"]]
        candidates.append(_candidate("type_executive_presence", "The Executive Presence", source, behaviours, min(t.confidence for t in source)))
    if has("credible") and has("structured_thinker"):
        source = [by_id["credible"], by_id["structured_thinker"]]
        candidates.append(_candidate("type_trusted_expert", "The Trusted Expert", source, behaviours, min(t.confidence for t in source)))
    if has("composed") and has("calm"):
        source = [by_id["composed"], by_id["calm"]]
        candidates.append(_candidate("type_calm_professional", "The Calm Professional", source, behaviours, min(t.confidence for t in source)))
    if has("rushed") and has("clear_communicator", 50):
        source = [by_id["rushed"], by_id["clear_communicator"]]
        candidates.append(_candidate("type_rushed_achiever", "The Rushed Achiever", source, behaviours, min(t.confidence for t in source)))
    if has("flat") and has("credible", 50):
        source = [by_id["flat"], by_id["credible"]]
        candidates.append(_candidate("type_quiet_analyst", "The Quiet Analyst", source, behaviours, min(t.confidence for t in source)))
    if has("persuasive") and has("energetic"):
        source = [by_id["persuasive"], by_id["energetic"]]
        candidates.append(_candidate("type_persuasive_operator", "The Persuasive Operator", source, behaviours, min(t.confidence for t in source)))

    if not candidates:
        visible = [trait for trait in traits if trait.confidence >= 0.4]
        candidates.append(
            _candidate(
                "type_developing_voice",
                "The Developing Voice",
                sorted(visible, key=lambda t: t.confidence, reverse=True)[:2],
                behaviours,
                min(0.55, max((trait.confidence for trait in visible), default=0.35)),
            )
        )

    return candidates[:3]


def build_psychological_inference(
    *,
    metrics: Metrics,
    scores: Scores,
    audio_quality: AudioQuality,
    uncertainty: Uncertainty,
    duration_ms: int,
    scenario: str,
    asr_confidence: float | None = None,
) -> PsychologicalInference:
    """Build deterministic micro-behaviour, trait, and report-candidate inference."""
    values = _metric_values(metrics, scores, audio_quality, duration_ms)
    values["transcript.overall_asr_confidence"] = asr_confidence

    evidence_by_signal, active_signals = _evaluate_signals(values)
    uncertainty_factor, reasons = _uncertainty_factor(active_signals, audio_quality, duration_ms)
    evidence_by_id = {
        evidence_by_signal[signal_id].evidence_id: evidence_by_signal[signal_id]
        for signal_id in active_signals
        if signal_id in evidence_by_signal and evidence_by_signal[signal_id].observed_value is not None
    }

    behaviours = _build_micro_behaviours(
        evidence_by_signal,
        active_signals,
        uncertainty_factor,
    )
    traits = _build_traits(
        behaviours,
        evidence_by_id,
        uncertainty_factor,
        scenario,
    )
    primary_candidates, report_candidates = _build_candidates(traits, behaviours)

    suppressed_traits = [
        trait.trait_id for trait in traits if trait.suppress_from_report
    ]
    visible_confidences = [
        trait.confidence for trait in traits if not trait.suppress_from_report
    ]
    overall_confidence = round(
        sum(visible_confidences) / len(visible_confidences)
        if visible_confidences
        else 0.0,
        2,
    )

    all_reasons = list(dict.fromkeys(list(uncertainty.reasons) + reasons))
    inference_uncertainty = Uncertainty(
        overall_confidence_label=_confidence_label(overall_confidence),  # type: ignore[arg-type]
        suppressed_traits=list(dict.fromkeys(uncertainty.suppressed_traits + suppressed_traits)),
        reasons=all_reasons,
    )

    return PsychologicalInference(
        micro_behaviours=behaviours,
        traits=traits,
        evidence_chain=list(evidence_by_id.values()),
        primary_candidates=primary_candidates,
        report_candidates=report_candidates,
        overall_inference_confidence=overall_confidence,
        suppressed_traits=list(dict.fromkeys(suppressed_traits)),
        uncertainty=inference_uncertainty,
    )
