"""Authority v2 scoring: six dimensions, derived axes, bonuses and penalties."""

from __future__ import annotations

from dataclasses import dataclass

from schemas import (
    DerivedAxes,
    DimensionScores,
    ScoreBonuses,
    ScoreComponents,
    ScorePenalties,
    Scores,
)
from services.acoustic_metrics import AcousticAnalysisResult


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class ScoringResult:
    scores: Scores
    legacy_authority_score: float
    dimension_map: dict[str, int]
    delivery_score: float
    content_score: float


def _pace_score(wpm: float) -> float:
    if 115 <= wpm <= 155:
        return 85
    if 105 <= wpm < 115 or 155 < wpm <= 170:
        return 72
    if 95 <= wpm < 105 or 170 < wpm <= 180:
        return 55
    if 85 <= wpm < 95 or 180 < wpm <= 195:
        return 38
    return 25


def _pause_control_score(avg_pause: float) -> float:
    if 0.30 <= avg_pause <= 0.75:
        return 85
    if 0.22 <= avg_pause < 0.30 or 0.75 < avg_pause <= 1.0:
        return 68
    if 0.16 <= avg_pause < 0.22 or 1.0 < avg_pause <= 1.3:
        return 48
    return 28


def _rhythm_score(pause_frequency: float) -> float:
    if 0.18 <= pause_frequency <= 0.35:
        return 85
    if 0.35 < pause_frequency <= 0.45:
        return 65
    if 0.45 < pause_frequency <= 0.55:
        return 45
    if pause_frequency < 0.18:
        return 50
    return 28


def _vocal_control_score(pitch_variation: float) -> float:
    if 32 <= pitch_variation <= 60:
        return 85
    if 24 <= pitch_variation < 32 or 60 < pitch_variation <= 72:
        return 65
    if 18 <= pitch_variation < 24 or 72 < pitch_variation <= 85:
        return 45
    return 25


def _energy_control_score(energy_mean: float, energy_variation: float) -> float:
    if 45 <= energy_mean <= 60:
        energy_base = 80
    elif 40 <= energy_mean < 45 or 60 < energy_mean <= 66:
        energy_base = 65
    elif 35 <= energy_mean < 40 or 66 < energy_mean <= 72:
        energy_base = 48
    else:
        energy_base = 30

    if 10 <= energy_variation <= 22:
        bonus = 5
    elif 7 <= energy_variation < 10:
        bonus = 0
    elif 4 <= energy_variation < 7:
        bonus = -12
    else:
        bonus = -20

    return _clamp(energy_base + bonus, 20, 90)


def _silence_control_score(silence_ratio: float) -> float:
    if 0.10 <= silence_ratio <= 0.24:
        return 85
    if 0.24 < silence_ratio <= 0.30 or 0.07 <= silence_ratio < 0.10:
        return 65
    if 0.30 < silence_ratio <= 0.36 or 0.05 <= silence_ratio < 0.07:
        return 45
    return 25


def compute_dimension_scores(
    voice_metrics: dict,
    cognitive_metrics: dict,
    delivery_metrics: dict,
    linguistic: dict | None = None,
    acoustic: AcousticAnalysisResult | None = None,
) -> tuple[DimensionScores, float, float, dict[str, float]]:
    """Map v1 measurement stack into v2 six dimensions."""
    wpm = delivery_metrics.get("words_per_minute", 0)
    filler_density = delivery_metrics.get("filler_density", 0)

    pitch_variation = voice_metrics.get("pitch_variation", 0)
    energy_mean = voice_metrics.get("energy_mean", 0)
    energy_variation = voice_metrics.get("energy_variation", 0)
    silence_ratio = voice_metrics.get("silence_ratio", 0)
    avg_pause = voice_metrics.get("avg_pause_duration", 0)
    pause_frequency = voice_metrics.get("pause_frequency", 0)

    pace = _pace_score(wpm)
    pause_control = _pause_control_score(avg_pause)
    rhythm = _rhythm_score(pause_frequency)
    vocal_control = _vocal_control_score(pitch_variation)
    energy_control = _energy_control_score(energy_mean, energy_variation)
    silence_control = _silence_control_score(silence_ratio)

    delivery_score = (
        pace + pause_control + rhythm + vocal_control + energy_control + silence_control
    ) / 6

    clarity_cog = cognitive_metrics["clarity"]["score"]
    persuasion_cog = cognitive_metrics["persuasion"]["score"]
    coherence_cog = cognitive_metrics["coherence"]["score"]
    idea_strength = cognitive_metrics["idea_strength"]["score"]
    conciseness = cognitive_metrics["conciseness"]["score"]

    content_score = (
        clarity_cog + persuasion_cog + coherence_cog + idea_strength + conciseness
    ) / 5

    filler_penalty = 0
    if filler_density > 0.10:
        filler_penalty = 20
    elif filler_density > 0.05:
        filler_penalty = 12
    elif filler_density > 0.02:
        filler_penalty = 6

    command = _clamp(
        (pace + pause_control + rhythm) / 3 - filler_penalty * 0.3,
        20,
        95,
    )
    clarity = _clamp((clarity_cog * 0.6 + delivery_score * 0.4) - filler_penalty * 0.4, 20, 95)
    composure = _clamp((silence_control + vocal_control + rhythm) / 3, 20, 95)
    presence = _clamp((energy_control + vocal_control) / 2, 20, 95)
    persuasion = _clamp((persuasion_cog * 0.55 + presence * 0.45), 20, 95)

    opening = linguistic.get("opening_strength_score", 0.5) if linguistic else 0.5
    closing = linguistic.get("closing_strength_score", 0.5) if linguistic else 0.5
    structure_det = linguistic.get("structure_score") if linguistic else None
    structure_base = (
        coherence_cog * 0.4 + conciseness * 0.3 + idea_strength * 0.3
    ) * 0.7 + (opening + closing) * 50 * 0.3
    if structure_det is not None:
        structure_base = structure_base * 0.55 + structure_det * 100 * 0.45
    structure = _clamp(structure_base, 20, 95)

    dimensions = DimensionScores(
        command=int(round(command)),
        clarity=int(round(clarity)),
        composure=int(round(composure)),
        presence=int(round(presence)),
        persuasion=int(round(persuasion)),
        structure=int(round(structure)),
    )

    penalties = {
        "filler_penalty": min(10.0, filler_penalty / 2),
        "rambling_penalty": 0.0,
        "monotony_penalty": 0.0,
        "rising_ending_penalty": 0.0,
        "audio_quality_penalty": 0.0,
    }

    if conciseness < 45:
        penalties["rambling_penalty"] = 4.0
    if linguistic and linguistic.get("rambling_score"):
        penalties["rambling_penalty"] = max(
            penalties["rambling_penalty"],
            min(10.0, float(linguistic["rambling_score"]) * 10),
        )
    if energy_variation < 6:
        penalties["monotony_penalty"] = min(8.0, (8 - energy_variation) * 1.2)
    if acoustic and acoustic.derived.monotony_index is not None:
        penalties["monotony_penalty"] = max(
            penalties["monotony_penalty"],
            min(8.0, acoustic.derived.monotony_index * 10),
        )
    terminal_rise = voice_metrics.get("terminal_rise_ratio")
    if terminal_rise is not None and terminal_rise > 0.3:
        penalties["rising_ending_penalty"] = min(6.0, float(terminal_rise) * 8)

    bonuses = {
        "opening_strength": round(max(0.0, (opening - 0.5) * 6), 1),
        "ending_strength": round(max(0.0, (closing - 0.5) * 6), 1),
        "consistency_bonus": 0.0,
    }

    return dimensions, delivery_score, content_score, {**penalties, **bonuses}


def compute_derived_axes(dimension_scores: DimensionScores, delivery_metrics: dict) -> DerivedAxes:
    """Derive trust, dominance, nervousness, and readiness axes."""
    filler_density = delivery_metrics.get("filler_density", 0)
    nervousness = int(
        _clamp(
            100
            - dimension_scores.composure * 0.5
            - dimension_scores.command * 0.2
            + filler_density * 120,
            10,
            90,
        )
    )

    return DerivedAxes(
        trust_warmth=int(
            _clamp(
                dimension_scores.clarity * 0.35
                + dimension_scores.structure * 0.25
                + (100 - nervousness) * 0.4,
                20,
                95,
            )
        ),
        dominance_status=int(
            _clamp(
                dimension_scores.command * 0.45 + dimension_scores.presence * 0.35 + dimension_scores.persuasion * 0.2,
                20,
                95,
            )
        ),
        nervousness=nervousness,
        interview_readiness=int(
            _clamp(
                dimension_scores.clarity * 0.3
                + dimension_scores.structure * 0.3
                + dimension_scores.composure * 0.4,
                20,
                95,
            )
        ),
        leadership_readiness=int(
            _clamp(
                dimension_scores.command * 0.35
                + dimension_scores.structure * 0.25
                + dimension_scores.presence * 0.2
                + dimension_scores.persuasion * 0.2,
                20,
                95,
            )
        ),
    )


def compute_authority_score(
    voice_metrics: dict,
    cognitive_metrics: dict,
    delivery_metrics: dict,
    linguistic: dict | None = None,
    audio_quality_penalty: float = 0.0,
    acoustic: AcousticAnalysisResult | None = None,
) -> ScoringResult:
    """Compute v2 authority score with v1-compatible guardrails."""
    # TODO(Milestone 3): replace blended cognitive/deterministic scoring with calibrated model
    dimensions, delivery_score, content_score, adjustments = compute_dimension_scores(
        voice_metrics, cognitive_metrics, delivery_metrics, linguistic, acoustic
    )

    weighted_base = (
        0.22 * dimensions.command
        + 0.20 * dimensions.clarity
        + 0.17 * dimensions.composure
        + 0.15 * dimensions.presence
        + 0.14 * dimensions.persuasion
        + 0.12 * dimensions.structure
    )

    bonuses = ScoreBonuses(
        opening_strength=adjustments.get("opening_strength", 0.0),
        ending_strength=adjustments.get("ending_strength", 0.0),
        consistency_bonus=adjustments.get("consistency_bonus", 0.0),
    )
    penalties = ScorePenalties(
        filler_penalty=adjustments.get("filler_penalty", 0.0),
        rambling_penalty=adjustments.get("rambling_penalty", 0.0),
        monotony_penalty=adjustments.get("monotony_penalty", 0.0),
        rising_ending_penalty=adjustments.get("rising_ending_penalty", 0.0),
        audio_quality_penalty=audio_quality_penalty,
    )

    bonus_total = bonuses.opening_strength + bonuses.ending_strength + bonuses.consistency_bonus
    penalty_total = (
        penalties.filler_penalty
        + penalties.rambling_penalty
        + penalties.monotony_penalty
        + penalties.rising_ending_penalty
        + penalties.audio_quality_penalty
    )

    failure = cognitive_metrics.get("failure", False)
    final_score = weighted_base + bonus_total - penalty_total

    if failure:
        final_score = min(final_score, 45)

    # Delivery caps preserved from v1
    if delivery_score < 35:
        final_score = min(final_score, 42)
    elif delivery_score < 45:
        final_score = min(final_score, 50)
    elif delivery_score < 55:
        final_score = min(final_score, 58)
    elif delivery_score < 62:
        final_score = min(final_score, 64)

    wpm = delivery_metrics.get("words_per_minute", 0)
    pause_frequency = voice_metrics.get("pause_frequency", 0)
    silence_ratio = voice_metrics.get("silence_ratio", 0)
    avg_pause = voice_metrics.get("avg_pause_duration", 0)
    pitch_variation = voice_metrics.get("pitch_variation", 0)
    energy_variation = voice_metrics.get("energy_variation", 0)

    red_flags = 0
    if wpm < 95 or wpm > 185:
        red_flags += 1
    if pause_frequency > 0.50:
        red_flags += 1
    if silence_ratio > 0.30:
        red_flags += 1
    if avg_pause > 1.1:
        red_flags += 1
    if pitch_variation < 22 or pitch_variation > 80:
        red_flags += 1
    if energy_variation < 6:
        red_flags += 1

    if red_flags >= 4:
        final_score = min(final_score, 46)
    elif red_flags >= 3:
        final_score = min(final_score, 52)
    elif red_flags >= 2:
        final_score = min(final_score, 60)

    final_score = _clamp(final_score, 25, 95)
    authority_int = int(round(final_score))

    # TODO(v2.3): isotonic calibration against human-rated corpus
    percentile = round(_clamp((authority_int - 25) / 70, 0.05, 0.95), 2)

    derived_axes = compute_derived_axes(dimensions, delivery_metrics)
    dimension_map = dimensions.model_dump()

    scores = Scores(
        authority_score=authority_int,
        authority_percentile_estimate=percentile,
        score_confidence=0.79 if not failure else 0.45,
        dimension_scores=dimensions,
        derived_axes=derived_axes,
        score_components=ScoreComponents(
            weighted_base=round(weighted_base, 1),
            bonuses=bonuses,
            penalties=penalties,
        ),
    )

    return ScoringResult(
        scores=scores,
        legacy_authority_score=round(final_score, 2),
        dimension_map=dimension_map,
        delivery_score=delivery_score,
        content_score=content_score,
    )


# Backward-compatible entry point for legacy imports
def compute_authority_score_legacy(voice_metrics, cognitive_metrics, delivery_metrics):
    return compute_authority_score(
        voice_metrics, cognitive_metrics, delivery_metrics
    ).legacy_authority_score
