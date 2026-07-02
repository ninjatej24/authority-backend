"""Authority scoring and deterministic calibration v2."""

from __future__ import annotations

from dataclasses import dataclass

from schemas import (
    CalibrationMetadata,
    DerivedAxes,
    DimensionScoreDetail,
    DimensionScores,
    FairnessAdjustments,
    ScoreBonuses,
    ScoreCap,
    ScoreComponentItem,
    ScoreComponents,
    ScoreExplanation,
    ScorePenalties,
    Scores,
)
from services.acoustic_metrics import AcousticAnalysisResult


DIMENSION_WEIGHTS = {
    "command": 0.22,
    "clarity": 0.20,
    "composure": 0.17,
    "presence": 0.15,
    "persuasion": 0.14,
    "structure": 0.12,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _score_01(value: float | None, default: float = 0.5) -> float:
    if value is None:
        return default
    return _clamp(float(value), 0.0, 1.0)


def _cognitive(cognitive_metrics: dict, key: str, default: float = 50.0) -> float:
    item = cognitive_metrics.get(key, {})
    if isinstance(item, dict):
        return float(item.get("score", default))
    return default


def _metric(source: dict | None, key: str, default: float = 0.0) -> float:
    if not source:
        return default
    value = source.get(key, default)
    if value is None:
        return default
    return float(value)


def _pace_score(wpm: float) -> float:
    if 120 <= wpm <= 155:
        return 88
    if 105 <= wpm < 120 or 155 < wpm <= 170:
        return 74
    if 90 <= wpm < 105 or 170 < wpm <= 185:
        return 55
    if 75 <= wpm < 90 or 185 < wpm <= 205:
        return 34
    return 22


def _pause_ownership_score(avg_pause: float, pause_frequency: float, silence_ratio: float) -> float:
    avg = 84 if 0.30 <= avg_pause <= 0.85 else 66 if 0.20 <= avg_pause <= 1.1 else 44 if 0.12 <= avg_pause <= 1.45 else 26
    frequency = 84 if 0.10 <= pause_frequency <= 0.32 else 62 if pause_frequency <= 0.45 else 42 if pause_frequency <= 0.62 else 24
    silence = 84 if 0.08 <= silence_ratio <= 0.26 else 64 if silence_ratio <= 0.34 else 38 if silence_ratio <= 0.45 else 22
    return avg * 0.4 + frequency * 0.35 + silence * 0.25


def _relative_pitch_score(pitch_variation: float, monotony: float) -> float:
    if 5.0 <= pitch_variation <= 12.5:
        base = 84
    elif 3.0 <= pitch_variation < 5.0 or 12.5 < pitch_variation <= 16.0:
        base = 66
    elif 1.5 <= pitch_variation < 3.0 or 16.0 < pitch_variation <= 20.0:
        base = 46
    else:
        base = 30
    return _clamp(base - monotony * 18, 20, 92)


def _energy_variation_score(energy_variation: float, dynamic_emphasis: float, projection_index: float) -> float:
    variation = 84 if 5.0 <= energy_variation <= 16.0 else 62 if 3.0 <= energy_variation < 5.0 or 16.0 < energy_variation <= 22.0 else 38
    return _clamp(variation * 0.45 + dynamic_emphasis * 100 * 0.35 + projection_index * 100 * 0.20, 20, 94)


def _filler_penalty_per_min(filler_per_min: float) -> float:
    if filler_per_min < 4.0:
        return max(0.0, (filler_per_min - 2.0) * 0.8)
    if filler_per_min < 8.0:
        return 2.0 + (filler_per_min - 4.0) * 1.25
    return min(10.0, 7.0 + (filler_per_min - 8.0) * 0.75)


def _score_band(score: int) -> tuple[str, str, str, str]:
    if score <= 38:
        return (
            "bottom_10",
            "Fragile authority signal",
            "Hard to follow, fragile control, heavy disruption, or weak opening and close.",
            "bottom 10%",
        )
    if score <= 52:
        return (
            "developing",
            "Developing authority signal",
            "Understandable but noticeably tentative, rushed, flat, or disorganised.",
            "next 25%",
        )
    if score <= 66:
        return (
            "competent",
            "Competent but not yet magnetic",
            "Generally competent; listeners understand but may not naturally defer.",
            "middle 30%",
        )
    if score <= 80:
        return (
            "strong",
            "Recognisably authoritative",
            "Clear, composed, persuasive, and recognisably authoritative.",
            "next 25%",
        )
    if score <= 90:
        return (
            "excellent",
            "High authority signal",
            "Clear and self-possessed enough that listeners are likely to trust the floor.",
            "top 8%",
        )
    return (
        "elite",
        "Rare authority signal",
        "Rare; people are likely to trust this speaker with the floor immediately.",
        "top 2%",
    )


def _calibrate(latent_score: float) -> tuple[float, list[str]]:
    anchors = [
        (20.0, 20.0),
        (38.0, 38.0),
        (52.0, 52.0),
        (66.0, 66.0),
        (80.0, 80.0),
        (90.0, 90.0),
        (97.0, 97.0),
    ]
    if latent_score <= anchors[0][0]:
        return anchors[0][1], ["clamped_to_lower_anchor"]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if latent_score <= x1:
            ratio = (latent_score - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0), ["deterministic_monotonic_anchor_mapping"]
    return 97.0, ["clamped_to_elite_anchor"]


def _percentile(score: int) -> float:
    if score <= 38:
        return round(0.02 + (score - 20) / 18 * 0.08, 2)
    if score <= 52:
        return round(0.10 + (score - 39) / 13 * 0.25, 2)
    if score <= 66:
        return round(0.35 + (score - 53) / 13 * 0.30, 2)
    if score <= 80:
        return round(0.65 + (score - 67) / 13 * 0.25, 2)
    if score <= 90:
        return round(0.90 + (score - 81) / 9 * 0.08, 2)
    return round(0.98 + (score - 91) / 6 * 0.02, 2)


@dataclass
class ScoringResult:
    scores: Scores
    legacy_authority_score: float
    dimension_map: dict[str, int]
    delivery_score: float
    content_score: float


def compute_dimension_scores(
    voice_metrics: dict,
    cognitive_metrics: dict,
    delivery_metrics: dict,
    linguistic: dict | None = None,
    acoustic: AcousticAnalysisResult | None = None,
) -> tuple[DimensionScores, dict[str, DimensionScoreDetail], float, float, dict[str, float], list[ScoreComponentItem], list[ScoreComponentItem], FairnessAdjustments]:
    """Map deterministic measurements into the final six Authority dimensions."""
    wpm = _metric(delivery_metrics, "words_per_minute")
    filler_density = _metric(delivery_metrics, "filler_density")
    duration_seconds = _metric(voice_metrics, "duration_seconds", 0.0)
    filler_per_min = filler_density * max(wpm, 1.0)

    pitch_variation = _metric(voice_metrics, "pitch_variation")
    energy_variation = _metric(voice_metrics, "energy_variation")
    silence_ratio = _metric(voice_metrics, "silence_ratio")
    avg_pause = _metric(voice_metrics, "avg_pause_duration")
    pause_frequency = _metric(voice_metrics, "pause_frequency")
    terminal_rise = _metric(voice_metrics, "terminal_rise_ratio")

    clarity_cog = _cognitive(cognitive_metrics, "clarity")
    persuasion_cog = _cognitive(cognitive_metrics, "persuasion")
    coherence_cog = _cognitive(cognitive_metrics, "coherence")
    idea_strength = _cognitive(cognitive_metrics, "idea_strength")
    conciseness = _cognitive(cognitive_metrics, "conciseness")

    opening = _score_01(linguistic.get("opening_strength_score") if linguistic else None)
    closing = _score_01(linguistic.get("closing_strength_score") if linguistic else None)
    structure_det = _score_01(linguistic.get("structure_score") if linguistic else None)
    rambling = _score_01(linguistic.get("rambling_score") if linguistic else None, 0.2)
    specificity = _score_01(linguistic.get("specificity_score") if linguistic else None)
    concreteness = _score_01(linguistic.get("concreteness_score") if linguistic else None)
    hedges = _metric(linguistic, "hedges_per_100_words")
    certainty = _metric(linguistic, "certainty_markers_per_100_words")
    filler_linguistic = _metric(linguistic, "filler_words_per_min", filler_per_min)
    filler_per_min = max(filler_per_min, filler_linguistic)

    derived = acoustic.derived if acoustic else None
    monotony = _score_01(getattr(derived, "monotony_index", None), 0.0)
    hesitation = _score_01(getattr(derived, "hesitation_cluster_score", None), 0.0)
    dynamic_emphasis = _score_01(getattr(derived, "dynamic_emphasis_score", None), 0.5)
    speech_continuity = _score_01(getattr(derived, "speech_continuity_score", None), 0.5)
    projection = _score_01(getattr(derived, "projection_index", None), 0.5)
    rhythm_index = _score_01(getattr(derived, "rhythm_index", None), 0.5)
    confidence_drops = int(getattr(derived, "confidence_drop_count", 0) or 0)

    pace = _pace_score(wpm)
    pause_ownership = _pause_ownership_score(avg_pause, pause_frequency, silence_ratio)
    pitch_control = _relative_pitch_score(pitch_variation, monotony)
    energy_control = _energy_variation_score(energy_variation, dynamic_emphasis, projection)
    filler_penalty = _filler_penalty_per_min(filler_per_min)
    rising_cluster = terminal_rise >= 0.35 and closing < 0.5 and certainty < 2.0

    command = (
        pause_ownership * 0.28
        + (100 - min(100, terminal_rise * 120)) * 0.20
        + opening * 100 * 0.14
        + closing * 100 * 0.18
        + (100 - min(100, hedges * 8)) * 0.10
        + pace * 0.10
        - filler_penalty * 0.65
        - (10 if rising_cluster else 0)
    )
    clarity = (
        clarity_cog * 0.26
        + (100 - min(100, filler_per_min * 6)) * 0.20
        + (100 - rambling * 100) * 0.16
        + specificity * 100 * 0.12
        + speech_continuity * 100 * 0.10
        + structure_det * 100 * 0.16
    )
    composure = (
        rhythm_index * 100 * 0.24
        + (100 - hesitation * 100) * 0.22
        + pace * 0.18
        + pitch_control * 0.14
        + pause_ownership * 0.14
        + (100 - min(100, hedges * 9)) * 0.08
    )
    presence = (
        energy_control * 0.36
        + pitch_control * 0.22
        + dynamic_emphasis * 100 * 0.22
        + (100 - monotony * 100) * 0.20
    )
    persuasion = (
        persuasion_cog * 0.26
        + certainty * 10 * 0.14
        + concreteness * 100 * 0.14
        + specificity * 100 * 0.14
        + presence * 0.18
        + command * 0.14
    )
    structure = (
        opening * 100 * 0.22
        + structure_det * 100 * 0.28
        + closing * 100 * 0.20
        + coherence_cog * 0.12
        + conciseness * 0.08
        + idea_strength * 0.06
        + (100 - rambling * 100) * 0.04
    )

    raw_dimensions = {
        "command": command,
        "clarity": clarity,
        "composure": composure,
        "presence": presence,
        "persuasion": persuasion,
        "structure": structure,
    }
    dimensions = DimensionScores(**{key: int(round(_clamp(value, 20, 97))) for key, value in raw_dimensions.items()})
    delivery_score = (pace + pause_ownership + pitch_control + energy_control) / 4
    content_score = (clarity_cog + persuasion_cog + coherence_cog + idea_strength + conciseness) / 5

    confidence = 0.82
    missing_reasons = []
    for label, value in {
        "pace": wpm,
        "pause timing": avg_pause,
        "pitch variation": pitch_variation,
        "energy variation": energy_variation,
    }.items():
        if value <= 0:
            missing_reasons.append(f"Missing or weak {label} measurement")
            confidence -= 0.06
    if duration_seconds and duration_seconds < 12:
        missing_reasons.append("Short usable speech limits dimension confidence")
        confidence -= 0.12

    detail_templates = {
        "command": (
            ["owned pauses", "clean opening", "strong close", "direct certainty language"],
            ["rising endings", "weak close", "high hedging", "clustered fillers"],
        ),
        "clarity": (
            ["clear transcript meaning", "low rambling", "specific wording", "speech continuity"],
            ["filler burden", "rambling", "low specificity"],
        ),
        "composure": (
            ["steady rhythm", "low hesitation clustering", "stable pace"],
            ["hesitation clustering", "pace pressure", "unstable pauses"],
        ),
        "presence": (
            ["dynamic emphasis", "relative pitch movement", "energy contrast"],
            ["monotony", "low energy variation", "flat emphasis"],
        ),
        "persuasion": (
            ["certainty language", "specific proof", "dynamic emphasis"],
            ["low listener pull", "weak ending", "vague support"],
        ),
        "structure": (
            ["strong opening", "clear sequence", "strong close"],
            ["rambling", "weak opening", "weak close"],
        ),
    }
    details: dict[str, DimensionScoreDetail] = {}
    for key, score in dimensions.model_dump().items():
        positives, negatives = detail_templates[key]
        details[key] = DimensionScoreDetail(
            score=score,
            confidence=round(_clamp(confidence, 0.25, 0.95), 2),
            positive_contributors=positives[: 2 if score < 70 else 4],
            negative_contributors=negatives[: 3 if score < 70 else 1],
            uncertainty_reasons=missing_reasons[:],
        )

    penalties = {
        "filler_penalty": round(filler_penalty, 2),
        "rambling_penalty": round(min(10.0, max(0.0, (rambling - 0.35) * 16)), 2),
        "monotony_penalty": round(min(8.0, max(0.0, (monotony - 0.35) * 14 + (1 - dynamic_emphasis) * 2)), 2),
        "rising_ending_penalty": round(min(6.0, max(0.0, (terminal_rise - 0.25) * 12) + (2.0 if rising_cluster else 0.0)), 2),
        "mid_recording_collapse_penalty": round(min(8.0, confidence_drops * 3.0), 2),
    }
    bonuses = {
        "opening_strength": round(min(3.0, max(0.0, (opening - 0.72) * 10.5)), 2),
        "ending_strength": round(min(3.0, max(0.0, (closing - 0.72) * 10.5)), 2),
        "consistency_bonus": round(min(2.0, max(0.0, (speech_continuity - 0.78) * 6.5 + (rhythm_index - 0.75) * 3.0)), 2),
    }

    penalty_items = [
        ScoreComponentItem(id=key, label=key.replace("_", " ").title(), value=value, reason="Thresholded deterministic scoring penalty.")
        for key, value in penalties.items()
        if value > 0
    ]
    bonus_items = [
        ScoreComponentItem(id=key, label=key.replace("_", " ").title(), value=value, reason="Small deterministic bonus for unusually clean supporting evidence.")
        for key, value in bonuses.items()
        if value > 0
    ]

    fairness = FairnessAdjustments(
        applied_adjustments=["used_relative_pitch_variation", "used_relative_energy_variation"],
        suppressed_features=["absolute_pitch", "microphone_loudness", "accent_sensitive_asr_as_primary_penalty"],
        reasons=[
            "Natural pitch and microphone loudness are not treated as authority by themselves.",
            "ASR confidence affects score confidence more than direct score punishment.",
        ],
    )

    return dimensions, details, delivery_score, content_score, {**penalties, **bonuses}, bonus_items, penalty_items, fairness


def compute_derived_axes(dimension_scores: DimensionScores, delivery_metrics: dict) -> DerivedAxes:
    filler_density = _metric(delivery_metrics, "filler_density")
    nervousness = int(
        round(
            _clamp(
                100
                - dimension_scores.composure * 0.46
                - dimension_scores.command * 0.22
                + filler_density * 145,
                8,
                94,
            )
        )
    )
    return DerivedAxes(
        trust_warmth=int(round(_clamp(dimension_scores.clarity * 0.32 + dimension_scores.structure * 0.24 + (100 - nervousness) * 0.32 + dimension_scores.composure * 0.12, 20, 96))),
        dominance_status=int(round(_clamp(dimension_scores.command * 0.50 + dimension_scores.presence * 0.28 + dimension_scores.persuasion * 0.22, 20, 97))),
        nervousness=nervousness,
        interview_readiness=int(round(_clamp(dimension_scores.clarity * 0.30 + dimension_scores.structure * 0.32 + dimension_scores.composure * 0.25 + dimension_scores.command * 0.13, 20, 97))),
        leadership_readiness=int(round(_clamp(dimension_scores.command * 0.34 + dimension_scores.structure * 0.22 + dimension_scores.presence * 0.20 + dimension_scores.persuasion * 0.16 + dimension_scores.composure * 0.08, 20, 97))),
    )


def _confidence(
    *,
    failure: bool,
    audio_quality_usable: bool,
    asr_confidence: float | None,
    duration_ms: int | None,
    acoustic: AcousticAnalysisResult | None,
    audio_quality_penalty: float,
) -> tuple[float, str, list[str], float]:
    confidence = 0.86
    reasons: list[str] = []
    if failure:
        confidence -= 0.25
        reasons.append("Transcript scoring was unavailable")
    if not audio_quality_usable:
        confidence -= 0.25
        reasons.append("Audio quality was not fully usable")
    if audio_quality_penalty > 0:
        confidence -= min(0.12, audio_quality_penalty / 100)
        reasons.append("Audio quality penalty reduced score confidence")
    if asr_confidence is not None and asr_confidence < 0.65:
        confidence -= 0.14
        reasons.append("ASR confidence was low")
    speech_seconds = acoustic.speaking_seconds if acoustic else 0.0
    if speech_seconds and speech_seconds < 8:
        confidence -= 0.16
        reasons.append("Very short usable speech")
    elif duration_ms is not None and duration_ms < 8000:
        confidence -= 0.12
        reasons.append("Short recording")
    if acoustic and acoustic.derived.monotony_index is None:
        confidence -= 0.05
        reasons.append("Some derived acoustic metrics were unavailable")

    confidence = round(_clamp(confidence, 0.25, 0.95), 2)
    label = "high" if confidence >= 0.8 else "medium_high" if confidence >= 0.65 else "medium" if confidence >= 0.45 else "low"
    low_confidence_penalty = 0.0 if confidence >= 0.7 else round((0.7 - confidence) * 10, 2)
    return confidence, label, reasons, low_confidence_penalty


def _caps(
    dimensions: DimensionScores,
    penalties: ScorePenalties,
    *,
    failure: bool,
    audio_quality_usable: bool,
    speech_seconds: float,
    score_confidence: float,
) -> list[ScoreCap]:
    caps: list[ScoreCap] = []
    if failure:
        caps.append(ScoreCap(id="transcript_failure_cap", label="Transcript Failure Cap", value=45, reason="Unavailable transcript/content scoring prevents confident high scoring."))
    if not audio_quality_usable or penalties.audio_quality_penalty >= 10:
        caps.append(ScoreCap(id="poor_audio_cap", label="Poor Audio Cap", value=72, reason="Poor audio reduces confidence and prevents elite scoring."))
    if speech_seconds and speech_seconds < 8:
        caps.append(ScoreCap(id="very_short_speech_cap", label="Very Short Speech Cap", value=58, reason="Too little usable speech to support a high score."))
    elif speech_seconds and speech_seconds < 15:
        caps.append(ScoreCap(id="short_speech_cap", label="Short Speech Cap", value=72, reason="Short usable speech limits score certainty."))
    if penalties.filler_penalty >= 7:
        caps.append(ScoreCap(id="severe_filler_cap", label="Severe Filler Cap", value=82, reason="Severe filler burden prevents elite scoring."))
    if dimensions.command < 65 or dimensions.clarity < 65:
        caps.append(ScoreCap(id="command_clarity_elite_cap", label="Command Or Clarity Cap", value=89, reason="Scores above 90 require both command and clarity."))
    if any(value >= 7 for value in (penalties.filler_penalty, penalties.rambling_penalty, penalties.monotony_penalty, penalties.mid_recording_collapse_penalty)):
        caps.append(ScoreCap(id="major_red_flag_cap", label="Major Red Flag Cap", value=91, reason="Major penalty clusters prevent 92+ scoring."))
    if score_confidence < 0.55:
        caps.append(ScoreCap(id="low_confidence_cap", label="Low Confidence Cap", value=76, reason="Low confidence prevents inflated precision."))
    return caps


def compute_authority_score(
    voice_metrics: dict,
    cognitive_metrics: dict,
    delivery_metrics: dict,
    linguistic: dict | None = None,
    audio_quality_penalty: float = 0.0,
    acoustic: AcousticAnalysisResult | None = None,
    *,
    audio_quality_usable: bool = True,
    asr_confidence: float | None = None,
    duration_ms: int | None = None,
) -> ScoringResult:
    """Compute deterministic Authority Score v2 with calibration and explainability."""
    (
        dimensions,
        dimension_details,
        delivery_score,
        content_score,
        adjustments,
        bonus_items,
        penalty_items,
        fairness,
    ) = compute_dimension_scores(voice_metrics, cognitive_metrics, delivery_metrics, linguistic, acoustic)

    weighted_base = round(sum(getattr(dimensions, key) * weight for key, weight in DIMENSION_WEIGHTS.items()), 2)
    confidence, confidence_label, confidence_reasons, low_confidence_penalty = _confidence(
        failure=bool(cognitive_metrics.get("failure", False)),
        audio_quality_usable=audio_quality_usable,
        asr_confidence=asr_confidence,
        duration_ms=duration_ms,
        acoustic=acoustic,
        audio_quality_penalty=audio_quality_penalty,
    )

    speech_seconds = acoustic.speaking_seconds if acoustic else _metric(voice_metrics, "duration_seconds")
    short_speech_penalty = 0.0
    if speech_seconds and speech_seconds < 8:
        short_speech_penalty = 8.0
    elif speech_seconds and speech_seconds < 15:
        short_speech_penalty = 4.0

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
        audio_quality_penalty=round(audio_quality_penalty, 2),
        short_speech_penalty=short_speech_penalty,
        low_confidence_penalty=low_confidence_penalty,
        mid_recording_collapse_penalty=adjustments.get("mid_recording_collapse_penalty", 0.0),
    )
    if short_speech_penalty:
        penalty_items.append(ScoreComponentItem(id="short_speech_penalty", label="Short Speech Penalty", value=short_speech_penalty, reason="Short usable speech cannot support a high-confidence score."))
    if low_confidence_penalty:
        penalty_items.append(ScoreComponentItem(id="low_confidence_penalty", label="Low Confidence Penalty", value=low_confidence_penalty, reason="Low scoring confidence reduces precision and score inflation."))
    if audio_quality_penalty:
        penalty_items.append(ScoreComponentItem(id="audio_quality_penalty", label="Audio Quality Penalty", value=round(audio_quality_penalty, 2), reason="Poor signal quality limits reliable scoring."))

    bonus_total = bonuses.opening_strength + bonuses.ending_strength + bonuses.consistency_bonus
    penalty_total = sum(
        (
            penalties.filler_penalty,
            penalties.rambling_penalty,
            penalties.monotony_penalty,
            penalties.rising_ending_penalty,
            penalties.audio_quality_penalty,
            penalties.short_speech_penalty,
            penalties.low_confidence_penalty,
            penalties.mid_recording_collapse_penalty,
        )
    )
    latent_score = round(weighted_base + bonus_total - penalty_total, 2)
    calibrated, notes = _calibrate(latent_score)
    caps = _caps(
        dimensions,
        penalties,
        failure=bool(cognitive_metrics.get("failure", False)),
        audio_quality_usable=audio_quality_usable,
        speech_seconds=speech_seconds,
        score_confidence=confidence,
    )
    capped_score = min(calibrated, *(cap.value for cap in caps)) if caps else calibrated
    final_score = int(round(_clamp(capped_score, 25, 97)))
    band, band_label, interpretation, rarity = _score_band(final_score)

    derived_axes = compute_derived_axes(dimensions, delivery_metrics)
    component_summary = [
        ScoreComponentItem(id="weighted_base", label="Weighted Base", value=weighted_base, reason="Six final dimensions combined with v2 Authority weights."),
        *bonus_items,
        *penalty_items,
        ScoreComponentItem(id="calibration_adjustment", label="Calibration Adjustment", value=round(capped_score - latent_score, 2), reason="Deterministic pre-human-corpus calibration and caps."),
    ]

    scores = Scores(
        authority_score=final_score,
        authority_percentile_estimate=_percentile(final_score),
        score_confidence=confidence,
        dimension_scores=dimensions,
        dimension_details=dimension_details,
        derived_axes=derived_axes,
        score_components=ScoreComponents(
            weighted_base=weighted_base,
            bonuses=bonuses,
            penalties=penalties,
            bonus_items=bonus_items,
            penalty_items=penalty_items,
            caps_applied=caps,
            calibration_adjustment=round(capped_score - latent_score, 2),
            final_score=final_score,
        ),
        calibration_metadata=CalibrationMetadata(
            latent_score=latent_score,
            calibrated_score=final_score,
            calibration_notes=notes,
        ),
        fairness_adjustments=fairness,
        score_explanation=ScoreExplanation(
            confidence_label=confidence_label,  # type: ignore[arg-type]
            confidence_reasons=confidence_reasons,
            component_summary=component_summary,
        ),
        score_band=band,
        score_band_label=band_label,
        score_interpretation=interpretation,
        score_rarity_label=rarity,
    )

    return ScoringResult(
        scores=scores,
        legacy_authority_score=round(float(final_score), 2),
        dimension_map=dimensions.model_dump(),
        delivery_score=delivery_score,
        content_score=content_score,
    )


def compute_authority_score_legacy(voice_metrics, cognitive_metrics, delivery_metrics):
    return compute_authority_score(voice_metrics, cognitive_metrics, delivery_metrics).legacy_authority_score
