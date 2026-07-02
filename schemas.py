"""Pydantic models for the authority.v2 analysis response contract."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


# --- Request / metadata ---


class RequestMetadata(BaseModel):
    scenario: str = "benchmark"
    prompt_id: str = "authority_benchmark_v1"
    language: str = "en"
    duration_ms: int = 0
    device_context: str | None = None
    user_id: str | None = None


# --- Audio quality ---


class AudioQuality(BaseModel):
    usable: bool = True
    snr_estimate_db: float | None = None
    clipping_detected: bool = False
    background_noise_level: Literal["low", "medium", "high", "unknown"] = "unknown"
    single_speaker_likelihood: float | None = None
    quality_warnings: list[str] = Field(default_factory=list)


# --- Transcript ---


class TranscriptWord(BaseModel):
    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = None
    is_filler: bool = False


class TranscriptSegment(BaseModel):
    segment_id: str
    start_ms: int
    end_ms: int
    text: str
    role: Literal["opening", "body", "closing", "other"] = "other"


class Transcript(BaseModel):
    full_text: str = ""
    speaker_language_confidence: float | None = None
    asr_model: str = "whisper-1"
    overall_asr_confidence: float | None = None
    words: list[TranscriptWord] = Field(default_factory=list)
    segments: list[TranscriptSegment] = Field(default_factory=list)


# --- Scores ---


class DimensionScores(BaseModel):
    command: int
    clarity: int
    composure: int
    presence: int
    persuasion: int
    structure: int


class DimensionScoreDetail(BaseModel):
    score: int
    confidence: float = 0.0
    positive_contributors: list[str] = Field(default_factory=list)
    negative_contributors: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    uncertainty_reasons: list[str] = Field(default_factory=list)


class DerivedAxes(BaseModel):
    trust_warmth: int
    dominance_status: int
    nervousness: int
    interview_readiness: int
    leadership_readiness: int


class ScoreBonuses(BaseModel):
    opening_strength: float = 0.0
    ending_strength: float = 0.0
    consistency_bonus: float = 0.0


class ScorePenalties(BaseModel):
    filler_penalty: float = 0.0
    rambling_penalty: float = 0.0
    monotony_penalty: float = 0.0
    rising_ending_penalty: float = 0.0
    audio_quality_penalty: float = 0.0
    short_speech_penalty: float = 0.0
    low_confidence_penalty: float = 0.0
    mid_recording_collapse_penalty: float = 0.0


class ScoreComponentItem(BaseModel):
    id: str
    label: str
    value: float
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)


class ScoreCap(BaseModel):
    id: str
    label: str
    value: float
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)


class ScoreComponents(BaseModel):
    weighted_base: float
    bonuses: ScoreBonuses
    penalties: ScorePenalties
    bonus_items: list[ScoreComponentItem] = Field(default_factory=list)
    penalty_items: list[ScoreComponentItem] = Field(default_factory=list)
    caps_applied: list[ScoreCap] = Field(default_factory=list)
    calibration_adjustment: float = 0.0
    final_score: int | None = None


class CalibrationMetadata(BaseModel):
    calibration_version: str = "authority_score_v2.0"
    method: str = "deterministic_v2_pre_human_corpus"
    latent_score: float = 0.0
    calibrated_score: float = 0.0
    calibration_notes: list[str] = Field(default_factory=list)
    future_human_corpus_ready: bool = True


class FairnessAdjustments(BaseModel):
    applied_adjustments: list[str] = Field(default_factory=list)
    suppressed_features: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class ScoreExplanation(BaseModel):
    confidence_label: Literal["low", "medium", "medium_high", "high"] = "medium"
    confidence_reasons: list[str] = Field(default_factory=list)
    component_summary: list[ScoreComponentItem] = Field(default_factory=list)


class ScenarioAdjustment(BaseModel):
    scenario_used: str = "benchmark"
    scenario_weight_version: str = "scenario_weights_v1"
    scenario_adjustments: dict[str, float] = Field(default_factory=dict)
    dimension_adjustments: dict[str, float] = Field(default_factory=dict)
    major_weight_changes: list[str] = Field(default_factory=list)


class Scores(BaseModel):
    authority_score: int
    authority_percentile_estimate: float | None = None
    score_confidence: float | None = None
    dimension_scores: DimensionScores
    dimension_details: dict[str, DimensionScoreDetail] = Field(default_factory=dict)
    derived_axes: DerivedAxes
    score_components: ScoreComponents
    calibration_metadata: CalibrationMetadata = Field(default_factory=CalibrationMetadata)
    fairness_adjustments: FairnessAdjustments = Field(default_factory=FairnessAdjustments)
    score_explanation: ScoreExplanation = Field(default_factory=ScoreExplanation)
    score_band: str | None = None
    score_band_label: str | None = None
    score_interpretation: str | None = None
    score_rarity_label: str | None = None
    scenario_used: str = "benchmark"
    scenario_weight_version: str = "scenario_weights_v1"
    scenario_adjustments: ScenarioAdjustment = Field(default_factory=ScenarioAdjustment)


# --- Metrics ---


class RawAcousticMetrics(BaseModel):
    words_per_minute: float | None = None
    syllables_per_second: float | None = None
    pause_frequency_per_min: float | None = None
    avg_pause_ms: float | None = None
    longest_pause_ms: float | None = None
    mid_phrase_pause_rate: float | None = None
    f0_median_hz: float | None = None
    f0_range_semitones: float | None = None
    f0_variability_semitones: float | None = None
    terminal_rise_ratio: float | None = None
    loudness_mean_db_relative: float | None = None
    loudness_variation_db: float | None = None
    hnr: float | None = None
    jitter_local: float | None = None
    shimmer_local: float | None = None
    # Milestone 3 enhanced metrics
    pitch_mean_hz: float | None = None
    pitch_std_hz: float | None = None
    pitch_slope: float | None = None
    pitch_stability: float | None = None
    pitch_dynamics: float | None = None
    pitch_resets: int | None = None
    terminal_slope: float | None = None
    terminal_rising: float | None = None
    terminal_falling: float | None = None
    terminal_rising_ratio: float | None = None
    terminal_falling_ratio: float | None = None
    energy_mean: float | None = None
    energy_peak: float | None = None
    energy_std: float | None = None
    energy_slope: float | None = None
    dynamic_emphasis: float | None = None
    loudness_stability: float | None = None
    emphasis_bursts: int | None = None
    projection_segments: int | None = None
    energy_cv: float | None = None
    voicing_ratio: float | None = None
    voice_breaks: int | None = None
    breathiness_proxy: float | None = None
    strain_proxy: float | None = None
    cpp_proxy: float | None = None


class LinguisticMetrics(BaseModel):
    filler_words_per_min: float | None = None
    hedges_per_100_words: float | None = None
    certainty_markers_per_100_words: float | None = None
    passive_voice_ratio: float | None = None
    apology_markers: int | None = None
    self_doubt_markers: int | None = None
    repetition_rate: float | None = None
    specificity_score: float | None = None
    concreteness_score: float | None = None
    rambling_score: float | None = None
    opening_strength_score: float | None = None
    closing_strength_score: float | None = None
    structure_score: float | None = None


class RhythmMetrics(BaseModel):
    speech_rate: float | None = None
    words_per_minute: float | None = None
    pause_cadence: float | None = None
    speech_continuity: float | None = None
    hesitation_windows: int | None = None
    rhythm_consistency: float | None = None
    burst_speaking_segments: int | None = None
    slow_down_segments: int | None = None
    speed_up_segments: int | None = None
    articulation_rate: float | None = None


class ArticulationMetrics(BaseModel):
    articulation_rate: float | None = None
    phoneme_timing_consistency: float | None = None
    speech_precision: float | None = None
    word_duration_mean_ms: float | None = None
    word_duration_std_ms: float | None = None
    word_duration_cv: float | None = None
    clarity_proxy: float | None = None
    articulation_stability: float | None = None


class VADMetrics(BaseModel):
    speech_ratio: float | None = None
    total_speech_duration_ms: int | None = None
    total_silence_duration_ms: int | None = None
    pause_durations_ms: list[float] = Field(default_factory=list)
    long_pauses_ms: list[float] = Field(default_factory=list)
    mid_sentence_pauses_ms: list[float] = Field(default_factory=list)
    end_of_sentence_pauses_ms: list[float] = Field(default_factory=list)
    avg_pause_duration_ms: float | None = None
    pause_frequency_per_minute: float | None = None
    vad_backend: str | None = None


class DerivedMetrics(BaseModel):
    monotony_index: float | None = None
    hesitation_cluster_score: float | None = None
    dynamic_emphasis_score: float | None = None
    speech_continuity_score: float | None = None
    confidence_drop_count: int | None = None
    # Milestone 3 derived indices
    vocal_command_index: float | None = None
    composure_index: float | None = None
    rhythm_index: float | None = None
    projection_index: float | None = None
    authority_signal_index: float | None = None


class Metrics(BaseModel):
    raw_acoustic: RawAcousticMetrics
    linguistic: LinguisticMetrics
    derived: DerivedMetrics
    # Milestone 3 additional metric categories
    rhythm: RhythmMetrics = Field(default_factory=RhythmMetrics)
    articulation: ArticulationMetrics = Field(default_factory=ArticulationMetrics)
    vad: VADMetrics = Field(default_factory=VADMetrics)


# --- Perception profile ---


class PerceptionHighlight(BaseModel):
    title: str
    explanation: str


class PerceptionReads(BaseModel):
    emotional: str
    professional: str
    social_status: str
    interview: str
    leadership: str


class PerceptionProfile(BaseModel):
    headline: str
    how_you_currently_come_across: str
    biggest_strength: PerceptionHighlight
    biggest_drag: PerceptionHighlight
    listener_assumptions: list[str]
    reads: PerceptionReads


# --- Evidence & moments ---


class EvidenceItem(BaseModel):
    id: str
    trait: str
    direction: Literal["positive", "negative"]
    headline: str
    why_it_matters: str
    signals: list[str]


class MetricEvidenceItem(BaseModel):
    metric_name: str
    value: float | int | str | bool | None = None
    confidence: float
    source: str
    calculation_method: str
    window_used: list[int] | None = None
    raw_inputs: dict = Field(default_factory=dict)
    timestamp: str | None = None
    notes: str = ""


class MetricEvidenceBundle(BaseModel):
    audio_quality: list[MetricEvidenceItem] = Field(default_factory=list)
    pitch_contour: list[MetricEvidenceItem] = Field(default_factory=list)
    energy_contour: list[MetricEvidenceItem] = Field(default_factory=list)
    voice_quality: list[MetricEvidenceItem] = Field(default_factory=list)
    rhythm: list[MetricEvidenceItem] = Field(default_factory=list)
    articulation: list[MetricEvidenceItem] = Field(default_factory=list)
    vad: list[MetricEvidenceItem] = Field(default_factory=list)
    derived_indices: list[MetricEvidenceItem] = Field(default_factory=list)
    window_features: list[MetricEvidenceItem] = Field(default_factory=list)


# --- Psychological inference ---


class PsychologicalEvidenceSignal(BaseModel):
    evidence_id: str
    metric: str
    observed_value: float | int | str | bool | None = None
    expected_range: str
    direction: Literal["supporting", "contradicting", "uncertainty"]
    weight: float
    why_it_matters_psychologically: str


class MicroBehaviour(BaseModel):
    id: str
    label: str
    confidence: float
    supporting_metrics: list[str] = Field(default_factory=list)
    contradicting_metrics: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    uncertainty_reason: str | None = None


class PsychologicalTrait(BaseModel):
    trait_id: str
    label: str
    score: int
    confidence: float
    confidence_label: Literal["low", "medium", "medium_high", "high"]
    supporting_behaviours: list[str] = Field(default_factory=list)
    contradicting_behaviours: list[str] = Field(default_factory=list)
    supporting_metrics: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    evidence_chain: list[PsychologicalEvidenceSignal] = Field(default_factory=list)
    uncertainty_reason: str | None = None
    suppress_from_report: bool = False
    scenario_weight_adjustments: dict[str, float] = Field(default_factory=dict)


class InferenceCandidate(BaseModel):
    candidate_id: str
    label: str
    trait_ids: list[str] = Field(default_factory=list)
    behaviour_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float


class PsychologicalPrimaryCandidates(BaseModel):
    primary_strength_candidate: InferenceCandidate | None = None
    primary_limiter_candidate: InferenceCandidate | None = None
    core_tension_candidate: InferenceCandidate | None = None


class PsychologicalReportCandidates(BaseModel):
    authority_type_candidates: list[InferenceCandidate] = Field(default_factory=list)
    strongest_positive_impression: InferenceCandidate | None = None
    strongest_negative_impression: InferenceCandidate | None = None
    highest_confidence_trait: str | None = None
    lowest_confidence_trait: str | None = None
    report_priority_order: list[str] = Field(default_factory=list)
    hidden_cost_candidates: list[InferenceCandidate] = Field(default_factory=list)
    highest_leverage_candidates: list[InferenceCandidate] = Field(default_factory=list)


class PsychologicalInference(BaseModel):
    micro_behaviours: list[MicroBehaviour] = Field(default_factory=list)
    traits: list[PsychologicalTrait] = Field(default_factory=list)
    evidence_chain: list[PsychologicalEvidenceSignal] = Field(default_factory=list)
    primary_candidates: PsychologicalPrimaryCandidates = Field(
        default_factory=PsychologicalPrimaryCandidates
    )
    report_candidates: PsychologicalReportCandidates = Field(
        default_factory=PsychologicalReportCandidates
    )
    overall_inference_confidence: float = 0.0
    suppressed_traits: list[str] = Field(default_factory=list)
    uncertainty: Uncertainty = Field(default_factory=lambda: Uncertainty())


class Moment(BaseModel):
    moment_id: str
    type: str
    start_ms: int
    end_ms: int
    severity: Literal["highlight", "low", "medium", "high"]
    headline: str
    summary: str
    dimension_impact: dict[str, float] = Field(default_factory=dict)
    preview_visible_free: bool = False


# --- Coaching ---


class Recommendations(BaseModel):
    highest_leverage_issue: str
    fastest_improvement_tip: str
    coaching_summary: str


class Drill(BaseModel):
    drill_id: str
    title: str
    goal: str
    instructions: list[str]
    duration_min: int
    difficulty: Literal["beginner", "intermediate", "advanced"]
    target_metrics: list[str]


# --- Progress, paywall, uncertainty, safety ---


class Progress(BaseModel):
    comparison_available: bool = False
    baseline_analysis_id: str | None = None
    delta_authority_score: float | None = None
    dimension_deltas: dict[str, float] | None = None


class FreePreview(BaseModel):
    show_score: bool = True
    show_headline: bool = True
    show_strength: bool = True
    show_drag: bool = True
    show_fast_tip: bool = True
    show_single_visible_moment: bool = True


class Paywall(BaseModel):
    free_preview: FreePreview = Field(default_factory=FreePreview)
    locked_modules: list[str] = Field(
        default_factory=lambda: [
            "full_transcript",
            "full_dimension_breakdown",
            "timeline_analysis",
            "full_perception_profile",
            "personalised_training_plan",
            "weekly_progress",
        ]
    )


class Uncertainty(BaseModel):
    overall_confidence_label: Literal[
        "low", "medium", "medium_high", "high"
    ] = "medium"
    suppressed_traits: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class Safety(BaseModel):
    responsible_framing: str = (
        "These results describe likely listener impressions from this recording, "
        "not fixed personality traits."
    )
    limitations: list[str] = Field(
        default_factory=lambda: [
            "Short single-sample recording",
            "Accent and microphone quality can affect some measures",
            "Some inferences are probabilistic rather than certain",
        ]
    )


# --- Report ---


class ReportMirror(BaseModel):
    headline: str | None = None
    identity_read: str | None = None
    one_line_identity_read: str | None = None
    core_tension: str | None = None
    emotional_tone: str | None = None
    authority_type: str | None = None
    confidence_label: Literal["low", "medium", "medium_high", "high"] = "low"
    confidence_level: Literal["low", "medium", "medium_high", "high"] = "low"
    evidence_ids: list[str] = Field(default_factory=list)


class ReportDiagnosis(BaseModel):
    strongest_dimension: str | None = None
    limiting_dimension: str | None = None
    core_behavioural_pattern: str | None = None
    social_consequence: str | None = None
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] | None = None
    primary_strength_dimension: str | None = None
    primary_limiting_dimension: str | None = None
    core_pattern: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ReportPerceptionRead(BaseModel):
    label: str | None = None
    text: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ReportPerceptionMap(BaseModel):
    first_impression: ReportPerceptionRead | None = None
    professional_read: ReportPerceptionRead | None = None
    leadership_read: ReportPerceptionRead | None = None
    interview_read: ReportPerceptionRead | None = None
    social_status_read: ReportPerceptionRead | None = None
    emotional_read: ReportPerceptionRead | None = None
    trust_read: ReportPerceptionRead | None = None
    persuasion_read: ReportPerceptionRead | None = None


class ReportScenarioSummary(BaseModel):
    scenario_id: str = "benchmark"
    description: str | None = None
    why_dimensions_changed: list[str] = Field(default_factory=list)
    scenario_expectations: list[str] = Field(default_factory=list)
    adjusted_strengths: list[str] = Field(default_factory=list)
    adjusted_weaknesses: list[str] = Field(default_factory=list)
    highest_leverage_fix: str | None = None
    coaching_explanation: str | None = None
    perception_emphasis: list[str] = Field(default_factory=list)


class ReportHiddenCost(BaseModel):
    dimension: str | None = None
    cost_id: str | None = None
    consequence: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ReportHighestLeverageFix(BaseModel):
    issue: str | None = None
    plain_english: str | None = None
    why_this_matters: str | None = None
    expected_score_lift: Literal["low", "medium", "high"] | None = None
    target_dimensions: list[str] = Field(default_factory=list)
    first_drill_id: str | None = None
    selection_score: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)


class ReportTrainingPrescription(BaseModel):
    drill_id: str | None = None
    title: str | None = None
    why_chosen: str | None = None
    instructions: list[str] = Field(default_factory=list)
    target_metrics: list[str] = Field(default_factory=list)
    success_signal: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ReportRetestPlan(BaseModel):
    recommended_retest_after_days: int | None = None
    focus_metric: str | None = None
    compare_metrics: list[str] = Field(default_factory=list)
    same_prompt_recommended: bool = True
    success_definition: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ReportAuthorityType(BaseModel):
    type_id: str | None = None
    label: str | None = None
    description: str | None = None
    top_dimensions: list[str] = Field(default_factory=list)
    growth_dimensions: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ReportShareCard(BaseModel):
    authority_score: int | None = None
    authority_type: str | None = None
    top_strength: str | None = None
    growth_area: str | None = None
    one_line_identity_read: str | None = None
    percentile_label: str | None = None
    share_safety: str = "public_safe"
    hidden_private_findings: list[str] = Field(default_factory=list)


class ReportTechnicalAppendix(BaseModel):
    metrics: dict[str, Any] = Field(default_factory=dict)
    audio_quality_warnings: list[str] = Field(default_factory=list)
    score_components: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)


class ReportEvidenceCard(BaseModel):
    evidence_id: str
    signal: str
    what_happened: str
    why_it_matters: str
    listener_interpretation: str
    related_dimension: str
    confidence: float
    timestamp: list[int] | None = None


class ReportTimelineItem(BaseModel):
    moment_id: str
    type: str
    headline: str
    summary: str
    listener_interpretation: str
    dimension_impact: dict[str, float] = Field(default_factory=dict)
    confidence: float
    start_ms: int
    end_ms: int
    evidence_ids: list[str] = Field(default_factory=list)
    severity: Literal["highlight", "low", "medium", "high"]
    preview_visible_free: bool = False


class ReportDimensionReport(BaseModel):
    dimension: str
    score: int
    label: str
    meaning: str
    why: list[str] = Field(default_factory=list)
    listener_consequence: str
    one_improvement_cue: str
    linked_evidence: list[str] = Field(default_factory=list)
    confidence: float


class ReportValidation(BaseModel):
    valid: bool = True
    evidence_ids_checked: list[str] = Field(default_factory=list)
    moment_ids_checked: list[str] = Field(default_factory=list)
    drill_ids_checked: list[str] = Field(default_factory=list)
    orphan_links: list[str] = Field(default_factory=list)
    duplicate_sections: list[str] = Field(default_factory=list)


class AuthorityReport(BaseModel):
    mirror: ReportMirror | None = None
    diagnosis: ReportDiagnosis | None = None
    perception_map: ReportPerceptionMap | None = None
    evidence_chain: list[ReportEvidenceCard] = Field(default_factory=list)
    timeline: list[ReportTimelineItem] = Field(default_factory=list)
    dimension_reports: dict[str, ReportDimensionReport] = Field(default_factory=dict)
    hidden_cost: ReportHiddenCost | None = None
    highest_leverage_fix: ReportHighestLeverageFix | None = None
    training_prescription: ReportTrainingPrescription | None = None
    retest_plan: ReportRetestPlan | None = None
    authority_type: ReportAuthorityType | None = None
    share_card: ReportShareCard | None = None
    technical_appendix: ReportTechnicalAppendix | None = None
    scenario_summary: ReportScenarioSummary | None = None
    diagnostic_reasoning: DiagnosticReasoning | None = None
    primary_diagnosis: DiagnosticDiagnosis | None = None
    secondary_diagnosis: DiagnosticDiagnosis | None = None
    contradictions: list[DiagnosticContradiction] = Field(default_factory=list)
    hidden_cost_reasoning: HiddenCostReasoning | None = None
    dimension_reasoning: dict[str, DimensionReasoning] = Field(default_factory=dict)
    trait_reasoning: dict[str, TraitReasoning] = Field(default_factory=dict)
    highest_leverage_reasoning: HighestLeverageReasoning | None = None
    coaching_engine: CoachingEngine | None = None
    validation: ReportValidation = Field(default_factory=ReportValidation)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)


# --- Diagnostic reasoning ---


class DiagnosticDiagnosis(BaseModel):
    diagnosis_id: str
    diagnosis_name: str
    confidence: float
    severity: Literal["low", "medium", "high"]
    supporting_traits: list[str] = Field(default_factory=list)
    contradicting_traits: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    supporting_moment_ids: list[str] = Field(default_factory=list)
    affected_dimensions: list[str] = Field(default_factory=list)


class DiagnosticContradiction(BaseModel):
    contradiction_id: str
    strength: str
    limiter: str
    why_it_happens: list[str] = Field(default_factory=list)
    listener_effect: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float


class HiddenCostReasoning(BaseModel):
    cost_id: str | None = None
    source_signal: str | None = None
    interpretation: str | None = None
    consequence: str | None = None
    listener_effect: str | None = None
    affected_dimensions: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    moment_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class HighestLeverageReasoning(BaseModel):
    issue_id: str | None = None
    plain_reason: str | None = None
    affected_dimensions: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    expected_score_lift: Literal["low", "medium", "high"] | None = None
    recommended_first_drill: str | None = None
    confidence: float = 0.0
    severity: float = 0.0
    authority_impact: float = 0.0
    trainability: float = 0.0
    evidence_confidence: float = 0.0
    scenario_relevance: float = 1.0
    selection_score: float = 0.0


class DimensionReasoning(BaseModel):
    dimension: str
    score: int
    why_score_is_high: list[str] = Field(default_factory=list)
    why_score_is_low: list[str] = Field(default_factory=list)
    largest_positive_signal: str | None = None
    largest_negative_signal: str | None = None
    biggest_metric_contributor: str | None = None
    biggest_linguistic_contributor: str | None = None
    biggest_behavioural_contributor: str | None = None
    confidence: float = 0.0
    supporting_evidence_ids: list[str] = Field(default_factory=list)


class TraitReasoning(BaseModel):
    trait_id: str
    label: str
    positive_evidence: list[str] = Field(default_factory=list)
    negative_evidence: list[str] = Field(default_factory=list)
    confidence: float
    suppression_reason: str | None = None
    supporting_metrics: list[str] = Field(default_factory=list)
    supporting_moments: list[str] = Field(default_factory=list)


class DiagnosticReasoning(BaseModel):
    primary_diagnosis: DiagnosticDiagnosis | None = None
    secondary_diagnosis: DiagnosticDiagnosis | None = None
    suppressed_diagnoses: list[DiagnosticDiagnosis] = Field(default_factory=list)
    contradictions: list[DiagnosticContradiction] = Field(default_factory=list)
    hidden_cost_reasoning: HiddenCostReasoning | None = None
    dimension_reasoning: dict[str, DimensionReasoning] = Field(default_factory=dict)
    trait_reasoning: dict[str, TraitReasoning] = Field(default_factory=dict)
    highest_leverage_reasoning: HighestLeverageReasoning | None = None
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)


# --- Deterministic coaching engine ---


class CoachingDrillDefinition(BaseModel):
    drill_id: str
    title: str
    category: str
    description: str
    target_behaviours: list[str] = Field(default_factory=list)
    target_metrics: list[str] = Field(default_factory=list)
    target_dimensions: list[str] = Field(default_factory=list)
    expected_authority_impact: float
    expected_difficulty: Literal["beginner", "intermediate", "advanced"]
    estimated_duration_min: int
    trainability_score: float
    prerequisites: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)


class CoachingRootCause(BaseModel):
    root_cause_id: str
    label: str
    contributing_signals: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float
    affected_dimensions: list[str] = Field(default_factory=list)


class ExpectedImprovement(BaseModel):
    drill_id: str
    authority_score: float = 0.0
    command: float = 0.0
    clarity: float = 0.0
    composure: float = 0.0
    presence: float = 0.0
    persuasion: float = 0.0
    structure: float = 0.0
    confidence: float = 0.0


class InterventionCandidate(BaseModel):
    drill_id: str
    title: str
    score: float
    severity: float
    authority_impact: float
    trainability: float
    confidence: float
    scenario_relevance: float
    required_evidence: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    expected_impact: ExpectedImprovement
    why_selected: str | None = None
    why_not_selected: str | None = None


class SelectedInterventions(BaseModel):
    primary_drill: InterventionCandidate | None = None
    secondary_drill: InterventionCandidate | None = None


class CoachingReasoningChain(BaseModel):
    detected: list[str] = Field(default_factory=list)
    contributing_factors: list[str] = Field(default_factory=list)
    root_issue: str | None = None
    highest_leverage_intervention: str | None = None
    reason: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class DrillDependency(BaseModel):
    before: str
    after: str
    reason: str


class CoachingEngine(BaseModel):
    drill_library: list[CoachingDrillDefinition] = Field(default_factory=list)
    drill_library_size: int = 0
    root_causes: list[CoachingRootCause] = Field(default_factory=list)
    intervention_candidates: list[InterventionCandidate] = Field(default_factory=list)
    selected_interventions: SelectedInterventions = Field(default_factory=SelectedInterventions)
    suppressed_interventions: list[InterventionCandidate] = Field(default_factory=list)
    reasoning_chain: CoachingReasoningChain = Field(default_factory=CoachingReasoningChain)
    expected_improvements: dict[str, ExpectedImprovement] = Field(default_factory=dict)
    dependency_graph: list[DrillDependency] = Field(default_factory=list)
    future_training_queue: list[InterventionCandidate] = Field(default_factory=list)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)




# --- Top-level response ---


class AuthorityV2Response(BaseModel):
    schema_version: Literal["authority.v2"] = "authority.v2"
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    request: RequestMetadata
    audio_quality: AudioQuality
    transcript: Transcript
    scores: Scores
    metrics: Metrics
    perception_profile: PerceptionProfile
    evidence: list[EvidenceItem]
    metric_evidence: MetricEvidenceBundle = Field(default_factory=MetricEvidenceBundle)
    moments: list[Moment]
    recommendations: Recommendations
    drills: list[Drill]
    psychological_inference: PsychologicalInference = Field(
        default_factory=PsychologicalInference
    )
    report: AuthorityReport = Field(default_factory=AuthorityReport)
    coaching_engine: CoachingEngine = Field(default_factory=CoachingEngine)
    progress: Progress = Field(default_factory=Progress)
    paywall: Paywall = Field(default_factory=Paywall)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    safety: Safety = Field(default_factory=Safety)
