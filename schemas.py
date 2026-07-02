"""Pydantic models for the authority.v2 analysis response contract."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
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


class ScoreComponents(BaseModel):
    weighted_base: float
    bonuses: ScoreBonuses
    penalties: ScorePenalties


class Scores(BaseModel):
    authority_score: int
    authority_percentile_estimate: float | None = None
    score_confidence: float | None = None
    dimension_scores: DimensionScores
    derived_axes: DerivedAxes
    score_components: ScoreComponents


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
    moments: list[Moment]
    recommendations: Recommendations
    drills: list[Drill]
    progress: Progress = Field(default_factory=Progress)
    paywall: Paywall = Field(default_factory=Paywall)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    safety: Safety = Field(default_factory=Safety)
