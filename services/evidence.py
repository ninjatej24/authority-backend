"""Evidence collection for metric provenance and explainability."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


class MetricSource(Enum):
    """Source of metric calculation."""
    PRAAT = "praat"
    WEBRTC_VAD = "webrtc_vad"
    OPENAI_WHISPER = "openai_whisper"
    RULE_BASED = "rule_based"
    STATISTICAL = "statistical"
    DERIVED = "derived"
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Confidence level for metric reliability."""
    HIGH = "high"
    MEDIUM = "high"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class MetricEvidence:
    """Evidence for a single metric calculation."""
    metric_name: str
    value: float | int | str | bool | None
    confidence: float  # 0.0 to 1.0
    source: MetricSource
    calculation_method: str
    window_used: tuple[int, int] | None = None  # (start_ms, end_ms) if windowed
    raw_inputs: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    notes: str = ""


@dataclass
class EvidenceCollection:
    """Collection of all metric evidence for an analysis."""
    audio_quality: list[MetricEvidence] = field(default_factory=list)
    pitch_contour: list[MetricEvidence] = field(default_factory=list)
    energy_contour: list[MetricEvidence] = field(default_factory=list)
    voice_quality: list[MetricEvidence] = field(default_factory=list)
    rhythm: list[MetricEvidence] = field(default_factory=list)
    articulation: list[MetricEvidence] = field(default_factory=list)
    vad: list[MetricEvidence] = field(default_factory=list)
    derived_indices: list[MetricEvidence] = field(default_factory=list)
    window_features: list[MetricEvidence] = field(default_factory=list)
    
    def all_evidence(self) -> list[MetricEvidence]:
        """Get all evidence across all categories."""
        return (
            self.audio_quality +
            self.pitch_contour +
            self.energy_contour +
            self.voice_quality +
            self.rhythm +
            self.articulation +
            self.vad +
            self.derived_indices +
            self.window_features
        )
    
    def get_evidence_for_metric(self, metric_name: str) -> list[MetricEvidence]:
        """Get all evidence for a specific metric name."""
        return [e for e in self.all_evidence() if e.metric_name == metric_name]


def create_evidence(
    metric_name: str,
    value: float | int | str | bool | None,
    confidence: float,
    source: MetricSource,
    calculation_method: str,
    window_used: tuple[int, int] | None = None,
    raw_inputs: dict[str, Any] | None = None,
    notes: str = "",
) -> MetricEvidence:
    """Create a MetricEvidence object with validation."""
    confidence = max(0.0, min(1.0, confidence))
    
    return MetricEvidence(
        metric_name=metric_name,
        value=value,
        confidence=confidence,
        source=source,
        calculation_method=calculation_method,
        window_used=window_used,
        raw_inputs=raw_inputs or {},
        notes=notes,
    )


def add_audio_quality_evidence(
    collection: EvidenceCollection,
    snr_db: float | None,
    clipping_detected: bool,
    background_noise_level: str,
    single_speaker_likelihood: float | None,
    usable: bool,
) -> None:
    """Add evidence for audio quality metrics."""
    if snr_db is not None:
        collection.audio_quality.append(create_evidence(
            metric_name="snr_estimate_db",
            value=snr_db,
            confidence=0.8,
            source=MetricSource.STATISTICAL,
            calculation_method="Energy-based SNR estimation from frame analysis",
            raw_inputs={"method": "frame_energy_ratio"},
        ))
    
    collection.audio_quality.append(create_evidence(
        metric_name="clipping_detected",
        value=clipping_detected,
        confidence=0.95,
        source=MetricSource.STATISTICAL,
        calculation_method="Peak amplitude threshold detection",
        raw_inputs={"threshold": 0.99},
    ))
    
    collection.audio_quality.append(create_evidence(
        metric_name="background_noise_level",
        value=background_noise_level,
        confidence=0.75 if snr_db is not None else 0.5,
        source=MetricSource.RULE_BASED,
        calculation_method="SNR-based noise classification",
        raw_inputs={"snr_db": snr_db},
    ))
    
    if single_speaker_likelihood is not None:
        collection.audio_quality.append(create_evidence(
            metric_name="single_speaker_likelihood",
            value=single_speaker_likelihood,
            confidence=0.6,
            source=MetricSource.STATISTICAL,
            calculation_method="Energy stability coefficient of variation",
            notes="Placeholder until pyannote integration",
        ))
    
    collection.audio_quality.append(create_evidence(
        metric_name="usable",
        value=usable,
        confidence=0.9,
        source=MetricSource.RULE_BASED,
        calculation_method="Multi-factor quality gate (duration, SNR, clipping, signal level)",
    ))


def add_pitch_contour_evidence(
    collection: EvidenceCollection,
    pitch_contour: dict[str, float],
    confidence: float = 0.85,
) -> None:
    """Add evidence for pitch contour metrics."""
    for metric_name, value in pitch_contour.items():
        collection.pitch_contour.append(create_evidence(
            metric_name=metric_name,
            value=value,
            confidence=confidence,
            source=MetricSource.PRAAT,
            calculation_method="Praat pitch extraction with semitone normalization",
            raw_inputs={"reference": "speaker_median"},
        ))


def add_energy_contour_evidence(
    collection: EvidenceCollection,
    energy_contour: dict[str, float],
    confidence: float = 0.85,
) -> None:
    """Add evidence for energy contour metrics."""
    for metric_name, value in energy_contour.items():
        collection.energy_contour.append(create_evidence(
            metric_name=metric_name,
            value=value,
            confidence=confidence,
            source=MetricSource.PRAAT,
            calculation_method="Praat intensity analysis with statistical derivatives",
            raw_inputs={"method": "intensity_values"},
        ))


def add_voice_quality_evidence(
    collection: EvidenceCollection,
    voice_quality: dict[str, float],
    confidence: float = 0.7,
) -> None:
    """Add evidence for voice quality metrics."""
    for metric_name, value in voice_quality.items():
        source = MetricSource.PRAAT if metric_name in ["hnr", "jitter", "shimmer"] else MetricSource.DERIVED
        method = "Praat voice quality extraction" if source == MetricSource.PRAAT else "Derived from spectral and pitch analysis"
        
        collection.voice_quality.append(create_evidence(
            metric_name=metric_name,
            value=value,
            confidence=confidence,
            source=source,
            calculation_method=method,
        ))


def add_rhythm_evidence(
    collection: EvidenceCollection,
    rhythm_analysis: dict,
    confidence: float = 0.8,
) -> None:
    """Add evidence for rhythm analysis metrics."""
    for metric_name, value in rhythm_analysis.items():
        collection.rhythm.append(create_evidence(
            metric_name=metric_name,
            value=value,
            confidence=confidence,
            source=MetricSource.RULE_BASED,
            calculation_method="Word timestamp analysis with rate change detection",
            raw_inputs={"method": "inter_word_intervals"},
        ))


def add_articulation_evidence(
    collection: EvidenceCollection,
    articulation_analysis: dict,
    confidence: float = 0.75,
) -> None:
    """Add evidence for articulation metrics."""
    for metric_name, value in articulation_analysis.items():
        collection.articulation.append(create_evidence(
            metric_name=metric_name,
            value=value,
            confidence=confidence,
            source=MetricSource.STATISTICAL,
            calculation_method="Word duration variability analysis as phoneme timing proxy",
            raw_inputs={"method": "word_duration_cv"},
            notes="Phoneme-level approximation via word timing consistency",
        ))


def add_vad_evidence(
    collection: EvidenceCollection,
    vad_result: dict,
    confidence: float = 0.85,
) -> None:
    """Add evidence for VAD metrics."""
    for metric_name, value in vad_result.items():
        if isinstance(value, (int, float)):
            collection.vad.append(create_evidence(
                metric_name=metric_name,
                value=value,
                confidence=confidence,
                source=MetricSource.WEBRTC_VAD,
                calculation_method="WebRTC VAD with frame-based speech detection",
                raw_inputs={"aggressiveness": 2, "frame_duration_ms": 30},
            ))


def add_derived_indices_evidence(
    collection: EvidenceCollection,
    derived_indices: dict,
    confidence: float = 0.7,
) -> None:
    """Add evidence for derived indices."""
    for metric_name, value in derived_indices.items():
        if metric_name == "confidence":
            continue  # Skip confidence itself
        
        collection.derived_indices.append(create_evidence(
            metric_name=metric_name,
            value=value,
            confidence=confidence,
            source=MetricSource.DERIVED,
            calculation_method="Weighted composite of acoustic, rhythm, and articulation metrics",
            raw_inputs={"type": "composite_index"},
            notes="Engineering feature for psychological inference and calibration",
        ))


def add_window_feature_evidence(
    collection: EvidenceCollection,
    window: dict,
    window_index: int,
    confidence: float = 0.8,
) -> None:
    """Add evidence for a single window feature."""
    window_start = window.get("start_ms", 0)
    window_end = window.get("end_ms", 0)
    
    for metric_name, value in window.items():
        if metric_name in ["start_ms", "end_ms"]:
            continue
        
        collection.window_features.append(create_evidence(
            metric_name=f"window_{window_index}_{metric_name}",
            value=value,
            confidence=confidence,
            source=MetricSource.DERIVED,
            calculation_method="Sliding window analysis (3s window, 1s hop)",
            window_used=(window_start, window_end),
            raw_inputs={"window_index": window_index},
        ))


def calculate_overall_confidence(collection: EvidenceCollection) -> float:
    """Calculate overall confidence from all evidence."""
    all_evidence = collection.all_evidence()
    if not all_evidence:
        return 0.5
    
    confidences = [e.confidence for e in all_evidence]
    return round(sum(confidences) / len(confidences), 2)
