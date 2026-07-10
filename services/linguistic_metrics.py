"""Deterministic transcript-based linguistic and structural metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass

from schemas import LinguisticMetrics, TranscriptWord
from services.lexicons import (
    APOLOGY_PHRASES,
    CAUSAL_MARKERS,
    CERTAINTY_PHRASES,
    HEDGE_PHRASES,
    PASSIVE_PATTERNS,
    SELF_DOUBT_PHRASES,
    SEQUENCE_MARKERS,
    THROAT_CLEARING,
    count_fillers_in_text,
    count_phrase_occurrences,
    is_filler_token,
)

OPENING_PORTION = 0.20
CLOSING_PORTION = 0.20
OPENING_MAX_SECONDS = 12.0
CLOSING_MAX_SECONDS = 12.0


@dataclass
class DeliveryMetrics:
    words_per_minute: float
    filler_density: float
    filler_count: int
    word_count: int


def _count_words(text: str) -> int:
    return len(text.split())


def _per_100_words(count: int, word_count: int) -> float | None:
    if word_count <= 0:
        return None
    return round((count / word_count) * 100, 2)


def compute_delivery_metrics(
    text: str,
    duration_seconds: float,
    words: list[TranscriptWord] | None = None,
    *,
    speaking_seconds: float | None = None,
) -> DeliveryMetrics:
    """Compute WPM and filler density from transcript and speaking time."""
    word_count = _count_words(text)
    active_seconds = max(speaking_seconds or duration_seconds, 1.0)
    wpm = (word_count / active_seconds) * 60

    if words:
        filler_count = sum(1 for word in words if word.is_filler)
        # Include multi-word fillers that token-level flags may miss.
        filler_count = max(filler_count, count_fillers_in_text(text.lower()))
    else:
        filler_count = count_fillers_in_text(text.lower())

    filler_density = filler_count / max(word_count, 1)

    return DeliveryMetrics(
        words_per_minute=round(wpm, 2),
        filler_density=round(filler_density, 4),
        filler_count=filler_count,
        word_count=word_count,
    )


def _span_by_time_or_portion(
    text: str,
    words: list[TranscriptWord],
    *,
    from_start: bool,
) -> str:
    if words:
        duration_ms = words[-1].end_ms
        span_ms = int(min(OPENING_MAX_SECONDS * 1000, duration_ms * OPENING_PORTION))
        if from_start:
            return " ".join(w.text for w in words if w.start_ms <= span_ms)
        cutoff = max(duration_ms - span_ms, 0)
        return " ".join(w.text for w in words if w.start_ms >= cutoff)

    tokens = text.split()
    span_count = max(int(len(tokens) * OPENING_PORTION), 1)
    if from_start:
        return " ".join(tokens[:span_count])
    return " ".join(tokens[-span_count:])


def _score_opening(opening: str) -> float:
    if not opening.strip():
        return 0.2
    lower = opening.lower()
    words = _count_words(opening)
    fillers = count_fillers_in_text(lower)
    hedges = count_phrase_occurrences(lower, HEDGE_PHRASES)
    throat = sum(lower.count(token) for token in THROAT_CLEARING)
    penalty = (fillers + hedges + throat) / max(words, 1)

    thesis_patterns = [
        r"\bthe point is\b",
        r"\bwhat matters is\b",
        r"\bi believe\b",
        r"\bmy view is\b",
        r"\bwe should\b",
        r"\bthe answer is\b",
        r"\bbecause\b",
    ]
    directness = 0.55
    if any(re.search(pattern, lower) for pattern in thesis_patterns):
        directness = 1.0
    elif re.search(r"\b(is|are|will|because|point)\b", lower):
        directness = 0.75

    return round(max(0.2, min(1.0, directness - penalty)), 2)


def _score_closing(closing: str) -> float:
    if not closing.strip():
        return 0.2
    lower = closing.lower()
    words = _count_words(closing)
    fillers = count_fillers_in_text(lower)
    hedges = count_phrase_occurrences(lower, HEDGE_PHRASES)
    penalty = (fillers + hedges) / max(words, 1)

    decisive = 0.55
    if closing.strip().endswith((".", "!", "?")):
        decisive = 0.85
    if any(marker in lower for marker in ["therefore", "in conclusion", "the point is", "finally"]):
        decisive = min(1.0, decisive + 0.15)
    if lower.endswith(("and", "but", "so", "um", "uh")):
        decisive -= 0.2

    return round(max(0.2, min(1.0, decisive - penalty)), 2)


def _repetition_rate(text: str) -> float:
    tokens = [re.sub(r"[^\w]", "", token.lower()) for token in text.split()]
    tokens = [token for token in tokens if token]
    if len(tokens) < 2:
        return 0.0
    unique = len(set(tokens))
    return round(1 - (unique / len(tokens)), 2)


def _passive_voice_ratio(text: str) -> float | None:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return None
    passive_count = 0
    for sentence in sentences:
        lower = sentence.lower()
        if any(re.search(pattern, lower) for pattern in PASSIVE_PATTERNS):
            passive_count += 1
    return round(passive_count / len(sentences), 2)


def _specificity_score(text: str) -> float:
    numbers = len(re.findall(r"\b\d+\b", text))
    named = len(re.findall(r"\b[A-Z][a-z]+\b", text))
    causal = count_phrase_occurrences(text.lower(), CAUSAL_MARKERS)
    sequence = sum(1 for marker in SEQUENCE_MARKERS if re.search(rf"\b{marker}\b", text.lower()))
    examples = len(re.findall(r"\b(for example|such as|like when)\b", text.lower()))
    words = max(_count_words(text), 1)
    score = (numbers * 2 + named + causal * 2 + sequence + examples * 2) / words
    return round(min(1.0, score), 2)


def _concreteness_score(text: str) -> float:
    concrete_nouns = len(
        re.findall(
            r"\b(project|team|client|product|company|result|number|percent|goal|plan)\b",
            text.lower(),
        )
    )
    numbers = len(re.findall(r"\b\d+\b", text))
    words = max(_count_words(text), 1)
    return round(min(1.0, (concrete_nouns + numbers * 2) / words), 2)


def _rambling_score(text: str) -> float:
    tokens = text.split()
    if len(tokens) < 20:
        return 0.0
    repetition = _repetition_rate(text)
    hedge_load = count_phrase_occurrences(text.lower(), HEDGE_PHRASES) / max(len(tokens), 1)
    connector_load = len(re.findall(r"\b(and|so|like|um|uh)\b", text.lower())) / max(len(tokens), 1)
    return round(min(1.0, repetition * 0.5 + hedge_load * 2 + connector_load), 2)


def _structure_score(
    opening_score: float,
    closing_score: float,
    specificity: float,
    repetition: float,
    rambling: float | None,
) -> float:
    rambling_penalty = rambling if rambling is not None else 0.0
    return round(
        max(
            0.0,
            min(
                1.0,
                opening_score * 0.3
                + closing_score * 0.25
                + specificity * 0.25
                + (1 - repetition) * 0.1
                - rambling_penalty * 0.1,
            ),
        ),
        2,
    )


def build_linguistic_metrics(
    text: str,
    delivery: DeliveryMetrics,
    duration_seconds: float,
    words: list[TranscriptWord] | None = None,
    *,
    asr_confidence: float | None = None,
    cognitive: dict | None = None,
    acoustic_hesitations: int | None = None,
    disfluency_confidence: float | None = None,
) -> LinguisticMetrics:
    """Build authority.v2 linguistic metrics from deterministic transcript rules."""
    text_lower = text.lower()
    word_count = delivery.word_count
    duration_minutes = max(duration_seconds / 60, 1 / 60)

    filler_per_min = delivery.filler_count / duration_minutes
    acoustic_count = max(acoustic_hesitations or 0, 0)
    confirmed_disfluencies = delivery.filler_count + acoustic_count
    hedges = count_phrase_occurrences(text_lower, HEDGE_PHRASES)
    certainty = count_phrase_occurrences(text_lower, CERTAINTY_PHRASES)

    opening = _span_by_time_or_portion(text, words or [], from_start=True)
    closing = _span_by_time_or_portion(text, words or [], from_start=False)
    opening_score = _score_opening(opening)
    closing_score = _score_closing(closing)

    repetition = _repetition_rate(text)
    specificity = _specificity_score(text)
    concreteness = _concreteness_score(text)
    rambling = _rambling_score(text)
    structure = _structure_score(opening_score, closing_score, specificity, repetition, rambling)

    # TODO(Milestone 3): optionally blend deterministic structure with cognitive scores
    if cognitive and asr_confidence is not None and asr_confidence < 0.5:
        pass  # keep deterministic only when ASR is weak

    return LinguisticMetrics(
        filler_words_per_min=round(filler_per_min, 1),
        lexical_fillers=delivery.filler_count,
        acoustic_hesitations=acoustic_count,
        confirmed_disfluencies=confirmed_disfluencies,
        disfluency_confidence=disfluency_confidence,
        hedges_per_100_words=_per_100_words(hedges, word_count),
        certainty_markers_per_100_words=_per_100_words(certainty, word_count),
        passive_voice_ratio=_passive_voice_ratio(text),
        apology_markers=count_phrase_occurrences(text_lower, APOLOGY_PHRASES),
        self_doubt_markers=count_phrase_occurrences(text_lower, SELF_DOUBT_PHRASES),
        repetition_rate=repetition,
        specificity_score=specificity,
        concreteness_score=concreteness,
        rambling_score=rambling,
        opening_strength_score=opening_score,
        closing_strength_score=closing_score,
        structure_score=structure,
    )
