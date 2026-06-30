"""Shared deterministic lexicons for transcript analysis."""

from __future__ import annotations

import re

# Single-token fillers
FILLER_TOKENS = frozenset(
    {
        "um",
        "uh",
        "er",
        "erm",
        "like",
        "basically",
        "actually",
        "literally",
    }
)

# Multi-word filler phrases (longest first for greedy matching)
FILLER_PHRASES = [
    "you know",
    "sort of",
    "kind of",
]

HEDGE_PHRASES = [
    "i think",
    "maybe",
    "probably",
    "possibly",
    "kind of",
    "sort of",
    "i guess",
    "i suppose",
    "perhaps",
    "hopefully",
    "in a way",
]

CERTAINTY_PHRASES = [
    "the point is",
    "what matters is",
    "the reason is",
    "i would",
    "i recommend",
    "i believe",
    "my view is",
    "the answer is",
    "the key is",
    "therefore",
    "we will",
    "clearly",
    "definitely",
]

APOLOGY_PHRASES = ["sorry", "apologize", "apologies", "my bad"]

SELF_DOUBT_PHRASES = [
    "sorry",
    "i'm not sure",
    "i am not sure",
    "i don't know",
    "i do not know",
    "this might be wrong",
    "maybe i'm wrong",
    "maybe i am wrong",
    "i guess",
    "not confident",
    "i doubt",
]

PASSIVE_PATTERNS = [
    r"\b(am|is|are|was|were|be|been|being)\s+\w+ed\b",
    r"\bwas\s+\w+ed\b",
    r"\bwere\s+\w+ed\b",
    r"\bbeen\s+\w+ed\b",
]

THROAT_CLEARING = ["uh", "um", "er", "erm", "ah"]

SEQUENCE_MARKERS = ["first", "second", "third", "finally", "next", "then"]

CAUSAL_MARKERS = ["because", "therefore", "so that", "as a result", "since"]


def count_phrase_occurrences(text_lower: str, phrases: list[str]) -> int:
    return sum(text_lower.count(phrase) for phrase in phrases)


def is_filler_token(text: str) -> bool:
    normalized = re.sub(r"[^\w\s']", "", text.lower()).strip()
    if normalized in FILLER_TOKENS:
        return True
    return any(
        re.search(rf"\b{re.escape(phrase)}\b", normalized) for phrase in FILLER_PHRASES
    )


def count_fillers_in_text(text_lower: str) -> int:
    count = 0
    for token in text_lower.split():
        if is_filler_token(token):
            count += 1
    for phrase in FILLER_PHRASES:
        count += text_lower.count(phrase)
    return count
