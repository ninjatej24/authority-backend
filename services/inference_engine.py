"""GPT cognitive analysis and perception inference."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from schemas import (
    DerivedMetrics,
    EvidenceItem,
    PerceptionHighlight,
    PerceptionProfile,
    PerceptionReads,
    Uncertainty,
)

load_dotenv()


def _get_client() -> OpenAI:
    """Lazy-create OpenAI client only when needed."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _clamp_score(value, default=50) -> int:
    try:
        value = float(value)
        return int(max(0, min(100, round(value))))
    except Exception:
        return default


def _clean_json_text(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3:
            content = "\n".join(lines[1:-1]).strip()
    return content


def _fallback_cognitive() -> dict:
    return {
        "clarity": {
            "score": 50,
            "reason": "Fallback: transcript could not be reliably evaluated.",
        },
        "persuasion": {
            "score": 45,
            "reason": "Fallback: persuasive strength could not be reliably evaluated.",
        },
        "coherence": {
            "score": 50,
            "reason": "Fallback: coherence could not be reliably evaluated.",
        },
        "idea_strength": {
            "score": 45,
            "reason": "Fallback: idea strength could not be reliably evaluated.",
        },
        "conciseness": {
            "score": 50,
            "reason": "Fallback: conciseness could not be reliably evaluated.",
        },
        "failure": False,
    }


def _normalize_cognitive(parsed: dict | None) -> dict:
    fallback = _fallback_cognitive()
    if not isinstance(parsed, dict):
        return fallback

    normalized: dict = {}
    for key in ["clarity", "persuasion", "coherence", "idea_strength", "conciseness"]:
        entry = parsed.get(key, {})
        if not isinstance(entry, dict):
            entry = {}
        normalized[key] = {
            "score": _clamp_score(entry.get("score"), fallback[key]["score"]),
            "reason": str(entry.get("reason", fallback[key]["reason"])).strip()
            or fallback[key]["reason"],
        }
    normalized["failure"] = bool(parsed.get("failure", False))
    return normalized


def analyze_cognition(transcript: str, context: str = "initial", prompt_text: str = "") -> dict:
    """GPT-based transcript content scoring (preserved from v1)."""
    transcript = (transcript or "").strip()
    context = (context or "initial").strip().lower()
    prompt_text = (prompt_text or "").strip()

    if not transcript:
        return {
            "clarity": {"score": 20, "reason": "No transcript was available to evaluate."},
            "persuasion": {"score": 20, "reason": "No transcript was available to evaluate."},
            "coherence": {"score": 20, "reason": "No transcript was available to evaluate."},
            "idea_strength": {"score": 20, "reason": "No transcript was available to evaluate."},
            "conciseness": {"score": 20, "reason": "No transcript was available to evaluate."},
            "failure": True,
        }

    if context == "impromptu":
        context_instructions = f"""
This transcript comes from an IMPROMPTU speaking challenge.
Prompt: {prompt_text}
Judge as an under-pressure response, not a polished speech.
"""
    else:
        context_instructions = f"""
This transcript comes from a general speaking evaluation.
Prompt: {prompt_text}
"""

    prompt = f"""
You are an elite communication analyst. Evaluate ONLY transcript content.
Be strict. Do not inflate scores.

{context_instructions}

Return ONLY valid JSON:
{{
  "clarity": {{"score": number, "reason": "string"}},
  "persuasion": {{"score": number, "reason": "string"}},
  "coherence": {{"score": number, "reason": "string"}},
  "idea_strength": {{"score": number, "reason": "string"}},
  "conciseness": {{"score": number, "reason": "string"}},
  "failure": boolean
}}

Transcript:
{transcript}
"""

    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a strict evaluator. Return only valid JSON with no markdown.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content or ""
    try:
        parsed = json.loads(_clean_json_text(content))
    except Exception:
        return _fallback_cognitive()

    return _normalize_cognitive(parsed)


def build_derived_metrics(
    voice_metrics: dict,
    delivery: dict,
    linguistic_opening: float | None,
    linguistic_closing: float | None,
) -> DerivedMetrics:
    """Rule-based derived composites with TODOs for sliding-window analysis."""
    pitch_variation = voice_metrics.get("pitch_variation", 0)
    energy_variation = voice_metrics.get("energy_variation", 0)
    pause_frequency = voice_metrics.get("pause_frequency", 0)
    filler_density = delivery.get("filler_density", 0)

    monotony_index = None
    if pitch_variation and energy_variation:
        flat_pitch = max(0.0, 1.0 - min(pitch_variation / 40, 1.0))
        flat_energy = max(0.0, 1.0 - min(energy_variation / 15, 1.0))
        monotony_index = round((flat_pitch + flat_energy) / 2, 2)

    hesitation_cluster = round(
        min(1.0, pause_frequency * 1.5 + filler_density * 8),
        2,
    )

    dynamic_emphasis = None
    if pitch_variation and energy_variation:
        dynamic_emphasis = round(
            min(1.0, (pitch_variation / 60 + energy_variation / 20) / 2),
            2,
        )

    speech_continuity = round(
        max(0.0, min(1.0, voice_metrics.get("speech_density", 0.8))),
        2,
    )

    # TODO(v2.2): sliding 3s windows for confidence_drop_count
    confidence_drop_count = 1 if hesitation_cluster > 0.5 else 0

    return DerivedMetrics(
        monotony_index=monotony_index,
        hesitation_cluster_score=hesitation_cluster,
        dynamic_emphasis_score=dynamic_emphasis,
        speech_continuity_score=speech_continuity,
        confidence_drop_count=confidence_drop_count,
    )


def _headline_for_score(score: int) -> str:
    if score >= 90:
        return "You sound like someone people naturally defer to: clear, intentional, and fully self-possessed."
    if score >= 80:
        return "You sound composed, decisive, and easy to trust with the floor."
    if score >= 64:
        return "You sound capable and credible, though your delivery still gives away some hesitation."
    if score >= 48:
        return "You sound thoughtful and promising, but not yet in full control of the room."
    return "You currently sound uneasy and underpowered; listeners may focus on your uncertainty more than your point."


def build_perception_profile(
    authority_score: int,
    dimension_scores: dict[str, int],
    cognitive: dict,
    delivery: dict,
    voice_metrics: dict,
) -> PerceptionProfile:
    """Rule-based perception copy mapped from measured cues."""
    clarity = dimension_scores.get("clarity", 50)
    command = dimension_scores.get("command", 50)
    structure = dimension_scores.get("structure", 50)
    composure = dimension_scores.get("composure", 50)

    wpm = delivery.get("words_per_minute", 0)
    filler_density = delivery.get("filler_density", 0)

    if structure >= clarity and structure >= command:
        strength_title = "Clear structure"
        strength_expl = cognitive.get("coherence", {}).get(
            "reason", "You establish your point and keep the listener oriented."
        )
    elif clarity >= command:
        strength_title = "Verbal clarity"
        strength_expl = cognitive.get("clarity", {}).get(
            "reason", "Your meaning is generally easy to follow."
        )
    else:
        strength_title = "Delivery control"
        strength_expl = "Parts of your delivery show deliberate pacing and vocal control."

    if filler_density > 0.05 or wpm > 170:
        drag_title = "Rushed or hesitant delivery"
        drag_expl = "Pace and filler patterns may make some lines sound less final than they could."
    elif composure < 55:
        drag_title = "Composure under pressure"
        drag_expl = "Pause and stability patterns suggest mild tension in parts of the recording."
    else:
        drag_title = "Executive finality"
        drag_expl = "Some landings could sound more decisive to maximize perceived command."

    reads = PerceptionReads(
        emotional="mostly calm with mild tension"
        if composure >= 55
        else "noticeable tension in stretches of the recording",
        professional="competent but not maximally executive"
        if authority_score < 75
        else "polished and professionally assured",
        social_status="respectable, not yet fully high-status"
        if authority_score < 70
        else "likely read as high-status in group settings",
        interview="hireable, though some answers would benefit from cleaner finality"
        if authority_score < 72
        else "interview-ready with strong closure",
        leadership="trusted for contribution, not yet automatically deferred to"
        if command < 72
        else "likely deferred to in leadership settings",
    )

    return PerceptionProfile(
        headline=_headline_for_score(authority_score),
        how_you_currently_come_across=(
            "Listeners are likely to see you as capable, though parts of your delivery "
            "still ask for permission instead of taking the floor."
            if authority_score < 75
            else "Listeners are likely to read you as composed, credible, and in control of your message."
        ),
        biggest_strength=PerceptionHighlight(title=strength_title, explanation=strength_expl),
        biggest_drag=PerceptionHighlight(title=drag_title, explanation=drag_expl),
        listener_assumptions=[
            "You know what you want to say",
            "You are still building full command under pressure"
            if authority_score < 70
            else "You can hold attention when you choose to",
        ],
        reads=reads,
    )


def build_evidence(
    dimension_scores: dict[str, int],
    cognitive: dict,
    delivery: dict,
    voice_metrics: dict,
    linguistic: dict | None = None,
) -> list[EvidenceItem]:
    """Construct evidence items from top positive and negative cues."""
    evidence: list[EvidenceItem] = []
    index = 1

    wpm = delivery.get("words_per_minute", 0)
    filler_density = delivery.get("filler_density", 0)

    if dimension_scores.get("command", 0) >= 60:
        evidence.append(
            EvidenceItem(
                id=f"ev_{index}",
                trait="command",
                direction="positive",
                headline="Controlled pacing in parts of the recording",
                why_it_matters="Steady pace helps listeners read confidence and intent.",
                signals=[f"words_per_minute={wpm:.0f}", "moderate pause pattern"],
            )
        )
        index += 1
    elif wpm > 170 or filler_density > 0.04:
        evidence.append(
            EvidenceItem(
                id=f"ev_{index}",
                trait="command",
                direction="negative",
                headline="Delivery rushed or cluttered with fillers",
                why_it_matters="High pace plus fillers can read as approval-seeking or loss of control.",
                signals=["elevated filler_density", f"words_per_minute={wpm:.0f}"],
            )
        )
        index += 1

    clarity_reason = cognitive.get("clarity", {}).get("reason", "")
    if dimension_scores.get("clarity", 0) >= 65:
        evidence.append(
            EvidenceItem(
                id=f"ev_{index}",
                trait="clarity",
                direction="positive",
                headline="Meaning lands clearly",
                why_it_matters="Clarity supports credibility before charisma.",
                signals=["clear thesis", clarity_reason[:80]],
            )
        )
        index += 1

    if voice_metrics.get("energy_variation", 0) < 8:
        evidence.append(
            EvidenceItem(
                id=f"ev_{index}",
                trait="presence",
                direction="negative",
                headline="Limited vocal energy variation",
                why_it_matters="Flat energy can reduce persuasive pull and memorability.",
                signals=["low loudness_variation", "monotony risk"],
            )
        )
        index += 1

    opening = linguistic.get("opening_strength_score") if linguistic else None
    if opening and opening >= 0.7:
        evidence.append(
            EvidenceItem(
                id=f"ev_{index}",
                trait="structure",
                direction="positive",
                headline="Strong opening",
                why_it_matters="Early decisiveness anchors first impressions.",
                signals=["clear thesis", "low opening filler burden"],
            )
        )

    # TODO(v2.3): multivariate trait inference with confidence gating per trait

    return evidence


def build_uncertainty(
    audio_usable: bool,
    transcript_empty: bool,
    cognitive_failure: bool,
    *,
    duration_ms: int = 0,
    approximate_timestamps: bool = False,
    asr_confidence: float | None = None,
    audio_warnings: list[str] | None = None,
    missing_metrics: list[str] | None = None,
) -> Uncertainty:
    reasons: list[str] = list(audio_warnings or [])
    suppressed: list[str] = []
    label = "medium_high"

    if not audio_usable:
        reasons.append("Audio quality limited reliable acoustic inference")
        suppressed.extend(["hnr", "jitter_local", "shimmer_local", "terminal_rise_ratio"])
        label = "low"
    if transcript_empty:
        reasons.append("Transcript was empty or unusable")
        label = "low"
    if cognitive_failure:
        reasons.append("Transcript content could not be reliably evaluated")
        if label == "medium_high":
            label = "medium"
    if duration_ms and duration_ms < 8000:
        reasons.append("Short recording limits pause and moment analysis")
        if label == "medium_high":
            label = "medium"
    if approximate_timestamps:
        reasons.append("Word timestamps are approximate; timeline moments may be less precise")
        if label == "medium_high":
            label = "medium"
    if asr_confidence is not None and asr_confidence < 0.6:
        reasons.append("Low ASR confidence reduces reliability of linguistic metrics")
        suppressed.append("passive_voice_ratio")
        if label == "medium_high":
            label = "medium"
    if missing_metrics:
        for metric in missing_metrics:
            reasons.append(f"Metric unavailable: {metric}")

    if label == "low":
        suppressed = list(dict.fromkeys(suppressed))

    return Uncertainty(
        overall_confidence_label=label,  # type: ignore[arg-type]
        suppressed_traits=suppressed,
        reasons=reasons,
    )
