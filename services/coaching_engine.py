"""Coaching recommendations and drill selection."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from schemas import Drill, Recommendations

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _safe_parse(content: str) -> dict | None:
    try:
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3:
                content = "\n".join(lines[1:-1]).strip()
        return json.loads(content)
    except Exception:
        return None


def _fallback_feedback(context: str = "initial") -> dict:
    if context == "impromptu":
        return {
            "strengths": [
                "You produced an answer under pressure without completely losing direction.",
                "There was a visible attempt to organize the response.",
            ],
            "weaknesses": [
                "The main point did not land early enough.",
                "The answer needs a cleaner finish instead of trailing off.",
            ],
            "main_issue": "Lack of fast structure under pressure",
            "fixes": [
                "State your answer earlier, then support it with one strong idea.",
                "Finish with a final sentence that closes the thought.",
            ],
            "drills": [
                "Answer random prompts in 30 seconds: point, support, finish.",
                "Practice one-sentence answers first, then expand once the point is clear.",
            ],
        }

    return {
        "strengths": [
            "You communicated a central idea.",
            "Your speech had a basic logical direction.",
        ],
        "weaknesses": [
            "Delivery lacks control in pacing and pauses.",
            "The message could use more specificity.",
        ],
        "main_issue": "Lack of controlled delivery and strong idea development",
        "fixes": [
            "Slow your speaking rate and pause deliberately after each sentence.",
            "Add one specific example to support your main point.",
        ],
        "drills": [
            "Speak for 30 seconds with a 1-second pause after every sentence.",
            "Explain one idea using exactly 3 clear sentences.",
        ],
    }


def generate_feedback(
    transcript: str,
    voice_metrics: dict,
    delivery_metrics: dict,
    cognitive_metrics: dict,
    authority_score: float,
    context: str = "initial",
    prompt_text: str = "",
) -> dict:
    """GPT coaching feedback (preserved from v1 feedback_engine)."""
    transcript = (transcript or "").strip()
    context = (context or "initial").strip().lower()
    prompt_text = (prompt_text or "").strip()

    context_block = (
        f"IMPROMPTU mode. Prompt: {prompt_text}"
        if context == "impromptu"
        else f"General analysis. Prompt: {prompt_text}"
    )

    prompt = f"""
You are an elite speaking coach. Base feedback on REAL metrics. No fluff.

{context_block}

Transcript: {transcript}
Voice Metrics: {voice_metrics}
Delivery Metrics: {delivery_metrics}
Cognitive Analysis: {cognitive_metrics}
Authority Score: {authority_score}

Return ONLY JSON:
{{
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "main_issue": "...",
  "fixes": ["...", "..."],
  "drills": ["...", "..."]
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You are a strict, data-driven speaking coach. Output ONLY valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    parsed = _safe_parse(response.choices[0].message.content or "")
    if not parsed:
        return _fallback_feedback(context)

    for key in ["strengths", "weaknesses", "main_issue", "fixes", "drills"]:
        if key not in parsed:
            return _fallback_feedback(context)

    return parsed


def build_recommendations(feedback: dict, delivery_metrics: dict) -> Recommendations:
    """Map GPT feedback into v2 recommendations block."""
    wpm = delivery_metrics.get("words_per_minute", 0)
    main_issue = feedback.get("main_issue", "delivery control")
    fixes = feedback.get("fixes", [])
    fastest_tip = fixes[0] if fixes else "Pause deliberately after key statements."

    if wpm > 170:
        leverage = "pace control"
    elif delivery_metrics.get("filler_density", 0) > 0.04:
        leverage = "filler reduction"
    else:
        leverage = main_issue.lower()

    return Recommendations(
        highest_leverage_issue=leverage,
        fastest_improvement_tip=fastest_tip,
        coaching_summary=(
            "Your content has a foundation to build on. "
            "The fastest score lift comes from tightening delivery control on your highest-leverage weakness."
        ),
    )


DIMENSION_TARGET_METRICS = {
    "command": "terminal_rise_ratio",
    "clarity": "filler_words_per_min",
    "composure": "hesitation_cluster_score",
    "presence": "monotony_index",
    "persuasion": "dynamic_emphasis_score",
    "structure": "structure_score",
}


def _weakest_dimension_metric(dimension_map: dict[str, int]) -> str:
    """Map the lowest-scoring dimension to a v2 metric identifier."""
    if not dimension_map:
        return "command"
    weakest = min(dimension_map, key=dimension_map.get)
    return DIMENSION_TARGET_METRICS.get(weakest, weakest)


def build_drills(feedback: dict, delivery_metrics: dict, dimension_map: dict) -> list[Drill]:
    """Convert feedback drills into structured v2 drill objects."""
    raw_drills = feedback.get("drills", [])
    target_metric = "command"
    if delivery_metrics.get("filler_density", 0) > 0.04:
        target_metric = "filler_words_per_min"
    elif delivery_metrics.get("words_per_minute", 0) > 170:
        target_metric = "words_per_minute"

    drills: list[Drill] = []
    templates = [
        ("drop_the_landing_v1", "Drop the landing", "reduce rising declarative endings"),
        ("pause_control_v1", "Pause with purpose", "own clause-final pauses"),
    ]

    for index, instruction in enumerate(raw_drills[:2]):
        template = templates[index] if index < len(templates) else (
            f"custom_drill_{index + 1}",
            f"Practice drill {index + 1}",
            feedback.get("main_issue", "delivery improvement"),
        )
        drills.append(
            Drill(
                drill_id=template[0],
                title=template[1],
                goal=template[2],
                instructions=[instruction],
                duration_min=4,
                difficulty="beginner",
                target_metrics=[target_metric, _weakest_dimension_metric(dimension_map)],
            )
        )

    # TODO(v2.5): deterministic drill library selector (severity × impact × trainability)

    return drills


# Moment detection lives in services.moments (Milestone 2 deterministic layer).
