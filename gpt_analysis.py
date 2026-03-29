from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _clamp_score(value, default=50):
    try:
        value = float(value)
        if value < 0:
            return 0
        if value > 100:
            return 100
        return round(value)
    except Exception:
        return default


def _clean_json_text(content: str) -> str:
    content = content.strip()

    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3:
            content = "\n".join(lines[1:-1]).strip()

    return content


def _fallback_response():
    return {
        "clarity": {
            "score": 50,
            "reason": "Fallback: transcript could not be reliably evaluated."
        },
        "persuasion": {
            "score": 45,
            "reason": "Fallback: persuasive strength could not be reliably evaluated."
        },
        "coherence": {
            "score": 50,
            "reason": "Fallback: coherence could not be reliably evaluated."
        },
        "idea_strength": {
            "score": 45,
            "reason": "Fallback: idea strength could not be reliably evaluated."
        },
        "conciseness": {
            "score": 50,
            "reason": "Fallback: conciseness could not be reliably evaluated."
        },
        "failure": False
    }


def _normalize_response(parsed):
    fallback = _fallback_response()

    if not isinstance(parsed, dict):
        return fallback

    normalized = {}

    for key in ["clarity", "persuasion", "coherence", "idea_strength", "conciseness"]:
        entry = parsed.get(key, {})
        if not isinstance(entry, dict):
            entry = {}

        normalized[key] = {
            "score": _clamp_score(entry.get("score"), fallback[key]["score"]),
            "reason": str(entry.get("reason", fallback[key]["reason"])).strip() or fallback[key]["reason"]
        }

    normalized["failure"] = bool(parsed.get("failure", False))

    return normalized


def analyze_cognition(transcript):
    transcript = (transcript or "").strip()

    if not transcript:
        return {
            "clarity": {"score": 20, "reason": "No transcript was available to evaluate."},
            "persuasion": {"score": 20, "reason": "No transcript was available to evaluate."},
            "coherence": {"score": 20, "reason": "No transcript was available to evaluate."},
            "idea_strength": {"score": 20, "reason": "No transcript was available to evaluate."},
            "conciseness": {"score": 20, "reason": "No transcript was available to evaluate."},
            "failure": True
        }

    prompt = f"""
You are an elite communication analyst.

Your job is to evaluate ONLY the transcript content.
Do not judge vocal tone, energy, charisma, pauses, pacing, pitch, or confidence from the text.
Only judge what is present in the words themselves.

Be strict.
Do not inflate scores.

Score ranges:
- 25–40 = poor
- 40–55 = weak
- 55–70 = decent
- 70–82 = strong
- 82+ = rare

Important:
- Most normal responses should land between 40 and 70.
- Do not give high scores just because the transcript is grammatically correct.
- Repetition, vagueness, weak examples, weak logic, and generic wording should lower scores.
- Only give 80+ if the transcript is unusually clear, sharp, well-structured, and impactful.

Evaluate these dimensions:

1. Clarity
- Is the meaning easy to understand?
- Are the statements direct rather than vague?

2. Persuasion
- Does the transcript make a convincing point?
- Does it create impact, not just explanation?

3. Coherence
- Does the transcript flow logically from one idea to the next?
- Does it read like one connected thought rather than fragments?

4. Idea Strength
- Is there a meaningful point here?
- Is the content substantial rather than generic or obvious?

5. Conciseness
- Is the transcript tight and efficient?
- Or is it repetitive, bloated, or unnecessarily wordy?

Failure rule:
Set "failure": true ONLY if the transcript is mostly empty, rambling, meaningless, or has no real point.
Do NOT mark failure just because the transcript is average or weak.

Return ONLY valid JSON in this exact format:

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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a strict evaluator. Return only valid JSON with no markdown."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content or ""

    try:
        parsed = json.loads(_clean_json_text(content))
    except Exception:
        return _fallback_response()

    return _normalize_response(parsed)