from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _safe_parse(content: str):
    try:
        content = content.strip()

        if content.startswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3:
                content = "\n".join(lines[1:-1]).strip()

        return json.loads(content)
    except Exception:
        return None


def _fallback(context: str = "initial"):
    if context == "impromptu":
        return {
            "strengths": [
                "You were able to produce an answer under pressure without completely losing direction.",
                "There was at least a visible attempt to organize the response instead of speaking in fragments."
            ],
            "weaknesses": [
                "The response was not structured tightly enough, so the main point did not land early enough.",
                "Your answer under pressure needs a cleaner finish instead of trailing off or wandering."
            ],
            "main_issue": "Lack of fast structure under pressure",
            "fixes": [
                "State your answer earlier, then support it with one strong supporting idea instead of exploring too many directions.",
                "Finish with a final sentence that closes the thought instead of fading out."
            ],
            "drills": [
                "Answer random prompts in 30 seconds using a 3-part structure: point, support, finish.",
                "Practice giving one-sentence answers first, then expand only once the main point is clear."
            ]
        }

    return {
        "strengths": [
            "You communicated a clear central idea.",
            "Your speech had a basic logical direction."
        ],
        "weaknesses": [
            "Your delivery lacks control in pacing and pauses.",
            "Your message lacks depth and specificity."
        ],
        "main_issue": "Lack of controlled delivery and strong idea development",
        "fixes": [
            "Slow down your speaking rate and pause deliberately after each sentence.",
            "Add one specific example to support your main point."
        ],
        "drills": [
            "Speak for 30 seconds and insert a 1-second pause after every sentence.",
            "Explain one idea using exactly 3 clear sentences."
        ]
    }


def generate_feedback(
    transcript,
    voice_metrics,
    delivery_metrics,
    cognitive_metrics,
    authority_score,
    context="initial",
    prompt_text="",
):
    transcript = (transcript or "").strip()
    context = (context or "initial").strip().lower()
    prompt_text = (prompt_text or "").strip()

    if context == "impromptu":
        context_instructions = f"""
This response comes from IMPROMPTU mode.

The speaker had limited preparation time and was asked to respond to this prompt:

{prompt_text}

For IMPROMPTU mode, your feedback should focus more on:
- how quickly the speaker formed a clear point
- whether they stayed on the prompt
- whether the response had visible structure under pressure
- whether they drifted, circled, or delayed their main point
- whether they finished cleanly

For IMPROMPTU mode, avoid treating the response like a polished prepared speech.
Judge it as a live under-pressure answer.
"""
    else:
        context_instructions = f"""
This response comes from general speech analysis mode.

Original prompt:
{prompt_text}
"""

    prompt = f"""
You are an elite speaking coach.

You MUST base your feedback on REAL DATA from metrics.
Do NOT guess. Do NOT be generic.

{context_instructions}

---

Transcript:
{transcript}

Voice Metrics:
{voice_metrics}

Delivery Metrics:
{delivery_metrics}

Cognitive Analysis:
{cognitive_metrics}

Authority Score:
{authority_score}

---

CRITICAL RULES:

- You MUST reference specific metrics where relevant
  (e.g. words_per_minute, energy_mean, pitch_variation, silence_ratio)

- Do NOT say vague things like:
  ❌ "be more confident"
  ❌ "improve delivery"

- Instead say:
  ✅ "Your speaking rate is 88 WPM which is too slow and reduces energy"
  ✅ "Your energy_mean is 45 which is below optimal range (~55–70)"

- Be direct. Be sharp. No fluff.

- DO NOT over-focus on pitch_mean by itself.
- DO NOT claim something from the transcript that is not actually supported.
- For impromptu mode, prioritize structure under pressure, speed of thought, and staying on topic.

---

TASK:

1. EXACTLY 2 strengths
- Must reference something REAL (content or metrics)

2. EXACTLY 2 weaknesses
- Must explain WHY it hurts performance
- Use metrics if possible

3. MAIN ISSUE
- The single biggest bottleneck

4. EXACTLY 2 FIXES
- Must be BEHAVIOURAL (what to do differently)
- Concrete actions

5. EXACTLY 2 DRILLS
- Repeatable exercises
- Must directly fix weaknesses

---

RETURN ONLY JSON:

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
                "content": "You are a strict, data-driven speaking coach. Output ONLY valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    parsed = _safe_parse(response.choices[0].message.content)

    if not parsed:
      return _fallback(context)

    required_keys = ["strengths", "weaknesses", "main_issue", "fixes", "drills"]

    for key in required_keys:
        if key not in parsed:
            return _fallback(context)

    return parsed