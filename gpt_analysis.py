from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_cognition(transcript):

    prompt = f"""
You are an elite communication analyst.

You must STRICTLY evaluate the speech. Do NOT inflate scores.

Most people score between 50–75.
Scores above 80 are rare and require exceptional communication.
Scores below 50 should be given if the speech is weak, unclear, or rambling.

Evaluate the transcript across these dimensions:

Clarity:
- Is the message easy to understand?

Persuasion:
- Is the speech convincing, confident, and impactful?

Coherence:
- Does the speech flow logically and connect well?

Idea Strength:
- Is there a strong, meaningful point being made?

Conciseness:
- Is the speech tight and efficient, or rambling and bloated?

---

Also detect FAILURE:
Mark failure = true if:
- No clear point is made
- Speech is mostly rambling or filler
- Very low meaningful content

---

Return ONLY valid JSON in this format:

{{
  "clarity": {{"score": number, "reason": "string"}},
  "persuasion": {{"score": number, "reason": "string"}},
  "coherence": {{"score": number, "reason": "string"}},
  "idea_strength": {{"score": number, "reason": "string"}},
  "conciseness": {{"score": number, "reason": "string"}},
  "failure": boolean
}}

Speech:
{transcript}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        parsed = json.loads(content)
    except Exception:
        # fallback in case GPT messes up JSON
        parsed = {
            "clarity": {"score": 60, "reason": "Parsing fallback"},
            "persuasion": {"score": 60, "reason": "Parsing fallback"},
            "coherence": {"score": 60, "reason": "Parsing fallback"},
            "idea_strength": {"score": 60, "reason": "Parsing fallback"},
            "conciseness": {"score": 60, "reason": "Parsing fallback"},
            "failure": False
        }

    return parsed