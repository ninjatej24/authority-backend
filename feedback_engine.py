from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_feedback(transcript, voice_metrics, delivery_metrics, cognitive_metrics, authority_score):

    prompt = f"""
You are an elite speaking coach.

Your job is to give DIRECT, SPECIFIC, and ACTIONABLE feedback.
Avoid generic advice.

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

Your task:

1. Identify EXACTLY 2 strengths
- Must be specific
- Must reference actual behavior

2. Identify EXACTLY 2 weaknesses
- Be honest and direct
- Explain WHY it hurts their communication

3. Identify the SINGLE biggest issue ("main_issue")
- The #1 thing holding them back

4. Give EXACTLY 2 actionable fixes
- Must be behavioral (what to DO differently)
- Not vague advice

5. Give EXACTLY 2 drills
- Practical speaking exercises they can repeat

---

Rules:
- No generic advice like "be more confident"
- No fluff
- Be concise but sharp
- Feedback should feel like a real coach, not AI

---

Return ONLY valid JSON:

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
        messages=[
            {"role": "system", "content": "You are a direct, high-level speaking coach."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    content = response.choices[0].message.content

    try:
        parsed = json.loads(content)
    except Exception:
        # fallback (prevents crashes)
        parsed = {
            "strengths": ["Clear attempt to communicate an idea", "Some structure is present"],
            "weaknesses": ["Lacks precision", "Delivery could be more controlled"],
            "main_issue": "Lack of clarity",
            "fixes": ["Slow down and simplify sentences", "Pause between key points"],
            "drills": ["Explain a topic in 3 sentences", "Practice speaking with pauses"]
        }

    return parsed