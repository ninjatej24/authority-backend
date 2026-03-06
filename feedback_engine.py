from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_feedback(transcript, voice_metrics, delivery_metrics, cognitive_metrics, authority_score):

    prompt = f"""
You are an elite speaking coach analyzing a recorded speech.

Transcript:
{transcript}

Voice Metrics:
{voice_metrics}

Delivery Metrics:
{delivery_metrics}

Cognitive Metrics:
{cognitive_metrics}

Authority Score:
{authority_score}

Your task:

1. Identify 2 strengths in the speaker's communication.
2. Identify 2 weaknesses limiting authority or clarity.
3. Suggest 2 specific speaking drills that would improve their performance.

Return ONLY JSON in this format:

{{
"strengths": ["...", "..."],
"weaknesses": ["...", "..."],
"drills": ["...", "..."]
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional speaking coach."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    import json

    return json.loads(response.choices[0].message.content)