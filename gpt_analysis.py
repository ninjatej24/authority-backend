from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_cognition(transcript):

    prompt = f"""
You are an expert communication analyst.

Evaluate the following speech transcript across these dimensions.

Score each from 0 to 100.

Clarity – how clearly the idea is communicated  
Persuasion – how convincing the speech is  
Articulation – vocabulary quality and precision  
Idea Density – how much meaningful information is delivered  
Structure – logical organization of thoughts  

Return ONLY valid JSON in this format:

{{
 "clarity": number,
 "persuasion": number,
 "articulation": number,
 "idea_density": number,
 "structure": number
}}

Speech:
{transcript}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)