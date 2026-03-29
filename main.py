from fastapi import FastAPI, UploadFile, File
from openai import OpenAI
from dotenv import load_dotenv
from audio_analysis import extract_voice_metrics
from gpt_analysis import analyze_cognition
from scoring import compute_authority_score
from feedback_engine import generate_feedback

import tempfile
import os
import subprocess

# =========================
# 🎧 AUDIO CONVERSION
# =========================
def convert_to_wav(input_path):
    output_path = input_path + ".wav"

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        output_path
    ], check=True)

    return output_path


# =========================
# 🔐 SETUP
# =========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()


# =========================
# 🚀 MAIN ENDPOINT
# =========================
@app.post("/analyze")
async def analyze_voice(file: UploadFile = File(...)):

    # Save uploaded file
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Convert to WAV
    wav_path = convert_to_wav(tmp_path)

    # =========================
    # 🎤 VOICE METRICS
    # =========================
    voice_metrics = extract_voice_metrics(wav_path)

    # =========================
    # 🧠 TRANSCRIPTION
    # =========================
    with open(wav_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

    text = transcript.text or ""

    # =========================
    # 🧠 COGNITIVE ANALYSIS
    # =========================
    cognitive_metrics = analyze_cognition(text)

    # =========================
    # 📊 DELIVERY METRICS
    # =========================
    word_count = len(text.split())
    duration = max(voice_metrics.get("duration_seconds", 1), 1)

    wpm = (word_count / duration) * 60

    filler_words = ["um", "uh", "like", "you know", "sort of", "kind of"]
    transcript_lower = text.lower()

    filler_count = sum(transcript_lower.count(word) for word in filler_words)
    filler_density = filler_count / max(word_count, 1)

    delivery_metrics = {
        "words_per_minute": wpm,
        "filler_density": filler_density
    }

    # =========================
    # 🎯 SCORING
    # =========================
    authority_score = compute_authority_score(
        voice_metrics,
        cognitive_metrics,
        delivery_metrics
    )

    # =========================
    # 💬 FEEDBACK
    # =========================
    feedback = generate_feedback(
        text,
        voice_metrics,
        delivery_metrics,
        cognitive_metrics,
        authority_score
    )

    # =========================
    # 🧪 DEBUG LOGS (IMPORTANT)
    # =========================
    print("\n===== DEBUG =====")
    print("Transcript:", text)
    print("Voice Metrics:", voice_metrics)
    print("Delivery Metrics:", delivery_metrics)
    print("Cognitive Metrics:", cognitive_metrics)
    print("Authority Score:", authority_score)
    print("=================\n")

    # =========================
    # 📤 RESPONSE
    # =========================
    return {
        "transcript": text,
        "voice_metrics": voice_metrics,
        "cognitive_metrics": cognitive_metrics,
        "delivery_metrics": delivery_metrics,
        "authority_score": authority_score,
        "feedback": feedback
    }