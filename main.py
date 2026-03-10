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

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create FastAPI app
app = FastAPI()


@app.post("/analyze")
async def analyze_voice(file: UploadFile = File(...)):

    # Preserve file extension
    suffix = os.path.splitext(file.filename)[1]

    # Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Convert audio to WAV (needed for Parselmouth)
    wav_path = convert_to_wav(tmp_path)

    voice_metrics = extract_voice_metrics(wav_path)

    # Open saved audio file
    audio_file = open(wav_path, "rb")

    # Send to Whisper
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

    cognitive_metrics = analyze_cognition(transcript.text)

    authority_score = compute_authority_score(voice_metrics, cognitive_metrics)

    word_count = len(transcript.text.split())

    duration = voice_metrics["duration_seconds"]

    wpm = (word_count / duration) * 60

    # Detect filler words
    filler_words = ["um", "uh", "like", "you know", "sort of", "kind of"]

    transcript_lower = transcript.text.lower()

    filler_count = sum(transcript_lower.count(word) for word in filler_words)

    filler_density = filler_count / max(word_count, 1)

    feedback = generate_feedback(
    transcript.text,
    voice_metrics,
    {
        "words_per_minute": wpm,
        "filler_density": filler_density
    },
    cognitive_metrics,
    authority_score
)

    return {
    "transcript": transcript.text,
    "voice_metrics": voice_metrics,
    "cognitive_metrics": cognitive_metrics,
    "delivery_metrics": {
        "words_per_minute": wpm,
        "filler_density": filler_density
    },
    "authority_score": authority_score,
    "feedback": feedback
}