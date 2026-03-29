import parselmouth
import numpy as np


def extract_voice_metrics(audio_path):
    snd = parselmouth.Sound(audio_path)
    duration = float(snd.get_total_duration())

    if duration <= 0:
        return {
            "duration_seconds": 0.0,
            "pitch_mean": 0.0,
            "pitch_variation": 0.0,
            "energy_mean": 0.0,
            "energy_variation": 0.0,
            "silence_ratio": 0.0,
            "avg_pause_duration": 0.0,
            "pause_frequency": 0.0,
            "speech_density": 0.0
        }

    # =========================
    # 🎤 PITCH
    # =========================
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    pitch_values = pitch_values[pitch_values > 0]

    if len(pitch_values) == 0:
        pitch_mean = 0.0
        pitch_std = 0.0
    else:
        pitch_mean = float(np.mean(pitch_values))
        pitch_std = float(np.std(pitch_values))

    # =========================
    # 🔊 INTENSITY
    # =========================
    intensity = snd.to_intensity()
    intensity_values = intensity.values[0]

    if len(intensity_values) == 0:
        energy_mean = 0.0
        energy_std = 0.0
        silence_ratio = 0.0
        avg_pause_duration = 0.0
        pause_frequency = 0.0
        speech_density = 1.0

        return {
            "duration_seconds": duration,
            "pitch_mean": pitch_mean,
            "pitch_variation": pitch_std,
            "energy_mean": energy_mean,
            "energy_variation": energy_std,
            "silence_ratio": silence_ratio,
            "avg_pause_duration": avg_pause_duration,
            "pause_frequency": pause_frequency,
            "speech_density": speech_density
        }

    energy_mean = float(np.mean(intensity_values))
    energy_std = float(np.std(intensity_values))

    # =========================
    # ⏸️ PAUSE / SILENCE DETECTION
    # =========================
    # More adaptive than percentile-only.
    # We set silence threshold as a fraction of mean intensity,
    # but keep it within a sensible range so different recordings
    # don't all collapse to the same silence ratio.
    threshold = energy_mean * 0.6

    # keep threshold from becoming absurdly low or high
    threshold = max(threshold, np.percentile(intensity_values, 10))
    threshold = min(threshold, np.percentile(intensity_values, 40))

    silence_frames = intensity_values < threshold

    frame_duration = float(intensity.dx)
    silence_duration = float(np.sum(silence_frames) * frame_duration)
    silence_ratio = silence_duration / duration

    # Detect meaningful pause segments
    pauses = []
    current_pause = 0.0

    for is_silent in silence_frames:
        if is_silent:
            current_pause += frame_duration
        else:
            if current_pause >= 0.18: # ignore tiny micro-pauses
                pauses.append(current_pause)
            current_pause = 0.0

    if current_pause >= 0.18:
        pauses.append(current_pause)

    pause_count = len(pauses)
    avg_pause_duration = float(np.mean(pauses)) if pauses else 0.0
    pause_frequency = pause_count / duration

    # =========================
    # 🧠 SPEECH DENSITY
    # =========================
    speech_duration = max(duration - silence_duration, 0.0)
    speech_density = speech_duration / duration

    return {
        "duration_seconds": duration,

        # pitch
        "pitch_mean": pitch_mean,
        "pitch_variation": pitch_std,

        # energy
        "energy_mean": energy_mean,
        "energy_variation": energy_std,

        # pauses / rhythm
        "silence_ratio": float(silence_ratio),
        "avg_pause_duration": avg_pause_duration,
        "pause_frequency": float(pause_frequency),
        "speech_density": float(speech_density)
    }