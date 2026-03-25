import parselmouth
import numpy as np

def extract_voice_metrics(audio_path):

    snd = parselmouth.Sound(audio_path)

    duration = snd.get_total_duration()

    # =========================
    # 🎤 PITCH
    # =========================
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]

    pitch_mean = np.mean(pitch_values)
    pitch_std = np.std(pitch_values)

    # =========================
    # 🔊 INTENSITY
    # =========================
    intensity = snd.to_intensity()
    intensity_values = intensity.values[0]

    energy_mean = np.mean(intensity_values)
    energy_std = np.std(intensity_values)  # 🔥 dynamic range

    # =========================
    # ⏸️ PAUSE DETECTION
    # =========================
    threshold = np.percentile(intensity_values, 25)  # silence threshold

    silence_frames = intensity_values < threshold

    silence_duration = np.sum(silence_frames) * intensity.dx
    silence_ratio = silence_duration / duration

    # Detect pause segments
    pauses = []
    current_pause = 0

    for val in silence_frames:
        if val:
            current_pause += intensity.dx
        else:
            if current_pause > 0.15:  # ignore micro pauses
                pauses.append(current_pause)
            current_pause = 0

    if current_pause > 0.15:
        pauses.append(current_pause)

    pause_count = len(pauses)
    avg_pause_duration = np.mean(pauses) if pauses else 0

    pause_frequency = pause_count / duration

    # =========================
    # 🧠 SPEECH DENSITY
    # =========================
    speech_duration = duration - silence_duration
    speech_density = speech_duration / duration

    return {
        "duration_seconds": float(duration),

        # pitch
        "pitch_mean": float(pitch_mean),
        "pitch_variation": float(pitch_std),

        # energy
        "energy_mean": float(energy_mean),
        "energy_variation": float(energy_std),

        # pauses
        "silence_ratio": float(silence_ratio),
        "avg_pause_duration": float(avg_pause_duration),
        "pause_frequency": float(pause_frequency),

        # rhythm proxy
        "speech_density": float(speech_density)
    }