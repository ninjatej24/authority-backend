import parselmouth
import numpy as np

def extract_voice_metrics(audio_path):

    snd = parselmouth.Sound(audio_path)

    duration = snd.get_total_duration()

    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]

    pitch_mean = np.mean(pitch_values)
    pitch_std = np.std(pitch_values)

    intensity = snd.to_intensity()
    intensity_values = intensity.values[0]

    energy_mean = np.mean(intensity_values)

    return {
        "duration_seconds": float(duration),
        "pitch_mean": float(pitch_mean),
        "pitch_variation": float(pitch_std),
        "energy_mean": float(energy_mean)
    }