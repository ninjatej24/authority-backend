"""Audio normalization, conversion, silence trimming, and quality assessment."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import numpy as np
import parselmouth

from schemas import AudioQuality

TARGET_SAMPLE_RATE = 16000
MIN_USABLE_DURATION_S = 1.0
SHORT_RECORDING_S = 8.0


@dataclass
class PreprocessResult:
    wav_path: str
    duration_ms: int
    audio_quality: AudioQuality


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            capture_output=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def convert_to_wav(input_path: str) -> str:
    """Convert uploaded audio to mono 16 kHz WAV via ffmpeg."""
    return preprocess_audio(input_path).wav_path


def preprocess_audio(input_path: str | os.PathLike[str]) -> PreprocessResult:
    """
    Normalize audio to mono 16 kHz WAV, trim leading/trailing silence,
    and assess quality.
    """
    input_path = str(input_path)
    output_path = _processed_output_path(input_path)

    if _ffmpeg_available():
        _run_ffmpeg_preprocess(input_path, output_path)
    elif input_path.lower().endswith(".wav"):
        output_path = input_path
    else:
        raise RuntimeError("ffmpeg is required to convert non-WAV uploads")

    duration_ms, audio_quality = _assess_processed_audio(output_path)
    return PreprocessResult(
        wav_path=output_path,
        duration_ms=duration_ms,
        audio_quality=audio_quality,
    )


def _processed_output_path(input_path: str) -> str:
    base, _ = os.path.splitext(input_path)
    return f"{base}.processed.wav"


def _run_ffmpeg_preprocess(input_path: str, output_path: str) -> None:
    # Trim silence at both ends, normalize to mono 16 kHz.
    silence_filter = (
        "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-45dB:"
        "stop_periods=1:stop_duration=0.2:stop_threshold=-45dB"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-af",
            silence_filter,
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-ac",
            "1",
            output_path,
        ],
        check=True,
        capture_output=True,
    )


def _load_samples(wav_path: str) -> tuple[np.ndarray, int, float]:
    snd = parselmouth.Sound(wav_path)
    samples = snd.values[0] if snd.n_channels >= 1 else np.array([])
    sample_rate = int(snd.sampling_frequency)
    duration = float(snd.get_total_duration())
    return samples, sample_rate, duration


def _estimate_snr_db(samples: np.ndarray, sample_rate: int) -> float | None:
    """Approximate SNR from low-energy vs high-energy frames."""
    if len(samples) == 0:
        return None

    frame_size = max(int(sample_rate * 0.02), 1)
    energies: list[float] = []
    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if len(frame) == 0:
            continue
        energies.append(float(np.mean(frame**2)))

    if not energies:
        return None

    sorted_energies = sorted(energies)
    noise_count = max(len(sorted_energies) // 10, 1)
    signal_count = max(len(sorted_energies) // 10, 1)

    noise_energy = float(np.mean(sorted_energies[:noise_count]))
    signal_energy = float(np.mean(sorted_energies[-signal_count:]))

    if noise_energy <= 0:
        return None

    ratio = max(signal_energy / noise_energy, 1.0)
    return round(10 * np.log10(ratio), 1)


def _detect_clipping(samples: np.ndarray, threshold: float = 0.99) -> bool:
    if len(samples) == 0:
        return False
    peak = float(np.max(np.abs(samples)))
    if peak <= 0:
        return False
    return peak >= threshold


def _noise_level_from_snr(snr: float | None) -> str:
    if snr is None:
        return "unknown"
    if snr >= 20:
        return "low"
    if snr >= 12:
        return "medium"
    return "high"


def _estimate_single_speaker_likelihood(samples: np.ndarray, sample_rate: int) -> float | None:
    """
    Placeholder speaker-count proxy from long-window energy stability.
    TODO(v2.3): pyannote or WebRTC-based diarization for real estimates.
    """
    if len(samples) < sample_rate:
        return None

    frame_size = max(int(sample_rate * 0.5), 1)
    energies = []
    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if len(frame) == 0:
            continue
        energies.append(float(np.mean(frame**2)))

    if len(energies) < 2:
        return 0.9

    mean_energy = float(np.mean(energies))
    if mean_energy <= 0:
        return 0.5

    coefficient_of_variation = float(np.std(energies) / mean_energy)
    # Stable single-speaker delivery tends to have moderate, not chaotic, energy swings.
    likelihood = 1.0 - min(coefficient_of_variation, 0.6)
    return round(max(0.5, min(0.99, likelihood)), 2)


def assess_audio_quality(wav_path: str) -> AudioQuality:
    """Assess quality on an already-normalized WAV file."""
    _, audio_quality = _assess_processed_audio(wav_path)
    return audio_quality


def _assess_processed_audio(wav_path: str) -> tuple[int, AudioQuality]:
    warnings: list[str] = []

    try:
        samples, sample_rate, duration = _load_samples(wav_path)
    except Exception:
        return 0, AudioQuality(
            usable=False,
            background_noise_level="unknown",
            quality_warnings=["Could not load audio for quality assessment"],
        )

    duration_ms = int(duration * 1000)

    if duration <= 0:
        return 0, AudioQuality(
            usable=False,
            background_noise_level="unknown",
            quality_warnings=["Recording has zero duration"],
        )

    if duration < MIN_USABLE_DURATION_S:
        warnings.append("Recording is very short; metrics may be unreliable")

    if duration < SHORT_RECORDING_S:
        warnings.append("Short recording limits pause and moment analysis")

    snr = _estimate_snr_db(samples, sample_rate)
    clipping = _detect_clipping(samples)
    noise_level = _noise_level_from_snr(snr)

    if snr is not None and snr < 12:
        warnings.append("Low signal-to-noise ratio may reduce metric reliability")
    if clipping:
        warnings.append("Clipping detected; loudness metrics may be distorted")
    if float(np.max(np.abs(samples))) < 0.01:
        warnings.append("Very low signal level detected; recording may be near-silent")

    usable = (
        duration >= MIN_USABLE_DURATION_S
        and float(np.max(np.abs(samples))) >= 0.01
        and not clipping
        and (snr is None or snr >= 8)
    )

    speaker_likelihood = _estimate_single_speaker_likelihood(samples, sample_rate)

    # TODO(v2.3): LUFS normalization and room-noise classification

    return duration_ms, AudioQuality(
        usable=usable,
        snr_estimate_db=snr,
        clipping_detected=clipping,
        background_noise_level=noise_level,  # type: ignore[arg-type]
        single_speaker_likelihood=speaker_likelihood,
        quality_warnings=warnings,
    )
