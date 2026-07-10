"""Speech-to-text with word-level timestamps via OpenAI."""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from schemas import Transcript, TranscriptSegment, TranscriptWord
from services.lexicons import is_filler_token

ASR_MODEL = "whisper-1"


@dataclass
class TranscriptionResult:
    transcript: Transcript
    approximate_timestamps: bool = False
    timestamp_source: str = "estimated"
    raw_response: object | None = None


def _ms(seconds: float) -> int:
    return max(int(seconds * 1000), 0)


def _read_field(obj, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _coerce_word(word, *, source: str) -> TranscriptWord | None:
    text = str(_read_field(word, "word", _read_field(word, "text", ""))).strip()
    if not text:
        return None
    start_raw = _read_field(word, "start", _read_field(word, "start_ms", 0))
    end_raw = _read_field(word, "end", _read_field(word, "end_ms", start_raw))
    start = float(start_raw)
    end = float(end_raw)
    if start > 1000 or end > 1000:
        start_ms = int(start)
        end_ms = int(end)
    else:
        start_ms = _ms(start)
        end_ms = _ms(end)
    confidence = _read_field(word, "confidence", None)
    return TranscriptWord(
        text=text,
        start_ms=max(start_ms, 0),
        end_ms=max(end_ms, start_ms),
        confidence=float(confidence) if confidence is not None else None,
        is_filler=is_filler_token(text),
        timestamp_source=source,  # type: ignore[arg-type]
    )


def _response_words(response: object) -> list[TranscriptWord]:
    raw_words = _read_field(response, "words", []) or []
    words = [_coerce_word(word, source="real") for word in raw_words]
    return [word for word in words if word is not None]


def _words_from_segments(segments: list, *, approximate: bool) -> tuple[list[TranscriptWord], bool, str]:
    words: list[TranscriptWord] = []
    used_interpolation = approximate
    timestamp_source = "segment"

    for segment in segments:
        segment_words = _read_field(segment, "words", []) or []
        if segment_words:
            for word in segment_words:
                coerced = _coerce_word(word, source="segment")
                if coerced:
                    words.append(coerced)
            continue

        used_interpolation = True
        timestamp_source = "interpolated"
        text = str(_read_field(segment, "text", "")).strip()
        start = float(_read_field(segment, "start", 0))
        end = float(_read_field(segment, "end", start))
        tokens = text.split()
        if not tokens:
            continue

        duration = max(end - start, 0.001)
        step = duration / len(tokens)
        for index, token in enumerate(tokens):
            word_start = start + index * step
            word_end = word_start + step
            words.append(
                TranscriptWord(
                    text=token,
                    start_ms=_ms(word_start),
                    end_ms=_ms(word_end),
                    confidence=None,
                    is_filler=is_filler_token(token),
                    timestamp_source="interpolated",
                )
            )

    if words and all(word.timestamp_source == "real" for word in words):
        timestamp_source = "real"
    return words, used_interpolation, timestamp_source


def _build_segments(
    words: list[TranscriptWord], full_text: str, duration_ms: int
) -> list[TranscriptSegment]:
    if not words:
        return []

    opening_end = min(duration_ms, 12_000)
    closing_start = max(duration_ms - 12_000, 0)

    segments: list[TranscriptSegment] = []
    current_role = "opening"
    current_start = words[0].start_ms
    current_tokens: list[str] = []

    for word in words:
        if word.start_ms >= opening_end and current_role == "opening":
            if current_tokens:
                segments.append(
                    TranscriptSegment(
                        segment_id=f"seg_{len(segments) + 1}",
                        start_ms=current_start,
                        end_ms=word.start_ms,
                        text=" ".join(current_tokens),
                        role="opening",
                        timestamp_source=_segment_source(words, current_start, word.start_ms),
                    )
                )
            current_role = "body"
            current_start = word.start_ms
            current_tokens = []

        if word.start_ms >= closing_start and current_role == "body":
            if current_tokens:
                segments.append(
                    TranscriptSegment(
                        segment_id=f"seg_{len(segments) + 1}",
                        start_ms=current_start,
                        end_ms=word.start_ms,
                        text=" ".join(current_tokens),
                        role="body",
                        timestamp_source=_segment_source(words, current_start, word.start_ms),
                    )
                )
            current_role = "closing"
            current_start = word.start_ms
            current_tokens = []

        current_tokens.append(word.text)

    if current_tokens:
        segments.append(
            TranscriptSegment(
                segment_id=f"seg_{len(segments) + 1}",
                start_ms=current_start,
                end_ms=words[-1].end_ms,
                text=" ".join(current_tokens),
                role=current_role,  # type: ignore[arg-type]
                timestamp_source=_segment_source(words, current_start, words[-1].end_ms),
            )
        )

    if not segments and full_text.strip():
        segments.append(
            TranscriptSegment(
                segment_id="seg_1",
                start_ms=words[0].start_ms,
                end_ms=words[-1].end_ms,
                text=full_text.strip(),
                role="other",
                timestamp_source=_segment_source(words, words[0].start_ms, words[-1].end_ms),
            )
        )

    return segments


def _segment_source(words: list[TranscriptWord], start_ms: int, end_ms: int) -> str:
    selected = [word.timestamp_source for word in words if word.end_ms >= start_ms and word.start_ms <= end_ms]
    if not selected:
        return "estimated"
    if all(source == "real" for source in selected):
        return "real"
    if any(source == "interpolated" for source in selected):
        return "interpolated"
    if any(source == "estimated" for source in selected):
        return "estimated"
    return "segment"


def _average_confidence(words: list[TranscriptWord]) -> float | None:
    values = [word.confidence for word in words if word.confidence is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def transcribe_audio(
    client: OpenAI,
    wav_path: str,
    duration_ms: int,
    language: str = "en",
) -> TranscriptionResult:
    """Transcribe audio using OpenAI Whisper with verbose JSON when available."""
    approximate_timestamps = False

    with open(wav_path, "rb") as audio_file:
        try:
            response = client.audio.transcriptions.create(
                model=ASR_MODEL,
                file=audio_file,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
        except Exception:
            audio_file.seek(0)
            response = client.audio.transcriptions.create(
                model=ASR_MODEL,
                file=audio_file,
                language=language,
                response_format="verbose_json",
            )

    full_text = _read_field(response, "text", "") or ""
    segments_raw = _read_field(response, "segments", None) or []
    words = _response_words(response)
    timestamp_source = "real" if words else "estimated"
    if not words:
        words, approximate_timestamps, timestamp_source = _words_from_segments(segments_raw, approximate=False)

    if not words and full_text.strip():
        approximate_timestamps = True
        timestamp_source = "estimated"
        tokens = full_text.split()
        step_ms = max(duration_ms // max(len(tokens), 1), 1)
        words = [
            TranscriptWord(
                text=token,
                start_ms=index * step_ms,
                end_ms=min((index + 1) * step_ms, duration_ms),
                confidence=None,
                is_filler=is_filler_token(token),
                timestamp_source="estimated",
            )
            for index, token in enumerate(tokens)
        ]

    detected_language = getattr(response, "language", None) or language
    transcript = Transcript(
        full_text=full_text.strip(),
        speaker_language_confidence=0.93 if detected_language else None,
        asr_model=ASR_MODEL,
        overall_asr_confidence=_average_confidence(words),
        words=words,
        segments=_build_segments(words, full_text, duration_ms),
    )

    # TODO(v2.2): Montreal Forced Aligner refinement for sharper word boundaries

    return TranscriptionResult(
        transcript=transcript,
        approximate_timestamps=approximate_timestamps,
        timestamp_source=timestamp_source,
        raw_response=response,
    )
