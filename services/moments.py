"""Deterministic timestamped moment detection."""

from __future__ import annotations

from schemas import Moment, TranscriptWord
from services.acoustic_metrics import WindowFeature


def build_moments(
    words: list[TranscriptWord],
    duration_ms: int,
    windows: list[WindowFeature],
    delivery_metrics: dict,
    linguistic: dict,
) -> list[Moment]:
    """Detect timestamped moments from sliding-window acoustic features."""
    if duration_ms <= 0:
        return []

    moments: list[Moment] = []
    moment_index = 1

    if windows:
        strongest = max(windows, key=lambda w: w.command_score + w.presence_score)
        weakest = min(windows, key=lambda w: w.composure_score + w.clarity_score)

        moments.append(
            _make_moment(
                moment_index,
                "strongest_moment",
                strongest,
                severity="highlight",
                headline="Most commanding section",
                summary=(
                    "This stretch may have sounded relatively controlled, with steadier pace "
                    f"around {strongest.wpm:.0f} WPM and stronger vocal presence."
                ),
                dimension_impact={"command": 0.14, "presence": 0.1},
                preview_visible_free=True,
            )
        )
        moment_index += 1

        moments.append(
            _make_moment(
                moment_index,
                "weakest_moment",
                weakest,
                severity="medium",
                headline="Least steady section",
                summary=(
                    "Pause and filler patterns here may have made this stretch sound less settled "
                    "than the rest of the recording."
                ),
                dimension_impact={"composure": -0.12, "clarity": -0.08},
                preview_visible_free=False,
            )
        )
        moment_index += 1

        for window in windows:
            if window.filler_rate > 0.12:
                moments.append(
                    _make_moment(
                        moment_index,
                        "filler_cluster",
                        window,
                        severity="medium",
                        headline="Filler cluster detected",
                        summary=(
                            "Filler density may have increased here, which can make statements "
                            "sound less final."
                        ),
                        dimension_impact={"clarity": -0.1, "command": -0.08},
                        preview_visible_free=False,
                    )
                )
                moment_index += 1
                break

        for window in windows:
            if window.rushing:
                moments.append(
                    _make_moment(
                        moment_index,
                        "rushing_moment",
                        window,
                        severity="medium",
                        headline="Pace may have rushed here",
                        summary=(
                            f"Speaking rate in this window reached about {window.wpm:.0f} WPM, "
                            "which may have sounded hurried."
                        ),
                        dimension_impact={"command": -0.1, "composure": -0.06},
                        preview_visible_free=False,
                    )
                )
                moment_index += 1
                break

        for window in windows:
            if window.monotone:
                moments.append(
                    _make_moment(
                        moment_index,
                        "monotone_stretch",
                        window,
                        severity="low",
                        headline="Flatter vocal colour here",
                        summary=(
                            "Pitch and loudness variation were relatively limited in this stretch, "
                            "which may have reduced emphasis."
                        ),
                        dimension_impact={"presence": -0.1, "persuasion": -0.06},
                        preview_visible_free=False,
                    )
                )
                moment_index += 1
                break

        drop = _find_confidence_drop(windows)
        if drop:
            moments.append(
                _make_moment(
                    moment_index,
                    "confidence_drop",
                    drop,
                    severity="medium",
                    headline="Certainty may have dipped here",
                    summary=(
                        "This section shows a local decline in composure relative to neighbouring "
                        "windows, possibly from pauses or fillers."
                    ),
                    dimension_impact={"composure": -0.14, "clarity": -0.08},
                    preview_visible_free=False,
                )
            )
            moment_index += 1

        hesitation = max(windows, key=lambda w: w.pause_ms)
        if hesitation.pause_ms > 700:
            moments.append(
                _make_moment(
                    moment_index,
                    "hesitation_cluster",
                    hesitation,
                    severity="medium",
                    headline="Hesitation cluster here",
                    summary=(
                        "Extended pausing in this stretch may have sounded like mid-thought searching."
                    ),
                    dimension_impact={"composure": -0.12, "command": -0.07},
                    preview_visible_free=False,
                )
            )
            moment_index += 1

    closing_score = linguistic.get("closing_strength_score")
    opening_score = linguistic.get("opening_strength_score")

    if words and closing_score is not None:
        closing_start = max(duration_ms - int(min(12, duration_ms / 1000) * 1000), 0)
        if closing_score >= 0.7:
            moments.append(
                Moment(
                    moment_id=f"m{moment_index}",
                    type="strong_ending",
                    start_ms=closing_start,
                    end_ms=duration_ms,
                    severity="highlight",
                    headline="Stronger closing section",
                    summary=(
                        "The ending may have landed with clearer closure and fewer trailing fillers."
                    ),
                    dimension_impact={"structure": 0.12, "command": 0.08},
                    preview_visible_free=False,
                )
            )
            moment_index += 1
        elif closing_score < 0.5:
            moments.append(
                Moment(
                    moment_id=f"m{moment_index}",
                    type="weak_ending",
                    start_ms=closing_start,
                    end_ms=duration_ms,
                    severity="medium",
                    headline="Ending may have trailed off",
                    summary=(
                        "The close of the recording may have sounded less decisive than the body."
                    ),
                    dimension_impact={"structure": -0.1, "command": -0.08},
                    preview_visible_free=False,
                )
            )
            moment_index += 1

    if words and opening_score is not None and opening_score >= 0.75:
        opening_end = min(int(duration_ms * 0.2), 12_000)
        moments.append(
            Moment(
                moment_id=f"m{moment_index}",
                type="decisive_moment",
                start_ms=words[0].start_ms,
                end_ms=opening_end,
                severity="highlight",
                headline="Direct opening",
                summary=(
                    "The opening may have established the topic with relatively direct framing."
                ),
                dimension_impact={"structure": 0.1, "command": 0.08},
                preview_visible_free=True,
            )
        )

    return moments[:8]


def _make_moment(
    index: int,
    moment_type: str,
    window: WindowFeature,
    *,
    severity: str,
    headline: str,
    summary: str,
    dimension_impact: dict[str, float],
    preview_visible_free: bool,
) -> Moment:
    return Moment(
        moment_id=f"m{index}",
        type=moment_type,
        start_ms=window.start_ms,
        end_ms=window.end_ms,
        severity=severity,  # type: ignore[arg-type]
        headline=headline,
        summary=summary,
        dimension_impact=dimension_impact,
        preview_visible_free=preview_visible_free,
    )


def _find_confidence_drop(windows: list[WindowFeature]) -> WindowFeature | None:
    if len(windows) < 3:
        return None
    best_drop = None
    best_delta = 0.0
    for index in range(1, len(windows) - 1):
        prev_score = windows[index - 1].composure_score
        cur_score = windows[index].composure_score
        delta = prev_score - cur_score
        if delta > best_delta and delta >= 0.15:
            best_delta = delta
            best_drop = windows[index]
    return best_drop
