"""Fact-led report composition quality tests."""

from __future__ import annotations

import re

from schemas import (
    AudioQuality,
    DimensionScores,
    Moment,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    Uncertainty,
)
from services.deterministic_coaching import build_deterministic_coaching
from services.diagnostic_reasoning import build_diagnostic_reasoning
from services.report_builder import build_report
from tests.test_psychological_inference import _infer, _metrics, _scores
from tests.test_report_builder import _evidence


FORBIDDEN_PHRASES = {
    "concrete anchors",
    "process detail",
    "the message explained the idea more than it proved it",
    "change the behaviour",
    "useful signal is behavioural",
    "the ceiling is",
    "selected focus",
    "deterministic drill",
    "winning diagnosis",
    "supported by x evidence items",
    "observed as",
    "hypothesis",
    "backend",
    "good communication",
    "effective communication",
    "strong communication",
    "clear communication",
    "powerful communication",
    "better communication",
    "repeatable strength",
    "local change in control",
}


def _transcript(text: str, *, confidence: float = 0.94, timestamp_source: str = "real", duration_ms: int = 45000) -> Transcript:
    tokens = text.split()
    step = max(duration_ms // max(len(tokens), 1), 1)
    words = [
        TranscriptWord(
            text=token,
            start_ms=index * step,
            end_ms=min((index + 1) * step, duration_ms),
            confidence=confidence,
            is_filler=token.lower().strip(".,!?") in {"um", "uh", "er"},
            timestamp_source=timestamp_source,  # type: ignore[arg-type]
        )
        for index, token in enumerate(tokens)
    ]
    first_cut = max(1, len(tokens) // 5)
    last_cut = max(first_cut + 1, len(tokens) - len(tokens) // 5)
    segments = [
        TranscriptSegment(
            segment_id="seg_1",
            start_ms=words[0].start_ms,
            end_ms=words[first_cut - 1].end_ms,
            text=" ".join(tokens[:first_cut]),
            role="opening",
            timestamp_source=timestamp_source,  # type: ignore[arg-type]
        ),
        TranscriptSegment(
            segment_id="seg_2",
            start_ms=words[first_cut].start_ms,
            end_ms=words[last_cut - 1].end_ms,
            text=" ".join(tokens[first_cut:last_cut]),
            role="body",
            timestamp_source=timestamp_source,  # type: ignore[arg-type]
        ),
        TranscriptSegment(
            segment_id="seg_3",
            start_ms=words[last_cut].start_ms,
            end_ms=words[-1].end_ms,
            text=" ".join(tokens[last_cut:]),
            role="closing",
            timestamp_source=timestamp_source,  # type: ignore[arg-type]
        ),
    ]
    return Transcript(
        full_text=text,
        speaker_language_confidence=confidence,
        asr_model="test",
        overall_asr_confidence=confidence,
        words=words,
        segments=segments,
    )


def _moment(moment_type: str, *, timestamp_source: str = "real") -> Moment:
    return Moment(
        moment_id=f"m_{moment_type}",
        type=moment_type,
        start_ms=12000,
        end_ms=18000,
        timestamp_source=timestamp_source,  # type: ignore[arg-type]
        severity="medium",
        headline=moment_type.replace("_", " ").title(),
        summary="Local behaviour changed in this span.",
        listener_interpretation="Listeners hear this as a local change in control.",
        why_it_matters="The moment shows where the recording changes.",
        confidence=0.82,
        supporting_evidence_ids=["fact_ev_thin_proof"],
        supporting_metrics=["window.wpm"],
        transcript_span="sample span",
        dimension_impact={"composure": -0.12, "structure": -0.08},
        importance_score=0.84,
    )


def _report(kind: str, *, weak_transcript: bool = False, weak_timestamps: bool = False):
    if kind == "thin_proof":
        text = (
            "The main reason communication matters is that communicating my ideas better would help my future. "
            "It would help me explain myself clearly and create more opportunities."
        )
        metrics = _metrics(
            linguistic={
                "specificity_score": 0.12,
                "concreteness_score": 0.08,
                "opening_strength_score": 0.82,
                "closing_strength_score": 0.72,
                "structure_score": 0.74,
            }
        )
        moments = [_moment("strong_opening", timestamp_source="estimated" if weak_timestamps else "real")]
    elif kind == "hesitation":
        text = (
            "The answer is that onboarding is slowing revenue. For example last month three customers waited two weeks, "
            "so I would simplify the first call and assign one owner."
        )
        metrics = _metrics(
            raw={"words_per_minute": 188.0, "mid_phrase_pause_rate": 0.48},
            linguistic={
                "specificity_score": 0.76,
                "concreteness_score": 0.64,
                "acoustic_hesitations": 3,
                "lexical_fillers": 0,
                "filler_words_per_min": 1.0,
            },
            derived={"hesitation_cluster_score": 0.72, "composure_index": 0.32},
            rhythm={"speed_up_segments": 2, "burst_speaking_segments": 1, "rhythm_consistency": 0.38},
        )
        moments = [_moment("hesitation_cluster")]
    elif kind == "weak_close":
        text = (
            "The plan is simple. We should clarify the goal, assign the owner, and review progress every Friday. "
            "That keeps the team moving and"
        )
        metrics = _metrics(
            raw={"terminal_rising_ratio": 0.62, "terminal_falling_ratio": 0.12},
            linguistic={
                "specificity_score": 0.72,
                "concreteness_score": 0.58,
                "closing_strength_score": 0.24,
                "opening_strength_score": 0.8,
                "structure_score": 0.72,
            },
        )
        moments = [_moment("weak_closing")]
    elif kind == "unclear":
        text = "I guess there are a few things and the point depends because it changes and then there is another thing to consider."
        metrics = _metrics(
            linguistic={
                "structure_score": 0.28,
                "rambling_score": 0.62,
                "opening_strength_score": 0.28,
                "specificity_score": 0.5,
                "concreteness_score": 0.45,
            }
        )
        moments = [_moment("weak_opening")]
    else:
        raise AssertionError(kind)

    transcript = _transcript(
        text,
        confidence=0.42 if weak_transcript else 0.94,
        timestamp_source="estimated" if weak_timestamps else "real",
    )
    audio_quality = AudioQuality(usable=True, background_noise_level="low")
    uncertainty = Uncertainty(overall_confidence_label="medium_high", reasons=[])
    scores = _scores().model_copy(
        update={
            "score_confidence": 0.82,
            "dimension_scores": DimensionScores(
                command=58 if kind == "weak_close" else 64,
                clarity=58 if kind == "thin_proof" else 70,
                composure=44 if kind == "hesitation" else 68,
                presence=66,
                persuasion=46 if kind == "thin_proof" else 68,
                structure=42 if kind == "unclear" else 68,
            ),
        }
    )
    inference = _infer(metrics, audio_quality=audio_quality, duration_ms=45000)
    diagnostic = build_diagnostic_reasoning(
        metrics=metrics,
        psychological_inference=inference,
        evidence=_evidence(),
        moments=moments,
        scores=scores,
        audio_quality=audio_quality,
        uncertainty=uncertainty,
        duration_ms=45000,
        scenario="benchmark",
    )
    coaching = build_deterministic_coaching(
        metrics=metrics,
        scores=scores,
        psychological_inference=inference,
        diagnostic_reasoning=diagnostic,
        report=None,
        audio_quality=audio_quality,
        uncertainty=uncertainty,
        duration_ms=45000,
        scenario="benchmark",
    )
    return build_report(
        scores=scores,
        metrics=metrics,
        psychological_inference=inference,
        diagnostic_reasoning=diagnostic,
        coaching_engine=coaching,
        evidence=_evidence(),
        moments=moments,
        uncertainty=uncertainty,
        audio_quality=audio_quality,
        duration_ms=45000,
        scenario="benchmark",
        transcript=transcript,
    )


def _major_sections(report):
    reads = report.perception_map.model_dump().values()
    return [
        ("mirror", report.mirror.evidence_ids),
        ("diagnosis", report.diagnosis.evidence_ids),
        *[(f"read_{index}", read["evidence_ids"]) for index, read in enumerate(reads) if read],
        ("hidden_cost", report.hidden_cost.evidence_ids),
        ("highest_leverage_fix", report.highest_leverage_fix.evidence_ids),
        ("training_prescription", report.training_prescription.evidence_ids),
        ("retest_plan", report.retest_plan.evidence_ids),
    ]


def _strings(report):
    reads = [read["text"] for read in report.perception_map.model_dump().values() if read] if report.perception_map else []
    values = [
        report.insufficient_sample.title if report.insufficient_sample else None,
        report.insufficient_sample.explanation if report.insufficient_sample else None,
        report.insufficient_sample.retry_instruction if report.insufficient_sample else None,
        report.mirror.headline if report.mirror else None,
        report.mirror.identity_read if report.mirror else None,
        report.diagnosis.core_pattern if report.diagnosis else None,
        report.diagnosis.social_consequence if report.diagnosis else None,
        *reads,
        report.hidden_cost.consequence if report.hidden_cost else None,
        report.highest_leverage_fix.plain_english if report.highest_leverage_fix else None,
        report.highest_leverage_fix.why_this_matters if report.highest_leverage_fix else None,
        report.training_prescription.why_chosen if report.training_prescription else None,
        report.training_prescription.success_signal if report.training_prescription else None,
        *(report.training_prescription.instructions if report.training_prescription else []),
        *[item.signal for item in report.evidence_chain],
        *[item.what_happened for item in report.evidence_chain],
        *[item.why_it_matters for item in report.evidence_chain],
        *[item.headline for item in report.timeline],
    ]
    return [value for value in values if value]


def _norm(text: str | None) -> set[str]:
    return set(re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split()) - {"the", "a", "and", "to", "of", "in", "it", "is"}


def _similarity(a: str | None, b: str | None) -> float:
    left, right = _norm(a), _norm(b)
    return len(left & right) / max(len(left | right), 1)


def test_major_claims_link_to_evidence_cards_with_recording_fact_ids():
    report = _report("thin_proof")
    cards = {card.evidence_id: card for card in report.evidence_chain}

    for _, evidence_ids in _major_sections(report):
        assert evidence_ids
        assert all(evidence_id in cards for evidence_id in evidence_ids)
        assert any(cards[evidence_id].recording_fact_ids for evidence_id in evidence_ids)


def test_no_section_contains_claims_without_source_facts():
    report = _report("hesitation")

    assert all(ids for _, ids in _major_sections(report))
    assert all(card.recording_fact_ids for card in report.evidence_chain)


def test_mirror_and_diagnosis_are_not_near_duplicates():
    report = _report("thin_proof")

    assert _similarity(report.mirror.headline, report.diagnosis.core_pattern) < 0.72


def test_perception_reads_are_not_near_duplicates():
    report = _report("thin_proof")
    reads = [read["text"] for read in report.perception_map.model_dump().values() if read]

    assert 2 <= len(reads) <= 3
    for index, text in enumerate(reads):
        assert all(_similarity(text, other) < 0.72 for other in reads[index + 1 :])


def test_hidden_cost_does_not_repeat_diagnosis():
    report = _report("hesitation")

    assert _similarity(report.hidden_cost.consequence, report.diagnosis.core_pattern) < 0.7


def test_highest_leverage_fix_does_not_repeat_hidden_cost():
    report = _report("weak_close")

    assert _similarity(report.highest_leverage_fix.plain_english, report.hidden_cost.consequence) < 0.72


def test_training_prescription_contains_concrete_instructions():
    report = _report("thin_proof")

    assert report.training_prescription.title
    assert len(report.training_prescription.instructions) >= 3
    assert re.search(r"\b(5|6|8|minute|minutes|reps)\b", " ".join(report.training_prescription.instructions + [str(report.training_prescription.duration_min)]))
    assert report.training_prescription.success_signal
    assert "retest" in report.training_prescription.success_signal.lower()


def test_drill_target_matches_diagnosis_target():
    report = _report("thin_proof")
    drill = next(item for item in report.coaching_engine.drill_library if item.drill_id == report.training_prescription.drill_id)

    assert "grounded_specificity" in drill.target_behaviours or drill.category == "specificity"


def test_evidence_headline_body_and_what_happened_are_not_duplicates():
    report = _report("hesitation")

    for card in report.evidence_chain:
        assert _similarity(card.signal, card.what_happened) < 0.8
        assert _similarity(card.signal, card.why_it_matters) < 0.8


def test_timeline_labels_describe_actual_local_change():
    report = _report("hesitation")

    assert report.timeline
    assert all(any(word in item.headline.lower() for word in ["increased", "dropped", "recovered", "aligned", "settled", "opened", "closing", "emphasis", "pause"]) for item in report.timeline)


def test_initial_benchmark_cannot_say_most_improved_section():
    report = _report("hesitation")

    assert "most improved section" not in " ".join(_strings(report)).lower()


def test_unsupported_claim_evidence_names_actual_claim_content():
    report = _report("thin_proof")
    text = " ".join(_strings(report)).lower()

    assert "communication matters" in text or "communicating my ideas" in text


def test_acoustic_hesitation_is_not_presented_as_confirmed_filler():
    report = _report("hesitation")
    text = " ".join(_strings(report)).lower()

    assert "hesitation" in text
    assert "filler words appeared" not in text
    assert "lexical fillers appeared" not in text


def test_weak_transcript_suppresses_transcript_specific_claims():
    report = _report("thin_proof", weak_transcript=True)
    text = " ".join(_strings(report)).lower()

    assert report.report_mode == "insufficient"
    assert "communicating my ideas" not in text
    assert report.diagnosis is None
    assert report.perception_map is None


def test_weak_timestamps_suppress_exact_proof_moments():
    report = _report("thin_proof", weak_timestamps=True)

    assert all(item.timestamp_source in {"real", "segment"} for item in report.timeline)
    assert all(item.transcript_span is None or item.timestamp_source in {"real", "segment"} for item in report.timeline)


def test_maximum_evidence_and_timeline_limits_are_enforced():
    report = _report("hesitation")

    assert len(report.evidence_chain) <= 3
    assert len(report.timeline) <= 3


def test_optional_perception_reads_are_omitted_when_unsupported():
    report = _report("thin_proof")
    payload = report.perception_map.model_dump()

    assert sum(1 for value in payload.values() if value) <= 3
    assert payload["leadership_read"] is None
    assert payload["emotional_read"] is None


def test_no_forbidden_vague_or_internal_phrases_appear():
    report = _report("weak_close")
    copy = " ".join(_strings(report)).lower()

    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in copy


def test_no_raw_metric_names_appear_in_user_facing_report():
    report = _report("hesitation")
    copy = " ".join(_strings(report))

    assert not re.search(r"\b(raw_acoustic|linguistic|derived|rhythm|vad|[a-z]+_[a-z0-9_]+)\b", copy)


def test_same_normalized_sentence_does_not_appear_in_multiple_sections():
    report = _report("thin_proof")
    normalized = [" ".join(_norm(text)) for text in _strings(report) if text and len(_norm(text)) >= 5]

    assert len(normalized) == len(set(normalized))


def test_thin_proof_diagnosis_selects_proof_focused_drill():
    report = _report("thin_proof")

    assert report.primary_diagnosis.diagnosis_id == "thin_proof"
    assert report.training_prescription.drill_id == "one_point_one_proof_v1"


def test_hesitation_diagnosis_selects_pause_or_control_drill():
    report = _report("hesitation")
    drill = next(item for item in report.coaching_engine.drill_library if item.drill_id == report.training_prescription.drill_id)

    assert report.primary_diagnosis.diagnosis_id == "hesitation_control"
    assert drill.category in {"pause_ownership", "composure", "pace_regulation", "filler_reduction"}


def test_unclear_path_diagnosis_selects_structure_drill():
    report = _report("unclear")
    drill = next(item for item in report.coaching_engine.drill_library if item.drill_id == report.training_prescription.drill_id)

    assert report.primary_diagnosis.diagnosis_id == "unclear_path"
    assert drill.category in {"opening_strength", "structure_compression"}


def test_existing_api_response_fields_remain_compatible():
    report = _report("weak_close")
    payload = report.model_dump()

    for key in [
        "mirror",
        "diagnosis",
        "perception_map",
        "evidence_chain",
        "timeline",
        "dimension_reports",
        "hidden_cost",
        "highest_leverage_fix",
        "training_prescription",
        "retest_plan",
        "technical_appendix",
        "share_card",
        "uncertainty",
    ]:
        assert key in payload


def test_three_realistic_fixtures_produce_different_reports_and_coaching():
    thin = _report("thin_proof")
    hesitation = _report("hesitation")
    close = _report("weak_close")

    assert {thin.primary_diagnosis.diagnosis_id, hesitation.primary_diagnosis.diagnosis_id, close.primary_diagnosis.diagnosis_id} == {
        "thin_proof",
        "hesitation_control",
        "weak_close",
    }
    assert len({thin.training_prescription.drill_id, hesitation.training_prescription.drill_id, close.training_prescription.drill_id}) == 3
    assert len({thin.mirror.headline, hesitation.mirror.headline, close.mirror.headline}) == 3


def test_strength_copy_references_concrete_listener_behaviour():
    report = _report("hesitation")
    strengths = [card for card in report.evidence_chain if card.direction == "positive"]

    assert strengths
    strength_text = " ".join([strengths[0].signal, strengths[0].what_happened, strengths[0].listener_interpretation]).lower()
    assert "repeatable strength" not in strength_text
    assert "listener" in strength_text
    assert any(word in strength_text for word in ["opening", "pace", "example", "ending", "pause", "emphasis", "frame"])


def test_timeline_forms_listener_perception_narrative():
    report = _report("hesitation")

    assert report.timeline
    item = report.timeline[0]
    assert "Expectation:" in item.summary
    assert "Behaviour:" in item.summary
    assert item.listener_interpretation.startswith("Interpretation:")
    assert item.why_it_matters and "Authority impact:" in item.why_it_matters
    assert item.why_it_matters and "Carry-forward:" in item.why_it_matters


def test_evidence_cards_explain_listener_consequence_like_exhibits():
    report = _report("thin_proof")

    primary = report.evidence_chain[0]
    assert primary.why_it_matters.startswith("Because ")
    assert "listener" in primary.why_it_matters.lower()
    assert "exhibit" in primary.why_it_matters.lower()
    assert primary.listener_interpretation


def test_transcript_insertions_are_grammatical_or_suppressed():
    report = _report("weak_close")
    text = " ".join(_strings(report)).lower()

    assert "team moving and" not in text
    assert "you said  " not in text
    assert "and without turning" not in text


def test_context_reads_use_distinct_listener_jobs():
    report = _report("weak_close")
    payload = report.perception_map.model_dump()
    selected = {
        key: value["text"]
        for key, value in payload.items()
        if key in {"professional_read", "leadership_read", "interview_read", "social_status_read"} and value
    }

    assert len(selected) >= 2
    assert any("colleague" in text.lower() for text in selected.values())
    assert any("leadership signal" in text.lower() for text in selected.values())
    texts = list(selected.values())
    for index, text in enumerate(texts):
        assert all(_similarity(text, other) < 0.72 for other in texts[index + 1 :])


def test_mirror_and_diagnosis_answer_different_listener_questions():
    report = _report("thin_proof")

    assert "gets the point early" in report.mirror.identity_read.lower()
    assert "claim arrives before demonstration" in report.diagnosis.core_pattern.lower()
    assert _similarity(report.mirror.identity_read, report.diagnosis.core_pattern) < 0.64


def test_hidden_cost_and_fix_derive_from_primary_listener_state():
    report = _report("hesitation")

    assert "search for words" in report.hidden_cost.consequence.lower()
    assert "pause before the next claim" in report.highest_leverage_fix.plain_english.lower()
    assert _similarity(report.hidden_cost.consequence, report.highest_leverage_fix.plain_english) < 0.7


def test_reports_from_different_recordings_are_measurably_different():
    thin = _report("thin_proof")
    hesitation = _report("hesitation")
    close = _report("weak_close")

    report_bodies = [
        " ".join(_strings(report)).lower()
        for report in (thin, hesitation, close)
    ]
    for index, body in enumerate(report_bodies):
        assert all(_similarity(body, other) < 0.6 for other in report_bodies[index + 1 :])
