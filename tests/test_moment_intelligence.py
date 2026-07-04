"""Milestone 13 deterministic moment intelligence tests."""

from __future__ import annotations

from schemas import AudioQuality, CoachingDrillDefinition, CoachingEngine, ExpectedImprovement, EvidenceItem, InterventionCandidate, SelectedInterventions, TranscriptWord, Uncertainty
from services.acoustic_metrics import WindowFeature
from services.deterministic_coaching import build_deterministic_coaching
from services.moment_intelligence import attach_coaching_relevance, build_moment_intelligence
from services.report_builder import build_report
from tests.test_diagnostic_reasoning import _diagnostic, _softened_expert_scores
from tests.test_psychological_inference import _infer, _metrics
from tests.test_report_builder import _evidence


def _words() -> list[TranscriptWord]:
    tokens = "I believe this plan works because the team can move faster and close with confidence".split()
    words = []
    for index, token in enumerate(tokens):
        start = 1000 + index * 900
        words.append(TranscriptWord(text=token, start_ms=start, end_ms=start + 500, confidence=0.95, is_filler=token == "because"))
    return words


def _windows() -> list[WindowFeature]:
    return [
        WindowFeature(0, 3000, 0.74, 0.76, 0.7, 0.68, 0.02, 142, 3.0, 4.0, 520, False, False, dynamic_emphasis=0.62),
        WindowFeature(3000, 6000, 0.5, 0.48, 0.43, 0.52, 0.16, 196, 2.2, 3.2, 920, False, True, dynamic_emphasis=0.35, hesitation_cluster=True),
        WindowFeature(6000, 9000, 0.68, 0.7, 0.72, 0.74, 0.02, 148, 3.2, 4.8, 480, False, False, dynamic_emphasis=0.82),
        WindowFeature(9000, 12000, 0.57, 0.62, 0.64, 0.44, 0.01, 134, 0.6, 0.8, 360, True, False, dynamic_emphasis=0.15),
        WindowFeature(12000, 15000, 0.78, 0.74, 0.76, 0.7, 0.01, 140, 2.8, 4.2, 610, False, False, dynamic_emphasis=0.66),
    ]


def _bundle(*, scenario: str = "benchmark", audio_quality: AudioQuality | None = None, evidence: list[EvidenceItem] | None = None):
    metrics = _metrics(
        linguistic={
            "opening_strength_score": 0.82,
            "closing_strength_score": 0.42,
            "structure_score": 0.72,
        }
    )
    return build_moment_intelligence(
        words=_words(),
        duration_ms=18000,
        windows=_windows(),
        linguistic=metrics.linguistic,
        evidence=evidence or _evidence(),
        scores=_softened_expert_scores(),
        audio_quality=audio_quality or AudioQuality(usable=True, background_noise_level="low"),
        uncertainty=Uncertainty(overall_confidence_label="medium_high", reasons=[]),
        scenario=scenario,
    )


def _types(bundle):
    return {moment.type for moment in bundle.moments}


def test_detects_core_moments_and_authority_arc():
    bundle = _bundle()
    types = _types(bundle)

    assert "strongest_moment" in types
    assert "weakest_moment" in types
    assert "confidence_drop" in types
    assert "confidence_recovery" in types
    assert "rushing_moment" in types
    assert "hesitation_cluster" in types
    assert "filler_cluster" in types
    assert "pause_ownership_moment" in types
    assert "monotone_stretch" in types
    assert "strong_opening" in types
    assert "weak_closing" in types
    assert bundle.authority_arc.authority_arc is not None
    assert bundle.authority_arc.arc_confidence > 0


def test_best_and_costly_sentence_include_transcript_links():
    bundle = _bundle()
    by_type = {moment.type: moment for moment in bundle.moments}

    assert by_type["best_sentence"].transcript_span
    assert by_type["best_sentence"].word_ids
    assert by_type["most_costly_sentence"].transcript_span
    assert by_type["most_costly_sentence"].word_ids
    assert by_type["most_costly_sentence"].playback_available is True


def test_dimension_evolution_and_prioritisation_are_populated():
    bundle = _bundle()

    assert len(bundle.dimension_evolution) == len(_windows())
    assert all(snapshot.command >= 0 for snapshot in bundle.dimension_evolution)
    assert bundle.top_premium_moments
    assert bundle.top_free_moment in {moment.moment_id for moment in bundle.moments}
    assert all(moment.importance_score > 0 for moment in bundle.moments)


def test_every_moment_links_to_evidence_and_dimensions():
    bundle = _bundle()

    for moment in bundle.moments:
        assert moment.supporting_evidence_ids
        assert moment.supporting_dimension_scores
        assert moment.supporting_metrics
        assert moment.listener_interpretation
        assert moment.why_it_matters


def test_scenario_weighting_changes_moment_relevance_without_changing_metrics():
    benchmark = _bundle(scenario="benchmark")
    leadership = _bundle(scenario="leadership")

    bench_command = next(moment for moment in benchmark.moments if moment.type == "most_commanding_moment")
    leader_command = next(moment for moment in leadership.moments if moment.type == "most_commanding_moment")
    assert leader_command.scenario_relevance >= bench_command.scenario_relevance
    assert bench_command.supporting_metrics == leader_command.supporting_metrics


def test_poor_audio_suppresses_moment_generation():
    bundle = _bundle(audio_quality=AudioQuality(usable=False, quality_warnings=["Very low signal level"]))

    assert bundle.moments == []
    assert bundle.suppressed_moments == ["poor_audio_or_unreliable_windows"]


def test_output_is_deterministic():
    first = _bundle().model_dump()
    second = _bundle().model_dump()

    assert first == second


def test_coaching_relevance_links_selected_drill_to_moments():
    coaching = CoachingEngine(
        drill_library=[
            CoachingDrillDefinition(
                drill_id="pause_ownership_v1",
                title="Pause Ownership",
                category="pause_ownership",
                description="Practise holding the pause before the claim.",
                target_metrics=["window.pause_ms"],
                target_dimensions=["command", "composure"],
                expected_authority_impact=0.8,
                expected_difficulty="beginner",
                estimated_duration_min=4,
                trainability_score=0.9,
            )
        ],
        selected_interventions=SelectedInterventions(
            primary_drill=InterventionCandidate(
                drill_id="pause_ownership_v1",
                title="Pause Ownership",
                score=0.8,
                severity=0.7,
                authority_impact=0.8,
                trainability=0.9,
                confidence=0.75,
                scenario_relevance=1.0,
                required_evidence=["window.pause_ms"],
                supporting_evidence_ids=["ev_command_1"],
                expected_impact=ExpectedImprovement(drill_id="pause_ownership_v1", command=2.0, confidence=0.7),
            )
        ),
    )
    linked = attach_coaching_relevance(_bundle(), coaching)

    assert any(moment.coaching_relevance for moment in linked.moments)


def test_report_timeline_consumes_moment_intelligence():
    metrics = _metrics()
    scores = _softened_expert_scores()
    audio_quality = AudioQuality(usable=True, background_noise_level="low")
    uncertainty = Uncertainty(overall_confidence_label="medium_high", reasons=[])
    inference = _infer(metrics, audio_quality=audio_quality, duration_ms=60000)
    bundle = _bundle()
    diagnostic = _diagnostic(scores=scores, metrics=metrics, audio_quality=audio_quality, uncertainty=uncertainty, duration_ms=60000, scenario="benchmark", inference=inference, evidence=_evidence(), moments=bundle.moments)
    coaching = build_deterministic_coaching(metrics=metrics, scores=scores, psychological_inference=inference, diagnostic_reasoning=diagnostic, report=None, audio_quality=audio_quality, uncertainty=uncertainty, duration_ms=60000, scenario="benchmark")
    bundle = attach_coaching_relevance(bundle, coaching)
    report = build_report(scores=scores, metrics=metrics, psychological_inference=inference, diagnostic_reasoning=diagnostic, coaching_engine=coaching, evidence=_evidence(), moments=bundle.moments, uncertainty=uncertainty, audio_quality=audio_quality, duration_ms=60000, scenario="benchmark", moment_intelligence=bundle)

    assert report.moment_intelligence.authority_arc.authority_arc == bundle.authority_arc.authority_arc
    assert report.timeline[0].supporting_metrics
    assert report.timeline[0].moment_group
    assert report.timeline[0].evidence_ids
