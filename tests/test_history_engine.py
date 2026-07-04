"""Milestone 15 persistence and longitudinal history tests."""

from __future__ import annotations

import pytest

from services.history_engine import build_history
from services.persistence import (
    DuplicateBenchmarkError,
    InMemoryAuthorityRepository,
    benchmark_from_response,
    persist_analysis,
)
from tests.test_pipeline_validator import _response


def _stored_response(analysis_id: str, score: int, *, created_at: str, scenario: str = "benchmark", user_id: str = "u1", command: int | None = None, presence: int | None = None):
    response = _response(scenario=scenario).model_copy(update={"analysis_id": analysis_id, "created_at": created_at})
    request = response.request.model_copy(update={"user_id": user_id, "scenario": scenario})
    dims = response.scores.dimension_scores.model_copy(
        update={
            "command": command if command is not None else score,
            "clarity": score,
            "composure": score,
            "presence": presence if presence is not None else score,
            "persuasion": score,
            "structure": score,
        }
    )
    scores = response.scores.model_copy(update={"authority_score": score, "dimension_scores": dims})
    return response.model_copy(update={"request": request, "scores": scores})


def test_first_benchmark_is_persistable_without_recalculation():
    repo = InMemoryAuthorityRepository()
    response = _stored_response("a1", 60, created_at="2026-07-01T10:00:00Z")
    stored = persist_analysis(response, repository=repo)
    history = build_history(repo.list_benchmarks("u1"), user_id="u1")

    assert stored.snapshot.authority_score == response.scores.authority_score
    assert stored.report["share_card"] == response.report.share_card.model_dump()
    assert history.history_summary.benchmark_count == 1
    assert history.authority_journey.stage == "Beginning"
    assert history.validation["valid"] is True


def test_duplicate_rejection_and_history_ordering():
    repo = InMemoryAuthorityRepository()
    second = _stored_response("a2", 66, created_at="2026-07-02T10:00:00Z")
    first = _stored_response("a1", 60, created_at="2026-07-01T10:00:00Z")
    persist_analysis(second, repository=repo)
    persist_analysis(first, repository=repo)

    with pytest.raises(DuplicateBenchmarkError):
        persist_analysis(first, repository=repo)

    ordered = repo.list_benchmarks("u1")
    assert [item.snapshot.analysis_id for item in ordered] == ["a1", "a2"]


def test_second_and_tenth_benchmark_aggregation_trends_and_averages():
    repo = InMemoryAuthorityRepository()
    for index in range(10):
        persist_analysis(
            _stored_response(
                f"a{index + 1}",
                50 + index * 3,
                created_at=f"2026-07-{index + 1:02d}T10:00:00Z",
                command=48 + index * 4,
                presence=45 + index * 2,
            ),
            repository=repo,
        )

    history = build_history(repo.list_benchmarks("u1"), user_id="u1")

    assert history.history_summary.benchmark_count == 10
    assert history.history_summary.current_authority == 77
    assert history.history_summary.previous_authority == 74
    assert history.history_summary.best_authority == 77
    assert history.history_summary.rolling_average == 71
    assert history.history_summary.rolling_30_day_average is not None
    assert history.history_summary.highest_ever_command == 84
    assert history.history_summary.most_improved_dimension == "command"
    assert history.history_summary.trend_direction == "improving"
    assert history.authority_journey.improvement_velocity == 3


def test_identity_evolution_plateau_and_persistent_weakness():
    repo = InMemoryAuthorityRepository()
    for index, score in enumerate([70, 71, 70, 71], start=1):
        persist_analysis(
            _stored_response(f"p{index}", score, created_at=f"2026-07-{index:02d}T10:00:00Z", command=55, presence=72),
            repository=repo,
        )
    history = build_history(repo.list_benchmarks("u1"), user_id="u1")

    assert history.authority_journey.plateau_detected is True
    assert "command" in history.authority_journey.persistent_weaknesses
    assert history.authority_journey.stage in {"Finding Command", "Building Consistency", "Authority Emerging"}


def test_training_scenario_retest_weekly_and_monthly_history():
    repo = InMemoryAuthorityRepository()
    persist_analysis(_stored_response("b1", 58, created_at="2026-07-01T10:00:00Z", scenario="interview"), repository=repo)
    persist_analysis(_stored_response("b2", 65, created_at="2026-07-04T10:00:00Z", scenario="interview"), repository=repo)
    persist_analysis(_stored_response("b3", 62, created_at="2026-07-03T10:00:00Z", scenario="sales"), repository=repo)
    history = build_history(repo.list_benchmarks("u1"), user_id="u1")

    assert history.training_history.started_drills
    assert history.training_history.completion_streak >= 1
    assert history.scenario_history.scenario_counts["interview"] == 2
    assert history.scenario_history.scenario_improvement["interview"] == 7
    assert history.retest_history.baseline_benchmark_id == "b1"
    assert history.weekly_summary.drill_completion_summary["started"] >= 1
    assert history.monthly_summary.benchmark_count == 3


def test_history_output_is_deterministic_and_schema_validates():
    repo = InMemoryAuthorityRepository()
    responses = [
        _stored_response("d1", 61, created_at="2026-07-01T10:00:00Z"),
        _stored_response("d2", 64, created_at="2026-07-02T10:00:00Z"),
    ]
    for response in responses:
        persist_analysis(response, repository=repo)
    first = build_history(repo.list_benchmarks("u1"), user_id="u1").model_dump()
    second = build_history(repo.list_benchmarks("u1"), user_id="u1").model_dump()

    assert first == second
    assert benchmark_from_response(responses[0]).snapshot.analysis_id == "d1"
