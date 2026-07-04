"""Deterministic longitudinal history aggregation for Authority."""

from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean, pstdev

from schemas import (
    AuthorityBenchmark,
    AuthorityHistory,
    AuthorityJourney,
    DrillCompletion,
    LongitudinalSummary,
    MonthlyHistorySummary,
    RetestHistory,
    ScenarioHistory,
    TrainingHistory,
    UserProfile,
    WeeklyHistorySummary,
)
from services.persistence import validate_history_integrity


DIMENSIONS = ("command", "clarity", "composure", "presence", "persuasion", "structure")
SCENARIOS = ("benchmark", "interview", "sales", "leadership", "meeting", "presentation", "podcast", "founder_pitch")


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def _ordered(benchmarks: list[AuthorityBenchmark]) -> list[AuthorityBenchmark]:
    unique: dict[str, AuthorityBenchmark] = {}
    for item in benchmarks:
        unique[item.snapshot.analysis_id] = item
    return sorted(unique.values(), key=lambda item: _parse_time(item.snapshot.created_at))


def _days_since(value: str | None) -> int | None:
    parsed = _parse_time(value)
    if parsed == datetime.min.replace(tzinfo=timezone.utc):
        return None
    return max(0, (datetime.now(timezone.utc) - parsed).days)


def _window_average(items: list[AuthorityBenchmark], days: int) -> float | None:
    if not items:
        return None
    latest = _parse_time(items[-1].snapshot.created_at)
    selected = [
        item.snapshot.authority_score
        for item in items
        if (latest - _parse_time(item.snapshot.created_at)).days <= days
    ]
    return round(mean(selected), 2) if selected else None


def _rolling_dimension_averages(items: list[AuthorityBenchmark]) -> dict[str, float]:
    result = {}
    for dimension in DIMENSIONS:
        values = [item.snapshot.dimension_scores.get(dimension) for item in items if item.snapshot.dimension_scores.get(dimension) is not None]
        if values:
            result[dimension] = round(mean(values), 2)
    return result


def _dimension_change(items: list[AuthorityBenchmark]) -> dict[str, float]:
    if len(items) < 2:
        return {}
    first = items[0].snapshot.dimension_scores
    last = items[-1].snapshot.dimension_scores
    return {
        dimension: round((last.get(dimension, 0) - first.get(dimension, 0)), 2)
        for dimension in DIMENSIONS
        if dimension in first and dimension in last
    }


def _trend(scores: list[int]) -> str:
    if len(scores) < 2:
        return "insufficient_history"
    delta = scores[-1] - scores[0]
    if delta >= 4:
        return "improving"
    if delta <= -4:
        return "declining"
    return "stable"


def _summary(items: list[AuthorityBenchmark], training: TrainingHistory) -> LongitudinalSummary:
    if not items:
        return LongitudinalSummary()
    scores = [item.snapshot.authority_score for item in items]
    dimension_change = _dimension_change(items)
    most_improved = max(dimension_change, key=dimension_change.get) if dimension_change else None
    most_stagnant = min(dimension_change, key=lambda key: abs(dimension_change[key])) if dimension_change else None
    volatility = round(min(1.0, (pstdev(scores) / 25 if len(scores) > 1 else 0.0)), 2)
    consistency = round(1.0 - volatility, 2)
    scenarios = [item.snapshot.scenario for item in items]
    favourite = max(set(scenarios), key=scenarios.count) if scenarios else None
    return LongitudinalSummary(
        current_authority=scores[-1],
        previous_authority=scores[-2] if len(scores) > 1 else None,
        best_authority=max(scores),
        rolling_average=round(mean(scores[-5:]), 2),
        rolling_30_day_average=_window_average(items, 30),
        rolling_90_day_average=_window_average(items, 90),
        rolling_dimension_averages=_rolling_dimension_averages(items),
        highest_ever_command=max((item.snapshot.dimension_scores.get("command", 0) for item in items), default=None),
        highest_ever_presence=max((item.snapshot.dimension_scores.get("presence", 0) for item in items), default=None),
        most_improved_dimension=most_improved,
        most_stagnant_dimension=most_stagnant,
        consistency=consistency,
        volatility=volatility,
        trend_direction=_trend(scores),  # type: ignore[arg-type]
        training_adherence=round(len(training.completed_drills) / max(1, len(training.started_drills)), 2),
        benchmark_count=len(items),
        drill_count=len(training.started_drills),
        scenario_count=len(set(scenarios)),
        practice_streak=training.completion_streak,
        days_since_benchmark=_days_since(items[-1].snapshot.created_at),
        days_since_drill=_days_since(training.last_completed_drill and next((item.completed_at for item in reversed(training.completed_drills) if item.drill_id == training.last_completed_drill), None)),
        days_since_practice=_days_since(training.completed_drills[-1].completed_at if training.completed_drills else None),
        last_scenario=items[-1].snapshot.scenario,
        favourite_scenario=favourite,
    )


def _training_history(items: list[AuthorityBenchmark]) -> TrainingHistory:
    started: list[DrillCompletion] = []
    for item in items:
        drill_id = item.snapshot.primary_drill_id
        if drill_id:
            started.append(
                DrillCompletion(
                    drill_id=drill_id,
                    status="started",
                    completed_at=item.snapshot.created_at,
                    duration_min=0,
                    source_analysis_id=item.snapshot.analysis_id,
                )
            )
    completed = [
        drill.model_copy(update={"status": "completed"})
        for drill in started[:-1]
    ]
    last_completed = completed[-1].drill_id if completed else None
    current = started[-1].drill_id if started else None
    progression = list(dict.fromkeys(drill.drill_id for drill in started))
    dependencies = {drill_id: drill_id in {item.drill_id for item in completed} for drill_id in progression}
    return TrainingHistory(
        completed_drills=completed,
        started_drills=started,
        abandoned_drills=[],
        practice_duration_min=sum(item.duration_min for item in completed),
        last_completed_drill=last_completed,
        completion_streak=len(completed),
        current_focus=current,
        training_progression=progression,
        dependency_completion=dependencies,
    )


def _scenario_history(items: list[AuthorityBenchmark]) -> ScenarioHistory:
    counts = {scenario: 0 for scenario in SCENARIOS}
    scores_by_scenario: dict[str, list[int]] = {scenario: [] for scenario in SCENARIOS}
    confidence_by_scenario: dict[str, list[float]] = {scenario: [] for scenario in SCENARIOS}
    for item in items:
        scenario = item.snapshot.scenario
        counts[scenario] = counts.get(scenario, 0) + 1
        scores_by_scenario.setdefault(scenario, []).append(item.snapshot.authority_score)
        confidence_by_scenario.setdefault(scenario, []).append(item.snapshot.confidence)
    used = {scenario: count for scenario, count in counts.items() if count > 0}
    best_scores = {scenario: max(values) for scenario, values in scores_by_scenario.items() if values}
    improvement = {
        scenario: round(values[-1] - values[0], 2)
        for scenario, values in scores_by_scenario.items()
        if len(values) >= 2
    }
    averages = {scenario: mean(values) for scenario, values in scores_by_scenario.items() if values}
    return ScenarioHistory(
        scenario_counts=used,
        scenario_best_scores=best_scores,
        scenario_improvement=improvement,
        best_scenario=max(averages, key=averages.get) if averages else None,
        weakest_scenario=min(averages, key=averages.get) if averages else None,
        most_practised=max(used, key=used.get) if used else None,
        least_practised=min(used, key=used.get) if used else None,
        scenario_confidence={scenario: round(mean(values), 2) for scenario, values in confidence_by_scenario.items() if values},
    )


def _retest_history(items: list[AuthorityBenchmark]) -> RetestHistory:
    if not items:
        return RetestHistory()
    current = items[-1]
    same_scenario = [item for item in items[:-1] if item.snapshot.scenario == current.snapshot.scenario]
    best = max(same_scenario, key=lambda item: item.snapshot.authority_score) if same_scenario else None
    recommended = same_scenario[-1] if same_scenario else None
    exclusions = []
    if len(items) > 1 and not same_scenario:
        exclusions.append("cross_scenario_restriction")
    return RetestHistory(
        baseline_benchmark_id=items[0].snapshot.analysis_id,
        best_comparison_benchmark_id=best.snapshot.analysis_id if best else None,
        recommended_comparison_benchmark_id=recommended.snapshot.analysis_id if recommended else None,
        comparison_confidence=round(min(current.snapshot.confidence, recommended.snapshot.confidence), 2) if recommended else 0.0,
        comparison_eligibility=recommended is not None,
        comparison_exclusions=exclusions,
        cross_scenario_restriction=True,
        same_prompt_recommendation=True,
    )


def _weekly_summary(items: list[AuthorityBenchmark], training: TrainingHistory) -> WeeklyHistorySummary:
    if not items:
        return WeeklyHistorySummary()
    latest = _parse_time(items[-1].snapshot.created_at)
    week = [item for item in items if (latest - _parse_time(item.snapshot.created_at)).days <= 7]
    improvement = week[-1].snapshot.authority_score - week[0].snapshot.authority_score if len(week) >= 2 else 0.0
    return WeeklyHistorySummary(
        weekly_improvement=max(0.0, improvement),
        weekly_regression=abs(min(0.0, improvement)),
        weekly_consistency=round(1.0 - min(1.0, (pstdev([item.snapshot.authority_score for item in week]) / 25 if len(week) > 1 else 0.0)), 2),
        new_milestone="first_benchmark" if len(items) == 1 else "highest_authority_score" if items[-1].snapshot.authority_score >= max(item.snapshot.authority_score for item in items) else None,
        recommended_focus=items[-1].snapshot.primary_drill_id,
        drill_completion_summary={"completed": len(training.completed_drills), "started": len(training.started_drills), "abandoned": len(training.abandoned_drills)},
        practice_summary=f"{len(training.completed_drills)} completed drills from stored benchmarks.",
    )


def _monthly_summary(items: list[AuthorityBenchmark]) -> MonthlyHistorySummary:
    if not items:
        return MonthlyHistorySummary()
    latest = _parse_time(items[-1].snapshot.created_at)
    month = [item for item in items if (latest - _parse_time(item.snapshot.created_at)).days <= 30]
    improvement = month[-1].snapshot.authority_score - month[0].snapshot.authority_score if len(month) >= 2 else 0.0
    return MonthlyHistorySummary(
        monthly_improvement=improvement,
        monthly_consistency=round(1.0 - min(1.0, (pstdev([item.snapshot.authority_score for item in month]) / 25 if len(month) > 1 else 0.0)), 2),
        best_monthly_score=max(item.snapshot.authority_score for item in month),
        recommended_focus=items[-1].snapshot.primary_drill_id,
        benchmark_count=len(month),
    )


def _journey(items: list[AuthorityBenchmark], summary: LongitudinalSummary) -> AuthorityJourney:
    if not items:
        return AuthorityJourney()
    current = items[-1]
    dims = current.snapshot.dimension_scores
    low_dimension = min(dims, key=dims.get) if dims else None
    high_dimensions = sorted(dims, key=dims.get, reverse=True)[:2]
    count = len(items)
    score = current.snapshot.authority_score
    stability = summary.consistency
    velocity = 0.0 if summary.previous_authority is None else score - summary.previous_authority
    if count == 1:
        stage = "Beginning"
    elif stability < 0.68:
        stage = "Building Consistency"
    elif low_dimension == "command":
        stage = "Finding Command"
    elif low_dimension == "presence":
        stage = "Developing Presence"
    elif score >= 78 and stability >= 0.75:
        stage = "Established Authority"
    else:
        stage = "Authority Emerging"
    scores = [item.snapshot.authority_score for item in items[-4:]]
    plateau = len(scores) >= 4 and max(scores) - min(scores) <= 2
    moment_types = [
        moment.get("type")
        for item in items
        for moment in item.moment_intelligence.get("moments", [])
        if isinstance(moment, dict)
    ]
    collapse = next((moment for moment in ("confidence_drop", "rushing_moment", "weak_closing") if moment_types.count(moment) >= 2), None)
    strength = next((moment for moment in ("strongest_moment", "pause_ownership_moment", "strong_closing") if moment_types.count(moment) >= 2), None)
    return AuthorityJourney(
        stage=stage,  # type: ignore[arg-type]
        authority_journey=f"{stage}: {count} stored benchmark{'s' if count != 1 else ''}.",
        identity_evolution=current.snapshot.authority_type,
        communication_trajectory=summary.trend_direction,
        emerging_strengths=high_dimensions,
        persistent_weaknesses=[low_dimension] if low_dimension else [],
        behaviour_stabilisation=stability,
        recurring_collapse_pattern=collapse,
        recurring_strength_pattern=strength,
        improvement_velocity=round(velocity, 2),
        plateau_detected=plateau,
    )


def build_history(benchmarks: list[AuthorityBenchmark], *, user_id: str | None = None) -> AuthorityHistory:
    """Build deterministic longitudinal aggregates from stored benchmarks."""
    ordered = _ordered(benchmarks)
    training = _training_history(ordered)
    summary = _summary(ordered, training)
    scenario_history = _scenario_history(ordered)
    weekly = _weekly_summary(ordered, training)
    monthly = _monthly_summary(ordered)
    journey = _journey(ordered, summary)
    profile = UserProfile(
        user_id=user_id,
        created_at=ordered[0].snapshot.created_at if ordered else None,
        latest_analysis_id=ordered[-1].snapshot.analysis_id if ordered else None,
        benchmark_count=len(ordered),
    )
    return AuthorityHistory(
        user_profile=profile,
        benchmarks=ordered,
        history_summary=summary,
        training_history=training,
        scenario_history=scenario_history,
        retest_history=_retest_history(ordered),
        weekly_summary=weekly,
        monthly_summary=monthly,
        authority_journey=journey,
        validation=validate_history_integrity(ordered),
    )
