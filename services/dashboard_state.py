"""Build deterministic frontend dashboard state from persisted history."""

from __future__ import annotations

from schemas import AuthorityHistory, DashboardState


def build_dashboard_state(history: AuthorityHistory) -> DashboardState:
    """Produce the Today-screen state from stored benchmark history."""
    latest = history.benchmarks[-1] if history.benchmarks else None
    summary = history.history_summary
    training = history.training_history
    journey = history.authority_journey
    retest = history.retest_history
    if not latest:
        return DashboardState()

    drill = latest.snapshot.primary_drill_id or training.current_focus
    momentum = "new_baseline"
    if summary.trend_direction == "improving":
        momentum = "improving"
    elif summary.trend_direction == "declining":
        momentum = "declining"
    elif summary.benchmark_count > 1:
        momentum = "steady"

    mission = drill or "record_next_benchmark"
    growth_signal = None
    if summary.most_improved_dimension:
        growth_signal = f"{summary.most_improved_dimension}_improving"
    elif journey.emerging_strengths:
        growth_signal = f"{journey.emerging_strengths[0]}_strength"

    return DashboardState(
        today_mission=mission,
        highest_leverage_drill=drill,
        active_training_stage=journey.stage,
        momentum=momentum,  # type: ignore[arg-type]
        next_retest={
            "recommended_comparison_benchmark_id": retest.recommended_comparison_benchmark_id,
            "baseline_benchmark_id": retest.baseline_benchmark_id,
            "same_prompt_recommendation": retest.same_prompt_recommendation,
            "comparison_eligible": retest.comparison_eligibility,
        },
        growth_signal=growth_signal,
        practice_recommendation=drill or "complete_next_benchmark",
        training_queue=latest.snapshot.future_drill_ids,
        weekly_summary=history.weekly_summary,
        identity_summary=journey.identity_evolution or latest.snapshot.authority_type,
        authority_snapshot=latest.snapshot,
    )
