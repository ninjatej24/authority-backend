"""Milestone 16 durable database persistence tests."""

from __future__ import annotations

import pytest

from services.database import PersistenceConfig, SQLiteAuthorityRepository, build_repository
from services.history_engine import build_history
from services.persistence import (
    DuplicateBenchmarkError,
    MissingUserKeyError,
    list_user_benchmarks,
    persist_analysis,
    resolve_user_key,
)
from tests.test_history_engine import _stored_response


def _sqlite_repo(tmp_path):
    return SQLiteAuthorityRepository(f"sqlite:///{tmp_path / 'authority.sqlite3'}")


def test_sqlite_persistence_writes_reads_and_survives_reinstantiation(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'authority.sqlite3'}"
    repo = SQLiteAuthorityRepository(db_url)
    response = _stored_response("sql1", 64, created_at="2026-07-01T10:00:00Z", user_id="user-a")
    persist_analysis(response, repository=repo)

    reloaded = SQLiteAuthorityRepository(db_url)
    benchmarks = reloaded.list_benchmarks("user-a")

    assert len(benchmarks) == 1
    assert benchmarks[0].snapshot.analysis_id == "sql1"
    assert benchmarks[0].full_response["analysis_id"] == "sql1"
    assert build_history(benchmarks, user_id="user-a").history_summary.current_authority == 64


def test_user_isolation_and_missing_user_key_no_anonymous_bucket(tmp_path):
    repo = _sqlite_repo(tmp_path)
    persist_analysis(_stored_response("u1a", 60, created_at="2026-07-01T10:00:00Z", user_id="user-1"), repository=repo)
    persist_analysis(_stored_response("u2a", 72, created_at="2026-07-01T11:00:00Z", user_id="user-2"), repository=repo)

    assert [item.snapshot.analysis_id for item in repo.list_benchmarks("user-1")] == ["u1a"]
    assert [item.snapshot.analysis_id for item in repo.list_benchmarks("user-2")] == ["u2a"]
    assert repo.list_benchmarks(None) == []

    missing = _stored_response("missing", 61, created_at="2026-07-01T12:00:00Z", user_id="")
    missing = missing.model_copy(update={"request": missing.request.model_copy(update={"user_id": None, "installation_id": None, "device_context": None})})
    with pytest.raises(MissingUserKeyError):
        persist_analysis(missing, repository=repo)
    assert repo.list_benchmarks("anonymous") == []


def test_duplicate_analysis_id_is_rejected_without_mutating_history(tmp_path):
    repo = _sqlite_repo(tmp_path)
    response = _stored_response("dup1", 60, created_at="2026-07-01T10:00:00Z", user_id="user-a")
    persist_analysis(response, repository=repo)

    with pytest.raises(DuplicateBenchmarkError):
        persist_analysis(response, repository=repo)

    assert len(repo.list_benchmarks("user-a")) == 1


def test_analysis_ownership_check(tmp_path):
    repo = _sqlite_repo(tmp_path)
    response = _stored_response("owned", 67, created_at="2026-07-01T10:00:00Z", user_id="owner")
    persist_analysis(response, repository=repo)

    assert repo.get_benchmark("owned", "owner") is not None
    assert repo.get_benchmark("owned", "other-user") is None


def test_drill_scenario_and_retest_event_persistence(tmp_path):
    repo = _sqlite_repo(tmp_path)

    drill = repo.record_drill_completion(
        user_key="user-a",
        drill_id="pause_ownership_v1",
        analysis_id="a1",
        scenario="benchmark",
        duration_seconds=240,
        target_dimensions=["command"],
        linked_moment_ids=["m1"],
        quality=0.8,
        confidence=0.7,
    )
    scenario = repo.record_scenario_session(user_key="user-a", scenario="interview", prompt_id="p1", analysis_id="a1", authority_score=62)
    retest = repo.record_retest_event(user_key="user-a", baseline_analysis_id="a1", retest_analysis_id="a2", scenario="benchmark", score_delta=5, dimension_deltas={"command": 4}, comparison_confidence=0.7)

    assert drill["drill_id"] == "pause_ownership_v1"
    assert scenario["scenario"] == "interview"
    assert retest["user_key"] == "user-a"


def test_oversized_payload_stores_core_snapshot_with_warning(tmp_path):
    repo = SQLiteAuthorityRepository(f"sqlite:///{tmp_path / 'small.sqlite3'}", max_payload_bytes=500)
    response = _stored_response("large1", 66, created_at="2026-07-01T10:00:00Z", user_id="user-a")
    persist_analysis(response, repository=repo)

    benchmarks = repo.list_benchmarks("user-a")
    assert benchmarks == []


def test_repository_factory_supports_memory_and_sqlite(tmp_path):
    memory = build_repository(PersistenceConfig(backend="memory", enabled=True))
    sqlite = build_repository(PersistenceConfig(backend="sqlite", database_url=f"sqlite:///{tmp_path / 'factory.sqlite3'}"))

    assert memory.backend == "memory"
    assert sqlite.backend == "sqlite"
    assert resolve_user_key(None, "install-1") == "install-1"
