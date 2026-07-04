"""Deterministic persistence boundary for completed Authority analyses."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Iterable, Protocol

from schemas import AuthorityBenchmark, AuthoritySnapshot, AuthorityV2Response, PersistenceStatus


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


class MissingUserKeyError(ValueError):
    """Raised when no stable user or installation key is available."""


def resolve_user_key(user_id: str | None = None, installation_id: str | None = None, device_context: str | None = None) -> str | None:
    """Resolve a stable persistence key without creating a shared anonymous bucket."""
    for value in (user_id, installation_id, device_context):
        normalized = (value or "").strip()
        if normalized:
            return normalized
    return None


class DuplicateBenchmarkError(ValueError):
    """Raised when a benchmark analysis_id is already stored for a user."""


class AuthorityRepository(Protocol):
    backend: str

    def list_benchmarks(self, user_key: str | None, *, limit: int | None = None) -> list[AuthorityBenchmark]: ...

    def persist(self, benchmark: AuthorityBenchmark) -> AuthorityBenchmark: ...


class InMemoryAuthorityRepository:
    """Simple process-local repository used until a real database is wired in."""

    backend = "memory"

    def __init__(self) -> None:
        self._lock = Lock()
        self._benchmarks: dict[str, list[AuthorityBenchmark]] = {}

    def clear(self) -> None:
        with self._lock:
            self._benchmarks.clear()

    def list_benchmarks(self, user_key_value: str | None, *, limit: int | None = None) -> list[AuthorityBenchmark]:
        if not user_key_value:
            return []
        with self._lock:
            items = deepcopy(self._benchmarks.get(user_key_value, []))
        ordered = sorted(items, key=lambda item: _parse_time(item.snapshot.created_at))
        return ordered[-limit:] if limit else ordered

    def persist(self, benchmark: AuthorityBenchmark) -> AuthorityBenchmark:
        key = benchmark.snapshot.user_id
        if not key:
            raise MissingUserKeyError("missing_stable_user_key")
        with self._lock:
            existing = self._benchmarks.setdefault(key, [])
            if any(item.snapshot.analysis_id == benchmark.snapshot.analysis_id for item in existing):
                raise DuplicateBenchmarkError(benchmark.snapshot.analysis_id)
            existing.append(deepcopy(benchmark))
        return benchmark


class _DefaultRepositoryProxy:
    def __init__(self) -> None:
        self._repo = None

    @property
    def backend(self) -> str:
        return getattr(self._get(), "backend", "unknown")

    def _get(self):
        if self._repo is None:
            try:
                from services.database import build_repository

                self._repo = build_repository()
            except Exception:
                self._repo = InMemoryAuthorityRepository()
        return self._repo

    def clear(self) -> None:
        repo = self._get()
        if hasattr(repo, "clear"):
            repo.clear()

    def persist(self, benchmark: AuthorityBenchmark) -> AuthorityBenchmark:
        return self._get().persist(benchmark)

    def list_benchmarks(self, user_key_value: str | None, *, limit: int | None = None) -> list[AuthorityBenchmark]:
        return self._get().list_benchmarks(user_key_value, limit=limit)

    def get_benchmark(self, analysis_id: str, user_key_value: str):
        repo = self._get()
        return repo.get_benchmark(analysis_id, user_key_value) if hasattr(repo, "get_benchmark") else next((item for item in repo.list_benchmarks(user_key_value) if item.snapshot.analysis_id == analysis_id), None)

    def record_drill_completion(self, **kwargs):
        repo = self._get()
        if hasattr(repo, "record_drill_completion"):
            return repo.record_drill_completion(**kwargs)
        return {"user_key": kwargs.get("user_key"), "drill_id": kwargs.get("drill_id")}

    def record_scenario_session(self, **kwargs):
        repo = self._get()
        if hasattr(repo, "record_scenario_session"):
            return repo.record_scenario_session(**kwargs)
        return {"user_key": kwargs.get("user_key"), "scenario": kwargs.get("scenario")}

    def record_retest_event(self, **kwargs):
        repo = self._get()
        if hasattr(repo, "record_retest_event"):
            return repo.record_retest_event(**kwargs)
        return {"user_key": kwargs.get("user_key")}


DEFAULT_REPOSITORY = _DefaultRepositoryProxy()


def snapshot_from_authority_response(response: AuthorityV2Response) -> AuthoritySnapshot:
    key = resolve_user_key(
        response.request.user_id,
        response.request.installation_id,
        response.request.device_context,
    )
    if not key:
        raise MissingUserKeyError("missing_stable_user_key")
    primary = response.coaching_engine.selected_interventions.primary_drill
    queue = response.coaching_engine.future_training_queue
    report_type = response.report.authority_type.label if response.report.authority_type else None
    report_drill = (
        response.report.training_prescription.drill_id
        if response.report.training_prescription
        else None
    )
    return AuthoritySnapshot(
        analysis_id=response.analysis_id,
        user_id=key,
        created_at=response.created_at,
        scenario=response.request.scenario,
        authority_score=response.scores.authority_score,
        dimension_scores=response.scores.dimension_scores.model_dump(),
        derived_axes=response.scores.derived_axes.model_dump(),
        authority_type=report_type,
        confidence=response.scores.score_confidence or 0.0,
        audio_usable=response.audio_quality.usable,
        primary_drill_id=primary.drill_id if primary else report_drill,
        future_drill_ids=[item.drill_id for item in queue],
        evidence_ids=[item.id for item in response.evidence],
        moment_ids=[item.moment_id for item in response.moments],
    )


def benchmark_from_response(response: AuthorityV2Response) -> AuthorityBenchmark:
    report = response.report
    return AuthorityBenchmark(
        snapshot=snapshot_from_authority_response(response),
        full_response=response.model_dump(),
        report=report.model_dump(),
        progress=response.progress.model_dump(),
        coaching=response.coaching_engine.model_dump(),
        moment_intelligence=response.moment_intelligence.model_dump(),
        explainability=response.explainability.model_dump(),
        timeline=[item.model_dump() for item in report.timeline],
        share_card=report.share_card.model_dump() if report.share_card else {},
        validation=response.pipeline_validation.model_dump(),
        polished_report=response.polished_report.model_dump(),
        audio_quality=response.audio_quality.model_dump(),
        technical_appendix=report.technical_appendix.model_dump() if report.technical_appendix else {},
    )


def persist_analysis(
    response: AuthorityV2Response,
    *,
    repository: AuthorityRepository | None = None,
) -> AuthorityBenchmark:
    """Persist a completed benchmark exactly as produced by the pipeline."""
    repo = repository or DEFAULT_REPOSITORY
    benchmark = benchmark_from_response(response)
    return repo.persist(benchmark)


def list_user_benchmarks(
    user_id: str | None,
    installation_id: str | None = None,
    device_context: str | None = None,
    *,
    repository: AuthorityRepository | None = None,
    limit: int | None = None,
) -> list[AuthorityBenchmark]:
    repo = repository or DEFAULT_REPOSITORY
    key = resolve_user_key(user_id, installation_id, device_context)
    return repo.list_benchmarks(key, limit=limit) if key else []


def persistence_status_for_missing_key() -> PersistenceStatus:
    return PersistenceStatus(
        persisted=False,
        user_key_present=False,
        warnings=["History was not saved because no stable user_id or installation_id was provided"],
        audit_events=["missing_user_key"],
    )


def validate_history_integrity(benchmarks: Iterable[AuthorityBenchmark]) -> dict:
    ordered = list(benchmarks)
    ids = [item.snapshot.analysis_id for item in ordered]
    duplicate_ids = sorted({analysis_id for analysis_id in ids if ids.count(analysis_id) > 1})
    timestamps = [_parse_time(item.snapshot.created_at) for item in ordered]
    future = datetime.now(timezone.utc)
    future_ids = [
        item.snapshot.analysis_id
        for item, parsed in zip(ordered, timestamps)
        if parsed > future
    ]
    ordering_valid = timestamps == sorted(timestamps)
    missing = [
        item.snapshot.analysis_id
        for item in ordered
        if not item.report or not item.coaching or not item.validation
    ]
    return {
        "valid": not duplicate_ids and not future_ids and ordering_valid and not missing,
        "duplicate_benchmark_ids": duplicate_ids,
        "future_benchmark_ids": future_ids,
        "history_ordering_valid": ordering_valid,
        "missing_payload_ids": missing,
        "benchmark_count": len(ordered),
    }
