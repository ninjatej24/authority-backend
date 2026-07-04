"""Database-backed persistence adapters for Authority history."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from schemas import AuthorityBenchmark


DEFAULT_SQLITE_PATH = "authority_history.sqlite3"


@dataclass(frozen=True)
class PersistenceConfig:
    backend: str = "sqlite"
    database_url: str = f"sqlite:///{DEFAULT_SQLITE_PATH}"
    enabled: bool = True
    max_history_items: int = 100
    connect_timeout: float = 5.0
    max_payload_bytes: int = 1_500_000


def load_persistence_config() -> PersistenceConfig:
    backend = os.getenv("PERSISTENCE_BACKEND", "sqlite").strip().lower()
    enabled = os.getenv("ENABLE_PERSISTENCE", "true").strip().lower() not in {"0", "false", "no"}
    return PersistenceConfig(
        backend=backend,
        database_url=os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}"),
        enabled=enabled,
        max_history_items=int(os.getenv("MAX_HISTORY_ITEMS", "100")),
        connect_timeout=float(os.getenv("DB_CONNECT_TIMEOUT", "5")),
        max_payload_bytes=int(os.getenv("MAX_PERSISTED_PAYLOAD_BYTES", "1500000")),
    )


def sqlite_path_from_url(database_url: str) -> str:
    if database_url == ":memory:":
        return database_url
    if database_url.startswith("sqlite:///"):
        return database_url.removeprefix("sqlite:///")
    if database_url.startswith("sqlite://"):
        return database_url.removeprefix("sqlite://")
    return database_url


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _loads(value: str | bytes | None, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


class SQLiteAuthorityRepository:
    """Durable SQLite repository storing completed Authority outputs."""

    backend = "sqlite"

    def __init__(self, database_url: str | None = None, *, connect_timeout: float = 5.0, max_payload_bytes: int = 1_500_000) -> None:
        self.database_url = database_url or f"sqlite:///{DEFAULT_SQLITE_PATH}"
        self.path = sqlite_path_from_url(self.database_url)
        self.connect_timeout = connect_timeout
        self.max_payload_bytes = max_payload_bytes
        self._lock = Lock()
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=self.connect_timeout)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self._connect() as conn:
            create_tables(conn)

    def clear(self) -> None:
        with self._lock, self._connect() as conn:
            for table in (
                "authority_analyses",
                "user_profiles",
                "drill_completions",
                "scenario_sessions",
                "retest_events",
                "weekly_summaries",
                "monthly_summaries",
                "analysis_audit_events",
                "pipeline_failures",
            ):
                conn.execute(f"DELETE FROM {table}")

    def persist(self, benchmark: AuthorityBenchmark) -> AuthorityBenchmark:
        payload = benchmark.model_dump()
        full_response_json = _json(payload)
        oversized = len(full_response_json.encode("utf-8")) > self.max_payload_bytes
        if oversized:
            full_response_json = _json(
                {
                    "snapshot": payload.get("snapshot"),
                    "warning": "full_response_omitted_payload_too_large",
                }
            )
        snap = benchmark.snapshot
        report = benchmark.report or {}
        full = benchmark.full_response or {}
        request = full.get("request") or {}
        share_card = benchmark.share_card or {}
        appendix = benchmark.technical_appendix or {}
        with self._lock, self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO user_profiles(user_key, user_id, created_at, latest_analysis_id, benchmark_count)
                    VALUES (?, ?, ?, ?, 1)
                    ON CONFLICT(user_key) DO UPDATE SET
                        latest_analysis_id=excluded.latest_analysis_id,
                        benchmark_count=benchmark_count + 1
                    """,
                    (snap.user_id, snap.user_id, snap.created_at, snap.analysis_id),
                )
                conn.execute(
                    """
                    INSERT INTO authority_analyses(
                        analysis_id, user_key, created_at, scenario, prompt_id, duration_ms,
                        authority_score, dimension_scores_json, authority_type, score_confidence,
                        audio_quality_json, full_response_json, report_json, polished_report_json,
                        coaching_engine_json, progress_json, moment_intelligence_json,
                        pipeline_validation_json, explainability_json, share_card_json,
                        technical_appendix_json, schema_version, oversized_payload
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snap.analysis_id,
                        snap.user_id,
                        snap.created_at,
                        snap.scenario,
                        request.get("prompt_id") or "",
                        request.get("duration_ms") or 0,
                        snap.authority_score,
                        _json(snap.dimension_scores),
                        snap.authority_type,
                        snap.confidence,
                        _json(benchmark.audio_quality),
                        full_response_json,
                        _json(benchmark.report),
                        _json(benchmark.polished_report),
                        _json(benchmark.coaching),
                        _json(benchmark.progress),
                        _json(benchmark.moment_intelligence),
                        _json(benchmark.validation),
                        _json(benchmark.explainability),
                        _json(share_card),
                        _json(appendix),
                        full.get("schema_version", "authority.v2"),
                        1 if oversized else 0,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO scenario_sessions(user_key, scenario, prompt_id, prompt_hash, analysis_id, created_at, completed_at, authority_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (snap.user_id, snap.scenario, "", "", snap.analysis_id, snap.created_at, snap.created_at, snap.authority_score),
                )
                progress = benchmark.progress or {}
                comparison = progress.get("comparison") or {}
                if comparison.get("comparison_target_id"):
                    conn.execute(
                        """
                        INSERT INTO retest_events(user_key, baseline_analysis_id, retest_analysis_id, scenario, prompt_id, created_at, score_delta, dimension_deltas_json, comparison_confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            snap.user_id,
                            comparison.get("comparison_target_id"),
                            snap.analysis_id,
                            snap.scenario,
                            "",
                            snap.created_at,
                            comparison.get("authority_score_delta"),
                            _json(progress.get("dimension_deltas") or {}),
                            (progress.get("confidence") or {}).get("confidence", 0.0),
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                if "authority_analyses.analysis_id" in str(exc) or "UNIQUE" in str(exc):
                    from services.persistence import DuplicateBenchmarkError

                    raise DuplicateBenchmarkError(snap.analysis_id) from exc
                raise
        return benchmark

    def list_benchmarks(self, user_key: str | None, *, limit: int | None = None) -> list[AuthorityBenchmark]:
        if not user_key:
            return []
        sql = "SELECT full_response_json FROM authority_analyses WHERE user_key = ? ORDER BY created_at ASC"
        params: tuple[Any, ...] = (user_key,)
        if limit:
            sql += " LIMIT ?"
            params = (user_key, limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        benchmarks = []
        for row in rows:
            data = _loads(row["full_response_json"], {})
            if "snapshot" in data and "report" in data:
                benchmarks.append(AuthorityBenchmark.model_validate(data))
        return benchmarks

    def get_benchmark(self, analysis_id: str, user_key: str) -> AuthorityBenchmark | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT full_response_json FROM authority_analyses WHERE analysis_id = ? AND user_key = ?",
                (analysis_id, user_key),
            ).fetchone()
        if not row:
            return None
        data = _loads(row["full_response_json"], {})
        return AuthorityBenchmark.model_validate(data) if "snapshot" in data and "report" in data else None

    def record_drill_completion(self, *, user_key: str, drill_id: str, analysis_id: str | None = None, scenario: str | None = None, completed_at: str | None = None, duration_seconds: int | None = None, target_dimensions: list[str] | None = None, linked_moment_ids: list[str] | None = None, quality: float | None = None, confidence: float | None = None) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO drill_completions(user_key, drill_id, analysis_id, scenario, completed_at, duration_seconds, target_dimensions_json, linked_moment_ids_json, quality, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_key, drill_id, analysis_id, scenario, completed_at, duration_seconds, _json(target_dimensions or []), _json(linked_moment_ids or []), quality, confidence),
            )
        return {"drill_completion_id": cursor.lastrowid, "user_key": user_key, "drill_id": drill_id}

    def record_scenario_session(self, *, user_key: str, scenario: str, prompt_id: str | None = None, prompt_hash: str | None = None, analysis_id: str | None = None, created_at: str | None = None, completed_at: str | None = None, authority_score: int | None = None) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scenario_sessions(user_key, scenario, prompt_id, prompt_hash, analysis_id, created_at, completed_at, authority_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_key, scenario, prompt_id, prompt_hash, analysis_id, created_at, completed_at, authority_score),
            )
        return {"scenario_session_id": cursor.lastrowid, "user_key": user_key, "scenario": scenario}

    def record_retest_event(self, *, user_key: str, baseline_analysis_id: str, retest_analysis_id: str, scenario: str | None = None, prompt_id: str | None = None, created_at: str | None = None, score_delta: float | None = None, dimension_deltas: dict[str, Any] | None = None, comparison_confidence: float | None = None) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO retest_events(user_key, baseline_analysis_id, retest_analysis_id, scenario, prompt_id, created_at, score_delta, dimension_deltas_json, comparison_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_key, baseline_analysis_id, retest_analysis_id, scenario, prompt_id, created_at, score_delta, _json(dimension_deltas or {}), comparison_confidence),
            )
        return {"retest_event_id": cursor.lastrowid, "user_key": user_key}


def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_key TEXT PRIMARY KEY,
            user_id TEXT,
            created_at TEXT,
            latest_analysis_id TEXT,
            benchmark_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS authority_analyses (
            analysis_id TEXT PRIMARY KEY,
            user_key TEXT NOT NULL,
            created_at TEXT NOT NULL,
            scenario TEXT,
            prompt_id TEXT,
            duration_ms INTEGER,
            authority_score INTEGER,
            dimension_scores_json TEXT,
            authority_type TEXT,
            score_confidence REAL,
            audio_quality_json TEXT,
            full_response_json TEXT,
            report_json TEXT,
            polished_report_json TEXT,
            coaching_engine_json TEXT,
            progress_json TEXT,
            moment_intelligence_json TEXT,
            pipeline_validation_json TEXT,
            explainability_json TEXT,
            share_card_json TEXT,
            technical_appendix_json TEXT,
            schema_version TEXT,
            oversized_payload INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_authority_user_created ON authority_analyses(user_key, created_at);
        CREATE INDEX IF NOT EXISTS idx_authority_scenario ON authority_analyses(scenario);
        CREATE TABLE IF NOT EXISTS drill_completions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_key TEXT NOT NULL,
            drill_id TEXT NOT NULL,
            analysis_id TEXT,
            scenario TEXT,
            completed_at TEXT,
            duration_seconds INTEGER,
            target_dimensions_json TEXT,
            linked_moment_ids_json TEXT,
            quality REAL,
            confidence REAL
        );
        CREATE TABLE IF NOT EXISTS scenario_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_key TEXT NOT NULL,
            scenario TEXT,
            prompt_id TEXT,
            prompt_hash TEXT,
            analysis_id TEXT,
            created_at TEXT,
            completed_at TEXT,
            authority_score INTEGER
        );
        CREATE TABLE IF NOT EXISTS retest_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_key TEXT NOT NULL,
            baseline_analysis_id TEXT,
            retest_analysis_id TEXT,
            scenario TEXT,
            prompt_id TEXT,
            created_at TEXT,
            score_delta REAL,
            dimension_deltas_json TEXT,
            comparison_confidence REAL
        );
        CREATE TABLE IF NOT EXISTS weekly_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_key TEXT NOT NULL,
            week_start TEXT,
            summary_json TEXT
        );
        CREATE TABLE IF NOT EXISTS monthly_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_key TEXT NOT NULL,
            month_start TEXT,
            summary_json TEXT
        );
        CREATE TABLE IF NOT EXISTS analysis_audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_key TEXT,
            analysis_id TEXT,
            event_type TEXT,
            created_at TEXT,
            metadata_json TEXT
        );
        CREATE TABLE IF NOT EXISTS pipeline_failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_key TEXT,
            analysis_id TEXT,
            failure_type TEXT,
            created_at TEXT,
            metadata_json TEXT
        );
        """
    )


def build_repository(config: PersistenceConfig | None = None):
    cfg = config or load_persistence_config()
    if not cfg.enabled or cfg.backend == "memory":
        from services.persistence import InMemoryAuthorityRepository

        return InMemoryAuthorityRepository()
    if cfg.backend in {"sqlite", "postgres"}:
        return SQLiteAuthorityRepository(cfg.database_url, connect_timeout=cfg.connect_timeout, max_payload_bytes=cfg.max_payload_bytes)
    from services.persistence import InMemoryAuthorityRepository

    return InMemoryAuthorityRepository()
