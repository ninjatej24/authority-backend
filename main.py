"""Authority Analysis Engine API."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from openai import OpenAI

from services.dashboard_state import build_dashboard_state
from services.history_engine import build_history
from services.persistence import DEFAULT_REPOSITORY, list_user_benchmarks, resolve_user_key
from services.response_builder import AnalyzeRequest, run_analysis

load_dotenv()


def _get_client() -> OpenAI:
    """Lazy-create OpenAI client only when needed."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


app = FastAPI()


@app.post("/analyze")
async def analyze_voice(
    file: UploadFile = File(...),
    context: str = Form("initial"),
    title: str = Form("Speech Analysis"),
    prompt: str = Form(""),
    drill_id: str | None = Form(None),
    module_slug: str | None = Form(None),
    skill: str | None = Form(None),
    user_id: str | None = Form(None),
    installation_id: str | None = Form(None),
    x_user_id: str | None = Header(None),
    x_installation_id: str | None = Header(None),
):
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".m4a"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    request = AnalyzeRequest(
        file_path=tmp_path,
        original_suffix=suffix,
        context=context,
        title=title,
        prompt=prompt,
        drill_id=drill_id,
        module_slug=module_slug,
        skill=skill,
        user_id=user_id or x_user_id,
        installation_id=installation_id or x_installation_id,
    )

    response = run_analysis(_get_client(), request)
    return response.model_dump()


def _stable_key(
    user_id: str | None = None,
    installation_id: str | None = None,
    x_user_id: str | None = None,
    x_installation_id: str | None = None,
) -> str | None:
    return resolve_user_key(user_id or x_user_id, installation_id or x_installation_id)


def _history_payload(user_key: str | None, limit: int | None = None) -> dict[str, Any]:
    if not user_key:
        history = build_history([], user_id=None)
        return {
            "history": history.model_dump(),
            "recent_benchmarks": [],
            "warnings": ["History unavailable because no stable user_id or installation_id was provided"],
        }
    try:
        benchmarks = list_user_benchmarks(user_key, repository=DEFAULT_REPOSITORY, limit=limit)
        history = build_history(benchmarks, user_id=user_key)
        return {
            "history": history.model_dump(),
            "recent_benchmarks": [item.snapshot.model_dump() for item in benchmarks[-(limit or 10):]],
            "warnings": [],
        }
    except Exception:
        history = build_history([], user_id=user_key)
        return {
            "history": history.model_dump(),
            "recent_benchmarks": [],
            "warnings": ["History unavailable because the persistence backend could not be read"],
        }


@app.get("/history")
async def get_history(
    user_id: str | None = Query(None),
    installation_id: str | None = Query(None),
    x_user_id: str | None = Header(None),
    x_installation_id: str | None = Header(None),
    limit: int = Query(20),
):
    return _history_payload(_stable_key(user_id, installation_id, x_user_id, x_installation_id), limit)


@app.get("/dashboard-state")
async def get_dashboard_state(
    user_id: str | None = Query(None),
    installation_id: str | None = Query(None),
    x_user_id: str | None = Header(None),
    x_installation_id: str | None = Header(None),
):
    payload = _history_payload(_stable_key(user_id, installation_id, x_user_id, x_installation_id), None)
    history = build_history(
        list_user_benchmarks(_stable_key(user_id, installation_id, x_user_id, x_installation_id), repository=DEFAULT_REPOSITORY)
        if _stable_key(user_id, installation_id, x_user_id, x_installation_id)
        else [],
        user_id=_stable_key(user_id, installation_id, x_user_id, x_installation_id),
    )
    return {
        "dashboard_state": build_dashboard_state(history).model_dump(),
        "history_summary": history.history_summary.model_dump(),
        "warnings": payload["warnings"],
    }


@app.get("/analysis/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    user_id: str | None = Query(None),
    installation_id: str | None = Query(None),
    x_user_id: str | None = Header(None),
    x_installation_id: str | None = Header(None),
):
    key = _stable_key(user_id, installation_id, x_user_id, x_installation_id)
    if not key:
        raise HTTPException(status_code=400, detail="stable user_id or installation_id required")
    benchmark = DEFAULT_REPOSITORY.get_benchmark(analysis_id, key)
    if not benchmark:
        raise HTTPException(status_code=404, detail="analysis not found for user")
    return benchmark.full_response or benchmark.model_dump()


@app.get("/progress-history")
async def get_progress_history(
    user_id: str | None = Query(None),
    installation_id: str | None = Query(None),
    x_user_id: str | None = Header(None),
    x_installation_id: str | None = Header(None),
):
    key = _stable_key(user_id, installation_id, x_user_id, x_installation_id)
    payload = _history_payload(key, None)
    history = build_history(list_user_benchmarks(key, repository=DEFAULT_REPOSITORY) if key else [], user_id=key)
    return {
        "history_summary": history.history_summary.model_dump(),
        "authority_journey": history.authority_journey.model_dump(),
        "retest_history": history.retest_history.model_dump(),
        "weekly_summary": history.weekly_summary.model_dump(),
        "monthly_summary": history.monthly_summary.model_dump(),
        "warnings": payload["warnings"],
    }


@app.post("/drills/complete")
async def complete_drill(
    payload: dict[str, Any] = Body(default_factory=dict),
    user_id: str | None = Query(None),
    installation_id: str | None = Query(None),
    x_user_id: str | None = Header(None),
    x_installation_id: str | None = Header(None),
):
    key = _stable_key(user_id or payload.get("user_id"), installation_id or payload.get("installation_id"), x_user_id, x_installation_id)
    if not key:
        raise HTTPException(status_code=400, detail="stable user_id or installation_id required")
    drill_id = payload.get("drill_id")
    if not drill_id:
        raise HTTPException(status_code=400, detail="drill_id required")
    result = DEFAULT_REPOSITORY.record_drill_completion(
        user_key=key,
        drill_id=drill_id,
        analysis_id=payload.get("analysis_id"),
        scenario=payload.get("scenario"),
        completed_at=payload.get("completed_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        duration_seconds=payload.get("duration_seconds"),
        target_dimensions=payload.get("target_dimensions") or [],
        linked_moment_ids=payload.get("linked_moment_ids") or [],
        quality=payload.get("quality"),
        confidence=payload.get("confidence"),
    )
    return {"stored": True, **result}
