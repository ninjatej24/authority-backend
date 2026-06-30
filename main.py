"""Authority Analysis Engine API."""

from __future__ import annotations

import os
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from openai import OpenAI

from services.response_builder import AnalyzeRequest, run_analysis

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
    )

    response = run_analysis(client, request)

    print("\n===== DEBUG =====")
    print("Context:", context)
    print("Title:", title)
    print("Authority Score:", response.scores.authority_score)
    print("Schema:", response.schema_version)
    print("=================\n")

    return response.model_dump()
