"""
FastAPI application — the pipeline ingestion endpoint (plan §1-§2).

Run locally:
    uvicorn app.main:app --reload

Then open the interactive Swagger docs at:
    http://127.0.0.1:8000/docs

Endpoints:
    GET  /health      -> liveness + model status
    POST /analyze     -> multipart image + JSON metadata -> TriageReport
"""
from __future__ import annotations

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from app import __version__
from app.config import settings
from app.pipeline import preprocessing
from app.pipeline.orchestrator import analyze_image
from app.schemas import PatientMetadata, TriageReport

app = FastAPI(
    title="Anemia-Screen Pipeline",
    version=__version__,
    description=(
        "AI-powered conjunctiva anemia screening & triage pipeline.\n\n"
        "Upload a conjunctiva (inner-eyelid) image plus patient metadata; the "
        "pipeline runs a quality gate, MobileNetV3 inference, Grad-CAM "
        "explainability, and a WHO clinical rules engine to return an actionable "
        "triage report."
    ),
)

# Permissive CORS so a notebook / local dashboard can call the API during demos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
def health() -> dict:
    """Liveness probe + model status (trained vs. demo weights)."""
    # Import lazily so /health works even if torch is mid-install.
    try:
        from app.pipeline.inference import get_classifier

        clf = get_classifier()
        model_info = {
            "arch": settings.model_arch,
            "device": str(clf.device),
            "trained": clf.trained,
            "classes": list(settings.class_names),
        }
    except Exception as exc:  # pragma: no cover
        model_info = {"error": f"model not ready: {exc}"}

    return {"status": "ok", "version": __version__, "model": model_info}


@app.post("/analyze", response_model=TriageReport, tags=["pipeline"])
async def analyze(
    image: UploadFile = File(..., description="Conjunctiva image (jpg/png)."),
    metadata: str = Form(
        "{}",
        description='Patient metadata as a JSON string, e.g. '
        '{"age_months": 300, "is_pregnant": true, "symptoms": ["fatigue"]}',
    ),
    with_gradcam: bool = Form(True, description="Include Grad-CAM heatmap."),
) -> TriageReport:
    """
    Analyze a single conjunctiva image and return a structured triage report.
    """
    # Parse + validate metadata.
    try:
        meta_dict = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid metadata JSON: {exc}")
    try:
        patient = PatientMetadata(**meta_dict)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json()))

    # Read + decode the image.
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Empty image upload.")
    try:
        bgr = preprocessing.decode_image(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return analyze_image(bgr, patient, with_gradcam=with_gradcam)


@app.get("/", tags=["system"])
def root() -> dict:
    return {
        "name": "Anemia-Screen Pipeline",
        "docs": "/docs",
        "health": "/health",
        "analyze": "POST /analyze (multipart: image + metadata)",
    }
