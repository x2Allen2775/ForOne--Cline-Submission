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
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app import __version__
from app.config import settings
from app.pipeline import preprocessing
from app.pipeline.orchestrator import analyze_image
from app.schemas import PatientMetadata, TriageReport

STATIC_DIR = Path(__file__).resolve().parent / "static"

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
    # Disable the default CDN-backed docs; we serve an offline version below.
    docs_url=None,
    redoc_url=None,
)

# Serve the self-contained web UI assets (no internet/CDN required).
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/docs", include_in_schema=False)
def swagger_docs():
    """Swagger UI served from LOCAL assets (works offline, no CDN)."""
    from fastapi.openapi.docs import get_swagger_ui_html

    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} · API docs",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
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

        info = {}
        for modality in ("eye", "palm"):
            clf = get_classifier(modality=modality)
            info[modality] = {
                "arch": settings.model_arch,
                "device": str(clf.device),
                "trained": clf.trained,
                "classes": list(settings.class_names),
            }
        model_info = info
    except Exception as exc:  # pragma: no cover
        model_info = {"error": f"model not ready: {exc}"}

    return {"status": "ok", "version": __version__, "models": model_info}


@app.post("/analyze", response_model=TriageReport, tags=["pipeline"])
async def analyze(
    image: UploadFile = File(..., description="Conjunctiva or palm/nail image (jpg/png/heic)."),
    metadata: str = Form(
        "{}",
        description='Patient metadata as a JSON string, e.g. '
        '{"age_months": 300, "is_pregnant": true, "symptoms": ["fatigue"]}',
    ),
    with_gradcam: bool = Form(True, description="Include Grad-CAM heatmap."),
    modality: str = Form(
        "eye",
        description="Screening modality: 'eye' (conjunctiva, default) or 'palm' (palm/nail bed).",
    ),
) -> TriageReport:
    """
    Analyze a single image and return a structured triage report.

    ``modality`` selects the model + preprocessing: 'eye' = conjunctiva
    (inner lower eyelid), 'palm' = palm/nail-bed photo.
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

    if modality not in ("eye", "palm"):
        raise HTTPException(status_code=422, detail="modality must be 'eye' or 'palm'.")

    # Read + decode the image.
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Empty image upload.")
    try:
        bgr = preprocessing.decode_image(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return analyze_image(bgr, patient, with_gradcam=with_gradcam, modality=modality)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Serve the self-contained web UI (works fully offline, no CDN)."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api", tags=["system"])
def api_info() -> dict:
    """Machine-readable API index."""
    return {
        "name": "Anemia-Screen Pipeline",
        "ui": "/",
        "docs": "/docs",
        "health": "/health",
        "analyze": "POST /analyze (multipart: image + metadata)",
    }
