"""
Pipeline Orchestrator.

Wires the stages together in the exact order from the plan's architecture
(plan §1):

    decode -> quality gate -> (reject | preprocess -> inference -> gradcam -> rules)

Returns a fully-populated :class:`TriageReport`. Importing torch-heavy stages is
deferred so lightweight callers (e.g. unit tests for the rules engine) don't pay
the import cost.
"""
from __future__ import annotations

import time

import numpy as np

from app.config import settings
from app.pipeline import preprocessing, quality_gate
from app.pipeline.rules_engine import generate_triage_report
from app.schemas import (
    PatientMetadata,
    TriageAction,
    TriageReport,
    Urgency,
    QualityReport,
    TriageDecision,
)


def _rejected_report(
    patient: PatientMetadata, quality: QualityReport, elapsed_ms: float
) -> TriageReport:
    return TriageReport(
        status="rejected",
        patient_id=patient.patient_id,
        quality=quality,
        vision=None,
        triage=TriageDecision(
            action_code=TriageAction.retake_photo,
            action="RETAKE PHOTO: sample unusable — please recapture the image.",
            urgency=Urgency.none,
            rationale="; ".join(quality.reasons)
            or "Image failed the quality gate.",
            rule_fired="quality_gate_reject",
        ),
        gradcam_png_base64=None,
        processing_ms=round(elapsed_ms, 2),
        warnings=["Quality gate rejected the image; no inference was performed."],
    )


def analyze_image(
    image_bgr: np.ndarray,
    patient: PatientMetadata,
    with_gradcam: bool = True,
) -> TriageReport:
    """
    Run the full pipeline on a decoded BGR image + patient metadata.
    """
    start = time.perf_counter()
    warnings: list[str] = []

    # ── Stage 1: Quality Gate ──
    quality = quality_gate.assess_quality(image_bgr)
    if not quality.passed:
        elapsed = (time.perf_counter() - start) * 1000
        return _rejected_report(patient, quality, elapsed)

    # ── Stage 2: Preprocessing ──
    pre = preprocessing.preprocess(image_bgr)

    # ── Stage 3: Inference (deferred torch import) ──
    from app.pipeline.inference import get_classifier

    clf = get_classifier()
    vision = clf.predict(pre["tensor_chw"])
    if not vision.model_trained:
        warnings.append(
            "Running on UNTRAINED demo weights — predictions are not clinically "
            "valid until a real checkpoint is trained and saved."
        )

    # ── Stage 4: Grad-CAM (explainability) ──
    gradcam_b64 = None
    if with_gradcam:
        from app.pipeline.gradcam import generate_gradcam

        target_idx = settings.class_names.index(vision.risk_class.value)
        gradcam_b64 = generate_gradcam(
            model=clf.model,
            tensor_chw=pre["tensor_chw"],
            base_bgr=pre["normalized_bgr"],
            device=clf.device,
            target_class=target_idx,
        )
        if gradcam_b64 is None:
            warnings.append("Grad-CAM heatmap could not be generated.")

    # ── Stage 5: WHO Rules Engine ──
    triage = generate_triage_report(
        vision.risk_class, patient, p_anemic=vision.p_anemic
    )

    elapsed = (time.perf_counter() - start) * 1000
    return TriageReport(
        status="ok",
        patient_id=patient.patient_id,
        quality=quality,
        vision=vision,
        triage=triage,
        gradcam_png_base64=gradcam_b64,
        processing_ms=round(elapsed, 2),
        warnings=warnings,
    )
