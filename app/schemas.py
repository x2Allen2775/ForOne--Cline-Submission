"""
Pydantic data contracts for the Anemia-Screen pipeline.

These models define the public shape of the ``POST /analyze`` request metadata
and the structured triage report returned to the caller. Keeping them explicit
makes the FastAPI Swagger docs self-documenting (great for the demo).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────
class Sex(str, Enum):
    male = "male"
    female = "female"
    other = "other"
    unknown = "unknown"


class RiskClass(str, Enum):
    """Vision model output band (3-class, plan §4)."""

    normal = "Normal"
    mild_moderate = "Mild/Moderate"
    severe = "Severe"


class HbBand(str, Enum):
    """4-level WHO Hb classification used when a measured Hb is available."""

    normal = "normal"
    mild = "mild"
    moderate = "moderate"
    severe = "severe"


class Group(str, Enum):
    """Beneficiary group — selects the WHO cut-off set (Anemia Loop spec §6)."""

    default = "default"
    pregnant = "pregnant"
    child_6_59m = "child_6_59m"


class TriageAction(str, Enum):
    """Actionable triage outcomes (Anemia Loop spec §9)."""

    refer_now = "REFER_NOW"
    confirm_48h = "CONFIRM_48H"
    confirm_1week = "CONFIRM_1WEEK"
    treat_and_recheck = "TREAT_AND_RECHECK"
    routine = "ROUTINE"
    retake_photo = "RETAKE_PHOTO"


class Urgency(str, Enum):
    high = "High"
    medium = "Medium"
    low = "Low"
    none = "None"  # used when the sample is rejected


# ──────────────────────────────────────────────────────────────────────────────
# Request metadata (sent alongside the image as a JSON string)
# ──────────────────────────────────────────────────────────────────────────────
class PatientMetadata(BaseModel):
    """Clinical metadata that accompanies a conjunctiva image."""

    patient_id: Optional[str] = Field(
        default=None, description="Local identifier for the patient / record."
    )
    age_months: int = Field(
        default=0, ge=0, le=1500, description="Patient age in months."
    )
    sex: Sex = Field(default=Sex.unknown, description="Biological sex.")
    is_pregnant: bool = Field(
        default=False, description="Whether the patient is currently pregnant."
    )
    trimester: Optional[int] = Field(
        default=None, ge=1, le=3, description="Pregnancy trimester (1-3) if known."
    )
    symptoms: list[str] = Field(
        default_factory=list,
        description="Reported symptoms, e.g. ['breathlessness', 'fatigue'].",
    )
    measured_hb: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=25.0,
        description="Confirmed hemoglobin (g/dL) from a hemoglobinometer, if known. "
        "When present it overrides the image score for triage.",
    )
    ifa_adherence: Optional[bool] = Field(
        default=None, description="Whether the patient is adhering to IFA supplementation."
    )
    notes: Optional[str] = Field(default=None, description="Free-text field notes.")

    @field_validator("symptoms", mode="before")
    @classmethod
    def _normalize_symptoms(cls, v):
        """Lower-case and strip symptoms so rule matching is robust."""
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        return [str(s).strip().lower() for s in v if str(s).strip()]

    @property
    def age_years(self) -> float:
        return round(self.age_months / 12.0, 1)

    @property
    def group(self) -> "Group":
        """Resolve the beneficiary group used to select WHO Hb cut-offs."""
        if self.is_pregnant:
            return Group.pregnant
        if 6 <= self.age_months <= 59:
            return Group.child_6_59m
        return Group.default


# ──────────────────────────────────────────────────────────────────────────────
# Response sub-structures
# ──────────────────────────────────────────────────────────────────────────────
class QualityReport(BaseModel):
    passed: bool
    blur_variance: float
    brightness_mean: float
    overexposed_fraction: float
    underexposed_fraction: float
    glare_fraction: float = 0.0
    tissue_fraction: float
    reasons: list[str] = Field(default_factory=list)


class VisionPrediction(BaseModel):
    risk_class: RiskClass
    confidence: float = Field(ge=0.0, le=1.0)
    # Probability the patient is anemic (mild/moderate + severe), used by triage.
    p_anemic: float = Field(default=0.0, ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model_trained: bool = Field(
        description="False when running on untrained demo weights."
    )


class TriageDecision(BaseModel):
    action_code: TriageAction = Field(description="Machine-readable triage action.")
    action: str = Field(description="Human-readable instruction for the worker.")
    urgency: Urgency
    rationale: str
    rule_fired: str
    hb_band: Optional[HbBand] = Field(
        default=None, description="WHO Hb band if a measured Hb was provided."
    )
    recheck_in_weeks: Optional[int] = Field(
        default=None, description="Suggested re-screen window (weeks)."
    )
    flags: list[str] = Field(
        default_factory=list,
        description="Additional clinical flags, e.g. 'preeclampsia_check'.",
    )


# Mandatory clinical disclaimer string.
CLINICAL_DISCLAIMER: str = (
    "DISCLAIMER: This screening tool is an AI-assisted risk assessment and is "
    "NOT a medical diagnosis. Non-invasive pallor screening cannot replace "
    "laboratory blood tests. All results must be confirmed with a standard hemoglobin "
    "test (e.g. CBC or hemoglobinometer) before initiating clinical treatment."
)


# ──────────────────────────────────────────────────────────────────────────────
# Top-level response
# ──────────────────────────────────────────────────────────────────────────────
class TriageReport(BaseModel):
    """The structured clinical recommendation returned by ``/analyze``."""

    status: str = Field(description="'ok' or 'rejected'.")
    patient_id: Optional[str] = None
    quality: QualityReport
    vision: Optional[VisionPrediction] = None
    triage: Optional[TriageDecision] = None
    gradcam_png_base64: Optional[str] = Field(
        default=None, description="Base64-encoded Grad-CAM heatmap overlay (PNG)."
    )
    heatmap_png_base64: Optional[str] = Field(
        default=None, description="Base64-encoded pure Grad-CAM heatmap (PNG)."
    )
    processing_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = Field(
        default=CLINICAL_DISCLAIMER,
        description="Mandatory clinical disclaimer indicating this is screening, not diagnosis.",
    )
