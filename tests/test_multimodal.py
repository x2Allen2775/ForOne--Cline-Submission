"""Unit tests for multimodal screening, NLP symptom analysis, and Grad-CAM fixes."""
from pathlib import Path
import numpy as np
import pytest

from app.clinical_nlp import analyze_symptom_text
from app.pipeline.rules_engine import generate_multimodal_triage_decision
from app.schemas import PatientMetadata, RiskClass, QualityReport, VisionPrediction, TriageDecision, TriageAction, Urgency, TriageReport, CLINICAL_DISCLAIMER


def _dummy_quality(passed=True):
    return QualityReport(
        passed=passed,
        blur_variance=100.0,
        brightness_mean=120.0,
        overexposed_fraction=0.01,
        underexposed_fraction=0.01,
        glare_fraction=0.01,
        tissue_fraction=0.5,
        reasons=[] if passed else ["too blurry"],
    )


def _dummy_vision(risk_class=RiskClass.mild_moderate, p_anemic=0.65):
    return VisionPrediction(
        risk_class=risk_class,
        confidence=0.85,
        p_anemic=p_anemic,
        probabilities={"Normal": 0.1, "Mild/Moderate": 0.65, "Severe": 0.25},
        model_trained=True,
    )


def _dummy_report(risk_class=RiskClass.mild_moderate, p_anemic=0.65, passed=True):
    q = _dummy_quality(passed)
    v = _dummy_vision(risk_class, p_anemic) if passed else None
    return TriageReport(
        status="ok" if passed else "rejected",
        quality=q,
        vision=v,
        triage=TriageDecision(
            action_code=TriageAction.confirm_48h if passed else TriageAction.retake_photo,
            action="Test action",
            urgency=Urgency.high if passed else Urgency.none,
            rationale="Test rationale",
            rule_fired="test_rule",
        ),
    )


# ── 1. NLP Symptom Analyzer Tests ──
def test_nlp_extracts_danger_signs():
    res = analyze_symptom_text("Patient feels dizzy and has severe shortness of breath with fainting")
    assert "breathlessness" in res["matched_symptoms"]
    assert "fainting" in res["matched_symptoms"]
    assert res["has_danger_signs"] is True
    assert res["clinical_urgency"] == "High"


def test_nlp_handles_negations():
    res = analyze_symptom_text("Patient reports fatigue and dizziness, denies chest pain and no breathlessness")
    assert "fatigue" in res["matched_symptoms"]
    assert "dizziness" in res["matched_symptoms"]
    assert "chest pain" not in res["matched_symptoms"]
    assert "breathlessness" not in res["matched_symptoms"]
    assert res["has_danger_signs"] is False


def test_nlp_detects_pregnancy_and_hb():
    res = analyze_symptom_text("2nd trimester pregnant female, reports tiredness, hb 9.1")
    assert res["detected_pregnant"] is True
    assert res["detected_trimester"] == 2
    assert res["detected_hb"] == 9.1
    assert "fatigue" in res["matched_symptoms"]


def test_nlp_blank_or_none():
    res = analyze_symptom_text("none")
    assert res["matched_symptoms"] == []
    assert res["clinical_urgency"] == "Low"


# ── 2. Multimodal Synthesis (Eyelid Primary Factor) ──
def test_multimodal_eyelid_primary_weight():
    # Eyelid high risk (p=0.8), palm normal (p=0.1)
    eye_rep = _dummy_report(risk_class=RiskClass.severe, p_anemic=0.8)
    palm_rep = _dummy_report(risk_class=RiskClass.normal, p_anemic=0.1)
    patient = PatientMetadata(age_months=300)

    decision, synth = generate_multimodal_triage_decision(eye_rep, palm_rep, patient)
    # 0.70 * 0.8 + 0.30 * 0.1 = 0.56 + 0.03 = 0.59
    assert synth["primary_site"] == "eyelid"
    assert synth["weights"]["eyelid"] == 0.70
    assert synth["fused_risk"] == "Severe"
    assert decision.action_code in (TriageAction.confirm_48h, TriageAction.confirm_1week)


def test_multimodal_eyelid_rejected_fallback_to_palm():
    eye_rep = _dummy_report(passed=False)
    palm_rep = _dummy_report(risk_class=RiskClass.severe, p_anemic=0.85, passed=True)
    patient = PatientMetadata(age_months=300)

    decision, synth = generate_multimodal_triage_decision(eye_rep, palm_rep, patient)
    assert synth["mode"] == "palm_fallback"
    assert "eyelid_quality_failed_fallback_palm" in decision.flags


# ── 3. Disclaimer Verification ──
def test_disclaimer_present():
    rep = _dummy_report()
    assert CLINICAL_DISCLAIMER in rep.disclaimer
    assert "NOT a medical diagnosis" in rep.disclaimer
