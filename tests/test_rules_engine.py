"""Tests for the WHO/MoHFW clinical rules engine (safety-first + config-driven).

Mirrors the Anemia Loop decision table (spec §9).
"""
from app.pipeline.rules_engine import generate_triage_report
from app.schemas import HbBand, PatientMetadata, RiskClass, TriageAction, Urgency


def _p(**kw) -> PatientMetadata:
    return PatientMetadata(**kw)


# ── Rule 1: danger signs ────────────────────────────────────────────────────────
def test_danger_sign_forces_refer_now_even_if_vision_normal():
    d = generate_triage_report(RiskClass.normal, _p(symptoms=["breathlessness"]), p_anemic=0.05)
    assert d.action_code is TriageAction.refer_now
    assert d.urgency is Urgency.high
    assert d.rule_fired == "danger_sign_override"


def test_fainting_is_danger_sign_case_insensitive():
    d = generate_triage_report(RiskClass.normal, _p(symptoms=["Fainting"]), p_anemic=0.0)
    assert d.action_code is TriageAction.refer_now


# ── Rules 2-4: measured Hb overrides the image ─────────────────────────────────
def test_measured_hb_severe_refers_now():
    d = generate_triage_report(RiskClass.normal, _p(age_months=300, measured_hb=6.5), p_anemic=0.0)
    assert d.action_code is TriageAction.refer_now
    assert d.hb_band is HbBand.severe
    assert d.rule_fired == "measured_hb_severe"


def test_measured_hb_moderate_treat_and_recheck():
    d = generate_triage_report(RiskClass.normal, _p(age_months=300, measured_hb=8.0), p_anemic=0.9)
    assert d.action_code is TriageAction.treat_and_recheck
    assert d.hb_band is HbBand.moderate
    assert d.recheck_in_weeks is not None


def test_measured_hb_child_gets_syrup_language():
    d = generate_triage_report(RiskClass.normal, _p(age_months=36, measured_hb=9.5), p_anemic=0.9)
    assert d.action_code is TriageAction.treat_and_recheck
    assert "syrup" in d.action.lower()


def test_measured_hb_normal_routine_overrides_high_image_risk():
    # Hb is normal even though the image screams anemia -> trust the blood test.
    d = generate_triage_report(RiskClass.severe, _p(age_months=300, measured_hb=12.5), p_anemic=0.95)
    assert d.action_code is TriageAction.routine
    assert d.hb_band is HbBand.normal


def test_pregnant_hb_bands_use_pregnant_cutoffs():
    # 10.5 g/dL -> mild for pregnant (normal_min 11.0)
    d = generate_triage_report(RiskClass.normal, _p(is_pregnant=True, age_months=312, measured_hb=10.5), p_anemic=0.0)
    assert d.hb_band is HbBand.mild
    assert d.action_code is TriageAction.treat_and_recheck


# ── Rule 5: high image risk => confirm 48h ─────────────────────────────────────
def test_high_image_risk_confirms_48h():
    d = generate_triage_report(RiskClass.mild_moderate, _p(age_months=300), p_anemic=0.8)
    assert d.action_code is TriageAction.confirm_48h
    assert d.urgency is Urgency.high


def test_severe_vision_without_hb_confirms_48h_not_refer():
    # Image alone never refers; it recommends a confirmatory test.
    d = generate_triage_report(RiskClass.severe, _p(age_months=300), p_anemic=0.85)
    assert d.action_code is TriageAction.confirm_48h


# ── Rule 6: medium risk / >=2 symptoms => confirm 1 week ───────────────────────
def test_medium_image_risk_confirms_1week():
    d = generate_triage_report(RiskClass.mild_moderate, _p(age_months=300), p_anemic=0.4)
    assert d.action_code is TriageAction.confirm_1week


def test_two_symptoms_escalate_even_with_low_risk():
    d = generate_triage_report(RiskClass.normal, _p(age_months=300, symptoms=["fatigue", "dizziness"]), p_anemic=0.1)
    assert d.action_code is TriageAction.confirm_1week


# ── Rule 7: routine ────────────────────────────────────────────────────────────
def test_low_risk_no_symptoms_routine():
    d = generate_triage_report(RiskClass.normal, _p(age_months=300), p_anemic=0.05)
    assert d.action_code is TriageAction.routine
    assert d.urgency is Urgency.low


# ── Pre-eclampsia flag ─────────────────────────────────────────────────────────
def test_preeclampsia_flag_for_pregnant_swelling_plus_headache():
    d = generate_triage_report(RiskClass.normal, _p(is_pregnant=True, age_months=312, symptoms=["swelling", "headache"]), p_anemic=0.1)
    assert "preeclampsia_check" in d.flags


def test_no_preeclampsia_flag_when_not_pregnant():
    d = generate_triage_report(RiskClass.normal, _p(age_months=300, symptoms=["swelling", "headache"]), p_anemic=0.1)
    assert "preeclampsia_check" not in d.flags


# ── Backwards-compatible: accepts string + derives p_anemic from class ─────────
def test_accepts_string_class_and_derives_p_anemic():
    d = generate_triage_report("Severe", _p(age_months=300))
    assert d.action_code is TriageAction.confirm_48h

