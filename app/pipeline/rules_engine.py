"""
WHO/MoHFW Clinical Rules Engine — the triage "secret sauce".

Combines the AI vision risk + symptoms + (optional) measured hemoglobin + the
beneficiary group into an *actionable* triage decision, following the **Anemia
Loop** decision table (spec §9) and **Anemia Mukt Bharat** test-and-treat logic.

Design principles (both the hackathon plan §5 and the Anemia Loop spec):
  * SAFETY FIRST — danger signs and severe findings are evaluated before milder
    protocols; the system can never "downgrade" a dangerous presentation.
  * IMAGE IS A SCREEN, NOT A DIAGNOSIS — an image alone never triggers a referral.
    Only a measured Hb < 7 g/dL or a danger sign yields REFER_NOW. A high image
    risk yields a *confirmatory Hb test* recommendation, not treatment.
  * CONFIG-DRIVEN — every threshold comes from ``clinical_config.yaml`` so WHO
    2011 / WHO 2024 / MoHFW updates need no code change.

The engine is pure/deterministic (no I/O beyond the cached config), which makes
it trivially testable.
"""
from __future__ import annotations

from app.clinical import get_clinical_config
from app.schemas import (
    HbBand,
    PatientMetadata,
    RiskClass,
    TriageAction,
    TriageDecision,
    Urgency,
)

# Map the machine-readable action to an urgency level.
_URGENCY = {
    TriageAction.refer_now: Urgency.high,
    TriageAction.confirm_48h: Urgency.high,
    TriageAction.confirm_1week: Urgency.medium,
    TriageAction.treat_and_recheck: Urgency.medium,
    TriageAction.routine: Urgency.low,
    TriageAction.retake_photo: Urgency.none,
}


def _matched_danger_signs(symptoms: list[str]) -> list[str]:
    cfg = get_clinical_config()
    return sorted({s for s in symptoms if s in cfg.danger_signs})


def _preeclampsia_flag(patient: PatientMetadata) -> bool:
    """Pregnant + swelling + (headache or blurred vision) => BP-check flag."""
    if not patient.is_pregnant:
        return False
    cfg = get_clinical_config()
    s = set(patient.symptoms)
    has_swelling = bool(s & cfg.preeclampsia_swelling)
    has_neuro = bool(s & cfg.preeclampsia_neuro)
    return has_swelling and has_neuro


def _p_anemic_from_class(risk: RiskClass) -> float:
    """Fallback p_anemic when the caller passes only a class (e.g. tests)."""
    return {
        RiskClass.normal: 0.1,
        RiskClass.mild_moderate: 0.6,
        RiskClass.severe: 0.9,
    }[risk]



def _decision(
    action_code: TriageAction,
    action: str,
    rationale: str,
    rule_fired: str,
    hb_band: HbBand | None = None,
    recheck_in_weeks: int | None = None,
    flags: list[str] | None = None,
) -> TriageDecision:
    return TriageDecision(
        action_code=action_code,
        action=action,
        urgency=_URGENCY[action_code],
        rationale=rationale,
        rule_fired=rule_fired,
        hb_band=hb_band,
        recheck_in_weeks=recheck_in_weeks,
        flags=flags or [],
    )


def generate_triage_report(
    ai_risk_class: RiskClass | str,
    patient: PatientMetadata,
    p_anemic: float | None = None,
) -> TriageDecision:
    """
    Produce an actionable triage decision following the Anemia Loop decision table
    (spec §9). Rules are evaluated in strict priority order (most urgent first):

      1. Danger sign            -> REFER_NOW
      2. Measured Hb < 7        -> REFER_NOW
      3. Measured Hb mild/mod   -> TREAT_AND_RECHECK   (Hb overrides the image)
      4. Measured Hb normal     -> ROUTINE
      5. High image risk        -> CONFIRM_48H         (confirm with a blood test)
      6. Low image risk / >=2 symptoms -> CONFIRM_1WEEK
      7. Otherwise              -> ROUTINE
    """
    cfg = get_clinical_config()
    risk = ai_risk_class if isinstance(ai_risk_class, RiskClass) else RiskClass(ai_risk_class)
    if p_anemic is None:
        p_anemic = _p_anemic_from_class(risk)
    symptoms = patient.symptoms or []

    extra_flags: list[str] = []
    if _preeclampsia_flag(patient):
        extra_flags.append("preeclampsia_check")

    # ── Rule 1: Danger-sign override (SAFETY FIRST) ────────────────────────────
    danger = _matched_danger_signs(symptoms)
    if danger:
        return _decision(
            TriageAction.refer_now,
            "REFER NOW: same-day referral to a health facility.",
            "Danger sign(s) reported: " + ", ".join(danger) + ". Immediate "
            "evaluation is required independent of the image score.",
            "danger_sign_override",
            flags=extra_flags,
        )

    recheck = cfg.recheck_weeks_min

    # ── Rules 2-4: Measured Hb overrides the image (confirmatory test exists) ───
    if patient.measured_hb is not None:
        band = cfg.cutoffs_for(patient.group).band(patient.measured_hb)
        if band is HbBand.severe:
            return _decision(
                TriageAction.refer_now,
                "REFER NOW: confirmed severe anemia (Hb < 7 g/dL).",
                f"Measured Hb {patient.measured_hb:.1f} g/dL is in the SEVERE band "
                f"for group '{patient.group.value}'. Same-day referral per AMB.",
                "measured_hb_severe",
                hb_band=band,
                flags=extra_flags,
            )
        if band in (HbBand.moderate, HbBand.mild):
            ifa = (
                "pediatric IFA syrup"
                if patient.group.value == "child_6_59m"
                else "daily IFA tablets"
            )
            return _decision(
                TriageAction.treat_and_recheck,
                f"Start {ifa} per MoHFW protocol, counsel on diet, and re-check Hb "
                f"in {cfg.recheck_weeks_min}-{cfg.recheck_weeks_max} weeks.",
                f"Measured Hb {patient.measured_hb:.1f} g/dL is in the {band.value.upper()} "
                "band. Treat-and-recheck under the Anemia Mukt Bharat test-and-treat "
                "flow. The app counsels and reminds only; dosing follows the ANM/MO.",
                "measured_hb_mild_moderate",
                hb_band=band,
                recheck_in_weeks=recheck,
                flags=extra_flags,
            )
        return _decision(
            TriageAction.routine,
            "ROUTINE: normal Hb. Continue prophylactic IFA and re-screen at next visit.",
            f"Measured Hb {patient.measured_hb:.1f} g/dL is NORMAL for group "
            f"'{patient.group.value}'.",
            "measured_hb_normal",
            hb_band=band,
            flags=extra_flags,
        )

    # ── Rule 5: High image risk => confirm with a blood test within 48h ────────
    if p_anemic >= cfg.high_threshold or risk is RiskClass.severe:
        return _decision(
            TriageAction.confirm_48h,
            "Confirm hemoglobin with a blood test within 48 hours; start protocol "
            "only if confirmed.",
            f"High image anemia risk (p={p_anemic:.2f}). An image is a SCREEN, not a "
            "diagnosis — a confirmatory Hb measurement is required before treatment.",
            "image_high_risk_confirm_48h",
            recheck_in_weeks=recheck,
            flags=extra_flags,
        )

    # ── Rule 6: Medium risk OR >=2 symptoms => confirm within a week ────────────
    if p_anemic >= cfg.low_threshold or len(symptoms) >= cfg.symptom_escalation_count:
        return _decision(
            TriageAction.confirm_1week,
            "Confirm hemoglobin with a blood test within one week; counsel on "
            "iron-rich diet meanwhile.",
            f"Borderline image risk (p={p_anemic:.2f}) and/or {len(symptoms)} "
            "symptom(s). Confirmatory testing is advised.",
            "image_medium_risk_confirm_1week",
            recheck_in_weeks=recheck,
            flags=extra_flags,
        )

    # ── Rule 7: Low risk, no symptoms => routine ───────────────────────────────
    return _decision(
        TriageAction.routine,
        "ROUTINE: prophylactic IFA, diet counselling, and re-screen at next visit.",
        f"Low image anemia risk (p={p_anemic:.2f}) with no danger signs. Continue "
        "preventive supplementation per Anemia Mukt Bharat.",
        "routine_low_risk",
        flags=extra_flags,
    )

