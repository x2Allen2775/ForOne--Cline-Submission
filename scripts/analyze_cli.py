"""
Terminal & Interactive Dual-Modality Analyzer for Anemia-Screen.

Runs the FULL multimodal screening pipeline:
  1. Opens a folder/file selection UI to pick Eyelid (primary) and Palm (secondary) images.
  2. Prompts in terminal for patient symptoms (free text).
  3. Analyzes symptoms using clinical NLP (extracts danger signs, fatigue, pallor, etc.).
  4. Runs quality gate, MobileNetV3 inference, and Grad-CAM explainability on both sites.
  5. Synthesizes an integrated WHO triage decision with Eyelids as the PRIMARY factor.
  6. Prints a complete color-coded summary ending with the mandatory Clinical Disclaimer.
  7. Displays a 3-image panel for both Eyelid and Palm along with Grad-CAM heatmaps.
  8. Saves all report figures and standalone heatmaps to disk.

Usage:
    # Interactive intake (opens UI selector + terminal symptom prompt)
    python run_pipeline.py
    python -m scripts.analyze_cli

    # Provide folder directly
    python -m scripts.analyze_cli --folder data/sample_patient

    # Provide individual files
    python -m scripts.analyze_cli --eye data/sample/pale_pregnant.png --palm "Palm/Anemic-260 (10).png"

    # Single-image legacy mode (backward compatible)
    python -m scripts.analyze_cli --image data/sample/pale_pregnant.png --modality eye
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.clinical_nlp import analyze_symptom_text
from app.config import settings
from app.pipeline.rules_engine import generate_multimodal_triage_decision
from app.schemas import CLINICAL_DISCLAIMER, PatientMetadata, RiskClass, TriageDecision, TriageReport


# ── ANSI colours (auto-disabled when output is not a TTY) ────────────────────────
class C:
    _tty = sys.stdout.isatty()

    @staticmethod
    def _w(code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if C._tty else str(s)

    @staticmethod
    def bold(s):   return C._w("1", s)
    @staticmethod
    def dim(s):    return C._w("2", s)
    @staticmethod
    def red(s):    return C._w("91", s)
    @staticmethod
    def green(s):  return C._w("92", s)
    @staticmethod
    def yellow(s): return C._w("93", s)
    @staticmethod
    def cyan(s):   return C._w("96", s)
    @staticmethod
    def magenta(s): return C._w("95", s)
    @staticmethod
    def grey(s):   return C._w("90", s)


URGENCY_COLOR = {
    "High": C.red,
    "Medium": C.yellow,
    "Low": C.green,
    "None": C.grey,
}


def _bar(fraction: float, width: int = 10) -> str:
    """Return a small unicode bar for a 0..1 fraction."""
    fraction = max(0.0, min(1.0, float(fraction)))
    filled = int(round(fraction * width))
    return "▇" * filled + "▁" * (width - filled)


def _decode_b64_bgr(b64: str) -> np.ndarray:
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def load_metadata(image_path: Path, args) -> dict:
    """Build metadata from CLI flags, falling back to a sidecar <image>.json."""
    meta: dict = {}
    sidecar = image_path.with_suffix(".json")
    if sidecar.exists() and not getattr(args, "ignore_sidecar", False):
        try:
            meta = json.loads(sidecar.read_text())
            print(C.dim(f"[cli] loaded metadata from {sidecar.name}"))
        except json.JSONDecodeError:
            print(C.yellow(f"[cli] WARN: bad sidecar JSON {sidecar.name}; ignoring."))

    if getattr(args, "age_months", None) is not None:
        meta["age_months"] = args.age_months
    if getattr(args, "sex", None) is not None:
        meta["sex"] = args.sex
    if getattr(args, "pregnant", False):
        meta["is_pregnant"] = True
    if getattr(args, "trimester", None) is not None:
        meta["trimester"] = args.trimester
    if getattr(args, "symptoms", None) is not None:
        meta["symptoms"] = [s.strip() for s in args.symptoms.split(",") if s.strip()]
    if getattr(args, "hb", None) is not None:
        meta["measured_hb"] = args.hb
    return meta


# ──────────────────────────────────────────────────────────────────────────────
# Terminal Reports
# ──────────────────────────────────────────────────────────────────────────────
def print_disclaimer_block() -> None:
    """Print the standardized clinical disclaimer at the end of the report."""
    line = "═" * 63
    print(C.yellow(line))
    print(C.yellow(C.bold(" ⚠ MANDATORY CLINICAL DISCLAIMER:")))
    print(C.dim(
        "   This screening tool is an AI-assisted risk assessment and is NOT a medical\n"
        "   diagnosis. Non-invasive pallor analysis cannot replace laboratory blood tests.\n"
        "   All screening results MUST be confirmed with a standard hemoglobin test\n"
        "   (e.g., CBC or certified hemoglobinometer) before initiating clinical treatment."
    ))
    print(C.yellow(line) + "\n")


def print_summary(image_path: Path, img_shape, report: TriageReport, patient: PatientMetadata, modality: str = "eye") -> None:
    """Print single-image triage summary to the terminal."""
    rd = report.model_dump(mode="json")
    q = rd["quality"]
    v = rd.get("vision")
    t = rd.get("triage")
    blur_min = (
        settings.palm_blur_min_variance if modality == "palm"
        else settings.blur_min_variance
    )

    line = "═" * 63
    print("\n" + C.cyan(line))
    print(C.cyan(C.bold(f"        ANEMIA-SCREEN · TRIAGE REPORT ({modality.upper()})")))
    print(C.cyan(line))

    h, w = img_shape[:2]
    print(f" {C.bold('Input image')} : {image_path}  ({w}×{h} px)")
    preg = "yes" if patient.is_pregnant else "no"
    tri = f"(T{patient.trimester})" if patient.trimester else ""
    print(
        f" {C.bold('Patient')}     : age {patient.age_months} mo "
        f"({patient.age_years} yr) · {patient.sex.value} · pregnant={preg}{tri} "
        f"· group={patient.group.value}"
    )
    print(f" {C.bold('Symptoms')}    : {', '.join(patient.symptoms) or '(none)'}")
    hb_txt = f"{patient.measured_hb} g/dL" if patient.measured_hb is not None else "(not provided)"
    print(f" {C.bold('Measured Hb')} : {hb_txt}")

    # Quality gate
    passed = q["passed"]
    status = C.green("PASS ✓") if passed else C.red("REJECT ✗")
    print("\n " + C.bold("── QUALITY GATE ") + "─" * 26 + f"  {status}")
    print(f"   blur (Laplacian var) : {q['blur_variance']:>8.1f}   (min {blur_min:.0f})")
    print(f"   brightness mean      : {q['brightness_mean']:>8.1f}   (window {settings.brightness_min:.0f}–{settings.brightness_max:.0f})")
    print(f"   over / under exposed : {q['overexposed_fraction']*100:>5.1f}% / {q['underexposed_fraction']*100:.1f}%")
    print(f"   glare fraction       : {q['glare_fraction']*100:>7.1f}%   (max {settings.glare_fraction_max*100:.0f}%)")
    print(f"   tissue fraction      : {q['tissue_fraction']*100:>7.1f}%   (min {settings.min_tissue_fraction*100:.0f}%)")
    if q["reasons"]:
        for r in q["reasons"]:
            print("   " + C.red("• " + r))

    # Vision model
    if v:
        print("\n " + C.bold("── VISION MODEL (MobileNetV3) ") + "─" * 12)
        print(f"   risk class   : {C.bold(v['risk_class'])}   (confidence {v['confidence']*100:.1f}%)")
        print(f"   P(anemic)    : {v['p_anemic']:.3f}")
        print("   probabilities:")
        for name, p in v["probabilities"].items():
            print(f"     {name:<14s} {_bar(p)}  {p*100:5.1f}%")

    # Triage
    if t:
        uc = URGENCY_COLOR.get(t["urgency"], C.grey)
        print("\n " + C.bold("── WHO/MoHFW TRIAGE ") + "─" * 22)
        print(f"   {C.bold('ACTION')}   : {uc(C.bold(t['action_code']))}   [urgency: {uc(t['urgency'].upper())}]")
        print(f"   detail   : {t['action']}")
        if t.get("hb_band"):
            print(f"   Hb band  : {t['hb_band']}")
        if t.get("recheck_in_weeks"):
            print(f"   recheck  : in {t['recheck_in_weeks']} weeks")
        print(f"   flags    : {', '.join(t['flags']) or '(none)'}")
        print(f"   rule     : {C.dim(t['rule_fired'])}")
        print(f"   rationale: {C.dim(t['rationale'])}")

    for wmsg in rd.get("warnings", []):
        print("\n " + C.yellow("⚠ " + wmsg))
    print(f"\n {C.dim('processing: %.0f ms' % rd.get('processing_ms', 0))}")

    # Mandatory disclaimer at the end
    print_disclaimer_block()


def print_multimodal_summary(
    eye_path: Path,
    palm_path: Path,
    eye_report: TriageReport,
    palm_report: TriageReport,
    final_decision: TriageDecision,
    synthesis: dict[str, Any],
    patient: PatientMetadata,
    nlp_results: dict[str, Any],
) -> None:
    """Print the complete color-coded multimodal triage summary."""
    line = "═" * 63
    print("\n" + C.cyan(line))
    print(C.cyan(C.bold("    ANEMIA-SCREEN · MULTIMODAL CLINICAL TRIAGE REPORT")))
    print(C.cyan(line))

    preg = "yes" if patient.is_pregnant else "no"
    tri = f"(T{patient.trimester})" if patient.trimester else ""
    print(
        f" {C.bold('Patient')}          : age {patient.age_months} mo "
        f"({patient.age_years} yr) · {patient.sex.value} · pregnant={preg}{tri} "
        f"· group={patient.group.value}"
    )
    print(f" {C.bold('Eyelid Image')}     : {eye_path.name} {C.magenta('[PRIMARY SCREENING FACTOR - 70%]')}")
    print(f" {C.bold('Palm Image')}       : {palm_path.name} {C.dim('[SECONDARY SUPPORTIVE FACTOR - 30%]')}")
    hb_txt = f"{patient.measured_hb} g/dL" if patient.measured_hb is not None else "(not provided)"
    print(f" {C.bold('Measured Hb')}      : {hb_txt}")

    # NLP Symptom intake
    print("\n " + C.bold("── SYMPTOM ANALYSIS (Clinical NLP) ") + "─" * 15)
    raw_s = nlp_results.get("raw_text") or "(none entered)"
    print(f"   input text      : {C.dim(raw_s)}")
    matched = nlp_results.get("matched_symptoms", [])
    print(f"   matched symptoms: {C.bold(', '.join(matched)) if matched else '(none)'}")
    danger = nlp_results.get("danger_signs", [])
    if danger:
        print("   " + C.red(C.bold(f"🚨 DANGER SIGNS DETECTED: {', '.join(danger)} (Forces immediate referral)")))
    else:
        print(f"   danger signs    : {C.green('none detected')}")
    if nlp_results.get("preeclampsia_risk"):
        print("   " + C.yellow("⚠ Pre-eclampsia warning: Swelling + neurological symptoms in pregnant patient"))
    print(f"   NLP summary     : {nlp_results.get('summary')}")

    # Eyelid assessment (Primary)
    print("\n " + C.bold("── 1. EYELID CONJUNCTIVA ") + C.magenta(C.bold("[PRIMARY FACTOR]")) + " ─" * 12)
    eq = eye_report.quality
    ev = eye_report.vision
    estatus = C.green("PASS ✓") if eq.passed else C.red("REJECT ✗")
    print(f"   quality gate    : {estatus} (blur {eq.blur_variance:.1f}, tissue {eq.tissue_fraction*100:.1f}%)")
    if ev:
        print(f"   vision risk     : {C.bold(ev.risk_class.value)} (confidence {ev.confidence*100:.1f}%, P(anemia)={ev.p_anemic:.3f})")
        for name, p in ev.probabilities.items():
            print(f"     {name:<14s} {_bar(p)} {p*100:5.1f}%")
    else:
        print("   " + C.red("vision inference unavailable (image rejected)"))

    # Palm assessment (Secondary)
    print("\n " + C.bold("── 2. PALM / NAIL BED ") + C.dim("[SECONDARY SUPPORTIVE]") + " ─" * 13)
    pq = palm_report.quality
    pv = palm_report.vision
    pstatus = C.green("PASS ✓") if pq.passed else C.red("REJECT ✗")
    print(f"   quality gate    : {pstatus} (blur {pq.blur_variance:.1f})")
    if pv:
        print(f"   vision risk     : {C.bold(pv.risk_class.value)} (confidence {pv.confidence*100:.1f}%, P(anemia)={pv.p_anemic:.3f})")
        for name, p in pv.probabilities.items():
            print(f"     {name:<14s} {_bar(p)} {p*100:5.1f}%")
    else:
        print("   " + C.yellow("vision inference unavailable (image rejected)"))

    # Multimodal Synthesis
    print("\n " + C.bold("── MULTIMODAL SYNTHESIS ") + "─" * 20)
    fused_p = synthesis.get("fused_p_anemic", 0.0)
    fused_risk = synthesis.get("fused_risk", "Unknown")
    concordance = synthesis.get("concordance", "n/a")
    print(f"   integrated risk : {C.bold(fused_risk)} (fused P(anemia)={fused_p:.3f})")
    print(f"   primary factor  : {C.magenta('Eyelid Conjunctival Pallor (70% weight)')}")
    print(f"   site agreement  : {concordance.upper()} between eyelid and palm")
    print(f"   clinical note   : {synthesis.get('note', '')}")

    # Final WHO / MoHFW Triage Decision
    uc = URGENCY_COLOR.get(final_decision.urgency.value, C.grey)
    print("\n " + C.bold("── WHO/MoHFW ACTIONABLE TRIAGE ") + "─" * 13)
    print(f"   {C.bold('ACTION')}   : {uc(C.bold(final_decision.action_code.value))}   [urgency: {uc(final_decision.urgency.value.upper())}]")
    print(f"   detail   : {final_decision.action}")
    if final_decision.hb_band:
        print(f"   Hb band  : {final_decision.hb_band.value}")
    if final_decision.recheck_in_weeks:
        print(f"   recheck  : in {final_decision.recheck_in_weeks} weeks")
    print(f"   flags    : {', '.join(final_decision.flags) or '(none)'}")
    print(f"   rule     : {C.dim(final_decision.rule_fired)}")
    print(f"   rationale: {C.dim(final_decision.rationale)}")

    # Mandatory disclaimer at the end
    print_disclaimer_block()


# ──────────────────────────────────────────────────────────────────────────────
# Visual Windows (3-image panels & heatmaps)
# ──────────────────────────────────────────────────────────────────────────────
def _rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def show_multimodal_window(
    eye_original_bgr: np.ndarray,
    eye_cropped_bgr: np.ndarray | None,
    eye_gradcam_bgr: np.ndarray | None,
    eye_report: TriageReport,
    palm_original_bgr: np.ndarray,
    palm_cropped_bgr: np.ndarray | None,
    palm_gradcam_bgr: np.ndarray | None,
    palm_report: TriageReport,
    final_decision: TriageDecision,
    synthesis: dict[str, Any],
    patient: PatientMetadata,
    out_dir: Path,
    show: bool = True,
) -> None:
    """
    Renders and pops up a comprehensive dual 3-panel window:
      Row 1: Eyelid (Primary):   Original · Cropped/Preprocessed · Grad-CAM Heatmap
      Row 2: Palm (Secondary):   Original · Cropped/Preprocessed · Grad-CAM Heatmap
      Bottom: Prominent Medical Disclaimer Banner.
    """
    import matplotlib
    if show:
        for backend in ("macosx", "TkAgg"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue
    else:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    verdict = f"{final_decision.action_code.value} · {final_decision.urgency.value.upper()}"
    p_info = f"Age: {patient.age_months}mo | Pregnant: {'Yes' if patient.is_pregnant else 'No'} | Symptoms: {', '.join(patient.symptoms) or 'None'}"
    fig.suptitle(
        f"Anemia-Screen Dual-Modality Triage — {verdict}\n{p_info}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # ── Row 1: Eyelid (Primary) ──
    ev = eye_report.vision
    e_title = f"Eyelid [PRIMARY] — {ev.risk_class.value} (P={ev.p_anemic:.2f})" if ev else "Eyelid [PRIMARY] (Rejected)"

    axes[0, 0].imshow(_rgb(eye_original_bgr))
    axes[0, 0].set_title(f"1 · Eyelid Original\n({e_title})", fontsize=11, fontweight="bold", color="#8b0000")

    if eye_cropped_bgr is not None:
        axes[0, 1].imshow(_rgb(eye_cropped_bgr))
        axes[0, 1].set_title("2 · Eyelid Preprocessed\n(Color balanced & normalized)", fontsize=11)
    else:
        axes[0, 1].text(0.5, 0.5, "Image rejected\n(quality gate)", ha="center", va="center", color="red")
        axes[0, 1].set_title("2 · Eyelid Preprocessed", fontsize=11)

    if eye_gradcam_bgr is not None:
        axes[0, 2].imshow(_rgb(eye_gradcam_bgr))
        axes[0, 2].set_title(f"3 · Eyelid Grad-CAM Heatmap\n(Attention Overlay · {ev.risk_class.value if ev else ''})", fontsize=11, fontweight="bold")
    else:
        axes[0, 2].imshow(_rgb(eye_original_bgr))
        reasons = "\n".join(eye_report.quality.reasons) or "Quality gate rejected"
        axes[0, 2].text(0.5, 0.05, reasons, ha="center", va="bottom", transform=axes[0, 2].transAxes, color="red", fontsize=9, bbox=dict(facecolor="white", alpha=0.8))
        axes[0, 2].set_title("3 · Eyelid Grad-CAM (n/a)", fontsize=11)

    # ── Row 2: Palm (Secondary) ──
    pv = palm_report.vision
    p_title = f"Palm [SECONDARY] — {pv.risk_class.value} (P={pv.p_anemic:.2f})" if pv else "Palm [SECONDARY] (Rejected)"

    axes[1, 0].imshow(_rgb(palm_original_bgr))
    axes[1, 0].set_title(f"4 · Palm Original\n({p_title})", fontsize=11, fontweight="bold", color="#333333")

    if palm_cropped_bgr is not None:
        axes[1, 1].imshow(_rgb(palm_cropped_bgr))
        axes[1, 1].set_title("5 · Palm Preprocessed\n(Normalized)", fontsize=11)
    else:
        axes[1, 1].text(0.5, 0.5, "Image rejected\n(quality gate)", ha="center", va="center", color="red")
        axes[1, 1].set_title("5 · Palm Preprocessed", fontsize=11)

    if palm_gradcam_bgr is not None:
        axes[1, 2].imshow(_rgb(palm_gradcam_bgr))
        axes[1, 2].set_title(f"6 · Palm Grad-CAM Heatmap\n(Attention Overlay · {pv.risk_class.value if pv else ''})", fontsize=11, fontweight="bold")
    else:
        axes[1, 2].imshow(_rgb(palm_original_bgr))
        reasons = "\n".join(palm_report.quality.reasons) or "Quality gate rejected"
        axes[1, 2].text(0.5, 0.05, reasons, ha="center", va="bottom", transform=axes[1, 2].transAxes, color="red", fontsize=9, bbox=dict(facecolor="white", alpha=0.8))
        axes[1, 2].set_title("6 · Palm Grad-CAM (n/a)", fontsize=11)

    for row in axes:
        for ax in row:
            ax.axis("off")

    # Bottom Medical Disclaimer Banner
    fig.text(
        0.5, 0.02,
        "⚠️ CLINICAL DISCLAIMER: Screening & triage decision-support only — NOT a medical diagnosis. "
        "Confirm every abnormal finding with laboratory hemoglobin testing.",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#8b0000",
        bbox=dict(facecolor="#fff0f0", edgecolor="#d9534f", boxstyle="round,pad=0.5", linewidth=1.5),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.94])

    out_dir.mkdir(parents=True, exist_ok=True)
    master_path = out_dir / "multimodal_triage_dashboard.png"
    fig.savefig(str(master_path), dpi=130, bbox_inches="tight")
    print(C.dim(f"[cli] master multimodal figure saved → {master_path}"))

    # Also save separate individual 3-image panels
    _save_single_3panel(eye_original_bgr, eye_cropped_bgr, eye_gradcam_bgr, eye_report, out_dir / "eyelid_3panel.png", "Eyelid (Primary)")
    _save_single_3panel(palm_original_bgr, palm_cropped_bgr, palm_gradcam_bgr, palm_report, out_dir / "palm_3panel.png", "Palm (Secondary)")

    if not show:
        plt.close(fig)
        return

    print(C.dim("[cli] opening multimodal visual window… (close it to exit)"))
    _bring_window_to_front(fig)
    plt.show(block=True)


def _save_single_3panel(
    original_bgr: np.ndarray,
    cropped_bgr: np.ndarray | None,
    gradcam_bgr: np.ndarray | None,
    report: TriageReport,
    save_path: Path,
    title_prefix: str,
) -> None:
    """Save an isolated 3-panel figure for one modality."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    v = report.vision
    stat = f"{v.risk_class.value} (P={v.p_anemic:.2f})" if v else "Rejected"
    fig.suptitle(f"{title_prefix} 3-Panel Screening — {stat}", fontsize=13, fontweight="bold")

    axes[0].imshow(_rgb(original_bgr))
    axes[0].set_title("1 · Original input")

    if cropped_bgr is not None:
        axes[1].imshow(_rgb(cropped_bgr))
        axes[1].set_title("2 · Preprocessed / normalized")
    else:
        axes[1].text(0.5, 0.5, "Image rejected", ha="center", va="center", color="red")
        axes[1].set_title("2 · Preprocessed")

    if gradcam_bgr is not None:
        axes[2].imshow(_rgb(gradcam_bgr))
        axes[2].set_title("3 · Grad-CAM heatmap overlay")
    else:
        axes[2].imshow(_rgb(original_bgr))
        reasons = "\n".join(report.quality.reasons) or "unavailable"
        axes[2].text(0.5, 0.05, reasons, ha="center", va="bottom", transform=axes[2].transAxes, color="red", fontsize=8, bbox=dict(facecolor="white", alpha=0.8))
        axes[2].set_title("3 · Grad-CAM (n/a)")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(C.dim(f"[cli] saved 3-panel figure → {save_path}"))


def _bring_window_to_front(fig) -> None:
    """Bring the matplotlib window to the foreground on macOS / desktop."""
    try:
        fig.canvas.draw()
        mgr = fig.canvas.manager
        if hasattr(mgr, "window"):
            win = mgr.window
            if hasattr(win, "lift"):
                win.lift()
                win.attributes("-topmost", True)
                win.after(600, lambda: win.attributes("-topmost", False))
        if platform.system() == "Darwin":
            subprocess.run(
                ["osascript", "-e",
                 f'tell application "System Events" to set frontmost of the first process whose unix id is {os.getpid()} to true'],
                capture_output=True, timeout=2,
            )
    except Exception:
        pass


def show_window(
    original_bgr: np.ndarray,
    cropped_bgr: np.ndarray | None,
    gradcam_bgr: np.ndarray | None,
    report: TriageReport,
    save_path: Path,
    show: bool,
) -> None:
    """Legacy single-image 3-panel window with disclaimer banner."""
    import matplotlib
    if show:
        for backend in ("macosx", "TkAgg"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue
    else:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rd = report.model_dump(mode="json")
    t = rd.get("triage") or {}
    verdict = f"{t.get('action_code', rd['status'])} · {t.get('urgency', '')}"

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    fig.suptitle(f"Anemia-Screen  —  {verdict}", fontsize=14, fontweight="bold")

    axes[0].imshow(_rgb(original_bgr))
    axes[0].set_title("1 · Original input")

    if cropped_bgr is not None:
        axes[1].imshow(_rgb(cropped_bgr))
        axes[1].set_title("2 · Cropped / preprocessed")
    else:
        axes[1].text(0.5, 0.5, "no preprocessing\n(image rejected)",
                     ha="center", va="center", fontsize=11)
        axes[1].set_title("2 · Cropped / preprocessed")

    if gradcam_bgr is not None:
        axes[2].imshow(_rgb(gradcam_bgr))
        axes[2].set_title("3 · Grad-CAM overlay")
    else:
        reasons = "\n".join((rd.get("quality") or {}).get("reasons", [])) or "unavailable"
        axes[2].imshow(_rgb(original_bgr))
        axes[2].set_title("3 · Grad-CAM (n/a)")
        axes[2].text(0.5, 0.04, reasons, ha="center", va="bottom",
                     transform=axes[2].transAxes, fontsize=8, color="red",
                     wrap=True, bbox=dict(facecolor="white", alpha=0.7, pad=3))

    for ax in axes:
        ax.axis("off")

    fig.text(
        0.5, 0.02,
        "⚠️ CLINICAL DISCLAIMER: Screening & triage decision-support only — NOT a medical diagnosis.",
        ha="center", va="center", fontsize=9, color="#8b0000",
        bbox=dict(facecolor="#fff0f0", edgecolor="#d9534f", boxstyle="round,pad=0.3"),
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
    print(C.dim(f"[cli] figure saved → {save_path}"))

    if not show:
        plt.close(fig)
        return

    print(C.dim("[cli] opening window… (close it to exit)"))
    _bring_window_to_front(fig)
    plt.show(block=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline Flow
# ──────────────────────────────────────────────────────────────────────────────
def run_multimodal_pipeline(
    eye_path: Path,
    palm_path: Path,
    args,
    patient_data: dict[str, Any] | None = None,
    symptom_text: str | None = None,
) -> int:
    """
    Run complete multimodal pipeline on both Eyelid (primary) and Palm (secondary).
    """
    from app.imageio import read_image
    from app.pipeline import preprocessing
    from app.pipeline.orchestrator import analyze_image

    print(C.cyan("\n[pipeline] Loading screening images…"))
    try:
        eye_bgr = read_image(eye_path)
        print(f"  ✓ Eyelid loaded : {eye_path} ({eye_bgr.shape[1]}×{eye_bgr.shape[0]} px)")
    except Exception as exc:
        print(C.red(f"  ✗ Could not decode eyelid image: {exc}"))
        return 1

    try:
        palm_bgr = read_image(palm_path)
        print(f"  ✓ Palm loaded   : {palm_path} ({palm_bgr.shape[1]}×{palm_bgr.shape[0]} px)")
    except Exception as exc:
        print(C.red(f"  ✗ Could not decode palm image: {exc}"))
        return 1

    # Load baseline metadata from sidecar or flags
    meta = load_metadata(eye_path, args)
    if patient_data:
        if "age_years" in patient_data:
            meta["age_months"] = int(round(patient_data["age_years"] * 12))
        if "sex" in patient_data:
            meta["sex"] = patient_data["sex"]
        if "is_pregnant" in patient_data:
            meta["is_pregnant"] = patient_data["is_pregnant"]
        if "trimester" in patient_data and patient_data["trimester"]:
            meta["trimester"] = patient_data["trimester"]
        if patient_data.get("measured_hb") is not None:
            meta["measured_hb"] = patient_data["measured_hb"]
        if patient_data.get("symptoms_text") and not symptom_text:
            symptom_text = patient_data["symptoms_text"]

    # Explicit CLI flags override
    if getattr(args, "age_months", None) is not None:
        meta["age_months"] = args.age_months
    if getattr(args, "sex", None) is not None:
        meta["sex"] = args.sex
    if getattr(args, "pregnant", False):
        meta["is_pregnant"] = True
    if getattr(args, "trimester", None) is not None:
        meta["trimester"] = args.trimester
    if getattr(args, "hb", None) is not None:
        meta["measured_hb"] = args.hb

    patient = PatientMetadata(**meta)

    # Symptom text NLP analysis
    if symptom_text is None and getattr(args, "symptoms", None):
        symptom_text = args.symptoms

    nlp_res = analyze_symptom_text(symptom_text)
    if nlp_res["matched_symptoms"]:
        patient.symptoms = sorted(set(patient.symptoms + nlp_res["matched_symptoms"]))
    if nlp_res["detected_pregnant"]:
        patient.is_pregnant = True
    if nlp_res["detected_trimester"]:
        patient.trimester = nlp_res["detected_trimester"]
    if nlp_res["detected_hb"] is not None and patient.measured_hb is None:
        patient.measured_hb = nlp_res["detected_hb"]

    with_gradcam = not getattr(args, "no_gradcam", False)

    # 1. Analyze Eyelid (Primary factor)
    print(C.dim("[pipeline] Analyzing Eyelid (Primary factor — Conjunctival Pallor)…"))
    eye_report = analyze_image(eye_bgr, patient, with_gradcam=with_gradcam, modality="eye")

    # 2. Analyze Palm (Secondary factor)
    print(C.dim("[pipeline] Analyzing Palm (Secondary factor — Palmar Pallor)…"))
    palm_report = analyze_image(palm_bgr, patient, with_gradcam=with_gradcam, modality="palm")

    # 3. Multimodal Synthesis (Eyelid 70% primary weight)
    final_decision, synthesis = generate_multimodal_triage_decision(
        eye_report=eye_report,
        palm_report=palm_report,
        patient=patient,
    )

    # 4. Print full terminal summary
    print_multimodal_summary(
        eye_path=eye_path,
        palm_path=palm_path,
        eye_report=eye_report,
        palm_report=palm_report,
        final_decision=final_decision,
        synthesis=synthesis,
        patient=patient,
        nlp_results=nlp_res,
    )

    out_dir = Path(getattr(args, "out_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract visual panels
    eye_cropped_bgr = None
    eye_gradcam_bgr = None
    if eye_report.status == "ok":
        try:
            eye_cropped_bgr = preprocessing.preprocess(eye_bgr, modality="eye")["normalized_bgr"]
        except Exception:
            pass
    if eye_report.gradcam_png_base64:
        eye_gradcam_bgr = _decode_b64_bgr(eye_report.gradcam_png_base64)
    if eye_report.heatmap_png_base64:
        heat = _decode_b64_bgr(eye_report.heatmap_png_base64)
        cv2.imwrite(str(out_dir / "eyelid_heatmap.png"), heat)
        print(C.dim(f"[cli] saved standalone eyelid heatmap → {out_dir / 'eyelid_heatmap.png'}"))

    palm_cropped_bgr = None
    palm_gradcam_bgr = None
    if palm_report.status == "ok":
        try:
            palm_cropped_bgr = preprocessing.preprocess(palm_bgr, modality="palm")["normalized_bgr"]
        except Exception:
            pass
    if palm_report.gradcam_png_base64:
        palm_gradcam_bgr = _decode_b64_bgr(palm_report.gradcam_png_base64)
    if palm_report.heatmap_png_base64:
        heat = _decode_b64_bgr(palm_report.heatmap_png_base64)
        cv2.imwrite(str(out_dir / "palm_heatmap.png"), heat)
        print(C.dim(f"[cli] saved standalone palm heatmap → {out_dir / 'palm_heatmap.png'}"))

    # 5. Render dual 3-panel window + disclaimer
    show_multimodal_window(
        eye_original_bgr=eye_bgr,
        eye_cropped_bgr=eye_cropped_bgr,
        eye_gradcam_bgr=eye_gradcam_bgr,
        eye_report=eye_report,
        palm_original_bgr=palm_bgr,
        palm_cropped_bgr=palm_cropped_bgr,
        palm_gradcam_bgr=palm_gradcam_bgr,
        palm_report=palm_report,
        final_decision=final_decision,
        synthesis=synthesis,
        patient=patient,
        out_dir=out_dir,
        show=not getattr(args, "no_window", False),
    )

    # 6. JSON output
    json_path = out_dir / "multimodal_triage_report.json"
    full_json = {
        "status": "ok",
        "primary_modality": "eyelid",
        "patient": patient.model_dump(mode="json"),
        "symptom_nlp": nlp_res,
        "eyelid": eye_report.model_dump(mode="json", exclude={"gradcam_png_base64", "heatmap_png_base64"}),
        "palm": palm_report.model_dump(mode="json", exclude={"gradcam_png_base64", "heatmap_png_base64"}),
        "synthesis": synthesis,
        "final_triage": final_decision.model_dump(mode="json"),
        "disclaimer": CLINICAL_DISCLAIMER,
    }
    json_path.write_text(json.dumps(full_json, indent=2))
    print(C.dim(f"[cli] saved JSON report → {json_path}"))

    if getattr(args, "json", False):
        print(json.dumps(full_json, indent=2))

    # Auto-open preview on macOS if requested
    if getattr(args, "open", False):
        master_img = out_dir / "multimodal_triage_dashboard.png"
        try:
            opener = "open" if platform.system() == "Darwin" else "xdg-open"
            subprocess.run([opener, str(master_img)], check=False)
        except Exception:
            pass

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Anemia-Screen: Dual-modality (Eyelid + Palm) intake, symptom NLP & WHO triage."
    )
    # Multimodal image arguments (individual files)
    ap.add_argument("--eye", "--eyelid", default=None, help="Path to Eyelid (conjunctiva) image.")
    ap.add_argument("--palm", default=None, help="Path to Palm / hand image.")

    # Legacy single image argument
    ap.add_argument("--image", default=None, help="Path to a single image (legacy mode).")
    ap.add_argument("--modality", choices=["eye", "palm"], default="eye",
                    help="modality for single-image mode (default=eye).")

    # Patient metadata flags
    ap.add_argument("--age-months", type=int, default=None)
    ap.add_argument("--age-years", type=float, default=None)
    ap.add_argument("--sex", choices=["female", "male", "other", "unknown"], default=None)
    ap.add_argument("--pregnant", action="store_true", help="Mark patient as pregnant.")
    ap.add_argument("--trimester", type=int, choices=[1, 2, 3], default=None)
    ap.add_argument("--symptoms", default=None, help="Symptom text or comma-separated list.")
    ap.add_argument("--hb", type=float, default=None, help="Measured hemoglobin (g/dL).")

    # Options
    ap.add_argument("--ignore-sidecar", action="store_true", help="Do not read <image>.json.")
    ap.add_argument("--no-window", action="store_true", help="Headless: print + save only.")
    ap.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM entirely.")
    ap.add_argument("--open", action="store_true",
                    help="Also open the saved report in OS image viewer (macOS Preview).")
    ap.add_argument("--json", action="store_true", help="Also dump full JSON.")
    ap.add_argument("--out-dir", default="outputs", help="Where to save figures and artifacts.")
    args = ap.parse_args()

    # Case 1: Legacy single-image mode explicitly requested via --image
    if args.image is not None:
        image_path = Path(args.image)
        if not image_path.exists():
            print(C.red(f"[cli] image not found: {image_path}"))
            return 1
        from app.imageio import read_image
        from app.pipeline import preprocessing
        from app.pipeline.orchestrator import analyze_image

        try:
            bgr = read_image(image_path)
        except Exception as exc:
            print(C.red(f"[cli] could not decode image: {exc}"))
            return 1

        meta = load_metadata(image_path, args)
        patient = PatientMetadata(**meta)

        with_gradcam = not args.no_gradcam
        print(C.dim(f"[cli] running pipeline… (modality={args.modality})"))
        report = analyze_image(bgr, patient, with_gradcam=with_gradcam, modality=args.modality)
        print_summary(image_path, bgr.shape, report, patient, modality=args.modality)

        if args.json:
            print(json.dumps(report.model_dump(mode="json"), indent=2))

        cropped_bgr = None
        gradcam_bgr = None
        if report.status == "ok":
            try:
                cropped_bgr = preprocessing.preprocess(bgr, modality=args.modality)["normalized_bgr"]
            except Exception:
                pass
        if report.gradcam_png_base64:
            gradcam_bgr = _decode_b64_bgr(report.gradcam_png_base64)

        save_path = Path(args.out_dir) / f"{image_path.stem}_report.png"
        show_window(
            original_bgr=bgr,
            cropped_bgr=cropped_bgr,
            gradcam_bgr=gradcam_bgr,
            report=report,
            save_path=save_path,
            show=not args.no_window,
        )
        return 0

    # Case 2: Multimodal mode (Dual Eyelid + Palm)
    from scripts.ui_selector import select_images_and_patient_gui, prompt_images_cli, prompt_patient_intake_cli

    eye_p: Path | None = Path(args.eye) if args.eye else None
    palm_p: Path | None = Path(args.palm) if args.palm else None
    patient_data: dict[str, Any] | None = None

    # If images were not both provided on CLI, try GUI selection dialog
    if not (eye_p and palm_p and eye_p.exists() and palm_p.exists()):
        if not args.no_window:
            print(C.cyan("\n[cli] Opening Intake Dialog (Select Eyelid & Palm images individually)…"))
            gui_res = select_images_and_patient_gui()
            if gui_res:
                eye_p = gui_res["eye_path"]
                palm_p = gui_res["palm_path"]
                patient_data = gui_res

    # If still not provided (e.g. headless or GUI canceled), use terminal prompts
    if not (eye_p and palm_p and eye_p.exists() and palm_p.exists()):
        eye_p, palm_p = prompt_images_cli()
        if not (eye_p and palm_p and eye_p.exists() and palm_p.exists()):
            print(C.red("[cli] Both Eyelid and Palm images are required. Exiting."))
            return 1

    # If patient factors weren't already provided by GUI or CLI flags, prompt in terminal
    if patient_data is None:
        defaults = {
            "age_years": args.age_years or (args.age_months / 12.0 if args.age_months else 25.0),
            "sex": args.sex or "female",
            "is_pregnant": args.pregnant,
            "trimester": args.trimester,
            "measured_hb": args.hb,
            "symptoms_text": args.symptoms or "",
        }
        if not args.no_window and args.age_months is None and args.sex is None and not args.pregnant:
            patient_data = prompt_patient_intake_cli(defaults=defaults)
        else:
            patient_data = defaults

    print(f"\n{C.green('✓ Selected Eyelid image')}: {eye_p}")
    print(f"{C.green('✓ Selected Palm image')  }: {palm_p}")
    preg_str = f"yes (T{patient_data.get('trimester')})" if patient_data.get('is_pregnant') else "no"
    hb_str = f"{patient_data.get('measured_hb')} g/dL" if patient_data.get('measured_hb') is not None else "(none)"
    print(
        f"{C.green('✓ Patient Factors')}      : Age {patient_data.get('age_years', 25):.0f} yr · "
        f"{patient_data.get('sex', 'female')} · Pregnant={preg_str} · Hb={hb_str}"
    )

    return run_multimodal_pipeline(
        eye_path=eye_p,
        palm_path=palm_p,
        args=args,
        patient_data=patient_data,
        symptom_text=patient_data.get("symptoms_text"),
    )


if __name__ == "__main__":
    sys.exit(main())
