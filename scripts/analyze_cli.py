"""
Terminal-first single-image analyzer for Anemia-Screen.

Runs the FULL pipeline in-process (no server / no web frontend) on one image,
prints a complete color-coded summary to the terminal, and pops up a matplotlib
window with three panels: original · cropped/preprocessed · Grad-CAM overlay.

Usage:
    # auto-read the sidecar <image>.json for metadata if present
    python -m scripts.analyze_cli --image data/sample/pale_pregnant.png

    # provide metadata on the command line
    python -m scripts.analyze_cli --image eye.jpg --age-months 312 --pregnant \
        --symptoms fatigue,breathlessness --hb 9.5

    python -m scripts.analyze_cli --image eye.jpg --no-window   # headless
    python -m scripts.analyze_cli --image eye.jpg --json        # also dump JSON
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from app.config import settings


# ── ANSI colours (auto-disabled when output is not a TTY) ────────────────────────
class C:
    _tty = sys.stdout.isatty()

    @staticmethod
    def _w(code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if C._tty else s

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


def load_metadata(image_path: Path, args) -> dict:
    """Build metadata from CLI flags, falling back to a sidecar <image>.json."""
    # Start from sidecar if present.
    meta: dict = {}
    sidecar = image_path.with_suffix(".json")
    if sidecar.exists() and not args.ignore_sidecar:
        try:
            meta = json.loads(sidecar.read_text())
            print(C.dim(f"[cli] loaded metadata from {sidecar.name}"))
        except json.JSONDecodeError:
            print(C.yellow(f"[cli] WARN: bad sidecar JSON {sidecar.name}; ignoring."))

    # CLI flags override sidecar values when explicitly provided.
    if args.age_months is not None:
        meta["age_months"] = args.age_months
    if args.sex is not None:
        meta["sex"] = args.sex
    if args.pregnant:
        meta["is_pregnant"] = True
    if args.trimester is not None:
        meta["trimester"] = args.trimester
    if args.symptoms is not None:
        meta["symptoms"] = [s.strip() for s in args.symptoms.split(",") if s.strip()]
    if args.hb is not None:
        meta["measured_hb"] = args.hb
    return meta


def print_summary(image_path: Path, img_shape, report, patient, modality: str = "eye") -> None:
    """Print the full color-coded triage summary to the terminal."""
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
    print(C.cyan(C.bold("        ANEMIA-SCREEN · TRIAGE REPORT")))
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

    # ── Quality gate ──
    passed = q["passed"]
    status = C.green("PASS ✓") if passed else C.red("REJECT ✗")
    print("\n " + C.bold("── QUALITY GATE ") + "─" * 26 + f"  {status}")
    print(f"   blur (Laplacian var) : {q['blur_variance']:>8.1f}   (min {blur_min:.0f})")
    print(f"   brightness mean      : {q['brightness_mean']:>8.1f}   (window {settings.brightness_min:.0f}–{settings.brightness_max:.0f})")
    print(f"   over / under exposed : {q['overexposed_fraction']*100:>5.1f}% / {q['underexposed_fraction']*100:.1f}%")
    print(f"   glare fraction       : {q['glare_fraction']*100:>7.1f}%   (max {settings.glare_fraction_max*100:.0f}%)")
    print(f"   conjunctiva tissue   : {q['tissue_fraction']*100:>7.1f}%   (min {settings.min_tissue_fraction*100:.0f}%)")
    if q["reasons"]:
        for r in q["reasons"]:
            print("   " + C.red("• " + r))

    # ── Vision model ──
    if v:
        print("\n " + C.bold("── VISION MODEL (MobileNetV3) ") + "─" * 12)
        print(f"   risk class   : {C.bold(v['risk_class'])}   (confidence {v['confidence']*100:.1f}%)")
        print(f"   P(anemic)    : {v['p_anemic']:.3f}")
        print("   probabilities:")
        for name, p in v["probabilities"].items():
            print(f"     {name:<14s} {_bar(p)}  {p*100:5.1f}%")
        if not v["model_trained"]:
            print("   " + C.yellow("⚠ UNTRAINED demo weights — not clinically valid yet"))

    # ── Triage ──
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

    # ── Warnings + timing ──
    for wmsg in rd.get("warnings", []):
        print("\n " + C.yellow("⚠ " + wmsg))
    print(f"\n {C.dim('processing: %.0f ms' % rd.get('processing_ms', 0))}")
    print(C.cyan(line) + "\n")


def _decode_b64_bgr(b64: str) -> np.ndarray:
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def show_window(
    original_bgr: np.ndarray,
    cropped_bgr: np.ndarray | None,
    gradcam_bgr: np.ndarray | None,
    report,
    save_path: Path,
    show: bool,
) -> None:
    """
    Build and (optionally) pop up a 3-panel matplotlib window:
      original · cropped/preprocessed · Grad-CAM overlay.
    Always saves the figure to ``save_path``.
    """
    import matplotlib

    if show:
        # Force a GUI backend so a real window appears (macOS native, Tk fallback).
        for backend in ("macosx", "TkAgg"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue
    else:
        matplotlib.use("Agg", force=True)  # headless render-to-file
    import matplotlib.pyplot as plt

    rd = report.model_dump(mode="json")
    t = rd.get("triage") or {}
    verdict = f"{t.get('action_code', rd['status'])} · {t.get('urgency', '')}"

    def _rgb(bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
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
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
    print(C.dim(f"[cli] figure saved → {save_path}"))

    if not show:
        plt.close(fig)
        return

    print(C.dim("[cli] opening window… (close it to exit)"))
    # Draw first so the window has content, then try to raise it to the front —
    # on macOS the matplotlib window often opens *behind* the terminal.
    try:
        fig.canvas.draw()
        mgr = fig.canvas.manager
        if hasattr(mgr, "window"):
            win = mgr.window
            # Tk backend
            if hasattr(win, "lift"):
                try:
                    win.lift()
                    win.attributes("-topmost", True)
                    win.after(500, lambda: win.attributes("-topmost", False))
                except Exception:
                    pass
        # Bring the Python process to the foreground on macOS.
        import platform
        if platform.system() == "Darwin":
            import subprocess
            try:
                subprocess.run(
                    ["osascript", "-e",
                     'tell application "System Events" to set frontmost of '
                     'the first process whose unix id is %d to true' % __import__("os").getpid()],
                    capture_output=True, timeout=3,
                )
            except Exception:
                pass
    except Exception:
        pass

    plt.show(block=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Analyze a single conjunctiva image: terminal summary + Grad-CAM window."
    )
    ap.add_argument("--image", required=True, help="Path to the image.")
    ap.add_argument("--modality", choices=["eye", "palm"], default="eye",
                    help="eye=conjunctiva (default), palm=palm/nail-bed photo.")
    ap.add_argument("--age-months", type=int, default=None)
    ap.add_argument("--sex", choices=["male", "female", "other", "unknown"], default=None)
    ap.add_argument("--pregnant", action="store_true", help="Mark patient as pregnant.")
    ap.add_argument("--trimester", type=int, choices=[1, 2, 3], default=None)
    ap.add_argument("--symptoms", default=None, help="Comma-separated, e.g. 'fatigue,fainting'.")
    ap.add_argument("--hb", type=float, default=None, help="Measured hemoglobin (g/dL).")
    ap.add_argument("--ignore-sidecar", action="store_true", help="Do not read <image>.json.")
    ap.add_argument("--no-window", action="store_true", help="Headless: print + save only.")
    ap.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM entirely.")
    ap.add_argument("--open", action="store_true",
                    help="Also open the saved report PNG in the default viewer (macOS Preview).")
    ap.add_argument("--json", action="store_true", help="Also print the raw JSON report.")
    ap.add_argument("--out-dir", default="outputs", help="Where to save the figure.")
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(C.red(f"[cli] image not found: {image_path}"))
        return 1

    # HEIC-aware loader (handles iPhone photos via pillow-heif + Pillow fallback).
    from app.imageio import read_image

    try:
        bgr = read_image(image_path)
    except Exception as exc:
        print(C.red(f"[cli] could not decode image: {exc}"))
        return 1

    # Deferred heavy imports (torch) so --help stays fast.
    from app.pipeline.orchestrator import analyze_image
    from app.pipeline import preprocessing
    from app.schemas import PatientMetadata

    meta = load_metadata(image_path, args)
    patient = PatientMetadata(**meta)

    with_gradcam = not args.no_gradcam
    print(C.dim(f"[cli] running pipeline… (modality={args.modality})"))
    report = analyze_image(bgr, patient, with_gradcam=with_gradcam, modality=args.modality)

    # Terminal summary.
    print_summary(image_path, bgr.shape, report, patient, modality=args.modality)

    if args.json:
        print(json.dumps(report.model_dump(mode="json"), indent=2))

    # Build the visual panels.
    cropped_bgr = None
    gradcam_bgr = None
    if report.status == "ok":
        try:
            cropped_bgr = preprocessing.preprocess(bgr, modality=args.modality)["normalized_bgr"]
        except Exception as exc:  # pragma: no cover
            print(C.yellow(f"[cli] could not render preprocessed panel ({exc})"))
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

    # Guaranteed fallback: open the saved PNG in the OS image viewer.
    if args.open:
        import platform
        import subprocess
        try:
            opener = {"Darwin": "open", "Windows": "start", "Linux": "xdg-open"}.get(
                platform.system(), "open"
            )
            subprocess.run([opener, str(save_path)], check=False)
        except Exception as exc:
            print(C.yellow(f"[cli] could not auto-open figure ({exc})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

