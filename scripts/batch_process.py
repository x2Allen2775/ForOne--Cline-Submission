"""
Batch Processing Script (plan §3 Hour 6, demo §6 macro-level).

Iterates over a folder of images, sends each to the running API's ``/analyze``
endpoint (or runs the pipeline in-process), and writes a summarized triage CSV —
the "health ministry dashboard" export used in the pitch.

Two modes:
  * --mode api     : POST each image to a running FastAPI server (default).
  * --mode local   : import the pipeline and run in-process (no server needed).

Metadata: optionally provide a sidecar JSON per image (same stem, .json) or a
single --metadata-json applied to all images.

Usage:
    # against a running server
    python -m scripts.batch_process --input data/sample --out outputs/triage.csv

    # fully local (no server)
    python -m scripts.batch_process --mode local --input data/sample \
        --out outputs/triage.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def _load_metadata(img_path: Path, default: dict) -> dict:
    sidecar = img_path.with_suffix(".json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            print(f"[batch] WARN: bad sidecar JSON for {img_path.name}; using default.")
    return default


def _rows_from_report(img_name: str, report: dict) -> dict:
    vision = report.get("vision") or {}
    triage = report.get("triage") or {}
    quality = report.get("quality") or {}
    return {
        "image": img_name,
        "status": report.get("status"),
        "risk_class": vision.get("risk_class", ""),
        "p_anemic": vision.get("p_anemic", ""),
        "confidence": vision.get("confidence", ""),
        "model_trained": vision.get("model_trained", ""),
        "action_code": triage.get("action_code", ""),
        "urgency": triage.get("urgency", ""),
        "action": triage.get("action", ""),
        "hb_band": triage.get("hb_band", "") or "",
        "flags": "; ".join(triage.get("flags", []) or []),
        "rule_fired": triage.get("rule_fired", ""),
        "quality_passed": quality.get("passed", ""),
        "quality_reasons": "; ".join(quality.get("reasons", [])),
        "processing_ms": report.get("processing_ms", ""),
    }


def run_api(images: list[Path], url: str, default_meta: dict, with_gradcam: bool,
            modality: str = "eye") -> list[dict]:
    import requests

    rows = []
    for p in images:
        meta = _load_metadata(p, default_meta)
        with open(p, "rb") as fh:
            files = {"image": (p.name, fh, "application/octet-stream")}
            data = {
                "metadata": json.dumps(meta),
                "with_gradcam": str(with_gradcam).lower(),
                "modality": modality,
            }
            resp = requests.post(url, files=files, data=data, timeout=120)
        if resp.status_code != 200:
            print(f"[batch] {p.name}: HTTP {resp.status_code} {resp.text[:160]}")
            rows.append({"image": p.name, "status": f"error_{resp.status_code}"})
            continue
        rows.append(_rows_from_report(p.name, resp.json()))
        print(f"[batch] {p.name}: {rows[-1].get('risk_class','?')} "
              f"({rows[-1].get('urgency','?')})")
    return rows


def run_local(images: list[Path], default_meta: dict, with_gradcam: bool,
              modality: str = "eye") -> list[dict]:
    from app.imageio import read_image
    from app.pipeline.orchestrator import analyze_image
    from app.schemas import PatientMetadata

    rows = []
    for p in images:
        meta = _load_metadata(p, default_meta)
        try:
            bgr = read_image(p)
        except Exception:
            rows.append({"image": p.name, "status": "error_decode"})
            continue
        report = analyze_image(bgr, PatientMetadata(**meta), with_gradcam=with_gradcam,
                               modality=modality)
        rows.append(_rows_from_report(p.name, report.model_dump()))
        print(f"[batch] {p.name}: {rows[-1].get('risk_class','?')} "
              f"({rows[-1].get('urgency','?')})")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch anemia triage over an image folder.")
    ap.add_argument("--input", required=True, help="Folder of images.")
    ap.add_argument("--out", default="outputs/triage_report.csv")
    ap.add_argument("--mode", choices=["api", "local"], default="api")
    ap.add_argument("--url", default="http://127.0.0.1:8000/analyze")
    ap.add_argument("--metadata-json", default="{}", help="Default metadata JSON for all images.")
    ap.add_argument("--modality", choices=["eye", "palm"], default="eye",
                    help="eye=conjunctiva (default), palm=palm/nail-bed images.")
    ap.add_argument("--no-gradcam", action="store_true", help="Skip heatmaps (faster).")
    args = ap.parse_args()

    folder = Path(args.input)
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"[batch] No images found under {folder}")
        return 1
    default_meta = json.loads(args.metadata_json)
    with_gradcam = not args.no_gradcam

    print(f"[batch] processing {len(images)} images in '{args.mode}' mode "
          f"(modality={args.modality})...")
    if args.mode == "api":
        rows = run_api(images, args.url, default_meta, with_gradcam, args.modality)
    else:
        rows = run_local(images, default_meta, with_gradcam, args.modality)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image", "status", "risk_class", "p_anemic", "confidence", "model_trained",
        "action_code", "urgency", "action", "hb_band", "flags", "rule_fired",
        "quality_passed", "quality_reasons", "processing_ms",
    ]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Quick summary (the "flag the urgent cases" moment in the pitch).
    refer = sum(1 for r in rows if r.get("action_code") == "REFER_NOW")
    high = sum(1 for r in rows if r.get("urgency") == "High")
    rejected = sum(1 for r in rows if r.get("status") == "rejected")
    print(f"[batch] wrote {out_path} — {len(rows)} rows "
          f"({refer} REFER_NOW, {high} high-urgency, {rejected} rejected).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
