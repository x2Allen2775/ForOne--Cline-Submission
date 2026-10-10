"""
Generate synthetic conjunctiva-like sample images + metadata sidecars.

These are NOT real medical images — they are simple procedurally-generated
reddish "eyelid" patches that pass/fail the quality gate in predictable ways, so
the whole pipeline (and the demo + batch script) works *before* you download the
real Eyes-defy-anemia / CP-AnemiC datasets.

Usage:
    python -m scripts.make_sample_images            # writes to data/sample/
    python -m scripts.make_sample_images --n 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from app.config import SAMPLE_DIR


def _reddish_patch(h: int, w: int, redness: int, blur: int = 0, dark: float = 1.0) -> np.ndarray:
    """Create a reddish gradient 'conjunctiva' patch (BGR)."""
    rng = np.random.default_rng(redness + h + w)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Base reddish tone: high R (BGR index 2), lower G/B. 'redness' = how red/healthy.
    base_r = np.clip(redness, 60, 255)
    base_g = int(base_r * 0.35)
    base_b = int(base_r * 0.33)
    img[:, :, 2] = base_r
    img[:, :, 1] = base_g
    img[:, :, 0] = base_b
    # Add fine, high-contrast vascular texture so a non-blurred patch has a high
    # Laplacian variance (passes the focus gate). Darker-red vessels on red tissue.
    vessel_dark = (int(base_b * 0.4), int(base_g * 0.4), int(min(255, base_r * 0.7)))
    vessel_bright = (int(base_b), int(base_g), int(min(255, base_r + 50)))
    for _ in range(120):
        x1 = int(rng.integers(0, w))
        y1 = int(rng.integers(0, h))
        x2 = int(np.clip(x1 + rng.integers(-25, 25), 0, w - 1))
        y2 = int(np.clip(y1 + rng.integers(-8, 8), 0, h - 1))
        color = vessel_dark if rng.random() < 0.5 else vessel_bright
        cv2.line(img, (x1, y1), (x2, y2), color, 1)
    # Surround with skin-ish border so auto-crop has something to isolate.
    canvas = np.full((h + 80, w + 80, 3), (120, 150, 190), dtype=np.uint8)
    canvas[40:40 + h, 40:40 + w] = img
    # Apply blur + illumination to the WHOLE frame (a bad photo is bad everywhere),
    # so the quality gate reacts realistically.
    if blur > 0:
        k = blur * 2 + 1
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)
    canvas = np.clip(canvas.astype(np.float32) * dark, 0, 255).astype(np.uint8)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=str(SAMPLE_DIR))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # (name, redness(health), blur, darkness, metadata)
    specs = [
        ("healthy_adult", 210, 0, 1.0,
         {"patient_id": "S001", "age_months": 360, "sex": "female", "symptoms": ["mild fatigue"]}),
        ("pale_pregnant", 120, 0, 1.0,
         {"patient_id": "S002", "age_months": 312, "sex": "female", "is_pregnant": True,
          "trimester": 2, "symptoms": ["tiredness"]}),
        ("severe_pale_child", 80, 0, 1.0,
         {"patient_id": "S003", "age_months": 36, "sex": "male", "symptoms": ["weakness"]}),
        ("red_flag_case", 150, 0, 1.0,
         {"patient_id": "S004", "age_months": 300, "sex": "female",
          "symptoms": ["breathlessness", "fainting"]}),
        ("blurry_reject", 150, 8, 1.0,
         {"patient_id": "S005", "age_months": 280, "sex": "male", "symptoms": []}),
        ("too_dark_reject", 150, 0, 0.25,
         {"patient_id": "S006", "age_months": 240, "sex": "female", "symptoms": []}),
    ]
    specs = specs[: max(1, args.n)]

    for name, redness, blur, dark, meta in specs:
        img = _reddish_patch(240, 320, redness, blur, dark)
        img_path = out / f"{name}.png"
        cv2.imwrite(str(img_path), img)
        (out / f"{name}.json").write_text(json.dumps(meta, indent=2))
        print(f"[samples] wrote {img_path.name} + sidecar")

    print(f"[samples] done -> {out}")


if __name__ == "__main__":
    main()
