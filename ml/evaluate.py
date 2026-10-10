"""
Quick evaluation: run a TRAINED model over its validation split using the SAME
preprocessing path as serving (orchestrator), and report an honest
confusion matrix + sensitivity/specificity per modality.

Usage:
    python -m ml.evaluate --modality palm --max-images 200
    python -m ml.evaluate --modality eye  --max-images 200
"""
from __future__ import annotations

import argparse

import numpy as np

from app.config import PROJECT_ROOT, settings
from app.pipeline import preprocessing, quality_gate
from app.pipeline.inference import get_classifier
from ml.dataset import load_manifest_csv, patient_wise_split


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a trained anemia model.")
    ap.add_argument("--modality", choices=["eye", "palm"], default="palm")
    ap.add_argument("--max-images", type=int, default=200,
                    help="Cap evaluation images for speed.")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = PROJECT_ROOT / "data" / "manifests" / f"{args.modality}.csv"
    samples = load_manifest_csv(manifest)
    train_s, val_s = patient_wise_split(samples, args.val_frac, args.seed)
    if args.max_images and len(val_s) > args.max_images:
        # Random representative sample — NOT the first N (manifests are sorted
        # by label, so a head slice would be heavily class-skewed).
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(val_s), args.max_images, replace=False)
        val_s = [val_s[i] for i in sorted(idx)]
    print(f"[eval] modality={args.modality} val images={len(val_s)}")

    from app.imageio import read_image

    clf = get_classifier(modality=args.modality)
    print(f"[eval] model trained={clf.trained} device={clf.device}")
    if not clf.trained:
        print("[eval] ERROR: model is on untrained demo weights — train first!")
        return 1

    names = list(settings.class_names)
    cm = np.zeros((len(names), len(names)), dtype=int)  # rows=true, cols=pred
    rejected = 0

    for s in val_s:
        try:
            img = read_image(s.image_path)
        except Exception:
            continue
        q = quality_gate.assess_quality(img, modality=args.modality)
        if not q.passed:
            rejected += 1
            continue
        pre = preprocessing.preprocess(img, modality=args.modality)
        vision = clf.predict(pre["tensor_chw"])
        pred = names.index(vision.risk_class.value)
        cm[s.label_idx, pred] += 1

    print(f"\n[eval] confusion matrix (rows=true, cols=pred), order={names}")
    print(cm)
    # Anemic = Mild/Moderate + Severe (positive class).
    tp = cm[1:, 1:].sum()
    fn = cm[1:, 0].sum()
    fp = cm[0, 1:].sum()
    tn = cm[0, 0]
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    print(f"\n[eval] anemic-positive: sensitivity={sens:.3f} ({tp}/{tp+fn})  "
          f"specificity={spec:.3f} ({tn}/{tn+fp})")
    print(f"[eval] quality-gate rejections: {rejected}/{len(val_s)}")
    print("[eval] NOTE: tune thresholds on a held-out set; report honestly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
