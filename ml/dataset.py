"""
Dataset utilities (plan §1 Hour 1, §2 Hour 2).

Expected on-disk layout (you will populate ``data/raw`` after downloading the
``Eyes-defy-anemia`` / ``CP-AnemiC`` datasets):

    data/raw/
        labels.csv          # columns: image_path, hb (g/dL) OR label, patient_id
        images/...

Two labelling modes are supported:
  * ``hb`` column present  -> we derive the 3-class label with WHO thresholds.
  * ``label`` column present -> used directly (must be one of the class names).

IMPORTANT (anti-leakage): splits are **patient-wise**, so all images from a given
patient land in exactly one split. Mixing a patient's images across train/val is
the classic way to inflate medical-ML scores.

Downloading the data (do this yourself; both require (free) registration):
  * Eyes-defy-anemia : https://www.kaggle.com/datasets (search "Eyes defy anemia")
  * CP-AnemiC        : conjunctival pallor anemia dataset (search "CP-AnemiC")
Place images + a labels.csv under data/raw/ and point --data-dir at it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import settings


# ── WHO Hb -> class mapping (plan §4) ────────────────────────────────────────────
def hb_to_class(hb: float) -> str:
    """Map a hemoglobin value (g/dL) to the 3-class WHO-aligned label."""
    if hb < 7.0:
        return "Severe"
    if hb < 11.0:  # 7.0 - 10.9
        return "Mild/Moderate"
    return "Normal"


@dataclass
class Sample:
    image_path: Path
    label_idx: int
    patient_id: str


def _label_to_idx(label: str) -> int:
    names = list(settings.class_names)
    if label not in names:
        raise ValueError(f"Unknown label '{label}'. Expected one of {names}.")
    return names.index(label)


def load_manifest(data_dir: str | Path) -> list[Sample]:
    """Read ``labels.csv`` and resolve image paths + integer labels."""
    data_dir = Path(data_dir)
    csv_path = data_dir / "labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected {csv_path}. See module docstring for the required layout."
        )
    df = pd.read_csv(csv_path)

    if "label" in df.columns:
        labels = df["label"].astype(str)
    elif "hb" in df.columns:
        labels = df["hb"].astype(float).map(hb_to_class)
    else:
        raise ValueError("labels.csv must contain either a 'label' or 'hb' column.")

    if "patient_id" in df.columns:
        patient_ids = df["patient_id"].astype(str)
    else:
        # Fall back to filename stem as a pseudo patient id (no leakage control!).
        patient_ids = df["image_path"].map(lambda p: Path(p).stem)

    samples: list[Sample] = []
    for path, label, pid in zip(df["image_path"], labels, patient_ids):
        img_path = (data_dir / path) if not Path(path).is_absolute() else Path(path)
        samples.append(
            Sample(image_path=img_path, label_idx=_label_to_idx(label), patient_id=str(pid))
        )
    return samples


def load_manifest_csv(csv_path: str | Path) -> list[Sample]:
    """
    Read a unified manifest CSV produced by ``ml.build_manifest`` (columns:
    image_path [absolute], label, patient_id, ...). Rows with a missing image
    file are skipped.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {csv_path}. Run 'python -m ml.build_manifest' first."
        )
    df = pd.read_csv(csv_path)

    if "label" in df.columns:
        labels = df["label"].astype(str)
    elif "hb" in df.columns:
        labels = df["hb"].astype(float).map(hb_to_class)
    else:
        raise ValueError("Manifest must contain a 'label' or 'hb' column.")

    patient_ids = (
        df["patient_id"].astype(str) if "patient_id" in df.columns
        else df["image_path"].map(lambda p: Path(p).stem)
    )

    samples: list[Sample] = []
    missing = 0
    for path, label, pid in zip(df["image_path"], labels, patient_ids):
        img_path = Path(path)
        if not img_path.exists():
            missing += 1
            continue
        samples.append(
            Sample(image_path=img_path, label_idx=_label_to_idx(label), patient_id=str(pid))
        )
    if missing:
        print(f"[dataset] WARN: {missing} manifest rows skipped (image file missing).")
    return samples


def patient_wise_split(
    samples: list[Sample], val_frac: float = 0.2, seed: int = 42
) -> tuple[list[Sample], list[Sample]]:
    """Split samples by patient id so no patient appears in both splits."""
    rng = np.random.default_rng(seed)
    patients = sorted({s.patient_id for s in samples})
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_frac))
    val_patients = set(patients[:n_val])
    train = [s for s in samples if s.patient_id not in val_patients]
    val = [s for s in samples if s.patient_id in val_patients]
    return train, val


def _augment(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Geometric + illumination augmentation for TRAINING only.

    Deliberately applied BEFORE the serving preprocessing (crop → colour
    normalize → resize/tensor) so the model sees the exact same distribution at
    train time and serve time. Train/serve preprocessing mismatch is the
    classic way a model scores well in training but degenerates in production.
    """
    import cv2

    # Horizontal flip (50%).
    if rng.random() < 0.5:
        bgr = cv2.flip(bgr, 1)
    # Small rotation (±15°, 50%).
    if rng.random() < 0.5:
        angle = float(rng.uniform(-15, 15))
        h, w = bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        bgr = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    # Brightness/contrast jitter (±20%, 50%).
    if rng.random() < 0.5:
        alpha = 1.0 + float(rng.uniform(-0.2, 0.2))   # contrast
        beta = float(rng.uniform(-25, 25))             # brightness
        bgr = np.clip(bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    # Slight defocus (20%) — mimics imperfect focus without destroying detail.
    if rng.random() < 0.2:
        k = 3
        bgr = cv2.GaussianBlur(bgr, (k, k), 0)
    return bgr


class AnemiaDataset:
    """
    Torch-style dataset yielding (tensor, label_idx).

    Uses the EXACT serving preprocessing (``app.pipeline.preprocessing.preprocess``:
    modality-aware crop → gray-world + CLAHE colour normalization → resize →
    ImageNet-normalize → CHW tensor) so training and inference distributions match.
    Training additionally applies augmentation BEFORE that pipeline.
    """

    def __init__(self, samples: list[Sample], train: bool, modality: str = "eye",
                 seed: int = 42):
        # NB: do NOT store unpicklable modules on the instance (DataLoader workers).
        self.samples = samples
        self.train = train
        self.modality = modality
        self._rng = np.random.default_rng(seed if not train else seed + 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        import torch

        from app.imageio import read_image
        from app.pipeline import preprocessing as pp

        s = self.samples[i]
        img = read_image(s.image_path)  # HEIC-aware BGR loader
        if self.train:
            img = _augment(img, self._rng)
        pre = pp.preprocess(img, modality=self.modality)
        x = torch.from_numpy(np.ascontiguousarray(pre["tensor_chw"]))
        return x, s.label_idx
