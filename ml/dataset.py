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


def build_transforms(train: bool):
    """albumentations transforms (brightness/contrast/rotation/blur per plan)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    size = settings.input_size
    if train:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=int(size * 1.15)),
                A.PadIfNeeded(size, size, border_mode=0),
                A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0), ratio=(0.8, 1.25)),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                A.HueSaturationValue(10, 15, 10, p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.Normalize(mean=settings.norm_mean, std=settings.norm_std),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(size, size, border_mode=0),
            A.CenterCrop(size, size),
            A.Normalize(mean=settings.norm_mean, std=settings.norm_std),
            ToTensorV2(),
        ]
    )


class AnemiaDataset:
    """Torch-style dataset yielding (tensor, label_idx)."""

    def __init__(self, samples: list[Sample], train: bool):
        import cv2

        self._cv2 = cv2
        self.samples = samples
        self.tf = build_transforms(train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = self._cv2.imread(str(s.image_path), self._cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {s.image_path}")
        img = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB)
        out = self.tf(image=img)
        return out["image"], s.label_idx
