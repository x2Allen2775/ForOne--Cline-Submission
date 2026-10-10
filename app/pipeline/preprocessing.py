"""
Preprocessing Module (plan §2, Hour 2).

Responsibilities:
  1. Decode raw image bytes into an OpenCV BGR array.
  2. Auto-crop to the conjunctiva region of interest (ROI) using an HSV red-mask,
     so the model focuses on the vascular bed rather than skin/background.
  3. Normalise colour/illumination (gray-world white balance + CLAHE on the L
     channel) to reduce lighting variance between phones/clinics.
  4. Produce a model-ready tensor (resized, ImageNet-normalised, CHW float32).

The ROI crop is intentionally conservative: if no convincing red region is found
we fall back to a centre crop so the pipeline never hard-fails here (the Quality
Gate is responsible for rejecting genuinely unusable frames).
"""
from __future__ import annotations

import cv2
import numpy as np

from app.config import settings


def decode_image(data: bytes) -> np.ndarray:
    """
    Decode raw image bytes into a BGR uint8 array.

    Delegates to :func:`app.imageio.decode_image`, which handles HEIC/HEIF
    (iPhone photos) and other formats OpenCV can't read.
    """
    from app.imageio import decode_image as _decode

    return _decode(data)


def _red_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(
        hsv, np.array(settings.red_hsv_low1), np.array(settings.red_hsv_high1)
    )
    mask2 = cv2.inRange(
        hsv, np.array(settings.red_hsv_low2), np.array(settings.red_hsv_high2)
    )
    mask = cv2.bitwise_or(mask1, mask2)
    # Clean up speckle noise and fill small holes.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def auto_crop_conjunctiva(bgr: np.ndarray) -> np.ndarray:
    """
    Crop to the largest reddish region (approx. the conjunctiva). Falls back to a
    centred 70% crop when no sufficiently large red blob is found.
    """
    h, w = bgr.shape[:2]
    mask = _red_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area_frac = cv2.contourArea(largest) / float(h * w)
        if area_frac >= settings.min_tissue_fraction:
            x, y, bw, bh = cv2.boundingRect(largest)
            pad_x = int(bw * settings.crop_padding)
            pad_y = int(bh * settings.crop_padding)
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(w, x + bw + pad_x)
            y1 = min(h, y + bh + pad_y)
            crop = bgr[y0:y1, x0:x1]
            if crop.size > 0:
                return crop

    # Fallback: centre crop (70% of each dimension).
    ch, cw = int(h * 0.7), int(w * 0.7)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return bgr[y0 : y0 + ch, x0 : x0 + cw]


def normalize_color(bgr: np.ndarray) -> np.ndarray:
    """
    Reduce illumination/colour variance:
      * Gray-world white balance (per-channel mean equalisation).
      * CLAHE on the L channel in LAB space for local contrast.
    """
    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    gray_mean = float(means.mean()) + 1e-6
    scale = gray_mean / (means + 1e-6)
    img = np.clip(img * scale, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def to_model_tensor(bgr: np.ndarray) -> np.ndarray:
    """
    Resize + ImageNet-normalise + convert to CHW float32 (no batch dim).

    Returns a numpy array; callers wrap it in a torch tensor so this module stays
    import-light and torch-optional for pure image utilities.
    """
    size = settings.input_size
    resized = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array(settings.norm_mean, dtype=np.float32)
    std = np.array(settings.norm_std, dtype=np.float32)
    rgb = (rgb - mean) / std
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32)  # CHW


def _center_crop(bgr: np.ndarray, frac: float = 0.85) -> np.ndarray:
    """Simple centred crop — used for non-conjunctiva modalities (palm/nail)."""
    h, w = bgr.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return bgr[y0:y0 + ch, x0:x0 + cw]


def preprocess(bgr: np.ndarray, modality: str = "eye") -> dict[str, np.ndarray]:
    """
    Full preprocessing pass. Returns a dict with intermediate artefacts so the
    demo notebook can visualise each stage.

    modality:
      * 'eye'  -> auto-crop the reddish conjunctiva ROI (default).
      * 'palm' -> centre-crop only (palm/nail bed is not reddish-masked).
    """
    if modality == "palm":
        cropped = _center_crop(bgr)
    else:
        cropped = auto_crop_conjunctiva(bgr)
    normalized = normalize_color(cropped)
    tensor = to_model_tensor(normalized)
    return {
        "cropped_bgr": cropped,
        "normalized_bgr": normalized,
        "tensor_chw": tensor,
    }
