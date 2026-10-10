"""
Quality Gate (plan §3, Hour 3).

Rejects unusable conjunctiva photos *before* they reach the model, preventing
false positives/negatives from blurry, over/under-exposed, or off-target images.

Checks performed:
  1. Blur     — variance of the Laplacian (low variance == out of focus).
  2. Exposure — mean luminance window + fraction of clipped (0 / 255) pixels.
  3. Tissue   — fraction of reddish (conjunctiva-like) pixels present.

All thresholds live in :mod:`app.config` so they can be tuned without code edits.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.config import settings
from app.schemas import QualityReport


def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """Coerce an input array to a 3-channel BGR uint8 image."""
    img = image
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _normalize_resolution(bgr: np.ndarray) -> np.ndarray:
    """
    Downscale so the longest edge == ``settings.qc_work_max_edge``.

    Blur (variance of Laplacian) is resolution-dependent: a huge 12-MP phone
    photo has each edge smeared across many pixels, which deflates the variance
    and would wrongly flag a sharp photo as blurry. Normalising to a standard
    working size makes the thresholds resolution-independent.
    """
    max_edge = settings.qc_work_max_edge
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return bgr
    scale = max_edge / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)


def laplacian_variance(gray: np.ndarray) -> float:
    """Return the variance of the Laplacian — a classic focus measure."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _reddish_tissue_fraction(bgr: np.ndarray) -> float:
    """Fraction of pixels that fall into the conjunctiva-like red HSV bands."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(
        hsv, np.array(settings.red_hsv_low1), np.array(settings.red_hsv_high1)
    )
    mask2 = cv2.inRange(
        hsv, np.array(settings.red_hsv_low2), np.array(settings.red_hsv_high2)
    )
    mask = cv2.bitwise_or(mask1, mask2)
    return float(np.count_nonzero(mask)) / float(mask.size)


def _glare_fraction(bgr: np.ndarray) -> float:
    """
    Fraction of pixels that look like specular glare (very bright AND desaturated).
    Flash/sunlight glare on the moist conjunctiva is a common failure mode
    (Anemia Loop spec §6.1, §11), so we detect and reject it.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    glare = (v >= 245) & (s <= 30)
    return float(np.count_nonzero(glare)) / float(glare.size)


def assess_quality(image: np.ndarray, modality: str = "eye") -> QualityReport:
    """
    Evaluate an image and decide whether it is fit for inference.

    Parameters
    ----------
    image:
        Image as a numpy array (BGR, RGB-as-BGR, grayscale, or RGBA). Callers in
        this project pass OpenCV BGR arrays.
    modality:
        'eye' (conjunctiva) applies the reddish-tissue + glare checks; 'palm'
        (palm/nail bed) skips those conjunctiva-specific checks.

    Returns
    -------
    QualityReport with per-check metrics and an overall ``passed`` flag.
    """
    bgr = _to_bgr_uint8(image)
    # Normalise resolution so blur/exposure metrics are size-independent
    # (e.g. 12-MP iPhone photos vs. small web images).
    bgr = _normalize_resolution(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    blur_var = laplacian_variance(gray)
    brightness = float(gray.mean())
    # Modality-specific blur threshold: palms are smooth skin with naturally
    # low edge content, so their threshold is much lower than conjunctiva's.
    blur_min = (
        settings.palm_blur_min_variance if modality == "palm"
        else settings.blur_min_variance
    )

    total = gray.size
    overexposed = float(np.count_nonzero(gray >= 250)) / total
    underexposed = float(np.count_nonzero(gray <= 5)) / total
    tissue = _reddish_tissue_fraction(bgr)
    glare = _glare_fraction(bgr)

    reasons: list[str] = []
    if blur_var < blur_min:
        reasons.append(
            f"Image too blurry (focus={blur_var:.1f} < {blur_min:.0f})."
        )
    if brightness < settings.brightness_min:
        reasons.append(
            f"Image too dark (brightness={brightness:.1f} < {settings.brightness_min:.0f})."
        )
    if brightness > settings.brightness_max:
        reasons.append(
            f"Image too bright (brightness={brightness:.1f} > {settings.brightness_max:.0f})."
        )
    if (overexposed + underexposed) > settings.exposure_clip_fraction_max:
        reasons.append(
            "Excessive clipping (over={:.0%}, under={:.0%}).".format(
                overexposed, underexposed
            )
        )
    # Conjunctiva-specific checks only apply to the 'eye' modality.
    if modality == "eye":
        if tissue < settings.min_tissue_fraction:
            reasons.append(
                f"Little conjunctiva tissue detected (red fraction={tissue:.2%} < "
                f"{settings.min_tissue_fraction:.2%}). Aim the camera at the inner eyelid."
            )
        if glare > settings.glare_fraction_max:
            reasons.append(
                f"Too much glare/reflection (glare={glare:.1%} > "
                f"{settings.glare_fraction_max:.0%}). Avoid direct flash; use diffuse light."
            )

    return QualityReport(
        passed=len(reasons) == 0,
        blur_variance=round(blur_var, 2),
        brightness_mean=round(brightness, 2),
        overexposed_fraction=round(overexposed, 4),
        underexposed_fraction=round(underexposed, 4),
        glare_fraction=round(glare, 4),
        tissue_fraction=round(tissue, 4),
        reasons=reasons,
    )
