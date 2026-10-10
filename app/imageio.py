"""
Robust image loading for Anemia-Screen.

Phones (especially iPhones) save photos as **HEIC/HEIF**, which OpenCV's
``cv2.imread`` / ``cv2.imdecode`` cannot decode. This module provides loaders that:

  1. Try OpenCV first (fast path for jpg/png/bmp/webp).
  2. Fall back to Pillow (+ ``pillow-heif`` if installed) for HEIC/HEIF and any
     other format OpenCV misses.
  3. Honour EXIF orientation so sideways phone photos are upright.

All loaders return an OpenCV-style **BGR uint8** numpy array.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Register HEIF/HEIC support with Pillow if available (no hard dependency).
_HEIF_OK = False
try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    _HEIF_OK = True
except Exception:  # pragma: no cover - optional dependency
    _HEIF_OK = False

HEIC_SUFFIXES = {".heic", ".heif"}


def heif_available() -> bool:
    """Whether HEIC/HEIF decoding is available in this environment."""
    return _HEIF_OK


def _pillow_to_bgr(data: bytes) -> np.ndarray:
    """Decode arbitrary image bytes via Pillow -> BGR uint8 (EXIF-corrected)."""
    import io

    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)  # respect phone rotation
        im = im.convert("RGB")
        rgb = np.array(im)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def decode_image(data: bytes) -> np.ndarray:
    """
    Decode raw image bytes into a BGR uint8 array.

    Tries OpenCV first, then falls back to Pillow (handles HEIC/HEIF when
    ``pillow-heif`` is installed).
    """
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    # OpenCV failed (e.g. HEIC) -> try Pillow.
    try:
        return _pillow_to_bgr(data)
    except Exception as exc:
        hint = ""
        if not _HEIF_OK:
            hint = (
                " If this is an iPhone HEIC photo, install HEIC support with "
                "'pip install pillow-heif'."
            )
        raise ValueError(
            f"Could not decode image bytes (unsupported/corrupt file).{hint}"
        ) from exc


def read_image(path: str | Path) -> np.ndarray:
    """
    Read an image file from disk into a BGR uint8 array (HEIC-aware).

    Raises FileNotFoundError if the path does not exist, ValueError if the file
    cannot be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # For HEIC/HEIF go straight to Pillow; otherwise try OpenCV first.
    if path.suffix.lower() not in HEIC_SUFFIXES:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            return img

    return decode_image(path.read_bytes())
