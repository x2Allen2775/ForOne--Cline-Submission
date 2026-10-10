"""Tests for the Quality Gate (blur / brightness / tissue checks)."""
import cv2
import numpy as np

from app.pipeline.quality_gate import assess_quality, laplacian_variance


def _reddish(h=200, w=260, val=200, blur=0, scale=1.0):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 2] = val  # strong red channel (BGR)
    img[:, :, 1] = int(val * 0.3)
    img[:, :, 0] = int(val * 0.3)
    # Dense sharp texture so a focused image has high Laplacian variance.
    for i in range(0, w, 3):
        img[:, i] = (20, 20, 250)
    for j in range(0, h, 5):
        img[j, :] = (10, 10, 180)
    img = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    if blur:
        k = blur * 2 + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


def test_good_image_passes():
    report = assess_quality(_reddish())
    assert report.passed, report.reasons
    assert report.tissue_fraction > 0.1


def test_blurry_image_rejected():
    report = assess_quality(_reddish(blur=9))
    assert not report.passed
    assert any("blurry" in r.lower() for r in report.reasons)


def test_dark_image_rejected():
    report = assess_quality(_reddish(scale=0.1))
    assert not report.passed
    assert any("dark" in r.lower() for r in report.reasons)


def test_no_tissue_rejected():
    # A plain gray frame has no reddish conjunctiva region.
    gray = np.full((200, 260, 3), 128, dtype=np.uint8)
    # add texture so it's not rejected for blur
    gray[::4, :] = 60
    report = assess_quality(gray)
    assert not report.passed
    assert any("tissue" in r.lower() for r in report.reasons)


def test_laplacian_variance_monotonic_with_blur():
    sharp = cv2.cvtColor(_reddish(blur=0), cv2.COLOR_BGR2GRAY)
    blurry = cv2.cvtColor(_reddish(blur=9), cv2.COLOR_BGR2GRAY)
    assert laplacian_variance(sharp) > laplacian_variance(blurry)


def test_large_sharp_image_passes_after_resolution_normalization():
    # A big (phone-sized) but sharp image must NOT be wrongly flagged as blurry;
    # the quality gate downscales to a standard working size first. We add fine
    # random reddish noise so the image stays sharp at any scale (like a real photo).
    rng = np.random.default_rng(0)
    big = np.zeros((3000, 4000, 3), dtype=np.uint8)
    big[:, :, 2] = rng.integers(140, 230, size=(3000, 4000))  # red channel texture
    big[:, :, 1] = rng.integers(40, 90, size=(3000, 4000))
    big[:, :, 0] = rng.integers(40, 90, size=(3000, 4000))
    report = assess_quality(big)
    assert report.passed, report.reasons
    assert report.blur_variance >= 100
