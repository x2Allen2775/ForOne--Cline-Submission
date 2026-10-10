"""Tests for the preprocessing stage (decode, crop, normalize, tensor)."""
import cv2
import numpy as np

from app.config import settings
from app.pipeline import preprocessing


def _reddish(h=240, w=320, val=200):
    canvas = np.full((h, w, 3), (120, 150, 190), dtype=np.uint8)  # skin-ish border
    y0, x0 = h // 4, w // 4
    patch = np.zeros((h // 2, w // 2, 3), dtype=np.uint8)
    patch[:, :, 2] = val
    patch[:, :, 1] = int(val * 0.3)
    patch[:, :, 0] = int(val * 0.3)
    canvas[y0 : y0 + h // 2, x0 : x0 + w // 2] = patch
    return canvas


def test_decode_roundtrip():
    img = _reddish()
    ok, buf = cv2.imencode(".png", img)
    assert ok
    decoded = preprocessing.decode_image(buf.tobytes())
    assert decoded.shape == img.shape


def test_decode_bad_bytes_raises():
    import pytest

    with pytest.raises(ValueError):
        preprocessing.decode_image(b"not an image")


def test_auto_crop_reduces_to_red_region():
    img = _reddish()
    crop = preprocessing.auto_crop_conjunctiva(img)
    assert crop.size > 0
    # cropped region should be smaller than original
    assert crop.shape[0] <= img.shape[0] and crop.shape[1] <= img.shape[1]


def test_to_model_tensor_shape_and_dtype():
    img = _reddish()
    t = preprocessing.to_model_tensor(img)
    assert t.shape == (3, settings.input_size, settings.input_size)
    assert t.dtype == np.float32


def test_full_preprocess_keys():
    out = preprocessing.preprocess(_reddish())
    assert {"cropped_bgr", "normalized_bgr", "tensor_chw"} <= set(out)
