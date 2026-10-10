"""Tests for the robust image loader (OpenCV + Pillow/HEIC fallback)."""
import cv2
import numpy as np
import pytest

from app.imageio import decode_image, read_image


def _png_bytes(h=40, w=60):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 2] = 200
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_decode_standard_png_bytes():
    out = decode_image(_png_bytes())
    assert out.shape[2] == 3
    assert out.dtype == np.uint8


def test_decode_bad_bytes_raises_valueerror():
    with pytest.raises(ValueError):
        decode_image(b"definitely not an image")


def test_read_image_file(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(_png_bytes())
    out = read_image(p)
    assert out.shape[2] == 3


def test_read_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_image("/no/such/file.png")


def test_pillow_fallback_reads_png(tmp_path):
    # Force the Pillow path by decoding bytes OpenCV *can* read is covered above;
    # here we ensure decode_image handles a JPEG round-trip too.
    img = np.full((30, 30, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    out = decode_image(buf.tobytes())
    assert out.shape == (30, 30, 3)
