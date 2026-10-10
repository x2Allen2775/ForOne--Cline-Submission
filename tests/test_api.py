"""End-to-end API tests using FastAPI's TestClient (in-process, no server)."""
import io
import json

import cv2
import numpy as np
import pytest

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _png_bytes(redness=200, blur=0, scale=1.0, h=240, w=320):
    canvas = np.full((h, w, 3), (120, 150, 190), dtype=np.uint8)
    y0, x0 = h // 4, w // 4
    patch = np.zeros((h // 2, w // 2, 3), dtype=np.uint8)
    patch[:, :, 2] = redness
    patch[:, :, 1] = int(redness * 0.3)
    patch[:, :, 0] = int(redness * 0.3)
    # Dense, high-contrast vascular texture so a focused image has high
    # Laplacian variance (passes the blur gate).
    for i in range(0, patch.shape[1], 3):
        patch[:, i] = (20, 20, 250)
    for j in range(0, patch.shape[0], 5):
        patch[j, :] = (10, 10, 180)
    canvas[y0 : y0 + h // 2, x0 : x0 + w // 2] = patch
    canvas = np.clip(canvas.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    if blur:
        k = blur * 2 + 1
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)
    ok, buf = cv2.imencode(".png", canvas)
    assert ok
    return buf.tobytes()


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # Per-modality model status (eye + palm).
    assert "models" in body
    assert "eye" in body["models"]
    assert "palm" in body["models"]
    for m in body["models"].values():
        assert "trained" in m and "classes" in m


def test_root_serves_web_ui():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "Anemia-Screen" in r.text


def test_docs_served_with_local_assets():
    r = client.get("/docs")
    assert r.status_code == 200
    # Must reference the LOCAL swagger assets, not an external CDN.
    assert "/static/swagger-ui-bundle.js" in r.text
    assert "cdn.jsdelivr.net" not in r.text


def test_analyze_good_image_returns_ok_report():
    meta = {"age_months": 300, "sex": "female", "symptoms": ["fatigue"]}
    files = {"image": ("good.png", io.BytesIO(_png_bytes()), "image/png")}
    data = {"metadata": json.dumps(meta), "with_gradcam": "false"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["vision"]["risk_class"] in {"Normal", "Mild/Moderate", "Severe"}
    assert body["triage"]["urgency"] in {"High", "Medium", "Low"}
    assert body["quality"]["passed"] is True


def test_analyze_red_flag_forces_refer_now():
    meta = {"age_months": 300, "symptoms": ["breathlessness"]}
    files = {"image": ("good.png", io.BytesIO(_png_bytes()), "image/png")}
    data = {"metadata": json.dumps(meta), "with_gradcam": "false"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 200, r.text
    triage = r.json()["triage"]
    assert triage["urgency"] == "High"
    assert triage["action_code"] == "REFER_NOW"


def test_analyze_measured_hb_severe_refers_now():
    meta = {"age_months": 300, "measured_hb": 6.2}
    files = {"image": ("good.png", io.BytesIO(_png_bytes()), "image/png")}
    data = {"metadata": json.dumps(meta), "with_gradcam": "false"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 200, r.text
    triage = r.json()["triage"]
    assert triage["action_code"] == "REFER_NOW"
    assert triage["hb_band"] == "severe"


def test_analyze_blurry_image_rejected():
    files = {"image": ("blur.png", io.BytesIO(_png_bytes(blur=9)), "image/png")}
    data = {"metadata": "{}", "with_gradcam": "false"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "rejected"
    assert body["quality"]["passed"] is False


def test_analyze_bad_metadata_json_returns_422():
    files = {"image": ("good.png", io.BytesIO(_png_bytes()), "image/png")}
    data = {"metadata": "{not json}", "with_gradcam": "false"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 422


def test_analyze_gradcam_included_when_requested():
    files = {"image": ("good.png", io.BytesIO(_png_bytes()), "image/png")}
    data = {"metadata": "{}", "with_gradcam": "true"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    if body["status"] == "ok":
        # heatmap may be None only if grad-cam unavailable; assert key present
        assert "gradcam_png_base64" in body


def test_analyze_palm_modality():
    files = {"image": ("palm.png", io.BytesIO(_png_bytes()), "image/png")}
    data = {"metadata": "{}", "with_gradcam": "false", "modality": "palm"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    # Palm modality skips the conjunctiva-tissue check, so a reddish synthetic
    # image should pass and produce a full vision result.
    assert body["status"] == "ok"
    assert body["vision"]["risk_class"] in {"Normal", "Mild/Moderate", "Severe"}


def test_analyze_bad_modality_returns_422():
    files = {"image": ("good.png", io.BytesIO(_png_bytes()), "image/png")}
    data = {"metadata": "{}", "with_gradcam": "false", "modality": "ear"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 422
