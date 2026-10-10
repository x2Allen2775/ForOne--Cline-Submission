"""
Explainability (plan §4) — Grad-CAM heatmaps.

Produces a human-readable heatmap showing *where* the model looked when deciding
the anemia risk class. The heatmap is overlaid on the (preprocessed) conjunctiva
image and returned as a base64-encoded PNG so it can travel inside the JSON
triage report — no filesystem needed.

Uses the ``grad-cam`` package (imported as ``pytorch_grad_cam``). If the package
or a suitable target layer is unavailable, we degrade gracefully and return None
rather than breaking the pipeline.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
import torch

from app.config import settings


def _find_target_layer(model: torch.nn.Module):
    """
    Locate a reasonable conv layer for CAM.
    For timm MobileNetV3 the final convolutional block in model.blocks
    retains the 2D spatial feature map before global_pool collapses it to 1x1.
    """
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        last_block = model.blocks[-1]
        last_conv = None
        for m in last_block.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        if last_conv is not None:
            return last_conv
        return last_block

    # Fallback: scan all Conv2d modules, choosing the last one before pooling
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    if len(convs) >= 2:
        return convs[-2]
    return convs[-1] if convs else None


def _overlay_heatmap(base_bgr: np.ndarray, cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Overlay a [0,1] CAM onto a BGR image. Returns (overlay_bgr, heat_bgr)."""
    size = settings.input_size
    base = cv2.resize(base_bgr, (size, size), interpolation=cv2.INTER_AREA)
    cam_min = float(np.min(cam))
    cam_max = float(np.max(cam))
    if cam_max > cam_min:
        cam_norm = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = np.clip(cam, 0, 1)
    cam_uint8 = np.uint8(255 * np.clip(cam_norm, 0, 1))
    heat = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    return overlay, heat


def png_base64(image_bgr: np.ndarray) -> str:
    """Encode a BGR image as a base64 PNG string."""
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to PNG-encode image.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def generate_gradcam_artifacts(
    model: torch.nn.Module,
    tensor_chw: np.ndarray,
    base_bgr: np.ndarray,
    device: torch.device,
    target_class: int | None = None,
) -> dict:
    """
    Compute Grad-CAM and return rich artifacts:
      - overlay_bgr: base image + heatmap overlay
      - heatmap_bgr: pure color-mapped heatmap
      - overlay_png_b64: base64 encoded overlay PNG
      - heatmap_png_b64: base64 encoded heatmap PNG
      - cam_raw: 2D float array (0..1)
    """
    result = {
        "overlay_bgr": None,
        "heatmap_bgr": None,
        "overlay_png_b64": None,
        "heatmap_png_b64": None,
        "cam_raw": None,
    }
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except Exception as exc:  # pragma: no cover
        print(f"[gradcam] grad-cam unavailable ({exc}); skipping heatmap.")
        return result

    target_layer = _find_target_layer(model)
    if target_layer is None:  # pragma: no cover
        print("[gradcam] No Conv2d layer found; skipping heatmap.")
        return result

    input_tensor = (
        torch.from_numpy(np.ascontiguousarray(tensor_chw)).unsqueeze(0).to(device)
    )
    targets = (
        [ClassifierOutputTarget(target_class)] if target_class is not None else None
    )

    try:
        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    except Exception as exc:
        print(f"[gradcam] primary device failed ({exc}); retrying on CPU.")
        try:
            model_cpu = model.to("cpu")
            input_cpu = input_tensor.to("cpu")
            with GradCAM(model=model_cpu, target_layers=[_find_target_layer(model_cpu)]) as cam:
                grayscale_cam = cam(input_tensor=input_cpu, targets=targets)[0]
            model.to(device)
        except Exception as exc2:  # pragma: no cover
            print(f"[gradcam] CPU retry failed ({exc2}); skipping heatmap.")
            return result

    overlay, heat = _overlay_heatmap(base_bgr, grayscale_cam)
    result["overlay_bgr"] = overlay
    result["heatmap_bgr"] = heat
    result["overlay_png_b64"] = png_base64(overlay)
    result["heatmap_png_b64"] = png_base64(heat)
    result["cam_raw"] = grayscale_cam
    return result


def generate_gradcam(
    model: torch.nn.Module,
    tensor_chw: np.ndarray,
    base_bgr: np.ndarray,
    device: torch.device,
    target_class: int | None = None,
) -> str | None:
    """
    Compute a Grad-CAM overlay and return it as a base64 PNG, or ``None`` if the
    computation cannot be performed.
    """
    arts = generate_gradcam_artifacts(
        model=model,
        tensor_chw=tensor_chw,
        base_bgr=base_bgr,
        device=device,
        target_class=target_class,
    )
    return arts["overlay_png_b64"]
