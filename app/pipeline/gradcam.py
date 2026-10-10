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
    Locate a reasonable conv layer for CAM. For timm MobileNetV3 the final
    conv block works well; we search for the last Conv2d as a robust default.
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def _overlay_heatmap(base_bgr: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Overlay a [0,1] CAM onto a BGR image."""
    size = settings.input_size
    base = cv2.resize(base_bgr, (size, size), interpolation=cv2.INTER_AREA)
    cam_uint8 = np.uint8(255 * np.clip(cam, 0, 1))
    heat = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    return overlay


def png_base64(image_bgr: np.ndarray) -> str:
    """Encode a BGR image as a base64 PNG string."""
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to PNG-encode image.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


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

    Parameters
    ----------
    model:        the torch model (in eval mode).
    tensor_chw:   preprocessed CHW float32 array for the same image.
    base_bgr:     the (normalised) BGR image to overlay the heatmap on.
    device:       torch device the model lives on.
    target_class: optional class index to explain (defaults to argmax).
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except Exception as exc:  # pragma: no cover
        print(f"[gradcam] grad-cam unavailable ({exc}); skipping heatmap.")
        return None

    target_layer = _find_target_layer(model)
    if target_layer is None:  # pragma: no cover
        print("[gradcam] No Conv2d layer found; skipping heatmap.")
        return None

    input_tensor = (
        torch.from_numpy(np.ascontiguousarray(tensor_chw)).unsqueeze(0).to(device)
    )
    targets = (
        [ClassifierOutputTarget(target_class)] if target_class is not None else None
    )

    # Grad-CAM needs gradients; use CPU fallback if MPS lacks an op during backward.
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
            model.to(device)  # restore
        except Exception as exc2:  # pragma: no cover
            print(f"[gradcam] CPU retry failed ({exc2}); skipping heatmap.")
            return None

    overlay = _overlay_heatmap(base_bgr, grayscale_cam)
    return png_base64(overlay)
