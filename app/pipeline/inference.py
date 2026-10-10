"""
Inference Module (plan §4-§5).

Loads a MobileNetV3 classifier (via ``timm``) with a 3-class head and runs it on
the best available device (MPS on Apple Silicon > CUDA > CPU).

Graceful degradation
--------------------
Before the real model is trained, there is no checkpoint on disk. Rather than
crash, the ``AnemiaClassifier`` loads ImageNet-pretrained MobileNetV3 weights with
a freshly-initialised 3-class head and flags ``trained=False`` in every result.
This lets the **entire pipeline run end-to-end today**; once you train and drop a
checkpoint at ``settings.checkpoint_path``, it is picked up automatically and
``trained`` becomes True — no code change required.

An ONNX path is also supported for portable/fast inference.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from app.config import settings
from app.schemas import RiskClass, VisionPrediction


def resolve_device(pref: str | None = None) -> torch.device:
    """Pick the best torch device."""
    pref = pref or settings.device
    if pref and pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes: int | None = None) -> torch.nn.Module:
    """Create a MobileNetV3 model with a ``num_classes`` head via timm."""
    import timm

    num_classes = num_classes or settings.num_classes
    model = timm.create_model(
        settings.model_arch,
        pretrained=True,  # ImageNet init; overwritten if a checkpoint is loaded
        num_classes=num_classes,
    )
    return model


def _checkpoint_for(modality: str) -> Path:
    """Resolve the checkpoint path for a given modality (eye/palm)."""
    if modality == "palm":
        return Path(settings.palm_checkpoint_path)
    return Path(settings.eye_checkpoint_path)


class AnemiaClassifier:
    """MobileNetV3-based anemia risk classifier with graceful fallback.

    One instance per modality ('eye' = conjunctiva, 'palm' = palm/nail bed).
    """

    def __init__(self, device: str | None = None, modality: str = "eye"):
        self.device = resolve_device(device)
        self.modality = modality
        self.class_names = list(settings.class_names)
        self.trained = False
        self._model: torch.nn.Module | None = None
        self._load()

    # ── loading ────────────────────────────────────────────────────────────────
    def _load(self) -> None:
        ckpt = _checkpoint_for(self.modality)
        self._model = build_model()
        if ckpt.exists():
            try:
                state = torch.load(ckpt, map_location="cpu")
                # Support both raw state_dicts and {"model": state_dict, ...}.
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                self._model.load_state_dict(state)
                self.trained = True
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[inference] WARN: failed to load checkpoint ({exc}); "
                      "falling back to untrained demo weights.")
                self.trained = False
        self._model.eval().to(self.device)

    # ── inference ──────────────────────────────────────────────────────────────
    @property
    def model(self) -> torch.nn.Module:
        assert self._model is not None
        return self._model

    @torch.no_grad()
    def predict(self, tensor_chw: np.ndarray) -> VisionPrediction:
        """
        Run a forward pass on a single preprocessed CHW float32 array.
        """
        x = torch.from_numpy(np.ascontiguousarray(tensor_chw)).unsqueeze(0)
        x = x.to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        idx = int(np.argmax(probs))
        prob_map = {
            name: round(float(p), 4) for name, p in zip(self.class_names, probs)
        }
        # p_anemic = probability of any anemia (everything that is not "Normal").
        p_anemic = float(
            sum(p for name, p in zip(self.class_names, probs) if name != "Normal")
        )
        return VisionPrediction(
            risk_class=RiskClass(self.class_names[idx]),
            confidence=round(float(probs[idx]), 4),
            p_anemic=round(p_anemic, 4),
            probabilities=prob_map,
            model_trained=self.trained,
        )


# ── module-level singletons (lazy, one per modality) ─────────────────────────────
_classifiers: dict[str, AnemiaClassifier] = {}


def get_classifier(modality: str = "eye") -> AnemiaClassifier:
    """Return a lazily-instantiated shared classifier for the given modality."""
    if modality not in _classifiers:
        _classifiers[modality] = AnemiaClassifier(modality=modality)
    return _classifiers[modality]
