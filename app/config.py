"""
Central configuration for the Anemia-Screen pipeline.

All tunable thresholds live here so the Quality Gate, inference, and clinical
logic can be adjusted without touching business code. Values can be overridden
with environment variables (prefix ``ANEMIA_``) or a local ``.env`` file.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
CHECKPOINT_DIR = PROJECT_ROOT / "ml" / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


class Settings(BaseSettings):
    """Runtime configuration (env-overridable)."""

    model_config = SettingsConfigDict(
        env_prefix="ANEMIA_",
        env_file=".env",
        extra="ignore",
    )

    # ── Class taxonomy (WHO-aligned, see plan §4) ──
    # Index order is fixed; the model head outputs logits in this order.
    class_names: tuple[str, str, str] = ("Normal", "Mild/Moderate", "Severe")

    # ── Model / inference ──
    model_arch: str = "mobilenetv3_large_100"
    input_size: int = 224
    # Default (eye) checkpoint — kept for backwards compatibility.
    checkpoint_path: str = str(CHECKPOINT_DIR / "anemia_eye_mobilenetv3.pt")
    onnx_path: str = str(CHECKPOINT_DIR / "anemia_eye_mobilenetv3.onnx")
    # Per-modality checkpoints (eye = conjunctiva, palm = palm/nail bed).
    eye_checkpoint_path: str = str(CHECKPOINT_DIR / "anemia_eye_mobilenetv3.pt")
    palm_checkpoint_path: str = str(CHECKPOINT_DIR / "anemia_palm_mobilenetv3.pt")
    # Prefer MPS on Apple Silicon, else CUDA, else CPU. "auto" resolves at runtime.
    device: str = "auto"

    # ── Quality Gate thresholds (plan §3, Hour 3) ──
    # Downscale the longest edge to this before measuring quality, so blur/exposure
    # metrics are resolution-independent (12-MP phone photos vs. small images).
    qc_work_max_edge: int = 1024
    # Variance of the Laplacian below this => image considered blurry.
    # Calibrated to the training data distribution (p1≈17, p5≈22, median≈1500).
    blur_min_variance: float = 50.0
    # PALM modality: palms are smooth skin with naturally low edge content
    # (measured median blur variance ≈ 22.6, p5 ≈ 3.6), so a much lower,
    # modality-specific threshold is required.
    palm_blur_min_variance: float = 3.0
    # Mean luminance (0-255) must sit inside this window.
    # Clinical conjunctiva crops (CP-AnemiC) are dark; p5 brightness ≈ 17.
    brightness_min: float = 15.0
    brightness_max: float = 215.0
    # Fraction of over/under-exposed pixels tolerated before rejection.
    # Conjunctiva crops often sit on a BLACK background (up to ~80% dark pixels
    # by design), so this is calibrated high; a fully-black frame still fails.
    exposure_clip_fraction_max: float = 0.85
    # Fraction of specular-glare pixels tolerated before rejection.
    glare_fraction_max: float = 0.08
    # Minimum fraction of the frame that must look like conjunctiva (reddish) tissue.
    min_tissue_fraction: float = 0.02

    # ── Preprocessing ──
    # HSV ranges (OpenCV scale: H 0-179, S/V 0-255) that isolate the reddish
    # vascular conjunctiva so we can auto-crop to the region of interest.
    red_hsv_low1: tuple[int, int, int] = (0, 40, 40)
    red_hsv_high1: tuple[int, int, int] = (12, 255, 255)
    red_hsv_low2: tuple[int, int, int] = (168, 40, 40)
    red_hsv_high2: tuple[int, int, int] = (179, 255, 255)
    crop_padding: float = 0.08  # pad the detected ROI by this fraction each side

    # ── ImageNet normalization (timm/MobileNetV3 defaults) ──
    norm_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


settings = get_settings()
