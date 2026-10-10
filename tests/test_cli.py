"""Smoke tests for the terminal CLI analyzer (headless, no pop-up window)."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "healthy_adult.png"


def _ensure_sample() -> Path:
    """Create a minimal reddish sample if the demo images aren't generated yet."""
    if SAMPLE.exists():
        return SAMPLE
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    h, w = 240, 320
    canvas = np.full((h, w, 3), (120, 150, 190), dtype=np.uint8)
    patch = np.zeros((h // 2, w // 2, 3), dtype=np.uint8)
    patch[:, :, 2] = 200
    for i in range(0, patch.shape[1], 3):
        patch[:, i] = (20, 20, 250)
    canvas[h // 4 : h // 4 + h // 2, w // 4 : w // 4 + w // 2] = patch
    cv2.imwrite(str(SAMPLE), canvas)
    return SAMPLE


def test_cli_headless_runs_and_saves_figure(tmp_path):
    img = _ensure_sample()
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.analyze_cli",
            "--image", str(img),
            "--age-months", "300",
            "--symptoms", "fatigue",
            "--no-window",
            "--out-dir", str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "ANEMIA-SCREEN · TRIAGE REPORT" in result.stdout
    assert "QUALITY GATE" in result.stdout
    # A report figure must have been written.
    figs = list(out_dir.glob("*_report.png"))
    assert figs, "expected a saved report figure"


def test_cli_missing_image_errors():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.analyze_cli", "--image", "does_not_exist.png", "--no-window"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 1
