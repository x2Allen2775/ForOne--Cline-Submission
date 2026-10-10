"""
Export a trained checkpoint to ONNX (plan §2 — ONNX inference portability).

Usage:
    python -m ml.export_onnx
    python -m ml.export_onnx --checkpoint ml/checkpoints/anemia_mobilenetv3.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.config import settings
from app.pipeline.inference import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Export anemia model to ONNX.")
    ap.add_argument("--checkpoint", default=settings.checkpoint_path)
    ap.add_argument("--output", default=settings.onnx_path)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = build_model()
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        print(f"[export] loaded checkpoint {ckpt_path}")
    else:
        print(f"[export] WARNING: no checkpoint at {ckpt_path}; exporting demo weights.")
    model.eval()

    dummy = torch.randn(1, 3, settings.input_size, settings.input_size)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"[export] wrote ONNX model -> {out_path}")

    # Optional sanity check with onnxruntime if available.
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        logits = sess.run(None, {"input": dummy.numpy().astype(np.float32)})[0]
        print(f"[export] onnxruntime check OK, output shape = {logits.shape}")
    except Exception as exc:  # pragma: no cover
        print(f"[export] onnxruntime check skipped ({exc}).")


if __name__ == "__main__":
    main()
