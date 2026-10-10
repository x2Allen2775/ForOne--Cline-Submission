"""
Training script (plan §4, Hours 2-4). Colab/Kaggle/local friendly.

Trains MobileNetV3 (timm) for 3-class anemia classification with:
  * patient-wise train/val split (anti-leakage),
  * albumentations augmentation,
  * class-weighted cross-entropy (handles imbalance),
  * best-checkpoint saving to ``settings.checkpoint_path``.

Once a checkpoint is written here, the inference pipeline picks it up
automatically (no code change). Until then the API runs on demo weights.

Usage:
    python -m ml.train --data-dir data/raw --epochs 15 --batch-size 32
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.config import settings
from app.pipeline.inference import build_model, resolve_device
from ml.dataset import AnemiaDataset, load_manifest, patient_wise_split


def _class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [
        total / (num_classes * max(1, counts.get(c, 0))) for c in range(num_classes)
    ]
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(crit(logits, y)) * x.size(0)
            correct += int((logits.argmax(1) == y).sum())
            total += x.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train anemia MobileNetV3 classifier.")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device()
    print(f"[train] device = {device}")

    samples = load_manifest(args.data_dir)
    train_s, val_s = patient_wise_split(samples, args.val_frac, args.seed)
    print(f"[train] {len(train_s)} train / {len(val_s)} val samples "
          f"({len({s.patient_id for s in samples})} patients)")

    train_ds = AnemiaDataset(train_s, train=True)
    val_ds = AnemiaDataset(val_s, train=False)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model().to(device)
    weights = _class_weights([s.label_idx for s in train_s], settings.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = Path(settings.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += float(loss) * x.size(0)
        scheduler.step()

        train_loss = running / max(1, len(train_ds))
        val_loss, val_acc = evaluate(model, val_dl, device)
        print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": settings.model_arch,
                    "classes": list(settings.class_names),
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"[train] saved best checkpoint (val_acc={val_acc:.4f}) -> {ckpt_path}")

    print(f"[train] done. best val_acc = {best_acc:.4f}")


if __name__ == "__main__":
    main()
