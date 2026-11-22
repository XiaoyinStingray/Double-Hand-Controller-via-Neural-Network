"""Evaluate a trained gesture checkpoint on any gesture dataset root.

Useful for the GT -> MediaPipe robustness comparison:
    python scripts/eval_gesture.py --checkpoint runs/gesture_transformer/best.pt \
        --data data/gesture_mpholistic --split val

Outputs metrics.json, report.txt, confusion.png, preds.npz in --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.gesture_dataset import GestureWindowDataset  # noqa: E402
from src.models.gesture import build_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit-batches", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def save_confusion(cm, classes, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes,
                annot_kws={"fontsize": 6})
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt["model"]
    classes = ckpt["classes"]
    T = int(ckpt["meta"]["T"])
    in_dim = int(ckpt["meta"]["feature_dim"])
    num_classes = len(classes)

    model = build_model(model_name, in_dim=in_dim, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = GestureWindowDataset(args.data, args.split, T=T)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    preds, ys = [], []
    probs = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            logits = model(x.to(device, non_blocking=True))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float16))
            ys.append(y.numpy())
            if args.limit_batches and i + 1 >= args.limit_batches:
                break

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    out = args.out or (ROOT / "runs" / "eval" /
                       f"{args.checkpoint.parent.name}_{args.data.name}_{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, labels=list(range(num_classes)),
                                   target_names=classes, digits=3, zero_division=0)
    (out / "report.txt").write_text(report, encoding="utf-8")
    save_confusion(cm, classes, out / "confusion.png")
    np.savez(out / "preds.npz", y_true=y_true, y_pred=y_pred,
             probs=np.concatenate(probs),
             video_idx=ds.video_idx, start=ds.start, center=ds.center)
    metrics = {
        "checkpoint": str(args.checkpoint),
        "model": model_name,
        "data": str(args.data),
        "split": args.split,
        "windows": int(len(y_true)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"{args.checkpoint.parent.name} on {args.data.name}/{args.split}: "
          f"windows {len(y_true)} | acc {acc:.4f} | macro-F1 {macro_f1:.4f} | "
          f"weighted-F1 {weighted_f1:.4f}")
    print(report)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
