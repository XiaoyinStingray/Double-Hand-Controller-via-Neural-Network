"""Train a V2 gesture classifier on precomputed windows.

Usage:
    python scripts/train_gesture.py --model mlp --epochs 30
    python scripts/train_gesture.py --model transformer --epochs 60
    python scripts/train_gesture.py --model gru --limit-batches 20 --epochs 1  # smoke

The official test split is NOT evaluated here; use a separate eval step later.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.gesture_dataset import GestureWindowDataset  # noqa: E402
from src.models.gesture import build_model, count_parameters  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data" / "gesture")
    p.add_argument("--model", choices=["mlp", "gru", "tcn", "transformer"], default="mlp")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--limit-batches", type=int, default=0,
                   help="stop each epoch after N batches (smoke test)")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, limit_batches=0):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    n = 0
    preds, ys = [], []

    with torch.set_grad_enabled(train):
        for i, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * len(y)
            n += len(y)
            preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
            if limit_batches and i + 1 >= limit_batches:
                break

    if not preds:
        return 0.0, 0.0, 0.0, np.zeros(0), np.zeros(0)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    return (
        total_loss / max(n, 1),
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro", zero_division=0),
        y_true,
        y_pred,
    )


def save_confusion(y_true, y_pred, classes, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
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
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    meta = json.loads((args.data / "meta.json").read_text(encoding="utf-8"))
    classes = [meta["classes"][str(i)] for i in range(len(meta["classes"]))]
    T = int(meta["T"])
    in_dim = int(meta["feature_dim"])

    out = args.out or (ROOT / "runs" / "gesture" / args.model)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = GestureWindowDataset(args.data, "train", T=T)
    val_ds = GestureWindowDataset(args.data, "val", T=T)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = build_model(args.model, in_dim=in_dim, num_classes=len(classes)).to(device)
    n_params = count_parameters(model)

    weight = None
    if not args.no_class_weights:
        w = np.load(args.data / "class_weights_train.npy")
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"device {device} | model {args.model} | params {n_params/1e3:.1f}K")
    print(f"train {len(train_ds)} windows | val {len(val_ds)} windows | T={T} | "
          f"in_dim={in_dim} | classes={len(classes)}")
    print(f"class weights {'on' if weight is not None else 'off'} | out {out}")

    history = []
    best_f1 = -1.0
    best_epoch = -1
    best_path = out / "best.pt"
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, args.limit_batches)
        va_loss, va_acc, va_f1, _, _ = run_epoch(
            model, val_loader, criterion, device, None, args.limit_batches)
        scheduler.step()

        dt = time.perf_counter() - t0
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "train_f1": tr_f1, "val_loss": va_loss, "val_acc": va_acc,
            "val_f1": va_f1, "lr": optimizer.param_groups[0]["lr"], "seconds": dt,
        })
        print(f"epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"f1 {tr_f1:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f} "
              f"f1 {va_f1:.4f} | {dt:.1f}s")

        if va_f1 > best_f1:
            best_f1, best_epoch = va_f1, epoch
            patience_left = args.patience
            torch.save({
                "model": args.model,
                "state_dict": model.state_dict(),
                "classes": classes,
                "meta": {"T": T, "feature_dim": in_dim},
                "args": {k: str(v) for k, v in vars(args).items()},
            }, best_path)
        else:
            patience_left -= 1
            if args.patience and patience_left <= 0:
                print(f"early stop at epoch {epoch} (best {best_epoch}, f1 {best_f1:.4f})")
                break

    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # final report on val using the best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    _, va_acc, va_f1, y_true, y_pred = run_epoch(
        model, val_loader, criterion, device, None, 0)

    report = classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                   target_names=classes, digits=3, zero_division=0)
    (out / "report_val.txt").write_text(report, encoding="utf-8")
    save_confusion(y_true, y_pred, classes, out / "confusion_val.png")
    with open(out / "args.json", "w", encoding="utf-8") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)

    print(f"\nbest epoch {best_epoch} | val acc {va_acc:.4f} | val macro-F1 {va_f1:.4f}")
    print(report)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
