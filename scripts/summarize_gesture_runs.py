"""Summarize gesture training runs and plot validation curves.

Usage:
    python scripts/summarize_gesture_runs.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def load_history(path: Path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            if k != "epoch":
                r[k] = float(r[k])
        r["epoch"] = int(r["epoch"])
    return rows


def eval_metric(short: str, which: str):
    path = RUNS / "eval" / f"{short}_{which}_val" / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(p for p in RUNS.glob("gesture_*") if (p / "history.csv").exists())
    if not models:
        print("no runs found")
        return 0

    table = []
    curves = {}
    for m in models:
        rows = load_history(m / "history.csv")
        best = max(rows, key=lambda r: r["val_f1"])
        short = m.name.replace("gesture_", "")
        table.append({
            "model": short,
            "epoch": best["epoch"],
            "val_acc": best["val_acc"],
            "val_f1": best["val_f1"],
            "gt": eval_metric(short, "gt"),
            "mp": eval_metric(short, "mph"),
        })
        curves[short] = rows

    print(f"{'model':12s} {'best':>4s} {'val_acc':>8s} {'val_F1':>8s} "
          f"{'GT_acc':>8s} {'GT_F1':>8s} {'MP_acc':>8s} {'MP_F1':>8s}")
    for r in table:
        gt = r["gt"] or {}
        mp = r["mp"] or {}
        print(f"{r['model']:12s} {r['epoch']:4d} {r['val_acc']:8.4f} {r['val_f1']:8.4f} "
              f"{gt.get('accuracy', float('nan')):8.4f} "
              f"{gt.get('macro_f1', float('nan')):8.4f} "
              f"{mp.get('accuracy', float('nan')):8.4f} "
              f"{mp.get('macro_f1', float('nan')):8.4f}")

    for metric, fname in (("val_f1", "gesture_curves_f1.png"),
                          ("val_acc", "gesture_curves_acc.png")):
        plt.figure(figsize=(8, 5))
        for name, rows in curves.items():
            plt.plot([r["epoch"] for r in rows], [r[metric] for r in rows],
                     label=name, linewidth=1.6)
        plt.xlabel("epoch")
        plt.ylabel("val macro-F1" if metric == "val_f1" else "val accuracy")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(RUNS / fname, dpi=150)
        plt.close()
        print("saved", RUNS / fname)

    lines = ["| model | best epoch | val acc | val macro-F1 | GT acc | GT F1 | MP acc | MP F1 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in table:
        gt = r["gt"] or {}
        mp = r["mp"] or {}
        lines.append(
            f"| {r['model']} | {r['epoch']} | {r['val_acc']:.4f} | {r['val_f1']:.4f} | "
            f"{gt.get('accuracy', 0):.4f} | {gt.get('macro_f1', 0):.4f} | "
            f"{mp.get('accuracy', 0):.4f} | {mp.get('macro_f1', 0):.4f} |"
        )
    (RUNS / "gesture_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved", RUNS / "gesture_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
