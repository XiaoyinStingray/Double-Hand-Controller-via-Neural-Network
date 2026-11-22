"""Continuous (segmental) evaluation from saved window predictions.

Reads preds.npz produced by scripts/eval_gesture.py and computes:
  * frame-level accuracy (all frames / hand-present frames)
  * predicted gesture segments after median smoothing
  * segmental F1 @ IoU (greedy matching)
  * false activations per minute

Usage:
    python scripts/eval_continuous.py \
        --preds runs/eval/transformer_gt_val/preds.npz \
        --data data/gesture --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.ipn_landmarks import load_segments  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--ann-root", type=Path,
                   default=ROOT / "data" / "raw" / "IPN_HandS_annotations")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--min-len", type=int, default=5,
                   help="minimum predicted segment length in frames")
    p.add_argument("--smooth", type=int, default=5,
                   help="median filter size over window predictions (odd)")
    p.add_argument("--conf", type=float, default=0.0,
                   help="drop predicted segments whose mean softmax confidence is below this")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def iou_1d(a, b) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return inter / union if union > 0 else 0.0


def frames_to_segments(pred: np.ndarray, min_len: int,
                       conf_thr: float = 0.0,
                       centers: np.ndarray | None = None,
                       win_conf: np.ndarray | None = None):
    segs = []
    n = len(pred)
    i = 0
    while i < n:
        c = int(pred[i])
        j = i
        while j + 1 < n and pred[j + 1] == c:
            j += 1
        if c != 0 and (j - i + 1) >= min_len:
            keep = True
            if conf_thr > 0 and centers is not None and win_conf is not None:
                m = (centers >= i) & (centers <= j)
                keep = bool(m.any()) and float(win_conf[m].mean()) >= conf_thr
            if keep:
                segs.append((c, i, j))
        i = j + 1
    return segs


def expand_to_frames(centers: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    if len(centers) == 0:
        return np.zeros(n, dtype=np.int16)
    frames = np.arange(n)
    pos = np.searchsorted(centers, frames)
    pos = np.clip(pos, 0, len(centers) - 1)
    left = np.clip(pos - 1, 0, len(centers) - 1)
    take_left = np.abs(frames - centers[left]) <= np.abs(centers[pos] - frames)
    return np.where(take_left, values[left], values[pos]).astype(np.int16)


def match_segments(gt, pred, iou_thr):
    pairs = []
    for i, (c1, s1, e1) in enumerate(gt):
        for j, (c2, s2, e2) in enumerate(pred):
            if c1 != c2:
                continue
            v = iou_1d((s1, e1), (s2, e2))
            if v >= iou_thr:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_g, used_p = set(), set()
    for _, i, j in pairs:
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
    return used_g, used_p


def main():
    args = parse_args()
    meta = json.loads((args.data / "meta.json").read_text(encoding="utf-8"))
    classes = [meta["classes"][str(i)] for i in range(len(meta["classes"]))]
    num_classes = len(classes)
    videos = meta["splits"][args.split]["videos"]

    offsets = np.load(args.data / f"offsets_{args.split}.npy")
    frame_labels = np.load(args.data / f"per_frame_labels_{args.split}.npy")
    presence = np.load(args.data / f"presence_{args.split}.npy")
    p = np.load(args.preds)
    y_true, y_pred = p["y_true"], p["y_pred"]
    probs = p["probs"] if "probs" in p.files else None
    vid, center = p["video_idx"], p["center"]

    assert len(videos) == len(offsets) - 1, "video list / offsets mismatch"

    gt_cnt = np.zeros(num_classes, dtype=np.int64)
    pred_cnt = np.zeros(num_classes, dtype=np.int64)
    tp_cnt = np.zeros(num_classes, dtype=np.int64)
    frame_correct = 0
    frame_total = 0
    hand_correct = 0
    hand_total = 0
    fp_total = 0
    total_frames = 0

    for vi, video in enumerate(videos):
        base = int(offsets[vi])
        n = int(offsets[vi + 1] - base)
        total_frames += n
        sel = np.where(vid == vi)[0]
        if len(sel) == 0:
            continue
        sel = sel[np.argsort(center[sel])]
        c = center[sel].astype(np.int64)
        pred = y_pred[sel].astype(np.int16)
        conf = None
        if probs is not None:
            pr = probs[sel]
            pred = pr.argmax(axis=1).astype(np.int16)
            conf = pr.max(axis=1).astype(np.float32)
        elif args.conf > 0:
            print("warning: --conf ignored (preds.npz has no probs)")
            args.conf = 0.0
        if args.smooth > 1:
            pred = median_filter(pred, size=args.smooth, mode="nearest").astype(np.int16)
        frame_pred = expand_to_frames(c, pred, n)
        lab = frame_labels[base:base + n]
        pres = presence[base:base + n]

        frame_correct += int((frame_pred == lab).sum())
        frame_total += n
        if pres.sum():
            hand_correct += int((frame_pred[pres == 1] == lab[pres == 1]).sum())
            hand_total += int(pres.sum())

        # ground-truth gesture segments on the 0-based frame timeline
        gt = [(cid, s - 1, e - 1) for cid, s, e in load_segments(args.ann_root, video)
              if cid != 0]
        pr = frames_to_segments(frame_pred, args.min_len, args.conf, c, conf)

        used_g, used_p = match_segments(gt, pr, args.iou)
        for i, (cid, _s, _e) in enumerate(gt):
            gt_cnt[cid] += 1
            if i in used_g:
                tp_cnt[cid] += 1
        for j, (cid, _s, _e) in enumerate(pr):
            pred_cnt[cid] += 1
            if j not in used_p:
                fp_total += 1

    per_class = []
    f1s = []
    for cid in range(1, num_classes):
        tp = int(tp_cnt[cid])
        fp = int(pred_cnt[cid] - tp)
        fn = int(gt_cnt[cid] - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        per_class.append({
            "id": cid, "code": classes[cid], "gt": int(gt_cnt[cid]),
            "pred": int(pred_cnt[cid]), "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
        })

    tp_all, fp_all, fn_all = int(tp_cnt.sum()), int(pred_cnt.sum() - tp_cnt.sum()), int(gt_cnt.sum() - tp_cnt.sum())
    micro_p = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    micro_r = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    macro_f1 = float(np.mean([x["f1"] for x in per_class]))

    minutes = total_frames / 30.0 / 60.0
    result = {
        "preds": str(args.preds),
        "data": str(args.data),
        "split": args.split,
        "iou": args.iou,
        "min_len": args.min_len,
        "smooth": args.smooth,
        "conf": args.conf,
        "windows": int(len(y_true)),
        "frames": int(total_frames),
        "frame_accuracy_all": frame_correct / max(frame_total, 1),
        "frame_accuracy_hand": hand_correct / max(hand_total, 1),
        "segments_gt": int(gt_cnt.sum()),
        "segments_pred": int(pred_cnt.sum()),
        "tp": tp_all, "fp": fp_all, "fn": fn_all,
        "precision": micro_p, "recall": micro_r,
        "micro_f1": micro_f1, "macro_f1": macro_f1,
        "false_activations_per_min": fp_total / max(minutes, 1e-9),
        "per_class": per_class,
    }

    out = args.out or (args.preds.parent / "continuous")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "continuous.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    lines = [
        f"{args.preds.parent.name} / {args.split}",
        f"IoU {args.iou} | min_len {args.min_len} | smooth {args.smooth} | "
        f"conf {args.conf}",
        f"frames {total_frames} ({minutes:.1f} min) | windows {len(y_true)}",
        f"frame acc (all) {result['frame_accuracy_all']:.4f} | "
        f"frame acc (hand) {result['frame_accuracy_hand']:.4f}",
        f"segments GT {result['segments_gt']} | pred {result['segments_pred']} | "
        f"TP {tp_all} FP {fp_all} FN {fn_all}",
        f"precision {micro_p:.4f} | recall {micro_r:.4f} | "
        f"micro-F1 {micro_f1:.4f} | macro-F1 {macro_f1:.4f}",
        f"false activations {result['false_activations_per_min']:.2f}/min",
        "",
        f"{'code':5s} {'gt':>5s} {'pred':>5s} {'tp':>5s} {'fp':>5s} {'fn':>5s} "
        f"{'P':>6s} {'R':>6s} {'F1':>6s}",
    ]
    for x in per_class:
        lines.append(f"{x['code']:5s} {x['gt']:5d} {x['pred']:5d} {x['tp']:5d} "
                     f"{x['fp']:5d} {x['fn']:5d} {x['precision']:6.3f} "
                     f"{x['recall']:6.3f} {x['f1']:6.3f}")
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
