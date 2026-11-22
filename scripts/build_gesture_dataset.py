"""Build the V2 gesture-window dataset from IPN HandS landmarks.

Outputs (per split):
    frames_<split>.npy        (num_frames, 85) float32
    per_frame_labels_<split>.npy
    presence_<split>.npy      (num_frames,) uint8
    hand_count_<split>.npy    (num_frames,) uint8 (0/1/2)
    side_code_<split>.npy     (num_frames,) uint8 (0 none, 1 Left, 2 Right)
    offsets_<split>.npy       (num_videos + 1,) int64
    windows_<split>.npz       video_idx, start, center, label
    class_weights_<split>.npy
    meta.json

Windows are per-video only (never cross a video boundary). By default the
center frame's annotation label is used; no-hand centers become NOG.
Use --drop-no-hand to drop them instead.

Usage:
    python scripts/build_gesture_dataset.py --limit-videos 2 --out data/gesture_sanity
    python scripts/build_gesture_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.ipn_landmarks import (  # noqa: E402
    FEATURE_DIM,
    FPS,
    build_video,
    class_counts,
    class_weights,
    load_classes,
    load_frame_counts,
    load_landmarks,
    load_segments,
    load_split,
    make_windows,
    resolve_landmarks_dir,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ann-root", type=Path,
                   default=ROOT / "data" / "raw" / "IPN_HandS_annotations")
    p.add_argument("--landmarks", type=Path,
                   default=ROOT / "data" / "raw" / "IPN_HandS_landmarks_GT")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "gesture")
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--val-videos", type=int, default=15,
                   help="held-out videos carved from the official train split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--limit-videos", type=int, default=0,
                   help="sanity mode: only first N train videos, no val/test")
    p.add_argument("--drop-no-hand", action="store_true",
                   help="drop windows whose center frame has no hand")
    return p.parse_args()


def fmt_counts(counts, classes):
    return "  ".join(f"{classes[i]}:{c}" for i, c in enumerate(counts) if c)


def build_split(name, videos, args, classes, frame_counts, lm_dir):
    t0 = time.perf_counter()
    feats, per_labels, presence, hand_count, side_code = [], [], [], [], []
    offsets = [0]
    w_video, w_start, w_center, w_label = [], [], [], []

    for vi, video in enumerate(videos):
        data = load_landmarks(lm_dir / f"{video}.json")
        segs = load_segments(args.ann_root, video)
        vf = build_video(data, segs, width=args.width, height=args.height)
        n = len(vf.features)
        if n != frame_counts.get(video, n):
            print(f"  [warn] {video}: {n} frames != {frame_counts.get(video)}")

        feats.append(vf.features)
        per_labels.append(vf.labels)
        presence.append(vf.presence)
        hand_count.append(vf.hand_count)
        side_code.append(vf.side_code)
        offsets.append(offsets[-1] + n)

        starts, centers, labels = make_windows(
            vf.labels, vf.presence, T=args.T, stride=args.stride,
            require_hand_center=args.drop_no_hand,
            force_nog_center=not args.drop_no_hand,
        )
        w_video.append(np.full(len(starts), vi, dtype=np.int32))
        w_start.append(starts)
        w_center.append(centers)
        w_label.append(labels)

    frames = np.concatenate(feats) if feats else np.zeros((0, FEATURE_DIM), np.float32)
    per_labels = np.concatenate(per_labels) if per_labels else np.zeros(0, np.int16)
    presence = np.concatenate(presence) if presence else np.zeros(0, np.uint8)
    hand_count = np.concatenate(hand_count) if hand_count else np.zeros(0, np.uint8)
    side_code = np.concatenate(side_code) if side_code else np.zeros(0, np.uint8)
    offsets = np.asarray(offsets, dtype=np.int64)

    w_video = np.concatenate(w_video) if w_video else np.zeros(0, np.int32)
    w_start = np.concatenate(w_start) if w_start else np.zeros(0, np.int64)
    w_center = np.concatenate(w_center) if w_center else np.zeros(0, np.int64)
    w_label = np.concatenate(w_label) if w_label else np.zeros(0, np.int16)

    lengths = np.diff(offsets)
    if len(w_start):
        assert np.all(lengths[w_video] >= w_start + args.T), "window crosses video end"

    assert frames.shape == (len(per_labels), FEATURE_DIM), frames.shape
    assert np.isfinite(frames).all(), "non-finite feature values"
    assert per_labels.min(initial=0) >= 0 and per_labels.max(initial=0) < len(classes)
    assert w_label.min(initial=0) >= 0 and w_label.max(initial=0) < len(classes)

    out = args.out
    np.save(out / f"frames_{name}.npy", frames)
    np.save(out / f"per_frame_labels_{name}.npy", per_labels)
    np.save(out / f"presence_{name}.npy", presence)
    np.save(out / f"hand_count_{name}.npy", hand_count)
    np.save(out / f"side_code_{name}.npy", side_code)
    np.save(out / f"offsets_{name}.npy", offsets)
    np.save(out / f"class_weights_{name}.npy", class_weights(w_label, len(classes)))
    np.savez(out / f"windows_{name}.npz",
             video_idx=w_video, start=w_start, center=w_center, label=w_label)

    frame_cls = class_counts(per_labels, len(classes))
    win_cls = class_counts(w_label, len(classes))
    hc = np.bincount(hand_count, minlength=3).astype(int).tolist()
    sc = np.bincount(side_code, minlength=3).astype(int).tolist()

    stats = {
        "num_videos": len(videos),
        "frames": int(len(frames)),
        "windows": int(len(w_label)),
        "presence_ratio": float(presence.mean()) if len(presence) else 0.0,
        "hand_count": {"0": hc[0], "1": hc[1], "2": hc[2]},
        "side": {"none": sc[0], "Left": sc[1], "Right": sc[2]},
        "frame_class_counts": frame_cls,
        "window_class_counts": win_cls,
        "class_weights": class_weights(w_label, len(classes)).tolist(),
    }

    print(f"\n[{name}] videos {stats['num_videos']} | frames {stats['frames']} | "
          f"windows {stats['windows']} | presence {stats['presence_ratio']:.2f} | "
          f"{time.perf_counter() - t0:.1f}s")
    print(f"  hands   0:{hc[0]} 1:{hc[1]} 2:{hc[2]} | "
          f"side none:{sc[0]} L:{sc[1]} R:{sc[2]}")
    print(f"  frames  {fmt_counts(frame_cls, classes)}")
    print(f"  windows {fmt_counts(win_cls, classes)}")
    return stats


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lm_dir = resolve_landmarks_dir(args.landmarks)
    classes = load_classes(args.ann_root)
    frame_counts = load_frame_counts(args.ann_root)

    print(f"ann       {args.ann_root}")
    print(f"landmarks {lm_dir}")
    print(f"out       {args.out}")
    print(f"T={args.T} stride={args.stride} "
          f"no_hand={'drop' if args.drop_no_hand else 'NOG'}")

    train = load_split(args.ann_root, "train")
    test = load_split(args.ann_root, "test")

    if args.limit_videos:
        splits = {"train": train[: args.limit_videos]}
    else:
        rng = np.random.default_rng(args.seed)
        n_val = max(1, min(args.val_videos, len(train) - 1))
        val = sorted(rng.choice(train, size=n_val, replace=False).tolist())
        val_set = set(val)
        splits = {
            "train": [v for v in train if v not in val_set],
            "val": val,
            "test": test,
        }

    stats = {}
    for name, videos in splits.items():
        stats[name] = build_split(name, videos, args, classes, frame_counts, lm_dir)

    meta = {
        "version": 1,
        "T": args.T,
        "stride": args.stride,
        "fps": FPS,
        "feature_dim": FEATURE_DIM,
        "feature_layout": {
            "pos": [0, 42],
            "vel": [42, 84],
            "presence": 84,
        },
        "hand_policy": "2hands->Right; 1hand->any; 0hands->none",
        "no_hand_policy": "drop" if args.drop_no_hand else "nog",
        "val_videos": args.val_videos,
        "seed": args.seed,
        "classes": {str(k): v for k, v in classes.items()},
        "splits": {name: {"videos": videos, **stats[name]}
                   for name, videos in splits.items()},
    }
    with open(args.out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    total_w = sum(s["windows"] for s in stats.values())
    total_f = sum(s["frames"] for s in stats.values())
    print(f"\nTotal: videos {sum(s['num_videos'] for s in stats.values())} | "
          f"frames {total_f} | windows {total_w}")
    print(f"meta: {args.out / 'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
