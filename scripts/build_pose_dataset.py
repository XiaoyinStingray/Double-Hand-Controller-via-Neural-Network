"""Build a YOLO-pose dataset from IPN HandS landmarks + extracted frames.

For each selected frame:
  * read the 21 keypoints from the HandS JSON at handS_idx
  * bbox = keypoint min/max + 15% padding, clipped to the image
  * write a YOLO-pose label next to the image:
        class cx cy w h  px1 py1 v1 ... px21 py21 v21
  * frames without a hand get an empty label (background)

Outputs under --out:
  train.txt / val.txt   image paths relative to data/
  pose.yaml             path=data, names={0:hand}, kpt_shape=[21,3]
  stats.json

The image files stay in data/frames; labels are written next to them, so
Ultralytics finds <frame>.txt automatically.

Usage (smoke):
    python scripts/build_pose_dataset.py --out data/pose_smoke \
        --train-per-video 50 --val-per-video 30 \
        --max-train-videos 20 --max-val-videos 5

Usage (full):
    python scripts/build_pose_dataset.py --out data/pose
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FRAMES = DATA / "frames"
LM = DATA / "raw" / "IPN_HandS_landmarks_GT" / "landmarks_gt"
META = DATA / "gesture" / "meta.json"
MANIFEST = FRAMES / "manifest.csv"

W, H = 640, 480
NUM_KPTS = 21


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DATA / "pose")
    p.add_argument("--train-per-video", type=int, default=200)
    p.add_argument("--val-per-video", type=int, default=100)
    p.add_argument("--max-train-videos", type=int, default=0)
    p.add_argument("--max-val-videos", type=int, default=0)
    p.add_argument("--pad", type=float, default=0.15)
    p.add_argument("--min-box", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pose_lines(item, pad: float, min_box: float):
    """Return (label lines, skipped_hands) for one frame."""
    hands = item.get("hand_landmarks") or []
    lines = []
    skipped = 0
    for hand in hands:
        lms = hand.get("landmarks") or {}
        if len(lms) != NUM_KPTS:
            skipped += 1
            continue
        pts = np.array([[lms[str(i)][0] * W, lms[str(i)][1] * H] for i in range(NUM_KPTS)],
                       dtype=np.float64)
        if not np.isfinite(pts).all():
            skipped += 1
            continue

        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        pad_px = pad * max(x1 - x0, y1 - y0)
        x0, y0 = max(0.0, x0 - pad_px), max(0.0, y0 - pad_px)
        x1, y1 = min(W - 1.0, x1 + pad_px), min(H - 1.0, y1 + pad_px)
        bw, bh = x1 - x0, y1 - y0
        if bw < min_box or bh < min_box:
            skipped += 1
            continue

        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        vals = [cx / W, cy / H, bw / W, bh / H]
        for x, y in pts:
            inside = (0.0 <= x < W) and (0.0 <= y < H)
            vals.extend([min(max(x / W, 0.0), 1.0), min(max(y / H, 0.0), 1.0),
                         2 if inside else 0])
        lines.append("0 " + " ".join(f"{v:.6f}" for v in vals))
    return lines, skipped


def select_indices(n: int, k: int, seed: int = 0):
    k = min(k, n)
    if k <= 0:
        return np.zeros(0, dtype=int)
    # even temporal coverage + a small deterministic jitter
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return np.unique(idx)


def build_split(name, videos, per_video, max_videos, args, manifest_by_video,
                stats):
    if max_videos:
        videos = videos[:max_videos]
    out = args.out
    image_paths = []
    cls_counter = Counter()
    n_hands = 0
    n_bg = 0
    n_skipped = 0

    for vi, video in enumerate(videos, 1):
        rows = sorted(manifest_by_video.get(video, []),
                      key=lambda r: int(r["handS_idx"]))
        if not rows:
            print(f"  [warn] no manifest rows for {video}")
            continue
        lms_path = LM / f"{video}.json"
        if not lms_path.exists():
            print(f"  [warn] no landmarks for {video}")
            continue
        lms = json.loads(lms_path.read_text(encoding="utf-8"))

        sel = select_indices(len(rows), per_video, args.seed)
        got = 0
        for i in sel:
            row = rows[int(i)]
            idx = int(row["handS_idx"])
            key = f"{video}_{idx:06d}.jpg"
            item = lms.get(key)
            if item is None:
                n_skipped += 1
                continue

            lines, skipped = pose_lines(item, args.pad, args.min_box)
            n_skipped += skipped
            img_abs = ROOT / row["path"]
            if not img_abs.exists():
                n_skipped += 1
                continue
            label_abs = img_abs.with_suffix(".txt")
            label_abs.write_text(("\n".join(lines) + "\n") if lines else "",
                                 encoding="utf-8")

            image_paths.append(img_abs.as_posix())
            cls_counter[int(row["class_id"])] += 1
            if lines:
                n_hands += len(lines)
            else:
                n_bg += 1
            got += 1

        print(f"  [{name}] {vi}/{len(videos)} {video}: {got} frames")

    list_path = out / f"{name}.txt"
    list_path.write_text("\n".join(image_paths) + ("\n" if image_paths else ""),
                         encoding="utf-8")
    stats[name] = {
        "videos": len(videos),
        "images": len(image_paths),
        "hands": n_hands,
        "backgrounds": n_bg,
        "skipped": n_skipped,
        "class_counts": {str(k): v for k, v in sorted(cls_counter.items())},
    }
    print(f"\n[{name}] images {len(image_paths)} | hands {n_hands} | "
          f"bg {n_bg} | skipped {n_skipped} | {list_path}")
    return list_path


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads(META.read_text(encoding="utf-8"))
    splits = meta["splits"]
    train_videos = splits["train"]["videos"]
    val_videos = splits["val"]["videos"]

    manifest_by_video = defaultdict(list)
    with open(MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            manifest_by_video[row["video"]].append(row)
    print(f"manifest videos {len(manifest_by_video)} | "
          f"train {len(train_videos)} | val {len(val_videos)}")

    stats = {}
    build_split("train", train_videos, args.train_per_video,
                args.max_train_videos, args, manifest_by_video, stats)
    build_split("val", val_videos, args.val_per_video,
                args.max_val_videos, args, manifest_by_video, stats)

    yaml_data = {
        "path": DATA.as_posix(),
        "train": f"{args.out.name}/train.txt",
        "val": f"{args.out.name}/val.txt",
        "names": {0: "hand"},
        "kpt_shape": [NUM_KPTS, 3],
        "flip_idx": list(range(NUM_KPTS)),
    }
    yaml_path = args.out / "pose.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False),
                         encoding="utf-8")
    (args.out / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"yaml {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
