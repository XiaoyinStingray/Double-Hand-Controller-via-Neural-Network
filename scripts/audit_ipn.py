"""Audit the IPN Hand / IPN HandS dataset layout and annotations.

Checks:
  * every split video has an annotation file and a frame count
  * segments are contiguous and cover [1, frames_ipnhands]
  * class distribution (segments / frames)
  * (optional) landmarks: one JSON per video, frame keys, hand counts

Usage:
    python scripts/audit_ipn.py
    python scripts/audit_ipn.py --landmarks data/raw/IPN_HandS_landmarks_GT

Exit code is non-zero if an inconsistency is found.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANN = ROOT / "data" / "raw" / "IPN_HandS_annotations"


def load_classes(root: Path):
    out = {}
    with open(root / "classes.txt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["id"])] = row["code"]
    return out


def load_frame_counts(root: Path):
    out = {}
    with open(root / "frame_counts.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["video"]] = row
    return out


def load_split(root: Path, name: str):
    path = root / "splits" / f"{name}_list.txt"
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def read_segments(path: Path):
    segs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            segs.append(tuple(int(x) for x in line.split(",")))
    return segs


def audit_annotations(root: Path):
    classes = load_classes(root)
    frame_counts = load_frame_counts(root)
    train = load_split(root, "train")
    test = load_split(root, "test")
    videos = train + test

    ann_dir = root / "temporal_annotations"
    ann_files = list(ann_dir.glob("*.txt"))
    ann_videos = {p.stem for p in ann_files}

    problems = []
    if len(ann_videos) != len(videos):
        problems.append(f"annotations {len(ann_videos)} != videos {len(videos)}")
    for v in sorted(set(videos) - ann_videos):
        problems.append(f"missing annotation: {v}")
    for v in sorted(ann_videos - set(videos)):
        problems.append(f"annotation not in split: {v}")
    for v in sorted(set(videos) - set(frame_counts)):
        problems.append(f"missing frame count: {v}")

    seg_count = Counter()
    frame_count = Counter()
    for p in ann_files:
        video = p.stem
        segs = read_segments(p)
        if not segs:
            problems.append(f"empty annotation: {video}")
            continue
        if segs[0][1] != 1:
            problems.append(f"{video}: starts at frame {segs[0][1]}")
        for i in range(1, len(segs)):
            if segs[i][1] != segs[i - 1][2] + 1:
                problems.append(f"{video}: gap before segment {i}")
                break
        expected = int(frame_counts[video]["frames_ipnhands"]) if video in frame_counts else None
        if expected is not None and segs[-1][2] != expected:
            problems.append(f"{video}: ends at {segs[-1][2]}, expected {expected}")
        for cid, start, end in segs:
            if cid not in classes:
                problems.append(f"{video}: unknown class id {cid}")
                continue
            seg_count[cid] += 1
            frame_count[cid] += end - start + 1

    total = sum(frame_count.values())
    print(f"videos        {len(videos)}  (train {len(train)}, test {len(test)})")
    print(f"frames total  {total}")
    print(f"\n{'id':>2}  {'code':5s} {'segments':>8s} {'frames':>8s} {'share':>7s}")
    for cid in sorted(classes):
        share = 100.0 * frame_count[cid] / total if total else 0.0
        print(f"{cid:2d}  {classes[cid]:5s} {seg_count[cid]:8d} "
              f"{frame_count[cid]:8d} {share:6.1f}%")
    return problems, videos, frame_counts


def audit_landmarks(landmarks_dir: Path, videos, frame_counts):
    print(f"\nlandmarks     {landmarks_dir}")
    if not landmarks_dir.exists():
        print("  not downloaded yet, skipped")
        return []

    json_files = list(landmarks_dir.rglob("*.json"))
    by_video = {p.stem: p for p in json_files}
    problems = []
    print(f"  json files   {len(json_files)}")
    for v in sorted(set(videos) - set(by_video)):
        problems.append(f"missing landmarks: {v}")

    frames_total = 0
    hand_hist = Counter()
    side_hist = Counter()
    bad_len = 0
    bad_keys = 0
    for v in videos:
        path = by_video.get(v)
        if path is None:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        expected = int(frame_counts[v]["frames_ipnhands"]) if v in frame_counts else len(data)
        if len(data) != expected:
            bad_keys += 1
            problems.append(f"{v}: {len(data)} landmark frames != {expected}")
        for frame_name, item in data.items():
            frames_total += 1
            hands = item.get("hand_landmarks") or []
            hand_hist[len(hands)] += 1
            for hand in hands:
                if len(hand.get("landmarks", {})) != 21:
                    bad_len += 1
                side_hist[hand.get("handedness", "?")] += 1
        # key continuity: <video>_000001.jpg ... _<N>.jpg
        nums = sorted(int(m.group(1)) for m in
                      (re.search(r"_(\d+)\.jpg$", k) for k in data) if m)
        if nums and (nums[0] != 1 or nums[-1] != expected or len(nums) != expected):
            problems.append(f"{v}: frame index range {nums[0]}..{nums[-1]} "
                            f"({len(nums)} keys)")

    print(f"  frames       {frames_total}")
    print(f"  hands/frame  " + ", ".join(f"{k}:{hand_hist[k]}" for k in sorted(hand_hist)))
    print(f"  handedness   " + ", ".join(f"{k}:{side_hist[k]}" for k in sorted(side_hist)))
    if bad_len:
        problems.append(f"{bad_len} hands without exactly 21 keypoints")
    if bad_keys:
        problems.append(f"{bad_keys} videos with wrong frame key count")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ANN,
                        help="extracted IPN_HandS_annotations directory")
    parser.add_argument("--landmarks", type=Path, default=None,
                        help="landmarks_gt directory (optional)")
    args = parser.parse_args()

    print(f"annotations   {args.root}")
    problems, videos, frame_counts = audit_annotations(args.root)
    if args.landmarks is not None:
        problems += audit_landmarks(args.landmarks, videos, frame_counts)

    print()
    if problems:
        print(f"PROBLEMS ({len(problems)}):")
        for p in problems[:30]:
            print("  -", p)
        if len(problems) > 30:
            print(f"  ... {len(problems) - 30} more")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
