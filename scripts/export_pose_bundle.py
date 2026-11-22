"""Bundle the local pose dataset into a portable YOLO layout + tar.

Output layout:
    pose_bundle/
      images/train/<video>/<frame>.jpg
      images/val/<video>/<frame>.jpg
      labels/train/<video>/<frame>.txt
      labels/val/<video>/<frame>.txt
      weights/yolo11n-pose.pt
      pose.yaml                 (path=/gdata1/zhaojy/datasets/ipn_pose)

Images/labels are hard-linked when possible (same volume), else copied.
The tar is written with bsdtar/tar and is intended for /gdata1.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pose-dir", type=Path, default=ROOT / "data" / "pose")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "pose_bundle")
    p.add_argument("--tar", type=Path, default=ROOT / "data" / "pose_bundle.tar")
    p.add_argument("--weights", type=Path, default=ROOT / "weights" / "yolo11n-pose.pt")
    p.add_argument("--cluster-root", default="/gdata1/zhaojy/datasets/ipn_pose")
    p.add_argument("--skip-tar", action="store_true")
    args = p.parse_args()

    data_root = (ROOT / "data").resolve()
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    counts = {}
    for split in ("train", "val"):
        list_path = args.pose_dir / f"{split}.txt"
        lines = [x for x in list_path.read_text(encoding="utf-8").splitlines() if x]
        n = 0
        for line in lines:
            src = Path(line)
            if not src.is_absolute():
                src = ROOT / src
            try:
                rel = src.resolve().relative_to(data_root)
            except ValueError:
                print(f"[warn] outside data/: {src}")
                continue
            parts = rel.parts
            # frames/<video>/<frame>.jpg -> <video>/<frame>.jpg
            if parts[0] == "frames":
                parts = parts[1:]
            dst = args.out / "images" / split / Path(*parts)
            label = src.with_suffix(".txt")
            if not label.exists():
                print(f"[warn] missing label: {label}")
                continue
            link_or_copy(src, dst)
            link_or_copy(label, args.out / "labels" / split / Path(*parts).with_suffix(".txt"))
            n += 1
        counts[split] = n
        print(f"[{split}] {n} images")

    wdst = args.out / "weights" / args.weights.name
    wdst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.weights, wdst)

    yaml_data = {
        "path": args.cluster_root,
        "train": "images/train",
        "val": "images/val",
        "names": {0: "hand"},
        "kpt_shape": [21, 3],
        "flip_idx": list(range(21)),
    }
    (args.out / "pose.yaml").write_text(
        yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")

    total_mb = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6
    print(f"bundle {args.out} | {counts} | {total_mb:.1f} MB")

    if not args.skip_tar:
        if args.tar.exists():
            args.tar.unlink()
        print(f"[tar] {args.tar}")
        subprocess.run(["tar", "-cf", str(args.tar), "-C", str(args.out), "."],
                       check=True)
        print(f"tar {args.tar} ({args.tar.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
