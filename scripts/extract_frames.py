"""Extract IPN Hand frames on the IPN HandS timeline.

Why not use raw decoded frame indices?
    Some source videos are 29.97 fps while IPN HandS is built with
    `ffmpeg -vf fps=30`, so frame i in HandS is NOT frame i of the container
    for those videos. We always resample to 30 fps first, then keep every
    K-th frame:

        handS_idx = 1 + K * (out_idx - 1)

    With K=3 this is 10 fps, K=4 -> 7.5 fps, K=6 -> 5 fps.

Output:
    data/frames/<video>/<video>_000001.jpg
    data/frames/manifest.csv   (video, split, out_idx, handS_idx, class_id, code, path)

Usage:
    python scripts/extract_frames.py --split train --stride 3
    python scripts/extract_frames.py --video "1CM1_1_R_#217" --stride 3
    python scripts/extract_frames.py --split test --limit 2 --stride 3
"""

from __future__ import annotations

import argparse
import bisect
import csv
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ANN = RAW / "IPN_HandS_annotations"
VIDEOS = RAW / "ipn_mp4" / "videos"
DEFAULT_OUT = ROOT / "data" / "frames"

# Four training videos have damaged/truncated AVI streams: the container reports
# the HandS frame count, but fewer frames actually decode, so `fps=30` yields
# 3-4 fewer 10 fps frames (tail only). Existing frames still align with the
# landmarks. Keep this list explicit rather than silently ignoring mismatches.
KNOWN_SHORT = {
    "1CM1_4_R_#229",
    "1CM1_4_R_#230",
    "1CM1_4_R_#231",
    "1CM1_4_R_#232",
}


def find_ffmpeg() -> str:
    local = sorted((ROOT / "tools" / "ffmpeg").rglob("ffmpeg.exe"))
    if local:
        return str(local[0])
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise SystemExit(
        "ffmpeg not found. Put a static build under tools/ffmpeg/ "
        "or install it and make sure it is on PATH."
    )


def load_split(name: str):
    path = ANN / "splits" / f"{name}_list.txt"
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_frame_counts():
    out = {}
    with open(ANN / "frame_counts.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["video"]] = int(row["frames_ipnhands"])
    return out


def load_classes():
    out = {}
    with open(ANN / "classes.txt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["id"])] = row["code"]
    return out


def load_segments(video: str):
    path = ANN / "temporal_annotations" / f"{video}.txt"
    segs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            segs.append(tuple(int(x) for x in line.split(",")))
    return segs


def extract_one(ffmpeg: str, video: str, stride: int, out_dir: Path,
                force: bool, frame_counts: dict):
    dst = out_dir / video
    expected = 1 + (frame_counts[video] - 1) // stride
    existing = sorted(dst.glob("*.jpg")) if dst.exists() else []

    if not force and len(existing) == expected:
        return expected, expected, True

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    vf = f"fps=30,select=not(mod(n\\,{stride}))"
    cmd = [
        ffmpeg, "-v", "error", "-i", str(VIDEOS / f"{video}.avi"),
        "-vf", vf, "-fps_mode", "vfr", "-q:v", "2",
        # A few containers report one extra frame (e.g. 1CM42_6_R_#161);
        # cap to the HandS length so indices and landmarks stay aligned.
        "-frames:v", str(expected), "-start_number", "1",
        str(dst / f"{video}_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    got = len(list(dst.glob("*.jpg")))
    return got, expected, got == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--video", default=None, help="single video name (no .avi)")
    parser.add_argument("--stride", type=int, default=3,
                        help="keep every N-th 30fps frame (3 -> 10 fps)")
    parser.add_argument("--limit", type=int, default=0, help="only first N videos")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    args.out = args.out.resolve()

    if args.video:
        test_set = set(load_split("test"))
        split = "test" if args.video in test_set else "train"
        videos = [(args.video, split)]
    else:
        train = load_split("train")
        test = load_split("test")
        if args.split == "train":
            videos = [(v, "train") for v in train]
        elif args.split == "test":
            videos = [(v, "test") for v in test]
        else:
            videos = [(v, "train") for v in train] + [(v, "test") for v in test]

    if args.limit:
        videos = videos[: args.limit]

    ffmpeg = find_ffmpeg()
    classes = load_classes()
    frame_counts = load_frame_counts()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.csv"

    total_frames = 0
    failures = []
    known_short = []
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "split", "out_idx", "handS_idx",
                         "class_id", "code", "path"])
        for i, (video, split) in enumerate(videos, 1):
            segs = load_segments(video)
            starts = [s for _, s, _ in segs]
            got, expected, ok = extract_one(ffmpeg, video, args.stride, args.out,
                                            args.force, frame_counts)
            tag = ""
            if not ok:
                if video in KNOWN_SHORT:
                    known_short.append((video, got, expected))
                    tag = "  [known-short]"
                else:
                    failures.append((video, got, expected))
                    tag = "  <-- MISMATCH"
            print(f"[{i}/{len(videos)}] {video}: {got}/{expected}{tag}")

            for out_idx in range(1, got + 1):
                handS_idx = 1 + args.stride * (out_idx - 1)
                j = bisect.bisect_right(starts, handS_idx) - 1
                cid = segs[j][0] if j >= 0 else 0
                writer.writerow([
                    video, split, out_idx, handS_idx, cid, classes.get(cid, "?"),
                    (args.out / video / f"{video}_{out_idx:06d}.jpg").relative_to(ROOT).as_posix(),
                ])
            total_frames += got

    print(f"\nvideos {len(videos)} | frames {total_frames} | manifest {manifest_path}")
    if known_short:
        print(f"known-short streams ({len(known_short)}):")
        for v, got, exp in known_short:
            print(f"  {v}: {got}/{exp} (damaged tail)")
    if failures:
        print(f"MISMATCHES ({len(failures)}):")
        for v, got, exp in failures:
            print(f"  {v}: {got} != {exp}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
