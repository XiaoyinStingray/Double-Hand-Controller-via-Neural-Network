"""Download pretrained weights / model assets into ./weights (gitignored).

Usage:
    python scripts/download_models.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"

ASSETS = {
    # COCO detect model, used as a starting point / smoke test
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt",
    # COCO pose model (17 body keypoints), starting point for the 21-kpt hand head
    "yolo11n-pose.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt",
    # MediaPipe hand landmarker (Tasks API, supports 2 hands + handedness)
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


def download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[get ] {dst.name} <- {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)
    print(f"[ ok ] {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for name, url in ASSETS.items():
        try:
            download(url, WEIGHTS / name)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
