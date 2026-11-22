"""IPN HandS landmark utilities (pure NumPy).

The official HandS timeline is 30 fps, 1-based frame indices. Each frame has
0-2 hands with 21 keypoints in MediaPipe order.

Single-hand V2 policy:
    * 0 hands -> no hand (presence=0)
    * 1 hand  -> use it (Left or Right)
    * 2 hands -> prefer the anatomical "Right" hand

Per-frame feature (85 dims):
    [0:42]   positions: 21 x (x, y), wrist-origin, palm-scaled, left mirrored
    [42:84]  velocity:  same normalization, per second
    [84]     presence:  1 if a hand was selected, else 0

The features are dimensionless (palm length = 1), so they are comparable
across subjects and camera distances.
"""

from __future__ import annotations

import bisect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

NUM_KEYPOINTS = 21
FEATURE_DIM = NUM_KEYPOINTS * 2 * 2 + 1  # 85
FPS = 30.0
_INDEX_RE = re.compile(r"_(\d+)\.jpg$")


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------
def load_classes(ann_root: Path) -> Dict[int, str]:
    import csv

    out: Dict[int, str] = {}
    with open(ann_root / "classes.txt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["id"])] = row["code"]
    return out


def load_frame_counts(ann_root: Path) -> Dict[str, int]:
    import csv

    out: Dict[str, int] = {}
    with open(ann_root / "frame_counts.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["video"]] = int(row["frames_ipnhands"])
    return out


def load_split(ann_root: Path, name: str) -> List[str]:
    path = ann_root / "splits" / f"{name}_list.txt"
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_segments(ann_root: Path, video: str) -> List[Tuple[int, int, int]]:
    path = ann_root / "temporal_annotations" / f"{video}.txt"
    segs: List[Tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            segs.append(tuple(int(x) for x in line.split(",")))  # type: ignore[arg-type]
    return segs


def resolve_landmarks_dir(root: Path) -> Path:
    """Accept either the JSON folder itself or its parent (e.g. IPN_HandS_landmarks_GT)."""
    if list(root.glob("*.json")):
        return root
    jsons = list(root.rglob("*.json"))
    if not jsons:
        raise SystemExit(f"no landmark JSON files under {root}")
    return jsons[0].parent


# --------------------------------------------------------------------------
# frame parsing
# --------------------------------------------------------------------------
def frame_index(frame_name: str) -> int:
    m = _INDEX_RE.search(frame_name)
    if not m:
        raise ValueError(f"unexpected frame name: {frame_name}")
    return int(m.group(1))


def sorted_frames(data: dict) -> List[Tuple[int, dict]]:
    return sorted(((frame_index(k), v) for k, v in data.items()), key=lambda x: x[0])


def select_hand(item: dict) -> Optional[dict]:
    hands = item.get("hand_landmarks") or []
    if not hands:
        return None
    if len(hands) == 1:
        return hands[0]
    for hand in hands:
        if hand.get("handedness") == "Right":
            return hand
    return hands[0]


def landmarks_to_pixels(hand: dict, width: int = 640, height: int = 480) -> np.ndarray:
    pts = np.empty((NUM_KEYPOINTS, 2), dtype=np.float64)
    lms = hand["landmarks"]
    for i in range(NUM_KEYPOINTS):
        x, y, _z = lms[str(i)]
        pts[i, 0] = x * width
        pts[i, 1] = y * height
    return pts


def normalize_pixels(pts: np.ndarray, side: str, min_palm_px: float = 5.0) -> Optional[np.ndarray]:
    """Wrist-origin, palm-scaled; mirror left hands to a right-hand canonical form."""
    palm = float(np.linalg.norm(pts[0] - pts[9]))
    if not np.isfinite(palm) or palm < min_palm_px:
        return None
    q = (pts - pts[0]) / palm
    if side == "Left":
        q[:, 0] *= -1.0
    return q


@dataclass
class VideoFrames:
    features: np.ndarray    # (N, 85) float32
    labels: np.ndarray      # (N,) int16
    hand_count: np.ndarray  # (N,) uint8
    side_code: np.ndarray   # (N,) uint8: 0 none, 1 Left, 2 Right
    presence: np.ndarray    # (N,) uint8


def build_video(data: dict, segments: List[Tuple[int, int, int]],
                width: int = 640, height: int = 480,
                v_clip: float = 30.0) -> VideoFrames:
    """Convert one HandS JSON into per-frame features and labels."""
    frames = sorted_frames(data)
    n = len(frames)
    feats = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int16)
    hand_count = np.zeros(n, dtype=np.uint8)
    side_code = np.zeros(n, dtype=np.uint8)
    presence = np.zeros(n, dtype=np.uint8)

    starts = [s for _, s, _ in segments]
    dt = 1.0 / FPS
    prev_q: Optional[np.ndarray] = None
    prev_side: Optional[str] = None

    for i, (idx, item) in enumerate(frames):
        j = bisect.bisect_right(starts, idx) - 1
        labels[i] = segments[j][0] if j >= 0 else 0

        hands = item.get("hand_landmarks") or []
        hand_count[i] = min(len(hands), 2)
        hand = select_hand(item)
        if hand is None:
            prev_q = None
            prev_side = None
            continue

        side = hand.get("handedness", "")
        pts = landmarks_to_pixels(hand, width, height)
        q = normalize_pixels(pts, side)
        if q is None:
            prev_q = None
            prev_side = None
            continue

        feats[i, :42] = q.reshape(-1)
        feats[i, 84] = 1.0
        presence[i] = 1
        side_code[i] = 1 if side == "Left" else 2

        if prev_q is not None and prev_side == side:
            vel = (q - prev_q) / dt
            feats[i, 42:84] = np.clip(vel, -v_clip, v_clip).reshape(-1)
        prev_q = q
        prev_side = side

    return VideoFrames(feats, labels, hand_count, side_code, presence)


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------
def make_windows(labels: np.ndarray, presence: np.ndarray,
                 T: int = 16, stride: int = 2,
                 require_hand_center: bool = False,
                 force_nog_center: bool = True):
    """Return (starts, centers, window_labels) for full in-video windows.

    require_hand_center: drop windows whose center frame has no hand.
    force_nog_center:   otherwise, label no-hand centers as NOG (0).
    """
    n = len(labels)
    if n < T:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.int16)
    starts = np.arange(0, n - T + 1, stride, dtype=np.int64)
    centers = starts + T // 2
    if require_hand_center:
        keep = presence[centers] > 0
        starts, centers = starts[keep], centers[keep]
    wlabels = labels[centers].copy()
    if force_nog_center and not require_hand_center:
        wlabels[presence[centers] == 0] = 0
    return starts, centers, wlabels


def class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    w = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    return w.astype(np.float32)


def class_counts(labels: np.ndarray, num_classes: int) -> List[int]:
    return np.bincount(labels, minlength=num_classes).astype(int).tolist()


def load_landmarks(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
