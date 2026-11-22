"""V0 -- single-hand gesture mouse (rule-based, no training).

Slice A: camera + MediaPipe landmarks + FPS.
Slice B: normalized features + 4 gesture rules + temporal stabilizer.

This slice still does NOT move the mouse. Its only job is to produce stable
labels; cursor/click come in slice C/E.

Usage:
    conda run -n hand python scripts/v0_handmouse.py
    conda run -n hand python scripts/v0_handmouse.py --headless --frames 300

Keys: Esc / q to quit.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

# Keep MediaPipe/LiteRT INFO logs out of the demo console.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "weights" / "hand_landmarker.task"

CAMERA_INDEX = 0
FRAME_W, FRAME_H, TARGET_FPS = 640, 480, 30
NUM_HANDS = 1

# MediaPipe's handedness assumes a particular mirror convention. On this
# camera + MediaPipe 1.2.3, feeding an already-mirrored frame yields the
# opposite label, so swap it. Keep this in ONE place: V2 reuses side_of().
SWAP_HANDEDNESS = True

# --- gesture thresholds (tune after watching the overlay) ---
FINGER_STRAIGHT_DEG = 25.0   # joint angle below this = straight
FINGER_FAR_RATIO = 1.1       # tip must be farther from wrist than pip
THUMB_OUT_DIST = 0.55        # thumb tip to pinky MCP, in palm-size units
FIST_INDEX_DEG = 100.0       # index PIP above this = tightly curled (fist)
PINCH_ENTER, PINCH_EXIT = 0.25, 0.35
STABLE_FRAMES = 3
MIN_HOLD_S = 0.2

FINGER_MCP = (5, 9, 13, 17)
FINGER_PIP = (6, 10, 14, 18)
FINGER_TIP = (8, 12, 16, 20)
PALM_IDX = (0, 5, 9, 13, 17)

COLORS = {
    "POINT": (0, 255, 0),
    "PINCH": (0, 0, 255),
    "OPEN": (255, 160, 0),
    "FIST": (140, 140, 140),
    "NONE": (0, 200, 0),
}


@dataclass
class HandFeatures:
    side: str
    pts: np.ndarray       # (21,2) pixels
    q: np.ndarray         # (21,2) wrist-origin, palm-scaled, right-hand canonical
    fingers: np.ndarray   # (5,) bool, index 0 = thumb
    angles: np.ndarray    # (5,) PIP joint angles in degrees (index 0 unused)
    pinch: float
    palm: np.ndarray      # (2,) pixels
    tip: np.ndarray       # (2,) pixels, index fingertip
    v_palm: np.ndarray    # (2,) px/s (EMA)


def side_of(result, index: int) -> str:
    """Return the anatomical side ("Left"/"Right"), applying the swap once."""
    cats = result.handedness[index]
    if not cats:
        return "?"
    raw = cats[0].category_name
    if SWAP_HANDEDNESS:
        return {"Left": "Right", "Right": "Left"}.get(raw, raw)
    return raw


def best_hand_index(result):
    """V0 uses one hand: pick the most confident one."""
    if not result.hand_landmarks:
        return None
    scores = [cats[0].score if cats else 0.0 for cats in result.handedness]
    return int(np.argmax(scores))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 180.0
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _finger_state(pts: np.ndarray, mcp: int, pip: int, tip: int):
    """Return (pip angle in degrees, extended?) -- both scale-free."""
    angle = _angle_deg(pts[pip] - pts[mcp], pts[tip] - pts[pip])
    d_tip = float(np.linalg.norm(pts[tip] - pts[0]))
    d_pip = float(np.linalg.norm(pts[pip] - pts[0]))
    extended = (angle < FINGER_STRAIGHT_DEG
                and d_tip > FINGER_FAR_RATIO * max(d_pip, 1e-6))
    return angle, extended


class PalmVelocity:
    """EMA-smoothed palm velocity; later used by the scroll action."""

    def __init__(self, tau: float = 0.1) -> None:
        self.tau = tau
        self.prev = None
        self.prev_t = None
        self.v = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self.prev = None
        self.prev_t = None
        self.v[:] = 0.0

    def update(self, center: np.ndarray, t: float) -> np.ndarray:
        if self.prev is not None and self.prev_t is not None:
            dt = t - self.prev_t
            if dt > 1e-4:
                raw = (center - self.prev) / dt
                alpha = 1.0 - math.exp(-dt / self.tau)
                self.v += alpha * (raw - self.v)
        self.prev = center.copy()
        self.prev_t = t
        return self.v


def extract_features(result, index: int, w: int, h: int, t: float,
                     velocity: PalmVelocity) -> HandFeatures | None:
    hand = result.hand_landmarks[index]
    if len(hand) < 21:
        return None

    pts = np.array([[lm.x * w, lm.y * h] for lm in hand], dtype=np.float64)
    palm_size = float(np.linalg.norm(pts[0] - pts[9]))
    if palm_size < 1e-6:
        return None

    q = (pts - pts[0]) / palm_size
    side = side_of(result, index)
    if side == "Left":          # canonicalize to a right hand for V2
        q[:, 0] *= -1.0

    fingers = np.zeros(5, dtype=bool)
    angles = np.zeros(5, dtype=np.float64)
    for k, (mcp, pip, tip) in enumerate(zip(FINGER_MCP, FINGER_PIP, FINGER_TIP), start=1):
        angle, extended = _finger_state(pts, mcp, pip, tip)
        angles[k] = angle
        fingers[k] = extended
    fingers[0] = float(np.linalg.norm(q[4] - q[17])) > THUMB_OUT_DIST

    pinch = float(np.linalg.norm(q[4] - q[8]))
    palm = pts[list(PALM_IDX)].mean(axis=0)
    v_palm = velocity.update(palm, t).copy()

    return HandFeatures(side, pts, q, fingers, angles, pinch, palm, pts[8], v_palm)


class GestureRules:
    """Raw per-frame label. PINCH uses hysteresis; FIST has top priority."""

    def __init__(self) -> None:
        self.pinch_active = False

    def reset(self) -> None:
        self.pinch_active = False

    def update(self, feat: HandFeatures) -> str:
        if feat.pinch < PINCH_ENTER:
            self.pinch_active = True
        elif feat.pinch > PINCH_EXIT:
            self.pinch_active = False

        four = feat.fingers[1:]
        # FIST vs PINCH ambiguity: a pinch with the other three fingers
        # curled also looks like "all four folded". The index PIP angle
        # separates them: pinch ~40-70 deg, tight fist >100 deg.
        fist_like = not np.any(four)
        if fist_like and not (self.pinch_active and feat.angles[1] < FIST_INDEX_DEG):
            return "FIST"
        if self.pinch_active:
            return "PINCH"
        if int(four.sum()) == 1 and four[0]:
            return "POINT"
        if int(four.sum()) >= 3:
            return "OPEN"
        return "NONE"


class Stabilizer:
    """Require N agreeing frames, then hold the new label for a minimum time."""

    def __init__(self, enter_frames: int = STABLE_FRAMES,
                 exit_frames: int = STABLE_FRAMES,
                 min_hold_s: float = MIN_HOLD_S) -> None:
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.min_hold_s = min_hold_s
        self.stable = "NONE"
        self.candidate = None
        self.count = 0
        self.last_change = -1e9

    def update(self, raw: str, now: float) -> str:
        if raw == self.stable:
            self.candidate = None
            self.count = 0
            return self.stable

        if raw != self.candidate:
            self.candidate = raw
            self.count = 1
        else:
            self.count += 1

        need = self.exit_frames if raw == "NONE" else self.enter_frames
        if self.count >= need and (now - self.last_change) >= self.min_hold_s:
            self.stable = raw
            self.last_change = now
            self.candidate = None
            self.count = 0
        return self.stable

    def held(self, now: float) -> float:
        return now - self.last_change


def build_landmarker() -> vision.HandLandmarker:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"model not found: {MODEL_PATH}\n"
            "run: python scripts/download_models.py"
        )
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    return cap


def to_mp_image(frame_bgr: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def draw_result(frame, result, feat, raw: str, stable: str,
                fps: float, detect_ms: float, switches: int) -> None:
    """Draw landmarks + debug text. English only: cv2 cannot render CJK."""
    h, w = frame.shape[:2]
    color = COLORS.get(stable, (0, 200, 0))

    for i, hand in enumerate(result.hand_landmarks):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(frame, pts[conn.start], pts[conn.end], color, 2)
        for p in pts:
            cv2.circle(frame, p, 3, (0, 0, 255), -1)
        cv2.putText(frame, side_of(result, i), (pts[0][0] - 20, pts[0][1] + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if feat is not None:
        cv2.circle(frame, tuple(feat.tip.astype(int)), 8, color, 2)
        bits = "".join("1" if f else "0" for f in feat.fingers)
        cv2.putText(frame,
                    f"pinch {feat.pinch:.2f}  aIdx {feat.angles[1]:3.0f}  "
                    f"fingers {bits}  vY {feat.v_palm[1]:+6.0f} px/s",
                    (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (220, 220, 220), 1)

    cv2.putText(frame,
                f"FPS {fps:5.1f}  detect {detect_ms:4.1f} ms  "
                f"hands {len(result.hand_landmarks)}  switches {switches}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"{raw:5s} -> {stable}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--headless", action="store_true",
                        help="no window; useful for a quick smoke test")
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after N frames (0 = run until quit)")
    args = parser.parse_args()

    cap = open_camera(args.camera)
    landmarker = build_landmarker()
    rules = GestureRules()
    stabilizer = Stabilizer()
    velocity = PalmVelocity()

    print(f"camera {args.camera} | backend {cap.getBackendName()} | "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    frame_times: deque[float] = deque(maxlen=30)
    label_count: Counter = Counter()
    t0 = time.perf_counter()
    first_frame_t = None
    last_frame_t = None
    last_ts_ms = -1
    last_hand_t = -1e9
    prev_stable = stabilizer.stable
    n = 0
    n_with_hand = 0
    n_switches = 0
    detect_sum = 0.0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("warning: frame grab failed, stopping")
                break

            frame = cv2.flip(frame, 1)   # selfie view: hand right -> screen right

            ts_ms = int((time.perf_counter() - t0) * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            t_detect = time.perf_counter()
            result = landmarker.detect_for_video(to_mp_image(frame), ts_ms)
            detect_ms = (time.perf_counter() - t_detect) * 1000

            now = time.perf_counter()
            feat = None
            raw = "NONE"
            index = best_hand_index(result)
            if index is not None:
                feat = extract_features(result, index, frame.shape[1],
                                        frame.shape[0], now, velocity)
                if feat is not None:
                    raw = rules.update(feat)
                    last_hand_t = now
                else:
                    index = None

            if index is None:
                velocity.reset()
                if now - last_hand_t > 0.3:
                    rules.reset()

            stable = stabilizer.update(raw, now)
            if stable != prev_stable:
                n_switches += 1
                prev_stable = stable

            frame_times.append(now)
            if first_frame_t is None:
                first_frame_t = now
            last_frame_t = now
            if len(frame_times) >= 2:
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])

            n += 1
            detect_sum += detect_ms
            if index is not None:
                n_with_hand += 1
            label_count[stable] += 1

            if args.headless:
                if n % 30 == 0:
                    pinch = f"{feat.pinch:.2f}" if feat is not None else "-"
                    bits = ("".join("1" if f else "0" for f in feat.fingers)
                            if feat is not None else "-")
                    print(f"  frame {n:4d} | fps {fps:5.1f} | "
                          f"{raw:5s} -> {stable:5s} | pinch {pinch} | {bits}")
            else:
                draw_result(frame, result, feat, raw, stable,
                            fps, detect_ms, n_switches)
                cv2.imshow("V0 preview", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            if args.frames and n >= args.frames:
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t0
    if n:
        loop_s = (last_frame_t - first_frame_t) if n > 1 else 0.0
        avg_fps = (n - 1) / loop_s if loop_s > 0 else 0.0
        switch_rate = n_switches / loop_s if loop_s > 0 else 0.0
        print(f"\nframes {n} | avg fps {avg_fps:5.1f} | "
              f"detect {detect_sum / n:4.1f} ms | "
              f"hand ratio {n_with_hand / n:.2f} | "
              f"switches {n_switches} ({switch_rate:.2f}/s)")
        top = ", ".join(f"{k}:{v}" for k, v in label_count.most_common())
        print(f"labels  {top}")


if __name__ == "__main__":
    main()
