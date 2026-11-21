"""V0 / slice A -- camera + MediaPipe hand landmarks + FPS.

This slice only validates the capture pipeline (camera, landmarks, handedness,
frame rate). It deliberately does NOT control the mouse yet.

Usage:
    conda run -n hand python scripts/v0_handmouse.py
    conda run -n hand python scripts/v0_handmouse.py --headless --frames 120

Keys: Esc / q to quit.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
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


def build_landmarker() -> vision.HandLandmarker:
    """Create the MediaPipe hand landmarker in VIDEO mode (needs timestamps)."""
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

    # These are requests, not guarantees; the actual values are printed below.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    return cap


def to_mp_image(frame_bgr: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def side_of(result, index: int) -> str:
    """Return the anatomical side ("Left"/"Right"), applying the swap once."""
    cats = result.handedness[index]
    if not cats:
        return "?"
    raw = cats[0].category_name
    if SWAP_HANDEDNESS:
        return {"Left": "Right", "Right": "Left"}.get(raw, raw)
    return raw


def draw_result(frame, result, fps: float, detect_ms: float) -> None:
    """Draw landmarks in-place. English text only: cv2 cannot render CJK."""
    h, w = frame.shape[:2]

    for i, hand in enumerate(result.hand_landmarks):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(frame, pts[conn.start], pts[conn.end], (0, 200, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 3, (0, 0, 255), -1)

        cats = result.handedness[i]
        if cats:
            side, score = side_of(result, i), cats[0].score
            cv2.putText(frame, f"{side} {score:.2f}",
                        (pts[0][0] - 30, pts[0][1] + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame,
                f"FPS {fps:5.1f}  detect {detect_ms:4.1f} ms  "
                f"hands {len(result.hand_landmarks)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, "Esc / q: quit", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


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

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera {args.camera} | backend {cap.getBackendName()} | "
          f"{actual_w}x{actual_h}")

    # Rolling window is more stable than an exponential average when fps is low.
    frame_times: deque[float] = deque(maxlen=30)
    t0 = time.perf_counter()
    first_frame_t = None
    last_frame_t = None
    last_ts_ms = -1
    n = 0
    n_with_hand = 0
    detect_sum = 0.0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("warning: frame grab failed, stopping")
                break

            # Mirror first: the preview then behaves like a selfie camera.
            frame = cv2.flip(frame, 1)

            # MediaPipe VIDEO mode requires strictly increasing millisecond stamps.
            ts_ms = int((time.perf_counter() - t0) * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            t_detect = time.perf_counter()
            result = landmarker.detect_for_video(to_mp_image(frame), ts_ms)
            detect_ms = (time.perf_counter() - t_detect) * 1000

            now = time.perf_counter()
            if first_frame_t is None:
                first_frame_t = now
            last_frame_t = now
            frame_times.append(now)
            if len(frame_times) >= 2:
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])

            n += 1
            detect_sum += detect_ms
            if result.hand_landmarks:
                n_with_hand += 1

            if args.headless:
                if n % 30 == 0:
                    print(f"  frame {n:4d} | fps {fps:5.1f} | "
                          f"detect {detect_ms:4.1f} ms | "
                          f"hands {len(result.hand_landmarks)}")
            else:
                draw_result(frame, result, fps, detect_ms)
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
        print(f"\nframes {n} | total {elapsed:5.1f} s (incl. init) | "
              f"avg fps {avg_fps:5.1f} | "
              f"detect {detect_sum / n:4.1f} ms | "
              f"hand ratio {n_with_hand / n:.2f}")


if __name__ == "__main__":
    main()
