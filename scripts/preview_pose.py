"""Preview YOLO-pose + a lightweight multi-hand tracker (EMA smoothing).

Usage:
    python scripts/preview_pose.py
    python scripts/preview_pose.py --headless --frames 60
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tracking.simple_tracker import HandObservation, SimpleHandTracker  # noqa: E402

# MediaPipe hand topology (21 edges)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
COLORS = [(0, 255, 0), (0, 160, 255), (255, 128, 0), (255, 0, 200), (0, 255, 255)]


def draw_tracks(frame, tracks):
    for tr in tracks:
        color = COLORS[tr.track_id % len(COLORS)]
        x1, y1, x2, y2 = [int(v) for v in tr.box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"hand #{tr.track_id} {tr.score:.2f}",
                    (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        pts = tr.kpts.astype(int)
        for a, b in CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 1)
        for p in pts:
            cv2.circle(frame, tuple(p), 2, (0, 0, 255), -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=ROOT / "runs" / "pose" / "full" / "best.pt")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--frames", type=int, default=0)
    args = p.parse_args()

    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    model = YOLO(str(args.model))
    tracker = SimpleHandTracker()
    times = deque(maxlen=30)
    t_first = None
    t_last = None
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            t0 = time.perf_counter()
            res = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                device=args.device, verbose=False)[0]
            dt = (time.perf_counter() - t0) * 1000
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            times.append(now)
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

            observations = []
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                kpts = (res.keypoints.xy.cpu().numpy()
                        if res.keypoints is not None else np.zeros((len(boxes), 21, 2)))
                for box, score, kp in zip(boxes, scores, kpts):
                    observations.append(HandObservation(box, kp, float(score)))

            tracks = tracker.update(observations, now)
            n += 1

            if args.headless:
                if n % 30 == 0:
                    print(f"frame {n:4d} | fps {fps:5.1f} | infer {dt:4.1f} ms | "
                          f"tracks {len(tracks)}")
            else:
                draw_tracks(frame, tracks)
                cv2.putText(frame, f"FPS {fps:5.1f}  infer {dt:4.1f} ms  "
                                   f"tracks {len(tracks)}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(frame, "Esc / q: quit", (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv2.imshow("YOLO pose + tracker", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if args.frames and n >= args.frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if n and t_first is not None and t_last is not None and t_last > t_first:
        print(f"frames {n} | avg fps {(n - 1) / (t_last - t_first):5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
