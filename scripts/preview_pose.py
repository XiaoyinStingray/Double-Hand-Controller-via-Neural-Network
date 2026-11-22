"""Preview the trained YOLO-pose hand model on the webcam.

Usage:
    python scripts/preview_pose.py
    python scripts/preview_pose.py --headless --frames 60
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


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
    times = deque(maxlen=30)
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
            times.append(time.perf_counter())
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
            n_hands = 0 if res.boxes is None else len(res.boxes)

            n += 1
            if args.headless:
                if n % 30 == 0:
                    print(f"frame {n:4d} | fps {fps:5.1f} | infer {dt:4.1f} ms | hands {n_hands}")
            else:
                vis = res.plot()
                cv2.putText(vis, f"FPS {fps:5.1f}  infer {dt:4.1f} ms  hands {n_hands}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(vis, "Esc / q: quit", (10, vis.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv2.imshow("YOLO pose preview", vis)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if args.frames and n >= args.frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if n:
        print(f"frames {n} | avg fps {(n - 1) / (times[-1] - times[0]):5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
