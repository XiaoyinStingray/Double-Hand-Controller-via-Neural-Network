"""V1 + V2 live preview: YOLO-pose -> tracker -> features -> GRU ONNX.

No mouse control yet. Shows boxes, track IDs, GRU gesture/confidence and
latency. Use `--no-gru` for the detector/tracker-only view.

Usage:
    python scripts/v1v2_live.py
    python scripts/v1v2_live.py --headless --frames 90
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.hand_features import TrackFeatureState  # noqa: E402
from src.tracking.simple_tracker import HandObservation, SimpleHandTracker  # noqa: E402

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
COLORS = [(0, 255, 0), (0, 160, 255), (255, 128, 0), (255, 0, 200), (0, 255, 255)]


def plausible(box, kpts, kconf, min_box, min_kpt_conf, min_inside):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < min_box or h < min_box:
        return False
    ar = w / max(h, 1e-6)
    if ar < 0.30 or ar > 3.0:
        return False
    if float(kconf.mean()) < min_kpt_conf:
        return False
    inside = ((kpts[:, 0] >= x1) & (kpts[:, 0] <= x2) &
              (kpts[:, 1] >= y1) & (kpts[:, 1] <= y2)).mean()
    if inside < min_inside:
        return False
    palm = float(np.linalg.norm(kpts[0] - kpts[9]))
    return palm >= 0.10 * math.hypot(w, h)


def assign_sides(tracks):
    if not tracks:
        return {}
    if len(tracks) == 1:
        return {tracks[0].track_id: "Right"}
    ordered = sorted(tracks, key=lambda t: (t.box[0] + t.box[2]) / 2)
    return {ordered[0].track_id: "Left", ordered[-1].track_id: "Right"}


def draw(frame, tracks, labels):
    for tr in tracks:
        color = COLORS[tr.track_id % len(COLORS)]
        x1, y1, x2, y2 = [int(v) for v in tr.box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = labels.get(tr.track_id, f"#{tr.track_id}")
        cv2.putText(frame, text, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        pts = tr.kpts.astype(int)
        for a, b in CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 1)
        for p in pts:
            cv2.circle(frame, tuple(p), 2, (0, 0, 255), -1)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=ROOT / "runs" / "pose" / "full" / "best.pt")
    p.add_argument("--onnx", type=Path,
                   default=ROOT / "runs" / "export" / "gesture_gru.onnx")
    p.add_argument("--meta", type=Path, default=ROOT / "data" / "gesture" / "meta.json")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--min-kpt-conf", type=float, default=0.30)
    p.add_argument("--min-box", type=float, default=30.0)
    p.add_argument("--min-inside", type=float, default=0.60)
    p.add_argument("--min-hits", type=int, default=2)
    p.add_argument("--max-age", type=float, default=1.0)
    p.add_argument("--max-tracks", type=int, default=2)
    p.add_argument("--conf-gesture", type=float, default=0.50)
    p.add_argument("--device", default="0")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--no-gru", action="store_true")
    args = p.parse_args()

    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")
    if not args.meta.exists():
        raise SystemExit(f"meta not found: {args.meta}")
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    classes = [meta["classes"][str(i)] for i in range(len(meta["classes"]))]
    T = int(meta["T"])

    use_gru = not args.no_gru and args.onnx.exists()
    if not args.no_gru and not args.onnx.exists():
        print(f"[warn] onnx not found: {args.onnx}; running detector only")
    sess = None
    in_name = out_name = None
    if use_gru:
        sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    model = YOLO(str(args.model))
    tracker = SimpleHandTracker(max_tracks=args.max_tracks,
                                min_hits=args.min_hits,
                                max_age=args.max_age)
    states = {}
    last_label = {}
    times = deque(maxlen=30)
    t_first = None
    t_last = None
    n = 0

    print(f"model {args.model.name} | gru {'on' if use_gru else 'off'} | T={T}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            t0 = time.perf_counter()
            res = model.predict(frame, imgsz=640, conf=args.conf,
                                device=args.device, verbose=False)[0]
            infer_ms = (time.perf_counter() - t0) * 1000
            now = time.perf_counter()

            obs = []
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                kpts = (res.keypoints.xy.cpu().numpy()
                        if res.keypoints is not None else np.zeros((len(boxes), 21, 2)))
                kconfs = None
                if res.keypoints is not None:
                    kconfs = getattr(res.keypoints, "conf", None)
                    if kconfs is not None:
                        kconfs = kconfs.cpu().numpy()
                if kconfs is None or len(kconfs) != len(boxes):
                    kconfs = np.ones((len(boxes), 21), dtype=np.float32)
                for box, score, kp, kc in zip(boxes, scores, kpts, kconfs):
                    if plausible(box, kp, kc, args.min_box,
                                 args.min_kpt_conf, args.min_inside):
                        obs.append(HandObservation(box, kp, float(score)))

            tracks = tracker.update(obs, now)
            sides = assign_sides(tracks)
            gru_ms = 0.0

            for tr in tracks:
                side = sides.get(tr.track_id, "Right")
                st = states.get(tr.track_id)
                if st is None:
                    st = TrackFeatureState(T=T)
                    states[tr.track_id] = st

                observed = (now - tr.last_t) < 1e-3
                if not observed:
                    if now - st.last_t > 0.30:
                        last_label[tr.track_id] = f"#{tr.track_id} LOST"
                    continue

                if now - st.last_t > 0.10:
                    st.reset()
                st.update(tr.kpts, side, now)

                if use_gru and st.ready():
                    t1 = time.perf_counter()
                    logits = sess.run([out_name], {in_name: st.batch()})[0][0]
                    gru_ms = (time.perf_counter() - t1) * 1000
                    prob = softmax(logits)
                    idx = int(prob.argmax())
                    conf = float(prob[idx])
                    name = classes[idx]
                    if conf < args.conf_gesture:
                        name = "low"
                    st.label, st.conf = name, conf
                    last_label[tr.track_id] = f"#{tr.track_id} {name} {conf:.2f} {side[0]}"
                elif use_gru:
                    last_label[tr.track_id] = f"#{tr.track_id} warmup"
                else:
                    last_label[tr.track_id] = f"#{tr.track_id} {side[0]}"

            # clean stale per-track states
            for tid in list(states):
                if tid not in sides and now - states[tid].last_t > 2.0:
                    states.pop(tid, None)
                    last_label.pop(tid, None)

            if t_first is None:
                t_first = now
            t_last = now
            times.append(now)
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
            n += 1

            if args.headless:
                if n % 30 == 0:
                    lab = ", ".join(last_label.get(t.track_id, f"#{t.track_id}") for t in tracks)
                    print(f"frame {n:4d} | fps {fps:5.1f} | infer {infer_ms:4.1f} ms | "
                          f"gru {gru_ms:4.2f} ms | tracks {len(tracks)} | {lab}")
            else:
                draw(frame, tracks, last_label)
                cv2.putText(frame,
                            f"FPS {fps:5.1f}  infer {infer_ms:4.1f} ms  "
                            f"gru {gru_ms:4.2f} ms  tracks {len(tracks)}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
                cv2.putText(frame, "Esc / q: quit", (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv2.imshow("V1+V2 live", frame)
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
