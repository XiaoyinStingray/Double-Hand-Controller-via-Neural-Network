"""V1 + V2 live pipeline: YOLO-pose -> tracker -> features -> GRU ONNX.

Optional layers on top of the same pipeline:
  --mouse          geometric mouse (cursor / click / drag / scroll)
  --demo hockey    two-hand air hockey
  --demo 3d        pinch-driven 3D object viewer
  --demo fruit     two-hand fruit slicing

Usage:
    python scripts/v1v2_live.py
    python scripts/v1v2_live.py --mouse --dry-run
    python scripts/v1v2_live.py --demo 3d --mouse
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
from src.features.rules import GestureRules, Stabilizer  # noqa: E402
from src.tracking.simple_tracker import HandObservation, SimpleHandTracker  # noqa: E402
from src.app.hockey import AirHockey  # noqa: E402
from src.app.manip3d import Manip3D  # noqa: E402
from src.app.fruit import FruitNinja  # noqa: E402

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
    # hybrid mouse layer
    p.add_argument("--mouse", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--active-x", type=float, nargs=2, default=(0.25, 0.75),
                   metavar=("X0", "X1"))
    p.add_argument("--active-y", type=float, nargs=2, default=(0.30, 0.70),
                   metavar=("Y0", "Y1"))
    p.add_argument("--mincutoff", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.012)
    p.add_argument("--deadzone", type=float, default=1.5)
    p.add_argument("--scroll-gain", type=float, default=0.25)
    p.add_argument("--scroll-deadzone", type=float, default=12.0)
    p.add_argument("--scroll-expo", type=float, default=1.5)
    p.add_argument("--scroll-max-speed", type=float, default=72.0)
    p.add_argument("--scroll-invert", action="store_true")
    p.add_argument("--pinch-latch", type=float, default=0.42)
    p.add_argument("--pinch-unlatch", type=float, default=0.52)
    p.add_argument("--drag-move", type=float, default=12.0)
    p.add_argument("--drag-gain", type=float, default=2.0)
    p.add_argument("--raw-pinch-enter", type=float, default=0.22)
    p.add_argument("--raw-pinch-exit", type=float, default=0.30)
    p.add_argument("--click-hold", type=float, default=0.04)
    p.add_argument("--click-context", type=float, default=0.6)
    p.add_argument("--pause-hold", type=float, default=0.4)
    p.add_argument("--resume-hold", type=float, default=0.8)
    p.add_argument("--cursor-source", choices=["tip", "mcp"], default="tip")
    p.add_argument("--stable-frames", type=int, default=3)
    p.add_argument("--min-hold", type=float, default=0.15)
    # two-hand demo / mode switching
    p.add_argument("--game", action="store_true",
                   help="alias for --demo hockey")
    p.add_argument("--demo", choices=["hockey", "3d", "fruit"], default=None)
    p.add_argument("--hockey-target", type=int, default=3)
    p.add_argument("--hockey-time", type=float, default=90.0)
    p.add_argument("--hockey-speed", type=float, default=240.0)
    p.add_argument("--fruit-lives", type=int, default=3)
    p.add_argument("--fruit-time", type=float, default=90.0)
    p.add_argument("--enter-hold", type=float, default=0.6)
    p.add_argument("--exit-hold", type=float, default=0.4)
    p.add_argument("--exit-lost", type=float, default=1.0)
    args = p.parse_args()

    if args.demo is None and args.game:
        args.demo = "hockey"
    args.game = args.demo is not None

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

    if args.headless and args.mouse and not args.dry_run:
        raise SystemExit("--headless with --mouse requires --dry-run")
    mouse = None
    mapper = None
    if args.mouse:
        from src.control.cursor import FixedRegionMapper
        from src.control.mouse import MouseController
        mouse = MouseController(
            dry_run=args.dry_run,
            drag_move_px=args.drag_move,
            drag_gain=args.drag_gain,
            raw_pinch_enter=args.raw_pinch_enter,
            raw_pinch_exit=args.raw_pinch_exit,
            click_min_hold_s=args.click_hold,
            click_context_s=args.click_context,
            deadzone_px=args.deadzone,
            scroll_gain=args.scroll_gain,
            scroll_deadzone=args.scroll_deadzone,
            scroll_expo=args.scroll_expo,
            scroll_max_speed=args.scroll_max_speed,
            scroll_invert=args.scroll_invert,
            mincutoff=args.mincutoff,
            beta=args.beta,
            cursor_source=args.cursor_source,
            pinch_latch=args.pinch_latch,
            pinch_unlatch=args.pinch_unlatch,
        )
        mapper = FixedRegionMapper(tuple(args.active_x), tuple(args.active_y))

    rules = {}
    stabs = {}

    def stable_of(tid, feat, now):
        r = rules.get(tid)
        if r is None:
            r = rules[tid] = GestureRules()
        s = stabs.get(tid)
        if s is None:
            s = stabs[tid] = Stabilizer(enter_frames=args.stable_frames,
                                        exit_frames=args.stable_frames,
                                        min_hold_s=args.min_hold)
        if feat is None:
            return s.update("NONE", now)
        return s.update(r.update(feat), now)

    mouse_state = "RUNNING"
    demo = None
    mode = "DEMO" if (args.game and not args.mouse) else "MOUSE"
    need_reset = True
    enter_since = None
    lost_since = None
    last_now = None

    states = {}
    last_label = {}
    times = deque(maxlen=30)
    t_first = None
    t_last = None
    n = 0

    print(f"model {args.model.name} | gru {'on' if use_gru else 'off'} | T={T}")
    print(f"mouse {'on' if args.mouse else 'off'} | dry_run={args.dry_run} | "
          f"demo {args.demo or 'off'} | mode {mode}")
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
            feats = {}

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
                    if tr.track_id in rules:
                        rules[tr.track_id].reset()
                    if tr.track_id in stabs:
                        stabs[tr.track_id].reset()
                feat = st.update(tr.kpts, side, now)
                if feat is not None:
                    feats[tr.track_id] = feat

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

            geoms = {tr.track_id: stable_of(tr.track_id, feats.get(tr.track_id), now)
                     for tr in tracks}

            if args.game and demo is None:
                fw, fh = frame.shape[1], frame.shape[0]
                if args.demo == "hockey":
                    demo = AirHockey(fw, fh, target=args.hockey_target,
                                     duration=args.hockey_time,
                                     base_speed=args.hockey_speed)
                elif args.demo == "3d":
                    demo = Manip3D(fw, fh)
                else:
                    demo = FruitNinja(fw, fh, duration=args.fruit_time,
                                      lives=args.fruit_lives)
                need_reset = True

            primary = max(tracks, key=lambda t: (t.box[0] + t.box[2]) / 2) if tracks else None
            pfeat = feats.get(primary.track_id) if primary is not None else None
            geom = geoms.get(primary.track_id, "NONE") if primary is not None else "NONE"

            # single-hand mouse <-> two-hand demo
            if args.game:
                if mode == "MOUSE":
                    both_open = (len(feats) == 2 and len(geoms) == 2 and
                                 all(g == "OPEN" for g in geoms.values()))
                    if both_open:
                        if enter_since is None:
                            enter_since = now
                        if now - enter_since >= args.enter_hold:
                            mode = "DEMO"
                            need_reset = True
                            enter_since = None
                            lost_since = None
                            if mouse is not None:
                                mouse.on_hand_lost()
                                mouse_state = "RUNNING"
                    else:
                        enter_since = None
                else:
                    if len(feats) < 2:
                        if lost_since is None:
                            lost_since = now
                    else:
                        lost_since = None
                    fist_out = any(
                        g == "FIST" and stabs[tid].held(now) >= args.exit_hold
                        for tid, g in geoms.items())
                    if fist_out or (lost_since is not None and
                                    now - lost_since >= args.exit_lost):
                        mode = "MOUSE"
                        lost_since = None
                        if mouse is not None:
                            mouse.on_hand_lost()
                            mouse_state = "RUNNING"

            if mode == "DEMO":
                if need_reset:
                    demo.reset(now)
                    need_reset = False
                inputs = {}
                for tid, f in feats.items():
                    side = sides.get(tid)
                    if side is None:
                        continue
                    px = f.palm[0] / max(frame.shape[1], 1)
                    py = f.palm[1] / max(frame.shape[0], 1)
                    if side == "Left":
                        u = 2.0 * px
                    else:
                        u = 2.0 * (px - 0.5)
                    v = py
                    inputs[side] = {"u": u, "v": v, "tip": f.tip,
                                    "palm": f.palm, "pinch": float(f.pinch),
                                    "geom": geoms.get(tid, "NONE")}
                demo.set_inputs(inputs, now)
                dt = 0.0 if last_now is None else now - last_now
                demo.update(now, dt)
                last_now = now
            elif mouse is not None:
                if pfeat is None:
                    mouse.on_hand_lost()
                if mouse_state == "RUNNING":
                    if geom == "FIST" and stabs[primary.track_id].held(now) >= args.pause_hold:
                        mouse.pause()
                        mouse_state = "PAUSED"
                    else:
                        mouse.update(geom, pfeat, mapper,
                                     frame.shape[1], frame.shape[0], now)
                elif geom == "OPEN" and stabs[primary.track_id].held(now) >= args.resume_hold:
                    mouse.resume()
                    mouse_state = "RUNNING"

            # clean stale per-track states
            for tid in list(states):
                if tid not in sides and now - states[tid].last_t > 2.0:
                    states.pop(tid, None)
                    last_label.pop(tid, None)
                    rules.pop(tid, None)
                    stabs.pop(tid, None)

            if t_first is None:
                t_first = now
            t_last = now
            times.append(now)
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
            n += 1

            if args.headless:
                if n % 30 == 0:
                    lab = ", ".join(last_label.get(t.track_id, f"#{t.track_id}") for t in tracks)
                    act = (mouse.current_action(now, geom)
                           if mouse is not None and mode == "MOUSE" else "-")
                    gs = (f" {args.demo} {demo.status()}"
                          if demo is not None and mode == "DEMO" else "")
                    print(f"frame {n:4d} | fps {fps:5.1f} | infer {infer_ms:4.1f} ms | "
                          f"gru {gru_ms:4.2f} ms | tracks {len(tracks)} | {lab} | "
                          f"{mode}{gs} | {mouse_state} {act}")
            else:
                if mode == "DEMO":
                    demo.draw_world(frame)
                    draw(frame, tracks, last_label)
                    demo.draw_hud(frame, now)
                    cv2.putText(frame, f"FPS {fps:5.1f}  infer {infer_ms:4.1f} ms",
                                (frame.shape[1] - 250, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                    hint = demo.hint
                else:
                    draw(frame, tracks, last_label)
                    cv2.putText(frame,
                                f"FPS {fps:5.1f}  infer {infer_ms:4.1f} ms  "
                                f"gru {gru_ms:4.2f} ms  tracks {len(tracks)}",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
                    extra = f"geom {geom}"
                    if mouse is not None:
                        extra += f" | {mouse_state} {mouse.current_action(now, geom)}"
                        if mouse.cursor is not None:
                            extra += f" | {int(mouse.cursor[0])},{int(mouse.cursor[1])}"
                    cv2.putText(frame, extra, (10, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
                    if pfeat is not None:
                        bits = "".join("1" if f else "0" for f in pfeat.fingers)
                        cv2.putText(frame,
                                    f"pinch {pfeat.pinch:.2f} dKn {pfeat.d_knuckle:.2f} "
                                    f"aIdx {pfeat.angles[1]:3.0f} {bits}",
                                    (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (220, 220, 220), 1)
                    hint = "Esc/q quit | p pause | r resume"
                    if args.game:
                        hint += f" | both hands open = {args.demo}"
                hint_y = frame.shape[0] - 34 if mode == "DEMO" else frame.shape[0] - 12
                cv2.putText(frame, hint, (10, hint_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("V1+V2 live", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if args.game and key == ord("g"):
                    mode = "MOUSE" if mode == "DEMO" else "DEMO"
                    need_reset = True
                    enter_since = None
                    lost_since = None
                    if mouse is not None:
                        mouse.on_hand_lost()
                        mouse_state = "RUNNING"
                if mouse is not None and key == ord("p"):
                    mouse.pause()
                    mouse_state = "PAUSED"
                if args.game and mode == "DEMO" and key == ord("r"):
                    demo.reset(now)
                elif mouse is not None and key == ord("r"):
                    mouse.resume()
                    mouse_state = "RUNNING"

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
