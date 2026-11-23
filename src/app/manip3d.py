"""Pinch-driven 3D object viewer rendered over the camera frame.

One pinched hand spins the object like a trackball; two pinched hands move it
and scale it. Release keeps the angular momentum. No 3D library: the mesh is
projected and shaded by hand.
"""

import math

import cv2
import numpy as np

FOCAL = 520.0
CAM_Z = 3.2
BASE_COLOR = (255, 170, 60)
AMBIENT = 0.50


def _icosphere(subdiv=1):
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    def norm(v):
        n = math.sqrt(sum(c * c for c in v))
        return [c / n for c in v]

    verts = [norm(v) for v in verts]
    for _ in range(subdiv):
        cache = {}
        new_faces = []

        def mid(a, b):
            key = (a, b) if a < b else (b, a)
            if key not in cache:
                v = [(verts[a][i] + verts[b][i]) / 2.0 for i in range(3)]
                verts.append(norm(v))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces
    return np.asarray(verts), np.asarray(faces)


class Manip3D:
    def __init__(self, width: int, height: int, subdiv: int = 1):
        self.w = int(width)
        self.h = int(height)
        self.verts, self.faces = _icosphere(subdiv)
        self.light = np.asarray([-0.35, -0.55, 0.75], dtype=np.float64)
        self.light /= np.linalg.norm(self.light)
        self.hint = "one hand pinch: spin | two hands: move + scale | r reset | fist 0.8s exits | g mouse | q quit"
        self.reset(0.0)

    def reset(self, now: float) -> None:
        self.center = np.asarray([self.w * 0.5, self.h * 0.48], dtype=np.float64)
        self.center_v = np.zeros(2)
        self.yaw = 0.0
        self.pitch = -0.25
        self.roll = 0.0
        self.yaw_v = 0.0
        self.pitch_v = 0.0
        self.scale = 1.0
        self.pinched = {}
        self.prev_tip = {}
        self.prev_two = None
        self.grabbed = False
        self.inputs = {}

    def set_inputs(self, inputs, now: float) -> None:
        self.inputs = inputs

    def _rotation(self):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        return rz @ rx @ ry

    def update(self, now: float, dt: float) -> None:
        dt = min(max(dt, 0.0), 0.1)
        pinched = {}
        for side, h in self.inputs.items():
            was = self.pinched.get(side, False)
            th = 0.32 if was else 0.22
            is_p = bool(h["pinch"] < th)
            self.pinched[side] = is_p
            if is_p:
                pinched[side] = np.asarray(h["tip"], dtype=np.float64)

        if len(pinched) == 1:
            side, tip = next(iter(pinched.items()))
            prev = self.prev_tip.get(side)
            if prev is not None:
                d = tip - prev
                self.yaw += d[0] * 0.012
                self.pitch += d[1] * 0.012
                if dt > 1e-3:
                    self.yaw_v = 0.8 * self.yaw_v + 0.2 * (d[0] * 0.012 / dt)
                    self.pitch_v = 0.8 * self.pitch_v + 0.2 * (d[1] * 0.012 / dt)
            self.prev_tip = {side: tip}
            self.prev_two = None
        elif len(pinched) == 2:
            pts = list(pinched.values())
            mid = (pts[0] + pts[1]) / 2.0
            dvec = pts[1] - pts[0]
            dist = float(np.linalg.norm(dvec))
            ang = math.atan2(float(dvec[1]), float(dvec[0]))
            if self.prev_two is not None:
                pm, pd, pa = self.prev_two
                self.center += mid - pm
                self.center_v = (mid - pm) / max(dt, 1e-3)
                if pd > 8.0:
                    self.scale = min(max(self.scale * (dist / pd), 0.45), 2.3)
                dang = (ang - pa + math.pi) % (2 * math.pi) - math.pi
                self.roll += dang
            self.prev_two = (mid, dist, ang)
            self.prev_tip = {}
        else:
            self.prev_tip = {}
            self.prev_two = None
            self.center += self.center_v * dt
            self.yaw += (self.yaw_v + 0.18) * dt
            self.pitch += self.pitch_v * dt
            damp = math.exp(-dt / 0.55)
            self.yaw_v *= damp
            self.pitch_v *= damp
            self.center_v *= math.exp(-dt / 0.40)

        self.grabbed = bool(pinched)
        self.pitch = min(max(self.pitch, -1.25), 1.25)
        self.center[0] = min(max(self.center[0], 80), self.w - 80)
        self.center[1] = min(max(self.center[1], 70), self.h - 70)

    # ---------------- drawing ----------------
    def draw_world(self, frame) -> None:
        frame[:] = (frame * 0.38).astype(frame.dtype)

        # floor shadow
        cx, cy = int(self.center[0]), int(self.center[1])
        cv2.ellipse(frame, (cx, int(cy + 185 * self.scale)),
                    (int(105 * self.scale), int(24 * self.scale)),
                    0, 0, 360, (18, 18, 18), -1)

        rot = self._rotation()
        p = (self.verts @ rot.T) * self.scale
        denom = CAM_Z - p[:, 2]
        denom = np.maximum(denom, 0.4)
        sx = self.center[0] + FOCAL * p[:, 0] / denom
        sy = self.center[1] + FOCAL * p[:, 1] / denom
        pts = np.stack([sx, sy], axis=1)

        fz = p[self.faces][:, :, 2].mean(axis=1)
        order = np.argsort(fz)
        base = np.asarray(BASE_COLOR, dtype=np.float64)
        for fi in order:
            face = self.faces[fi]
            v0, v1, v2 = self.verts[face[0]], self.verts[face[1]], self.verts[face[2]]
            n = np.cross(v1 - v0, v2 - v0)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n = rot @ (n / nn)
            if n[2] <= 0.0:
                continue
            shade = AMBIENT + (1.0 - AMBIENT) * max(0.0, float(n @ self.light))
            color = tuple(int(min(255, c * shade)) for c in base)
            poly = pts[face].astype(np.int32)
            cv2.fillPoly(frame, [poly], color)
            edge = tuple(int(min(255, c * (shade * 0.75))) for c in base)
            cv2.polylines(frame, [poly], True, edge, 1, cv2.LINE_AA)

        hull = cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.int32))
        cv2.polylines(frame, [hull], True, (120, 80, 20), 7, cv2.LINE_AA)
        cv2.polylines(frame, [hull], True, (255, 220, 140), 2, cv2.LINE_AA)

        # hand markers
        colors = {"Left": (0, 200, 255), "Right": (255, 160, 0)}
        for side, h in self.inputs.items():
            tx, ty = int(h["tip"][0]), int(h["tip"][1])
            c = colors.get(side, (255, 255, 255))
            if self.pinched.get(side):
                cv2.circle(frame, (tx, ty), 14, c, 3, cv2.LINE_AA)
                cv2.circle(frame, (tx, ty), 3, c, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (tx, ty), 14, c, 1, cv2.LINE_AA)
        if len(self.pinched) == 2 and all(self.pinched.get(s, False) for s in self.pinched):
            a = self.inputs["Left"]["tip"]
            b = self.inputs["Right"]["tip"]
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     (200, 200, 200), 1, cv2.LINE_AA)

    def draw_hud(self, frame, now: float) -> None:
        state = "GRAB" if self.grabbed else "IDLE"
        text = f"{state}  scale {self.scale:.2f}  yaw {math.degrees(self.yaw):4.0f}"
        cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 4)
        cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

    def draw(self, frame, now: float) -> None:
        self.draw_world(frame)
        self.draw_hud(frame, now)

    def status(self) -> str:
        return f"{'GRAB' if self.grabbed else 'IDLE'} s{self.scale:.2f}"
