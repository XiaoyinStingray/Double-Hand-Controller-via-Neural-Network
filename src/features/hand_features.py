"""Per-hand features from 21 keypoints (shared by V0/V1/V2)."""

from dataclasses import dataclass
from collections import deque
import math

import numpy as np

FINGER_MCP = (5, 9, 13, 17)
FINGER_PIP = (6, 10, 14, 18)
FINGER_TIP = (8, 12, 16, 20)
PALM_IDX = (0, 5, 9, 13, 17)

FINGER_STRAIGHT_DEG = 25.0
FINGER_FAR_RATIO = 1.1
THUMB_OUT_DIST = 0.55


def _angle(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 180.0
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(c))


@dataclass
class HandFeatures:
    side: str
    q: np.ndarray          # (21,2) wrist-origin, palm-scaled, mirrored
    vel: np.ndarray        # (21,2) per-second
    fingers: np.ndarray    # (5,) bool
    angles: np.ndarray     # (5,) deg
    pinch: float
    d_knuckle: float
    palm: np.ndarray       # (2,) px
    tip: np.ndarray        # (2,) px
    v_palm: np.ndarray     # (2,) px/s

    def vector(self):
        return np.concatenate([self.q.ravel(), self.vel.ravel(), [1.0]]).astype(np.float32)


def extract_features(pts, side, prev_q, dt, v_palm):
    """pts: (21,2) pixel keypoints. Returns HandFeatures or None."""
    palm_size = float(np.linalg.norm(pts[0] - pts[9]))
    if not np.isfinite(palm_size) or palm_size < 1e-6:
        return None
    q = (pts - pts[0]) / palm_size
    if side == "Left":
        q[:, 0] *= -1.0

    fingers = np.zeros(5, dtype=bool)
    angles = np.zeros(5, dtype=np.float64)
    for k, (mcp, pip, tip) in enumerate(zip(FINGER_MCP, FINGER_PIP, FINGER_TIP), start=1):
        ang = _angle(pts[pip] - pts[mcp], pts[tip] - pts[pip])
        angles[k] = ang
        far = np.linalg.norm(pts[tip] - pts[0]) > FINGER_FAR_RATIO * max(
            np.linalg.norm(pts[pip] - pts[0]), 1e-6)
        fingers[k] = (ang < FINGER_STRAIGHT_DEG) and far
    fingers[0] = float(np.linalg.norm(q[4] - q[17])) > THUMB_OUT_DIST

    pinch = float(np.linalg.norm(q[4] - q[8]))
    d_knuckle = float(min(np.linalg.norm(q[4] - q[5]), np.linalg.norm(q[4] - q[6])))

    if prev_q is None or dt <= 0:
        vel = np.zeros((21, 2), dtype=np.float64)
    else:
        vel = np.clip((q - prev_q) / dt, -30.0, 30.0)

    palm = pts[list(PALM_IDX)].mean(axis=0)
    return HandFeatures(side, q, vel, fingers, angles, pinch, d_knuckle,
                        palm, pts[8], np.asarray(v_palm, dtype=np.float64))


class PalmVelocity:
    def __init__(self, tau=0.1):
        self.tau = tau
        self.prev = None
        self.prev_t = None
        self.v = np.zeros(2, dtype=np.float64)

    def reset(self):
        self.prev = None
        self.prev_t = None
        self.v[:] = 0.0

    def update(self, center, t):
        if self.prev is not None and self.prev_t is not None:
            dt = t - self.prev_t
            if dt > 1e-4:
                raw = (center - self.prev) / dt
                alpha = 1.0 - math.exp(-dt / self.tau)
                self.v += alpha * (raw - self.v)
        self.prev = np.asarray(center, dtype=np.float64).copy()
        self.prev_t = t
        return self.v


class TrackFeatureState:
    """One per track: previous q + sliding window of 85-d vectors."""

    def __init__(self, T=16):
        self.T = T
        self.side = None
        self.prev_q = None
        self.prev_t = None
        self.window = deque(maxlen=T)
        self.vel = PalmVelocity()
        self.last_t = -1e9
        self.label = ""
        self.conf = 0.0

    def reset(self):
        self.prev_q = None
        self.prev_t = None
        self.window.clear()
        self.vel.reset()

    def update(self, pts, side, now):
        if self.side != side:
            self.side = side
            self.reset()
        dt = 1.0 / 30.0 if self.prev_t is None else min(
            max(now - self.prev_t, 1.0 / 120.0), 1.0 / 10.0)
        vp = self.vel.update(pts[list(PALM_IDX)].mean(axis=0), now)
        feat = extract_features(pts, side, self.prev_q, dt, vp)
        if feat is None:
            return None
        self.prev_q = feat.q
        self.prev_t = now
        self.last_t = now
        self.window.append(feat.vector())
        return feat

    def ready(self):
        return len(self.window) == self.T

    def batch(self):
        return np.stack(self.window)[None].astype(np.float32)
