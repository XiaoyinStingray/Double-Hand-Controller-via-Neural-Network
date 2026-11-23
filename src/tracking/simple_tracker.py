"""Lightweight multi-hand tracker with per-track EMA smoothing.

Used by the live preview and (later) the two-hand game. It does not try to
classify handedness; it keeps temporal identity by IoU matching and smooths
box/keypoints so the drawing/game does not jitter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class HandObservation:
    box: np.ndarray      # (4,) x1,y1,x2,y2 in pixels
    kpts: np.ndarray     # (21,2) pixels
    score: float


@dataclass
class HandTrack:
    track_id: int
    box: np.ndarray
    kpts: np.ndarray
    score: float
    last_t: float
    hits: int = 1
    history: list = field(default_factory=list)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _ema(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * new + (1.0 - alpha) * old


class SimpleHandTracker:
    def __init__(self, iou_thr: float = 0.2, max_age: float = 0.3,
                 alpha_box: float = 0.6, alpha_kpt: float = 0.5):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.alpha_box = alpha_box
        self.alpha_kpt = alpha_kpt
        self.tracks: list[HandTrack] = []
        self._next_id = 0

    def _new_track(self, obs: HandObservation, now: float) -> HandTrack:
        tr = HandTrack(self._next_id, obs.box.copy(), obs.kpts.copy(),
                       obs.score, now, hits=1)
        self._next_id += 1
        return tr

    def update(self, obs_list: list[HandObservation], now: float) -> list[HandTrack]:
        # drop stale tracks
        self.tracks = [t for t in self.tracks if now - t.last_t <= self.max_age]

        if not obs_list:
            return sorted(self.tracks, key=lambda t: (t.box[0] + t.box[2]) / 2)

        if not self.tracks:
            self.tracks = [self._new_track(o, now) for o in obs_list]
            return sorted(self.tracks, key=lambda t: (t.box[0] + t.box[2]) / 2)

        # cost = 1 - IoU; reject weak matches
        n_t, n_o = len(self.tracks), len(obs_list)
        cost = np.ones((n_t, n_o), dtype=np.float64)
        for i, tr in enumerate(self.tracks):
            for j, ob in enumerate(obs_list):
                v = _iou(tr.box, ob.box)
                cost[i, j] = 1.0 - v if v >= self.iou_thr else 10.0

        rows, cols = linear_sum_assignment(cost)
        matched_t, matched_o = set(), set()
        for i, j in zip(rows, cols):
            if cost[i, j] >= 10.0:
                continue
            tr, ob = self.tracks[i], obs_list[j]
            tc = np.array([(tr.box[0] + tr.box[2]) / 2, (tr.box[1] + tr.box[3]) / 2])
            ts = np.array([tr.box[2] - tr.box[0], tr.box[3] - tr.box[1]])
            oc = np.array([(ob.box[0] + ob.box[2]) / 2, (ob.box[1] + ob.box[3]) / 2])
            os_ = np.array([ob.box[2] - ob.box[0], ob.box[3] - ob.box[1]])
            nc = _ema(tc, oc, self.alpha_box)
            ns = _ema(ts, os_, self.alpha_box)
            tr.box = np.array([nc[0] - ns[0] / 2, nc[1] - ns[1] / 2,
                               nc[0] + ns[0] / 2, nc[1] + ns[1] / 2])
            tr.kpts = _ema(tr.kpts, ob.kpts, self.alpha_kpt)
            tr.score = ob.score
            tr.last_t = now
            tr.hits += 1
            matched_t.add(i)
            matched_o.add(j)

        for j, ob in enumerate(obs_list):
            if j not in matched_o:
                self.tracks.append(self._new_track(ob, now))

        return sorted(self.tracks, key=lambda t: (t.box[0] + t.box[2]) / 2)
