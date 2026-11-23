"""Lightweight multi-hand tracker with per-track EMA smoothing.

Key behaviors for the demo:
  * IoU matching keeps temporal identity (max_age seconds).
  * A new track must be seen min_hits times before it is reported, which
    suppresses one-frame background false positives.
  * Track IDs are recycled, so entering/leaving does not grow #0 -> #57.
  * At most max_tracks hands are tracked (2 for the two-hand game).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class HandObservation:
    box: np.ndarray      # (4,) x1,y1,x2,y2 pixels
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
    def __init__(self, iou_thr: float = 0.2, max_age: float = 1.0,
                 alpha_box: float = 0.6, alpha_kpt: float = 0.5,
                 max_tracks: int = 2, min_hits: int = 2):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.alpha_box = alpha_box
        self.alpha_kpt = alpha_kpt
        self.max_tracks = max_tracks
        self.min_hits = min_hits
        self.tracks: list[HandTrack] = []
        self._next_id = 0
        self._free_ids: list[int] = []

    # ---------------- id pool ----------------
    def _alloc_id(self) -> int:
        if self._free_ids:
            self._free_ids.sort()
            return self._free_ids.pop(0)
        tid = self._next_id
        self._next_id += 1
        return tid

    def _release(self, tid: int) -> None:
        if tid not in self._free_ids:
            self._free_ids.append(tid)

    def _new_track(self, obs: HandObservation, now: float) -> HandTrack:
        return HandTrack(self._alloc_id(), obs.box.copy(), obs.kpts.copy(),
                         obs.score, now, hits=1)

    def _confirmed(self) -> list[HandTrack]:
        out = [t for t in self.tracks if t.hits >= self.min_hits]
        return sorted(out, key=lambda t: (t.box[0] + t.box[2]) / 2)

    # ---------------- per-frame update ----------------
    def update(self, obs_list: list[HandObservation], now: float) -> list[HandTrack]:
        # only the strongest max_tracks observations can create tracks
        obs_list = sorted(obs_list, key=lambda o: o.score, reverse=True)[: self.max_tracks]

        # expire old tracks (confirmed tracks survive brief occlusions)
        for tr in list(self.tracks):
            if now - tr.last_t > self.max_age:
                self._release(tr.track_id)
                self.tracks.remove(tr)

        if not obs_list:
            # one-frame false positives should not occupy a slot/ID
            for tr in list(self.tracks):
                if tr.hits < self.min_hits:
                    self._release(tr.track_id)
                    self.tracks.remove(tr)
            return self._confirmed()

        if not self.tracks:
            for obs in obs_list:
                if len(self.tracks) < self.max_tracks:
                    self.tracks.append(self._new_track(obs, now))
            return self._confirmed()

        # matched cost = 1 - IoU; weak pairs are rejected
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

        # unmatched tracks: keep confirmed ones for max_age; drop unconfirmed
        kept = []
        for i, tr in enumerate(self.tracks):
            if i in matched_t:
                kept.append(tr)
            elif tr.hits >= self.min_hits and now - tr.last_t <= self.max_age:
                kept.append(tr)
            else:
                self._release(tr.track_id)
        self.tracks = kept

        # unmatched observations: create a track if a slot is free
        for j, ob in enumerate(obs_list):
            if j not in matched_o and len(self.tracks) < self.max_tracks:
                self.tracks.append(self._new_track(ob, now))

        return self._confirmed()
