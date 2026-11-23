"""Geometric gesture rules + stabilizer for the live mouse layer."""

import numpy as np

PINCH_ENTER = 0.25
PINCH_EXIT = 0.35
STABLE_FRAMES = 3
MIN_HOLD_S = 0.15


class GestureRules:
    def __init__(self):
        self.pinch_active = False

    def reset(self):
        self.pinch_active = False

    def update(self, feat):
        pinch_shape = (feat.pinch < PINCH_ENTER and
                       feat.pinch <= feat.d_knuckle * 1.15)
        if pinch_shape:
            self.pinch_active = True
        elif feat.pinch > PINCH_EXIT or feat.pinch > feat.d_knuckle * 1.30:
            self.pinch_active = False

        four = feat.fingers[1:]
        if self.pinch_active:
            return "PINCH"
        if not np.any(four):
            return "FIST"
        if int(four.sum()) == 1 and four[0]:
            return "POINT"
        if int(four.sum()) >= 3:
            return "OPEN"
        return "NONE"


class Stabilizer:
    def __init__(self, enter_frames=STABLE_FRAMES, exit_frames=STABLE_FRAMES,
                 min_hold_s=MIN_HOLD_S):
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.min_hold_s = min_hold_s
        self.stable = "NONE"
        self.candidate = None
        self.count = 0
        self.last_change = -1e9

    def update(self, raw, now):
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

    def held(self, now):
        return now - self.last_change
