"""1-Euro filter for 2D screen coordinates."""

from __future__ import annotations

import math

import numpy as np


class OneEuroFilter2D:
    """Speed-adaptive low-pass filter (Casiez et al., CHI 2012).

    Slow movement -> strong smoothing (less jitter).
    Fast movement -> weak smoothing (less lag).
    """

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.007,
                 dcutoff: float = 1.0):
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x: np.ndarray | None = None
        self.dx = np.zeros(2, dtype=np.float64)
        self.t_prev: float | None = None

    def reset(self) -> None:
        self.x = None
        self.dx[:] = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, value, t: float) -> np.ndarray:
        v = np.asarray(value, dtype=np.float64)
        if self.x is None or self.t_prev is None:
            self.x = v.copy()
            self.t_prev = t
            return self.x.copy()

        dt = t - self.t_prev
        if dt <= 0:
            return self.x.copy()
        dt = min(max(dt, 1.0 / 120.0), 1.0 / 10.0)

        raw_dx = (v - self.x) / dt
        a_d = self._alpha(self.dcutoff, dt)
        self.dx = a_d * raw_dx + (1.0 - a_d) * self.dx

        cutoff = self.mincutoff + self.beta * np.abs(self.dx)
        a = np.array([self._alpha(c, dt) for c in cutoff])
        self.x = a * v + (1.0 - a) * self.x
        self.t_prev = t
        return self.x.copy()
