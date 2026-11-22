"""Camera -> screen coordinate mapping for V0 (fixed active region)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FixedRegionMapper:
    """Map the central camera region to the full primary screen.

    The user points inside the active region; values outside are clamped.
    A 5-point homography mapper will replace this in slice D.
    """

    x_range: tuple = (0.25, 0.75)
    y_range: tuple = (0.30, 0.70)

    def map(self, tip_xy, frame_w: int, frame_h: int,
            screen_w: int, screen_h: int):
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        nx = (float(tip_xy[0]) / frame_w - x0) / (x1 - x0)
        ny = (float(tip_xy[1]) / frame_h - y0) / (y1 - y0)
        nx = float(np.clip(nx, 0.0, 1.0))
        ny = float(np.clip(ny, 0.0, 1.0))
        return nx * (screen_w - 1), ny * (screen_h - 1)

    def region_px(self, frame_w: int, frame_h: int):
        """Active region in pixels, for the preview overlay."""
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        return (int(x0 * frame_w), int(y0 * frame_h),
                int(x1 * frame_w), int(y1 * frame_h))
