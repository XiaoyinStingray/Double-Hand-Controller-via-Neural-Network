"""Mouse action layer: move, pinch click/drag, scroll, pause.

The controller is UI-agnostic: the caller passes the stable gesture label and
features each frame. Use dry_run=True to log actions without touching the OS
mouse (for tests and first-time tuning).
"""

from __future__ import annotations

import ctypes
import math
import time

import numpy as np

from src.control.one_euro import OneEuroFilter2D

try:
    import pyautogui as _pyautogui
except Exception:  # pragma: no cover - only relevant on headless machines
    _pyautogui = None


def set_dpi_awareness() -> None:
    """Make pyautogui coordinates match the physical screen on Windows."""
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)  # per-monitor v2
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor
    except Exception:
        pass


class MouseController:
    def __init__(self, dry_run: bool = False,
                 click_cooldown_s: float = 0.3,
                 click_min_hold_s: float = 0.04,
                 click_context_s: float = 0.6,
                 deadzone_px: float = 1.5,
                 scroll_speed: float = 90.0,
                 scroll_interval_s: float = 0.03,
                 scroll_invert: bool = False,
                 mincutoff: float = 1.0,
                 beta: float = 0.012,
                 cursor_source: str = "tip",
                 pinch_latch: float = 0.42,
                 pinch_unlatch: float = 0.52,
                 raw_pinch_enter: float = 0.22,
                 raw_pinch_exit: float = 0.30,
                 drag_move_px: float = 12.0,
                 drag_gain: float = 2.0):
        self.dry_run = dry_run
        self.click_cooldown_s = click_cooldown_s
        self.click_min_hold_s = click_min_hold_s
        self.click_context_s = click_context_s
        self.deadzone_px = deadzone_px
        self.scroll_speed = scroll_speed
        self.scroll_interval_s = scroll_interval_s
        self.scroll_invert = scroll_invert
        self.cursor_source = cursor_source
        self.pinch_latch = pinch_latch
        self.pinch_unlatch = pinch_unlatch
        self.raw_pinch_enter = raw_pinch_enter
        self.raw_pinch_exit = raw_pinch_exit
        self.drag_move_px = drag_move_px
        self.drag_gain = drag_gain

        set_dpi_awareness()
        if _pyautogui is not None:
            try:
                _pyautogui.FAILSAFE = True
                _pyautogui.PAUSE = 0
            except Exception:
                pass
            try:
                self.screen_w, self.screen_h = _pyautogui.size()
            except Exception:
                self.screen_w, self.screen_h = 2560, 1440
        else:
            self.screen_w, self.screen_h = 2560, 1440
        if not dry_run and _pyautogui is None:
            raise RuntimeError("pyautogui unavailable; use --dry-run")

        self.filter = OneEuroFilter2D(mincutoff=mincutoff, beta=beta)
        self.last_sent = None
        self.cursor = None
        self.was_pointing = False

        self.raw_pinch = False
        self.pinch_start = 0.0
        self.pinch_origin_palm = None
        self.dragging = False
        self.drag_origin_palm = None
        self.drag_origin_cursor = None
        self.last_click = -1e9
        self.last_scroll = -1e9
        self.pinch_latched = False
        self.latched_cursor = None
        self.last_point_t = -1e9
        self.click_allowed = False

        self.paused = False
        self.action = "IDLE"
        self.action_t = 0.0

    # ---------------- low-level actions ----------------
    def _set_action(self, name: str, now: float) -> None:
        self.action = name
        self.action_t = now

    def _move(self, x: float, y: float) -> None:
        if not self.dry_run and _pyautogui is not None:
            _pyautogui.moveTo(int(round(x)), int(round(y)), duration=0)

    def _click(self) -> None:
        if self.dry_run:
            print("[dry] CLICK")
        elif _pyautogui is not None:
            _pyautogui.click()

    def _down(self) -> None:
        if self.dry_run:
            print("[dry] MOUSE DOWN (drag start)")
        elif _pyautogui is not None:
            _pyautogui.mouseDown(button="left")

    def _up(self) -> None:
        if self.dry_run:
            print("[dry] MOUSE UP (drag end)")
        elif _pyautogui is not None:
            _pyautogui.mouseUp(button="left")

    def _scroll(self, ticks: int) -> None:
        if self.dry_run:
            print(f"[dry] SCROLL {ticks:+d}")
        elif _pyautogui is not None:
            _pyautogui.scroll(ticks)

    # ---------------- state changes ----------------
    def release_all(self) -> None:
        if self.dragging:
            self._up()
        self.dragging = False
        self.raw_pinch = False
        self.pinch_origin_palm = None
        self.drag_origin_palm = None
        self.drag_origin_cursor = None
        self.pinch_latched = False
        self.latched_cursor = None
        self.click_allowed = False

    def pause(self) -> None:
        self.release_all()
        self.paused = True
        self.cursor = None
        self.last_sent = None
        self.last_point_t = -1e9
        self.filter.reset()

    def resume(self) -> None:
        self.paused = False
        self.cursor = None
        self.last_sent = None
        self.last_point_t = -1e9
        self.filter.reset()

    def on_hand_lost(self) -> None:
        """Immediate safe release; never emit a click on tracking loss."""
        if self.dragging:
            self._up()
        self.dragging = False
        self.raw_pinch = False
        self.pinch_origin_palm = None
        self.drag_origin_palm = None
        self.drag_origin_cursor = None
        self.pinch_latched = False
        self.latched_cursor = None
        self.click_allowed = False
        self.was_pointing = False
        self.cursor = None
        self.last_sent = None
        self.last_point_t = -1e9
        self.filter.reset()

    # ---------------- per-frame update ----------------
    def update(self, stable: str, feat, mapper, frame_w: int, frame_h: int,
               now: float) -> None:
        if self.paused or feat is None:
            return

        # Pre-pinch latch: the index fingertip moves while forming a pinch.
        # Freeze the cursor BEFORE the gesture is fully classified so the
        # click lands where the user aimed.
        if feat.pinch < self.pinch_latch and not self.pinch_latched:
            self.pinch_latched = True
            self.latched_cursor = self.cursor
            self._set_action("HOLD", now)
        elif self.pinch_latched and feat.pinch > self.pinch_unlatch:
            self.pinch_latched = False
            self.filter.reset()
            self.last_sent = self.latched_cursor

        if stable == "POINT":
            self.last_point_t = now

        # Raw pinch (bypasses the 3-frame stabilizer): tap = click on release;
        # move the palm while pinching = drag. No time-based drag.
        threshold = self.raw_pinch_exit if self.raw_pinch else self.raw_pinch_enter
        raw_pinch = feat.pinch < threshold
        if raw_pinch and not self.raw_pinch and feat.pinch > feat.d_knuckle * 1.15:
            raw_pinch = False

        if raw_pinch and not self.raw_pinch:
            self.raw_pinch = True
            self.pinch_start = now
            self.pinch_origin_palm = feat.palm.copy()
            self.click_allowed = (self.cursor is not None and
                                  (now - self.last_point_t) <= self.click_context_s)
        elif raw_pinch and self.raw_pinch:
            if self.dragging:
                delta = feat.palm - self.drag_origin_palm
                target = self.drag_origin_cursor + self.drag_gain * delta
                tx = float(np.clip(target[0], 0, self.screen_w - 1))
                ty = float(np.clip(target[1], 0, self.screen_h - 1))
                self._move(tx, ty)
                self.cursor = (tx, ty)
                self.last_sent = (tx, ty)
                self._set_action("DRAG", now)
            elif self.click_allowed and self.cursor is not None:
                moved = float(np.linalg.norm(feat.palm - self.pinch_origin_palm))
                if moved >= self.drag_move_px and (now - self.pinch_start) >= 0.05:
                    self._down()
                    self.dragging = True
                    self.drag_origin_palm = feat.palm.copy()
                    self.drag_origin_cursor = self.cursor
                    self._set_action("DRAG", now)
        elif not raw_pinch and self.raw_pinch:
            self.raw_pinch = False
            if self.dragging:
                self._up()
                self.dragging = False
                self._set_action("DROP", now)
            elif (self.click_allowed and
                  now - self.pinch_start >= self.click_min_hold_s and
                  now - self.last_click >= self.click_cooldown_s):
                self._click()
                self.last_click = now
                self._set_action("CLICK", now)
            self.click_allowed = False
            self.pinch_origin_palm = None

        # cursor follows a stable POINT only, and freezes during pre-pinch
        if stable == "POINT" and not self.pinch_latched:
            if not self.was_pointing:
                self.filter.reset()
                self.last_sent = None
                self.was_pointing = True
            point = feat.tip if self.cursor_source == "tip" else feat.pts[5]
            sx, sy = mapper.map(point, frame_w, frame_h,
                                self.screen_w, self.screen_h)
            fx, fy = self.filter((sx, sy), now)
            if self.last_sent is None or math.hypot(
                    fx - self.last_sent[0], fy - self.last_sent[1]) >= self.deadzone_px:
                self._move(fx, fy)
                self.last_sent = (fx, fy)
            self.cursor = (fx, fy)
            self._set_action("MOVE", now)
        else:
            self.was_pointing = False

        # open palm + vertical velocity = wheel
        if stable == "OPEN":
            vy = float(feat.v_palm[1])
            if abs(vy) > self.scroll_speed and (
                    now - self.last_scroll) >= self.scroll_interval_s:
                mag = max(1, min(4, int(abs(vy) / self.scroll_speed)))
                ticks = mag if vy < 0 else -mag
                if self.scroll_invert:
                    ticks = -ticks
                self._scroll(ticks)
                self.last_scroll = now
                self._set_action(f"SCROLL {ticks:+d}", now)

    def current_action(self, now: float, fallback: str = "IDLE") -> str:
        if self.paused:
            return "PAUSED"
        if self.dragging:
            return "DRAG"
        if self.pinch_latched and not self.raw_pinch:
            return "HOLD"
        if now - self.action_t <= 0.5:
            return self.action
        return fallback
