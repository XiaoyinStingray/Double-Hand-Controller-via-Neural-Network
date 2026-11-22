"""Mouse action layer: move, pinch click/drag, scroll, pause.

The controller is UI-agnostic: the caller passes the stable gesture label and
features each frame. Use dry_run=True to log actions without touching the OS
mouse (for tests and first-time tuning).
"""

from __future__ import annotations

import ctypes
import math
import time

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
                 drag_hold_s: float = 0.35,
                 click_cooldown_s: float = 0.3,
                 click_min_hold_s: float = 0.12,
                 click_context_s: float = 0.6,
                 deadzone_px: float = 1.5,
                 scroll_speed: float = 90.0,
                 scroll_interval_s: float = 0.03,
                 scroll_invert: bool = False,
                 mincutoff: float = 1.0,
                 beta: float = 0.012,
                 cursor_source: str = "tip",
                 pinch_latch: float = 0.42,
                 pinch_unlatch: float = 0.52):
        self.dry_run = dry_run
        self.drag_hold_s = drag_hold_s
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

        self.pinch_active = False
        self.pinch_start = 0.0
        self.dragging = False
        self.last_click = -1e9
        self.last_scroll = -1e9
        self.pinch_latched = False
        self.latched_cursor = None
        self.last_point_t = -1e9
        self.click_armed = False
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
        self.pinch_active = False
        self.pinch_latched = False
        self.latched_cursor = None
        self.click_armed = False
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
        self.pinch_active = False
        self.pinch_latched = False
        self.latched_cursor = None
        self.click_armed = False
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

        # pinch: short pinch = click on release; hold = drag.
        # Require recent pointing context and a minimum pinch duration so a
        # transient misclassification cannot fire a click.
        is_pinch = stable == "PINCH"
        if is_pinch and not self.pinch_active:
            self.pinch_active = True
            self.pinch_start = now
            self.click_armed = False
            self.click_allowed = (self.cursor is not None and
                                  (now - self.last_point_t) <= self.click_context_s)
        elif is_pinch and self.pinch_active:
            if self.click_allowed and not self.click_armed and (
                    now - self.pinch_start) >= self.click_min_hold_s:
                self.click_armed = True
            if self.click_allowed and not self.dragging and (
                    now - self.pinch_start) >= self.drag_hold_s:
                self._down()
                self.dragging = True
                self._set_action("DRAG", now)
        elif not is_pinch and self.pinch_active:
            self.pinch_active = False
            if self.dragging:
                self._up()
                self.dragging = False
                self._set_action("DROP", now)
            elif (self.click_armed and
                  now - self.last_click >= self.click_cooldown_s):
                self._click()
                self.last_click = now
                self._set_action("CLICK", now)
            self.click_armed = False
            self.click_allowed = False

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
        if self.pinch_latched and not self.pinch_active:
            return "HOLD"
        if now - self.action_t <= 0.5:
            return self.action
        return fallback
