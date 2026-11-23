"""Two-hand fruit slicing rendered over the camera frame.

Each index fingertip leaves a blade trail. A fast tip segment that crosses a
fruit splits it and adds combo score; bombs cost a life. Ninety seconds or
three misses per round.
"""

from collections import deque
from dataclasses import dataclass
import math
import random

import cv2
import numpy as np

GRAVITY = 1100.0
TIP_SPEED_PX = 14.0

FRUIT_STYLES = [
    ((40, 120, 40), (60, 60, 220)),    # watermelon
    ((20, 140, 255), (60, 190, 255)),  # orange
    ((30, 230, 250), (90, 250, 255)),  # lemon
    ((60, 150, 60), (120, 220, 140)),  # kiwi
]


@dataclass
class Fruit:
    x: float
    y: float
    vx: float
    vy: float
    r: float
    outer: tuple
    inner: tuple
    bomb: bool = False


@dataclass
class Half:
    x: float
    y: float
    vx: float
    vy: float
    r: float
    ang: float
    va: float
    color: tuple
    life: float = 1.4


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    r: float
    color: tuple
    life: float


class FruitNinja:
    def __init__(self, width: int, height: int,
                 duration: float = 90.0,
                 lives: int = 3,
                 spawn_interval: float = 0.65,
                 seed: int = 0):
        self.w = int(width)
        self.h = int(height)
        self.duration = duration
        self.start_lives = lives
        self.spawn_interval = spawn_interval
        self.rng = random.Random(seed)
        self.hint = "fast index fingers slice | r restart | fist exit | g mouse | q quit"
        self.trail = {"Left": deque(maxlen=10), "Right": deque(maxlen=10)}
        self.prev_tip = {}
        self.reset(0.0)

    def reset(self, now: float) -> None:
        self.score = 0
        self.lives = self.start_lives
        self.combo = 0
        self.last_slice = -1e9
        self.fruits = []
        self.halves = []
        self.particles = []
        self.trail = {"Left": deque(maxlen=10), "Right": deque(maxlen=10)}
        self.prev_tip = {}
        self.start_t = now
        self.next_spawn = now + 0.6
        self.over = False
        self.over_t = 0.0
        self.reason = ""

    def _finish(self, now: float, reason: str) -> None:
        if self.over:
            return
        self.over = True
        self.over_t = now
        self.reason = reason

    def _spawn(self, now: float) -> None:
        bomb = self.rng.random() < 0.10 and (now - self.start_t) > 8.0
        x = self.rng.uniform(70, self.w - 70)
        y = self.h + 30
        vy = -self.rng.uniform(690, 880)
        vx = (self.w * 0.5 - x) * self.rng.uniform(0.25, 0.60)
        r = self.rng.uniform(23, 33)
        if bomb:
            self.fruits.append(Fruit(x, y, vx, vy, 26, (30, 30, 30),
                                     (60, 60, 60), True))
        else:
            outer, inner = FRUIT_STYLES[self.rng.randrange(len(FRUIT_STYLES))]
            self.fruits.append(Fruit(x, y, vx, vy, r, outer, inner, False))

    # ---------------- input / slicing ----------------
    def set_inputs(self, inputs, now: float) -> None:
        if self.over:
            return
        for side in ("Left", "Right"):
            if side not in inputs:
                self.prev_tip.pop(side, None)
                self.trail[side].clear()
                continue
            tip = np.asarray(inputs[side]["tip"], dtype=np.float64)
            self.trail[side].append((float(tip[0]), float(tip[1])))
            prev = self.prev_tip.get(side)
            self.prev_tip[side] = tip
            if prev is None:
                continue
            seg = float(np.linalg.norm(tip - prev))
            if seg >= TIP_SPEED_PX:
                self._slice(prev, tip, now)

    def _slice(self, a, b, now: float) -> None:
        removed = []
        for i, f in enumerate(self.fruits):
            if _seg_dist(a, b, (f.x, f.y)) <= f.r + 3:
                removed.append(i)
                if f.bomb:
                    self.lives -= 1
                    self._burst(f, 24, (40, 40, 255), 220.0)
                    if self.lives <= 0:
                        self._finish(now, "BOMB")
                        break
                else:
                    self._split(f, a, b)
                    if now - self.last_slice <= 0.55:
                        self.combo = min(self.combo + 1, 9)
                    else:
                        self.combo = 1
                    self.last_slice = now
                    self.score += self.combo
        for i in reversed(removed):
            if i < len(self.fruits):
                self.fruits.pop(i)

    def _split(self, f: Fruit, a, b) -> None:
        ang = math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))
        nx, ny = -math.sin(ang), math.cos(ang)
        for s in (-1, 1):
            self.halves.append(Half(f.x + s * nx * f.r * 0.35,
                                    f.y + s * ny * f.r * 0.35,
                                    f.vx + s * nx * 130.0,
                                    f.vy * 0.55 + s * ny * 90.0,
                                    f.r, ang if s < 0 else ang + math.pi,
                                    s * 2.4, f.inner))
        self._burst(f, 16, f.inner, 180.0)

    def _burst(self, f: Fruit, n: int, color, speed: float) -> None:
        for _ in range(n):
            ang = self.rng.uniform(0, 2 * math.pi)
            sp = self.rng.uniform(0.3, 1.0) * speed
            self.particles.append(Particle(
                f.x, f.y, math.cos(ang) * sp, math.sin(ang) * sp,
                self.rng.uniform(2, 5), color, self.rng.uniform(0.35, 0.7)))

    # ---------------- per-frame ----------------
    def update(self, now: float, dt: float) -> None:
        if self.over:
            if now - self.over_t >= 4.0:
                self.reset(now)
            return
        dt = min(max(dt, 0.0), 0.10)
        if now - self.start_t >= self.duration:
            self._finish(now, "TIME")
            return

        if now >= self.next_spawn:
            self._spawn(now)
            self.next_spawn = now + self.spawn_interval

        alive = []
        for f in self.fruits:
            f.vy += GRAVITY * dt
            f.x += f.vx * dt
            f.y += f.vy * dt
            if f.y - f.r > self.h + 30:
                if not f.bomb:
                    self.lives -= 1
                    if self.lives <= 0:
                        self._finish(now, "MISS")
                        return
                continue
            alive.append(f)
        self.fruits = alive

        for h in self.halves:
            h.vy += GRAVITY * dt
            h.x += h.vx * dt
            h.y += h.vy * dt
            h.ang += h.va * dt
            h.life -= dt
        self.halves = [h for h in self.halves if h.life > 0]

        for p in self.particles:
            p.vy += GRAVITY * 0.5 * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.life -= dt
        self.particles = [p for p in self.particles if p.life > 0]

        if now - self.last_slice > 0.55:
            self.combo = 0

    # ---------------- drawing ----------------
    def _draw_fruit(self, frame, f: Fruit) -> None:
        c = (int(f.x), int(f.y))
        if f.bomb:
            cv2.circle(frame, c, int(f.r), (20, 20, 20), -1)
            cv2.circle(frame, c, int(f.r), (80, 80, 80), 2)
            top = (c[0] + int(f.r * 0.3), c[1] - int(f.r))
            cv2.line(frame, (c[0], c[1] - int(f.r)), top, (120, 120, 120), 3)
            cv2.circle(frame, top, 5, (0, 120, 255), -1)
            return
        cv2.circle(frame, c, int(f.r), f.outer, -1)
        cv2.circle(frame, c, int(f.r * 0.78), f.inner, -1)
        cv2.circle(frame, (c[0] - int(f.r * 0.25), c[1] - int(f.r * 0.25)),
                   max(2, int(f.r * 0.12)), (245, 245, 245), -1)

    def draw_world(self, frame) -> None:
        frame[:] = (frame * 0.55).astype(frame.dtype)

        for p in self.particles:
            a = max(0.0, min(1.0, p.life / 0.6))
            col = tuple(int(c * a) for c in p.color)
            cv2.circle(frame, (int(p.x), int(p.y)), int(p.r), col, -1)

        for f in self.fruits:
            self._draw_fruit(frame, f)

        for h in self.halves:
            c = (int(h.x), int(h.y))
            axes = (int(h.r), int(h.r))
            deg = math.degrees(h.ang) % 360
            cv2.ellipse(frame, c, axes, deg, 0, 180, h.color, -1)
            dx = h.r * math.cos(math.radians(deg))
            dy = h.r * math.sin(math.radians(deg))
            cv2.line(frame, (int(c[0] - dx), int(c[1] - dy)),
                     (int(c[0] + dx), int(c[1] + dy)), (240, 240, 240), 2)

        colors = {"Left": (0, 200, 255), "Right": (255, 160, 0)}
        for side, tr in self.trail.items():
            if len(tr) < 2:
                continue
            pts = np.asarray(tr, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (30, 30, 30), 7, cv2.LINE_AA)
            cv2.polylines(frame, [pts], False, colors[side], 4, cv2.LINE_AA)
            cv2.polylines(frame, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, tuple(pts[-1]), 6, colors[side], -1)

    def draw_hud(self, frame, now: float) -> None:
        left = max(0.0, self.duration - (now - self.start_t))
        text = f"SCORE {self.score}   LIVES {self.lives}   TIME {left:4.1f}s"
        cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68, (0, 0, 0), 5)
        cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68, (255, 255, 255), 2)
        if self.combo > 1:
            msg = f"COMBO x{self.combo}"
            size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.putText(frame, msg, ((self.w - size[0]) // 2, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(frame, msg, ((self.w - size[0]) // 2, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if self.over:
            msg = f"{self.reason}   SCORE {self.score}"
            size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 3)[0]
            cv2.putText(frame, msg, ((self.w - size[0]) // 2, self.h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 6)
            cv2.putText(frame, msg, ((self.w - size[0]) // 2, self.h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 3)

    def draw(self, frame, now: float) -> None:
        self.draw_world(frame)
        self.draw_hud(frame, now)

    def status(self) -> str:
        state = "OVER" if self.over else "PLAY"
        return f"{state} score {self.score} lives {self.lives}"


def _seg_dist(a, b, p) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(p[0]), float(p[1])
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / den))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))
