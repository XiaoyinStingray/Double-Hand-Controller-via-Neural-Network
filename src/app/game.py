"""Two-hand catch game rendered on top of the camera frame.

Left/right hand x controls the left/right paddle. Balls fall in each half;
catching one scores, missing costs a life. No extra dependency beyond OpenCV.
"""

from dataclasses import dataclass
import math
import random

import cv2

BALL_R = 9
PADDLE_W = 110
PADDLE_H = 14


@dataclass
class Ball:
    x: float
    y: float
    vy: float
    side: str


class HandGame:
    def __init__(self, width: int, height: int,
                 max_lives: int = 3,
                 duration: float = 60.0,
                 ball_interval: float = 1.1,
                 base_speed: float = 150.0,
                 speed_gain: float = 7.0,
                 max_speed: float = 360.0,
                 seed: int = 0):
        self.w = int(width)
        self.h = int(height)
        self.max_lives = max_lives
        self.duration = duration
        self.ball_interval = ball_interval
        self.base_speed = base_speed
        self.speed_gain = speed_gain
        self.max_speed = max_speed
        self.rng = random.Random(seed)
        self.paddle_y = self.h - 34.0
        self.paddle_x = {"Left": self.w * 0.25, "Right": self.w * 0.75}
        self.target = dict(self.paddle_x)
        self.balls = []
        self.score = 0
        self.lives = max_lives
        self.over = False
        self.over_t = 0.0
        self.start_t = 0.0
        self.next_spawn = 0.0
        self.spawn_side = "Left"

    # ---------------- state ----------------
    def reset(self, now: float) -> None:
        self.paddle_x = {"Left": self.w * 0.25, "Right": self.w * 0.75}
        self.target = dict(self.paddle_x)
        self.balls = []
        self.score = 0
        self.lives = self.max_lives
        self.over = False
        self.start_t = now
        self.next_spawn = now + 0.8
        self.spawn_side = "Left"

    def set_targets(self, targets) -> None:
        for side, x in targets.items():
            lo, hi = self._limits(side)
            self.target[side] = min(max(float(x), lo), hi)

    def _limits(self, side: str):
        half = self.w / 2.0
        lo = PADDLE_W / 2.0 + 4.0
        hi = half - PADDLE_W / 2.0 - 4.0
        if side == "Right":
            lo += half
            hi += half
        return lo, hi

    def _spawn(self, now: float) -> None:
        side = self.spawn_side
        self.spawn_side = "Right" if side == "Left" else "Left"
        lo, hi = self._limits(side)
        x = self.rng.uniform(lo + BALL_R, hi - BALL_R)
        speed = min(self.base_speed + self.speed_gain * self.score, self.max_speed)
        self.balls.append(Ball(x, -BALL_R, speed, side))
        self.next_spawn = now + self.ball_interval

    def _finish(self, now: float) -> None:
        self.over = True
        self.over_t = now

    # ---------------- per-frame ----------------
    def update(self, now: float, dt: float) -> None:
        if self.over:
            if now - self.over_t >= 3.0:
                self.reset(now)
            return

        dt = min(max(dt, 0.0), 0.10)
        alpha = 1.0 - math.exp(-dt / 0.08)
        for side in ("Left", "Right"):
            self.paddle_x[side] += alpha * (self.target[side] - self.paddle_x[side])

        if now - self.start_t >= self.duration:
            self._finish(now)
            return
        if now >= self.next_spawn:
            self._spawn(now)

        alive = []
        for b in self.balls:
            b.y += b.vy * dt
            if b.y + BALL_R < self.paddle_y:
                alive.append(b)
                continue
            if abs(b.x - self.paddle_x[b.side]) <= PADDLE_W / 2.0 + BALL_R:
                self.score += 1
                continue
            if b.y - BALL_R > self.h:
                self.lives -= 1
                if self.lives <= 0:
                    self._finish(now)
                    return
                continue
            alive.append(b)
        self.balls = alive

    # ---------------- drawing ----------------
    def draw_world(self, frame) -> None:
        frame[:] = (frame * 0.55).astype(frame.dtype)
        mid = self.w // 2
        for y in range(0, self.h, 24):
            cv2.line(frame, (mid, y), (mid, y + 12), (90, 90, 90), 1)

        colors = {"Left": (0, 200, 255), "Right": (255, 160, 0)}
        for side, x in self.paddle_x.items():
            x0 = int(x - PADDLE_W / 2)
            x1 = int(x + PADDLE_W / 2)
            y0 = int(self.paddle_y)
            cv2.rectangle(frame, (x0, y0), (x1, y0 + PADDLE_H),
                          colors[side], -1)
            cv2.putText(frame, side[0], (int(x) - 6, y0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[side], 1)

        for b in self.balls:
            p = (int(b.x), int(b.y))
            cv2.circle(frame, p, BALL_R + 2, (0, 140, 255), 1)
            cv2.circle(frame, p, BALL_R, (0, 255, 255), -1)

    def draw_hud(self, frame, now: float) -> None:
        left = max(0.0, self.duration - (now - self.start_t))
        text = f"SCORE {self.score}   LIVES {self.lives}   TIME {left:4.1f}s"
        cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4)
        cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        if self.over:
            msg = f"GAME OVER   SCORE {self.score}"
            size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            org = ((self.w - size[0]) // 2, self.h // 2)
            cv2.putText(frame, msg, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0), 6)
            cv2.putText(frame, msg, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 255), 3)

    def draw(self, frame, now: float) -> None:
        self.draw_world(frame)
        self.draw_hud(frame, now)

    def status(self) -> str:
        state = "OVER" if self.over else "PLAY"
        return f"{state} score {self.score} lives {self.lives}"
