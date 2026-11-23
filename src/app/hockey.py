"""Two-hand air hockey rendered on top of the camera frame.

Each hand defends one side: the left hand moves the left paddle, the right hand
moves the right paddle. The puck bounces off the top/bottom walls and the
paddles; first to `target` goals wins. Paddle motion adds momentum to the puck.
"""

from collections import deque
import math
import random

import cv2

PUCK_R = 11
PADDLE_R = 32
GOAL_H = 180
HIT_PAD = 10


class AirHockey:
    def __init__(self, width: int, height: int,
                 target: int = 3,
                 duration: float = 90.0,
                 base_speed: float = 240.0,
                 max_speed: float = 520.0,
                 seed: int = 0):
        self.w = int(width)
        self.h = int(height)
        self.target = target
        self.duration = duration
        self.base_speed = base_speed
        self.max_speed = max_speed
        self.rng = random.Random(seed)
        self.trail = deque(maxlen=12)
        self.hint = "paddle follows palm | fist 0.4s exits | g mouse | q quit"
        self.reset(0.0)

    # ---------------- state ----------------
    def reset(self, now: float) -> None:
        self.score = {"Left": 0, "Right": 0}
        self.paddle = {"Left": [self.w * 0.25, self.h * 0.5],
                       "Right": [self.w * 0.75, self.h * 0.5]}
        self.paddle_v = {"Left": [0.0, 0.0], "Right": [0.0, 0.0]}
        self.target_pos = {"Left": list(self.paddle["Left"]),
                           "Right": list(self.paddle["Right"])}
        self.puck = [self.w * 0.5, self.h * 0.5]
        self.puck_v = [0.0, 0.0]
        self.trail.clear()
        self.serve_dir = 1 if self.rng.random() < 0.5 else -1
        self.serve_at = now + 1.5
        self.start_t = now
        self.last_touch = now
        self.over = False
        self.over_t = 0.0
        self.winner = ""

    def set_inputs(self, inputs, now: float) -> None:
        """inputs: {side: {"u": 0..1, "v": 0..1, ...}} for each half."""
        for side, h in inputs.items():
            if side not in self.target_pos or "u" not in h:
                continue
            u = min(max(float(h["u"]), 0.0), 1.0)
            v = min(max(float(h["v"]), 0.0), 1.0)
            half = self.w / 2.0
            x0 = PADDLE_R if side == "Left" else half + PADDLE_R
            x1 = half - PADDLE_R if side == "Left" else self.w - PADDLE_R
            y0, y1 = PADDLE_R, self.h - PADDLE_R
            self.target_pos[side][0] = x0 + u * (x1 - x0)
            self.target_pos[side][1] = y0 + v * (y1 - y0)

    def _finish(self, now: float) -> None:
        self.over = True
        self.over_t = now
        if self.score["Left"] > self.score["Right"]:
            self.winner = "LEFT"
        elif self.score["Right"] > self.score["Left"]:
            self.winner = "RIGHT"
        else:
            self.winner = "DRAW"

    def _goal(self, scorer: str, now: float) -> None:
        self.score[scorer] += 1
        self.last_touch = now
        if self.score[scorer] >= self.target:
            self._finish(now)
            return
        self.puck = [self.w * 0.5, self.h * 0.5]
        self.puck_v = [0.0, 0.0]
        self.trail.clear()
        self.serve_dir = -1 if scorer == "Right" else 1
        self.serve_at = now + 1.0

    # ---------------- per-frame ----------------
    def update(self, now: float, dt: float) -> None:
        if self.over:
            if now - self.over_t >= 4.0:
                self.reset(now)
            return

        dt = min(max(dt, 0.0), 0.10)
        if now - self.start_t >= self.duration:
            self._finish(now)
            return

        alpha = 1.0 - math.exp(-dt / 0.045)
        for side in ("Left", "Right"):
            px, py = self.paddle[side]
            tx, ty = self.target_pos[side]
            nx = px + alpha * (tx - px)
            ny = py + alpha * (ty - py)
            if dt > 0:
                k = 1.0 - math.exp(-dt / 0.08)
                self.paddle_v[side][0] += k * ((nx - px) / dt - self.paddle_v[side][0])
                self.paddle_v[side][1] += k * ((ny - py) / dt - self.paddle_v[side][1])
            self.paddle[side] = [nx, ny]

        if now < self.serve_at:
            return
        if self.puck_v[0] == 0.0 and self.puck_v[1] == 0.0:
            self.puck_v = [self.serve_dir * self.base_speed,
                           self.rng.uniform(-0.30, 0.30) * self.base_speed]
            self.last_touch = now
        elif now - self.last_touch > 10.0:
            # safety: puck stuck behind a paddle or in a corner
            self.puck = [self.w * 0.5, self.h * 0.5]
            self.puck_v = [0.0, 0.0]
            self.trail.clear()
            self.serve_at = now + 1.0
            self.last_touch = now
            return

        self.puck[0] += self.puck_v[0] * dt
        self.puck[1] += self.puck_v[1] * dt
        self.trail.append((self.puck[0], self.puck[1]))

        if self.puck[1] - PUCK_R < 0:
            self.puck[1] = PUCK_R
            self.puck_v[1] = abs(self.puck_v[1])
        elif self.puck[1] + PUCK_R > self.h:
            self.puck[1] = self.h - PUCK_R
            self.puck_v[1] = -abs(self.puck_v[1])

        best = None
        best_pen = 0.0
        for side in ("Left", "Right"):
            px, py = self.paddle[side]
            dist = math.hypot(self.puck[0] - px, self.puck[1] - py)
            pen = PUCK_R + PADDLE_R + HIT_PAD - dist
            if pen > best_pen:
                best_pen = pen
                best = side
        if best is not None:
            self._hit_paddle(best, now)

        if self.puck[0] - PUCK_R < 0:
            self._goal("Right", now)
        elif self.puck[0] + PUCK_R > self.w:
            self._goal("Left", now)

    def _hit_paddle(self, side: str, now: float) -> None:
        px, py = self.paddle[side]
        dx = self.puck[0] - px
        dy = self.puck[1] - py
        dist = math.hypot(dx, dy)
        r_sum = PUCK_R + PADDLE_R
        if dist >= r_sum + HIT_PAD:
            return
        if dist < 1e-6:
            nx, ny = (1.0 if side == "Left" else -1.0), 0.0
        else:
            nx, ny = dx / dist, dy / dist

        self.puck[0] = px + nx * (r_sum + 0.5)
        self.puck[1] = py + ny * (r_sum + 0.5)

        vn = self.puck_v[0] * nx + self.puck_v[1] * ny
        if vn < 0:
            self.puck_v[0] -= 2 * vn * nx
            self.puck_v[1] -= 2 * vn * ny

        pvx, pvy = self.paddle_v[side]
        pvn = pvx * nx + pvy * ny
        self.puck_v[0] += 0.45 * pvn * nx
        self.puck_v[1] += 0.45 * pvn * ny

        speed = math.hypot(self.puck_v[0], self.puck_v[1])
        if speed < 1e-6:
            self.puck_v = [nx * self.base_speed, ny * self.base_speed]
            speed = self.base_speed
        speed = min(speed * 1.02, self.max_speed)
        speed = max(speed, self.base_speed * 0.85)
        scale = speed / max(math.hypot(self.puck_v[0], self.puck_v[1]), 1e-6)
        self.puck_v[0] *= scale
        self.puck_v[1] *= scale
        self.last_touch = now

    # ---------------- drawing ----------------
    def draw_world(self, frame) -> None:
        frame[:] = (frame * 0.50).astype(frame.dtype)
        colors = {"Left": (0, 200, 255), "Right": (255, 160, 0)}

        cv2.rectangle(frame, (2, 2), (self.w - 3, self.h - 3), (70, 70, 70), 2)
        mid = self.w // 2
        for y in range(0, self.h, 26):
            cv2.line(frame, (mid, y), (mid, y + 13), (70, 70, 70), 1)
        cv2.circle(frame, (mid, self.h // 2), 60, (70, 70, 70), 1)

        gy0 = self.h // 2 - GOAL_H // 2
        gy1 = self.h // 2 + GOAL_H // 2
        cv2.line(frame, (3, gy0), (3, gy1), colors["Left"], 6)
        cv2.line(frame, (self.w - 4, gy0), (self.w - 4, gy1), colors["Right"], 6)

        for i, (tx, ty) in enumerate(self.trail):
            a = (i + 1) / len(self.trail)
            cv2.circle(frame, (int(tx), int(ty)), max(2, int(PUCK_R * a)),
                       (0, 140, 200), -1)

        for side in ("Left", "Right"):
            px, py = self.paddle[side]
            cv2.circle(frame, (int(px), int(py)), PADDLE_R, colors[side], -1)
            cv2.circle(frame, (int(px), int(py)), PADDLE_R, (255, 255, 255), 2)

        cv2.circle(frame, (int(self.puck[0]), int(self.puck[1])), PUCK_R + 2,
                   (0, 0, 0), -1)
        cv2.circle(frame, (int(self.puck[0]), int(self.puck[1])), PUCK_R,
                   (0, 255, 255), -1)

    def draw_hud(self, frame, now: float) -> None:
        text = f"L {self.score['Left']} : {self.score['Right']} R"
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        org = ((self.w - size[0]) // 2, 36)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 5)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)

        left = max(0.0, self.duration - (now - self.start_t))
        hint = f"first to {self.target}  |  {left:4.1f}s"
        cv2.putText(frame, hint, (12, self.h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)

        if self.over:
            msg = f"{self.winner} WINS  {self.score['Left']} : {self.score['Right']}"
            size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 3)[0]
            org = ((self.w - size[0]) // 2, self.h // 2)
            cv2.putText(frame, msg, org, cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                        (0, 0, 0), 6)
            cv2.putText(frame, msg, org, cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                        (0, 255, 255), 3)
        elif now < self.serve_at:
            msg = "READY"
            size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.putText(frame, msg, ((self.w - size[0]) // 2, self.h // 2 + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    def draw(self, frame, now: float) -> None:
        self.draw_world(frame)
        self.draw_hud(frame, now)

    def status(self) -> str:
        state = "OVER" if self.over else "PLAY"
        return f"{state} L{self.score['Left']}:{self.score['Right']}R"
