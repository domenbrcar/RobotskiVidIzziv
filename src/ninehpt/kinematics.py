# Kinematične meritve za izbrano roko.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import DetectedHand


@dataclass
class _PointState:
    position_m: np.ndarray | None = None
    velocity_mps: float = 0.0
    acceleration_mps2: float = 0.0
    path_m: float = 0.0
    last_time_s: float | None = None
    last_speed_mps: float = 0.0


class KinematicsTracker:
    """Tracks path length, speed and acceleration for palm, thumb and index."""

    def __init__(self, meters_per_pixel: float, config: dict):
        self.scale = float(meters_per_pixel)
        self.alpha = float(config["kinematic_smoothing_alpha"])
        self.max_gap = float(config["kinematic_max_gap_seconds"])
        self.max_speed = float(config["kinematic_max_speed_mps"])
        self.max_acc = float(config["kinematic_max_acc_mps2"])
        self.points = {
            "hand": _PointState(),
            "thumb": _PointState(),
            "index": _PointState(),
        }

    def _update_point(self, name: str, point_px: np.ndarray | None, time_s: float) -> dict[str, float]:
        state = self.points[name]
        if point_px is None or self.scale <= 0:
            return self._row(state, valid=False)
        position = np.asarray(point_px, dtype=np.float32) * self.scale
        if state.position_m is None or state.last_time_s is None:
            state.position_m = position
            state.last_time_s = time_s
            return self._row(state, valid=True)
        dt = max(0.0, time_s - state.last_time_s)
        if dt <= 1e-6 or dt > self.max_gap:
            state.position_m = position
            state.last_time_s = time_s
            state.last_speed_mps = 0.0
            state.velocity_mps = 0.0
            state.acceleration_mps2 = 0.0
            return self._row(state, valid=True)
        step = float(np.linalg.norm(position - state.position_m))
        raw_speed = step / dt
        if raw_speed > self.max_speed:
            state.position_m = position
            state.last_time_s = time_s
            return self._row(state, valid=False)
        speed = self.alpha * raw_speed + (1.0 - self.alpha) * state.velocity_mps
        raw_acc = (speed - state.last_speed_mps) / dt
        raw_acc = float(np.clip(raw_acc, -self.max_acc, self.max_acc))
        acc = self.alpha * raw_acc + (1.0 - self.alpha) * state.acceleration_mps2
        state.path_m += step
        state.position_m = position
        state.last_time_s = time_s
        state.last_speed_mps = speed
        state.velocity_mps = speed
        state.acceleration_mps2 = acc
        return self._row(state, valid=True)

    @staticmethod
    def _row(state: _PointState, valid: bool) -> dict[str, float]:
        x_m = float(state.position_m[0]) if state.position_m is not None else 0.0
        y_m = float(state.position_m[1]) if state.position_m is not None else 0.0
        return {
            "x_m": x_m,
            "y_m": y_m,
            "path_m": state.path_m,
            "velocity_mps": state.velocity_mps if valid else 0.0,
            "acceleration_mps2": state.acceleration_mps2 if valid else 0.0,
            "valid": 1.0 if valid else 0.0,
        }

    def update(self, hand: DetectedHand | None, time_s: float) -> dict[str, dict[str, float]]:
        if hand is None:
            return {
                "hand": self._update_point("hand", None, time_s),
                "thumb": self._update_point("thumb", None, time_s),
                "index": self._update_point("index", None, time_s),
            }
        return {
            "hand": self._update_point("hand", hand.palm_center, time_s),
            "thumb": self._update_point("thumb", hand.thumb_tip, time_s),
            "index": self._update_point("index", hand.index_tip, time_s),
        }
