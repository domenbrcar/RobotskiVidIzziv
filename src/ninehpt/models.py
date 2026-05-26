# Podatkovne strukture, uporabljene skozi celotno 9HPT obdelavo.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VideoMetadata:
    video_id: str
    patient_id: str
    camera_id: str
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float


@dataclass
class GridCandidate:
    centers: np.ndarray
    confidence: float
    spacing_px: float
    center: np.ndarray
    source_frame: int = -1


@dataclass
class CalibrationResult:
    target_centers: np.ndarray | None
    target_rois: list[tuple[int, int, int, int]]
    target_grid_id: str
    target_reason: str
    target_lock_time_s: float
    target_confidence: float
    baseline_frame_index: int
    grid_spacing_px: float
    meters_per_pixel: float
    calibration_debug: dict[str, Any] = field(default_factory=dict)

    @property
    def target_grid_locked(self) -> bool:
        return self.target_centers is not None and len(self.target_rois) == 9


@dataclass
class DetectedHand:
    landmarks_px: np.ndarray
    confidence: float
    handedness: str
    score: float = 0.0

    @property
    def wrist(self) -> np.ndarray:
        return self.landmarks_px[0]

    @property
    def thumb_tip(self) -> np.ndarray:
        return self.landmarks_px[4]

    @property
    def index_tip(self) -> np.ndarray:
        return self.landmarks_px[8]

    @property
    def palm_center(self) -> np.ndarray:
        return np.nanmean(self.landmarks_px[[0, 5, 9, 13, 17]], axis=0)

    @property
    def pinch_center(self) -> np.ndarray:
        return (self.thumb_tip + self.index_tip) / 2.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xy_min = np.nanmin(self.landmarks_px, axis=0)
        xy_max = np.nanmax(self.landmarks_px, axis=0)
        return (float(xy_min[0]), float(xy_min[1]), float(xy_max[0]), float(xy_max[1]))


@dataclass
class HandSelectionState:
    selected_hand: DetectedHand | None = None
    last_frame_index: int = -10_000
    candidate_hand: DetectedHand | None = None
    candidate_frames: int = 0
    last_good_hand: DetectedHand | None = None
    last_good_frame: int = -10_000
    locked_away_start_frame: int = -10_000
    confidence: float = 0.0
    selected_hand_source: str = ""
    rejected_switch_reason: str = ""


@dataclass
class RoiBaseline:
    gray: np.ndarray
    hsv: np.ndarray
    edges: np.ndarray
    histogram: np.ndarray
    center_patch: np.ndarray
    brightness: float
    center_brightness: float
    noise_level: float
    change_mean: float
    change_std: float
    occupied_threshold: float
    empty_threshold: float


@dataclass
class RoiRuntimeState:
    confirmed_state: str
    display_state: str
    change_score_ema: float = 0.0
    post_insert_score_ema: float = 0.0
    change_from_baseline: float = 0.0
    change_from_pre_entry: float = 0.0
    center_disappearance: float = 0.0
    center_brightness: float = 0.0
    dark_score: float = 0.0
    blue_score: float = 0.0
    occupied_frames: int = 0
    empty_frames: int = 0
    occluded_frames: int = 0
    last_state_change_frame: int = -10_000
    pre_visit_snapshot: np.ndarray | None = None
    pre_entry_change_score: float = 0.0
    confidence: float = 0.0
    hand_inside: bool = False
    hand_recently_left: bool = False
    last_stop_score: float = 0.0
    rejected_reason: str = ""


@dataclass
class PinEvent:
    event_number: int
    event_type: str
    roi_index: int
    frame: int
    time_s: float
    previous_state: str
    new_state: str
    change_score: float
    confidence: float
    hand_recently_left: bool
    source: str
