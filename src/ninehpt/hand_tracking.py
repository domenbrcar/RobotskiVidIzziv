"""Stable patient-hand tracking on top of MediaPipe detections."""

from __future__ import annotations

import math
import os

import cv2
import numpy as np

from .models import DetectedHand, HandSelectionState

try:  # MediaPipe is optional for unit tests and documentation builds.
    import mediapipe as mp
except Exception:  # pragma: no cover - depends on the runtime image.
    mp = None


try:
    _MP_SOLUTIONS_HANDS = mp.solutions.hands if mp is not None and hasattr(mp, "solutions") else None
except Exception:  # pragma: no cover - runtime dependent.
    _MP_SOLUTIONS_HANDS = None

if _MP_SOLUTIONS_HANDS is not None:
    HAND_CONNECTIONS = tuple((int(a), int(b)) for a, b in _MP_SOLUTIONS_HANDS.HAND_CONNECTIONS)
else:
    HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
    )


class MediaPipeHandDetector:
    """Thin MediaPipe wrapper returning hand landmarks in pixel coordinates."""

    def __init__(self, config: dict):
        self.available = mp is not None
        self._hands = None
        self._landmarker = None
        self._mode = "none"
        if not self.available:
            return
        model_path = config.get("hand_landmarker_model_path", "")
        if _MP_SOLUTIONS_HANDS is not None:
            self._hands = _MP_SOLUTIONS_HANDS.Hands(
                static_image_mode=False,
                max_num_hands=int(config["hand_max_num_hands"]),
                model_complexity=int(config["hand_model_complexity"]),
                min_detection_confidence=float(config["hand_min_detection_confidence"]),
                min_tracking_confidence=float(config["hand_min_tracking_confidence"]),
            )
            self._mode = "solutions"
            return
        if model_path and os.path.exists(model_path) and hasattr(mp, "tasks"):
            try:
                base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
                options = mp.tasks.vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_hands=int(config["hand_max_num_hands"]),
                    min_hand_detection_confidence=float(config["hand_min_detection_confidence"]),
                    min_hand_presence_confidence=float(config["hand_min_detection_confidence"]),
                    min_tracking_confidence=float(config["hand_min_tracking_confidence"]),
                )
                self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
                self._mode = "tasks"
                return
            except Exception:
                self._landmarker = None
        self.available = False

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
        if self._landmarker is not None:
            self._landmarker.close()

    def detect(self, frame_bgr: np.ndarray, frame_index: int = 0, fps: float = 25.0) -> list[DetectedHand]:
        if self._hands is None and self._landmarker is None:
            return []
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._mode == "solutions":
            result = self._hands.process(rgb)
            landmark_list = result.multi_hand_landmarks or []
            handedness_list = result.multi_handedness or []
        else:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(round(frame_index / max(fps, 1e-6) * 1000.0))
            result = self._landmarker.detect_for_video(image, timestamp_ms)
            landmark_list = result.hand_landmarks or []
            handedness_list = result.handedness or []
        if not landmark_list:
            return []
        detections: list[DetectedHand] = []
        for index, landmarks in enumerate(landmark_list):
            items = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
            points = np.array([[lm.x * width, lm.y * height] for lm in items], dtype=np.float32)
            label = "unknown"
            score = 0.5
            if index < len(handedness_list):
                handedness = handedness_list[index]
                if hasattr(handedness, "classification") and handedness.classification:
                    cls = handedness.classification[0]
                    label = cls.label
                    score = float(cls.score)
                elif handedness:
                    cls = handedness[0]
                    label = getattr(cls, "category_name", "unknown")
                    score = float(getattr(cls, "score", 0.5))
            detections.append(DetectedHand(points, score, label))
        return detections


def _grid_center_and_scale(target_centers: np.ndarray | None, frame_shape: tuple[int, ...] | None = None) -> tuple[np.ndarray, float]:
    if target_centers is None or len(target_centers) == 0:
        if frame_shape is None:
            return np.array([0.0, 0.0], dtype=np.float32), 100.0
        height, width = frame_shape[:2]
        return np.array([width / 2.0, height / 2.0], dtype=np.float32), float(math.hypot(width, height) * 0.15)
    centers = np.asarray(target_centers, dtype=np.float32).reshape(-1, 2)
    spacing_candidates = []
    grid = centers.reshape(3, 3, 2) if len(centers) == 9 else None
    if grid is not None:
        for row in range(3):
            for col in range(2):
                spacing_candidates.append(float(np.linalg.norm(grid[row, col + 1] - grid[row, col])))
        for row in range(2):
            for col in range(3):
                spacing_candidates.append(float(np.linalg.norm(grid[row + 1, col] - grid[row, col])))
    spacing = float(np.median(spacing_candidates)) if spacing_candidates else 100.0
    return np.mean(centers, axis=0), max(spacing, 25.0)


def _orientation_score(hand: DetectedHand, target_center: np.ndarray, spacing: float) -> float:
    wrist_to_pinch = hand.pinch_center - hand.wrist
    wrist_to_target = target_center - hand.wrist
    denom = float(np.linalg.norm(wrist_to_pinch) * np.linalg.norm(wrist_to_target))
    if denom <= 1e-6:
        direction = 0.0
    else:
        direction = float(np.dot(wrist_to_pinch, wrist_to_target) / denom)
    pinch_distance = float(np.linalg.norm(hand.pinch_center - target_center))
    thumb_index_distance = float(np.linalg.norm(hand.thumb_tip - hand.index_tip))
    directed = (direction + 1.0) / 2.0
    close = math.exp(-pinch_distance / max(4.0 * spacing, 1.0))
    pinch_shape = math.exp(-thumb_index_distance / max(2.3 * spacing, 1.0))
    return float(np.clip(0.45 * directed + 0.40 * close + 0.15 * pinch_shape, 0.0, 1.0))


def _hand_score(hand: DetectedHand, target_centers: np.ndarray | None, previous: DetectedHand | None, frame_shape: tuple[int, ...] | None, config: dict) -> float:
    target_center, spacing = _grid_center_and_scale(target_centers, frame_shape)
    palm_distance = float(np.linalg.norm(hand.palm_center - target_center))
    pinch_distance = float(np.linalg.norm(hand.pinch_center - target_center))
    distance_score = math.exp(-min(palm_distance, pinch_distance) / max(4.5 * spacing, 1.0))
    orientation = _orientation_score(hand, target_center, spacing)
    continuity = 0.0
    if previous is not None:
        jump = float(np.linalg.norm(hand.palm_center - previous.palm_center))
        continuity = math.exp(-jump / max(2.5 * spacing, 1.0))
    score = 0.34 * float(hand.confidence) + 0.34 * distance_score + 0.20 * orientation + 0.12 * continuity
    return float(np.clip(score, 0.0, 1.0))


def _hold_previous(state: HandSelectionState, frame_index: int, fps: float, config: dict, reason: str) -> tuple[DetectedHand | None, HandSelectionState]:
    hold_frames = int(round(config["hand_hold_seconds"] * max(fps, 1.0)))
    if state.last_good_hand is not None and frame_index - state.last_good_frame <= hold_frames:
        state.selected_hand = state.last_good_hand
        state.last_frame_index = frame_index
        state.confidence = max(0.0, state.confidence * 0.96)
        state.selected_hand_source = "held_last_good"
        state.rejected_switch_reason = reason
        return state.selected_hand, state
    state.selected_hand = None
    state.selected_hand_source = "missing"
    state.rejected_switch_reason = reason
    state.confidence = 0.0
    return None, state


def select_patient_hand(
    detections: list[DetectedHand],
    state: HandSelectionState,
    frame_index: int,
    fps: float,
    target_centers: np.ndarray | None,
    frame_shape: tuple[int, ...] | None,
    config: dict,
) -> tuple[DetectedHand | None, HandSelectionState]:
    """Select one patient hand with hysteresis and short dropout retention."""
    if not detections:
        return _hold_previous(state, frame_index, fps, config, "ni_zaznane_roke")

    for hand in detections:
        hand.score = _hand_score(hand, target_centers, state.last_good_hand, frame_shape, config)
    detections = sorted(detections, key=lambda item: item.score, reverse=True)
    best = detections[0]

    target_center, spacing = _grid_center_and_scale(target_centers, frame_shape)
    max_jump = max(45.0, config["hand_max_jump_ratio"] * (math.hypot(frame_shape[1], frame_shape[0]) if frame_shape else 1000.0))
    switch_margin = float(config["hand_switch_margin"])
    confirm_frames = int(config["hand_switch_confirm_frames"])

    previous = state.last_good_hand
    if previous is None:
        state.selected_hand = best
        state.last_good_hand = best
        state.last_good_frame = frame_index
        state.last_frame_index = frame_index
        state.confidence = best.score
        state.selected_hand_source = "initial_detection"
        state.rejected_switch_reason = ""
        return best, state

    distances_to_previous = [float(np.linalg.norm(hand.palm_center - previous.palm_center)) for hand in detections]
    nearest_index = int(np.argmin(distances_to_previous))
    nearest = detections[nearest_index]
    nearest_distance = distances_to_previous[nearest_index]
    nearest_score = nearest.score
    previous_score_estimate = _hand_score(previous, target_centers, previous, frame_shape, config)

    previous_visible = nearest_distance <= max_jump * 1.15
    best_is_previous = nearest is best or float(np.linalg.norm(best.palm_center - previous.palm_center)) <= max_jump
    hands_overlap = len(detections) > 1 and abs(detections[0].score - detections[1].score) < 0.08

    if previous_visible and (best_is_previous or hands_overlap or best.score < nearest_score + switch_margin):
        state.selected_hand = nearest
        state.last_good_hand = nearest
        state.last_good_frame = frame_index
        state.last_frame_index = frame_index
        state.confidence = nearest.score
        state.candidate_hand = None
        state.candidate_frames = 0
        state.selected_hand_source = "continuous_detection"
        state.rejected_switch_reason = "ohranjena_kontinuiteta"
        return nearest, state

    best_jump = float(np.linalg.norm(best.palm_center - previous.palm_center))
    best_near_target = float(np.linalg.norm(best.pinch_center - target_center)) <= 5.0 * spacing
    switch_candidate = best.score >= previous_score_estimate + switch_margin and best_jump <= max_jump * 2.2 and best_near_target
    if switch_candidate:
        if state.candidate_hand is not None and np.linalg.norm(best.palm_center - state.candidate_hand.palm_center) <= max_jump:
            state.candidate_frames += 1
        else:
            state.candidate_hand = best
            state.candidate_frames = 1
        if state.candidate_frames >= confirm_frames:
            state.selected_hand = best
            state.last_good_hand = best
            state.last_good_frame = frame_index
            state.last_frame_index = frame_index
            state.confidence = best.score
            state.selected_hand_source = "confirmed_switch"
            state.rejected_switch_reason = ""
            state.candidate_hand = None
            state.candidate_frames = 0
            return best, state
        return _hold_previous(state, frame_index, fps, config, "preklop_se_ni_potrjen")

    if frame_index - state.last_good_frame > int(round(config["hand_hold_seconds"] * max(fps, 1.0))):
        state.selected_hand = best
        state.last_good_hand = best
        state.last_good_frame = frame_index
        state.last_frame_index = frame_index
        state.confidence = best.score
        state.selected_hand_source = "reselected_after_timeout"
        state.rejected_switch_reason = "prejsnja_roka_predolgo_izgubljena"
        return best, state

    return _hold_previous(state, frame_index, fps, config, "nova_roka_ni_dovolj_boljsa")


def draw_selected_hand(frame: np.ndarray, hand: DetectedHand | None) -> np.ndarray:
    """Draw only the selected stable skeleton."""
    if hand is None:
        return frame
    out = frame.copy()
    points = np.asarray(hand.landmarks_px, dtype=np.float32)
    for start, end in HAND_CONNECTIONS:
        p1 = tuple(np.round(points[start]).astype(int))
        p2 = tuple(np.round(points[end]).astype(int))
        cv2.line(out, p1, p2, (58, 210, 132), 2, cv2.LINE_AA)
    for idx in (0, 4, 8, 12, 16, 20):
        p = tuple(np.round(points[idx]).astype(int))
        cv2.circle(out, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, p, 4, (58, 210, 132), 1, cv2.LINE_AA)
    return out
