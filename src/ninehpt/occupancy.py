# Vizualni avtomat stanj zasedenosti zatičev za zaklenjeno 9HPT mrežo.

from __future__ import annotations

import cv2
import numpy as np

from .config import EMPTY, OCCUPIED
from .grid_detection import compute_grid_spacing_px
from .models import DetectedHand, PinEvent, RoiBaseline, RoiRuntimeState


def _extract_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi
    return frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]


def _resize_bgr(patch: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if patch.size == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(patch, size, interpolation=cv2.INTER_AREA)


def _resize_to_baseline(patch: np.ndarray, baseline: RoiBaseline) -> np.ndarray:
    height, width = baseline.gray.shape[:2]
    return _resize_bgr(patch, (width, height))


def _center_patch(patch: np.ndarray, fraction: float = 0.48) -> np.ndarray:
    if patch.size == 0:
        return patch
    height, width = patch.shape[:2]
    cx1 = int(round(width * (1.0 - fraction) / 2.0))
    cx2 = int(round(width * (1.0 + fraction) / 2.0))
    cy1 = int(round(height * (1.0 - fraction) / 2.0))
    cy2 = int(round(height * (1.0 + fraction) / 2.0))
    return patch[max(0, cy1):min(height, cy2), max(0, cx1):min(width, cx2)]


def _center_brightness_bgr(patch: np.ndarray) -> float:
    center = _center_patch(patch)
    if center.size == 0:
        return 0.0
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def _skin_fraction(patch: np.ndarray) -> float:
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    skin = (((h <= 25) | (h >= 165)) & (s >= 28) & (s <= 180) & (v >= 45))
    return float(np.mean(skin))


def _roi_histogram(hsv: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    hist = cv2.normalize(hist, None).flatten()
    return hist.astype(np.float32)


def _gray_difference(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(gray_a, gray_b)) / 255.0)


def _edge_image(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray, 40, 100)


def _build_baseline_from_patch(patch: np.ndarray, changes: list[float], config: dict) -> RoiBaseline:
    patch = _resize_bgr(patch, (32, 32))
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    edges = _edge_image(gray)
    noise_mean = float(np.mean(changes)) if changes else 0.018
    noise_std = float(np.std(changes)) if len(changes) > 1 else 0.012
    occupied_threshold = max(
        config["min_occupied_threshold"],
        noise_mean + config["occupied_threshold_std_factor"] * noise_std,
    )
    empty_threshold = max(
        config["min_empty_threshold"],
        noise_mean + config["empty_threshold_std_factor"] * noise_std,
    )
    return RoiBaseline(
        gray=gray,
        hsv=hsv,
        edges=edges,
        histogram=_roi_histogram(hsv),
        center_patch=cv2.cvtColor(_center_patch(patch), cv2.COLOR_BGR2GRAY),
        brightness=float(np.mean(hsv[:, :, 2])),
        center_brightness=_center_brightness_bgr(patch),
        noise_level=noise_mean + noise_std,
        change_mean=noise_mean,
        change_std=noise_std,
        occupied_threshold=float(occupied_threshold),
        empty_threshold=float(empty_threshold),
    )


def initialize_roi_baseline(frames: list[np.ndarray], rois: list[tuple[int, int, int, int]], config: dict) -> list[RoiBaseline]:
    """Create one empty-hole baseline per ROI from bright, skin-free samples."""
    baselines: list[RoiBaseline] = []
    if not frames:
        raise ValueError("Baseline initialization needs at least one frame.")
    for roi in rois:
        candidates: list[tuple[float, np.ndarray]] = []
        for frame in frames:
            patch = _extract_roi(frame, roi)
            if patch.size == 0:
                continue
            if _skin_fraction(patch) > config["baseline_skin_reject_fraction"]:
                continue
            candidates.append((_center_brightness_bgr(patch), _resize_bgr(patch, (32, 32))))
        if not candidates:
            candidates = [(_center_brightness_bgr(_extract_roi(frames[0], roi)), _resize_bgr(_extract_roi(frames[0], roi), (32, 32)))]
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [patch for _, patch in candidates[: max(1, min(len(candidates), int(config["baseline_min_frames"])))]]
        median_patch = np.median(np.stack(selected, axis=0).astype(np.float32), axis=0).astype(np.uint8)
        median_gray = cv2.cvtColor(median_patch, cv2.COLOR_BGR2GRAY)
        changes = [_gray_difference(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), median_gray) for patch in selected]
        baselines.append(_build_baseline_from_patch(median_patch, changes, config))
    return baselines


def compute_roi_change_score(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    baseline: RoiBaseline,
    pre_entry_snapshot: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    """Return visual evidence for the current ROI against its empty baseline."""
    patch_raw = _extract_roi(frame, roi)
    patch = _resize_to_baseline(patch_raw, baseline)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    edges = _edge_image(gray)
    change_from_baseline = _gray_difference(gray, baseline.gray)
    if pre_entry_snapshot is not None:
        pre = _resize_bgr(pre_entry_snapshot, (gray.shape[1], gray.shape[0]))
        change_from_pre = _gray_difference(gray, cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY))
    else:
        change_from_pre = 0.0
    hist = _roi_histogram(hsv)
    hist_change = float(cv2.compareHist(baseline.histogram, hist, cv2.HISTCMP_BHATTACHARYYA))
    edge_change = float(np.mean(cv2.absdiff(edges, baseline.edges)) / 255.0)
    center_brightness = _center_brightness_bgr(patch)
    center_disappearance = max(0.0, (baseline.center_brightness - center_brightness) / 255.0)
    v = hsv[:, :, 2]
    center_hsv = cv2.cvtColor(_center_patch(patch), cv2.COLOR_BGR2HSV)
    center_h = center_hsv[:, :, 0]
    center_s = center_hsv[:, :, 1]
    center_v = center_hsv[:, :, 2]
    dark_score = float(np.mean((center_v < max(70, baseline.center_brightness * 0.68)) & (center_v < baseline.center_brightness - 18)))
    blue_score = float(np.mean((center_h >= 88) & (center_h <= 134) & (center_s > 45) & (center_v > 25)))
    color_shift = float(np.mean(np.abs(hsv.astype(np.float32) - baseline.hsv.astype(np.float32))) / 255.0)
    color_or_dark = max(blue_score, dark_score, color_shift * 0.55)
    change_score = float(np.clip(0.58 * change_from_baseline + 0.18 * hist_change + 0.12 * edge_change + 0.12 * color_or_dark, 0.0, 1.0))
    post_insert_score = float(np.clip(change_score + 0.42 * center_disappearance + 0.20 * dark_score + 0.16 * blue_score, 0.0, 1.0))
    return {
        "patch": patch,
        "change_score": change_score,
        "post_insert_score": post_insert_score,
        "change_from_baseline": change_from_baseline,
        "change_from_pre_entry": change_from_pre,
        "center_disappearance": center_disappearance,
        "hist_change_score": hist_change,
        "edge_change_score": edge_change,
        "color_or_dark_object_score": color_or_dark,
        "blue_score": blue_score,
        "dark_object_score": dark_score,
        "brightness": float(np.mean(v)),
        "center_brightness": center_brightness,
        "skin_fraction": _skin_fraction(patch_raw),
    }


def detect_hand_visit_to_roi(hand: DetectedHand | None, roi: tuple[int, int, int, int], spacing_px: float, config: dict) -> bool:
    if hand is None:
        return False
    x1, y1, x2, y2 = roi
    expansion = max(6.0, spacing_px * config["roi_visit_expansion_factor"])
    points = [hand.pinch_center, hand.thumb_tip, hand.index_tip, hand.palm_center]
    return any((x1 - expansion <= p[0] <= x2 + expansion and y1 - expansion <= p[1] <= y2 + expansion) for p in points)


class PegOccupancyTracker:
    """Current visual occupancy tracker for the nine locked hole ROIs."""

    def __init__(
        self,
        rois: list[tuple[int, int, int, int]],
        baselines: list[RoiBaseline],
        target_centers: np.ndarray,
        fps: float,
        config: dict,
    ):
        if len(rois) != 9 or len(baselines) != 9:
            raise ValueError("Peg tracker expects exactly nine ROIs and baselines.")
        self.rois = rois
        self.baselines = baselines
        self.target_centers = np.asarray(target_centers, dtype=np.float32)
        self.spacing_px = compute_grid_spacing_px(self.target_centers)
        self.fps = max(float(fps), 1.0)
        self.config = config
        self.states = [RoiRuntimeState(confirmed_state=EMPTY, display_state=EMPTY) for _ in range(9)]
        self.events: list[PinEvent] = []
        self._event_number = 0
        self._grid_active = False
        self._grid_entry_frame = -10_000
        self._grid_exit_frame = -10_000
        self._grid_entry_snapshots: list[np.ndarray | None] = [None] * 9
        self._previous_brightness: list[float | None] = [None] * 9
        self._previous_scores: list[float] = [0.0] * 9
        self._global_light_frames = 0
        # Zadnje potrjene vstavitve s signalom, da lahko zavrnemo nerealne skoke štetja.
        self._accepted_insert_events: list[tuple[int, float]] = []
        self._grid_box = self._build_grid_box()

    def _build_grid_box(self) -> tuple[int, int, int, int]:
        centers = self.target_centers.reshape(-1, 2)
        margin = self.spacing_px * (1.0 + self.config["grid_box_margin_spacing_factor"])
        return (
            int(np.floor(np.min(centers[:, 0]) - margin)),
            int(np.floor(np.min(centers[:, 1]) - margin)),
            int(np.ceil(np.max(centers[:, 0]) + margin)),
            int(np.ceil(np.max(centers[:, 1]) + margin)),
        )

    def _hand_in_grid(self, hand: DetectedHand | None, frame: np.ndarray) -> bool:
        x1, y1, x2, y2 = self._grid_box
        points = []
        if hand is not None:
            points = [hand.pinch_center, hand.thumb_tip, hand.index_tip, hand.palm_center]
        landmark_inside = any((x1 <= p[0] <= x2 and y1 <= p[1] <= y2) for p in points)
        grid_patch = _extract_roi(frame, (max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)))
        skin_inside = _skin_fraction(grid_patch) >= self.config["grid_skin_freeze_fraction"]
        return bool(landmark_inside or skin_inside)

    def _add_event(self, roi_index: int, frame_index: int, time_s: float, previous: str, new: str, state: RoiRuntimeState, source: str) -> None:
        self._event_number += 1
        event_type = "insert" if new == OCCUPIED else "remove"
        self.events.append(PinEvent(
            event_number=self._event_number,
            event_type=event_type,
            roi_index=roi_index,
            frame=frame_index,
            time_s=time_s,
            previous_state=previous,
            new_state=new,
            change_score=state.post_insert_score_ema,
            confidence=state.confidence,
            hand_recently_left=state.hand_recently_left,
            source=source,
        ))

    def _can_accept_insert(self, frame_index: int, insert_score: float) -> tuple[bool, str]:
        min_score = float(self.config.get("confirmed_insert_min_score", 0.0))
        if insert_score < min_score:
            return False, "sibek_vizualni_signal_zatica"

        window_frames = max(1, int(round(float(self.config["peg_count_jump_window_seconds"]) * self.fps)))
        max_increase = max(1, int(self.config["max_peg_count_increase_per_window"]))
        cutoff = frame_index - window_frames
        self._accepted_insert_events = [
            (frame, score)
            for frame, score in self._accepted_insert_events
            if frame > cutoff
        ]

        pair_window_frames = max(1, int(round(float(self.config.get("fast_pair_window_seconds", 0.0)) * self.fps)))
        pair_min_score = float(self.config.get("fast_pair_min_score", min_score))
        recent_pair_events = [
            (frame, score)
            for frame, score in self._accepted_insert_events
            if frame_index - frame <= pair_window_frames
        ]
        weak_fast_pair = any(
            insert_score < pair_min_score or previous_score < pair_min_score
            for _, previous_score in recent_pair_events
        )
        if weak_fast_pair:
            return False, "prehitra_sibka_dvojna_potrditev"

        if len(self._accepted_insert_events) >= max_increase:
            return False, "prehitro_skupinsko_stetje"
        return True, ""

    def _register_insert(self, frame_index: int, insert_score: float) -> None:
        self._accepted_insert_events.append((frame_index, float(insert_score)))

    def _detail_looks_occupied(self, detail: dict[str, float | np.ndarray], baseline: RoiBaseline, state: RoiRuntimeState, recently_left_grid: bool) -> bool:
        occupied_thr = max(baseline.occupied_threshold, self.config["min_occupied_threshold"])
        baseline_changed = float(detail["change_from_baseline"]) >= max(occupied_thr * 0.52, self.config["pre_visit_change_threshold"])
        pre_changed = float(detail["change_from_pre_entry"]) >= max(self.config["pre_visit_change_threshold"], occupied_thr * 0.36) or not recently_left_grid
        object_like = float(detail["center_disappearance"]) >= 0.030 and (
            float(detail["dark_object_score"]) >= 0.12
            or float(detail["blue_score"]) >= 0.05
            or float(detail["center_disappearance"]) >= 0.18
        )
        strong_score = float(detail["post_insert_score"]) >= occupied_thr
        stable_score = (
            float(detail["post_insert_score"]) >= self.config["stable_visual_min_score"]
            and float(detail["change_from_baseline"]) >= self.config["stable_visual_min_baseline_change"]
        )
        return bool((strong_score and baseline_changed and pre_changed and object_like) or (stable_score and object_like))

    def _reset_group_reference(self, candidate_indices: list[int], details: list[dict[str, float | np.ndarray]], frame_index: int) -> None:
        for idx in candidate_indices:
            state = self.states[idx]
            patch = details[idx]["patch"]
            if isinstance(patch, np.ndarray):
                self.baselines[idx] = _build_baseline_from_patch(patch, [0.0], self.config)
            state.change_score_ema = 0.0
            state.post_insert_score_ema = 0.0
            state.change_from_baseline = 0.0
            state.change_from_pre_entry = 0.0
            state.center_disappearance = 0.0
            state.dark_score = 0.0
            state.blue_score = 0.0
            state.occupied_frames = 0
            state.empty_frames = 0
            state.display_state = state.confirmed_state
            state.pre_visit_snapshot = None
            state.pre_entry_change_score = 0.0
            state.last_state_change_frame = frame_index
            state.rejected_reason = "osvezena_referenca_skupinska_sprememba"
            self._previous_brightness[idx] = None
            self._previous_scores[idx] = 0.0

    def _global_light_event(self, details: list[dict[str, float | np.ndarray]]) -> bool:
        changed_scores = 0
        changed_brightness = 0
        for idx, detail in enumerate(details):
            score = float(detail["change_score"])
            brightness = float(detail["brightness"])
            if abs(score - self._previous_scores[idx]) >= self.config["global_light_change_score_jump"]:
                changed_scores += 1
            previous_brightness = self._previous_brightness[idx]
            if previous_brightness is not None and abs(brightness - previous_brightness) >= self.config["global_light_brightness_jump"]:
                changed_brightness += 1
            self._previous_scores[idx] = score
            self._previous_brightness[idx] = brightness
        is_global = (
            changed_scores >= self.config["global_light_change_roi_count"]
            or changed_brightness >= self.config["global_light_change_roi_count"]
        )
        self._global_light_frames = 2 if is_global else max(0, self._global_light_frames - 1)
        return self._global_light_frames > 0

    def update(
        self,
        frame: np.ndarray,
        frame_index: int,
        time_s: float,
        selected_hand: DetectedHand | None = None,
        target_locked: bool = True,
    ) -> dict:
        if not target_locked:
            return self._result(False, False, -1, "target_not_locked", [])

        details = [
            compute_roi_change_score(frame, roi, baseline, self.states[idx].pre_visit_snapshot)
            for idx, (roi, baseline) in enumerate(zip(self.rois, self.baselines))
        ]
        hand_in_grid = self._hand_in_grid(selected_hand, frame)
        if hand_in_grid and not self._grid_active:
            self._grid_entry_frame = frame_index
            self._grid_entry_snapshots = [detail["patch"].copy() for detail in details]
            for idx, snapshot in enumerate(self._grid_entry_snapshots):
                self.states[idx].pre_visit_snapshot = snapshot
                self.states[idx].pre_entry_change_score = float(details[idx]["change_score"])
        if not hand_in_grid and self._grid_active:
            self._grid_exit_frame = frame_index
        self._grid_active = hand_in_grid

        global_light = self._global_light_event(details)
        wait_after_exit = int(round(self.config["post_hand_exit_wait_seconds"] * self.fps))
        recently_left_grid = 0 <= frame_index - self._grid_exit_frame <= int(round(self.config["grid_interaction_recent_seconds"] * self.fps))
        freeze_updates = hand_in_grid or global_light or frame_index - self._grid_exit_frame < wait_after_exit
        empty_candidate_indices = [
            idx
            for idx, (state, detail, baseline) in enumerate(zip(self.states, details, self.baselines))
            if state.confirmed_state != OCCUPIED and self._detail_looks_occupied(detail, baseline, state, recently_left_grid)
        ]
        enough_empty_slots = sum(1 for state in self.states if state.confirmed_state != OCCUPIED) >= int(self.config["reference_reset_min_empty_slots"])
        if (
            not hand_in_grid
            and not recently_left_grid
            and enough_empty_slots
            and len(empty_candidate_indices) >= int(self.config["reference_reset_candidate_count"])
        ):
            self._reset_group_reference(empty_candidate_indices, details, frame_index)
            return self._result(hand_in_grid, global_light, -1, "osvezena_referenca_skupinska_sprememba", details)

        alpha = float(self.config["change_ema_alpha"])
        occupied_confirm_frames = max(1, int(round(self.config["occupied_confirm_seconds"] * self.fps)))
        visual_occupied_frames = max(1, int(round(self.config["visual_occupied_fallback_seconds"] * self.fps)))
        empty_confirm_frames = max(1, int(round(self.config["empty_confirm_seconds"] * self.fps)))
        visual_empty_frames = max(1, int(round(self.config["visual_empty_fallback_seconds"] * self.fps)))
        min_hold_frames = max(1, int(round(self.config["min_state_hold_seconds"] * self.fps)))
        active_roi = -1

        for idx, (state, detail, baseline) in enumerate(zip(self.states, details, self.baselines)):
            state.hand_inside = detect_hand_visit_to_roi(selected_hand, self.rois[idx], self.spacing_px, self.config) or hand_in_grid
            state.hand_recently_left = recently_left_grid
            state.change_score_ema = alpha * float(detail["change_score"]) + (1.0 - alpha) * state.change_score_ema
            state.post_insert_score_ema = alpha * float(detail["post_insert_score"]) + (1.0 - alpha) * state.post_insert_score_ema
            state.change_from_baseline = float(detail["change_from_baseline"])
            state.change_from_pre_entry = float(detail["change_from_pre_entry"])
            state.center_disappearance = float(detail["center_disappearance"])
            state.center_brightness = float(detail["center_brightness"])
            state.dark_score = float(detail["dark_object_score"])
            state.blue_score = float(detail["blue_score"])
            state.rejected_reason = ""
            if state.hand_inside:
                active_roi = idx
            if freeze_updates:
                state.display_state = state.confirmed_state
                state.occupied_frames = 0
                state.empty_frames = 0
                state.rejected_reason = "zamrznjeno_roka" if hand_in_grid else "zamrznjeno_svetloba"
                continue

            frames_since_change = frame_index - state.last_state_change_frame
            occupied_thr = max(baseline.occupied_threshold, self.config["min_occupied_threshold"])
            empty_thr = max(baseline.empty_threshold, self.config["min_empty_threshold"])
            baseline_changed = state.change_from_baseline >= max(occupied_thr * 0.52, self.config["pre_visit_change_threshold"])
            pre_changed = state.change_from_pre_entry >= max(self.config["pre_visit_change_threshold"], occupied_thr * 0.36) or not state.hand_recently_left
            object_like = state.center_disappearance >= 0.030 and (
                state.dark_score >= 0.12 or state.blue_score >= 0.05 or state.center_disappearance >= 0.18
            )
            stable_visual_occupied = (
                state.post_insert_score_ema >= self.config["stable_visual_min_score"]
                and state.change_from_baseline >= self.config["stable_visual_min_baseline_change"]
                and object_like
            )
            grid_visual_occupied = state.post_insert_score_ema >= occupied_thr and baseline_changed and pre_changed and object_like
            strong_visual_occupied = state.post_insert_score_ema >= occupied_thr * 1.18 and baseline_changed and object_like

            raw_close_to_baseline = (
                state.change_from_baseline <= empty_thr * self.config["removal_return_baseline_factor"]
                and state.center_disappearance <= max(0.030, self.config["center_disappearance_threshold"])
                and state.dark_score < 0.12
                and state.blue_score < 0.05
            )
            ema_close = state.post_insert_score_ema <= max(empty_thr, occupied_thr - self.config["threshold_hysteresis"])
            strong_return = state.change_from_baseline <= max(self.config["min_empty_threshold"], baseline.change_mean + 2.4 * baseline.change_std)
            empty_candidate = raw_close_to_baseline and (ema_close or strong_return)

            if state.confirmed_state != OCCUPIED:
                occupied_candidate = grid_visual_occupied or strong_visual_occupied or stable_visual_occupied
                if occupied_candidate:
                    state.occupied_frames += 1
                else:
                    state.occupied_frames = 0
                    state.rejected_reason = "ni_lokalnega_objekta"
                needed = occupied_confirm_frames if state.hand_recently_left else visual_occupied_frames
                if state.occupied_frames >= needed and frames_since_change >= min_hold_frames:
                    can_insert, rejection_reason = self._can_accept_insert(frame_index, state.post_insert_score_ema)
                    if not can_insert:
                        state.display_state = state.confirmed_state
                        state.rejected_reason = rejection_reason
                        continue
                    previous = state.confirmed_state
                    state.confirmed_state = OCCUPIED
                    state.display_state = OCCUPIED
                    state.empty_frames = 0
                    state.last_state_change_frame = frame_index
                    state.confidence = float(np.clip(state.post_insert_score_ema / max(occupied_thr, 1e-6), 0.0, 1.0))
                    self._register_insert(frame_index, state.post_insert_score_ema)
                    self._add_event(idx, frame_index, time_s, previous, OCCUPIED, state, "visual_roi_state")
                else:
                    state.display_state = state.confirmed_state
            else:
                if empty_candidate:
                    state.empty_frames += 1
                else:
                    state.empty_frames = 0
                    state.rejected_reason = "zatic_se_vizualno_ostaja"
                needed = empty_confirm_frames if state.hand_recently_left else visual_empty_frames
                if state.empty_frames >= needed and frames_since_change >= min_hold_frames:
                    previous = state.confirmed_state
                    state.confirmed_state = EMPTY
                    state.display_state = EMPTY
                    state.occupied_frames = 0
                    state.last_state_change_frame = frame_index
                    state.confidence = float(np.clip(1.0 - state.change_from_baseline / max(empty_thr * 2.0, 1e-6), 0.0, 1.0))
                    self._add_event(idx, frame_index, time_s, previous, EMPTY, state, "visual_roi_state")
                else:
                    state.display_state = state.confirmed_state

        return self._result(hand_in_grid, global_light, active_roi, "", details)

    def _result(self, hand_in_grid: bool, global_light: bool, active_roi: int, rejected_reason: str, details: list[dict]) -> dict:
        states = [state.confirmed_state for state in self.states]
        display_states = [state.display_state for state in self.states]
        diagnostics = []
        for idx, state in enumerate(self.states):
            diagnostics.append({
                "roi": idx,
                "state": state.confirmed_state,
                "display_state": state.display_state,
                "change_score_ema": state.change_score_ema,
                "post_insert_score_ema": state.post_insert_score_ema,
                "change_from_baseline": state.change_from_baseline,
                "change_from_pre_entry": state.change_from_pre_entry,
                "center_disappearance": state.center_disappearance,
                "center_brightness": state.center_brightness,
                "dark_score": state.dark_score,
                "blue_score": state.blue_score,
                "occupied_frames": state.occupied_frames,
                "empty_frames": state.empty_frames,
                "hand_inside": int(state.hand_inside),
                "hand_recently_left": int(state.hand_recently_left),
                "confidence": state.confidence,
                "rejected_reason": state.rejected_reason,
            })
        return {
            "peg_count": int(sum(1 for state in states if state == OCCUPIED)),
            "roi_states": states,
            "roi_display_states": display_states,
            "roi_confidences": [state.confidence for state in self.states],
            "global_light_event": int(global_light),
            "hand_in_grid": int(hand_in_grid),
            "active_roi": active_roi,
            "rejected_reason": rejected_reason,
            "roi_diagnostics": diagnostics,
        }
