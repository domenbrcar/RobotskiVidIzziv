"""Target grid calibration and light-sequence reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import cv2
import numpy as np

from .config import KNOWN_HOLE_SPACING_MM
from .grid_detection import compute_grid_spacing_px, detect_candidate_3x3_grids, make_rois_from_centers
from .models import CalibrationResult, GridCandidate


@dataclass
class _GridTrack:
    track_id: str
    centers: np.ndarray
    spacing_px: float
    center: np.ndarray
    first_frame: int
    last_frame: int
    detections: int = 1
    confidence_sum: float = 0.0
    light_scores: list[tuple[int, float, float]] = field(default_factory=list)
    local_change_score: float = 0.0
    max_occupied_like_count: int = 0

    @property
    def confidence(self) -> float:
        return self.confidence_sum / max(1, self.detections)


def _frame_diagonal(frame_shape: tuple[int, ...]) -> float:
    height, width = frame_shape[:2]
    return math.hypot(width, height)


def _merge_candidate(tracks: list[_GridTrack], candidate: GridCandidate, frame_index: int, config: dict, frame_shape: tuple[int, ...]) -> None:
    diagonal = _frame_diagonal(frame_shape)
    merge_distance = diagonal * config["track_merge_distance_ratio"]
    best_track: _GridTrack | None = None
    best_distance = float("inf")
    for track in tracks:
        distance = float(np.linalg.norm(candidate.center - track.center))
        spacing_ok = abs(candidate.spacing_px - track.spacing_px) <= max(10.0, 0.45 * track.spacing_px)
        if spacing_ok and distance < merge_distance and distance < best_distance:
            best_track = track
            best_distance = distance
    if best_track is None:
        track_id = f"grid_{len(tracks)}"
        tracks.append(_GridTrack(
            track_id=track_id,
            centers=np.asarray(candidate.centers, dtype=np.float32),
            spacing_px=float(candidate.spacing_px),
            center=np.asarray(candidate.center, dtype=np.float32),
            first_frame=frame_index,
            last_frame=frame_index,
            confidence_sum=float(candidate.confidence),
        ))
        return

    alpha = 0.25
    best_track.centers = (1.0 - alpha) * best_track.centers + alpha * np.asarray(candidate.centers, dtype=np.float32)
    best_track.center = (1.0 - alpha) * best_track.center + alpha * np.asarray(candidate.center, dtype=np.float32)
    best_track.spacing_px = float((1.0 - alpha) * best_track.spacing_px + alpha * candidate.spacing_px)
    best_track.last_frame = frame_index
    best_track.detections += 1
    best_track.confidence_sum += float(candidate.confidence)


def _filter_stable_tracks(tracks: list[_GridTrack], frame_shape: tuple[int, ...], config: dict) -> tuple[list[_GridTrack], list[dict[str, Any]]]:
    diagonal = _frame_diagonal(frame_shape)
    min_spacing = diagonal * config["candidate_min_spacing_ratio"] * 0.85
    max_spacing = diagonal * config["track_max_spacing_ratio"]
    accepted: list[_GridTrack] = []
    rejected: list[dict[str, Any]] = []
    for track in tracks:
        reason = ""
        if track.detections < config["min_grid_detections"]:
            reason = "premalo_zaznav"
        elif track.confidence < config["track_min_confidence"]:
            reason = "nizka_geometrijska_zanesljivost"
        elif not min_spacing <= track.spacing_px <= max_spacing:
            reason = "neprepricljiv_razmik"
        if reason:
            rejected.append({
                "grid_id": track.track_id,
                "reason": reason,
                "detections": track.detections,
                "confidence": round(track.confidence, 4),
                "spacing_px": round(track.spacing_px, 2),
            })
        else:
            accepted.append(track)
    return accepted, rejected


def _roi_mean_brightness(frame: np.ndarray, roi: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = roi
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]) / 255.0)


def _light_score(frame: np.ndarray, track: _GridTrack, config: dict) -> float:
    rois = make_rois_from_centers(track.centers, frame.shape, config)
    brightness = np.array([_roi_mean_brightness(frame, roi) for roi in rois], dtype=np.float32)
    if brightness.size == 0:
        return 0.0
    # Osvetljena ploščica ima hkrati svetle centre in majhno razpršenost med luknjami.
    mean_bright = float(np.mean(brightness))
    bright_fraction = float(np.mean(brightness > 0.55))
    uniformity = max(0.0, 1.0 - float(np.std(brightness)) * 2.0)
    return float(np.clip(0.72 * mean_bright + 0.20 * bright_fraction + 0.08 * uniformity, 0.0, 1.0))


def _estimate_light_threshold(scores: list[float], config: dict) -> float:
    if not scores:
        return 0.55
    values = np.asarray(scores, dtype=np.float32)
    low = float(np.percentile(values, 25))
    high = float(np.percentile(values, 75))
    threshold = (low + high) / 2.0 if high - low >= 0.08 else float(np.percentile(values, 60))
    return float(np.clip(threshold, config["target_light_threshold_min"], config["target_light_threshold_max"]))


def _classify_reason(transitions: list[dict[str, Any]], selected_id: str) -> str:
    on_sets = [set(item["on_grids"]) for item in transitions]
    saw_both_on = any(len(items) >= 2 for items in on_sets)
    saw_all_off = any(len(items) == 0 for items in on_sets)
    saw_selected_alone = any(items == {selected_id} for items in on_sets)
    if saw_both_on and saw_all_off and saw_selected_alone:
        return "both_on_both_off_one_on"
    if saw_both_on and saw_selected_alone:
        return "video_started_with_both_on_then_one_stayed_lit"
    if saw_selected_alone:
        return "late_start_target_stayed_lit"
    return "fallback_stable_grid_with_roi_changes"


def _find_target_by_light_sequence(tracks: list[_GridTrack], fps: float, config: dict) -> tuple[_GridTrack | None, str, float, int, list[dict[str, Any]], float]:
    all_scores = [score for track in tracks for _, _, score in track.light_scores]
    threshold = _estimate_light_threshold(all_scores, config)
    frame_ids = sorted({frame_index for track in tracks for frame_index, _, _ in track.light_scores})
    by_track = {track.track_id: track for track in tracks}
    score_lookup = {
        (track.track_id, frame_index): score
        for track in tracks
        for frame_index, _, score in track.light_scores
    }

    transitions: list[dict[str, Any]] = []
    previous_on: set[str] | None = None
    alone_runs: dict[str, tuple[int, int]] = {}
    current_alone: str | None = None
    current_start = 0
    current_length = 0

    for frame_index in frame_ids:
        time_s = frame_index / max(fps, 1e-6)
        on_grids = [track.track_id for track in tracks if score_lookup.get((track.track_id, frame_index), 0.0) >= threshold]
        on_set = set(on_grids)
        if previous_on is None or on_set != previous_on:
            transitions.append({"frame": frame_index, "time_s": round(time_s, 3), "on_grids": sorted(on_grids)})
            previous_on = on_set
        alone_id = on_grids[0] if len(on_grids) == 1 else None
        if alone_id != current_alone:
            if current_alone is not None:
                best = alone_runs.get(current_alone, (0, current_start))
                if current_length > best[0]:
                    alone_runs[current_alone] = (current_length, current_start)
            current_alone = alone_id
            current_start = frame_index
            current_length = 1 if alone_id else 0
        elif alone_id:
            current_length += 1
    if current_alone is not None:
        best = alone_runs.get(current_alone, (0, current_start))
        if current_length > best[0]:
            alone_runs[current_alone] = (current_length, current_start)

    if alone_runs:
        stride_frames = np.median(np.diff(frame_ids)) if len(frame_ids) > 1 else 1.0
        min_run_frames = max(1.0, config["target_stays_on_seconds"] * fps / max(stride_frames, 1.0))
        best_id, (best_len, best_start) = max(alone_runs.items(), key=lambda item: item[1][0])
        if best_len >= min_run_frames or len(tracks) == 1:
            selected = by_track[best_id]
            sorted_runs = sorted((length for length, _ in alone_runs.values()), reverse=True)
            margin = (sorted_runs[0] - sorted_runs[1]) / max(sorted_runs[0], 1) if len(sorted_runs) > 1 else 1.0
            confidence = float(np.clip(0.62 + 0.25 * min(1.0, best_len / max(min_run_frames, 1.0)) + 0.13 * margin, 0.0, 0.97))
            return selected, _classify_reason(transitions, selected.track_id), confidence, int(best_start), transitions, threshold

    if not tracks:
        return None, "no_stable_grid", 0.0, 0, transitions, threshold
    selected = max(tracks, key=lambda track: (track.local_change_score, track.confidence, track.detections))
    return selected, "fallback_stable_grid_with_roi_changes", float(np.clip(0.45 + selected.local_change_score, 0.0, 0.78)), selected.first_frame, transitions, threshold


def _estimate_local_changes(tracks: list[_GridTrack], frames: list[np.ndarray], config: dict) -> None:
    if not tracks or len(frames) < 2:
        return
    roi_config = dict(config)
    roi_config["roi_radius_spacing_factor"] = 0.34
    for track in tracks:
        rois = make_rois_from_centers(track.centers, frames[0].shape, roi_config)
        baseline_values = []
        for roi in rois:
            baseline_values.append(_roi_mean_brightness(frames[0], roi))
        baseline = np.asarray(baseline_values, dtype=np.float32)
        max_local = 0.0
        max_count = 0
        for frame in frames[1:]:
            current = np.asarray([_roi_mean_brightness(frame, roi) for roi in rois], dtype=np.float32)
            diff = np.abs(current - baseline)
            local_count = int(np.sum(diff > 0.13))
            max_count = max(max_count, local_count)
            if diff.size:
                top = np.sort(diff)[-min(3, len(diff)) :]
                max_local = max(max_local, float(np.mean(top)))
        track.local_change_score = float(np.clip(max_local, 0.0, 1.0))
        track.max_occupied_like_count = max_count


def calibrate_target_grid_from_light_sequence(
    frames: list[np.ndarray],
    frame_indices: list[int],
    fps: float,
    config: dict,
) -> CalibrationResult:
    """Detect stable 3x3 grids and select the target from lighting and ROI evidence."""
    if not frames or not frame_indices:
        return CalibrationResult(None, [], "", "no_frames", 0.0, 0.0, 0, 0.0, 0.0, {"rejected_tracks": []})

    tracks: list[_GridTrack] = []
    detections_debug: list[dict[str, Any]] = []
    for frame, frame_index in zip(frames, frame_indices):
        candidates = detect_candidate_3x3_grids(frame, config)
        detections_debug.append({"frame": frame_index, "candidate_count": len(candidates)})
        for candidate in candidates:
            candidate.source_frame = frame_index
            _merge_candidate(tracks, candidate, frame_index, config, frame.shape)

    accepted, rejected = _filter_stable_tracks(tracks, frames[0].shape, config)
    _estimate_local_changes(accepted, frames, config)
    for track in accepted:
        for frame, frame_index in zip(frames, frame_indices):
            track.light_scores.append((frame_index, frame_index / max(fps, 1e-6), _light_score(frame, track, config)))

    selected, reason, confidence, lock_frame, transitions, threshold = _find_target_by_light_sequence(accepted, fps, config)
    if selected is None:
        return CalibrationResult(
            None,
            [],
            "",
            reason,
            0.0,
            0.0,
            0,
            0.0,
            0.0,
            {"detections": detections_debug, "rejected_tracks": rejected, "light_state_transitions": transitions},
        )

    if accepted:
        activity_best = max(accepted, key=lambda track: track.local_change_score)
        if activity_best is not selected:
            margin = activity_best.local_change_score - selected.local_change_score
            can_override = reason.startswith("fallback") or (reason.startswith("late_start") and confidence < 0.80)
            if can_override and activity_best.local_change_score >= config["activity_override_min_score"] and margin >= config["activity_override_margin"]:
                selected = activity_best
                reason = "fallback_stable_grid_with_roi_changes"
                confidence = max(confidence, float(np.clip(0.58 + selected.local_change_score, 0.0, 0.86)))
                lock_frame = selected.first_frame

    centers = np.asarray(selected.centers, dtype=np.float32)
    spacing_px = compute_grid_spacing_px(centers)
    meters_per_pixel = (KNOWN_HOLE_SPACING_MM / 1000.0) / spacing_px if spacing_px > 0 else 0.0
    target_rois = make_rois_from_centers(centers, frames[0].shape, config)
    baseline_frame_index = int(lock_frame + round(max(fps, 1.0) * config["baseline_wait_seconds"]))
    debug_tracks = []
    for track in accepted:
        debug_tracks.append({
            "grid_id": track.track_id,
            "detections": track.detections,
            "confidence": round(track.confidence, 4),
            "spacing_px": round(track.spacing_px, 2),
            "local_change_score": round(track.local_change_score, 4),
            "max_occupied_like_count": track.max_occupied_like_count,
            "mean_light_score": round(float(np.mean([score for _, _, score in track.light_scores])) if track.light_scores else 0.0, 4),
        })
    return CalibrationResult(
        target_centers=centers,
        target_rois=target_rois,
        target_grid_id=selected.track_id,
        target_reason=reason,
        target_lock_time_s=lock_frame / max(fps, 1e-6),
        target_confidence=confidence,
        baseline_frame_index=baseline_frame_index,
        grid_spacing_px=spacing_px,
        meters_per_pixel=meters_per_pixel,
        calibration_debug={
            "selected_target": {
                "grid_id": selected.track_id,
                "reason": reason,
                "confidence": round(confidence, 4),
                "spacing_px": round(spacing_px, 2),
                "meters_per_pixel": meters_per_pixel,
                "baseline_frame_index": baseline_frame_index,
            },
            "light_threshold": threshold,
            "light_state_transitions": transitions,
            "tracks": debug_tracks,
            "rejected_tracks": rejected,
            "detections": detections_debug,
        },
    )
