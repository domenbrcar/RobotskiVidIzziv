# Celoten potek obdelave videa za analizo 9HPT.

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .calibration import calibrate_target_grid_from_light_sequence
from .config import DEFAULT_CONFIG, EMPTY
from .grid_detection import compute_grid_spacing_px, make_rois_from_centers
from .gui import DashboardRenderer
from .hand_tracking import MediaPipeHandDetector, draw_selected_hand, select_patient_hand
from .kinematics import KinematicsTracker
from .models import CalibrationResult, DetectedHand, HandSelectionState, VideoMetadata
from .occupancy import PegOccupancyTracker, initialize_roi_baseline
from .utils import build_video_metadata, ensure_dir, format_float


def _read_video_properties(video_path: str, config: dict) -> tuple[VideoMetadata, cv2.VideoCapture]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Videoposnetka ni mogoče odpreti: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or config["gui_fps_fallback"]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return build_video_metadata(video_path, fps, frame_count, width, height), cap


def _read_frames_at(video_path: str, frame_indices: list[int]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def _sample_calibration_frames(video_path: str, metadata: VideoMetadata, config: dict, fallback: bool = False) -> tuple[list[np.ndarray], list[int]]:
    stride = int(config["calibration_fallback_stride_frames"] if fallback else config["calibration_stride_frames"])
    max_frame = metadata.frame_count if fallback else min(metadata.frame_count, int(round(metadata.fps * config["calibration_max_seconds"])))
    indices = list(range(0, max_frame, max(1, stride)))
    frames = _read_frames_at(video_path, indices)
    if len(frames) != len(indices):
        indices = indices[: len(frames)]
    return frames, indices


def _read_baseline_frames(video_path: str, metadata: VideoMetadata, start_frame: int, config: dict) -> list[np.ndarray]:
    if metadata.frame_count <= 0:
        return []
    start = min(max(0, int(start_frame)), metadata.frame_count - 1)
    stride = max(1, int(config["baseline_stride_frames"]))
    local_count = max(int(config["baseline_min_frames"]), int(round(metadata.fps * config["baseline_window_seconds"])))
    local_stop = min(metadata.frame_count, start + local_count * stride)
    local_indices = list(range(start, local_stop, stride))
    global_indices = list(range(0, metadata.frame_count, stride * 2))
    indices = sorted(set(local_indices + global_indices))
    return _read_frames_at(video_path, indices)


def _processing_start_frame(calibration: CalibrationResult) -> int:
    selected = calibration.target_grid_id
    transitions = calibration.calibration_debug.get("light_state_transitions", [])
    for transition in transitions:
        on_grids = set(transition.get("on_grids", []))
        if selected and on_grids == {selected}:
            return int(transition.get("frame", 0))
    return 0


def _candidate_grid_tracks(calibration: CalibrationResult) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    for track in calibration.calibration_debug.get("tracks", []):
        centers = np.asarray(track.get("centers", []), dtype=np.float32)
        if centers.shape != (9, 2):
            continue
        spacing_px = float(track.get("spacing_px", 0.0)) or compute_grid_spacing_px(centers)
        tracks.append({
            "grid_id": str(track.get("grid_id", "")),
            "centers": centers,
            "spacing_px": spacing_px,
            "first_frame": int(track.get("first_frame", 0)),
        })
    return tracks


def _expanded_grid_box(centers: np.ndarray, spacing_px: float, frame_shape: tuple[int, ...], expansion_factor: float) -> tuple[int, int, int, int]:
    points = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    # Vključi roko tik pred luknjami, ne samo centre lukenj.
    margin = max(8.0, float(spacing_px) * float(expansion_factor))
    height, width = frame_shape[:2]
    return (
        max(0, int(np.floor(np.min(points[:, 0]) - margin))),
        max(0, int(np.floor(np.min(points[:, 1]) - margin))),
        min(width, int(np.ceil(np.max(points[:, 0]) + margin))),
        min(height, int(np.ceil(np.max(points[:, 1]) + margin))),
    )


def _patch_skin_fraction_bgr(patch: np.ndarray) -> float:
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    skin = (((h <= 25) | (h >= 165)) & (s >= 28) & (s <= 180) & (v >= 45))
    return float(np.mean(skin))


def _hand_activity_score(hand: DetectedHand, centers: np.ndarray, spacing_px: float, frame_shape: tuple[int, ...], config: dict) -> float:
    box = _expanded_grid_box(centers, spacing_px, frame_shape, float(config["hand_target_box_expansion_factor"]))
    points = [hand.pinch_center, hand.thumb_tip, hand.index_tip, hand.palm_center]
    inside_points = sum(1 for point in points if box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])
    centers_np = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    min_hole_distance = min(float(np.linalg.norm(point - center)) for point in points for center in centers_np)
    grid_center = np.mean(centers_np, axis=0)
    center_distance = min(float(np.linalg.norm(point - grid_center)) for point in (hand.pinch_center, hand.palm_center))
    # Roka je uporabna, če je blizu posamezne luknje.
    near_hole = float(np.exp(-min_hole_distance / max(2.5 * spacing_px, 1.0)))
    # Doda stabilnost, ko je roka med luknjami.
    near_grid = float(np.exp(-center_distance / max(4.5 * spacing_px, 1.0)))
    inside = min(1.0, inside_points / 2.0)
    return float(hand.confidence) * (0.52 * near_hole + 0.30 * near_grid + 0.18 * inside)


def _sample_target_activity(video_path: str, metadata: VideoMetadata, tracks: list[dict[str, Any]], detector: MediaPipeHandDetector, config: dict) -> dict[str, Any]:
    scores = {track["grid_id"]: 0.0 for track in tracks}
    hit_frames = {track["grid_id"]: 0 for track in tracks}
    sampled_frames = 0
    scan_frames = min(metadata.frame_count, int(round(metadata.fps * float(config["hand_target_scan_seconds"]))))
    stride_frames = max(1, int(round(metadata.fps * float(config["hand_target_scan_stride_seconds"]))))
    previous_gray: np.ndarray | None = None
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    try:
        while frame_index < scan_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride_frames != 0:
                frame_index += 1
                continue

            sampled_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_scores = {track["grid_id"]: 0.0 for track in tracks}

            for track in tracks:
                grid_id = track["grid_id"]
                box = _expanded_grid_box(track["centers"], track["spacing_px"], frame.shape, float(config["hand_target_box_expansion_factor"]))
                x1, y1, x2, y2 = box
                patch = frame[y1:y2, x1:x2]
                skin_score = min(1.0, _patch_skin_fraction_bgr(patch) * 4.0)
                motion_score = 0.0
                if previous_gray is not None and x2 > x1 and y2 > y1:
                    diff = cv2.absdiff(gray[y1:y2, x1:x2], previous_gray[y1:y2, x1:x2])
                    motion_score = min(1.0, float(np.mean(diff)) / 255.0 * 8.0)
                frame_scores[grid_id] += 0.45 * motion_score + 0.35 * skin_score

            detections = detector.detect(frame, frame_index, metadata.fps) if detector.available else []
            for hand in detections:
                for track in tracks:
                    grid_id = track["grid_id"]
                    frame_scores[grid_id] += 1.35 * _hand_activity_score(hand, track["centers"], track["spacing_px"], frame.shape, config)

            for grid_id, score in frame_scores.items():
                scores[grid_id] += float(score)
                if score >= 0.35:
                    hit_frames[grid_id] += 1

            previous_gray = gray
            frame_index += 1
    finally:
        cap.release()

    return {
        "scores": {grid_id: round(score, 4) for grid_id, score in scores.items()},
        "hit_frames": hit_frames,
        "sampled_frames": sampled_frames,
        "mediapipe_available": int(detector.available),
    }


def _activity_override_target(selected_id: str, activity: dict[str, Any], config: dict) -> str | None:
    scores = {str(key): float(value) for key, value in activity.get("scores", {}).items()}
    hit_frames = {str(key): int(value) for key, value in activity.get("hit_frames", {}).items()}
    valid_ids = [grid_id for grid_id in scores if hit_frames.get(grid_id, 0) >= int(config["hand_target_min_hit_frames"])]
    if not valid_ids:
        return None
    best_id = max(valid_ids, key=lambda grid_id: scores[grid_id])
    if best_id == selected_id:
        return None
    best_score = scores[best_id]
    selected_score = scores.get(selected_id, 0.0)
    enough_score = best_score >= float(config["hand_target_override_min_score"])
    enough_margin = best_score >= selected_score + float(config["hand_target_override_margin"])
    enough_ratio = selected_score <= 0.01 or best_score >= selected_score * float(config["hand_target_override_ratio"])
    return best_id if enough_score and enough_margin and enough_ratio else None


def _replace_calibration_target(calibration: CalibrationResult, track: dict[str, Any], metadata: VideoMetadata, config: dict, activity: dict[str, Any]) -> CalibrationResult:
    centers = np.asarray(track["centers"], dtype=np.float32).reshape(9, 2)
    spacing_px = compute_grid_spacing_px(centers)
    meters_per_pixel = (float(config["known_hole_spacing_mm"]) / 1000.0) / spacing_px if spacing_px > 0 else 0.0
    target_rois = make_rois_from_centers(centers, (metadata.height, metadata.width, 3), config)
    reason = f"{calibration.target_reason}_hand_activity_override"
    confidence = float(np.clip(max(calibration.target_confidence, 0.72), 0.0, 0.95))
    debug = dict(calibration.calibration_debug)
    debug["hand_activity"] = activity
    debug["selected_target"] = {
        **dict(debug.get("selected_target", {})),
        "grid_id": track["grid_id"],
        "reason": reason,
        "confidence": round(confidence, 4),
        "spacing_px": round(spacing_px, 2),
        "meters_per_pixel": meters_per_pixel,
        "baseline_frame_index": calibration.baseline_frame_index,
    }
    return CalibrationResult(
        target_centers=centers,
        target_rois=target_rois,
        target_grid_id=track["grid_id"],
        target_reason=reason,
        target_lock_time_s=calibration.target_lock_time_s,
        target_confidence=confidence,
        baseline_frame_index=calibration.baseline_frame_index,
        grid_spacing_px=spacing_px,
        meters_per_pixel=meters_per_pixel,
        calibration_debug=debug,
    )


def _refine_target_with_hand_activity(video_path: str, metadata: VideoMetadata, calibration: CalibrationResult, detector: MediaPipeHandDetector, config: dict) -> CalibrationResult:
    if not calibration.target_grid_locked:
        return calibration
    tracks = _candidate_grid_tracks(calibration)
    if len(tracks) < 2:
        return calibration
    activity = _sample_target_activity(video_path, metadata, tracks, detector, config)
    override_id = _activity_override_target(calibration.target_grid_id, activity, config)
    if override_id is None:
        calibration.calibration_debug["hand_activity"] = activity
        return calibration
    chosen = next(track for track in tracks if track["grid_id"] == override_id)
    return _replace_calibration_target(calibration, chosen, metadata, config, activity)


def _target_grid_patient_side(calibration: CalibrationResult, metadata: VideoMetadata, config: dict) -> str:
    if not calibration.target_grid_locked or calibration.target_centers is None:
        return "right"
    center = np.mean(np.asarray(calibration.target_centers, dtype=np.float32).reshape(-1, 2), axis=0)
    axis_by_camera = config.get("patient_side_axis_by_camera", {})
    flip_by_camera = config.get("patient_side_flip_by_camera", {})
    axis = str(axis_by_camera.get(metadata.camera_id, "x")).lower()
    if axis == "y":
        value = float(center[1])
        midpoint = float(metadata.height) / 2.0
    else:
        value = float(center[0])
        midpoint = float(metadata.width) / 2.0
    side = "right" if value >= midpoint else "left"
    if bool(flip_by_camera.get(metadata.camera_id, False)):
        side = "left" if side == "right" else "right"
    return side


def _append_history(histories: dict[str, list[float]], time_s: float, kinematics: dict[str, dict[str, float]], config: dict) -> None:
    alpha = float(config["gui_smoothing_alpha"])
    histories.setdefault("time", []).append(time_s)
    mapping = {
        "hand_path": ("hand", "path_m"),
        "hand_velocity": ("hand", "velocity_mps"),
        "hand_acceleration": ("hand", "acceleration_mps2"),
        "thumb_path": ("thumb", "path_m"),
        "thumb_velocity": ("thumb", "velocity_mps"),
        "thumb_acceleration": ("thumb", "acceleration_mps2"),
        "index_path": ("index", "path_m"),
        "index_velocity": ("index", "velocity_mps"),
        "index_acceleration": ("index", "acceleration_mps2"),
    }
    for key, (part, value_key) in mapping.items():
        raw = float(kinematics.get(part, {}).get(value_key, 0.0))
        previous = histories.get(key, [raw])[-1] if histories.get(key) else raw
        histories.setdefault(key, []).append(alpha * raw + (1.0 - alpha) * previous)
    max_len = int(max(10, config["graph_history_seconds"] * 2.5 * 60))
    for key in list(histories):
        if len(histories[key]) > max_len:
            histories[key] = histories[key][-max_len:]


def _numeric_column(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, np.nan)))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=np.float32)


def _save_time_graph(path: Path, title: str, y_label: str, time_s: np.ndarray, series: list[tuple[str, np.ndarray, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    plotted = False
    for label, values, color in series:
        valid = np.isfinite(time_s) & np.isfinite(values)
        if np.count_nonzero(valid) < 2:
            continue
        ax.plot(time_s[valid], values[valid], label=label, color=color, linewidth=1.8)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.28)
    if plotted and len(series) > 1:
        ax.legend(loc="best", frameon=False)
    if not plotted:
        ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_trajectory(path: Path, title: str, rows: list[dict[str, Any]], part: str, color: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_m = _numeric_column(rows, f"{part}_x_m")
    y_m = _numeric_column(rows, f"{part}_y_m")
    valid = _numeric_column(rows, f"{part}_valid") >= 0.5
    mask = valid & np.isfinite(x_m) & np.isfinite(y_m)

    fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=150)
    if np.count_nonzero(mask) >= 2:
        ax.plot(x_m[mask], y_m[mask], color=color, linewidth=1.8)
        ax.scatter(x_m[mask][0], y_m[mask][0], color="#2ca02c", s=30, label="Start", zorder=3)
        ax.scatter(x_m[mask][-1], y_m[mask][-1], color="#d62728", s=30, label="End", zorder=3)
        ax.legend(loc="best", frameon=False)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No valid trajectory", transform=ax.transAxes, ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("X position [m]")
    ax.set_ylabel("Y position [m]")
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_output_figures(patient_dir: Path, kinematic_rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    graphs_dir = ensure_dir(patient_dir / "graphs")
    trajectories_dir = ensure_dir(patient_dir / "trajectories")
    time_s = _numeric_column(kinematic_rows, "time_s")
    parts = {
        "hand": ("Hand", "#1f77b4"),
        "thumb": ("Thumb", "#d8a620"),
        "index": ("Index finger", "#2ca58d"),
    }
    metrics = {
        "path_m": ("path", "Path", "Path [m]"),
        "velocity_mps": ("velocity", "Velocity", "Velocity [m/s]"),
        "acceleration_mps2": ("acceleration", "Acceleration", "Acceleration [m/s²]"),
    }

    for part, (part_label, color) in parts.items():
        for metric_key, (metric_slug, metric_label, y_label) in metrics.items():
            values = _numeric_column(kinematic_rows, f"{part}_{metric_key}")
            _save_time_graph(
                graphs_dir / f"{part}_{metric_slug}_over_time.png",
                f"{part_label} {metric_label} Over Time",
                y_label,
                time_s,
                [(part_label, values, color)],
            )

    for metric_key, (metric_slug, metric_label, y_label) in metrics.items():
        combined_series = [
            (part_label, _numeric_column(kinematic_rows, f"{part}_{metric_key}"), color)
            for part, (part_label, color) in parts.items()
        ]
        _save_time_graph(
            graphs_dir / f"combined_{metric_slug}_over_time.png",
            f"Combined {metric_label} Over Time",
            y_label,
            time_s,
            combined_series,
        )

    for part, (part_label, color) in parts.items():
        _save_trajectory(trajectories_dir / f"{part}_trajectory.png", f"{part_label} Trajectory", kinematic_rows, part, color)
    return graphs_dir, trajectories_dir


def process_video(video_path: str, output_root: str, config: dict | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG if config is None else config)
    metadata, cap = _read_video_properties(video_path, config)
    frames, indices = _sample_calibration_frames(video_path, metadata, config, fallback=False)
    calibration = calibrate_target_grid_from_light_sequence(frames, indices, metadata.fps, config)
    if not calibration.target_grid_locked:
        frames, indices = _sample_calibration_frames(video_path, metadata, config, fallback=True)
        calibration = calibrate_target_grid_from_light_sequence(frames, indices, metadata.fps, config)

    activity_detector = MediaPipeHandDetector(config)
    try:
        calibration = _refine_target_with_hand_activity(video_path, metadata, calibration, activity_detector, config)
    finally:
        activity_detector.close()

    processing_start = _processing_start_frame(calibration) if calibration.target_grid_locked else 0
    baseline_frames = _read_baseline_frames(video_path, metadata, max(processing_start, calibration.baseline_frame_index), config) if calibration.target_grid_locked else []
    baselines = initialize_roi_baseline(baseline_frames or frames[:1], calibration.target_rois, config) if calibration.target_grid_locked else []
    target_grid_side = _target_grid_patient_side(calibration, metadata, config)

    patient_dir = ensure_dir(Path(output_root) / f"patient_{metadata.patient_id}" / metadata.video_id)
    gui_video_dir = ensure_dir(patient_dir / "gui_video")
    gui_path = gui_video_dir / f"gui_{metadata.video_id}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*str(config["video_codec"]))
    writer = cv2.VideoWriter(str(gui_path), fourcc, metadata.fps, (int(config["gui_width"]), int(config["gui_height"])))
    renderer = DashboardRenderer(int(config["gui_width"]), int(config["gui_height"]), metadata.fps, config)
    detector = MediaPipeHandDetector(config)
    hand_state = HandSelectionState()
    peg_tracker = (
        PegOccupancyTracker(calibration.target_rois, baselines, calibration.target_centers, metadata.fps, config)
        if calibration.target_grid_locked
        else None
    )
    kinematic_tracker = KinematicsTracker(calibration.meters_per_pixel, config)
    histories: dict[str, list[float]] = {}
    kinematic_rows: list[dict[str, Any]] = []
    max_peg_count = 0
    final_peg_count = 0
    hand_confidences: list[float] = []

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            time_s = frame_index / max(metadata.fps, 1e-6)
            detections = detector.detect(frame, frame_index, metadata.fps)
            target_locked_now = bool(calibration.target_grid_locked and frame_index >= processing_start)
            selected_hand, hand_state = select_patient_hand(
                detections,
                hand_state,
                frame_index,
                metadata.fps,
                calibration.target_centers,
                frame.shape,
                config,
                lock_switches=target_locked_now,
            )
            if selected_hand is not None:
                hand_confidences.append(float(hand_state.confidence))
            if peg_tracker is not None:
                peg_result = peg_tracker.update(frame, frame_index, time_s, selected_hand, target_locked_now)
            else:
                peg_result = {
                    "peg_count": 0,
                    "roi_states": [EMPTY] * 9,
                    "roi_display_states": [EMPTY] * 9,
                    "roi_confidences": [0.0] * 9,
                    "global_light_event": 0,
                    "hand_in_grid": 0,
                    "active_roi": -1,
                    "rejected_reason": "target_not_locked",
                    "roi_diagnostics": [],
                }
            max_peg_count = max(max_peg_count, int(peg_result["peg_count"]))
            final_peg_count = int(peg_result["peg_count"])
            kinematics = kinematic_tracker.update(selected_hand, time_s)
            _append_history(histories, time_s, kinematics, config)
            annotated = draw_selected_hand(frame, selected_hand)
            gui_frame = renderer.render(
                annotated,
                metadata,
                frame_index,
                time_s,
                int(peg_result["peg_count"]),
                list(peg_result["roi_display_states"]),
                kinematics,
                histories,
                target_grid_side,
            )
            writer.write(gui_frame)

            kin_row = {"frame": frame_index, "time_s": format_float(time_s, 3)}
            for part in ("hand", "thumb", "index"):
                for key, value in kinematics.get(part, {}).items():
                    kin_row[f"{part}_{key}"] = format_float(value, 5)
            kinematic_rows.append(kin_row)
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        detector.close()

    graphs_dir, trajectories_dir = _write_output_figures(patient_dir, kinematic_rows)

    summary = {
        "video_id": metadata.video_id,
        "patient_id": metadata.patient_id,
        "camera_id": metadata.camera_id,
        "target_locked": int(calibration.target_grid_locked),
        "target_reason": calibration.target_reason,
        "target_confidence": format_float(calibration.target_confidence, 4),
        "target_lock_time_s": format_float(calibration.target_lock_time_s, 3),
        "baseline_frame_index": calibration.baseline_frame_index,
        "grid_spacing_px": format_float(calibration.grid_spacing_px, 3),
        "meters_per_pixel": format_float(calibration.meters_per_pixel, 8),
        "max_peg_count": max_peg_count,
        "final_peg_count": final_peg_count,
        "reached_9": int(max_peg_count == 9),
        "target_patient_side": target_grid_side,
        "hand_mean_confidence": format_float(float(np.mean(hand_confidences)) if hand_confidences else 0.0, 5),
        "frames_processed": frame_index,
        "mediapipe_available": int(detector.available),
        "output_gui_path": str(gui_path),
        "output_graphs_path": str(graphs_dir),
        "output_trajectories_path": str(trajectories_dir),
    }
    return {
        **summary,
        "gui_path": str(gui_path),
        "graphs_path": str(graphs_dir),
        "trajectories_path": str(trajectories_dir),
    }
