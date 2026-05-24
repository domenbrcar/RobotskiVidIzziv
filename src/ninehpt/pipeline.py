"""End-to-end video processing pipeline for 9HPT analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .calibration import calibrate_target_grid_from_light_sequence
from .config import DEFAULT_CONFIG, EMPTY
from .gui import DashboardRenderer
from .hand_tracking import MediaPipeHandDetector, draw_selected_hand, select_patient_hand
from .kinematics import KinematicsTracker
from .models import CalibrationResult, HandSelectionState, VideoMetadata
from .occupancy import PegOccupancyTracker, initialize_roi_baseline
from .utils import build_video_metadata, ensure_dir, format_float, write_csv


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


def _append_history(histories: dict[str, list[float]], time_s: float, kinematics: dict[str, dict[str, float]], config: dict) -> None:
    alpha = float(config["gui_smoothing_alpha"])
    histories.setdefault("time", []).append(time_s)
    mapping = {
        "hand_path": ("hand", "path_m"),
        "hand_velocity": ("hand", "velocity_mps"),
        "hand_acceleration": ("hand", "acceleration_mps2"),
    }
    for key, (part, value_key) in mapping.items():
        raw = float(kinematics.get(part, {}).get(value_key, 0.0))
        previous = histories.get(key, [raw])[-1] if histories.get(key) else raw
        histories.setdefault(key, []).append(alpha * raw + (1.0 - alpha) * previous)
    max_len = int(max(10, config["graph_history_seconds"] * 2.5 * 60))
    for key in list(histories):
        if len(histories[key]) > max_len:
            histories[key] = histories[key][-max_len:]


def _event_rows(events: list) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        rows.append({
            "event_number": event.event_number,
            "event_type": event.event_type,
            "roi_index": event.roi_index,
            "frame": event.frame,
            "time_s": format_float(event.time_s, 3),
            "previous_state": event.previous_state,
            "new_state": event.new_state,
            "change_score": format_float(event.change_score, 4),
            "confidence": format_float(event.confidence, 4),
            "hand_recently_left": int(event.hand_recently_left),
            "source": event.source,
        })
    return rows


def _write_calibration_debug(path: Path, calibration: CalibrationResult) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(calibration.calibration_debug, handle, indent=2, ensure_ascii=False)


def process_video(video_path: str, output_root: str, config: dict | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG if config is None else config)
    metadata, cap = _read_video_properties(video_path, config)
    frames, indices = _sample_calibration_frames(video_path, metadata, config, fallback=False)
    calibration = calibrate_target_grid_from_light_sequence(frames, indices, metadata.fps, config)
    if not calibration.target_grid_locked:
        frames, indices = _sample_calibration_frames(video_path, metadata, config, fallback=True)
        calibration = calibrate_target_grid_from_light_sequence(frames, indices, metadata.fps, config)

    processing_start = _processing_start_frame(calibration) if calibration.target_grid_locked else 0
    baseline_frames = _read_baseline_frames(video_path, metadata, max(processing_start, calibration.baseline_frame_index), config) if calibration.target_grid_locked else []
    baselines = initialize_roi_baseline(baseline_frames or frames[:1], calibration.target_rois, config) if calibration.target_grid_locked else []

    patient_dir = ensure_dir(Path(output_root) / f"patient_{metadata.patient_id}" / metadata.video_id)
    videos_dir = ensure_dir(patient_dir / "videos")
    gui_path = videos_dir / f"gui_{metadata.video_id}.mp4"
    states_path = patient_dir / f"gui_pin_states_{metadata.video_id}.csv"
    events_path = patient_dir / f"gui_pin_events_{metadata.video_id}.csv"
    kinematics_path = patient_dir / f"gui_kinematics_{metadata.video_id}.csv"
    summary_path = patient_dir / f"gui_summary_{metadata.video_id}.csv"
    debug_path = patient_dir / f"gui_calibration_debug_{metadata.video_id}.json"

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
    state_rows: list[dict[str, Any]] = []
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
            selected_hand, hand_state = select_patient_hand(
                detections,
                hand_state,
                frame_index,
                metadata.fps,
                calibration.target_centers,
                frame.shape,
                config,
            )
            if selected_hand is not None:
                hand_confidences.append(float(hand_state.confidence))
            target_locked_now = bool(calibration.target_grid_locked and frame_index >= processing_start)
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
                selected_hand,
            )
            writer.write(gui_frame)

            row: dict[str, Any] = {
                "frame": frame_index,
                "time_s": format_float(time_s, 3),
                "target_locked": int(target_locked_now),
                "peg_count": int(peg_result["peg_count"]),
                "global_light_event": int(peg_result["global_light_event"]),
                "hand_in_grid": int(peg_result["hand_in_grid"]),
                "selected_hand_confidence": format_float(hand_state.confidence, 4),
                "selected_hand_source": hand_state.selected_hand_source,
                "rejected_switch_reason": hand_state.rejected_switch_reason,
                "active_roi": peg_result["active_roi"],
                "rejected_reason": peg_result["rejected_reason"],
            }
            for idx, state in enumerate(peg_result["roi_states"]):
                row[f"roi_{idx}_state"] = state
            for idx, state in enumerate(peg_result["roi_display_states"]):
                row[f"roi_{idx}_display_state"] = state
            for diag in peg_result["roi_diagnostics"]:
                idx = diag["roi"]
                for key in (
                    "change_score_ema",
                    "post_insert_score_ema",
                    "change_from_baseline",
                    "change_from_pre_entry",
                    "center_disappearance",
                    "center_brightness",
                    "dark_score",
                    "blue_score",
                    "occupied_frames",
                    "empty_frames",
                    "hand_inside",
                    "hand_recently_left",
                    "confidence",
                    "rejected_reason",
                ):
                    value = diag.get(key, "")
                    row[f"roi_{idx}_{key}"] = format_float(value, 4) if isinstance(value, float) else value
            state_rows.append(row)

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

    state_fields = list(state_rows[0].keys()) if state_rows else ["frame", "time_s", "peg_count"]
    kin_fields = list(kinematic_rows[0].keys()) if kinematic_rows else ["frame", "time_s"]
    events = peg_tracker.events if peg_tracker is not None else []
    write_csv(states_path, state_fields, state_rows)
    write_csv(kinematics_path, kin_fields, kinematic_rows)
    write_csv(events_path, [
        "event_number", "event_type", "roi_index", "frame", "time_s", "previous_state",
        "new_state", "change_score", "confidence", "hand_recently_left", "source",
    ], _event_rows(events))
    _write_calibration_debug(debug_path, calibration)

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
        "hand_mean_confidence": format_float(float(np.mean(hand_confidences)) if hand_confidences else 0.0, 5),
        "frames_processed": frame_index,
        "mediapipe_available": int(detector.available),
        "output_gui_path": str(gui_path),
    }
    write_csv(summary_path, list(summary.keys()), [summary])
    return {
        **summary,
        "summary_path": str(summary_path),
        "states_path": str(states_path),
        "events_path": str(events_path),
        "kinematics_path": str(kinematics_path),
        "debug_path": str(debug_path),
        "gui_path": str(gui_path),
    }


def write_processing_report(output_root: str, rows: list[dict[str, Any]]) -> Path:
    report_path = Path(output_root) / "processing_report.csv"
    if rows:
        fields = list(rows[0].keys())
    else:
        fields = ["video_id", "patient_id", "target_locked", "max_peg_count", "final_peg_count"]
    write_csv(report_path, fields, rows)
    return report_path
