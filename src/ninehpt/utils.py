# Pomožne funkcije za imena datotek in poti v datotečnem sistemu.

from __future__ import annotations

import os
import re
from pathlib import Path

from .models import VideoMetadata


VIDEO_RE = re.compile(r"patient_(?P<patient>\d+)camP_(?P<camera>\d+)_")


def parse_video_identity(path: str | os.PathLike[str]) -> tuple[str, str, str]:
    video_id = Path(path).stem
    match = VIDEO_RE.search(video_id)
    if not match:
        patient_match = re.search(r"patient_(\d+)", video_id)
        patient_id = patient_match.group(1) if patient_match else "unknown"
        return video_id, patient_id, "unknown"
    return video_id, match.group("patient"), f"P{match.group('camera')}"


def build_video_metadata(path: str, fps: float, frame_count: int, width: int, height: int) -> VideoMetadata:
    video_id, patient_id, camera_id = parse_video_identity(path)
    safe_fps = fps if fps and fps > 0 else 25.0
    return VideoMetadata(
        video_id=video_id,
        patient_id=patient_id,
        camera_id=camera_id,
        path=str(path),
        fps=safe_fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_s=frame_count / safe_fps if frame_count > 0 else 0.0,
    )


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def find_video_files(root: str | os.PathLike[str]) -> list[str]:
    root_path = Path(root)
    if root_path.is_file() and root_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        return [str(root_path)]
    videos: list[str] = []
    for suffix in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        videos.extend(str(path) for path in root_path.rglob(suffix))
    return sorted(videos)


def format_float(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""
