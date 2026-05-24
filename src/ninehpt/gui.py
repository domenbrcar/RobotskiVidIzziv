"""Slovenian dashboard renderer for final 9HPT output videos."""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import OCCUPIED
from .models import DetectedHand, VideoMetadata
from .utils import format_float


class DashboardRenderer:
    """Render a clean analysis dashboard around the processed video."""

    def __init__(self, width: int, height: int, fps: float, config: dict):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.config = config
        self.left_w = 270
        self.right_w = 330
        self.gap = 18
        self.center_x = self.left_w + self.gap
        self.center_w = self.width - self.left_w - self.right_w - 2 * self.gap
        self.video_h = int(self.height * 0.66)
        self.progress_y = self.video_h + 18
        self.bg = (22, 24, 28)
        self.panel = (33, 36, 41)
        self.panel_alt = (39, 43, 49)
        self.text = (235, 238, 242)
        self.muted = (156, 165, 174)
        self.green = (46, 190, 116)
        self.red = (214, 78, 74)
        self.blue = (92, 157, 230)
        self.yellow = (230, 188, 88)
        self.fonts = self._load_fonts()

    @staticmethod
    def _load_fonts() -> dict[str, ImageFont.ImageFont]:
        candidates = ["DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"]
        bold_candidates = ["DejaVuSans-Bold.ttf", "Arialbd.ttf", "LiberationSans-Bold.ttf"]

        def load(size: int, bold: bool = False) -> ImageFont.ImageFont:
            names = bold_candidates if bold else candidates
            for name in names:
                try:
                    return ImageFont.truetype(name, size=size)
                except OSError:
                    continue
            return ImageFont.load_default()

        return {
            "title": load(20, True),
            "section": load(15, True),
            "body": load(13),
            "small": load(11),
            "value": load(16, True),
            "graph": load(10),
        }

    def render(
        self,
        frame_bgr: np.ndarray,
        metadata: VideoMetadata,
        frame_index: int,
        time_s: float,
        peg_count: int,
        roi_states: list[str],
        kinematics: dict[str, dict[str, float]],
        histories: dict[str, list[float]],
        selected_hand: DetectedHand | None,
    ) -> np.ndarray:
        image = Image.new("RGB", (self.width, self.height), self.bg)
        draw = ImageDraw.Draw(image)
        self._draw_left(draw, metadata, frame_index, time_s, peg_count, kinematics, selected_hand)
        self._draw_video(image, frame_bgr)
        self._draw_progress(draw, peg_count, roi_states)
        self._draw_right(draw, histories)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def _draw_panel(self, draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], title: str | None = None) -> None:
        draw.rounded_rectangle(xy, radius=8, fill=self.panel)
        if title:
            draw.text((xy[0] + 14, xy[1] + 10), title, fill=self.text, font=self.fonts["section"])

    def _draw_kv(self, draw: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str, color: tuple[int, int, int] | None = None) -> int:
        draw.text((x, y), label, fill=self.muted, font=self.fonts["small"])
        draw.text((x, y + 15), value, fill=color or self.text, font=self.fonts["body"])
        return y + 39

    def _draw_left(
        self,
        draw: ImageDraw.ImageDraw,
        metadata: VideoMetadata,
        frame_index: int,
        time_s: float,
        peg_count: int,
        kinematics: dict[str, dict[str, float]],
        selected_hand: DetectedHand | None,
    ) -> None:
        x = 14
        draw.text((x, 16), "Analiza 9HPT", fill=self.text, font=self.fonts["title"])
        top = 52
        self._draw_panel(draw, (x, top, self.left_w - 12, 224), "Podatki testa")
        y = top + 42
        y = self._draw_kv(draw, x + 14, y, "Pacient", f"patient_{metadata.patient_id}")
        y = self._draw_kv(draw, x + 14, y, "Kamera", metadata.camera_id)
        y = self._draw_kv(draw, x + 14, y, "Čas videa", f"{format_float(metadata.duration_s, 1)} s")

        self._draw_panel(draw, (x, 238, self.left_w - 12, 360), None)
        y = 252
        y = self._draw_kv(draw, x + 14, y, "Čas", f"{time_s:0.2f} s")
        y = self._draw_kv(draw, x + 14, y, "Sličica", str(frame_index))
        self._draw_kv(draw, x + 14, y, "Zatiči", f"{peg_count} / 9", self.green if peg_count == 9 else self.text)

        y = 388
        self._draw_motion_block(draw, x, y, "Kinematika roke", kinematics.get("hand", {}))
        y += 104
        self._draw_motion_block(draw, x, y, "Kinematika palca", kinematics.get("thumb", {}))
        y += 104
        self._draw_motion_block(draw, x, y, "Kinematika kazalca", kinematics.get("index", {}))

        source = "stabilna" if selected_hand is not None else "ni zaznave"
        draw.text((x + 14, self.height - 34), f"Roka: {source}", fill=self.muted, font=self.fonts["small"])

    def _draw_motion_block(self, draw: ImageDraw.ImageDraw, x: int, y: int, title: str, values: dict[str, float]) -> None:
        self._draw_panel(draw, (x, y, self.left_w - 12, y + 92), title)
        draw.text((x + 14, y + 36), f"pot {format_float(values.get('path_m', 0.0), 3)} m", fill=self.text, font=self.fonts["body"])
        draw.text((x + 14, y + 56), f"hitrost {format_float(values.get('velocity_mps', 0.0), 3)} m/s", fill=self.text, font=self.fonts["body"])
        draw.text((x + 14, y + 74), f"pospešek {format_float(values.get('acceleration_mps2', 0.0), 2)} m/s²", fill=self.muted, font=self.fonts["small"])

    def _draw_video(self, image: Image.Image, frame_bgr: np.ndarray) -> None:
        x = self.center_x
        y = 18
        area_w = self.center_w
        area_h = self.video_h - 28
        if frame_bgr.size == 0:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = min(area_w / max(w, 1), area_h / max(h, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        video = Image.fromarray(resized)
        bx = x + (area_w - new_w) // 2
        by = y + (area_h - new_h) // 2
        ImageDraw.Draw(image).rounded_rectangle((x, y, x + area_w, y + area_h), radius=8, fill=(9, 11, 14))
        image.paste(video, (bx, by))

    def _draw_progress(self, draw: ImageDraw.ImageDraw, peg_count: int, roi_states: list[str]) -> None:
        x = self.center_x
        y = self.progress_y
        self._draw_panel(draw, (x, y, x + self.center_w, self.height - 18), None)
        draw.text((x + 18, y + 14), "Zatiči", fill=self.text, font=self.fonts["section"])
        draw.text((x + self.center_w - 82, y + 13), f"{peg_count} / 9", fill=self.text, font=self.fonts["value"])
        grid_x = x + 42
        grid_y = y + 54
        step = 46
        radius = 14
        for idx in range(9):
            row, col = divmod(idx, 3)
            cx = grid_x + col * step
            cy = grid_y + row * step
            occupied = idx < len(roi_states) and roi_states[idx] == OCCUPIED
            fill = self.green if occupied else self.red
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill, outline=(245, 247, 249), width=2)
        legend_x = grid_x + step * 3 + 30
        draw.ellipse((legend_x, grid_y - 12, legend_x + 20, grid_y + 8), fill=self.red)
        draw.text((legend_x + 28, grid_y - 14), "prazno", fill=self.muted, font=self.fonts["body"])
        draw.ellipse((legend_x, grid_y + 24, legend_x + 20, grid_y + 44), fill=self.green)
        draw.text((legend_x + 28, grid_y + 22), "zasedeno", fill=self.muted, font=self.fonts["body"])

    def _draw_right(self, draw: ImageDraw.ImageDraw, histories: dict[str, list[float]]) -> None:
        x = self.width - self.right_w + 12
        panel_w = self.right_w - 26
        top = 18
        graph_h = (self.height - 54) // 3
        self._draw_graph(draw, (x, top, x + panel_w, top + graph_h), "Pot skozi čas", histories.get("time", []), histories.get("hand_path", []), self.green)
        top += graph_h + 12
        self._draw_graph(draw, (x, top, x + panel_w, top + graph_h), "Hitrost skozi čas", histories.get("time", []), histories.get("hand_velocity", []), self.blue)
        top += graph_h + 12
        self._draw_graph(draw, (x, top, x + panel_w, top + graph_h), "Pospešek skozi čas", histories.get("time", []), histories.get("hand_acceleration", []), self.yellow)

    def _draw_graph(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[int, int, int, int],
        title: str,
        times: list[float],
        values: list[float],
        color: tuple[int, int, int],
    ) -> None:
        self._draw_panel(draw, xy, title)
        x1, y1, x2, y2 = xy
        plot = (x1 + 16, y1 + 42, x2 - 14, y2 - 20)
        draw.rectangle(plot, outline=(67, 72, 80), width=1)
        if len(values) < 2:
            return
        window = float(self.config["graph_history_seconds"])
        t_end = times[-1] if times else len(values) / max(self.fps, 1.0)
        pairs = [(t, v) for t, v in zip(times, values) if t >= t_end - window]
        if len(pairs) < 2:
            return
        t_values = np.array([p[0] for p in pairs], dtype=np.float32)
        y_values = np.array([p[1] for p in pairs], dtype=np.float32)
        finite = np.isfinite(y_values)
        if not np.any(finite):
            return
        y_values = y_values[finite]
        t_values = t_values[finite]
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        if math.isclose(y_min, y_max):
            y_max = y_min + 1.0
        points = []
        for t, v in zip(t_values, y_values):
            px = plot[0] + (float(t - t_values[0]) / max(float(t_values[-1] - t_values[0]), 1e-6)) * (plot[2] - plot[0])
            py = plot[3] - (float(v - y_min) / max(y_max - y_min, 1e-6)) * (plot[3] - plot[1])
            points.append((px, py))
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        draw.text((plot[0], plot[3] + 4), format_float(y_min, 2), fill=self.muted, font=self.fonts["graph"])
        draw.text((plot[2] - 48, plot[3] + 4), format_float(y_max, 2), fill=self.muted, font=self.fonts["graph"])
