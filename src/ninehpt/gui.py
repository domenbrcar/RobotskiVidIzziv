# Slovenski prikazovalnik nadzorne plošče za končne 9HPT videe.

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import OCCUPIED
from .models import VideoMetadata
from .utils import format_float


class DashboardRenderer:
    """Izriše urejeno nadzorno ploščo okoli obdelanega videa."""

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
        self.blue = (92, 157, 230)
        self.green = self.blue
        self.red = (248, 250, 252)
        self.yellow = (230, 188, 88)
        self.teal = (78, 196, 178)
        self.fonts = self._load_fonts()
        self._success_delay_s = 0.5
        self._full_grid_since_s: float | None = None
        self._seen_unfilled_grid = False
        self._success_visible = False

    @staticmethod
    def _load_fonts() -> dict[str, ImageFont.ImageFont]:
        candidates = [
            "DejaVuSans.ttf",
            "Arial.ttf",
            "LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
        bold_candidates = [
            "DejaVuSans-Bold.ttf",
            "Arialbd.ttf",
            "LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]

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
            "body": load(14),
            "small": load(11),
            "value": load(16, True),
            "success_count": load(23, True),
            "success_body": load(19, True),
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
        target_grid_side: str = "right",
    ) -> np.ndarray:
        image = Image.new("RGB", (self.width, self.height), self.bg)
        draw = ImageDraw.Draw(image)
        self._draw_left(draw, metadata, frame_index, time_s, kinematics)
        self._draw_video(image, frame_bgr)
        self._draw_progress(draw, roi_states, target_grid_side, peg_count, time_s)
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
        kinematics: dict[str, dict[str, float]],
    ) -> None:
        x = 14
        draw.text((x, 16), "Analiza 9HPT", fill=self.text, font=self.fonts["title"])
        top = 52
        self._draw_panel(draw, (x, top, self.left_w - 12, 224), "Podatki testa")
        y = top + 44
        patient_label = str(int(metadata.patient_id)) if str(metadata.patient_id).isdigit() else str(metadata.patient_id)
        y = self._draw_kv(draw, x + 14, y, "Pacient", patient_label)
        y = self._draw_kv(draw, x + 14, y, "Kamera", metadata.camera_id)
        y = self._draw_kv(draw, x + 14, y, "Čas videa", f"{format_float(metadata.duration_s, 1)} s")

        self._draw_panel(draw, (x, 240, self.left_w - 12, 346), "Potek")
        y = 278
        y = self._draw_kv(draw, x + 14, y, "Čas", f"{time_s:0.2f} s")
        self._draw_kv(draw, x + 14, y, "Sličica", str(frame_index))

        y = 362
        self._draw_motion_block(draw, x, y, "Kinematika roke", kinematics.get("hand", {}))
        y += 108
        self._draw_motion_block(draw, x, y, "Kinematika palca", kinematics.get("thumb", {}))
        y += 108
        self._draw_motion_block(draw, x, y, "Kinematika kazalca", kinematics.get("index", {}))

    def _draw_motion_block(self, draw: ImageDraw.ImageDraw, x: int, y: int, title: str, values: dict[str, float]) -> None:
        self._draw_panel(draw, (x, y, self.left_w - 12, y + 96), title)
        draw.text((x + 14, y + 36), f"pot {format_float(values.get('path_m', 0.0), 3)} m", fill=self.text, font=self.fonts["body"])
        draw.text((x + 14, y + 58), f"hitrost {format_float(values.get('velocity_mps', 0.0), 3)} m/s", fill=self.text, font=self.fonts["body"])
        draw.text((x + 14, y + 80), f"pospešek {format_float(values.get('acceleration_mps2', 0.0), 2)} m/s²", fill=self.text, font=self.fonts["body"])

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

    def _draw_progress_grid(
        self,
        draw: ImageDraw.ImageDraw,
        center: tuple[int, int],
        roi_states: list[str],
        step: int,
        radius: int,
    ) -> None:
        center_x, center_y = center
        for idx in range(9):
            row, col = divmod(idx, 3)
            display_row = col
            display_col = 2 - row
            cx = center_x + (display_col - 1) * step
            cy = center_y + (display_row - 1) * step
            occupied = idx < len(roi_states) and roi_states[idx] == OCCUPIED
            fill = self.green if occupied else self.red
            outline = (218, 224, 231) if occupied else (255, 255, 255)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill, outline=outline, width=2)

    def _centered_text(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        y: int,
        text: str,
        font: ImageFont.ImageFont,
        fill: tuple[int, int, int],
    ) -> int:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = int(round(box[0] + (box[2] - box[0] - text_w) / 2))
        draw.text((x, y), text, fill=fill, font=font)
        return y + text_h

    def _draw_success_progress(self, draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = xy
        inner = (x1 + 24, y1 + 18, x2 - 24, y2 - 18)
        draw.rounded_rectangle(inner, radius=8, fill=(29, 40, 49), outline=self.blue, width=2)

        center_x = int(round((inner[0] + inner[2]) / 2))
        badge_y = inner[1] + 44
        badge_r = 32
        draw.ellipse(
            (center_x - badge_r, badge_y - badge_r, center_x + badge_r, badge_y + badge_r),
            fill=self.blue,
            outline=(218, 224, 231),
            width=2,
        )
        badge_text = "9/9"
        badge_bbox = draw.textbbox((0, 0), badge_text, font=self.fonts["success_count"])
        badge_x = center_x - (badge_bbox[2] - badge_bbox[0]) // 2
        badge_text_y = badge_y - (badge_bbox[3] - badge_bbox[1]) // 2 - 2
        draw.text((badge_x, badge_text_y), badge_text, fill=(255, 255, 255), font=self.fonts["success_count"])

        message = "Uspešno vnešeno in zaznano vseh devet zatičev."
        text_box = (inner[0] + 22, inner[1], inner[2] - 22, inner[3])
        self._centered_text(draw, text_box, badge_y + badge_r + 26, message, self.fonts["success_body"], self.text)

    def _draw_progress(
        self,
        draw: ImageDraw.ImageDraw,
        roi_states: list[str],
        target_grid_side: str = "right",
        peg_count: int | None = None,
        time_s: float = 0.0,
    ) -> None:
        x = self.center_x
        y = self.progress_y
        panel_xy = (x, y, x + self.center_w, self.height - 18)
        self._draw_panel(draw, panel_xy, None)
        if self._success_visible:
            self._draw_success_progress(draw, panel_xy)
            return

        filled_slots = sum(1 for state in roi_states[:9] if state == OCCUPIED)
        full_grid = len(roi_states) >= 9 and filled_slots == 9 and (peg_count is None or peg_count == 9)
        if full_grid and self._seen_unfilled_grid:
            if self._full_grid_since_s is None:
                self._full_grid_since_s = time_s
            if time_s - self._full_grid_since_s >= self._success_delay_s:
                self._success_visible = True
                self._draw_success_progress(draw, panel_xy)
                return
        else:
            self._full_grid_since_s = None

        if not full_grid:
            self._seen_unfilled_grid = True

        panel_h = self.height - 18 - y
        step = min(54, max(42, (panel_h - 36) // 3))
        radius = 16
        element_radius = step + radius
        gap = max(20, min(32, self.center_w // 22))
        element_width = element_radius * 2
        group_width = element_width * 3 + gap * 2
        group_x = int(round(x + (self.center_w - group_width) / 2))
        cy = int(round(y + panel_h / 2))
        left_center = (group_x + element_radius, cy)
        circle_center = (group_x + element_width + gap + element_radius, cy)
        right_center = (group_x + 2 * (element_width + gap) + element_radius, cy)
        empty_states = []
        left_states = roi_states if target_grid_side == "left" else empty_states
        right_states = roi_states if target_grid_side != "left" else empty_states

        self._draw_progress_grid(draw, left_center, left_states, step, radius)
        circle_radius = element_radius
        cx, cy = circle_center
        draw.ellipse(
            (cx - circle_radius, cy - circle_radius, cx + circle_radius, cy + circle_radius),
            fill=self.panel_alt,
            outline=(82, 88, 98),
            width=2,
        )
        self._draw_progress_grid(draw, right_center, right_states, step, radius)

    def _draw_right(self, draw: ImageDraw.ImageDraw, histories: dict[str, list[float]]) -> None:
        x = self.width - self.right_w + 12
        panel_w = self.right_w - 26
        top = 18
        graph_h = (self.height - 54) // 3
        self._draw_graph(
            draw,
            (x, top, x + panel_w, top + graph_h),
            "Pot skozi čas",
            histories.get("time", []),
            [
                ("Roka", histories.get("hand_path", []), self.blue),
                ("Palec", histories.get("thumb_path", []), self.yellow),
                ("Kazalec", histories.get("index_path", []), self.teal),
            ],
        )
        top += graph_h + 12
        self._draw_graph(
            draw,
            (x, top, x + panel_w, top + graph_h),
            "Hitrost skozi čas",
            histories.get("time", []),
            [
                ("Roka", histories.get("hand_velocity", []), self.blue),
                ("Palec", histories.get("thumb_velocity", []), self.yellow),
                ("Kazalec", histories.get("index_velocity", []), self.teal),
            ],
        )
        top += graph_h + 12
        self._draw_graph(
            draw,
            (x, top, x + panel_w, top + graph_h),
            "Pospešek skozi čas",
            histories.get("time", []),
            [
                ("Roka", histories.get("hand_acceleration", []), self.blue),
                ("Palec", histories.get("thumb_acceleration", []), self.yellow),
                ("Kazalec", histories.get("index_acceleration", []), self.teal),
            ],
        )

    def _draw_graph(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[int, int, int, int],
        title: str,
        times: list[float],
        series: list[tuple[str, list[float], tuple[int, int, int]]],
        y_unit: str | None = None,
    ) -> None:
        self._draw_panel(draw, xy, title)
        x1, y1, x2, y2 = xy
        if y_unit is None:
            if "Hitrost" in title:
                y_unit = "m/s"
                y_label = "hitrost"
            elif "Pospe" in title:
                y_unit = "m/s²"
                y_label = "pospešek"
            else:
                y_unit = "m"
                y_label = "pot"
        elif "Hitrost" in title:
            y_label = "hitrost"
        elif "Pospe" in title:
            y_label = "pospešek"
        else:
            y_label = "pot"
        legend_x = x1 + 14
        legend_y = y1 + 34
        for label, _, color in series:
            draw.line((legend_x, legend_y + 6, legend_x + 14, legend_y + 6), fill=color, width=3)
            draw.text((legend_x + 18, legend_y), label, fill=self.text, font=self.fonts["graph"])
            legend_x += 76

        plot = (x1 + 42, y1 + 64, x2 - 14, y2 - 38)
        draw.rectangle(plot, outline=(67, 72, 80), width=1)
        draw.text((plot[0], y1 + 49), f"{y_label} [{y_unit}]", fill=self.muted, font=self.fonts["graph"])
        x_label = "čas [s]"
        x_label_bbox = draw.textbbox((0, 0), x_label, font=self.fonts["graph"])
        x_label_w = x_label_bbox[2] - x_label_bbox[0]
        draw.text((plot[0] + (plot[2] - plot[0] - x_label_w) / 2, plot[3] + 22), x_label, fill=self.muted, font=self.fonts["graph"])
        if len(times) < 2:
            return
        window = float(self.config["graph_history_seconds"])
        t_end = times[-1]
        visible_times = np.array([t for t in times if t >= t_end - window], dtype=np.float32)
        if len(visible_times) < 2:
            return
        t_min = float(visible_times[0])
        t_max = float(visible_times[-1])
        prepared_series: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]] = []
        all_y_values: list[float] = []
        for _, values, color in series:
            pairs = [(t, v) for t, v in zip(times, values) if t_min <= t <= t_max and math.isfinite(float(v))]
            if not pairs:
                continue
            t_values = np.array([p[0] for p in pairs], dtype=np.float32)
            y_values = np.array([p[1] for p in pairs], dtype=np.float32)
            finite = np.isfinite(y_values)
            if not np.any(finite):
                continue
            t_values = t_values[finite]
            y_values = y_values[finite]
            all_y_values.extend(float(v) for v in y_values)
            if len(t_values) >= 2:
                prepared_series.append((t_values, y_values, color))
        if not all_y_values:
            return
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        if math.isclose(y_min, y_max):
            y_max = y_min + 1.0
        for t_values, y_values, color in prepared_series:
            points = []
            for t, v in zip(t_values, y_values):
                px = plot[0] + (float(t - t_min) / max(t_max - t_min, 1e-6)) * (plot[2] - plot[0])
                py = plot[3] - (float(v - y_min) / max(y_max - y_min, 1e-6)) * (plot[3] - plot[1])
                points.append((px, py))
            if len(points) >= 2:
                draw.line(points, fill=color, width=2)
        draw.text((plot[0] - 34, plot[3] - 8), format_float(y_min, 2), fill=self.muted, font=self.fonts["graph"])
        draw.text((plot[0] - 34, plot[1] - 4), format_float(y_max, 2), fill=self.muted, font=self.fonts["graph"])
        draw.text((plot[0], plot[3] + 5), format_float(t_min, 1), fill=self.muted, font=self.fonts["graph"])
        draw.text((plot[2] - 34, plot[3] + 5), format_float(t_max, 1), fill=self.muted, font=self.fonts["graph"])
