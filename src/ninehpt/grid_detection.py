"""Candidate 3x3 grid detection for 9HPT calibration."""

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np

from .models import GridCandidate


def _deduplicate_points(points: list[tuple[float, float]], min_distance: float) -> np.ndarray:
    unique: list[np.ndarray] = []
    for point in sorted(points, key=lambda item: (item[1], item[0])):
        p = np.array(point, dtype=np.float32)
        if all(np.linalg.norm(p - q) >= min_distance for q in unique):
            unique.append(p)
    return np.array(unique, dtype=np.float32) if unique else np.empty((0, 2), dtype=np.float32)


def _contour_points(frame: np.ndarray, config: dict) -> list[tuple[float, float]]:
    height, width = frame.shape[:2]
    area = float(height * width)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bright_threshold = max(135, int(np.percentile(blur, 94)))
    dark_threshold = min(95, int(np.percentile(blur, 9)))
    masks = []
    _, bright = cv2.threshold(blur, bright_threshold, 255, cv2.THRESH_BINARY)
    _, dark = cv2.threshold(blur, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    masks.extend([bright, dark])
    points: list[tuple[float, float]] = []
    min_area = config["candidate_min_area_ratio"] * area
    max_area = config["candidate_max_area_ratio"] * area
    kernel = np.ones((3, 3), np.uint8)
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < min_area or contour_area > max_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * math.pi * contour_area / (perimeter * perimeter)
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            if circularity < 0.35 or not 0.45 <= aspect <= 2.15:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            if 2 <= cx <= width - 3 and 2 <= cy <= height - 3:
                points.append((cx, cy))
    return points


def _hough_points(frame: np.ndarray, config: dict) -> list[tuple[float, float]]:
    height, width = frame.shape[:2]
    min_radius = max(3, int(min(width, height) * 0.007))
    max_radius = max(min_radius + 1, int(min(width, height) * 0.045))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(8, int(min(width, height) * 0.025)),
        param1=80,
        param2=16,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []
    return [(float(x), float(y)) for x, y, _ in np.squeeze(circles, axis=0)]


def _match_lattice(points: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray, tolerance: float) -> tuple[np.ndarray | None, float]:
    matched = []
    used: set[int] = set()
    errors = []
    for row in range(3):
        for col in range(3):
            predicted = origin + col * u + row * v
            distances = np.linalg.norm(points - predicted, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] > tolerance or nearest in used:
                return None, float("inf")
            used.add(nearest)
            matched.append(points[nearest])
            errors.append(float(distances[nearest]))
    return np.array(matched, dtype=np.float32).reshape(3, 3, 2), float(np.mean(errors))


def _canonical_grid_order(centers: np.ndarray) -> np.ndarray:
    ordered_by_y = np.asarray(centers, dtype=np.float32)[np.argsort(np.asarray(centers)[:, 1])]
    rows = []
    for row_index in range(3):
        row = ordered_by_y[row_index * 3 : (row_index + 1) * 3]
        rows.append(row[np.argsort(row[:, 0])])
    return np.stack(rows, axis=0).astype(np.float32)


def _candidate_from_lattice(grid: np.ndarray, mean_error: float, tolerance: float, source_frame: int) -> GridCandidate | None:
    grid = _canonical_grid_order(grid.reshape(9, 2))
    horizontal = [np.linalg.norm(grid[r, c + 1] - grid[r, c]) for r in range(3) for c in range(2)]
    vertical = [np.linalg.norm(grid[r + 1, c] - grid[r, c]) for r in range(2) for c in range(3)]
    distances = np.array(horizontal + vertical, dtype=np.float32)
    spacing = float(np.median(distances))
    if spacing <= 1:
        return None
    consistency = float(np.std(distances) / spacing)
    if consistency > 0.36:
        return None
    confidence = max(0.0, 1.0 - mean_error / max(tolerance, 1e-6)) * max(0.0, 1.0 - consistency)
    centers = grid.reshape(9, 2)
    return GridCandidate(centers=centers, confidence=float(confidence), spacing_px=spacing, center=np.mean(centers, axis=0), source_frame=source_frame)


def _deduplicate_candidates(candidates: list[GridCandidate]) -> list[GridCandidate]:
    result: list[GridCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for existing in result:
            if np.linalg.norm(candidate.center - existing.center) < 0.45 * existing.spacing_px and abs(candidate.spacing_px - existing.spacing_px) < 0.35 * existing.spacing_px:
                duplicate = True
                break
        if not duplicate:
            result.append(candidate)
    return result


def _detect_grids_from_points(points: list[tuple[float, float]], frame_shape: tuple[int, int], config: dict) -> list[GridCandidate]:
    height, width = frame_shape
    diagonal = math.hypot(width, height)
    points_np = _deduplicate_points(points, max(4.0, diagonal * 0.008))
    if len(points_np) < 9:
        return []
    if len(points_np) > config["max_candidate_points"]:
        frame_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        points_np = points_np[np.argsort(np.linalg.norm(points_np - frame_center, axis=1))[: config["max_candidate_points"]]]
    min_spacing = diagonal * config["candidate_min_spacing_ratio"]
    max_spacing = diagonal * config["candidate_max_spacing_ratio"]
    angle_min = math.radians(config["candidate_min_angle_degrees"])
    angle_max = math.radians(config["candidate_max_angle_degrees"])
    candidates: list[GridCandidate] = []
    for origin_index, origin in enumerate(points_np):
        distances = np.linalg.norm(points_np - origin, axis=1)
        neighbor_indices = [int(idx) for idx in np.argsort(distances) if idx != origin_index and min_spacing <= distances[idx] <= max_spacing][: int(config["candidate_neighbor_limit"])]
        for neighbor_u, neighbor_v in itertools.permutations(neighbor_indices, 2):
            u = points_np[neighbor_u] - origin
            v = points_np[neighbor_v] - origin
            len_u = float(np.linalg.norm(u))
            len_v = float(np.linalg.norm(v))
            if not 0.58 <= len_u / max(len_v, 1e-6) <= 1.72:
                continue
            cosine = float(np.dot(u, v) / max(len_u * len_v, 1e-6))
            angle = math.acos(max(-1.0, min(1.0, cosine)))
            if not angle_min <= angle <= angle_max:
                continue
            tolerance = max(5.0, min(len_u, len_v) * config["candidate_match_tolerance_ratio"])
            grid, mean_error = _match_lattice(points_np, origin, u, v, tolerance)
            if grid is None:
                continue
            candidate = _candidate_from_lattice(grid, mean_error, tolerance, source_frame=-1)
            if candidate and candidate.confidence >= config["candidate_min_confidence"]:
                candidates.append(candidate)
    return _deduplicate_candidates(candidates)


def detect_candidate_3x3_grids(frame: np.ndarray, config: dict) -> list[GridCandidate]:
    contour_points = _contour_points(frame, config)
    contour_candidates = _detect_grids_from_points(contour_points, frame.shape[:2], config)
    if contour_candidates:
        return contour_candidates
    return _detect_grids_from_points(contour_points + _hough_points(frame, config), frame.shape[:2], config)


def compute_grid_spacing_px(centers: np.ndarray) -> float:
    grid = np.asarray(centers, dtype=np.float32).reshape(3, 3, 2)
    distances = []
    for row in range(3):
        for col in range(2):
            distances.append(np.linalg.norm(grid[row, col + 1] - grid[row, col]))
    for row in range(2):
        for col in range(3):
            distances.append(np.linalg.norm(grid[row + 1, col] - grid[row, col]))
    valid = [float(value) for value in distances if value > 1.0]
    return float(np.median(valid)) if valid else 0.0


def make_rois_from_centers(centers: np.ndarray, frame_shape: tuple[int, int, int] | tuple[int, int], config: dict) -> list[tuple[int, int, int, int]]:
    height, width = frame_shape[:2]
    spacing = compute_grid_spacing_px(centers)
    radius = max(6, int(round(spacing * config["roi_radius_spacing_factor"])))
    rois = []
    for cx, cy in np.asarray(centers, dtype=np.float32).reshape(9, 2):
        rois.append((
            max(0, int(round(cx - radius))),
            max(0, int(round(cy - radius))),
            min(width, int(round(cx + radius))),
            min(height, int(round(cy + radius))),
        ))
    return rois
