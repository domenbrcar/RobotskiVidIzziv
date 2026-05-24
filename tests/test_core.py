import unittest

import cv2
import numpy as np

from src.ninehpt.config import DEFAULT_CONFIG, EMPTY, OCCUPIED
from src.ninehpt.grid_detection import compute_grid_spacing_px, detect_candidate_3x3_grids, make_rois_from_centers
from src.ninehpt.hand_tracking import select_patient_hand
from src.ninehpt.models import DetectedHand, HandSelectionState
from src.ninehpt.occupancy import PegOccupancyTracker, compute_roi_change_score, initialize_roi_baseline
from src.ninehpt.utils import parse_video_identity


def _synthetic_grid_frame() -> tuple[np.ndarray, np.ndarray]:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    centers = []
    start = np.array([230, 150], dtype=np.float32)
    u = np.array([56, 4], dtype=np.float32)
    v = np.array([-5, 55], dtype=np.float32)
    for row in range(3):
        for col in range(3):
            point = start + col * u + row * v
            centers.append(point)
            cv2.circle(frame, tuple(np.round(point).astype(int)), 12, (238, 238, 238), -1)
    return frame, np.asarray(centers, dtype=np.float32)


def _make_hand(palm: tuple[float, float]) -> DetectedHand:
    points = np.zeros((21, 2), dtype=np.float32)
    points[:] = np.array(palm, dtype=np.float32)
    points[0] = np.array([palm[0] - 18, palm[1] + 25])
    points[4] = np.array([palm[0] + 5, palm[1] - 10])
    points[8] = np.array([palm[0] + 12, palm[1] - 12])
    points[5] = np.array([palm[0] - 4, palm[1] - 6])
    points[9] = np.array([palm[0], palm[1]])
    points[13] = np.array([palm[0] + 5, palm[1] + 5])
    points[17] = np.array([palm[0] + 8, palm[1] + 12])
    return DetectedHand(points, 0.90, "Right")


class CorePipelineTests(unittest.TestCase):
    def test_video_identity(self):
        video_id, patient_id, camera_id = parse_video_identity("data/patient_003/patient_003camP_1_20231005_14_13_03.mp4")
        self.assertEqual(video_id, "patient_003camP_1_20231005_14_13_03")
        self.assertEqual(patient_id, "003")
        self.assertEqual(camera_id, "P1")

    def test_detect_synthetic_grid(self):
        frame, expected_centers = _synthetic_grid_frame()
        candidates = detect_candidate_3x3_grids(frame, DEFAULT_CONFIG)
        self.assertTrue(candidates)
        best = candidates[0]
        self.assertGreater(best.confidence, 0.35)
        self.assertAlmostEqual(compute_grid_spacing_px(best.centers), compute_grid_spacing_px(expected_centers), delta=8.0)

    def test_roi_change_score_uses_visual_object_not_empty_light(self):
        frame, centers = _synthetic_grid_frame()
        rois = make_rois_from_centers(centers, frame.shape, DEFAULT_CONFIG)
        baselines = initialize_roi_baseline([frame] * 6, rois, DEFAULT_CONFIG)
        empty_score = compute_roi_change_score(frame, rois[0], baselines[0])
        occupied = frame.copy()
        cv2.circle(occupied, tuple(np.round(centers[0]).astype(int)), 11, (90, 40, 18), -1)
        occupied_score = compute_roi_change_score(occupied, rois[0], baselines[0])
        self.assertLess(float(empty_score["post_insert_score"]), 0.08)
        self.assertGreater(float(occupied_score["post_insert_score"]), float(empty_score["post_insert_score"]) + 0.08)
        self.assertGreater(float(occupied_score["center_disappearance"]), 0.03)

    def test_occupancy_is_current_state_without_forced_completion(self):
        config = dict(DEFAULT_CONFIG)
        config["occupied_confirm_seconds"] = 0.08
        config["visual_occupied_fallback_seconds"] = 0.08
        config["empty_confirm_seconds"] = 0.08
        config["visual_empty_fallback_seconds"] = 0.08
        frame, centers = _synthetic_grid_frame()
        rois = make_rois_from_centers(centers, frame.shape, config)
        baselines = initialize_roi_baseline([frame] * 8, rois, config)
        tracker = PegOccupancyTracker(rois, baselines, centers, 25.0, config)
        result = None
        for idx in range(8):
            result = tracker.update(frame, idx, idx / 25.0, None, True)
        self.assertEqual(result["peg_count"], 0)

        one_peg = frame.copy()
        cv2.circle(one_peg, tuple(np.round(centers[4]).astype(int)), 11, (80, 35, 18), -1)
        for idx in range(8, 18):
            result = tracker.update(one_peg, idx, idx / 25.0, None, True)
        self.assertEqual(result["peg_count"], 1)
        self.assertEqual(result["roi_states"][4], OCCUPIED)
        self.assertNotEqual(result["peg_count"], 9)

        for idx in range(18, 30):
            result = tracker.update(frame, idx, idx / 25.0, None, True)
        self.assertEqual(result["peg_count"], 0)
        self.assertTrue(all(state == EMPTY for state in result["roi_states"]))

    def test_hand_selection_requires_confirmed_switch(self):
        frame_shape = (480, 640, 3)
        _, centers = _synthetic_grid_frame()
        state = HandSelectionState()
        first = _make_hand((300, 230))
        selected, state = select_patient_hand([first], state, 0, 25.0, centers, frame_shape, DEFAULT_CONFIG)
        self.assertIs(selected, first)
        intruder = _make_hand((120, 110))
        selected, state = select_patient_hand([intruder], state, 1, 25.0, centers, frame_shape, DEFAULT_CONFIG)
        self.assertIsNotNone(selected)
        self.assertIn(state.selected_hand_source, {"held_last_good", "continuous_detection"})


if __name__ == "__main__":
    unittest.main()
