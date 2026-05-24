"""Central configuration for the 9HPT analysis pipeline."""

KNOWN_HOLE_SPACING_MM = 32.0

EMPTY = "EMPTY"
OCCUPIED = "OCCUPIED"
OCCLUDED = "OCCLUDED"
UNKNOWN = "UNKNOWN"

DEFAULT_CONFIG = {
    "known_hole_spacing_mm": KNOWN_HOLE_SPACING_MM, # Razmik med luknjami za pretvorbo v metre.
    "calibration_stride_frames": 10, # Korak vzorčenja sličic pri kalibraciji.
    "calibration_max_seconds": 45.0, # Največji čas videa za začetno kalibracijo.
    "calibration_fallback_stride_frames": 20, # Redkejši korak pri rezervnem pregledu celega videa.
    "target_stays_on_seconds": 1.0, # Čas potrditve mreže, ki ostane osvetljena.
    "target_light_threshold_min": 0.35,
    "target_light_threshold_max": 0.78,
    "min_grid_detections": 3,
    "max_candidate_points": 42,
    "candidate_min_area_ratio": 0.000035,
    "candidate_max_area_ratio": 0.009,
    "candidate_min_spacing_ratio": 0.020,
    "candidate_max_spacing_ratio": 0.26,
    "candidate_match_tolerance_ratio": 0.46,
    "candidate_min_angle_degrees": 45.0,
    "candidate_max_angle_degrees": 135.0,
    "candidate_min_confidence": 0.35,
    "candidate_neighbor_limit": 8,
    "track_min_confidence": 0.55, # Zavrnitev naključnih mrež iz besedila, refleksij ali posode.
    "track_max_spacing_ratio": 0.14, # Največji smiseln razmik mreže glede na diagonalo slike.
    "track_merge_distance_ratio": 0.08,
    "activity_override_min_score": 0.30,
    "activity_override_margin": 0.22,
    "roi_radius_spacing_factor": 0.30, # Velikost območja okoli posamezne luknje.
    "roi_visit_expansion_factor": 1.80,
    "roi_near_expansion_factor": 3.20,
    "baseline_wait_seconds": 0.10, # Zamik po zaklepu mreže pred gradnjo prazne reference.
    "baseline_window_seconds": 2.50, # Dolžina okna za referenco praznih lukenj.
    "baseline_stride_frames": 5,
    "baseline_min_frames": 5,
    "baseline_skin_reject_fraction": 0.22,
    "change_ema_alpha": 0.35, # EMA za stabilizacijo vizualnih ROI signalov.
    "min_occupied_threshold": 0.085, # Najmanjši lokalni signal za zasedeno luknjo.
    "min_empty_threshold": 0.085,
    "occupied_threshold_std_factor": 1.25,
    "empty_threshold_std_factor": 1.55,
    "threshold_hysteresis": 0.04,
    "pre_visit_change_threshold": 0.025,
    "center_disappearance_threshold": 0.035,
    "occupied_confirm_seconds": 0.16, # Čas stabilnega signala za potrditev zatiča.
    "visual_occupied_fallback_seconds": 0.16,
    "stable_visual_min_score": 0.070,
    "stable_visual_min_baseline_change": 0.050,
    "stable_visual_global_reject_count": 5,
    "empty_confirm_seconds": 0.16, # Čas stabilnega signala za potrditev prazne luknje.
    "visual_empty_fallback_seconds": 0.42,
    "post_hand_exit_wait_seconds": 0.04, # Kratek zamik po odmiku roke pred posodobitvijo ROI.
    "min_state_hold_seconds": 0.22,
    "grid_box_margin_spacing_factor": 0.15625,
    "grid_interaction_recent_seconds": 2.60,
    "grid_skin_freeze_fraction": 0.12, # Delež kože v mreži, pri katerem zamrznemo štetje.
    "removal_return_baseline_factor": 1.45,
    "removal_prechange_min_factor": 0.35,
    "global_light_change_roi_count": 5,
    "global_light_change_score_jump": 0.18,
    "global_light_brightness_jump": 18.0,
    "hand_max_num_hands": 4,
    "hand_detection_stride_frames": 1,
    "hand_model_complexity": 0,
    "hand_landmarker_model_path": "/opt/models/hand_landmarker.task",
    "hand_min_detection_confidence": 0.45,
    "hand_min_tracking_confidence": 0.45,
    "hand_hold_seconds": 1.10, # Čas ohranitve zadnje zanesljive roke.
    "hand_switch_margin": 0.18,
    "hand_switch_confirm_frames": 6, # Zaporedne sličice za zanesljiv preklop roke.
    "hand_max_jump_ratio": 0.28,
    "kinematic_smoothing_alpha": 0.45,
    "kinematic_max_gap_seconds": 0.24,
    "kinematic_max_speed_mps": 3.5,
    "kinematic_max_acc_mps2": 80.0,
    "gui_smoothing_alpha": 0.32, # Blago glajenje grafov samo za prikaz.
    "gui_width": 1280,
    "gui_height": 720,
    "gui_fps_fallback": 25.0,
    "graph_history_seconds": 12.0,
    "video_codec": "mp4v",
}
