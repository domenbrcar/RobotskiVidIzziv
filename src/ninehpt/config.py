# Centralna konfiguracija za 9HPT obdelavo.

KNOWN_HOLE_SPACING_MM = 32.0

EMPTY = "EMPTY"
OCCUPIED = "OCCUPIED"
OCCLUDED = "OCCLUDED"
UNKNOWN = "UNKNOWN"

DEFAULT_CONFIG = {
    # Znan razmik med sosednjimi luknjami za pretvorbo pikslov v metre.
    "known_hole_spacing_mm": KNOWN_HOLE_SPACING_MM,
    # Korak vzorčenja sličic pri osnovni kalibraciji mrež.
    "calibration_stride_frames": 10,
    # Najdaljši začetni del videa, ki ga uporabimo za kalibracijo.
    "calibration_max_seconds": 45.0,
    # Redkejši korak pri rezervnem pregledu celega videa.
    "calibration_fallback_stride_frames": 20,
    # Minimalen čas, ko mora ciljna mreža ostati osvetljena.
    "target_stays_on_seconds": 1.0,
    # Spodnja meja svetlobnega praga pri izbiri ciljne mreže.
    "target_light_threshold_min": 0.35,
    # Zgornja meja svetlobnega praga pri izbiri ciljne mreže.
    "target_light_threshold_max": 0.78,
    # Najmanj zaznav iste mreže, da jo sprejmemo kot stabilno.
    "min_grid_detections": 3,
    # Največ kandidatnih točk pri iskanju 3x3 mreže.
    "max_candidate_points": 42,
    # Najmanjša dovoljena površina kandidatne luknje glede na sliko.
    "candidate_min_area_ratio": 0.000035,
    # Največja dovoljena površina kandidatne luknje glede na sliko.
    "candidate_max_area_ratio": 0.009,
    # Najmanjši dovoljen razmik med luknjami glede na diagonalo slike.
    "candidate_min_spacing_ratio": 0.020,
    # Največji dovoljen razmik med luknjami glede na diagonalo slike.
    "candidate_max_spacing_ratio": 0.26,
    # Toleranca pri sestavljanju pravilne 3x3 geometrije.
    "candidate_match_tolerance_ratio": 0.46,
    # Najmanjši kot med osema mreže.
    "candidate_min_angle_degrees": 45.0,
    # Največji kot med osema mreže.
    "candidate_max_angle_degrees": 135.0,
    # Najnižja geometrijska zanesljivost kandidatne mreže.
    "candidate_min_confidence": 0.35,
    # Največ sosedov pri sestavljanju kandidatov mreže.
    "candidate_neighbor_limit": 8,
    # Najnižja povprečna zanesljivost sledenja mreži.
    "track_min_confidence": 0.55,
    # Največji smiseln razmik stabilne mreže glede na diagonalo slike.
    "track_max_spacing_ratio": 0.14,
    # Razdalja, pri kateri združimo zaznave iste mreže.
    "track_merge_distance_ratio": 0.08,
    # Najmanjši lokalni ROI signal za preglasitev z aktivnostjo.
    "activity_override_min_score": 0.30,
    # Potrebna razlika aktivnosti za preglasitev svetlobne izbire.
    "activity_override_margin": 0.22,
    # Dolžina začetnega preverjanja, pri kateri mreži roka dejansko dela.
    "hand_target_scan_seconds": 14.0,
    # Korak vzorčenja sličic za preverjanje aktivnosti roke.
    "hand_target_scan_stride_seconds": 0.20,
    # Najmanjša skupna aktivnost roke za preglasitev svetlobne izbire.
    "hand_target_override_min_score": 2.80,
    # Absolutna prednost druge mreže pred trenutno izbrano.
    "hand_target_override_margin": 1.10,
    # Relativna prednost druge mreže pred trenutno izbrano.
    "hand_target_override_ratio": 1.20,
    # Najmanj sličic z aktivnostjo pri mreži za zanesljivo preglasitev.
    "hand_target_min_hit_frames": 3,
    # Razširitev območja mreže pri iskanju roke in gibanja.
    "hand_target_box_expansion_factor": 2.10,
    # Velikost ROI okoli posamezne luknje glede na razmik mreže.
    "roi_radius_spacing_factor": 0.30,
    # Razširitev ROI pri preverjanju, ali je roka obiskala luknjo.
    "roi_visit_expansion_factor": 1.80,
    # Zamik po zaklepu mreže pred gradnjo prazne reference.
    "baseline_wait_seconds": 0.10,
    # Dolžina okna za gradnjo reference praznih lukenj.
    "baseline_window_seconds": 2.50,
    # Korak sličic pri vzorčenju baseline referenc.
    "baseline_stride_frames": 5,
    # Najmanj vzorcev za baseline posamezne luknje.
    "baseline_min_frames": 5,
    # Največji delež kože v baseline vzorcu, da ga še sprejmemo.
    "baseline_skin_reject_fraction": 0.22,
    # EMA faktor za glajenje vizualnih ROI signalov.
    "change_ema_alpha": 0.35,
    # Najnižji prag lokalnega signala za zasedeno luknjo.
    "min_occupied_threshold": 0.085,
    # Najnižji prag lokalnega signala za prazno luknjo.
    "min_empty_threshold": 0.085,
    # Faktor šuma baseline za prag zasedenosti.
    "occupied_threshold_std_factor": 1.25,
    # Faktor šuma baseline za prag praznosti.
    "empty_threshold_std_factor": 1.55,
    # Histereza med zasedenim in praznim stanjem.
    "threshold_hysteresis": 0.04,
    # Minimalna sprememba pred obiskom roke.
    "pre_visit_change_threshold": 0.025,
    # Minimalno izginjanje svetlega centra luknje.
    "center_disappearance_threshold": 0.035,
    # Čas stabilnega signala za potrditev zatiča.
    "occupied_confirm_seconds": 0.16,
    # Rezervni čas potrditve zatiča brez jasnega odhoda roke.
    "visual_occupied_fallback_seconds": 0.16,
    # Okno za preverjanje fizično smiselne hitrosti naraščanja štetja.
    "peg_count_jump_window_seconds": 1.0,
    # Največ novih zatičev v enem oknu; skok za 3+ je lažen.
    "max_peg_count_increase_per_window": 2,
    # Najmanjši vizualni dokaz, da novo stanje res predstavlja zatič.
    "confirmed_insert_min_score": 0.50,
    # Čas, v katerem mora biti hitra dvojna potrditev posebej zanesljiva.
    "fast_pair_window_seconds": 0.50,
    # Najmanjši signal za oba zatiča pri hitri dvojni potrditvi.
    "fast_pair_min_score": 0.83,
    # Število hkratnih novih kandidatov, ki pomeni spremembo reference.
    "reference_reset_candidate_count": 3,
    # Najmanj nepotrjenih lukenj za osvežitev reference.
    "reference_reset_min_empty_slots": 3,
    # Minimalen stabilen vizualni signal za rezervno potrditev.
    "stable_visual_min_score": 0.070,
    # Minimalna sprememba od baseline pri rezervni potrditvi.
    "stable_visual_min_baseline_change": 0.050,
    # Čas stabilnega signala za potrditev prazne luknje.
    "empty_confirm_seconds": 0.16,
    # Rezervni čas potrditve prazne luknje brez jasnega odhoda roke.
    "visual_empty_fallback_seconds": 0.42,
    # Kratek zamik po odmiku roke pred posodobitvijo ROI.
    "post_hand_exit_wait_seconds": 0.04,
    # Minimalen čas, ko mora stanje ostati nespremenjeno.
    "min_state_hold_seconds": 0.22,
    # Rob okoli mreže za zaznavo roke v celotni mreži.
    "grid_box_margin_spacing_factor": 0.15625,
    # Čas, ko odhod roke še šteje kot nedavna interakcija z mrežo.
    "grid_interaction_recent_seconds": 2.60,
    # Delež kože v mreži, pri katerem zamrznemo štetje.
    "grid_skin_freeze_fraction": 0.12,
    # Faktor vrnitve proti baseline pri odstranitvi zatiča.
    "removal_return_baseline_factor": 1.45,
    # Število ROI, ki morajo skočiti za globalni svetlobni dogodek.
    "global_light_change_roi_count": 5,
    # Minimalen skok lokalnega signala za globalni svetlobni dogodek.
    "global_light_change_score_jump": 0.18,
    # Minimalen skok svetlosti za globalni svetlobni dogodek.
    "global_light_brightness_jump": 18.0,
    # Največ rok, ki jih MediaPipe išče.
    "hand_max_num_hands": 4,
    # Kompleksnost MediaPipe modela roke.
    "hand_model_complexity": 0,
    # Pot do MediaPipe hand landmarker modela v Dockerju.
    "hand_landmarker_model_path": "/opt/models/hand_landmarker.task",
    # Najnižja zanesljivost zaznave roke.
    "hand_min_detection_confidence": 0.45,
    # Najnižja zanesljivost sledenja roki.
    "hand_min_tracking_confidence": 0.45,
    # Čas ohranitve zadnje zanesljive roke.
    "hand_hold_seconds": 1.10,
    # Čas po zaklepu mreže, preden sme sistem izbrati drugo roko.
    "locked_hand_reselect_seconds": 3.50,
    # Delovno območje okoli mreže, kjer izbrano roko še obravnavamo kot pravo.
    "locked_hand_work_zone_spacing_factor": 5.80,
    # Ožje območje, v katerem roke ne zamenjamo samo zaradi malo boljše bližine.
    "locked_hand_keep_zone_spacing_factor": 3.20,
    # Koliko bližje mreži mora biti druga roka za varen preklop.
    "locked_hand_grid_switch_margin_spacing": 1.15,
    # Zaporedne sličice, potrebne za preklop na roko, ki je bližje mreži.
    "locked_hand_grid_switch_confirm_frames": 8,
    # Največja razdalja stabilnih točk dlani, da gre še za isto roko.
    "hand_identity_max_spacing_factor": 3.40,
    # Prednost nove roke, potrebna za preklop.
    "hand_switch_margin": 0.18,
    # Zaporedne sličice, potrebne za potrjen preklop roke.
    "hand_switch_confirm_frames": 6,
    # Največji dovoljen skok roke glede na diagonalo slike.
    "hand_max_jump_ratio": 0.28,
    # EMA faktor za glajenje kinematike.
    "kinematic_smoothing_alpha": 0.45,
    # Največja vrzel med sličicami za zvezno kinematiko.
    "kinematic_max_gap_seconds": 0.24,
    # Največja dovoljena hitrost roke.
    "kinematic_max_speed_mps": 3.5,
    # Največji dovoljen pospešek roke.
    "kinematic_max_acc_mps2": 80.0,
    # EMA faktor za glajenje grafov v GUI.
    "gui_smoothing_alpha": 0.32,
    # Širina izhodnega GUI videa.
    "gui_width": 1280,
    # Višina izhodnega GUI videa.
    "gui_height": 720,
    # Nadomestni FPS, če ga video ne poda.
    "gui_fps_fallback": 25.0,
    # Os slike, po kateri določimo levo/desno stran pacienta.
    "patient_side_axis_by_camera": {"P0": "y", "P1": "y", "P2": "y"},
    # Obrat strani po kameri; spodnja mreža na sliki je pacientova leva.
    "patient_side_flip_by_camera": {"P0": True, "P1": True, "P2": True},
    # Dolžina zgodovine grafov v sekundah.
    "graph_history_seconds": 12.0,
    # Kodek izhodnega MP4 videa.
    "video_codec": "mp4v",
}
