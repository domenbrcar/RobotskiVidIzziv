"""Command-line entry point for the final 9HPT analysis pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ninehpt.config import DEFAULT_CONFIG
from src.ninehpt.pipeline import process_video, write_processing_report
from src.ninehpt.utils import find_video_files, parse_video_identity


def _collect_videos(args: argparse.Namespace) -> list[str]:
    if args.video:
        videos = find_video_files(args.video)
    elif args.patient:
        videos = find_video_files(args.patient)
    elif args.folder:
        videos = find_video_files(args.folder)
    else:
        raise SystemExit("Podaj -video, -patient ali -folder.")
    if args.one_per_patient:
        selected: dict[str, str] = {}
        for video in videos:
            _, patient_id, _ = parse_video_identity(video)
            selected.setdefault(patient_id, video)
        videos = [selected[key] for key in sorted(selected)]
    if args.process_n:
        videos = videos[: int(args.process_n)]
    return videos


def _evaluation_notes(row: dict) -> str:
    max_count = int(row.get("max_peg_count", 0))
    final_count = int(row.get("final_peg_count", 0))
    if max_count == 9 and final_count == 0:
        return "full_cycle_observed"
    if max_count == 9:
        return "full_insertion_partial_removal"
    if max_count > 0:
        return "partial_cycle_or_limited_view"
    return "no_clear_peg_cycle"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analiza 9HPT videoposnetkov.")
    parser.add_argument("-folder", help="Mapa z mapami patient_XXX.", default=None)
    parser.add_argument("-patient", help="Mapa enega pacienta.", default=None)
    parser.add_argument("-video", help="Pot do enega videoposnetka.", default=None)
    parser.add_argument("-output", help="Izhodna mapa.", default="output/final")
    parser.add_argument("--process-n", type=int, default=None, help="Omeji število obdelanih videov.")
    parser.add_argument("--one-per-patient", action="store_true", help="Obdelaj največ prvi video na pacienta.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    videos = _collect_videos(args)
    if not videos:
        print("Ni najdenih videoposnetkov.")
        return 1
    rows = []
    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] Obdelujem: {video}", flush=True)
        try:
            result = process_video(video, args.output, DEFAULT_CONFIG)
            row = {
                "patient_id": result.get("patient_id", ""),
                "video_id": result.get("video_id", ""),
                "target_locked": result.get("target_locked", 0),
                "target_reason": result.get("target_reason", ""),
                "max_peg_count": result.get("max_peg_count", 0),
                "final_peg_count": result.get("final_peg_count", 0),
                "reached_9": result.get("reached_9", 0),
                "hand_mean_confidence": result.get("hand_mean_confidence", ""),
                "notes": _evaluation_notes(result),
            }
            rows.append(row)
            print(
                f"  -> mreža={row['target_locked']} max={row['max_peg_count']} final={row['final_peg_count']} "
                f"reason={row['target_reason']}",
                flush=True,
            )
        except Exception as exc:
            video_id, patient_id, _ = parse_video_identity(video)
            rows.append({
                "patient_id": patient_id,
                "video_id": video_id,
                "target_locked": 0,
                "target_reason": "processing_error",
                "max_peg_count": 0,
                "final_peg_count": 0,
                "reached_9": 0,
                "hand_mean_confidence": "",
                "notes": str(exc),
            })
            print(f"  -> NAPAKA: {exc}", flush=True)
    report_path = write_processing_report(args.output, rows)
    print(f"Pregled shranjen: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
