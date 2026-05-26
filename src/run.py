# Ukazni zagon končne 9HPT obdelave.

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ninehpt.config import DEFAULT_CONFIG
from src.ninehpt.pipeline import process_video
from src.ninehpt.utils import find_video_files, parse_video_identity


VIDEO_TIME_RE = re.compile(r"_(\d{8})_(\d{2})_(\d{2})_(\d{2})$")


def _video_chronological_key(video_path: str) -> tuple[str, str]:
    stem = Path(video_path).stem
    match = VIDEO_TIME_RE.search(stem)
    # Časovni ključ iz imena; če ga ni, ostane stabilno sortiranje po imenu.
    timestamp = "".join(match.groups()) if match else stem
    return timestamp, stem


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
            current = selected.get(patient_id)
            if current is None or _video_chronological_key(video) < _video_chronological_key(current):
                selected[patient_id] = video
        videos = [selected[key] for key in sorted(selected)]
    if args.process_n:
        videos = videos[: int(args.process_n)]
    return videos


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
    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] Obdelujem: {video}", flush=True)
        try:
            process_video(video, args.output, DEFAULT_CONFIG)
        except Exception as exc:
            print(f"  -> NAPAKA: {exc}", flush=True)
    print(f"Končano. Izhodi so v: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
