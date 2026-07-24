"""Run the NeuraRoads ADAS pipeline on a dashcam video file.

Produces a fully annotated output video (detection, tracking, distance, speed,
TTC, lanes, BEV and ADAS warnings) and a per-frame performance log.

Usage::

    python src/inference/inference_video.py --source data/raw/videos/dashcam_samples/drive.mp4
    python src/inference/inference_video.py --source clip.mp4 --output out.mp4 --no-preview
    python src/inference/inference_video.py --source clip.mp4 --jetson       # Jetson overlay
    python src/inference/inference_video.py --source clip.mp4 --no-detector  # before training
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.realtime_pipeline import RealtimePipeline
from utils.config_loader import resolve_path
from utils.logger import get_logger

log = get_logger(__name__)


def default_output_for(source: str) -> Path:
    """Derive a default annotated-output path from a source video path."""
    stem = Path(source).stem
    return resolve_path("src/results/videos/output_videos") / f"{stem}_annotated.mp4"


def main() -> int:
    """CLI entry: annotate a single video end to end."""
    parser = argparse.ArgumentParser(description="NeuraRoads ADAS - video inference.")
    parser.add_argument("--source", type=str, required=True, help="Path to the input video.")
    parser.add_argument("--output", type=str, default=None, help="Annotated output path.")
    parser.add_argument("--no-preview", action="store_true", help="Disable the live window.")
    parser.add_argument("--preview", action="store_true", help="Force-enable the live window.")
    parser.add_argument("--max-frames", type=int, default=None, help="Process at most N frames.")
    parser.add_argument("--jetson", action="store_true", help="Apply jetson_config overlay.")
    parser.add_argument("--no-detector", action="store_true",
                        help="Run without object detection (before the model is trained).")
    args = parser.parse_args()

    source = str(resolve_path(args.source))
    if not Path(source).is_file():
        log.error("Source video not found: {}", source)
        return 1

    output = str(resolve_path(args.output)) if args.output else str(default_output_for(args.source))
    show_preview = True if args.preview else (False if args.no_preview else None)

    rt = RealtimePipeline(
        overlay="jetson_config" if args.jetson else None,
        allow_no_detector=args.no_detector,
    )
    log.info("Processing {} -> {}", source, output)
    try:
        summary = rt.run(source=source, output_path=output, show_preview=show_preview,
                         max_frames=args.max_frames)
    except Exception as exc:
        log.error("Inference failed: {}", exc)
        return 1

    log.info("Done. Avg FPS: {:.1f} over {} frames.", summary.get("avg_fps", 0.0),
             int(summary.get("frames", 0)))
    log.info("Annotated video: {}", summary.get("output", output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
