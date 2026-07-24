"""Jetson Nano inference entry point (applies the jetson_config overlay).

Same pipeline as the desktop, but with the Nano-tuned overlay merged in
(YOLOv8n + TensorRT INT8, smaller input, classical lanes, reduced cadence). Runs
headless by default and writes an annotated MP4.

Usage (on the Jetson):

    python3 src/deployment/jetson/jetson_inference.py --source clip.mp4
    python3 src/deployment/jetson/jetson_inference.py --source /dev/video0 --preview
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# `src` root is two levels up: src/deployment/jetson/ -> src
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.realtime_pipeline import RealtimePipeline
from utils.config_loader import resolve_path
from utils.logger import get_logger

log = get_logger(__name__)


def main() -> int:
    """CLI entry: run the Nano-optimised ADAS pipeline."""
    parser = argparse.ArgumentParser(description="NeuraRoads ADAS - Jetson Nano inference.")
    parser.add_argument("--source", type=str, required=True, help="Video path or camera index.")
    parser.add_argument("--output", type=str, default=None, help="Annotated output path.")
    parser.add_argument("--preview", action="store_true", help="Show a preview window.")
    parser.add_argument("--no-detector", action="store_true", help="Run without detection.")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    # Allow a webcam index like "0".
    source: object = args.source
    if str(args.source).isdigit():
        source = int(args.source)
    elif not Path(str(resolve_path(args.source))).exists():
        log.error("Source not found: {}", args.source)
        return 1
    else:
        source = str(resolve_path(args.source))

    output = args.output
    if output is None and not isinstance(source, int):
        output = str(resolve_path("src/results/videos/output_videos") /
                     f"{Path(str(source)).stem}_jetson.mp4")

    log.info("Jetson inference (jetson_config overlay). Source={} Output={}", source, output)
    rt = RealtimePipeline(overlay="jetson_config", allow_no_detector=args.no_detector)
    try:
        summary = rt.run(source=source, output_path=output, show_preview=args.preview,
                         max_frames=args.max_frames)
    except Exception as exc:
        log.error("Jetson inference failed: {}", exc)
        return 1
    log.info("Done. Avg FPS: {:.1f}.", summary.get("avg_fps", 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
