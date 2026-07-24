"""Run the NeuraRoads ADAS pipeline live from a webcam / capture device.

Uses wall-clock frame timing (so speed/TTC reflect real motion) and shows the
annotated stream in a window. Press ``q`` or ``Esc`` to stop.

Usage::

    python src/inference/inference_webcam.py                 # default camera 0
    python src/inference/inference_webcam.py --camera 1 --record
    python src/inference/inference_webcam.py --no-detector    # before training
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.realtime_pipeline import RealtimePipeline
from utils.config_loader import resolve_path
from utils.logger import get_logger

log = get_logger(__name__)


def main() -> int:
    """CLI entry: run live webcam ADAS."""
    parser = argparse.ArgumentParser(description="NeuraRoads ADAS - webcam inference.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--record", action="store_true", help="Also save the annotated stream.")
    parser.add_argument("--output", type=str, default=None, help="Recording path (if --record).")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    parser.add_argument("--jetson", action="store_true", help="Apply jetson_config overlay.")
    parser.add_argument("--no-detector", action="store_true",
                        help="Run without object detection (before the model is trained).")
    args = parser.parse_args()

    output = None
    if args.record:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = str(resolve_path(args.output) if args.output else
                     resolve_path("src/results/videos/output_videos") / f"webcam_{ts}.mp4")

    rt = RealtimePipeline(
        overlay="jetson_config" if args.jetson else None,
        allow_no_detector=args.no_detector,
    )
    log.info("Starting webcam {} (press 'q' to quit).", args.camera)
    try:
        summary = rt.run(source=args.camera, output_path=output, show_preview=True,
                         max_frames=args.max_frames)
    except Exception as exc:
        log.error("Webcam inference failed: {}", exc)
        return 1
    log.info("Stopped. Avg FPS: {:.1f}.", summary.get("avg_fps", 0.0))
    if output:
        log.info("Recording saved: {}", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
