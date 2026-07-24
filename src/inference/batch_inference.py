"""Batch-process a folder of dashcam videos through the NeuraRoads ADAS pipeline.

Annotates every video in an input directory, writing outputs to a results
directory and a combined CSV summary of per-video FPS. The pipeline is rebuilt
per video (fresh tracker/speed/lane state) but the heavy model load is shared
via Ultralytics' internal caching.

Usage::

    python src/inference/batch_inference.py --input data/raw/videos/test_videos
    python src/inference/batch_inference.py --input clips/ --output out/ --no-preview
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.realtime_pipeline import RealtimePipeline
from utils.config_loader import resolve_path
from utils.logger import get_logger

log = get_logger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def find_videos(input_dir: Path) -> List[Path]:
    """Return sorted video files directly under ``input_dir``."""
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def main() -> int:
    """CLI entry: annotate every video in a folder."""
    parser = argparse.ArgumentParser(description="NeuraRoads ADAS - batch inference.")
    parser.add_argument("--input", type=str, required=True, help="Folder of input videos.")
    parser.add_argument("--output", type=str, default=None, help="Output folder.")
    parser.add_argument("--no-preview", action="store_true", help="Disable live windows.")
    parser.add_argument("--jetson", action="store_true", help="Apply jetson_config overlay.")
    parser.add_argument("--no-detector", action="store_true", help="Run without detection.")
    args = parser.parse_args()

    input_dir = resolve_path(args.input)
    if not input_dir.is_dir():
        log.error("Input folder not found: {}", input_dir)
        return 1
    videos = find_videos(input_dir)
    if not videos:
        log.error("No videos found in {} (extensions: {}).", input_dir, sorted(VIDEO_EXTS))
        return 1

    out_dir = resolve_path(args.output) if args.output else resolve_path("src/results/videos/output_videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Batch: {} videos from {} -> {}", len(videos), input_dir, out_dir)

    rt = RealtimePipeline(
        overlay="jetson_config" if args.jetson else None,
        allow_no_detector=args.no_detector,
    )

    rows = []
    for i, vid in enumerate(videos, 1):
        out_path = out_dir / f"{vid.stem}_annotated.mp4"
        log.info("[{}/{}] {}", i, len(videos), vid.name)
        try:
            summary = rt.run(source=str(vid), output_path=str(out_path),
                             show_preview=not args.no_preview)
            rows.append({"video": vid.name, "avg_fps": summary.get("avg_fps", 0.0),
                         "frames": summary.get("frames", 0), "output": str(out_path)})
        except Exception as exc:
            log.error("Failed on {}: {}", vid.name, exc)
            rows.append({"video": vid.name, "avg_fps": 0.0, "frames": 0, "output": "FAILED"})

    try:
        import pandas as pd

        csv = out_dir / "batch_summary.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        log.info("Batch summary -> {}", csv)
    except Exception as exc:
        log.debug("Could not write batch summary: {}", exc)

    ok = sum(1 for r in rows if r["output"] != "FAILED")
    log.info("Batch complete: {}/{} succeeded.", ok, len(videos))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
