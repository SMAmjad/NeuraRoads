"""Benchmark the NeuraRoads pipeline: FPS, per-stage timings and system load.

Runs the full pipeline over a sample video (or synthetic frames) and reports a
per-stage breakdown so you can see where the frame budget goes and whether the
60 FPS / 30-40 FPS targets are met on this hardware. Also useful to compare
PyTorch vs ONNX vs TensorRT backends after export.

Usage::

    python src/scripts/benchmark.py --source data/raw/videos/dashcam_samples/drive.mp4 --frames 300
    python src/scripts/benchmark.py --synthetic --frames 200 --no-detector
    python src/scripts/benchmark.py --jetson --source clip.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from pipeline.inference_pipeline import NeuraRoadsPipeline
from utils.config_loader import resolve_path
from utils.logger import get_logger
from utils.metrics import system_stats
from utils.video_processor import VideoReader

log = get_logger(__name__)


def _print_table(pipeline: NeuraRoadsPipeline, frames: int, elapsed: float) -> None:
    """Pretty-print the benchmark results (rich table if available)."""
    avgs = pipeline.perf.averages()
    fps = frames / elapsed if elapsed > 0 else 0.0
    sysinfo = system_stats()
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="NeuraRoads Pipeline Benchmark")
        table.add_column("Stage", style="cyan")
        table.add_column("ms/frame", justify="right", style="magenta")
        table.add_column("% of frame", justify="right")
        total_ms = sum(avgs.values()) or 1.0
        for stage, ms in sorted(avgs.items(), key=lambda kv: -kv[1]):
            table.add_row(stage, f"{ms:.2f}", f"{100 * ms / total_ms:.1f}%")
        Console().print(table)
        Console().print(f"[bold green]Throughput: {fps:.1f} FPS[/] over {frames} frames "
                        f"({elapsed:.1f}s). GPU mem={sysinfo.get('gpu_mem_mb', 0):.0f}MB "
                        f"CPU={sysinfo.get('cpu_percent', 0):.0f}%")
    except Exception:
        log.info("Throughput: {:.1f} FPS over {} frames ({:.1f}s).", fps, frames, elapsed)
        for stage, ms in sorted(avgs.items(), key=lambda kv: -kv[1]):
            log.info("  {:<10s} {:.2f} ms", stage, ms)


def main() -> int:
    """CLI entry: benchmark the pipeline."""
    parser = argparse.ArgumentParser(description="Benchmark the NeuraRoads pipeline.")
    parser.add_argument("--source", type=str, default=None, help="Sample video (else --synthetic).")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic random frames.")
    parser.add_argument("--frames", type=int, default=200, help="Frames to benchmark.")
    parser.add_argument("--jetson", action="store_true", help="Apply jetson_config overlay.")
    parser.add_argument("--no-detector", action="store_true", help="Skip object detection.")
    args = parser.parse_args()

    overlay = "jetson_config" if args.jetson else None
    pipeline = NeuraRoadsPipeline(overlay=overlay, allow_no_detector=args.no_detector,
                                  frame_rate=30.0)
    pipeline.warmup()

    reader: Optional[VideoReader] = None
    if not args.synthetic and args.source:
        src = str(resolve_path(args.source))
        if not Path(src).is_file():
            log.error("Source not found: {}", src)
            return 1
        reader = VideoReader(src, threaded=True)

    log.info("Benchmarking {} frames ({})...", args.frames,
             "synthetic" if reader is None else Path(str(args.source)).name)
    t0 = time.perf_counter()
    count = 0
    if reader is not None:
        for frame in reader:
            pipeline.process_frame(frame, 1.0 / 30.0)
            count += 1
            if count >= args.frames:
                break
        reader.release()
    else:
        frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
        for _ in range(args.frames):
            pipeline.process_frame(frame.copy(), 1.0 / 30.0)
            count += 1
    elapsed = time.perf_counter() - t0

    _print_table(pipeline, count, elapsed)
    mdir = resolve_path("src/results/metrics/performance_logs")
    pipeline.perf.save_csv(mdir / "benchmark.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
