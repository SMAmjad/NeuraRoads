"""Extract frames from videos - for building datasets or generating test images.

Can sample every Nth frame, at a target FPS, or a fixed number of evenly-spaced
frames. Used to turn raw dashcam clips into still images for annotation or for
quick qualitative testing.

Usage::

    python src/preprocessing/frame_extractor.py --source clip.mp4 --every 15
    python src/preprocessing/frame_extractor.py --source clip.mp4 --fps 2 --output frames/
    python src/preprocessing/frame_extractor.py --source clip.mp4 --count 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cv2

from utils.config_loader import resolve_path
from utils.logger import get_logger
from utils.video_processor import VideoReader

log = get_logger(__name__)


def extract_frames(
    source: str,
    output_dir: Path,
    every: Optional[int] = None,
    target_fps: Optional[float] = None,
    count: Optional[int] = None,
    prefix: Optional[str] = None,
    img_ext: str = "jpg",
) -> int:
    """Extract frames from a video according to one sampling strategy.

    Exactly one of ``every`` / ``target_fps`` / ``count`` selects the strategy
    (checked in that priority order; defaults to every 30th frame).

    Args:
        source: Input video path.
        output_dir: Directory to write frames into (created if needed).
        every: Save every Nth frame.
        target_fps: Save frames to approximate this output FPS.
        count: Save this many evenly-spaced frames across the whole video.
        prefix: Filename prefix (defaults to the video stem).
        img_ext: Output image extension.

    Returns:
        Number of frames written.
    """
    reader = VideoReader(source, threaded=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or Path(source).stem

    if count is not None and reader.frame_count > 0:
        step = max(1, reader.frame_count // count)
    elif target_fps is not None:
        step = max(1, int(round(reader.fps / max(target_fps, 0.1))))
    else:
        step = max(1, every or 30)

    log.info("Extracting from {} every {} frames (~{} total).",
             Path(source).name, step, (reader.frame_count // step if reader.frame_count else "?"))

    idx = saved = 0
    try:
        for frame in reader:
            if idx % step == 0:
                out = output_dir / f"{prefix}_{idx:06d}.{img_ext}"
                cv2.imwrite(str(out), frame)
                saved += 1
                if count is not None and saved >= count:
                    break
            idx += 1
    finally:
        reader.release()
    log.info("Wrote {} frames -> {}", saved, output_dir)
    return saved


def main() -> int:
    """CLI entry: extract frames from a single video."""
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("--source", type=str, required=True, help="Input video path.")
    parser.add_argument("--output", type=str, default=None, help="Output directory.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--every", type=int, help="Save every Nth frame.")
    group.add_argument("--fps", type=float, help="Approximate output FPS.")
    group.add_argument("--count", type=int, help="Total evenly-spaced frames.")
    parser.add_argument("--prefix", type=str, default=None, help="Filename prefix.")
    args = parser.parse_args()

    source = str(resolve_path(args.source))
    if not Path(source).is_file():
        log.error("Video not found: {}", source)
        return 1
    output = resolve_path(args.output) if args.output else \
        resolve_path("data/raw/frames") / Path(source).stem

    try:
        extract_frames(source, output, every=args.every, target_fps=args.fps,
                       count=args.count, prefix=args.prefix)
    except Exception as exc:
        log.error("Frame extraction failed: {}", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
