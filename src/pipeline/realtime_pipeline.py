"""Real-time runner that drives :class:`NeuraRoadsPipeline` over a video/webcam.

Handles threaded I/O, an optional live preview, writing the annotated output at
the configured FPS, and *adaptive quality* - if the measured FPS drops below the
floor for a sustained period it automatically reduces the lane/BEV cadence to
recover, then relaxes again when there is headroom. This is the component that
keeps the system responsive on both the GTX 1660 SUPER and the Jetson Nano.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2

from pipeline.inference_pipeline import NeuraRoadsPipeline
from utils.config_loader import load_config, resolve_path
from utils.logger import get_logger
from utils.video_processor import VideoReader, VideoWriter

log = get_logger(__name__)


class RealtimePipeline:
    """Streams frames through the ADAS pipeline with adaptive quality control."""

    def __init__(
        self,
        config_name: str = "model_config",
        overlay: Optional[str] = None,
        adas_config: str = "adas_thresholds",
        allow_no_detector: bool = False,
    ) -> None:
        """Args mirror :class:`NeuraRoadsPipeline`; the pipeline is built per-run
        once the source FPS/size are known."""
        self.config_name = config_name
        self.overlay = overlay
        self.adas_config = adas_config
        self.allow_no_detector = allow_no_detector
        self.cfg = load_config(config_name, overlay)
        self._pipeline: Optional[NeuraRoadsPipeline] = None

    # -- adaptive quality ---------------------------------------------------
    def _adapt(self, pipeline: NeuraRoadsPipeline, fps: float) -> None:
        """Nudge lane/BEV cadence based on measured FPS."""
        aq = self.cfg.get("pipeline", {}).get("adaptive_quality", {})
        if not aq.get("enabled", True):
            return
        min_fps = float(aq.get("min_fps", 25))
        if fps and fps < min_fps:
            pipeline.lane_every = min(4, pipeline.lane_every + 1)
            pipeline.bev_every = min(4, pipeline.bev_every + 1)
        elif fps > min_fps * 1.5:
            pipeline.lane_every = max(1, pipeline.lane_every - 1)
            pipeline.bev_every = max(1, pipeline.bev_every - 1)

    # -- run ----------------------------------------------------------------
    def run(
        self,
        source: Union[str, Path, int],
        output_path: Optional[Union[str, Path]] = None,
        show_preview: Optional[bool] = None,
        max_frames: Optional[int] = None,
        save_metrics: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Process a source end to end.

        Args:
            source: Video path or webcam index.
            output_path: Where to write the annotated video (None = don't write).
            show_preview: Show a live window (defaults to config ``video.show_preview``).
            max_frames: Stop after this many frames (None = whole source).
            save_metrics: Save a per-frame performance CSV (defaults to config).

        Returns:
            A summary dict (avg FPS, per-stage timings, frame count, output path).
        """
        vcfg = self.cfg.get("video", {})
        show_preview = vcfg.get("show_preview", True) if show_preview is None else show_preview
        save_metrics = vcfg.get("save_metrics", True) if save_metrics is None else save_metrics
        threaded = self.cfg.get("pipeline", {}).get("threading", {}).get("enabled", True)
        queue_size = int(self.cfg.get("pipeline", {}).get("threading", {}).get("queue_size", 8))

        reader = VideoReader(source, queue_size=queue_size, threaded=threaded)
        pipeline = NeuraRoadsPipeline(
            self.config_name, self.overlay, self.adas_config,
            frame_rate=reader.fps, allow_no_detector=self.allow_no_detector,
        )
        pipeline.warmup()
        self._pipeline = pipeline

        writer: Optional[VideoWriter] = None
        # output_fps null/0 => match the source FPS (correct playback speed).
        out_fps = float(vcfg.get("output_fps") or reader.fps)
        codec = vcfg.get("output_codec", "mp4v")

        aq = self.cfg.get("pipeline", {}).get("adaptive_quality", {})
        check_interval = float(aq.get("check_interval_s", 2.0))
        last_check = time.perf_counter()

        win = "NeuraRoads ADAS"
        if show_preview:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        count = 0
        try:
            for frame in reader:
                dt = None if reader.is_webcam else (1.0 / reader.fps)
                annotated, state = pipeline.process_frame(frame, dt)

                if output_path is not None:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = VideoWriter(resolve_path(output_path), out_fps, (w, h),
                                             codec=codec, threaded=threaded)
                    writer.write(annotated)

                if show_preview:
                    cv2.imshow(win, annotated)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        log.info("Preview stop requested.")
                        break

                count += 1
                now = time.perf_counter()
                if now - last_check >= check_interval:
                    self._adapt(pipeline, pipeline.perf.fps)
                    last_check = now
                    log.debug("FPS {:.1f} | lane_every={} bev_every={}",
                              pipeline.perf.fps, pipeline.lane_every, pipeline.bev_every)
                if max_frames is not None and count >= max_frames:
                    break
        finally:
            reader.release()
            if writer is not None:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()

        pipeline.perf.log_summary()
        summary = pipeline.perf.summary()
        summary["frames"] = float(count)
        if output_path is not None:
            summary["output"] = str(resolve_path(output_path))
        if save_metrics:
            mdir = resolve_path(vcfg.get("metrics_dir", "src/results/metrics/performance_logs"))
            stem = "webcam" if reader.is_webcam else Path(str(source)).stem
            pipeline.perf.save_csv(mdir / f"perf_{stem}.csv")
        return summary
