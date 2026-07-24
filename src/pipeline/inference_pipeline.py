"""The NeuraRoads perception + ADAS pipeline (single-frame engine).

:class:`NeuraRoadsPipeline` wires every module together into one
``process_frame`` call that returns a fully annotated frame plus the structured
state behind it. It is the reusable engine shared by the video, webcam and batch
inference scripts and by the real-time runner.

Per-frame flow::

    detect -> track -> distance -> speed(ego+objects) -> lane
          -> collision assess -> lane departure -> warnings -> alert
          -> BEV -> HUD render

Heavy stages (lane, BEV) honour a configurable cadence so the FPS budget is
protected; the last result is reused on skipped frames for smooth output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from adas.alert_manager import AlertManager
from adas.collision_detector import CollisionDetector
from adas.lane_departure import LaneDepartureWarning
from adas.warning_system import WarningSystem
from core.bev_transformer import BEVTransformer
from core.detector import Detector
from core.distance_estimator import DistanceEstimator
from core.lane_detector import LaneDetector, LaneResult
from core.speed_estimator import SpeedEstimator
from core.tracker import ObjectTracker
from utils.calibration import CameraCalibration
from utils.config_loader import load_config, resolve_device, apply_torch_runtime
from utils.logger import get_logger
from utils.metrics import PerformanceTracker
from utils.visualization import Visualizer

log = get_logger(__name__)


class NeuraRoadsPipeline:
    """End-to-end ADAS pipeline configured from ``model_config`` + ``adas_thresholds``."""

    def __init__(
        self,
        config_name: str = "model_config",
        overlay: Optional[str] = None,
        adas_config: str = "adas_thresholds",
        frame_size: Optional[Tuple[int, int]] = None,
        frame_rate: float = 30.0,
        allow_no_detector: bool = False,
    ) -> None:
        """Build the pipeline and all sub-modules.

        Args:
            config_name: Base model config name.
            overlay: Optional overlay config (e.g. ``"jetson_config"``).
            adas_config: ADAS thresholds config name.
            frame_size: Initial ``(width, height)``; refined on first frame.
            frame_rate: Source FPS (affects tracker + speed dt default).
            allow_no_detector: If True and no trained weights exist, run without
                object detection (lane + ego speed + HUD still work). Useful for
                testing the pipeline before the model is trained.
        """
        self.cfg = load_config(config_name, overlay)
        self.adas_cfg = load_config(adas_config)
        self.device = resolve_device(self.cfg.get("device"))
        apply_torch_runtime(self.cfg.get("device", {}))
        self.frame_rate = float(frame_rate)

        # Processing resolution (protects FPS; falls back to native).
        pcfg = self.cfg.get("pipeline", {})
        self.proc_w = int(pcfg.get("process_width", 1280))
        self.proc_h = int(pcfg.get("process_height", 720))
        cad = pcfg.get("cadence", {})
        self.lane_every = max(1, int(cad.get("lane_every", 1)))
        self.bev_every = max(1, int(cad.get("bev_every", 1)))
        self.det_every = max(1, int(cad.get("detection_every", 1)))

        self._frame_size = frame_size
        self.calib: Optional[CameraCalibration] = None

        # --- Detector (optional if not yet trained) ---
        self.detector: Optional[Detector] = None
        try:
            self.detector = Detector(self.cfg, self.device)
        except FileNotFoundError as exc:
            if not allow_no_detector:
                raise
            log.warning("Running WITHOUT object detection: {}", exc)

        # --- Core + ADAS modules ---
        self.tracker = ObjectTracker(self.cfg.get("tracker", {}), self.frame_rate)
        self.speed = SpeedEstimator(self.cfg.get("speed_estimator", {}))
        self.lane = LaneDetector(self.cfg, self.device)
        self.collision = CollisionDetector(self.cfg, self.adas_cfg)
        self.lane_departure = LaneDepartureWarning(self.adas_cfg)
        self.warnings = WarningSystem(self.adas_cfg)
        self.alerts = AlertManager(self.adas_cfg)
        self.visualizer = Visualizer(self.cfg.get("visualization", {}), self.cfg.get("bev", {}))
        self.perf = PerformanceTracker()

        # Filled once the frame size is known.
        self.distance: Optional[DistanceEstimator] = None
        self.bev: Optional[BEVTransformer] = None

        self._frame_idx = 0
        self._last_lane: Optional[LaneResult] = None
        self._last_bev: Optional[np.ndarray] = None
        if frame_size is not None:
            self._init_size(frame_size)

    # -- setup depending on frame size -------------------------------------
    def _init_size(self, frame_size: Tuple[int, int]) -> None:
        """Initialise calibration-dependent modules for a given frame size."""
        self._frame_size = frame_size
        self.calib = CameraCalibration.from_config(frame_size=frame_size)
        self.distance = DistanceEstimator(self.cfg.get("distance_estimator", {}), self.calib)
        colors = {k: tuple(v) for k, v in self.cfg.get("visualization", {}).get("colors", {}).items()}
        self.bev = BEVTransformer(self.cfg.get("bev", {}), self.calib, colors)
        # Give the ego-speed estimator the calibration (enables metric, scale-free
        # ego speed) plus any explicit fallback flow->kmh scale from calibration.
        self.speed.ego.set_calibration(self.calib)
        try:
            spd_cal = load_config("camera_calibration").get("speed_calibration", {})
            self.speed.ego.set_scale(spd_cal.get("flow_to_kmh_scale"))
        except Exception:
            pass
        log.info("Pipeline initialised for frame size {}x{}.", frame_size[0], frame_size[1])

    def _fit_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a frame to the processing resolution, preserving aspect ratio."""
        h, w = frame.shape[:2]
        scale = min(self.proc_w / w, self.proc_h / h)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return frame

    # -- per-frame processing ----------------------------------------------
    def process_frame(self, frame: np.ndarray, dt: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process one BGR frame and return ``(annotated_frame, state)``.

        Args:
            frame: Input BGR frame.
            dt: Seconds since the previous frame (defaults to ``1/frame_rate``).

        Returns:
            Tuple of the annotated frame and the structured per-frame state.
        """
        self.perf.frame_start()
        frame = self._fit_frame(frame)
        h, w = frame.shape[:2]
        if self.calib is None or self._frame_size != (w, h):
            self._init_size((w, h))
        dt = (1.0 / self.frame_rate) if dt is None else max(1e-3, dt)
        self._frame_idx += 1

        # 1) Detection + tracking
        with self.perf.stage("detect"):
            detections = self.detector.detect(frame) if self.detector is not None else []
        with self.perf.stage("track"):
            tracks = self.tracker.update(detections, frame)

        # 2) Distance + speed
        with self.perf.stage("distance"):
            if self.distance is not None:
                self.distance.estimate(tracks)
        with self.perf.stage("speed"):
            # Mask detected vehicles out of the ego optical-flow ROI.
            ego_speed = self.speed.update_ego(frame, dt, [o.box for o in tracks])
            self.speed.estimate(tracks, dt)

        # 3) Lane detection (cadence-controlled, reuse between)
        with self.perf.stage("lane"):
            if self._frame_idx % self.lane_every == 0 or self._last_lane is None:
                self._last_lane = self.lane.detect(frame)
            lane_result = self._last_lane

        # 4) ADAS decisions
        with self.perf.stage("adas"):
            assessment = self.collision.assess(tracks, (w, h))
            lane_state = self.lane_departure.update(lane_result)
            events, primary = self.warnings.evaluate(assessment, lane_state, lane_result)
            alert = self.alerts.update(primary)

        # 5) BEV (cadence-controlled)
        with self.perf.stage("bev"):
            if self.bev is not None and (self._frame_idx % self.bev_every == 0 or self._last_bev is None):
                self._last_bev = self.bev.render(tracks)

        # 6) Assemble render state + draw HUD
        with self.perf.stage("render"):
            state = self._build_state(tracks, lane_result, lane_state, alert, events,
                                      assessment, ego_speed)
            annotated = self.visualizer.render(frame, state)

        fps = self.perf.frame_end()
        state["fps"] = fps
        return annotated, state

    def _build_state(self, tracks, lane_result, lane_state, alert, events, assessment,
                     ego_speed) -> Dict[str, Any]:
        """Assemble the dict consumed by the visualizer and callers."""
        max_closest = int(self.cfg.get("visualization", {}).get("hud", {}).get("closest_objects_count", 3))
        max_warn = int(self.adas_cfg.get("icon_behaviour", {}).get("max_active_warnings", 4))
        closest = [
            {"class_name": o.class_name, "distance_m": o.distance_m, "color_key": o.color_key}
            for o in assessment.closest[:max_closest]
        ]
        lane_render = None
        if lane_result is not None and lane_result.is_valid():
            lane_render = {
                "lines": lane_result.lines,
                "fill": lane_result.fill,
                "drive_mask": lane_result.drive_mask,
                "lane_mask": lane_result.lane_mask,
                "leaving": lane_state.active,
            }
        return {
            "objects": [o.as_dict() for o in tracks],
            "lane": lane_render,
            "bev": self._last_bev,
            "alert": alert,
            "closest": closest,
            "warnings": self.warnings.active_warning_messages(events, max_warn),
            "ego_speed_kmh": round(ego_speed, 1),
            "fps": self.perf.fps,
        }

    def reset(self) -> None:
        """Reset all stateful modules (call between separate videos)."""
        self.tracker.reset()
        self.speed.reset()
        self.lane.reset()
        self.lane_departure.reset()
        self.alerts.reset()
        if self.distance is not None:
            self.distance.reset()
        self._frame_idx = 0
        self._last_lane = None
        self._last_bev = None

    def warmup(self) -> None:
        """Warm up the detector (first-frame latency)."""
        if self.detector is not None:
            self.detector.warmup()
