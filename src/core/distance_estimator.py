"""Monocular distance estimation using known real-world object heights (pinhole).

For each tracked object we know its true physical height (client-provided, in
``model_config.distance_estimator.real_heights_m``) and the camera focal length
(from :class:`~utils.calibration.CameraCalibration`). The pinhole relation

    distance = focal_length_px * real_height_m / bbox_height_px

gives a robust monocular distance. Results are clamped to a sane range and
exponentially smoothed per track ID so the on-screen number never jumps.

A flat-road, ground-contact fallback is also available and is blended in for
objects standing on the road, which stabilises distance when bounding-box
heights are noisy (e.g. partial occlusion in heavy traffic).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.tracker import TrackedObject
from utils.calibration import CameraCalibration
from utils.logger import get_logger

log = get_logger(__name__)


class DistanceEstimator:
    """Estimates and smooths per-object distance in metres."""

    def __init__(self, cfg: Dict[str, Any], calibration: CameraCalibration) -> None:
        """Create the estimator.

        Args:
            cfg: The ``distance_estimator`` block of ``model_config``.
            calibration: Active camera calibration (already rescaled to frame).
        """
        self.cfg = cfg or {}
        self.calib = calibration
        self.method = str(self.cfg.get("method", "pinhole")).lower()
        raw_heights = self.cfg.get("real_heights_m", {})
        self.real_heights: Dict[int, float] = {int(k): float(v) for k, v in raw_heights.items()}
        self.default_height = float(self.cfg.get("default_height_m", 1.5))
        self.min_d = float(self.cfg.get("min_distance_m", 1.0))
        self.max_d = float(self.cfg.get("max_distance_m", 250.0))
        self.alpha = float(self.cfg.get("smoothing_alpha", 0.4))
        self.use_width_fallback = bool(self.cfg.get("use_width_fallback", True))
        # Per-track smoothed distance memory.
        self._ema: Dict[int, float] = {}

    def height_for(self, class_id: int) -> float:
        """Return the configured real height (m) for a class, or the default."""
        return self.real_heights.get(int(class_id), self.default_height)

    def _raw_distance(self, obj: TrackedObject) -> float:
        """Compute the unsmoothed distance (metres) for one object."""
        real_h = self.height_for(obj.class_id)
        px_h = obj.height
        d_height = self.calib.distance_from_height(px_h, real_h)

        # Ground-contact estimate from the box bottom row (flat-road model).
        _, y_bottom = obj.bottom_center
        d_ground = self.calib.ground_distance_from_row(y_bottom)

        # Blend: trust height model, nudge toward ground model when both finite.
        estimates = [d for d in (d_height, d_ground) if np.isfinite(d) and d > 0]
        if not estimates:
            # Width fallback for oddly-shaped/very-wide boxes.
            if self.use_width_fallback:
                d_w = self.calib.distance_from_width(obj.width, real_h * 0.5)
                if np.isfinite(d_w) and d_w > 0:
                    return float(np.clip(d_w, self.min_d, self.max_d))
            return self.max_d
        if len(estimates) == 2:
            # Weighted toward the height estimate (0.7) which is class-aware.
            dist = 0.7 * d_height + 0.3 * d_ground
        else:
            dist = estimates[0]
        return float(np.clip(dist, self.min_d, self.max_d))

    def _smooth(self, track_id: int, value: float) -> float:
        """Exponentially smooth ``value`` for ``track_id``."""
        prev = self._ema.get(track_id)
        if prev is None:
            self._ema[track_id] = value
            return value
        smoothed = self.alpha * value + (1.0 - self.alpha) * prev
        self._ema[track_id] = smoothed
        return smoothed

    def estimate(self, objects: Sequence[TrackedObject]) -> List[TrackedObject]:
        """Fill ``distance_m`` on each tracked object (in place) and return them.

        Args:
            objects: Tracked objects for the current frame.

        Returns:
            The same list, with ``distance_m`` populated.
        """
        seen = set()
        for obj in objects:
            try:
                raw = self._raw_distance(obj)
                obj.distance_m = round(self._smooth(obj.track_id, raw), 2)
                seen.add(obj.track_id)
            except Exception as exc:  # never break the stream
                log.debug("Distance estimate failed for track {}: {}", obj.track_id, exc)
                obj.distance_m = None
        # Forget tracks that disappeared to avoid unbounded memory growth.
        for tid in list(self._ema.keys()):
            if tid not in seen:
                self._ema.pop(tid, None)
        return list(objects)

    def reset(self) -> None:
        """Clear per-track smoothing memory."""
        self._ema.clear()
