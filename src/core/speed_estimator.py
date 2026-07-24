"""Speed estimation: per-object (distance-derivative + Kalman) and ego (optical flow).

Client-approved technique. For each tracked object a 1-D constant-velocity
Kalman filter runs on its measured distance; the filtered velocity is the range
rate (closing speed). Combined with the ego speed it yields the object's
absolute over-ground speed. A final EMA keeps the displayed number stable.

Ego speed is estimated from sparse optical flow of the static road surface in the
lower image ROI, scaled to km/h by a calibratable factor. Everything is smooth
and needs no external sensor, matching the monocular constraint.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.tracker import TrackedObject
from utils.logger import get_logger

log = get_logger(__name__)

MPS_TO_KMH = 3.6


class _KalmanCV1D:
    """Minimal 1-D constant-velocity Kalman filter (state = [position, velocity])."""

    def __init__(self, q: float, r: float, x0: float) -> None:
        """Args: q: process noise; r: measurement noise; x0: initial position."""
        self.x = np.array([x0, 0.0], dtype=np.float64)          # [pos, vel]
        self.P = np.diag([1.0, 10.0]).astype(np.float64)
        self.q = float(q)
        self.r = float(r)

    def step(self, z: float, dt: float) -> Tuple[float, float]:
        """Predict with ``dt`` then update with measurement ``z``.

        Returns:
            Tuple ``(filtered_position, filtered_velocity)``.
        """
        dt = max(1e-3, float(dt))
        # Predict
        F = np.array([[1.0, dt], [0.0, 1.0]])
        self.x = F @ self.x
        # Process noise (piecewise white-noise acceleration model).
        G = np.array([[0.5 * dt * dt], [dt]])
        Q = (G @ G.T) * self.q
        self.P = F @ self.P @ F.T + Q
        # Update
        H = np.array([[1.0, 0.0]])
        y = z - (H @ self.x)[0]
        S = (H @ self.P @ H.T)[0, 0] + self.r
        K = (self.P @ H.T).flatten() / S
        self.x = self.x + K * y
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        return float(self.x[0]), float(self.x[1])


class EgoSpeedEstimator:
    """Estimates ego-vehicle speed (km/h) from optical flow of the STATIC road.

    Two improvements over a naive flow-magnitude approach make this reliable in
    traffic:

    * **Calibration-based metric scaling** - when a :class:`CameraCalibration`
      is provided, each tracked ground feature's image row is converted to a real
      ground distance. Because every point on the flat road moves by the *same*
      metric amount per frame under forward motion, the median of those per-frame
      distance changes IS the ego displacement - giving a correctly-scaled speed
      with no magic factor. Falls back to ``flow_magnitude * scale`` if no calib.
    * **Vehicle masking** - detected object boxes are excluded from the feature
      ROI so we track the static road, not other moving cars (the main cause of
      badly under-estimated ego speed in dense traffic).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Args: cfg: The ``speed_estimator.ego`` block of ``model_config``."""
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", True))
        self.method = str(self.cfg.get("method", "optical_flow")).lower()
        self.max_corners = int(self.cfg.get("max_corners", 200))
        self.quality = float(self.cfg.get("quality_level", 0.01))
        self.min_dist = int(self.cfg.get("min_corner_distance", 12))
        self.roi_top = float(self.cfg.get("road_roi_top", 0.60))
        self.roi_bottom = float(self.cfg.get("road_roi_bottom", 0.94))
        self.roi_x_frac = float(self.cfg.get("road_roi_x_frac", 0.6))
        self.scale = float(self.cfg.get("flow_to_kmh_scale", 0.65))
        self.ema_alpha = float(self.cfg.get("ema_alpha", 0.25))
        self.max_speed_kmh = float(self.cfg.get("max_ego_speed_kmh", 200.0))
        self.max_ground_disp_m = float(self.cfg.get("max_ground_disp_m", 5.0))
        self.manual_speed = float(self.cfg.get("manual_speed_kmh", 0.0))
        # Run corner-finding + optical flow on a downscaled frame (4x fewer
        # pixels at 0.5) for a big speed-stage win; ground rows are mapped back
        # to full-res before metric conversion so accuracy is unchanged.
        self.flow_scale = float(self.cfg.get("flow_downscale", 0.5))
        self.calib = None
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self.speed_kmh: float = 0.0

    def set_scale(self, scale: Optional[float]) -> None:
        """Override the fallback flow->km/h scale (e.g. from camera calibration)."""
        if scale is not None:
            self.scale = float(scale)

    def set_calibration(self, calib) -> None:
        """Provide camera calibration to enable metric (scale-free) ego speed."""
        self.calib = calib

    def _feature_points(self, gray: np.ndarray, boxes: Optional[Sequence] = None,
                        s: float = 1.0) -> np.ndarray:
        """Find corners in the lower-central road ROI, excluding vehicle boxes.

        ``gray`` is at the (possibly downscaled) flow resolution; ``s`` is the
        scale from full-frame to flow coords so vehicle boxes are scaled to match.
        """
        h, w = gray.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        y0, y1 = int(h * self.roi_top), int(h * self.roi_bottom)
        half = self.roi_x_frac * w / 2.0
        x0, x1 = int(w / 2 - half), int(w / 2 + half)
        mask[y0:y1, x0:x1] = 255
        if boxes:
            for b in boxes:
                if not all(np.isfinite(v) for v in b):
                    continue
                bx1, by1, bx2, by2 = [int(v * s) for v in b]
                cv2.rectangle(mask, (bx1, by1), (bx2, by2), 0, -1)  # drop moving vehicles
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_corners, qualityLevel=self.quality,
            minDistance=self.min_dist, mask=mask,
        )
        return pts if pts is not None else np.zeros((0, 1, 2), np.float32)

    def update(self, frame: np.ndarray, dt: float,
               boxes: Optional[Sequence] = None) -> float:
        """Update and return the smoothed ego speed (km/h) for this frame.

        Args:
            frame: Current BGR frame.
            dt: Seconds since the previous frame.
            boxes: Detected object boxes ``[x1,y1,x2,y2]`` to exclude from flow.
        """
        if not self.enabled:
            return 0.0
        if self.method == "manual":
            self.speed_kmh = self.manual_speed
            return self.speed_kmh

        # Downscale for corner-finding + flow (big speed win); map rows back
        # to full-res via `inv` before any metric ground-distance conversion.
        s = self.flow_scale if self.flow_scale and self.flow_scale < 1.0 else 1.0
        if frame.ndim == 3:
            small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else frame
        inv = 1.0 / s
        inst = self.speed_kmh
        if self._prev_gray is not None and self._prev_pts is not None and len(self._prev_pts):
            nxt = status = None
            try:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, self._prev_pts, None)
            except cv2.error as exc:
                log.debug("Ego optical flow error: {}", exc)
            if nxt is not None and status is not None:
                st = status.flatten() == 1
                go = self._prev_pts[st].reshape(-1, 2)
                gn = nxt[st].reshape(-1, 2)
                dt_s = max(1e-3, dt)
                if self.calib is not None and len(gn) >= 5:
                    # Metric: change in ground distance == ego displacement.
                    disp = []
                    for (ox, oy), (nx, ny) in zip(go, gn):
                        if ny <= oy:                      # road streams downward
                            continue
                        d_old = self.calib.ground_distance_from_row(oy * inv)
                        d_new = self.calib.ground_distance_from_row(ny * inv)
                        if np.isfinite(d_old) and np.isfinite(d_new) and d_old > d_new:
                            m = d_old - d_new
                            if 0.0 < m < self.max_ground_disp_m:
                                disp.append(m)
                    if len(disp) >= 5:
                        inst = (float(np.median(disp)) / dt_s) * MPS_TO_KMH
                elif len(gn) >= 5:
                    # Fallback: flow magnitude * calibratable scale.
                    flow = gn - go
                    mag = float(np.median(np.abs(flow[:, 1])))
                    inst = (mag / dt_s) * self.scale
                inst = float(np.clip(inst, 0.0, self.max_speed_kmh))

        self.speed_kmh = (1 - self.ema_alpha) * self.speed_kmh + self.ema_alpha * inst
        self.speed_kmh = float(max(0.0, self.speed_kmh))
        self._prev_gray = gray
        self._prev_pts = self._feature_points(gray, boxes, s)
        return self.speed_kmh

    def reset(self) -> None:
        """Clear optical-flow history."""
        self._prev_gray = None
        self._prev_pts = None
        self.speed_kmh = 0.0


class SpeedEstimator:
    """Per-object relative + absolute speed with Kalman filtering, plus ego speed."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Args: cfg: The ``speed_estimator`` block of ``model_config``."""
        self.cfg = cfg or {}
        self.q = float(self.cfg.get("kalman_process_noise", 1.0))
        self.r = float(self.cfg.get("kalman_measurement_noise", 4.0))
        self.ema_alpha = float(self.cfg.get("ema_alpha", 0.3))
        self.min_age = int(self.cfg.get("min_track_age", 3))
        self.max_rel = float(self.cfg.get("max_rel_speed_kmh", 250.0))
        self.ego = EgoSpeedEstimator(self.cfg.get("ego", {}))

        self._filters: Dict[int, _KalmanCV1D] = {}
        self._age: Dict[int, int] = {}
        self._ema_speed: Dict[int, float] = {}
        self.ego_speed_kmh: float = 0.0

    def update_ego(self, frame: np.ndarray, dt: float,
                   boxes: Optional[Sequence] = None) -> float:
        """Update and return the ego speed (km/h), excluding ``boxes`` from flow."""
        self.ego_speed_kmh = self.ego.update(frame, dt, boxes)
        return self.ego_speed_kmh

    def estimate(self, objects: Sequence[TrackedObject], dt: float) -> List[TrackedObject]:
        """Fill ``speed_kmh`` (absolute) + range-rate on each object in place.

        Args:
            objects: Tracked objects with ``distance_m`` already populated.
            dt: Seconds elapsed since the previous frame.

        Returns:
            The same list with speed fields populated (and ``extra`` updated).
        """
        seen = set()
        for obj in objects:
            tid = obj.track_id
            seen.add(tid)
            if obj.distance_m is None or not np.isfinite(obj.distance_m):
                obj.speed_kmh = None
                continue

            self._age[tid] = self._age.get(tid, 0) + 1
            kf = self._filters.get(tid)
            if kf is None:
                kf = _KalmanCV1D(self.q, self.r, obj.distance_m)
                self._filters[tid] = kf
            _, vel_mps = kf.step(obj.distance_m, dt)  # d(distance)/dt

            range_rate_kmh = float(np.clip(vel_mps * MPS_TO_KMH, -self.max_rel, self.max_rel))
            closing_kmh = -range_rate_kmh  # +ve => approaching
            # Absolute over-ground object speed (monocular approximation).
            abs_speed = max(0.0, self.ego_speed_kmh + range_rate_kmh)

            # Smooth the displayed absolute speed.
            prev = self._ema_speed.get(tid, abs_speed)
            abs_speed = self.ema_alpha * abs_speed + (1 - self.ema_alpha) * prev
            self._ema_speed[tid] = abs_speed

            obj.extra["range_rate_mps"] = vel_mps
            obj.extra["closing_speed_kmh"] = closing_kmh
            if self._age[tid] >= self.min_age:
                obj.speed_kmh = round(abs_speed, 1)
            else:
                obj.speed_kmh = None  # not yet trustworthy

        # Prune vanished tracks.
        for tid in list(self._filters.keys()):
            if tid not in seen:
                self._filters.pop(tid, None)
                self._age.pop(tid, None)
                self._ema_speed.pop(tid, None)
        return list(objects)

    def reset(self) -> None:
        """Clear all per-track and ego speed state."""
        self._filters.clear()
        self._age.clear()
        self._ema_speed.clear()
        self.ego.reset()
        self.ego_speed_kmh = 0.0
