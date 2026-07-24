"""Bird's-Eye-View (top-down) mini-map renderer.

Two complementary projections are used:

* Objects are placed in metric BEV space directly from calibration: an object's
  forward distance comes from the distance estimator and its lateral offset is
  ``(x_img - cx) * distance / fx``. This is robust and needs no perfect
  homography.
* A ground-plane homography (derived from the configured source trapezoid) is
  available to warp lane geometry / road texture when desired.

The output is a small BGR canvas drawn into a corner of the annotated frame,
showing the ego vehicle, range rings and colour-coded objects.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.tracker import TrackedObject
from utils.calibration import CameraCalibration
from utils.logger import get_logger

log = get_logger(__name__)

BGR = Tuple[int, int, int]


class BEVTransformer:
    """Renders a metric top-down view of tracked objects around the ego vehicle."""

    def __init__(self, cfg: Dict[str, Any], calibration: CameraCalibration,
                 colors: Optional[Dict[str, BGR]] = None) -> None:
        """Create the BEV renderer.

        Args:
            cfg: The ``bev`` block of ``model_config``.
            calibration: Active camera calibration (for lateral projection).
            colors: Optional visualization colour map (danger keys -> BGR).
        """
        self.cfg = cfg or {}
        self.calib = calibration
        self.enabled = bool(self.cfg.get("enabled", True))
        w, h = self.cfg.get("output_size", [300, 400])
        self.width, self.height = int(w), int(h)
        rng = self.cfg.get("range_m", {})
        self.forward_m = float(rng.get("forward", 60.0))
        self.lateral_m = float(rng.get("lateral", 20.0))
        self.draw_grid = bool(self.cfg.get("draw_grid", True))
        self.ego_color: BGR = tuple(self.cfg.get("ego_color", [0, 255, 255]))
        self.update_every = int(self.cfg.get("update_every", 1))
        self.colors = colors or {}
        # Ego-lane width (m) drawn as the road corridor, and per-class real
        # footprints (width_m, length_m) so objects render as scaled top-down
        # vehicle icons rather than featureless dots.
        self.lane_width_m = float(self.cfg.get("lane_width_m", 3.7))
        default_sizes: Dict[str, Sequence[float]] = {
            "default": [1.9, 4.5], "_ego": [1.9, 4.6],
            "Car": [1.9, 4.5], "Bus": [2.6, 12.0], "Truck": [2.6, 9.0],
            "Train": [3.0, 20.0], "Bike": [0.8, 2.2], "Bicycle": [0.7, 1.8],
            "Pedestrian": [0.6, 0.6], "Ped on Bike": [0.8, 2.0],
            "Traffic Light": [0.6, 0.6], "Sign Board": [0.6, 0.6],
        }
        self.object_sizes: Dict[str, Sequence[float]] = {
            **default_sizes, **(self.cfg.get("object_sizes_m", {}) or {})
        }
        self._homography: Optional[np.ndarray] = None
        self._last_canvas: Optional[np.ndarray] = None
        self._frame_idx = 0

    # -- homography (optional, for warping lane/road) -----------------------
    def compute_homography(self, frame_size: Tuple[int, int],
                           src_trapezoid: Dict[str, float]) -> np.ndarray:
        """Compute and cache an image->BEV homography from a source trapezoid.

        Args:
            frame_size: ``(width, height)`` of the source frame.
            src_trapezoid: Fractions ``top_y/bottom_y/top_width/bottom_width``.

        Returns:
            The 3x3 homography matrix mapping image points to BEV canvas points.
        """
        w, h = frame_size
        ty = src_trapezoid.get("top_y", 0.62) * h
        by = src_trapezoid.get("bottom_y", 0.98) * h
        tw = src_trapezoid.get("top_width", 0.20) * w
        bw = src_trapezoid.get("bottom_width", 1.00) * w
        cx = w / 2.0
        src = np.float32([
            [cx - tw / 2, ty], [cx + tw / 2, ty],
            [cx + bw / 2, by], [cx - bw / 2, by],
        ])
        dst = np.float32([
            [0, 0], [self.width, 0],
            [self.width, self.height], [0, self.height],
        ])
        self._homography = cv2.getPerspectiveTransform(src, dst)
        return self._homography

    def warp(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Warp a frame into the BEV canvas using the cached homography."""
        if self._homography is None:
            return None
        return cv2.warpPerspective(frame, self._homography, (self.width, self.height))

    # -- metric placement ---------------------------------------------------
    def _metric_to_canvas(self, forward_m: float, lateral_m: float) -> Tuple[int, int]:
        """Map a metric ``(forward, lateral)`` point to canvas pixel coords."""
        x = self.width / 2.0 + (lateral_m / self.lateral_m) * (self.width / 2.0)
        y = self.height - (forward_m / self.forward_m) * self.height
        return int(np.clip(x, 0, self.width - 1)), int(np.clip(y, 0, self.height - 1))

    def object_lateral_m(self, obj: TrackedObject) -> float:
        """Estimate an object's lateral offset (m, +right) from calibration."""
        x_img, _ = obj.bottom_center
        dist = obj.distance_m if obj.distance_m else self.forward_m
        return (x_img - self.calib.cx) * dist / max(self.calib.fx, 1e-6)

    # -- rendering ----------------------------------------------------------
    def _px_size(self, w_m: float, l_m: float) -> Tuple[int, int]:
        """Convert a real ``(width_m, length_m)`` footprint to canvas ``(w, h)`` px."""
        wpx = int(round((w_m / max(self.lateral_m, 1e-6)) * (self.width / 2.0)))
        lpx = int(round((l_m / max(self.forward_m, 1e-6)) * self.height))
        return max(4, wpx), max(6, lpx)

    def _blank_canvas(self) -> np.ndarray:
        """Create the top-down road background: corridor + range rings."""
        canvas = np.full((self.height, self.width, 3), 24, np.uint8)
        cxp = self.width // 2
        # Ego-lane road corridor (shaded band down the middle) - grounds the view
        # as a road seen from directly above.
        half_lane = int((self.lane_width_m / 2.0 / max(self.lateral_m, 1e-6)) * (self.width / 2.0))
        cv2.rectangle(canvas, (cxp - half_lane, 0), (cxp + half_lane, self.height), (40, 40, 40), -1)
        cv2.line(canvas, (cxp - half_lane, 0), (cxp - half_lane, self.height), (75, 75, 75), 1)
        cv2.line(canvas, (cxp + half_lane, 0), (cxp + half_lane, self.height), (75, 75, 75), 1)
        if self.draw_grid:
            # Forward range rings every 10 m.
            for m in range(10, int(self.forward_m) + 1, 10):
                y = int(self.height - (m / self.forward_m) * self.height)
                cv2.line(canvas, (0, y), (self.width, y), (55, 55, 55), 1)
                cv2.putText(canvas, f"{m}m", (4, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, (115, 115, 115), 1, cv2.LINE_AA)
            cv2.line(canvas, (cxp, 0), (cxp, self.height), (55, 55, 55), 1)
        cv2.putText(canvas, "TOP VIEW", (self.width - 78, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (150, 150, 150), 1, cv2.LINE_AA)
        return canvas

    def _draw_ego(self, canvas: np.ndarray) -> None:
        """Draw the ego vehicle as a car body (with heading) at the bottom centre."""
        cxp = self.width // 2
        w_m, l_m = self.object_sizes.get("_ego", [1.9, 4.6])
        wpx, lpx = self._px_size(w_m, l_m)
        y2 = self.height - 6
        y1 = y2 - lpx
        x1, x2 = cxp - wpx // 2, cxp + wpx // 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.ego_color, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)
        # Windshield band + forward heading tick (points up = ahead of ego).
        cv2.line(canvas, (x1, y1 + max(3, lpx // 4)), (x2, y1 + max(3, lpx // 4)), (0, 0, 0), 1)
        cv2.line(canvas, (cxp, y1), (cxp, y1 - 10), self.ego_color, 2)

    def _draw_object(self, canvas: np.ndarray, obj: TrackedObject, x: int, y: int) -> None:
        """Draw one tracked object as a scaled, colour-coded top-down vehicle icon."""
        w_m, l_m = self.object_sizes.get(obj.class_name, self.object_sizes.get("default", [1.9, 4.5]))
        wpx, lpx = self._px_size(float(w_m), float(l_m))
        col = self.colors.get(obj.color_key, (0, 200, 0))
        x1, y1 = x - wpx // 2, y - lpx // 2
        x2, y2 = x + wpx // 2, y + lpx // 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(canvas, str(obj.track_id), (min(x2 + 3, self.width - 14), y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    def render(self, objects: Sequence[TrackedObject]) -> Optional[np.ndarray]:
        """Render the BEV canvas for the current frame.

        Honors ``update_every`` (returns the cached canvas on skipped frames so
        the mini-map stays visually stable without recomputing every frame).

        Args:
            objects: Tracked objects with ``distance_m`` populated.

        Returns:
            A BGR canvas of size ``(height, width)`` or ``None`` if disabled.
        """
        if not self.enabled:
            return None
        self._frame_idx += 1
        if self._last_canvas is not None and self.update_every > 1 and \
                (self._frame_idx % self.update_every) != 0:
            return self._last_canvas

        canvas = self._blank_canvas()
        for obj in objects:
            if obj.distance_m is None or not np.isfinite(obj.distance_m):
                continue
            if obj.distance_m > self.forward_m:
                continue
            lateral = self.object_lateral_m(obj)
            if abs(lateral) > self.lateral_m:
                continue
            x, y = self._metric_to_canvas(obj.distance_m, lateral)
            self._draw_object(canvas, obj, x, y)
        self._draw_ego(canvas)  # ego on top so it is never occluded
        self._last_canvas = canvas
        return canvas
