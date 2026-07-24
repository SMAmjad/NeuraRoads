"""Camera calibration model: intrinsics, distortion and ground-plane geometry.

Wraps ``configs/camera_calibration.yaml`` into a :class:`CameraCalibration`
object used by the distance estimator (pinhole focal length), the ego-speed
scaler and the BEV transformer (ground-plane homography). Intrinsics are
auto-rescaled when the input video resolution differs from the resolution the
camera was calibrated at.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from utils.config_loader import load_config
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class CameraCalibration:
    """Camera intrinsics + mounting geometry, scaled to the active frame size.

    Attributes:
        fx, fy: Focal lengths in pixels (for the current frame size).
        cx, cy: Principal point in pixels (for the current frame size).
        dist_coeffs: OpenCV distortion vector ``[k1,k2,p1,p2,k3]`` (zeros if off).
        camera_height_m: Camera height above the road (metres).
        pitch_deg: Downward pitch of the camera (degrees).
        horizon_y_frac: Fractional image row of the road horizon.
        frame_size: Active ``(width, height)`` these values correspond to.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray
    camera_height_m: float
    pitch_deg: float
    horizon_y_frac: float
    frame_size: Tuple[int, int]
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        frame_size: Optional[Tuple[int, int]] = None,
        config_name: str = "camera_calibration",
        overlay: Optional[str] = None,
    ) -> "CameraCalibration":
        """Build a calibration from the YAML config, rescaled to ``frame_size``.

        Args:
            frame_size: Active ``(width, height)`` of the video being processed.
                If ``None``, the calibrated resolution is used unchanged.
            config_name: Calibration config name.
            overlay: Optional overlay config (e.g. jetson).

        Returns:
            A :class:`CameraCalibration` scaled to ``frame_size``.
        """
        raw = load_config(config_name, overlay)
        calib_res = tuple(raw.get("calibrated_resolution", [1920, 1080]))
        intr = raw.get("intrinsics", {})
        fx = float(intr.get("fx", 1371.0))
        fy = float(intr.get("fy", fx))
        cx = float(intr.get("cx", calib_res[0] / 2.0))
        cy = float(intr.get("cy", calib_res[1] / 2.0))

        # Derive fx from FOV if focal length missing/zero.
        if fx <= 0:
            hfov = math.radians(float(intr.get("horizontal_fov_deg", 70.0)))
            fx = fy = (calib_res[0] / 2.0) / max(math.tan(hfov / 2.0), 1e-6)

        dist = raw.get("distortion", {})
        if dist.get("enabled", False):
            dist_coeffs = np.array(
                [dist.get("k1", 0.0), dist.get("k2", 0.0), dist.get("p1", 0.0),
                 dist.get("p2", 0.0), dist.get("k3", 0.0)],
                dtype=np.float64,
            )
        else:
            dist_coeffs = np.zeros(5, dtype=np.float64)

        mount = raw.get("mounting", {})
        obj = cls(
            fx=fx, fy=fy, cx=cx, cy=cy,
            dist_coeffs=dist_coeffs,
            camera_height_m=float(mount.get("height_m", 1.25)),
            pitch_deg=float(mount.get("pitch_deg", 0.0)),
            horizon_y_frac=float(mount.get("horizon_y_frac", 0.52)),
            frame_size=tuple(calib_res),
            _raw=raw,
        )
        if frame_size is not None:
            obj.rescale(frame_size)
        return obj

    # -- geometry helpers ---------------------------------------------------
    def rescale(self, frame_size: Tuple[int, int]) -> "CameraCalibration":
        """Rescale intrinsics in place to a new ``(width, height)`` frame size.

        Args:
            frame_size: The new active frame ``(width, height)``.

        Returns:
            ``self`` (for chaining).
        """
        w, h = int(frame_size[0]), int(frame_size[1])
        cw, ch = self.frame_size
        if (w, h) == (cw, ch):
            return self
        sx, sy = w / float(cw), h / float(ch)
        self.fx *= sx
        self.fy *= sy
        self.cx *= sx
        self.cy *= sy
        self.frame_size = (w, h)
        log.debug("Rescaled intrinsics to {}x{} (sx={:.3f}, sy={:.3f}).", w, h, sx, sy)
        return self

    @property
    def camera_matrix(self) -> np.ndarray:
        """3x3 intrinsic matrix ``K`` for the active frame size."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def has_distortion(self) -> bool:
        """Whether non-zero distortion coefficients are configured."""
        return bool(np.any(np.abs(self.dist_coeffs) > 1e-9))

    def horizon_row(self) -> int:
        """Pixel row of the road horizon for the active frame."""
        return int(self.horizon_y_frac * self.frame_size[1])

    def distance_from_height(self, pixel_height: float, real_height_m: float) -> float:
        """Pinhole distance from a known real object height and its pixel height.

        ``distance = f * H_real / h_pixels``.

        Args:
            pixel_height: Bounding-box height in pixels (must be > 0).
            real_height_m: Real-world object height in metres.

        Returns:
            Estimated distance in metres (``inf`` if ``pixel_height <= 0``).
        """
        if pixel_height <= 0:
            return float("inf")
        return (self.fy * real_height_m) / pixel_height

    def distance_from_width(self, pixel_width: float, real_width_m: float) -> float:
        """Pinhole distance from a known real object width and its pixel width."""
        if pixel_width <= 0:
            return float("inf")
        return (self.fx * real_width_m) / pixel_width

    def ground_distance_from_row(self, image_row: float) -> float:
        """Distance to a ground point imaged at ``image_row`` via the flat-road model.

        Uses the camera height, pitch and focal length assuming a flat road. Rows
        at or above the horizon return ``inf``.

        Args:
            image_row: Pixel row (y) of the object's ground-contact point.

        Returns:
            Longitudinal distance in metres.
        """
        # Angle below the optical axis for this row.
        theta = math.atan2((image_row - self.cy), self.fy)
        total_angle = math.radians(self.pitch_deg) + theta
        if total_angle <= 1e-4:  # at/above horizon
            return float("inf")
        return self.camera_height_m / math.tan(total_angle)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the active intrinsics/geometry to a plain dict."""
        return {
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "dist_coeffs": self.dist_coeffs.tolist(),
            "camera_height_m": self.camera_height_m,
            "pitch_deg": self.pitch_deg,
            "horizon_y_frac": self.horizon_y_frac,
            "frame_size": list(self.frame_size),
        }
