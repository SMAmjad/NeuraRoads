"""Forward-collision + pedestrian danger assessment (pure geometry / kinematics).

This module does the safety-critical *maths* only - it fills each tracked
object's ``ttc_s`` and ``color_key`` and reports which objects are threats. It
deliberately knows nothing about icons or messages; the :mod:`adas.warning_system`
turns the assessment into user-facing warnings using the ADAS config.

TTC (time-to-collision) is ``distance / closing_speed`` using the Kalman-filtered
range rate from the speed estimator, so it is smooth and reacts early.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.tracker import TrackedObject
from utils.logger import get_logger

log = get_logger(__name__)

KMH_TO_MPS = 1.0 / 3.6


@dataclass
class CollisionAssessment:
    """Result of one frame's danger assessment.

    Attributes:
        fcws_level: ``"normal" | "prompt" | "warning"``.
        fcws_target: The primary in-path threat object (or None).
        pedestrian_targets: Pedestrian threats within range, nearest first.
        closest: All collision-relevant objects sorted by distance (for the HUD).
    """

    fcws_level: str = "normal"
    fcws_target: Optional[TrackedObject] = None
    pedestrian_targets: List[TrackedObject] = field(default_factory=list)
    closest: List[TrackedObject] = field(default_factory=list)


class CollisionDetector:
    """Assigns TTC + danger colour to objects and finds forward/pedestrian threats."""

    def __init__(self, model_cfg: Dict[str, Any], adas_cfg: Dict[str, Any]) -> None:
        """Create the detector.

        Args:
            model_cfg: Full ``model_config`` (for class groups).
            adas_cfg: Full ``adas_thresholds`` config.
        """
        self.groups = model_cfg.get("detector", {}).get("groups", {})
        self.collision_ids = set(self.groups.get("collision_relevant", []))
        self.pedestrian_ids = set(self.groups.get("pedestrians", []))

        self.bands: List[Dict[str, Any]] = adas_cfg.get("box_color_bands", [])
        self.fcws = adas_cfg.get("fcws", {})
        self.ped_cfg = adas_cfg.get("pedestrian", {})

        self.fcws_enabled = bool(self.fcws.get("enabled", True))
        self.in_path_frac = float(self.fcws.get("in_path_width_frac", 0.55))
        self.in_path_min_y = float(self.fcws.get("in_path_min_y_frac", 0.35))
        levels = self.fcws.get("levels", {})
        self.prompt_dist = levels.get("prompt", {}).get("distance_m", [10.0, 20.0])
        self.prompt_ttc = levels.get("prompt", {}).get("ttc_s", [3.0, 5.0])
        self.warn_dist = levels.get("warning", {}).get("distance_m", [0.0, 10.0])
        self.warn_ttc = levels.get("warning", {}).get("ttc_s", [0.0, 3.0])
        # A vehicle you are merely FOLLOWING (gap not shrinking) must not raise a
        # forward-collision alert on proximity alone. Distance-based triggers
        # therefore require the gap to be closing, except inside a hard "too
        # close regardless" floor (genuine tailgating).
        self.min_closing_kmh = float(self.fcws.get("min_closing_kmh", 4.0))
        self.hard_warn_distance_m = float(self.fcws.get("hard_warn_distance_m", 6.0))

        self.ped_enabled = bool(self.ped_cfg.get("enabled", True))
        self.ped_trigger = float(self.ped_cfg.get("trigger_distance_m", 15.0))
        self.ped_critical = float(self.ped_cfg.get("critical_distance_m", 6.0))
        self.ped_path_frac = float(self.ped_cfg.get("in_path_width_frac", 0.7))
        self.ped_color_far = str(self.ped_cfg.get("color_far", "too_close"))
        self.ped_color_near = str(self.ped_cfg.get("color_near", "danger"))

    # -- per-object helpers -------------------------------------------------
    @staticmethod
    def ttc(distance_m: Optional[float], closing_kmh: Optional[float]) -> float:
        """Time-to-collision (s). ``inf`` if not approaching or unknown."""
        if distance_m is None or closing_kmh is None or closing_kmh <= 0:
            return float("inf")
        closing_mps = closing_kmh * KMH_TO_MPS
        if closing_mps <= 1e-3:
            return float("inf")
        return float(distance_m / closing_mps)

    def _color_band(self, distance_m: Optional[float], ttc_s: float) -> str:
        """Pick the colour key for the stricter (nearer/sooner) of distance/TTC."""
        d = distance_m if distance_m is not None and np.isfinite(distance_m) else 1e9
        for band in self.bands:
            if d <= band.get("distance_max_m", 1e9) or ttc_s <= band.get("ttc_max_s", 1e9):
                return band.get("color_key", "safe")
        return "safe"

    def _in_path(self, obj: TrackedObject, frame_size: Tuple[int, int], width_frac: float) -> bool:
        """Whether ``obj`` lies in the ego forward corridor."""
        w, h = frame_size
        cx, _ = obj.center
        _, y_bottom = obj.bottom_center
        half = width_frac * w / 2.0
        return abs(cx - w / 2.0) <= half and y_bottom >= self.in_path_min_y * h

    # -- main assessment ----------------------------------------------------
    def assess(self, objects: Sequence[TrackedObject],
               frame_size: Tuple[int, int]) -> CollisionAssessment:
        """Assess all objects; fills ``ttc_s``/``color_key`` and returns threats.

        Args:
            objects: Tracked objects with distance + speed already populated.
            frame_size: ``(width, height)`` of the processing frame.

        Returns:
            A :class:`CollisionAssessment`.
        """
        assessment = CollisionAssessment()
        fcws_best: Optional[Tuple[float, TrackedObject]] = None  # (severity_key, obj)
        peds: List[TrackedObject] = []

        for obj in objects:
            closing = obj.extra.get("closing_speed_kmh")
            ttc_val = self.ttc(obj.distance_m, closing)
            obj.ttc_s = round(ttc_val, 1) if np.isfinite(ttc_val) else None
            obj.color_key = self._color_band(obj.distance_m, ttc_val)

            # Pedestrian threats.
            if self.ped_enabled and obj.class_id in self.pedestrian_ids and obj.distance_m is not None:
                if obj.distance_m <= self.ped_trigger and self._in_path(obj, frame_size, self.ped_path_frac):
                    # Colour ramps orange -> red as they approach.
                    obj.color_key = self.ped_color_near if obj.distance_m <= self.ped_critical else self.ped_color_far
                    peds.append(obj)

            # Forward-collision candidates (vehicles + anything hittable ahead).
            if self.fcws_enabled and obj.class_id in self.collision_ids and obj.distance_m is not None:
                if self._in_path(obj, frame_size, self.in_path_frac):
                    # Severity ordering key: lower is more dangerous.
                    sev = min(obj.distance_m, (ttc_val * 5.0 if np.isfinite(ttc_val) else 1e9))
                    if fcws_best is None or sev < fcws_best[0]:
                        fcws_best = (sev, obj)

        # FCWS level from the primary threat.
        if fcws_best is not None:
            obj = fcws_best[1]
            assessment.fcws_target = obj
            assessment.fcws_level = self._fcws_level(obj)
        assessment.pedestrian_targets = sorted(peds, key=lambda o: o.distance_m or 1e9)

        assessment.closest = sorted(
            [o for o in objects if o.class_id in self.collision_ids and o.distance_m is not None],
            key=lambda o: o.distance_m or 1e9,
        )
        return assessment

    def _fcws_level(self, obj: TrackedObject) -> str:
        """Classify an in-path object into normal/prompt/warning.

        TTC-based triggers already imply the gap is closing (TTC is ``inf`` for a
        non-approaching object). Distance-based triggers additionally require a
        real closing speed so that steadily *following* a car never fires an
        alert - the exception being the hard ``hard_warn_distance_m`` floor.
        """
        d = obj.distance_m if obj.distance_m is not None else 1e9
        ttc_val = obj.ttc_s if obj.ttc_s is not None else float("inf")
        closing = obj.extra.get("closing_speed_kmh") or 0.0
        is_closing = closing >= self.min_closing_kmh

        # Imminent: short TTC (closing) or dangerously close no matter what.
        if ttc_val <= self.warn_ttc[1] or d <= self.hard_warn_distance_m:
            return "warning"
        # Caution: a short-ish TTC, or moderately close AND still closing in.
        if ttc_val <= self.prompt_ttc[1]:
            return "prompt"
        if d <= self.prompt_dist[1] and is_closing:
            return "prompt"
        return "normal"
