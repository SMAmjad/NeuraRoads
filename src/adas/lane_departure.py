"""Lane-departure detection with hysteresis, debounce and curve suppression.

Consumes the ego lane offset from :class:`~core.lane_detector.LaneResult`
(``-1``=left line .. ``+1``=right line) and decides whether the ego vehicle is
drifting out of lane. Three safeguards prevent false warnings:

* **Debounce** - the offset must exceed the threshold for N consecutive frames.
* **Hysteresis** - once triggered, the warning only clears after the offset
  recovers below a lower reset threshold.
* **Curve suppression** - departures are suppressed on sharp curves/roundabouts
  where large offsets are expected (configurable), avoiding nuisance alerts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.lane_detector import LaneResult
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class LaneDepartureState:
    """Current lane-departure decision.

    Attributes:
        side: ``"left"``, ``"right"`` or ``None`` (no departure).
        active: Whether a departure warning is currently latched.
        offset: The latest ego offset used (``None`` if unavailable).
        suppressed_by_curve: True when a would-be warning was suppressed.
    """

    side: Optional[str] = None
    active: bool = False
    offset: Optional[float] = None
    suppressed_by_curve: bool = False


class LaneDepartureWarning:
    """Stateful lane-departure detector (one instance per video stream)."""

    def __init__(self, adas_cfg: Dict[str, Any]) -> None:
        """Args: adas_cfg: full ``adas_thresholds`` config."""
        cfg = adas_cfg.get("lane_departure", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.warn_offset = float(cfg.get("warn_offset", 0.72))
        self.reset_offset = float(cfg.get("reset_offset", 0.55))
        self.min_frames = int(cfg.get("min_frames", 3))
        self.suppress_on_curve = bool(cfg.get("suppress_on_curve", True))
        self.curve_min_m = float(cfg.get("curve_curvature_min_m", 60.0))

        self._counter = 0
        self._active_side: Optional[str] = None

    def update(self, lane: Optional[LaneResult]) -> LaneDepartureState:
        """Advance the state machine with the latest lane result.

        Args:
            lane: The current (smoothed) lane result, or None.

        Returns:
            The current :class:`LaneDepartureState`.
        """
        if not self.enabled or lane is None or lane.ego_offset is None:
            self._counter = 0
            self._active_side = None
            return LaneDepartureState()

        offset = lane.ego_offset
        # Suppress on sharp curves/roundabouts (large offset is expected there).
        on_curve = self.suppress_on_curve and lane.curvature_m < self.curve_min_m and lane.curvature_m > 0
        if on_curve:
            self._counter = 0
            self._active_side = None
            return LaneDepartureState(side=None, active=False, offset=offset, suppressed_by_curve=True)

        # +offset means ego is left of centre -> drifting LEFT toward left line.
        side = "left" if offset > 0 else "right"
        magnitude = abs(offset)

        if self._active_side is not None:
            # Latched: clear only when recovered below reset threshold.
            if magnitude < self.reset_offset:
                self._active_side = None
                self._counter = 0
                return LaneDepartureState(side=None, active=False, offset=offset)
            return LaneDepartureState(side=self._active_side, active=True, offset=offset)

        # Not latched: debounce before firing.
        if magnitude >= self.warn_offset:
            self._counter += 1
            if self._counter >= self.min_frames:
                self._active_side = side
                return LaneDepartureState(side=side, active=True, offset=offset)
        else:
            self._counter = 0
        return LaneDepartureState(side=None, active=False, offset=offset)

    def reset(self) -> None:
        """Reset the state machine (e.g. between videos)."""
        self._counter = 0
        self._active_side = None
