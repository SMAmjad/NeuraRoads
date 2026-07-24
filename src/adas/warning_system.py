"""Aggregates danger assessments into prioritised, user-facing warning events.

Takes the pure :class:`~adas.collision_detector.CollisionAssessment`, the
:class:`~adas.lane_departure.LaneDepartureState` and the lane geometry, then
produces :class:`WarningEvent` objects with the icons, messages, colours and
pulse styles defined in ``adas_thresholds.yaml``. Priority ordering is:

    Collision > Pedestrian > Lane Departure > Direction

The single highest-priority event is the "primary" one shown center-top; the
rest populate the bottom-bar active-warnings list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from adas.collision_detector import CollisionAssessment
from adas.lane_departure import LaneDepartureState
from core.lane_detector import LaneResult
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class WarningEvent:
    """A single ADAS warning ready for display.

    Attributes:
        kind: ``"collision" | "pedestrian" | "lane_departure" | "direction"``.
        level: Sub-level string (e.g. ``"warning"``, ``"prompt"``, ``"normal"``).
        message: The text shown to the driver.
        icon: PNG filename (in ``data/icons``).
        color_key: Visualization colour key.
        priority: Higher wins the center-top slot.
        pulse: ``"none" | "slow" | "fast"``.
        is_alert: Whether this belongs in the active-warnings list (vs info).
        screen_tint: Optional full-screen tint BGR (danger flash).
        screen_tint_alpha: Tint strength 0..1.
    """

    kind: str
    level: str
    message: str
    icon: str
    color_key: str = "safe"
    priority: int = 0
    pulse: str = "none"
    is_alert: bool = False
    screen_tint: Optional[List[int]] = None
    screen_tint_alpha: float = 0.0


class WarningSystem:
    """Builds prioritised :class:`WarningEvent` lists from raw assessments."""

    def __init__(self, adas_cfg: Dict[str, Any]) -> None:
        """Args: adas_cfg: full ``adas_thresholds`` config."""
        self.cfg = adas_cfg
        self.fcws = adas_cfg.get("fcws", {})
        self.ped_cfg = adas_cfg.get("pedestrian", {})
        self.lane_cfg = adas_cfg.get("lane_departure", {})
        self.dir_cfg = adas_cfg.get("direction", {})
        self.prio = adas_cfg.get("priority", {})

    # -- individual builders ------------------------------------------------
    def _collision_event(self, assessment: CollisionAssessment) -> Optional[WarningEvent]:
        """Build the forward-collision event (or None if no in-path target)."""
        if assessment.fcws_target is None:
            return None
        levels = self.fcws.get("levels", {})
        lvl = assessment.fcws_level
        spec = levels.get(lvl, {})
        if lvl == "warning":
            return WarningEvent(
                kind="collision", level=lvl, message=spec.get("message", "COLLISION WARNING"),
                icon=spec.get("icon", "fcws-warning.png"), color_key=spec.get("color", "danger"),
                priority=int(self.prio.get("collision", 40)), pulse=spec.get("pulse", "fast"),
                is_alert=True, screen_tint=spec.get("screen_tint"),
                screen_tint_alpha=float(spec.get("screen_tint_alpha", 0.0)),
            )
        if lvl == "prompt":
            return WarningEvent(
                kind="collision", level=lvl, message=spec.get("message", "CAUTION"),
                icon=spec.get("icon", "FCWS-prompt.png"), color_key=spec.get("color", "getting_close"),
                priority=int(self.prio.get("collision", 40)), pulse=spec.get("pulse", "slow"),
                is_alert=True,
            )
        # normal: a safe lead vehicle is present -> low-priority "road clear".
        spec = levels.get("normal", {})
        return WarningEvent(
            kind="collision", level="normal", message=spec.get("message", "ROAD CLEAR"),
            icon=spec.get("icon", "FCWS-normal.png"), color_key=spec.get("color", "safe"),
            priority=12, pulse="none", is_alert=False,
        )

    def _pedestrian_event(self, assessment: CollisionAssessment) -> Optional[WarningEvent]:
        """Build the pedestrian-alert event (nearest pedestrian in range)."""
        if not assessment.pedestrian_targets:
            return None
        nearest = assessment.pedestrian_targets[0]
        near = nearest.distance_m is not None and nearest.distance_m <= float(
            self.ped_cfg.get("critical_distance_m", 6.0))
        color = self.ped_cfg.get("color_near" if near else "color_far", "too_close")
        return WarningEvent(
            kind="pedestrian", level="critical" if near else "alert",
            message=self.ped_cfg.get("message", "PEDESTRIAN ALERT"),
            icon=self.ped_cfg.get("icon", "warn.png"), color_key=color,
            priority=int(self.prio.get("pedestrian", 30)),
            pulse=self.ped_cfg.get("pulse", "slow"), is_alert=True,
        )

    def _lane_event(self, lane_state: LaneDepartureState) -> Optional[WarningEvent]:
        """Build the lane-departure event when latched active."""
        if not lane_state.active or lane_state.side is None:
            return None
        side_cfg = self.lane_cfg.get(lane_state.side, {})
        return WarningEvent(
            kind="lane_departure", level=lane_state.side,
            message=side_cfg.get("message", "LANE DEPARTURE"),
            icon=side_cfg.get("icon", "LTA-left_lanes.png"),
            color_key=side_cfg.get("color", "too_close"),
            priority=int(self.prio.get("lane_departure", 20)), pulse="slow", is_alert=True,
        )

    def _direction_event(self, lane: Optional[LaneResult]) -> WarningEvent:
        """Build the baseline direction/geometry info event (always present)."""
        prio = int(self.prio.get("direction", 10))
        straight = self.dir_cfg.get("straight", {})
        if lane is None or lane.direction in ("unknown", "straight"):
            return WarningEvent(kind="direction", level="straight",
                                message=straight.get("message", "ROAD AHEAD CLEAR"),
                                icon=straight.get("icon", "straight.png"),
                                color_key=straight.get("color", "safe"), priority=prio)
        roundabout_max = float(self.dir_cfg.get("roundabout_curvature_max_m", 30.0))
        is_round = 0 < lane.curvature_m < roundabout_max
        key = "left_turn" if lane.direction == "left" else "right_turn"
        spec = self.dir_cfg.get(key, {})
        msg = "ROUNDABOUT AHEAD" if is_round else spec.get("message", "CURVE AHEAD")
        return WarningEvent(kind="direction", level=lane.direction, message=msg,
                            icon=spec.get("icon", "straight.png"),
                            color_key=spec.get("color", "lane_normal"), priority=prio)

    # -- aggregation --------------------------------------------------------
    def evaluate(self, assessment: CollisionAssessment, lane_state: LaneDepartureState,
                 lane: Optional[LaneResult]) -> Tuple[List[WarningEvent], Optional[WarningEvent]]:
        """Produce the sorted event list and the primary (highest-priority) event.

        Args:
            assessment: Collision/pedestrian assessment for the frame.
            lane_state: Lane-departure state.
            lane: Lane geometry result (for direction).

        Returns:
            Tuple ``(events_sorted_desc, primary_event)``.
        """
        events: List[WarningEvent] = []
        for ev in (self._collision_event(assessment), self._pedestrian_event(assessment),
                   self._lane_event(lane_state), self._direction_event(lane)):
            if ev is not None:
                events.append(ev)
        events.sort(key=lambda e: e.priority, reverse=True)
        primary = events[0] if events else None
        return events, primary

    @staticmethod
    def active_warning_messages(events: List[WarningEvent], max_n: int) -> List[str]:
        """Return up to ``max_n`` alert-level messages for the bottom bar."""
        return [e.message for e in events if e.is_alert][:max_n]
