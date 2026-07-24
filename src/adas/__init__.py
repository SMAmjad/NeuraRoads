"""NeuraRoads ADAS decision layer: collision, lane departure, warnings, alerts.

The layer is a clean pipeline: :class:`CollisionDetector` +
:class:`LaneDepartureWarning` produce raw assessments, :class:`WarningSystem`
turns them into prioritised :class:`WarningEvent` objects, and
:class:`AlertManager` handles their temporal on-screen presentation.
"""
from __future__ import annotations

from adas.alert_manager import AlertManager
from adas.collision_detector import CollisionAssessment, CollisionDetector
from adas.lane_departure import LaneDepartureState, LaneDepartureWarning
from adas.warning_system import WarningEvent, WarningSystem

__all__ = [
    "CollisionDetector",
    "CollisionAssessment",
    "LaneDepartureWarning",
    "LaneDepartureState",
    "WarningSystem",
    "WarningEvent",
    "AlertManager",
]
