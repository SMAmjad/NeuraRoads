"""NeuraRoads core perception modules.

Detection, tracking, distance, speed, lane detection and bird's-eye view. Each
module is independently usable and configured from ``model_config``.
"""
from __future__ import annotations

from core.bev_transformer import BEVTransformer
from core.detector import Detection, Detector
from core.distance_estimator import DistanceEstimator
from core.lane_detector import LaneDetector, LaneResult
from core.speed_estimator import EgoSpeedEstimator, SpeedEstimator
from core.tracker import ObjectTracker, TrackedObject

__all__ = [
    "Detector",
    "Detection",
    "ObjectTracker",
    "TrackedObject",
    "DistanceEstimator",
    "SpeedEstimator",
    "EgoSpeedEstimator",
    "LaneDetector",
    "LaneResult",
    "BEVTransformer",
]
