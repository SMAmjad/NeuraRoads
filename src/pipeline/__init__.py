"""NeuraRoads pipeline package: the single-frame engine and the real-time runner."""
from __future__ import annotations

from pipeline.inference_pipeline import NeuraRoadsPipeline
from pipeline.realtime_pipeline import RealtimePipeline

__all__ = ["NeuraRoadsPipeline", "RealtimePipeline"]
