"""NeuraRoads shared utilities: config, logging, calibration, video I/O, metrics, HUD.

Importing this package is side-effect free beyond configuring logging on first
use. See individual modules for details.
"""
from __future__ import annotations

from utils.calibration import CameraCalibration
from utils.config_loader import (
    ConfigLoader,
    apply_torch_runtime,
    deep_merge,
    get_in,
    load_config,
    resolve_device,
    resolve_path,
)
from utils.logger import configure_logging, get_logger
from utils.metrics import (
    PerformanceTracker,
    box_center,
    box_iou,
    iou_matrix,
    system_stats,
)
from utils.video_processor import FPSMeter, VideoReader, VideoWriter, letterbox
from utils.visualization import Visualizer

__all__ = [
    "CameraCalibration",
    "ConfigLoader",
    "load_config",
    "resolve_device",
    "resolve_path",
    "get_in",
    "deep_merge",
    "apply_torch_runtime",
    "configure_logging",
    "get_logger",
    "PerformanceTracker",
    "box_iou",
    "box_center",
    "iou_matrix",
    "system_stats",
    "VideoReader",
    "VideoWriter",
    "FPSMeter",
    "letterbox",
    "Visualizer",
]
