"""Tests for the YOLOv8 detector wrapper (model-dependent tests skip w/o weights)."""
from __future__ import annotations

import numpy as np
import pytest

from core.detector import Detection, Detector


def test_detection_dataclass_geometry():
    """Detection exposes correct geometry helpers."""
    d = Detection(np.array([10, 20, 110, 220], np.float32), 0.9, 2, "Car")
    assert d.width == 100.0 and d.height == 200.0
    assert d.center == (60.0, 120.0)
    assert d.class_name == "Car"


def test_missing_weights_raises(model_cfg, has_trained_weights):
    """With no trained weights, constructing the detector raises a clear error."""
    if has_trained_weights:
        pytest.skip("weights present; skip the missing-weights path")
    with pytest.raises(FileNotFoundError):
        Detector(model_cfg)


def test_class_names_are_readable(model_cfg):
    """Config class names map indices to human-readable names."""
    names = {int(k): v for k, v in model_cfg["detector"]["class_names"].items()}
    assert names[2] == "Car"
    assert names[9] == "Truck"
    assert names[4] == "Pedestrian"
    assert len(names) == 10


def test_detect_on_frame(model_cfg, has_trained_weights):
    """End-to-end detection when a trained model is available (else skipped)."""
    if not has_trained_weights:
        pytest.skip("no trained weights yet - run train_yolo.py first")
    det = Detector(model_cfg)
    out = det.detect(np.zeros((640, 640, 3), np.uint8))
    assert isinstance(out, list)
