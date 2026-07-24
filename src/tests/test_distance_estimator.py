"""Tests for the pinhole distance estimator."""
from __future__ import annotations

import numpy as np

from core.distance_estimator import DistanceEstimator
from core.tracker import TrackedObject
from utils.calibration import CameraCalibration


def _obj(box, cls_id, name):
    return TrackedObject(1, np.array(box, np.float32), cls_id, name, 0.9)


def test_pinhole_distance_matches_formula(model_cfg):
    """distance = fy * real_height / pixel_height (within blending tolerance)."""
    calib = CameraCalibration.from_config(frame_size=(1280, 720))
    est = DistanceEstimator(model_cfg["distance_estimator"], calib)
    # Car (1.5m real) with 100px box height near frame bottom.
    obj = _obj([600, 400, 700, 500], 2, "Car")
    est.estimate([obj])
    expected = calib.fy * 1.5 / 100.0
    assert obj.distance_m is not None
    # Blended with ground model, so allow a generous band.
    assert 0.3 * expected <= obj.distance_m <= 2.0 * expected


def test_farther_objects_have_larger_distance(model_cfg):
    """Smaller (farther) boxes yield larger distances."""
    calib = CameraCalibration.from_config(frame_size=(1280, 720))
    est = DistanceEstimator(model_cfg["distance_estimator"], calib)
    near = _obj([600, 400, 700, 520], 2, "Car")   # 120px tall
    far = _obj([600, 400, 640, 440], 2, "Car")    # 40px tall
    est.estimate([near, far])
    assert far.distance_m > near.distance_m


def test_distance_is_clamped(model_cfg):
    """Distance never exceeds the configured maximum."""
    calib = CameraCalibration.from_config(frame_size=(1280, 720))
    est = DistanceEstimator(model_cfg["distance_estimator"], calib)
    tiny = _obj([600, 300, 601, 301], 2, "Car")   # 1px tall -> very far
    est.estimate([tiny])
    assert tiny.distance_m <= est.max_d


def test_height_lookup(model_cfg):
    """Real heights come from config; unknown class falls back to default."""
    calib = CameraCalibration.from_config(frame_size=(1280, 720))
    est = DistanceEstimator(model_cfg["distance_estimator"], calib)
    assert est.height_for(9) == 3.8   # Truck
    assert est.height_for(4) == 1.7   # Pedestrian
    assert est.height_for(999) == est.default_height
