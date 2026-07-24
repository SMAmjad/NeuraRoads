"""Tests for the ByteTrack object tracker wrapper."""
from __future__ import annotations

import numpy as np

from core.detector import Detection
from core.tracker import ObjectTracker, TrackedObject


def _det(box, cls_id, name, conf=0.9):
    return Detection(np.array(box, np.float32), conf, cls_id, name)


def test_ids_are_stable_across_frames(model_cfg):
    """A slowly moving object keeps the same track ID across frames."""
    tracker = ObjectTracker(model_cfg["tracker"], frame_rate=30)
    frame = np.zeros((720, 1280, 3), np.uint8)
    t1 = tracker.update([_det([600, 400, 700, 520], 2, "Car")], frame)
    t2 = tracker.update([_det([604, 402, 705, 524], 2, "Car")], frame)
    assert t1 and t2
    assert t1[0].track_id == t2[0].track_id
    assert t1[0].class_name == "Car"


def test_empty_detections_returns_empty(model_cfg):
    """No detections -> no tracks, no crash."""
    tracker = ObjectTracker(model_cfg["tracker"], frame_rate=30)
    assert tracker.update([], np.zeros((100, 100, 3), np.uint8)) == []


def test_reset_restarts(model_cfg):
    """reset() rebuilds the tracker without error."""
    tracker = ObjectTracker(model_cfg["tracker"], frame_rate=30)
    frame = np.zeros((720, 1280, 3), np.uint8)
    for _ in range(3):
        tracker.update([_det([600, 400, 700, 520], 2, "Car")], frame)
    tracker.reset()
    assert isinstance(tracker.update([_det([600, 400, 700, 520], 2, "Car")], frame), list)


def test_tracked_object_geometry():
    """TrackedObject geometry helpers are correct."""
    o = TrackedObject(1, np.array([10, 20, 30, 60], np.float32), 2, "Car", 0.9)
    assert o.center == (20.0, 40.0)
    assert o.bottom_center == (20.0, 60.0)
    assert o.width == 20.0 and o.height == 40.0
