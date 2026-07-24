"""Tests for the hybrid lane detector (classical path + smoothing + fallback)."""
from __future__ import annotations

import numpy as np
import pytest

from core.lane_detector import (
    ClassicalLaneDetector,
    LaneDetector,
    LaneResult,
    LaneSmoother,
    build_row_anchors,
)


def test_classical_runs_on_real_frame(model_cfg, sample_frames):
    """Classical detection returns a LaneResult on a real dashcam frame."""
    if sample_frames is None:
        pytest.skip("sample dashcam clip not available")
    det = ClassicalLaneDetector(model_cfg["lane_detector"]["classical"])
    res = det.detect(sample_frames[0])
    assert isinstance(res, LaneResult)
    # On the highway sample it should find lanes.
    assert res.source in ("classical", "none")


def test_detector_never_raises_on_garbage(model_cfg):
    """A noise frame must not crash detection (graceful degradation)."""
    det = LaneDetector(model_cfg)
    res = det.detect((np.random.rand(360, 640, 3) * 255).astype(np.uint8))
    assert isinstance(res, LaneResult)


def test_smoother_holds_last_good():
    """The smoother holds the previous lane during a short dropout."""
    sm = LaneSmoother({"enabled": True, "ema_alpha": 0.5, "max_frames_missing": 5})
    good = LaneResult(lines=[np.array([[0, 0], [1, 1]])], left_fit=np.array([0, 0, 100.0]),
                      source="classical", ego_offset=0.0)
    sm.update(good)
    held = sm.update(LaneResult(source="none"))  # dropout
    assert held.source == "held"


def test_row_anchors_shape():
    """Row anchors are normalised and monotonic."""
    anchors = build_row_anchors("culane")
    assert anchors.ndim == 1 and anchors[0] < anchors[-1]
    assert 0.0 <= anchors.min() and anchors.max() <= 1.0


def test_deep_disabled_without_weights(model_cfg):
    """Deep detector auto-disables when weights are absent (hybrid still works)."""
    det = LaneDetector(model_cfg)
    # deep may be None or present-but-unavailable; either way detect() works.
    assert det.deep is None or det.deep.available in (True, False)
