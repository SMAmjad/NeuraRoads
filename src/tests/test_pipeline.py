"""Integration tests for the end-to-end pipeline (runs without a trained model)."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.inference_pipeline import NeuraRoadsPipeline


def test_pipeline_processes_synthetic_frame():
    """process_frame returns an annotated frame + state on a synthetic input."""
    p = NeuraRoadsPipeline(allow_no_detector=True, frame_rate=30)
    frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    annotated, state = p.process_frame(frame, 1 / 30)
    assert annotated.shape[2] == 3
    assert "objects" in state and "fps" in state and "ego_speed_kmh" in state


def test_pipeline_processes_real_frames(sample_frames):
    """The pipeline runs on real dashcam frames and produces a HUD-sized frame."""
    if sample_frames is None:
        pytest.skip("sample dashcam clip not available")
    p = NeuraRoadsPipeline(allow_no_detector=True, frame_rate=25)
    for f in sample_frames:
        annotated, state = p.process_frame(f, 1 / 25)
    assert annotated.shape[0] > 0 and annotated.shape[1] > 0
    assert isinstance(state["warnings"], list)


def test_pipeline_reset_is_clean():
    """reset() clears state without error and processing resumes."""
    p = NeuraRoadsPipeline(allow_no_detector=True, frame_rate=30)
    frame = (np.random.rand(360, 640, 3) * 255).astype(np.uint8)
    p.process_frame(frame, 1 / 30)
    p.reset()
    annotated, _ = p.process_frame(frame, 1 / 30)
    assert annotated is not None


def test_alert_always_present(sample_frames):
    """An ADAS icon/alert is always produced (baseline direction event)."""
    if sample_frames is None:
        pytest.skip("sample dashcam clip not available")
    p = NeuraRoadsPipeline(allow_no_detector=True, frame_rate=25)
    _, state = p.process_frame(sample_frames[0], 1 / 25)
    assert state["alert"] is not None
