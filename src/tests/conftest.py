"""Pytest configuration + shared fixtures for the NeuraRoads test suite.

Puts the ``src`` package root on ``sys.path`` so tests can ``import core`` /
``import utils`` etc. regardless of the working directory, and provides small
reusable fixtures (configs, a synthetic frame, a couple of real dashcam frames).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# --- make `src` importable ---------------------------------------------------
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.config_loader import PROJECT_ROOT, load_config  # noqa: E402


@pytest.fixture(scope="session")
def model_cfg():
    """The full model_config dict."""
    return load_config("model_config")


@pytest.fixture(scope="session")
def adas_cfg():
    """The full adas_thresholds dict."""
    return load_config("adas_thresholds")


@pytest.fixture()
def synthetic_frame():
    """A deterministic 720x1280 BGR frame for size-agnostic tests."""
    rng = np.random.default_rng(0)
    return (rng.random((720, 1280, 3)) * 255).astype(np.uint8)


@pytest.fixture(scope="session")
def sample_frames():
    """Two consecutive real dashcam frames, or ``None`` if the clip is missing."""
    import cv2

    clip = PROJECT_ROOT / "data" / "raw" / "videos" / "dashcam_samples" / "lane.mp4"
    if not clip.is_file():
        return None
    cap = cv2.VideoCapture(str(clip))
    ok1, f1 = cap.read()
    ok2, f2 = cap.read()
    cap.release()
    return (f1, f2) if (ok1 and ok2) else None


@pytest.fixture(scope="session")
def has_trained_weights():
    """Whether trained detector weights exist (to skip model-dependent tests)."""
    cfg = load_config("model_config")
    from utils.config_loader import resolve_path

    return resolve_path(cfg["detector"]["weights"]).is_file()
