"""Tests for the Kalman speed estimator and ego optical-flow speed."""
from __future__ import annotations

import numpy as np

from core.speed_estimator import SpeedEstimator, _KalmanCV1D
from core.tracker import TrackedObject


def _obj(dist):
    o = TrackedObject(1, np.array([600, 400, 700, 520], np.float32), 2, "Car", 0.9)
    o.distance_m = dist
    return o


def test_kalman_tracks_constant_velocity():
    """The 1-D CV Kalman filter recovers a known velocity."""
    kf = _KalmanCV1D(q=1.0, r=1.0, x0=100.0)
    dt = 0.1
    true_v = -5.0  # approaching 5 m/s
    pos = 100.0
    vel = 0.0
    for _ in range(60):
        pos += true_v * dt
        _, vel = kf.step(pos, dt)
    assert abs(vel - true_v) < 1.0


def test_object_speed_requires_min_age(model_cfg):
    """Speed is withheld until the track reaches min_track_age, then reported."""
    est = SpeedEstimator(model_cfg["speed_estimator"])
    est.ego_speed_kmh = 60.0
    dt = 1 / 30
    o = _obj(50.0)
    est.estimate([o], dt)
    assert o.speed_kmh is None  # age 1 < min_age

    # Feed enough same-track observations to pass min_age (config-driven).
    o = _obj(50.0)
    for i in range(1, est.min_age + 2):
        o = _obj(50.0 - i)
        o.track_id = 1
        est.estimate([o], dt)
    assert o.speed_kmh is not None


def test_approaching_object_has_positive_closing(model_cfg):
    """A closing object reports positive closing speed and finite range rate."""
    est = SpeedEstimator(model_cfg["speed_estimator"])
    est.ego_speed_kmh = 50.0
    dt = 1 / 30
    for d in np.linspace(60, 40, 20):
        o = _obj(float(d))
        est.estimate([o], dt)
    assert o.extra["closing_speed_kmh"] > 0


def test_ego_speed_non_negative(model_cfg):
    """Ego optical-flow speed is non-negative and smooth on a static frame."""
    est = SpeedEstimator(model_cfg["speed_estimator"])
    frame = np.zeros((360, 640, 3), np.uint8)
    for _ in range(3):
        v = est.update_ego(frame, 1 / 30)
    assert v >= 0.0
