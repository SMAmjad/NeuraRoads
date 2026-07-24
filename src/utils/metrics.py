"""Runtime performance tracking, per-stage timing and shared geometry helpers.

Two responsibilities:

* :class:`PerformanceTracker` / :class:`StageTimer` measure the time each
  pipeline stage takes, compute FPS and dump a CSV performance log.
* Small vectorised box-geometry helpers (IoU, centres, containment) shared by
  the distance / speed / collision / tracking modules so the maths lives in one
  place.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Union

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


# ===========================================================================
# Geometry helpers (a.k.a. detection metrics)
# ===========================================================================
def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-Union of two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def box_center(box: Sequence[float]) -> tuple[float, float]:
    """Return the ``(cx, cy)`` centre of an ``[x1, y1, x2, y2]`` box."""
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def box_wh(box: Sequence[float]) -> tuple[float, float]:
    """Return ``(width, height)`` of an ``[x1, y1, x2, y2]`` box."""
    return max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])


def box_bottom_center(box: Sequence[float]) -> tuple[float, float]:
    """Return the ground-contact point ``(cx, y2)`` of a box."""
    return (box[0] + box[2]) / 2.0, box[3]


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU matrix between two arrays of boxes ``(N,4)`` and ``(M,4)``.

    Args:
        boxes_a: ``(N, 4)`` array of ``[x1,y1,x2,y2]``.
        boxes_b: ``(M, 4)`` array of ``[x1,y1,x2,y2]``.

    Returns:
        ``(N, M)`` IoU matrix.
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


# ===========================================================================
# Performance tracking
# ===========================================================================
class StageTimer:
    """Context manager that records elapsed time for one named stage.

    Example::

        with tracker.stage("detect"):
            dets = detector(frame)
    """

    def __init__(self, tracker: "PerformanceTracker", name: str) -> None:
        self.tracker = tracker
        self.name = name
        self._t0 = 0.0

    def __enter__(self) -> "StageTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.tracker.record(self.name, (time.perf_counter() - self._t0) * 1000.0)


class PerformanceTracker:
    """Collects per-stage timings and end-to-end FPS over a run.

    Keeps a rolling window of recent frame times for a live FPS read plus running
    totals for a final summary and optional CSV export.
    """

    def __init__(self, window: int = 60) -> None:
        """Args: window: Number of recent frames used for the rolling FPS."""
        self._stage_ms: Dict[str, float] = defaultdict(float)
        self._stage_ms_total: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self._frame_times: Deque[float] = deque(maxlen=window)
        self._rows: List[Dict[str, float]] = []
        self._frame_t0: Optional[float] = None
        self.frame_index: int = 0

    def stage(self, name: str) -> StageTimer:
        """Return a :class:`StageTimer` context manager for ``name``."""
        return StageTimer(self, name)

    def record(self, name: str, ms: float) -> None:
        """Record ``ms`` milliseconds spent in stage ``name`` for this frame."""
        self._stage_ms[name] = ms
        self._stage_ms_total[name] += ms
        self._counts[name] += 1

    def frame_start(self) -> None:
        """Mark the beginning of a frame (resets per-frame stage buffer)."""
        self._frame_t0 = time.perf_counter()
        self._stage_ms.clear()

    def frame_end(self) -> float:
        """Close the current frame, update FPS and log a CSV row. Returns FPS."""
        if self._frame_t0 is None:
            return 0.0
        dt = time.perf_counter() - self._frame_t0
        self._frame_times.append(dt)
        self.frame_index += 1
        row = {"frame": float(self.frame_index), "total_ms": dt * 1000.0, "fps": (1.0 / dt if dt > 0 else 0.0)}
        row.update(self._stage_ms)
        self._rows.append(row)
        return row["fps"]

    @property
    def fps(self) -> float:
        """Rolling average FPS over the recent window."""
        if not self._frame_times:
            return 0.0
        avg = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg if avg > 0 else 0.0

    def averages(self) -> Dict[str, float]:
        """Mean milliseconds per stage over the whole run."""
        return {k: self._stage_ms_total[k] / max(1, self._counts[k]) for k in self._stage_ms_total}

    def summary(self) -> Dict[str, float]:
        """Return a summary dict (avg FPS + per-stage means)."""
        out: Dict[str, float] = {"avg_fps": self.fps, "frames": float(self.frame_index)}
        for k, v in self.averages().items():
            out[f"{k}_ms"] = v
        return out

    def save_csv(self, path: Union[str, Path]) -> Optional[Path]:
        """Write the per-frame timing log to CSV. Returns the path or None."""
        if not self._rows:
            return None
        try:
            import pandas as pd

            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self._rows).to_csv(out, index=False)
            log.info("Saved performance log ({} frames) -> {}", len(self._rows), out)
            return out
        except Exception as exc:  # never crash on logging
            log.warning("Could not save performance CSV: {}", exc)
            return None

    def log_summary(self) -> None:
        """Emit a human-readable summary line to the log."""
        s = self.summary()
        parts = [f"avg_fps={s['avg_fps']:.1f}", f"frames={int(s['frames'])}"]
        for k, v in self.averages().items():
            parts.append(f"{k}={v:.1f}ms")
        log.info("Performance: {}", " | ".join(parts))


def system_stats() -> Dict[str, float]:
    """Return current CPU%, RAM% and (if available) GPU memory used (MB).

    Uses psutil for CPU/RAM and torch for GPU memory. Any missing dependency is
    silently skipped.

    Returns:
        Dict with keys ``cpu_percent``, ``ram_percent`` and optionally
        ``gpu_mem_mb`` / ``gpu_util_percent``.
    """
    stats: Dict[str, float] = {}
    try:
        import psutil

        stats["cpu_percent"] = psutil.cpu_percent()
        stats["ram_percent"] = psutil.virtual_memory().percent
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            stats["gpu_mem_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return stats
