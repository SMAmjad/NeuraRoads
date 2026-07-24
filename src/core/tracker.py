"""Multi-object tracking via ByteTrack (Ultralytics implementation).

Consumes a list of :class:`~core.detector.Detection` per frame and returns
:class:`TrackedObject` instances carrying a stable unique ``track_id``. ByteTrack
was chosen for its near-zero overhead (no appearance network) and strong ID
stability in dense traffic - the best fit for the FPS budget.

The :class:`TrackedObject` is the pipeline's central per-object record: the
distance, speed, TTC and danger modules enrich it in place, keyed by track ID.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.detector import Detection
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TrackedObject:
    """A tracked object with a persistent ID, enriched through the pipeline.

    Attributes:
        track_id: Persistent unique ID across frames.
        box: ``[x1, y1, x2, y2]`` in pixels.
        class_id: Integer class index.
        class_name: Human-readable class name.
        confidence: Latest detection confidence.
        distance_m: Estimated distance (metres), filled by distance estimator.
        speed_kmh: Relative speed (km/h, +approaching), filled by speed estimator.
        ttc_s: Time-to-collision (seconds), filled by collision detector.
        color_key: Visualization colour key, filled by the ADAS layer.
    """

    track_id: int
    box: np.ndarray
    class_id: int
    class_name: str
    confidence: float
    distance_m: Optional[float] = None
    speed_kmh: Optional[float] = None
    ttc_s: Optional[float] = None
    color_key: str = "safe"
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        return (float(self.box[0] + self.box[2]) / 2.0, float(self.box[1] + self.box[3]) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (float(self.box[0] + self.box[2]) / 2.0, float(self.box[3]))

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])

    def as_dict(self) -> Dict[str, Any]:
        """Flatten to a plain dict for the visualizer / logging."""
        return {
            "track_id": self.track_id,
            "box": self.box,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "distance_m": self.distance_m,
            "speed_kmh": self.speed_kmh,
            "ttc_s": self.ttc_s,
            "color_key": self.color_key,
        }


def _tracker_args(cfg: Dict[str, Any]) -> SimpleNamespace:
    """Build the SimpleNamespace of args Ultralytics trackers expect."""
    return SimpleNamespace(
        tracker_type=cfg.get("type", "bytetrack"),
        track_high_thresh=float(cfg.get("track_high_thresh", 0.5)),
        track_low_thresh=float(cfg.get("track_low_thresh", 0.1)),
        new_track_thresh=float(cfg.get("new_track_thresh", 0.6)),
        track_buffer=int(cfg.get("track_buffer", 30)),
        match_thresh=float(cfg.get("match_thresh", 0.8)),
        fuse_score=bool(cfg.get("fuse_score", True)),
        # BoT-SORT extras (ignored by ByteTrack, needed if type=botsort).
        gmc_method=cfg.get("gmc_method", "sparseOptFlow"),
        proximity_thresh=float(cfg.get("proximity_thresh", 0.5)),
        appearance_thresh=float(cfg.get("appearance_thresh", 0.25)),
        with_reid=bool(cfg.get("with_reid", False)),
        model=cfg.get("reid_model", "auto"),
    )


class ObjectTracker:
    """Wraps Ultralytics ByteTrack/BoT-SORT with a Detection-in / Track-out API."""

    def __init__(self, cfg: Dict[str, Any], frame_rate: float = 30.0) -> None:
        """Create the tracker.

        Args:
            cfg: The ``tracker`` block of ``model_config``.
            frame_rate: Video FPS (affects track buffer lifetime).
        """
        self.cfg = cfg or {}
        self.type = str(self.cfg.get("type", "bytetrack")).lower()
        self.min_box_area = float(self.cfg.get("min_box_area", 10))
        self.frame_rate = float(frame_rate)
        self._args = _tracker_args(self.cfg)
        self._tracker = self._build(self._args, self.frame_rate)
        log.info("Tracker ready: {} @ {:.0f} FPS (buffer={}).",
                 self.type, self.frame_rate, self._args.track_buffer)

    def _build(self, args: SimpleNamespace, frame_rate: float):
        """Instantiate the underlying Ultralytics tracker object."""
        if self.type == "botsort":
            from ultralytics.trackers.bot_sort import BOTSORT

            return BOTSORT(args, frame_rate=int(frame_rate))
        from ultralytics.trackers.byte_tracker import BYTETracker

        return BYTETracker(args, frame_rate=int(frame_rate))

    def reset(self) -> None:
        """Reset all track state (e.g. between separate videos)."""
        self._tracker = self._build(self._args, self.frame_rate)

    def update(self, detections: Sequence[Detection],
               frame: Optional[np.ndarray] = None) -> List[TrackedObject]:
        """Advance tracking by one frame.

        Args:
            detections: Detections for the current frame.
            frame: The current BGR frame (used by BoT-SORT GMC; optional for
                ByteTrack).

        Returns:
            List of :class:`TrackedObject` with stable IDs. Detections without a
            confirmed track this frame are omitted (standard ByteTrack behaviour).
        """
        # Keep a class-name lookup so we can restore readable names by index.
        names: Dict[int, str] = {}
        if not detections:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.float32)
        else:
            boxes, confs, clss = [], [], []
            for d in detections:
                w = d.box[2] - d.box[0]
                h = d.box[3] - d.box[1]
                if w * h < self.min_box_area:
                    continue
                boxes.append(d.box)
                confs.append(d.confidence)
                clss.append(d.class_id)
                names[d.class_id] = d.class_name
            xyxy = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            conf = np.asarray(confs, dtype=np.float32)
            cls = np.asarray(clss, dtype=np.float32)

        # Build the xywh view ByteTracker reads.
        if len(xyxy):
            xywh = np.zeros_like(xyxy)
            xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
            xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
            xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
            xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        else:
            xywh = np.zeros((0, 4), dtype=np.float32)

        results = SimpleNamespace(conf=conf, cls=cls, xywh=xywh, xyxy=xyxy)
        try:
            tracks = self._tracker.update(results, frame)
        except Exception as exc:
            log.error("Tracker update failed: {}", exc)
            return []

        out: List[TrackedObject] = []
        for row in np.asarray(tracks):
            # row = [x1, y1, x2, y2, track_id, score, cls, idx]
            if len(row) < 7:
                continue
            x1, y1, x2, y2 = row[0:4]
            track_id = int(row[4])
            score = float(row[5])
            class_id = int(row[6])
            out.append(TrackedObject(
                track_id=track_id,
                box=np.array([x1, y1, x2, y2], dtype=np.float32),
                class_id=class_id,
                class_name=names.get(class_id, f"cls{class_id}"),
                confidence=score,
            ))
        return out
