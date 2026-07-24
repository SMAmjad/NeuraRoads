"""Hybrid lane detection: deep (UFLDv2) primary + robust classical fallback + smoothing.

Design goals (client-critical): reliable lanes on straight/curved roads,
roundabouts, heavy traffic, night, rain, faded markings and unmarked roads, with
NO flicker and NO false departure warnings.

Architecture
------------
* :class:`ClassicalLaneDetector` - the always-available workhorse and fallback.
  Multi-colour-space masks (HLS white/yellow + LAB) fused with edges inside a
  perspective-warped ROI, sliding-window polynomial fit, and a road-EDGE mode
  for unmarked roads.
* :class:`DeepLaneDetector` - a UFLDv2 wrapper that activates only when trained
  weights are present; on any load/inference problem it disables itself so the
  hybrid transparently falls back to classical.
* :class:`LaneSmoother` - temporal Kalman/EMA on the polynomial coefficients,
  holding the last good lane for a few frames so output stays smooth.
* :class:`LaneDetector` - the manager that ties them together and computes the
  ego lane offset, curvature and turn/roundabout direction consumed by the ADAS
  layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.config_loader import resolve_device, resolve_path
from utils.logger import get_logger

log = get_logger(__name__)

# UFLDv2 grid / row-anchor configuration per training dataset. These match the
# official Ultra-Fast-Lane-Detection-v2 CULane/TuSimple setups so official
# checkpoints load. Values: (num_grid_row, num_row_anchors, num_lanes).
_UFLDV2_CFG: Dict[str, Dict[str, Any]] = {
    "culane": {"num_grid_row": 200, "num_row": 72, "num_lanes": 4,
               "crop_ratio": 0.6, "row_anchor_start": 0.42},
    "tusimple": {"num_grid_row": 100, "num_row": 56, "num_lanes": 4,
                 "crop_ratio": 0.8, "row_anchor_start": 0.16},
}


def build_row_anchors(dataset: str) -> np.ndarray:
    """Return the normalised row-anchor positions (0..1) for a UFLDv2 dataset."""
    conf = _UFLDV2_CFG.get(dataset, _UFLDV2_CFG["culane"])
    start = conf["row_anchor_start"]
    return np.linspace(start, 1.0, conf["num_row"], dtype=np.float32)


# ===========================================================================
# Result container
# ===========================================================================
@dataclass
class LaneResult:
    """Per-frame lane detection output in original image coordinates.

    Attributes:
        lines: List of polylines (each ``(N, 2)`` int array) to draw.
        fill: Optional drivable-area polygon ``(M, 2)`` for shading.
        ego_offset: Lateral position in lane, ``-1``=left line, ``0``=centre,
            ``+1``=right line. ``None`` if unknown.
        curvature_m: Estimated radius of curvature (m); large => straight.
        direction: ``"straight" | "left" | "right" | "roundabout" | "unknown"``.
        source: ``"deep" | "classical" | "held" | "none"``.
        confidence: 0..1 quality score.
    """

    lines: List[np.ndarray] = field(default_factory=list)
    fill: Optional[np.ndarray] = None
    left_fit: Optional[np.ndarray] = None
    right_fit: Optional[np.ndarray] = None
    ego_offset: Optional[float] = None
    curvature_m: float = float("inf")
    direction: str = "unknown"
    source: str = "none"
    confidence: float = 0.0
    # Deep (YOLOP) segmentation masks in ORIGINAL frame coords (uint8 0/1):
    # drivable area and lane lines. Rendered directly for a rich overlay.
    drive_mask: Optional[np.ndarray] = None
    lane_mask: Optional[np.ndarray] = None

    def is_valid(self) -> bool:
        """Whether a usable lane was found this frame."""
        if self.source in ("none",):
            return False
        return (self.left_fit is not None or self.right_fit is not None
                or len(self.lines) > 0 or self.fill is not None
                or self.drive_mask is not None or self.lane_mask is not None)


# ===========================================================================
# Perspective helpers
# ===========================================================================
class _Warper:
    """Caches forward/inverse perspective transforms for a fixed frame size."""

    def __init__(self, frame_size: Tuple[int, int], roi: Dict[str, float]) -> None:
        w, h = frame_size
        self.size = (w, h)
        top_y = roi.get("roi_top_y", 0.60) * h
        bot_y = roi.get("roi_bottom_y", 0.95) * h
        top_w = roi.get("roi_top_width", 0.18) * w
        bot_w = roi.get("roi_bottom_width", 0.90) * w
        cx = w / 2.0
        self.src = np.float32([
            [cx - top_w / 2, top_y], [cx + top_w / 2, top_y],
            [cx + bot_w / 2, bot_y], [cx - bot_w / 2, bot_y],
        ])
        self.dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def warp(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(img, self.M, self.size, flags=cv2.INTER_LINEAR)

    def unwarp_points(self, pts: np.ndarray) -> np.ndarray:
        """Map warped ``(N,2)`` points back to the original image."""
        if len(pts) == 0:
            return pts
        p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.Minv).reshape(-1, 2)


# ===========================================================================
# Classical detector (robust workhorse + fallback)
# ===========================================================================
class ClassicalLaneDetector:
    """Colour + edge + sliding-window lane detector with road-edge fallback."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Args: cfg: the ``lane_detector.classical`` block."""
        self.cfg = cfg or {}
        self.use_color = bool(self.cfg.get("use_color_masks", True))
        self.canny_low = int(self.cfg.get("canny_low", 50))
        self.canny_high = int(self.cfg.get("canny_high", 150))
        self.edge_fallback = bool(self.cfg.get("road_edge_fallback", True))
        # Plausibility / draw gating (kills "lanes flying into the sky"):
        #  * only draw the reliable NEAR field, trimming the far/horizon zone
        #    where tiny poly errors get magnified on unwarp.
        #  * reject fits that cross, are implausibly wide/narrow, or wander
        #    near-horizontally (fitted to buildings/kerbs instead of the lane).
        self.draw_top_frac = float(self.cfg.get("draw_top_frac", 0.35))
        self.min_lane_width_frac = float(self.cfg.get("min_lane_width_frac", 0.25))
        self.max_lane_width_frac = float(self.cfg.get("max_lane_width_frac", 1.30))
        self.max_lane_dx_frac = float(self.cfg.get("max_lane_dx_frac", 0.55))
        self.max_single_dx_frac = float(self.cfg.get("max_single_dx_frac", 0.30))
        self.require_pair = bool(self.cfg.get("require_pair", True))
        # Colour is the primary cue; only fold in raw edges when colour is sparse.
        self.use_edges_assist = bool(self.cfg.get("use_edges_assist", True))
        self.min_color_pixels = int(self.cfg.get("min_color_pixels", 1500))
        # Temporal accumulation of the warped mask: dashed markings are sparse in
        # any single frame, so we fade-blend recent frames to knit the dashes into
        # continuous lines and recover BOTH boundaries (not just one).
        self.accum_decay = float(self.cfg.get("accum_decay", 0.8))
        self.accum_thresh = int(self.cfg.get("accum_thresh", 40))
        self._accum: Optional[np.ndarray] = None
        self._warper: Optional[_Warper] = None
        self._frame_size: Optional[Tuple[int, int]] = None

    def _ensure_warper(self, frame_size: Tuple[int, int]) -> _Warper:
        if self._warper is None or self._frame_size != frame_size:
            self._warper = _Warper(frame_size, self.cfg)
            self._frame_size = frame_size
            self._accum = None
        return self._warper

    def _accumulate(self, warped: np.ndarray) -> np.ndarray:
        """Fade-blend the warped mask over time so dashed lines become solid."""
        warped_f = warped.astype(np.float32)
        if self._accum is None or self._accum.shape != warped_f.shape:
            self._accum = warped_f
        else:
            self._accum = np.maximum(warped_f, self._accum * self.accum_decay)
        return (self._accum >= self.accum_thresh).astype(np.uint8) * 255

    def reset(self) -> None:
        """Clear temporal accumulation (call between separate videos)."""
        self._accum = None

    def _binary_mask(self, frame: np.ndarray) -> np.ndarray:
        """Isolate lane MARKINGS via colour, robust to night/rain/faded lines.

        Lane paint is bright white/yellow; keying on colour (after CLAHE) avoids
        importing the tree/guard-rail/vehicle edges that a raw full-frame Canny
        would add - those clutter the fit and make lanes "fly" off the road. Raw
        edges are reserved for :meth:`_road_edge_fallback` (unmarked roads only).
        """
        # Normalise brightness (helps night / glare / rain).
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        norm = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)

        combined = np.zeros(gray.shape, np.uint8)
        if self.use_color:
            hls = cv2.cvtColor(norm, cv2.COLOR_BGR2HLS)
            white = cv2.inRange(hls, (0, 150, 0), (255, 255, 90))
            yellow = cv2.inRange(hls, (15, 40, 60), (45, 220, 255))
            yellow_b = cv2.inRange(b, 150, 255)
            for m in (white, yellow, yellow_b):
                combined = cv2.bitwise_or(combined, m)

        # If colour found little (faded markings / dusk), add thin bright edges as
        # a gentle assist - Sobel-x on the normalised luma keeps mostly vertical
        # marking gradients rather than horizontal clutter.
        if self.use_edges_assist and cv2.countNonZero(combined) < self.min_color_pixels:
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, self.canny_low, self.canny_high)
            combined = cv2.bitwise_or(combined, edges)

        # Morphological open then close to drop speckle and knit dashes together.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)
        return combined

    def _sliding_window(self, binary_warped: np.ndarray,
                        nwindows: int = 9, margin: int = 80, minpix: int = 40
                        ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Fit 2nd-order polynomials to left/right lanes in warped binary space.

        Returns ``(left_fit, right_fit, confidence)`` where fits are in warped
        coordinates (``x = a*y^2 + b*y + c``).
        """
        h, w = binary_warped.shape
        histogram = np.sum(binary_warped[h // 2:, :], axis=0)
        if histogram.max() < 1:
            return None, None, 0.0
        mid = w // 2
        leftx_base = int(np.argmax(histogram[:mid]))
        rightx_base = int(np.argmax(histogram[mid:]) + mid)

        window_h = h // nwindows
        nonzero = binary_warped.nonzero()
        nz_y, nz_x = np.array(nonzero[0]), np.array(nonzero[1])
        leftx_cur, rightx_cur = leftx_base, rightx_base
        left_idx: List[np.ndarray] = []
        right_idx: List[np.ndarray] = []

        for win in range(nwindows):
            y_lo = h - (win + 1) * window_h
            y_hi = h - win * window_h
            good_l = ((nz_y >= y_lo) & (nz_y < y_hi) &
                      (nz_x >= leftx_cur - margin) & (nz_x < leftx_cur + margin)).nonzero()[0]
            good_r = ((nz_y >= y_lo) & (nz_y < y_hi) &
                      (nz_x >= rightx_cur - margin) & (nz_x < rightx_cur + margin)).nonzero()[0]
            left_idx.append(good_l)
            right_idx.append(good_r)
            if len(good_l) > minpix:
                leftx_cur = int(np.mean(nz_x[good_l]))
            if len(good_r) > minpix:
                rightx_cur = int(np.mean(nz_x[good_r]))

        left_idx_arr = np.concatenate(left_idx) if left_idx else np.array([], int)
        right_idx_arr = np.concatenate(right_idx) if right_idx else np.array([], int)

        left_fit = right_fit = None
        conf_l = conf_r = 0.0
        if len(left_idx_arr) > 200:
            left_fit = np.polyfit(nz_y[left_idx_arr], nz_x[left_idx_arr], 2)
            conf_l = min(1.0, len(left_idx_arr) / 4000.0)
        if len(right_idx_arr) > 200:
            right_fit = np.polyfit(nz_y[right_idx_arr], nz_x[right_idx_arr], 2)
            conf_r = min(1.0, len(right_idx_arr) / 4000.0)
        confidence = (conf_l + conf_r) / 2.0
        return left_fit, right_fit, confidence

    def detect(self, frame: np.ndarray) -> LaneResult:
        """Detect lanes classically. Returns a :class:`LaneResult`."""
        h, w = frame.shape[:2]
        warper = self._ensure_warper((w, h))
        try:
            binary = self._binary_mask(frame)
            warped = self._accumulate(warper.warp(binary))
            left_fit, right_fit, conf = self._sliding_window(warped)
            if left_fit is None and right_fit is None and self.edge_fallback:
                return self._road_edge_fallback(frame, warper)
            return self._build_result(frame, warper, left_fit, right_fit, conf, "classical")
        except Exception as exc:
            log.debug("Classical lane detection error: {}", exc)
            return LaneResult(source="none")

    def _road_edge_fallback(self, frame: np.ndarray, warper: _Warper) -> LaneResult:
        """Estimate lane bounds from dominant road EDGES when markings are absent."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
        warped = warper.warp(edges)
        left_fit, right_fit, conf = self._sliding_window(warped, margin=110, minpix=25)
        res = self._build_result(frame, warper, left_fit, right_fit, conf * 0.6, "classical")
        res.direction = res.direction  # keep
        return res

    def _gate_fits(self, left_fit: Optional[np.ndarray], right_fit: Optional[np.ndarray],
                   w: int, h: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Reject implausible fits (crossing / bad width / near-horizontal wander).

        Works in warped space where genuine lanes are near-vertical and roughly
        parallel. Returns cleaned ``(left_fit, right_fit)`` (either may be None).
        """
        y_b, y_t = float(h - 1), float(self.draw_top_frac * h)
        max_dx = self.max_lane_dx_frac * w

        def x_at(fit: Optional[np.ndarray], y: float) -> float:
            return float(np.polyval(fit, y))

        def slope_ok(fit: Optional[np.ndarray]) -> bool:
            # A real lane barely wanders horizontally across the drawn height.
            return fit is not None and abs(x_at(fit, y_t) - x_at(fit, y_b)) <= max_dx

        left_fit = left_fit if slope_ok(left_fit) else None
        right_fit = right_fit if slope_ok(right_fit) else None
        if left_fit is not None and right_fit is not None:
            lb, rb = x_at(left_fit, y_b), x_at(right_fit, y_b)
            lt, rt = x_at(left_fit, y_t), x_at(right_fit, y_t)
            width_b = rb - lb
            crossing = lt >= rt or rb <= lb
            bad_width = not (self.min_lane_width_frac * w <= width_b <= self.max_lane_width_frac * w)
            if crossing or bad_width:
                return None, None  # messy pair -> let the smoother hold the last good lane
            return left_fit, right_fit
        if self.require_pair:
            # "Clean lane or nothing": a lone line (e.g. a kerb, guard-rail or
            # dashboard edge) reads as messy; the smoother bridges brief gaps.
            return None, None
        # Otherwise a lone line is drawn only if it is strongly vertical.
        strict = self.max_single_dx_frac * w
        if left_fit is not None and abs(x_at(left_fit, y_t) - x_at(left_fit, y_b)) > strict:
            left_fit = None
        if right_fit is not None and abs(x_at(right_fit, y_t) - x_at(right_fit, y_b)) > strict:
            right_fit = None
        return left_fit, right_fit

    def _build_result(self, frame: np.ndarray, warper: _Warper,
                      left_fit: Optional[np.ndarray], right_fit: Optional[np.ndarray],
                      conf: float, source: str) -> LaneResult:
        """Turn polynomial fits (warped) into an image-space :class:`LaneResult`."""
        h, w = frame.shape[:2]
        left_fit, right_fit = self._gate_fits(left_fit, right_fit, w, h)
        if left_fit is None and right_fit is None:
            return LaneResult(source="none")
        # Only draw the reliable near field; skipping the top of the warped image
        # avoids extrapolating to the horizon where lanes "fly" off the road.
        ploty = np.linspace(self.draw_top_frac * h, h - 1, 30)
        lines: List[np.ndarray] = []
        left_pts = right_pts = None
        if left_fit is not None:
            lx = np.polyval(left_fit, ploty)
            left_pts = warper.unwarp_points(np.stack([lx, ploty], axis=1))
            lines.append(left_pts.astype(np.int32))
        if right_fit is not None:
            rx = np.polyval(right_fit, ploty)
            right_pts = warper.unwarp_points(np.stack([rx, ploty], axis=1))
            lines.append(right_pts.astype(np.int32))

        fill = None
        ego_offset = None
        if left_pts is not None and right_pts is not None:
            fill = np.vstack([left_pts, right_pts[::-1]]).astype(np.int32)
            # Ego offset from lane centre at the bottom row.
            lane_center_x = (left_pts[-1][0] + right_pts[-1][0]) / 2.0
            half_lane = max(1.0, (right_pts[-1][0] - left_pts[-1][0]) / 2.0)
            ego_offset = float(np.clip((w / 2.0 - lane_center_x) / half_lane, -1.5, 1.5))

        curvature, direction = self._curvature(left_fit, right_fit, h)
        return LaneResult(lines=lines, fill=fill, left_fit=left_fit, right_fit=right_fit,
                          ego_offset=ego_offset, curvature_m=curvature, direction=direction,
                          source=source, confidence=float(conf))

    @staticmethod
    def _curvature(left_fit: Optional[np.ndarray], right_fit: Optional[np.ndarray],
                   h: int) -> Tuple[float, str]:
        """Estimate radius of curvature and a coarse turn direction."""
        fits = [f for f in (left_fit, right_fit) if f is not None]
        if not fits:
            return float("inf"), "unknown"
        y = h - 1
        curvs, signs = [], []
        for f in fits:
            a, b = f[0], f[1]
            denom = abs(2 * a) if abs(a) > 1e-6 else 1e-6
            curvs.append(((1 + (2 * a * y + b) ** 2) ** 1.5) / denom)
            signs.append(a)
        radius = float(np.mean(curvs))
        mean_a = float(np.mean(signs))
        direction = "straight"
        if radius < 800:  # curved (px-space radius heuristic)
            direction = "right" if mean_a > 0 else "left"
        return radius, direction


def _build_ufldv2_model(backbone: str, dataset: str):
    """Construct a UFLDv2 row-anchor lane model (torch imported lazily).

    A compact, faithful re-implementation of Ultra-Fast-Lane-Detection-v2's
    classification head on a torchvision ResNet backbone. Loading official
    checkpoints uses ``strict=False`` so partially-matching weights still work;
    the hybrid manager gates on confidence and falls back when needed.

    Args:
        backbone: ``"resnet18"`` or ``"resnet34"``.
        dataset: ``"culane"`` or ``"tusimple"`` (selects grid dims).

    Returns:
        An ``nn.Module`` returning a dict with ``loc_row`` and ``exist_row``.
    """
    import torch
    import torch.nn as nn
    import torchvision

    conf = _UFLDV2_CFG.get(dataset, _UFLDV2_CFG["culane"])
    num_grid, num_row, num_lanes = conf["num_grid_row"], conf["num_row"], conf["num_lanes"]

    class _UFLDv2(nn.Module):
        """UFLDv2 net: ResNet backbone + pooled classification/existence heads."""

        def __init__(self) -> None:
            super().__init__()
            net = getattr(torchvision.models, backbone)(weights=None)
            self.backbone = nn.Sequential(
                net.conv1, net.bn1, net.relu, net.maxpool,
                net.layer1, net.layer2, net.layer3, net.layer4,
            )
            feat_c = 512  # resnet18/34 final channels
            self.pool = nn.Conv2d(feat_c, 8, 1)
            self.pooled = nn.AdaptiveAvgPool2d((10, 25))
            in_dim = 8 * 10 * 25
            self.cls = nn.Sequential(
                nn.Linear(in_dim, 2048), nn.ReLU(),
                nn.Linear(2048, num_grid * num_row * num_lanes),
            )
            self.exist = nn.Sequential(
                nn.Linear(in_dim, 512), nn.ReLU(),
                nn.Linear(512, 2 * num_row * num_lanes),
            )

        def forward(self, x):  # type: ignore[override]
            f = self.backbone(x)
            f = self.pooled(self.pool(f))
            f = torch.flatten(f, 1)
            loc = self.cls(f).view(-1, num_grid, num_row, num_lanes)
            ext = self.exist(f).view(-1, 2, num_row, num_lanes)
            return {"loc_row": loc, "exist_row": ext}

    return _UFLDv2()


def decode_lanes(out: Dict[str, Any], row_anchors: np.ndarray,
                 frame_size: Tuple[int, int], dataset: str,
                 conf_threshold: float) -> List[np.ndarray]:
    """Decode UFLDv2 row-anchor logits into image-space lane point lists.

    Args:
        out: Model output dict with ``loc_row`` and ``exist_row``.
        row_anchors: Normalised row positions (0..1), length ``num_row``.
        frame_size: Target ``(width, height)`` for output points.
        dataset: Dataset key selecting grid dims.
        conf_threshold: Minimum existence probability to keep a point.

    Returns:
        List of ``(N, 2)`` int arrays, one per detected lane.
    """
    import torch

    conf = _UFLDV2_CFG.get(dataset, _UFLDV2_CFG["culane"])
    num_grid, num_lanes = conf["num_grid_row"], conf["num_lanes"]
    w, h = frame_size
    loc = out["loc_row"]          # (1, num_grid, num_row, num_lanes)
    ext = out["exist_row"]        # (1, 2, num_row, num_lanes)
    prob = torch.softmax(loc[0], dim=0)                      # over grid
    idx = torch.arange(1, num_grid + 1, device=loc.device).view(-1, 1, 1).float()
    expected = torch.sum(prob * idx, dim=0) - 1             # (num_row, num_lanes)
    exist = torch.softmax(ext[0], dim=0)[1]                 # (num_row, num_lanes)

    expected_np = expected.detach().cpu().numpy()
    exist_np = exist.detach().cpu().numpy()
    lanes: List[np.ndarray] = []
    for lane in range(num_lanes):
        pts = []
        for r, y_frac in enumerate(row_anchors):
            if exist_np[r, lane] < conf_threshold:
                continue
            x = float(expected_np[r, lane]) / max(num_grid - 1, 1) * w
            y = float(y_frac) * h
            if 0 <= x <= w:
                pts.append([x, y])
        if len(pts) >= 2:
            lanes.append(np.asarray(pts, np.float32))
    return lanes


# ===========================================================================
# Deep detector (UFLDv2) - optional, auto-disables without weights
# ===========================================================================
class DeepLaneDetector:
    """UFLDv2 deep lane detector. Active only when trained weights are present.

    This wrapper loads a UFLDv2 checkpoint and produces lane point sets. If the
    weights file is missing or loading/inference fails, :attr:`available` becomes
    False and the hybrid manager falls back to the classical detector.
    """

    def __init__(self, cfg: Dict[str, Any], device: str) -> None:
        """Args: cfg: ``lane_detector.deep`` block; device: torch device str."""
        self.cfg = cfg or {}
        self.device = device
        self.enabled = bool(self.cfg.get("enabled", True))
        self.conf_threshold = float(self.cfg.get("conf_threshold", 0.4))
        self.input_size = tuple(self.cfg.get("input_size", [320, 1600]))  # (h, w)
        self.available = False
        self._model = None
        self._row_anchors: Optional[np.ndarray] = None
        if self.enabled:
            self._try_load()

    def _try_load(self) -> None:
        """Attempt to build the net and load weights; disable on any failure."""
        weights = self.cfg.get("weights")
        wpath = resolve_path(weights) if weights else None
        if wpath is None or not wpath.is_file():
            log.info("UFLDv2 weights not found ({}); using classical lane detection.",
                     str(wpath) if wpath else "unset")
            return
        try:
            import torch

            self._model = _build_ufldv2_model(
                backbone=self.cfg.get("backbone", "resnet34"),
                dataset=self.cfg.get("dataset", "culane"),
            ).to(self.device)
            ckpt = torch.load(str(wpath), map_location=self.device)
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = self._model.load_state_dict(state, strict=False)
            self._model.eval()
            self._row_anchors = build_row_anchors(self.cfg.get("dataset", "culane"))
            self.available = True
            log.info("UFLDv2 loaded (missing={}, unexpected={}).", len(missing), len(unexpected))
        except Exception as exc:
            log.warning("UFLDv2 unavailable ({}); falling back to classical.", exc)
            self.available = False
            self._model = None

    def detect(self, frame: np.ndarray) -> Optional[LaneResult]:
        """Run deep lane inference; returns None to signal fallback."""
        if not self.available or self._model is None:
            return None
        try:
            import torch

            h0, w0 = frame.shape[:2]
            inp = cv2.resize(frame, (self.input_size[1], self.input_size[0]))
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = (inp - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            tensor = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                out = self._model(tensor)
            lanes = decode_lanes(out, self._row_anchors, (w0, h0),
                                 self.cfg.get("dataset", "culane"), self.conf_threshold)
            if not lanes:
                return None
            lines = [np.asarray(l, np.int32) for l in lanes if len(l) >= 2]
            if not lines:
                return None
            return LaneResult(lines=lines, source="deep", confidence=0.9)
        except Exception as exc:
            log.debug("UFLDv2 inference error: {}; falling back.", exc)
            return None


# ===========================================================================
# Deep detector (YOLOP ONNX) - robust primary trained on BDD100K
# ===========================================================================
class YOLOPLaneDetector:
    """YOLOP ONNX lane detector: drivable-area + lane-line segmentation.

    YOLOP (trained on BDD100K: day/night/rain/city/highway) predicts a drivable
    area mask and a lane-line mask in one pass. This is far more robust than
    classical CV on curves, faded/absent markings and poor light - exactly the
    client's hard cases. We ignore YOLOP's own detection head (we use YOLOv8).

    Runs on GPU via onnxruntime (CUDA/TensorRT providers) and auto-disables on
    any load/inference error so the hybrid manager falls back to classical.
    """

    _MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    _STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, cfg: Dict[str, Any], device: str) -> None:
        """Args: cfg: ``lane_detector.deep`` block; device: torch device str."""
        self.cfg = cfg or {}
        self.device = device
        self.enabled = bool(self.cfg.get("enabled", True))
        size = self.cfg.get("input_size", [640, 640])
        self.in_h, self.in_w = int(size[0]), int(size[1])
        self.lane_thresh = float(self.cfg.get("lane_conf", 0.5))
        self.drive_thresh = float(self.cfg.get("drive_conf", 0.5))
        self.mask_ema = float(self.cfg.get("mask_ema_alpha", 0.5))
        self.available = False
        self._sess = None
        self._in_name = "images"
        self._ema_drive: Optional[np.ndarray] = None
        self._ema_lane: Optional[np.ndarray] = None
        self._size: Optional[Tuple[int, int]] = None
        if self.enabled:
            self._try_load()

    def _try_load(self) -> None:
        """Load the ONNX session; disable on any failure."""
        weights = self.cfg.get("weights")
        wpath = resolve_path(weights) if weights else None
        if wpath is None or not wpath.is_file():
            log.info("YOLOP weights not found ({}); using classical lane detection.",
                     str(wpath) if wpath else "unset")
            return
        try:
            import onnxruntime as ort

            avail = ort.get_available_providers()
            want = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if "cuda" in str(self.device) and "CUDAExecutionProvider" in avail
                    else ["CPUExecutionProvider"])
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._sess = ort.InferenceSession(str(wpath), sess_options=so, providers=want)
            self._in_name = self._sess.get_inputs()[0].name
            self.available = True
            log.info("YOLOP lane model loaded ({}) providers={}.", wpath.name,
                     self._sess.get_providers())
        except Exception as exc:
            log.warning("YOLOP unavailable ({}); falling back to classical.", exc)
            self.available = False
            self._sess = None

    def reset(self) -> None:
        """Clear temporal mask smoothing (call between videos)."""
        self._ema_drive = self._ema_lane = None

    # -- pre / post processing ---------------------------------------------
    def _letterbox(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, int, int, int]:
        """Aspect-preserving resize + pad to the network input square."""
        h, w = frame.shape[:2]
        r = min(self.in_h / h, self.in_w / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (self.in_h - nh) // 2, (self.in_w - nw) // 2
        out = np.full((self.in_h, self.in_w, 3), 114, np.uint8)
        out[top:top + nh, left:left + nw] = resized
        return out, r, left, top, nw, nh

    def _unletterbox(self, mask: np.ndarray, left: int, top: int, nw: int, nh: int,
                     frame_size: Tuple[int, int]) -> np.ndarray:
        """Crop the padding and resize a network-space mask back to the frame.

        Uses linear interpolation on a float mask so upscaling a small (e.g. 320)
        network output to the frame gives smooth edges, not blocky stairsteps;
        the caller re-thresholds after temporal EMA.
        """
        crop = mask[top:top + nh, left:left + nw].astype(np.float32)
        w, h = frame_size
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    def _smooth(self, prev: Optional[np.ndarray], cur: np.ndarray) -> np.ndarray:
        """EMA-blend consecutive masks to remove per-frame flicker."""
        cf = cur.astype(np.float32)
        if prev is None or prev.shape != cf.shape:
            return cf
        return self.mask_ema * cf + (1.0 - self.mask_ema) * prev

    def detect(self, frame: np.ndarray) -> Optional[LaneResult]:
        """Run YOLOP; return a mask-backed :class:`LaneResult` (or None to fall back)."""
        if not self.available or self._sess is None:
            return None
        try:
            h, w = frame.shape[:2]
            inp, r, left, top, nw, nh = self._letterbox(frame)
            x = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = (x - self._MEAN) / self._STD
            x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
            _, da, ll = self._sess.run(None, {self._in_name: x})
            # Foreground vs background, cropped to the valid (unpadded) region.
            # ALL heavy per-pixel work (EMA, threshold, geometry) stays at NETWORK
            # resolution - ~16x fewer pixels than the frame - and only the final
            # masks are upscaled once. This is the key to a cheap render.
            drive = (da[0, 1] > da[0, 0]).astype(np.float32)[top:top + nh, left:left + nw]
            lane = (ll[0, 1] > ll[0, 0]).astype(np.float32)[top:top + nh, left:left + nw]
            self._ema_drive = self._smooth(self._ema_drive, drive)
            self._ema_lane = self._smooth(self._ema_lane, lane)
            db_s = (self._ema_drive >= self.drive_thresh).astype(np.uint8)
            lb_s = (self._ema_lane >= self.lane_thresh).astype(np.uint8)
            if int(db_s.sum()) < 60 and int(lb_s.sum()) < 25:
                return None  # empty -> let classical try

            # Geometry on the small masks (in-lane ratios are resolution-free).
            ego_offset = self._ego_offset(lb_s, db_s, (nw, nh))
            curv, direction = self._geometry(db_s, (nw, nh))
            # Upscale the smoothed masks to the frame ONCE (linear + threshold =>
            # smooth edges without any full-resolution float math per frame).
            drive_b = (cv2.resize(self._ema_drive, (w, h), interpolation=cv2.INTER_LINEAR)
                       >= self.drive_thresh).astype(np.uint8)
            lane_b = (cv2.resize(self._ema_lane, (w, h), interpolation=cv2.INTER_LINEAR)
                      >= self.lane_thresh).astype(np.uint8)
            return LaneResult(
                lines=[], fill=None, drive_mask=drive_b, lane_mask=lane_b,
                ego_offset=ego_offset, curvature_m=curv, direction=direction,
                source="deep", confidence=0.9,
            )
        except Exception as exc:
            log.debug("YOLOP inference error: {}; falling back.", exc)
            return None

    @staticmethod
    def _ego_offset(lane_b: np.ndarray, drive_b: np.ndarray,
                    frame_size: Tuple[int, int]) -> Optional[float]:
        """Ego lateral position in-lane from the two nearest lane lines (fallback
        to the drivable-area centre). ``-1``=on left line .. ``+1``=on right."""
        w, h = frame_size
        cx = w / 2.0
        y0, y1 = int(0.80 * h), int(0.97 * h)
        xs = np.where(lane_b[y0:y1] > 0)[1]
        if xs.size > 30:
            left = xs[xs < cx]
            right = xs[xs > cx]
            if left.size > 5 and right.size > 5:
                lx, rx = float(left.max()), float(right.min())
                center = (lx + rx) / 2.0
                half = max((rx - lx) / 2.0, 1.0)
                return float(np.clip((cx - center) / half, -1.5, 1.5))
        # Fallback: centre of the drivable band near the bottom.
        cols = np.where(drive_b[int(0.90 * h)] > 0)[0]
        if cols.size > 20:
            center = (float(cols.min()) + float(cols.max())) / 2.0
            half = max((float(cols.max()) - float(cols.min())) / 2.0, 1.0)
            return float(np.clip((cx - center) / half, -1.5, 1.5))
        return None

    @staticmethod
    def _geometry(drive_b: np.ndarray, frame_size: Tuple[int, int]) -> Tuple[float, str]:
        """Coarse curve direction from how the drivable centre shifts with depth."""
        w, h = frame_size
        def band_center(row: int) -> Optional[float]:
            cols = np.where(drive_b[row] > 0)[0]
            return float(cols.mean()) if cols.size > 10 else None
        top, bot = band_center(int(0.55 * h)), band_center(int(0.92 * h))
        if top is None or bot is None:
            return float("inf"), "unknown"
        dx = top - bot
        if abs(dx) < 0.06 * w:
            return 500.0, "straight"
        return 120.0, ("right" if dx > 0 else "left")


# ===========================================================================
# Temporal smoothing
# ===========================================================================
class LaneSmoother:
    """Smooths lane polynomial coefficients over time and holds the last good lane."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Args: cfg: the ``lane_detector.smoothing`` block."""
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", True))
        self.alpha = float(self.cfg.get("ema_alpha", 0.35))
        self.max_missing = int(self.cfg.get("max_frames_missing", 8))
        self._left: Optional[np.ndarray] = None
        self._right: Optional[np.ndarray] = None
        self._last: Optional[LaneResult] = None
        self._missing = 0

    def _ema(self, prev: Optional[np.ndarray], new: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if new is None:
            return prev
        if prev is None:
            return new
        return self.alpha * new + (1 - self.alpha) * prev

    def update(self, result: LaneResult) -> LaneResult:
        """Blend ``result`` with history; hold last good lane during dropouts."""
        if not self.enabled:
            return result
        if not result.is_valid():
            self._missing += 1
            if self._last is not None and self._missing <= self.max_missing:
                held = LaneResult(**{**self._last.__dict__})
                held.source = "held"
                held.confidence *= 0.8
                return held
            return result
        self._missing = 0
        # Smooth polynomial coefficients if both present (classical path).
        if result.left_fit is not None or result.right_fit is not None:
            self._left = self._ema(self._left, result.left_fit)
            self._right = self._ema(self._right, result.right_fit)
            result.left_fit = self._left if self._left is not None else result.left_fit
            result.right_fit = self._right if self._right is not None else result.right_fit
        self._last = result
        return result

    def reset(self) -> None:
        """Clear smoothing history."""
        self._left = self._right = None
        self._last = None
        self._missing = 0


# ===========================================================================
# Hybrid manager
# ===========================================================================
class LaneDetector:
    """Hybrid lane detector: deep primary, classical fallback, temporal smoothing."""

    def __init__(self, cfg: Dict[str, Any], device: Optional[str] = None) -> None:
        """Create the hybrid detector.

        Args:
            cfg: The full ``model_config`` dict (needs ``lane_detector`` + device).
            device: torch device; auto-resolved if None.
        """
        lane_cfg = cfg.get("lane_detector", {})
        self.mode = str(lane_cfg.get("mode", "hybrid")).lower()
        self.device = device or resolve_device(cfg.get("device"))
        self.classical = ClassicalLaneDetector(lane_cfg.get("classical", {}))
        self.smoother = LaneSmoother(lane_cfg.get("smoothing", {}))
        self.curve_min_m = float(lane_cfg.get("smoothing", {}).get("curvature_min_m", 50.0))
        self.deep = None
        if self.mode in ("hybrid", "deep"):
            deep_cfg = lane_cfg.get("deep", {})
            arch = str(deep_cfg.get("architecture", "yolop")).lower()
            # YOLOP (BDD100K) is the robust default; UFLDv2 kept as an option.
            self.deep = (YOLOPLaneDetector(deep_cfg, self.device) if arch == "yolop"
                         else DeepLaneDetector(deep_cfg, self.device))
        log.info("LaneDetector mode='{}' arch='{}' (deep_available={}).",
                 self.mode, str(lane_cfg.get("deep", {}).get("architecture", "yolop")).lower(),
                 bool(self.deep and self.deep.available))

    def detect(self, frame: np.ndarray) -> LaneResult:
        """Detect lanes for one frame with fallback + smoothing applied.

        Args:
            frame: BGR frame.

        Returns:
            A smoothed :class:`LaneResult`.
        """
        result: Optional[LaneResult] = None
        if self.mode in ("hybrid", "deep") and self.deep is not None and self.deep.available:
            result = self.deep.detect(frame)
        if (result is None or not result.is_valid()) and self.mode != "deep":
            result = self.classical.detect(frame)
        if result is None:
            result = LaneResult(source="none")
        return self.smoother.update(result)

    def reset(self) -> None:
        """Reset temporal state (e.g. between videos)."""
        self.smoother.reset()
        self.classical.reset()
        if self.deep is not None and hasattr(self.deep, "reset"):
            self.deep.reset()
