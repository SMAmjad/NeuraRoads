"""HUD renderer: bounding boxes, lanes, BEV, top/bottom bars and ADAS icons.

:class:`Visualizer` turns the structured per-frame state produced by the
pipeline into the final annotated frame described in the project spec:

* Top bar: ``FPS | NEURAROADS ADAS | ego speed | time``.
* Center view: colour-coded boxes labelled with name / ID / distance / speed /
  TTC, lane lines + shaded drivable area.
* Top-right: bird's-eye-view mini-map.
* Center-top: the single active ADAS icon (fade + pulse) and its warning text.
* Bottom bar: bar graph of the closest objects + active-warnings list.

All colours, fonts and layout come from ``model_config.visualization`` so
nothing is hardcoded. The renderer is purely presentational - danger levels,
icon selection and fade/pulse state are decided upstream (``src/adas``) and
passed in via the ``alert`` dict.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.config_loader import resolve_path
from utils.logger import get_logger

log = get_logger(__name__)

BGR = Tuple[int, int, int]
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Visualizer:
    """Draws the complete NeuraRoads HUD onto frames.

    Args:
        vis_cfg: The ``visualization`` block of ``model_config``.
        bev_cfg: The ``bev`` block (for corner placement/size).
    """

    def __init__(self, vis_cfg: Dict[str, Any], bev_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = vis_cfg or {}
        self.bev_cfg = bev_cfg or {}
        self.colors: Dict[str, BGR] = {
            k: tuple(v) for k, v in self.cfg.get("colors", {}).items()  # type: ignore
        }
        self.box_thickness = int(self.cfg.get("box_thickness", 2))
        self.font_scale = float(self.cfg.get("font_scale", 0.5))
        self.lane_fill_alpha = float(self.cfg.get("lane_fill_alpha", 0.25))
        self.hud = self.cfg.get("hud", {})
        self.icons_dir = resolve_path(self.cfg.get("icons_dir", "data/icons"))
        self._icons: Dict[str, np.ndarray] = {}
        self._load_icons()

    # -- asset loading ------------------------------------------------------
    def _load_icons(self) -> None:
        """Load every PNG in the icons dir as an RGBA image (cached)."""
        if not self.icons_dir.is_dir():
            log.warning("Icons dir not found: {}", self.icons_dir)
            return
        for png in self.icons_dir.glob("*.png"):
            img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 3 and img.shape[2] == 3:  # add opaque alpha
                img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
            self._icons[png.name] = img
        log.info("Loaded {} ADAS icons from {}.", len(self._icons), self.icons_dir)

    def color(self, key: str, default: BGR = (255, 255, 255)) -> BGR:
        """Resolve a colour key from config to a BGR tuple."""
        return self.colors.get(key, default)

    # -- low-level primitives ----------------------------------------------
    @staticmethod
    def _overlay_rgba(frame: np.ndarray, rgba: np.ndarray, x: int, y: int,
                      alpha_scale: float = 1.0) -> None:
        """Alpha-blend an RGBA image onto ``frame`` at top-left ``(x, y)`` in place."""
        h, w = rgba.shape[:2]
        H, W = frame.shape[:2]
        if x >= W or y >= H or x + w <= 0 or y + h <= 0:
            return
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        rx0, ry0 = x0 - x, y0 - y
        roi = frame[y0:y1, x0:x1]
        icon = rgba[ry0:ry0 + (y1 - y0), rx0:rx0 + (x1 - x0)]
        if icon.size == 0:
            return
        alpha = (icon[:, :, 3:4].astype(np.float32) / 255.0) * float(np.clip(alpha_scale, 0, 1))
        roi[:] = (icon[:, :, :3].astype(np.float32) * alpha +
                  roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

    def _draw_icon(self, frame: np.ndarray, name: str, x: int, y: int,
                   size: Tuple[int, int], alpha: float = 1.0) -> None:
        """Draw a named icon resized to ``size`` at ``(x, y)`` with ``alpha``."""
        icon = self._icons.get(name)
        if icon is None:
            return
        resized = cv2.resize(icon, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)
        self._overlay_rgba(frame, resized, int(x), int(y), alpha)

    def _text(self, frame: np.ndarray, text: str, org: Tuple[int, int],
              scale: Optional[float] = None, color: BGR = (255, 255, 255),
              thickness: int = 1, bg: Optional[BGR] = None) -> None:
        """Draw text with an optional filled background box for readability."""
        scale = self.font_scale if scale is None else scale
        (tw, th), base = cv2.getTextSize(text, _FONT, scale, thickness)
        x, y = org
        if bg is not None:
            cv2.rectangle(frame, (x - 2, y - th - base - 2), (x + tw + 2, y + base - 1), bg, -1)
        cv2.putText(frame, text, (x, y - 1), _FONT, scale, color, thickness, cv2.LINE_AA)

    # -- boxes --------------------------------------------------------------
    def draw_boxes(self, frame: np.ndarray, objects: Sequence[Dict[str, Any]]) -> None:
        """Draw colour-coded bounding boxes with rich labels.

        Each object dict may contain: ``box`` ``[x1,y1,x2,y2]`` (required),
        ``label``/``class_name``, ``track_id``, ``distance_m``, ``speed_kmh``,
        ``ttc_s`` and ``color_key`` (one of the visualization colour keys).
        """
        for obj in objects:
            box = obj.get("box")
            if box is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            col = self.color(obj.get("color_key", "safe"))
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, self.box_thickness)

            name = obj.get("class_name", obj.get("label", "obj"))
            tid = obj.get("track_id")
            header = f"{name}" + (f" #{int(tid)}" if tid is not None else "")
            self._text(frame, header, (x1 + 2, max(14, y1 - 4)),
                       scale=self.font_scale, color=(255, 255, 255),
                       thickness=1, bg=col)

            # Second line: distance / speed / TTC (only what is available).
            # Skip it for tiny/distant boxes to avoid label clutter when several
            # far vehicles cluster near the horizon.
            bits: List[str] = []
            if (y2 - y1) >= 22:
                if obj.get("distance_m") is not None and np.isfinite(obj["distance_m"]):
                    bits.append(f"{obj['distance_m']:.0f}m")
                if obj.get("speed_kmh") is not None:
                    bits.append(f"{obj['speed_kmh']:.0f}km/h")
                if obj.get("ttc_s") is not None and np.isfinite(obj["ttc_s"]):
                    bits.append(f"TTC {obj['ttc_s']:.1f}s")
            if bits:
                self._text(frame, "  ".join(bits), (x1 + 2, min(frame.shape[0] - 2, y2 + 14)),
                           scale=self.font_scale * 0.9, color=(255, 255, 255),
                           thickness=1, bg=(0, 0, 0))

    # -- lanes --------------------------------------------------------------
    def draw_lanes(self, frame: np.ndarray, lane: Optional[Dict[str, Any]]) -> None:
        """Draw lane lines and shade the drivable area.

        ``lane`` may contain: ``lines`` (list of Nx2 point arrays), ``fill``
        (Nx2 polygon of the drivable area) and ``leaving`` (bool -> red lines).
        """
        if not lane:
            return
        leaving = bool(lane.get("leaving", False))
        line_col = self.color("lane_leaving") if leaving else self.color("lane_normal")

        # Deep (YOLOP) path: shade the drivable-area mask and paint the lane-line
        # mask. This is the robust primary; the classical polygon path is below.
        drive_mask = lane.get("drive_mask")
        lane_mask = lane.get("lane_mask")
        if drive_mask is not None:
            dm = np.asarray(drive_mask)
            if dm.shape[:2] == frame.shape[:2]:
                rows = np.where(dm.any(axis=1))[0]
                if rows.size:
                    # Blend ONLY the band of rows the drivable area covers (lower
                    # ~half), uint8 + cv2 (no full-frame copy or float math).
                    y0, y1 = int(rows[0]), int(rows[-1]) + 1
                    a = min(0.9, self.lane_fill_alpha + 0.15)
                    roi = frame[y0:y1]
                    ov = roi.copy()
                    ov[dm[y0:y1] > 0] = self.color("drivable")
                    cv2.addWeighted(ov, a, roi, 1.0 - a, 0.0, roi)
        if lane_mask is not None:
            lm = np.asarray(lane_mask)
            if lm.shape[:2] == frame.shape[:2] and lm.any():
                # Dilate the thin mask so the lane lines read clearly.
                lm = cv2.dilate(lm.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
                frame[lm > 0] = line_col

        fill = lane.get("fill")
        if fill is not None and len(fill) >= 3:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.asarray(fill, dtype=np.int32)], self.color("lane_normal"))
            cv2.addWeighted(overlay, self.lane_fill_alpha, frame, 1 - self.lane_fill_alpha, 0, frame)

        for pts in lane.get("lines", []) or []:
            arr = np.asarray(pts, dtype=np.int32)
            if arr.shape[0] >= 2:
                cv2.polylines(frame, [arr], False, line_col, 4, cv2.LINE_AA)

    # -- BEV ----------------------------------------------------------------
    def draw_bev(self, frame: np.ndarray, bev_img: Optional[np.ndarray]) -> None:
        """Composite the BEV mini-map into the configured corner with a border."""
        if bev_img is None:
            return
        corner = self.hud.get("bev_corner", "top_right")
        bh, bw = bev_img.shape[:2]
        H, W = frame.shape[:2]
        margin = 12
        y0 = 8 + self._top_bar_height(H)
        x0 = (W - bw - margin) if "right" in corner else margin
        y1, x1 = y0 + bh, x0 + bw
        if y1 > H or x1 > W:
            return
        frame[y0:y1, x0:x1] = bev_img[:, :, :3] if bev_img.ndim == 3 else bev_img
        cv2.rectangle(frame, (x0 - 1, y0 - 1), (x1, y1), (200, 200, 200), 1)
        self._text(frame, "BEV", (x0 + 4, y0 + 16), 0.5, (255, 255, 255), 1, (0, 0, 0))

    # -- top / bottom bars --------------------------------------------------
    def _top_bar_height(self, H: int) -> int:
        return max(28, int(H * 0.045))

    def draw_top_bar(self, frame: np.ndarray, fps: float, ego_speed_kmh: float) -> None:
        """Draw the top status bar: FPS | title | ego speed | time."""
        if not self.hud.get("top_bar", True):
            return
        H, W = frame.shape[:2]
        bh = self._top_bar_height(H)
        # Blend ONLY the bar strip, not a full-frame copy (10x cheaper render).
        roi = frame[0:bh, 0:W]
        bar = np.empty_like(roi)
        bar[:] = self.color("hud_bg", (20, 20, 20))
        cv2.addWeighted(bar, 0.6, roi, 0.4, 0, roi)
        ty = int(bh * 0.68)
        self._text(frame, f"FPS {fps:4.1f}", (10, ty), 0.6, self.color("safe"), 2)
        title = "NEURAROADS ADAS"
        (tw, _), _ = cv2.getTextSize(title, _FONT, 0.7, 2)
        self._text(frame, title, (W // 2 - tw // 2, ty), 0.7, (255, 255, 255), 2)
        self._text(frame, f"EGO {ego_speed_kmh:5.1f} km/h", (W - 320, ty), 0.6, self.color("ego"), 2)
        self._text(frame, datetime.now().strftime("%H:%M:%S"), (W - 90, ty), 0.6, (255, 255, 255), 2)

    def draw_bottom_bar(self, frame: np.ndarray, closest: Sequence[Dict[str, Any]],
                        warnings: Sequence[str]) -> None:
        """Draw the bottom bar: closest-object bar graph + active warnings list."""
        if not self.hud.get("bottom_bar", True):
            return
        H, W = frame.shape[:2]
        bh = int(H * 0.16)
        y0 = H - bh
        # Blend ONLY the bottom strip (not a full-frame copy).
        roi = frame[y0:H, 0:W]
        bar = np.empty_like(roi)
        bar[:] = self.color("hud_bg", (20, 20, 20))
        cv2.addWeighted(bar, 0.55, roi, 0.45, 0, roi)

        # Left: closest objects as horizontal bars (nearer => longer + redder).
        self._text(frame, "CLOSEST", (12, y0 + 20), 0.5, (200, 200, 200), 1)
        max_range = 60.0
        bar_x, bar_w = 12, int(W * 0.32)
        for i, obj in enumerate(closest):
            yy = y0 + 34 + i * 22
            dist = float(obj.get("distance_m", max_range))
            frac = float(np.clip(1.0 - dist / max_range, 0.05, 1.0))
            col = self.color(obj.get("color_key", "safe"))
            cv2.rectangle(frame, (bar_x, yy), (bar_x + int(bar_w * frac), yy + 14), col, -1)
            label = f"{obj.get('class_name', 'obj')} {dist:.0f}m"
            self._text(frame, label, (bar_x + bar_w + 8, yy + 12), 0.45, (255, 255, 255), 1)

        # Right: active warnings list.
        self._text(frame, "ACTIVE WARNINGS", (int(W * 0.62), y0 + 20), 0.5, (200, 200, 200), 1)
        if not warnings:
            self._text(frame, "- none -", (int(W * 0.62), y0 + 42), 0.5, self.color("safe"), 1)
        for i, msg in enumerate(warnings):
            self._text(frame, f"! {msg}", (int(W * 0.62), y0 + 42 + i * 20), 0.5,
                       self.color("danger"), 1)

    # -- central ADAS icon + warning ---------------------------------------
    def draw_alert(self, frame: np.ndarray, alert: Optional[Dict[str, Any]]) -> None:
        """Draw the single active ADAS icon (center-top) with fade + message.

        ``alert`` fields: ``icon`` (png filename), ``message``, ``color_key``,
        ``alpha`` (0..1 fade), ``pulse_scale`` (>=1 icon zoom), ``screen_tint``
        (optional BGR) and ``screen_tint_alpha``.
        """
        if not alert:
            return
        H, W = frame.shape[:2]

        tint = alert.get("screen_tint")
        if tint is not None:
            self.apply_screen_tint(frame, tuple(tint), float(alert.get("screen_tint_alpha", 0.1)) *
                                   float(alert.get("alpha", 1.0)))

        if not self.hud.get("icon_center_top", True):
            return
        icon_name = alert.get("icon")
        alpha = float(alert.get("alpha", 1.0))
        pulse = float(alert.get("pulse_scale", 1.0))
        base = int(min(W, H) * 0.09)
        size = (int(base * pulse), int(base * pulse))
        x = W // 2 - size[0] // 2
        y = self._top_bar_height(H) + 8
        if icon_name:
            self._draw_icon(frame, icon_name, x, y, size, alpha)

        msg = alert.get("message")
        if msg:
            col = self.color(alert.get("color_key", "danger"))
            (tw, _), _ = cv2.getTextSize(msg, _FONT, 0.8, 2)
            self._text(frame, msg, (W // 2 - tw // 2, y + size[1] + 26), 0.8, col, 2, (0, 0, 0))

    @staticmethod
    def apply_screen_tint(frame: np.ndarray, color: BGR, alpha: float) -> None:
        """Blend a translucent full-screen colour tint (danger flash) in place."""
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if alpha <= 0:
            return
        overlay = np.full_like(frame, color, dtype=np.uint8)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # -- orchestrator -------------------------------------------------------
    def render(self, frame: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
        """Render the full HUD for one frame and return the annotated frame.

        Args:
            frame: The BGR frame to draw on (modified in place and returned).
            state: Per-frame state with optional keys ``objects``, ``lane``,
                ``bev``, ``alert``, ``closest``, ``warnings``, ``fps``,
                ``ego_speed_kmh``.

        Returns:
            The annotated frame.
        """
        self.draw_lanes(frame, state.get("lane"))
        self.draw_boxes(frame, state.get("objects", []))
        self.draw_bev(frame, state.get("bev"))
        self.draw_top_bar(frame, state.get("fps", 0.0), state.get("ego_speed_kmh", 0.0))
        self.draw_bottom_bar(frame, state.get("closest", []), state.get("warnings", []))
        self.draw_alert(frame, state.get("alert"))
        return frame
