"""Camera calibration tool - writes configs/camera_calibration.yaml.

Two modes:

* ``chessboard`` - full intrinsic + distortion calibration from a folder of
  chessboard photos (most accurate).
* ``fov`` - quick focal-length estimate from a known horizontal field of view
  and resolution (good enough when you only know the lens spec).

Both can also set the mounting geometry (camera height, pitch, horizon row)
which feeds the distance / BEV models.

Usage::

    python src/scripts/calibrate_camera.py fov --hfov 70 --width 1920 --height 1080 \
        --cam-height 1.25 --horizon 0.52
    python src/scripts/calibrate_camera.py chessboard --images calib/ --cols 9 --rows 6 --square 0.025
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cv2
import numpy as np
import yaml

from utils.config_loader import PROJECT_ROOT, resolve_path
from utils.logger import get_logger

log = get_logger(__name__)

CALIB_YAML = PROJECT_ROOT / "configs" / "camera_calibration.yaml"


def _load_yaml() -> Dict[str, Any]:
    return yaml.safe_load(CALIB_YAML.read_text(encoding="utf-8")) or {}


def _save_yaml(doc: Dict[str, Any]) -> None:
    CALIB_YAML.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log.info("Updated {}", CALIB_YAML)


def _apply_mounting(doc: Dict[str, Any], args: argparse.Namespace) -> None:
    """Update mounting geometry in the config from CLI args (if provided)."""
    mount = doc.setdefault("mounting", {})
    if args.cam_height is not None:
        mount["height_m"] = float(args.cam_height)
    if args.pitch is not None:
        mount["pitch_deg"] = float(args.pitch)
    if args.horizon is not None:
        mount["horizon_y_frac"] = float(args.horizon)


def calibrate_fov(args: argparse.Namespace) -> int:
    """Estimate intrinsics from HFOV + resolution and write the config."""
    doc = _load_yaml()
    w, h = int(args.width), int(args.height)
    fx = (w / 2.0) / math.tan(math.radians(args.hfov) / 2.0)
    doc["calibrated_resolution"] = [w, h]
    doc["intrinsics"] = {"fx": round(fx, 2), "fy": round(fx, 2),
                         "cx": w / 2.0, "cy": h / 2.0, "horizontal_fov_deg": float(args.hfov)}
    doc.setdefault("distortion", {"enabled": False, "k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0})
    _apply_mounting(doc, args)
    log.info("FOV calibration: fx=fy={:.1f}px for {}x{} @ {} deg HFOV.", fx, w, h, args.hfov)
    _save_yaml(doc)
    return 0


def calibrate_chessboard(args: argparse.Namespace) -> int:
    """Full intrinsic + distortion calibration from chessboard images."""
    images_dir = resolve_path(args.images)
    files = [p for p in images_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not files:
        log.error("No calibration images in {}", images_dir)
        return 1

    cols, rows = int(args.cols), int(args.rows)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(args.square)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    img_size = None
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
    log.info("Found the board in {}/{} images.", len(objpoints), len(files))
    if len(objpoints) < 5:
        log.error("Need >=5 successful detections; got {}.", len(objpoints))
        return 1

    ret, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    log.info("Calibration RMS reprojection error: {:.4f}", ret)

    doc = _load_yaml()
    doc["calibrated_resolution"] = [int(img_size[0]), int(img_size[1])]
    doc["intrinsics"] = {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                         "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                         "horizontal_fov_deg": round(2 * math.degrees(math.atan(img_size[0] / (2 * K[0, 0]))), 2)}
    d = dist.flatten().tolist() + [0, 0, 0, 0, 0]
    doc["distortion"] = {"enabled": True, "k1": d[0], "k2": d[1], "p1": d[2], "p2": d[3], "k3": d[4]}
    _apply_mounting(doc, args)
    _save_yaml(doc)
    return 0


def main() -> int:
    """CLI entry with fov / chessboard subcommands."""
    parser = argparse.ArgumentParser(description="NeuraRoads camera calibration.")
    parser.add_argument("--cam-height", type=float, default=None, help="Camera height (m).")
    parser.add_argument("--pitch", type=float, default=None, help="Camera pitch (deg).")
    parser.add_argument("--horizon", type=float, default=None, help="Horizon row fraction (0..1).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fov = sub.add_parser("fov", help="Estimate intrinsics from HFOV.")
    fov.add_argument("--hfov", type=float, required=True)
    fov.add_argument("--width", type=int, required=True)
    fov.add_argument("--height", type=int, required=True)

    cb = sub.add_parser("chessboard", help="Full calibration from chessboard images.")
    cb.add_argument("--images", type=str, required=True)
    cb.add_argument("--cols", type=int, default=9, help="Inner corners per row.")
    cb.add_argument("--rows", type=int, default=6, help="Inner corners per column.")
    cb.add_argument("--square", type=float, default=0.025, help="Square size (m).")

    args = parser.parse_args()
    try:
        return calibrate_fov(args) if args.cmd == "fov" else calibrate_chessboard(args)
    except Exception as exc:
        log.error("Calibration failed: {}", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
