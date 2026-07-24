"""Export a trained NeuraRoads detector to ONNX / TensorRT for fast inference.

The exported files are written to ``models/weights`` with the names the detector
config already looks for (``neuraroads_yolov8m.onnx`` / ``.engine``), so the
inference pipeline picks the fastest available backend automatically.

Usage::

    python src/training/export_model.py --format onnx           # portable, GPU via ORT
    python src/training/export_model.py --format engine --half   # TensorRT FP16 (fastest)
    python src/training/export_model.py --format both
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.config_loader import load_config, resolve_device, resolve_path
from utils.logger import get_logger

log = get_logger(__name__)


def _device_arg(device_str: str) -> Any:
    if device_str.startswith("cuda"):
        return int(device_str.split(":")[1]) if ":" in device_str else 0
    return "cpu"


def export(weights: Path, fmt: str, imgsz: int, half: bool, device: Any,
           int8: bool, dst_dir: Path, dst_stem: str) -> Path:
    """Export ``weights`` to ``fmt`` and copy the result into ``dst_dir``.

    Args:
        weights: Path to the trained ``.pt``.
        fmt: ``"onnx"`` or ``"engine"``.
        imgsz: Export image size.
        half: FP16 export (GPU only).
        device: Ultralytics device arg.
        int8: INT8 quantised TensorRT engine (needs calibration data).
        dst_dir: Destination directory (``models/weights``).
        dst_stem: Destination filename stem.

    Returns:
        Path to the exported file in ``dst_dir``.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    log.info("Exporting {} -> {} (imgsz={}, half={}, int8={})", weights.name, fmt, imgsz, half, int8)
    exported = model.export(format=fmt, imgsz=imgsz, half=half, device=device,
                            int8=int8, dynamic=False, simplify=True)
    exported = Path(exported)
    dst_dir.mkdir(parents=True, exist_ok=True)
    ext = ".engine" if fmt == "engine" else f".{fmt}"
    dst = dst_dir / f"{dst_stem}{'_int8' if (int8 and fmt == 'engine') else ''}{ext}"
    shutil.copy2(exported, dst)
    log.info("Saved -> {}", dst)
    return dst


def main() -> int:
    """CLI entry: export the trained detector to the requested format(s)."""
    parser = argparse.ArgumentParser(description="Export NeuraRoads detector.")
    parser.add_argument("--weights", type=str, default=None, help="Path to best.pt.")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "engine", "both"])
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--half", action="store_true", help="FP16 export (GPU).")
    parser.add_argument("--int8", action="store_true", help="INT8 TensorRT engine.")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_config("model_config")
    weights = resolve_path(args.weights or cfg["detector"]["weights"])
    if not weights.is_file():
        log.error("Weights not found: {}. Train first with train_yolo.py.", weights)
        return 1

    imgsz = args.imgsz or int(cfg["detector"].get("imgsz", 640))
    device = _device_arg(resolve_device(cfg.get("device")) if args.device == "auto" else
                         ("cpu" if args.device == "cpu" else f"cuda:{args.device}"))
    dst_dir = resolve_path("models/weights")
    stem = f"neuraroads_{cfg['detector'].get('architecture', 'yolov8m')}"

    formats = ["onnx", "engine"] if args.format == "both" else [args.format]
    for fmt in formats:
        if fmt == "engine" and device == "cpu":
            log.warning("TensorRT engine export requires a GPU; skipping.")
            continue
        try:
            export(weights, fmt, imgsz, args.half, device, args.int8, dst_dir, stem)
        except Exception as exc:
            log.error("Export to {} failed: {}", fmt, exc)
            if fmt == "engine":
                log.error("TensorRT export needs the 'tensorrt' package installed for your GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
