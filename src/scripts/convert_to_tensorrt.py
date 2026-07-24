"""Build a TensorRT engine from the trained detector (FP16 or INT8).

TensorRT gives the biggest inference speed-up on both the GTX 1660 SUPER (FP16)
and the Jetson Nano (INT8). INT8 needs a calibration image set - the val split
works well. The resulting ``.engine`` is copied to ``models/weights`` with the
name the detector config already looks for, so inference uses it automatically.

Usage::

    python src/scripts/convert_to_tensorrt.py --half                       # FP16 (desktop)
    python src/scripts/convert_to_tensorrt.py --int8 --jetson              # INT8 (Jetson Nano)
    python src/scripts/convert_to_tensorrt.py --weights path/to/best.pt --half

Note: requires the NVIDIA ``tensorrt`` Python package to be installed for your
platform. On the Jetson it ships with JetPack; on the desktop install the wheel
matching your CUDA before running this.
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


def _check_tensorrt() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except Exception:
        return False


def main() -> int:
    """CLI entry: export the detector to a TensorRT engine."""
    parser = argparse.ArgumentParser(description="Convert NeuraRoads detector to TensorRT.")
    parser.add_argument("--weights", type=str, default=None, help="Path to best.pt.")
    parser.add_argument("--imgsz", type=int, default=None, help="Engine input size.")
    parser.add_argument("--half", action="store_true", help="FP16 engine.")
    parser.add_argument("--int8", action="store_true", help="INT8 engine (needs calibration).")
    parser.add_argument("--jetson", action="store_true", help="Use jetson_config (arch/imgsz/paths).")
    parser.add_argument("--workspace", type=int, default=4, help="Builder workspace (GB).")
    args = parser.parse_args()

    if not _check_tensorrt():
        log.error("The 'tensorrt' package is not installed. Install the wheel matching your "
                  "CUDA/JetPack, then re-run. (ONNX export via export_model.py works without it.)")
        return 1

    overlay = "jetson_config" if args.jetson else None
    cfg = load_config("model_config", overlay)
    weights = resolve_path(args.weights or cfg["detector"]["weights"])
    if not weights.is_file():
        log.error("Weights not found: {}. Train first with train_yolo.py.", weights)
        return 1

    imgsz = args.imgsz or int(cfg["detector"].get("imgsz", 640))
    device = _device_arg(resolve_device(cfg.get("device")))
    arch = cfg["detector"].get("architecture", "yolov8m")

    export_kwargs = dict(format="engine", imgsz=imgsz, device=device,
                         half=args.half, int8=args.int8, workspace=args.workspace,
                         dynamic=False, simplify=True, verbose=False)
    if args.int8:
        # Ultralytics uses the dataset's val split for INT8 calibration.
        export_kwargs["data"] = str(resolve_path("data/annotations/data.yaml"))
        log.info("INT8 calibration will use the dataset val split.")

    from ultralytics import YOLO

    log.info("Building TensorRT engine from {} (imgsz={}, half={}, int8={})...",
             weights.name, imgsz, args.half, args.int8)
    try:
        engine = Path(YOLO(str(weights)).export(**export_kwargs))
    except Exception as exc:
        log.error("TensorRT export failed: {}", exc)
        return 1

    dst_dir = resolve_path("models/weights")
    dst_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_int8" if args.int8 else ""
    dst = dst_dir / f"neuraroads_{arch}{suffix}.engine"
    shutil.copy2(engine, dst)
    log.info("Engine ready -> {}", dst)
    log.info("The detector will now auto-use this engine (backend=auto).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
