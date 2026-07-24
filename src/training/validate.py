"""Validate a trained NeuraRoads detector and report per-class metrics.

Runs Ultralytics validation on the val (or test) split and prints/saves mAP and
per-class AP with the human-readable class names, so results are reported as
``Car``, ``Truck``, ``Pedestrian`` ... rather than numeric indices.

Usage::

    python src/training/validate.py
    python src/training/validate.py --weights models/trained/neuraroads_yolov8m/weights/best.pt
    python src/training/validate.py --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.config_loader import load_config, resolve_device, resolve_path
from utils.logger import get_logger

log = get_logger(__name__)


def _device_arg(device_str: str) -> Any:
    """torch device string -> Ultralytics device arg."""
    if device_str.startswith("cuda"):
        return int(device_str.split(":")[1]) if ":" in device_str else 0
    return "cpu"


def validate(weights: str, data: str, split: str, device: Any,
             imgsz: int, save_dir: Path) -> Dict[str, float]:
    """Run validation and return a metrics summary dict.

    Args:
        weights: Path to the trained ``.pt`` weights.
        data: Path to the dataset YAML.
        split: ``"val"`` or ``"test"``.
        device: Ultralytics device arg.
        imgsz: Validation image size.
        save_dir: Directory for plots / outputs.

    Returns:
        Dict with overall + per-class metrics.
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(data=data, split=split, device=device, imgsz=imgsz,
                        project=str(save_dir), name="validation", exist_ok=True, plots=True)

    names = model.names
    summary: Dict[str, float] = {
        "mAP50-95": float(metrics.box.map),
        "mAP50": float(metrics.box.map50),
        "mAP75": float(metrics.box.map75),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    log.info("Overall: mAP50-95={:.4f} mAP50={:.4f} P={:.3f} R={:.3f}",
             summary["mAP50-95"], summary["mAP50"], summary["precision"], summary["recall"])

    # Per-class AP50 with readable names.
    try:
        for i, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
            cname = names.get(int(i), f"cls{int(i)}")
            summary[f"AP50/{cname}"] = float(ap)
            log.info("  {:<14s} AP50={:.4f}", cname, float(ap))
    except Exception as exc:
        log.debug("Per-class metrics unavailable: {}", exc)
    return summary


def main() -> int:
    """CLI entry: validate a trained model on the chosen split."""
    parser = argparse.ArgumentParser(description="Validate NeuraRoads detector.")
    parser.add_argument("--weights", type=str, default=None, help="Path to best.pt.")
    parser.add_argument("--data", type=str, default=None, help="Dataset YAML.")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_config("model_config")
    weights = args.weights or cfg["detector"]["weights"]
    wpath = resolve_path(weights)
    if not wpath.is_file():
        log.error("Weights not found: {}. Train first with train_yolo.py.", wpath)
        return 1

    data = str(resolve_path(args.data or "data/annotations/data.yaml"))
    device = _device_arg(resolve_device(cfg.get("device")) if args.device == "auto" else
                         ("cpu" if args.device == "cpu" else f"cuda:{args.device}"))
    save_dir = resolve_path("src/results/metrics")

    try:
        summary = validate(str(wpath), data, args.split, device, args.imgsz, save_dir)
    except Exception as exc:
        log.error("Validation failed: {}", exc)
        return 1

    # Persist a compact CSV.
    try:
        import pandas as pd

        out = save_dir / f"validation_{args.split}.csv"
        pd.DataFrame([summary]).to_csv(out, index=False)
        log.info("Saved metrics -> {}", out)
    except Exception as exc:
        log.debug("Could not save metrics CSV: {}", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
