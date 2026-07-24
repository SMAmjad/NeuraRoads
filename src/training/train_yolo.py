"""Train the NeuraRoads YOLOv8 detector FROM SCRATCH on the 10-class dataset.

No pretrained COCO weights are used: the model is instantiated from the
architecture YAML (random init) and trained on ``data/annotations/data.yaml``.
All hyperparameters come from ``src/training/hyperparameters.yaml`` and may be
overridden on the command line.

Usage::

    python src/training/train_yolo.py                 # full run (150 epochs)
    python src/training/train_yolo.py --epochs 50 --batch 8
    python src/training/train_yolo.py --resume        # resume last run
    python src/training/train_yolo.py --device cpu     # force CPU

Output: ``models/trained/neuraroads_yolov8m/weights/best.pt``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# --- make the `src` package root importable when run as a script -------------
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml

from utils.config_loader import PROJECT_ROOT, load_config, resolve_device, resolve_path
from utils.logger import get_logger

log = get_logger(__name__)

HYP_PATH = _SRC / "training" / "hyperparameters.yaml"


def load_hyperparameters(path: Path = HYP_PATH) -> Dict[str, Any]:
    """Load the training hyperparameters YAML into a dict."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as e.g. ``2h05m`` / ``12m30s`` / ``45s``."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def attach_progress_callbacks(model, total_epochs: int) -> None:
    """Attach callbacks that log a whole-run progress bar after each epoch.

    Ultralytics already prints a per-epoch *iteration* bar; this adds the missing
    big-picture view: which epoch of N finished, how long it took, cumulative
    elapsed time, a run-level progress bar and an ETA for the whole run.

    Args:
        model: The Ultralytics ``YOLO`` model to attach callbacks to.
        total_epochs: Total number of epochs (for the bar + ETA).
    """
    import time

    state: Dict[str, Any] = {"run_start": None, "epoch_start": None}

    def _on_train_start(trainer) -> None:
        state["run_start"] = time.time()

    def _on_epoch_start(trainer) -> None:
        state["epoch_start"] = time.time()

    def _on_fit_epoch_end(trainer) -> None:
        now = time.time()
        ep = int(getattr(trainer, "epoch", 0)) + 1
        total = int(getattr(trainer, "epochs", total_epochs) or total_epochs)
        epoch_dur = now - (state["epoch_start"] or now)
        elapsed = now - (state["run_start"] or now)
        avg = elapsed / max(ep, 1)
        eta = avg * max(0, total - ep)
        frac = ep / max(total, 1)
        bar_len = 24
        filled = int(round(bar_len * frac))
        bar = "#" * filled + "-" * (bar_len - filled)

        # Pull val metrics if available this epoch.
        metrics = getattr(trainer, "metrics", {}) or {}
        extra = ""
        m50 = metrics.get("metrics/mAP50(B)")
        m5095 = metrics.get("metrics/mAP50-95(B)")
        if isinstance(m50, (int, float)):
            extra = f" | mAP50={m50:.3f} mAP50-95={m5095:.3f}" if isinstance(m5095, (int, float)) \
                else f" | mAP50={m50:.3f}"

        log.info("EPOCH {}/{}  [{}] {:5.1f}%  | this epoch {} | elapsed {} | ETA {}{}",
                 ep, total, bar, frac * 100.0,
                 _fmt_duration(epoch_dur), _fmt_duration(elapsed), _fmt_duration(eta), extra)

    model.add_callback("on_train_start", _on_train_start)
    model.add_callback("on_train_epoch_start", _on_epoch_start)
    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)


def to_ultralytics_device(device_str: str) -> Any:
    """Convert a torch device string to the value Ultralytics expects.

    ``"cuda:0"`` -> ``0``; ``"cpu"`` -> ``"cpu"``.
    """
    if device_str.startswith("cuda"):
        return int(device_str.split(":")[1]) if ":" in device_str else 0
    return "cpu"


def build_train_args(hyp: Dict[str, Any], args: argparse.Namespace, device: Any) -> Dict[str, Any]:
    """Assemble the keyword args passed to ``YOLO.train`` from hyp + CLI."""
    data_path = resolve_path(hyp.get("data", "data/annotations/data.yaml"))
    train_args: Dict[str, Any] = {
        "data": str(data_path),
        "epochs": args.epochs if args.epochs is not None else hyp.get("epochs", 150),
        "imgsz": args.imgsz if args.imgsz is not None else hyp.get("imgsz", 640),
        "batch": args.batch if args.batch is not None else hyp.get("batch", -1),
        "device": device,
        "workers": hyp.get("workers", 8),
        "seed": hyp.get("seed", 42),
        "deterministic": hyp.get("deterministic", True),
        "pretrained": hyp.get("pretrained", False),  # FROM SCRATCH
        "optimizer": hyp.get("optimizer", "SGD"),
        "lr0": hyp.get("lr0", 0.01),
        "lrf": hyp.get("lrf", 0.01),
        "momentum": hyp.get("momentum", 0.937),
        "weight_decay": hyp.get("weight_decay", 0.0005),
        "warmup_epochs": hyp.get("warmup_epochs", 3.0),
        "warmup_momentum": hyp.get("warmup_momentum", 0.8),
        "warmup_bias_lr": hyp.get("warmup_bias_lr", 0.1),
        "cos_lr": hyp.get("cos_lr", True),
        "box": hyp.get("box", 7.5),
        "cls": hyp.get("cls", 0.5),
        "dfl": hyp.get("dfl", 1.5),
        "patience": hyp.get("patience", 30),
        "amp": hyp.get("amp", True),
        "cache": hyp.get("cache", False),
        "close_mosaic": hyp.get("close_mosaic", 10),
        "save_period": hyp.get("save_period", 10),
        "val": hyp.get("val", True),
        "plots": hyp.get("plots", True),
        # Augmentation
        "hsv_h": hyp.get("hsv_h", 0.015), "hsv_s": hyp.get("hsv_s", 0.7), "hsv_v": hyp.get("hsv_v", 0.4),
        "degrees": hyp.get("degrees", 0.0), "translate": hyp.get("translate", 0.1),
        "scale": hyp.get("scale", 0.5), "shear": hyp.get("shear", 0.0),
        "perspective": hyp.get("perspective", 0.0), "flipud": hyp.get("flipud", 0.0),
        "fliplr": hyp.get("fliplr", 0.5), "mosaic": hyp.get("mosaic", 1.0),
        "mixup": hyp.get("mixup", 0.1), "copy_paste": hyp.get("copy_paste", 0.0),
        # Output
        "project": str(resolve_path(hyp.get("project", "models/trained"))),
        "name": args.name or hyp.get("name", "neuraroads_yolov8m"),
        "exist_ok": args.exist_ok or hyp.get("exist_ok", False),
        "resume": args.resume,
    }
    return train_args


def main() -> int:
    """Parse CLI args, train the model from scratch and report final metrics."""
    parser = argparse.ArgumentParser(description="Train NeuraRoads YOLOv8 from scratch.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size (-1 = auto).")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size.")
    parser.add_argument("--device", type=str, default=None, help="'auto', 'cpu', or GPU index.")
    parser.add_argument("--name", type=str, default=None, help="Run name / output dir.")
    parser.add_argument("--arch", type=str, default=None, help="Override architecture (e.g. yolov8n).")
    parser.add_argument("--resume", action="store_true", help="Resume the last run.")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing run dir.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics is not installed. Activate the project venv.")
        return 1

    hyp = load_hyperparameters()
    arch = args.arch or hyp.get("architecture", "yolov8m")

    # Device: honor CLI, else the model_config auto-detect.
    if args.device in (None, "auto"):
        device = to_ultralytics_device(resolve_device(load_config("model_config").get("device")))
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = int(args.device)

    log.info("Training {} FROM SCRATCH (pretrained={}) on device {}.",
             arch, hyp.get("pretrained", False), device)

    # From-scratch: build from the architecture YAML (random init).
    model = YOLO(f"{arch}.yaml")

    train_args = build_train_args(hyp, args, device)
    log.info("Dataset: {} | epochs={} | imgsz={} | batch={}",
             train_args["data"], train_args["epochs"], train_args["imgsz"], train_args["batch"])

    # Whole-run progress bar + ETA after every epoch (on top of Ultralytics' own
    # per-epoch iteration bar).
    attach_progress_callbacks(model, int(train_args["epochs"]))

    try:
        results = model.train(**train_args)
    except Exception as exc:
        log.error("Training failed: {}", exc)
        if "out of memory" in str(exc).lower():
            log.error("CUDA OOM on 6 GB - retry with a smaller --batch (e.g. 8 or 4).")
        return 1

    best = resolve_path(train_args["project"]) / train_args["name"] / "weights" / "best.pt"
    log.info("Training complete. Best weights: {}", best)
    try:
        metrics = model.val(data=train_args["data"], device=device)
        log.info("Final val mAP50-95={:.4f} mAP50={:.4f}",
                 float(metrics.box.map), float(metrics.box.map50))
    except Exception as exc:
        log.warning("Post-training validation skipped: {}", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
