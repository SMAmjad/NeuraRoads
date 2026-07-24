"""Validate and summarise the YOLO dataset, and (re)generate data.yaml.

Checks the ``images/{train,val,test}`` <-> ``labels/{train,val,test}`` pairing,
reports per-split image/label counts and the class distribution, flags orphan
images/labels, and rewrites ``data/annotations/data.yaml`` to match what's on
disk. Run this before training to catch dataset problems early.

Usage::

    python src/scripts/prepare_dataset.py
    python src/scripts/prepare_dataset.py --root data/datasets --write-yaml
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml

from utils.config_loader import PROJECT_ROOT, load_config, resolve_path
from utils.logger import get_logger

log = get_logger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _class_names() -> Dict[int, str]:
    return {int(k): str(v) for k, v in load_config("model_config")["detector"]["class_names"].items()}


def check_split(images_dir: Path, labels_dir: Path, names: Dict[int, str]) -> Dict[str, object]:
    """Check one split's pairing and class counts. Returns a stats dict."""
    imgs = {p.stem: p for p in images_dir.glob("*") if p.suffix.lower() in IMG_EXTS} if images_dir.is_dir() else {}
    lbls = {p.stem: p for p in labels_dir.glob("*.txt")} if labels_dir.is_dir() else {}
    missing_labels = sorted(set(imgs) - set(lbls))
    orphan_labels = sorted(set(lbls) - set(imgs))
    counts: Counter = Counter()
    for stem, lp in lbls.items():
        for line in lp.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                counts[int(float(parts[0]))] += 1
    return {
        "images": len(imgs), "labels": len(lbls),
        "missing_labels": missing_labels, "orphan_labels": orphan_labels,
        "class_counts": {names.get(k, f"cls{k}"): counts.get(k, 0) for k in names},
    }


def write_data_yaml(root: Path, names: Dict[int, str]) -> Path:
    """(Re)write data/annotations/data.yaml to match the on-disk splits."""
    out = PROJECT_ROOT / "data" / "annotations" / "data.yaml"
    doc = {
        "path": str(root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": {int(k): v for k, v in names.items()},
    }
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log.info("Wrote {}", out)
    return out


def main() -> int:
    """CLI entry: verify dataset structure + distribution and optionally rewrite data.yaml."""
    parser = argparse.ArgumentParser(description="Prepare / verify the YOLO dataset.")
    parser.add_argument("--root", type=str, default="data/datasets", help="Dataset root.")
    parser.add_argument("--write-yaml", action="store_true", help="Rewrite data.yaml.")
    args = parser.parse_args()

    root = resolve_path(args.root)
    names = _class_names()
    if not root.is_dir():
        log.error("Dataset root not found: {}", root)
        return 1

    total = Counter()
    problems = 0
    for split in ("train", "val", "test"):
        stats = check_split(root / "images" / split, root / "labels" / split, names)
        log.info("[{}] images={} labels={} missing_labels={} orphan_labels={}",
                 split, stats["images"], stats["labels"],
                 len(stats["missing_labels"]), len(stats["orphan_labels"]))
        problems += len(stats["missing_labels"]) + len(stats["orphan_labels"])
        for cname, c in stats["class_counts"].items():
            total[cname] += c

    log.info("Total class distribution (all splits):")
    rare: List[str] = []
    grand = sum(total.values()) or 1
    for cname in names.values():
        c = total.get(cname, 0)
        log.info("  {:<14s}: {:>7d}  ({:.2f}%)", cname, c, 100 * c / grand)
        if c < 500:
            rare.append(cname)
    if rare:
        log.warning("Under-represented classes (<500 instances): {}", rare)
        log.warning("Consider data_augmentation.py --classes <ids> to balance before training.")

    if args.write_yaml:
        write_data_yaml(root, names)

    log.info("Dataset check done ({} pairing problems).", problems)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
