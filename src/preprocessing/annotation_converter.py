"""Annotation utilities: COCO->YOLO conversion, class remapping and dataset verification.

The NeuraRoads dataset is already in YOLO format, so the most useful operations
here are verifying label integrity against the 10 classes and remapping class
ids if a new data source uses a different ordering. A COCO->YOLO converter is
included for importing additional data.

Usage::

    python src/preprocessing/annotation_converter.py verify
    python src/preprocessing/annotation_converter.py verify --labels data/datasets/labels/train
    python src/preprocessing/annotation_converter.py remap --labels dir --map "5:4,6:7"
    python src/preprocessing/annotation_converter.py coco2yolo --json ann.json --images imgs --out labels
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.config_loader import load_config, resolve_path
from utils.logger import get_logger

log = get_logger(__name__)


def _class_names() -> Dict[int, str]:
    """Load the authoritative id->name map from model_config."""
    return {int(k): str(v) for k, v in load_config("model_config")["detector"]["class_names"].items()}


def verify_labels(labels_dir: Path, num_classes: int = 10) -> Dict[str, int]:
    """Validate YOLO label files and return a per-class instance count.

    Checks each ``.txt``: 5 columns, integer class in ``[0, num_classes)`` and
    normalised coordinates in ``[0, 1]``. Problems are logged, not raised.

    Args:
        labels_dir: Directory of YOLO ``.txt`` label files.
        num_classes: Expected number of classes.

    Returns:
        Mapping ``class_name -> instance_count``.
    """
    names = _class_names()
    counts: Counter = Counter()
    bad_files = 0
    files = list(labels_dir.glob("*.txt"))
    for f in files:
        try:
            for ln, line in enumerate(f.read_text().splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    log.warning("{}:{} expected 5 cols, got {}", f.name, ln, len(parts))
                    bad_files += 1
                    break
                cid = int(float(parts[0]))
                coords = [float(x) for x in parts[1:]]
                if not (0 <= cid < num_classes):
                    log.warning("{}:{} class {} out of range", f.name, ln, cid)
                if any(not (0.0 <= c <= 1.0) for c in coords):
                    log.warning("{}:{} coords not normalised: {}", f.name, ln, coords)
                counts[names.get(cid, f"cls{cid}")] += 1
        except Exception as exc:
            log.warning("Could not parse {}: {}", f.name, exc)
            bad_files += 1

    log.info("Verified {} label files ({} problematic).", len(files), bad_files)
    for name in names.values():
        log.info("  {:<14s}: {}", name, counts.get(name, 0))
    return dict(counts)


def remap_labels(labels_dir: Path, mapping: Dict[int, int], in_place: bool = True,
                 out_dir: Optional[Path] = None) -> int:
    """Remap class ids in every YOLO label file.

    Args:
        labels_dir: Source label directory.
        mapping: ``{old_id: new_id}``. Ids not present are left unchanged.
        in_place: Overwrite files when True; otherwise write to ``out_dir``.
        out_dir: Destination when ``in_place`` is False.

    Returns:
        Number of files modified.
    """
    dst = labels_dir if in_place else (out_dir or labels_dir.parent / f"{labels_dir.name}_remap")
    if not in_place:
        dst.mkdir(parents=True, exist_ok=True)
    modified = 0
    for f in labels_dir.glob("*.txt"):
        out_lines: List[str] = []
        changed = False
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                old = int(float(parts[0]))
                if old in mapping:
                    parts[0] = str(mapping[old])
                    changed = True
                out_lines.append(" ".join(parts))
        (dst / f.name).write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
        modified += int(changed)
    log.info("Remapped {} files -> {}", modified, dst)
    return modified


def coco_to_yolo(json_path: Path, images_dir: Path, out_dir: Path) -> int:
    """Convert a COCO detection JSON to YOLO ``.txt`` labels.

    Args:
        json_path: COCO annotations JSON.
        images_dir: Directory of the referenced images (for sizes fallback).
        out_dir: Output directory for YOLO labels.

    Returns:
        Number of label files written.
    """
    data = json.loads(Path(json_path).read_text())
    images = {img["id"]: img for img in data.get("images", [])}
    # Map COCO category ids to a contiguous 0-based index.
    cats = sorted(data.get("categories", []), key=lambda c: c["id"])
    cat_index = {c["id"]: i for i, c in enumerate(cats)}
    log.info("COCO categories -> YOLO indices: {}",
             {c["name"]: cat_index[c["id"]] for c in cats})

    by_image: Dict[int, List[str]] = {}
    for ann in data.get("annotations", []):
        img = images.get(ann["image_id"])
        if img is None:
            continue
        w, h = img["width"], img["height"]
        x, y, bw, bh = ann["bbox"]  # COCO: top-left x,y + w,h
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw, nh = bw / w, bh / h
        cid = cat_index.get(ann["category_id"], 0)
        by_image.setdefault(ann["image_id"], []).append(
            f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for img_id, img in images.items():
        stem = Path(img["file_name"]).stem
        lines = by_image.get(img_id, [])
        (out_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        written += 1
    log.info("Wrote {} YOLO label files -> {}", written, out_dir)
    return written


def _parse_map(spec: str) -> Dict[int, int]:
    """Parse a ``"old:new,old:new"`` remap spec into a dict."""
    out: Dict[int, int] = {}
    for pair in spec.split(","):
        old, new = pair.split(":")
        out[int(old)] = int(new)
    return out


def main() -> int:
    """CLI entry with verify / remap / coco2yolo subcommands."""
    parser = argparse.ArgumentParser(description="NeuraRoads annotation utilities.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="Validate YOLO labels + class distribution.")
    v.add_argument("--labels", type=str, default="data/datasets/labels/train")

    r = sub.add_parser("remap", help="Remap class ids in YOLO labels.")
    r.add_argument("--labels", type=str, required=True)
    r.add_argument("--map", type=str, required=True, help='e.g. "5:4,6:7"')
    r.add_argument("--out", type=str, default=None, help="Write to a new dir instead of in place.")

    c = sub.add_parser("coco2yolo", help="Convert COCO JSON to YOLO labels.")
    c.add_argument("--json", type=str, required=True)
    c.add_argument("--images", type=str, required=True)
    c.add_argument("--out", type=str, required=True)

    args = parser.parse_args()
    try:
        if args.cmd == "verify":
            verify_labels(resolve_path(args.labels))
        elif args.cmd == "remap":
            remap_labels(resolve_path(args.labels), _parse_map(args.map),
                         in_place=args.out is None,
                         out_dir=resolve_path(args.out) if args.out else None)
        elif args.cmd == "coco2yolo":
            coco_to_yolo(resolve_path(args.json), resolve_path(args.images), resolve_path(args.out))
    except Exception as exc:
        log.error("{} failed: {}", args.cmd, exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
