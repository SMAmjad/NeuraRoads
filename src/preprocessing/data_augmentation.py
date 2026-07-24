"""Offline dataset augmentation with Albumentations (bbox-safe, YOLO format).

Ultralytics already augments online during training, so this is mainly for
*offline* balancing - generating extra copies of images containing rare classes
(e.g. Train, Bicycle) to reduce class imbalance before training. Augmentations
are photometric + mild geometric, chosen to stay realistic for dashcam scenes
(night/rain/weather simulation) while keeping boxes valid.

Usage::

    python src/preprocessing/data_augmentation.py --images data/datasets/images/train \
        --labels data/datasets/labels/train --out-images aug/images --out-labels aug/labels \
        --classes 8,0 --multiplier 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cv2

from utils.logger import get_logger
from utils.config_loader import resolve_path

log = get_logger(__name__)


def build_transform():
    """Build the Albumentations pipeline (YOLO bbox params)."""
    import albumentations as A

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=20, p=0.5),
            A.OneOf([  # weather / low-light realism
                A.RandomRain(blur_value=3, brightness_coefficient=0.9, p=1.0),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
                A.RandomShadow(p=1.0),
                A.GaussNoise(var_limit=(10, 50), p=1.0),
            ], p=0.5),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=5,
                               border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_ids"], min_visibility=0.3),
    )


def _read_yolo_label(path: Path) -> Tuple[List[List[float]], List[int]]:
    """Read a YOLO label file into ``(bboxes, class_ids)``."""
    bboxes, cls = [], []
    if not path.is_file():
        return bboxes, cls
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            cls.append(int(float(p[0])))
            bboxes.append([float(x) for x in p[1:]])
    return bboxes, cls


def _write_yolo_label(path: Path, bboxes: Sequence[Sequence[float]], class_ids: Sequence[int]) -> None:
    """Write ``(bboxes, class_ids)`` to a YOLO label file."""
    lines = [f"{c} " + " ".join(f"{v:.6f}" for v in box) for box, c in zip(bboxes, class_ids)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def augment_dataset(
    images_dir: Path,
    labels_dir: Path,
    out_images: Path,
    out_labels: Path,
    multiplier: int = 2,
    target_classes: Optional[Sequence[int]] = None,
    img_ext: str = "jpg",
) -> int:
    """Generate augmented copies of images (optionally only those with target classes).

    Args:
        images_dir: Source images.
        labels_dir: Source YOLO labels (same stems as images).
        out_images: Output image dir.
        out_labels: Output label dir.
        multiplier: Number of augmented copies per selected image.
        target_classes: Only augment images containing one of these class ids
            (None = all images).
        img_ext: Output image extension.

    Returns:
        Number of augmented images written.
    """
    transform = build_transform()
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    target = set(target_classes) if target_classes else None

    written = 0
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    for img_path in images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        bboxes, class_ids = _read_yolo_label(label_path)
        if not bboxes:
            continue
        if target is not None and not (set(class_ids) & target):
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        for k in range(multiplier):
            try:
                res = transform(image=image, bboxes=bboxes, class_ids=class_ids)
            except Exception as exc:
                log.debug("Augment failed for {}: {}", img_path.name, exc)
                continue
            if not res["bboxes"]:
                continue
            stem = f"{img_path.stem}_aug{k}"
            cv2.imwrite(str(out_images / f"{stem}.{img_ext}"), res["image"])
            _write_yolo_label(out_labels / f"{stem}.txt", res["bboxes"], res["class_ids"])
            written += 1
    log.info("Wrote {} augmented images -> {}", written, out_images)
    return written


def main() -> int:
    """CLI entry: offline-augment a YOLO dataset."""
    parser = argparse.ArgumentParser(description="Offline dataset augmentation.")
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--out-images", type=str, required=True)
    parser.add_argument("--out-labels", type=str, required=True)
    parser.add_argument("--multiplier", type=int, default=2)
    parser.add_argument("--classes", type=str, default=None,
                        help='Comma-separated class ids to target, e.g. "8,0".')
    args = parser.parse_args()

    target = [int(c) for c in args.classes.split(",")] if args.classes else None
    try:
        augment_dataset(resolve_path(args.images), resolve_path(args.labels),
                        resolve_path(args.out_images), resolve_path(args.out_labels),
                        multiplier=args.multiplier, target_classes=target)
    except Exception as exc:
        log.error("Augmentation failed: {}", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
