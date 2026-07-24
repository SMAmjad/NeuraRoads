"""Download the pretrained YOLOP lane-detection model (drivable area + lane lines).

The OBJECT DETECTOR is trained from scratch, so nothing is downloaded for it.
This script fetches the deep lane model that powers the robust primary branch of
the hybrid lane detector: **YOLOP** (trained on BDD100K - day/night/rain/city/
highway), which segments the drivable area and lane lines in one ONNX pass and
runs GPU-accelerated via onnxruntime. Without it the system still runs on the
classical CV detector (automatic fallback); YOLOP is what makes lanes robust on
curves, roundabouts, faded/absent markings and poor light.

The weights are committed in the official YOLOP repo (no Google Drive), so this
downloads them directly. Two sizes are fetched:

* ``yolop-320-320.onnx`` - balanced default (~43 ms on a GTX 1660).
* ``yolop-640-640.onnx`` - max quality for a stronger GPU.

Usage::

    python src/scripts/download_pretrained.py            # download both YOLOP models
    python src/scripts/download_pretrained.py --list     # show expected paths
    python src/scripts/download_pretrained.py --url <U>  # override source URL (640)
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Dict

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.config_loader import load_config, resolve_path  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

# Direct, unauthenticated download URLs (official hustvl/YOLOP repo).
_YOLOP_URLS: Dict[str, str] = {
    "yolop-320-320.onnx": "https://github.com/hustvl/YOLOP/raw/main/weights/yolop-320-320.onnx",
    "yolop-640-640.onnx": "https://github.com/hustvl/YOLOP/raw/main/weights/yolop-640-640.onnx",
}


def _weights_dir() -> Path:
    """Directory where the deep lane weights are expected (from config)."""
    cfg = load_config("model_config")
    return resolve_path(cfg["lane_detector"]["deep"]["weights"]).parent


def _download(url: str, dst: Path) -> bool:
    """Download ``url`` to ``dst`` with a simple progress log. Returns success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading {} -> {}", url, dst)

    def _hook(block_num: int, block_size: int, total: int) -> None:
        if total > 0 and block_num % 100 == 0:
            pct = min(100.0, 100.0 * block_num * block_size / total)
            log.info("  {:.0f}% ({:.1f} MB)", pct, block_num * block_size / 1e6)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
            f.write(r.read())
        log.info("Saved {} ({:.1f} MB).", dst.name, dst.stat().st_size / 1e6)
        return True
    except Exception as exc:
        log.error("Download failed: {}", exc)
        if dst.exists():
            dst.unlink(missing_ok=True)
        return False


def main() -> int:
    """CLI entry: list expected asset paths or download the YOLOP lane models."""
    parser = argparse.ArgumentParser(description="Download the pretrained YOLOP lane model.")
    parser.add_argument("--list", action="store_true", help="List expected asset paths.")
    parser.add_argument("--url", type=str, default=None,
                        help="Override the source URL for the 640 model (mirror).")
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = parser.parse_args()

    wdir = _weights_dir()
    log.info("Object detector: trained FROM SCRATCH (no download needed).")
    log.info("YOLOP lane models expected under: {}", wdir)
    for name in _YOLOP_URLS:
        log.info("  {} present: {}", name, (wdir / name).is_file())
    if args.list:
        return 0

    urls = dict(_YOLOP_URLS)
    if args.url:
        urls["yolop-640-640.onnx"] = args.url

    ok = True
    for name, url in urls.items():
        dst = wdir / name
        if dst.is_file() and not args.force:
            log.info("{} already present - skipping.", name)
            continue
        ok = _download(url, dst) and ok
    if ok:
        log.info("Done. Lane detection will use YOLOP automatically on next run.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
