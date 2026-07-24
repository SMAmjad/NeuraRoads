"""FastAPI routes for the NeuraRoads ADAS inference service.

Endpoints:

* ``GET  /health``          - liveness + device/model info.
* ``GET  /config``          - the active (sanitised) configuration.
* ``POST /infer/image``     - annotate a single uploaded image, return PNG.
* ``POST /infer/video``     - annotate an uploaded video, return the MP4.

The heavy pipeline is created once and reused (see :mod:`deployment.api.app`).
Video processing is synchronous - fine for short clips; for long videos run the
CLI (``inference_video.py``) or extend this with a background-job queue.
"""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response

from utils.config_loader import resolve_path
from utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()


def _pipeline(request: Request):
    """Fetch the shared pipeline from app state (built at startup)."""
    pipe = getattr(request.app.state, "pipeline", None)
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")
    return pipe


@router.get("/health")
def health(request: Request) -> Dict[str, Any]:
    """Liveness probe with device + detector status."""
    pipe = getattr(request.app.state, "pipeline", None)
    return {
        "status": "ok",
        "device": getattr(pipe, "device", "unknown"),
        "detector_loaded": bool(pipe and pipe.detector is not None),
        "lane_mode": getattr(getattr(pipe, "lane", None), "mode", "unknown"),
    }


@router.get("/config")
def config(request: Request) -> Dict[str, Any]:
    """Return the active detector class map and key runtime settings."""
    pipe = _pipeline(request)
    return {
        "classes": pipe.cfg["detector"]["class_names"],
        "imgsz": pipe.cfg["detector"]["imgsz"],
        "target_fps": pipe.cfg["pipeline"]["target_fps"],
        "device": pipe.device,
    }


@router.post("/infer/image")
async def infer_image(request: Request, file: UploadFile = File(...)) -> Response:
    """Annotate one uploaded image and return it as PNG."""
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    pipe = _pipeline(request)
    annotated, _ = pipe.process_frame(frame, 1.0 / 30.0)
    pipe.reset()  # single image -> no temporal carry-over
    ok, buf = cv2.imencode(".png", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Encoding failed.")
    return Response(content=buf.tobytes(), media_type="image/png")


@router.post("/infer/video")
async def infer_video(request: Request, file: UploadFile = File(...)) -> FileResponse:
    """Annotate an uploaded video and return the resulting MP4."""
    from pipeline.realtime_pipeline import RealtimePipeline

    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    tmp_in = Path(tempfile.gettempdir()) / f"nr_{uuid.uuid4().hex}{suffix}"
    with tmp_in.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    out_dir = resolve_path("src/results/videos/output_videos")
    out_path = out_dir / f"api_{uuid.uuid4().hex}.mp4"
    allow_no_detector = bool(getattr(request.app.state, "allow_no_detector", False))
    overlay = getattr(request.app.state, "overlay", None)

    try:
        rt = RealtimePipeline(overlay=overlay, allow_no_detector=allow_no_detector)
        rt.run(source=str(tmp_in), output_path=str(out_path), show_preview=False)
    except Exception as exc:
        log.error("API video inference failed: {}", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    finally:
        tmp_in.unlink(missing_ok=True)

    if not out_path.is_file():
        raise HTTPException(status_code=500, detail="No output produced.")
    return FileResponse(str(out_path), media_type="video/mp4", filename=out_path.name)
