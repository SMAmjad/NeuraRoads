"""FastAPI application factory for the NeuraRoads ADAS service.

Builds a single shared :class:`NeuraRoadsPipeline` at startup (so the model is
loaded once) and mounts the routes from :mod:`deployment.api.endpoints`.

Run with::

    uvicorn deployment.api.app:app --host 0.0.0.0 --port 8000
    # or, before the model is trained:
    NR_ALLOW_NO_DETECTOR=1 uvicorn deployment.api.app:app --reload

Environment variables:
    NR_ALLOW_NO_DETECTOR  "1" to run lane/ego/HUD only (no trained model needed).
    NR_OVERLAY            Optional config overlay name (e.g. "jetson_config").
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# --- make `src` importable when launched via uvicorn from anywhere ----------
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deployment.api.endpoints import router
from pipeline.inference_pipeline import NeuraRoadsPipeline
from utils.logger import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the pipeline on startup; release resources on shutdown."""
    allow_no_detector = os.environ.get("NR_ALLOW_NO_DETECTOR", "0") == "1"
    overlay = os.environ.get("NR_OVERLAY") or None
    app.state.allow_no_detector = allow_no_detector
    app.state.overlay = overlay
    log.info("Initialising pipeline (allow_no_detector={}, overlay={})...", allow_no_detector, overlay)
    try:
        app.state.pipeline = NeuraRoadsPipeline(overlay=overlay, allow_no_detector=allow_no_detector,
                                                frame_rate=30.0)
        app.state.pipeline.warmup()
        log.info("Pipeline ready on {}.", app.state.pipeline.device)
    except Exception as exc:
        log.error("Failed to initialise pipeline: {}", exc)
        app.state.pipeline = None
    yield
    app.state.pipeline = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="NeuraRoads ADAS API",
        description="Real-time ADAS inference: detection, tracking, distance, "
                    "speed, TTC, lanes, BEV and collision/lane/pedestrian warnings.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("deployment.api.app:app", host="0.0.0.0", port=8000, reload=False)
