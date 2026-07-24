# =============================================================================
# NeuraRoads ADAS - inference API container (CUDA 12.1)
# -----------------------------------------------------------------------------
# Serves the FastAPI inference service on GPU. Build from the PROJECT ROOT:
#
#   docker build -f src/deployment/docker/Dockerfile -t neuraroads:latest .
#   docker run --gpus all -p 8000:8000 \
#       -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data neuraroads:latest
#
# Requires the NVIDIA Container Toolkit on the host for `--gpus all`.
# =============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 \
    PYTHONPATH=/app/src

# System deps: python + OpenCV runtime libs (libGL/glib for cv2).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install PyTorch (CUDA 12.1) first, then the rest, then reconcile OpenCV.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir opencv-python-headless==4.10.0.84

# Project source + configs + assets.
COPY src /app/src
COPY configs /app/configs
COPY data/annotations /app/data/annotations
COPY data/icons /app/data/icons
# models/ and data/datasets are mounted at runtime (kept out of the image).

EXPOSE 8000

# Healthcheck hits the API liveness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "deployment.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
