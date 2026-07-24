#!/usr/bin/env bash
# =============================================================================
# NeuraRoads ADAS - Jetson Nano dependency installer (JetPack 4.6+, Python 3.8)
# -----------------------------------------------------------------------------
# The Jetson uses NVIDIA's own aarch64 PyTorch wheel (NOT the desktop cu121
# wheel) and the system OpenCV/TensorRT that ship with JetPack. This script
# installs the remaining pure-Python deps and wires everything up.
#
#   bash src/deployment/jetson/install_jetson.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
echo "[NeuraRoads/Jetson] Project root: $PROJECT_ROOT"

# 1) System packages (OpenCV, TensorRT and CUDA come with JetPack).
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev \
    libjpeg-dev zlib1g-dev libpython3-dev ffmpeg

# 2) Create a venv that can see the system site-packages (for cv2/tensorrt).
python3 -m pip install --upgrade pip
python3 -m venv --system-site-packages venv_jetson
# shellcheck disable=SC1091
source venv_jetson/bin/activate

# 3) PyTorch for Jetson: install the NVIDIA wheel matching your JetPack.
#    See https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 and set
#    TORCH_WHL to the correct URL for your JetPack, then:
if [ -n "${TORCH_WHL:-}" ]; then
  echo "[NeuraRoads/Jetson] Installing PyTorch from $TORCH_WHL"
  pip install "$TORCH_WHL"
else
  echo "[NeuraRoads/Jetson] NOTE: set TORCH_WHL to the NVIDIA Jetson torch wheel URL"
  echo "                    for your JetPack, then re-run, or install it manually."
fi

# 4) Remaining Python deps (skip torch/opencv - provided above / by JetPack).
pip install ultralytics lapx numpy'<2' scipy filterpy shapely pyyaml loguru \
    tqdm rich onnx psutil pandas || true

# 5) Ultralytics on Jetson pulls opencv-python; prefer the system cv2.
pip uninstall -y opencv-python opencv-python-headless || true

echo "[NeuraRoads/Jetson] Verifying..."
python3 - <<'PY'
try:
    import torch, cv2
    print("torch", torch.__version__, "CUDA:", torch.cuda.is_available())
    print("cv2", cv2.__version__)
except Exception as e:
    print("Verification note:", e)
PY

echo "[NeuraRoads/Jetson] Done. Next: bash src/deployment/jetson/optimize_jetson.sh"
