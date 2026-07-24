#!/usr/bin/env bash
# =============================================================================
# NeuraRoads ADAS - environment setup (Linux / WSL / Jetson reference)
# -----------------------------------------------------------------------------
# On the primary Windows dev machine the venv already exists (./venv). This
# script reproduces that environment on Linux / WSL / a fresh machine, including
# the OpenCV reconcile step (albumentations pulls the headless build, which
# strips cv2.imshow needed for the preview window).
#
# Usage:
#   bash src/scripts/setup_environment.sh            # CUDA 12.1 (desktop GPU)
#   CUDA=cpu bash src/scripts/setup_environment.sh   # CPU-only
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
echo "[NeuraRoads] Project root: $PROJECT_ROOT"

PYTHON="${PYTHON:-python3}"
CUDA="${CUDA:-cu121}"          # cu121 | cpu
VENV_DIR="${VENV_DIR:-venv}"

# 1) Create the project venv (leave any old .venv untouched).
if [ ! -d "$VENV_DIR" ]; then
  echo "[NeuraRoads] Creating venv at ./$VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# 2) Install PyTorch (CUDA build first so pip doesn't fall back to CPU wheels).
if [ "$CUDA" = "cpu" ]; then
  echo "[NeuraRoads] Installing CPU PyTorch"
  pip install torch==2.5.1 torchvision==0.20.1
else
  echo "[NeuraRoads] Installing CUDA ($CUDA) PyTorch"
  pip install "torch==2.5.1+${CUDA}" "torchvision==0.20.1+${CUDA}" \
    --index-url "https://download.pytorch.org/whl/${CUDA}"
fi

# 3) Install the rest of the requirements.
pip install -r requirements.txt

# 4) Reconcile OpenCV: keep the FULL build (with imshow), drop the headless one.
echo "[NeuraRoads] Reconciling OpenCV (keeping full opencv-python)"
pip uninstall -y opencv-python opencv-python-headless || true
pip install --no-cache-dir opencv-python==4.10.0.84

# 5) Verify torch + CUDA.
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "[NeuraRoads] Setup complete. Activate with: source $VENV_DIR/bin/activate"
