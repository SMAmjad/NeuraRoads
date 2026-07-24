#!/usr/bin/env bash
# =============================================================================
# NeuraRoads ADAS - Jetson Nano performance optimisation
# -----------------------------------------------------------------------------
# 1) Sets max power mode + locks clocks.
# 2) Adds swap (TensorRT engine builds are memory-hungry on the 4GB Nano).
# 3) Builds the INT8 TensorRT engine from the trained YOLOv8n weights.
#
#   sudo bash src/deployment/jetson/optimize_jetson.sh
# Reads knobs from configs/jetson_config.yaml (the `jetson:` block).
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
echo "[NeuraRoads/Jetson] Optimising. Project root: $PROJECT_ROOT"

# 1) Max performance: MAXN power mode + locked clocks.
if command -v nvpmodel >/dev/null 2>&1; then
  sudo nvpmodel -m 0 || true            # 0 = MAXN (10W)
fi
if command -v jetson_clocks >/dev/null 2>&1; then
  sudo jetson_clocks || true
fi

# 2) Swap for TensorRT builder memory pressure (4 GB Nano).
SWAP_GB="${SWAP_GB:-4}"
if [ ! -f /var/swapfile ]; then
  echo "[NeuraRoads/Jetson] Creating ${SWAP_GB}G swap"
  sudo fallocate -l "${SWAP_GB}G" /var/swapfile
  sudo chmod 600 /var/swapfile
  sudo mkswap /var/swapfile
  sudo swapon /var/swapfile
fi

# 3) Build the INT8 TensorRT engine from the trained YOLOv8n weights.
#    Falls back to FP16 if INT8 calibration data is unavailable.
PY="${PY:-python3}"
if [ -d "venv_jetson" ]; then
  # shellcheck disable=SC1091
  source venv_jetson/bin/activate
fi

echo "[NeuraRoads/Jetson] Building TensorRT engine (INT8) ..."
$PY src/scripts/convert_to_tensorrt.py --jetson --int8 || \
  $PY src/scripts/convert_to_tensorrt.py --jetson --half || \
  echo "[NeuraRoads/Jetson] Engine build skipped (train the model first, or install tensorrt)."

echo "[NeuraRoads/Jetson] Optimisation complete."
echo "Run: python3 src/deployment/jetson/jetson_inference.py --source <video>"
