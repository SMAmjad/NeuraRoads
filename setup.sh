#!/usr/bin/env bash
# =============================================================================
# NeuraRoads ADAS - top-level setup entry point.
# Delegates to the full environment setup script.
#
#   bash setup.sh            # CUDA 12.1 GPU build
#   CUDA=cpu bash setup.sh   # CPU-only build
#
# On Windows use the venv that already ships with the project (./venv), or run
# the equivalent commands documented in docs/INSTALLATION_GUIDE.md.
# =============================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$PROJECT_ROOT/src/scripts/setup_environment.sh"
