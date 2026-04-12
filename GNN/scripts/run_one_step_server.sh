#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON:-python}"
PRESET="${PRESET:-server}"
DEVICE="${DEVICE:-cuda}"
CACHE="${CACHE:-${REPO_ROOT}/Real_game_of_life/GNN/cache/frame_graphs_dynamic.pt}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/Real_game_of_life/GNN/runs/one_step_${PRESET}_$(date +%Y%m%d_%H%M%S)}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m Real_game_of_life.GNN.run_full_one_step \
  --preset "${PRESET}" \
  --cache "${CACHE}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --build-cache-if-missing \
  "$@"
