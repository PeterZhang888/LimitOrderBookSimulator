#!/usr/bin/env bash
# Audit the conventional four-candidate grid written by the launch helper.

set -Eeuo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 /absolute/GRID_ROOT" >&2; exit 2; }
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
GRID_ROOT="$(cd "$1" && pwd -P)"

python3 "${PROJECT_DIR}/scripts/summarize_activity_grid.py" \
    --candidate "0.50=${GRID_ROOT}/scale_0p50" \
    --candidate "0.75=${GRID_ROOT}/scale_0p75" \
    --candidate "1.00=${GRID_ROOT}/scale_1p00" \
    --candidate "1.25=${GRID_ROOT}/scale_1p25" \
    --output "${GRID_ROOT}/pilot_grid_selection.json"

echo "selection=${GRID_ROOT}/pilot_grid_selection.json"
echo "If status is passed_candidate_selected, resume with:"
echo "  bash ${PROJECT_DIR}/resume_selected_activity_run.sh ${GRID_ROOT}"
