#!/usr/bin/env bash
# Resume the expensive matrix only from the grid auditor's passed candidate.

set -Eeuo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 /absolute/GRID_ROOT" >&2; exit 2; }
: "${SELECTION_ROOT:?export the same passed R27 selection root}"
: "${POOL_ROOT:?export the same completed R26 pool root}"
command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable" >&2; exit 2; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
GRID_ROOT="$(cd "$1" && pwd -P)"
SELECTION_ROOT="$(cd "${SELECTION_ROOT}" && pwd -P)"
POOL_ROOT="$(cd "${POOL_ROOT}" && pwd -P)"
SELECTION_JSON="${GRID_ROOT}/pilot_grid_selection.json"
JOBS_FILE="${GRID_ROOT}/pilot_jobs.tsv"
[[ -s "${SELECTION_JSON}" && -s "${JOBS_FILE}" ]] || {
    echo "ERROR: grid audit or job ledger is absent in ${GRID_ROOT}" >&2; exit 2;
}

IFS=$'\t' read -r ACTIVITY_SCALE RESULT_DIR BUILD_DIR < <(
python3 - "${SELECTION_JSON}" "${JOBS_FILE}" <<'PY'
import csv
import json
import math
import pathlib
import sys

selection = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if selection.get("status") != "passed_candidate_selected":
    raise SystemExit("grid has no passed candidate; full matrix is forbidden")
chosen = selection.get("selected")
if not isinstance(chosen, dict):
    raise SystemExit("selected candidate record is malformed")
scale = float(chosen["activity_scale"])
result_root = str(pathlib.Path(chosen["result_root"]).resolve())
with pathlib.Path(sys.argv[2]).open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
matches = [row for row in rows if math.isclose(
    float(row["activity_scale"]), scale, rel_tol=0.0, abs_tol=1.0e-12
)]
if len(matches) != 1 or str(pathlib.Path(matches[0]["result_root"]).resolve()) != result_root:
    raise SystemExit("selected result does not match the immutable job ledger")
print(f"{scale:.2f}\t{result_root}\t{matches[0]['build_dir']}")
PY
)

[[ -s "${RESULT_DIR}/full_training_adequacy/directional_pilot/directional_pilot_handoff.json" ]] || {
    echo "ERROR: selected candidate has no passed handoff" >&2; exit 2;
}

cd "${PROJECT_DIR}"
FULL_JOB="$(sbatch --parsable \
    --time=06:00:00 \
    --export="ALL,SELECTION_ROOT=${SELECTION_ROOT},POOL_ROOT=${POOL_ROOT},ACTIVITY_SCALE=${ACTIVITY_SCALE},RESULT_DIR=${RESULT_DIR},BUILD_DIR=${BUILD_DIR},PILOT_ONLY=off,RESUME=on,ALLOW_PILOT_REJECTION=off" \
    "${PROJECT_DIR}/submit_queue_reactive_full_validation_hpc.sh")"
FULL_JOB="${FULL_JOB%%;*}"
[[ "${FULL_JOB}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: sbatch returned invalid job id '${FULL_JOB}'" >&2; exit 2;
}
printf '%s\n' "${FULL_JOB}" >"${GRID_ROOT}/full_job_id.txt"
echo "FULL_JOB=${FULL_JOB}"
echo "activity_scale=${ACTIVITY_SCALE}"
echo "result_dir=${RESULT_DIR}"
echo "Monitor:"
echo "  squeue -j ${FULL_JOB} -o \"%.18i %.24j %.10T %.12M %.50R\""
