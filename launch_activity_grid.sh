#!/usr/bin/env bash
# Submit the four bounded directional pilots from a Seagull login node.

set -Eeuo pipefail

: "${SELECTION_ROOT:?export the passed selection root}"
: "${POOL_ROOT:?export the completed five-day pool root}"
command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable" >&2; exit 2; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SELECTION_ROOT="$(cd "${SELECTION_ROOT}" && pwd -P)"
POOL_ROOT="$(cd "${POOL_ROOT}" && pwd -P)"
GRID_TAG="${GRID_TAG:-$(date +%Y%m%dT%H%M%S)}"
GRID_ROOT="${GRID_ROOT:-${PROJECT_DIR}/results/seagull/activity_grid_${GRID_TAG}}"
JOBS_FILE="${GRID_ROOT}/pilot_jobs.tsv"

[[ -s "${SELECTION_ROOT}/selection/training_selection_freeze.json" ]] || {
    echo "ERROR: invalid SELECTION_ROOT=${SELECTION_ROOT}" >&2; exit 2;
}
[[ -s "${POOL_ROOT}/heldout_common.csv" ]] || {
    echo "ERROR: invalid POOL_ROOT=${POOL_ROOT}" >&2; exit 2;
}
[[ ! -e "${JOBS_FILE}" ]] || {
    echo "ERROR: grid already exists: ${GRID_ROOT}" >&2; exit 2;
}
mkdir -p "${GRID_ROOT}" "${PROJECT_DIR}/slurm"
printf 'activity_scale\tjob_id\tresult_root\tbuild_dir\n' >"${JOBS_FILE}"
cd "${PROJECT_DIR}"

for scale in 0.50 0.75 1.00 1.25; do
    tag="${scale/./p}"
    result_root="${GRID_ROOT}/scale_${tag}"
    build_dir="${PROJECT_DIR}/build-seagull-activity-scale_${tag}"
    job_id="$(sbatch --parsable \
        --time=01:30:00 \
        --export="ALL,SELECTION_ROOT=${SELECTION_ROOT},POOL_ROOT=${POOL_ROOT},ACTIVITY_SCALE=${scale},RESULT_DIR=${result_root},BUILD_DIR=${build_dir},PILOT_ONLY=on,RESUME=off,ALLOW_PILOT_REJECTION=on" \
        "${PROJECT_DIR}/submit_queue_reactive_full_validation_hpc.sh")"
    job_id="${job_id%%;*}"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || {
        echo "ERROR: sbatch returned invalid job id '${job_id}'" >&2; exit 2;
    }
    printf '%s\t%s\t%s\t%s\n' \
        "${scale}" "${job_id}" "${result_root}" "${build_dir}" >>"${JOBS_FILE}"
    echo "submitted scale=${scale} job=${job_id}"
done

JOB_LIST="$(awk 'NR>1 {value=value sep $2; sep=","} END {print value}' "${JOBS_FILE}")"
echo "GRID_ROOT=${GRID_ROOT}"
echo "JOBS_FILE=${JOBS_FILE}"
echo "Monitor:"
echo "  squeue -j ${JOB_LIST} -o \"%.18i %.24j %.10T %.12M %.50R\""
echo "After all four finish:"
echo "  bash ${PROJECT_DIR}/summarize_activity_grid.sh ${GRID_ROOT}"
