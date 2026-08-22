#!/usr/bin/env bash
#SBATCH --job-name=lob-omp-window-only
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results/seagull/${SLURM_JOB_ID}_openmp_window_only}"
REPETITIONS=7
DURATION_SECONDS=23400
CORES_PER_NODE=16
BACKGROUND_MODEL=queue-reactive-v1

CONTROL_BUILD_DIR="$PROJECT_DIR/build-comparison/all-phases"
TREATMENT_BUILD_DIR="$PROJECT_DIR/build-comparison/window-only"
test -x "$CONTROL_BUILD_DIR/lob_mpi"
test -x "$TREATMENT_BUILD_DIR/lob_mpi"

source "$PROJECT_DIR/hpc/seagull/common.sh"

common_arguments=(
  --partition cyclic
  --synchronous-observations
  --disable-persistent-risk-collective
  --openmp-schedule dynamic1
  --shared-inventory-policy gross_pooled
)

BUILD_DIR="$CONTROL_BUILD_DIR"
run_variant all_phases 32 2 "${common_arguments[@]}"

BUILD_DIR="$TREATMENT_BUILD_DIR"
run_variant window_only 32 2 "${common_arguments[@]}"

python3 "$PROJECT_DIR/scripts/summarize_openmp_window_comparison.py" \
  "$RESULT_ROOT"
