#!/usr/bin/env bash
set -Eeuo pipefail

[[ $# -eq 1 ]] || {
  printf 'Usage: bash scripts/submit_experiment.sh EXPERIMENT_DIRECTORY\n' >&2
  exit 2
}

EXPERIMENT_DIR=$(cd "$1" && pwd)
PROJECT_DIR=$(cd "$EXPERIMENT_DIR/../.." && pwd)
SUBMISSION_FILE="$EXPERIMENT_DIR/submit_seagull.sh"
EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")

[[ -f "$SUBMISSION_FILE" ]] || {
  printf 'ERROR: submission file is missing: %s\n' "$SUBMISSION_FILE" >&2
  exit 1
}
for executable in \
  "$PROJECT_DIR/build-mpi/lob_mpi" \
  "$PROJECT_DIR/build-openmp/lob_openmp"
do
  [[ -x "$executable" ]] || {
    printf 'ERROR: compile the repository before submitting this experiment.\n' >&2
    printf 'Run: bash %s/compile.sh\n' "$EXPERIMENT_DIR" >&2
    exit 1
  }
done
bash "$PROJECT_DIR/scripts/verify_build_source.sh" \
  "$PROJECT_DIR/build-mpi" "$PROJECT_DIR/build-openmp"

mkdir -p "$PROJECT_DIR/slurm" "$PROJECT_DIR/results/runs"
cd "$PROJECT_DIR"

case "$EXPERIMENT_NAME" in
  01_strong_scaling|06_mpi_openmp)
    exec bash "$SUBMISSION_FILE"
    ;;
  *)
    exec sbatch --chdir="$PROJECT_DIR" "$SUBMISSION_FILE"
    ;;
esac
