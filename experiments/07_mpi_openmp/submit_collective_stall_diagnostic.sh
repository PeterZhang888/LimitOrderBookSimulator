#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_DIR"

mkdir -p slurm results/runs
job=$(
  sbatch --parsable \
    experiments/07_mpi_openmp/run_collective_stall_diagnostic.sbatch
)
job="${job%%;*}"
[[ "$job" =~ ^[0-9]+$ ]] || {
  printf 'ERROR: invalid job identifier: %s\n' "$job" >&2
  exit 1
}

printf 'STALL_DIAGNOSTIC_JOB=%s\n' "$job" \
  > results/runs/LATEST_COLLECTIVE_STALL_DIAGNOSTIC.env
printf 'Submitted focused 32-rank collective diagnostic: %s\n' "$job"
squeue -j "$job" -o "%.18i %.30j %.10T %.12M %.45R"
