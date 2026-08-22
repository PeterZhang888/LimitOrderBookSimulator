#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_DIR"

mkdir -p slurm results/seagull
STATE_FILE="$PROJECT_DIR/results/seagull/LATEST_FIXED_OWNERSHIP_MATRIX.env"
: > "$STATE_FILE"

for total_cores in 16 32 64; do
  nodes=$((total_cores / 16))
  job=$(
    sbatch --parsable \
      --job-name="lob-fixed-${total_cores}c" \
      --nodes="$nodes" \
      --ntasks="$total_cores" \
      --ntasks-per-node=16 \
      --export="ALL,TOTAL_CORES=$total_cores" \
      experiments/07_mpi_openmp/run_fixed_ownership_matrix.sbatch
  )
  job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || {
    printf 'ERROR: invalid job identifier: %s\n' "$job" >&2
    exit 1
  }
  printf 'MATRIX_JOB_%s=%s\n' "$total_cores" "$job" \
    >> "$STATE_FILE"
  printf 'Submitted %s-core permanent-ownership matrix: %s\n' \
    "$total_cores" "$job"
done

source "$STATE_FILE"
squeue -j "$MATRIX_JOB_16,$MATRIX_JOB_32,$MATRIX_JOB_64" \
  -o "%.18i %.30j %.10T %.12M %.45R"
