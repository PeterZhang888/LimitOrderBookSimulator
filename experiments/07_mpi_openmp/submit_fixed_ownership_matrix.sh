#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_DIR"

mkdir -p slurm results/runs
STATE_FILE="$PROJECT_DIR/results/runs/LATEST_FIXED_OWNERSHIP_MATRIX.env"
: > "$STATE_FILE"
read -r -a TOTAL_CORE_VALUES <<< "${TOTAL_CORES_OVERRIDE:-16 32}"

job_ids=()
for total_cores in "${TOTAL_CORE_VALUES[@]}"; do
  case "$total_cores" in
    16|32|64) ;;
    *)
      printf 'ERROR: total-core overrides must contain 16, 32, or 64.\n' >&2
      exit 1
      ;;
  esac
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
  job_ids+=("$job")
  printf 'Submitted %s-core permanent-ownership matrix: %s\n' \
    "$total_cores" "$job"
done

job_list=$(IFS=,; printf '%s' "${job_ids[*]}")
squeue -j "$job_list" \
  -o "%.18i %.30j %.10T %.12M %.45R"
