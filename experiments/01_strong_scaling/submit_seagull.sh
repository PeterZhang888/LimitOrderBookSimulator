#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RANK_COUNTS=(1 2 4 8 16 32 64 128 256)
ASSET_COUNTS=(201 1000 2000 5000 10000)

# This file is both the login-node submission driver and the worker executed
# by each Slurm job. Run it with `bash`, not `sbatch`: the driver gives each
# rank count only the number of nodes it needs and records every submitted job.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  cd "$PROJECT_DIR"
  mkdir -p slurm results/seagull

  BASE_CONFIG="${BASE_CONFIG:-$PROJECT_DIR/examples/synthetic/templates.csv}"
  BACKGROUND_MODEL="${BACKGROUND_MODEL:-legacy}"
  REPETITIONS="${REPETITIONS:-7}"
  DURATION_SECONDS="${DURATION_SECONDS:-23400}"
  CORES_PER_NODE="${CORES_PER_NODE:-16}"

  if (( CORES_PER_NODE < 1 )); then
    printf 'ERROR: CORES_PER_NODE must be positive.\n' >&2
    exit 1
  fi
  if [[ ! -x "$PROJECT_DIR/build-mpi/lob_mpi" ]]; then
    printf 'ERROR: build-mpi/lob_mpi is missing; run scripts/build_seagull.sh first.\n' >&2
    exit 1
  fi
  if [[ ! -s "$BASE_CONFIG" ]]; then
    printf 'ERROR: synthetic configuration is missing: %s\n' "$BASE_CONFIG" >&2
    exit 1
  fi

  export PROJECT_DIR BASE_CONFIG BACKGROUND_MODEL
  export REPETITIONS DURATION_SECONDS CORES_PER_NODE

  campaign="strong_scaling_$(date -u +%Y%m%dT%H%M%SZ)"
  campaign_root="$PROJECT_DIR/results/seagull/$campaign"
  manifest="$campaign_root/submitted_jobs.csv"
  mkdir -p "$campaign_root"
  printf 'assets,mpi_ranks,nodes,job_id,result_directory\n' > "$manifest"

  for assets in "${ASSET_COUNTS[@]}"; do
    for ranks in "${RANK_COUNTS[@]}"; do
      nodes=$(((ranks + CORES_PER_NODE - 1) / CORES_PER_NODE))
      result_root="$campaign_root/assets_${assets}/ranks_${ranks}"
      job_id=$(
        ASSET_COUNT="$assets" SCALING_RANKS="$ranks" \
        RESULT_ROOT="$result_root" \
        sbatch --parsable \
          --chdir="$PROJECT_DIR" \
          --job-name="lob-a${assets}-r${ranks}" \
          --nodes="$nodes" \
          --ntasks="$ranks" \
          --cpus-per-task=1 \
          --time=2-00:00:00 \
          --exclusive \
          --hint=nomultithread \
          --output="$PROJECT_DIR/slurm/%x-%j.out" \
          --error="$PROJECT_DIR/slurm/%x-%j.err" \
          --export=ALL \
          "$PROJECT_DIR/experiments/01_strong_scaling/submit_seagull.sh"
      )
      job_id="${job_id%%;*}"
      if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
        printf 'ERROR: invalid job ID for %d books and %d ranks: %s\n' \
          "$assets" "$ranks" "$job_id" >&2
        exit 1
      fi
      printf '%d,%d,%d,%s,%s\n' \
        "$assets" "$ranks" "$nodes" "$job_id" "$result_root" \
        >> "$manifest"
    done
  done

  printf 'Submitted the complete strong-scaling campaign.\n'
  printf 'Job manifest: %s\n' "$manifest"
  column -s, -t < "$manifest"
  exit 0
fi

if [[ -z "${SCALING_RANKS:-}" || -z "${ASSET_COUNT:-}" ]]; then
  printf 'ERROR: submit this experiment with:\n' >&2
  printf '  bash experiments/01_strong_scaling/submit_seagull.sh\n' >&2
  exit 1
fi

source "$PROJECT_DIR/hpc/seagull/common.sh"
run_variant "a${ASSET_COUNT}_r${SCALING_RANKS}" "$SCALING_RANKS" 1 \
  --partition cyclic --synchronous-observations \
  --disable-persistent-risk-collective \
  --shared-inventory-policy gross_pooled
