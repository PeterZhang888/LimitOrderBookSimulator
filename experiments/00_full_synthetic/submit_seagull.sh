#!/usr/bin/env bash
#SBATCH --job-name=lob-full-synthetic
#SBATCH --nodes=16
#SBATCH --ntasks=256
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
BASE_CONFIG="${BASE_CONFIG:-$PROJECT_DIR/examples/synthetic/templates.csv}"
ASSET_COUNT="${ASSET_COUNT:-10000}"
BACKGROUND_MODEL=legacy
REPETITIONS="${REPETITIONS:-1}"
DURATION_SECONDS="${DURATION_SECONDS:-23400}"

source "$PROJECT_DIR/hpc/seagull/common.sh"

run_variant full_10000 256 1 \
  --partition cyclic --synchronous-observations \
  --disable-persistent-risk-collective \
  --shared-inventory-policy gross_pooled
