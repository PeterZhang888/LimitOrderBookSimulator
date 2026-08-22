#!/usr/bin/env bash
#SBATCH --job-name=lob-full-10000-r64
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
BASE_CONFIG="${BASE_CONFIG:-$PROJECT_DIR/examples/synthetic/templates.csv}"
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build-mpi}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results/seagull/${SLURM_JOB_ID}_full_10000_r64}"
ASSET_COUNT=10000
BACKGROUND_MODEL=legacy
REPETITIONS=1
DURATION_SECONDS=23400
CORES_PER_NODE=16

source "$PROJECT_DIR/hpc/seagull/common.sh"

run_variant full_10000_r64 64 1 \
  --partition cyclic \
  --synchronous-observations \
  --disable-persistent-risk-collective \
  --shared-inventory-policy gross_pooled
