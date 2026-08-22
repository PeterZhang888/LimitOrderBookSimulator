#!/usr/bin/env bash
#SBATCH --job-name=lob-weak
#SBATCH --nodes=16
#SBATCH --ntasks=256
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
BACKGROUND_MODEL=legacy
ASSET_COUNT=1
REPETITIONS="${REPETITIONS:-3}"
source "$PROJECT_DIR/hpc/seagull/common.sh"
for books_per_rank in 16 40; do
  for ranks in 1 2 4 8 16 32 64 128 256; do
    ASSET_COUNT=$((books_per_rank * ranks))
    INPUT_ARGS=(--base-config "$BASE_CONFIG" --assets "$ASSET_COUNT")
    run_variant "bpr${books_per_rank}_r${ranks}" "$ranks" 1 \
      --partition cyclic --synchronous-observations \
      --disable-persistent-risk-collective \
      --shared-inventory-policy gross_pooled
  done
done
