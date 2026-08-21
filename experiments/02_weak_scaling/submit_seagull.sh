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
: "${BASE_CONFIG:?set BASE_CONFIG to the synthetic template CSV}"
ASSET_COUNT=40
source "$PROJECT_DIR/hpc/seagull/common.sh"
for ranks in 1 2 4 8 16 32 64 128 256; do
  ASSET_COUNT=$((40 * ranks))
  INPUT_ARGS=(--base-config "$BASE_CONFIG" --assets "$ASSET_COUNT")
  run_variant "r${ranks}" "$ranks" 1 \
    --partition cyclic --synchronous-observations \
    --disable-persistent-risk-collective \
    --shared-inventory-policy gross_pooled
done
