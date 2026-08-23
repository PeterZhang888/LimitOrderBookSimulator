#!/usr/bin/env bash
#SBATCH --job-name=lob-empirical
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
read -r -a RANK_VALUES <<< \
  "${RANK_COUNTS_OVERRIDE:-1 2 4 8 16 32 64 128 256}"
source "$PROJECT_DIR/hpc/seagull/common.sh"
for ranks in "${RANK_VALUES[@]}"; do
  [[ "$ranks" =~ ^[1-9][0-9]*$ ]] || {
    printf 'ERROR: rank overrides must be positive integers.\n' >&2
    exit 1
  }
  run_variant "r${ranks}" "$ranks" 1 \
    --partition cyclic --synchronous-observations \
    --disable-persistent-risk-collective \
    --shared-quote-multiplier 2.00 \
    --shared-inventory-policy gross_pooled
done
