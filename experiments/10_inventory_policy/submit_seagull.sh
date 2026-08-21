#!/usr/bin/env bash
#SBATCH --job-name=lob-inventory
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
source "$PROJECT_DIR/hpc/seagull/common.sh"
base=(--partition cyclic --synchronous-observations
      --disable-persistent-risk-collective --shared-quote-relative)
for eta in 0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.50 2.75 3.00 3.25 3.50 3.75 4.00; do
  label=${eta/./p}
  run_variant "asset_local_eta_${label}" 64 1 "${base[@]}" \
    --shared-inventory-policy asset_local \
    --shared-quote-multiplier "$eta"
  run_variant "gross_pooled_eta_${label}" 64 1 "${base[@]}" \
    --shared-inventory-policy gross_pooled \
    --shared-quote-multiplier "$eta"
done
