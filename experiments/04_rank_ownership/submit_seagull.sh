#!/usr/bin/env bash
#SBATCH --job-name=lob-ownership
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -Eeuo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
source "$PROJECT_DIR/hpc/seagull/common.sh"
for ranks in 16 32 64; do
  run_variant "r${ranks}_cyclic" "$ranks" 1 \
    --partition cyclic --synchronous-observations \
    --disable-persistent-risk-collective \
    --shared-quote-multiplier 2.00
  run_variant "r${ranks}_weighted" "$ranks" 1 \
    --partition weighted --synchronous-observations \
    --disable-persistent-risk-collective \
    --shared-quote-multiplier 2.00
done
