#!/usr/bin/env bash
#SBATCH --job-name=lob-risk
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -Eeuo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
source "$PROJECT_DIR/hpc/seagull/common.sh"
base=(--partition cyclic --disable-persistent-risk-collective
      --shared-quote-multiplier 2.00)
run_variant blocking 64 1 "${base[@]}"
run_variant nonblocking 64 1 "${base[@]}" \
  --nonblocking-risk-collective
run_variant lookahead 64 1 "${base[@]}" \
  --risk-lookahead-max-windows 30
run_variant nonblocking_lookahead 64 1 "${base[@]}" \
  --nonblocking-risk-collective --risk-lookahead-max-windows 30
