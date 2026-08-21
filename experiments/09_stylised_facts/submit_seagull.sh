#!/usr/bin/env bash
#SBATCH --job-name=lob-facts
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
REPETITIONS="${REPETITIONS:-1}"
source "$PROJECT_DIR/hpc/seagull/common.sh"
run_variant validation 64 1 \
  --partition cyclic --synchronous-observations \
  --disable-persistent-risk-collective \
  --shared-inventory-policy gross_pooled
