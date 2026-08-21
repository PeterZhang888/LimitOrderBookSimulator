#!/usr/bin/env bash
#SBATCH --job-name=lob-buffering
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -Eeuo pipefail
source "$PROJECT_DIR/hpc/seagull/common.sh"
run_variant synchronous 16 1 \
  --partition cyclic --synchronous-observations \
  --disable-persistent-risk-collective
run_variant buffered 16 1 \
  --partition cyclic --disable-persistent-risk-collective
