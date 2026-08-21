#!/usr/bin/env bash
#SBATCH --job-name=lob-fusion
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
: "${CLUSTER_CSV:?set CLUSTER_CSV for cluster output}"
source "$PROJECT_DIR/hpc/seagull/common.sh"
run_variant baseline 64 1 \
  --partition cyclic --synchronous-observations \
  --disable-persistent-risk-collective
run_variant fused 64 1 \
  --partition cyclic --synchronous-observations \
  --disable-persistent-risk-collective \
  --fuse-metric-cluster-scans
