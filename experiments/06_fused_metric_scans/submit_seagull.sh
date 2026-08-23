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
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
RANKS="${RANKS:-64}"
CLUSTER_CSV="${CLUSTER_CSV:-$PROJECT_DIR/data/empirical/clusters.csv}"
[[ -s "$CLUSTER_CSV" ]] || {
  printf 'ERROR: cluster assignment is missing: %s\n' "$CLUSTER_CSV" >&2
  exit 1
}
source "$PROJECT_DIR/hpc/seagull/common.sh"
run_variant baseline "$RANKS" 1 \
  --partition cyclic --synchronous-observations \
  --disable-persistent-risk-collective \
  --shared-quote-multiplier 2.00
run_variant fused "$RANKS" 1 \
  --partition cyclic --synchronous-observations \
  --disable-persistent-risk-collective \
  --shared-quote-multiplier 2.00 \
  --fuse-metric-cluster-scans
