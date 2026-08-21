#!/usr/bin/env bash
#SBATCH --job-name=lob-openmp
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
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
source "$PROJECT_DIR/hpc/seagull/common.sh"
base=(--partition cyclic --synchronous-observations
      --disable-persistent-risk-collective --openmp-schedule dynamic1)
run_variant c16_r16_t1 16 1 "${base[@]}"
run_variant c16_r8_t2 8 2 "${base[@]}"
run_variant c16_r4_t4 4 4 "${base[@]}"
run_variant c16_r2_t8 2 8 "${base[@]}"
run_variant c16_r1_t16 1 16 "${base[@]}"
run_openmp_variant c16_openmp_t16 16 "${base[@]}"
run_variant c64_r64_t1 64 1 "${base[@]}"
run_variant c64_r32_t2 32 2 "${base[@]}"
run_variant c64_r16_t4 16 4 "${base[@]}"
run_variant c64_r8_t8 8 8 "${base[@]}"
run_variant c64_r4_t16 4 16 "${base[@]}"
run_variant c64_r32_t2_persistent 32 2 "${base[@]}" \
  --persistent-openmp-team
if [[ -n "${PARTITION_COST_CSV:-}" ]]; then
  run_variant c16_r8_t2_weighted_static 8 2 \
    --partition cyclic --synchronous-observations \
    --disable-persistent-risk-collective \
    --openmp-schedule weighted-static \
    --partition-cost-csv "$PARTITION_COST_CSV"
  run_variant c16_r4_t4_weighted_static 4 4 \
    --partition cyclic --synchronous-observations \
    --disable-persistent-risk-collective \
    --openmp-schedule weighted-static \
    --partition-cost-csv "$PARTITION_COST_CSV"
  run_variant c64_r8_t8_weighted_static 8 8 \
    --partition cyclic --synchronous-observations \
    --disable-persistent-risk-collective \
    --openmp-schedule weighted-static \
    --partition-cost-csv "$PARTITION_COST_CSV"
fi
