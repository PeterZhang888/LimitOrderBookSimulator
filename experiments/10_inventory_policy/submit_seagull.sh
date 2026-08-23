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
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
REPETITIONS=1
RANKS="${RANKS:-64}"
read -r -a OPENING_SEEDS <<< \
  "${OPENING_SEEDS_OVERRIDE:-30300130 30300131 30300132 30300133 30300134}"
read -r -a FULL_SEEDS <<< \
  "${FULL_SEEDS_OVERRIDE:-20200130 20200131 20200132 20200133 20200134}"
read -r -a ETA_VALUES <<< \
  "${ETA_VALUES_OVERRIDE:-0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.50 2.75 3.00 3.25 3.50 3.75 4.00}"
RUN_OPENING_PREFIX="${RUN_OPENING_PREFIX:-1}"
[[ "$RUN_OPENING_PREFIX" == 0 || "$RUN_OPENING_PREFIX" == 1 ]] || {
  printf 'ERROR: RUN_OPENING_PREFIX must be 0 or 1.\n' >&2
  exit 1
}
source "$PROJECT_DIR/hpc/seagull/common.sh"
base=(--partition cyclic --synchronous-observations
      --disable-persistent-risk-collective)

# The opening-prefix table uses the five pre-declared quote-engine paths.
if [[ "$RUN_OPENING_PREFIX" == 1 ]]; then
  DURATION_SECONDS=1800
  for seed in "${OPENING_SEEDS[@]}"; do
    SEED=$seed
    run_variant "opening/asset_local_eta_2p00_seed_${seed}" "$RANKS" 1 "${base[@]}" \
      --shared-inventory-policy asset_local \
      --shared-quote-multiplier 2.00
    run_variant "opening/gross_pooled_eta_2p00_seed_${seed}" "$RANKS" 1 "${base[@]}" \
      --shared-inventory-policy gross_pooled \
      --shared-quote-multiplier 2.00
  done
fi

# The main comparison uses five matched full-session paths at every value of
# the participation control. Asset-local and Gross-pooled always share a seed.
DURATION_SECONDS=23400
for eta in "${ETA_VALUES[@]}"; do
  eta_label=${eta/./p}
  for seed in "${FULL_SEEDS[@]}"; do
    SEED=$seed
    run_variant "full/asset_local_eta_${eta_label}_seed_${seed}" \
      "$RANKS" 1 "${base[@]}" \
      --shared-inventory-policy asset_local \
      --shared-quote-multiplier "$eta"
    run_variant "full/gross_pooled_eta_${eta_label}_seed_${seed}" \
      "$RANKS" 1 "${base[@]}" \
      --shared-inventory-policy gross_pooled \
      --shared-quote-multiplier "$eta"
  done
done
