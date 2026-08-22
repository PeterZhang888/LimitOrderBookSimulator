#!/usr/bin/env bash
#SBATCH --job-name=lob-fixed-matrix
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}}"
TOTAL_CORES="${TOTAL_CORES:?submit through submit_fixed_ownership_matrix.sh}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results/seagull/${JOB_TAG}_fixed_ownership_${TOTAL_CORES}core}"
REPETITIONS=1
DURATION_SECONDS=23400
SEED=20200130
CORES_PER_NODE=16
BACKGROUND_MODEL=queue-reactive-v1
BUILD_DIR="$PROJECT_DIR/build-mpi"

case "$TOTAL_CORES" in
  16|32|64) ;;
  *)
    printf 'ERROR: TOTAL_CORES must be 16, 32, or 64.\n' >&2
    exit 1
    ;;
esac
EXPECTED_NODES=$((TOTAL_CORES / CORES_PER_NODE))
if (( ${SLURM_NTASKS:-0} != TOTAL_CORES
      || ${SLURM_JOB_NUM_NODES:-0} != EXPECTED_NODES )); then
  printf 'ERROR: requested allocation does not match %d total cores.\n' \
    "$TOTAL_CORES" >&2
  exit 1
fi

export OMP_WAIT_POLICY=ACTIVE
export OMP_MAX_ACTIVE_LEVELS=1
unset GOMP_SPINCOUNT BASE_CONFIG ASSET_COUNT CLUSTER_CSV || true
source "$PROJECT_DIR/hpc/seagull/common.sh"

test -x "$BUILD_DIR/lob_mpi" || {
  printf 'ERROR: missing executable: %s\nRun scripts/build_seagull.sh first.\n' \
    "$BUILD_DIR/lob_mpi" >&2
  exit 1
}
if ! help_output=$("$BUILD_DIR/lob_mpi" --help 2>&1); then
  printf 'ERROR: cannot execute %s:\n%s\n' \
    "$BUILD_DIR/lob_mpi" "$help_output" >&2
  exit 1
fi
grep -F -- '--persistent-fixed-book-ownership' \
  <<< "$help_output" >/dev/null || {
  printf 'ERROR: executable predates permanent book ownership; rebuild it.\n' >&2
  exit 1
}
if [[ -n "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=no)" ]]; then
  printf 'ERROR: tracked source files have local modifications.\n' >&2
  exit 1
fi

common_arguments=(
  --partition cyclic
  --synchronous-observations
  --disable-persistent-risk-collective
  --risk-lookahead-max-windows 0
  --shared-inventory-policy gross_pooled
  --shared-quote-multiplier 2.00
)

mkdir -p "$RESULT_ROOT"
PARTITION_COST_CSV="$RESULT_ROOT/partition_costs.csv"
if [[ ! -s "$PARTITION_COST_CSV" ]]; then
  work_profile="$RESULT_ROOT/cost_preparation/asset_work.csv"
  run_variant cost_preparation "$TOTAL_CORES" 1 \
    "${common_arguments[@]}" \
    --openmp-schedule dynamic1 \
    --asset-work-csv "$work_profile"
  bash "$PROJECT_DIR/scripts/create_partition_costs.sh" \
    "$work_profile" "$PARTITION_COST_CSV"
fi

thread_values=(1 2 4 8 16)
rank_values=(
  "$TOTAL_CORES"
  "$((TOTAL_CORES / 2))"
  "$((TOTAL_CORES / 4))"
  "$((TOTAL_CORES / 8))"
  "$((TOTAL_CORES / 16))"
)
labels=()
for index in 0 1 2 3 4; do
  labels+=("mpi_${rank_values[$index]}x${thread_values[$index]}")
done

printf 'block,position,variant,ranks,threads\n' > "$RESULT_ROOT/run_order.csv"
for block in 1 2 3 4 5 6 7; do
  rotation=$(((block - 1) % 5))
  for position_index in 0 1 2 3 4; do
    index=$(((rotation + position_index) % 5))
    ranks="${rank_values[$index]}"
    threads="${thread_values[$index]}"
    variant="${labels[$index]}"
    position=$((position_index + 1))
    printf '%d,%d,%s,%d,%d\n' \
      "$block" "$position" "$variant" "$ranks" "$threads" \
      >> "$RESULT_ROOT/run_order.csv"
    if (( threads == 1 )); then
      run_variant "$variant/block_$block" "$ranks" "$threads" \
        "${common_arguments[@]}" \
        --openmp-schedule dynamic1
    else
      run_variant "$variant/block_$block" "$ranks" "$threads" \
        "${common_arguments[@]}" \
        --openmp-schedule weighted-static \
        --partition-cost-csv "$PARTITION_COST_CSV" \
        --persistent-fixed-book-ownership
    fi
  done
done

python3 "$PROJECT_DIR/scripts/summarize_fixed_decomposition_matrix.py" \
  "$RESULT_ROOT" "$TOTAL_CORES"

printf 'PERMANENT OWNERSHIP %s-CORE MATRIX: PASS\n' "$TOTAL_CORES"
printf 'RESULT_ROOT=%s\n' "$RESULT_ROOT"
