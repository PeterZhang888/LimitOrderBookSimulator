#!/usr/bin/env bash
#SBATCH --job-name=lob-16core-fixed-owner
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results/runs/${JOB_TAG}_16core_fixed_ownership}"
REPETITIONS=1
DURATION_SECONDS=23400
SEED=20200130
CORES_PER_NODE=16
BACKGROUND_MODEL=queue-reactive-v1
BUILD_DIR="$PROJECT_DIR/build-mpi"
OPENMP_BUILD_DIR="$PROJECT_DIR/build-openmp"

export OMP_WAIT_POLICY=ACTIVE
export OMP_MAX_ACTIVE_LEVELS=1
unset GOMP_SPINCOUNT BASE_CONFIG ASSET_COUNT CLUSTER_CSV || true
source "$PROJECT_DIR/hpc/seagull/common.sh"

for executable in "$BUILD_DIR/lob_mpi" "$OPENMP_BUILD_DIR/lob_openmp"; do
  test -x "$executable" || {
    printf 'ERROR: missing executable: %s\nRun scripts/build_seagull.sh first.\n' \
      "$executable" >&2
    exit 1
  }
  if ! help_output=$("$executable" --help 2>&1); then
    printf 'ERROR: cannot execute %s during preflight:\n%s\n' \
      "$executable" "$help_output" >&2
    exit 1
  fi
  grep -F -- '--persistent-fixed-book-ownership' \
    <<< "$help_output" >/dev/null || {
    printf 'ERROR: %s predates fixed book ownership; rebuild it.\n' \
      "$executable" >&2
    exit 1
  }
done
if [[ -n "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=no)" ]]; then
  printf 'ERROR: tracked source files have local modifications.\n' >&2
  exit 1
fi
command -v taskset >/dev/null || {
  printf 'ERROR: taskset is required for MPI-free OpenMP placement.\n' >&2
  exit 1
}

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
  run_variant cost_preparation 16 1 \
    "${common_arguments[@]}" \
    --openmp-schedule dynamic1 \
    --asset-work-csv "$work_profile"
  bash "$PROJECT_DIR/scripts/create_partition_costs.sh" \
    "$work_profile" "$PARTITION_COST_CSV"
fi

run_fixed_openmp() {
  local label=$1
  local variant_dir="$RESULT_ROOT/$label"
  local mpi_placement="$RESULT_ROOT/mpi_16x1/block_1/cpu_placement.txt"
  test -s "$mpi_placement" || {
    printf 'ERROR: block-1 MPI placement must be recorded first.\n' >&2
    return 1
  }
  local cpu_list
  cpu_list=$(awk -F'|' '{print $3}' "$mpi_placement" | paste -sd, -)
  mkdir -p "$variant_dir"
  OMP_NUM_THREADS=16 \
  OMP_DYNAMIC=FALSE \
  OMP_PLACES=cores \
  OMP_PROC_BIND=spread \
  taskset -c "$cpu_list" \
  bash -c '
    placement=$1
    validator=$2
    shift 2
    allowed=$(awk '\''/^Cpus_allowed_list:/ {print $2}'\'' /proc/self/status)
    printf "%s|0|%s\n" "$(hostname -s)" "$allowed" > "$placement"
    python3 "$validator" "$placement" 1 16 1
    exec "$@"
  ' bash \
    "$variant_dir/cpu_placement.txt" \
    "$PROJECT_DIR/scripts/validate_cpu_placement.py" \
    "$OPENMP_BUILD_DIR/lob_openmp" \
    --duration-seconds "$DURATION_SECONDS" \
    --window-ms 1000 \
    "${INPUT_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${SCIENTIFIC_ARGS[@]}" \
    --seed "$SEED" \
    --metrics-csv "$variant_dir/metrics_1.csv" \
    --asset-summary-csv "$variant_dir/assets_1.csv" \
    --asset-summary-interval-ms 1000 \
    --threads 16 \
    "${common_arguments[@]}" \
    --openmp-schedule weighted-static \
    --partition-cost-csv "$PARTITION_COST_CSV" \
    --persistent-fixed-book-ownership \
    --thread-ownership-csv "$variant_dir/thread_ownership.csv" \
    | tee "$variant_dir/run_1.txt"
}

control=mpi_16x1
treatment=openmp_1x16_fixed
printf 'block,position,variant\n' > "$RESULT_ROOT/run_order.csv"

for block in 1 2 3 4 5 6 7; do
  if (( block % 2 == 1 )); then
    order=("$control" "$treatment")
  else
    order=("$treatment" "$control")
  fi
  for position_index in 0 1; do
    variant="${order[$position_index]}"
    position=$((position_index + 1))
    printf '%d,%d,%s\n' "$block" "$position" "$variant" \
      >> "$RESULT_ROOT/run_order.csv"
    case "$variant" in
      mpi_16x1)
        run_variant "$variant/block_$block" 16 1 \
          "${common_arguments[@]}" \
          --openmp-schedule dynamic1
        ;;
      openmp_1x16_fixed)
        run_fixed_openmp "$variant/block_$block"
        ;;
      *)
        printf 'ERROR: unknown layout: %s\n' "$variant" >&2
        exit 1
        ;;
    esac
  done
done

python3 "$PROJECT_DIR/scripts/summarize_fixed_ownership_pair.py" \
  "$RESULT_ROOT" "$control" "$treatment"

printf 'PERSISTENT FIXED BOOK OWNERSHIP PAIR: PASS\n'
printf 'RESULT_ROOT=%s\n' "$RESULT_ROOT"
