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
STALL_THRESHOLD_SECONDS=120
MPI_HEALTH_CHECK_ITERATIONS=100
MPI_HEALTH_CHECK_MAX_MEAN_MS=2.0
MAX_ATTEMPTS_PER_ACCEPTED_RUN=10

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
grep -F -- '--mpi-health-check-iterations' \
  <<< "$help_output" >/dev/null || {
  printf 'ERROR: executable predates the MPI health check; rebuild it.\n' >&2
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
  --mpi-health-check-iterations "$MPI_HEALTH_CHECK_ITERATIONS"
  --mpi-health-check-max-mean-ms "$MPI_HEALTH_CHECK_MAX_MEAN_MS"
)

mkdir -p "$RESULT_ROOT"
ATTEMPT_LOG="$RESULT_ROOT/attempts.csv"
printf '%s\n' \
  'slot,ranks,threads,attempt,status,health_mean_ms,execution_seconds,result_path' \
  > "$ATTEMPT_LOG"

run_accepted_variant() {
  local slot=$1
  local ranks=$2
  local threads=$3
  shift 3
  local canonical_dir="$RESULT_ROOT/$slot"
  local attempt

  [[ ! -e "$canonical_dir" ]] || {
    printf 'ERROR: accepted-result directory already exists: %s\n' \
      "$canonical_dir" >&2
    return 1
  }

  for ((attempt=1; attempt<=MAX_ATTEMPTS_PER_ACCEPTED_RUN; attempt++)); do
    local attempt_label="_attempts/$slot/attempt_$attempt"
    local attempt_dir="$RESULT_ROOT/$attempt_label"
    local attempt_arguments=()
    local argument
    for argument in "$@"; do
      attempt_arguments+=("${argument//@RUN_DIR@/$attempt_dir}")
    done

    local run_status=0
    if run_variant "$attempt_label" "$ranks" "$threads" \
      "${attempt_arguments[@]}"
    then
      run_status=0
    else
      run_status=$?
    fi

    local log="$attempt_dir/run_1.txt"
    local health_mean_ms=""
    local health_status=""
    if [[ -f "$log" ]]; then
      health_mean_ms=$(
        sed -n \
          's/^mpi_health_check .*max_mean_collective_ms=\([^ ]*\).*$/\1/p' \
          "$log" | tail -n 1
      )
      health_status=$(
        sed -n 's/^mpi_health_check .*status=\([^ ]*\).*$/\1/p' \
          "$log" | tail -n 1
      )
    fi

    if (( run_status != 0 )); then
      if [[ "$health_status" == REJECT ]]; then
        printf '%s,%d,%d,%d,%s,%s,,%s\n' \
          "$slot" "$ranks" "$threads" "$attempt" \
          preflight_rejected "$health_mean_ms" "$attempt_dir" \
          >> "$ATTEMPT_LOG"
        printf '%s\n' \
          "RETRY: $slot attempt $attempt failed the MPI health check" \
          "(mean $health_mean_ms ms; limit $MPI_HEALTH_CHECK_MAX_MEAN_MS ms)."
        continue
      fi
      printf 'ERROR: %s attempt %d failed for a reason other than the declared MPI health gate.\n' \
        "$slot" "$attempt" >&2
      return "$run_status"
    fi

    [[ "$health_status" == PASS ]] || {
      printf 'ERROR: %s attempt %d did not record a passing MPI health check.\n' \
        "$slot" "$attempt" >&2
      return 1
    }

    local execution_seconds
    execution_seconds=$(
      sed -n 's/.* execution_seconds=\([0-9.]*\).*/\1/p' "$log"
    )
    [[ "$execution_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
      printf 'ERROR: %s attempt %d has no valid execution_seconds.\n' \
        "$slot" "$attempt" >&2
      return 1
    }

    if python3 - "$execution_seconds" "$STALL_THRESHOLD_SECONDS" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
    then
      printf '%s,%d,%d,%d,%s,%s,%s,%s\n' \
        "$slot" "$ranks" "$threads" "$attempt" \
        postrun_rejected "$health_mean_ms" "$execution_seconds" \
        "$attempt_dir" >> "$ATTEMPT_LOG"
      printf '%s\n' \
        "RETRY: $slot attempt $attempt took $execution_seconds seconds" \
        "(post-run limit $STALL_THRESHOLD_SECONDS seconds)."
      continue
    fi

    mkdir -p "$(dirname "$canonical_dir")"
    mv "$attempt_dir" "$canonical_dir"
    printf '%s,%d,%d,%d,%s,%s,%s,%s\n' \
      "$slot" "$ranks" "$threads" "$attempt" accepted \
      "$health_mean_ms" "$execution_seconds" "$canonical_dir" \
      >> "$ATTEMPT_LOG"
    return 0
  done

  printf 'ERROR: %s did not obtain an accepted run in %d attempts.\n' \
    "$slot" "$MAX_ATTEMPTS_PER_ACCEPTED_RUN" >&2
  return 1
}

PARTITION_COST_CSV="$RESULT_ROOT/partition_costs.csv"
if [[ ! -s "$PARTITION_COST_CSV" ]]; then
  work_profile="$RESULT_ROOT/cost_preparation/asset_work.csv"
  run_accepted_variant cost_preparation "$TOTAL_CORES" 1 \
    "${common_arguments[@]}" \
    --openmp-schedule dynamic1 \
    --asset-work-csv '@RUN_DIR@/asset_work.csv'
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
      run_accepted_variant "$variant/block_$block" "$ranks" "$threads" \
        "${common_arguments[@]}" \
        --openmp-schedule dynamic1
    else
      run_accepted_variant "$variant/block_$block" "$ranks" "$threads" \
        "${common_arguments[@]}" \
        --openmp-schedule weighted-static \
        --partition-cost-csv "$PARTITION_COST_CSV" \
        --persistent-fixed-book-ownership
    fi
  done
done

python3 "$PROJECT_DIR/scripts/summarize_fixed_decomposition_matrix.py" \
  "$RESULT_ROOT" "$TOTAL_CORES"

printf 'PERMANENT OWNERSHIP %s-CORE MATRIX: COMPLETE\n' "$TOTAL_CORES"
printf 'RESULT_ROOT=%s\n' "$RESULT_ROOT"
