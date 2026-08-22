#!/usr/bin/env bash
#SBATCH --job-name=lob-16core-mpi-openmp
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results/seagull/${SLURM_JOB_ID}_16core_mpi_openmp}"
REPETITIONS=1
DURATION_SECONDS=23400
SEED=20200130
CORES_PER_NODE=16
BACKGROUND_MODEL=queue-reactive-v1
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build-mpi}"

test -x "$BUILD_DIR/lob_mpi" || {
  printf 'ERROR: missing MPI executable: %s\nRun scripts/build_seagull.sh first.\n' \
    "$BUILD_DIR/lob_mpi" >&2
  exit 1
}
test -x "$PROJECT_DIR/build-openmp/lob_openmp" || {
  printf 'ERROR: missing MPI-free OpenMP executable: %s\nRun scripts/build_seagull.sh first.\n' \
    "$PROJECT_DIR/build-openmp/lob_openmp" >&2
  exit 1
}
if [[ -n "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=no)" ]]; then
  printf 'ERROR: tracked source files have local modifications.\n' >&2
  exit 1
fi

export OMP_WAIT_POLICY=ACTIVE
export OMP_MAX_ACTIVE_LEVELS=1
unset GOMP_SPINCOUNT || true

source "$PROJECT_DIR/hpc/seagull/common.sh"
command -v taskset >/dev/null || {
  printf 'ERROR: taskset is required for the MPI-free OpenMP placement.\n' >&2
  exit 1
}

common_arguments=(
  --partition cyclic
  --synchronous-observations
  --disable-persistent-risk-collective
  --risk-lookahead-max-windows 0
  --openmp-schedule dynamic1
  --shared-inventory-policy gross_pooled
  --shared-quote-multiplier 2.00
)

run_openmp_checked() {
  local label=$1
  shift
  local variant_dir="$RESULT_ROOT/$label"
  local mpi_placement="$RESULT_ROOT/mpi_16x1/block_1/cpu_placement.txt"
  test -s "$mpi_placement" || {
    printf 'ERROR: the block-1 MPI placement must be recorded first.\n' >&2
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
    "$PROJECT_DIR/build-openmp/lob_openmp" \
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
    "$@" | tee "$variant_dir/run_1.txt"
}

control=mpi_16x1
treatment=openmp_1x16
mkdir -p "$RESULT_ROOT"
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
          "${common_arguments[@]}"
        ;;
      openmp_1x16)
        run_openmp_checked "$variant/block_$block" \
          "${common_arguments[@]}"
        ;;
      *)
        printf 'ERROR: unknown layout: %s\n' "$variant" >&2
        exit 1
        ;;
    esac
  done
done

python3 "$PROJECT_DIR/scripts/summarize_layout_pair.py" \
  "$RESULT_ROOT" \
  "$control" "$treatment" \
  16 1 1 16
