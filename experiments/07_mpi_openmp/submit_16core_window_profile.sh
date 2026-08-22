#!/usr/bin/env bash
#SBATCH --job-name=lob-16core-window-profile
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
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results/seagull/${JOB_TAG}_16core_window_profile}"
REPETITIONS=1
DURATION_SECONDS=23400
SEED=20200130
CORES_PER_NODE=16
BACKGROUND_MODEL=queue-reactive-v1
BUILD_DIR="$PROJECT_DIR/build-mpi"
OPENMP_BUILD_DIR="$PROJECT_DIR/build-openmp"
BLOCK_COUNT=3

# Load the compiler/MPI runtime before executing either binary during the
# preflight. A batch job does not inherit the environment loaded inside the
# separate build script.
export OMP_WAIT_POLICY=ACTIVE
export OMP_MAX_ACTIVE_LEVELS=1
unset GOMP_SPINCOUNT || true
# This diagnostic is specifically the full 1,480-book empirical case. Do not
# inherit optional synthetic-template or cluster-output overrides.
unset BASE_CONFIG ASSET_COUNT CLUSTER_CSV || true
source "$PROJECT_DIR/hpc/seagull/common.sh"

test -x "$BUILD_DIR/lob_mpi" || {
  printf 'ERROR: missing MPI executable: %s\nRun scripts/build_seagull.sh first.\n' \
    "$BUILD_DIR/lob_mpi" >&2
  exit 1
}
test -x "$OPENMP_BUILD_DIR/lob_openmp" || {
  printf 'ERROR: missing MPI-free OpenMP executable: %s\nRun scripts/build_seagull.sh first.\n' \
    "$OPENMP_BUILD_DIR/lob_openmp" >&2
  exit 1
}
if [[ -n "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=no)" ]]; then
  printf 'ERROR: tracked source files have local modifications.\n' >&2
  exit 1
fi
for executable in "$BUILD_DIR/lob_mpi" "$OPENMP_BUILD_DIR/lob_openmp"; do
  if ! help_output=$("$executable" --help 2>&1); then
    printf 'ERROR: cannot execute %s during preflight:\n%s\n' \
      "$executable" "$help_output" >&2
    exit 1
  fi
  grep -F -- '--window-phase-profile-csv' <<< "$help_output" >/dev/null || {
    printf 'ERROR: %s is stale; rebuild with scripts/build_seagull.sh.\n' \
      "$executable" >&2
    exit 1
  }
done

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

run_profiled_openmp() {
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
    --window-phase-profile-csv "$variant_dir/window_phase_profile.csv" \
    --threads 16 \
    "$@" | tee "$variant_dir/run_1.txt"
}

control=mpi_16x1
treatment=openmp_1x16
mkdir -p "$RESULT_ROOT"
printf 'block,position,variant\n' > "$RESULT_ROOT/run_order.csv"

for block in 1 2 3; do
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
        profile="$RESULT_ROOT/$variant/block_$block/window_phase_profile.csv"
        run_variant "$variant/block_$block" 16 1 \
          "${common_arguments[@]}" \
          --window-phase-profile-csv "$profile"
        ;;
      openmp_1x16)
        run_profiled_openmp "$variant/block_$block" \
          "${common_arguments[@]}"
        ;;
      *)
        printf 'ERROR: unknown layout: %s\n' "$variant" >&2
        exit 1
        ;;
    esac
  done
done

python3 "$PROJECT_DIR/scripts/verify_window_profile_campaign.py" \
  "$RESULT_ROOT" "$control" "$treatment" 16 1 1 16 \
  --blocks "$BLOCK_COUNT" --expected-assets 1480 \
  --expected-duration-seconds "$DURATION_SECONDS"

profile_arguments=()
for block in 1 2 3; do
  profile_arguments+=(
    --control-csv "$RESULT_ROOT/$control/block_$block/window_phase_profile.csv"
    --treatment-csv "$RESULT_ROOT/$treatment/block_$block/window_phase_profile.csv"
  )
done
python3 "$PROJECT_DIR/scripts/summarize_window_profile.py" \
  "$RESULT_ROOT/profile_analysis" \
  --control-label "$control" \
  --control-ranks 16 \
  --treatment-label "$treatment" \
  --treatment-ranks 1 \
  --expected-windows 23400 \
  "${profile_arguments[@]}"

printf 'WINDOW PHASE PROFILE: PASS\n'
printf 'RESULT_ROOT=%s\n' "$RESULT_ROOT"
