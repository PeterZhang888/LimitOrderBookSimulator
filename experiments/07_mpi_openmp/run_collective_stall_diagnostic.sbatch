#!/usr/bin/env bash
#SBATCH --job-name=lob-collective-diagnostic
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results/seagull/${JOB_TAG}_collective_stall_diagnostic}"
BUILD_DIR="$PROJECT_DIR/build-mpi"
BACKGROUND_MODEL=queue-reactive-v1
DURATION_SECONDS=23400
SEED=20200130
CORES_PER_NODE=16
REPETITIONS=12
STALL_THRESHOLD_SECONDS=120
RANKS=32
THREADS=1
TASKS_PER_NODE=16

export OMP_WAIT_POLICY=ACTIVE
export OMP_MAX_ACTIVE_LEVELS=1
unset GOMP_SPINCOUNT BASE_CONFIG ASSET_COUNT CLUSTER_CSV || true
source "$PROJECT_DIR/hpc/seagull/common.sh"

[[ ${SLURM_NTASKS:-0} -eq $RANKS \
   && ${SLURM_JOB_NUM_NODES:-0} -eq 2 ]] || {
  printf 'ERROR: this diagnostic requires 32 ranks on two nodes.\n' >&2
  exit 1
}
test -x "$BUILD_DIR/lob_mpi" || {
  printf 'ERROR: missing executable: %s\nRun scripts/build_seagull.sh first.\n' \
    "$BUILD_DIR/lob_mpi" >&2
  exit 1
}
if [[ -n "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=no)" ]]; then
  printf 'ERROR: tracked source files have local modifications.\n' >&2
  exit 1
fi
if ! help_output=$("$BUILD_DIR/lob_mpi" --help 2>&1); then
  printf 'ERROR: cannot execute %s:\n%s\n' \
    "$BUILD_DIR/lob_mpi" "$help_output" >&2
  exit 1
fi
for option in --boundary-arrival-csv --window-phase-profile-csv; do
  grep -F -- "$option" <<< "$help_output" >/dev/null || {
    printf 'ERROR: executable lacks %s; rebuild it.\n' "$option" >&2
    exit 1
  }
done

common_arguments=(
  --partition cyclic
  --synchronous-observations
  --disable-persistent-risk-collective
  --risk-lookahead-max-windows 0
  --openmp-schedule dynamic1
  --shared-inventory-policy gross_pooled
  --shared-quote-multiplier 2.00
)
mpi_mapping="ppr:${TASKS_PER_NODE}:node:PE=${THREADS}"
mkdir -p "$RESULT_ROOT"
printf 'repetition,execution_seconds,classification\n' \
  > "$RESULT_ROOT/repetitions.csv"

reference_metrics=
reference_assets=
normal_run=
stall_run=

for ((rep=1; rep<=REPETITIONS; rep++)); do
  run_dir="$RESULT_ROOT/repetition_$rep"
  placement_dir="$run_dir/rank_placement"
  metrics="$run_dir/metrics.csv"
  assets="$run_dir/assets.csv"
  run_log="$run_dir/run.txt"
  mpi_log="$run_dir/mpi_stderr.txt"
  boundary_profile="$run_dir/boundary_arrivals.csv"
  window_profile="$run_dir/window_phase_profile.csv"
  mkdir -p "$placement_dir"

  env -u OMP_PLACES -u GOMP_CPU_AFFINITY \
    OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE \
    LOB_PLACEMENT_DIR="$placement_dir" \
    mpirun --np "$RANKS" \
      --map-by "$mpi_mapping" \
      --bind-to core \
      --report-bindings \
      "$PROJECT_DIR/hpc/seagull/record_rank_placement.sh" \
      "$BUILD_DIR/lob_mpi" \
      --duration-seconds "$DURATION_SECONDS" \
      --window-ms 1000 \
      "${INPUT_ARGS[@]}" \
      "${MODEL_ARGS[@]}" \
      "${SCIENTIFIC_ARGS[@]}" \
      --seed "$SEED" \
      --metrics-csv "$metrics" \
      --asset-summary-csv "$assets" \
      --asset-summary-interval-ms 1000 \
      --threads "$THREADS" \
      --boundary-arrival-csv "$boundary_profile" \
      --window-phase-profile-csv "$window_profile" \
      "${common_arguments[@]}" \
      2> >(tee "$mpi_log" >&2) | tee "$run_log"

  LC_ALL=C sort -t'|' -k1,1 -k2,2n \
    "$placement_dir"/rank_*.txt > "$run_dir/cpu_placement.txt"
  python3 "$PROJECT_DIR/scripts/validate_cpu_placement.py" \
    "$run_dir/cpu_placement.txt" "$RANKS" "$THREADS" "$TASKS_PER_NODE"

  execution_seconds=$(
    sed -n 's/.* execution_seconds=\([0-9.]*\).*/\1/p' "$run_log"
  )
  [[ "$execution_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    printf 'ERROR: repetition %d has no valid execution_seconds.\n' "$rep" >&2
    exit 1
  }

  if [[ -z "$reference_metrics" ]]; then
    reference_metrics="$metrics"
    reference_assets="$assets"
  else
    cmp -s "$reference_metrics" "$metrics" || {
      printf 'ERROR: metrics changed in repetition %d.\n' "$rep" >&2
      exit 1
    }
    cmp -s "$reference_assets" "$assets" || {
      printf 'ERROR: asset results changed in repetition %d.\n' "$rep" >&2
      exit 1
    }
  fi

  classification=$(python3 - "$execution_seconds" "$STALL_THRESHOLD_SECONDS" <<'PY'
import sys
print("stall" if float(sys.argv[1]) >= float(sys.argv[2]) else "normal")
PY
  )
  printf '%d,%s,%s\n' "$rep" "$execution_seconds" "$classification" \
    >> "$RESULT_ROOT/repetitions.csv"

  if [[ "$classification" == normal && -z "$normal_run" ]]; then
    normal_run="$run_dir"
  fi
  if [[ "$classification" == stall && -z "$stall_run" ]]; then
    stall_run="$run_dir"
    printf 'STALL_REPETITION=%d\nSTALL_EXECUTION_SECONDS=%s\n' \
      "$rep" "$execution_seconds" > "$RESULT_ROOT/CAPTURED_STALL.env"
  fi
  if [[ -n "$normal_run" && -n "$stall_run" ]]; then
    break
  fi
done

python3 "$PROJECT_DIR/scripts/summarize_boundary_arrivals.py" \
  "$RESULT_ROOT/boundary_summary.csv" \
  --expected-ranks "$RANKS" \
  --expected-boundaries "$((DURATION_SECONDS + 1))" \
  "$RESULT_ROOT"/repetition_*/boundary_arrivals.csv

if [[ -n "$normal_run" && -n "$stall_run" ]]; then
  python3 "$PROJECT_DIR/scripts/summarize_window_profile.py" \
    "$RESULT_ROOT/profile_analysis" \
    --control-label normal \
    --control-csv "$normal_run/window_phase_profile.csv" \
    --control-run "$normal_run/run.txt" \
    --control-ranks "$RANKS" \
    --treatment-label stall \
    --treatment-csv "$stall_run/window_phase_profile.csv" \
    --treatment-run "$stall_run/run.txt" \
    --treatment-ranks "$RANKS" \
    --expected-windows "$DURATION_SECONDS"
else
  printf 'No repetition crossed the %s-second stall threshold.\n' \
    "$STALL_THRESHOLD_SECONDS" > "$RESULT_ROOT/NO_STALL_CAPTURED.txt"
fi

printf 'COLLECTIVE STALL DIAGNOSTIC: COMPLETE\n'
printf 'RESULT_ROOT=%s\n' "$RESULT_ROOT"
