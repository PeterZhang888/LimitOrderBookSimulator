#!/usr/bin/env bash
set -Eeuo pipefail

: "${PROJECT_DIR:?set PROJECT_DIR to the cloned repository}"
source "$PROJECT_DIR/hpc/seagull/load_environment.sh"
cd "$PROJECT_DIR"

EMPIRICAL_DATA_DIR="${EMPIRICAL_DATA_DIR:-$PROJECT_DIR/data/empirical}"
UNIVERSE_CONFIG="${UNIVERSE_CONFIG:-$EMPIRICAL_DATA_DIR/universe.csv}"
BACKGROUND_POLICY_CSV="${BACKGROUND_POLICY_CSV:-$EMPIRICAL_DATA_DIR/background_policy.csv}"
VALUE_POLICY_CSV="${VALUE_POLICY_CSV:-$EMPIRICAL_DATA_DIR/value_policy.csv}"

BACKGROUND_MODEL="${BACKGROUND_MODEL:-queue-reactive-v1}"
MODEL_ARGS=(--background-model "$BACKGROUND_MODEL")
if [[ "$BACKGROUND_MODEL" == queue-reactive-v1 ]]; then
  [[ -s "$BACKGROUND_POLICY_CSV" ]] || {
    printf 'ERROR: background policy is missing: %s\n' \
      "$BACKGROUND_POLICY_CSV" >&2
    exit 1
  }
  MODEL_ARGS+=(--background-policy-csv "$BACKGROUND_POLICY_CSV")
fi
if [[ -s "$VALUE_POLICY_CSV" ]]; then
  MODEL_ARGS+=(--value-agent-policy-csv "$VALUE_POLICY_CSV")
fi

if [[ -n "${BASE_CONFIG:-}" ]]; then
  : "${ASSET_COUNT:?set ASSET_COUNT when BASE_CONFIG is used}"
  INPUT_ARGS=(--base-config "$BASE_CONFIG" --assets "$ASSET_COUNT")
  SCIENTIFIC_ARGS=()
else
  [[ -s "$UNIVERSE_CONFIG" ]] || {
    printf 'ERROR: empirical universe is missing: %s\n' \
      "$UNIVERSE_CONFIG" >&2
    exit 1
  }
  INPUT_ARGS=(--universe-config "$UNIVERSE_CONFIG")
  # Frozen empirical-market controls used by the thesis. Experiment files
  # change only their declared treatment; they do not silently inherit the
  # executable's generic demonstration defaults.
  SCIENTIFIC_ARGS=(
    --stochastic-baseline-normalization-seconds 23400
    --hawkes-activity-scale 0.3
    --local-mm-interval-ms 1000
    --local-mm-quantity-multiplier 1.0
    --local-mm-improvement-probability 0.25
    --local-mm-spread-elasticity 0.0
    --local-mm-max-improvement-probability 1.0
    --shared-quote-relative
    --shared-capacity-relative
    --shared-quote-levels 3
    --local-inventory-limit 800
    --global-risk-limit-per-asset 50
    --capacity-threshold 0.5
    --minimum-shared-quote-scale 0.05
    --shared-price-unit-usd 0.0001
    --shared-terminal-fallback-distance-ticks 100
  )
fi

BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build-mpi}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results/seagull/$SLURM_JOB_ID}"
REPETITIONS="${REPETITIONS:-7}"
DURATION_SECONDS="${DURATION_SECONDS:-23400}"
SEED="${SEED:-20200130}"
CORES_PER_NODE="${CORES_PER_NODE:-16}"

mkdir -p "$RESULT_ROOT"

ENVIRONMENT_FILE="$RESULT_ROOT/environment.txt"
if [[ ! -e "$ENVIRONMENT_FILE" ]]; then
  {
    printf 'recorded_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'project_dir=%s\n' "$PROJECT_DIR"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-not-in-slurm}"
    printf 'slurm_node_list=%s\n' "${SLURM_JOB_NODELIST:-not-in-slurm}"
    printf 'openmpi_module=%s\n' "$OPENMPI_MODULE"
    printf 'mpi_cxx=%s\n' "$LOB_MPI_CXX_COMPILER"
    printf 'openmp_cxx=%s\n' "$LOB_OPENMP_CXX_COMPILER"
    printf '\ncompiler\n'
    "$LOB_MPI_CXX_COMPILER" --version
    printf '\nmpi\n'
    mpirun --version
    printf '\ncmake\n'
    cmake --version
    printf '\ncpu\n'
    lscpu
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
      printf '\nslurm_job\n'
      scontrol show job -o "$SLURM_JOB_ID"
    fi
    printf '\nmodules\n'
    module list
  } > "$ENVIRONMENT_FILE" 2>&1
fi

run_variant() {
  local label=$1
  local ranks=$2
  local threads=$3
  shift 3
  local variant_dir="$RESULT_ROOT/$label"

  if (( CORES_PER_NODE < 1 )); then
    printf 'ERROR: CORES_PER_NODE must be positive.\n' >&2
    return 1
  fi
  if (( ranks < 1 || threads < 1 || threads > CORES_PER_NODE
        || CORES_PER_NODE % threads != 0 )); then
    printf 'ERROR: threads per rank must divide %d cores per node.\n' \
      "$CORES_PER_NODE" >&2
    return 1
  fi

  local total_cores=$((ranks * threads))
  local nodes=$(((total_cores + CORES_PER_NODE - 1) / CORES_PER_NODE))
  local tasks_per_node=$((CORES_PER_NODE / threads))

  mkdir -p "$variant_dir"

  for ((rep=1; rep<=REPETITIONS; rep++)); do
    local metrics="$variant_dir/metrics_${rep}.csv"
    local assets="$variant_dir/assets_${rep}.csv"
    local cluster="$variant_dir/clusters_${rep}.csv"
    local log="$variant_dir/run_${rep}.txt"
    local cluster_args=()
    if [[ -n "${CLUSTER_CSV:-}" ]]; then
      cluster_args=(
        --shock-cluster-csv "$CLUSTER_CSV"
        --cluster-metrics-csv "$cluster"
      )
    fi
    OMP_NUM_THREADS="$threads" \
    OMP_DYNAMIC=FALSE \
    OMP_PLACES=cores \
    OMP_PROC_BIND=close \
    srun --nodes="$nodes" --ntasks="$ranks" \
      --ntasks-per-node="$tasks_per_node" \
      --cpus-per-task="$threads" \
      --cpu-bind=cores \
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
      --threads "$threads" \
      "${cluster_args[@]}" \
      "$@" | tee "$log"
  done
}

run_openmp_variant() {
  local label=$1
  local threads=$2
  shift 2
  local variant_dir="$RESULT_ROOT/$label"
  mkdir -p "$variant_dir"

  for ((rep=1; rep<=REPETITIONS; rep++)); do
    OMP_NUM_THREADS="$threads" \
    OMP_DYNAMIC=FALSE \
    OMP_PLACES=cores \
    OMP_PROC_BIND=spread \
    srun --nodes=1 --ntasks=1 --cpus-per-task="$threads" \
      --cpu-bind=none \
      "$PROJECT_DIR/build-openmp/lob_openmp" \
      --duration-seconds "$DURATION_SECONDS" \
      --window-ms 1000 \
      "${INPUT_ARGS[@]}" \
      "${MODEL_ARGS[@]}" \
      "${SCIENTIFIC_ARGS[@]}" \
      --seed "$SEED" \
      --metrics-csv "$variant_dir/metrics_${rep}.csv" \
      --asset-summary-csv "$variant_dir/assets_${rep}.csv" \
      --asset-summary-interval-ms 1000 \
      --threads "$threads" \
      "$@" | tee "$variant_dir/run_${rep}.txt"
  done
}
