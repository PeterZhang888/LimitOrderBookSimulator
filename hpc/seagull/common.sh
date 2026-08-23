#!/usr/bin/env bash
set -Eeuo pipefail

: "${PROJECT_DIR:?set PROJECT_DIR to the cloned repository}"

# Formal runs use the Open MPI defaults supplied by the loaded Seagull module.
# Do not silently inherit transport, collective, progress, or UCX experiments
# from the shell that called sbatch.  A specialised diagnostic can set an
# intentional override after sourcing this file.
while IFS='=' read -r variable _; do
  case "$variable" in
    OMPI_MCA_*|PRTE_MCA_*|PMIX_MCA_*|UCX_*) unset "$variable" ;;
  esac
done < <(env)

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
  # The empirical Value Agent policy contains one row for each asset in the
  # empirical universe.  Never attach it to a synthetic template expansion,
  # whose asset count can differ (for example, the 10,000-book benchmark).
  if [[ -s "$VALUE_POLICY_CSV" ]]; then
    MODEL_ARGS+=(--value-agent-policy-csv "$VALUE_POLICY_CSV")
  fi
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
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results/runs/$SLURM_JOB_ID}"
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
    printf '\nopenmp_environment\n'
    printf 'OMP_WAIT_POLICY=%s\n' "${OMP_WAIT_POLICY:-unset}"
    printf 'OMP_MAX_ACTIVE_LEVELS=%s\n' \
      "${OMP_MAX_ACTIVE_LEVELS:-unset}"
    printf 'GOMP_SPINCOUNT=%s\n' "${GOMP_SPINCOUNT:-unset}"
    printf '\nmpi_runtime_overrides\n'
    env | LC_ALL=C sort |
      awk '/^(OMPI_MCA_|PRTE_MCA_|PMIX_MCA_|UCX_)/' || true
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

  local tasks_per_node=$((CORES_PER_NODE / threads))
  local mpi_mapping="ppr:${tasks_per_node}:node:PE=${threads}"
  local mpi_binding=core
  if (( threads == CORES_PER_NODE )); then
    # Seagull's 16 physical cores are split across two CPU packages. Open MPI
    # refuses a CORE binding that spans both packages. With exactly one rank
    # per node, retain Slurm's full-node CPU set and let OMP_PLACES=cores bind
    # the 16 OpenMP workers within that validated allocation.
    mpi_mapping="ppr:1:node"
    mpi_binding=none
  fi

  mkdir -p "$variant_dir"

  # Slurm owns process placement.  Do not apply OpenMP affinity to a
  # one-thread MPI rank: on Seagull, doing so can move every rank on a node
  # onto the first OpenMP place.  Threaded ranks retain explicit OpenMP
  # placement inside the multi-core mask assigned by Slurm.
  local omp_environment=(
    env -u OMP_PLACES -u GOMP_CPU_AFFINITY
    "OMP_NUM_THREADS=$threads"
    OMP_DYNAMIC=FALSE
  )
  if (( threads == 1 )); then
    omp_environment+=(OMP_PROC_BIND=FALSE)
  else
    omp_environment+=(OMP_PLACES=cores OMP_PROC_BIND=close)
  fi

  # Fail before starting a long simulation if two ranks on the same node
  # have been assigned the same CPU set.  This guard catches placement
  # failures such as all ranks being restricted to CPU 0.
  local placement_file="$variant_dir/cpu_placement.txt"
  "${omp_environment[@]}" \
  mpirun --np "$ranks" \
    --map-by "$mpi_mapping" \
    --bind-to "$mpi_binding" \
    bash -c '
      cpu_list=$(awk '\''/^Cpus_allowed_list:/ {print $2}'\'' /proc/self/status)
      printf "%s|%s|%s\n" "$(hostname -s)" "$OMPI_COMM_WORLD_RANK" "$cpu_list"
    ' | LC_ALL=C sort -t'|' -k1,1 -k2,2n > "$placement_file"

  python3 "$PROJECT_DIR/scripts/validate_cpu_placement.py" \
    "$placement_file" "$ranks" "$threads" "$tasks_per_node" || {
      printf 'ERROR: invalid CPU placement for %s.\n' "$label" >&2
      cat "$placement_file" >&2
      return 1
    }

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
    if "${omp_environment[@]}" \
      mpirun --np "$ranks" \
        --map-by "$mpi_mapping" \
        --bind-to "$mpi_binding" \
        --report-bindings \
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
    then
      :
    else
      local run_status=$?
      return "$run_status"
    fi
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
