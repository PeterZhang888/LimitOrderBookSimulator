#!/usr/bin/env bash
set -Eeuo pipefail

: "${PROJECT_DIR:?set PROJECT_DIR to the cloned repository}"
BACKGROUND_MODEL="${BACKGROUND_MODEL:-queue-reactive-v1}"
MODEL_ARGS=(--background-model "$BACKGROUND_MODEL")
if [[ "$BACKGROUND_MODEL" == queue-reactive-v1 ]]; then
  : "${BACKGROUND_POLICY_CSV:?set BACKGROUND_POLICY_CSV for the queue-reactive model}"
  MODEL_ARGS+=(--background-policy-csv "$BACKGROUND_POLICY_CSV")
fi
if [[ -n "${VALUE_POLICY_CSV:-}" ]]; then
  MODEL_ARGS+=(--value-agent-policy-csv "$VALUE_POLICY_CSV")
fi

if [[ -n "${BASE_CONFIG:-}" ]]; then
  : "${ASSET_COUNT:?set ASSET_COUNT when BASE_CONFIG is used}"
  INPUT_ARGS=(--base-config "$BASE_CONFIG" --assets "$ASSET_COUNT")
else
  : "${UNIVERSE_CONFIG:?set UNIVERSE_CONFIG or BASE_CONFIG}"
  INPUT_ARGS=(--universe-config "$UNIVERSE_CONFIG")
fi

BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build-mpi}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results/seagull/$SLURM_JOB_ID}"
REPETITIONS="${REPETITIONS:-7}"
DURATION_SECONDS="${DURATION_SECONDS:-23400}"
SEED="${SEED:-20200130}"

mkdir -p "$RESULT_ROOT"

run_variant() {
  local label=$1
  local ranks=$2
  local threads=$3
  shift 3
  local variant_dir="$RESULT_ROOT/$label"
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
    srun --ntasks="$ranks" --cpus-per-task="$threads" \
      --cpu-bind=cores \
      "$BUILD_DIR/lob_mpi" \
      --duration-seconds "$DURATION_SECONDS" \
      --window-ms 1000 \
      "${INPUT_ARGS[@]}" \
      "${MODEL_ARGS[@]}" \
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
      --seed "$SEED" \
      --metrics-csv "$variant_dir/metrics_${rep}.csv" \
      --asset-summary-csv "$variant_dir/assets_${rep}.csv" \
      --asset-summary-interval-ms 1000 \
      --threads "$threads" \
      "$@" | tee "$variant_dir/run_${rep}.txt"
  done
}
