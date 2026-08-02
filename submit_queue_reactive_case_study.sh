#!/usr/bin/env bash
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
# Final Seagull launcher for the validated 1,480-book queue-reactive case.
#
# Submit from the project directory after creating ./slurm:
#   sbatch --export=ALL,EXPERIMENT=preflight submit_queue_reactive_case_study.sh
#   sbatch --export=ALL,EXPERIMENT=science   submit_queue_reactive_case_study.sh
#   sbatch --export=ALL,EXPERIMENT=scaling   submit_queue_reactive_case_study.sh
#
# EXPERIMENT=science always repeats the short rank-equivalence preflight before
# starting the 40 predeclared full-day paths.  This script does not calibrate.
#SBATCH --job-name=lob-r36-case
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -Eeuo pipefail

: "${SLURM_JOB_ID:?submit this file with sbatch}"
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
BASE_DIR="${BASE_DIR:-/home/users/mschpc/2025/czhang4}"
POOL_ROOT="${POOL_ROOT:-${BASE_DIR}/coupled_lob_r26_grid_projection_20260801/results/seagull/five_day_pool_45477}"
SELECTION_ROOT="${SELECTION_ROOT:-${BASE_DIR}/coupled_lob_r27_target_protocol_20260801/results/seagull/queue_selection_45480}"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/lob_empirical_compact_20260724_r2}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-${PROJECT_DIR}/case_evidence}"
EXPERIMENT="${EXPERIMENT:-preflight}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_DIR}/results/seagull/queue_case_${SLURM_JOB_ID}}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-seagull-r36-${SLURM_JOB_ID}}"
BUILD_JOBS="${BUILD_JOBS:-16}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-21600}"

# Frozen scientific protocol.  Do not turn these into a parameter sweep.
readonly DURATION_SECONDS=23400
readonly WINDOW_MS=1000
readonly SHOCK_TIME_SECONDS=11700
readonly SHOCK_FRACTION=0.01
readonly SHOCK_TARGET_COUNT=0
readonly SHOCK_TARGET_SEED=314159
readonly SHOCK_TOP_DEPTH_MULTIPLE=1.0
readonly SCIENCE_RANKS=32
readonly SCIENCE_RISK_LIMITS="25,100"
readonly REFERENCE_RISK_LIMIT=100
readonly LOCAL_INVENTORY_LIMIT=100
readonly CAPACITY_THRESHOLD=0.5
readonly REPETITIONS=5
readonly BASE_SEED=20200130
readonly POST_SHOCK_HORIZON_SECONDS=1800
readonly SCALING_RANKS="1,2,4,8,16,32"
readonly SCALING_REPETITIONS=3
readonly PREFLIGHT_DURATION_SECONDS=300

case "${EXPERIMENT}" in
    preflight|science|scaling|all) ;;
    *)
        echo "ERROR: EXPERIMENT must be preflight, science, scaling, or all." >&2
        exit 2
        ;;
esac
if (( SLURM_NTASKS < SCIENCE_RANKS )); then
    echo "ERROR: this protocol requires at least ${SCIENCE_RANKS} allocated tasks." >&2
    exit 2
fi

SEAGULL_MODULES="${SEAGULL_MODULES:-gcc/15.2.0-gcc-8.5.0-r7c4jsu openmpi/5.0.9-gcc-15.2.0-2irqibq cmake/3.31.9-gcc-15.2.0-ylutpfi ninja/1.13.0-gcc-15.2.0-nukwcsd python/3.14.2-gcc-15.2.0-e63sscp}"
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/lmod.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/lmod.sh
fi
if ! type module >/dev/null 2>&1; then
    echo "ERROR: the module command is unavailable." >&2
    exit 2
fi
module purge
# Intentional word splitting: this is a whitespace-separated module list.
# shellcheck disable=SC2086
module load ${SEAGULL_MODULES}

for command_name in cmake ninja mpicxx mpirun python3 sha256sum ldd; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "ERROR: ${command_name} is unavailable after loading modules." >&2
        exit 2
    fi
done
for required in \
    "${POOL_ROOT}/pooling_provenance.json" \
    "${POOL_ROOT}/heldout_common.csv" \
    "${SELECTION_ROOT}/selection/training_selection_freeze.json" \
    "${SELECTION_ROOT}/full_training_configs/deployment_config.csv" \
    "${SELECTION_ROOT}/queue_reactive_policy/symbol_policy_mapping.csv" \
    "${SELECTION_ROOT}/liquidity_clusters/cluster_assignments.csv" \
    "${EVIDENCE_ROOT}/development_validation/heldout_run_manifest.json" \
    "${EVIDENCE_ROOT}/training/expanded_training_freeze.json" \
    "${EVIDENCE_ROOT}/provenance/queue_reactive_augmentation_provenance.json"
do
    if [[ ! -s "${required}" ]]; then
        echo "ERROR: required deployment input is missing or empty: ${required}" >&2
        exit 2
    fi
done

MPI_LIB_DIR="$(mpicxx --showme:libdirs | awk '{print $1}')"
if [[ -z "${MPI_LIB_DIR}" || ! -d "${MPI_LIB_DIR}" ]]; then
    echo "ERROR: cannot determine the OpenMPI library directory." >&2
    exit 2
fi
if ! compgen -G "${MPI_LIB_DIR}/libmpi.so*" >/dev/null; then
    echo "ERROR: ${MPI_LIB_DIR} contains no libmpi.so." >&2
    exit 2
fi
export LD_LIBRARY_PATH="${MPI_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited

mkdir -p "${PROJECT_DIR}/slurm" "${RESULT_DIR}"
cd "${PROJECT_DIR}"

echo "== SOURCE AND ALLOCATION =="
date --iso-8601=seconds
hostname -f
echo "job_id=${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} tasks=${SLURM_NTASKS} experiment=${EXPERIMENT}"
echo "pool_root=${POOL_ROOT}"
echo "selection_root=${SELECTION_ROOT}"
echo "data_root=${DATA_ROOT}"
sha256sum -c SOURCE_MANIFEST.sha256

DETERMINISTIC_BUILD_CONTRACT="${PROJECT_DIR}/scripts/seagull_deterministic_build.sh"
if [[ ! -r "${DETERMINISTIC_BUILD_CONTRACT}" ]]; then
    echo "ERROR: missing deterministic build contract: ${DETERMINISTIC_BUILD_CONTRACT}" >&2
    exit 2
fi
# shellcheck source=scripts/seagull_deterministic_build.sh
source "${DETERMINISTIC_BUILD_CONTRACT}"
lob_deterministic_configure_and_build \
    "${PROJECT_DIR}" "${BUILD_DIR}" "${MPI_LIB_DIR}" "${BUILD_JOBS}"

EXECUTABLE="${BUILD_DIR}/fragmented_mpi_lob"
if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "ERROR: expected executable is missing: ${EXECUTABLE}" >&2
    exit 2
fi
runtime_libraries="$(ldd "${EXECUTABLE}" 2>&1)"
echo "${runtime_libraries}"
if [[ "${runtime_libraries}" == *"not found"* ]]; then
    echo "ERROR: executable has unresolved runtime libraries." >&2
    exit 2
fi

echo "== RELEVANT MODEL TESTS =="
ctest --test-dir "${BUILD_DIR}" \
    -R '^(background_hawkes_stream|fragmented_model_semantics)$' \
    --output-on-failure

PORTABLE_ROOT="${RESULT_DIR}/portable_case"
python3 "${PROJECT_DIR}/scripts/prepare_portable_queue_case.py" \
    --project-root "${PROJECT_DIR}" \
    --evidence-root "${EVIDENCE_ROOT}" \
    --pool-root "${POOL_ROOT}" \
    --selection-root "${SELECTION_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --executable "${EXECUTABLE}" \
    --output-root "${PORTABLE_ROOT}" \
    | tee "${RESULT_DIR}/portable_case_preparation.json"

CASE_ARTIFACT="${PORTABLE_ROOT}/portable_queue_reactive_case.json"
mapfile -t CASE_PATHS < <(python3 - "${CASE_ARTIFACT}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "portable_queue_reactive_case_ready":
    raise SystemExit("portable case artifact is not ready")
runtime = payload.get("runtime_artifacts")
if not isinstance(runtime, dict):
    raise SystemExit("portable case artifact lacks runtime_artifacts")
for key in ("case_config", "background_policy_mapping", "value_policy", "cluster_map"):
    record = runtime.get(key)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise SystemExit(f"portable case artifact lacks {key}")
    print(record["path"])
PY
)
if (( ${#CASE_PATHS[@]} != 4 )); then
    echo "ERROR: could not resolve all four portable runtime paths." >&2
    exit 2
fi
UNIVERSE_CONFIG="${CASE_PATHS[0]}"
BACKGROUND_POLICY="${CASE_PATHS[1]}"
VALUE_POLICY="${CASE_PATHS[2]}"
CLUSTER_MAP="${CASE_PATHS[3]}"

RUNNER=(
    python3 "${PROJECT_DIR}/scripts/run_fragmented_mpi_experiments.py"
    --executable "${EXECUTABLE}"
    --universe-config "${UNIVERSE_CONFIG}"
    --background-model queue-reactive-v1
    --background-policy-csv "${BACKGROUND_POLICY}"
    --value-agent-policy-csv "${VALUE_POLICY}"
    --campaign-manifest "${CASE_ARTIFACT}"
    --mpirun mpirun
    --bind-to core
    --map-by slot
    --run-timeout-seconds "${RUN_TIMEOUT_SECONDS}"
)
FROZEN_CONTROLS=(
    --window-ms "${WINDOW_MS}"
    --hawkes-activity-scales 0.3
    --local-mm-intervals-ms 1000
    --local-mm-quantity-multipliers 1.0
    --local-mm-improvement-probabilities 0.25
    --local-mm-spread-elasticities 0.0
    --local-mm-max-improvement-probabilities 1.0
    --shared-quote-relative
    --shared-quote-multiplier 1.0
    --shared-quote-levels 1
    --local-inventory-limit "${LOCAL_INVENTORY_LIMIT}"
    --capacity-thresholds "${CAPACITY_THRESHOLD}"
)
SHOCK_CONTROLS=(
    --shock-time-seconds "${SHOCK_TIME_SECONDS}"
    --shock-fraction "${SHOCK_FRACTION}"
    --shock-target-count "${SHOCK_TARGET_COUNT}"
    --shock-target-seed "${SHOCK_TARGET_SEED}"
    --shock-cluster-csv "${CLUSTER_MAP}"
    --shock-top-depth-multiple "${SHOCK_TOP_DEPTH_MULTIPLE}"
)

checkpoint_flag() {
    local output="$1"
    if [[ -s "${output}" ]]; then
        printf '%s' '--resume'
    fi
}

run_preflight() {
    local output="${RESULT_DIR}/rank_equivalence_raw.csv"
    local resume_flag
    resume_flag="$(checkpoint_flag "${output}")"
    echo "== 300-SECOND 1-RANK/32-RANK STATE-HASH PREFLIGHT =="
    "${RUNNER[@]}" "${FROZEN_CONTROLS[@]}" \
        --duration-seconds "${PREFLIGHT_DURATION_SECONDS}" \
        --shock-time-seconds 150 \
        --ranks "1,${SCIENCE_RANKS}" \
        --risk-limits "${REFERENCE_RISK_LIMIT}" \
        --shared-mm-modes global \
        --shock-modes off \
        --repetitions 1 \
        --seed "${BASE_SEED}" \
        --seed-step 0 \
        --output "${output}" \
        --summary "${RESULT_DIR}/rank_equivalence_summary.csv" \
        ${resume_flag:+"${resume_flag}"}
}

run_scaling() {
    local output="${RESULT_DIR}/final_model_scaling_raw.csv"
    local resume_flag
    resume_flag="$(checkpoint_flag "${output}")"
    echo "== FINAL QUEUE-REACTIVE FULL-DAY STRONG SCALING =="
    "${RUNNER[@]}" "${FROZEN_CONTROLS[@]}" \
        --duration-seconds "${DURATION_SECONDS}" \
        --shock-time-seconds "${SHOCK_TIME_SECONDS}" \
        --ranks "${SCALING_RANKS}" \
        --risk-limits "${REFERENCE_RISK_LIMIT}" \
        --shared-mm-modes global \
        --shock-modes off \
        --repetitions "${SCALING_REPETITIONS}" \
        --seed "${BASE_SEED}" \
        --seed-step 0 \
        --output "${output}" \
        --summary "${RESULT_DIR}/final_model_scaling_summary.csv" \
        ${resume_flag:+"${resume_flag}"}
}

run_science() {
    local global_output="${RESULT_DIR}/science_global_raw.csv"
    local uncoupled_output="${RESULT_DIR}/science_uncoupled_raw.csv"
    local off_output="${RESULT_DIR}/science_shared_off_raw.csv"
    local resume_flag

    echo "== 20 PATHS: GLOBAL CAPACITY, TWO RISK LIMITS =="
    resume_flag="$(checkpoint_flag "${global_output}")"
    "${RUNNER[@]}" "${FROZEN_CONTROLS[@]}" "${SHOCK_CONTROLS[@]}" \
        --duration-seconds "${DURATION_SECONDS}" \
        --ranks "${SCIENCE_RANKS}" \
        --risk-limits "${SCIENCE_RISK_LIMITS}" \
        --shared-mm-modes global \
        --shock-modes on,off \
        --repetitions "${REPETITIONS}" \
        --seed "${BASE_SEED}" \
        --seed-step 1 \
        --metrics-dir "${RESULT_DIR}/science_metrics/global" \
        --shock-targets-dir "${RESULT_DIR}/science_targets/global" \
        --asset-summary-dir "${RESULT_DIR}/science_asset_summaries/global" \
        --output "${global_output}" \
        --summary "${RESULT_DIR}/science_global_summary.csv" \
        ${resume_flag:+"${resume_flag}"}

    echo "== 10 PATHS: UNCOUPLED SHARED-LIQUIDITY CONTROL =="
    resume_flag="$(checkpoint_flag "${uncoupled_output}")"
    "${RUNNER[@]}" "${FROZEN_CONTROLS[@]}" "${SHOCK_CONTROLS[@]}" \
        --duration-seconds "${DURATION_SECONDS}" \
        --ranks "${SCIENCE_RANKS}" \
        --risk-limits "${REFERENCE_RISK_LIMIT}" \
        --shared-mm-modes uncoupled \
        --shock-modes on,off \
        --repetitions "${REPETITIONS}" \
        --seed "${BASE_SEED}" \
        --seed-step 1 \
        --metrics-dir "${RESULT_DIR}/science_metrics/uncoupled" \
        --shock-targets-dir "${RESULT_DIR}/science_targets/uncoupled" \
        --asset-summary-dir "${RESULT_DIR}/science_asset_summaries/uncoupled" \
        --output "${uncoupled_output}" \
        --summary "${RESULT_DIR}/science_uncoupled_summary.csv" \
        ${resume_flag:+"${resume_flag}"}

    echo "== 10 PATHS: SHARED MARKET MAKER ABSENT =="
    resume_flag="$(checkpoint_flag "${off_output}")"
    "${RUNNER[@]}" "${FROZEN_CONTROLS[@]}" "${SHOCK_CONTROLS[@]}" \
        --duration-seconds "${DURATION_SECONDS}" \
        --ranks "${SCIENCE_RANKS}" \
        --risk-limits "${REFERENCE_RISK_LIMIT}" \
        --shared-mm-modes off \
        --shock-modes on,off \
        --repetitions "${REPETITIONS}" \
        --seed "${BASE_SEED}" \
        --seed-step 1 \
        --metrics-dir "${RESULT_DIR}/science_metrics/shared_off" \
        --shock-targets-dir "${RESULT_DIR}/science_targets/shared_off" \
        --asset-summary-dir "${RESULT_DIR}/science_asset_summaries/shared_off" \
        --output "${off_output}" \
        --summary "${RESULT_DIR}/science_shared_off_summary.csv" \
        ${resume_flag:+"${resume_flag}"}

    echo "== PAIRED MARKET-WIDE AND CLUSTER ANALYSIS =="
    python3 "${PROJECT_DIR}/scripts/analyze_fragmented_shared_liquidity_case.py" \
        --global-raw "${global_output}" \
        --uncoupled-raw "${uncoupled_output}" \
        --shared-off-raw "${off_output}" \
        --universe-input "${CASE_ARTIFACT}" \
        --shock-time-seconds "${SHOCK_TIME_SECONDS}" \
        --horizon-seconds "${POST_SHOCK_HORIZON_SECONDS}" \
        --rank "${SCIENCE_RANKS}" \
        --output-dir "${RESULT_DIR}/science_analysis"

    python3 "${PROJECT_DIR}/scripts/analyze_cluster_liquidity_heterogeneity.py" \
        --global-raw "${global_output}" \
        --uncoupled-raw "${uncoupled_output}" \
        --shared-off-raw "${off_output}" \
        --universe-config "${UNIVERSE_CONFIG}" \
        --cluster-assignments "${CLUSTER_MAP}" \
        --rank "${SCIENCE_RANKS}" \
        --output-dir "${RESULT_DIR}/science_cluster_analysis"
}

run_preflight
case "${EXPERIMENT}" in
    preflight) ;;
    science) run_science ;;
    scaling) run_scaling ;;
    all)
        run_scaling
        run_science
        ;;
esac

python3 - "${RESULT_DIR}" "${CASE_ARTIFACT}" "${EXPERIMENT}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
artifact = pathlib.Path(sys.argv[2]).resolve()
experiment = sys.argv[3]
files = []
for path in sorted(root.rglob("*.csv")):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    files.append({"path": str(path), "sha256": digest, "bytes": path.stat().st_size})
payload = {
    "schema_version": 1,
    "status": "queue_reactive_case_job_completed",
    "experiment": experiment,
    "case_artifact": str(artifact),
    "case_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    "csv_count": len(files),
    "csv_artifacts": files,
}
(root / "case_job_completion.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "result_dir=${RESULT_DIR}"
echo "completion_manifest=${RESULT_DIR}/case_job_completion.json"
date --iso-8601=seconds
