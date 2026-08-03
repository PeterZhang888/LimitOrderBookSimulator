#!/usr/bin/env bash
# Full-universe queue-reactive validation. The workflow evaluates the selected
# training iteration under the six-component protocol, freezes the model, and
# only then runs five 2020 development-validation seeds. Cached simulations are
# reused only when their exact command hashes match.
#
# Submit after phase 1 succeeds:
#   sbatch --export=ALL,SELECTION_ROOT=/abs/queue_selection_JOB,\
# POOL_ROOT=/abs/pool submit_queue_reactive_full_validation_hpc.sh
#SBATCH --job-name=lob-model-validate
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -Eeuo pipefail

: "${SLURM_JOB_ID:?submit this script with sbatch}"
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}"
: "${SELECTION_ROOT:?provide the passed phase-1 selection root}"
: "${POOL_ROOT:?provide the completed five-day pool root}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
SELECTION_ROOT="$(cd "${SELECTION_ROOT}" && pwd -P)"
POOL_ROOT="$(cd "${POOL_ROOT}" && pwd -P)"
RESULT_DIR="${RESULT_DIR:-${PROJECT_DIR}/results/seagull/queue_full_validation_${SLURM_JOB_ID}}"
MPI_RANKS_PER_RUN="${MPI_RANKS_PER_RUN:-32}"
RUN_WORKERS="${RUN_WORKERS:-1}"
RANK_EQUIVALENCE_DURATION="${RANK_EQUIVALENCE_DURATION:-300}"
MAX_REFINEMENT_ITERATIONS="${MAX_REFINEMENT_ITERATIONS:-1}"
MINIMUM_REFINEMENT_ITERATIONS="${MINIMUM_REFINEMENT_ITERATIONS:-1}"
GATE_PROTOCOL="${GATE_PROTOCOL:-marketwide-six-v2}"
ACTIVITY_SCALE="${ACTIVITY_SCALE:-${COUPLING_SCALE:-1.0}}"
BUILD_JOBS="${BUILD_JOBS:-16}"
RESUME="${RESUME:-off}"
PILOT_ONLY="${PILOT_ONLY:-on}"
ALLOW_PILOT_REJECTION="${ALLOW_PILOT_REJECTION:-on}"
SEAGULL_GCC_MODULE="gcc/15.2.0-gcc-8.5.0-r7c4jsu"
SEAGULL_MODULES="${SEAGULL_MODULES:-${SEAGULL_GCC_MODULE} openmpi/5.0.9-gcc-15.2.0-2irqibq cmake/3.31.9-gcc-15.2.0-ylutpfi ninja/1.13.0-gcc-15.2.0-nukwcsd python/3.14.2-gcc-15.2.0-e63sscp}"
TRAINING_DATES=(2019-01-30 2019-03-27 2019-07-30 2019-10-30 2019-12-30)

fail() { echo "ERROR: $*" >&2; exit 2; }
for pair in "MPI_RANKS_PER_RUN=${MPI_RANKS_PER_RUN}" "RUN_WORKERS=${RUN_WORKERS}"; do
    value="${pair#*=}"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "${pair%%=*} must be positive"
done
[[ "${MAX_REFINEMENT_ITERATIONS}" =~ ^[0-4]$ ]] || fail \
    "MAX_REFINEMENT_ITERATIONS must be an integer from zero to four"
[[ "${MINIMUM_REFINEMENT_ITERATIONS}" =~ ^[0-4]$ ]] || fail \
    "MINIMUM_REFINEMENT_ITERATIONS must be an integer from zero to four"
(( MINIMUM_REFINEMENT_ITERATIONS <= MAX_REFINEMENT_ITERATIONS )) || fail \
    "MINIMUM_REFINEMENT_ITERATIONS exceeds MAX_REFINEMENT_ITERATIONS"
case "${GATE_PROTOCOL}" in
    strict-nine-v1|marketwide-six-v2) ;;
    *) fail "GATE_PROTOCOL must be strict-nine-v1 or marketwide-six-v2" ;;
esac
(( RUN_WORKERS == 1 )) || fail \
    "this mpirun submission uses one MPI realization at a time; concurrent paths require disjoint Slurm job-array allocations"
(( MPI_RANKS_PER_RUN * RUN_WORKERS <= SLURM_NTASKS )) || fail \
    "rank/work matrix needs $((MPI_RANKS_PER_RUN * RUN_WORKERS)) tasks; allocation has ${SLURM_NTASKS}"
case "${RESUME}" in on|off) ;; *) fail "RESUME must be on or off" ;; esac
case "${PILOT_ONLY}" in on|off) ;; *) fail "PILOT_ONLY must be on or off" ;; esac
case "${ALLOW_PILOT_REJECTION}" in on|off) ;; *) fail "ALLOW_PILOT_REJECTION must be on or off" ;; esac
if [[ "${PILOT_ONLY}" == "off" && "${RESUME}" != "on" ]]; then
    fail "PILOT_ONLY=off requires RESUME=on and the exact passed pilot RESULT_DIR"
fi
if [[ "${RESUME}" == "on" ]]; then
    [[ -d "${RESULT_DIR}" ]] || fail \
        "RESUME=on requires the exact existing RESULT_DIR: ${RESULT_DIR}"
else
    [[ ! -e "${RESULT_DIR}" ]] || fail "RESULT_DIR already exists: ${RESULT_DIR}"
fi
mkdir -p "${RESULT_DIR}" "${PROJECT_DIR}/slurm"

SIX_COMPONENT_CERTIFICATE="${SIX_COMPONENT_CERTIFICATE:-${RESULT_DIR}/full_training_adequacy/full_universe_refinement/iteration_1/six_component_training_certificate.json}"
if [[ "${GATE_PROTOCOL}" == "marketwide-six-v2" ]]; then
    [[ "${RESUME}" == "on" && "${PILOT_ONLY}" == "off" ]] || fail \
        "marketwide-six-v2 continuation requires RESUME=on and PILOT_ONLY=off"
    [[ -s "${SIX_COMPONENT_CERTIFICATE}" ]] || fail \
        "six-component protocol certificate is missing: ${SIX_COMPONENT_CERTIFICATE}"
fi

# Refuse to spend cluster time on a truncated, mixed, or locally edited
# release. The package manifest covers every source and configuration file
# used by this workflow.
[[ -s "${PROJECT_DIR}/SOURCE_MANIFEST.sha256" ]] || fail \
    "SOURCE_MANIFEST.sha256 is missing"
(
    cd "${PROJECT_DIR}"
    sha256sum --quiet -c SOURCE_MANIFEST.sha256
) || fail "source-manifest verification failed"

if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/lmod.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/lmod.sh
fi
type module >/dev/null 2>&1 || fail "module command is unavailable"
module purge
# shellcheck disable=SC2086
module load ${SEAGULL_MODULES}
for name in cmake ninja mpicxx mpirun python3; do
    command -v "${name}" >/dev/null 2>&1 || fail "${name} is unavailable"
done
python3 - "${ACTIVITY_SCALE}" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError as error:
    raise SystemExit(f"ACTIVITY_SCALE is not numeric: {sys.argv[1]}") from error
if not math.isfinite(value) or not 0.5 <= value <= 1.25:
    raise SystemExit("ACTIVITY_SCALE must be finite and in [0.5,1.25]")
PY
MPI_LIB_DIR="$(mpicxx --showme:libdirs | awk '{print $1}')"
[[ -d "${MPI_LIB_DIR}" ]] || fail "cannot locate OpenMPI libraries"
export LD_LIBRARY_PATH="${MPI_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -s unlimited

# Build and test the current source inside the compute allocation before any
# empirical target is opened. Checkpoint resumption hashes the full invocation,
# including the executable path, so a resumed run must reuse the same BUILD_DIR.
# Do not launch concurrent validation jobs against one result directory.
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-seagull-model-validation}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/seagull_deterministic_build.sh"
lob_deterministic_configure_and_build \
    "${PROJECT_DIR}" "${BUILD_DIR}" "${MPI_LIB_DIR}" "${BUILD_JOBS}"
EXECUTABLE="${BUILD_DIR}/fragmented_mpi_lob"
[[ -x "${EXECUTABLE}" ]] || fail "validation executable was not built: ${EXECUTABLE}"
ctest --test-dir "${BUILD_DIR}" --output-on-failure \
    -R '^(background_hawkes_stream|fragmented_model_semantics)$'
python3 -m unittest \
    tests.test_strict_model_validation \
    tests.test_queue_reactive_calibration_driver \
    tests.test_liquidity_regime_protocol \
    tests.test_prepare_heldout_target_config \
    tests.test_resolve_queue_reactive_case_artifact.QueueReactiveCaseArtifactTest

SELECTION_FREEZE="${SELECTION_ROOT}/selection/training_selection_freeze.json"
FULL_CONFIG_ROOT="${SELECTION_ROOT}/full_training_configs"
POLICY_ROOT="${SELECTION_ROOT}/queue_reactive_policy"
CLUSTER_DIR="${SELECTION_ROOT}/liquidity_clusters"
HELDOUT_CONFIG="${POOL_ROOT}/heldout_common.csv"
POOLING_PROVENANCE="${POOL_ROOT}/pooling_provenance.json"
# heldout_common.csv is the runtime/opening configuration.  Its data_dir is
# deliberately inherited from pooled 2019 training and therefore must not be
# used as the 2020 empirical target. The hash-bound source config still has
# extraction-host paths, so create a separate evaluator-only config whose
# target_data_dir values are rebased onto heldout.target_root. Every manifest
# and target file is verified before any held-out result is evaluated.
HELDOUT_INPUT_DIR="${RESULT_DIR}/heldout_inputs"
HELDOUT_TARGET_CONFIG="${HELDOUT_INPUT_DIR}/heldout_target_config_20200130.csv"
mkdir -p "${HELDOUT_INPUT_DIR}"
python3 scripts/prepare_heldout_target_config.py \
    --pooling-provenance "${POOLING_PROVENANCE}" \
    --expected-date 2020-01-30 \
    --output "${HELDOUT_TARGET_CONFIG}" \
    >"${HELDOUT_INPUT_DIR}/preparation_stdout.json"
TRAINING_REFINEMENT_MANIFEST="${PROJECT_DIR}/config/training_refinement_seed.json"
for path in \
    "${SELECTION_FREEZE}" \
    "${FULL_CONFIG_ROOT}/deployment_config.csv" \
    "${POLICY_ROOT}/symbol_policy_mapping.csv" \
    "${CLUSTER_DIR}/cluster_assignments.csv" \
    "${HELDOUT_CONFIG}" \
    "${HELDOUT_TARGET_CONFIG}" \
    "${POOLING_PROVENANCE}" \
    "${TRAINING_REFINEMENT_MANIFEST}"
do
    [[ -s "${path}" ]] || fail "required phase input is missing: ${path}"
done

cd "${PROJECT_DIR}"
PRODUCTION_LAUNCHER="mpirun --bind-to core --map-by slot -np ${MPI_RANKS_PER_RUN}"
REFERENCE_LAUNCHER="mpirun --bind-to core --map-by slot -np 1"
mpirun --bind-to core --map-by slot --report-bindings \
    -np "${MPI_RANKS_PER_RUN}" /bin/true \
    >"${RESULT_DIR}/mpi_placement.txt" 2>&1
[[ -s "${RESULT_DIR}/mpi_placement.txt" ]] || fail \
    "OpenMPI produced no placement report"

EXPAND_ARGS=(
    python3 scripts/calibrate_queue_reactive_model.py expand-full-universe
    --freeze-record "${SELECTION_FREEZE}"
    --executable "${EXECUTABLE}"
    --full-deployment-config "${FULL_CONFIG_ROOT}/deployment_config.csv"
    --full-background-policy "${POLICY_ROOT}/symbol_policy_mapping.csv"
    --full-cluster-map "${CLUSTER_DIR}/cluster_assignments.csv"
    --launcher "${PRODUCTION_LAUNCHER}"
    --reference-launcher "${REFERENCE_LAUNCHER}"
    --rank-equivalence-duration "${RANK_EQUIVALENCE_DURATION}"
    --max-refinement-iterations "${MAX_REFINEMENT_ITERATIONS}"
    --minimum-refinement-iterations "${MINIMUM_REFINEMENT_ITERATIONS}"
    --gate-protocol "${GATE_PROTOCOL}"
    --initial-refinement-manifest "${TRAINING_REFINEMENT_MANIFEST}"
    --initial-coupling-scale "${ACTIVITY_SCALE}"
    --run-workers "${RUN_WORKERS}"
    --output-root "${RESULT_DIR}/full_training_adequacy"
)
if [[ "${GATE_PROTOCOL}" == "marketwide-six-v2" ]]; then
    EXPAND_ARGS+=(
        --six-component-protocol-certificate "${SIX_COMPONENT_CERTIFICATE}"
    )
fi
if [[ "${PILOT_ONLY}" == "on" ]]; then
    EXPAND_ARGS+=(--directional-pilot --directional-pilot-only)
fi
for day in "${TRAINING_DATES[@]}"; do
    compact="${day//-/}"
    config="${FULL_CONFIG_ROOT}/dated_config_${compact}.csv"
    [[ -s "${config}" ]] || fail "missing dated full-universe config: ${config}"
    EXPAND_ARGS+=(--training-config "${day}=${config}")
done
if [[ "${RESUME}" == "on" ]]; then
    EXPAND_ARGS+=(--resume)
fi
set +e
"${EXPAND_ARGS[@]}"
EXPAND_STATUS=$?
set -e

if (( EXPAND_STATUS != 0 )); then
    PILOT_DECISION="${RESULT_DIR}/full_training_adequacy/directional_pilot/pilot_decision.json"
    if [[ "${PILOT_ONLY}" == "on" \
          && "${ALLOW_PILOT_REJECTION}" == "on" \
          && -s "${PILOT_DECISION}" ]] \
       && python3 - "${PILOT_DECISION}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("status") == "rejected" else 1)
PY
    then
        echo "DIRECTIONAL PILOT: REJECTED BY FROZEN TRAINING GATE"
        echo "This is a completed scientific candidate, not an infrastructure failure."
        echo "No 25-run matrix or held-out validation was launched."
        echo "activity_scale=${ACTIVITY_SCALE}"
        echo "result_dir=${RESULT_DIR}"
        echo "pilot_decision=${PILOT_DECISION}"
        exit 0
    fi
    fail "calibration driver exited with status ${EXPAND_STATUS}"
fi

if [[ "${PILOT_ONLY}" == "on" ]]; then
    PILOT_HANDOFF="${RESULT_DIR}/full_training_adequacy/directional_pilot/directional_pilot_handoff.json"
    [[ -s "${PILOT_HANDOFF}" ]] || fail \
        "directional pilot completed without a passed handoff"
    echo "DIRECTIONAL PILOT: PASS"
    echo "No 25-run matrix or held-out validation was launched."
    echo "activity_scale=${ACTIVITY_SCALE}"
    echo "result_dir=${RESULT_DIR}"
    echo "pilot_handoff=${PILOT_HANDOFF}"
    exit 0
fi

PILOT_HANDOFF="${RESULT_DIR}/full_training_adequacy/directional_pilot/directional_pilot_handoff.json"
[[ -s "${PILOT_HANDOFF}" ]] || fail \
    "PILOT_ONLY=off requires the passed pilot handoff in RESULT_DIR"

EXPANDED_FREEZE="${RESULT_DIR}/full_training_adequacy/expanded_training_freeze.json"
[[ -s "${EXPANDED_FREEZE}" ]] || fail "expanded training freeze was not written"

HELDOUT_ARGS=(
    python3 scripts/calibrate_queue_reactive_model.py heldout
    --freeze-record "${EXPANDED_FREEZE}"
    --heldout-date 2020-01-30
    --heldout-opening-config "${HELDOUT_CONFIG}"
    --heldout-target-config "${HELDOUT_TARGET_CONFIG}"
    --heldout-seed 6599 --heldout-seed 2027 --heldout-seed 31337
    --heldout-seed 4242 --heldout-seed 9001
    --heldout-role development_validation
    --launcher "${PRODUCTION_LAUNCHER}"
    --run-workers "${RUN_WORKERS}"
    --output-root "${RESULT_DIR}/development_validation"
)
if [[ "${RESUME}" == "on" ]]; then
    HELDOUT_ARGS+=(--resume)
fi
"${HELDOUT_ARGS[@]}"

HANDOFF="${RESULT_DIR}/development_validation/heldout_run_manifest.json"
[[ -s "${HANDOFF}" ]] || fail "passed development-validation handoff is absent"
python3 scripts/resolve_queue_reactive_case_artifact.py \
    --artifact "${HANDOFF}" \
    --output "${RESULT_DIR}/case_study_artifact_resolution.json"

echo "QUEUE-REACTIVE FULL TRAINING AND VALIDATION: PASS"
echo "result_dir=${RESULT_DIR}"
echo "case_study_artifact=${HANDOFF}"
