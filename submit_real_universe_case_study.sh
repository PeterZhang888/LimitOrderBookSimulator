#!/usr/bin/env bash
# Run the final one-book-per-symbol shared-liquidity case study on Seagull.
#
# First build the audited empirical universe on compute nodes:
#   sbatch --export=ALL,ITCH_FILE=/shared/path/01302020.NASDAQ_ITCH50.gz \
#       submit_itch50_universe_calibration.sh
#
# Then submit this script from the login node (the simulation itself does no
# work there). The calibrated path requires the immutable handoff emitted by
# submit_cluster_value_agent_calibration.sh. In the canonical five-session
# protocol the financial case uses the report-linked
# `heldout_openings_frozen_backgrounds.csv`: training-period background and
# behavioural inputs with only the held-out opening state substituted.  For
# backward compatibility the old pooled-training UNIVERSE_CONFIG argument is
# accepted, but handoff mode always resolves and records the frozen held-out
# artifact before any simulation starts:
#   sbatch --export=ALL,EXPERIMENT=science,VALUE_AGENT=on,\
#       UNIVERSE_CONFIG=/absolute/path/calibration/heldout_openings_frozen_backgrounds.csv,\
#       CALIBRATION_HANDOFF_JSON=/absolute/path/calibration/calibration_handoff.json \
#       submit_real_universe_case_study.sh
# `CALIBRATION_METADATA` is optional launcher provenance in certified-handoff
# mode because compact empirical bundles intentionally omit it.  It remains
# mandatory for the explicitly uncalibrated legacy path, where no handoff
# otherwise certifies the requested universe.
#
# Select one workload with --export=ALL,EXPERIMENT=...:
#   scaling  : full-day, real-universe rank-invariance and strong scaling
#   pilot    : one-seed capacity sweep used to choose transparent scenarios
#   science  : paired global/uncoupled/off financial case study
#   sensitivity: shock-size, threshold, one-book, and value-agent checks
#   cadence  : optional 1 s versus 100 ms global shared-risk/value-decision
#              cadence sensitivity; the calibrated local-MM cadence is held fixed
#   all      : scaling + science + sensitivity + cadence (pilot remains separate)
#
# The core result has one displayed NASDAQ book per empirical symbol.  It does
# not divide NASDAQ ITCH flow across fabricated venues and it contains no ETF
# arbitrage mechanism.  QQQ, if selected, is simply one NASDAQ-traded symbol.
#
# Edit literal #SBATCH directives or override them with sbatch options.  Slurm
# reads these before Bash begins, so shell variables cannot control them.
#SBATCH --job-name=real-lob-case
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=4-00:00:00
#SBATCH --exclusive
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -Eeuo pipefail

: "${SLURM_JOB_ID:?submit this file with sbatch}"
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}"
: "${UNIVERSE_CONFIG:?provide UNIVERSE_CONFIG when submitting this job}"

# ---------------------------------------------------------------------------
# Scientific design controls. The main study has 8 cases/seed:
#   shared MM off: shock/control (2), uncoupled: shock/control (2), and
#   globally constrained: two capacities x shock/control (4).
# Five matched path seeds therefore give 40 full-day simulations. Capacity
# values are transparent scenario parameters, not dealer balance-sheet estimates.
# Run EXPERIMENT=pilot before changing SCIENCE_RISK_LIMITS if the reference
# setting has no material quote-scale response to the stress.
# ---------------------------------------------------------------------------
EXPERIMENT="${EXPERIMENT:-science}"
DURATION_SECONDS="${DURATION_SECONDS:-23400}"
WINDOW_MS="${WINDOW_MS:-1000}"
SHOCK_TIME_SECONDS="${SHOCK_TIME_SECONDS:-$(( DURATION_SECONDS / 2 ))}"
SHOCK_FRACTION="${SHOCK_FRACTION:-0.01}"
SHOCK_TARGET_COUNT="0"
# Positive: every sell stress is this multiple of contemporaneous bid depth
# immediately before t_s.  Fixed share quantity is not used in the core study.
SHOCK_TOP_DEPTH_MULTIPLE="${SHOCK_TOP_DEPTH_MULTIPLE:-1.0}"
SHARED_QUOTE_MULTIPLIER="${SHARED_QUOTE_MULTIPLIER:-1.0}"
SHARED_QUOTE_LEVELS="${SHARED_QUOTE_LEVELS:-1}"
SEED="${SEED:-20200130}"
MASK_SEED="${MASK_SEED:-314159}"
ALTERNATIVE_MASK_SEED="${ALTERNATIVE_MASK_SEED:-271828}"
REPETITIONS="${REPETITIONS:-5}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-21600}"
SCIENCE_RANKS="${SCIENCE_RANKS:-32}"
SCIENCE_RISK_LIMITS="${SCIENCE_RISK_LIMITS:-25,100}"
REFERENCE_RISK_LIMIT="${REFERENCE_RISK_LIMIT:-100}"
LOCAL_INVENTORY_LIMIT="${LOCAL_INVENTORY_LIMIT:-100}"
CAPACITY_THRESHOLD="${CAPACITY_THRESHOLD:-0.5}"
ALTERNATIVE_CAPACITY_THRESHOLD="${ALTERNATIVE_CAPACITY_THRESHOLD:-0.75}"
SHOCK_CLUSTER_CSV="${SHOCK_CLUSTER_CSV:-}"
POST_SHOCK_HORIZON_SECONDS="${POST_SHOCK_HORIZON_SECONDS:-1800}"

# The calibrated workflow uses the immutable handoff emitted by
# calibrate_cluster_value_agents.py.  It supplies the policy plus all four
# selected runtime controls.  A hand-entered policy is intentionally rejected:
# it would sever the policy from the global controls selected alongside it.
#
# An uncalibrated legacy run is allowed only when explicitly labelled with
# LEGACY_UNCALIBRATED_MODE=on and VALUE_AGENT=off.  It is useful for debugging
# or an ablation, but must not be reported as the calibrated empirical model.
VALUE_AGENT="${VALUE_AGENT:-on}"
VALUE_AGENT_POLICY_CSV="${VALUE_AGENT_POLICY_CSV:-}"
CALIBRATION_HANDOFF_JSON="${CALIBRATION_HANDOFF_JSON:-}"
CALIBRATION_METADATA="${CALIBRATION_METADATA:-}"
LEGACY_UNCALIBRATED_MODE="${LEGACY_UNCALIBRATED_MODE:-off}"
ALLOW_PRELIMINARY_MODEL="${ALLOW_PRELIMINARY_MODEL:-off}"

# Full-day real-universe rank-invariance/scaling; every rank must return the
# same state hash for this one coupled market realisation.
SCALING_RANKS="${SCALING_RANKS:-1,2,4,8,16,32}"
PILOT_RISK_LIMITS="${PILOT_RISK_LIMITS:-5,10,25,50,100,200,400,1000000000}"
PILOT_RANKS="${PILOT_RANKS:-${SCIENCE_RANKS}}"

# Optional cadence sensitivity.  It changes the global shared-risk, shared-MM,
# value-agent and metric schedule while retaining the calibrated local-MM
# interval.  It therefore measures a deliberate global-decision/communication
# sensitivity, not a re-fit of local liquidity behaviour.
CADENCE_WINDOWS_MS="${CADENCE_WINDOWS_MS:-1000,100}"
CADENCE_RANKS="${CADENCE_RANKS:-${SCIENCE_RANKS}}"
CADENCE_REPETITIONS="${CADENCE_REPETITIONS:-${REPETITIONS}}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
# A job-specific build tree prevents concurrent pilot/science allocations from
# racing on one CMake cache, Ninja database, or executable.
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-seagull-gcc15-ompi509-real-case-${SLURM_JOB_ID}}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_DIR}/results/seagull/real_universe_case_${SLURM_JOB_ID}}"
BUILD_JOBS="${BUILD_JOBS:-16}"

# Exact compatible Seagull modules.  This job uses mpirun rather than direct
# srun/PMIx launch because the latter previously crashed in PMIx on Seagull.
SEAGULL_MODULES="${SEAGULL_MODULES:-gcc/15.2.0-gcc-8.5.0-r7c4jsu openmpi/5.0.9-gcc-15.2.0-2irqibq cmake/3.31.9-gcc-15.2.0-ylutpfi ninja/1.13.0-gcc-15.2.0-nukwcsd python/3.14.2-gcc-15.2.0-e63sscp}"

validate_positive_integer() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
        echo "ERROR: ${name} must be a positive integer; observed '${value}'." >&2
        exit 2
    fi
}

validate_finite_positive() {
    local name="$1"
    local value="$2"
    python3 - "${name}" "${value}" <<'PY'
import math
import sys
name, text = sys.argv[1:]
try:
    value = float(text)
except ValueError:
    raise SystemExit(f"ERROR: {name} must be numeric; observed {text!r}.")
if not math.isfinite(value) or value <= 0.0:
    raise SystemExit(f"ERROR: {name} must be finite and positive; observed {text!r}.")
PY
}

normalise_list() {
    local value="$1"
    value="${value//:/,}"
    value="${value// /,}"
    while [[ "${value}" == *",,"* ]]; do value="${value//,,/,}"; done
    value="${value#,}"
    value="${value%,}"
    printf '%s' "${value}"
}

max_from_integer_list() {
    local values="$1"
    local maximum=0 item
    IFS=',' read -r -a list_items <<< "${values}"
    for item in "${list_items[@]}"; do
        validate_positive_integer "list entry" "${item}"
        if (( item > maximum )); then
            maximum="${item}"
        fi
    done
    printf '%s' "${maximum}"
}

validate_float_list() {
    local name="$1"
    local values="$2"
    python3 - "${name}" "${values}" <<'PY'
import math
import sys
name, text = sys.argv[1:]
parts = [item.strip() for item in text.split(",") if item.strip()]
if not parts:
    raise SystemExit(f"ERROR: {name} must be a non-empty comma-separated list.")
for item in parts:
    try:
        value = float(item)
    except ValueError:
        raise SystemExit(f"ERROR: {name} contains a non-numeric value: {item!r}.")
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"ERROR: {name} contains an invalid value: {item!r}.")
PY
}

case "${EXPERIMENT}" in
    scaling|pilot|science|sensitivity|cadence|all) ;;
    *)
        echo "ERROR: EXPERIMENT must be scaling, pilot, science, sensitivity, cadence, or all." >&2
        exit 2
        ;;
esac
case "${VALUE_AGENT}" in
    on|off) ;;
    *)
        echo "ERROR: VALUE_AGENT must be on or off." >&2
        exit 2
        ;;
esac
case "${LEGACY_UNCALIBRATED_MODE}" in
    on|off) ;;
    *)
        echo "ERROR: LEGACY_UNCALIBRATED_MODE must be on or off." >&2
        exit 2
        ;;
esac
case "${ALLOW_PRELIMINARY_MODEL}" in
    on|off) ;;
    *)
        echo "ERROR: ALLOW_PRELIMINARY_MODEL must be on or off." >&2
        exit 2
        ;;
esac
if [[ -n "${VALUE_AGENT_POLICY_CSV}" && -z "${CALIBRATION_HANDOFF_JSON}" ]]; then
    echo "ERROR: a calibrated VALUE_AGENT_POLICY_CSV requires CALIBRATION_HANDOFF_JSON; the handoff freezes its jointly selected runtime controls." >&2
    exit 2
fi
if [[ "${VALUE_AGENT}" == "on" && -z "${CALIBRATION_HANDOFF_JSON}" ]]; then
    echo "ERROR: VALUE_AGENT=on requires CALIBRATION_HANDOFF_JSON from the block-coordinate calibration workflow." >&2
    exit 2
fi
if [[ "${VALUE_AGENT}" == "off" && -n "${VALUE_AGENT_POLICY_CSV}" ]]; then
    echo "ERROR: VALUE_AGENT_POLICY_CSV was supplied while VALUE_AGENT=off." >&2
    exit 2
fi
if [[ -z "${CALIBRATION_HANDOFF_JSON}" && "${LEGACY_UNCALIBRATED_MODE}" != "on" ]]; then
    echo "ERROR: CALIBRATION_HANDOFF_JSON is required for the final empirical workflow. For an explicitly uncalibrated debugging/ablation run only, set LEGACY_UNCALIBRATED_MODE=on with VALUE_AGENT=off." >&2
    exit 2
fi
if [[ "${LEGACY_UNCALIBRATED_MODE}" == "on" && "${VALUE_AGENT}" != "off" ]]; then
    echo "ERROR: LEGACY_UNCALIBRATED_MODE=on is permitted only with VALUE_AGENT=off." >&2
    exit 2
fi
if [[ "${LEGACY_UNCALIBRATED_MODE}" == "on" && -n "${CALIBRATION_HANDOFF_JSON}" ]]; then
    echo "ERROR: do not combine LEGACY_UNCALIBRATED_MODE=on with CALIBRATION_HANDOFF_JSON." >&2
    exit 2
fi
if [[ "${LEGACY_UNCALIBRATED_MODE}" == "on" && -z "${CALIBRATION_METADATA}" ]]; then
    echo "ERROR: legacy uncalibrated mode requires CALIBRATION_METADATA from submit_itch50_universe_calibration.sh." >&2
    exit 2
fi

DURATION_SECONDS="$(normalise_list "${DURATION_SECONDS}")"
SCIENCE_RISK_LIMITS="$(normalise_list "${SCIENCE_RISK_LIMITS}")"
PILOT_RISK_LIMITS="$(normalise_list "${PILOT_RISK_LIMITS}")"
validate_positive_integer DURATION_SECONDS "${DURATION_SECONDS}"
validate_positive_integer REPETITIONS "${REPETITIONS}"
validate_positive_integer RUN_TIMEOUT_SECONDS "${RUN_TIMEOUT_SECONDS}"
validate_positive_integer SCIENCE_RANKS "${SCIENCE_RANKS}"
validate_positive_integer PILOT_RANKS "${PILOT_RANKS}"
validate_positive_integer CADENCE_RANKS "${CADENCE_RANKS}"
validate_positive_integer CADENCE_REPETITIONS "${CADENCE_REPETITIONS}"
validate_positive_integer SHARED_QUOTE_LEVELS "${SHARED_QUOTE_LEVELS}"
validate_positive_integer POST_SHOCK_HORIZON_SECONDS "${POST_SHOCK_HORIZON_SECONDS}"
validate_finite_positive WINDOW_MS "${WINDOW_MS}"
validate_finite_positive SHOCK_FRACTION "${SHOCK_FRACTION}"
validate_finite_positive SHOCK_TOP_DEPTH_MULTIPLE "${SHOCK_TOP_DEPTH_MULTIPLE}"
validate_finite_positive SHARED_QUOTE_MULTIPLIER "${SHARED_QUOTE_MULTIPLIER}"
validate_float_list SCIENCE_RISK_LIMITS "${SCIENCE_RISK_LIMITS}"
validate_float_list PILOT_RISK_LIMITS "${PILOT_RISK_LIMITS}"
validate_finite_positive REFERENCE_RISK_LIMIT "${REFERENCE_RISK_LIMIT}"

SCALING_RANKS="$(normalise_list "${SCALING_RANKS}")"
CADENCE_WINDOWS_MS="$(normalise_list "${CADENCE_WINDOWS_MS}")"
required_ranks="${SCIENCE_RANKS}"
case "${EXPERIMENT}" in
    scaling) required_ranks="$(max_from_integer_list "${SCALING_RANKS}")" ;;
    pilot) required_ranks="${PILOT_RANKS}" ;;
    cadence) required_ranks="${CADENCE_RANKS}" ;;
    all)
        scaling_max="$(max_from_integer_list "${SCALING_RANKS}")"
        required_ranks="${SCIENCE_RANKS}"
        (( scaling_max > required_ranks )) && required_ranks="${scaling_max}"
        (( CADENCE_RANKS > required_ranks )) && required_ranks="${CADENCE_RANKS}"
        ;;
esac
if (( required_ranks > SLURM_NTASKS )); then
    echo "ERROR: requested up to ${required_ranks} ranks but allocation has ${SLURM_NTASKS}." >&2
    exit 2
fi

# Validate the shock time and every optional cadence value with Python so that
# a decimal window cannot be accidentally interpreted as an integer shell test.
python3 - "${DURATION_SECONDS}" "${SHOCK_TIME_SECONDS}" "${CADENCE_WINDOWS_MS}" <<'PY'
import math
import sys
duration = float(sys.argv[1])
try:
    shock = float(sys.argv[2])
except ValueError:
    raise SystemExit("ERROR: SHOCK_TIME_SECONDS must be numeric.")
if not math.isfinite(shock) or not 0.0 <= shock < duration:
    raise SystemExit("ERROR: SHOCK_TIME_SECONDS must be inside the simulated session.")
for value in sys.argv[3].split(","):
    try:
        window = float(value)
    except ValueError:
        raise SystemExit(f"ERROR: CADENCE_WINDOWS_MS contains {value!r}, not a number.")
    if not math.isfinite(window) or window <= 0.0:
        raise SystemExit(f"ERROR: invalid cadence window {value!r}.")
PY

if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/lmod.sh ]]; then
    source /etc/profile.d/lmod.sh
fi
if ! type module >/dev/null 2>&1; then
    echo "ERROR: the module command is unavailable on this compute allocation." >&2
    exit 2
fi
module purge
# Intentional splitting: this is a whitespace-separated module list.
module load ${SEAGULL_MODULES}
for command_name in cmake ninja mpicxx mpirun python3 ldd; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "ERROR: ${command_name} is unavailable after loading SEAGULL_MODULES." >&2
        exit 2
    fi
done

MPI_LIB_DIR="$(mpicxx --showme:libdirs | awk '{print $1}')"
if [[ -z "${MPI_LIB_DIR}" || ! -d "${MPI_LIB_DIR}" ]] \
    || ! compgen -G "${MPI_LIB_DIR}/libmpi.so*" >/dev/null; then
    echo "ERROR: could not determine a usable OpenMPI library directory." >&2
    exit 2
fi
export LD_LIBRARY_PATH="${MPI_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited

DETERMINISTIC_BUILD_CONTRACT="${PROJECT_DIR}/scripts/seagull_deterministic_build.sh"
if [[ ! -r "${DETERMINISTIC_BUILD_CONTRACT}" ]]; then
    echo "ERROR: missing deterministic build contract: ${DETERMINISTIC_BUILD_CONTRACT}" >&2
    exit 2
fi
# shellcheck source=scripts/seagull_deterministic_build.sh
source "${DETERMINISTIC_BUILD_CONTRACT}"

cd "${PROJECT_DIR}"
UNIVERSE_CONFIG="$(python3 - "${UNIVERSE_CONFIG}" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
)"
if [[ ! -f "${UNIVERSE_CONFIG}" ]]; then
    echo "ERROR: UNIVERSE_CONFIG is not a regular file: ${UNIVERSE_CONFIG}" >&2
    exit 2
fi
REQUESTED_UNIVERSE_CONFIG="${UNIVERSE_CONFIG}"
if [[ -n "${CALIBRATION_METADATA}" ]]; then
    CALIBRATION_METADATA="$(python3 - "${CALIBRATION_METADATA}" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
)"
    if [[ ! -f "${CALIBRATION_METADATA}" ]]; then
        echo "ERROR: CALIBRATION_METADATA is not a regular file: ${CALIBRATION_METADATA}" >&2
        exit 2
    fi
fi
HANDOFF_MODE="off"
CALIBRATED_POLICY_PATH=""
CALIBRATED_HAWKES_ACTIVITY_SCALE=""
CALIBRATED_LOCAL_MM_ENABLED=""
CALIBRATED_LOCAL_MM_INTERVAL_MS=""
CALIBRATED_LOCAL_MM_QUANTITY_MULTIPLIER=""
CALIBRATED_LOCAL_MM_IMPROVEMENT_PROBABILITY=""
CALIBRATED_SHARED_MM_SELECTED=""
CALIBRATED_SHARED_TREATMENT_MULTIPLIER=""
CALIBRATED_SHARED_QUOTE_LEVELS=""
CALIBRATED_DECISION_WINDOW_MS=""
CALIBRATED_CASE_CONFIG_PATH=""
CALIBRATED_CONFIG_SHA256=""
CALIBRATED_TRAINING_CONFIG_SHA256=""
CALIBRATION_REPORT_PATH=""
CALIBRATION_REPORT_SHA256=""
CALIBRATED_CLUSTER_PATH=""
CALIBRATED_CLUSTER_SHA256=""
CALIBRATION_ARTIFACT_ROLE=""
CALIBRATED_BINARY_SHA256=""
CALIBRATED_POLICY_SHA256=""
CALIBRATION_BUILD_PROVENANCE_PATH=""
CALIBRATION_BINARY_PATH=""
CALIBRATION_CERTIFICATION_PATH=""
CALIBRATION_CERTIFICATION_SHA256=""
INPUT_SNAPSHOT_MANIFEST=""
CERTIFICATION_RECHECK_TMP=""
cleanup_certification_recheck() {
    if [[ -n "${CERTIFICATION_RECHECK_TMP:-}" ]]; then
        rm -f -- "${CERTIFICATION_RECHECK_TMP}"
    fi
}
trap cleanup_certification_recheck EXIT
if [[ -n "${CALIBRATION_HANDOFF_JSON}" ]]; then
    CALIBRATION_HANDOFF_JSON="$(python3 - "${CALIBRATION_HANDOFF_JSON}" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
)"
    if [[ ! -f "${CALIBRATION_HANDOFF_JSON}" ]]; then
        echo "ERROR: CALIBRATION_HANDOFF_JSON is not a regular file: ${CALIBRATION_HANDOFF_JSON}" >&2
        exit 2
    fi
    # NUL-delimited output avoids evaluating JSON/path content as shell code
    # and preserves valid paths containing spaces.  Use `read -d` rather than
    # mapfile so the script also works with Bash 3.2 installations.
    if ! {
        IFS= read -r -d '' CALIBRATED_POLICY_PATH
        IFS= read -r -d '' CALIBRATED_HAWKES_ACTIVITY_SCALE
        IFS= read -r -d '' CALIBRATED_LOCAL_MM_ENABLED
        IFS= read -r -d '' CALIBRATED_LOCAL_MM_INTERVAL_MS
        IFS= read -r -d '' CALIBRATED_LOCAL_MM_QUANTITY_MULTIPLIER
        IFS= read -r -d '' CALIBRATED_LOCAL_MM_IMPROVEMENT_PROBABILITY
        IFS= read -r -d '' CALIBRATED_SHARED_MM_SELECTED
        IFS= read -r -d '' CALIBRATED_SHARED_TREATMENT_MULTIPLIER
        IFS= read -r -d '' CALIBRATED_SHARED_QUOTE_LEVELS
        IFS= read -r -d '' CALIBRATED_DECISION_WINDOW_MS
        IFS= read -r -d '' CALIBRATED_CASE_CONFIG_PATH
        IFS= read -r -d '' CALIBRATED_CONFIG_SHA256
        IFS= read -r -d '' CALIBRATED_TRAINING_CONFIG_SHA256
        IFS= read -r -d '' CALIBRATION_REPORT_PATH
        IFS= read -r -d '' CALIBRATION_REPORT_SHA256
        IFS= read -r -d '' CALIBRATED_CLUSTER_PATH
        IFS= read -r -d '' CALIBRATED_CLUSTER_SHA256
        IFS= read -r -d '' CALIBRATION_ARTIFACT_ROLE
        IFS= read -r -d '' CALIBRATED_BINARY_SHA256
        IFS= read -r -d '' CALIBRATED_POLICY_SHA256
        IFS= read -r -d '' CALIBRATION_BUILD_PROVENANCE_PATH
        IFS= read -r -d '' CALIBRATION_BINARY_PATH
        IFS= read -r -d '' CALIBRATION_CERTIFICATION_PATH
    } < <(python3 - \
        "${CALIBRATION_HANDOFF_JSON}" "${UNIVERSE_CONFIG}" "${VALUE_AGENT}" \
        "${VALUE_AGENT_POLICY_CSV}" "${SHOCK_CLUSTER_CSV}" \
        "${ALLOW_PRELIMINARY_MODEL}" "${PROJECT_DIR}" <<'PY'
import csv
import hashlib
import json
import math
import pathlib
import shutil
import stat
import statistics
import subprocess
import sys

handoff_path = pathlib.Path(sys.argv[1]).expanduser().resolve()
universe_path = pathlib.Path(sys.argv[2]).expanduser().resolve()
value_agent_mode = sys.argv[3]
manual_policy_text = sys.argv[4]
manual_shock_cluster_text = sys.argv[5]
allow_preliminary = sys.argv[6]
project_root = pathlib.Path(sys.argv[7]).expanduser().resolve()
sys.path.insert(0, str(project_root / "scripts"))
import calibrate_cluster_value_agents as calibration_contract
import certification_cohort as cohort_contract

def fail(message: str) -> None:
    raise SystemExit(f"ERROR: calibration handoff validation failed: {message}")

def regular_file(value: object, label: str) -> pathlib.Path:
    if not isinstance(value, str) or not value:
        fail(f"{label} must be a non-empty path string")
    path = pathlib.Path(value).expanduser().resolve()
    if not path.is_file():
        fail(f"{label} is not a regular file: {path}")
    return path

def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def expected_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        fail(f"{label} must be a SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError:
        fail(f"{label} is not hexadecimal")
    return value.lower()

def exact_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{label} must be a non-negative JSON integer")
    return value

def exact_positive_int(value: object, label: str) -> int:
    result = exact_nonnegative_int(value, label)
    if result == 0:
        fail(f"{label} must be positive")
    return result

def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{label} must be finite")
    return result

def require_true_fields(payload: dict[str, object], fields: tuple[str, ...],
                        label: str) -> None:
    for field in fields:
        if payload.get(field) is not True:
            fail(f"{label}.{field} must be true")

def require_empty_list(value: object, label: str) -> None:
    if value != []:
        fail(f"{label} must be an empty JSON list")

def validated_summary_paths(
    evaluation: dict[str, object], *, expected_seeds: tuple[int, ...],
    symbols: tuple[str, ...], status_path: pathlib.Path, label: str,
) -> tuple[pathlib.Path, ...]:
    """Bind an evaluation report to complete, fresh simulator CSV evidence."""
    if exact_positive_int(evaluation.get("seed_count"), f"{label}.seed_count") \
            != len(expected_seeds):
        fail(f"{label} has the wrong seed count")
    raw_paths = evaluation.get("summary_paths")
    if not isinstance(raw_paths, list) or len(raw_paths) != len(expected_seeds):
        fail(f"{label} must contain one summary path per declared seed")
    wall_seconds = evaluation.get("seed_wall_seconds")
    if not isinstance(wall_seconds, list) or len(wall_seconds) != len(expected_seeds):
        fail(f"{label} must contain one wall time per declared seed")
    for index, seconds in enumerate(wall_seconds):
        if finite_number(seconds, f"{label}.seed_wall_seconds[{index}]") < 0.0:
            fail(f"{label} contains a negative wall time")
    require_empty_list(evaluation.get("errors"), f"{label}.errors")
    require_empty_list(
        evaluation.get("two_sided_integrity_failures"),
        f"{label}.two_sided_integrity_failures",
    )
    require_true_fields(
        evaluation,
        (
            "two_sided_integrity_passed",
            "finite_boundary_adequacy_passed",
            "value_boundary_adequacy_passed",
        ),
        label,
    )

    status_mtime = status_path.stat().st_mtime_ns
    status_root = status_path.parent.resolve()
    paths: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for index, (raw_value, seed) in enumerate(zip(raw_paths, expected_seeds)):
        if not isinstance(raw_value, str) or not raw_value:
            fail(f"{label}.summary_paths[{index}] is not a path string")
        raw_path = pathlib.Path(raw_value).expanduser()
        if not raw_path.is_absolute():
            fail(f"{label} summary paths must be absolute")
        try:
            file_stat = raw_path.lstat()
        except OSError as error:
            fail(f"cannot stat {label} summary {raw_path}: {error}")
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            fail(f"{label} summary is not a direct regular file: {raw_path}")
        if file_stat.st_size <= 0:
            fail(f"{label} summary is empty: {raw_path}")
        path = raw_path.resolve()
        if str(path) != raw_value:
            fail(f"{label} summary path is not canonical: {raw_value}")
        if path in seen:
            fail(f"{label} reuses summary evidence: {path}")
        seen.add(path)
        if path.parent.name != f"seed_{seed}":
            fail(
                f"{label} summary {path} is not associated with declared seed {seed}"
            )
        if status_root not in path.parents:
            fail(f"{label} summary lies outside its certified result root: {path}")
        if file_stat.st_mtime_ns > status_mtime:
            fail(f"{label} summary was modified after its status JSON: {path}")
        try:
            calibration_contract.summary_rows(
                path, symbols, required_expected_sample_count=23_400,
            )
        except (OSError, ValueError, RuntimeError) as error:
            fail(f"{label} summary evidence is invalid: {error}")
        paths.append(path)
    return tuple(paths)

def recompute_seed_evaluation(
    persisted: object, *, expected_seeds: tuple[int, ...],
    symbols: tuple[str, ...], targets: dict[str, object],
    status_path: pathlib.Path, label: str,
) -> dict[str, object]:
    """Recompute every acceptance statistic from the referenced CSV outputs."""
    if not isinstance(persisted, dict):
        fail(f"{label} is not an evaluation object")
    paths = validated_summary_paths(
        persisted, expected_seeds=expected_seeds, symbols=symbols,
        status_path=status_path, label=label,
    )
    try:
        fit, estimates = calibration_contract.weighted_moment_loss(
            paths, targets, symbols,
            required_expected_sample_count=23_400,
        )
        combined, _ = calibration_contract.weighted_moment_loss(
            paths, targets, symbols, uncertainty_mode="combined",
            required_expected_sample_count=23_400,
        )
        selection_score, selection_metric_scores = (
            calibration_contract.metric_balanced_robust_loss(estimates)
        )
        two_sided_passed, two_sided_failures = (
            calibration_contract.two_sided_execution_integrity(
                paths, symbols, required_expected_sample_count=23_400,
            )
        )
        boundary = calibration_contract.finite_boundary_adequacy(
            paths, symbols, required_expected_sample_count=23_400,
        )
        value_boundary = calibration_contract.value_boundary_adequacy(
            paths, symbols, required_expected_sample_count=23_400,
        )
    except (OSError, ValueError, RuntimeError) as error:
        fail(f"cannot recompute {label}: {error}")
    recomputed: dict[str, object] = {
        "fit_wsmrmse": fit,
        "combined_uncertainty_wsmrmse": combined,
        "selection_score": selection_score,
        "selection_metric_scores": selection_metric_scores,
        "two_sided_integrity_passed": two_sided_passed,
        "two_sided_integrity_failures": two_sided_failures,
        "finite_boundary_adequacy_passed": boundary.get("passed") is True,
        "finite_boundary_adequacy": boundary,
        "value_boundary_adequacy_passed": value_boundary.get("passed") is True,
        "value_boundary_adequacy": value_boundary,
        "seed_wall_seconds": persisted["seed_wall_seconds"],
        "summary_paths": [str(path) for path in paths],
        "errors": [],
        "moment_estimates": [
            {
                field: getattr(estimate, field)
                for field in estimate.__dataclass_fields__
            }
            for estimate in estimates
        ],
    }
    expected_report = calibration_contract.evaluation_report(recomputed)
    if persisted != expected_report:
        fail(
            f"{label} does not equal the evaluation recomputed from its "
            "complete summary CSV evidence"
        )
    expected_pairs = {
        (symbol, metric)
        for symbol in symbols for metric in calibration_contract.METRICS
    }
    observed_pairs = {
        (str(row.get("symbol", "")), str(row.get("metric", "")))
        for row in persisted.get("moment_estimates", [])
        if isinstance(row, dict)
    }
    if observed_pairs != expected_pairs or len(observed_pairs) != len(
            persisted.get("moment_estimates", [])):
        fail(f"{label} lacks unique full symbol-by-metric moment coverage")
    return recomputed

def contract_csv(path: pathlib.Path, label: str,
                 *, allow_empty: bool = False) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                fail(f"{label} has no CSV header")
            fields = tuple(str(field).strip() for field in reader.fieldnames)
            if any(not field for field in fields) or len(set(fields)) != len(fields):
                fail(f"{label} has an invalid or duplicate CSV header")
            rows: list[dict[str, str]] = []
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    fail(f"{label}:{line_number} has too many columns")
                rows.append({field: str(row.get(field) or "").strip()
                             for field in fields})
    except OSError as error:
        fail(f"cannot read {label}: {error}")
    if not rows and not allow_empty:
        fail(f"{label} has no data rows")
    return fields, rows

def verify_policy_cluster_contract(
    *, policy_path: pathlib.Path, assignments_path: pathlib.Path,
    validation_path: pathlib.Path, symbols: tuple[str, ...],
    required_cluster_count: int, required_training_representatives: int,
    required_validation_symbols: int, minimum_cluster_size: int,
) -> dict[str, object]:
    """Verify policy/cluster/sample content, not only their recorded hashes."""
    expected_symbols = set(symbols)
    expected_cluster_ids = set(range(required_cluster_count))
    assignment_fields, assignment_rows = contract_csv(
        assignments_path, "cluster assignments"
    )
    required_assignment_fields = {"symbol", "cluster_id", "is_representative"}
    if not required_assignment_fields.issubset(assignment_fields):
        fail("cluster assignments lack symbol, cluster_id, or is_representative")
    membership: dict[str, int] = {}
    cluster_order: dict[int, list[tuple[float, str]]] = {
        cluster: [] for cluster in expected_cluster_ids
    }
    marked: dict[int, list[str]] = {cluster: [] for cluster in expected_cluster_ids}
    for line_number, row in enumerate(assignment_rows, start=2):
        try:
            symbol = calibration_contract.normalise_symbol(
                row["symbol"], label=f"cluster assignments:{line_number}",
            )
            cluster_id = calibration_contract.parse_cluster_id(
                row["cluster_id"], label=f"cluster assignments:{line_number}",
            )
            is_marked = calibration_contract.parse_bool(
                row["is_representative"],
                label=f"cluster assignments:{line_number}:is_representative",
            )
        except (ValueError, RuntimeError) as error:
            fail(f"invalid cluster assignment: {error}")
        if cluster_id not in expected_cluster_ids:
            fail(f"cluster assignment uses out-of-contract id {cluster_id}")
        if symbol in membership:
            fail(f"duplicate cluster assignment for {symbol}")
        membership[symbol] = cluster_id
        try:
            distance = float(row.get("distance_to_centroid", "nan"))
        except ValueError:
            distance = math.nan
        if not math.isfinite(distance):
            distance = float(line_number)
        cluster_order[cluster_id].append((distance, symbol))
        if is_marked:
            marked[cluster_id].append(symbol)
    if set(membership) != expected_symbols or len(membership) != len(symbols):
        fail("cluster assignments do not cover the training universe exactly once")
    observed_cluster_ids = set(membership.values())
    if observed_cluster_ids != expected_cluster_ids:
        fail("cluster assignments do not contain exactly the canonical cluster ids")
    for cluster_id in sorted(expected_cluster_ids):
        if len(cluster_order[cluster_id]) < minimum_cluster_size:
            fail(f"cluster {cluster_id} violates the certified minimum size")
        if len(marked[cluster_id]) != 1:
            fail(f"cluster {cluster_id} must have exactly one marked centroid")

    validation_fields, validation_rows = contract_csv(
        validation_path, "validation sample",
        allow_empty=(required_validation_symbols == 0),
    )
    if not {"symbol", "cluster_id"}.issubset(validation_fields):
        fail("validation sample lacks symbol or cluster_id")
    validation: dict[int, list[str]] = {
        cluster: [] for cluster in expected_cluster_ids
    }
    validation_seen: set[str] = set()
    for line_number, row in enumerate(validation_rows, start=2):
        try:
            symbol = calibration_contract.normalise_symbol(
                row["symbol"], label=f"validation sample:{line_number}",
            )
            cluster_id = calibration_contract.parse_cluster_id(
                row["cluster_id"], label=f"validation sample:{line_number}",
            )
        except (ValueError, RuntimeError) as error:
            fail(f"invalid validation-sample row: {error}")
        if symbol in validation_seen:
            fail(f"duplicate validation-sample symbol {symbol}")
        if symbol not in membership or membership[symbol] != cluster_id:
            fail(f"validation-sample cluster disagrees with assignments for {symbol}")
        if symbol in marked.get(cluster_id, []):
            fail(f"validation symbol {symbol} is also the marked representative")
        validation_seen.add(symbol)
        validation[cluster_id].append(symbol)
    if any(len(validation[cluster]) != required_validation_symbols
           for cluster in expected_cluster_ids):
        fail("validation sample must contain exactly three symbols per cluster")

    representatives: dict[int, tuple[str, ...]] = {}
    for cluster_id in sorted(expected_cluster_ids):
        validation_set = set(validation[cluster_id])
        ordered = [
            symbol for _, symbol in sorted(cluster_order[cluster_id])
            if symbol not in validation_set
        ]
        centroid = marked[cluster_id][0]
        ordered = [centroid] + [symbol for symbol in ordered if symbol != centroid]
        selected = tuple(ordered[:required_training_representatives])
        if len(selected) != required_training_representatives:
            fail(f"cluster {cluster_id} lacks exactly three training representatives")
        if set(selected).intersection(validation_set):
            fail(f"cluster {cluster_id} training and validation samples overlap")
        representatives[cluster_id] = selected

    policy_fields, policy_rows = contract_csv(policy_path, "value-agent policy")
    if policy_fields != tuple(calibration_contract.POLICY_FIELDS):
        fail("value-agent policy does not use the certified exact CSV schema")
    policy_by_symbol: dict[str, dict[str, object]] = {}
    policy_by_cluster: dict[int, tuple[bool, float, float]] = {}
    for line_number, row in enumerate(policy_rows, start=2):
        try:
            symbol = calibration_contract.normalise_symbol(
                row["symbol"], label=f"value-agent policy:{line_number}",
            )
            cluster_id = calibration_contract.parse_cluster_id(
                row["cluster_id"], label=f"value-agent policy:{line_number}",
            )
            enabled = calibration_contract.parse_bool(
                row["enabled"], label=f"value-agent policy:{line_number}:enabled",
            )
            threshold = calibration_contract.finite_float(
                row["value_threshold_bps"],
                label=f"value-agent policy:{line_number}:threshold",
            )
            depth = calibration_contract.finite_float(
                row["value_depth_participation"],
                label=f"value-agent policy:{line_number}:depth",
            )
        except (ValueError, RuntimeError) as error:
            fail(f"invalid value-agent policy row: {error}")
        if symbol in policy_by_symbol:
            fail(f"duplicate value-agent policy row for {symbol}")
        if symbol not in membership or membership[symbol] != cluster_id:
            fail(f"value-agent policy cluster disagrees with assignments for {symbol}")
        if row["cluster_label"] != f"liquidity_{cluster_id:02d}":
            fail(f"value-agent policy has the wrong cluster label for {symbol}")
        if row["policy_source"] != "selected_block_coordinate_cluster_wmm":
            fail(f"value-agent policy has an uncertified source for {symbol}")
        if enabled:
            if (threshold not in CANONICAL_CERTIFICATION_PROFILE[
                    "value_thresholds_bps"]
                    or depth not in CANONICAL_CERTIFICATION_PROFILE[
                        "value_depth_participations"]):
                fail(
                    f"enabled value-agent policy lies outside the certified "
                    f"candidate grid for {symbol}"
                )
        elif (not math.isclose(threshold, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
              or not math.isclose(depth, 0.25, rel_tol=0.0, abs_tol=1.0e-12)):
            fail(
                f"disabled value-agent policy is not the canonical baseline "
                f"for {symbol}"
            )
        cluster_policy = (enabled, threshold, depth)
        prior = policy_by_cluster.setdefault(cluster_id, cluster_policy)
        if prior != cluster_policy:
            fail(f"cluster {cluster_id} has inconsistent per-symbol policies")
        policy_by_symbol[symbol] = {
            "cluster_id": cluster_id, "enabled": enabled,
            "threshold_bps": threshold, "depth_participation": depth,
        }
    if set(policy_by_symbol) != expected_symbols or len(policy_by_symbol) != len(symbols):
        fail("value-agent policy does not cover the training universe exactly once")
    if set(policy_by_cluster) != expected_cluster_ids:
        fail("value-agent policy does not contain every certified cluster")
    return {
        "membership": membership,
        "representatives": representatives,
        "validation": {
            cluster: tuple(sorted(validation[cluster]))
            for cluster in expected_cluster_ids
        },
        "policy_by_cluster": policy_by_cluster,
    }

def source_semantics_digest(root: pathlib.Path) -> str:
    files = [root / "CMakeLists.txt"]
    files.extend(sorted((root / "include").rglob("*.hpp")))
    files.extend(sorted((root / "src").rglob("*.cpp")))
    if not files or any(not path.is_file() for path in files):
        fail(f"incomplete simulator source tree below {root}")
    value = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        value.update(len(relative).to_bytes(8, "big"))
        value.update(relative)
        value.update(len(content).to_bytes(8, "big"))
        value.update(content)
    return value.hexdigest()

WORKFLOW_SEMANTICS_FILES = (
    "submit_five_day_pooled_training.sh",
    "submit_cluster_value_agent_calibration.sh",
    "submit_real_universe_case_study.sh",
    "scripts/derive_hawkes_rates.py",
    "scripts/pool_multiday_empirical_universe.py",
    "scripts/cluster_empirical_universe.py",
    "scripts/intersect_empirical_universe_configs.py",
    "scripts/calibrate_cluster_value_agents.py",
    "scripts/run_fragmented_mpi_experiments.py",
    "scripts/analyze_capacity_pilot.py",
    "scripts/analyze_fragmented_shared_liquidity_case.py",
    "scripts/seagull_deterministic_build.sh",
    "scripts/certification_cohort.py",
    "scripts/verify_global_calibration_certification.py",
    "tests/test_global_calibration_certification_verifier.py",
    "config/certification_symbols_1480.txt",
    "config/certification_symbols_1480_origin.json",
)

def workflow_semantics_digest(root: pathlib.Path) -> str:
    files = [root / relative for relative in WORKFLOW_SEMANTICS_FILES]
    if any(not path.is_file() for path in files):
        fail(f"incomplete workflow source tree below {root}")
    value = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        value.update(len(relative).to_bytes(8, "big"))
        value.update(relative)
        value.update(len(content).to_bytes(8, "big"))
        value.update(content)
    return value.hexdigest()

SIMULATOR_EMPIRICAL_INPUT_FILENAMES = (
    "limit_buy_quantity_distribution.txt",
    "limit_sell_quantity_distribution.txt",
    "market_buy_quantity_distribution.txt",
    "market_sell_quantity_distribution.txt",
    "cancel_bid_quantity_distribution.txt",
    "cancel_ask_quantity_distribution.txt",
    "limit_buy_distance_distribution.txt",
    "limit_sell_distance_distribution.txt",
    "cancel_bid_distance_distribution.txt",
    "cancel_ask_distance_distribution.txt",
)

def empirical_bundle_digest(config: pathlib.Path) -> str:
    try:
        with config.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    except OSError as error:
        fail(f"cannot read empirical bundle configuration: {error}")
    if not rows:
        fail("empirical bundle configuration is empty")
    value = hashlib.sha256()

    def add_file(identity: str, path: pathlib.Path) -> None:
        if not path.is_file():
            fail(f"empirical input is not a regular file: {path}")
        name = identity.encode("utf-8")
        content = path.read_bytes()
        value.update(len(name).to_bytes(8, "big"))
        value.update(name)
        value.update(len(content).to_bytes(8, "big"))
        value.update(content)

    try:
        ordered = sorted(rows, key=lambda row: int(row["book_id"]))
    except (KeyError, TypeError, ValueError):
        fail("empirical bundle configuration has invalid book_id values")
    for row in ordered:
        book_id = int(row["book_id"])
        symbol = str(row.get("symbol", "")).strip().upper()
        data_dir = pathlib.Path(str(row.get("data_dir", ""))).expanduser()
        rates = pathlib.Path(str(row.get("hawkes_rates_file", ""))).expanduser()
        if not symbol or not data_dir.is_absolute() or not rates.is_absolute():
            fail(f"empirical input paths/symbol are invalid at book {book_id}")
        data_dir = data_dir.resolve()
        rates = rates.resolve()
        add_file(f"{book_id}:{symbol}:hawkes_rates", rates)
        for filename in SIMULATOR_EMPIRICAL_INPUT_FILENAMES:
            add_file(f"{book_id}:{symbol}:mark:{filename}", data_dir / filename)
        manifests = sorted(data_dir.glob("itch_manifest_*.json"))
        if len(manifests) != 1:
            fail(
                f"{data_dir} needs exactly one ITCH manifest for provenance; "
                f"found {len(manifests)}"
            )
        add_file(f"{book_id}:{symbol}:manifest", manifests[0])
    return value.hexdigest()

CANONICAL_CERTIFICATION_PROFILE = {
    "profile_id": "development_validation_gate_v18",
    "certification_profile_enforced": True,
    "validation_role": "development_validation_after_protocol_revision",
    "independent_final_holdout": False,
    "required_session_duration_seconds": 23400,
    "required_training_dates": [
        "2019-01-30", "2019-03-27", "2019-07-30", "2019-10-30", "2019-12-30"
    ],
    "required_validation_date": "2020-01-30",
    "required_common_symbol_count": 1480,
    "required_common_symbol_order_sha256": (
        "2f57f37762772d9523fb9916fe2376a9578e337d20971fe39aa44d578f5691d3"
    ),
    "cohort_identity": {
        "cohort_file": "config/certification_symbols_1480.txt",
        "cohort_symbol_count": 1480,
        "cohort_symbol_order_sha256": (
            "2f57f37762772d9523fb9916fe2376a9578e337d20971fe39aa44d578f5691d3"
        ),
        "canonical_order": "QQQ_first_then_lexicographic",
        "origin_manifest": "config/certification_symbols_1480_origin.json",
        "selection_role": "development_validation_balanced_panel",
        "heldout_availability_conditioned": True,
        "heldout_target_values_used": False,
        "independent_final_holdout": False,
        "original_intersection_symbol_count": 1509,
        "fixed_price_grid_excluded_symbol_count": 29,
        "final_symbol_count": 1480,
        "r16_pooled_training_universe_csv_sha256": (
            "13fb1700643f408787708190d7af752d5bd7e107d1009e6b4f6a686c0dc155ef"
        ),
        "r16_pooling_provenance_sha256": (
            "ab908a56b5962f946c7f7fd4f2906876b1497a62f428960bfb7f71352032edca"
        ),
        "interpretation": (
            "fixed development-validation balanced panel conditioned on "
            "symbol availability and opening-price-grid compatibility on "
            "2020-01-30; no held-out target value entered cohort selection"
        ),
    },
    "required_cluster_count": 10,
    "empirical_target_session": {
        "session_start": "09:30:00",
        "session_end": "16:00:00",
        "duration_seconds": 23400,
        "snapshot_interval_ms": 1000,
        "full_session_observations": 23400,
    },
    "required_training_representatives_per_cluster": 3,
    "required_validation_symbols_per_cluster": 3,
    "stage1_duration_seconds": 300,
    "stage2_duration_seconds": 3600,
    "asset_summary_interval_ms": 1000,
    "required_stage1_seeds": [1729],
    "required_stage2_seeds": [1729, 7919],
    "required_stage3_seeds": [1729, 7919, 1103, 6599, 2027],
    "shared_quote_candidate_count": 4,
    "shared_quote_stage1_survivor_cap": 6,
    "shared_quote_stage1_promoted_candidates": 4,
    "local_flow_stage1_refinement_leaders": 6,
    "stage1_refinement_candidates": 32,
    "shared_quote_stage2_survivor_cap": 2,
    "shared_quote_stage2_promoted_candidates": 2,
    "shared_quote_stage3_survivor_cap": 1,
    "shared_quote_stage3_promoted_candidates": 1,
    "local_flow_stage1_promotion": "all_structurally_eligible",
    "local_flow_stage2_promotion": "all_structurally_eligible",
    "local_flow_stage3_selection": (
        "best_training_fit_among_structurally_eligible"
    ),
    "value_policy_stage1_promotion": (
        "all_structurally_eligible_threshold_depth_policies_plus_disabled_baseline"
    ),
    "value_policy_stage2_promotion": (
        "all_structurally_eligible_threshold_depth_policies_plus_disabled_baseline"
    ),
    "value_policy_stage1_survivors_per_depth": 6,
    "value_policy_stage2_survivors_per_depth": 6,
    "value_policy_stage3_candidates_per_cluster": 25,
    "structural_preflight": {
        "required_candidate_roles": [
            "background_only",
            "enabled_local_mm_reference",
        ],
        "duration_seconds": 3600,
        "seeds": [1729, 7919],
        "empirical_admissibility_metrics": [
            "mean_bid_depth",
            "mean_ask_depth",
        ],
        "maximum_robust_score": 2.0,
        "maximum_metric_score": 3.0,
        "maximum_symbol_metric_absolute_robust_residual": 6.0,
        "gross_symbol_metric_failures_role": (
            "diagnostic_only_during_structural_preflight"
        ),
        "zero_gross_symbol_metric_failures_required": False,
        "strict_gross_symbol_gate_retained_for_development_validation": False,
        "two_sided_integrity_required": True,
        "finite_boundary_adequacy_required": True,
        "both_candidates_must_pass": True,
        "spread_excluded_because_local_mm_is_spread_repair": True,
        "training_targets_only": True,
    },
    "value_thresholds_bps": [5.0, 8.0, 10.0, 15.0, 25.0, 40.0],
    "value_depth_participations": [0.05, 0.1, 0.25, 0.5],
    "hawkes_activity_scales": [0.3],
    "local_mm_intervals_ms": [500.0, 1000.0, 2000.0],
    "local_mm_quantity_multipliers": [0.5, 1.0, 2.0],
    "local_mm_improvement_probabilities": [0.0, 0.25, 0.5, 1.0],
    "shared_quote_multipliers": [0.5, 1.0, 2.0],
    "shared_treatment_multiplier": 1.0,
    "background_event_rate_acceptance_required": True,
    "build_provenance_required": True,
    "workflow_source_semantics_required": True,
    "clustering_protocol": {
        "algorithm": (
            "deterministic_farthest_first_lloyd_kmeans_"
            "with_minimum_size_repair"
        ),
        "seed": 20200130,
        "minimum_cluster_size": 6,
        "features": [
            "event_rate_per_second", "mean_spread_ticks", "mean_top_depth",
            "return_variance", "opening_mid_price_ticks",
        ],
    },
    "pooling_protocol": {
        "activity_scale": 0.3,
        "hawkes_beta": 10.0,
        "balance_directional_volume": True,
        "balance_best_depth": True,
        "balance_strength": 1.0,
        "excitation_structure": "diagonal_self_excitation_only",
        "self_excitation_amplitude": 0.20,
        "cross_excitation_amplitude": 0.0,
        "simulator_tick_size_price_units": 100,
        "minimum_opening_bid_price_units": 10000,
        "minimum_common_symbols": 20,
        "quote_quantity_fraction": 0.5,
        "minimum_quote_quantity": 10,
        "maximum_quote_quantity": 1000,
        "pool_label": "five_2019_sessions",
        "runtime_configuration_schema_version": 5,
        "runtime_configuration_schema_sha256": (
            calibration_contract.configuration_schema_sha256(
                calibration_contract.RUNTIME_CONFIG_FIELDS
            )
        ),
        "pooled_homeostatic_fields": [
            "target_spread_ticks", "target_mean_bid_depth",
            "target_mean_ask_depth",
        ],
        "latent_value_fields": [
            "fundamental_volatility_bps_sqrt_second",
            "fundamental_move_probability_per_second",
            "fundamental_conditional_kurtosis",
        ],
        "frozen_training_derived_fields": [
            "target_spread_ticks", "target_mean_bid_depth",
            "target_mean_ask_depth",
            "fundamental_volatility_bps_sqrt_second",
            "fundamental_move_probability_per_second",
            "fundamental_conditional_kurtosis",
        ],
        "heldout_target_files_used_for_runtime_configuration": False,
    },
    "marketwide_validation_required": True,
    "heldout_validation_acceptance_protocol": {
        "authoritative_empirical_fit_scope": "full_universe_marketwide",
        "stratified": {
            "required": True,
            "scope": "three_nonrepresentative_symbols_per_cluster",
            "required_symbol_count": 30,
            "execution_integrity_required": True,
            "two_sided_clock_required": True,
            "empirical_coverage_required": True,
            "background_boundary_adequacy_required": True,
            "value_boundary_adequacy_required": True,
            "empirical_fit_computation_required": True,
            "empirical_fit_acceptance_role": (
                "required_reported_diagnostic_only"
            ),
        },
        "marketwide": {
            "required": True,
            "scope": "all_1480_common_symbols",
            "required_symbol_count": 1480,
            "execution_integrity_required": True,
            "two_sided_clock_required": True,
            "background_boundary_adequacy_required": True,
            "value_boundary_adequacy_required": True,
            "empirical_fit_computation_required": True,
            "empirical_fit_acceptance_role": (
                "authoritative_certification_gate"
            ),
            "maximum_robust_score": 2.0,
            "maximum_metric_score": 3.0,
            "maximum_symbol_metric_absolute_robust_residual": 6.0,
        },
        "heldout_information_used_for_selection": False,
        "thresholds_changed_from_v17": False,
        "seeds_changed_from_v17": False,
    },
    "model_semantics": {
        "local_market_maker": (
            "owned_queue_and_spread_reactive_one_tick_limit_quotes"
        ),
        "value_agent": (
            "contrarian_market_order_protected_at_perceived_fundamental_"
            "and_sized_as_a_cluster_calibrated_fraction_of_displayed_"
            "opposite_side_depth_"
            "against_rank_independent_sparse_"
            "training_moment_latent_value"
        ),
        "finite_book_reserve": "final_displayed_share_not_owner_zero_share",
    },
    "nested_policy_selection": {
        "disabled_baseline_promoted_through_stage2": True,
        "each_depth_participation_stratum_promoted_through_stage2": True,
        "all_threshold_depth_policies_promoted_through_stage2": True,
        "complete_grid_eligibility_required_at_each_stage": True,
        "full_day_selection": "best_training_fit_among_eligible_candidates",
        "heldout_information_used_for_selection": False,
    },
    "full_universe_training_adequacy": {
        "required_before_development_validation": True,
        "scope": "all_common_symbols_on_every_training_date",
        "required_training_dates": [
            "2019-01-30", "2019-03-27", "2019-07-30",
            "2019-10-30", "2019-12-30",
        ],
        "session_duration_seconds": 23400,
        "seeds": [
            3424815697, 1799108475, 2301941028,
            3637917665, 3007455382,
        ],
        "seed_derivation": (
            "first_32_bits_sha256(development_validation_gate_v17:"
            "training_adequacy:{i}), i=0,...,4"
        ),
        "seed_set_inherited_from_profile_id": (
            "development_validation_gate_v17"
        ),
        "every_training_day_must_pass": True,
        "maximum_aggregate_robust_score": 2.0,
        "maximum_day_robust_score": 2.0,
        "maximum_day_metric_score": 3.0,
        "two_sided_integrity_required": True,
        "finite_boundary_adequacy_required": True,
        "development_validation_targets_opened": False,
    },
    "gross_symbol_metric_failures_role": (
        "diagnostic_outliers_under_cluster_level_calibration"
    ),
    "gross_symbol_metric_failures_required_for_acceptance": False,
    "simulated_two_sided_fraction_required": 1.0,
    "maximum_robust_score": 2.0,
    "maximum_metric_score": 3.0,
    "maximum_symbol_metric_absolute_robust_residual": 6.0,
    "maximum_two_sided_shortfall_diagnostic": 0.01,
    "finite_boundary_adequacy": {
        "model_adequacy_gate": True,
        "source_attribution_required": True,
        "background_gate_scope": (
            "per_symbol_pooled_across_predeclared_seeds_and_"
            "market_aggregate_pooled_across_symbols_and_seeds"
        ),
        "maximum_asset_event_ratio": 0.05,
        "maximum_asset_quantity_ratio": 0.05,
        "maximum_run_event_ratio": 0.01,
        "maximum_run_quantity_ratio": 0.01,
        "event_ratio": (
            "background_boundary_truncation_events / background_event_count"
        ),
        "quantity_ratio": (
            "background_boundary_truncated_quantity / "
            "(background_market_requested_quantity + "
            "background_cancel_requested_quantity)"
        ),
        "value_event_ratio": (
            "value_boundary_truncation_events / value_order_count"
        ),
        "value_quantity_ratio": (
            "value_boundary_truncated_quantity / value_requested_quantity"
        ),
        "development_validation_sources_required": [
            "background", "value",
        ],
        "per_seed_ratios_role": "diagnostic_only",
        "zero_denominator_rule": (
            "passes only when the corresponding numerator is zero"
        ),
    },
    "cluster_training_finite_boundary_adequacy": {
        "scope": (
            "value_agent_source_only; per_symbol_date_and_cluster_"
            "candidate_pooled_across_predeclared_stage_seeds"
        ),
        "reason": (
            "the background and local-flow model is frozen by block one; "
            "block two must not divide value-agent boundary events by "
            "background denominators or reclassify a three-symbol cluster "
            "as a market-wide aggregate"
        ),
        "maximum_symbol_date_event_ratio": 0.05,
        "maximum_symbol_date_quantity_ratio": 0.05,
        "maximum_cluster_candidate_event_ratio": 0.05,
        "maximum_cluster_candidate_quantity_ratio": 0.05,
        "per_seed_ratios_role": "diagnostic_only",
        "background_boundary_role": "diagnostic_frozen_block_one",
        "development_validation_requires_background_and_value_gates": True,
    },
    "stochastic_stream_identity": "stable_hash_of_symbol_not_subset_book_id",
}

def canonical_profile_digest() -> str:
    encoded = json.dumps(
        CANONICAL_CERTIFICATION_PROFILE,
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def positive_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        fail(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError):
        fail(f"{label} must be numeric")
    if not math.isfinite(result) or result <= 0.0:
        fail(f"{label} must be finite and positive")
    return result

def nonnegative_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        fail(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError):
        fail(f"{label} must be numeric")
    if not math.isfinite(result) or result < 0.0:
        fail(f"{label} must be finite and non-negative")
    return result

def probability(value: object, label: str) -> float:
    result = nonnegative_float(value, label)
    if result > 1.0:
        fail(f"{label} must be at most one")
    return result

def positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        fail(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError):
        fail(f"{label} must be an integer")
    if result <= 0 or str(result) != str(value):
        fail(f"{label} must be a positive integer")
    return result

def boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        fail(f"{label} must be a JSON boolean")
    return value

HAWKES_EVENT_TYPES = (
    "limit_buy", "limit_sell", "market_buy", "market_sell",
    "cancel_bid", "cancel_ask",
)

def verified_artifact_record(
    value: object, label: str,
) -> tuple[pathlib.Path, str]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    path = regular_file(value.get("path"), f"{label}.path")
    recorded_hash = expected_digest(value.get("sha256"), f"{label}.sha256")
    if digest(path) != recorded_hash:
        fail(f"{label} changed after rate derivation")
    return path, recorded_hash

def verify_rate_derivation(
    value: object, label: str, expected_settings: dict[str, object],
) -> tuple[pathlib.Path, str, pathlib.Path, str]:
    if not isinstance(value, dict):
        fail(f"{label} lacks rate_derivation")
    if value.get("schema_version") != 1 or value.get("status") != "passed":
        fail(f"{label} rate derivation was not successfully audited")
    if positive_int(
            value.get("event_types_checked"),
            f"{label}.rate_derivation.event_types_checked",
        ) != len(HAWKES_EVENT_TYPES):
        fail(f"{label} rate derivation does not cover all six event types")
    if value.get("stationary_reconstruction_equals_target_per_type") is not True:
        fail(f"{label} does not certify stationary target reconstruction")
    if value.get("observed_rates_equal_manifest_counts_per_duration") is not True:
        fail(f"{label} does not certify manifest-derived observed rates")
    if value.get("stationary_targets_equal_declared_transforms_per_type") is not True:
        fail(f"{label} does not certify the declared reduced-book transforms")
    if value.get(
            "reported_reconstruction_equals_configured_rate_equation_per_type"
        ) is not True:
        fail(f"{label} does not certify its configured Hawkes-rate equation")
    positive_int(
        value.get("manifest_duration_seconds"),
        f"{label}.rate_derivation.manifest_duration_seconds",
    )
    relative_tolerance = nonnegative_float(
        value.get("relative_tolerance"),
        f"{label}.rate_derivation.relative_tolerance",
    )
    absolute_tolerance = nonnegative_float(
        value.get("absolute_tolerance"),
        f"{label}.rate_derivation.absolute_tolerance",
    )
    if relative_tolerance != 1.0e-12 or absolute_tolerance != 1.0e-12:
        fail(f"{label} rate derivation uses unsupported tolerances")
    if value.get("transform_settings") != expected_settings:
        fail(f"{label} rate derivation has noncanonical transform settings")
    recorded_maximum_error = nonnegative_float(
        value.get("maximum_absolute_stationary_reconstruction_error"),
        f"{label}.rate_derivation.maximum_error",
    )
    recorded_observed_error = nonnegative_float(
        value.get("maximum_absolute_observed_rate_error"),
        f"{label}.rate_derivation.maximum_observed_rate_error",
    )
    recorded_target_error = nonnegative_float(
        value.get("maximum_absolute_stationary_target_error"),
        f"{label}.rate_derivation.maximum_stationary_target_error",
    )
    recorded_reported_reconstruction_error = nonnegative_float(
        value.get("maximum_absolute_reported_reconstruction_error"),
        f"{label}.rate_derivation.maximum_reported_reconstruction_error",
    )
    if (recorded_observed_error > absolute_tolerance
            or recorded_target_error > absolute_tolerance):
        fail(f"{label} independent rate-derivation audit exceeds tolerance")
    manifest_path, manifest_hash = verified_artifact_record(
        value.get("manifest"), f"{label}.rate_derivation.manifest",
    )
    rate_path, rate_hash = verified_artifact_record(
        value.get("generated_hawkes_rates"),
        f"{label}.rate_derivation.generated_hawkes_rates",
    )
    try:
        with rate_path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    except OSError as error:
        fail(f"cannot read {label} generated Hawkes rates: {error}")
    if [row.get("event_type") for row in rows] != list(HAWKES_EVENT_TYPES):
        fail(f"{label} generated Hawkes rates have the wrong event order")
    target_rates = [
        nonnegative_float(
            row.get("stationary_target_rate"),
            f"{label}.{row.get('event_type')}.stationary_target_rate",
        )
        for row in rows
    ]
    observed_maximum_error = 0.0
    observed_reported_reconstruction_error = 0.0
    for index, row in enumerate(rows):
        event = str(row["event_type"])
        for field in (
            "observed_rate_per_second", "stationary_target_rate",
            "configured_mu", "stationary_reconstructed_rate",
        ):
            nonnegative_float(
                row.get(field), f"{label}.{event}.{field}"
            )
        target = target_rates[index]
        configured_mu = float(row["configured_mu"])
        reconstructed = float(row["stationary_reconstructed_rate"])
        if expected_settings["cross_excitation_amplitude"] != 0.0:
            fail(f"{label} requests unsupported cross excitation")
        endogenous = (
            expected_settings["self_excitation_amplitude"] * target
            / expected_settings["kernel_beta"]
        )
        computed_reconstruction = (
            expected_settings["activity_scale"] * configured_mu + endogenous
        )
        observed_reported_reconstruction_error = max(
            observed_reported_reconstruction_error,
            abs(reconstructed - computed_reconstruction),
        )
        observed_maximum_error = max(
            observed_maximum_error, abs(computed_reconstruction - target)
        )
        if not math.isclose(
                reconstructed, computed_reconstruction,
                rel_tol=relative_tolerance, abs_tol=absolute_tolerance):
            fail(
                f"{label} reported reconstruction disagrees with configured_mu "
                f"for {event}"
            )
        if not math.isclose(
                computed_reconstruction, target,
                rel_tol=relative_tolerance, abs_tol=absolute_tolerance):
            fail(
                f"{label} generated Hawkes rates do not reconstruct the "
                f"stationary target for {event}"
            )
    if not math.isclose(
            recorded_maximum_error, observed_maximum_error,
            rel_tol=1.0e-12, abs_tol=1.0e-15):
        fail(f"{label} recorded rate-derivation error disagrees with its CSV")
    if not math.isclose(
            recorded_reported_reconstruction_error,
            observed_reported_reconstruction_error,
            rel_tol=1.0e-12, abs_tol=1.0e-15):
        fail(f"{label} reported reconstruction audit disagrees with its CSV")
    return manifest_path, manifest_hash, rate_path, rate_hash

try:
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    fail(f"cannot read JSON from {handoff_path}: {error}")
if not isinstance(handoff, dict) or handoff.get("schema_version") != 1:
    fail("unsupported handoff schema_version (expected 1)")
certification = handoff.get("certification")
if not isinstance(certification, dict):
    fail("artifact lacks explicit calibration certification")
execution_passed = boolean(
    certification.get("execution_integrity_passed"),
    "certification.execution_integrity_passed",
)
full_two_sided_passed = boolean(
    certification.get("full_two_sided_book_passed"),
    "certification.full_two_sided_book_passed",
)
coverage_passed = boolean(
    certification.get("coverage_passed"), "certification.coverage_passed",
)
boundary_adequacy_passed = boolean(
    certification.get("finite_boundary_adequacy_passed"),
    "certification.finite_boundary_adequacy_passed",
)
empirical_passed = boolean(
    certification.get("empirical_fit_passed"),
    "certification.empirical_fit_passed",
)
if certification.get("empirical_fit_acceptance_scope") != (
        "full_universe_marketwide"):
    fail("certification empirical-fit acceptance scope is not market-wide")
stratified_structural_passed = boolean(
    certification.get("stratified_structural_adequacy_passed"),
    "certification.stratified_structural_adequacy_passed",
)
stratified_empirical_passed = boolean(
    certification.get("stratified_empirical_fit_passed"),
    "certification.stratified_empirical_fit_passed",
)
if certification.get("stratified_empirical_fit_acceptance_role") != (
        "required_reported_diagnostic_only"):
    fail("certification gives the stratified fit an unsupported acceptance role")
stratified_empirical_failure_reasons = certification.get(
    "stratified_empirical_fit_failure_reasons"
)
if (not isinstance(stratified_empirical_failure_reasons, list)
        or not all(isinstance(value, str)
                   for value in stratified_empirical_failure_reasons)):
    fail("certification stratified empirical-fit failures are malformed")
marketwide_empirical_passed = boolean(
    certification.get("marketwide_empirical_fit_passed"),
    "certification.marketwide_empirical_fit_passed",
)
if certification.get("marketwide_empirical_fit_acceptance_role") != (
        "authoritative_certification_gate"):
    fail("certification does not make the market-wide fit authoritative")
if empirical_passed is not marketwide_empirical_passed:
    fail("global empirical-fit result disagrees with the authoritative market-wide fit")
provenance_passed = boolean(
    certification.get("provenance_integrity_passed"),
    "certification.provenance_integrity_passed",
)
certified = boolean(
    certification.get("certified_for_case_study"),
    "certification.certified_for_case_study",
)
runtime_profile_matched = boolean(
    certification.get("runtime_matches_certification_profile"),
    "certification.runtime_matches_certification_profile",
)
marketwide_completed = boolean(
    certification.get("marketwide_validation_completed"),
    "certification.marketwide_validation_completed",
)
training_adequacy_passed = boolean(
    certification.get("training_full_universe_adequacy_passed"),
    "certification.training_full_universe_adequacy_passed",
)
complete_clock_passed = boolean(
    certification.get("complete_two_sided_clock_passed"),
    "certification.complete_two_sided_clock_passed",
)
if certification.get("validation_role") != CANONICAL_CERTIFICATION_PROFILE["validation_role"]:
    fail("certification has an unsupported validation_role")
if certification.get("independent_final_holdout") is not False:
    fail("2020 development validation must not be labelled an independent final holdout")
handoff_cohort_identity = handoff.get("cohort_identity")
try:
    cohort_contract.require_identity_record(
        handoff_cohort_identity, label="handoff.cohort_identity",
    )
except cohort_contract.CohortIdentityError as error:
    fail(str(error))
if (certification.get("cohort_identity_verified") is not True
        or certification.get("cohort_identity") != handoff_cohort_identity):
    fail("certification does not bind the exact handoff cohort identity")
profile = handoff.get("certification_profile")
if profile != CANONICAL_CERTIFICATION_PROFILE:
    fail("artifact does not use the immutable case-study certification profile")
if handoff.get("observed_runtime_profile") != CANONICAL_CERTIFICATION_PROFILE:
    fail("calibration search/runtime does not match the canonical profile")
observed_survivors = handoff.get("observed_survivor_counts")
if not isinstance(observed_survivors, dict):
    fail("handoff lacks observed survivor counts")
observed_shared = observed_survivors.get("global_shared_quote")
if not isinstance(observed_shared, dict):
    fail("handoff lacks global shared-quote survivor counts")
shared_stage_contract = (
    (
        "stage1_screen",
        CANONICAL_CERTIFICATION_PROFILE[
            "shared_quote_stage1_survivor_cap"
        ],
        CANONICAL_CERTIFICATION_PROFILE[
            "shared_quote_stage1_promoted_candidates"
        ],
        CANONICAL_CERTIFICATION_PROFILE["shared_quote_candidate_count"],
    ),
    (
        "stage2_refinement",
        CANONICAL_CERTIFICATION_PROFILE[
            "shared_quote_stage2_survivor_cap"
        ],
        CANONICAL_CERTIFICATION_PROFILE[
            "shared_quote_stage2_promoted_candidates"
        ],
        CANONICAL_CERTIFICATION_PROFILE[
            "shared_quote_stage1_promoted_candidates"
        ],
    ),
    (
        "stage3_full",
        CANONICAL_CERTIFICATION_PROFILE[
            "shared_quote_stage3_survivor_cap"
        ],
        CANONICAL_CERTIFICATION_PROFILE[
            "shared_quote_stage3_promoted_candidates"
        ],
        CANONICAL_CERTIFICATION_PROFILE[
            "shared_quote_stage2_promoted_candidates"
        ],
    ),
)
for stage, expected_cap, expected_promoted, expected_evaluated in (
        shared_stage_contract):
    counts = observed_shared.get(stage)
    if not isinstance(counts, dict):
        fail(f"handoff lacks shared-quote {stage} counts")
    observed_cap = positive_int(
        counts.get("configured_ranked_survivor_count"),
        f"observed shared-quote {stage} survivor cap",
    )
    evaluated = positive_int(
        counts.get("evaluated_candidates"),
        f"observed shared-quote {stage} evaluated candidates",
    )
    eligible = positive_int(
        counts.get("eligible_candidates"),
        f"observed shared-quote {stage} eligible candidates",
    )
    promoted = positive_int(
        counts.get("promoted_candidates"),
        f"observed shared-quote {stage} promoted candidates",
    )
    if (observed_cap != expected_cap
            or evaluated != expected_evaluated
            or promoted != expected_promoted
            or not promoted <= eligible <= evaluated):
        fail(
            f"shared-quote {stage} counts violate the canonical "
            f"cap/trajectory: observed cap={observed_cap}, "
            f"evaluated={evaluated}, eligible={eligible}, promoted={promoted}"
        )
profile_hash = expected_digest(
    handoff.get("certification_profile_sha256"),
    "certification_profile_sha256",
)
if profile_hash != canonical_profile_digest():
    fail("certification profile SHA-256 is not the canonical case-study gate")
expected_runtime_schema = {
    "schema_version": 5,
    "fields": list(calibration_contract.RUNTIME_CONFIG_FIELDS),
    "sha256": calibration_contract.configuration_schema_sha256(
        calibration_contract.RUNTIME_CONFIG_FIELDS
    ),
    "pooled_homeostatic_fields": list(
        calibration_contract.POOLED_HOMEOSTATIC_FIELDS
    ),
    "latent_value_fields": list(calibration_contract.LATENT_VALUE_FIELDS),
    "frozen_training_derived_fields": list(
        calibration_contract.FROZEN_TRAINING_DERIVED_FIELDS
    ),
    "heldout_target_files_used": False,
}
if handoff.get("runtime_configuration_schema") != expected_runtime_schema:
    fail(
        "handoff lacks the certified queue-reactive runtime configuration schema"
    )
if certification.get("certification_profile_id") != profile["profile_id"]:
    fail("certification profile id disagrees with the handoff profile")
if certification.get("certification_profile_sha256") != profile_hash:
    fail("certification profile hash disagrees with the handoff profile")
if not runtime_profile_matched:
    fail("calibration runtime does not match the immutable case-study profile")
if not marketwide_completed:
    fail("full-universe development validation is required for any case-study run")
if not training_adequacy_passed:
    fail("full-universe training adequacy is required before a case-study run")
if complete_clock_passed != coverage_passed:
    fail("complete two-sided clock flag disagrees with coverage compatibility flag")
if execution_passed and not full_two_sided_passed:
    fail("execution integrity cannot pass when a book becomes one-sided")
all_checks = (
    execution_passed and full_two_sided_passed
    and complete_clock_passed and coverage_passed and empirical_passed
    and stratified_structural_passed
    and boundary_adequacy_passed
    and runtime_profile_matched and training_adequacy_passed
    and marketwide_completed and provenance_passed
)
if certified != all_checks:
    fail("certified_for_case_study disagrees with component checks")
if not certified and allow_preliminary != "on":
    fail(
        "calibration is preliminary and failed held-out certification; set "
        "ALLOW_PRELIMINARY_MODEL=on only for an explicitly labelled diagnostic run"
    )
if certified and handoff.get("artifact_role") != "certified_calibration_handoff":
    fail("certified artifact has the wrong artifact_role")
if not certified and handoff.get("artifact_role") != "preliminary_not_certified":
    fail("uncertified artifact is not clearly labelled preliminary")

report_path = regular_file(handoff.get("calibration_report"), "calibration_report")
report_hash = expected_digest(
    handoff.get("calibration_report_sha256"), "calibration_report_sha256"
)
if digest(report_path) != report_hash:
    fail("calibration report SHA-256 does not match the handoff")
training_adequacy = handoff.get("full_universe_training_adequacy")
if not isinstance(training_adequacy, dict):
    fail("handoff lacks full-universe training adequacy provenance")
required_common_symbols = CANONICAL_CERTIFICATION_PROFILE[
    "required_common_symbol_count"
]
if training_adequacy.get("symbols") != required_common_symbols:
    fail(
        "full-universe training adequacy does not cover exactly "
        f"{required_common_symbols} symbols"
    )
training_gate = CANONICAL_CERTIFICATION_PROFILE[
    "full_universe_training_adequacy"
]
if (training_adequacy.get("training_dates")
        != CANONICAL_CERTIFICATION_PROFILE["required_training_dates"]
        or training_adequacy.get("duration_seconds")
            != training_gate["session_duration_seconds"]
        or training_adequacy.get("seeds") != training_gate["seeds"]):
    fail(
        "full-universe training adequacy has the wrong dates, duration, "
        "or independent seeds"
    )
if boolean(
    training_adequacy.get("passed"),
    "full_universe_training_adequacy.passed",
) is not training_adequacy_passed:
    fail("training adequacy provenance disagrees with certification")
training_adequacy_status = regular_file(
    training_adequacy.get("status_json"),
    "full_universe_training_adequacy.status_json",
)
training_adequacy_status_hash = expected_digest(
    training_adequacy.get("status_sha256"),
    "full_universe_training_adequacy.status_sha256",
)
if digest(training_adequacy_status) != training_adequacy_status_hash:
    fail("full-universe training adequacy status SHA-256 does not match")
try:
    training_status_payload = json.loads(
        training_adequacy_status.read_text(encoding="utf-8")
    )
except (OSError, json.JSONDecodeError) as error:
    fail(f"cannot read training adequacy status: {error}")
if not isinstance(training_status_payload, dict):
    fail("training adequacy status is not a JSON object")
if training_status_payload.get("schema_version") != 1:
    fail("training adequacy status has an unsupported schema")
if training_status_payload.get("scope") != (
        "all_common_symbols_on_every_training_date"):
    fail("training adequacy status has the wrong evaluation scope")
if training_status_payload.get("passed") is not True:
    fail("persisted full-universe training adequacy status did not pass")
if (training_status_payload.get("symbol_count") != required_common_symbols
        or training_status_payload.get("required_symbol_count")
            != required_common_symbols):
    fail(
        "persisted full-universe training adequacy status has the wrong "
        "symbol cardinality"
    )
if (training_status_payload.get("training_dates")
        != CANONICAL_CERTIFICATION_PROFILE["required_training_dates"]
        or training_status_payload.get("duration_seconds")
            != training_gate["session_duration_seconds"]
        or training_status_payload.get("seeds") != training_gate["seeds"]):
    fail(
        "persisted full-universe training adequacy status has the wrong "
        "dates, duration, or independent seeds"
    )
if training_status_payload.get("development_validation_targets_opened") is not False:
    fail("training adequacy status does not preserve the held-out leakage barrier")
if training_status_payload.get("cohort_identity") != handoff_cohort_identity:
    fail("training adequacy status does not bind the exact cohort identity")
require_true_fields(
    training_status_payload,
    (
        "selection_parameters_frozen_before_evaluation",
        "aggregate_selection_score_passed",
        "every_training_day_empirical_fit_passed",
        "execution_integrity_passed",
        "finite_boundary_adequacy_passed",
        "value_boundary_adequacy_passed",
    ),
    "training adequacy status",
)
if training_status_payload.get("training_day_count") != len(
        CANONICAL_CERTIFICATION_PROFILE["required_training_dates"]):
    fail("training adequacy status has the wrong dated-evaluation count")
require_empty_list(
    training_status_payload.get("failure_reasons"),
    "training adequacy status.failure_reasons",
)
marketwide_adequacy = handoff.get("heldout_marketwide_validation")
if not isinstance(marketwide_adequacy, dict):
    fail("handoff lacks held-out market-wide validation provenance")
if (marketwide_adequacy.get("passed") is not True
        or marketwide_adequacy.get("symbols") != required_common_symbols
        or marketwide_adequacy.get("validation_date")
            != CANONICAL_CERTIFICATION_PROFILE["required_validation_date"]
        or marketwide_adequacy.get("duration_seconds")
            != CANONICAL_CERTIFICATION_PROFILE[
                "required_session_duration_seconds"
            ]
        or marketwide_adequacy.get("seeds")
            != CANONICAL_CERTIFICATION_PROFILE["required_stage3_seeds"]):
    fail(
        "held-out market-wide validation has the wrong pass state, symbol "
        "count, date, duration, or seeds"
    )
if marketwide_adequacy.get("empirical_fit_acceptance_role") != (
        "authoritative_certification_gate"):
    fail("held-out market-wide provenance is not the authoritative fit gate")
marketwide_status = regular_file(
    marketwide_adequacy.get("status_json"),
    "heldout_marketwide_validation.status_json",
)
marketwide_status_hash = expected_digest(
    marketwide_adequacy.get("status_sha256"),
    "heldout_marketwide_validation.status_sha256",
)
if digest(marketwide_status) != marketwide_status_hash:
    fail("held-out market-wide validation status SHA-256 does not match")
try:
    marketwide_status_payload = json.loads(
        marketwide_status.read_text(encoding="utf-8")
    )
except (OSError, json.JSONDecodeError) as error:
    fail(f"cannot read held-out market-wide validation status: {error}")
if (not isinstance(marketwide_status_payload, dict)
        or marketwide_status_payload.get("schema_version") != 2
        or marketwide_status_payload.get("scope") != "full_universe_marketwide"
        or marketwide_status_payload.get("passed") is not True
        or marketwide_status_payload.get("symbol_count")
            != required_common_symbols
        or marketwide_status_payload.get("required_symbol_count")
            != required_common_symbols
        or marketwide_status_payload.get("validation_date")
            != CANONICAL_CERTIFICATION_PROFILE["required_validation_date"]
        or marketwide_status_payload.get("duration_seconds")
            != CANONICAL_CERTIFICATION_PROFILE[
                "required_session_duration_seconds"
            ]
        or marketwide_status_payload.get("seeds")
            != CANONICAL_CERTIFICATION_PROFILE["required_stage3_seeds"]):
    fail(
        "persisted held-out market-wide validation status has the wrong "
        "pass state, symbol count, date, duration, or seeds"
    )
if marketwide_status_payload.get("cohort_identity") != handoff_cohort_identity:
    fail("market-wide validation status does not bind the exact cohort identity")
if marketwide_status_payload.get("empirical_fit_acceptance_role") != (
        "authoritative_certification_gate"):
    fail("market-wide status is not the authoritative empirical-fit gate")
require_true_fields(
    marketwide_status_payload,
    (
        "execution_integrity_passed",
        "full_two_sided_book_passed",
        "coverage_passed",
        "structural_adequacy_passed",
        "finite_boundary_adequacy_passed",
        "background_boundary_adequacy_passed",
        "value_boundary_adequacy_passed",
        "empirical_fit_passed",
        "certified_for_case_study",
    ),
    "held-out market-wide validation status",
)
require_empty_list(
    marketwide_status_payload.get("failure_reasons"),
    "held-out market-wide validation status.failure_reasons",
)
calibration_semantics_hash = expected_digest(
    handoff.get("simulator_source_semantics_sha256"),
    "simulator_source_semantics_sha256",
)
if source_semantics_digest(project_root) != calibration_semantics_hash:
    fail(
        "current simulator source semantics differ from the source calibrated "
        "and validated by this artifact"
    )
calibration_workflow_hash = expected_digest(
    handoff.get("workflow_source_semantics_sha256"),
    "workflow_source_semantics_sha256",
)
if workflow_semantics_digest(project_root) != calibration_workflow_hash:
    fail(
        "current calibration/analysis workflow differs from the workflow "
        "that produced this artifact"
    )
calibration_binary_hash = expected_digest(
    handoff.get("calibration_binary_sha256"), "calibration_binary_sha256",
)
build_provenance = handoff.get("calibration_build_provenance")
if not isinstance(build_provenance, dict):
    fail("handoff lacks calibration build provenance")
build_path = regular_file(build_provenance.get("path"), "build provenance path")
build_hash = expected_digest(
    build_provenance.get("sha256"), "build provenance SHA-256",
)
if digest(build_path) != build_hash:
    fail("calibration build-provenance file changed after fitting")
try:
    persisted_build = json.loads(build_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    fail(f"cannot parse calibration build provenance: {error}")
calibration_binary_path = regular_file(
    persisted_build.get("binary"), "calibration executable",
)
if digest(calibration_binary_path) != calibration_binary_hash:
    fail("persisted calibration executable changed after fitting")
for key in (
    "artifact_role", "cmake_build_type", "binary", "binary_sha256",
    "simulator_source_semantics_sha256", "workflow_source_semantics_sha256",
    "compiler", "mpi", "deterministic_build_contract",
):
    if persisted_build.get(key) != build_provenance.get(key):
        fail(f"embedded build provenance disagrees for {key}")
if build_provenance.get("binary_sha256") != calibration_binary_hash:
    fail("build provenance calibration-binary hash disagrees with handoff")
if build_provenance.get("simulator_source_semantics_sha256") != calibration_semantics_hash:
    fail("build provenance simulator-source hash disagrees with handoff")
if build_provenance.get("workflow_source_semantics_sha256") != calibration_workflow_hash:
    fail("build provenance workflow-source hash disagrees with handoff")
build_contract = build_provenance.get("deterministic_build_contract")
current_contract_path = (
    project_root / "scripts" / "seagull_deterministic_build.sh"
).resolve()
if not isinstance(build_contract, dict):
    fail("build provenance lacks deterministic build contract")
current_compiler = shutil.which("mpicxx")
current_ninja = shutil.which("ninja")
if current_compiler is None or current_ninja is None:
    fail("deterministic build tools disappeared after module loading")
current_mpi_lib = subprocess.check_output(
    [current_compiler, "--showme:libdirs"], text=True
).split()[0]
expected_build_contract = {
    "version": "seagull_release_mpi_v1",
    "path": str(current_contract_path),
    "sha256": digest(current_contract_path),
    "compiler_path": str(pathlib.Path(current_compiler).resolve()),
    "ninja_path": str(pathlib.Path(current_ninja).resolve()),
    "mpi_lib_dir": str(pathlib.Path(current_mpi_lib).resolve()),
    "source_date_epoch": "1577836800",
    "cmake_build_type": "Release",
    "lob_require_mpi": True,
    "lob_build_tests": True,
    "interprocedural_optimization": False,
}
if build_contract != expected_build_contract:
    fail(
        "current case-study toolchain/build contract differs from calibration"
    )
training_path = regular_file(
    handoff.get("training_universe_config"), "training_universe_config"
)
training_hash = expected_digest(
    handoff.get("training_universe_config_sha256"),
    "training_universe_config_sha256",
)
if digest(training_path) != training_hash:
    fail("training-universe SHA-256 does not match the handoff")
try:
    training_fields_checked, training_identity_rows = (
        calibration_contract.load_universe_config(training_path)
    )
except (OSError, ValueError, RuntimeError) as error:
    fail(f"cannot read training universe for target verification: {error}")
if tuple(training_fields_checked) != tuple(calibration_contract.RUNTIME_CONFIG_FIELDS):
    fail("training universe does not use certified runtime schema version 5")
training_symbols = tuple(
    str(row.get("symbol", "")).strip().upper()
    for row in training_identity_rows
)
if (not training_symbols or any(not symbol for symbol in training_symbols)
        or len(set(training_symbols)) != len(training_symbols)):
    fail("training universe has invalid symbols for target verification")
if len(training_symbols) != required_common_symbols:
    fail(
        "certified training universe must contain exactly "
        f"{required_common_symbols} symbols; observed {len(training_symbols)}"
    )
try:
    observed_training_cohort = cohort_contract.validate_csv(
        training_path,
        label="certified training universe",
        project_root=project_root,
    )
except cohort_contract.CohortIdentityError as error:
    fail(str(error))
if observed_training_cohort.get("symbol_order_sha256") != (
        CANONICAL_CERTIFICATION_PROFILE["required_common_symbol_order_sha256"]):
    fail("training universe cohort digest disagrees with the canonical profile")
pooled_homeostatic_targets = {
    row["symbol"]: tuple(
        row[field]
        for field in calibration_contract.FROZEN_TRAINING_DERIVED_FIELDS
    )
    for row in training_identity_rows
}
verified_training_days: list[calibration_contract.TrainingDay] = []
training_days = handoff.get("training_days")
if not isinstance(training_days, list) or len(training_days) != 5:
    fail("handoff does not contain the five canonical training-day records")
if [entry.get("date") if isinstance(entry, dict) else None for entry in training_days] \
        != CANONICAL_CERTIFICATION_PROFILE["required_training_dates"]:
    fail("handoff training dates do not match the canonical profile")
for entry in training_days:
    if not isinstance(entry, dict):
        fail("handoff training-day provenance is malformed")
    daily_config = regular_file(entry.get("universe_config"), "daily training config")
    daily_config_hash = expected_digest(
        entry.get("universe_config_sha256"), "daily training config SHA-256",
    )
    if digest(daily_config) != daily_config_hash:
        fail("a daily training configuration changed after calibration")
    try:
        daily_fields, daily_rows = calibration_contract.load_universe_config(
            daily_config
        )
    except (OSError, ValueError, RuntimeError) as error:
        fail(f"cannot validate a daily runtime configuration: {error}")
    if tuple(daily_fields) != tuple(calibration_contract.RUNTIME_CONFIG_FIELDS):
        fail("a daily training configuration has an unsupported schema")
    try:
        cohort_contract.validate_symbols(
            (row["symbol"] for row in daily_rows),
            label=f"daily training universe {entry['date']}",
            project_root=project_root,
        )
    except cohort_contract.CohortIdentityError as error:
        fail(str(error))
    daily_targets = {
        row["symbol"]: tuple(
            row[field]
            for field in calibration_contract.FROZEN_TRAINING_DERIVED_FIELDS
        )
        for row in daily_rows
    }
    if daily_targets != pooled_homeostatic_targets:
        fail(
            "a daily runtime does not use the frozen pooled "
            "spread/depth/latent-value targets"
        )
    daily_bundle_hash = expected_digest(
        entry.get("empirical_input_bundle_sha256"),
        "daily empirical-input bundle SHA-256",
    )
    if empirical_bundle_digest(daily_config) != daily_bundle_hash:
        fail("a daily training empirical-input bundle changed after calibration")
    recorded_target_hash = expected_digest(
        entry.get("target_artifact_bundle_sha256"),
        "daily target-artifact bundle SHA-256",
    )
    target_root = pathlib.Path(str(entry.get("target_root", ""))).expanduser().resolve()
    if not target_root.is_dir():
        fail(f"daily target root is not a directory: {target_root}")
    try:
        observed_target_hash = calibration_contract.target_artifact_bundle_sha256(
            target_root, str(entry["date"]), training_symbols,
            (300, 3600, None),
        )
    except (OSError, ValueError, RuntimeError) as error:
        fail(f"cannot recompute daily target bundle: {error}")
    if observed_target_hash != recorded_target_hash:
        fail("a daily target-artifact bundle changed after calibration")
    verified_training_days.append(calibration_contract.TrainingDay(
        date=str(entry["date"]),
        universe_config=daily_config,
        target_root=target_root,
        fields=tuple(daily_fields),
        rows=tuple(dict(row) for row in daily_rows),
        universe_config_sha256=daily_config_hash,
    ))
development_target_hash = expected_digest(
    handoff.get("development_validation_target_bundle_sha256"),
    "development_validation_target_bundle_sha256",
)
development_date = handoff.get("development_validation_date")
if development_date != CANONICAL_CERTIFICATION_PROFILE["required_validation_date"]:
    fail("handoff has the wrong development-validation date")
development_target_root = pathlib.Path(
    str(handoff.get("development_validation_target_root", ""))
).expanduser().resolve()
if not development_target_root.is_dir():
    fail("development-validation target root is not a directory")
try:
    observed_development_target_hash = (
        calibration_contract.target_artifact_bundle_sha256(
            development_target_root, str(development_date), training_symbols,
            (None,),
        )
    )
except (OSError, ValueError, RuntimeError) as error:
    fail(f"cannot recompute development-validation target bundle: {error}")
if observed_development_target_hash != development_target_hash:
    fail("development-validation target-artifact bundle changed after calibration")

# A JSON boolean is not evidence that 1,480 books were actually evaluated.
# Re-open all 30 declared full-day seed summaries, verify their exact symbol
# and fixed-clock coverage, and recompute every WMM and structural-adequacy
# statistic before accepting the status artifacts.
training_day_records = training_status_payload.get(
    "evaluation", {}
).get("training_day_evaluations") if isinstance(
    training_status_payload.get("evaluation"), dict
) else None
if not isinstance(training_day_records, list):
    fail("training adequacy status lacks dated evaluation evidence")
expected_training_dates = tuple(
    CANONICAL_CERTIFICATION_PROFILE["required_training_dates"]
)
if tuple(
    record.get("date") if isinstance(record, dict) else None
    for record in training_day_records
) != expected_training_dates:
    fail("training adequacy evidence does not contain the five exact dates")
training_evaluations: list[
    tuple[calibration_contract.TrainingDay, dict[str, object]]
] = []
all_training_summaries: set[pathlib.Path] = set()
training_seeds = tuple(training_gate["seeds"])
for training_day, record in zip(verified_training_days, training_day_records):
    if not isinstance(record, dict):
        fail(f"training adequacy evidence for {training_day.date} is malformed")
    try:
        day_targets = calibration_contract.load_targets(
            training_day.target_root, training_day.date, training_symbols,
        )
    except (OSError, ValueError, RuntimeError) as error:
        fail(f"cannot load certified training targets for {training_day.date}: {error}")
    recomputed_day = recompute_seed_evaluation(
        record.get("evaluation"), expected_seeds=training_seeds,
        symbols=training_symbols, targets=day_targets,
        status_path=training_adequacy_status,
        label=f"full-universe training {training_day.date}",
    )
    day_paths = {
        pathlib.Path(value).resolve()
        for value in recomputed_day["summary_paths"]
    }
    if all_training_summaries.intersection(day_paths):
        fail("training adequacy reuses a seed summary across dates")
    all_training_summaries.update(day_paths)
    training_evaluations.append((training_day, recomputed_day))
try:
    recomputed_training = calibration_contract.aggregate_training_day_evaluations(
        training_evaluations, seed_count=len(training_seeds),
    )
    recomputed_training_report = calibration_contract.evaluation_report(
        recomputed_training
    )
except (OSError, ValueError, RuntimeError) as error:
    fail(f"cannot recompute full-universe training adequacy: {error}")
if training_status_payload.get("evaluation") != recomputed_training_report:
    fail(
        "full-universe training aggregate does not equal the result "
        "recomputed from all five dates and 25 seed summaries"
    )
try:
    recomputed_training_gate = (
        calibration_contract.full_universe_training_adequacy_summary(
            recomputed_training,
            maximum_score=CANONICAL_CERTIFICATION_PROFILE[
                "maximum_robust_score"
            ],
            maximum_metric_score=CANONICAL_CERTIFICATION_PROFILE[
                "maximum_metric_score"
            ],
            maximum_symbol_metric_absolute_residual=(
                CANONICAL_CERTIFICATION_PROFILE[
                    "maximum_symbol_metric_absolute_robust_residual"
                ]
            ),
        )
    )
except (OSError, ValueError, RuntimeError) as error:
    fail(f"cannot recompute the full-universe training gate: {error}")
for key, value in recomputed_training_gate.items():
    if training_status_payload.get(key) != value:
        fail(f"training adequacy status disagrees with recomputed {key}")
if recomputed_training_gate.get("passed") is not True:
    fail("recomputed full-universe training adequacy did not pass")

try:
    marketwide_targets = calibration_contract.load_targets(
        development_target_root, str(development_date), training_symbols,
    )
except (OSError, ValueError, RuntimeError) as error:
    fail(f"cannot load certified development-validation targets: {error}")
recomputed_marketwide = recompute_seed_evaluation(
    marketwide_status_payload.get("evaluation"),
    expected_seeds=tuple(CANONICAL_CERTIFICATION_PROFILE["required_stage3_seeds"]),
    symbols=training_symbols, targets=marketwide_targets,
    status_path=marketwide_status,
    label="held-out full-universe market-wide validation",
)
try:
    recomputed_marketwide_coverage = (
        calibration_contract.two_sided_coverage_summary(
            recomputed_marketwide,
            CANONICAL_CERTIFICATION_PROFILE[
                "maximum_two_sided_shortfall_diagnostic"
            ],
        )
    )
    recomputed_marketwide_fit = calibration_contract.empirical_fit_summary(
        recomputed_marketwide,
        maximum_score=CANONICAL_CERTIFICATION_PROFILE["maximum_robust_score"],
        maximum_metric_score=CANONICAL_CERTIFICATION_PROFILE[
            "maximum_metric_score"
        ],
        maximum_symbol_metric_absolute_residual=(
            CANONICAL_CERTIFICATION_PROFILE[
                "maximum_symbol_metric_absolute_robust_residual"
            ]
        ),
    )
except (OSError, ValueError, RuntimeError) as error:
    fail(f"cannot recompute the held-out market-wide gates: {error}")
if marketwide_status_payload.get("coverage_summary") != (
        recomputed_marketwide_coverage):
    fail("market-wide coverage summary disagrees with its seed CSV evidence")
if marketwide_status_payload.get("empirical_fit") != recomputed_marketwide_fit:
    fail("market-wide empirical-fit summary disagrees with its seed CSV evidence")
if certification.get("marketwide_empirical_fit") != recomputed_marketwide_fit:
    fail("global certification disagrees with recomputed market-wide fit")
if marketwide_empirical_passed is not (
        recomputed_marketwide_fit.get("passed") is True):
    fail("global market-wide empirical-fit flag disagrees with recomputed evidence")
expected_marketwide_boundary = {
    "background": recomputed_marketwide["finite_boundary_adequacy"],
    "value": recomputed_marketwide["value_boundary_adequacy"],
}
if marketwide_status_payload.get("finite_boundary_adequacy") != (
        expected_marketwide_boundary):
    fail("market-wide boundary status disagrees with its seed CSV evidence")
if (recomputed_marketwide.get("two_sided_integrity_passed") is not True
        or recomputed_marketwide.get("finite_boundary_adequacy_passed") is not True
        or recomputed_marketwide.get("value_boundary_adequacy_passed") is not True
        or recomputed_marketwide_fit.get("passed") is not True):
    fail("recomputed held-out full-universe validation did not pass")
pool_record = handoff.get("pooling_provenance")
if not isinstance(pool_record, dict):
    fail("handoff lacks pooling provenance")
pool_path = regular_file(pool_record.get("path"), "pooling provenance")
pool_hash = expected_digest(pool_record.get("sha256"), "pooling provenance SHA-256")
if digest(pool_path) != pool_hash:
    fail("pooling provenance changed after calibration")
try:
    persisted_pool = json.loads(pool_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    fail(f"cannot parse pooling provenance: {error}")
if not isinstance(persisted_pool, dict):
    fail("pooling provenance is not an object")
producer_source_verification = pool_record.get("producer_source_verification")
if not isinstance(producer_source_verification, dict):
    fail("embedded pooling provenance lacks producer-source verification")
embedded_pool = {key: value for key, value in pool_record.items()
                 if key not in {
                     "path", "sha256", "producer_source_verification",
                 }}
if persisted_pool != embedded_pool:
    fail("embedded pooling provenance disagrees with its source artifact")
producer_root = pathlib.Path(str(
    producer_source_verification.get("producer_project_root", "")
)).expanduser().resolve()
try:
    observed_producer_verification = (
        calibration_contract.validate_pooling_producer_workflow_source(
            persisted_pool,
            producer_project_root=producer_root,
            consumer_project_root=project_root,
        )
    )
except (calibration_contract.CalibrationError, OSError, ValueError) as error:
    fail(f"cannot verify pooling producer source: {error}")
if observed_producer_verification != producer_source_verification:
    fail("embedded pooling producer-source verification disagrees with source trees")
if (producer_source_verification.get(
        "consumer_workflow_source_semantics_sha256")
        != calibration_workflow_hash):
    fail("pool consumer workflow hash disagrees with the calibrated workflow")
if persisted_pool.get("schema_version") != 7:
    fail("pooling provenance does not use the audited rate-derivation schema")
expected_pool_schema = {
    "schema_version": 5,
    "source_fields": list(calibration_contract.BASE_CONFIG_FIELDS),
    "runtime_fields": list(calibration_contract.RUNTIME_CONFIG_FIELDS),
    "runtime_fields_sha256": calibration_contract.configuration_schema_sha256(
        calibration_contract.RUNTIME_CONFIG_FIELDS
    ),
    "pooled_homeostatic_fields": list(
        calibration_contract.POOLED_HOMEOSTATIC_FIELDS
    ),
    "latent_value_fields": list(calibration_contract.LATENT_VALUE_FIELDS),
    "frozen_training_derived_fields": list(
        calibration_contract.FROZEN_TRAINING_DERIVED_FIELDS
    ),
    "queue_reactive_target_fields": list(
        calibration_contract.QUEUE_REACTIVE_TARGET_FIELDS
    ),
    "positive_queue_reactive_targets_required": True,
    "same_pooled_targets_in_all_runtime_sessions": True,
    "heldout_target_files_used": False,
}
if persisted_pool.get("configuration_schema") != expected_pool_schema:
    fail("pooling provenance has an unsupported runtime configuration schema")
if persisted_pool.get("quote_improvement_runtime_approximation") != (
        calibration_contract.QUOTE_IMPROVEMENT_RUNTIME_APPROXIMATION):
    fail(
        "pooling provenance lacks the fail-closed quote-improvement "
        "compatibility preflight"
    )
pooling_method = persisted_pool.get("pooling")
if (not isinstance(pooling_method, dict)
        or pooling_method.get(
            "heldout_targets_used_for_runtime_configuration"
        ) is not False):
    fail("pooling provenance does not enforce the held-out target barrier")
pooling_profile = CANONICAL_CERTIFICATION_PROFILE["pooling_protocol"]
expected_hawkes = {
    "activity_scale": pooling_profile["activity_scale"],
    "kernel_beta": pooling_profile["hawkes_beta"],
    "balance_directional_volume": pooling_profile[
        "balance_directional_volume"
    ],
    "balance_best_depth": pooling_profile["balance_best_depth"],
    "balance_strength": pooling_profile["balance_strength"],
    "excitation_structure": pooling_profile["excitation_structure"],
    "self_excitation_amplitude": pooling_profile[
        "self_excitation_amplitude"
    ],
    "cross_excitation_amplitude": pooling_profile[
        "cross_excitation_amplitude"
    ],
}
if pooling_method.get("hawkes") != expected_hawkes:
    fail(
        "pooling provenance does not certify the canonical balanced "
        "reduced-book event-rate settings"
    )
opening_eligibility = persisted_pool.get("opening_price_grid_eligibility")
if (not isinstance(opening_eligibility, dict)
        or opening_eligibility.get("simulator_tick_size_price_units")
            != pooling_profile["simulator_tick_size_price_units"]
        or opening_eligibility.get("minimum_opening_bid_price_units")
            != pooling_profile["minimum_opening_bid_price_units"]):
    fail("pooling provenance has noncanonical opening price-grid settings")
expected_pooling_parameters = {
    "minimum_common_symbols": pooling_profile["minimum_common_symbols"],
    "quote_quantity_fraction": pooling_profile["quote_quantity_fraction"],
    "minimum_quote_quantity": pooling_profile["minimum_quote_quantity"],
    "maximum_quote_quantity": pooling_profile["maximum_quote_quantity"],
    "pool_label": pooling_profile["pool_label"],
}
if persisted_pool.get("pooling_parameters") != expected_pooling_parameters:
    fail("pooling provenance has noncanonical immutable pooling parameters")
pool_symbols = persisted_pool.get("symbols")
if (not isinstance(pool_symbols, list)
        or [
            str(item.get("symbol", "")).strip().upper()
            if isinstance(item, dict) else ""
            for item in pool_symbols
        ] != list(training_symbols)):
    fail("pooling provenance rate audits do not cover the training universe")
for symbol_record in pool_symbols:
    symbol = str(symbol_record["symbol"]).strip().upper()
    label = f"pooling symbol {symbol}"
    (
        pooled_manifest_path, _pooled_manifest_hash,
        pooled_rate_path, pooled_rate_hash,
    ) = verify_rate_derivation(
        symbol_record.get("rate_derivation"), label, expected_hawkes,
    )
    if pathlib.Path(str(symbol_record.get("pooled_manifest", ""))).resolve() \
            != pooled_manifest_path:
        fail(f"{label} manifest path disagrees with its rate derivation")
    outer_pooled_rate = regular_file(
        symbol_record.get("pooled_hawkes_rates"),
        f"{label}.pooled_hawkes_rates",
    )
    outer_pooled_hash = expected_digest(
        symbol_record.get("pooled_hawkes_rates_sha256"),
        f"{label}.pooled_hawkes_rates_sha256",
    )
    if (outer_pooled_rate != pooled_rate_path
            or outer_pooled_hash != pooled_rate_hash):
        fail(f"{label} pooled rate path/hash disagrees with its audit")
    source_records = symbol_record.get("sources")
    if (not isinstance(source_records, list)
            or [
                item.get("trading_date") if isinstance(item, dict) else None
                for item in source_records
            ] != CANONICAL_CERTIFICATION_PROFILE["required_training_dates"]):
        fail(f"{label} lacks five ordered daily rate-derivation records")
    for source_record in source_records:
        day = str(source_record["trading_date"])
        daily_label = f"{label} {day}"
        (
            daily_manifest_path, daily_manifest_hash,
            daily_rate_path, daily_rate_hash,
        ) = verify_rate_derivation(
            source_record.get("rate_derivation"),
            daily_label, expected_hawkes,
        )
        outer_manifest = regular_file(
            source_record.get("manifest"), f"{daily_label}.manifest",
        )
        outer_manifest_hash = expected_digest(
            source_record.get("manifest_sha256"),
            f"{daily_label}.manifest_sha256",
        )
        outer_generated_rate = regular_file(
            source_record.get("generated_hawkes_rates"),
            f"{daily_label}.generated_hawkes_rates",
        )
        outer_generated_hash = expected_digest(
            source_record.get("generated_hawkes_rates_sha256"),
            f"{daily_label}.generated_hawkes_rates_sha256",
        )
        source_rate = regular_file(
            source_record.get("source_hawkes_rates"),
            f"{daily_label}.source_hawkes_rates",
        )
        source_rate_hash = expected_digest(
            source_record.get("source_hawkes_rates_sha256"),
            f"{daily_label}.source_hawkes_rates_sha256",
        )
        if digest(source_rate) != source_rate_hash:
            fail(f"{daily_label} source Hawkes rates changed after pooling")
        if (outer_manifest != daily_manifest_path
                or outer_manifest_hash != daily_manifest_hash
                or outer_generated_rate != daily_rate_path
                or outer_generated_hash != daily_rate_hash):
            fail(f"{daily_label} outer rate provenance disagrees with its audit")
pooled_record = persisted_pool.get("pooled_configuration")
if not isinstance(pooled_record, dict):
    fail("pooling provenance lacks pooled_configuration")
if (pathlib.Path(str(pooled_record.get("path", ""))).expanduser().resolve()
        != training_path or pooled_record.get("sha256") != training_hash):
    fail("pooling provenance does not identify the certified training universe")
pool_training_days = persisted_pool.get("training_days")
if not isinstance(pool_training_days, list) or len(pool_training_days) != 5:
    fail("pooling provenance lacks five training-day records")
for handoff_day, pool_day in zip(training_days, pool_training_days):
    if not isinstance(pool_day, dict):
        fail("pooling training-day record is malformed")
    if (pool_day.get("date") != handoff_day.get("date")
            or pool_day.get("common_config_sha256")
                != handoff_day.get("universe_config_sha256")
            or pathlib.Path(str(pool_day.get("target_root", ""))).expanduser().resolve()
                != pathlib.Path(str(handoff_day.get("target_root", ""))).expanduser().resolve()):
        fail("pooling and calibration training-day provenance disagree")
frozen_path_from_handoff = regular_file(
    handoff.get("frozen_heldout_opening_config"),
    "frozen_heldout_opening_config",
)
frozen_hash_from_handoff = expected_digest(
    handoff.get("frozen_heldout_opening_config_sha256"),
    "frozen_heldout_opening_config_sha256",
)
if digest(frozen_path_from_handoff) != frozen_hash_from_handoff:
    fail("frozen held-out opening configuration changed after validation")
frozen_bundle_hash = expected_digest(
    handoff.get("frozen_empirical_input_bundle_sha256"),
    "frozen_empirical_input_bundle_sha256",
)
if empirical_bundle_digest(frozen_path_from_handoff) != frozen_bundle_hash:
    fail("frozen empirical-input bundle changed after validation")
if not universe_path.is_file():
    fail(f"UNIVERSE_CONFIG is not a regular file: {universe_path}")
requested_universe_hash = digest(universe_path)

policy_path = regular_file(handoff.get("value_agent_policy_csv"), "value_agent_policy_csv")
policy_hash = expected_digest(
    handoff.get("value_agent_policy_sha256"), "value_agent_policy_sha256"
)
if digest(policy_path) != policy_hash:
    fail("value-agent policy SHA-256 does not match the handoff")
cluster_path = regular_file(handoff.get("shock_cluster_csv"), "shock_cluster_csv")
cluster_hash = expected_digest(
    handoff.get("shock_cluster_csv_sha256"), "shock_cluster_csv_sha256"
)
if digest(cluster_path) != cluster_hash:
    fail("shock-cluster CSV SHA-256 does not match the handoff")
validation_sample_path = regular_file(
    handoff.get("validation_sample_csv"), "validation_sample_csv",
)
validation_sample_hash = expected_digest(
    handoff.get("validation_sample_sha256"), "validation_sample_sha256",
)
if digest(validation_sample_path) != validation_sample_hash:
    fail("validation-sample CSV changed after calibration")
cluster_manifest_record = handoff.get("cluster_manifest")
if not isinstance(cluster_manifest_record, dict):
    fail("handoff lacks cluster-manifest provenance")
cluster_manifest_path = regular_file(
    cluster_manifest_record.get("path"), "cluster manifest",
)
cluster_manifest_hash = expected_digest(
    cluster_manifest_record.get("sha256"), "cluster manifest SHA-256",
)
if digest(cluster_manifest_path) != cluster_manifest_hash:
    fail("cluster manifest changed after calibration")
try:
    observed_cluster_manifest = calibration_contract.validate_cluster_manifest(
        cluster_manifest_path,
        assignments_path=cluster_path,
        validation_path=validation_sample_path,
        universe_config_path=training_path,
    )
except (OSError, ValueError, RuntimeError) as error:
    fail(f"cluster provenance is no longer valid: {error}")
normalized_cluster_record = dict(cluster_manifest_record)
normalized_cluster_record["path"] = str(cluster_manifest_path)
if observed_cluster_manifest != normalized_cluster_record:
    fail("cluster manifest content disagrees with the handoff")
verified_cluster_contract = verify_policy_cluster_contract(
    policy_path=policy_path,
    assignments_path=cluster_path,
    validation_path=validation_sample_path,
    symbols=training_symbols,
    required_cluster_count=CANONICAL_CERTIFICATION_PROFILE[
        "required_cluster_count"
    ],
    required_training_representatives=CANONICAL_CERTIFICATION_PROFILE[
        "required_training_representatives_per_cluster"
    ],
    required_validation_symbols=CANONICAL_CERTIFICATION_PROFILE[
        "required_validation_symbols_per_cluster"
    ],
    minimum_cluster_size=CANONICAL_CERTIFICATION_PROFILE[
        "clustering_protocol"
    ]["minimum_cluster_size"],
)
try:
    pool_heldout_record = persisted_pool.get("heldout")
    if not isinstance(pool_heldout_record, dict):
        fail("pooling provenance lacks the frozen held-out config record")
    pool_heldout_path = regular_file(
        pool_heldout_record.get("common_config"),
        "pooling frozen held-out opening universe",
    )
    pool_training_records = persisted_pool.get("training_days")
    if (not isinstance(pool_training_records, list)
            or len(pool_training_records) != 5):
        fail("pooling provenance lacks five source-training records")
    source_sessions = {}
    for source_record in pool_training_records:
        if not isinstance(source_record, dict):
            fail("pooling source-training record is malformed")
        source_date = str(source_record.get("date", ""))
        source_path = regular_file(
            source_record.get("source_config"),
            f"pooling source universe {source_date}",
        )
        source_hash = expected_digest(
            source_record.get("source_config_sha256"),
            f"pooling source universe {source_date} SHA-256",
        )
        if digest(source_path) != source_hash:
            fail(f"pooling source universe changed for {source_date}")
        source_sessions[source_date] = cohort_contract.symbols_from_csv(
            source_path,
            label=f"pooling source universe {source_date}",
        )
    heldout_source_path = regular_file(
        pool_heldout_record.get("source_config"),
        "pooling held-out source universe",
    )
    heldout_source_hash = expected_digest(
        pool_heldout_record.get("source_config_sha256"),
        "pooling held-out source universe SHA-256",
    )
    if digest(heldout_source_path) != heldout_source_hash:
        fail("pooling held-out source universe changed")
    source_sessions[
        CANONICAL_CERTIFICATION_PROFILE["required_validation_date"]
    ] = cohort_contract.symbols_from_csv(
        heldout_source_path,
        label="pooling held-out source universe",
    )
    opening_grid = persisted_pool.get("opening_price_grid_eligibility")
    if not isinstance(opening_grid, dict):
        fail("pooling provenance lacks opening-price-grid evidence")
    raw_exclusions = opening_grid.get("excluded_symbols")
    if (not isinstance(raw_exclusions, list)
            or any(not isinstance(entry, dict) for entry in raw_exclusions)):
        fail("pooling fixed-grid exclusion evidence is malformed")
    observed_input_selection = (
        cohort_contract.certification_pool_input_selection(
            source_sessions=source_sessions,
            excluded_symbols=(entry.get("symbol", "") for entry in raw_exclusions),
            final_symbols=training_symbols,
            project_root=project_root,
        )
    )
    cohort_contract.require_pool_input_selection_record(
        persisted_pool.get("certification_input_selection"),
        expected=observed_input_selection,
        label="pooling provenance certification_input_selection",
    )
    if handoff.get("certification_input_selection") != observed_input_selection:
        fail("calibration handoff does not bind the verified pool input shape")
    expected_pool_cohort_identity = {
        **observed_training_cohort,
        "original_intersection_symbol_count": 1509,
        "fixed_price_grid_excluded_symbol_count": 29,
        "artifact_checks": {
            "pooled_training_universe": observed_training_cohort,
            "heldout_common": cohort_contract.validate_csv(
                pool_heldout_path,
                label="pooling frozen held-out opening universe",
                project_root=project_root,
            ),
            "training_days": {
                day.date: cohort_contract.validate_csv(
                    day.universe_config,
                    label=f"pooling training universe {day.date}",
                    project_root=project_root,
                )
                for day in verified_training_days
            },
        },
    }
    if persisted_pool.get("certification_cohort_required") is not True:
        fail("pooling provenance did not require the immutable cohort")
    if persisted_pool.get("cohort_identity") != expected_pool_cohort_identity:
        fail("pooling provenance cohort identity does not match its exact CSVs")
    expected_handoff_cohort_identity = {
        "schema_version": 1,
        **observed_training_cohort,
        "artifact_checks": {
            "pooled_training_universe": observed_training_cohort,
            "training_days": {
                day.date: cohort_contract.validate_csv(
                    day.universe_config,
                    label=f"calibration training universe {day.date}",
                    project_root=project_root,
                )
                for day in verified_training_days
            },
            "heldout_opening_universe": cohort_contract.validate_csv(
                pool_heldout_path,
                label="calibration held-out opening universe",
                project_root=project_root,
            ),
            "cluster_assignments": cohort_contract.validate_csv(
                cluster_path,
                label="certified cluster assignments",
                project_root=project_root,
            ),
            "full_universe_policy": cohort_contract.validate_csv(
                policy_path,
                label="certified full-universe policy",
                project_root=project_root,
            ),
            "frozen_heldout_runtime_universe": cohort_contract.validate_csv(
                frozen_path_from_handoff,
                label="certified frozen held-out runtime universe",
                project_root=project_root,
            ),
        },
    }
except cohort_contract.CohortIdentityError as error:
    fail(str(error))
if handoff_cohort_identity != expected_handoff_cohort_identity:
    fail("handoff cohort identity disagrees with one or more materialized artifacts")
if manual_shock_cluster_text:
    manual_cluster = pathlib.Path(manual_shock_cluster_text).expanduser().resolve()
    if not manual_cluster.is_file():
        fail(f"SHOCK_CLUSTER_CSV is not a regular file: {manual_cluster}")
    if digest(manual_cluster) != cluster_hash:
        fail("SHOCK_CLUSTER_CSV does not match the cluster CSV certified by the handoff")
if manual_policy_text:
    manual_policy = pathlib.Path(manual_policy_text).expanduser().resolve()
    if not manual_policy.is_file():
        fail(f"VALUE_AGENT_POLICY_CSV is not a regular file: {manual_policy}")
    if digest(manual_policy) != policy_hash:
        fail("VALUE_AGENT_POLICY_CSV does not match the policy certified by the handoff")

controls = handoff.get("runtime_controls")
if not isinstance(controls, dict):
    fail("runtime_controls must be an object")
hawkes = positive_float(controls.get("hawkes_activity_scale"), "hawkes_activity_scale")
local_mm_enabled = boolean(
    controls.get("local_market_maker_enabled"), "local_market_maker_enabled",
)
local_interval = positive_float(controls.get("local_mm_interval_ms"), "local_mm_interval_ms")
local_quantity = positive_float(
    controls.get("local_mm_quantity_multiplier"), "local_mm_quantity_multiplier"
)
local_improvement_probability = probability(
    controls.get("local_mm_improvement_probability"),
    "local_mm_improvement_probability",
)
shared_mm_selected = boolean(
    controls.get("shared_market_maker_enabled"), "shared_market_maker_enabled",
)
if controls.get("shared_quote_mode") != "relative_to_empirical_symbol_quote_size":
    fail("unsupported shared_quote_mode")
selected_shared_multiplier = nonnegative_float(
    controls.get("shared_quote_multiplier"), "shared_quote_multiplier",
)
if shared_mm_selected != (selected_shared_multiplier > 0.0):
    fail("selected shared-MM enablement disagrees with its multiplier")
shared_levels = positive_int(controls.get("shared_quote_levels"), "shared_quote_levels")
decision_window = positive_float(controls.get("decision_window_ms"), "decision_window_ms")

enablement = handoff.get("agent_enablement")
if not isinstance(enablement, dict):
    fail("agent_enablement must be an object")
if (enablement.get("local_market_maker") is not local_mm_enabled
        or enablement.get("shared_market_maker") is not shared_mm_selected
        or enablement.get("value_agents") is not True):
    fail("agent enablement disagrees with selected nested model")
mechanisms = handoff.get("mechanism_treatments")
if not isinstance(mechanisms, dict) or not isinstance(
        mechanisms.get("shared_market_maker"), dict):
    fail("handoff lacks the explicit nonzero shared-MM mechanism treatment")
shared_treatment = mechanisms["shared_market_maker"]
if shared_treatment.get("enabled") is not True:
    fail("shared-MM mechanism treatment must remain enabled")
if shared_treatment.get("quote_mode") != "relative_to_empirical_symbol_quote_size":
    fail("shared-MM treatment must use the relative empirical quote mode")
shared_multiplier = positive_float(
    shared_treatment.get("quote_multiplier"),
    "mechanism_treatments.shared_market_maker.quote_multiplier",
)
selected_by_training_fit = boolean(
    shared_treatment.get("selected_by_training_fit"),
    "mechanism_treatments.shared_market_maker.selected_by_training_fit",
)
if selected_by_training_fit is not shared_mm_selected:
    fail("shared-MM treatment provenance disagrees with the selected model")
if shared_mm_selected:
    if not math.isclose(
            shared_multiplier, selected_shared_multiplier,
            rel_tol=1.0e-12, abs_tol=1.0e-12):
        fail("selected shared-MM treatment multiplier differs from the fitted model")
else:
    if shared_treatment.get("interpretation") != (
            "explicit nonzero case-study scenario; not calibrated"):
        fail("off-baseline shared-MM treatment is not labelled an uncalibrated scenario")

try:
    report = json.loads(report_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    fail(f"cannot parse certified calibration report: {error}")
if not isinstance(report, dict) or report.get("schema_version") != 2:
    fail("unsupported calibration report schema_version (expected 2)")
if report.get("certification") != certification:
    fail("report certification disagrees with the supplied artifact")
if report.get("cohort_identity") != handoff_cohort_identity:
    fail("report cohort identity disagrees with the handoff")
protocol = report.get("protocol")
artifacts = report.get("artifacts")
if not isinstance(protocol, dict) or not isinstance(artifacts, dict):
    fail("certified calibration report lacks protocol/artifacts objects")
if protocol.get("training_config_sha256") != training_hash:
    fail("report training-config hash disagrees with the handoff")
if protocol.get("runtime_configuration_schema") != expected_runtime_schema:
    fail("report runtime-configuration schema disagrees with the handoff")
if protocol.get("training_days") != training_days:
    fail("report training-day provenance disagrees with the handoff")
if protocol.get("simulator_source_semantics_sha256") != calibration_semantics_hash:
    fail("report simulator semantics hash disagrees with the handoff")
if protocol.get("workflow_source_semantics_sha256") != calibration_workflow_hash:
    fail("report workflow semantics hash disagrees with the handoff")
if protocol.get("binary_sha256") != calibration_binary_hash:
    fail("report calibration-binary hash disagrees with the handoff")
if protocol.get("calibration_build_provenance") != build_provenance:
    fail("report build provenance disagrees with the handoff")
if protocol.get("cluster_manifest") != cluster_manifest_record:
    fail("report cluster-manifest provenance disagrees with the handoff")
if protocol.get("pooling_provenance") != pool_record:
    fail("report pooling provenance disagrees with the handoff")
if protocol.get("cohort_identity") != handoff_cohort_identity:
    fail("report protocol cohort identity disagrees with the handoff")
if protocol.get("frozen_empirical_input_bundle_sha256") != frozen_bundle_hash:
    fail("report empirical-input bundle hash disagrees with the handoff")
if protocol.get("development_validation_target_bundle_sha256") != development_target_hash:
    fail("report development-target bundle hash disagrees with the handoff")
if pathlib.Path(str(protocol.get("development_validation_target_root", ""))).expanduser().resolve() \
        != development_target_root:
    fail("report development-target root disagrees with the handoff")
if report.get("certification_profile") != CANONICAL_CERTIFICATION_PROFILE:
    fail("report certification profile disagrees with the canonical handoff profile")
if report.get("certification_profile_sha256") != profile_hash:
    fail("report certification profile hash disagrees with the handoff")
if report.get("observed_runtime_profile") != CANONICAL_CERTIFICATION_PROFILE:
    fail("report search/runtime does not match the canonical profile")
validation_scope = report.get("validation_scope")
if not isinstance(validation_scope, dict):
    fail("report lacks an explicit development-validation scope")
if (validation_scope.get("role") != CANONICAL_CERTIFICATION_PROFILE["validation_role"]
        or validation_scope.get("independent_final_holdout") is not False):
    fail("report misstates the development-validation role")
report_policy = artifacts.get("full_universe_policy_csv")
if not isinstance(report_policy, str) or pathlib.Path(report_policy).expanduser().resolve() != policy_path:
    fail("report full-universe policy path disagrees with the handoff")
report_clusters = report.get("clusters")
expected_cluster_ids = set(verified_cluster_contract["representatives"])
if (not isinstance(report_clusters, dict)
        or set(report_clusters) != {str(cluster) for cluster in expected_cluster_ids}):
    fail("calibration report does not contain exactly the ten certified clusters")
for cluster_id in sorted(expected_cluster_ids):
    cluster_report = report_clusters.get(str(cluster_id))
    if not isinstance(cluster_report, dict):
        fail(f"calibration report cluster {cluster_id} is malformed")
    if (cluster_report.get("cluster_id") != cluster_id
            or cluster_report.get("cluster_label")
                != f"liquidity_{cluster_id:02d}"
            or cluster_report.get("representative_symbols")
                != list(verified_cluster_contract["representatives"][cluster_id])):
        fail(
            f"calibration report cluster {cluster_id} does not identify the "
            "three representatives derived from the certified assignments"
        )
    selected_policy = cluster_report.get("selected_policy")
    if not isinstance(selected_policy, dict):
        fail(f"calibration report cluster {cluster_id} lacks its selected policy")
    expected_enabled, expected_threshold, expected_depth = (
        verified_cluster_contract["policy_by_cluster"][cluster_id]
    )
    if selected_policy.get("enabled") is not expected_enabled:
        fail(f"calibration report policy enablement disagrees for cluster {cluster_id}")
    try:
        reported_threshold = finite_number(
            selected_policy.get("threshold_bps"),
            f"report cluster {cluster_id} threshold_bps",
        )
        reported_depth = finite_number(
            selected_policy.get("depth_participation"),
            f"report cluster {cluster_id} depth_participation",
        )
    except (TypeError, ValueError):
        fail(f"calibration report selected policy is invalid for cluster {cluster_id}")
    if (not math.isclose(reported_threshold, expected_threshold,
                         rel_tol=0.0, abs_tol=1.0e-12)
            or not math.isclose(reported_depth, expected_depth,
                                rel_tol=0.0, abs_tol=1.0e-12)):
        fail(f"calibration report selected policy disagrees for cluster {cluster_id}")

# The economic case study must use the exact held-out opening state that was
# evaluated after parameter selection, while retaining every non-opening field
# from the pooled training configuration.  The report is itself hash-certified
# by the handoff; validate both its artifact reference and the leakage barrier
# before selecting that artifact for execution.
case_config_path = regular_file(
    artifacts.get("frozen_heldout_opening_config_csv"),
    "report frozen_heldout_opening_config_csv",
)
case_config_hash = digest(case_config_path)
report_frozen_hash = expected_digest(
    artifacts.get("frozen_heldout_opening_config_sha256"),
    "report frozen_heldout_opening_config_sha256",
)
report_bundle_hash = expected_digest(
    artifacts.get("frozen_empirical_input_bundle_sha256"),
    "report frozen_empirical_input_bundle_sha256",
)
if case_config_path != frozen_path_from_handoff:
    fail("report frozen held-out configuration path disagrees with the handoff")
if case_config_hash != frozen_hash_from_handoff or report_frozen_hash != case_config_hash:
    fail("report frozen held-out configuration SHA-256 disagrees with the handoff")
if report_bundle_hash != frozen_bundle_hash:
    fail("report empirical-input bundle SHA-256 disagrees with the handoff")
barrier = protocol.get("heldout_leakage_barrier")
if not isinstance(barrier, dict):
    fail("report lacks heldout_leakage_barrier provenance")
allowed_opening_fields = barrier.get("heldout_fields_allowed")
expected_opening_fields = {
    "fundamental_price_ticks",
    "initial_best_bid_ticks",
    "initial_best_ask_ticks",
    "initial_best_bid_depth",
    "initial_best_ask_depth",
}
if (not isinstance(allowed_opening_fields, list)
        or set(allowed_opening_fields) != expected_opening_fields):
    fail("report has an unsupported held-out opening-field set")
stratified = report.get("heldout_stratified_validation")
if not isinstance(stratified, dict):
    fail("report does not contain completed held-out stratified validation")
expected_validation_symbols = tuple(
    symbol
    for cluster_id in sorted(verified_cluster_contract["validation"])
    for symbol in verified_cluster_contract["validation"][cluster_id]
)
if len(expected_validation_symbols) != (
        CANONICAL_CERTIFICATION_PROFILE["required_cluster_count"]
        * CANONICAL_CERTIFICATION_PROFILE[
            "required_validation_symbols_per_cluster"
        ]):
    fail("certified stratified validation does not contain exactly 30 symbols")
if (stratified.get("scope")
        != "one or more non-representative symbols from every cluster"
        or stratified.get("not_a_full_market_distributional_claim") is not True
        or stratified.get("symbols") != list(expected_validation_symbols)):
    fail("report held-out stratified validation has the wrong scope or symbols")
selected_local_record = report.get("global_local_flow_selection")
selected_shared_record = report.get("global_shared_quote_selection")
if (not isinstance(selected_local_record, dict)
        or not isinstance(selected_local_record.get("controls"), dict)
        or not isinstance(selected_shared_record, dict)
        or not isinstance(selected_shared_record.get("candidate"), dict)):
    fail("report lacks controls needed to reconstruct stratified validation")
expected_stratified_controls = {
    **selected_local_record["controls"],
    "shared_quote_mode": "relative_to_empirical_symbol_quote_size",
    "shared_quote_multiplier": selected_shared_multiplier,
    "shared_market_maker_enabled": shared_mm_selected,
    "value_agents_enabled": True,
}
if stratified.get("frozen_runtime_controls") != expected_stratified_controls:
    fail("held-out stratified validation did not use the frozen selected controls")
try:
    stratified_targets = {
        symbol: marketwide_targets[symbol]
        for symbol in expected_validation_symbols
    }
except KeyError as error:
    fail(f"held-out stratified symbol is absent from market-wide targets: {error}")
recomputed_stratified = recompute_seed_evaluation(
    stratified.get("evaluation"),
    expected_seeds=tuple(CANONICAL_CERTIFICATION_PROFILE["required_stage3_seeds"]),
    symbols=expected_validation_symbols,
    targets=stratified_targets,
    status_path=report_path,
    label="held-out 30-symbol stratified validation",
)
try:
    recomputed_stratified_shortfalls = (
        calibration_contract.two_sided_coverage_shortfalls(
            recomputed_stratified,
            CANONICAL_CERTIFICATION_PROFILE[
                "maximum_two_sided_shortfall_diagnostic"
            ],
        )
    )
    recomputed_stratified_coverage = (
        calibration_contract.two_sided_coverage_summary(
            recomputed_stratified,
            CANONICAL_CERTIFICATION_PROFILE[
                "maximum_two_sided_shortfall_diagnostic"
            ],
        )
    )
    recomputed_stratified_fit = calibration_contract.empirical_fit_summary(
        recomputed_stratified,
        maximum_score=CANONICAL_CERTIFICATION_PROFILE["maximum_robust_score"],
        maximum_metric_score=CANONICAL_CERTIFICATION_PROFILE[
            "maximum_metric_score"
        ],
        maximum_symbol_metric_absolute_residual=(
            CANONICAL_CERTIFICATION_PROFILE[
                "maximum_symbol_metric_absolute_robust_residual"
            ]
        ),
    )
except (OSError, ValueError, RuntimeError) as error:
    fail(f"cannot recompute held-out stratified gates: {error}")
stratified_execution_passed = (
    recomputed_stratified.get("two_sided_integrity_passed") is True
)
stratified_background_boundary_passed = (
    recomputed_stratified.get("finite_boundary_adequacy_passed") is True
)
stratified_value_boundary_passed = (
    recomputed_stratified.get("value_boundary_adequacy_passed") is True
)
stratified_coverage_passed = (
    not recomputed_stratified_shortfalls and stratified_execution_passed
)
stratified_structural_adequacy = (
    stratified_execution_passed
    and stratified_background_boundary_passed
    and stratified_value_boundary_passed
    and stratified_coverage_passed
)
recomputed_stratified_fit_passed = (
    recomputed_stratified_fit.get("passed") is True
)
recomputed_stratified_fit_failures = (
    calibration_contract.empirical_fit_failure_reasons(
        "held-out stratified", recomputed_stratified_fit,
    )
    if not recomputed_stratified_fit_passed else []
)
expected_stratified_certification = {
    "execution_integrity_passed": stratified_execution_passed,
    "full_two_sided_book_passed": stratified_execution_passed,
    "coverage_passed": stratified_coverage_passed,
    "finite_boundary_adequacy_passed": (
        stratified_background_boundary_passed
        and stratified_value_boundary_passed
    ),
    "background_boundary_adequacy_passed": (
        stratified_background_boundary_passed
    ),
    "value_boundary_adequacy_passed": stratified_value_boundary_passed,
    "structural_adequacy_passed": stratified_structural_adequacy,
    "empirical_fit_passed": recomputed_stratified_fit_passed,
    "empirical_fit_acceptance_role": "required_reported_diagnostic_only",
    "empirical_fit_failure_reasons": recomputed_stratified_fit_failures,
    "certified_for_case_study": stratified_structural_adequacy,
}
if stratified.get("coverage_summary") != recomputed_stratified_coverage:
    fail("stratified coverage summary disagrees with its five seed CSVs")
if stratified.get("empirical_fit") != recomputed_stratified_fit:
    fail("stratified empirical-fit summary disagrees with its five seed CSVs")
if stratified.get("certification") != expected_stratified_certification:
    fail("stratified certification flags disagree with recomputed evidence")
if not stratified_structural_adequacy:
    fail("recomputed held-out stratified structural validation did not pass")
if certification.get("stratified_empirical_fit") != recomputed_stratified_fit:
    fail("global certification disagrees with recomputed stratified fit")
if stratified_structural_passed is not stratified_structural_adequacy:
    fail("global certification disagrees with stratified structural adequacy")
if stratified_empirical_passed is not recomputed_stratified_fit_passed:
    fail("global certification disagrees with stratified empirical-fit diagnostic")
if stratified_empirical_failure_reasons != recomputed_stratified_fit_failures:
    fail("global certification loses stratified empirical-fit failure reasons")

# The same held-out target moment for a sampled symbol must be used in both
# the 30-symbol coupled run and the 1,480-symbol market-wide run.  Simulated
# outcomes may differ because the shared-risk state is genuinely global.
marketwide_moment_targets = {
    (str(row["symbol"]), str(row["metric"])): (
        row["target"], row["empirical_scale"], row["weight"]
    )
    for row in recomputed_marketwide["moment_estimates"]
}
for row in recomputed_stratified["moment_estimates"]:
    key = (str(row["symbol"]), str(row["metric"]))
    if marketwide_moment_targets.get(key) != (
            row["target"], row["empirical_scale"], row["weight"]):
        fail(f"stratified and market-wide target definitions disagree for {key}")

stratified_status_path = regular_file(
    artifacts.get("heldout_stratified_validation_status_json"),
    "held-out stratified validation status",
)
stratified_status_hash = expected_digest(
    artifacts.get("heldout_stratified_validation_status_sha256"),
    "held-out stratified validation status SHA-256",
)
if digest(stratified_status_path) != stratified_status_hash:
    fail("held-out stratified validation status SHA-256 does not match")
if stratified_status_path.parent != report_path.parent:
    fail("stratified validation status lies outside the calibration result root")
try:
    stratified_status_payload = json.loads(
        stratified_status_path.read_text(encoding="utf-8")
    )
except (OSError, json.JSONDecodeError) as error:
    fail(f"cannot read held-out stratified validation status: {error}")
expected_stratified_status = {
    "schema_version": 2,
    "scope": "pooled_stratified_sample",
    "cohort_identity": handoff_cohort_identity,
    "passed": stratified_structural_adequacy,
    "structural_adequacy_passed": stratified_structural_adequacy,
    **expected_stratified_certification,
    "finite_boundary_adequacy": {
        "background": recomputed_stratified["finite_boundary_adequacy"],
        "value": recomputed_stratified["value_boundary_adequacy"],
    },
    "failure_reasons": [],
    "empirical_fit_failure_reasons": recomputed_stratified_fit_failures,
    "interpretation": (
        "This required stratified probe certifies structural adequacy only; "
        "its empirical-fit score and failures are preserved as diagnostics. "
        "The full-universe market-wide fit is authoritative."
    ),
    "coverage_summary": recomputed_stratified_coverage,
    "coverage_shortfalls": recomputed_stratified_shortfalls,
    "empirical_fit": recomputed_stratified_fit,
    "evaluation": calibration_contract.evaluation_report(
        recomputed_stratified
    ),
}
if stratified_status_payload != expected_stratified_status:
    fail("stratified status JSON disagrees with recomputed report/CSV evidence")
stratified_handoff = handoff.get("heldout_stratified_validation")
expected_stratified_handoff = {
    "passed": stratified_structural_adequacy,
    "structural_adequacy_passed": stratified_structural_adequacy,
    "empirical_fit_passed": recomputed_stratified_fit_passed,
    "empirical_fit_acceptance_role": "required_reported_diagnostic_only",
    "empirical_fit_failure_reasons": recomputed_stratified_fit_failures,
    "status_json": str(stratified_status_path),
    "status_sha256": stratified_status_hash,
    "symbols": len(expected_validation_symbols),
    "validation_date": CANONICAL_CERTIFICATION_PROFILE[
        "required_validation_date"
    ],
    "duration_seconds": CANONICAL_CERTIFICATION_PROFILE[
        "required_session_duration_seconds"
    ],
    "seeds": CANONICAL_CERTIFICATION_PROFILE["required_stage3_seeds"],
}
if stratified_handoff != expected_stratified_handoff:
    fail("handoff stratified provenance disagrees with recomputed evidence")

def read_config(path: pathlib.Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = list(reader.fieldnames or ())
            rows = list(reader)
    except OSError as error:
        fail(f"cannot read {label}: {error}")
    if not fields or not rows:
        fail(f"{label} is empty")
    return fields, rows

training_fields, training_rows = read_config(training_path, "training universe")
case_fields, case_rows = read_config(case_config_path, "frozen held-out universe")
if tuple(training_fields) != tuple(calibration_contract.RUNTIME_CONFIG_FIELDS):
    fail("certified training universe has an unsupported runtime schema")
if training_fields != case_fields:
    fail("frozen held-out universe schema differs from the training universe")
if len(training_rows) != len(case_rows):
    fail("frozen held-out universe book count differs from the training universe")
for index, (training_row, case_row) in enumerate(
        zip(training_rows, case_rows)):
    if (training_row.get("book_id"), training_row.get("symbol")) != (
            case_row.get("book_id"), case_row.get("symbol")):
        fail(f"frozen held-out universe changes book identity at row {index + 2}")
    for field in training_fields:
        if field not in expected_opening_fields and training_row[field] != case_row[field]:
            fail(
                "frozen held-out universe changes non-opening field "
                f"{field!r} for {training_row.get('symbol', '<unknown>')}"
            )
try:
    _, validated_case_rows = calibration_contract.load_universe_config(
        case_config_path
    )
except (OSError, ValueError, RuntimeError) as error:
    fail(f"frozen held-out runtime configuration is invalid: {error}")
case_targets = {
    row["symbol"]: tuple(
        row[field]
        for field in calibration_contract.FROZEN_TRAINING_DERIVED_FIELDS
    )
    for row in validated_case_rows
}
if case_targets != pooled_homeostatic_targets:
    fail(
        "frozen held-out runtime does not use the frozen pooled "
        "spread/depth/latent-value targets"
    )

if requested_universe_hash not in {training_hash, case_config_hash}:
    fail(
        "UNIVERSE_CONFIG matches neither the pooled training universe nor the "
        "report-linked frozen held-out universe"
    )
if requested_universe_hash == training_hash and training_hash != case_config_hash:
    print(
        "NOTICE: handoff mode replaces the backward-compatible pooled-training "
        "UNIVERSE_CONFIG argument with the report-linked frozen held-out opening "
        "configuration for the financial case study.",
        file=sys.stderr,
    )
selected_local = report.get("global_local_flow_selection")
selected_shared = report.get("global_shared_quote_selection")
if not isinstance(selected_local, dict) or not isinstance(selected_shared, dict):
    fail("report lacks selected global controls")
report_controls = selected_local.get("controls")
shared_candidate = selected_shared.get("candidate")
if not isinstance(report_controls, dict) or not isinstance(shared_candidate, dict):
    fail("report selected-control fields are malformed")
for field, expected in (
    ("hawkes_activity_scale", hawkes),
    ("local_mm_interval_ms", local_interval),
    ("local_mm_quantity_multiplier", local_quantity),
):
    observed = positive_float(report_controls.get(field), f"report {field}")
    if not math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        fail(f"report {field} disagrees with the handoff")
reported_local_improvement_probability = probability(
    report_controls.get("local_mm_improvement_probability"),
    "report local_mm_improvement_probability",
)
if not math.isclose(
        reported_local_improvement_probability, local_improvement_probability,
        rel_tol=1.0e-12, abs_tol=1.0e-12):
    fail("report local_mm_improvement_probability disagrees with the handoff")
if boolean(report_controls.get("local_mm_enabled"), "report local_mm_enabled") \
        is not local_mm_enabled:
    fail("report local-MM enablement disagrees with the handoff")
if boolean(shared_candidate.get("enabled"), "report shared enabled") \
        is not shared_mm_selected:
    fail("report shared-MM enablement disagrees with the handoff")
report_shared_multiplier = nonnegative_float(
    shared_candidate.get("multiplier"), "report shared quote multiplier",
)
if not math.isclose(
        report_shared_multiplier, selected_shared_multiplier,
        rel_tol=1.0e-12, abs_tol=1.0e-12):
    fail("report shared quote multiplier disagrees with the handoff")
if value_agent_mode == "on" and not policy_path.is_file():
    fail("VALUE_AGENT=on requires the certified value-agent policy")
certification_path = (
    handoff_path.parent / "independent_global_calibration_certification.json"
)

for value in (
    str(policy_path), format(hawkes, ".12g"), str(int(local_mm_enabled)),
    format(local_interval, ".12g"), format(local_quantity, ".12g"),
    format(local_improvement_probability, ".12g"),
    str(int(shared_mm_selected)), format(shared_multiplier, ".12g"), str(shared_levels),
    format(decision_window, ".12g"), str(case_config_path), case_config_hash,
    training_hash, str(report_path), report_hash, str(cluster_path), cluster_hash,
    str(handoff["artifact_role"]),
    calibration_binary_hash,
    policy_hash,
    str(build_path), str(calibration_binary_path), str(certification_path),
):
    sys.stdout.buffer.write(value.encode("utf-8") + b"\0")
PY
)
    then
        echo "ERROR: calibration handoff loader returned an incomplete control record." >&2
        exit 2
    fi
    if [[ "${CALIBRATION_ARTIFACT_ROLE}" == "certified_calibration_handoff" ]]; then
        CALIBRATION_ROOT="$(dirname -- "${CALIBRATION_HANDOFF_JSON}")"
        if [[ -e "${CALIBRATION_ROOT}/preliminary_calibration_result.json" \
              || -e "${CALIBRATION_ROOT}/calibration_failure.json" ]]; then
            echo "ERROR: a preliminary/failure artifact coexists with the certified handoff." >&2
            exit 2
        fi
        CERTIFICATION_RECHECK_TMP="$(
            mktemp "${TMPDIR:-/tmp}/lob-case-certification.${SLURM_JOB_ID}.XXXXXX"
        )"
        if ! python3 "${PROJECT_DIR}/scripts/verify_global_calibration_certification.py" \
            --project-root "${PROJECT_DIR}" \
            --calibration-root "${CALIBRATION_ROOT}" \
            --binary "${CALIBRATION_BINARY_PATH}" \
            --build-provenance "${CALIBRATION_BUILD_PROVENANCE_PATH}" \
            > "${CERTIFICATION_RECHECK_TMP}"; then
            echo "ERROR: independent calibration re-verification failed; case study is blocked." >&2
            exit 2
        fi
        CALIBRATION_CERTIFICATION_SHA256="$(python3 - \
            "${CALIBRATION_CERTIFICATION_PATH}" \
            "${CERTIFICATION_RECHECK_TMP}" \
            "${CALIBRATION_HANDOFF_JSON}" <<'PY'
import hashlib
import json
import pathlib
import stat
import sys

stored_path = pathlib.Path(sys.argv[1])
recomputed_path = pathlib.Path(sys.argv[2])
handoff_path = pathlib.Path(sys.argv[3])
try:
    stored_status = stored_path.lstat()
except FileNotFoundError:
    raise SystemExit(
        "ERROR: mandatory independent certification artifact is missing: "
        f"{stored_path}"
    )
if stat.S_ISLNK(stored_status.st_mode) or not stat.S_ISREG(stored_status.st_mode):
    raise SystemExit(
        "ERROR: independent certification artifact must be a direct regular file"
    )
try:
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    recomputed = json.loads(recomputed_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"ERROR: cannot parse independent certification JSON: {error}")
if stored != recomputed:
    raise SystemExit(
        "ERROR: stored independent certification differs from fresh re-verification"
    )
if (not isinstance(stored, dict)
        or stored.get("artifact_role") != (
            "independent_global_calibration_certification"
        )
        or stored.get("status") != "PASS"):
    raise SystemExit("ERROR: independent certification artifact is not a PASS")
handoff_hash = hashlib.sha256(handoff_path.read_bytes()).hexdigest()
if stored.get("calibration_handoff_sha256") != handoff_hash:
    raise SystemExit("ERROR: independent certification is bound to another handoff")
print(hashlib.sha256(stored_path.read_bytes()).hexdigest())
PY
)"
        rm -f -- "${CERTIFICATION_RECHECK_TMP}"
        CERTIFICATION_RECHECK_TMP=""
    fi
    HANDOFF_MODE="on"
    UNIVERSE_CONFIG="${CALIBRATED_CASE_CONFIG_PATH}"
    if [[ "${VALUE_AGENT}" == "on" ]]; then
        VALUE_AGENT_POLICY_CSV="${CALIBRATED_POLICY_PATH}"
    fi
    # Even an identical user override is reduced to the canonical path carried
    # by the immutable handoff so provenance is unambiguous.
    SHOCK_CLUSTER_CSV="${CALIBRATED_CLUSTER_PATH}"
    if [[ "${EXPERIMENT}" != "cadence" ]]; then
        python3 - "${WINDOW_MS}" "${CALIBRATED_DECISION_WINDOW_MS}" <<'PY'
import math
import sys
observed, expected = map(float, sys.argv[1:])
if not math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
    raise SystemExit(
        "ERROR: WINDOW_MS must equal the decision_window_ms frozen in "
        "CALIBRATION_HANDOFF_JSON (use EXPERIMENT=cadence for a deliberate sensitivity)."
    )
PY
    fi
fi
if [[ "${EXPERIMENT}" == "science" || "${EXPERIMENT}" == "all" ]]; then
    if [[ "${HANDOFF_MODE}" != "on" \
          || "${CALIBRATION_ARTIFACT_ROLE}" != "certified_calibration_handoff" \
          || "${ALLOW_PRELIMINARY_MODEL}" != "off" ]]; then
        echo "ERROR: final science/all requires the certified calibration handoff; preliminary and legacy modes are diagnostic-only." >&2
        exit 2
    fi
    if [[ "${VALUE_AGENT}" != "on" || "${DURATION_SECONDS}" != "23400" \
          || "${WINDOW_MS}" != "1000" || "${REPETITIONS}" != "5" \
          || "${SEED}" != "20200130" || "${SCIENCE_RANKS}" != "32" \
          || "${SCIENCE_RISK_LIMITS}" != "25,100" \
          || "${MASK_SEED}" != "314159" \
          || "${REFERENCE_RISK_LIMIT}" != "100" ]]; then
        echo "ERROR: final science/all does not match the canonical duration, cadence, seeds, rank count, or risk-limit design." >&2
        exit 2
    fi
    python3 - \
        "${SHOCK_TIME_SECONDS}" "${SHOCK_FRACTION}" \
        "${SHOCK_TOP_DEPTH_MULTIPLE}" "${LOCAL_INVENTORY_LIMIT}" \
        "${CAPACITY_THRESHOLD}" "${POST_SHOCK_HORIZON_SECONDS}" <<'PY'
import math
import sys
labels = (
    "SHOCK_TIME_SECONDS", "SHOCK_FRACTION", "SHOCK_TOP_DEPTH_MULTIPLE",
    "LOCAL_INVENTORY_LIMIT", "CAPACITY_THRESHOLD",
    "POST_SHOCK_HORIZON_SECONDS",
)
expected = (11700.0, 0.01, 1.0, 100.0, 0.5, 1800.0)
for label, text, target in zip(labels, sys.argv[1:], expected):
    try:
        observed = float(text)
    except ValueError:
        raise SystemExit(f"ERROR: {label} is not numeric")
    if not math.isfinite(observed) or not math.isclose(
            observed, target, rel_tol=0.0, abs_tol=1.0e-12):
        raise SystemExit(
            f"ERROR: final science/all requires {label}={target:g}; "
            f"observed {text!r}."
        )
PY
fi
if [[ -z "${SHOCK_CLUSTER_CSV}" || ! -f "${SHOCK_CLUSTER_CSV}" ]]; then
    echo "ERROR: SHOCK_CLUSTER_CSV must be a readable cluster_assignments.csv." >&2
    exit 2
fi
if [[ "${VALUE_AGENT}" == "on" ]]; then
    VALUE_AGENT_POLICY_CSV="$(python3 - "${VALUE_AGENT_POLICY_CSV}" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
)"
    if [[ ! -f "${VALUE_AGENT_POLICY_CSV}" ]]; then
        echo "ERROR: VALUE_AGENT_POLICY_CSV is not a regular file: ${VALUE_AGENT_POLICY_CSV}" >&2
        exit 2
    fi
fi

mkdir -p "${RESULT_DIR}" "${PROJECT_DIR}/slurm"
if [[ "${HANDOFF_MODE}" == "on" \
      && "${CALIBRATION_ARTIFACT_ROLE}" == "certified_calibration_handoff" ]]; then
    INPUT_SNAPSHOT_DIR="${RESULT_DIR}/input_snapshot"
    INPUT_SNAPSHOT_MANIFEST="${INPUT_SNAPSHOT_DIR}/snapshot_manifest.json"
    ORIGINAL_CERTIFIED_UNIVERSE_CONFIG="${UNIVERSE_CONFIG}"
    ORIGINAL_CERTIFIED_POLICY_CSV="${VALUE_AGENT_POLICY_CSV}"
    ORIGINAL_CERTIFIED_CLUSTER_CSV="${SHOCK_CLUSTER_CSV}"
    python3 - \
        "${INPUT_SNAPSHOT_DIR}" \
        "${ORIGINAL_CERTIFIED_UNIVERSE_CONFIG}" "${CALIBRATED_CONFIG_SHA256}" \
        "${ORIGINAL_CERTIFIED_POLICY_CSV}" "${CALIBRATED_POLICY_SHA256}" \
        "${ORIGINAL_CERTIFIED_CLUSTER_CSV}" "${CALIBRATED_CLUSTER_SHA256}" \
        "${CALIBRATION_HANDOFF_JSON}" "${CALIBRATION_CERTIFICATION_PATH}" \
        "${CALIBRATION_CERTIFICATION_SHA256}" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import sys

snapshot_root = pathlib.Path(sys.argv[1])
specifications = (
    ("universe_config", pathlib.Path(sys.argv[2]), sys.argv[3], "universe_config.csv"),
    ("value_agent_policy", pathlib.Path(sys.argv[4]), sys.argv[5], "value_agent_policy.csv"),
    ("shock_clusters", pathlib.Path(sys.argv[6]), sys.argv[7], "shock_clusters.csv"),
)
handoff_path = pathlib.Path(sys.argv[8])
certification_path = pathlib.Path(sys.argv[9])
certification_hash = sys.argv[10]

def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

if snapshot_root.exists():
    raise SystemExit(f"ERROR: input snapshot already exists: {snapshot_root}")
snapshot_root.mkdir(mode=0o700)
records = []
for role, source, expected_hash, filename in specifications:
    try:
        source_status = source.lstat()
    except FileNotFoundError:
        raise SystemExit(f"ERROR: certified {role} disappeared before snapshot: {source}")
    if stat.S_ISLNK(source_status.st_mode) or not stat.S_ISREG(source_status.st_mode):
        raise SystemExit(f"ERROR: certified {role} is not a direct regular file: {source}")
    source_hash = digest(source)
    if source_hash != expected_hash:
        raise SystemExit(
            f"ERROR: certified {role} changed before snapshot: "
            f"expected={expected_hash} observed={source_hash}"
        )
    destination = snapshot_root / filename
    temporary = snapshot_root / f".{filename}.tmp"
    with source.open("rb") as input_file, temporary.open("xb") as output_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            output_file.write(block)
        output_file.flush()
        os.fsync(output_file.fileno())
    copied_hash = digest(temporary)
    if copied_hash != expected_hash:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"ERROR: snapshotted {role} failed SHA-256 verification")
    os.replace(temporary, destination)
    destination.chmod(0o400)
    records.append({
        "role": role,
        "original_path": str(source.resolve()),
        "certified_sha256": expected_hash,
        "snapshot_path": str(destination.resolve()),
        "snapshot_sha256": copied_hash,
    })

manifest = {
    "schema_version": 1,
    "artifact_role": "case_study_validated_input_snapshot",
    "calibration_handoff": str(handoff_path.resolve()),
    "calibration_handoff_sha256": digest(handoff_path),
    "independent_certification": str(certification_path.resolve()),
    "independent_certification_sha256": certification_hash,
    "inputs": records,
}
manifest_path = snapshot_root / "snapshot_manifest.json"
temporary_manifest = snapshot_root / ".snapshot_manifest.json.tmp"
temporary_manifest.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
)
with temporary_manifest.open("rb") as source:
    os.fsync(source.fileno())
os.replace(temporary_manifest, manifest_path)
manifest_path.chmod(0o400)
snapshot_root.chmod(0o500)
PY
    UNIVERSE_CONFIG="${INPUT_SNAPSHOT_DIR}/universe_config.csv"
    VALUE_AGENT_POLICY_CSV="${INPUT_SNAPSHOT_DIR}/value_agent_policy.csv"
    SHOCK_CLUSTER_CSV="${INPUT_SNAPSHOT_DIR}/shock_clusters.csv"
fi
python3 - "${UNIVERSE_CONFIG}" "${CALIBRATION_METADATA}" "${RESULT_DIR}/universe_input.json" \
    "${HANDOFF_MODE}" "${CALIBRATION_HANDOFF_JSON}" "${CALIBRATED_CONFIG_SHA256}" \
    "${CALIBRATED_TRAINING_CONFIG_SHA256}" "${REQUESTED_UNIVERSE_CONFIG}" \
    "${CALIBRATION_REPORT_PATH}" "${CALIBRATION_REPORT_SHA256}" \
    "${CALIBRATION_ARTIFACT_ROLE}" "${PROJECT_DIR}" \
    "${CALIBRATION_CERTIFICATION_PATH}" \
    "${CALIBRATION_CERTIFICATION_SHA256}" \
    "${INPUT_SNAPSHOT_MANIFEST}" \
    "${VALUE_AGENT_POLICY_CSV}" \
    "${SHOCK_CLUSTER_CSV}" <<'PY'
import csv
import hashlib
import json
import math
import pathlib
import stat
import sys

config_path = pathlib.Path(sys.argv[1])
metadata_text = sys.argv[2].strip()
metadata_path = pathlib.Path(metadata_text) if metadata_text else None
output_path = pathlib.Path(sys.argv[3])
handoff_mode = sys.argv[4]
handoff_path_text = sys.argv[5]
handoff_case_config_hash = sys.argv[6]
handoff_training_config_hash = sys.argv[7]
requested_config_path = pathlib.Path(sys.argv[8])
report_path_text = sys.argv[9]
report_hash = sys.argv[10]
artifact_role = sys.argv[11]
project_root = pathlib.Path(sys.argv[12]).resolve()
certification_path_text = sys.argv[13]
certification_hash = sys.argv[14]
snapshot_manifest_text = sys.argv[15]
snapshot_policy_path_text = sys.argv[16]
snapshot_cluster_path_text = sys.argv[17]
sys.path.insert(0, str(project_root / "scripts"))
import calibrate_cluster_value_agents as calibration_contract
import certification_cohort as cohort_contract

with config_path.open(newline="", encoding="utf-8") as source:
    reader = csv.DictReader(source)
    config_fields = list(reader.fieldnames or [])
    rows = list(reader)
if not rows:
    raise SystemExit("ERROR: empirical universe configuration is empty.")
expected_runtime_fields = list(calibration_contract.RUNTIME_CONFIG_FIELDS)
if config_fields != expected_runtime_fields:
    raise SystemExit("ERROR: final universe does not use runtime schema version 5.")
if rows[0].get("book_id") != "0" or rows[0].get("symbol") != "QQQ":
    raise SystemExit("ERROR: final universe must be QQQ-first at book_id 0.")
for expected, row in enumerate(rows):
    if row.get("book_id") != str(expected):
        raise SystemExit(f"ERROR: non-contiguous book_id at row {expected + 2}.")
    for field in ("data_dir", "hawkes_rates_file"):
        value = pathlib.Path(row.get(field, ""))
        exists_with_expected_kind = (
            (field == "data_dir" and value.is_dir())
            or (field == "hawkes_rates_file" and value.is_file())
        )
        if not value.is_absolute() or not exists_with_expected_kind:
            raise SystemExit(
                f"ERROR: row {expected + 2} has missing/non-absolute {field}: {value}"
            )
    for field in (
            "target_spread_ticks", "target_mean_bid_depth",
            "target_mean_ask_depth"):
        try:
            target = float(row.get(field, ""))
        except ValueError:
            raise SystemExit(
                f"ERROR: row {expected + 2} has nonnumeric {field}."
            )
        if not math.isfinite(target) or target <= 0.0:
            raise SystemExit(
                f"ERROR: row {expected + 2} has nonpositive {field}."
            )
    try:
        latent_volatility = float(row.get(
            "fundamental_volatility_bps_sqrt_second", ""
        ))
    except ValueError:
        raise SystemExit(
            f"ERROR: row {expected + 2} has nonnumeric latent volatility."
        )
    if not math.isfinite(latent_volatility) or latent_volatility < 0.0:
        raise SystemExit(
            f"ERROR: row {expected + 2} has negative latent volatility."
        )
    try:
        latent_move_probability = float(row.get(
            "fundamental_move_probability_per_second", ""
        ))
    except ValueError:
        raise SystemExit(
            f"ERROR: row {expected + 2} has nonnumeric latent move probability."
        )
    if (not math.isfinite(latent_move_probability)
            or not 0.0 <= latent_move_probability <= 1.0):
        raise SystemExit(
            f"ERROR: row {expected + 2} has out-of-range latent move probability."
        )
    try:
        latent_conditional_kurtosis = float(row.get(
            "fundamental_conditional_kurtosis", ""
        ))
    except ValueError:
        raise SystemExit(
            f"ERROR: row {expected + 2} has nonnumeric latent conditional kurtosis."
        )
    if (not math.isfinite(latent_conditional_kurtosis)
            or latent_conditional_kurtosis < 1.0):
        raise SystemExit(
            f"ERROR: row {expected + 2} has invalid latent conditional kurtosis."
        )
config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
schema_hash = hashlib.sha256(json.dumps(
    expected_runtime_fields, ensure_ascii=True, separators=(",", ":")
).encode("utf-8")).hexdigest()
record = {
    "schema_version": 4,
    "universe_config": str(config_path),
    "universe_config_sha256": config_hash,
    "book_count": len(rows),
    "books_per_asset": 1,
    "runtime_configuration_schema": {
        "schema_version": 5,
        "fields": expected_runtime_fields,
        "sha256": schema_hash,
        "pooled_homeostatic_fields": list(
            calibration_contract.POOLED_HOMEOSTATIC_FIELDS
        ),
        "latent_value_fields": list(calibration_contract.LATENT_VALUE_FIELDS),
        "frozen_training_derived_fields": list(
            calibration_contract.FROZEN_TRAINING_DERIVED_FIELDS
        ),
        "heldout_target_files_used": False,
    },
}
metadata = None
configuration = None
stored_hash = None
metadata_matches_config = None
if metadata_path is not None:
    record["calibration_metadata"] = str(metadata_path)
    record["calibration_metadata_sha256"] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"ERROR: cannot parse CALIBRATION_METADATA: {error}")
    configuration = metadata.get("configuration", {})
    if not isinstance(configuration, dict):
        raise SystemExit("ERROR: CALIBRATION_METADATA has no configuration object.")
    stored_hash = configuration.get("sha256")
    metadata_matches_config = stored_hash == config_hash
if handoff_mode == "on":
    if handoff_case_config_hash != config_hash:
        raise SystemExit("ERROR: report-linked held-out config hash does not match UNIVERSE_CONFIG.")
    handoff_path = pathlib.Path(handoff_path_text)
    report_path = pathlib.Path(report_path_text)
    if not handoff_path.is_file() or not report_path.is_file():
        raise SystemExit("ERROR: validated handoff/report path disappeared before provenance write.")
    try:
        handoff_payload = json.loads(handoff_path.read_text(encoding="utf-8"))
        observed_cohort_identity = cohort_contract.validate_symbols(
            (row.get("symbol", "") for row in rows),
            label="final case-study universe",
            project_root=project_root,
        )
        cohort_contract.require_identity_record(
            handoff_payload.get("cohort_identity"),
            label="case-study handoff cohort identity",
        )
    except (OSError, json.JSONDecodeError,
            cohort_contract.CohortIdentityError) as error:
        raise SystemExit(f"ERROR: case-study cohort identity failed: {error}")
    if handoff_payload.get("cohort_identity", {}).get(
            "symbol_order_sha256") != observed_cohort_identity.get(
                "symbol_order_sha256"):
        raise SystemExit(
            "ERROR: final case-study universe is not the handoff cohort."
        )
    record["cohort_identity"] = handoff_payload["cohort_identity"]
    if hashlib.sha256(report_path.read_bytes()).hexdigest() != report_hash:
        raise SystemExit("ERROR: calibration report changed after handoff validation.")
    record["calibration_provenance_mode"] = (
        "block_coordinate_certified_handoff"
        if artifact_role == "certified_calibration_handoff"
        else "block_coordinate_preliminary_explicit_override"
    )
    record["calibration_handoff"] = {
        "path": str(handoff_path),
        "sha256": hashlib.sha256(handoff_path.read_bytes()).hexdigest(),
        "report": str(report_path),
        "report_sha256": report_hash,
        "training_config_sha256": handoff_training_config_hash,
        "case_study_config_role": "frozen_training_backgrounds_with_heldout_openings",
        "case_study_config_sha256": handoff_case_config_hash,
        "requested_universe_config": str(requested_config_path),
        "independent_certification": certification_path_text,
        "independent_certification_sha256": certification_hash,
    }
    if artifact_role == "certified_calibration_handoff":
        snapshot_manifest = pathlib.Path(snapshot_manifest_text)
        try:
            snapshot_status = snapshot_manifest.lstat()
        except FileNotFoundError:
            raise SystemExit("ERROR: certified case-study input snapshot is missing.")
        if (stat.S_ISLNK(snapshot_status.st_mode)
                or not stat.S_ISREG(snapshot_status.st_mode)):
            raise SystemExit(
                "ERROR: certified input snapshot manifest must be a direct regular file."
            )
        try:
            snapshot_payload = json.loads(
                snapshot_manifest.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"ERROR: cannot parse input snapshot manifest: {error}")
        if (not isinstance(snapshot_payload, dict)
                or snapshot_payload.get("schema_version") != 1
                or snapshot_payload.get("artifact_role") != (
                    "case_study_validated_input_snapshot"
                )):
            raise SystemExit("ERROR: input snapshot manifest has the wrong contract.")

        def file_sha256(path: pathlib.Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        if pathlib.Path(snapshot_payload.get("calibration_handoff", "")).resolve() \
                != handoff_path.resolve():
            raise SystemExit("ERROR: input snapshot is bound to another handoff path.")
        if snapshot_payload.get("calibration_handoff_sha256") != file_sha256(
                handoff_path):
            raise SystemExit("ERROR: input snapshot is bound to another handoff hash.")
        certification_path = pathlib.Path(certification_path_text)
        try:
            certification_status = certification_path.lstat()
        except FileNotFoundError:
            raise SystemExit("ERROR: independent certification disappeared before launch.")
        if (stat.S_ISLNK(certification_status.st_mode)
                or not stat.S_ISREG(certification_status.st_mode)
                or file_sha256(certification_path) != certification_hash):
            raise SystemExit(
                "ERROR: independent certification changed after input validation."
            )
        if pathlib.Path(snapshot_payload.get("independent_certification", "")).resolve() \
                != certification_path.resolve():
            raise SystemExit(
                "ERROR: input snapshot is bound to another certification path."
            )
        if snapshot_payload.get("independent_certification_sha256") != certification_hash:
            raise SystemExit(
                "ERROR: input snapshot is bound to another certification hash."
            )

        expected_snapshot_paths = {
            "universe_config": config_path.resolve(),
            "value_agent_policy": pathlib.Path(snapshot_policy_path_text).resolve(),
            "shock_clusters": pathlib.Path(snapshot_cluster_path_text).resolve(),
        }
        snapshot_records = snapshot_payload.get("inputs")
        if not isinstance(snapshot_records, list) or len(snapshot_records) != 3:
            raise SystemExit("ERROR: input snapshot must contain exactly three inputs.")
        observed_roles = set()
        for snapshot_record in snapshot_records:
            if not isinstance(snapshot_record, dict):
                raise SystemExit("ERROR: input snapshot contains a malformed record.")
            role = snapshot_record.get("role")
            if role not in expected_snapshot_paths or role in observed_roles:
                raise SystemExit("ERROR: input snapshot roles are missing or duplicated.")
            observed_roles.add(role)
            snapshot_path = pathlib.Path(snapshot_record.get("snapshot_path", ""))
            try:
                input_status = snapshot_path.lstat()
            except FileNotFoundError:
                raise SystemExit(f"ERROR: snapshotted {role} input is missing.")
            if (stat.S_ISLNK(input_status.st_mode)
                    or not stat.S_ISREG(input_status.st_mode)
                    or snapshot_path.resolve() != expected_snapshot_paths[role]):
                raise SystemExit(f"ERROR: snapshotted {role} path is invalid.")
            observed_hash = file_sha256(snapshot_path)
            if (snapshot_record.get("snapshot_sha256") != observed_hash
                    or snapshot_record.get("certified_sha256") != observed_hash):
                raise SystemExit(f"ERROR: snapshotted {role} hash is invalid.")
        if observed_roles != set(expected_snapshot_paths):
            raise SystemExit("ERROR: input snapshot is incomplete.")
        record["validated_input_snapshot"] = {
            "path": str(snapshot_manifest.resolve()),
            "sha256": file_sha256(snapshot_manifest),
            "provenance": snapshot_payload,
        }
    # The source extractor's metadata normally describes the pre-intersection
    # universe.  The handoff, rather than that larger file, is the exact
    # certificate for the final common-symbol configuration.
    record["source_extractor_metadata_supplied"] = metadata_path is not None
    record["source_extractor_metadata_matches_final_config"] = metadata_matches_config
    if metadata_matches_config:
        assert configuration is not None
        if configuration.get("book_count") != len(rows):
            raise SystemExit("ERROR: matching CALIBRATION_METADATA has a different book count.")
        if configuration.get("qqq_book_id") != 0:
            raise SystemExit("ERROR: matching CALIBRATION_METADATA does not certify QQQ at book_id 0.")
else:
    if metadata_path is None or configuration is None:
        raise SystemExit(
            "ERROR: legacy uncalibrated mode requires CALIBRATION_METADATA."
        )
    if stored_hash != config_hash:
        raise SystemExit("ERROR: CALIBRATION_METADATA does not match UNIVERSE_CONFIG hash.")
    if configuration.get("book_count") != len(rows):
        raise SystemExit("ERROR: CALIBRATION_METADATA book count does not match UNIVERSE_CONFIG.")
    if configuration.get("qqq_book_id") != 0:
        raise SystemExit("ERROR: CALIBRATION_METADATA does not certify QQQ at book_id 0.")
    record["calibration_provenance_mode"] = "legacy_uncalibrated"
output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(record, sort_keys=True))
PY

echo "== REAL-UNIVERSE CASE STUDY ALLOCATION =="
date --iso-8601=seconds
hostname -f
echo "job_id=${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} tasks=${SLURM_NTASKS} node_list=${SLURM_JOB_NODELIST}"
echo "experiment=${EXPERIMENT} universe_config=${UNIVERSE_CONFIG} result_dir=${RESULT_DIR}"
if [[ "${HANDOFF_MODE}" == "on" ]]; then
    echo "requested_universe_config=${REQUESTED_UNIVERSE_CONFIG} case_study_input=frozen_heldout_openings"
fi
echo "value_agent=${VALUE_AGENT} policy=${VALUE_AGENT_POLICY_CSV:-none}"
echo "calibration_mode=${HANDOFF_MODE} handoff=${CALIBRATION_HANDOFF_JSON:-none}"
if [[ "${HANDOFF_MODE}" == "on" ]]; then
    echo "independent_certification=${CALIBRATION_CERTIFICATION_PATH:-none} sha256=${CALIBRATION_CERTIFICATION_SHA256:-none}"
    echo "selected_controls=hawkes:${CALIBRATED_HAWKES_ACTIVITY_SCALE} local_mm_enabled:${CALIBRATED_LOCAL_MM_ENABLED} local_mm_ms:${CALIBRATED_LOCAL_MM_INTERVAL_MS} local_mm_quantity:${CALIBRATED_LOCAL_MM_QUANTITY_MULTIPLIER} local_mm_improvement_probability:${CALIBRATED_LOCAL_MM_IMPROVEMENT_PROBABILITY} shared_mm_selected:${CALIBRATED_SHARED_MM_SELECTED}"
    echo "shared_mm_mechanism_treatment=relative_multiplier:${CALIBRATED_SHARED_TREATMENT_MULTIPLIER} selected_by_fit:${CALIBRATED_SHARED_MM_SELECTED}"
    echo "allow_preliminary_model=${ALLOW_PRELIMINARY_MODEL}"
fi
echo "duration=${DURATION_SECONDS}s window_ms=${WINDOW_MS} shock_time=${SHOCK_TIME_SECONDS}s shock_fraction=${SHOCK_FRACTION} shock_bid_depth_multiple=${SHOCK_TOP_DEPTH_MULTIPLE}"
echo "mpi_lib_dir=${MPI_LIB_DIR}"
module list 2>&1 || true

lob_deterministic_configure_and_build \
    "${PROJECT_DIR}" "${BUILD_DIR}" "${MPI_LIB_DIR}" "${BUILD_JOBS}"

EXECUTABLE="${BUILD_DIR}/fragmented_mpi_lob"
if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "ERROR: expected executable is missing: ${EXECUTABLE}" >&2
    exit 2
fi
echo "== EXECUTABLE RUNTIME LIBRARIES =="
ldd "${EXECUTABLE}"
if ldd "${EXECUTABLE}" | grep -q 'not found'; then
    echo "ERROR: executable has unresolved runtime libraries." >&2
    exit 2
fi
CASE_EXECUTABLE_SHA256="$(python3 - "${EXECUTABLE}" <<'PY'
import hashlib
import pathlib
import sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
echo "case_executable_sha256=${CASE_EXECUTABLE_SHA256}"
if [[ "${HANDOFF_MODE}" == "on" \
      && "${CASE_EXECUTABLE_SHA256}" != "${CALIBRATED_BINARY_SHA256}" ]]; then
    echo "ERROR: rebuilt case-study executable is not byte-identical to the executable calibrated and validated by the handoff." >&2
    echo "calibration_binary_sha256=${CALIBRATED_BINARY_SHA256}" >&2
    echo "case_binary_sha256=${CASE_EXECUTABLE_SHA256}" >&2
    exit 2
fi

python3 - "${RESULT_DIR}/universe_input.json" "${EXECUTABLE}" \
    "${CASE_EXECUTABLE_SHA256}" "${EXPERIMENT}" "${DURATION_SECONDS}" \
    "${WINDOW_MS}" "${SHOCK_TIME_SECONDS}" "${SHOCK_FRACTION}" \
    "${SHOCK_TARGET_COUNT}" "${SHOCK_TOP_DEPTH_MULTIPLE}" "${MASK_SEED}" \
    "${SCIENCE_RANKS}" "${SCIENCE_RISK_LIMITS}" \
    "${REFERENCE_RISK_LIMIT}" "${LOCAL_INVENTORY_LIMIT}" \
    "${CAPACITY_THRESHOLD}" "${REPETITIONS}" "${SEED}" \
    "${POST_SHOCK_HORIZON_SECONDS}" "${CADENCE_WINDOWS_MS}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import tempfile

manifest = pathlib.Path(sys.argv[1]).resolve()
executable = pathlib.Path(sys.argv[2]).resolve()
payload = json.loads(manifest.read_text(encoding="utf-8"))
profile = {
    "profile_id": "systemic_liquidity_case_v1",
    "experiment": sys.argv[4],
    "duration_seconds": int(sys.argv[5]),
    "decision_window_ms": float(sys.argv[6]),
    "shock_time_seconds": float(sys.argv[7]),
    "shock_fraction": float(sys.argv[8]),
    "shock_target_count": int(sys.argv[9]),
    "shock_top_depth_multiple": float(sys.argv[10]),
    "shock_target_seed": int(sys.argv[11]),
    "science_ranks": int(sys.argv[12]),
    "science_risk_limits": [float(value) for value in sys.argv[13].split(",")],
    "reference_risk_limit": float(sys.argv[14]),
    "local_inventory_limit": float(sys.argv[15]),
    "capacity_threshold": float(sys.argv[16]),
    "repetitions": int(sys.argv[17]),
    "base_seed": int(sys.argv[18]),
    "post_shock_horizon_seconds": float(sys.argv[19]),
    "cadence_windows_ms": [float(value) for value in sys.argv[20].split(",")],
}
encoded = json.dumps(
    profile, sort_keys=True, separators=(",", ":"), allow_nan=False,
).encode("utf-8")
payload["case_executable"] = str(executable)
payload["case_executable_sha256"] = sys.argv[3]
payload["case_study_protocol"] = profile
payload["case_study_protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
descriptor, temporary_name = tempfile.mkstemp(
    dir=manifest.parent, prefix=f".{manifest.name}.", suffix=".tmp", text=True,
)
temporary = pathlib.Path(temporary_name)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    os.replace(temporary, manifest)
except BaseException:
    temporary.unlink(missing_ok=True)
    raise
PY

RUNNER=(
    python3 "${PROJECT_DIR}/scripts/run_fragmented_mpi_experiments.py"
    --executable "${EXECUTABLE}"
    --universe-config "${UNIVERSE_CONFIG}"
    --mpirun mpirun
    --bind-to core
    --map-by slot
    --run-timeout-seconds "${RUN_TIMEOUT_SECONDS}"
    --campaign-manifest "${RESULT_DIR}/universe_input.json"
)
if [[ "${VALUE_AGENT}" == "on" ]]; then
    RUNNER+=(--value-agent-policy-csv "${VALUE_AGENT_POLICY_CSV}")
fi
RUNTIME_CONTROL_ARGS=()
if [[ "${HANDOFF_MODE}" == "on" ]]; then
    python3 - "${SHARED_QUOTE_LEVELS}" "${CALIBRATED_SHARED_QUOTE_LEVELS}" <<'PY'
import sys
if int(sys.argv[1]) != int(sys.argv[2]):
    raise SystemExit(
        "ERROR: SHARED_QUOTE_LEVELS may not override the value frozen in "
        "CALIBRATION_HANDOFF_JSON."
    )
PY
    RUNTIME_CONTROL_ARGS=(
        --hawkes-activity-scales "${CALIBRATED_HAWKES_ACTIVITY_SCALE}"
        --local-mm-intervals-ms "${CALIBRATED_LOCAL_MM_INTERVAL_MS}"
        --local-mm-quantity-multipliers "${CALIBRATED_LOCAL_MM_QUANTITY_MULTIPLIER}"
        --local-mm-improvement-probabilities "${CALIBRATED_LOCAL_MM_IMPROVEMENT_PROBABILITY}"
        --shared-quote-relative
        --shared-quote-multiplier "${CALIBRATED_SHARED_TREATMENT_MULTIPLIER}"
        --shared-quote-levels "${CALIBRATED_SHARED_QUOTE_LEVELS}"
    )
    if [[ "${CALIBRATED_LOCAL_MM_ENABLED}" == "0" ]]; then
        RUNTIME_CONTROL_ARGS+=(--disable-local-mm)
    fi
else
    # Explicit legacy mode retains the old relative shared-quote convention;
    # its default global controls are supplied by the runner and are not a
    # calibrated empirical specification.
    RUNTIME_CONTROL_ARGS=(
        --shared-quote-relative
        --shared-quote-multiplier "${SHARED_QUOTE_MULTIPLIER}"
        --shared-quote-levels "${SHARED_QUOTE_LEVELS}"
    )
fi
COMMON_CASE_ARGS=(
    --duration-seconds "${DURATION_SECONDS}"
    --window-ms "${WINDOW_MS}"
    --shock-time-seconds "${SHOCK_TIME_SECONDS}"
    --shock-fraction "${SHOCK_FRACTION}"
    --shock-target-count "${SHOCK_TARGET_COUNT}"
    --shock-target-seed "${MASK_SEED}"
    --shock-cluster-csv "${SHOCK_CLUSTER_CSV}"
    --shock-top-depth-multiple "${SHOCK_TOP_DEPTH_MULTIPLE}"
    --local-inventory-limit "${LOCAL_INVENTORY_LIMIT}"
    --capacity-thresholds "${CAPACITY_THRESHOLD}"
    "${RUNTIME_CONTROL_ARGS[@]}"
    --seed "${SEED}"
)
if [[ "${VALUE_AGENT}" == "off" ]]; then
    COMMON_CASE_ARGS+=(--disable-value-agent)
fi

run_scaling() {
    echo "== REAL-UNIVERSE FULL-DAY RANK-INVARIANCE AND STRONG SCALING =="
    "${RUNNER[@]}" "${COMMON_CASE_ARGS[@]}" \
        --ranks "${SCALING_RANKS}" \
        --risk-limits "${REFERENCE_RISK_LIMIT}" \
        --shared-mm-modes global \
        --shock-modes off \
        --repetitions "${REPETITIONS}" \
        --seed-step 0 \
        --output "${RESULT_DIR}/real_universe_scaling_raw.csv" \
        --summary "${RESULT_DIR}/real_universe_scaling_summary.csv"
}

run_pilot() {
    echo "== ONE-SEED CAPACITY PILOT: USE TO SET TIGHT/REFERENCE/LOOSE SCENARIOS =="
    "${RUNNER[@]}" "${COMMON_CASE_ARGS[@]}" \
        --ranks "${PILOT_RANKS}" \
        --risk-limits "${PILOT_RISK_LIMITS}" \
        --shared-mm-modes global \
        --shock-modes on,off \
        --repetitions 1 \
        --seed-step 0 \
        --metrics-dir "${RESULT_DIR}/pilot_metrics" \
        --shock-targets-dir "${RESULT_DIR}/pilot_targets" \
        --output "${RESULT_DIR}/capacity_pilot_raw.csv" \
        --summary "${RESULT_DIR}/capacity_pilot_summary.csv"
    python3 "${PROJECT_DIR}/scripts/analyze_capacity_pilot.py" \
        --raw "${RESULT_DIR}/capacity_pilot_raw.csv" \
        --shock-time-seconds "${SHOCK_TIME_SECONDS}" \
        --output "${RESULT_DIR}/capacity_pilot_diagnostics.csv"
    echo "Inspect capacity_pilot_diagnostics.csv; choose one tight, one moderate, and an uncoupled control."
    echo "Choose SCIENCE_RISK_LIMITS to span inactive/moderate/binding capacity, then run EXPERIMENT=science."
}

run_science() {
    echo "== PAIRED CASE STUDY: GLOBALLY CONSTRAINED SHARED MM =="
    "${RUNNER[@]}" "${COMMON_CASE_ARGS[@]}" \
        --ranks "${SCIENCE_RANKS}" \
        --risk-limits "${SCIENCE_RISK_LIMITS}" \
        --shared-mm-modes global \
        --shock-modes on,off \
        --repetitions "${REPETITIONS}" \
        --seed-step 1 \
        --metrics-dir "${RESULT_DIR}/science_metrics/global" \
        --shock-targets-dir "${RESULT_DIR}/science_targets/global" \
        --asset-summary-dir "${RESULT_DIR}/science_asset_summaries/global" \
        --output "${RESULT_DIR}/science_global_raw.csv" \
        --summary "${RESULT_DIR}/science_global_summary.csv"

    echo "== PRIMARY MECHANISM CONTROL: UNCOUPLED SHARED MM =="
    "${RUNNER[@]}" "${COMMON_CASE_ARGS[@]}" \
        --ranks "${SCIENCE_RANKS}" \
        --risk-limits "${REFERENCE_RISK_LIMIT}" \
        --shared-mm-modes uncoupled \
        --shock-modes on,off \
        --repetitions "${REPETITIONS}" \
        --seed-step 1 \
        --metrics-dir "${RESULT_DIR}/science_metrics/uncoupled" \
        --shock-targets-dir "${RESULT_DIR}/science_targets/uncoupled" \
        --asset-summary-dir "${RESULT_DIR}/science_asset_summaries/uncoupled" \
        --output "${RESULT_DIR}/science_uncoupled_raw.csv" \
        --summary "${RESULT_DIR}/science_uncoupled_summary.csv"

    echo "== MATCHED NEGATIVE CONTROL: SHARED MM OFF =="
    "${RUNNER[@]}" "${COMMON_CASE_ARGS[@]}" \
        --ranks "${SCIENCE_RANKS}" \
        --risk-limits "${REFERENCE_RISK_LIMIT}" \
        --shared-mm-modes off \
        --shock-modes on,off \
        --repetitions "${REPETITIONS}" \
        --seed-step 1 \
        --metrics-dir "${RESULT_DIR}/science_metrics/shared_off" \
        --shock-targets-dir "${RESULT_DIR}/science_targets/shared_off" \
        --asset-summary-dir "${RESULT_DIR}/science_asset_summaries/shared_off" \
        --output "${RESULT_DIR}/science_shared_off_raw.csv" \
        --summary "${RESULT_DIR}/science_shared_off_summary.csv"

    python3 "${PROJECT_DIR}/scripts/analyze_fragmented_shared_liquidity_case.py" \
        --global-raw "${RESULT_DIR}/science_global_raw.csv" \
        --uncoupled-raw "${RESULT_DIR}/science_uncoupled_raw.csv" \
        --shared-off-raw "${RESULT_DIR}/science_shared_off_raw.csv" \
        --universe-input "${RESULT_DIR}/universe_input.json" \
        --shock-time-seconds "${SHOCK_TIME_SECONDS}" \
        --horizon-seconds "${POST_SHOCK_HORIZON_SECONDS}" \
        --rank "${SCIENCE_RANKS}" \
        --output-dir "${RESULT_DIR}/science_analysis"

    # The primary endpoint above uses the 30-minute market-wide time series.
    # This separate diagnostic averages fixed-clock full-session per-asset
    # moments within the ten predeclared liquidity clusters.  It is useful for
    # heterogeneity description but must not replace the primary endpoint.
    python3 "${PROJECT_DIR}/scripts/analyze_cluster_liquidity_heterogeneity.py" \
        --global-raw "${RESULT_DIR}/science_global_raw.csv" \
        --uncoupled-raw "${RESULT_DIR}/science_uncoupled_raw.csv" \
        --shared-off-raw "${RESULT_DIR}/science_shared_off_raw.csv" \
        --universe-config "${UNIVERSE_CONFIG}" \
        --cluster-assignments "${SHOCK_CLUSTER_CSV}" \
        --rank "${SCIENCE_RANKS}" \
        --output-dir "${RESULT_DIR}/science_cluster_analysis"
}

run_sensitivity() {
    local multiplier threshold label value_mode
    echo "== SHOCK-SIZE SENSITIVITY AT REFERENCE CAPACITY =="
    for multiplier in 0.5 2.0; do
        label="lambda_${multiplier}"
        "${RUNNER[@]}" "${COMMON_CASE_ARGS[@]}" \
            --shock-top-depth-multiple "${multiplier}" \
            --ranks "${SCIENCE_RANKS}" --risk-limits "${REFERENCE_RISK_LIMIT}" \
            --shared-mm-modes global,uncoupled --shock-modes on,off \
            --repetitions "${REPETITIONS}" --seed-step 1 \
            --metrics-dir "${RESULT_DIR}/sensitivity/${label}/metrics" \
            --output "${RESULT_DIR}/sensitivity/${label}_raw.csv"
    done
    echo "== CAPACITY-THRESHOLD SENSITIVITY =="
    "${RUNNER[@]}" "${COMMON_CASE_ARGS[@]}" \
        --capacity-thresholds "${ALTERNATIVE_CAPACITY_THRESHOLD}" \
        --ranks "${SCIENCE_RANKS}" --risk-limits "${REFERENCE_RISK_LIMIT}" \
        --shared-mm-modes global,uncoupled --shock-modes on,off \
        --repetitions "${REPETITIONS}" --seed-step 1 \
        --metrics-dir "${RESULT_DIR}/sensitivity/u0_075/metrics" \
        --output "${RESULT_DIR}/sensitivity/u0_075_raw.csv"
    echo "== ONE-BOOK LOCAL-TO-GLOBAL PROPAGATION CHECK =="
    "${RUNNER[@]}" "${COMMON_CASE_ARGS[@]}" \
        --shock-target-count 1 \
        --ranks "${SCIENCE_RANKS}" --risk-limits "${REFERENCE_RISK_LIMIT}" \
        --shared-mm-modes global,uncoupled --shock-modes on,off \
        --repetitions "${REPETITIONS}" --seed-step 1 \
        --metrics-dir "${RESULT_DIR}/sensitivity/one_book/metrics" \
        --output "${RESULT_DIR}/sensitivity/one_book_raw.csv"
    echo "== TARGET-MASK COMPOSITION SENSITIVITY =="
    "${RUNNER[@]}" "${COMMON_CASE_ARGS[@]}" \
        --shock-target-seed "${ALTERNATIVE_MASK_SEED}" \
        --ranks "${SCIENCE_RANKS}" --risk-limits "${REFERENCE_RISK_LIMIT}" \
        --shared-mm-modes global,uncoupled --shock-modes on,off \
        --repetitions "${REPETITIONS}" --seed-step 1 \
        --metrics-dir "${RESULT_DIR}/sensitivity/alternative_mask/metrics" \
        --output "${RESULT_DIR}/sensitivity/alternative_mask_raw.csv"
    echo "== VALUE-AGENT-OFF ABLATION =="
    "${RUNNER[@]}" "${COMMON_CASE_ARGS[@]}" \
        --disable-value-agent \
        --ranks "${SCIENCE_RANKS}" --risk-limits "${REFERENCE_RISK_LIMIT}" \
        --shared-mm-modes global,uncoupled --shock-modes on,off \
        --repetitions "${REPETITIONS}" --seed-step 1 \
        --metrics-dir "${RESULT_DIR}/sensitivity/value_off/metrics" \
        --output "${RESULT_DIR}/sensitivity/value_off_raw.csv"
}

run_cadence() {
    local cadence_window cadence_root
    echo "== OPTIONAL GLOBAL-DECISION-CADENCE SENSITIVITY =="
    IFS=',' read -r -a cadence_values <<< "${CADENCE_WINDOWS_MS}"
    for cadence_window in "${cadence_values[@]}"; do
        cadence_root="${RESULT_DIR}/cadence_${cadence_window}ms"
        echo "== CADENCE ${cadence_window} ms: SHARED MM ON =="
        "${RUNNER[@]}" "${COMMON_CASE_ARGS[@]}" \
            --window-ms "${cadence_window}" \
            --ranks "${CADENCE_RANKS}" \
            --risk-limits "${REFERENCE_RISK_LIMIT}" \
            --shared-mm-modes global \
            --shock-modes on,off \
            --repetitions "${CADENCE_REPETITIONS}" \
            --seed-step 1 \
            --metrics-dir "${cadence_root}/metrics/shared_on" \
            --shock-targets-dir "${cadence_root}/targets/shared_on" \
            --output "${cadence_root}/shared_on_raw.csv" \
            --summary "${cadence_root}/shared_on_summary.csv"
        echo "== CADENCE ${cadence_window} ms: UNCOUPLED MECHANISM CONTROL =="
        "${RUNNER[@]}" "${COMMON_CASE_ARGS[@]}" \
            --window-ms "${cadence_window}" \
            --ranks "${CADENCE_RANKS}" \
            --risk-limits "${REFERENCE_RISK_LIMIT}" \
            --shared-mm-modes uncoupled \
            --shock-modes on,off \
            --repetitions "${CADENCE_REPETITIONS}" \
            --seed-step 1 \
            --metrics-dir "${cadence_root}/metrics/uncoupled" \
            --shock-targets-dir "${cadence_root}/targets/uncoupled" \
            --output "${cadence_root}/uncoupled_raw.csv" \
            --summary "${cadence_root}/uncoupled_summary.csv"
        echo "== CADENCE ${cadence_window} ms: SHARED MM OFF NEGATIVE CONTROL =="
        "${RUNNER[@]}" "${COMMON_CASE_ARGS[@]}" \
            --window-ms "${cadence_window}" --ranks "${CADENCE_RANKS}" \
            --risk-limits "${REFERENCE_RISK_LIMIT}" --shared-mm-modes off \
            --shock-modes on,off --repetitions "${CADENCE_REPETITIONS}" \
            --seed-step 1 --metrics-dir "${cadence_root}/metrics/off" \
            --output "${cadence_root}/off_raw.csv"
        python3 "${PROJECT_DIR}/scripts/analyze_fragmented_shared_liquidity_case.py" \
            --global-raw "${cadence_root}/shared_on_raw.csv" \
            --uncoupled-raw "${cadence_root}/uncoupled_raw.csv" \
            --shared-off-raw "${cadence_root}/off_raw.csv" \
            --universe-input "${RESULT_DIR}/universe_input.json" \
            --shock-time-seconds "${SHOCK_TIME_SECONDS}" \
            --horizon-seconds "${POST_SHOCK_HORIZON_SECONDS}" \
            --rank "${CADENCE_RANKS}" \
            --output-dir "${cadence_root}/analysis"
    done
}

case "${EXPERIMENT}" in
    scaling) run_scaling ;;
    pilot) run_pilot ;;
    science) run_science ;;
    sensitivity) run_sensitivity ;;
    cadence) run_cadence ;;
    all)
        run_scaling
        run_science
        run_sensitivity
        run_cadence
        ;;
esac

echo "== REAL-UNIVERSE CASE STUDY COMPLETE =="
echo "result_dir=${RESULT_DIR}"
date --iso-8601=seconds
