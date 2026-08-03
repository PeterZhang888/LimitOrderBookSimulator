#!/usr/bin/env bash
# Pool the five fixed 2019 ITCH training sessions into one direct-input
# template, while retaining every individual day for behavioural calibration.
#
# This is a normal Slurm submission script: submit it from seagull01 and it
# performs all work only after a compute node has been allocated.  It does not
# launch MPI. First extract the five training dates and the 2020-01-30
# development-validation date with `scripts/extract_itch50_symbols.py`, then
# submit:
#
#   sbatch --export=ALL,\
# TRAIN_20190130_ROOT=/shared/results/itch_universe_20190130_<job>,\
# TRAIN_20190327_ROOT=/shared/results/itch_universe_20190327_<job>,\
# TRAIN_20190730_ROOT=/shared/results/itch_universe_20190730_<job>,\
# TRAIN_20191030_ROOT=/shared/results/itch_universe_20191030_<job>,\
# TRAIN_20191230_ROOT=/shared/results/itch_universe_20191230_<job>,\
# HELDOUT_20200130_ROOT=/shared/results/itch_universe_20200130_<job> \
# submit_five_day_pooled_training.sh
#
# Its output `pooling_provenance.json`, together with this producer source-tree
# path, is consumed by submit_cluster_value_agent_calibration.sh.
#
# The script does NOT average opening books. It sums observed histogram counts,
# computes observed rates from total events / total session seconds, applies
# the declared reduced-book directional-volume and best-depth closures, audits
# the Hawkes stationary inversion, and retains a real held-out opening
# separately. Candidate policies are later scored on each individual training
# session with equal day-level weight.
#SBATCH --job-name=lob-five-day-pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -Eeuo pipefail

: "${SLURM_JOB_ID:?submit this file with sbatch}"
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}"
: "${TRAIN_20190130_ROOT:?provide the completed 2019-01-30 extraction result root}"
: "${TRAIN_20190327_ROOT:?provide the completed 2019-03-27 extraction result root}"
: "${TRAIN_20190730_ROOT:?provide the completed 2019-07-30 extraction result root}"
: "${TRAIN_20191030_ROOT:?provide the completed 2019-10-30 extraction result root}"
: "${TRAIN_20191230_ROOT:?provide the completed 2019-12-30 extraction result root}"
: "${HELDOUT_20200130_ROOT:?provide the completed 2020-01-30 extraction result root}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
POOL_RESULT_ROOT="${POOL_RESULT_ROOT:-${PROJECT_DIR}/results/seagull/five_day_pool_${SLURM_JOB_ID}}"
MINIMUM_COMMON_SYMBOLS="20"
SIMULATOR_TICK_SIZE_PRICE_UNITS="100"
MINIMUM_OPENING_BID_PRICE_UNITS="10000"
# These values define the analytical ITCH-rate inversion consumed by the
# canonical calibration profile and therefore are not submit-time treatments.
ACTIVITY_SCALE="0.30"
HAWKES_BETA="10.0"
BALANCE_STRENGTH="1.0"
QUOTE_QUANTITY_FRACTION="0.5"
MINIMUM_QUOTE_QUANTITY="10"
MAXIMUM_QUOTE_QUANTITY="1000"
POOL_LABEL="five_2019_sessions"
POOL_OVERWRITE="${POOL_OVERWRITE:-off}"
SEAGULL_GCC_MODULE="gcc/15.2.0-gcc-8.5.0-r7c4jsu"
SEAGULL_CALIBRATION_MODULES="${SEAGULL_CALIBRATION_MODULES:-${SEAGULL_GCC_MODULE} python/3.14.2-gcc-15.2.0-e63sscp}"

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

positive_integer() {
    local name="$1" value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] && (( value > 0 )) \
        || fail "${name} must be a positive integer; observed '${value}'."
}

finite_positive() {
    local name="$1" value="$2"
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

finite_range() {
    local name="$1" value="$2" minimum="$3" maximum="$4"
    python3 - "${name}" "${value}" "${minimum}" "${maximum}" <<'PY'
import math
import sys
name, text, lower_text, upper_text = sys.argv[1:]
try:
    value, lower, upper = float(text), float(lower_text), float(upper_text)
except ValueError:
    raise SystemExit(f"ERROR: {name} must be numeric; observed {text!r}.")
if not math.isfinite(value) or not lower <= value <= upper:
    raise SystemExit(f"ERROR: {name} must lie in [{lower:g}, {upper:g}]; observed {text!r}.")
PY
}

absolute_directory() {
    python3 - "$1" <<'PY'
import pathlib
import sys
path = pathlib.Path(sys.argv[1]).expanduser().resolve()
if not path.is_dir():
    raise SystemExit(f"ERROR: not a directory: {path}")
print(path)
PY
}

if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/lmod.sh ]]; then
    source /etc/profile.d/lmod.sh
fi
type module >/dev/null 2>&1 || fail "module command is unavailable"
module purge
# Intentional word splitting: this is a whitespace-separated module list.
module load ${SEAGULL_CALIBRATION_MODULES}
command -v python3 >/dev/null 2>&1 || fail "python3 is unavailable after module load"

positive_integer MINIMUM_COMMON_SYMBOLS "${MINIMUM_COMMON_SYMBOLS}"
positive_integer SIMULATOR_TICK_SIZE_PRICE_UNITS "${SIMULATOR_TICK_SIZE_PRICE_UNITS}"
positive_integer MINIMUM_OPENING_BID_PRICE_UNITS "${MINIMUM_OPENING_BID_PRICE_UNITS}"
(( MINIMUM_OPENING_BID_PRICE_UNITS % SIMULATOR_TICK_SIZE_PRICE_UNITS == 0 )) \
    || fail "MINIMUM_OPENING_BID_PRICE_UNITS must be a multiple of SIMULATOR_TICK_SIZE_PRICE_UNITS"
positive_integer MINIMUM_QUOTE_QUANTITY "${MINIMUM_QUOTE_QUANTITY}"
positive_integer MAXIMUM_QUOTE_QUANTITY "${MAXIMUM_QUOTE_QUANTITY}"
(( MINIMUM_QUOTE_QUANTITY <= MAXIMUM_QUOTE_QUANTITY )) \
    || fail "MINIMUM_QUOTE_QUANTITY must not exceed MAXIMUM_QUOTE_QUANTITY"
finite_positive ACTIVITY_SCALE "${ACTIVITY_SCALE}"
finite_positive HAWKES_BETA "${HAWKES_BETA}"
finite_positive QUOTE_QUANTITY_FRACTION "${QUOTE_QUANTITY_FRACTION}"
finite_range BALANCE_STRENGTH "${BALANCE_STRENGTH}" 0 5
case "${POOL_OVERWRITE}" in on|off) ;; *) fail "POOL_OVERWRITE must be on or off" ;; esac

if [[ -e "${POOL_RESULT_ROOT}" ]]; then
    if [[ ! -d "${POOL_RESULT_ROOT}" ]]; then
        fail "POOL_RESULT_ROOT is not a directory: ${POOL_RESULT_ROOT}"
    fi
    if [[ "${POOL_OVERWRITE}" == "off" ]] \
            && find "${POOL_RESULT_ROOT}" -mindepth 1 -print -quit | grep -q .; then
        fail "refusing to reuse non-empty POOL_RESULT_ROOT=${POOL_RESULT_ROOT}; set POOL_OVERWRITE=on only to replace the same pooling artifacts"
    fi
else
    mkdir -p "${POOL_RESULT_ROOT}"
fi
mkdir -p "${PROJECT_DIR}/slurm"
cd "${PROJECT_DIR}"

TRAINING_DATES=(2019-01-30 2019-03-27 2019-07-30 2019-10-30 2019-12-30)
TRAINING_ROOTS=(
    "${TRAIN_20190130_ROOT}"
    "${TRAIN_20190327_ROOT}"
    "${TRAIN_20190730_ROOT}"
    "${TRAIN_20191030_ROOT}"
    "${TRAIN_20191230_ROOT}"
)
POOL_ARGS=()
for (( index = 0; index < ${#TRAINING_DATES[@]}; ++index )); do
    day="${TRAINING_DATES[index]}"
    compact="${day//-/}"
    root="$(absolute_directory "${TRAINING_ROOTS[index]}")"
    config="${root}/nasdaq_common_plus_qqq_${compact}.csv"
    targets="${root}/empirical_data"
    [[ -f "${config}" ]] || fail "missing universe config for ${day}: ${config}"
    [[ -d "${targets}" ]] || fail "missing empirical target root for ${day}: ${targets}"
    POOL_ARGS+=(--training-day "${day}" "${config}")
    POOL_ARGS+=(--training-target-root "${day}" "${targets}")
done

HELDOUT_DATE=2020-01-30
HELDOUT_COMPACT="${HELDOUT_DATE//-/}"
HELDOUT_ROOT="$(absolute_directory "${HELDOUT_20200130_ROOT}")"
HELDOUT_CONFIG="${HELDOUT_ROOT}/nasdaq_common_plus_qqq_${HELDOUT_COMPACT}.csv"
HELDOUT_TARGET_ROOT="${HELDOUT_ROOT}/empirical_data"
[[ -f "${HELDOUT_CONFIG}" ]] || fail "missing held-out universe config: ${HELDOUT_CONFIG}"
[[ -d "${HELDOUT_TARGET_ROOT}" ]] || fail "missing held-out target root: ${HELDOUT_TARGET_ROOT}"

# The July 24 compact archive predates the exact two-sided clock metadata used
# by the certified calibration.  Audit every target and horizon with the same
# strict loader before pooling, so an obsolete empirical bundle fails here
# with one precise diagnostic instead of later in behavioural calibration.
CERTIFICATION_DATA_ROOT="$(python3 - \
    "${TRAINING_ROOTS[@]}" "${HELDOUT_ROOT}" <<'PY'
import pathlib
import sys

roots = [pathlib.Path(value).expanduser().resolve() for value in sys.argv[1:]]
parents = {root.parent for root in roots}
if len(parents) != 1:
    raise SystemExit(
        "ERROR: certified empirical preflight requires all six itch_DATE "
        "roots beneath one data directory"
    )
for root in roots:
    if root.name != f"itch_{root.name.removeprefix('itch_')}":
        raise SystemExit(f"ERROR: unexpected empirical day-root name: {root}")
print(next(iter(parents)))
PY
)"
EMPIRICAL_PREFLIGHT_REPORT="${PROJECT_DIR}/slurm/empirical_preflight_${SLURM_JOB_ID}.json"
python3 "${PROJECT_DIR}/scripts/preflight_empirical_calibration_inputs.py" \
    --data-root "${CERTIFICATION_DATA_ROOT}" \
    --symbols-file "${PROJECT_DIR}/config/certification_symbols_1480.txt" \
    --output "${EMPIRICAL_PREFLIGHT_REPORT}"

echo "== FIVE-DAY AUDITED REDUCED-BOOK EMPIRICAL POOLING =="
date --iso-8601=seconds
echo "job_id=${SLURM_JOB_ID} node_list=${SLURM_JOB_NODELIST}"
echo "training_dates=${TRAINING_DATES[*]}"
echo "heldout_date=${HELDOUT_DATE}"
echo "pool_result_root=${POOL_RESULT_ROOT}"
echo "simulator_tick_size_price_units=${SIMULATOR_TICK_SIZE_PRICE_UNITS}"
echo "minimum_opening_bid_price_units=${MINIMUM_OPENING_BID_PRICE_UNITS}"
module list 2>&1 || true

POOL_COMMAND=(
    python3 "${PROJECT_DIR}/scripts/pool_multiday_empirical_universe.py"
    "${POOL_ARGS[@]}"
    --heldout-date "${HELDOUT_DATE}"
    --heldout-config "${HELDOUT_CONFIG}"
    --heldout-target-root "${HELDOUT_TARGET_ROOT}"
    --output-root "${POOL_RESULT_ROOT}"
    --label "${POOL_LABEL}"
    --minimum-symbols "${MINIMUM_COMMON_SYMBOLS}"
    --require-certification-cohort
    --simulator-tick-size-price-units "${SIMULATOR_TICK_SIZE_PRICE_UNITS}"
    --minimum-opening-bid-price-units "${MINIMUM_OPENING_BID_PRICE_UNITS}"
    --activity-scale "${ACTIVITY_SCALE}"
    --hawkes-beta "${HAWKES_BETA}"
    --balance-strength "${BALANCE_STRENGTH}"
    --balance-directional-volume
    --balance-best-depth
    --quote-quantity-fraction "${QUOTE_QUANTITY_FRACTION}"
    --minimum-quote-quantity "${MINIMUM_QUOTE_QUANTITY}"
    --maximum-quote-quantity "${MAXIMUM_QUOTE_QUANTITY}"
)
if [[ "${POOL_OVERWRITE}" == "on" ]]; then
    POOL_COMMAND+=(--overwrite)
fi
# Do not open a log inside POOL_RESULT_ROOT before the pooler starts.  The
# pooler deliberately rejects a non-empty output directory, and `tee` opens
# its destination before the left-hand process executes.  Keep the live log
# in the job-log directory and move it into the completed artifact only after
# the pooler returns successfully.
POOL_RESULT_LOG="${PROJECT_DIR}/slurm/five_day_pool_${SLURM_JOB_ID}_result.json"
"${POOL_COMMAND[@]}" | tee "${POOL_RESULT_LOG}"
mv "${POOL_RESULT_LOG}" "${POOL_RESULT_ROOT}/pooling_result.json"
mv "${EMPIRICAL_PREFLIGHT_REPORT}" \
    "${POOL_RESULT_ROOT}/empirical_input_preflight.json"

echo "pooling_provenance=${POOL_RESULT_ROOT}/pooling_provenance.json"
echo "pooled_training_universe=${POOL_RESULT_ROOT}/pooled_training_universe.csv"
echo "heldout_common=${POOL_RESULT_ROOT}/heldout_common.csv"
date --iso-8601=seconds
