#!/usr/bin/env bash
# Run a leakage-safe four-asset weighted-moment calibration and hold-out test.
#
# This script performs substantial raw-ITCH extraction and simulation.  Invoke
# it only from an allocated compute node, for example through
# submit_four_asset_wmm_validation.sh; never run it on a login/head node.
#
# Usage:
#   $0 TRAINING_ITCH_GZ YYYY-MM-DD HELDOUT_ITCH_GZ YYYY-MM-DD \
#      [DATA_ROOT] [RESULT_ROOT]
#
# The public Nasdaq sample directory contains 01302020.NASDAQ_ITCH50.gz, but
# not a 01312020 file.  A Jan-31 holdout therefore requires the user's licensed
# historical archive staged on the cluster.  This script intentionally does not
# attempt to download or substitute a different day.

set -Eeuo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
    echo "usage: $0 TRAINING_ITCH_GZ YYYY-MM-DD HELDOUT_ITCH_GZ YYYY-MM-DD [DATA_ROOT] [RESULT_ROOT]" >&2
    exit 2
fi

training_archive=$1
training_date=$2
heldout_archive=$3
heldout_date=$4
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "${script_dir}/.." && pwd)
data_root=${5:-"${project_root}/data"}
result_root=${6:-"${project_root}/results/four_asset_wmm_${training_date//-/}_${heldout_date//-/}"}

require_iso_date() {
    local value=$1
    if ! [[ "${value}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "ERROR: expected an ISO date (YYYY-MM-DD), got '${value}'." >&2
        exit 2
    fi
}

require_archive() {
    local value=$1
    if [[ ! -f "${value}" ]]; then
        echo "ERROR: ITCH archive is missing: ${value}" >&2
        exit 1
    fi
    if ! gzip -t "${value}"; then
        echo "ERROR: ITCH archive failed gzip integrity validation: ${value}" >&2
        exit 1
    fi
}

require_iso_date "${training_date}"
require_iso_date "${heldout_date}"
if [[ "${training_date}" == "${heldout_date}" || "${training_date}" > "${heldout_date}" ]]; then
    echo "ERROR: training date must precede held-out date." >&2
    exit 2
fi
require_archive "${training_archive}"
require_archive "${heldout_archive}"

training_compact=${training_date//-/}
heldout_compact=${heldout_date//-/}
training_config="${project_root}/config/qqq_aapl_msft_amzn_train_${training_compact}.csv"
heldout_config="${project_root}/config/qqq_aapl_msft_amzn_holdout_${heldout_compact}.csv"
binary=${BINARY:-"${project_root}/build/sequential_multi_asset_lob"}

stage1_duration=${STAGE1_DURATION_SECONDS:-300}
stage2_duration=${STAGE2_DURATION_SECONDS:-3600}
stage3_duration=${STAGE3_DURATION_SECONDS:-23400}
stage1_seed_list=${STAGE1_SEEDS:-1729}
stage2_seed_list=${STAGE2_SEEDS:-1729,7919}
seed_list=${SEEDS:-1729,7919,1103,6599,2027}
threshold_list=${THRESHOLDS:-5,10,20}
response_list=${RESPONSE_STEPS:-2.5,5}
quantity_list=${BASE_QUANTITIES:-10,25,50}
volatility_list=${VOLATILITIES:-0,0.5}
stage1_top_candidates=${STAGE1_TOP_CANDIDATES:-12}
stage2_top_candidates=${STAGE2_TOP_CANDIDATES:-4}
coupling_mode=${COUPLING_MODE:-etf}
timeout_seconds=${TIMEOUT_SECONDS:-900}

read -r -a stage1_seeds <<< "${stage1_seed_list//,/ }"
read -r -a stage2_seeds <<< "${stage2_seed_list//,/ }"
read -r -a seeds <<< "${seed_list//,/ }"
read -r -a thresholds <<< "${threshold_list//,/ }"
read -r -a response_steps <<< "${response_list//,/ }"
read -r -a base_quantities <<< "${quantity_list//,/ }"
read -r -a volatilities <<< "${volatility_list//,/ }"
if (( ${#stage1_seeds[@]} == 0 || ${#stage2_seeds[@]} == 0 || ${#seeds[@]} == 0 || ${#thresholds[@]} == 0 \
      || ${#response_steps[@]} == 0 || ${#base_quantities[@]} == 0 \
      || ${#volatilities[@]} == 0 )); then
    echo "ERROR: calibration seed and candidate lists must be non-empty." >&2
    exit 2
fi

if (( ${#stage2_seeds[@]} < 2 || ${#seeds[@]} < 2 )); then
    echo "ERROR: STAGE2_SEEDS and SEEDS must each provide at least two independent seeds." >&2
    exit 2
fi
if [[ ! -x "${binary}" ]]; then
    echo "ERROR: sequential simulator is missing or not executable: ${binary}" >&2
    echo "Build it first with CMake on the allocated node." >&2
    exit 1
fi

mkdir -p "${data_root}" "${result_root}"

echo "== Stage 0: training-day ITCH extraction and background inputs =="
TARGET_WINDOW_SECONDS="${stage1_duration},${stage2_duration}" \
"${script_dir}/extract_calibrate_multi_asset_day.sh" \
    "${training_archive}" "${training_date}" "${data_root}"

echo "== Stage 0: held-out ITCH targets and opening state =="
TARGET_WINDOW_SECONDS="${stage1_duration},${stage2_duration}" \
"${script_dir}/extract_calibrate_multi_asset_day.sh" \
    "${heldout_archive}" "${heldout_date}" "${data_root}"

echo "== Frozen-background held-out configuration =="
python3 "${script_dir}/build_multi_asset_config.py" \
    --data-root "${data_root}" \
    --opening-date "${heldout_date}" \
    --calibration-date "${training_date}" \
    --weights-file "${project_root}/config/qqq_reduced_basket_weights_20190930.csv" \
    --output "${heldout_config}"

echo "== Stages 1–3: window-matched weighted moment calibration and held-out validation =="
python3 "${script_dir}/calibrate_and_validate_value_agent.py" \
    --binary "${binary}" \
    --training-config "${training_config}" \
    --heldout-config "${heldout_config}" \
    --training-date "${training_date}" \
    --heldout-date "${heldout_date}" \
    --target-root "${data_root}" \
    --output-dir "${result_root}" \
    --stage1-duration "${stage1_duration}" \
    --stage2-duration "${stage2_duration}" \
    --stage3-duration "${stage3_duration}" \
    --stage1-top-candidates "${stage1_top_candidates}" \
    --stage2-top-candidates "${stage2_top_candidates}" \
    --stage1-seeds "${stage1_seeds[@]}" \
    --stage2-seeds "${stage2_seeds[@]}" \
    --seeds "${seeds[@]}" \
    --thresholds "${thresholds[@]}" \
    --response-steps "${response_steps[@]}" \
    --base-quantities "${base_quantities[@]}" \
    --volatilities "${volatilities[@]}" \
    --coupling-mode "${coupling_mode}" \
    --timeout-seconds "${timeout_seconds}"

python3 "${script_dir}/summarize_value_validation.py" \
    --report "${result_root}/value_agent_calibration_report.json" \
    --target-root "${data_root}" \
    --result-root "${result_root}" \
    --output-csv "${result_root}/four_asset_wmm_metric_detail.csv" \
    --output-markdown "${result_root}/FOUR_ASSET_WMM_VALIDATION.md"

echo "four-asset WMM validation complete"
echo "report=${result_root}/value_agent_calibration_report.json"
echo "summary=${result_root}/FOUR_ASSET_WMM_VALIDATION.md"
