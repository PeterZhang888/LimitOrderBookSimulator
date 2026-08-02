#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "usage: $0 TRAINING_ITCH_GZ [DATA_ROOT] [RESULT_ROOT]" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "${script_dir}/.." && pwd)
training_archive=$1
data_root=${2:-"${project_root}/data"}
result_root=${3:-"${project_root}/results/value_agent_heldout"}
training_date=2019-12-30
heldout_date=2020-01-30
training_compact=20191230
heldout_compact=20200130
training_config="${project_root}/config/qqq_aapl_msft_amzn_train_${training_compact}.csv"
heldout_config="${project_root}/config/qqq_aapl_msft_amzn_holdout_${heldout_compact}.csv"
expected_training_bytes=3524013057

if [[ ! -f ${training_archive} ]]; then
    echo "training archive is missing: ${training_archive}" >&2
    exit 1
fi
actual_training_bytes=$(wc -c < "${training_archive}" | tr -d '[:space:]')
if [[ ${actual_training_bytes} != "${expected_training_bytes}" ]]; then
    echo "training archive size mismatch: expected ${expected_training_bytes}, got ${actual_training_bytes}" >&2
    exit 1
fi
if ! gzip -t "${training_archive}"; then
    echo "training archive failed gzip integrity validation: ${training_archive}" >&2
    exit 1
fi

for symbol in qqq aapl msft amzn; do
    target="${data_root}/itch_${heldout_compact}_${symbol}/market_targets_${symbol}_${heldout_compact}.csv"
    if [[ ! -f ${target} ]]; then
        echo "held-out target is missing: ${target}" >&2
        exit 1
    fi
done
opening="${data_root}/itch_${heldout_compact}_basket/opening_bbo_${heldout_compact}.csv"
if [[ ! -f ${opening} ]]; then
    echo "held-out opening BBO is missing: ${opening}" >&2
    exit 1
fi

"${script_dir}/extract_calibrate_multi_asset_day.sh" \
    "${training_archive}" "${training_date}" "${data_root}"

python3 "${script_dir}/build_multi_asset_config.py" \
    --data-root "${data_root}" \
    --opening-date "${heldout_date}" \
    --calibration-date "${training_date}" \
    --weights-file "${project_root}/config/qqq_reduced_basket_weights_20190930.csv" \
    --output "${heldout_config}"

python3 "${script_dir}/calibrate_and_validate_value_agent.py" \
    --binary "${project_root}/build/sequential_multi_asset_lob" \
    --training-config "${training_config}" \
    --heldout-config "${heldout_config}" \
    --training-date "${training_date}" \
    --heldout-date "${heldout_date}" \
    --target-root "${data_root}" \
    --output-dir "${result_root}"

python3 "${script_dir}/summarize_value_validation.py" \
    --report "${result_root}/value_agent_calibration_report.json" \
    --target-root "${data_root}" \
    --result-root "${result_root}" \
    --output-csv "${result_root}/value_agent_metric_detail.csv" \
    --output-markdown "${result_root}/VALUE_AGENT_VALIDATION.md"

python3 "${script_dir}/validate_selected_exact_mpi.py" \
    --report "${result_root}/value_agent_calibration_report.json" \
    --config "${heldout_config}" \
    --sequential-binary "${project_root}/build/sequential_multi_asset_lob" \
    --exact-binary "${project_root}/build/exact_mpi_multi_asset_lob" \
    --duration-seconds 300 \
    --output-dir "${result_root}/exact_rank_validation"

echo "chronological value-agent validation complete"
echo "training_config=${training_config}"
echo "heldout_config=${heldout_config}"
echo "report=${result_root}/value_agent_calibration_report.json"
echo "validation_summary=${result_root}/VALUE_AGENT_VALIDATION.md"
echo "exact_evidence=${result_root}/exact_rank_validation/selected_exact_mpi_validation.json"
