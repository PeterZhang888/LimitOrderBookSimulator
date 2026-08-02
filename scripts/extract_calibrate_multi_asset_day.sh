#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 PATH_TO_ITCH50_GZ YYYY-MM-DD [OUTPUT_ROOT]" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "${script_dir}/.." && pwd)
input_file=$1
trading_date=$2
output_root=${3:-"${project_root}/data"}
# The three-stage value-agent workflow uses matched ITCH prefixes.  Keep this
# configurable because this extractor is also used by older full-day-only
# workflows.  Commas and spaces are both accepted, for example "300,3600".
target_windows=${TARGET_WINDOW_SECONDS:-"300,3600"}

if [[ ! ${trading_date} =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "trading date must be YYYY-MM-DD" >&2
    exit 2
fi
date_compact=${trading_date//-/}
source_hash=$(shasum -a 256 "${input_file}" | awk '{print $1}')
target_windows=${target_windows//,/ }
read -r -a target_window_args <<< "${target_windows}"

python3 "${script_dir}/extract_itch50_symbols.py" \
    --input "${input_file}" \
    --input-sha256 "${source_hash}" \
    --symbols QQQ AAPL MSFT AMZN \
    --date "${trading_date}" \
    --start 09:30:00 \
    --end 16:00:00 \
    --snapshot-ms 1000 \
    --target-window-seconds "${target_window_args[@]}" \
    --output-root "${output_root}"

for symbol in QQQ AAPL MSFT AMZN; do
    lower=$(printf '%s' "${symbol}" | tr '[:upper:]' '[:lower:]')
    symbol_dir="${output_root}/itch_${date_compact}_${lower}"
    python3 "${script_dir}/derive_hawkes_rates.py" \
        --manifest "${symbol_dir}/itch_manifest_${lower}_${date_compact}.json" \
        --activity-scale 0.30 \
        --beta 10 \
        --balance-directional-volume \
        --balance-best-depth \
        --balance-strength 1.0 \
        --output "${symbol_dir}/hawkes_rates_${lower}_balanced_${date_compact}.csv"
done

python3 "${script_dir}/build_multi_asset_config.py" \
    --data-root "${output_root}" \
    --opening-date "${trading_date}" \
    --calibration-date "${trading_date}" \
    --weights-file "${project_root}/config/qqq_reduced_basket_weights_20190930.csv" \
    --output "${project_root}/config/qqq_aapl_msft_amzn_train_${date_compact}.csv"

echo "four-asset one-pass extraction and first-stage calibration complete"
echo "source_sha256=${source_hash}"
