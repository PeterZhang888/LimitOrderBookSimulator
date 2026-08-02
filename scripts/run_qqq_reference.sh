#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${project_dir}/build}"
output_dir="${1:-${project_dir}/results/qqq_full_day_provisional}"
data_dir="${project_dir}/data/itch_20200130_qqq"
rates_file="${data_dir}/hawkes_rates_qqq_provisional_20200130.csv"

"${build_dir}/sequential_multi_asset_lob" \
  --duration-seconds 23400 \
  --books 1 \
  --seed 12345 \
  --data-dir "${data_dir}" \
  --hawkes-rates-file "${rates_file}" \
  --quote-interval-ms 150 \
  --quote-quantity 75 \
  --quote-levels 7 \
  --quote-growth 2 \
  --output-dir "${output_dir}"

python3 "${project_dir}/scripts/compare_qqq_baseline.py" \
  --summary "${output_dir}/sequential_multi_asset_summary.csv" \
  --targets "${data_dir}/market_targets_qqq_20200130.csv" \
  --book-id 0 \
  --output "${output_dir}/comparison.csv"

