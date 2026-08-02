#!/usr/bin/env bash
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${project_dir}/build}"
output_root="${1:-${project_dir}/results/coupled_demo}"
mpi_ranks="${MPI_RANKS:-4}"
mpi_launcher="${MPI_LAUNCHER:-mpirun}"
data_dir="${project_dir}/data/itch_20200130_qqq"
rates_file="${data_dir}/hawkes_rates_qqq_provisional_20200130.csv"

common_args=(
  --duration-seconds 10
  --sample-interval-ms 1
  --books 2
  --seed 12345
  --data-dir "${data_dir}"
  --hawkes-rates-file "${rates_file}"
  --quote-interval-ms 150
  --quote-quantity 75
  --quote-levels 7
  --quote-growth 2
  --enable-shared-mm-hedging
  --exposure-threshold 500
  --max-hedge-quantity 1000
)
shock_args=(
  --shock-time-ns 5000000000
  --shock-book 1
  --shock-side sell
  --shock-quantity 5000
)

"${build_dir}/sequential_multi_asset_lob" \
  "${common_args[@]}" \
  --output-dir "${output_root}/control_sequential"

"${build_dir}/sequential_multi_asset_lob" \
  "${common_args[@]}" \
  "${shock_args[@]}" \
  --output-dir "${output_root}/shock_sequential"

"${mpi_launcher}" --bind-to none --map-by slot:OVERSUBSCRIBE -n "${mpi_ranks}" \
  "${build_dir}/exact_mpi_multi_asset_lob" \
  "${common_args[@]}" \
  --output-dir "${output_root}/control_exact_mpi"

"${mpi_launcher}" --bind-to none --map-by slot:OVERSUBSCRIBE -n "${mpi_ranks}" \
  "${build_dir}/exact_mpi_multi_asset_lob" \
  "${common_args[@]}" \
  "${shock_args[@]}" \
  --output-dir "${output_root}/shock_exact_mpi"

cmp "${output_root}/control_sequential/sequential_multi_asset_state_trace.csv" \
    "${output_root}/control_exact_mpi/exact_mpi_multi_asset_state_trace.csv"
cmp "${output_root}/shock_sequential/sequential_multi_asset_state_trace.csv" \
    "${output_root}/shock_exact_mpi/exact_mpi_multi_asset_state_trace.csv"

python3 "${project_dir}/scripts/analyze_liquidity_shock.py" \
  --control-trace "${output_root}/control_sequential/sequential_multi_asset_state_trace.csv" \
  --shock-trace "${output_root}/shock_sequential/sequential_multi_asset_state_trace.csv" \
  --shock-time-ns 5000000000 \
  --shock-book 1 \
  --recovery-window-samples 100 \
  --output-json "${output_root}/liquidity_shock_metrics.json" \
  --output-csv "${output_root}/liquidity_shock_metrics.csv"
