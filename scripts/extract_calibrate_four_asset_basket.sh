#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 PATH_TO_ITCH50_GZ [OUTPUT_ROOT]" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "${script_dir}/.." && pwd)
input_file=$1
output_root=${2:-"${project_root}/data"}

exec "${script_dir}/extract_calibrate_multi_asset_day.sh" \
    "${input_file}" 2020-01-30 "${output_root}"
