#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="${1:-${project_root}/build-gpu}"
result_file="${2:-${project_root}/gpu_agent_sweep.txt}"

cmake -S "${project_root}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release -DLOB_BUILD_TESTS=OFF
cmake --build "${build_dir}" --target gpu_agent_benchmark -j 4

: > "${result_file}"
for specification in \
    "1024 4096" \
    "8192 4096" \
    "65536 1024" \
    "262144 256" \
    "1048576 64"
do
    read -r agents steps <<< "${specification}"
    {
        echo "case_agents=${agents}"
        "${build_dir}/gpu_agent_benchmark" \
            --agents "${agents}" --steps "${steps}" --books 4
        echo
    } | tee -a "${result_file}"
done

echo "wrote ${result_file}"
