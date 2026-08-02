#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="$(hostname -s)"
if [[ "$HOST" == callan* && -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run simulation work on a Callan login/head node." >&2
    echo "Submit submit.sh with sbatch instead." >&2
    exit 2
fi

BUILD_DIR="${PROJECT_DIR}/build_local"
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLOB_REQUIRE_MPI=OFF \
    -DLOB_BUILD_TESTS=ON
cmake --build "$BUILD_DIR" -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
ctest --test-dir "$BUILD_DIR" --output-on-failure
rm -rf "$PROJECT_DIR/results/local_debug"
"$BUILD_DIR/distributed_lob" \
    --profile debug \
    --output-dir "$PROJECT_DIR/results/local_debug"
