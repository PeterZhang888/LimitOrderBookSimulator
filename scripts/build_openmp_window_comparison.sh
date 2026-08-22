#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_JOBS="${BUILD_JOBS:-16}"
source "$PROJECT_DIR/hpc/seagull/load_environment.sh"

if command -v ninja >/dev/null 2>&1; then
  GENERATOR=Ninja
else
  GENERATOR="Unix Makefiles"
fi

build_one() {
  local directory=$1
  local window_only=$2
  cmake -E remove_directory "$directory"
  cmake -S "$PROJECT_DIR" -B "$directory" -G "$GENERATOR" \
    -DCMAKE_CXX_COMPILER="$LOB_MPI_CXX_COMPILER" \
    -DMPI_CXX_COMPILER="$LOB_MPI_CXX_COMPILER" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLOB_REQUIRE_MPI=ON \
    -DLOB_FORCE_MPI_STUB=OFF \
    -DLOB_ENABLE_OPENMP=ON \
    -DLOB_OPENMP_WINDOW_ONLY="$window_only" \
    -DLOB_BUILD_TESTS=ON
  cmake --build "$directory" --parallel "$BUILD_JOBS"
  ctest --test-dir "$directory" --output-on-failure
  test -x "$directory/lob_mpi"
}

mkdir -p "$PROJECT_DIR/build-comparison"
build_one "$PROJECT_DIR/build-comparison/all-phases" OFF
build_one "$PROJECT_DIR/build-comparison/window-only" ON

printf 'Built matched OpenMP comparison executables:\n  %s\n  %s\n' \
  "$PROJECT_DIR/build-comparison/all-phases/lob_mpi" \
  "$PROJECT_DIR/build-comparison/window-only/lob_mpi"
