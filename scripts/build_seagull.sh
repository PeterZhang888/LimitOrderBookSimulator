#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_JOBS="${BUILD_JOBS:-16}"

module purge
module load openmpi

cmake -S "$PROJECT_DIR" -B "$PROJECT_DIR/build-mpi" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOB_REQUIRE_MPI=ON \
  -DLOB_FORCE_MPI_STUB=OFF \
  -DLOB_ENABLE_OPENMP=ON \
  -DLOB_BUILD_TESTS=ON
cmake --build "$PROJECT_DIR/build-mpi" -j "$BUILD_JOBS"
ctest --test-dir "$PROJECT_DIR/build-mpi" --output-on-failure

cmake -S "$PROJECT_DIR" -B "$PROJECT_DIR/build-openmp" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOB_REQUIRE_MPI=OFF \
  -DLOB_FORCE_MPI_STUB=ON \
  -DLOB_ENABLE_OPENMP=ON \
  -DLOB_BUILD_TESTS=OFF
cmake --build "$PROJECT_DIR/build-openmp" -j "$BUILD_JOBS"

printf 'Built:\n  %s\n  %s\n' \
  "$PROJECT_DIR/build-mpi/lob_mpi" \
  "$PROJECT_DIR/build-openmp/lob_openmp"
