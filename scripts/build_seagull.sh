#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_JOBS="${BUILD_JOBS:-16}"

module purge
module load openmpi

if command -v ninja >/dev/null 2>&1; then
  CMAKE_GENERATOR_NAME="Ninja"
else
  CMAKE_GENERATOR_NAME="Unix Makefiles"
fi

prepare_build_directory() {
  local build_dir="$1"
  local cache_file="$build_dir/CMakeCache.txt"
  local existing_generator=""

  if [[ -f "$cache_file" ]]; then
    existing_generator="$(
      sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$cache_file" | head -n 1
    )"
  fi

  if [[ -d "$build_dir" && "$existing_generator" != "$CMAKE_GENERATOR_NAME" ]]; then
    printf 'Removing incomplete or incompatible build directory: %s\n' "$build_dir"
    cmake -E remove_directory "$build_dir"
  fi
}

MPI_BUILD_DIR="$PROJECT_DIR/build-mpi"
OPENMP_BUILD_DIR="$PROJECT_DIR/build-openmp"

prepare_build_directory "$MPI_BUILD_DIR"
prepare_build_directory "$OPENMP_BUILD_DIR"

printf 'CMake generator: %s\n' "$CMAKE_GENERATOR_NAME"

cmake -S "$PROJECT_DIR" -B "$MPI_BUILD_DIR" -G "$CMAKE_GENERATOR_NAME" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOB_REQUIRE_MPI=ON \
  -DLOB_FORCE_MPI_STUB=OFF \
  -DLOB_ENABLE_OPENMP=ON \
  -DLOB_BUILD_TESTS=ON
cmake --build "$MPI_BUILD_DIR" --parallel "$BUILD_JOBS"
ctest --test-dir "$MPI_BUILD_DIR" --output-on-failure

cmake -S "$PROJECT_DIR" -B "$OPENMP_BUILD_DIR" -G "$CMAKE_GENERATOR_NAME" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOB_REQUIRE_MPI=OFF \
  -DLOB_FORCE_MPI_STUB=ON \
  -DLOB_ENABLE_OPENMP=ON \
  -DLOB_BUILD_TESTS=OFF
cmake --build "$OPENMP_BUILD_DIR" --parallel "$BUILD_JOBS"

printf 'Built:\n  %s\n  %s\n' \
  "$PROJECT_DIR/build-mpi/lob_mpi" \
  "$PROJECT_DIR/build-openmp/lob_openmp"
