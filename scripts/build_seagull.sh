#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_JOBS="${BUILD_JOBS:-16}"

bash "$PROJECT_DIR/scripts/validate_empirical_data.sh"
source "$PROJECT_DIR/hpc/seagull/load_environment.sh"

if command -v ninja >/dev/null 2>&1; then
  CMAKE_GENERATOR_NAME="Ninja"
else
  CMAKE_GENERATOR_NAME="Unix Makefiles"
fi

MPI_BUILD_DIR="$PROJECT_DIR/build-mpi"
OPENMP_BUILD_DIR="$PROJECT_DIR/build-openmp"

for BUILD_DIR in "$MPI_BUILD_DIR" "$OPENMP_BUILD_DIR"; do
  if [[ -d "$BUILD_DIR" ]]; then
    printf 'Removing previous build directory: %s\n' "$BUILD_DIR"
    cmake -E remove_directory "$BUILD_DIR"
  fi
done

printf 'CMake generator: %s\n' "$CMAKE_GENERATOR_NAME"
printf 'MPI C++ compiler: %s\n' "$LOB_MPI_CXX_COMPILER"
printf 'MPI-free C++ compiler: %s\n' "$LOB_OPENMP_CXX_COMPILER"
printf 'MPI runtime libraries: %s\n' "$LOB_MPI_LIBRARY_PATH"

cmake -S "$PROJECT_DIR" -B "$MPI_BUILD_DIR" -G "$CMAKE_GENERATOR_NAME" \
  -DCMAKE_CXX_COMPILER="$LOB_MPI_CXX_COMPILER" \
  -DMPI_CXX_COMPILER="$LOB_MPI_CXX_COMPILER" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOB_REQUIRE_MPI=ON \
  -DLOB_FORCE_MPI_STUB=OFF \
  -DLOB_ENABLE_OPENMP=ON \
  -DLOB_BUILD_TESTS=ON
cmake --build "$MPI_BUILD_DIR" --parallel "$BUILD_JOBS"
ctest --test-dir "$MPI_BUILD_DIR" --output-on-failure

cmake -S "$PROJECT_DIR" -B "$OPENMP_BUILD_DIR" -G "$CMAKE_GENERATOR_NAME" \
  -DCMAKE_CXX_COMPILER="$LOB_OPENMP_CXX_COMPILER" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOB_REQUIRE_MPI=OFF \
  -DLOB_FORCE_MPI_STUB=ON \
  -DLOB_ENABLE_OPENMP=ON \
  -DLOB_BUILD_TESTS=ON
cmake --build "$OPENMP_BUILD_DIR" --parallel "$BUILD_JOBS"
ctest --test-dir "$OPENMP_BUILD_DIR" --output-on-failure

printf 'Built:\n  %s\n  %s\n' \
  "$PROJECT_DIR/build-mpi/lob_mpi" \
  "$PROJECT_DIR/build-openmp/lob_openmp"
