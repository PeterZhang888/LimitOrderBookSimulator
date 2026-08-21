#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_JOBS="${BUILD_JOBS:-16}"

module purge
module load openmpi

MPI_CXX_COMPILER="$(command -v mpicxx)"
read -r OPENMP_CXX_COMPILER _ <<< "$(mpicxx --showme:command)"

if [[ -z "$MPI_CXX_COMPILER" || ! -x "$MPI_CXX_COMPILER" ]]; then
  printf 'ERROR: mpicxx is unavailable after loading OpenMPI.\n' >&2
  exit 1
fi

if [[ "$OPENMP_CXX_COMPILER" != /* ]]; then
  OPENMP_CXX_COMPILER="$(command -v "$OPENMP_CXX_COMPILER")"
fi

if [[ -z "$OPENMP_CXX_COMPILER" || ! -x "$OPENMP_CXX_COMPILER" ]]; then
  printf 'ERROR: the C++ compiler used by mpicxx could not be resolved.\n' >&2
  exit 1
fi

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
printf 'MPI C++ compiler: %s\n' "$MPI_CXX_COMPILER"
printf 'MPI-free C++ compiler: %s\n' "$OPENMP_CXX_COMPILER"

cmake -S "$PROJECT_DIR" -B "$MPI_BUILD_DIR" -G "$CMAKE_GENERATOR_NAME" \
  -DCMAKE_CXX_COMPILER="$MPI_CXX_COMPILER" \
  -DMPI_CXX_COMPILER="$MPI_CXX_COMPILER" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOB_REQUIRE_MPI=ON \
  -DLOB_FORCE_MPI_STUB=OFF \
  -DLOB_ENABLE_OPENMP=ON \
  -DLOB_BUILD_TESTS=ON
cmake --build "$MPI_BUILD_DIR" --parallel "$BUILD_JOBS"
ctest --test-dir "$MPI_BUILD_DIR" --output-on-failure

cmake -S "$PROJECT_DIR" -B "$OPENMP_BUILD_DIR" -G "$CMAKE_GENERATOR_NAME" \
  -DCMAKE_CXX_COMPILER="$OPENMP_CXX_COMPILER" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOB_REQUIRE_MPI=OFF \
  -DLOB_FORCE_MPI_STUB=ON \
  -DLOB_ENABLE_OPENMP=ON \
  -DLOB_BUILD_TESTS=OFF
cmake --build "$OPENMP_BUILD_DIR" --parallel "$BUILD_JOBS"

printf 'Built:\n  %s\n  %s\n' \
  "$PROJECT_DIR/build-mpi/lob_mpi" \
  "$PROJECT_DIR/build-openmp/lob_openmp"
