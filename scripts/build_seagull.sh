#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_JOBS="${BUILD_JOBS:-16}"
BUILD_LOG_DIR="$PROJECT_DIR/build-logs"
BUILD_LOG="$BUILD_LOG_DIR/seagull-build.log"

if [[ -n "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=no)" ]]; then
  printf 'ERROR: tracked source files have local modifications.\n' >&2
  exit 1
fi
SOURCE_COMMIT=$(git -C "$PROJECT_DIR" rev-parse HEAD)

mkdir -p "$BUILD_LOG_DIR"
: > "$BUILD_LOG"
exec 3>&1

report() {
  printf '%s\n' "$*" >&3
}

build_failed() {
  local status=$?
  trap - ERR
  report "Build and tests: FAIL"
  report "Relevant output:"
  tail -n 120 "$BUILD_LOG" >&3
  report "Complete log: $BUILD_LOG"
  exit "$status"
}

trap build_failed ERR
exec > "$BUILD_LOG" 2>&1

report "Validating frozen runtime inputs..."
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

report "Building and testing the MPI executable..."
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

report "Building and testing the MPI-free OpenMP executable..."
cmake -S "$PROJECT_DIR" -B "$OPENMP_BUILD_DIR" -G "$CMAKE_GENERATOR_NAME" \
  -DCMAKE_CXX_COMPILER="$LOB_OPENMP_CXX_COMPILER" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOB_REQUIRE_MPI=OFF \
  -DLOB_FORCE_MPI_STUB=ON \
  -DLOB_ENABLE_OPENMP=ON \
  -DLOB_BUILD_TESTS=ON
cmake --build "$OPENMP_BUILD_DIR" --parallel "$BUILD_JOBS"
ctest --test-dir "$OPENMP_BUILD_DIR" --output-on-failure

if [[ "$SOURCE_COMMIT" != "$(git -C "$PROJECT_DIR" rev-parse HEAD)"
      || -n "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=no)" ]]; then
  printf 'ERROR: tracked source changed while the executables were being built.\n' >&2
  exit 1
fi
printf '%s\n' "$SOURCE_COMMIT" > "$MPI_BUILD_DIR/source_commit.txt"
printf '%s\n' "$SOURCE_COMMIT" > "$OPENMP_BUILD_DIR/source_commit.txt"

trap - ERR
report "Build and tests: PASS"
report "MPI executable: $PROJECT_DIR/build-mpi/lob_mpi"
report "MPI-free OpenMP executable: $PROJECT_DIR/build-openmp/lob_openmp"
report "Complete log: $BUILD_LOG"
