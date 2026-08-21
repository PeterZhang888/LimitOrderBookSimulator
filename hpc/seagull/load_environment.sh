#!/usr/bin/env bash

module purge
module load openmpi

LOB_MPI_CXX_COMPILER="$(command -v mpicxx)"
read -r LOB_OPENMP_CXX_COMPILER _ <<< "$(mpicxx --showme:command)"

if [[ -z "$LOB_MPI_CXX_COMPILER" || ! -x "$LOB_MPI_CXX_COMPILER" ]]; then
  printf 'ERROR: mpicxx is unavailable after loading OpenMPI.\n' >&2
  return 1 2>/dev/null || exit 1
fi

if [[ "$LOB_OPENMP_CXX_COMPILER" != /* ]]; then
  LOB_OPENMP_CXX_COMPILER="$(command -v "$LOB_OPENMP_CXX_COMPILER")"
fi

if [[ -z "$LOB_OPENMP_CXX_COMPILER" || ! -x "$LOB_OPENMP_CXX_COMPILER" ]]; then
  printf 'ERROR: the C++ compiler used by mpicxx could not be resolved.\n' >&2
  return 1 2>/dev/null || exit 1
fi

read -r -a LOB_MPI_LIBRARY_DIRS <<< "$(mpicxx --showme:libdirs)"
if (( ${#LOB_MPI_LIBRARY_DIRS[@]} == 0 )); then
  printf 'ERROR: mpicxx did not report its runtime library directory.\n' >&2
  return 1 2>/dev/null || exit 1
fi

for LOB_MPI_LIBRARY_DIR in "${LOB_MPI_LIBRARY_DIRS[@]}"; do
  if [[ ! -d "$LOB_MPI_LIBRARY_DIR" ]]; then
    printf 'ERROR: MPI runtime library directory does not exist: %s\n' \
      "$LOB_MPI_LIBRARY_DIR" >&2
    return 1 2>/dev/null || exit 1
  fi
done

LOB_MPI_LIBRARY_PATH="$(
  IFS=:
  printf '%s' "${LOB_MPI_LIBRARY_DIRS[*]}"
)"
export LD_LIBRARY_PATH="${LOB_MPI_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
