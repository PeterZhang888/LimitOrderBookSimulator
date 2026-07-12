#!/bin/bash
set -euo pipefail

module purge
module load gcc/13.3.0-gcc-8.5.0-twhwkn6
module load mpi/2021.12

export I_MPI_CXX=g++

if ! command -v g++ >/dev/null 2>&1; then
    echo "ERROR: g++ not found after loading modules" >&2
    exit 1
fi

if ! command -v mpicxx >/dev/null 2>&1; then
    echo "ERROR: mpicxx not found after loading mpi/2021.12" >&2
    exit 1
fi

if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: mpirun not found after loading mpi/2021.12" >&2
    exit 1
fi

echo "g++: $(which g++)"
echo "mpicxx: $(which mpicxx)"
echo "mpirun: $(which mpirun)"
echo "mpicxx -show:"
mpicxx -show

make clean
make -j 4 all CXX=mpicxx

echo "Compile finished. Built executables:"
ls -lh strategy_1 strategy_2 strategy_3 tuner validator tester hawkes_full_mpi
