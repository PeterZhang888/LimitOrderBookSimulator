#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module purge
module load gcc/13.3.0-gcc-8.5.0-twhwkn6
module load mpi/2021.12
export I_MPI_CXX=g++

STRATEGY_ID=${1:-1}
BASE_SEED=${2:-12345}

mkdir -p results configs logs
srun ./tester "${STRATEGY_ID}" "${BASE_SEED}"
