#!/bin/bash
#SBATCH -J lob_validator
#SBATCH -o ./lob_validator_%j.out
#SBATCH -e ./lob_validator_%j.err
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --get-user-env

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module purge
module load gcc/13.3.0-gcc-8.5.0-twhwkn6
module load mpi/2021.12
export I_MPI_CXX=g++

STRATEGY_ID=${1:-1}
BASE_SEED=${BASE_SEED:-12345}

mkdir -p results configs logs
srun ./validator "${STRATEGY_ID}" "${BASE_SEED}"
