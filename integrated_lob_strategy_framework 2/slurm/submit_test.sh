#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash slurm/submit_test.sh STRATEGY_ID NTASKS" >&2
    echo "Example: bash slurm/submit_test.sh 1 100" >&2
    exit 1
fi

STRATEGY_ID="$1"
NTASKS="$2"
BASE_SEED="${BASE_SEED:-12345}"

sbatch \
  --job-name="lob_test_s${STRATEGY_ID}_${NTASKS}" \
  --output="./lob_test_s${STRATEGY_ID}_${NTASKS}_%j.out" \
  --error="./lob_test_s${STRATEGY_ID}_${NTASKS}_%j.err" \
  --partition=compute \
  --ntasks="${NTASKS}" \
  --cpus-per-task=1 \
  --mem-per-cpu=4G \
  --time=24:00:00 \
  --no-requeue \
  --export=NONE \
  --get-user-env \
  "$(dirname "$0")/submit_test_job.sh" "${STRATEGY_ID}" "${BASE_SEED}"
