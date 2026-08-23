#!/usr/bin/env bash
#SBATCH --job-name=lob-facts
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
REPETITIONS="${REPETITIONS:-1}"
DURATION_SECONDS="${DURATION_SECONDS:-23400}"
SEED="${SEED:-20200130}"
source "$PROJECT_DIR/hpc/seagull/common.sh"

BASE_RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results/seagull/$SLURM_JOB_ID}"
OUTPUT_DIR="$BASE_RESULT_ROOT/stylised_facts"
RESULT_ROOT="$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

run_variant simulation 16 1 \
  --partition cyclic \
  --synchronous-observations \
  --disable-persistent-risk-collective \
  --disable-shared-mm \
  --metrics-interval-ms 1000 \
  --return-panel-prefix "$OUTPUT_DIR/simulated_twice_midpoint" \
  --return-panel-interval-ms 1000

python3 - "$OUTPUT_DIR" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
paths = sorted(root.glob("simulated_twice_midpoint.rank*.csv"))
if len(paths) != 16:
    raise SystemExit(
        f"expected 16 rank-local return panels, observed {len(paths)}"
    )

symbols = set()
expected_observations = 23_401
for path in paths:
    with path.open(newline="", encoding="utf-8") as source:
        rows = csv.reader(source)
        header = next(rows, None)
        if not header or header[0] != "time_seconds" or len(header) < 2:
            raise SystemExit(f"invalid return-panel header: {path}")
        local_symbols = header[1:]
        if len(local_symbols) != len(set(local_symbols)):
            raise SystemExit(f"duplicate asset in return panel: {path}")
        overlap = symbols.intersection(local_symbols)
        if overlap:
            raise SystemExit(f"asset appears on multiple ranks: {sorted(overlap)[:3]}")
        symbols.update(local_symbols)
        count = 0
        first_time = None
        last_time = None
        for row in rows:
            if len(row) != len(header):
                raise SystemExit(f"short return-panel row: {path}")
            time = float(row[0])
            first_time = time if first_time is None else first_time
            last_time = time
            count += 1
        if count != expected_observations or first_time != 0.0 \
                or last_time != 23_400.0:
            raise SystemExit(
                f"incomplete full-session return panel: {path}; "
                f"rows={count}, first={first_time}, last={last_time}"
            )

if len(symbols) != 1_480:
    raise SystemExit(f"expected 1480 assets across return panels; observed {len(symbols)}")
print("Return-panel completeness: PASS")
print(f"Assets: {len(symbols)}")
print(f"Observations per asset: {expected_observations}")
PY

printf 'Full-session stylised-fact simulation completed.\n'
printf 'Result directory: %s\n' "$OUTPUT_DIR"
