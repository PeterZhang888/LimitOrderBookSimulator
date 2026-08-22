#!/usr/bin/env bash
#SBATCH --job-name=lob-stall-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}}"
CAPTURE_JOB="${1:?pass the completed diagnostic job number}"
RESULT_ROOT="$PROJECT_DIR/results/seagull/${CAPTURE_JOB}_collective_stall_diagnostic"

cd "$PROJECT_DIR"
test -s "$RESULT_ROOT/repetitions.csv" || {
  printf 'ERROR: repetitions file is missing under %s.\n' "$RESULT_ROOT" >&2
  exit 1
}
test -s "$RESULT_ROOT/CAPTURED_STALL.env" || {
  printf 'ERROR: job %s did not record a stalled repetition.\n' \
    "$CAPTURE_JOB" >&2
  exit 1
}

normal_rep=$(awk -F, 'NR > 1 && $3 == "normal" {print $1; exit}' \
  "$RESULT_ROOT/repetitions.csv")
[[ "$normal_rep" =~ ^[0-9]+$ ]] || {
  printf 'ERROR: no normal repetition was recorded.\n' >&2
  exit 1
}
source "$RESULT_ROOT/CAPTURED_STALL.env"
[[ "${STALL_REPETITION:-}" =~ ^[0-9]+$ ]] || {
  printf 'ERROR: the stalled repetition number is invalid.\n' >&2
  exit 1
}

python3 "$PROJECT_DIR/scripts/summarize_boundary_arrivals.py" \
  "$RESULT_ROOT/boundary_summary.csv" \
  --expected-ranks 32 \
  --expected-boundaries 23401 \
  "$RESULT_ROOT"/repetition_*/boundary_arrivals.csv

python3 "$PROJECT_DIR/scripts/summarize_window_profile.py" \
  "$RESULT_ROOT/profile_analysis" \
  --control-label normal \
  --control-csv \
    "$RESULT_ROOT/repetition_${normal_rep}/window_phase_profile.csv" \
  --control-run "$RESULT_ROOT/repetition_${normal_rep}/run.txt" \
  --control-ranks 32 \
  --treatment-label stall \
  --treatment-csv \
    "$RESULT_ROOT/repetition_${STALL_REPETITION}/window_phase_profile.csv" \
  --treatment-run "$RESULT_ROOT/repetition_${STALL_REPETITION}/run.txt" \
  --treatment-ranks 32 \
  --expected-windows 23400

printf 'COLLECTIVE STALL ANALYSIS: PASS\n'
printf 'RESULT_ROOT=%s\n' "$RESULT_ROOT"
