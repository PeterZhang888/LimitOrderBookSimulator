#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

test -x build-mpi/lob_mpi || {
  printf 'ERROR: build-mpi/lob_mpi is missing; run scripts/build_seagull.sh first.\n' >&2
  exit 1
}
test -x build-openmp/lob_openmp || {
  printf 'ERROR: build-openmp/lob_openmp is missing; run scripts/build_seagull.sh first.\n' >&2
  exit 1
}
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  printf 'ERROR: tracked files have local modifications.\n' >&2
  exit 1
fi

bash scripts/validate_empirical_data.sh
mkdir -p slurm results/seagull

campaign="release_validation_$(date -u +%Y%m%dT%H%M%SZ)"
campaign_root="$PROJECT_DIR/results/seagull/$campaign"
manifest="$campaign_root/submitted_jobs.csv"
state_file="$PROJECT_DIR/results/seagull/LATEST_RELEASE_VALIDATION.env"
mkdir -p "$campaign_root"
printf 'experiment,job_id,nodes,tasks,expected_runs,result_directory\n' > "$manifest"

job_ids=()

submit_case() {
  local experiment=$1
  local nodes=$2
  local tasks=$3
  local expected_runs=$4
  local script=$5
  shift 5
  local result_dir="$campaign_root/$experiment"
  local job

  job=$(
    env \
      PROJECT_DIR="$PROJECT_DIR" \
      RESULT_ROOT="$result_dir" \
      REPETITIONS=1 \
      "$@" \
      sbatch --parsable \
        --chdir="$PROJECT_DIR" \
        --job-name="lob-check-${experiment}" \
        --nodes="$nodes" \
        --ntasks="$tasks" \
        --ntasks-per-node=16 \
        --cpus-per-task=1 \
        --time=2-00:00:00 \
        --exclusive \
        --hint=nomultithread \
        --output="$PROJECT_DIR/slurm/%x-%j.out" \
        --error="$PROJECT_DIR/slurm/%x-%j.err" \
        --export=ALL \
        "$script"
  )
  job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || {
    printf 'ERROR: invalid job identifier for %s: %s\n' \
      "$experiment" "$job" >&2
    exit 1
  }

  job_ids+=("$job")
  printf '%s,%s,%d,%d,%d,%s\n' \
    "$experiment" "$job" "$nodes" "$tasks" "$expected_runs" \
    "$result_dir" >> "$manifest"
}

# Every case runs the complete 23,400-second session. Validation changes only
# the replication count and the maximum allocation size.
submit_case 00_full_synthetic 2 32 1 \
  experiments/00_full_synthetic/submit_seagull.sh \
  RANKS=32 ASSET_COUNT=10000 BACKGROUND_MODEL=legacy

submit_case 01_strong_scaling 2 32 1 \
  experiments/01_strong_scaling/submit_seagull.sh \
  SCALING_RANKS=32 ASSET_COUNT=10000 \
  BASE_CONFIG="$PROJECT_DIR/examples/synthetic/templates.csv" \
  BACKGROUND_MODEL=legacy

submit_case 02_weak_scaling 2 32 1 \
  experiments/02_weak_scaling/submit_seagull.sh \
  RANK_COUNTS_OVERRIDE=32 BOOKS_PER_RANK_OVERRIDE=16

submit_case 03_empirical_scaling 2 32 1 \
  experiments/03_empirical_scaling/submit_seagull.sh \
  RANK_COUNTS_OVERRIDE=32

submit_case 04_rank_ownership 2 32 2 \
  experiments/04_rank_ownership/submit_seagull.sh \
  RANK_COUNTS_OVERRIDE=32

submit_case 05_observation_buffering 1 16 2 \
  experiments/05_observation_buffering/submit_seagull.sh

submit_case 06_fused_metric_scans 2 32 2 \
  experiments/06_fused_metric_scans/submit_seagull.sh \
  RANKS=32

submit_case 07_mpi_openmp_16 1 16 6 \
  experiments/07_mpi_openmp/run_fixed_ownership_matrix.sbatch \
  TOTAL_CORES=16 BLOCK_COUNT=1

submit_case 07_mpi_openmp_32 2 32 6 \
  experiments/07_mpi_openmp/run_fixed_ownership_matrix.sbatch \
  TOTAL_CORES=32 BLOCK_COUNT=1

submit_case 08_risk_collectives 2 32 4 \
  experiments/08_risk_collectives/submit_seagull.sh \
  RANKS=32

submit_case 09_stylised_facts 1 16 1 \
  experiments/09_stylised_facts/submit_seagull.sh

submit_case 10_inventory_policy 2 32 4 \
  experiments/10_inventory_policy/submit_seagull.sh \
  RANKS=32 OPENING_SEEDS_OVERRIDE=30300130 \
  FULL_SEEDS_OVERRIDE=20200130 ETA_VALUES_OVERRIDE=2.00

job_list=$(IFS=,; printf '%s' "${job_ids[*]}")
{
  printf 'RELEASE_VALIDATION_CAMPAIGN=%q\n' "$campaign"
  printf 'RELEASE_VALIDATION_MANIFEST=%q\n' "$manifest"
  printf 'RELEASE_VALIDATION_JOBS=%q\n' "$job_list"
} > "$state_file"

printf 'Submitted full-session release validation.\n'
printf 'Manifest: %s\n' "$manifest"
column -s, -t < "$manifest"
squeue -j "$job_list" -o "%.18i %.30j %.10T %.12M %.45R"
