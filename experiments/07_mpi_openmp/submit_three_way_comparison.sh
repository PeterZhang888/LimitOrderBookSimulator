#!/usr/bin/env bash
#SBATCH --job-name=lob-omp-three-way
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results/seagull/${SLURM_JOB_ID}_openmp_three_way}"
REPETITIONS=1
DURATION_SECONDS=23400
SEED=20200130
CORES_PER_NODE=16
BACKGROUND_MODEL=queue-reactive-v1
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build-mpi}"

test -x "$BUILD_DIR/lob_mpi" || {
  printf 'ERROR: missing executable: %s\nRun scripts/build_seagull.sh first.\n' \
  "$BUILD_DIR/lob_mpi" >&2
  exit 1
}
if [[ -n "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=no)" ]]; then
  printf 'ERROR: tracked source files have local modifications.\n' >&2
  exit 1
fi

# Hold the OpenMP runtime policy fixed in every treatment.  GOMP_SPINCOUNT is
# removed so a login-shell setting cannot silently alter worker waiting.
export OMP_WAIT_POLICY=ACTIVE
export OMP_MAX_ACTIVE_LEVELS=1
unset GOMP_SPINCOUNT || true

source "$PROJECT_DIR/hpc/seagull/common.sh"

if ! "$BUILD_DIR/lob_mpi" --help | grep -q -- '--openmp-window-only'; then
  printf 'ERROR: the executable predates the three-way OpenMP implementation.\nRun scripts/build_seagull.sh again.\n' >&2
  exit 1
fi

common_arguments=(
  --partition cyclic
  --synchronous-observations
  --disable-persistent-risk-collective
  --openmp-schedule dynamic1
  --shared-inventory-policy gross_pooled
  --shared-quote-multiplier 2.00
)

# A six-sequence balanced order is used for the first six blocks.  The seventh
# block repeats the first sequence because seven system-noise repetitions are
# required.  All treatments use the same executable; only one runtime OpenMP
# treatment flag changes.
orders=(
  "all_phases window_only persistent"
  "window_only persistent all_phases"
  "persistent all_phases window_only"
  "all_phases persistent window_only"
  "persistent window_only all_phases"
  "window_only all_phases persistent"
  "all_phases window_only persistent"
)

mkdir -p "$RESULT_ROOT"
printf 'block,position,variant\n' > "$RESULT_ROOT/run_order.csv"

for order_index in "${!orders[@]}"; do
  block=$((order_index + 1))
  read -r -a block_order <<< "${orders[$order_index]}"
  for position_index in "${!block_order[@]}"; do
    position=$((position_index + 1))
    variant="${block_order[$position_index]}"
    printf '%d,%d,%s\n' "$block" "$position" "$variant" \
      >> "$RESULT_ROOT/run_order.csv"

    variant_arguments=("${common_arguments[@]}")
    case "$variant" in
      all_phases)
        ;;
      window_only)
        variant_arguments+=(--openmp-window-only)
        ;;
      persistent)
        variant_arguments+=(--persistent-openmp-team)
        ;;
      *)
        printf 'ERROR: unknown OpenMP treatment: %s\n' "$variant" >&2
        exit 1
        ;;
    esac

    run_variant "$variant/block_$block" 32 2 \
      "${variant_arguments[@]}"
  done
done

python3 "$PROJECT_DIR/scripts/summarize_openmp_three_way.py" \
  "$RESULT_ROOT"
