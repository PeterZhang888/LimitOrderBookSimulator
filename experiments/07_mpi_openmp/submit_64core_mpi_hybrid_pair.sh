#!/usr/bin/env bash
#SBATCH --job-name=lob-64core-mpi-hybrid
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results/seagull/${SLURM_JOB_ID}_64core_mpi_hybrid}"
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

export OMP_WAIT_POLICY=ACTIVE
export OMP_MAX_ACTIVE_LEVELS=1
unset GOMP_SPINCOUNT || true

source "$PROJECT_DIR/hpc/seagull/common.sh"

common_arguments=(
  --partition cyclic
  --synchronous-observations
  --disable-persistent-risk-collective
  --risk-lookahead-max-windows 0
  --openmp-schedule dynamic1
  --shared-inventory-policy gross_pooled
  --shared-quote-multiplier 2.00
)

control=mpi_64x1
treatment=hybrid_32x2_all_phases
mkdir -p "$RESULT_ROOT"
printf 'block,position,variant\n' > "$RESULT_ROOT/run_order.csv"

for block in 1 2 3 4 5 6 7; do
  if (( block % 2 == 1 )); then
    order=("$control" "$treatment")
  else
    order=("$treatment" "$control")
  fi
  for position_index in 0 1; do
    variant="${order[$position_index]}"
    position=$((position_index + 1))
    printf '%d,%d,%s\n' "$block" "$position" "$variant" \
      >> "$RESULT_ROOT/run_order.csv"
    case "$variant" in
      mpi_64x1)
        run_variant "$variant/block_$block" 64 1 \
          "${common_arguments[@]}"
        ;;
      hybrid_32x2_all_phases)
        run_variant "$variant/block_$block" 32 2 \
          "${common_arguments[@]}"
        ;;
      *)
        printf 'ERROR: unknown layout: %s\n' "$variant" >&2
        exit 1
        ;;
    esac
  done
done

python3 "$PROJECT_DIR/scripts/summarize_layout_pair.py" \
  "$RESULT_ROOT" \
  "$control" "$treatment" \
  64 1 32 2
