#!/usr/bin/env bash
set -Eeuo pipefail

: "${LOB_PLACEMENT_DIR:?set LOB_PLACEMENT_DIR for the current run}"
: "${OMPI_COMM_WORLD_RANK:?this wrapper must be launched by Open MPI}"

mkdir -p "$LOB_PLACEMENT_DIR"
cpu_list=$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)
[[ -n "$cpu_list" ]] || {
  printf 'ERROR: cannot read the allowed CPU list for rank %s.\n' \
    "$OMPI_COMM_WORLD_RANK" >&2
  exit 1
}

placement_file="$LOB_PLACEMENT_DIR/rank_${OMPI_COMM_WORLD_RANK}.txt"
printf '%s|%s|%s\n' \
  "$(hostname -s)" "$OMPI_COMM_WORLD_RANK" "$cpu_list" \
  > "$placement_file"

exec "$@"
