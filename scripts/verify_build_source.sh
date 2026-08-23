#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if (( $# == 0 )); then
  set -- "$PROJECT_DIR/build-mpi" "$PROJECT_DIR/build-openmp"
fi

current_commit=$(git -C "$PROJECT_DIR" rev-parse HEAD)
if [[ -n "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=no)" ]]; then
  printf 'ERROR: tracked source files have local modifications.\n' >&2
  exit 1
fi

for build_dir in "$@"; do
  stamp="$build_dir/source_commit.txt"
  if [[ ! -s "$stamp" || "$(<"$stamp")" != "$current_commit" ]]; then
    printf 'ERROR: %s was not built from the current source commit.\n' \
      "$build_dir" >&2
    printf 'Run scripts/build_seagull.sh again.\n' >&2
    exit 1
  fi
done
