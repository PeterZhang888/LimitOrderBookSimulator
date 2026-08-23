#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
state_file="$PROJECT_DIR/results/runs/LATEST_RELEASE_VALIDATION.env"
test -s "$state_file" || {
  printf 'ERROR: no release-validation state file was found.\n' >&2
  exit 1
}
source "$state_file"
test -s "$RELEASE_VALIDATION_MANIFEST" || {
  printf 'ERROR: validation manifest is missing: %s\n' \
    "$RELEASE_VALIDATION_MANIFEST" >&2
  exit 1
}

pending=0
failed=0
printf 'experiment,job_id,state,exit_code,completed_runs,expected_runs\n'
while IFS=, read -r experiment job nodes tasks expected_runs result_dir; do
  [[ "$job" == job_id ]] && continue
  record=$(
    sacct -X -n -P -j "$job" --format=JobIDRaw,State,ExitCode |
      awk -F'|' -v job="$job" '$1 == job {print $2 "|" $3; exit}'
  )
  if [[ -z "$record" ]]; then
    state=UNKNOWN
    exit_code=unknown
  else
    IFS='|' read -r state exit_code <<< "$record"
  fi

  completed_runs=0
  if [[ -d "$result_dir" ]]; then
    completed_runs=$(
      { grep -R -l -E '^lob_(mpi|openmp) ' --include='*.txt' \
          "$result_dir" 2>/dev/null || true; } | wc -l | tr -d ' '
    )
  fi
  printf '%s,%s,%s,%s,%s,%s\n' \
    "$experiment" "$job" "$state" "$exit_code" \
    "$completed_runs" "$expected_runs"

  case "$state" in
    COMPLETED)
      if (( completed_runs != expected_runs )); then
        failed=1
      elif ! python3 "$PROJECT_DIR/scripts/validate_release_result.py" \
          "$result_dir" "$expected_runs" "$experiment" >/dev/null; then
        failed=1
      fi
      ;;
    PENDING|RUNNING|CONFIGURING|COMPLETING)
      pending=1
      ;;
    *)
      failed=1
      ;;
  esac
done < "$RELEASE_VALIDATION_MANIFEST"

if (( failed )); then
  printf 'RELEASE VALIDATION: FAIL\n' >&2
  exit 1
fi
if (( pending )); then
  printf 'RELEASE VALIDATION: IN PROGRESS\n'
  exit 2
fi
printf 'RELEASE VALIDATION: PASS\n'
