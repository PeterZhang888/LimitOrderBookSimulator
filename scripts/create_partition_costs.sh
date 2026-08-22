#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 2 )); then
  printf 'Usage: %s ASSET_WORK_CSV OUTPUT_CSV\n' "$0" >&2
  exit 2
fi

INPUT=$1
OUTPUT=$2
[[ -s "$INPUT" ]] || {
  printf 'ERROR: per-book work profile is missing: %s\n' "$INPUT" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"
awk -F, '
  BEGIN { OFS="," }
  NR == 1 {
    for (i = 1; i <= NF; ++i) {
      if ($i == "symbol") symbol_col = i
      if ($i == "processing_nanoseconds") cost_col = i
    }
    if (!symbol_col || !cost_col) {
      print "ERROR: work profile lacks symbol or processing_nanoseconds" > "/dev/stderr"
      exit 1
    }
    print "symbol", "partition_weight"
    next
  }
  {
    value = $cost_col + 0
    if (value <= 0) value = 1
    print $symbol_col, value
  }
' "$INPUT" > "$OUTPUT"

rows=$(( $(wc -l < "$OUTPUT") - 1 ))
if (( rows != 1480 )); then
  printf 'ERROR: generated %d cost rows; expected 1480.\n' "$rows" >&2
  exit 1
fi

printf 'Created per-book scheduling costs: %s\n' "$OUTPUT"
