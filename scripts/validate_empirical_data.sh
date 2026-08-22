#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${EMPIRICAL_DATA_DIR:-$PROJECT_DIR/data/empirical}"
EXPECTED_ASSETS=1480

UNIVERSE="$DATA_DIR/universe.csv"
BACKGROUND="$DATA_DIR/background_policy.csv"
VALUE="$DATA_DIR/value_policy.csv"
CLUSTERS="$DATA_DIR/clusters.csv"

for file in "$UNIVERSE" "$BACKGROUND" "$VALUE" "$CLUSTERS"; do
  [[ -s "$file" ]] || {
    printf 'ERROR: required empirical input is missing: %s\n' "$file" >&2
    exit 1
  }
  rows=$(( $(wc -l < "$file") - 1 ))
  if (( rows != EXPECTED_ASSETS )); then
    printf 'ERROR: %s has %d asset rows; expected %d.\n' \
      "$file" "$rows" "$EXPECTED_ASSETS" >&2
    exit 1
  fi
done

TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/lob-data-validation.XXXXXX")
trap 'rm -rf "$TEMP_DIR"' EXIT

extract_symbols() {
  awk -F, 'NR > 1 {gsub(/\r$/, "", $1); print $1}' "$1" | sort
}

extract_symbols "$BACKGROUND" > "$TEMP_DIR/background.symbols"
extract_symbols "$VALUE" > "$TEMP_DIR/value.symbols"
extract_symbols "$CLUSTERS" > "$TEMP_DIR/clusters.symbols"
awk -F, 'NR > 1 {gsub(/\r$/, "", $2); print $2}' "$UNIVERSE" | sort \
  > "$TEMP_DIR/universe.symbols"

for list in "$TEMP_DIR"/*.symbols; do
  if [[ $(sort -u "$list" | wc -l) -ne $EXPECTED_ASSETS ]]; then
    printf 'ERROR: duplicate or missing asset names in %s.\n' "$list" >&2
    exit 1
  fi
  cmp -s "$TEMP_DIR/universe.symbols" "$list" || {
    printf 'ERROR: asset names disagree across the empirical inputs.\n' >&2
    exit 1
  }
done

required_distributions=(
  cancel_ask_distance_distribution.txt
  cancel_ask_quantity_distribution.txt
  cancel_bid_distance_distribution.txt
  cancel_bid_quantity_distribution.txt
  limit_buy_distance_distribution.txt
  limit_buy_quantity_distribution.txt
  limit_sell_distance_distribution.txt
  limit_sell_quantity_distribution.txt
  market_buy_quantity_distribution.txt
  market_sell_quantity_distribution.txt
)

expected_id=0
while IFS=, read -r book_id symbol data_path rates_path _; do
  book_id=${book_id//$'\r'/}
  symbol=${symbol//$'\r'/}
  data_path=${data_path//$'\r'/}
  rates_path=${rates_path//$'\r'/}
  [[ "$book_id" == "$expected_id" ]] || {
    printf 'ERROR: expected book_id %d but found %s.\n' \
      "$expected_id" "$book_id" >&2
    exit 1
  }
  [[ "$data_path" = /* ]] || data_path="$PROJECT_DIR/$data_path"
  [[ "$rates_path" = /* ]] || rates_path="$PROJECT_DIR/$rates_path"
  [[ -s "$rates_path" ]] || {
    printf 'ERROR: missing Hawkes-rate input for %s: %s\n' \
      "$symbol" "$rates_path" >&2
    exit 1
  }
  for name in "${required_distributions[@]}"; do
    [[ -s "$data_path/$name" ]] || {
      printf 'ERROR: missing distribution for %s: %s\n' \
        "$symbol" "$data_path/$name" >&2
      exit 1
    }
  done
  expected_id=$((expected_id + 1))
done < <(tail -n +2 "$UNIVERSE")

while IFS=, read -r symbol _ policy buy_improvement sell_improvement _; do
  for relative in "$policy" "$buy_improvement" "$sell_improvement"; do
    relative=${relative//$'\r'/}
    [[ -s "$DATA_DIR/$relative" ]] || {
      printf 'ERROR: missing queue policy for %s: %s\n' \
        "$symbol" "$DATA_DIR/$relative" >&2
      exit 1
    }
  done
done < <(tail -n +2 "$BACKGROUND")

if find "$DATA_DIR" -type f \
    \( -iname '*.nasdaq_itch50' -o -iname '*.itch' -o -iname '*.itch.gz' \) \
    -print -quit | grep -q .; then
  printf 'ERROR: raw exchange-message files must not be stored here.\n' >&2
  exit 1
fi

printf 'Empirical runtime inputs: PASS\n'
printf 'Assets: %d\n' "$expected_id"
printf 'Universe: %s\n' "$UNIVERSE"
