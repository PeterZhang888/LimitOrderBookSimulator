#!/usr/bin/env python3
"""Validate the complete output set produced by one release-check job."""

from __future__ import print_function

import csv
import pathlib
import re
import sys


SUMMARY = re.compile(r"^lob_(?:mpi|openmp) ")


def fail(message):
    raise SystemExit("ERROR: {}".format(message))


def summary_fields(path):
    matches = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if SUMMARY.match(line):
                matches.append(line.strip())
    if len(matches) != 1:
        return None
    fields = {}
    for token in matches[0].split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def validate_csv(path, first_header, expected_rows, final_time=None):
    if not path.is_file() or path.stat().st_size == 0:
        fail("missing or empty output: {}".format(path))
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        header = next(reader, None)
        if not header or header[0] != first_header:
            fail("invalid header in {}".format(path))
        count = 0
        last = None
        for row in reader:
            if len(row) != len(header):
                fail("incomplete row in {}".format(path))
            last = row
            count += 1
    if count != expected_rows:
        fail("{} has {} data rows; expected {}".format(
            path, count, expected_rows))
    if final_time is not None:
        if last is None or float(last[0]) != final_time:
            fail("{} does not end at {} seconds".format(path, final_time))


def main():
    if len(sys.argv) != 4:
        fail("usage: validate_release_result.py RESULT_DIR EXPECTED_RUNS EXPERIMENT")
    root = pathlib.Path(sys.argv[1])
    expected_runs = int(sys.argv[2])
    experiment = sys.argv[3]
    if not root.is_dir():
        fail("result directory is missing: {}".format(root))

    completed = []
    for path in sorted(root.rglob("run_*.txt")):
        fields = summary_fields(path)
        if fields is not None:
            completed.append((path, fields))
    if len(completed) != expected_runs:
        fail("{} contains {} completed runs; expected {}".format(
            root, len(completed), expected_runs))

    seen_outputs = set()
    for run_path, fields in completed:
        try:
            duration = int(fields["simulated_seconds"])
            assets = int(fields["assets"])
            int(fields["processed_orders"])
            int(fields["trades"])
            float(fields["shared_signed_mark_to_mid_pnl_usd"])
            float(fields["shared_signed_liquidation_pnl_usd"])
        except (KeyError, ValueError) as error:
            fail("incomplete simulator summary in {}: {}".format(run_path, error))
        if duration != 23400 or assets <= 0:
            fail("{} is not a complete 23,400-second run".format(run_path))
        suffix = run_path.stem[len("run_"):]
        metrics = run_path.with_name("metrics_{}.csv".format(suffix))
        asset_csv = run_path.with_name("assets_{}.csv".format(suffix))
        pair = (metrics.resolve(), asset_csv.resolve())
        if pair in seen_outputs:
            fail("duplicate output pair for {}".format(run_path))
        seen_outputs.add(pair)
        validate_csv(metrics, "time_seconds", duration + 1, float(duration))
        validate_csv(asset_csv, "asset_id", assets)

    if experiment.startswith("06_mpi_openmp"):
        for name in ("comparison.csv", "paired_comparisons.csv"):
            path = root / name
            if not path.is_file() or path.stat().st_size == 0:
                fail("missing OpenMP validation summary: {}".format(path))
    if experiment == "08_stylised_facts":
        panels = sorted(root.rglob("simulated_twice_midpoint.rank*.csv"))
        if len(panels) != 16:
            fail("stylised-fact validation produced {} rank panels; expected 16".format(
                len(panels)))

    print("{}: complete outputs for {} runs".format(experiment, expected_runs))


if __name__ == "__main__":
    main()
