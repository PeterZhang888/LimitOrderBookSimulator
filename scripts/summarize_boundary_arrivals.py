#!/usr/bin/env python3
"""Validate and summarize per-rank risk-boundary timing records."""

import argparse
import csv
import math
import pathlib
import statistics


HEADER = (
    "boundary_index",
    "time_seconds",
    "rank",
    "arrival_seconds",
    "work_interval_seconds",
    "work_interval_spread_seconds",
    "collective_seconds",
)


def fail(message):
    raise SystemExit(message)


def number(text, context):
    try:
        value = float(text)
    except (TypeError, ValueError):
        fail("{} is not numeric: {!r}".format(context, text))
    if not math.isfinite(value) or value < 0.0:
        fail("{} must be finite and nonnegative: {!r}".format(context, text))
    return value


def integer(text, context):
    value = number(text, context)
    result = int(value)
    if value != float(result):
        fail("{} is not an integer: {!r}".format(context, text))
    return result


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(path, expected_rank_count, expected_boundaries):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != HEADER:
            fail("unexpected header in {}".format(path))
        grouped = {}
        ranks = set()
        for line_number, row in enumerate(reader, start=2):
            context = "{}:{}".format(path, line_number)
            boundary = integer(row["boundary_index"], context)
            rank = integer(row["rank"], context)
            values = {
                name: number(row[name], context + " " + name)
                for name in HEADER
                if name not in ("boundary_index", "rank")
            }
            key = (boundary, rank)
            if key in grouped:
                fail("duplicate boundary/rank row in {}: {}".format(path, key))
            grouped[key] = values
            ranks.add(rank)
    if not grouped:
        fail("no records in {}".format(path))
    contiguous_ranks = set(range(max(ranks) + 1))
    if ranks != contiguous_ranks:
        fail("rank set is incomplete in {}".format(path))
    boundaries = sorted(set(key[0] for key in grouped))
    # The simulator increments its boundary counter before recording the
    # initial t=0 reduction, so a full session is numbered 1, ..., 23,401.
    if boundaries != list(range(1, boundaries[-1] + 1)):
        fail("boundary indices are incomplete in {}".format(path))
    if len(ranks) != expected_rank_count:
        fail("{} contains {} ranks; expected {}".format(
            path, len(ranks), expected_rank_count))
    if len(boundaries) != expected_boundaries:
        fail("{} contains {} boundaries; expected {}".format(
            path, len(boundaries), expected_boundaries))

    rank_collective_totals = {rank: 0.0 for rank in ranks}
    spreads = []
    maximum_collectives = []
    minimum_collectives = []
    for boundary in boundaries:
        rows = [grouped[(boundary, rank)] for rank in sorted(ranks)]
        if len(rows) != len(ranks):
            fail("rank coverage is incomplete at boundary {} in {}".format(
                boundary, path))
        calculated_spread = max(row["work_interval_seconds"] for row in rows) \
            - min(row["work_interval_seconds"] for row in rows)
        recorded_spreads = set(row["work_interval_spread_seconds"] for row in rows)
        if len(recorded_spreads) != 1:
            fail("recorded work spread differs across ranks at boundary {}".format(
                boundary))
        recorded_spread = next(iter(recorded_spreads))
        tolerance = 1.0e-12 + 1.0e-9 * max(calculated_spread, recorded_spread)
        if abs(calculated_spread - recorded_spread) > tolerance:
            fail("recorded work spread is inconsistent at boundary {}".format(
                boundary))
        spreads.append(recorded_spread)
        collectives = []
        for rank, row in zip(sorted(ranks), rows):
            elapsed = row["collective_seconds"]
            rank_collective_totals[rank] += elapsed
            collectives.append(elapsed)
        maximum_collectives.append(max(collectives))
        minimum_collectives.append(min(collectives))

    totals = list(rank_collective_totals.values())
    return {
        "file": str(path),
        "ranks": len(ranks),
        "boundaries": len(boundaries),
        "sum_work_interval_spread": sum(spreads),
        "median_work_interval_spread": statistics.median(spreads),
        "p99_work_interval_spread": percentile(spreads, 0.99),
        "maximum_work_interval_spread": max(spreads),
        "minimum_rank_collective_total": min(totals),
        "mean_rank_collective_total": statistics.mean(totals),
        "maximum_rank_collective_total": max(totals),
        "sum_boundary_minimum_collective": sum(minimum_collectives),
        "sum_boundary_maximum_collective": sum(maximum_collectives),
        "sum_collective_rank_spread": sum(
            high - low for low, high in zip(
                minimum_collectives, maximum_collectives)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_csv", type=pathlib.Path)
    parser.add_argument("--expected-ranks", required=True, type=int)
    parser.add_argument("--expected-boundaries", required=True, type=int)
    parser.add_argument("input_csv", nargs="+", type=pathlib.Path)
    args = parser.parse_args()
    if args.expected_ranks < 1 or args.expected_boundaries < 1:
        fail("expected ranks and boundaries must be positive")
    summaries = [
        summarize(path, args.expected_ranks, args.expected_boundaries)
        for path in args.input_csv
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0])
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print("boundary summary: {}".format(args.output_csv))


if __name__ == "__main__":
    main()
