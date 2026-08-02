#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Audit and summarize the paired diagnostic-collective communication A/B test."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


IDENTITY_FIELDS = (
    "assets", "ranks", "repetition", "seed", "risk_limit_per_asset",
    "shared_mm_mode", "shock_mode", "control_scenario",
)
SEMANTIC_FIELDS = (
    "state_hash", "processed_orders", "trades", "local_mm_refresh_boundaries",
    "shock_requested_quantity", "shock_executed_quantity",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def identity(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in IDENTITY_FIELDS)


def finite(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as error:
        raise SystemExit(f"invalid {field} in ablation row: {row}") from error
    if not math.isfinite(value):
        raise SystemExit(f"non-finite {field} in ablation row")
    return value


def index_rows(
    rows: list[dict[str, str]], label: str,
) -> dict[tuple[str, ...], dict[str, str]]:
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = identity(row)
        if key in indexed:
            raise SystemExit(f"duplicate {label} ablation identity: {key}")
        indexed[key] = row
    return indexed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = index_rows(read_rows(args.baseline), "baseline")
    optimized = index_rows(read_rows(args.optimized), "optimized")
    if baseline.keys() != optimized.keys():
        missing = sorted(baseline.keys() - optimized.keys())
        extra = sorted(optimized.keys() - baseline.keys())
        raise SystemExit(f"unpaired ablation rows: missing={missing}, extra={extra}")

    grouped: dict[tuple[int, int], list[tuple[dict[str, str], dict[str, str]]]] = (
        defaultdict(list)
    )
    for key in sorted(baseline):
        before, after = baseline[key], optimized[key]
        for field in SEMANTIC_FIELDS:
            if before.get(field, "") != after.get(field, ""):
                raise SystemExit(
                    f"semantic mismatch for {key}: {field} "
                    f"{before.get(field)!r} != {after.get(field)!r}"
                )
        grouped[(int(before["assets"]), int(before["ranks"]))].append(
            (before, after)
        )

    fields = [
        "assets", "ranks", "repetitions", "state_hash_verified",
        "baseline_median_wall_seconds", "optimized_median_wall_seconds",
        "wall_speedup", "wall_reduction_percent",
        "baseline_median_mpi_seconds", "optimized_median_mpi_seconds",
        "mpi_reduction_percent", "baseline_collective_calls",
        "optimized_collective_calls", "collective_reduction_percent",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (assets, ranks), pairs in sorted(grouped.items()):
            before_wall = statistics.median(
                finite(before, "wall_seconds") for before, _ in pairs
            )
            after_wall = statistics.median(
                finite(after, "wall_seconds") for _, after in pairs
            )
            before_mpi = statistics.median(
                finite(before, "max_communication_seconds") for before, _ in pairs
            )
            after_mpi = statistics.median(
                finite(after, "max_communication_seconds") for _, after in pairs
            )
            before_calls = statistics.median(
                finite(before, "collective_calls") for before, _ in pairs
            )
            after_calls = statistics.median(
                finite(after, "collective_calls") for _, after in pairs
            )
            writer.writerow({
                "assets": assets,
                "ranks": ranks,
                "repetitions": len(pairs),
                "state_hash_verified": 1,
                "baseline_median_wall_seconds": f"{before_wall:.9f}",
                "optimized_median_wall_seconds": f"{after_wall:.9f}",
                "wall_speedup": f"{before_wall / after_wall:.9f}",
                "wall_reduction_percent": f"{100.0 * (1.0 - after_wall / before_wall):.6f}",
                "baseline_median_mpi_seconds": f"{before_mpi:.9f}",
                "optimized_median_mpi_seconds": f"{after_mpi:.9f}",
                "mpi_reduction_percent": f"{100.0 * (1.0 - after_mpi / before_mpi):.6f}" if before_mpi else "0.000000",
                "baseline_collective_calls": f"{before_calls:.0f}",
                "optimized_collective_calls": f"{after_calls:.0f}",
                "collective_reduction_percent": f"{100.0 * (1.0 - after_calls / before_calls):.6f}",
            })
    print(f"COMMUNICATION ABLATION: PASS ({len(baseline)} paired runs)")
    print(f"summary={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
