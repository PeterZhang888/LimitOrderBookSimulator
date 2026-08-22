#!/usr/bin/env python3
"""Collect simulator result lines and summarize wall time by configuration."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


RESULT_PREFIXES = ("lob_mpi ", "lob_openmp ")


def parse_result(path: Path, root: Path) -> dict[str, str] | None:
    result_line = None
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if line.startswith(RESULT_PREFIXES):
                result_line = line.strip()
    if result_line is None:
        return None

    record = {
        "configuration": str(path.parent.relative_to(root)),
        "repetition": "",
        "source_log": str(path.relative_to(root)),
        "executable": result_line.split(maxsplit=1)[0],
    }
    match = re.fullmatch(r"run_(\d+)\.txt", path.name)
    if match:
        record["repetition"] = match.group(1)
    for item in result_line.split()[1:]:
        if "=" not in item:
            continue
        name, value = item.split("=", 1)
        record[name] = value
    return record


def write_raw(path: Path, records: list[dict[str, str]]) -> None:
    preferred = [
        "configuration", "repetition", "source_log", "executable",
        "ranks", "worker_threads", "partition", "openmp_schedule",
        "persistent_openmp_team", "buffered_observations",
        "persistent_risk_collective", "nonblocking_risk_collective",
        "risk_lookahead_max_windows", "wall_seconds", "processed_orders",
        "trades", "risk_collective_calls", "risk_boundaries",
        "risk_lookahead_skipped_boundaries", "shared_inventory_policy",
        "shared_quote_multiplier", "final_shared_gross_exposure",
        "maximum_shared_gross_exposure", "shared_terminal_absolute_inventory",
        "shared_signed_liquidation_pnl_usd", "shared_buy_quantity",
        "shared_sell_quantity",
    ]
    all_fields = set().union(*(record.keys() for record in records))
    fields = [name for name in preferred if name in all_fields]
    fields.extend(sorted(all_fields.difference(fields)))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_performance(path: Path, records: list[dict[str, str]]) -> None:
    grouped: dict[str, list[float]] = {}
    for record in records:
        value = record.get("wall_seconds")
        if value is None:
            continue
        grouped.setdefault(record["configuration"], []).append(float(value))
    fields = (
        "configuration", "repetitions", "minimum_wall", "median_wall",
        "maximum_wall", "maximum_minimum_ratio",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for configuration in sorted(grouped):
            values = grouped[configuration]
            minimum = min(values)
            maximum = max(values)
            writer.writerow({
                "configuration": configuration,
                "repetitions": len(values),
                "minimum_wall": f"{minimum:.9f}",
                "median_wall": f"{statistics.median(values):.9f}",
                "maximum_wall": f"{maximum:.9f}",
                "maximum_minimum_ratio": f"{maximum / minimum:.9f}",
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--performance-output", type=Path)
    args = parser.parse_args()

    root = args.result_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"result directory does not exist: {root}")
    records = [
        record for path in sorted(root.rglob("run_*.txt"))
        if (record := parse_result(path, root)) is not None
    ]
    if not records:
        raise SystemExit(f"no completed simulator result lines below {root}")

    raw_output = (args.raw_output or root / "raw_results.csv").resolve()
    performance_output = (
        args.performance_output or root / "performance_summary.csv"
    ).resolve()
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    performance_output.parent.mkdir(parents=True, exist_ok=True)
    write_raw(raw_output, records)
    write_performance(performance_output, records)
    print("Result summary: PASS")
    print(f"Completed runs: {len(records)}")
    print(f"Raw table: {raw_output}")
    print(f"Performance table: {performance_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
