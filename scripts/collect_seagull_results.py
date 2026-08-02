#!/usr/bin/env python3
"""Combine independently scheduled Seagull jobs into thesis-ready CSV tables."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"invalid or missing {field} in result row") from error


def median(rows: list[dict[str, str]], field: str) -> float:
    return statistics.median(as_float(row, field) for row in rows)


def mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.mean(as_float(row, field) for row in rows)


def sample_sd(rows: list[dict[str, str]], field: str) -> float:
    values = [as_float(row, field) for row in rows]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def experiment_from_path(root: Path, path: Path) -> str:
    parts = path.relative_to(root).parts
    try:
        index = parts.index("partials")
        return parts[index + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"cannot infer experiment from {path}") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    raw_paths = sorted((root / "partials").glob("*/*/raw.csv"))
    if not raw_paths:
        raise SystemExit(f"no partial raw.csv files found below {root / 'partials'}")

    rows: list[dict[str, str]] = []
    for path in raw_paths:
        experiment = experiment_from_path(root, path)
        for row in read_rows(path):
            row["experiment"] = experiment
            row["source_file"] = str(path.relative_to(root))
            rows.append(row)

    write_rows(root / "combined_raw.csv", rows)

    # Verify deterministic rank invariance wherever the same scientific case
    # was intentionally run with multiple MPI sizes.
    hash_groups: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in rows:
        if row["experiment"] not in {"validation", "strong", "full_day"}:
            continue
        key = tuple(
            row.get(field, "")
            for field in (
                "experiment",
                "assets",
                "simulated_seconds",
                "window_ms",
                "risk_limit_per_asset",
                "shared_mm_mode",
                "shock_mode",
                "seed",
            )
        )
        hash_groups[key].add(row["state_hash"])
    bad_hashes = {key: values for key, values in hash_groups.items() if len(values) != 1}
    if bad_hashes:
        first_key, first_values = next(iter(bad_hashes.items()))
        raise RuntimeError(
            "rank invariance failed during collection: "
            f"case={first_key} hashes={sorted(first_values)}"
        )

    performance_groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["experiment"] not in {"validation", "strong", "weak", "full_day"}:
            continue
        key = tuple(
            row.get(field, "")
            for field in (
                "experiment",
                "assets",
                "ranks",
                "simulated_seconds",
                "risk_limit_per_asset",
                "shared_mm_mode",
                "shock_mode",
            )
        )
        performance_groups[key].append(row)

    median_wall = {key: median(group, "wall_seconds") for key, group in performance_groups.items()}
    weak_baselines: dict[tuple[str, ...], float] = {}
    for key, wall in median_wall.items():
        experiment, _assets, ranks, duration, risk, mm_mode, shock_mode = key
        if experiment == "weak" and ranks == "1":
            weak_baselines[(experiment, duration, risk, mm_mode, shock_mode)] = wall
    performance_rows: list[dict[str, str]] = []
    for key in sorted(performance_groups):
        group = performance_groups[key]
        experiment, assets, ranks, duration, risk, mm_mode, shock_mode = key
        wall = median_wall[key]
        baseline_key = (
            experiment,
            assets,
            "1",
            duration,
            risk,
            mm_mode,
            shock_mode,
        )
        baseline = median_wall.get(baseline_key)
        if experiment == "weak":
            baseline = weak_baselines.get(
                (experiment, duration, risk, mm_mode, shock_mode)
            )
        speedup = baseline / wall if baseline is not None else float("nan")
        efficiency = speedup / int(ranks) if baseline is not None else float("nan")
        processed = median(group, "processed_orders")
        performance_rows.append(
            {
                "experiment": experiment,
                "assets": assets,
                "ranks": ranks,
                "simulated_seconds": duration,
                "repetitions": str(len(group)),
                "median_wall_seconds": f"{wall:.9f}",
                "speedup_vs_rank1": f"{speedup:.9f}",
                "parallel_efficiency": f"{efficiency:.9f}",
                "weak_scaling_efficiency": (
                    f"{speedup:.9f}" if experiment == "weak" else "nan"
                ),
                "median_processed_orders": f"{processed:.0f}",
                "median_orders_per_wall_second": f"{processed / wall:.3f}",
                "median_communication_fraction": (
                    f"{median(group, 'communication_fraction'):.9f}"
                ),
                "median_max_compute_seconds": (
                    f"{median(group, 'max_compute_seconds'):.9f}"
                ),
                "median_max_communication_seconds": (
                    f"{median(group, 'max_communication_seconds'):.9f}"
                ),
                "risk_limit_per_asset": risk,
                "shared_mm_mode": mm_mode,
                "shock_mode": shock_mode,
            }
        )
    write_rows(root / "performance_summary.csv", performance_rows)

    science_fields = (
        "peak_affected_fraction",
        "peak_mean_spread_bps",
        "final_mean_spread_bps",
        "final_mean_top_depth",
        "final_shared_gross_exposure",
        "minimum_shared_quote_scale",
        "withdrawal_windows",
    )
    science_groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["experiment"] != "science":
            continue
        key = tuple(
            row.get(field, "")
            for field in (
                "assets",
                "ranks",
                "simulated_seconds",
                "risk_limit_per_asset",
                "shared_mm_mode",
                "shock_mode",
            )
        )
        science_groups[key].append(row)

    science_rows: list[dict[str, str]] = []
    for key in sorted(science_groups):
        group = science_groups[key]
        assets, ranks, duration, risk, mm_mode, shock_mode = key
        output = {
            "assets": assets,
            "ranks": ranks,
            "simulated_seconds": duration,
            "risk_limit_per_asset": risk,
            "shared_mm_mode": mm_mode,
            "shock_mode": shock_mode,
            "independent_seeds": str(len({row.get('seed', '') for row in group})),
            "mean_wall_seconds": f"{mean(group, 'wall_seconds'):.9f}",
        }
        for field in science_fields:
            output[f"mean_{field}"] = f"{mean(group, field):.9f}"
            output[f"sd_{field}"] = f"{sample_sd(group, field):.9f}"
        science_rows.append(output)
    write_rows(root / "science_summary.csv", science_rows)

    experiments = sorted({row["experiment"] for row in rows})
    report_lines = [
        f"partial_csv_files={len(raw_paths)}",
        f"raw_rows={len(rows)}",
        f"experiments={','.join(experiments)}",
        f"rank_invariance_cases={len(hash_groups)}",
        f"rank_invariance_failures={len(bad_hashes)}",
        f"performance_rows={len(performance_rows)}",
        f"science_rows={len(science_rows)}",
        f"combined_raw={root / 'combined_raw.csv'}",
        f"performance_summary={root / 'performance_summary.csv'}",
        f"science_summary={root / 'science_summary.csv'}",
    ]
    (root / "campaign_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(report_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
