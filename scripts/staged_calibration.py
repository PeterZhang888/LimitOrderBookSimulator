#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Utilities for the three-stage eligibility-set calibration workflow.

The script is dependency-light. It uses scipy.stats.qmc.Sobol when SciPy is
available and falls back to a deterministic Halton design otherwise.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

PARAMETERS = [
    "market_maker_interval_ms",
    "market_maker_min_spread_ticks",
    "momentum_rate_per_second",
    "momentum_threshold_ticks",
    "informed_rate_per_second",
    "informed_signal_precision",
    "institutional_rate_per_second",
    "institutional_participation_cap",
]


def _van_der_corput(index: int, base: int) -> float:
    result = 0.0
    denominator = 1.0
    while index > 0:
        index, remainder = divmod(index, base)
        denominator *= base
        result += remainder / denominator
    return result


def _halton(n: int, dimension: int, seed: int) -> list[list[float]]:
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    offset = max(0, seed) * 17
    return [
        [_van_der_corput(offset + row + 1, primes[col]) for col in range(dimension)]
        for row in range(n)
    ]


def sobol_design(n: int, seed: int) -> list[list[float]]:
    if n <= 0:
        raise ValueError("n must be positive")
    try:
        from scipy.stats import qmc  # type: ignore

        sampler = qmc.Sobol(d=len(PARAMETERS), scramble=True, seed=seed)
        exponent = math.ceil(math.log2(n))
        points = sampler.random_base2(exponent)[:n]
        return points.tolist()
    except Exception:
        return _halton(n, len(PARAMETERS), seed)


def write_design(path: Path, points: Iterable[Iterable[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", *[f"u_{name}" for name in PARAMETERS]])
        for candidate_id, point in enumerate(points):
            values = [min(1.0, max(0.0, float(value))) for value in point]
            if len(values) != len(PARAMETERS):
                raise ValueError("Every design point must have eight coordinates")
            writer.writerow([candidate_id, *[f"{value:.17g}" for value in values]])


def collect_results(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("eligibility_result.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["result_path"] = str(path)
                rows.append(row)
    return rows


def _finite_distance(row: dict[str, str]) -> float:
    try:
        value = float(row.get("distance", "inf"))
    except (TypeError, ValueError):
        return math.inf
    return value if math.isfinite(value) else math.inf


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def promote(rows: list[dict[str, str]], top: int) -> list[list[float]]:
    eligible = [row for row in rows if row.get("eligible") == "1"]
    eligible.sort(key=_finite_distance)
    selected = eligible[:top]
    points: list[list[float]] = []
    for row in selected:
        points.append([float(row[f"u_{name}"]) for name in PARAMETERS])
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    design = subparsers.add_parser("design", help="Generate a Sobol design")
    design.add_argument("--n", type=int, required=True)
    design.add_argument("--seed", type=int, default=12345)
    design.add_argument("--output", type=Path, required=True)

    collect = subparsers.add_parser("collect", help="Collect candidate result files")
    collect.add_argument("--root", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--eligible-only", action="store_true")

    promote_parser = subparsers.add_parser(
        "promote", help="Promote the best eligible candidates to the next stage"
    )
    promote_parser.add_argument("--root", type=Path, required=True)
    promote_parser.add_argument("--top", type=int, default=20)
    promote_parser.add_argument("--output", type=Path, required=True)

    replicate_parser = subparsers.add_parser(
        "replicate", help="Expand each design point across independent seed replicates"
    )
    replicate_parser.add_argument("--input", type=Path, required=True)
    replicate_parser.add_argument("--replicates", type=int, required=True)
    replicate_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "design":
        write_design(args.output, sobol_design(args.n, args.seed))
    elif args.command == "collect":
        rows = collect_results(args.root)
        if args.eligible_only:
            rows = [row for row in rows if row.get("eligible") == "1"]
        rows.sort(key=_finite_distance)
        write_rows(args.output, rows)
    elif args.command == "promote":
        rows = collect_results(args.root)
        points = promote(rows, args.top)
        if not points:
            raise SystemExit("No eligible candidates were found")
        write_design(args.output, points)
    elif args.command == "replicate":
        if args.replicates <= 0:
            raise SystemExit("--replicates must be positive")
        with args.input.open(newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "candidate_id",
                *[f"u_{name}" for name in PARAMETERS],
                "configuration_id",
                "replicate_id",
            ])
            candidate_id = 0
            for configuration_id, row in enumerate(source_rows):
                point = [float(row[f"u_{name}"]) for name in PARAMETERS]
                for replicate_id in range(args.replicates):
                    writer.writerow([
                        candidate_id,
                        *[f"{value:.17g}" for value in point],
                        configuration_id,
                        replicate_id,
                    ])
                    candidate_id += 1


if __name__ == "__main__":
    main()
