#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Compare one sequential book row with its empirical market targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from typing import Mapping


TARGET_FIELDS = (
    "mean_spread_ticks",
    "mean_bid_depth",
    "mean_ask_depth",
    "mid_move_rate",
    "return_variance",
    "return_kurtosis",
    "absolute_return_acf1",
)


def load_target_rows(path: pathlib.Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as source:
        rows = csv.DictReader(source)
        result = {
            row["name"]: {
                "target": float(row["target"]),
                "scale": float(row["scale"]),
                "weight": float(row["weight"]),
            }
            for row in rows
        }
    missing = [name for name in TARGET_FIELDS if name not in result]
    if missing:
        raise ValueError(f"target CSV is missing: {', '.join(missing)}")
    return result


def load_simulation_row(path: pathlib.Path, book_id: int) -> dict[str, str]:
    with path.open(newline="") as source:
        matches = [
            row for row in csv.DictReader(source)
            if int(row["book_id"]) == book_id
        ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one summary row for book_id={book_id}")
    row = matches[0]
    if row.get("structurally_valid") != "1":
        raise ValueError("simulation row is not structurally valid")
    if int(row.get("sample_count", "-1")) != int(
        row.get("expected_sample_count", "-2")
    ):
        raise ValueError("simulation row does not contain every expected sample")
    return row


def compare(simulation: Mapping[str, str],
            targets: Mapping[str, Mapping[str, float]]) -> tuple[list[dict[str, float | str]], float]:
    rows: list[dict[str, float | str]] = []
    weighted_squared = 0.0
    total_weight = 0.0
    for name in TARGET_FIELDS:
        simulated = float(simulation[name])
        target = targets[name]["target"]
        scale = targets[name]["scale"]
        weight = targets[name]["weight"]
        if not all(math.isfinite(value) for value in (simulated, target, scale, weight)):
            raise ValueError(f"non-finite value for {name}")
        if scale <= 0.0 or weight < 0.0:
            raise ValueError(f"invalid scale/weight for {name}")
        standardized = (simulated - target) / scale
        contribution = weight * standardized * standardized
        rows.append({
            "name": name,
            "empirical_target": target,
            "simulated_value": simulated,
            "scale": scale,
            "weight": weight,
            "standardized_residual": standardized,
            "weighted_squared_residual": contribution,
        })
        weighted_squared += contribution
        total_weight += weight
    objective = math.sqrt(weighted_squared / total_weight) if total_weight > 0.0 else math.inf
    return rows, objective


def run(args: argparse.Namespace) -> dict[str, object]:
    summary_path = pathlib.Path(args.summary).resolve()
    target_path = pathlib.Path(args.targets).resolve()
    rows, objective = compare(
        load_simulation_row(summary_path, args.book_id),
        load_target_rows(target_path),
    )
    output_path = pathlib.Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result: dict[str, object] = {
        "book_id": args.book_id,
        "standardized_rmse": objective,
        "summary": str(summary_path),
        "targets": str(target_path),
        "comparison_csv": str(output_path),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="sequential multi-asset summary CSV")
    parser.add_argument("--targets", required=True, help="empirical market-target CSV")
    parser.add_argument("--book-id", type=int, default=0)
    parser.add_argument("--output", required=True, help="metric comparison CSV")
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
