#!/usr/bin/env python3
"""Certify that shortened preflight paths equal prefixes of full-day paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import tempfile
import os


class PrefixError(RuntimeError):
    """Raised when a shortened path is not an exact full-path prefix."""


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    except OSError as error:
        raise PrefixError(f"cannot read {path}: {error}") from error
    if not rows:
        raise PrefixError(f"empty CSV: {path}")
    return rows


def key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("risk_limit_per_asset", ""),
        row.get("seed", ""),
        row.get("shared_mm_mode", ""),
        row.get("shock_mode", ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-raw", type=pathlib.Path, required=True)
    parser.add_argument("--full-raw", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    short_rows = read_rows(args.short_raw)
    full_rows = read_rows(args.full_raw)
    short_index = {key(row): row for row in short_rows}
    if len(short_index) != len(short_rows):
        raise PrefixError("short raw CSV contains duplicate treatment paths")
    full_index = {key(row): row for row in full_rows}
    if len(full_index) != len(full_rows):
        raise PrefixError("full raw CSV contains duplicate treatment paths")

    comparisons: list[dict[str, object]] = []
    for short in short_rows:
        treatment = key(short)
        if any(not value for value in treatment):
            raise PrefixError(f"short raw CSV has incomplete treatment key: {treatment}")
        full = full_index.get(treatment)
        if full is None:
            raise PrefixError(f"full matrix lacks shortened treatment {treatment}")
        short_metrics = pathlib.Path(short.get("metrics_csv", ""))
        full_metrics = pathlib.Path(full.get("metrics_csv", ""))
        for label, row, path in (
            ("short", short, short_metrics), ("full", full, full_metrics),
        ):
            expected_hash = row.get("metrics_csv_sha256", "")
            if not path.is_file() or not expected_hash or sha256(path) != expected_hash:
                raise PrefixError(
                    f"{label} metrics artifact is missing or hash-invalid for "
                    f"treatment {treatment}: {path}"
                )
        short_horizon = short.get(
            "requested_stochastic_baseline_normalization_seconds", ""
        )
        full_horizon = full.get(
            "requested_stochastic_baseline_normalization_seconds", ""
        )
        full_duration = full.get("requested_duration_seconds", "")
        if not short_horizon or short_horizon != full_horizon:
            raise PrefixError(
                f"normalization horizons differ for treatment {treatment}"
            )
        try:
            fixed_horizon = float(short_horizon)
            production_duration = float(full_duration)
        except ValueError as error:
            raise PrefixError(
                f"invalid horizon metadata for treatment {treatment}"
            ) from error
        if fixed_horizon != production_duration:
            raise PrefixError(
                f"treatment {treatment} does not use the fixed full-session "
                "normalization horizon"
            )
        short_series = read_rows(short_metrics)
        full_series = read_rows(full_metrics)
        full_by_time = {row.get("time_seconds", ""): row for row in full_series}
        if len(full_by_time) != len(full_series):
            raise PrefixError(f"duplicate full-path times in {full_metrics}")
        if list(short_series[0]) != list(full_series[0]):
            raise PrefixError(f"metric schemas differ for treatment {treatment}")
        for short_observation in short_series:
            time_value = short_observation.get("time_seconds", "")
            full_observation = full_by_time.get(time_value)
            if full_observation is None:
                raise PrefixError(
                    f"full path lacks t={time_value} for treatment {treatment}"
                )
            if short_observation != full_observation:
                differing = [
                    field for field in short_observation
                    if short_observation[field] != full_observation.get(field)
                ]
                raise PrefixError(
                    f"short/full prefix differs for treatment {treatment}, "
                    f"t={time_value}, fields={','.join(differing)}"
                )
        comparisons.append({
            "risk_limit_per_asset": treatment[0],
            "seed": int(treatment[1]),
            "shared_mm_mode": treatment[2],
            "shock_mode": treatment[3],
            "short_metrics_csv": str(short_metrics.resolve()),
            "short_metrics_sha256": sha256(short_metrics),
            "full_metrics_csv": str(full_metrics.resolve()),
            "full_metrics_sha256": sha256(full_metrics),
            "matched_observations": len(short_series),
            "last_matched_time_seconds": short_series[-1]["time_seconds"],
        })

    payload = {
        "schema_version": 1,
        "status": "exact_truncated_full_prefix_passed",
        "short_raw": str(args.short_raw.resolve()),
        "short_raw_sha256": sha256(args.short_raw),
        "full_raw": str(args.full_raw.resolve()),
        "full_raw_sha256": sha256(args.full_raw),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=args.output.parent, prefix=f".{args.output.name}.", suffix=".tmp",
        text=True,
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(payload, destination, indent=2, sort_keys=True)
            destination.write("\n")
        os.replace(temporary, args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"prefix_certificate={args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrefixError as error:
        raise SystemExit(f"short/full prefix validation failed: {error}")
