#!/usr/bin/env python3
"""Fail-fast audit of every empirical target used by certified calibration.

This command deliberately invokes the same strict target loader as the
calibration driver.  It is intended to run on the login node before a Slurm
submission so an obsolete compact-data archive cannot consume an allocation
and fail several minutes later.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import tempfile
from collections.abc import Sequence

import calibrate_cluster_value_agents as calibration


DEFAULT_DATES = (
    "2019-01-30",
    "2019-03-27",
    "2019-07-30",
    "2019-10-30",
    "2019-12-30",
    "2020-01-30",
)
DEFAULT_HORIZONS: tuple[int | None, ...] = (300, 3600, None)


class PreflightError(RuntimeError):
    """Raised when the empirical bundle is not calibration-compatible."""


def symbol_list(path: pathlib.Path) -> tuple[str, ...]:
    if not path.is_file():
        raise PreflightError(f"symbol list is not a regular file: {path}")
    values = tuple(
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not values:
        raise PreflightError(f"symbol list is empty: {path}")
    if len(set(values)) != len(values):
        raise PreflightError(f"symbol list contains duplicates: {path}")
    return values


def config_symbols(path: pathlib.Path) -> tuple[str, ...]:
    if not path.is_file():
        raise PreflightError(f"universe config is missing: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "symbol" not in reader.fieldnames:
            raise PreflightError(f"universe config has no symbol column: {path}")
        symbols = tuple(
            str(row.get("symbol", "")).strip().upper() for row in reader
        )
    if not symbols or any(not symbol for symbol in symbols):
        raise PreflightError(f"universe config has an empty symbol: {path}")
    if len(set(symbols)) != len(symbols):
        raise PreflightError(f"universe config has duplicate symbols: {path}")
    return symbols


def canonical_target_digest(
    targets: dict[str, dict[str, calibration.TargetMoment]],
) -> str:
    rows = []
    for symbol in sorted(targets):
        for metric in sorted(targets[symbol]):
            moment = targets[symbol][metric]
            rows.append((
                symbol,
                metric,
                format(moment.target, ".17g"),
                format(moment.empirical_scale, ".17g"),
                format(moment.weight, ".17g"),
            ))
    encoded = json.dumps(
        rows, ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit(
    *,
    data_root: pathlib.Path,
    symbols: Sequence[str],
    dates: Sequence[str],
    horizons: Sequence[int | None] = DEFAULT_HORIZONS,
    expected_symbol_count: int,
) -> dict[str, object]:
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise PreflightError(f"data root is not a directory: {data_root}")
    if len(symbols) != expected_symbol_count:
        raise PreflightError(
            f"certification cohort has {len(symbols)} symbols; expected "
            f"{expected_symbol_count}"
        )
    expected = set(symbols)
    date_reports: list[dict[str, object]] = []
    checked_target_sets = 0
    for day in dates:
        compact = calibration.compact_date(day)
        day_root = data_root / f"itch_{compact}"
        config = day_root / f"nasdaq_common_plus_qqq_{compact}.csv"
        empirical_root = day_root / "empirical_data"
        observed = config_symbols(config)
        observed_set = set(observed)
        if not expected.issubset(observed_set):
            missing = sorted(expected.difference(observed))[:10]
            raise PreflightError(
                f"{config} does not contain the complete certification cohort; "
                f"missing={missing}, rows={len(observed)}"
            )
        horizon_reports: list[dict[str, object]] = []
        for horizon in horizons:
            loaded = calibration.load_targets(
                empirical_root,
                day,
                symbols,
                window_seconds=horizon,
            )
            if set(loaded) != expected:
                raise PreflightError(
                    f"target loader returned an incomplete cohort for "
                    f"{day}/{horizon}"
                )
            horizon_reports.append({
                "horizon_seconds": horizon,
                "symbols": len(loaded),
                "canonical_target_digest_sha256": canonical_target_digest(loaded),
            })
            checked_target_sets += len(loaded)
        date_reports.append({
            "date": day,
            "config": str(config),
            "target_root": str(empirical_root),
            "horizons": horizon_reports,
        })
    return {
        "schema_version": 1,
        "status": "passed",
        "data_root": str(data_root),
        "symbol_count": len(symbols),
        "dates": list(dates),
        "horizons_seconds": list(horizons),
        "checked_symbol_horizon_sets": checked_target_sets,
        "date_reports": date_reports,
    }


def write_json_atomic(path: pathlib.Path, payload: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        temporary = pathlib.Path(output.name)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=pathlib.Path)
    parser.add_argument("--symbols-file", required=True, type=pathlib.Path)
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES))
    parser.add_argument("--expected-symbol-count", type=int, default=1480)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit(
            data_root=args.data_root,
            symbols=symbol_list(args.symbols_file),
            dates=args.dates,
            expected_symbol_count=args.expected_symbol_count,
        )
    except (PreflightError, calibration.CalibrationError, OSError) as error:
        raise SystemExit(f"empirical calibration preflight failed: {error}") from error
    if args.output is not None:
        write_json_atomic(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
