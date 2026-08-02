#!/usr/bin/env python3
"""Build no-oracle dated configs around one frozen pooled market model.

Every runtime parameter, Hawkes-rate file and empirical mark directory comes
from ``--pooled-config``.  A dated opening file contributes only the opening
price/BBO/depth fields, while ``target_data_dir`` points the calibration
evaluator at that date's empirical targets.  This separation prevents a
training or held-out run from silently using same-day rates or marks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
from collections.abc import Mapping, Sequence
from datetime import date


OPENING_FIELDS = (
    "fundamental_price_ticks",
    "initial_best_bid_ticks",
    "initial_best_ask_ticks",
    "initial_best_bid_depth",
    "initial_best_ask_depth",
)
REQUIRED_PREFIX_TARGET_SECONDS = (300, 3_600)


class PreparationError(RuntimeError):
    """A dated config would violate the frozen-input contract."""


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except OSError as error:
        raise PreparationError(f"cannot read {path}: {error}") from error
    if not fields or not rows:
        raise PreparationError(f"empty CSV: {path}")
    return fields, rows


def parse_dated_path(value: str) -> tuple[str, pathlib.Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected ISO-DATE=PATH")
    raw_day, raw_path = value.split("=", 1)
    try:
        day = date.fromisoformat(raw_day).isoformat()
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {raw_day}") from error
    path = pathlib.Path(raw_path).expanduser().resolve()
    return day, path


def dated_map(values: Sequence[tuple[str, pathlib.Path]], label: str) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    for day, path in values:
        if day in result:
            raise PreparationError(f"duplicate {label} date: {day}")
        result[day] = path
    return result


def symbol_rows(path: pathlib.Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    fields, rows = read_csv(path)
    required = {"book_id", "symbol", *OPENING_FIELDS}
    missing = sorted(required.difference(fields))
    if missing:
        raise PreparationError(f"{path} lacks columns: {', '.join(missing)}")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        symbol = row["symbol"].strip().upper()
        if not symbol or symbol in result:
            raise PreparationError(f"invalid or duplicate symbol in {path}: {symbol!r}")
        row["symbol"] = symbol
        result[symbol] = row
    return fields, result


def positive(value: str, label: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise PreparationError(f"{label} is not numeric") from error
    if not math.isfinite(number) or number <= 0.0:
        raise PreparationError(f"{label} must be finite and positive")
    return number


def validate_opening(row: Mapping[str, str], day: str, symbol: str) -> None:
    fundamental = positive(
        row["fundamental_price_ticks"], f"{day}/{symbol} fundamental"
    )
    bid = positive(row["initial_best_bid_ticks"], f"{day}/{symbol} bid")
    ask = positive(row["initial_best_ask_ticks"], f"{day}/{symbol} ask")
    positive(row["initial_best_bid_depth"], f"{day}/{symbol} bid depth")
    positive(row["initial_best_ask_depth"], f"{day}/{symbol} ask depth")
    if not bid < ask or not bid <= fundamental <= ask:
        raise PreparationError(
            f"{day}/{symbol} opening must be two-sided and contain its midpoint"
        )


def target_directory(root: pathlib.Path, day: str, symbol: str) -> pathlib.Path:
    compact = day.replace("-", "")
    name = f"itch_{compact}_{symbol.lower()}"
    candidates = (root / "empirical_data" / name, root / name)
    directory = next((path for path in candidates if path.is_dir()), None)
    if directory is None:
        raise PreparationError(f"target directory is missing for {day}/{symbol}")
    stems = (f"market_targets_{symbol.lower()}_{compact}.csv",) + tuple(
        f"market_targets_{symbol.lower()}_{compact}_window_{seconds}s.csv"
        for seconds in REQUIRED_PREFIX_TARGET_SECONDS
    )
    missing = [name for name in stems if not (directory / name).is_file()]
    if missing:
        raise PreparationError(
            f"target artifacts are incomplete for {day}/{symbol}: {missing}"
        )
    manifest_path = directory / f"itch_manifest_{symbol.lower()}_{compact}.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(
            f"cannot read target manifest for {day}/{symbol}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise PreparationError(f"target manifest is not an object for {day}/{symbol}")
    windows = manifest.get("market_target_windows")
    if (
        manifest.get("trading_date") != day
        or manifest.get("symbol") != symbol
        or not isinstance(windows, dict)
    ):
        raise PreparationError(f"target manifest identity is invalid for {day}/{symbol}")
    for seconds in REQUIRED_PREFIX_TARGET_SECONDS:
        record = windows.get(str(seconds))
        expected_file = (
            f"market_targets_{symbol.lower()}_{compact}_window_{seconds}s.csv"
        )
        if (
            not isinstance(record, dict)
            or record.get("file") != expected_file
            or record.get("duration_seconds") != seconds
        ):
            raise PreparationError(
                f"target manifest lacks the certified {seconds}s prefix for "
                f"{day}/{symbol}"
            )
    return directory.resolve()


def selected_symbols(path: pathlib.Path | None, available: Sequence[str]) -> list[str]:
    if path is None:
        return list(available)
    try:
        values = [line.strip().upper() for line in path.read_text(encoding="utf-8").splitlines()]
    except OSError as error:
        raise PreparationError(f"cannot read symbols file {path}: {error}") from error
    if not values or any(not value for value in values) or len(values) != len(set(values)):
        raise PreparationError("symbols file must contain unique nonempty symbols")
    missing = sorted(set(values).difference(available))
    if missing:
        raise PreparationError(f"symbols are absent from pooled config: {missing[:10]}")
    wanted = set(values)
    return [symbol for symbol in available if symbol in wanted]


def write_csv(path: pathlib.Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, object]:
    pooled_path = args.pooled_config.expanduser().resolve()
    pooled_fields, pooled_by_symbol = symbol_rows(pooled_path)
    required_runtime = {"data_dir", "hawkes_rates_file"}
    missing_runtime = sorted(required_runtime.difference(pooled_fields))
    if missing_runtime:
        raise PreparationError(
            f"pooled config lacks columns: {', '.join(missing_runtime)}"
        )
    pooled_order = list(pooled_by_symbol)
    symbols = selected_symbols(
        args.symbols_file.expanduser().resolve() if args.symbols_file else None,
        pooled_order,
    )
    openings = dated_map(args.dated_opening_config, "opening")
    targets = dated_map(args.dated_target_root, "target")
    if set(openings) != set(targets):
        raise PreparationError("opening-config and target-root dates differ")
    if len(openings) != args.expected_date_count:
        raise PreparationError(
            f"expected {args.expected_date_count} dates, observed {len(openings)}"
        )
    forbidden = {date.fromisoformat(value).isoformat() for value in args.forbid_date}
    overlap = sorted(set(openings).intersection(forbidden))
    if overlap:
        raise PreparationError(f"forbidden dates supplied: {overlap}")

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise PreparationError(f"output root must be absent or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    deployment_path = output_root / "deployment_config.csv"
    deployment_rows: list[dict[str, object]] = []
    for book_id, symbol in enumerate(symbols):
        row: dict[str, object] = dict(pooled_by_symbol[symbol])
        row["book_id"] = book_id
        deployment_rows.append(row)
    write_csv(deployment_path, pooled_fields, deployment_rows)
    output_fields = list(pooled_fields)
    if "target_data_dir" not in output_fields:
        output_fields.append("target_data_dir")
    outputs: dict[str, object] = {}
    for day in sorted(openings):
        opening_path = openings[day]
        _, opening_by_symbol = symbol_rows(opening_path)
        missing = sorted(set(symbols).difference(opening_by_symbol))
        if missing:
            raise PreparationError(f"opening config {day} misses symbols: {missing[:10]}")
        rows: list[dict[str, object]] = []
        for book_id, symbol in enumerate(symbols):
            opening = opening_by_symbol[symbol]
            validate_opening(opening, day, symbol)
            row: dict[str, object] = dict(pooled_by_symbol[symbol])
            row["book_id"] = book_id
            for field in OPENING_FIELDS:
                row[field] = opening[field]
            row["target_data_dir"] = str(target_directory(targets[day], day, symbol))
            rows.append(row)
        output_path = output_root / f"dated_config_{day.replace('-', '')}.csv"
        write_csv(output_path, output_fields, rows)
        outputs[day] = {
            "path": str(output_path),
            "sha256": sha256(output_path),
            "opening_source": str(opening_path),
            "opening_source_sha256": sha256(opening_path),
            "target_root": str(targets[day]),
        }

    manifest = {
        "schema_version": 1,
        "role": "frozen_pooled_model_with_dated_openings_and_targets",
        "pooled_config": {"path": str(pooled_path), "sha256": sha256(pooled_path)},
        "deployment_config": {
            "path": str(deployment_path),
            "sha256": sha256(deployment_path),
        },
        "symbol_count": len(symbols),
        "dates": sorted(openings),
        "runtime_fields_inherited_from_pooled": True,
        "dated_fields": list(OPENING_FIELDS),
        "target_data_dir_is_evaluation_only": True,
        "same_day_rates_or_marks_used": False,
        "forbidden_dates": sorted(forbidden),
        "outputs": outputs,
    }
    manifest_path = output_root / "dated_config_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pooled-config", type=pathlib.Path, required=True)
    result.add_argument("--symbols-file", type=pathlib.Path)
    result.add_argument(
        "--dated-opening-config", action="append", type=parse_dated_path,
        required=True, metavar="DATE=PATH",
    )
    result.add_argument(
        "--dated-target-root", action="append", type=parse_dated_path,
        required=True, metavar="DATE=PATH",
    )
    result.add_argument("--expected-date-count", type=int, required=True)
    result.add_argument("--forbid-date", action="append", default=[])
    result.add_argument("--output-root", type=pathlib.Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(parser().parse_args(argv))
    except (PreparationError, OSError, ValueError) as error:
        print(f"dated queue-reactive config preparation failed: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
