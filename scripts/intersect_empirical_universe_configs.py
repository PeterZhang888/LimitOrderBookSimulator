#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Create matched training/held-out empirical-universe configurations.

Two independently extracted ITCH days need not accept exactly the same symbols:
one day can lack a valid two-sided opening or a complete derived artifact for a
stock.  A behavioural held-out test must use a predeclared common universe,
not silently lose those books while it is running.  This utility takes two
standard ``MultiAssetBookConfig`` CSVs, retains their symbol intersection in
training-day book order, resets ``book_id`` values to a common contiguous
sequence, and writes a provenance report.

It does *not* combine data from days.  The training output still points at the
training direct inputs, and the held-out output still points at the ordinary
held-out direct inputs.  The cluster calibration driver later copies only the
held-out opening state onto the training background.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import tempfile
from collections.abc import Iterable, Sequence
from typing import Any


REQUIRED_FIELDS = {"book_id", "symbol"}


class IntersectionError(ValueError):
    """Raised for malformed or incompatible empirical configurations."""


def normalise_symbol(value: object, *, source: pathlib.Path, line: int) -> str:
    symbol = str(value).strip().upper()
    if not symbol or any(character.isspace() for character in symbol):
        raise IntersectionError(f"invalid symbol {value!r} in {source}:{line}")
    return symbol


def read_config(path: pathlib.Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    if not path.is_file():
        raise IntersectionError(f"not a regular file: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = tuple(reader.fieldnames or ())
        if not fields:
            raise IntersectionError(f"CSV has no header: {path}")
        if len(set(fields)) != len(fields) or any(not field.strip() for field in fields):
            raise IntersectionError(f"CSV has duplicate or empty headers: {path}")
        missing = sorted(REQUIRED_FIELDS.difference(fields))
        if missing:
            raise IntersectionError(f"CSV {path} is missing fields: {', '.join(missing)}")
        rows: list[dict[str, str]] = []
        book_ids: set[int] = set()
        symbols: set[str] = set()
        for line, raw_row in enumerate(reader, start=2):
            if raw_row is None or None in raw_row:
                raise IntersectionError(f"malformed CSV row at {path}:{line}")
            row = {field: (raw_row.get(field) or "").strip() for field in fields}
            try:
                book_id = int(row["book_id"])
            except ValueError as error:
                raise IntersectionError(f"invalid book_id in {path}:{line}") from error
            if book_id < 0 or book_id in book_ids:
                raise IntersectionError(f"duplicate/negative book_id in {path}:{line}")
            symbol = normalise_symbol(row["symbol"], source=path, line=line)
            if symbol in symbols:
                raise IntersectionError(f"duplicate symbol {symbol} in {path}")
            row["book_id"] = str(book_id)
            row["symbol"] = symbol
            rows.append(row)
            book_ids.add(book_id)
            symbols.add(symbol)
    if not rows:
        raise IntersectionError(f"CSV has no data rows: {path}")
    if book_ids != set(range(len(rows))):
        raise IntersectionError(f"book_id values must be contiguous from zero in {path}")
    rows.sort(key=lambda row: int(row["book_id"]))
    return fields, rows


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(path: pathlib.Path,
               fields: Sequence[str],
               rows: Iterable[dict[str, str]],
               *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise IntersectionError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True,
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with open(descriptor, "w", newline="", encoding="utf-8", closefd=True) as output:
            writer = csv.DictWriter(output, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: pathlib.Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise IntersectionError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True,
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with open(descriptor, "w", encoding="utf-8", closefd=True) as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def intersect_rows(training_rows: Sequence[dict[str, str]],
                   heldout_rows: Sequence[dict[str, str]]) -> tuple[
                       list[dict[str, str]], list[dict[str, str]], list[str], list[str]
                   ]:
    """Return common configurations in training order plus day-specific exclusions."""
    heldout_by_symbol = {row["symbol"]: row for row in heldout_rows}
    training_symbols = {row["symbol"] for row in training_rows}
    heldout_symbols = set(heldout_by_symbol)
    common_symbols = [row["symbol"] for row in training_rows if row["symbol"] in heldout_by_symbol]
    if not common_symbols:
        raise IntersectionError("training and held-out configurations have no common symbols")
    training_by_symbol = {row["symbol"]: row for row in training_rows}
    common_training: list[dict[str, str]] = []
    common_heldout: list[dict[str, str]] = []
    for book_id, symbol in enumerate(common_symbols):
        train = dict(training_by_symbol[symbol])
        hold = dict(heldout_by_symbol[symbol])
        train["book_id"] = str(book_id)
        hold["book_id"] = str(book_id)
        common_training.append(train)
        common_heldout.append(hold)
    return (
        common_training,
        common_heldout,
        sorted(training_symbols.difference(heldout_symbols)),
        sorted(heldout_symbols.difference(training_symbols)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--heldout-config", required=True)
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--heldout-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--minimum-symbols", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.minimum_symbols <= 0:
        raise IntersectionError("--minimum-symbols must be positive")
    training_path = pathlib.Path(args.training_config).expanduser().resolve()
    heldout_path = pathlib.Path(args.heldout_config).expanduser().resolve()
    training_output = pathlib.Path(args.training_output).expanduser().resolve()
    heldout_output = pathlib.Path(args.heldout_output).expanduser().resolve()
    report_path = pathlib.Path(args.report).expanduser().resolve()
    if len({training_output, heldout_output, report_path}) != 3:
        raise IntersectionError("all outputs must be different paths")
    train_fields, training_rows = read_config(training_path)
    heldout_fields, heldout_rows = read_config(heldout_path)
    if train_fields != heldout_fields:
        raise IntersectionError(
            "training and held-out configurations must have identical headers"
        )
    common_training, common_heldout, only_training, only_heldout = intersect_rows(
        training_rows, heldout_rows
    )
    if len(common_training) < args.minimum_symbols:
        raise IntersectionError(
            f"common universe has {len(common_training)} symbols, below "
            f"--minimum-symbols={args.minimum_symbols}"
        )
    atomic_csv(training_output, train_fields, common_training, overwrite=args.overwrite)
    atomic_csv(heldout_output, heldout_fields, common_heldout, overwrite=args.overwrite)
    report = {
        "schema_version": 1,
        "training_input": str(training_path),
        "heldout_input": str(heldout_path),
        "training_input_sha256": sha256_file(training_path),
        "heldout_input_sha256": sha256_file(heldout_path),
        "training_output": str(training_output),
        "heldout_output": str(heldout_output),
        "common_symbol_count": len(common_training),
        "training_symbol_count": len(training_rows),
        "heldout_symbol_count": len(heldout_rows),
        "only_training_symbols": only_training,
        "only_heldout_symbols": only_heldout,
        "common_symbols_in_training_order": [row["symbol"] for row in common_training],
    }
    atomic_json(report_path, report, overwrite=args.overwrite)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (IntersectionError, OSError) as error:
        print(f"universe intersection failed: {error}", file=__import__("sys").stderr)
        return 1
    print(json.dumps({
        "common_symbol_count": report["common_symbol_count"],
        "report": args.report,
        "training_output": args.training_output,
        "heldout_output": args.heldout_output,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
