#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Assemble independently extracted ITCH symbol batches into one universe.

``extract_itch50_symbols.py`` reconstructs several visible books in one pass.
For an all-symbol ITCH universe it is deliberately run in small batches so
that active-order, broken-trade, and fixed-clock state remain bounded.  This
program performs the deterministic, auditable assembly step after all batches
have completed successfully:

* validates that every selected symbol has exactly one extraction result or a
  recorded invalid-opening exclusion;
* moves the valid per-symbol extraction directories into one flat data root;
* merges the batch opening-BBO files without silently accepting duplicates;
* writes the exact candidate catalog consumed by ``build_itch_universe_config``;
  and
* writes a small manifest describing the batch-to-universe transformation.

The data root and batch root must be on the same filesystem.  The program uses
atomic directory renames after all validation succeeds; it intentionally does
not copy or delete raw ITCH-derived data.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


OPENING_FIELDS = (
    "symbol",
    "best_bid_ticks",
    "best_ask_ticks",
    "best_bid_depth",
    "best_ask_depth",
    "mid_price_ticks",
)
EXCLUSION_FIELDS = ("symbol", "reason")


class AssemblyError(ValueError):
    """Raised when batch outputs cannot safely form one empirical universe."""


@dataclass(frozen=True)
class Move:
    symbol: str
    source: pathlib.Path
    destination: pathlib.Path


def compact_date(value: str) -> str:
    result = value.replace("-", "")
    if len(result) != 8 or not result.isdigit():
        raise AssemblyError("--trading-date must be YYYY-MM-DD")
    try:
        dt.date(int(result[:4]), int(result[4:6]), int(result[6:]))
    except ValueError as error:
        raise AssemblyError(f"invalid --trading-date: {value}") from error
    return result


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not symbol:
        raise AssemblyError("empty symbol")
    if any(character.isspace() for character in symbol) or "/" in symbol or "\\" in symbol:
        raise AssemblyError(f"unsafe symbol: {value!r}")
    return symbol


def read_symbols(path: pathlib.Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AssemblyError(f"cannot read selected symbols file {path}: {error}") from error
    result: list[str] = []
    seen: set[str] = set()
    for line_number, text in enumerate(lines, start=1):
        if not text.strip():
            continue
        symbol = normalise_symbol(text)
        if symbol in seen:
            raise AssemblyError(
                f"duplicate selected symbol {symbol!r} at line {line_number}"
            )
        seen.add(symbol)
        result.append(symbol)
    if not result:
        raise AssemblyError("selected symbols file is empty")
    if "QQQ" not in seen:
        raise AssemblyError(
            "selected universe does not contain QQQ; remove a restrictive "
            "--max-symbols cap or choose a QQQ-containing candidate list"
        )
    return result


def required_columns(reader: csv.DictReader[str], fields: Sequence[str], path: pathlib.Path) -> None:
    names = {name.strip().lower() for name in (reader.fieldnames or [])}
    missing = [field for field in fields if field not in names]
    if missing:
        raise AssemblyError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )


def canonical_row(row: Mapping[str, str | None]) -> dict[str, str]:
    return {
        str(key).strip().lower(): str(value or "").strip()
        for key, value in row.items()
        if key is not None
    }


def read_openings(path: pathlib.Path, selected: set[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required_columns(reader, OPENING_FIELDS, path)
            rows: list[dict[str, str]] = []
            for line_number, raw in enumerate(reader, start=2):
                row = canonical_row(raw)
                symbol = normalise_symbol(row.get("symbol", ""))
                if symbol not in selected:
                    raise AssemblyError(
                        f"{path}:{line_number} names non-selected symbol {symbol}"
                    )
                rows.append({field: symbol if field == "symbol" else row.get(field, "")
                             for field in OPENING_FIELDS})
            return rows
    except OSError as error:
        raise AssemblyError(f"cannot read opening BBO file {path}: {error}") from error


def read_exclusions(path: pathlib.Path, selected: set[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required_columns(reader, EXCLUSION_FIELDS, path)
            rows: list[dict[str, str]] = []
            for line_number, raw in enumerate(reader, start=2):
                row = canonical_row(raw)
                symbol = normalise_symbol(row.get("symbol", ""))
                reason = row.get("reason", "")
                if symbol not in selected:
                    raise AssemblyError(
                        f"{path}:{line_number} names non-selected symbol {symbol}"
                    )
                if not reason:
                    raise AssemblyError(f"{path}:{line_number} has an empty exclusion reason")
                rows.append({"symbol": symbol, "reason": reason})
            return rows
    except OSError as error:
        raise AssemblyError(f"cannot read exclusion file {path}: {error}") from error


def atomic_csv(path: pathlib.Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(fields))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: pathlib.Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def reject_existing(path: pathlib.Path, label: str) -> None:
    if path.exists():
        raise AssemblyError(f"refusing to replace existing {label}: {path}")


def build_plan(args: argparse.Namespace) -> tuple[
    list[str], list[Move], list[dict[str, str]], list[dict[str, str]], list[dict[str, object]]
]:
    symbols_path = pathlib.Path(args.symbols_file).resolve()
    symbols = read_symbols(symbols_path)
    selected = set(symbols)
    compact = compact_date(args.trading_date)
    batch_root = pathlib.Path(args.batch_root).resolve()
    if not batch_root.is_dir():
        raise AssemblyError(f"--batch-root is not a directory: {batch_root}")
    batches = sorted(path for path in batch_root.glob("batch_*") if path.is_dir())
    if not batches:
        raise AssemblyError(f"no batch_* directories found under {batch_root}")

    data_root = pathlib.Path(args.data_root).resolve()
    opening_out = pathlib.Path(args.opening_bbo_out).resolve()
    candidate_out = pathlib.Path(args.candidate_catalog_out).resolve()
    exclusions_out = pathlib.Path(args.exclusions_out).resolve()
    manifest_out = pathlib.Path(args.manifest_out).resolve()
    for path, label in (
        (data_root, "assembled data root"),
        (opening_out, "merged opening BBO output"),
        (candidate_out, "candidate catalog output"),
        (exclusions_out, "merged exclusions output"),
        (manifest_out, "assembly manifest output"),
    ):
        reject_existing(path, label)

    openings_by_symbol: dict[str, dict[str, str]] = {}
    exclusions_by_symbol: dict[str, dict[str, str]] = {}
    batch_summary: list[dict[str, object]] = []
    for batch in batches:
        opening_file = batch / f"itch_{compact}_basket" / f"opening_bbo_{compact}.csv"
        opening_rows = read_openings(opening_file, selected) if opening_file.is_file() else []
        for row in opening_rows:
            symbol = row["symbol"]
            if symbol in openings_by_symbol:
                raise AssemblyError(
                    f"duplicate opening BBO for {symbol}: batches contain more than one result"
                )
            openings_by_symbol[symbol] = row

        exclusion_file = batch / f"itch_{compact}_exclusions.csv"
        exclusion_rows = read_exclusions(exclusion_file, selected) if exclusion_file.is_file() else []
        for row in exclusion_rows:
            symbol = row["symbol"]
            if symbol in exclusions_by_symbol:
                raise AssemblyError(
                    f"duplicate extractor exclusion for {symbol}: batches overlap"
                )
            exclusions_by_symbol[symbol] = row

        batch_summary.append({
            "batch": batch.name,
            "path": str(batch),
            "opening_bbo_rows": len(opening_rows),
            "invalid_opening_rows": len(exclusion_rows),
        })

    moves: list[Move] = []
    missing: list[str] = []
    for symbol in symbols:
        name = f"itch_{compact}_{symbol.lower()}"
        matches = [batch / name for batch in batches if (batch / name).is_dir()]
        if len(matches) > 1:
            raise AssemblyError(f"duplicate extraction directory for {symbol}: {matches}")
        has_opening = symbol in openings_by_symbol
        excluded = symbol in exclusions_by_symbol
        if matches:
            if excluded:
                raise AssemblyError(
                    f"{symbol} has both an extraction directory and an invalid-opening exclusion"
                )
            if not has_opening:
                raise AssemblyError(
                    f"{symbol} has an extraction directory but no opening-BBO row"
                )
            moves.append(Move(symbol, matches[0], data_root / name))
            continue
        if has_opening:
            raise AssemblyError(f"{symbol} has an opening-BBO row but no extraction directory")
        if not excluded:
            missing.append(symbol)

    if missing:
        raise AssemblyError(
            "selected symbols have neither output nor a recorded invalid opening: "
            + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )
    if "QQQ" not in openings_by_symbol:
        raise AssemblyError("QQQ has no valid two-sided opening BBO")
    return symbols, moves, openings_by_symbol.values(), exclusions_by_symbol.values(), batch_summary


def assemble(args: argparse.Namespace) -> dict[str, object]:
    symbols, moves, opening_rows, exclusion_rows, batch_summary = build_plan(args)
    data_root = pathlib.Path(args.data_root).resolve()
    opening_out = pathlib.Path(args.opening_bbo_out).resolve()
    candidate_out = pathlib.Path(args.candidate_catalog_out).resolve()
    exclusions_out = pathlib.Path(args.exclusions_out).resolve()
    manifest_out = pathlib.Path(args.manifest_out).resolve()
    symbols_path = pathlib.Path(args.symbols_file).resolve()
    source_catalog = pathlib.Path(args.source_catalog).resolve() if args.source_catalog else None
    if source_catalog is not None and not source_catalog.is_file():
        raise AssemblyError(f"--source-catalog is not a file: {source_catalog}")

    data_root.mkdir(parents=True, exist_ok=False)
    moved: list[Move] = []
    try:
        for move in moves:
            # The batch and output roots are deliberately under the same Slurm
            # result directory.  rename is atomic and avoids duplicating a
            # potentially large empirical extraction tree.
            os.rename(move.source, move.destination)
            moved.append(move)
    except OSError as error:
        for move in reversed(moved):
            try:
                os.rename(move.destination, move.source)
            except OSError:
                pass
        try:
            data_root.rmdir()
        except OSError:
            pass
        raise AssemblyError(
            "could not atomically move batch output directories; make --data-root "
            "and --batch-root reside on the same filesystem"
        ) from error

    try:
        ordered_openings = sorted(opening_rows, key=lambda row: row["symbol"])
        ordered_exclusions = sorted(exclusion_rows, key=lambda row: row["symbol"])
        atomic_csv(opening_out, OPENING_FIELDS, ordered_openings)
        atomic_csv(candidate_out, ("symbol",), ({"symbol": symbol} for symbol in symbols))
        atomic_csv(exclusions_out, EXCLUSION_FIELDS, ordered_exclusions)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "trading_date": args.trading_date,
            "inputs": {
                "symbols_file": {
                    "path": str(symbols_path),
                    "sha256": sha256_file(symbols_path),
                },
                "source_catalog": (
                    {"path": str(source_catalog), "sha256": sha256_file(source_catalog)}
                    if source_catalog is not None else None
                ),
                "batch_root": str(pathlib.Path(args.batch_root).resolve()),
            },
            "outputs": {
                "data_root": str(data_root),
                "opening_bbo": str(opening_out),
                "candidate_catalog": str(candidate_out),
                "exclusions": str(exclusions_out),
            },
            "counts": {
                "selected_symbols": len(symbols),
                "valid_two_sided_openings": len(ordered_openings),
                "invalid_opening_exclusions": len(ordered_exclusions),
                "moved_symbol_directories": len(moved),
            },
            "batches": batch_summary,
            "invalid_opening_exclusions": ordered_exclusions,
        }
        atomic_json(manifest_out, manifest)
    except BaseException:
        # The merged CSVs may already exist, but the directories remain the
        # source of truth and are restored when possible.  The caller can
        # inspect the failure and choose a fresh result root rather than
        # having a partial empirical universe mistaken for a complete one.
        for move in reversed(moved):
            if move.destination.exists() and not move.source.exists():
                try:
                    os.rename(move.destination, move.source)
                except OSError:
                    pass
        raise
    return {
        "selected_symbols": len(symbols),
        "valid_two_sided_openings": len(ordered_openings),
        "invalid_opening_exclusions": len(ordered_exclusions),
        "data_root": str(data_root),
        "opening_bbo": str(opening_out),
        "candidate_catalog": str(candidate_out),
        "manifest": str(manifest_out),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols-file", required=True)
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--trading-date", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--opening-bbo-out", required=True)
    parser.add_argument("--candidate-catalog-out", required=True)
    parser.add_argument("--exclusions-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument(
        "--source-catalog",
        help="full startup selector catalog retained as upstream provenance",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = assemble(args)
    except AssemblyError as error:
        print(f"ITCH batch assembly failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
