#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Select an auditable real-instrument universe from NASDAQ ITCH 5.0 startup data.

The selector intentionally reads only the startup directory/trading-state
snapshot.  It stops at the ``Q`` (start of market hours) System Event, rather
than scanning or simulating the full trading day.  This is important because
the final Trading Action state at the end of a file is not the opening trading
state.

The currently supported policy, ``nasdaq-common-plus-qqq``, selects an
instrument when all of the following hold in the startup snapshot:

* NASDAQ market category is ``Q``, ``G``, or ``S``;
* Financial Status Indicator is ``N`` (normal);
* Stock Directory authenticity is ``P`` (production);
* the latest startup Trading Action state is ``T`` (trading);
* round-lot size is positive; and
* Issue Classification is ``C`` (common stock), with QQQ retained as an
  explicit ETP exception.

The resulting catalog records both eligible and ineligible directory entries,
including the precise exclusion reason.  This makes the selection criterion
reproducible and prevents a synthetic instrument universe from being confused
with the complete ITCH directory.  A fixed-symbol file may be supplied when a
cross-date cohort has already been declared.  In that mode every member must
pass this session's startup policy; the selector never substitutes or silently
drops a requested symbol.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import BinaryIO, Iterator, Sequence, TextIO


POLICY_NAME = "nasdaq-common-plus-qqq"
MARKET_CATEGORIES = frozenset(("Q", "G", "S"))
START_OF_MARKET_HOURS = "Q"
SAFE_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9.+$-]{0,7}\Z")

# NASDAQ TotalView-ITCH 5.0 message body sizes, including the one-byte type.
STOCK_DIRECTORY_MINIMUM_BYTES = 39
TRADING_ACTION_MINIMUM_BYTES = 25
SYSTEM_EVENT_MINIMUM_BYTES = 12


@dataclass
class TradingAction:
    """The latest pre-market-hours trading state for one stock locate."""

    state: str
    reason: str
    timestamp_ns: int


@dataclass
class DirectoryRecord:
    """Fields from a Stock Directory (R) message relevant to selection."""

    stock_locate: int
    timestamp_ns: int
    symbol: str
    market_category: str
    financial_status: str
    round_lot_size: int
    round_lots_only: str
    issue_classification: str
    issue_sub_type: str
    authenticity: str
    etp_flag: str
    etp_leverage_factor: int
    inverse_indicator: str
    trading_state: str = ""
    trading_reason: str = ""
    trading_action_timestamp_ns: int | None = None


@dataclass(frozen=True)
class StartupSnapshot:
    """Directory and trading-state data observed before market hours begin."""

    directories: tuple[DirectoryRecord, ...]
    market_hours_timestamp_ns: int
    records_scanned: int


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def sha256_file(path: pathlib.Path) -> str:
    """Return the SHA-256 of the exact bytes in *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_symbols_sha256(symbols: Sequence[str]) -> str:
    """Hash the exact UTF-8 newline-delimited canonical symbol sequence."""

    rendered = "".join(f"{symbol}\n" for symbol in symbols)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def normalise_requested_symbol(value: str) -> str:
    """Normalize one requested symbol and reject path-unsafe identifiers."""

    symbol = value.strip().upper()
    if not symbol:
        raise ValueError("empty symbol")
    if not symbol.isascii() or SAFE_SYMBOL.fullmatch(symbol) is None:
        raise ValueError(
            f"unsafe symbol {value!r}; expected 1--8 ASCII characters from "
            "A--Z, 0--9, '.', '+', '$', and '-' and an alphanumeric first character"
        )
    return symbol


def read_fixed_symbols(path: pathlib.Path) -> list[str]:
    """Read, normalize, deduplicate, and canonically order a fixed cohort."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read fixed-symbol file {path}: {exc}") from exc

    symbols: list[str] = []
    seen: dict[str, int] = {}
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            symbol = normalise_requested_symbol(raw)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if symbol in seen:
            raise ValueError(
                f"duplicate fixed symbol {symbol!r} at {path}:{line_number}; "
                f"first appeared at line {seen[symbol]}"
            )
        seen[symbol] = line_number
        symbols.append(symbol)

    if not symbols:
        raise ValueError(f"fixed-symbol file is empty: {path}")
    if "QQQ" not in seen:
        raise ValueError(
            f"fixed-symbol file must contain QQQ explicitly: {path}"
        )
    return ["QQQ", *sorted(symbol for symbol in symbols if symbol != "QQQ")]


def _ascii_text(field: bytes) -> str:
    """Decode a fixed-width ITCH text field without leaking NUL padding."""

    return field.rstrip(b" \x00").decode("ascii", errors="replace")


def _ascii_code(field: bytes) -> str:
    return field.decode("ascii", errors="replace")


def _timestamp_ns(message: bytes) -> int:
    # ITCH common header: type[0], locate[1:3], tracking[3:5], timestamp[5:11].
    return int.from_bytes(message[5:11], "big", signed=False)


@contextlib.contextmanager
def open_itch_binary(path: pathlib.Path) -> Iterator[BinaryIO]:
    """Open either raw or gzip-compressed ITCH input, based on its magic bytes."""

    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rb") as source:  # type: ignore[arg-type]
        yield source


def read_itch_record(source: BinaryIO, record_number: int) -> bytes | None:
    """Read one length-prefixed ITCH record, rejecting truncated input clearly."""

    length_bytes = source.read(2)
    if not length_bytes:
        return None
    if len(length_bytes) != 2:
        raise ValueError(
            f"truncated two-byte length prefix before ITCH record {record_number}"
        )
    size = int.from_bytes(length_bytes, "big", signed=False)
    if size == 0:
        raise ValueError(f"zero-length ITCH record {record_number}")
    message = source.read(size)
    if len(message) != size:
        raise ValueError(
            f"truncated ITCH record {record_number}: expected {size} bytes, "
            f"received {len(message)}"
        )
    return message


def parse_stock_directory(message: bytes, record_number: int) -> DirectoryRecord:
    """Parse fixed ITCH 5.0 R offsets, allowing future trailing extensions."""

    if len(message) < STOCK_DIRECTORY_MINIMUM_BYTES:
        raise ValueError(
            f"malformed Stock Directory record {record_number}: expected at least "
            f"{STOCK_DIRECTORY_MINIMUM_BYTES} bytes, received {len(message)}"
        )
    return DirectoryRecord(
        stock_locate=int.from_bytes(message[1:3], "big", signed=False),
        timestamp_ns=_timestamp_ns(message),
        symbol=_ascii_text(message[11:19]),
        market_category=_ascii_code(message[19:20]),
        financial_status=_ascii_code(message[20:21]),
        round_lot_size=int.from_bytes(message[21:25], "big", signed=False),
        round_lots_only=_ascii_code(message[25:26]),
        issue_classification=_ascii_code(message[26:27]),
        issue_sub_type=_ascii_text(message[27:29]),
        authenticity=_ascii_code(message[29:30]),
        # The remaining values are not selection inputs, but retaining them in
        # the catalog makes the common-stock/QQQ policy auditable.
        etp_flag=_ascii_code(message[33:34]),
        etp_leverage_factor=int.from_bytes(message[34:38], "big", signed=False),
        inverse_indicator=_ascii_code(message[38:39]),
    )


def parse_trading_action(message: bytes, record_number: int) -> tuple[int, str, TradingAction]:
    """Parse fixed ITCH 5.0 H offsets, allowing future trailing extensions."""

    if len(message) < TRADING_ACTION_MINIMUM_BYTES:
        raise ValueError(
            f"malformed Trading Action record {record_number}: expected at least "
            f"{TRADING_ACTION_MINIMUM_BYTES} bytes, received {len(message)}"
        )
    locate = int.from_bytes(message[1:3], "big", signed=False)
    symbol = _ascii_text(message[11:19])
    action = TradingAction(
        state=_ascii_code(message[19:20]),
        reason=_ascii_text(message[21:25]),
        timestamp_ns=_timestamp_ns(message),
    )
    return locate, symbol, action


def _attach_trading_actions(
    directories: dict[int, DirectoryRecord],
    actions_by_locate: dict[int, TradingAction],
    actions_by_symbol: dict[str, TradingAction],
) -> None:
    for record in directories.values():
        # Locate is authoritative.  Symbol fallback handles an unusual but
        # harmless ordering in which an H record appears before its R record.
        action = actions_by_locate.get(record.stock_locate)
        if action is None:
            action = actions_by_symbol.get(record.symbol)
        if action is not None:
            record.trading_state = action.state
            record.trading_reason = action.reason
            record.trading_action_timestamp_ns = action.timestamp_ns


def scan_startup_snapshot(input_path: pathlib.Path) -> StartupSnapshot:
    """Read R/H startup state through, and not beyond, System Event ``Q``."""

    directories: dict[int, DirectoryRecord] = {}
    actions_by_locate: dict[int, TradingAction] = {}
    actions_by_symbol: dict[str, TradingAction] = {}
    record_number = 0

    with open_itch_binary(input_path) as source:
        while True:
            message = read_itch_record(source, record_number + 1)
            if message is None:
                break
            record_number += 1
            kind = _ascii_code(message[0:1])

            if kind == "R":
                record = parse_stock_directory(message, record_number)
                directories[record.stock_locate] = record
                continue

            if kind == "H":
                locate, symbol, action = parse_trading_action(message, record_number)
                actions_by_locate[locate] = action
                if symbol:
                    actions_by_symbol[symbol] = action
                continue

            if kind == "S":
                if len(message) < SYSTEM_EVENT_MINIMUM_BYTES:
                    raise ValueError(
                        f"malformed System Event record {record_number}: expected at "
                        f"least {SYSTEM_EVENT_MINIMUM_BYTES} bytes, received {len(message)}"
                    )
                if _ascii_code(message[11:12]) == START_OF_MARKET_HOURS:
                    _attach_trading_actions(
                        directories, actions_by_locate, actions_by_symbol
                    )
                    ordered = tuple(
                        sorted(
                            directories.values(),
                            key=lambda item: (item.symbol, item.stock_locate),
                        )
                    )
                    return StartupSnapshot(
                        directories=ordered,
                        market_hours_timestamp_ns=_timestamp_ns(message),
                        records_scanned=record_number,
                    )

    raise ValueError(
        "no System Event Q (start of market hours) was found; refusing to use "
        "a full-day or end-of-day Trading Action state as the startup snapshot"
    )


def eligibility(record: DirectoryRecord) -> tuple[bool, str]:
    """Return the named-policy decision and an auditable reason/basis string."""

    failures: list[str] = []
    if record.market_category not in MARKET_CATEGORIES:
        failures.append("market_category_not_QGS")
    if record.financial_status != "N":
        failures.append("financial_status_not_N")
    if record.authenticity != "P":
        failures.append("authenticity_not_P")
    if record.trading_state != "T":
        failures.append("startup_trading_state_not_T")
    if record.round_lot_size <= 0:
        failures.append("nonpositive_round_lot")
    if record.issue_classification != "C" and record.symbol != "QQQ":
        failures.append("issue_not_common_or_qqq")
    if not record.symbol:
        failures.append("blank_symbol")
    if failures:
        return False, ";".join(failures)
    if record.symbol == "QQQ" and record.issue_classification != "C":
        return True, "eligible_qqq_exception"
    return True, "eligible_common_stock"


CATALOG_COLUMNS = (
    "policy",
    "symbol",
    "stock_locate",
    "directory_timestamp_ns",
    "market_category",
    "financial_status",
    "round_lot_size",
    "round_lots_only",
    "issue_classification",
    "issue_sub_type",
    "authenticity",
    "etp_flag",
    "etp_leverage_factor",
    "inverse_indicator",
    "startup_trading_state",
    "startup_trading_reason",
    "trading_action_timestamp_ns",
    "eligible",
    "selected",
    "eligibility_reason",
)


def _atomic_text_writer(path: pathlib.Path) -> tuple[pathlib.Path, TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    return pathlib.Path(temporary_name), os.fdopen(descriptor, "w", encoding="utf-8", newline="")


def write_catalog(
    output_path: pathlib.Path,
    records: Sequence[DirectoryRecord],
    selected_symbols: frozenset[str],
) -> int:
    """Write the complete startup catalog atomically and return eligible symbols."""

    temporary_path, target = _atomic_text_writer(output_path)
    eligible_symbols: set[str] = set()
    try:
        with target:
            writer = csv.DictWriter(target, fieldnames=CATALOG_COLUMNS)
            writer.writeheader()
            for record in records:
                is_eligible, reason = eligibility(record)
                if is_eligible:
                    eligible_symbols.add(record.symbol)
                writer.writerow({
                    "policy": POLICY_NAME,
                    "symbol": record.symbol,
                    "stock_locate": record.stock_locate,
                    "directory_timestamp_ns": record.timestamp_ns,
                    "market_category": record.market_category,
                    "financial_status": record.financial_status,
                    "round_lot_size": record.round_lot_size,
                    "round_lots_only": record.round_lots_only,
                    "issue_classification": record.issue_classification,
                    "issue_sub_type": record.issue_sub_type,
                    "authenticity": record.authenticity,
                    "etp_flag": record.etp_flag,
                    "etp_leverage_factor": record.etp_leverage_factor,
                    "inverse_indicator": record.inverse_indicator,
                    "startup_trading_state": record.trading_state,
                    "startup_trading_reason": record.trading_reason,
                    "trading_action_timestamp_ns": (
                        "" if record.trading_action_timestamp_ns is None
                        else record.trading_action_timestamp_ns
                    ),
                    "eligible": int(is_eligible),
                    "selected": int(is_eligible and record.symbol in selected_symbols),
                    "eligibility_reason": reason,
                })
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return len(eligible_symbols)


def write_symbol_list(output_path: pathlib.Path, symbols: Sequence[str]) -> None:
    """Write the deterministic, newline-delimited selected universe atomically."""

    temporary_path, target = _atomic_text_writer(output_path)
    try:
        with target:
            for symbol in symbols:
                target.write(f"{symbol}\n")
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_json(output_path: pathlib.Path, value: object) -> None:
    """Write deterministic JSON atomically."""

    temporary_path, target = _atomic_text_writer(output_path)
    try:
        with target:
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def validate_fixed_symbols(
    records: Sequence[DirectoryRecord], requested: Sequence[str]
) -> None:
    """Require every fixed-cohort member to exist and pass startup policy."""

    by_symbol: dict[str, list[DirectoryRecord]] = {}
    for record in records:
        by_symbol.setdefault(record.symbol, []).append(record)

    missing: list[str] = []
    ineligible: list[str] = []
    for symbol in requested:
        candidates = by_symbol.get(symbol, [])
        if not candidates:
            missing.append(symbol)
            continue
        if not any(eligibility(record)[0] for record in candidates):
            reasons = sorted({eligibility(record)[1] for record in candidates})
            ineligible.append(f"{symbol} ({' | '.join(reasons)})")

    diagnostics: list[str] = []
    if missing:
        diagnostics.append("absent from startup directory: " + ", ".join(missing))
    if ineligible:
        diagnostics.append("ineligible at startup: " + ", ".join(ineligible))
    if diagnostics:
        raise ValueError(
            "fixed-symbol universe is not valid for this ITCH session; "
            + "; ".join(diagnostics)
        )


def choose_symbols(
    records: Sequence[DirectoryRecord], max_symbols: int | None,
    fixed_symbols: Sequence[str] | None = None,
) -> list[str]:
    """Choose either the fixed cohort or eligible symbols deterministically."""

    if max_symbols is not None and fixed_symbols is not None:
        raise ValueError("max-symbol cap and fixed-symbol universe are mutually exclusive")
    if fixed_symbols is not None:
        validate_fixed_symbols(records, fixed_symbols)
        return list(fixed_symbols)

    eligible_symbols = sorted({
        record.symbol for record in records if eligibility(record)[0]
    })
    return eligible_symbols if max_symbols is None else eligible_symbols[:max_symbols]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a real NASDAQ common-stock-plus-QQQ universe from ITCH 5.0 "
            "startup Stock Directory and Trading Action messages."
        )
    )
    parser.add_argument(
        "--itch", required=True, type=pathlib.Path,
        help="raw or gzip-compressed NASDAQ TotalView-ITCH 5.0 input",
    )
    parser.add_argument(
        "--catalog-out", required=True, type=pathlib.Path,
        help="CSV path for all startup directory entries and selection decisions",
    )
    parser.add_argument(
        "--symbols-out", required=True, type=pathlib.Path,
        help="newline-delimited selected symbols, sorted deterministically",
    )
    parser.add_argument(
        "--policy", choices=(POLICY_NAME,), default=POLICY_NAME,
        help=f"selection policy (only {POLICY_NAME!r} is currently supported)",
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--max-symbols", type=_positive_int,
        help="optional deterministic cap on selected eligible symbols",
    )
    selection_group.add_argument(
        "--fixed-symbols", type=pathlib.Path,
        help=(
            "optional newline-delimited predeclared cohort; every normalized "
            "symbol must be startup-eligible, QQQ is mandatory, and output order "
            "is QQQ first followed by lexical order"
        ),
    )
    parser.add_argument(
        "--provenance-out", type=pathlib.Path,
        help="optional JSON path for content-bound selection provenance",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    input_path = args.itch.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"--itch does not name a readable file: {input_path}")
    catalog_out = args.catalog_out.expanduser().resolve()
    symbols_out = args.symbols_out.expanduser().resolve()
    provenance_out = (
        None if args.provenance_out is None
        else args.provenance_out.expanduser().resolve()
    )
    output_paths = [catalog_out, symbols_out]
    if provenance_out is not None:
        output_paths.append(provenance_out)
    if len(set(output_paths)) != len(output_paths):
        parser.error("catalog, symbol, and provenance outputs must be different paths")
    if input_path in output_paths:
        parser.error("selection outputs must not overwrite the ITCH input")

    fixed_path: pathlib.Path | None = None
    fixed_symbols: list[str] | None = None
    fixed_sha256: str | None = None
    if args.fixed_symbols is not None:
        fixed_path = args.fixed_symbols.expanduser().resolve()
        if not fixed_path.is_file():
            parser.error(
                f"--fixed-symbols does not name a readable regular file: {fixed_path}"
            )
        if fixed_path in output_paths:
            parser.error("selection outputs must not overwrite --fixed-symbols")
        try:
            fixed_symbols = read_fixed_symbols(fixed_path)
            fixed_sha256 = sha256_file(fixed_path)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))

    try:
        snapshot = scan_startup_snapshot(input_path)
        selected_symbols = choose_symbols(
            snapshot.directories, args.max_symbols, fixed_symbols
        )
        eligible_count = write_catalog(
            catalog_out, snapshot.directories, frozenset(selected_symbols)
        )
        write_symbol_list(symbols_out, selected_symbols)
        if fixed_symbols is not None:
            mode = "fixed_symbols"
        elif args.max_symbols is not None:
            mode = "eligible_capped"
        else:
            mode = "all_eligible"
        selected_sha256 = canonical_symbols_sha256(selected_symbols)
        provenance = {
            "schema_version": 1,
            "policy": args.policy,
            "mode": mode,
            "itch_input": str(input_path),
            "startup_records": snapshot.records_scanned,
            "directory_entries": len(snapshot.directories),
            "eligible_symbol_count": eligible_count,
            "market_hours_timestamp_ns": snapshot.market_hours_timestamp_ns,
            "max_symbols": args.max_symbols,
            "fixed_symbols_input": (
                None if fixed_path is None else {
                    "path": str(fixed_path),
                    "raw_sha256": fixed_sha256,
                    "normalized_count": len(fixed_symbols or ()),
                }
            ),
            "selected_symbols": {
                "path": str(symbols_out),
                "count": len(selected_symbols),
                "canonical_sha256": selected_sha256,
                "canonical_encoding": "UTF-8, one symbol per line, LF terminated",
                "canonical_order": (
                    "QQQ first, then lexicographic"
                    if mode == "fixed_symbols" else "lexicographic"
                ),
                "ordered_symbols": selected_symbols,
            },
        }
        if provenance_out is not None:
            write_json(provenance_out, provenance)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(
        "policy={policy} startup_records={records} directory_entries={entries} "
        "eligible_symbols={eligible} selected_symbols={selected} "
        "market_hours_timestamp_ns={timestamp} selection_mode={mode} "
        "selected_symbols_sha256={selected_sha256}".format(
            policy=args.policy,
            records=snapshot.records_scanned,
            entries=len(snapshot.directories),
            eligible=eligible_count,
            selected=len(selected_symbols),
            timestamp=snapshot.market_hours_timestamp_ns,
            mode=mode,
            selected_sha256=selected_sha256,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
