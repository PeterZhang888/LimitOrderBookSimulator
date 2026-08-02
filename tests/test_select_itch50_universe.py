#!/usr/bin/env python3
"""Deterministic unit test for startup-only ITCH universe selection."""

from __future__ import annotations

import csv
import contextlib
import gzip
import hashlib
import io
import json
import pathlib
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import select_itch50_universe as selector  # noqa: E402


def header(kind: str, locate: int, timestamp_ns: int) -> bytes:
    return (
        kind.encode("ascii")
        + locate.to_bytes(2, "big")
        + (1).to_bytes(2, "big")
        + timestamp_ns.to_bytes(6, "big")
    )


def stock_directory(
    symbol: str,
    locate: int,
    *,
    market_category: str = "Q",
    financial_status: str = "N",
    round_lot_size: int = 100,
    issue_classification: str = "C",
    authenticity: str = "P",
) -> bytes:
    message = (
        header("R", locate, 1)
        + symbol.ljust(8).encode("ascii")
        + market_category.encode("ascii")
        + financial_status.encode("ascii")
        + round_lot_size.to_bytes(4, "big")
        + b"N"
        + issue_classification.encode("ascii")
        + b"  "
        + authenticity.encode("ascii")
        + b"N"  # short-sale threshold indicator
        + b"N"  # IPO flag
        + b"1"  # LULD tier
        + b"N"  # ETP flag
        + (0).to_bytes(4, "big")
        + b"N"  # inverse indicator
    )
    assert len(message) == 39
    return message


def trading_action(symbol: str, locate: int, state: str, timestamp_ns: int) -> bytes:
    message = (
        header("H", locate, timestamp_ns)
        + symbol.ljust(8).encode("ascii")
        + state.encode("ascii")
        + b" "
        + b"TEST"
    )
    assert len(message) == 25
    return message


def system_event(code: str, timestamp_ns: int) -> bytes:
    message = header("S", 0, timestamp_ns) + code.encode("ascii")
    assert len(message) == 12
    return message


def write_fixture(path: pathlib.Path, messages: list[bytes]) -> None:
    with gzip.open(path, "wb") as output:
        for message in messages:
            output.write(len(message).to_bytes(2, "big"))
            output.write(message)


class StartupUniverseTest(unittest.TestCase):
    @staticmethod
    def startup_fixture(path: pathlib.Path) -> None:
        write_fixture(path, [
            stock_directory("QQQ", 4, issue_classification="E"),
            stock_directory("AAPL", 2),
            stock_directory("HALT", 3),
            trading_action("QQQ", 4, "T", 10),
            trading_action("AAPL", 2, "T", 11),
            trading_action("HALT", 3, "H", 12),
            system_event("Q", 20),
        ])

    def test_policy_uses_startup_state_and_qqq_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "startup.itch.gz"
            catalog = root / "catalog.csv"
            symbols = root / "symbols.txt"
            write_fixture(source, [
                stock_directory("QQQ", 4, issue_classification="E"),
                stock_directory("AAPL", 2),
                stock_directory("HALT", 3),
                stock_directory("ETF", 5, issue_classification="E"),
                stock_directory("BADLOT", 6, round_lot_size=0),
                trading_action("QQQ", 4, "T", 10),
                trading_action("AAPL", 2, "T", 11),
                trading_action("HALT", 3, "H", 12),
                trading_action("ETF", 5, "T", 13),
                trading_action("BADLOT", 6, "T", 14),
                system_event("Q", 20),
                # This is deliberately ignored: it is after start-of-market
                # hours and must not convert AAPL into an ineligible symbol.
                trading_action("AAPL", 2, "H", 21),
            ])

            result = selector.main([
                "--itch", str(source),
                "--catalog-out", str(catalog),
                "--symbols-out", str(symbols),
                "--max-symbols", "1",
            ])
            self.assertEqual(result, 0)
            with catalog.open(newline="") as source_file:
                rows = {row["symbol"]: row for row in csv.DictReader(source_file)}

            self.assertEqual(list(symbols.read_text().splitlines()), ["AAPL"])
            self.assertEqual(rows["AAPL"]["startup_trading_state"], "T")
            self.assertEqual(rows["AAPL"]["eligible"], "1")
            self.assertEqual(rows["AAPL"]["selected"], "1")
            self.assertEqual(rows["QQQ"]["eligible"], "1")
            self.assertEqual(rows["QQQ"]["selected"], "0")
            self.assertEqual(
                rows["QQQ"]["eligibility_reason"], "eligible_qqq_exception"
            )
            self.assertEqual(rows["HALT"]["eligible"], "0")
            self.assertIn("startup_trading_state_not_T", rows["HALT"]["eligibility_reason"])
            self.assertEqual(rows["ETF"]["eligible"], "0")
            self.assertIn("issue_not_common_or_qqq", rows["ETF"]["eligibility_reason"])
            self.assertEqual(rows["BADLOT"]["eligible"], "0")
            self.assertIn("nonpositive_round_lot", rows["BADLOT"]["eligibility_reason"])

    def test_fixed_cohort_is_normalized_validated_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "startup.itch.gz"
            fixed = root / "fixed.txt"
            catalog = root / "catalog.csv"
            symbols = root / "symbols.txt"
            provenance = root / "selection.json"
            self.startup_fixture(source)
            fixed.write_text("  aapl  \nqqq\n", encoding="utf-8")

            self.assertEqual(selector.main([
                "--itch", str(source),
                "--catalog-out", str(catalog),
                "--symbols-out", str(symbols),
                "--fixed-symbols", str(fixed),
                "--provenance-out", str(provenance),
            ]), 0)

            ordered = ["QQQ", "AAPL"]
            rendered = "QQQ\nAAPL\n"
            self.assertEqual(symbols.read_text(encoding="utf-8"), rendered)
            value = json.loads(provenance.read_text(encoding="utf-8"))
            self.assertEqual(value["mode"], "fixed_symbols")
            self.assertEqual(value["fixed_symbols_input"]["path"], str(fixed.resolve()))
            self.assertEqual(
                value["fixed_symbols_input"]["raw_sha256"],
                hashlib.sha256(fixed.read_bytes()).hexdigest(),
            )
            self.assertEqual(value["fixed_symbols_input"]["normalized_count"], 2)
            self.assertEqual(value["selected_symbols"]["ordered_symbols"], ordered)
            self.assertEqual(value["selected_symbols"]["count"], 2)
            self.assertEqual(
                value["selected_symbols"]["canonical_sha256"],
                hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            )
            with catalog.open(newline="", encoding="utf-8") as source_file:
                rows = {row["symbol"]: row for row in csv.DictReader(source_file)}
            self.assertEqual(rows["QQQ"]["selected"], "1")
            self.assertEqual(rows["AAPL"]["selected"], "1")
            self.assertEqual(rows["HALT"]["selected"], "0")

    def test_fixed_cohort_rejects_duplicate_after_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixed = pathlib.Path(temporary) / "fixed.txt"
            fixed.write_text("QQQ\naapl\n AAPL \n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate fixed symbol 'AAPL'"):
                selector.read_fixed_symbols(fixed)

    def test_fixed_cohort_rejects_unsafe_and_missing_qqq(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixed = pathlib.Path(temporary) / "fixed.txt"
            fixed.write_text("QQQ\n../AAPL\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe symbol"):
                selector.read_fixed_symbols(fixed)
            fixed.write_text("AAPL\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain QQQ"):
                selector.read_fixed_symbols(fixed)

    def test_fixed_cohort_rejects_absent_or_ineligible_startup_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "startup.itch.gz"
            self.startup_fixture(source)
            for contents, expected in (
                ("QQQ\nMISSING\n", "absent from startup directory: MISSING"),
                ("QQQ\nHALT\n", "ineligible at startup: HALT"),
            ):
                fixed = root / "fixed.txt"
                fixed.write_text(contents, encoding="utf-8")
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    with self.assertRaisesRegex(SystemExit, "2"):
                        selector.main([
                            "--itch", str(source),
                            "--catalog-out", str(root / "catalog.csv"),
                            "--symbols-out", str(root / "symbols.txt"),
                            "--fixed-symbols", str(fixed),
                        ])
                self.assertIn(expected, errors.getvalue())

    def test_fixed_cohort_and_cap_are_argparse_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            selector.build_argument_parser().parse_args([
                "--itch", "x", "--catalog-out", "c", "--symbols-out", "s",
                "--max-symbols", "10", "--fixed-symbols", "fixed.txt",
            ])

    def test_fixed_declaration_cannot_be_overwritten_by_an_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "startup.itch.gz"
            fixed = root / "fixed.txt"
            self.startup_fixture(source)
            fixed.write_text("QQQ\nAAPL\n", encoding="utf-8")
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                with self.assertRaises(SystemExit):
                    selector.main([
                        "--itch", str(source),
                        "--catalog-out", str(root / "catalog.csv"),
                        "--symbols-out", str(fixed),
                        "--fixed-symbols", str(fixed),
                    ])
            self.assertIn("must not overwrite --fixed-symbols", errors.getvalue())
            self.assertEqual(fixed.read_text(encoding="utf-8"), "QQQ\nAAPL\n")


if __name__ == "__main__":
    unittest.main()
