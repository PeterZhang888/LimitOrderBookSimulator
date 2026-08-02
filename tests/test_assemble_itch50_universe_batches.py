#!/usr/bin/env python3
"""Regression test for safe assembly of bounded ITCH extraction batches."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import assemble_itch50_universe_batches as assembler  # noqa: E402


class BatchAssemblyTest(unittest.TestCase):
    def test_moves_valid_book_and_retains_invalid_opening_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            selected = root / "selected.txt"
            selected.write_text("QQQ\nAAPL\n", encoding="utf-8")
            batch = root / "batches" / "batch_00000"
            qqq = batch / "itch_20200130_qqq"
            qqq.mkdir(parents=True)
            (qqq / "marker.txt").write_text("QQQ", encoding="utf-8")
            basket = batch / "itch_20200130_basket"
            basket.mkdir()
            with (basket / "opening_bbo_20200130.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(assembler.OPENING_FIELDS))
                writer.writeheader()
                writer.writerow({
                    "symbol": "QQQ", "best_bid_ticks": 100,
                    "best_ask_ticks": 101, "best_bid_depth": 20,
                    "best_ask_depth": 30, "mid_price_ticks": 100.5,
                })
            with (batch / "itch_20200130_exclusions.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(assembler.EXCLUSION_FIELDS))
                writer.writeheader()
                writer.writerow({"symbol": "AAPL", "reason": "not_two_sided_at_09:30:00"})

            result = assembler.assemble(argparse.Namespace(
                symbols_file=str(selected),
                batch_root=str(root / "batches"),
                trading_date="2020-01-30",
                data_root=str(root / "empirical_data"),
                opening_bbo_out=str(root / "opening.csv"),
                candidate_catalog_out=str(root / "candidates.csv"),
                exclusions_out=str(root / "exclusions.csv"),
                manifest_out=str(root / "assembly.json"),
                source_catalog=None,
            ))

            self.assertEqual(result["selected_symbols"], 2)
            self.assertEqual(result["valid_two_sided_openings"], 1)
            self.assertEqual(result["invalid_opening_exclusions"], 1)
            self.assertTrue((root / "empirical_data" / "itch_20200130_qqq" / "marker.txt").is_file())
            self.assertFalse(qqq.exists())
            with (root / "candidates.csv").open(newline="") as source:
                self.assertEqual(
                    [row["symbol"] for row in csv.DictReader(source)],
                    ["QQQ", "AAPL"],
                )
            with (root / "assembly.json").open() as source:
                manifest = json.load(source)
            self.assertEqual(manifest["counts"]["moved_symbol_directories"], 1)
            self.assertEqual(
                manifest["invalid_opening_exclusions"],
                [{"symbol": "AAPL", "reason": "not_two_sided_at_09:30:00"}],
            )


if __name__ == "__main__":
    unittest.main()
