#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Tests for explicit training/held-out ITCH universe intersection."""

from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "intersect_universe", ROOT / "scripts" / "intersect_empirical_universe_configs.py",
)
assert SPEC is not None and SPEC.loader is not None
INTERSECT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INTERSECT
SPEC.loader.exec_module(INTERSECT)


FIELDS = ("book_id", "symbol", "data_dir", "hawkes_rates_file")


def write_config(path: pathlib.Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class UniverseIntersectionTest(unittest.TestCase):
    def test_intersection_preserves_training_order_and_reindexes_each_day(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            training = root / "training.csv"
            heldout = root / "heldout.csv"
            write_config(training, [
                {"book_id": "0", "symbol": "QQQ", "data_dir": "/train/qqq", "hawkes_rates_file": "/train/qqq/rates"},
                {"book_id": "1", "symbol": "AAA", "data_dir": "/train/aaa", "hawkes_rates_file": "/train/aaa/rates"},
                {"book_id": "2", "symbol": "BBB", "data_dir": "/train/bbb", "hawkes_rates_file": "/train/bbb/rates"},
            ])
            write_config(heldout, [
                {"book_id": "0", "symbol": "BBB", "data_dir": "/hold/bbb", "hawkes_rates_file": "/hold/bbb/rates"},
                {"book_id": "1", "symbol": "QQQ", "data_dir": "/hold/qqq", "hawkes_rates_file": "/hold/qqq/rates"},
                {"book_id": "2", "symbol": "CCC", "data_dir": "/hold/ccc", "hawkes_rates_file": "/hold/ccc/rates"},
            ])
            training_out = root / "common_training.csv"
            heldout_out = root / "common_heldout.csv"
            report = root / "report.json"
            exit_code = INTERSECT.main([
                "--training-config", str(training),
                "--heldout-config", str(heldout),
                "--training-output", str(training_out),
                "--heldout-output", str(heldout_out),
                "--report", str(report),
                "--minimum-symbols", "2",
            ])
            self.assertEqual(exit_code, 0)
            with training_out.open(newline="", encoding="utf-8") as source:
                training_rows = list(csv.DictReader(source))
            with heldout_out.open(newline="", encoding="utf-8") as source:
                heldout_rows = list(csv.DictReader(source))
            self.assertEqual([row["symbol"] for row in training_rows], ["QQQ", "BBB"])
            self.assertEqual([row["symbol"] for row in heldout_rows], ["QQQ", "BBB"])
            self.assertEqual([row["book_id"] for row in heldout_rows], ["0", "1"])
            self.assertEqual(heldout_rows[0]["data_dir"], "/hold/qqq")
            with report.open(encoding="utf-8") as source:
                result = json.load(source)
            self.assertEqual(result["only_training_symbols"], ["AAA"])
            self.assertEqual(result["only_heldout_symbols"], ["CCC"])

    def test_mismatched_headers_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            training = root / "training.csv"
            heldout = root / "heldout.csv"
            write_config(training, [
                {"book_id": "0", "symbol": "AAA", "data_dir": "/train", "hawkes_rates_file": "/train/rates"},
            ])
            with heldout.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=("book_id", "symbol", "data_dir"))
                writer.writeheader()
                writer.writerow({"book_id": "0", "symbol": "AAA", "data_dir": "/hold"})
            with self.assertRaisesRegex(INTERSECT.IntersectionError, "identical headers"):
                INTERSECT.run(type("Args", (), {
                    "training_config": str(training), "heldout_config": str(heldout),
                    "training_output": str(root / "train_out.csv"),
                    "heldout_output": str(root / "hold_out.csv"),
                    "report": str(root / "report.json"), "minimum_symbols": 1,
                    "overwrite": False,
                })())


if __name__ == "__main__":
    unittest.main()
