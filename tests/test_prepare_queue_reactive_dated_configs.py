#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.

from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "prepare_queue_reactive_dated_configs.py"
SPEC = importlib.util.spec_from_file_location("dated_configs", PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def write_csv(path: pathlib.Path, fields: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(fields)
        writer.writerows(rows)


class DatedConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.fields = [
            "book_id", "symbol", "data_dir", "hawkes_rates_file",
            *module.OPENING_FIELDS, "cluster_id", "fixed_parameter",
        ]
        self.pooled = self.root / "pooled.csv"
        write_csv(self.pooled, self.fields, [[
            0, "AAA", "/pooled/marks", "/pooled/rates.csv",
            10000, 9900, 10100, 100, 110, 3, "frozen",
        ]])
        self.opening = self.root / "opening.csv"
        write_csv(self.opening, self.fields, [[
            0, "AAA", "/oracle/marks", "/oracle/rates.csv",
            12000, 11900, 12100, 200, 210, 9, "oracle",
        ]])
        self.targets = self.root / "targets"
        directory = self.targets / "empirical_data" / "itch_20190130_aaa"
        for suffix in ("", "_window_300s", "_window_3600s"):
            write_csv(
                directory / f"market_targets_aaa_20190130{suffix}.csv",
                ["name", "target", "scale", "weight"],
                [["mean_spread_ticks", 2, 1, 1]],
            )
        (directory / "itch_manifest_aaa_20190130.json").write_text(
            json.dumps({
                "trading_date": "2019-01-30",
                "symbol": "AAA",
                "market_target_windows": {
                    "300": {
                        "file": "market_targets_aaa_20190130_window_300s.csv",
                        "duration_seconds": 300,
                    },
                    "3600": {
                        "file": "market_targets_aaa_20190130_window_3600s.csv",
                        "duration_seconds": 3600,
                    },
                },
            }) + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def argv(self, output: pathlib.Path) -> list[str]:
        return [
            "--pooled-config", str(self.pooled),
            "--dated-opening-config", f"2019-01-30={self.opening}",
            "--dated-target-root", f"2019-01-30={self.targets}",
            "--expected-date-count", "1",
            "--forbid-date", "2020-01-30",
            "--output-root", str(output),
        ]

    def test_only_opening_and_target_pointer_vary(self) -> None:
        output = self.root / "output"
        self.assertEqual(module.main(self.argv(output)), 0)
        with (output / "deployment_config.csv").open(newline="") as source:
            deployment = next(csv.DictReader(source))
        self.assertEqual(deployment["fundamental_price_ticks"], "10000")
        self.assertEqual(deployment["data_dir"], "/pooled/marks")
        self.assertNotIn("target_data_dir", deployment)
        with (output / "dated_config_20190130.csv").open(newline="") as source:
            row = next(csv.DictReader(source))
        self.assertEqual(row["data_dir"], "/pooled/marks")
        self.assertEqual(row["hawkes_rates_file"], "/pooled/rates.csv")
        self.assertEqual(row["cluster_id"], "3")
        self.assertEqual(row["fixed_parameter"], "frozen")
        self.assertEqual(row["fundamental_price_ticks"], "12000")
        self.assertEqual(row["initial_best_bid_depth"], "200")
        self.assertIn("itch_20190130_aaa", row["target_data_dir"])

    def test_forbidden_date_fails_before_output(self) -> None:
        argv = self.argv(self.root / "forbidden")
        argv[argv.index("--dated-opening-config") + 1] = (
            f"2020-01-30={self.opening}"
        )
        argv[argv.index("--dated-target-root") + 1] = (
            f"2020-01-30={self.targets}"
        )
        self.assertEqual(module.main(argv), 1)

    def test_same_day_model_fields_are_never_copied(self) -> None:
        output = self.root / "second"
        self.assertEqual(module.main(self.argv(output)), 0)
        with (output / "dated_config_20190130.csv").open(newline="") as source:
            row = next(csv.DictReader(source))
        self.assertNotEqual(row["data_dir"], "/oracle/marks")
        self.assertNotEqual(row["hawkes_rates_file"], "/oracle/rates.csv")
        self.assertNotEqual(row["fixed_parameter"], "oracle")

    def test_missing_certified_3600_second_prefix_fails_closed(self) -> None:
        target = (
            self.targets / "empirical_data/itch_20190130_aaa"
            / "market_targets_aaa_20190130_window_3600s.csv"
        )
        target.unlink()
        self.assertEqual(module.main(self.argv(self.root / "missing_3600")), 1)


if __name__ == "__main__":
    unittest.main()
