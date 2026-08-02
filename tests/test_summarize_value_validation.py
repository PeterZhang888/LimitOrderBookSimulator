#!/usr/bin/env python3
"""Tests for the chronological validation table generator."""

from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SYMBOLS = ("QQQ", "AAPL", "MSFT", "AMZN")
METRICS = (
    "mean_spread_ticks", "mean_bid_depth", "mean_ask_depth",
    "mid_move_rate", "return_variance", "return_kurtosis",
    "absolute_return_acf1",
)


class SummarizeValueValidationTests(unittest.TestCase):
    def test_writes_complete_metric_and_markdown_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            targets = root / "data"
            results = root / "results"
            for date in ("20191230", "20200130"):
                for symbol in SYMBOLS:
                    directory = targets / f"itch_{date}_{symbol.lower()}"
                    directory.mkdir(parents=True)
                    target_path = directory / (
                        f"market_targets_{symbol.lower()}_{date}.csv"
                    )
                    with target_path.open("w", newline="") as output:
                        writer = csv.DictWriter(
                            output, fieldnames=("name", "target", "scale", "weight")
                        )
                        writer.writeheader()
                        for metric in METRICS:
                            writer.writerow({
                                "name": metric, "target": 1.0,
                                "scale": 1.0, "weight": 1.0,
                            })
            summary_fields = ("symbol", "structurally_valid", *METRICS)
            for split in ("coupled_training", "heldout_validation"):
                for seed, value in ((7, 2.0), (11, 4.0)):
                    directory = results / split / f"seed_{seed}"
                    directory.mkdir(parents=True)
                    with (directory / "sequential_multi_asset_summary.csv").open(
                        "w", newline=""
                    ) as output:
                        writer = csv.DictWriter(output, fieldnames=summary_fields)
                        writer.writeheader()
                        for symbol in SYMBOLS:
                            writer.writerow({
                                "symbol": symbol,
                                "structurally_valid": 1,
                                **{metric: value for metric in METRICS},
                            })
            report = {
                "protocol": {
                    "training_date": "2019-12-30",
                    "heldout_date": "2020-01-30",
                    "seeds": [7, 11],
                    "selection_uses_heldout_targets": False,
                },
                "selected_parameters": {
                    "threshold_bps": 10.0,
                    "response_step_bps": 5.0,
                    "base_order_quantity": 25,
                    "volatility_bps_sqrt_second": 0.0,
                },
                "coupled_training_mean_score": 1.0,
                "heldout_mean_score": 1.0,
                "coupled_training_wall_seconds": [3.0],
                "heldout_wall_seconds": [4.0],
            }
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report))
            detail = root / "detail.csv"
            markdown = root / "summary.md"
            subprocess.run([
                sys.executable,
                str(ROOT / "scripts" / "summarize_value_validation.py"),
                "--report", str(report_path),
                "--target-root", str(targets),
                "--result-root", str(results),
                "--output-csv", str(detail),
                "--output-markdown", str(markdown),
            ], check=True, stdout=subprocess.PIPE, text=True)
            with detail.open(newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 2 * len(SYMBOLS) * len(METRICS))
            self.assertTrue(all(float(row["simulated_mean"]) == 3.0 for row in rows))
            self.assertTrue(all(float(row["simulation_mc_se"]) == 1.0 for row in rows))
            self.assertTrue(all(float(row["importance_weight"]) == 1.0 for row in rows))
            text = markdown.read_text()
            self.assertIn("| Held out | 2", text)
            self.assertIn("| QQQ | 2 | 2 |", text)


if __name__ == "__main__":
    unittest.main()
