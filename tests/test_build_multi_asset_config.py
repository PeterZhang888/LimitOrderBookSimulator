#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.

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

import build_multi_asset_config as builder  # noqa: E402
import calibrate_and_validate_value_agent as calibration  # noqa: E402


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class MultiAssetConfigBuilderTest(unittest.TestCase):
    def test_weight_source_must_predate_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "weights.csv"
            write_csv(
                path,
                ["symbol", "raw_qqq_portfolio_weight", "source_as_of", "filing_date"],
                [
                    {"symbol": symbol, "raw_qqq_portfolio_weight": 0.1,
                     "source_as_of": "2019-09-30", "filing_date": "2020-01-02"}
                    for symbol in builder.SYMBOLS[1:]
                ],
            )
            with self.assertRaisesRegex(ValueError, "occurs after calibration"):
                builder.read_weights(path, "20191230")

    def test_heldout_opening_uses_frozen_training_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            openings = []
            prices = {"QQQ": 2000, "AAPL": 3000, "MSFT": 1600, "AMZN": 16000}
            for symbol in builder.SYMBOLS:
                price = prices[symbol]
                openings.append({
                    "symbol": symbol,
                    "clock": "09:30:00",
                    "best_bid_ticks": price - 1,
                    "best_ask_ticks": price + 1,
                    "best_bid_depth": 100,
                    "best_ask_depth": 120,
                    "mid_price_ticks": float(price),
                })
                lower = symbol.lower()
                training = root / "data" / f"itch_20191230_{lower}"
                write_csv(
                    training / f"market_targets_{lower}_20191230.csv",
                    ["name", "target", "scale", "weight"],
                    [{"name": "mean_spread_ticks", "target": 3.2,
                      "scale": 1.0, "weight": 1.0}],
                )
                for side, median in (("buy", 30), ("sell", 50)):
                    write_csv(
                        training / f"limit_{side}_quantity_distribution.txt",
                        ["quantity", "count"],
                        [{"quantity": median, "count": 10}],
                    )
                    write_csv(
                        training / f"limit_{side}_distance_distribution.txt",
                        ["distance_ticks", "count"],
                        [{"distance_ticks": 0, "count": 10}],
                    )
                with (training / f"itch_manifest_{lower}_20191230.json").open("w") as output:
                    json.dump({
                        "placement_counts": {
                            "improvement_eligible_limit_orders": 20,
                            "inside_spread_limit_orders": 3,
                        }
                    }, output)
            write_csv(
                root / "data/itch_20200130_basket/opening_bbo_20200130.csv",
                list(openings[0].keys()), openings,
            )
            weights = root / "weights.csv"
            write_csv(
                weights,
                ["symbol", "raw_qqq_portfolio_weight", "source_as_of", "filing_date"],
                [
                    {"symbol": "AAPL", "raw_qqq_portfolio_weight": 0.1160,
                     "source_as_of": "2019-09-30", "filing_date": "2019-12-20"},
                    {"symbol": "MSFT", "raw_qqq_portfolio_weight": 0.1069,
                     "source_as_of": "2019-09-30", "filing_date": "2019-12-20"},
                    {"symbol": "AMZN", "raw_qqq_portfolio_weight": 0.0814,
                     "source_as_of": "2019-09-30", "filing_date": "2019-12-20"},
                ],
            )
            rows = builder.build(argparse.Namespace(
                data_root=str(root / "data"),
                opening_date="2020-01-30",
                calibration_date="2019-12-30",
                weights_file=str(weights),
            ))

            self.assertEqual(len(rows), 4)
            self.assertIn("itch_20191230_qqq", str(rows[0]["data_dir"]))
            self.assertEqual(rows[0]["fundamental_price_ticks"], "2000.0")
            self.assertEqual(rows[0]["market_maker_quote_quantity"], 40)
            self.assertEqual(rows[0]["target_spread_ticks"], 3)
            self.assertAlmostEqual(rows[0]["quote_improvement_probability"], 0.15)
            self.assertAlmostEqual(float(rows[1]["beta"]), 1.5)
            self.assertEqual(rows[1]["basket_weight"], 0.1160)

    def test_heldout_protocol_rejects_refitted_backgrounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fieldnames = list(builder.FIELDNAMES)
            base_rows = []
            for index, symbol in enumerate(builder.SYMBOLS):
                lower = symbol.lower()
                base_rows.append({
                    "book_id": index, "symbol": symbol,
                    "data_dir": f"data/itch_20191230_{lower}",
                    "hawkes_rates_file": f"data/itch_20191230_{lower}/rates.csv",
                    "fundamental_price_ticks": 1000,
                    "initial_best_bid_ticks": 999,
                    "initial_best_ask_ticks": 1001,
                    "initial_best_bid_depth": 100,
                    "initial_best_ask_depth": 100,
                    "beta": 1.0, "basket_weight": 0.1 if index else 0.0,
                    "market_maker_quote_quantity": 50,
                    "target_spread_ticks": 2,
                    "quote_improvement_probability": 0.1,
                })
            training = root / "training.csv"
            heldout = root / "heldout.csv"
            write_csv(training, fieldnames, base_rows)
            write_csv(heldout, fieldnames, base_rows)
            calibration.validate_frozen_backgrounds(
                training, heldout, "2019-12-30", "2020-01-30"
            )

            leaked_rows = [dict(row) for row in base_rows]
            leaked_rows[0]["data_dir"] = "data/itch_20200130_qqq"
            write_csv(heldout, fieldnames, leaked_rows)
            with self.assertRaisesRegex(ValueError, "refits QQQ field data_dir"):
                calibration.validate_frozen_backgrounds(
                    training, heldout, "2019-12-30", "2020-01-30"
                )


if __name__ == "__main__":
    unittest.main()
