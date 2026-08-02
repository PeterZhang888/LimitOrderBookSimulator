#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import analyze_liquidity_shock as analysis  # noqa: E402


FIELDS = (
    "book_id", "symbol", "exchange_time_ns", "best_bid_ticks",
    "best_ask_ticks", "best_bid_depth", "best_ask_depth",
    "last_trade_price_ticks", "mid_price_ticks", "fundamental_value_ticks",
    "cumulative_aggressive_buy", "cumulative_aggressive_sell",
)


def write_trace(path: pathlib.Path, shocked: bool) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for book_id, symbol in ((0, "QQQ"), (1, "AAPL")):
            for second in range(1, 7):
                bid, ask, bid_depth, ask_depth = 1000, 1200, 500, 500
                if shocked and second in (3, 4):
                    if book_id == 1 and second == 3:
                        bid, bid_depth = 900, 100
                    if book_id == 0 and second == 4:
                        ask, ask_depth = 1300, 200
                writer.writerow({
                    "book_id": book_id, "symbol": symbol,
                    "exchange_time_ns": second * 1_000_000_000,
                    "best_bid_ticks": bid, "best_ask_ticks": ask,
                    "best_bid_depth": bid_depth, "best_ask_depth": ask_depth,
                    "last_trade_price_ticks": 0,
                    "mid_price_ticks": 0.5 * (bid + ask),
                    "fundamental_value_ticks": 1100,
                    "cumulative_aggressive_buy": 0,
                    "cumulative_aggressive_sell": 0,
                })


class ShockAnalysisTest(unittest.TestCase):
    def test_propagation_and_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            control, shock = root / "control.csv", root / "shock.csv"
            write_trace(control, False)
            write_trace(shock, True)
            args = argparse.Namespace(
                control_trace=str(control), shock_trace=str(shock),
                shock_time_ns=3_000_000_000, shock_book=1, tick_size=100,
                spread_tolerance_ticks=0.0, mid_tolerance_ticks=0.0,
                depth_relative_tolerance=0.0, depth_absolute_tolerance=0,
                recovery_window_samples=2,
            )
            report = analysis.analyze(args)
            self.assertEqual(report["system"]["affected_book_count"], 2)
            self.assertEqual(report["system"]["cross_asset_affected_book_count"], 1)
            self.assertEqual(report["system"]["first_cross_asset_response_seconds"], 1.0)
            self.assertEqual(report["system"]["system_recovery_seconds"], 2.0)
            aapl = report["books"][1]
            self.assertEqual(aapl["propagation_delay_seconds"], 0.0)
            self.assertEqual(aapl["peak_best_depth_loss"], 400)


if __name__ == "__main__":
    unittest.main()
