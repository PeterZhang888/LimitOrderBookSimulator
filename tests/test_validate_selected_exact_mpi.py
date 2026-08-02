#!/usr/bin/env python3

from __future__ import annotations

import csv
import pathlib
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import validate_selected_exact_mpi as validation  # noqa: E402


class SelectedExactValidationTest(unittest.TestCase):
    def test_model_arguments_reconstruct_selected_protocol(self) -> None:
        report = {
            "selected_parameters": {
                "threshold_bps": 10.0, "response_step_bps": 5.0,
                "base_order_quantity": 25,
                "volatility_bps_sqrt_second": 0.5,
            },
            "protocol": {
                "fixed_value_parameters": {
                    "max_order_quantity": 1000, "max_inventory": 2_000_000,
                    "decision_interval_ms": 1000.0,
                },
                "coupling_parameters": {
                    "arbitrage_trigger_bps": 5.0,
                    "arbitrage_release_bps": 2.5,
                    "shared_mm_exposure_threshold": 500.0,
                    "max_hedge_quantity": 1000,
                },
                "shared_mm_cross_book_hedging": False,
            },
        }
        arguments = validation.model_arguments(report)
        self.assertIn("--enable-value-agent", arguments)
        self.assertIn("--enable-etf-arbitrage", arguments)
        self.assertNotIn("--enable-shared-mm-hedging", arguments)

    def test_summary_normalization_ignores_execution_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "summary.csv"
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=["book_id", "trade_hash", "owner_rank",
                                "mpi_ranks", "wall_seconds"],
                )
                writer.writeheader()
                writer.writerow({
                    "book_id": 0, "trade_hash": 123, "owner_rank": 4,
                    "mpi_ranks": 5, "wall_seconds": 9.0,
                })
            self.assertEqual(
                validation.normalized_summary(path),
                [{"book_id": "0", "trade_hash": "123"}],
            )


if __name__ == "__main__":
    unittest.main()
