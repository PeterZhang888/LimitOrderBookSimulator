#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Regression tests for the chronological calibration leakage barriers."""

from __future__ import annotations

import csv
import importlib.util
import math
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "value_calibration",
    ROOT / "scripts" / "calibrate_and_validate_value_agent.py",
)
assert SPEC is not None and SPEC.loader is not None
CALIBRATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CALIBRATION
SPEC.loader.exec_module(CALIBRATION)


FIELDS = (
    "book_id", "symbol", "data_dir", "hawkes_rates_file",
    "fundamental_price_ticks", "initial_best_bid_ticks",
    "initial_best_ask_ticks", "initial_best_bid_depth",
    "initial_best_ask_depth", "beta", "basket_weight",
    "market_maker_quote_quantity", "target_spread_ticks",
    "quote_improvement_probability",
)


def write_config(path: pathlib.Path, *, mutate: tuple[str, str, str] | None = None) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for book_id, symbol in enumerate(CALIBRATION.SYMBOLS):
            row = {
                "book_id": book_id,
                "symbol": symbol,
                "data_dir": f"/data/itch_20191230_{symbol.lower()}",
                "hawkes_rates_file": (
                    f"/data/itch_20191230_{symbol.lower()}/"
                    f"hawkes_rates_{symbol.lower()}_balanced_20191230.csv"
                ),
                "fundamental_price_ticks": 10_000 + book_id,
                "initial_best_bid_ticks": 9_999 + book_id,
                "initial_best_ask_ticks": 10_001 + book_id,
                "initial_best_bid_depth": 100,
                "initial_best_ask_depth": 100,
                "beta": 1.0 + book_id,
                "basket_weight": 0.0 if book_id == 0 else 1.0 / 3.0,
                "market_maker_quote_quantity": 100,
                "target_spread_ticks": 2,
                "quote_improvement_probability": 0.01,
            }
            if mutate is not None and symbol == mutate[0]:
                row[mutate[1]] = mutate[2]
            writer.writerow(row)


class CalibrationProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.training = self.root / "training.csv"
        self.heldout = self.root / "heldout.csv"
        write_config(self.training)
        write_config(self.heldout)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def validate(self) -> None:
        CALIBRATION.validate_frozen_backgrounds(
            self.training,
            self.heldout,
            "2019-12-30",
            "2020-01-30",
        )

    def test_opening_state_and_price_ratio_beta_may_change(self) -> None:
        write_config(
            self.heldout,
            mutate=("AAPL", "fundamental_price_ticks", "12345"),
        )
        self.validate()
        write_config(self.heldout, mutate=("AAPL", "beta", "9.75"))
        self.validate()

    def test_fitted_fields_are_frozen(self) -> None:
        for field in (
            "data_dir",
            "hawkes_rates_file",
            "basket_weight",
            "market_maker_quote_quantity",
            "target_spread_ticks",
            "quote_improvement_probability",
        ):
            with self.subTest(field=field):
                write_config(self.heldout, mutate=("MSFT", field, "different"))
                with self.assertRaisesRegex(ValueError, "must be frozen"):
                    self.validate()

    def test_heldout_path_token_is_rejected(self) -> None:
        write_config(
            self.training,
            mutate=("AMZN", "data_dir", "/data/itch_20200130_amzn"),
        )
        write_config(
            self.heldout,
            mutate=("AMZN", "data_dir", "/data/itch_20200130_amzn"),
        )
        with self.assertRaisesRegex(ValueError, "exclusively"):
            self.validate()

    def write_summary(self, path: pathlib.Path, values: list[float], *,
                      structurally_valid: int = 1,
                      complete_samples: bool = True) -> None:
        fields = (
            "symbol", "structurally_valid", "sample_count",
            "expected_sample_count", *CALIBRATION.METRICS,
        )
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for symbol in CALIBRATION.SYMBOLS:
                writer.writerow({
                    "symbol": symbol,
                    "structurally_valid": structurally_valid,
                    "sample_count": 10,
                    "expected_sample_count": 10 if complete_samples else 11,
                    **{
                        metric: values[index]
                        for index, metric in enumerate(CALIBRATION.METRICS)
                    },
                })

    def targets(self, *, first_weight: float = 1.0) -> dict[str, dict[str, object]]:
        return {
            symbol: {
                metric: CALIBRATION.TargetMoment(
                    target=10.0,
                    empirical_scale=2.0,
                    weight=first_weight if metric == CALIBRATION.METRICS[0] else 1.0,
                )
                for metric in CALIBRATION.METRICS
            }
            for symbol in CALIBRATION.SYMBOLS
        }

    def test_weighted_seed_mean_and_mc_uncertainty_are_reported(self) -> None:
        first = self.root / "first.csv"
        second = self.root / "second.csv"
        self.write_summary(first, [12.0] * len(CALIBRATION.METRICS))
        self.write_summary(second, [16.0] * len(CALIBRATION.METRICS))
        score, estimates = CALIBRATION.weighted_moment_loss(
            [first, second], self.targets(),
        )
        self.assertAlmostEqual(score, 2.0)
        estimate = estimates[0]
        self.assertAlmostEqual(estimate.simulated_mean, 14.0)
        self.assertAlmostEqual(estimate.simulated_mean_se, 2.0)
        self.assertAlmostEqual(estimate.empirical_standardized_residual, 2.0)
        self.assertAlmostEqual(
            estimate.combined_uncertainty_residual, math.sqrt(2.0),
        )
        combined, _ = CALIBRATION.weighted_moment_loss(
            [first, second], self.targets(), uncertainty_mode="combined",
        )
        self.assertAlmostEqual(combined, math.sqrt(2.0))

    def test_target_weights_change_the_fit_objective(self) -> None:
        first = self.root / "weighted.csv"
        values = [14.0] + [10.0] * (len(CALIBRATION.METRICS) - 1)
        self.write_summary(first, values)
        score, _ = CALIBRATION.weighted_moment_loss(
            [first], self.targets(first_weight=3.0),
        )
        # Every asset has one z=2 moment with weight 3 and six zero-error
        # moments with weight 1: sqrt((4*3*2^2)/(4*(3+6))).
        self.assertAlmostEqual(score, math.sqrt(4.0 / 3.0))

    def test_residual_is_not_capped(self) -> None:
        path = self.root / "uncapped.csv"
        self.write_summary(path, [210.0] * len(CALIBRATION.METRICS))
        score, _ = CALIBRATION.weighted_moment_loss([path], self.targets())
        self.assertAlmostEqual(score, 100.0)

    def test_incomplete_samples_are_rejected(self) -> None:
        path = self.root / "incomplete.csv"
        self.write_summary(path, [10.0] * len(CALIBRATION.METRICS),
                           complete_samples=False)
        with self.assertRaisesRegex(ValueError, "incomplete fixed-clock"):
            CALIBRATION.weighted_moment_loss([path], self.targets())


if __name__ == "__main__":
    unittest.main()
