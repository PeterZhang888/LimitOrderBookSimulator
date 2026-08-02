#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import derive_hawkes_rates as rates  # noqa: E402


class HawkesRateDerivationTest(unittest.TestCase):
    def empirical_fixture(self, symbol: str) -> pathlib.Path:
        path = PROJECT_ROOT / "data" / f"itch_20200130_{symbol.lower()}"
        if not path.is_dir():
            self.skipTest(f"external ITCH fixture is not installed: {path}")
        return path

    def test_default_excitation_is_diagonal_and_matches_runtime_contract(self) -> None:
        alpha = rates.default_alpha()
        for row in range(6):
            for column in range(6):
                expected = rates.SELF_EXCITATION_AMPLITUDE if row == column else 0.0
                self.assertEqual(alpha[row][column], expected)
        self.assertEqual(
            rates.excitation_settings(),
            {
                "excitation_structure": "diagonal_self_excitation_only",
                "self_excitation_amplitude": 0.20,
                "cross_excitation_amplitude": 0.0,
            },
        )

    def test_stationary_rate_equation_is_recovered(self) -> None:
        observed = [47.0, 48.0, 0.9, 1.0, 46.0, 45.0]
        activity = 0.30
        beta = 10.0
        alpha = rates.default_alpha()
        mu = rates.derive(observed, activity, beta, alpha)
        for row in range(6):
            reconstructed = activity * mu[row] + sum(
                alpha[row][column] * observed[column] / beta
                for column in range(6)
            )
            self.assertAlmostEqual(reconstructed, observed[row], places=12)

    def test_sparse_acnb_like_target_has_nonnegative_exact_inversion(self) -> None:
        # Job 45257 failed because a fixed limit-buy -> market-buy coefficient
        # contributed more intensity than ACNB's sparse market-buy target.  A
        # diagonal kernel remains feasible for every nonnegative marginal-rate
        # vector and reconstructs this exact failure-class target.
        target = [
            0.018,
            0.017,
            6.4102564102564103e-05,
            8.0e-05,
            0.012,
            0.011,
        ]
        activity = 0.30
        beta = 10.0
        alpha = rates.default_alpha()
        mu = rates.derive(target, activity, beta, alpha)
        self.assertTrue(all(value >= 0.0 for value in mu))
        for row in range(6):
            reconstructed = activity * mu[row] + sum(
                alpha[row][column] * target[column] / beta
                for column in range(6)
            )
            self.assertAlmostEqual(reconstructed, target[row], places=15)

    def test_infeasible_cross_excitation_is_rejected_not_clipped(self) -> None:
        target = [0.018, 0.017, 6.4102564102564103e-05, 8.0e-05, 0.012, 0.011]
        alpha = rates.default_alpha()
        alpha[2][0] = 0.04
        with self.assertRaisesRegex(
            ValueError,
            "stationary target is infeasible.*market_buy",
        ):
            rates.derive(target, 0.30, 10.0, alpha)

    def test_best_depth_balance_raises_qqq_cancel_rates(self) -> None:
        data_dir = self.empirical_fixture("QQQ")
        observed = [47.037, 47.321, 0.889, 0.893, 45.98, 46.627]
        adjusted = rates.balance_best_depth(data_dir, observed)
        self.assertEqual(adjusted[:4], observed[:4])
        self.assertGreater(adjusted[4], observed[4])
        self.assertGreater(adjusted[5], observed[5])

    def test_directional_balance_equalises_expected_pair_volume(self) -> None:
        data_dir = self.empirical_fixture("AMZN")
        observed = [6.55, 6.39, 0.396, 0.410, 5.48, 5.32]
        adjusted = rates.balance_directional_volume(data_dir, observed)
        quantity_files = (
            "limit_buy_quantity_distribution.txt",
            "limit_sell_quantity_distribution.txt",
            "market_buy_quantity_distribution.txt",
            "market_sell_quantity_distribution.txt",
            "cancel_bid_quantity_distribution.txt",
            "cancel_ask_quantity_distribution.txt",
        )
        means = [
            rates.weighted_distribution(data_dir / filename, "quantity")[0]
            for filename in quantity_files
        ]
        for left, right in ((0, 1), (2, 3), (4, 5)):
            self.assertAlmostEqual(
                adjusted[left] * means[left],
                adjusted[right] * means[right],
                places=10,
            )
            self.assertAlmostEqual(
                adjusted[left] + adjusted[right],
                observed[left] + observed[right],
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
