#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Tests for quote-improvement inputs in the real-universe builder."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_itch_universe_config as builder  # noqa: E402


class QuoteImprovementBuilderTest(unittest.TestCase):
    @staticmethod
    def stats(total: int, zero: int) -> builder.DistributionStats:
        return builder.DistributionStats(
            total_count=total,
            weighted_median=0,
            mean=0.0,
            zero_count=zero,
            zero_fraction=zero / total,
        )

    def test_uses_combined_zero_count_not_the_eligible_rate(self) -> None:
        manifest = {
            "placement_counts": {
                "improvement_eligible_limit_orders": 20,
                "inside_spread_limit_orders": 3,
            }
        }
        probability = builder.quote_improvement_probability(
            manifest, self.stats(10, 1), self.stats(10, 5)
        )
        self.assertAlmostEqual(probability, 3 / 6)
        self.assertNotAlmostEqual(probability, 3 / 20)

    def test_rejects_inside_count_above_combined_zero_count(self) -> None:
        manifest = {
            "placement_counts": {
                "improvement_eligible_limit_orders": 20,
                "inside_spread_limit_orders": 7,
            }
        }
        with self.assertRaisesRegex(
            ValueError, "exceed_combined_zero_distance"
        ):
            builder.quote_improvement_probability(
                manifest, self.stats(10, 1), self.stats(10, 5)
            )


class RateDerivationBuilderTest(unittest.TestCase):
    @staticmethod
    def shift_field(path: pathlib.Path, field: str, amount: float) -> None:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = tuple(reader.fieldnames or ())
            rows = list(reader)
        rows[0][field] = str(float(rows[0][field]) + amount)
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def write_rates(
        self, root: pathlib.Path, *, reconstructed_offset: float = 0.0,
    ) -> tuple[pathlib.Path, pathlib.Path]:
        compact = "20200130"
        data_root = root / "data"
        builder.write_self_test_symbol(data_root, compact, "QQQ")
        directory = data_root / f"itch_{compact}_qqq"
        manifest = directory / f"itch_manifest_qqq_{compact}.json"
        path = directory / "rates.csv"
        builder.hawkes.run(argparse.Namespace(
            manifest=str(manifest), output=str(path),
            activity_scale=0.3, beta=10.0,
            balance_directional_volume=True, balance_best_depth=True,
            balance_strength=1.0,
        ))
        if reconstructed_offset:
            self.shift_field(
                path, "stationary_reconstructed_rate", reconstructed_offset
            )
        return path, manifest

    @staticmethod
    def validate(
        path: pathlib.Path, manifest: pathlib.Path,
    ) -> dict[str, object]:
        return builder.validate_generated_rates(
            path, label="fixture", manifest_path=manifest,
            activity_scale=0.3, kernel_beta=10.0,
            balance_directional_volume=True, balance_best_depth=True,
            balance_strength=1.0,
        )

    def test_accepts_transformed_targets_when_reconstruction_matches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = self.write_rates(pathlib.Path(directory))
            audit = self.validate(path, manifest)
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["event_types_checked"], 6)
        self.assertTrue(
            audit["stationary_reconstruction_equals_target_per_type"]
        )
        self.assertTrue(
            audit["observed_rates_equal_manifest_counts_per_duration"]
        )
        self.assertTrue(
            audit["stationary_targets_equal_declared_transforms_per_type"]
        )
        self.assertTrue(
            audit[
                "reported_reconstruction_equals_configured_rate_equation_per_type"
            ]
        )
        self.assertEqual(audit["transform_settings"], {
            "activity_scale": 0.3,
            "kernel_beta": 10.0,
            "balance_directional_volume": True,
            "balance_best_depth": True,
            "balance_strength": 1.0,
            **builder.hawkes.excitation_settings(),
        })

    def test_rejects_stationary_reconstruction_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = self.write_rates(
                pathlib.Path(directory), reconstructed_offset=0.01
            )
            with self.assertRaisesRegex(
                builder.UniverseBuildError,
                "reported stationary reconstruction disagrees",
            ):
                self.validate(path, manifest)

    def test_rejects_observed_rate_not_derived_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = self.write_rates(pathlib.Path(directory))
            self.shift_field(path, "observed_rate_per_second", 0.01)
            with self.assertRaisesRegex(
                builder.UniverseBuildError, "manifest count/duration",
            ):
                self.validate(path, manifest)

    def test_rejects_target_not_derived_from_declared_transforms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = self.write_rates(pathlib.Path(directory))
            self.shift_field(path, "stationary_target_rate", 0.01)
            with self.assertRaisesRegex(
                builder.UniverseBuildError,
                "declared reduced-book transforms",
            ):
                self.validate(path, manifest)

    def test_rejects_rate_file_under_wrong_transform_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = self.write_rates(pathlib.Path(directory))
            with self.assertRaisesRegex(
                builder.UniverseBuildError,
                "declared reduced-book transforms",
            ):
                builder.validate_generated_rates(
                    path, label="fixture", manifest_path=manifest,
                    activity_scale=0.3, kernel_beta=10.0,
                    balance_directional_volume=False,
                    balance_best_depth=False, balance_strength=0.0,
                )

    def test_balanced_reduced_book_derivation_is_the_default(self) -> None:
        args = builder.build_parser().parse_args([])
        self.assertTrue(args.balance_directional_volume)
        self.assertTrue(args.balance_best_depth)
        self.assertEqual(args.balance_strength, 1.0)
        self.assertEqual(args.rate_label, "universe_balanced")

    def test_end_to_end_builder_records_rate_audit_artifacts(self) -> None:
        builder.run_self_test()


if __name__ == "__main__":
    unittest.main()
