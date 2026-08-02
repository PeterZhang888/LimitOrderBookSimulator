#!/usr/bin/env python3
"""Regression tests for auditable multi-day empirical-input pooling."""

from __future__ import annotations

import csv
import json
import math
import pathlib
import shutil
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pool_multiday_empirical_universe as pooling  # noqa: E402


SYMBOLS = ("QQQ", "AAA")
QUANTITY_EVENTS = (
    "limit_buy", "limit_sell", "market_buy", "market_sell", "cancel_bid", "cancel_ask",
)
DISTANCE_EVENTS = ("limit_buy", "limit_sell", "cancel_bid", "cancel_ask")


def write_csv(path: pathlib.Path, fields: tuple[str, ...] | list[str],
              rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


class MultiDayPoolingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.training_one = self.write_day("2019-01-30", multiplier=1)
        self.training_two = self.write_day("2019-03-27", multiplier=3)
        self.heldout = self.write_day("2020-01-30", multiplier=5)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_five_day_launcher_uses_canonical_balanced_rate_derivation(self) -> None:
        launcher = (
            PROJECT_ROOT / "submit_five_day_pooled_training.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('BALANCE_STRENGTH="1.0"', launcher)
        self.assertIn("--balance-directional-volume", launcher)
        self.assertIn("--balance-best-depth", launcher)
        self.assertNotIn("--no-balance-directional-volume", launcher)
        self.assertNotIn("--no-balance-best-depth", launcher)
        self.assertIn("--require-certification-cohort", launcher)
        self.assertIn(
            "scripts/preflight_empirical_calibration_inputs.py", launcher,
        )
        self.assertIn("EMPIRICAL_PREFLIGHT_REPORT", launcher)
        self.assertIn("empirical_input_preflight.json", launcher)

    def test_certification_mode_rejects_same_count_wrong_cohort_before_output(self) -> None:
        output_root = self.root / "wrong_certification_cohort"
        result = pooling.main([
            "--training-day", "2019-01-30", str(self.training_one),
            "--training-day", "2019-03-27", str(self.training_two),
            "--heldout-date", "2020-01-30",
            "--heldout-config", str(self.heldout),
            "--output-root", str(output_root),
            "--minimum-symbols", "2",
            "--require-certification-cohort",
        ])
        self.assertEqual(result, 1)
        self.assertFalse((output_root / "pooling_provenance.json").exists())

    def write_day(self, trading_date: str, *, multiplier: int) -> pathlib.Path:
        compact = trading_date.replace("-", "")
        day_root = self.root / compact
        rows: list[dict[str, object]] = []
        for book_id, symbol in enumerate(SYMBOLS):
            directory = day_root / "empirical_data" / f"itch_{compact}_{symbol.lower()}"
            directory.mkdir(parents=True)
            event_counts: dict[str, int] = {}
            for event_index, event in enumerate(QUANTITY_EVENTS, start=1):
                total = multiplier * (10 + event_index + book_id)
                event_counts[event] = total
                quantities = (
                    (1, 2) if event in {"market_buy", "market_sell"}
                    else (10 + event_index, 20 + event_index)
                )
                write_csv(directory / f"{event}_quantity_distribution.txt",
                          ("quantity", "count"), [
                              {"quantity": quantities[0], "count": total // 2},
                              {"quantity": quantities[1], "count": total - total // 2},
                          ])
            for event_index, event in enumerate(DISTANCE_EVENTS, start=1):
                # The distance and quantity histograms describe the same
                # additions/cancellations and must have identical totals.
                total = event_counts[event]
                write_csv(directory / f"{event}_distance_distribution.txt",
                          ("distance_ticks", "count"), [
                              {"distance_ticks": 0, "count": total - 1},
                              {"distance_ticks": 1, "count": 1},
                          ])
            with (directory / f"itch_manifest_{symbol.lower()}_{compact}.json").open(
                "w", encoding="utf-8"
            ) as output:
                json.dump({
                    "trading_date": trading_date,
                    "session_start": "09:30:00",
                    "session_end": "16:00:00",
                    "valid_snapshots": 23_400,
                    "invalid_snapshots": 0,
                    "distribution_observation_counts": event_counts,
                    "placement_counts": {
                        "improvement_eligible_limit_orders": multiplier * 100,
                        "inside_spread_limit_orders": multiplier * 20,
                    },
                }, output)
            write_csv(
                directory / "input_rates.csv",
                ("event_type", "mu", "alpha", "beta"),
                [{"event_type": "limit_buy", "mu": 1, "alpha": 0, "beta": 10}],
            )
            write_csv(directory / f"market_targets_{symbol.lower()}_{compact}.csv",
                      ("name", "target", "scale", "weight"), [
                          {"name": "mean_spread_ticks", "target": multiplier + book_id + 1,
                           "scale": 1.0, "weight": 1.0},
                          {"name": "mean_bid_depth", "target": multiplier * 100 + book_id,
                           "scale": 2.0, "weight": 1.0},
                          {"name": "mean_ask_depth", "target": multiplier * 120 + book_id,
                           "scale": 2.0, "weight": 1.0},
                          {"name": "return_variance", "target": multiplier * 1.0e-6,
                           "scale": 1.0e-7, "weight": 1.0},
                          {"name": "mid_move_rate", "target": multiplier * 0.1,
                           "scale": 0.01, "weight": 1.0},
                          {"name": "return_kurtosis", "target": 20.0,
                           "scale": 2.0, "weight": 1.0},
                      ])
            rows.append({
                "book_id": book_id,
                "symbol": symbol,
                "data_dir": str(directory),
                "hawkes_rates_file": str(directory / "input_rates.csv"),
                "fundamental_price_ticks": 1_000_100 * (book_id + 1),
                "initial_best_bid_ticks": 1_000_000 * (book_id + 1),
                "initial_best_ask_ticks": 1_000_200 * (book_id + 1),
                "initial_best_bid_depth": 100 + multiplier,
                "initial_best_ask_depth": 120 + multiplier,
                "beta": 1.0 + book_id,
                "basket_weight": 0.0,
                "market_maker_quote_quantity": 100,
                "target_spread_ticks": 2,
                "quote_improvement_probability": 0.2,
            })
        config = day_root / "universe.csv"
        write_csv(config, pooling.CONFIG_FIELDS, rows)
        return config

    def test_pools_direct_inputs_but_retains_day_specific_configs(self) -> None:
        output_root = self.root / "pooled"
        result = pooling.main([
            "--training-day", "2019-01-30", str(self.training_one),
            "--training-day", "2019-03-27", str(self.training_two),
            "--training-target-root", "2019-01-30", str(self.training_one.parent),
            "--training-target-root", "2019-03-27", str(self.training_two.parent),
            "--heldout-date", "2020-01-30",
            "--heldout-config", str(self.heldout),
            "--heldout-target-root", str(self.heldout.parent),
            "--output-root", str(output_root),
            "--minimum-symbols", "2",
        ])
        self.assertEqual(result, 0)

        pooled_rows = read_csv(output_root / "pooled_training_universe.csv")
        self.assertEqual([row["symbol"] for row in pooled_rows], ["QQQ", "AAA"])
        self.assertEqual([row["book_id"] for row in pooled_rows], ["0", "1"])
        # The opening placeholder is an actual latest training-day opening,
        # rather than an invalid componentwise synthetic LOB state.
        self.assertEqual(pooled_rows[0]["initial_best_bid_depth"], "103")
        self.assertNotEqual(pooled_rows[0]["data_dir"], "")

        pooled_dir = pathlib.Path(pooled_rows[0]["data_dir"])
        with (pooled_dir / "itch_manifest_qqq_pooled_20190130_20190327.json").open(
            encoding="utf-8"
        ) as source:
            manifest = json.load(source)
        self.assertEqual(manifest["aggregation_duration_seconds"], 46_800)
        self.assertEqual(manifest["valid_snapshots"], 46_800)
        self.assertEqual(manifest["invalid_snapshots"], 0)
        self.assertEqual(
            manifest["distribution_observation_counts"]["limit_buy"],
            11 + 3 * 11,
        )
        targets = {
            row["name"]: row
            for row in read_csv(
                pooled_dir / "market_targets_qqq_pooled_20190130_20190327.csv"
            )
        }
        self.assertEqual(float(targets["two_sided_sample_fraction"]["target"]), 1.0)
        distribution = read_csv(pooled_dir / "limit_buy_quantity_distribution.txt")
        self.assertEqual(sum(float(row["count"]) for row in distribution), 44.0)
        rates = read_csv(pooled_dir / "hawkes_rates_qqq_pooled_20190130_20190327.csv")
        limit_buy = next(row for row in rates if row["event_type"] == "limit_buy")
        self.assertAlmostEqual(float(limit_buy["observed_rate_per_second"]), 44 / 46_800)
        self.assertNotAlmostEqual(
            float(limit_buy["stationary_target_rate"]),
            float(limit_buy["observed_rate_per_second"]),
        )
        self.assertAlmostEqual(
            float(limit_buy["stationary_reconstructed_rate"]),
            float(limit_buy["stationary_target_rate"]),
        )

        training_first = read_csv(
            output_root / "training_days" / "2019-01-30" / "universe_common.csv"
        )
        training_second = read_csv(
            output_root / "training_days" / "2019-03-27" / "universe_common.csv"
        )
        heldout_rows = read_csv(output_root / "heldout_common.csv")
        self.assertEqual(training_first[0]["data_dir"], str((
            self.root / "20190130" / "empirical_data" / "itch_20190130_qqq"
        ).resolve()))
        self.assertEqual(heldout_rows[0]["initial_best_bid_depth"], "105")
        # The held-out runtime is the checked pooled training template with
        # only its opening state replaced.  No held-out flow/mark parameter is
        # executable during validation.
        self.assertEqual(heldout_rows[0]["data_dir"], pooled_rows[0]["data_dir"])
        self.assertEqual(
            heldout_rows[0]["hawkes_rates_file"],
            pooled_rows[0]["hawkes_rates_file"],
        )
        self.assertEqual(
            heldout_rows[0]["quote_improvement_probability"],
            pooled_rows[0]["quote_improvement_probability"],
        )
        # These are five-day training estimates, not daily or held-out
        # outcome oracles.  For QQQ the two synthetic training targets are
        # bid=(100,300), ask=(120,360), spread=(2,4).
        for rows in (training_first, training_second, heldout_rows, pooled_rows):
            self.assertEqual(rows[0]["target_mean_bid_depth"], "200")
            self.assertEqual(rows[0]["target_mean_ask_depth"], "240")
            self.assertEqual(rows[0]["target_spread_ticks"], "3")
            # The latent value volatility is estimated only from the two
            # training-day one-second return variances: mean(1e-6, 3e-6).
            # It is frozen unchanged in every executable session, including
            # held-out validation, so no held-out outcome can leak into it.
            self.assertAlmostEqual(
                float(rows[0]["fundamental_volatility_bps_sqrt_second"]),
                10_000.0 * math.sqrt(2.0e-6),
            )
            self.assertAlmostEqual(
                float(rows[0]["fundamental_move_probability_per_second"]),
                0.2,
            )
            self.assertAlmostEqual(
                float(rows[0]["fundamental_conditional_kurtosis"]),
                4.0,
            )

        with (output_root / "pooling_provenance.json").open(encoding="utf-8") as source:
            provenance = json.load(source)
        self.assertEqual(provenance["schema_version"], 7)
        self.assertEqual(provenance["training_dates"], ["2019-01-30", "2019-03-27"])
        self.assertEqual(provenance["heldout_date"], "2020-01-30")
        self.assertIn("day_level_behavioural_wmm", provenance["method"])
        self.assertEqual(
            provenance["training_days"][0]["target_root"],
            str(self.training_one.parent.resolve()),
        )
        self.assertEqual(
            provenance["heldout"]["target_root"],
            str(self.heldout.parent.resolve()),
        )
        schema = provenance["configuration_schema"]
        self.assertEqual(schema["runtime_fields"], list(pooling.RUNTIME_CONFIG_FIELDS))
        self.assertFalse(schema["heldout_target_files_used"])
        self.assertFalse(
            provenance["pooling"][
                "heldout_targets_used_for_runtime_configuration"
            ]
        )
        self.assertEqual(
            provenance["pooling"]["hawkes"],
            {
                "activity_scale": 0.3,
                "kernel_beta": 10.0,
                "balance_directional_volume": True,
                "balance_best_depth": True,
                "balance_strength": 1.0,
                **pooling.hawkes.excitation_settings(),
            },
        )
        eligibility = provenance["opening_price_grid_eligibility"]
        self.assertEqual(eligibility["intersection_symbol_count"], 2)
        self.assertEqual(eligibility["eligible_symbol_count"], 2)
        self.assertEqual(eligibility["excluded_symbol_count"], 0)
        approximation = provenance["quote_improvement_runtime_approximation"]
        self.assertEqual(
            approximation, pooling.QUOTE_IMPROVEMENT_COMPATIBILITY
        )
        self.assertFalse(approximation["exact_joint_mark_calibration"])
        qqq_metadata = provenance["symbols"][0]
        expected_rate_settings = provenance["pooling"]["hawkes"]
        rate_audits = [
            qqq_metadata["rate_derivation"],
            *[
                source["rate_derivation"]
                for source in qqq_metadata["sources"]
            ],
        ]
        for audit in rate_audits:
            self.assertEqual(audit["schema_version"], 1)
            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["event_types_checked"], 6)
            self.assertTrue(
                audit["stationary_reconstruction_equals_target_per_type"]
            )
            self.assertTrue(
                audit["observed_rates_equal_manifest_counts_per_duration"]
            )
            self.assertTrue(
                audit[
                    "stationary_targets_equal_declared_transforms_per_type"
                ]
            )
            self.assertTrue(
                audit[
                    "reported_reconstruction_equals_configured_rate_equation_per_type"
                ]
            )
            self.assertLessEqual(
                audit["maximum_absolute_observed_rate_error"], 1.0e-12
            )
            self.assertLessEqual(
                audit["maximum_absolute_stationary_target_error"], 1.0e-12
            )
            self.assertEqual(
                audit["transform_settings"], expected_rate_settings
            )
            for artifact in ("manifest", "generated_hawkes_rates"):
                record = audit[artifact]
                self.assertEqual(
                    record["sha256"],
                    pooling.sha256_file(pathlib.Path(record["path"])),
                )
        self.assertEqual(
            qqq_metadata["pooled_hawkes_rates_sha256"],
            qqq_metadata["rate_derivation"]["generated_hawkes_rates"][
                "sha256"
            ],
        )
        for source in qqq_metadata["sources"]:
            audit = source["rate_derivation"]
            self.assertIn("_balanced_", source["generated_hawkes_rates"])
            self.assertEqual(
                source["manifest_sha256"], audit["manifest"]["sha256"]
            )
            self.assertEqual(
                source["generated_hawkes_rates_sha256"],
                audit["generated_hawkes_rates"]["sha256"],
            )
            self.assertEqual(
                source["source_hawkes_rates_sha256"],
                pooling.sha256_file(pathlib.Path(source["source_hawkes_rates"])),
            )
        pooled_check = qqq_metadata["quote_improvement_compatibility"]
        self.assertEqual(pooled_check["status"], "passed")
        self.assertFalse(pooled_check["probability_clamped"])
        self.assertAlmostEqual(
            pooled_check["quote_improvement_probability"], 80 / 88
        )
        self.assertAlmostEqual(
            pooled_check["descriptive_eligible_improvement_rate"], 0.2
        )
        self.assertEqual(
            pooled_check["side_allocation"],
            "proportional_to_observed_side_zero_counts",
        )
        for source in qqq_metadata["sources"]:
            self.assertEqual(
                source["quote_improvement_compatibility"]["status"], "passed"
            )
            self.assertEqual(
                source["quote_improvement_compatibility"][
                    "configured_input_semantics"
                ],
                "legacy_eligible_rate_v1_migrated",
            )
        heldout_provenance = provenance["heldout"]
        self.assertEqual(
            heldout_provenance["heldout_role"],
            "opening_state_and_validation_targets_only",
        )
        self.assertEqual(
            heldout_provenance["opening_fields_copied_from_heldout"],
            list(pooling.OPENING_FIELDS),
        )
        self.assertTrue(
            heldout_provenance["background_inputs_inherited_from_pooled"]
        )
        heldout_check = heldout_provenance["quote_improvement_compatibility"]
        self.assertEqual(
            heldout_check["status"], "frozen_from_pooled_training"
        )
        self.assertEqual(heldout_check["symbol_count"], 2)
        self.assertFalse(heldout_check["heldout_mark_inputs_instantiated"])

    def test_rejects_inside_count_above_combined_zero_count(self) -> None:
        directory = (
            self.training_one.parent / "empirical_data" / "itch_20190130_qqq"
        )
        for side, total in (("buy", 11), ("sell", 12)):
            write_csv(
                directory / f"limit_{side}_distance_distribution.txt",
                ("distance_ticks", "count"),
                [
                    {"distance_ticks": 0, "count": 1},
                    {"distance_ticks": 1, "count": total - 1},
                ],
            )
        result = pooling.main([
            "--training-day", "2019-01-30", str(self.training_one),
            "--training-day", "2019-03-27", str(self.training_two),
            "--heldout-date", "2020-01-30",
            "--heldout-config", str(self.heldout),
            "--output-root", str(self.root / "incompatible_improvement"),
            "--minimum-symbols", "2",
        ])
        self.assertEqual(result, 1)

    def test_accepts_asymmetric_side_zero_mass_using_combined_split(self) -> None:
        directory = (
            self.training_one.parent / "empirical_data" / "itch_20190130_qqq"
        )
        write_csv(
            directory / "limit_buy_distance_distribution.txt",
            ("distance_ticks", "count"),
            [
                {"distance_ticks": 0, "count": 1},
                {"distance_ticks": 1, "count": 10},
            ],
        )
        # Keep the reduced-book best-depth moment feasible despite the small
        # buy-side zero-distance mass used by this compatibility regression.
        write_csv(
            directory / "limit_buy_quantity_distribution.txt",
            ("quantity", "count"),
            [
                {"quantity": 100, "count": 5},
                {"quantity": 200, "count": 6},
            ],
        )
        manifest_path = directory / "itch_manifest_qqq_20190130.json"
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)
        manifest["placement_counts"]["inside_spread_limit_orders"] = 10
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        rows = read_csv(self.training_one)
        rows[0]["quote_improvement_probability"] = "0.1"
        write_csv(self.training_one, pooling.CONFIG_FIELDS, rows)

        output_root = self.root / "asymmetric_zero_mass"
        result = pooling.main([
            "--training-day", "2019-01-30", str(self.training_one),
            "--training-day", "2019-03-27", str(self.training_two),
            "--heldout-date", "2020-01-30",
            "--heldout-config", str(self.heldout),
            "--output-root", str(output_root),
            "--minimum-symbols", "2",
        ])
        self.assertEqual(result, 0)
        first_day = read_csv(
            output_root / "training_days" / "2019-01-30"
            / "universe_common.csv"
        )
        self.assertAlmostEqual(
            float(first_day[0]["quote_improvement_probability"]), 10 / 12
        )

    def test_ignores_incompatible_raw_heldout_improvement_background(self) -> None:
        directory = (
            self.heldout.parent / "empirical_data" / "itch_20200130_qqq"
        )
        write_csv(
            directory / "limit_sell_distance_distribution.txt",
            ("distance_ticks", "count"),
            [
                {"distance_ticks": 0, "count": 1},
                {"distance_ticks": 1, "count": 9},
            ],
        )
        output_root = self.root / "incompatible_heldout"
        result = pooling.main([
            "--training-day", "2019-01-30", str(self.training_one),
            "--training-day", "2019-03-27", str(self.training_two),
            "--heldout-date", "2020-01-30",
            "--heldout-config", str(self.heldout),
            "--output-root", str(output_root),
            "--minimum-symbols", "2",
        ])
        self.assertEqual(result, 0)
        pooled = read_csv(output_root / "pooled_training_universe.csv")
        heldout = read_csv(output_root / "heldout_common.csv")
        self.assertEqual(
            heldout[0]["quote_improvement_probability"],
            pooled[0]["quote_improvement_probability"],
        )
        self.assertEqual(heldout[0]["data_dir"], pooled[0]["data_dir"])

    def test_excludes_sub_dollar_opening_before_pooling_and_records_reason(self) -> None:
        heldout_rows = read_csv(self.heldout)
        aaa = next(row for row in heldout_rows if row["symbol"] == "AAA")
        aaa["fundamental_price_ticks"] = "9950"
        aaa["initial_best_bid_ticks"] = "9910"
        aaa["initial_best_ask_ticks"] = "9990"
        write_csv(self.heldout, pooling.CONFIG_FIELDS, heldout_rows)

        output_root = self.root / "price_grid_screened"
        result = pooling.main([
            "--training-day", "2019-01-30", str(self.training_one),
            "--training-day", "2019-03-27", str(self.training_two),
            "--heldout-date", "2020-01-30",
            "--heldout-config", str(self.heldout),
            "--output-root", str(output_root),
            "--minimum-symbols", "1",
        ])
        self.assertEqual(result, 0)
        self.assertEqual(
            [row["symbol"] for row in read_csv(
                output_root / "pooled_training_universe.csv"
            )],
            ["QQQ"],
        )
        with (output_root / "pooling_provenance.json").open(
            encoding="utf-8"
        ) as source:
            provenance = json.load(source)
        self.assertEqual(provenance["intersection_symbol_count"], 2)
        self.assertEqual(provenance["common_symbol_count"], 1)
        eligibility = provenance["opening_price_grid_eligibility"]
        self.assertEqual(eligibility["excluded_symbol_count"], 1)
        self.assertEqual(eligibility["excluded_symbols"][0]["symbol"], "AAA")
        reasons = {
            issue["reason"]
            for issue in eligibility["excluded_symbols"][0]["issues"]
        }
        self.assertIn("opening_bid_below_model_price_regime", reasons)
        self.assertIn("opening_bbo_off_simulator_price_grid", reasons)

    def test_heldout_target_changes_cannot_change_runtime_depth_anchors(self) -> None:
        heldout_target = (
            self.heldout.parent / "empirical_data" / "itch_20200130_qqq"
            / "market_targets_qqq_20200130.csv"
        )
        target_rows = read_csv(heldout_target)
        for row in target_rows:
            if row["name"] in {"mean_bid_depth", "mean_ask_depth"}:
                row["target"] = "999999"
        write_csv(
            heldout_target, ("name", "target", "scale", "weight"), target_rows
        )
        output_root = self.root / "heldout_target_mutated"
        result = pooling.main([
            "--training-day", "2019-01-30", str(self.training_one),
            "--training-day", "2019-03-27", str(self.training_two),
            "--heldout-date", "2020-01-30",
            "--heldout-config", str(self.heldout),
            "--heldout-target-root", str(self.heldout.parent),
            "--output-root", str(output_root),
            "--minimum-symbols", "2",
        ])
        self.assertEqual(result, 0)
        for relative in (
            "training_days/2019-01-30/universe_common.csv",
            "training_days/2019-03-27/universe_common.csv",
            "pooled_training_universe.csv",
            "heldout_common.csv",
        ):
            qqq = read_csv(output_root / relative)[0]
            self.assertEqual(qqq["target_mean_bid_depth"], "200")
            self.assertEqual(qqq["target_mean_ask_depth"], "240")

    def test_relocates_absolute_workstation_paths_when_bundle_moves(self) -> None:
        original_root = self.training_one.parent
        relocated_root = self.root / "transferred" / "itch_20190130"
        shutil.copytree(original_root, relocated_root)
        relocated_config = relocated_root / self.training_one.name
        shutil.rmtree(original_root)

        day = pooling.load_config("2019-01-30", str(relocated_config))
        rows = pooling.subset_rows(day, SYMBOLS)
        expected_data = (
            relocated_root / "empirical_data" / "itch_20190130_qqq"
        ).resolve()
        self.assertEqual(pathlib.Path(rows[0]["data_dir"]), expected_data)
        self.assertEqual(
            pathlib.Path(rows[0]["hawkes_rates_file"]),
            expected_data / "input_rates.csv",
        )

    def test_rejects_a_training_day_after_heldout(self) -> None:
        result = pooling.main([
            "--training-day", "2020-01-30", str(self.training_one),
            "--training-day", "2020-02-01", str(self.training_two),
            "--heldout-date", "2020-01-30",
            "--heldout-config", str(self.heldout),
            "--output-root", str(self.root / "invalid"),
            "--minimum-symbols", "2",
        ])
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
