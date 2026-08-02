#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Focused no-simulator tests for cluster-level value-agent calibration."""

from __future__ import annotations

import argparse
import csv
import contextlib
import importlib.util
import io
import json
import math
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cluster_value_calibration",
    ROOT / "scripts" / "calibrate_cluster_value_agents.py",
)
assert SPEC is not None and SPEC.loader is not None
CALIBRATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CALIBRATION
SPEC.loader.exec_module(CALIBRATION)


CONFIG_FIELDS = (
    "book_id",
    "symbol",
    "data_dir",
    "hawkes_rates_file",
    "fundamental_price_ticks",
    "fundamental_volatility_bps_sqrt_second",
    "fundamental_move_probability_per_second",
    "fundamental_conditional_kurtosis",
    "initial_best_bid_ticks",
    "initial_best_ask_ticks",
    "initial_best_bid_depth",
    "initial_best_ask_depth",
    "beta",
    "basket_weight",
    "market_maker_quote_quantity",
    "target_spread_ticks",
    "quote_improvement_probability",
    "target_mean_bid_depth",
    "target_mean_ask_depth",
)


def write_csv(path: pathlib.Path,
              fields: tuple[str, ...] | list[str],
              rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def config_rows() -> list[dict[str, object]]:
    symbols = ("AAA", "BBB", "CCC", "DDD")
    return [
        {
            "book_id": index,
            "symbol": symbol,
            "data_dir": f"/frozen/train/{symbol.lower()}",
            "hawkes_rates_file": f"/frozen/train/{symbol.lower()}/rates.csv",
            "fundamental_price_ticks": 10_000 + 100 * index,
            "fundamental_volatility_bps_sqrt_second": 2.0 + 0.1 * index,
            "fundamental_move_probability_per_second": 0.1 + 0.01 * index,
            "fundamental_conditional_kurtosis": 4.0 + index,
            "initial_best_bid_ticks": 9_990 + 100 * index,
            "initial_best_ask_ticks": 10_010 + 100 * index,
            "initial_best_bid_depth": 100 + index,
            "initial_best_ask_depth": 120 + index,
            "beta": 1.0,
            "basket_weight": 0.0,
            "market_maker_quote_quantity": 50,
            "target_spread_ticks": 2,
            "quote_improvement_probability": 0.1,
            "target_mean_bid_depth": 250.5 + index,
            "target_mean_ask_depth": 275.5 + index,
        }
        for index, symbol in enumerate(symbols)
    ]


class ClusterValueCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.training = self.root / "training.csv"
        self.heldout = self.root / "heldout.csv"
        self.training_rows = config_rows()
        self.heldout_rows = config_rows()
        empirical_root = self.root / "empirical_inputs"
        for training_row, heldout_row in zip(self.training_rows, self.heldout_rows):
            symbol = str(training_row["symbol"])
            data_dir = empirical_root / symbol.lower()
            data_dir.mkdir(parents=True, exist_ok=True)
            rates = data_dir / "rates.csv"
            rates.write_text("event_type,configured_mu\nlimit_buy,1\n", encoding="utf-8")
            for filename in CALIBRATION.SIMULATOR_EMPIRICAL_INPUT_FILENAMES:
                (data_dir / filename).write_text("value,count\n1,1\n", encoding="utf-8")
            (data_dir / f"itch_manifest_{symbol.lower()}_pooled.json").write_text(
                '{"schema_version":1}\n', encoding="utf-8",
            )
            for row in (training_row, heldout_row):
                row["data_dir"] = str(data_dir)
                row["hawkes_rates_file"] = str(rates)
        for row in self.heldout_rows:
            # Only these five values form the held-out opening observation.
            row["fundamental_price_ticks"] = int(row["fundamental_price_ticks"]) + 50
            row["initial_best_bid_ticks"] = int(row["initial_best_bid_ticks"]) + 50
            row["initial_best_ask_ticks"] = int(row["initial_best_ask_ticks"]) + 50
            row["initial_best_bid_depth"] = int(row["initial_best_bid_depth"]) + 3
            row["initial_best_ask_depth"] = int(row["initial_best_ask_depth"]) + 3
        write_csv(self.training, CONFIG_FIELDS, self.training_rows)
        write_csv(self.heldout, CONFIG_FIELDS, self.heldout_rows)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_targets(self, root: pathlib.Path, *, date: str,
                      symbols: tuple[str, ...], window: int | None) -> None:
        compact = CALIBRATION.compact_date(date)
        suffix = "" if window is None else f"_window_{window}s"
        for symbol_index, symbol in enumerate(symbols):
            directory = root / f"itch_{compact}_{symbol.lower()}"
            target_values = {
                metric: (
                    1.0 if metric == "two_sided_sample_fraction"
                    else 10.0 + symbol_index
                )
                for metric in CALIBRATION.METRICS
            }
            target_scales = {
                metric: (
                    0.005 if metric == "two_sided_sample_fraction" else 2.0
                )
                for metric in CALIBRATION.METRICS
            }
            rows = [
                {
                    "name": metric,
                    "target": target_values[metric],
                    "scale": target_scales[metric],
                    "weight": 1.0,
                }
                for metric in CALIBRATION.METRICS
            ]
            write_csv(
                directory / f"market_targets_{symbol.lower()}_{compact}{suffix}.csv",
                ("name", "target", "scale", "weight"), rows,
            )
            manifest_path = directory / f"itch_manifest_{symbol.lower()}_{compact}.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                manifest = {
                    "snapshot_interval_ms": 1000,
                    "trading_date": date,
                    "symbol": symbol,
                    "session_start": "09:30:00",
                    "session_end": "16:00:00",
                    "valid_snapshots": 23_400,
                    "invalid_snapshots": 0,
                    "aggregation_duration_seconds": 23_400,
                    "distribution_observation_counts": {
                        event: int(
                            target_values["background_event_rate"]
                            * 23_400
                            / len(CALIBRATION.BACKGROUND_EVENT_NAMES)
                        )
                        for event in CALIBRATION.BACKGROUND_EVENT_NAMES
                    },
                    "market_values": {
                        metric: target_values[metric]
                        for metric in CALIBRATION.METRICS
                        if metric != "background_event_rate"
                    },
                    "market_target_scales": {
                        metric: target_scales[metric]
                        for metric in CALIBRATION.METRICS
                        if metric != "background_event_rate"
                    },
                    "market_target_windows": {},
                }
            windows = manifest["market_target_windows"]
            if window is not None:
                windows[str(window)] = {
                    "file": f"market_targets_{symbol.lower()}_{compact}{suffix}.csv",
                    "duration_seconds": window,
                    "observations": window,
                    "valid_snapshots": window,
                    "invalid_snapshots": 0,
                    "values": {
                        metric: target_values[metric]
                        for metric in CALIBRATION.METRICS
                        if metric != "background_event_rate"
                    },
                    "scales": {
                        metric: target_scales[metric]
                        for metric in CALIBRATION.METRICS
                        if metric != "background_event_rate"
                    },
                }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def write_summary(self, path: pathlib.Path, *, values: dict[str, float],
                      sample_count: int = 10, expected_sample_count: int = 10,
                      invalid_sample_count: int = 0,
                      structurally_valid: int = 1,
                      two_sided_sample_fraction: float | None = None) -> None:
        fields = (
            "asset_id", "symbol", "sample_count", "expected_sample_count",
            "invalid_sample_count", "structurally_valid",
            *CALIBRATION.BOUNDARY_SUMMARY_FIELDS, *CALIBRATION.METRICS,
        )
        rows = []
        for index, (symbol, value) in enumerate(values.items()):
            rows.append({
                "asset_id": index,
                "symbol": symbol,
                "sample_count": sample_count,
                "expected_sample_count": expected_sample_count,
                "invalid_sample_count": invalid_sample_count,
                "structurally_valid": structurally_valid,
                "background_event_count": 100,
                "background_market_requested_quantity": 50,
                "background_cancel_requested_quantity": 50,
                "removal_boundary_truncation_events": 0,
                "removal_boundary_truncated_quantity": 0,
                "background_boundary_truncation_events": 0,
                "background_boundary_truncated_quantity": 0,
                "value_order_count": 0,
                "value_requested_quantity": 0,
                "value_boundary_truncation_events": 0,
                "value_boundary_truncated_quantity": 0,
                "other_boundary_truncation_events": 0,
                "other_boundary_truncated_quantity": 0,
                **{
                    metric: (
                        two_sided_sample_fraction
                        if metric == "two_sided_sample_fraction"
                        and two_sided_sample_fraction is not None
                        else 1.0 if metric == "two_sided_sample_fraction"
                        else value
                    )
                    for metric in CALIBRATION.METRICS
                },
            })
        write_csv(path, fields, rows)

    def test_target_loading_requires_horizon_matched_artifacts(self) -> None:
        root = self.root / "targets"
        self.write_targets(root, date="2019-01-30", symbols=("AAA", "BBB"), window=300)
        loaded = CALIBRATION.load_targets(
            root, "2019-01-30", ("AAA", "BBB"), window_seconds=300,
        )
        self.assertEqual(set(loaded), {"AAA", "BBB"})
        self.assertEqual(loaded["AAA"]["mean_spread_ticks"].weight, 1.0)
        with self.assertRaisesRegex(FileNotFoundError, "matched-prefix"):
            CALIBRATION.load_targets(
                root, "2019-01-30", ("AAA",), window_seconds=3_600,
            )

    def test_target_csv_values_and_scales_must_match_manifest(self) -> None:
        for window in (None, 300):
            for column in ("target", "scale"):
                with self.subTest(window=window, column=column):
                    root = self.root / f"target_map_{window}_{column}"
                    self.write_targets(
                        root, date="2019-01-30", symbols=("AAA",),
                        window=window,
                    )
                    suffix = "" if window is None else f"_window_{window}s"
                    target_path = (
                        root / "itch_20190130_aaa"
                        / f"market_targets_aaa_20190130{suffix}.csv"
                    )
                    rows = read_csv(target_path)
                    for row in rows:
                        if row["name"] == "mean_spread_ticks":
                            row[column] = str(float(row[column]) + 1.0)
                    write_csv(
                        target_path,
                        ("name", "target", "scale", "weight"),
                        rows,
                    )
                    with self.assertRaisesRegex(
                        CALIBRATION.CalibrationError,
                        "disagrees with extractor manifest",
                    ):
                        CALIBRATION.load_targets(
                            root, "2019-01-30", ("AAA",),
                            window_seconds=window,
                        )

    def test_certified_target_weights_must_remain_equal(self) -> None:
        root = self.root / "target_weight_contract"
        self.write_targets(
            root, date="2019-01-30", symbols=("AAA",), window=None,
        )
        target_path = (
            root / "itch_20190130_aaa"
            / "market_targets_aaa_20190130.csv"
        )
        rows = read_csv(target_path)
        for row in rows:
            if row["name"] == "mean_spread_ticks":
                row["weight"] = "2"
        write_csv(
            target_path,
            ("name", "target", "scale", "weight"),
            rows,
        )
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError,
            "certified extractor target weight must equal 1",
        ):
            CALIBRATION.load_targets(
                root, "2019-01-30", ("AAA",),
            )

    def test_direct_targets_must_match_manifest_accounting(self) -> None:
        replacements = {
            "two_sided_sample_fraction": "0.5",
            "background_event_rate": "999",
        }
        for window in (None, 300):
            for metric, replacement in replacements.items():
                with self.subTest(window=window, metric=metric):
                    root = self.root / f"direct_target_{window}_{metric}"
                    self.write_targets(
                        root, date="2019-01-30", symbols=("AAA",),
                        window=window,
                    )
                    suffix = "" if window is None else f"_window_{window}s"
                    target_path = (
                        root / "itch_20190130_aaa"
                        / f"market_targets_aaa_20190130{suffix}.csv"
                    )
                    rows = read_csv(target_path)
                    for row in rows:
                        if row["name"] == metric:
                            row["target"] = replacement
                    write_csv(
                        target_path,
                        ("name", "target", "scale", "weight"),
                        rows,
                    )
                    with self.assertRaisesRegex(
                        CALIBRATION.CalibrationError, "disagrees with",
                    ):
                        CALIBRATION.load_targets(
                            root, "2019-01-30", ("AAA",),
                            window_seconds=window,
                        )

    def test_target_loading_rejects_nonmatching_snapshot_cadence(self) -> None:
        root = self.root / "targets"
        self.write_targets(root, date="2019-01-30", symbols=("AAA",), window=300)
        manifest = root / "itch_20190130_aaa" / "itch_manifest_aaa_20190130.json"
        manifest.write_text(
            '{"snapshot_interval_ms": 60000, "market_target_windows": {"300": '
            '{"file": "market_targets_aaa_20190130_window_300s.csv", "observations": 5}}}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(CALIBRATION.CalibrationError, "requires 1000 ms"):
            CALIBRATION.load_targets(
                root, "2019-01-30", ("AAA",), window_seconds=300,
            )

    def test_target_snapshot_interval_requires_exact_numeric_integer(self) -> None:
        root = self.root / "target_exact_snapshot_interval"
        self.write_targets(
            root, date="2019-01-30", symbols=("AAA",), window=None,
        )
        manifest_path = (
            root / "itch_20190130_aaa" / "itch_manifest_aaa_20190130.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        manifest["snapshot_interval_ms"] = 1000.0
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertIn(
            "AAA", CALIBRATION.load_targets(root, "2019-01-30", ("AAA",)),
        )

        for invalid in (True, "1000", 1000.9, float("nan")):
            with self.subTest(snapshot_interval_ms=invalid):
                manifest["snapshot_interval_ms"] = invalid
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(
                    CALIBRATION.CalibrationError,
                    "no valid snapshot_interval_ms",
                ):
                    CALIBRATION.load_targets(root, "2019-01-30", ("AAA",))

    def test_empirical_event_counts_require_exact_nonnegative_integers(self) -> None:
        manifest = {
            "distribution_observation_counts": {
                event: 1 for event in CALIBRATION.BACKGROUND_EVENT_NAMES
            },
            "aggregation_duration_seconds": 23_400,
        }
        manifest_path = self.root / "event_count_manifest.json"
        accepted = CALIBRATION.empirical_background_event_rate_target(
            manifest, manifest_path=manifest_path,
        )
        self.assertAlmostEqual(
            accepted.target,
            len(CALIBRATION.BACKGROUND_EVENT_NAMES) / 23_400,
        )

        first_event = CALIBRATION.BACKGROUND_EVENT_NAMES[0]
        for invalid in (True, "1", 1.9, float("nan")):
            with self.subTest(event_count=invalid):
                malformed = {
                    **manifest,
                    "distribution_observation_counts": {
                        **manifest["distribution_observation_counts"],
                        first_event: invalid,
                    },
                }
                with self.assertRaisesRegex(
                    CALIBRATION.CalibrationError,
                    f"no valid {first_event} count",
                ):
                    CALIBRATION.empirical_background_event_rate_target(
                        malformed, manifest_path=manifest_path,
                    )

    def test_full_target_requires_canonical_session_and_exact_clock(self) -> None:
        root = self.root / "full_target_session"
        self.write_targets(
            root, date="2019-01-30", symbols=("AAA",), window=None,
        )
        manifest_path = (
            root / "itch_20190130_aaa" / "itch_manifest_aaa_20190130.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["session_end"] = "15:00:00"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError, "canonical 09:30:00-16:00:00",
        ):
            CALIBRATION.load_targets(root, "2019-01-30", ("AAA",))

        manifest["session_end"] = "16:00:00"
        manifest["valid_snapshots"] = 3_600
        manifest["invalid_snapshots"] = 0
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError, "expected exactly 23400",
        ):
            CALIBRATION.load_targets(root, "2019-01-30", ("AAA",))

    def test_target_rejects_noncanonical_event_rate_duration(self) -> None:
        root = self.root / "target_event_rate_duration"
        self.write_targets(
            root, date="2019-01-30", symbols=("AAA",), window=None,
        )
        manifest_path = (
            root / "itch_20190130_aaa" / "itch_manifest_aaa_20190130.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["aggregation_duration_seconds"] = 3_600
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError,
            "aggregation_duration_seconds=3600; expected 23400",
        ):
            CALIBRATION.load_targets(root, "2019-01-30", ("AAA",))

    def test_prefix_target_requires_exact_snapshot_accounting(self) -> None:
        root = self.root / "prefix_target_accounting"
        self.write_targets(
            root, date="2019-01-30", symbols=("AAA",), window=300,
        )
        manifest_path = (
            root / "itch_20190130_aaa" / "itch_manifest_aaa_20190130.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest["market_target_windows"]["300"]
        metadata["valid_snapshots"] = 298
        metadata["invalid_snapshots"] = 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError, "expected exactly 300",
        ):
            CALIBRATION.load_targets(
                root, "2019-01-30", ("AAA",), window_seconds=300,
            )

    def test_prefix_accounting_accepts_lossless_coverage_representation(self) -> None:
        root = self.root / "prefix_coverage_accounting"
        self.write_targets(
            root, date="2019-01-30", symbols=("AAA",), window=300,
        )
        manifest_path = (
            root / "itch_20190130_aaa" / "itch_manifest_aaa_20190130.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest["market_target_windows"]["300"]
        metadata.pop("valid_snapshots")
        metadata.pop("invalid_snapshots")
        metadata["values"]["two_sided_sample_fraction"] = 0.99
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        target_path = (
            root / "itch_20190130_aaa"
            / "market_targets_aaa_20190130_window_300s.csv"
        )
        target_rows = read_csv(target_path)
        for row in target_rows:
            if row["name"] == "two_sided_sample_fraction":
                row["target"] = "0.99"
        write_csv(
            target_path, ("name", "target", "scale", "weight"), target_rows,
        )
        loaded = CALIBRATION.load_targets(
            root, "2019-01-30", ("AAA",), window_seconds=300,
        )
        self.assertIn("AAA", loaded)

    def test_legacy_target_recovers_two_sided_fraction_from_manifest(self) -> None:
        root = self.root / "legacy_targets"
        self.write_targets(root, date="2019-01-30", symbols=("AAA",), window=300)
        directory = root / "itch_20190130_aaa"
        target_path = directory / "market_targets_aaa_20190130_window_300s.csv"
        rows = [
            row for row in read_csv(target_path)
            if row["name"] != "two_sided_sample_fraction"
        ]
        write_csv(target_path, ("name", "target", "scale", "weight"), rows)
        manifest_path = directory / "itch_manifest_aaa_20190130.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"valid_snapshots": 23_166, "invalid_snapshots": 234})
        metadata = manifest["market_target_windows"]["300"]
        metadata.update({
            "valid_snapshots": 297,
            "invalid_snapshots": 3,
        })
        metadata["values"]["two_sided_sample_fraction"] = 0.99
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        loaded = CALIBRATION.load_targets(
            root, "2019-01-30", ("AAA",), window_seconds=300,
        )
        coverage = loaded["AAA"]["two_sided_sample_fraction"]
        self.assertAlmostEqual(coverage.target, 0.99)
        self.assertGreaterEqual(coverage.empirical_scale, 0.005)

    def test_frozen_background_merge_allows_only_opening_fields_and_subset_resets_ids(self) -> None:
        train_fields, train_rows = CALIBRATION.load_universe_config(self.training)
        heldout_fields, heldout_rows = CALIBRATION.load_universe_config(self.heldout)
        merged = CALIBRATION.merge_frozen_heldout_config(
            train_fields, train_rows, heldout_fields, heldout_rows,
        )
        self.assertEqual(merged[0]["data_dir"], train_rows[0]["data_dir"])
        self.assertEqual(merged[0]["fundamental_price_ticks"], "10050")
        subset = CALIBRATION.subset_config_rows(merged, ("BBB", "DDD"))
        self.assertEqual([row["symbol"] for row in subset], ["BBB", "DDD"])
        self.assertEqual([row["book_id"] for row in subset], ["0", "1"])

        leaked_rows = [dict(row) for row in heldout_rows]
        leaked_rows[2]["hawkes_rates_file"] = "/leaked/heldout/rates.csv"
        leaked = self.root / "leaked.csv"
        write_csv(leaked, CONFIG_FIELDS, leaked_rows)
        leaked_fields, parsed_leaked = CALIBRATION.load_universe_config(leaked)
        with self.assertRaisesRegex(CALIBRATION.CalibrationError, "refits CCC field hawkes_rates_file"):
            CALIBRATION.merge_frozen_heldout_config(
                train_fields, train_rows, leaked_fields, parsed_leaked,
            )

    def test_raw_heldout_opening_source_copies_only_opening_fields(self) -> None:
        train_fields, train_rows = CALIBRATION.load_universe_config(self.training)
        source_rows = config_rows()
        source_rows[0]["data_dir"] = "/raw/heldout/aaa"
        source_rows[0]["hawkes_rates_file"] = "/raw/heldout/aaa/rates.csv"
        source_rows[0]["market_maker_quote_quantity"] = 999
        source_rows[0]["target_spread_ticks"] = 99
        source_rows[0]["quote_improvement_probability"] = 0.9
        source_rows[0]["target_mean_bid_depth"] = 9999
        source_rows[0]["target_mean_ask_depth"] = 8888
        source_rows[0]["fundamental_price_ticks"] = 10_777
        source_rows[0]["initial_best_bid_ticks"] = 10_767
        source_rows[0]["initial_best_ask_ticks"] = 10_787
        source_rows[0]["initial_best_bid_depth"] = 777
        source_rows[0]["initial_best_ask_depth"] = 888
        source = self.root / "raw_heldout_source.csv"
        write_csv(source, CONFIG_FIELDS, source_rows)
        source_fields, parsed_source = CALIBRATION.load_universe_config(source)
        frozen = CALIBRATION.freeze_training_backgrounds_with_heldout_openings(
            train_fields, train_rows, source_fields, parsed_source,
        )
        self.assertEqual(frozen[0]["data_dir"], train_rows[0]["data_dir"])
        self.assertEqual(
            frozen[0]["hawkes_rates_file"], train_rows[0]["hawkes_rates_file"]
        )
        self.assertEqual(frozen[0]["market_maker_quote_quantity"], "50")
        self.assertEqual(frozen[0]["target_spread_ticks"], "2")
        self.assertEqual(frozen[0]["quote_improvement_probability"], "0.1")
        self.assertEqual(frozen[0]["target_mean_bid_depth"], "250.5")
        self.assertEqual(frozen[0]["target_mean_ask_depth"], "275.5")
        self.assertEqual(frozen[0]["fundamental_price_ticks"], "10777")
        self.assertEqual(frozen[0]["initial_best_bid_depth"], "777")

    def test_certification_requires_identical_pooled_homeostatic_targets(self) -> None:
        fields, pooled = CALIBRATION.load_universe_config(self.training)
        day = CALIBRATION.TrainingDay(
            date="2019-01-30",
            universe_config=self.training,
            target_root=self.root,
            fields=fields,
            rows=tuple(dict(row) for row in pooled),
            universe_config_sha256=CALIBRATION.sha256_file(self.training),
        )
        _, heldout = CALIBRATION.load_universe_config(self.heldout)
        CALIBRATION.validate_frozen_homeostatic_targets((day,), pooled, heldout)
        heldout[0]["target_mean_bid_depth"] = "999"
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError, "non-pooled target_mean_bid_depth"
        ):
            CALIBRATION.validate_frozen_homeostatic_targets(
                (day,), pooled, heldout
            )

    def test_runtime_config_rejects_nonpositive_queue_target(self) -> None:
        rows = config_rows()
        rows[0]["target_mean_ask_depth"] = 0
        invalid = self.root / "invalid_queue_target.csv"
        write_csv(invalid, CONFIG_FIELDS, rows)
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError,
            "target_mean_ask_depth must be positive",
        ):
            CALIBRATION.load_universe_config(invalid)

    def test_legacy_pooling_provenance_is_rejected_before_calibration(self) -> None:
        legacy = self.root / "legacy_pooling_provenance.json"
        legacy.write_text(
            json.dumps({
                "schema_version": 1,
                "method": (
                    "multi_day_direct_input_pooling_with_day_level_"
                    "behavioural_wmm"
                ),
            }),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError,
            "regenerate the five-day pool",
        ):
            CALIBRATION.validate_pooling_provenance(
                legacy,
                training_days=(),
                pooled_config_path=self.training,
                heldout_config_path=self.heldout,
                heldout_target_root=self.root,
                producer_project_root=ROOT,
                project_root=ROOT,
            )

    def test_pooling_symbol_coverage_requires_unique_exact_provenance_set(
        self,
    ) -> None:
        fields, pooled_rows = CALIBRATION.load_universe_config(self.training)
        training_day = CALIBRATION.TrainingDay(
            date="2019-01-30",
            universe_config=self.training,
            target_root=self.root,
            fields=fields,
            rows=tuple(dict(row) for row in pooled_rows),
            universe_config_sha256=CALIBRATION.sha256_file(self.training),
        )
        records = [{"symbol": row["symbol"]} for row in pooled_rows]
        observed = CALIBRATION.validate_pooling_symbol_coverage(
            common_symbol_count=len(pooled_rows),
            symbol_records=records,
            pooled_rows=pooled_rows,
            training_days=(training_day,),
        )
        self.assertEqual(observed, tuple(row["symbol"] for row in pooled_rows))

        duplicate_records = [dict(record) for record in records]
        duplicate_records[-1]["symbol"] = duplicate_records[0]["symbol"]
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError,
            "duplicate pooled-symbol provenance record",
        ):
            CALIBRATION.validate_pooling_symbol_coverage(
                common_symbol_count=len(pooled_rows),
                symbol_records=duplicate_records,
                pooled_rows=pooled_rows,
                training_days=(training_day,),
            )

        wrong_records = [dict(record) for record in records]
        wrong_records[-1]["symbol"] = "ZZZ"
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError,
            "symbol records differ from the pooled runtime configuration",
        ):
            CALIBRATION.validate_pooling_symbol_coverage(
                common_symbol_count=len(pooled_rows),
                symbol_records=wrong_records,
                pooled_rows=pooled_rows,
                training_days=(training_day,),
            )

    def test_pooling_symbol_coverage_requires_each_training_config_exactly(
        self,
    ) -> None:
        fields, pooled_rows = CALIBRATION.load_universe_config(self.training)
        mismatched_rows = [dict(row) for row in pooled_rows]
        mismatched_rows[-1]["symbol"] = "ZZZ"
        training_day = CALIBRATION.TrainingDay(
            date="2019-01-30",
            universe_config=self.training,
            target_root=self.root,
            fields=fields,
            rows=tuple(mismatched_rows),
            universe_config_sha256=CALIBRATION.sha256_file(self.training),
        )
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError,
            "training config 2019-01-30 symbols differ",
        ):
            CALIBRATION.validate_pooling_symbol_coverage(
                common_symbol_count=len(pooled_rows),
                symbol_records=[
                    {"symbol": row["symbol"]} for row in pooled_rows
                ],
                pooled_rows=pooled_rows,
                training_days=(training_day,),
            )

    def test_reused_pool_is_bound_to_its_distinct_producer_tree(self) -> None:
        producer = self.root / "r3_pool_producer"
        consumer = self.root / "r4_calibration_consumer"
        for root in (producer, consumer):
            for relative in CALIBRATION.WORKFLOW_SEMANTICS_FILES:
                source = ROOT / relative
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())

        # Simulate a calibration-only R4 revision after the R3 pool was made.
        consumer_submit = consumer / "submit_cluster_value_agent_calibration.sh"
        consumer_submit.write_text(
            consumer_submit.read_text(encoding="utf-8")
            + "\n# simulated R4 calibration-only revision\n",
            encoding="utf-8",
        )
        producer_hash = CALIBRATION.workflow_source_semantics_sha256(producer)
        consumer_hash = CALIBRATION.workflow_source_semantics_sha256(consumer)
        self.assertNotEqual(producer_hash, consumer_hash)
        payload = {"workflow_source_semantics_sha256": producer_hash}

        verification = CALIBRATION.validate_pooling_producer_workflow_source(
            payload,
            producer_project_root=producer,
            consumer_project_root=consumer,
        )
        self.assertEqual(verification["status"], "producer_source_verified")
        self.assertEqual(
            verification[
                "observed_producer_workflow_source_semantics_sha256"
            ],
            producer_hash,
        )
        self.assertEqual(
            verification["consumer_workflow_source_semantics_sha256"],
            consumer_hash,
        )
        self.assertFalse(
            verification[
                "producer_and_consumer_workflow_semantics_identical"
            ]
        )

        # Declaring the modified consumer as the producer cannot satisfy the
        # old pool hash.
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError,
            "does not match the declared producer source tree",
        ):
            CALIBRATION.validate_pooling_producer_workflow_source(
                payload,
                producer_project_root=consumer,
                consumer_project_root=consumer,
            )

        # Nor can the declared producer be edited after the pool is produced.
        producer_submit = producer / "submit_cluster_value_agent_calibration.sh"
        producer_submit.write_text(
            producer_submit.read_text(encoding="utf-8")
            + "\n# tampered producer\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError,
            "does not match the declared producer source tree",
        ):
            CALIBRATION.validate_pooling_producer_workflow_source(
                payload,
                producer_project_root=producer,
                consumer_project_root=consumer,
            )

    def test_weighted_wmm_aggregates_symbols_and_seeds(self) -> None:
        first = self.root / "first.csv"
        second = self.root / "second.csv"
        self.write_summary(first, values={"AAA": 12.0, "BBB": 14.0})
        self.write_summary(second, values={"AAA": 16.0, "BBB": 18.0})
        targets = {
            symbol: {
                metric: CALIBRATION.TargetMoment(
                    target=1.0 if metric == "two_sided_sample_fraction" else 10.0,
                    empirical_scale=(
                        0.005 if metric == "two_sided_sample_fraction" else 2.0
                    ),
                    weight=3.0 if metric == "mean_spread_ticks" else 1.0,
                )
                for metric in CALIBRATION.METRICS
            }
            for symbol in ("AAA", "BBB")
        }
        score, estimates = CALIBRATION.weighted_moment_loss(
            (first, second), targets, ("AAA", "BBB"),
        )
        # Eight non-coverage moments include the direct background-event-rate
        # acceptance target. Two matched coverage moments contribute no loss.
        self.assertAlmostEqual(score, math.sqrt(130.0 / 22.0))
        aaa_spread = next(
            estimate for estimate in estimates
            if estimate.symbol == "AAA" and estimate.metric == "mean_spread_ticks"
        )
        self.assertAlmostEqual(aaa_spread.simulated_mean, 14.0)
        self.assertAlmostEqual(aaa_spread.simulated_mean_se, 2.0)

        local_score, local_estimates = CALIBRATION.weighted_moment_loss(
            (first, second), targets, ("AAA", "BBB"),
            metrics=CALIBRATION.LOCAL_FLOW_METRICS,
        )
        # Local-flow fitting checks the direct event-rate target while still
        # excluding mid-price incidence as a confounded activity proxy.
        self.assertAlmostEqual(local_score, math.sqrt(78.0 / 14.0))
        self.assertEqual(len(local_estimates), 2 * len(CALIBRATION.LOCAL_FLOW_METRICS))
        self.assertEqual(
            {estimate.metric for estimate in local_estimates},
            set(CALIBRATION.LOCAL_FLOW_METRICS),
        )
        with self.assertRaisesRegex(CALIBRATION.CalibrationError, "subset"):
            CALIBRATION.weighted_moment_loss(
                (first,), targets, ("AAA",), metrics=("not_a_moment",),
            )

    def test_metric_balanced_robust_loss_limits_count_and_outlier_dominance(self) -> None:
        """One populous or badly scaled metric must not own candidate selection."""

        def estimate(symbol: str, metric: str, residual: float):
            return CALIBRATION.MomentEstimate(
                symbol=symbol,
                metric=metric,
                target=0.0,
                empirical_scale=1.0,
                weight=1.0,
                simulated_mean=residual,
                simulated_sample_sd=0.0,
                simulated_mean_se=0.0,
                combined_scale=1.0,
                empirical_standardized_residual=residual,
                combined_uncertainty_residual=residual,
                objective_residual=residual,
                weighted_squared_residual=residual * residual,
                seed_count=1,
            )

        metrics = ("mean_spread_ticks", "mean_bid_depth")
        balanced = [
            estimate("AAA", "mean_spread_ticks", 1.0),
            estimate("BBB", "mean_bid_depth", 1.0),
        ]
        duplicated = [
            *(
                estimate(f"SPREAD_{index}", "mean_spread_ticks", 1.0)
                for index in range(100)
            ),
            estimate("BBB", "mean_bid_depth", 1.0),
        ]
        balanced_score, balanced_details = CALIBRATION.metric_balanced_robust_loss(
            balanced, metrics=metrics, huber_delta=2.0,
        )
        duplicated_score, duplicated_details = CALIBRATION.metric_balanced_robust_loss(
            duplicated, metrics=metrics, huber_delta=2.0,
        )
        self.assertAlmostEqual(duplicated_score, balanced_score)

        def detail_metrics(details: object) -> set[str]:
            if isinstance(details, dict):
                return {str(metric) for metric in details}
            return {str(row["metric"]) for row in details}  # type: ignore[index]

        self.assertEqual(detail_metrics(balanced_details), set(metrics))
        self.assertEqual(detail_metrics(duplicated_details), set(metrics))

        outlier_residual = 1_000_000.0
        outlier_score, _ = CALIBRATION.metric_balanced_robust_loss(
            [
                estimate("AAA", "mean_spread_ticks", outlier_residual),
                estimate("BBB", "mean_bid_depth", 1.0),
            ],
            metrics=metrics,
            huber_delta=2.0,
        )
        unbounded_quadratic_score = math.sqrt(
            (outlier_residual ** 2 + 1.0) / 2.0
        )
        self.assertTrue(math.isfinite(outlier_score))
        self.assertLess(outlier_score, unbounded_quadratic_score / 100.0)

    def test_structural_preflight_requires_depth_fit_under_fixed_thresholds(
        self,
    ) -> None:
        def depth_estimates(bid_ratio: float, ask_ratio: float):
            return [
                {
                    "symbol": "AAA",
                    "metric": "mean_bid_depth",
                    "target": 100.0,
                    "simulated_mean": 100.0 * bid_ratio,
                },
                {
                    "symbol": "AAA",
                    "metric": "mean_ask_depth",
                    "target": 120.0,
                    "simulated_mean": 120.0 * ask_ratio,
                },
                # A deliberately poor spread is irrelevant to this structural
                # check because the local market maker is the spread-repair
                # component calibrated after preflight.
                {
                    "symbol": "AAA",
                    "metric": "mean_spread_ticks",
                    "target": 2.0,
                    "simulated_mean": 200.0,
                },
            ]

        passing = CALIBRATION.structural_depth_fit_summary({
            "moment_estimates": depth_estimates(1.5, 1.0 / 1.5),
        })
        self.assertTrue(passing["passed"])
        self.assertEqual(
            passing["metrics"],
            list(CALIBRATION.STRUCTURAL_PREFLIGHT_DEPTH_METRICS),
        )
        self.assertTrue(
            passing["spread_excluded_because_local_mm_is_spread_repair"]
        )
        eligible_evaluation = {
            "fit_wsmrmse": 0.1,
            "selection_score": 0.1,
            "two_sided_integrity_passed": True,
            "finite_boundary_adequacy_passed": True,
            "errors": [],
            "structural_depth_fit": passing,
        }
        self.assertTrue(CALIBRATION.candidate_is_eligible({
            "candidate_index": 0,
            "evaluation": eligible_evaluation,
        }))

        failing = CALIBRATION.structural_depth_fit_summary({
            "moment_estimates": depth_estimates(1.5 ** 7, 1.0),
        })
        self.assertFalse(failing["passed"])
        self.assertGreater(failing["gross_symbol_metric_failure_count"], 0)
        eligible_evaluation["structural_depth_fit"] = failing
        self.assertFalse(CALIBRATION.candidate_is_eligible({
            "candidate_index": 0,
            "evaluation": eligible_evaluation,
        }))

        missing = CALIBRATION.structural_depth_fit_summary({
            "moment_estimates": [],
        })
        self.assertFalse(missing["passed"])
        self.assertIsNotNone(missing["error"])

        # In the real five-day workflow the aggregate deliberately keeps
        # moment estimates inside dated child evaluations.  Preflight must
        # consume that shape instead of treating the empty top-level list as
        # absence of depth evidence.
        aggregate = {
            "moment_estimates": [],
            "training_day_evaluations": [
                {
                    "date": "2019-01-30",
                    "evaluation": {
                        "moment_estimates": depth_estimates(1.0, 1.0),
                    },
                },
                {
                    "date": "2019-03-27",
                    "evaluation": {
                        "moment_estimates": depth_estimates(1.5, 1.0 / 1.5),
                    },
                },
            ],
        }
        self.assertTrue(
            CALIBRATION.structural_depth_fit_summary(aggregate)["passed"]
        )
        aggregate["training_day_evaluations"][1]["evaluation"][
            "moment_estimates"
        ] = depth_estimates(1.5 ** 7, 1.0)
        aggregate_failure = CALIBRATION.structural_depth_fit_summary(aggregate)
        self.assertFalse(aggregate_failure["passed"])
        self.assertEqual(
            aggregate_failure["gross_symbol_metric_failures"][0][
                "training_date"
            ],
            "2019-03-27",
        )
        systemic_failure = CALIBRATION.structural_depth_fit_summary(
            aggregate,
            require_zero_gross_symbol_failures=False,
        )
        self.assertFalse(systemic_failure["passed"])
        self.assertFalse(systemic_failure["aggregate_fit_passed"])

        # Structural preflight diagnoses isolated stock/date outliers but must
        # not demand that one uncalibrated reference policy already satisfy
        # the later frozen-development zero-outlier rule.  Aggregate and
        # per-metric thresholds remain mandatory.
        preflight_like = {
            "moment_estimates": [
                row
                for _ in range(20)
                for row in depth_estimates(1.0, 1.0)
            ] + depth_estimates(1.5 ** 7, 1.0),
        }
        diagnostic_only = CALIBRATION.structural_depth_fit_summary(
            preflight_like,
            require_zero_gross_symbol_failures=False,
        )
        self.assertTrue(diagnostic_only["passed"])
        self.assertTrue(diagnostic_only["aggregate_fit_passed"])
        self.assertGreater(
            diagnostic_only["gross_symbol_metric_failure_count"], 0,
        )
        self.assertEqual(
            diagnostic_only["gross_symbol_metric_failures_role"],
            "diagnostic_only_during_structural_preflight",
        )

    def test_one_sided_coverage_is_reported_but_fails_execution_screen(self) -> None:
        summary = self.root / "partly_one_sided.csv"
        self.write_summary(
            summary,
            values={"AAA": 10.0},
            sample_count=85,
            expected_sample_count=100,
            invalid_sample_count=15,
            structurally_valid=0,
            two_sided_sample_fraction=0.85,
        )
        targets = {
            "AAA": {
                metric: CALIBRATION.TargetMoment(
                    target=1.0 if metric == "two_sided_sample_fraction" else 10.0,
                    empirical_scale=(
                        0.01 if metric == "two_sided_sample_fraction" else 2.0
                    ),
                    weight=1.0,
                )
                for metric in CALIBRATION.METRICS
            }
        }
        score, estimates = CALIBRATION.weighted_moment_loss(
            (summary,), targets, ("AAA",),
        )
        self.assertTrue(math.isfinite(score))
        coverage = next(
            estimate for estimate in estimates
            if estimate.metric == "two_sided_sample_fraction"
        )
        self.assertAlmostEqual(coverage.simulated_mean, 0.85)
        integrity, integrity_failures = CALIBRATION.two_sided_execution_integrity(
            (summary,), ("AAA",),
        )
        self.assertFalse(integrity)
        self.assertEqual(integrity_failures[0]["invalid_sample_count"], 15)
        evaluation = {"moment_estimates": [
            CALIBRATION.asdict(estimate) for estimate in estimates
        ]}
        failures = CALIBRATION.two_sided_coverage_shortfalls(evaluation, 0.01)
        self.assertEqual([row["symbol"] for row in failures], ["AAA"])
        diagnostic = CALIBRATION.two_sided_coverage_summary(evaluation, 0.01)
        self.assertEqual(diagnostic["failing_symbol_count"], 1)
        self.assertEqual(diagnostic["within_tolerance_fraction"], 0.0)

    def test_summary_rejects_inconsistent_coverage_accounting(self) -> None:
        summary = self.root / "inconsistent_coverage.csv"
        self.write_summary(
            summary,
            values={"AAA": 10.0},
            sample_count=85,
            expected_sample_count=100,
            invalid_sample_count=15,
            structurally_valid=1,
            two_sided_sample_fraction=0.90,
        )
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError, "inconsistent two-sided coverage",
        ):
            CALIBRATION.summary_rows(summary, ("AAA",))

    def test_summary_rejects_inconsistent_structural_flag(self) -> None:
        summary = self.root / "inconsistent_structural_flag.csv"
        self.write_summary(
            summary,
            values={"AAA": 10.0},
            sample_count=85,
            expected_sample_count=100,
            invalid_sample_count=15,
            structurally_valid=1,
            two_sided_sample_fraction=0.85,
        )
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError,
            "inconsistent structurally_valid flag",
        ):
            CALIBRATION.summary_rows(summary, ("AAA",))

    def test_finite_boundary_adequacy_enforces_asset_run_and_zero_rules(self) -> None:
        summary = self.root / "boundary.csv"
        self.write_summary(summary, values={"AAA": 10.0, "BBB": 10.0})
        rows = read_csv(summary)
        # Each asset is below 5%, and the aggregate is exactly the 1% limit.
        for row in rows:
            row["removal_boundary_truncation_events"] = "1"
            row["removal_boundary_truncated_quantity"] = "1"
            row["background_boundary_truncation_events"] = "1"
            row["background_boundary_truncated_quantity"] = "1"
        write_csv(summary, list(rows[0]), rows)
        passed = CALIBRATION.finite_boundary_adequacy(
            (summary,), ("AAA", "BBB"), required_expected_sample_count=10,
        )
        self.assertTrue(passed["passed"])
        self.assertAlmostEqual(
            passed["aggregate_pooled"]["boundary_event_ratio"], 0.01
        )

        # One asset can satisfy the 5% asset limit while the run violates the
        # stricter aggregate 1% limit.
        rows[0]["removal_boundary_truncation_events"] = "5"
        rows[0]["background_boundary_truncation_events"] = "5"
        rows[1]["removal_boundary_truncation_events"] = "0"
        rows[1]["background_boundary_truncation_events"] = "0"
        write_csv(summary, list(rows[0]), rows)
        failed_run = CALIBRATION.finite_boundary_adequacy(
            (summary,), ("AAA", "BBB"), required_expected_sample_count=10,
        )
        self.assertFalse(failed_run["passed"])
        self.assertTrue(any(
            item["scope"] == "aggregate_seed_pool"
            and item["metric"] == "boundary_event_ratio"
            for item in failed_run["failures"]
        ))

        # A zero removal denominator is admissible only with zero truncated
        # quantity; a positive numerator is an accounting/model failure.
        rows[0]["removal_boundary_truncation_events"] = "0"
        rows[0]["background_boundary_truncation_events"] = "0"
        rows[0]["background_market_requested_quantity"] = "0"
        rows[0]["background_cancel_requested_quantity"] = "0"
        rows[0]["removal_boundary_truncated_quantity"] = "1"
        rows[0]["background_boundary_truncated_quantity"] = "1"
        rows[1]["removal_boundary_truncation_events"] = "0"
        rows[1]["removal_boundary_truncated_quantity"] = "0"
        rows[1]["background_boundary_truncation_events"] = "0"
        rows[1]["background_boundary_truncated_quantity"] = "0"
        write_csv(summary, list(rows[0]), rows)
        zero_denominator = CALIBRATION.finite_boundary_adequacy(
            (summary,), ("AAA", "BBB"), required_expected_sample_count=10,
        )
        self.assertFalse(zero_denominator["passed"])
        self.assertTrue(any(
            item["symbol"] == "AAA" and item["ratio"] is None
            for item in zero_denominator["failures"]
            if item["scope"] == "symbol_seed_pool"
        ))
        self.assertFalse(CALIBRATION.candidate_is_eligible({
            "candidate_index": 0,
            "evaluation": {
                "fit_wsmrmse": 0.1,
                "selection_score": 0.1,
                "two_sided_integrity_passed": True,
                "finite_boundary_adequacy_passed": False,
                "errors": [],
            },
        }))

    def test_cluster_training_boundary_pools_small_subset_but_keeps_asset_gate(
        self,
    ) -> None:
        def day_record(boundary_events: int, value_orders: int,
                       *, failures: list[dict] | None = None) -> dict:
            return {
                "source": "value",
                "failures": failures or [],
                "diagnostic_per_seed_failures": [],
                "aggregate_pooled": {
                    "boundary_truncation_events": boundary_events,
                    "source_event_count": value_orders,
                    "boundary_truncated_quantity": boundary_events,
                    "source_requested_quantity": 100000,
                    "run_count": 2,
                },
            }

        # A three-symbol block is not a market-wide aggregate.  Value-agent
        # contacts are pooled over declared seeds and dates and use value
        # orders—not background events—as their denominator.
        evaluation = {
            "value_boundary_adequacy": [
                {
                    "date": "2019-03-27",
                    "adequacy": day_record(0, 3532),
                },
                {
                    "date": "2019-07-30",
                    "adequacy": day_record(70, 3483, failures=[{
                            "scope": "aggregate_seed_pool",
                            "metric": "boundary_event_ratio",
                            "numerator": 70,
                            "denominator": 3483,
                            "ratio": 70 / 3483,
                            "maximum": 0.01,
                    }]),
                },
            ],
        }
        pooled = CALIBRATION.cluster_training_boundary_adequacy(evaluation)
        self.assertTrue(pooled["passed"])
        self.assertAlmostEqual(
            pooled["pooled_aggregate"]["boundary_event_ratio"],
            70 / 7015,
        )
        self.assertEqual(len(
            pooled["diagnostic_day_aggregate_failures"]
        ), 1)
        self.assertEqual(pooled["symbol_day_seed_pool_failures"], [])

        # A true symbol/day seed-pool instability remains disqualifying.
        evaluation["value_boundary_adequacy"][1]["adequacy"]["failures"].append({
            "scope": "symbol_seed_pool",
            "symbol": "EIDX",
            "metric": "boundary_event_ratio",
            "numerator": 100,
            "denominator": 1000,
            "ratio": 0.1,
            "maximum": 0.05,
        })
        asset_failure = CALIBRATION.cluster_training_boundary_adequacy(
            evaluation
        )
        self.assertFalse(asset_failure["passed"])
        self.assertEqual(
            len(asset_failure["symbol_day_seed_pool_failures"]), 1
        )

    def test_boundary_gates_use_matching_agent_source_denominators(self) -> None:
        summary = self.root / "source_separated_boundary.csv"
        self.write_summary(summary, values={"AAA": 10.0, "BBB": 10.0})
        rows = read_csv(summary)
        rows[0].update({
            "removal_boundary_truncation_events": "1",
            "removal_boundary_truncated_quantity": "1",
            "value_order_count": "10",
            "value_requested_quantity": "10",
            "value_boundary_truncation_events": "1",
            "value_boundary_truncated_quantity": "1",
        })
        write_csv(summary, list(rows[0]), rows)

        background = CALIBRATION.finite_boundary_adequacy(
            (summary,), ("AAA", "BBB"), required_expected_sample_count=10,
        )
        value = CALIBRATION.value_boundary_adequacy(
            (summary,), ("AAA", "BBB"), required_expected_sample_count=10,
        )
        self.assertTrue(background["passed"])
        self.assertEqual(
            background["aggregate_pooled"]["boundary_truncation_events"], 0,
        )
        self.assertFalse(value["passed"])
        self.assertEqual(
            value["symbol_pooled"][0]["boundary_event_ratio"], 0.1,
        )

    def test_summary_rejects_truncated_declared_horizon(self) -> None:
        summary = self.root / "truncated.csv"
        self.write_summary(
            summary, values={"AAA": 10.0},
            sample_count=99, expected_sample_count=99,
        )
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError, "wrong fixed-clock horizon",
        ):
            CALIBRATION.summary_rows(
                summary, ("AAA",), required_expected_sample_count=100,
            )

    def test_local_flow_stage1_and_stage2_promote_every_eligible_candidate(
        self,
    ) -> None:
        def candidate(index: int, score: float, *, eligible: bool = True):
            return {
                "candidate_index": index,
                "candidate": CALIBRATION.LocalFlowCandidate(
                    hawkes_activity_scale=0.3,
                    local_mm_enabled=True,
                    local_mm_interval_ms=1000.0,
                    local_mm_quantity_multiplier=1.0,
                    local_mm_improvement_probability=0.25,
                    label=f"candidate_{index}",
                ),
                "evaluation": {
                    "fit_wsmrmse": score,
                    "selection_score": score,
                    "two_sided_integrity_passed": eligible,
                    "finite_boundary_adequacy_passed": eligible,
                    "errors": [],
                },
            }

        evaluated = [
            candidate(0, 3.0),
            candidate(1, 1.0),
            candidate(2, 2.0),
            candidate(3, 0.1, eligible=False),
        ]
        for stage_name in ("stage1_screen", "stage2_refinement"):
            promoted = CALIBRATION.select_local_flow_stage_survivors(
                stage_name, evaluated,
            )
            self.assertEqual(
                [item["candidate_index"] for item in promoted],
                [1, 2, 0],
            )

    def test_local_flow_full_day_selects_best_eligible_candidate(self) -> None:
        def candidate(index: int, score: float, *, boundary_passed: bool = True):
            return {
                "candidate_index": index,
                "evaluation": {
                    "fit_wsmrmse": score,
                    "selection_score": score,
                    "two_sided_integrity_passed": True,
                    "finite_boundary_adequacy_passed": boundary_passed,
                    "errors": [],
                },
            }

        selected = CALIBRATION.select_local_flow_stage_survivors(
            "stage3_full",
            [
                candidate(0, 2.0),
                candidate(1, 1.0),
                candidate(2, 0.1, boundary_passed=False),
            ],
        )
        self.assertEqual([item["candidate_index"] for item in selected], [1])

    def test_local_flow_promotion_rejects_unknown_stage(self) -> None:
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError, "unknown local-flow calibration stage",
        ):
            CALIBRATION.select_local_flow_stage_survivors("stage4", [])

    def test_ranked_policy_keeps_complete_grid_until_full_day(self) -> None:
        def item(
            index: int, score: float, enabled: bool, threshold: float = 0.0,
            depth: float = 0.0,
        ):
            return {
                "candidate_index": index,
                "candidate": CALIBRATION.Candidate(
                    enabled=enabled,
                    threshold_bps=threshold if enabled else 0.0,
                    depth_participation=depth,
                    label=f"candidate_{index}",
                ),
                "evaluation": {
                    "fit_wsmrmse": score,
                    "selection_score": score,
                },
            }

        eligible = [
            item(1, 1.0, True, 5.0, 0.05),
            item(2, 2.0, True, 8.0, 0.05),
            item(3, 1.5, True, 5.0, 0.5),
            item(4, 2.5, True, 8.0, 0.5),
            item(0, 9.0, False),
        ]
        stage1 = CALIBRATION.ranked_policy_stage_survivors(
            "stage1_screen", eligible, 2, (0.05, 0.5), (5.0, 8.0),
        )
        self.assertEqual(
            {value["candidate_index"] for value in stage1}, {0, 1, 2, 3, 4},
        )
        stage2 = CALIBRATION.ranked_policy_stage_survivors(
            "stage2_refinement", eligible, 2, (0.05, 0.5), (5.0, 8.0),
        )
        self.assertEqual(
            {value["candidate_index"] for value in stage2}, {0, 1, 2, 3, 4},
        )
        stage3 = CALIBRATION.ranked_policy_stage_survivors(
            "stage3_full", eligible, 1, (0.05, 0.5), (5.0, 8.0),
        )
        self.assertEqual([value["candidate_index"] for value in stage3], [1])

    def test_ranked_policy_fails_closed_on_incomplete_grid_or_baseline(self) -> None:
        def item(
            index: int, threshold: float, depth: float,
            enabled: bool = True,
        ):
            return {
                "candidate_index": index,
                "candidate": CALIBRATION.Candidate(
                    enabled=enabled,
                    threshold_bps=threshold if enabled else 0.0,
                    depth_participation=depth,
                    label=f"candidate_{index}",
                ),
                "evaluation": {"fit_wsmrmse": 1.0, "selection_score": 1.0},
            }

        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError, "eligible value-policy grid is incomplete",
        ):
            CALIBRATION.ranked_policy_stage_survivors(
                "stage2_refinement",
                [item(1, 5.0, 0.05), item(0, 0.0, 0.0, False)],
                2,
                (0.05, 0.5),
                (5.0, 8.0),
            )
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError, "disabled value-policy baseline",
        ):
            CALIBRATION.ranked_policy_stage_survivors(
                "stage2_refinement",
                [
                    item(1, 5.0, 0.05), item(2, 8.0, 0.05),
                    item(3, 5.0, 0.5), item(4, 8.0, 0.5),
                ],
                2,
                (0.05, 0.5),
                (5.0, 8.0),
            )
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError, "eligible value-policy grid is incomplete",
        ):
            CALIBRATION.ranked_policy_stage_survivors(
                "stage3_full",
                [
                    item(0, 0.0, 0.0, False),
                    item(1, 5.0, 0.05), item(2, 8.0, 0.05),
                    item(3, 5.0, 0.5),
                ],
                1,
                (0.05, 0.5),
                (5.0, 8.0),
            )

    def test_ranked_policy_rejects_unknown_stage(self) -> None:
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError, "unknown ranked policy stage",
        ):
            CALIBRATION.ranked_policy_stage_survivors("stage4", [], 2)

    def test_shared_quote_survivor_trajectory_respects_grid_and_caps(self) -> None:
        self.assertEqual(
            CALIBRATION.ranked_survivor_trajectory(4, (6, 2, 1)),
            (4, 2, 1),
        )

    def test_v18_profile_records_complete_search_and_validation_contract(self) -> None:
        profile = CALIBRATION.certification_profile()
        self.assertEqual(profile["profile_id"], "development_validation_gate_v18")
        self.assertTrue(profile["certification_profile_enforced"])
        self.assertEqual(profile["required_common_symbol_count"], 1480)
        self.assertEqual(
            profile["required_common_symbol_order_sha256"],
            "2f57f37762772d9523fb9916fe2376a9578e337d20971fe39aa44d578f5691d3",
        )
        cohort_identity = profile["cohort_identity"]
        self.assertEqual(
            cohort_identity["selection_role"],
            "development_validation_balanced_panel",
        )
        self.assertTrue(cohort_identity["heldout_availability_conditioned"])
        self.assertFalse(cohort_identity["heldout_target_values_used"])
        self.assertFalse(cohort_identity["independent_final_holdout"])
        self.assertEqual(
            cohort_identity["original_intersection_symbol_count"], 1509,
        )
        self.assertEqual(
            cohort_identity["fixed_price_grid_excluded_symbol_count"], 29,
        )
        heldout_protocol = profile["heldout_validation_acceptance_protocol"]
        self.assertEqual(
            heldout_protocol["authoritative_empirical_fit_scope"],
            "full_universe_marketwide",
        )
        self.assertEqual(
            heldout_protocol["stratified"]["required_symbol_count"], 30,
        )
        self.assertEqual(
            heldout_protocol["stratified"]["empirical_fit_acceptance_role"],
            "required_reported_diagnostic_only",
        )
        self.assertTrue(
            heldout_protocol["stratified"]["empirical_coverage_required"]
        )
        self.assertEqual(
            heldout_protocol["marketwide"]["required_symbol_count"], 1480,
        )
        self.assertEqual(
            heldout_protocol["marketwide"]["empirical_fit_acceptance_role"],
            "authoritative_certification_gate",
        )
        self.assertEqual(
            heldout_protocol["marketwide"]["maximum_robust_score"], 2.0,
        )
        self.assertEqual(
            heldout_protocol["marketwide"]["maximum_metric_score"], 3.0,
        )
        self.assertFalse(heldout_protocol["thresholds_changed_from_v17"])
        self.assertFalse(heldout_protocol["seeds_changed_from_v17"])
        self.assertEqual(
            profile["full_universe_training_adequacy"]
            ["seed_set_inherited_from_profile_id"],
            "development_validation_gate_v17",
        )
        self.assertEqual(
            CALIBRATION.certification_profile_sha256(),
            "38a52e86eaa5cfed6a039c68b0cda471b60a1b0255e62d26a6ddcbde700bb475",
        )
        self.assertNotEqual(
            CALIBRATION.certification_profile_sha256(),
            "055fbc9a4f23266442f343654fbb83dc6bb3f90492ee13db7ffc21a157e25bad",
        )
        self.assertEqual(profile["shared_quote_candidate_count"], 4)
        self.assertEqual(profile["shared_quote_stage1_survivor_cap"], 6)
        self.assertEqual(profile["shared_quote_stage1_promoted_candidates"], 4)
        self.assertEqual(profile["shared_quote_stage2_survivor_cap"], 2)
        self.assertEqual(profile["shared_quote_stage2_promoted_candidates"], 2)
        self.assertEqual(profile["shared_quote_stage3_survivor_cap"], 1)
        self.assertEqual(profile["shared_quote_stage3_promoted_candidates"], 1)
        self.assertEqual(profile["empirical_target_session"], {
            "session_start": "09:30:00",
            "session_end": "16:00:00",
            "duration_seconds": 23_400,
            "snapshot_interval_ms": 1_000,
            "full_session_observations": 23_400,
        })
        self.assertEqual(
            profile["local_flow_stage1_promotion"],
            "all_structurally_eligible",
        )
        self.assertEqual(
            profile["local_flow_stage2_promotion"],
            "all_structurally_eligible",
        )
        self.assertEqual(
            profile["local_flow_stage3_selection"],
            "best_training_fit_among_structurally_eligible",
        )
        self.assertEqual(
            profile["value_policy_stage1_promotion"],
            "all_structurally_eligible_threshold_depth_policies_plus_"
            "disabled_baseline",
        )
        self.assertEqual(
            profile["value_policy_stage2_promotion"],
            "all_structurally_eligible_threshold_depth_policies_plus_"
            "disabled_baseline",
        )
        self.assertEqual(
            profile["value_policy_stage1_survivors_per_depth"], 6,
        )
        self.assertEqual(
            profile["value_policy_stage2_survivors_per_depth"], 6,
        )
        self.assertEqual(
            profile["value_policy_stage3_candidates_per_cluster"], 25,
        )
        self.assertEqual(
            profile["value_depth_participations"], [0.05, 0.1, 0.25, 0.5],
        )
        self.assertEqual(
            profile["value_thresholds_bps"],
            [5.0, 8.0, 10.0, 15.0, 25.0, 40.0],
        )
        self.assertTrue(
            profile["nested_policy_selection"]
            ["all_threshold_depth_policies_promoted_through_stage2"]
        )
        self.assertTrue(
            profile["nested_policy_selection"]
            ["complete_grid_eligibility_required_at_each_stage"]
        )
        self.assertEqual(
            profile["full_universe_training_adequacy"]["seeds"],
            [3424815697, 1799108475, 2301941028, 3637917665, 3007455382],
        )
        self.assertTrue(
            profile["full_universe_training_adequacy"]
            ["required_before_development_validation"]
        )
        structural = profile["structural_preflight"]
        self.assertEqual(
            structural["empirical_admissibility_metrics"],
            ["mean_bid_depth", "mean_ask_depth"],
        )
        self.assertEqual(structural["maximum_robust_score"], 2.0)
        self.assertEqual(structural["maximum_metric_score"], 3.0)
        self.assertEqual(
            structural["maximum_symbol_metric_absolute_robust_residual"],
            6.0,
        )
        self.assertFalse(
            structural["zero_gross_symbol_metric_failures_required"]
        )
        self.assertFalse(
            structural[
                "strict_gross_symbol_gate_retained_for_development_validation"
            ]
        )
        self.assertEqual(
            profile["model_semantics"]["value_agent"],
            (
                "contrarian_market_order_protected_at_perceived_fundamental_"
                "and_sized_as_a_cluster_calibrated_fraction_of_displayed_"
                "opposite_side_depth_"
                "against_rank_independent_sparse_"
                "training_moment_latent_value"
            ),
        )
        self.assertEqual(
            profile["model_semantics"]["local_market_maker"],
            "owned_queue_and_spread_reactive_one_tick_limit_quotes",
        )
        self.assertTrue(
            profile["nested_policy_selection"][
                "disabled_baseline_promoted_through_stage2"
            ]
        )
        self.assertFalse(
            profile["gross_symbol_metric_failures_required_for_acceptance"]
        )
        self.assertTrue(structural["both_candidates_must_pass"])
        self.assertTrue(
            structural["spread_excluded_because_local_mm_is_spread_repair"]
        )
        boundary = profile["finite_boundary_adequacy"]
        self.assertTrue(boundary["source_attribution_required"])
        self.assertEqual(boundary["maximum_asset_event_ratio"], 0.05)
        self.assertEqual(boundary["maximum_asset_quantity_ratio"], 0.05)
        self.assertEqual(boundary["maximum_run_event_ratio"], 0.01)
        self.assertEqual(boundary["maximum_run_quantity_ratio"], 0.01)
        cluster_boundary = profile[
            "cluster_training_finite_boundary_adequacy"
        ]
        self.assertEqual(
            cluster_boundary["maximum_symbol_date_event_ratio"], 0.05,
        )
        self.assertEqual(
            cluster_boundary["maximum_cluster_candidate_event_ratio"], 0.05,
        )
        self.assertEqual(
            cluster_boundary["per_seed_ratios_role"],
            "diagnostic_only",
        )
        self.assertTrue(
            cluster_boundary[
                "development_validation_requires_background_and_value_gates"
            ]
        )

    def test_runtime_profile_records_actual_acceptance_thresholds(self) -> None:
        canonical = CALIBRATION.certification_profile()
        observed = CALIBRATION.runtime_profile_with_acceptance_thresholds(
            canonical,
            maximum_robust_score=9.0,
            maximum_metric_score=11.0,
            maximum_two_sided_shortfall=0.2,
        )
        self.assertEqual(observed["maximum_robust_score"], 9.0)
        self.assertEqual(observed["maximum_metric_score"], 11.0)
        self.assertEqual(
            observed["maximum_two_sided_shortfall_diagnostic"], 0.2,
        )
        training = observed["full_universe_training_adequacy"]
        self.assertEqual(training["maximum_aggregate_robust_score"], 9.0)
        self.assertEqual(training["maximum_day_robust_score"], 9.0)
        self.assertEqual(training["maximum_day_metric_score"], 11.0)
        self.assertEqual(canonical["maximum_robust_score"], 2.0)
        self.assertEqual(
            canonical["full_universe_training_adequacy"]
            ["maximum_aggregate_robust_score"],
            2.0,
        )

    def test_noncanonical_diagnostic_thresholds_are_allowed_but_cannot_certify(
        self,
    ) -> None:
        base = [
            "--binary", "/bin/true",
            "--training-universe-config", str(self.training),
            "--heldout-opening-source-config", str(self.heldout),
            "--cluster-assignments", str(self.root / "assignments.csv"),
            "--validation-sample", str(self.root / "validation.csv"),
            "--training-date", "2019-01-30",
            "--heldout-date", "2020-01-30",
            "--training-target-root", str(self.root / "training_targets"),
            "--heldout-target-root", str(self.root / "heldout_targets"),
            "--output-dir", str(self.root / "output"),
            "--maximum-heldout-robust-score", "9",
            "--maximum-heldout-metric-score", "11",
            "--maximum-two-sided-coverage-shortfall", "0.2",
        ]
        diagnostic_parser = CALIBRATION.build_parser()
        diagnostic_args = diagnostic_parser.parse_args(base)
        CALIBRATION.validate_arguments(diagnostic_args, diagnostic_parser)

        provenance = [
            "--build-provenance", str(self.root / "build.json"),
            "--cluster-manifest", str(self.root / "cluster.json"),
            "--pooling-provenance", str(self.root / "pool.json"),
            "--pooling-producer-project-root", str(self.root / "producer"),
            "--require-certification-profile",
        ]
        for option, expected_message in (
            (
                ["--maximum-heldout-robust-score", "9"],
                "--maximum-heldout-robust-score is immutable",
            ),
            (
                ["--maximum-heldout-metric-score", "11"],
                "--maximum-heldout-metric-score is immutable",
            ),
            (
                ["--maximum-two-sided-coverage-shortfall", "0.2"],
                "--maximum-two-sided-coverage-shortfall is immutable",
            ),
        ):
            canonical_thresholds = [
                "--maximum-heldout-robust-score", "2",
                "--maximum-heldout-metric-score", "3",
                "--maximum-two-sided-coverage-shortfall", "0.01",
            ]
            changed_name = option[0]
            index = canonical_thresholds.index(changed_name)
            canonical_thresholds[index + 1] = option[1]
            parser = CALIBRATION.build_parser()
            args = parser.parse_args(
                base[:base.index("--maximum-heldout-robust-score")]
                + canonical_thresholds + provenance
            )
            error_output = io.StringIO()
            with contextlib.redirect_stderr(error_output):
                with self.assertRaises(SystemExit):
                    CALIBRATION.validate_arguments(args, parser)
            self.assertIn(expected_message, error_output.getvalue())

    def test_r4_candidate_and_stage_checkpoints_are_atomic_and_counted(self) -> None:
        progress_path = self.root / "calibration" / "calibration_progress.json"
        CALIBRATION.initialize_calibration_progress(
            progress_path, overwrite=False,
        )
        candidate = CALIBRATION.LocalFlowCandidate(
            hawkes_activity_scale=0.3,
            local_mm_interval_ms=1000.0,
            local_mm_quantity_multiplier=1.0,
            local_mm_improvement_probability=0.75,
            label="checkpoint_test",
        )
        evaluation = {
            "fit_wsmrmse": 1.25,
            "combined_uncertainty_wsmrmse": 1.5,
            "selection_score": 1.25,
            "selection_metric_scores": [],
            "two_sided_integrity_passed": True,
            "two_sided_integrity_failures": [],
            "finite_boundary_adequacy_passed": True,
            "finite_boundary_adequacy": {"passed": True},
            "seed_count": 1,
            "seed_wall_seconds": [0.1],
            "summary_paths": ["/tmp/summary.csv"],
            "errors": [],
            "moment_estimates": [],
        }
        candidate_dir = self.root / "calibration" / "candidate_001"
        reference = CALIBRATION.persist_candidate_evaluation(
            candidate_dir,
            block="global_local_flow",
            stage="stage1_screen",
            cluster_id=None,
            candidate_index=1,
            candidate=candidate,
            evaluation=evaluation,
            progress_path=progress_path,
            overwrite=False,
        )
        candidate_payload = json.loads(
            (candidate_dir / "candidate_evaluation.json").read_text(
                encoding="utf-8",
            )
        )
        self.assertTrue(reference["eligible"])
        self.assertTrue(candidate_payload["eligibility"]["eligible"])
        self.assertEqual(candidate_payload["candidate_index"], 1)

        checkpoint = CALIBRATION.persist_stage_checkpoint(
            self.root / "calibration" / "stage1_screen",
            block="global_local_flow",
            stage="stage1_screen",
            cluster_id=None,
            candidate_references=[reference],
            promoted_candidate_indices=[1],
            configured_ranked_survivor_count=6,
            progress_path=progress_path,
            overwrite=False,
        )
        self.assertEqual(checkpoint["observed_counts"], {
            "evaluated_candidates": 1,
            "eligible_candidates": 1,
            "promoted_candidates": 1,
            "configured_ranked_survivor_count": 6,
        })
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        self.assertEqual(progress["status"], "running")
        self.assertEqual(progress["event_count"], 2)

    def test_overwrite_revokes_stale_handoff_before_early_run_failure(self) -> None:
        output_root = self.root / "overwrite_attempt"
        output_root.mkdir()
        for filename in CALIBRATION.TERMINAL_CALIBRATION_ARTIFACT_FILENAMES:
            (output_root / filename).write_text(
                '{"stale":true}\n', encoding="utf-8",
            )
        retained_diagnostic = (
            output_root / "full_universe_training_adequacy" / "keep.json"
        )
        retained_diagnostic.parent.mkdir()
        retained_diagnostic.write_text('{"keep":true}\n', encoding="utf-8")

        args = argparse.Namespace(
            output_dir=str(output_root),
            overwrite=True,
            binary=str(self.root / "missing_simulator"),
        )
        with self.assertRaisesRegex(CALIBRATION.CalibrationError, "--binary"):
            CALIBRATION.run(args)

        for filename in CALIBRATION.TERMINAL_CALIBRATION_ARTIFACT_FILENAMES:
            self.assertFalse((output_root / filename).exists())
        self.assertTrue(retained_diagnostic.is_file())
        self.assertTrue((output_root / "calibration_progress.json").is_file())

    def test_overwrite_removes_handoff_first_and_never_recurses(self) -> None:
        output_root = self.root / "malformed_terminal_artifact"
        output_root.mkdir()
        handoff = output_root / "calibration_handoff.json"
        handoff.write_text('{"stale":true}\n', encoding="utf-8")
        malformed = output_root / "preliminary_calibration_result.json"
        malformed.mkdir()

        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError, "not a regular file",
        ):
            CALIBRATION.invalidate_terminal_calibration_artifacts(
                output_root, overwrite=True,
            )
        self.assertFalse(handoff.exists())
        self.assertTrue(malformed.is_dir())

    def test_r4_failure_manifest_preserves_progress_snapshot(self) -> None:
        output_root = self.root / "failed_calibration"
        progress_path = output_root / "calibration_progress.json"
        CALIBRATION.initialize_calibration_progress(
            progress_path, overwrite=False,
        )
        CALIBRATION.append_calibration_progress(
            progress_path,
            {"kind": "stage_checkpoint", "stage": "stage2_refinement"},
        )
        failure_path = CALIBRATION.persist_calibration_failure(
            output_root, RuntimeError("no structurally eligible candidates"),
        )
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["exception_type"], "RuntimeError")
        self.assertIn("no structurally eligible", failure["message"])
        self.assertEqual(
            failure["progress_checkpoint"]["snapshot"]["status"], "failed",
        )
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        self.assertEqual(progress["status"], "failed")
        self.assertEqual(progress["last_event"]["kind"], "calibration_failure")

    def test_gross_symbol_residual_is_retained_as_cluster_diagnostic(self) -> None:
        evaluation = {
            "selection_score": 0.2,
            "selection_metric_scores": [
                {"metric": "mean_spread_ticks", "score": 0.2},
            ],
            "moment_estimates": [
                {
                    "symbol": "GOOD", "metric": "mean_spread_ticks",
                    "target": 1.0, "simulated_mean": 1.0,
                },
                {
                    "symbol": "BAD", "metric": "mean_spread_ticks",
                    "target": 1.0, "simulated_mean": 100.0,
                },
            ],
        }
        result = CALIBRATION.empirical_fit_summary(
            evaluation,
            maximum_score=CALIBRATION.CERTIFICATION_MAXIMUM_ROBUST_SCORE,
            maximum_metric_score=CALIBRATION.CERTIFICATION_MAXIMUM_METRIC_SCORE,
            maximum_symbol_metric_absolute_residual=(
                CALIBRATION.CERTIFICATION_GROSS_RESIDUAL_LIMIT
            ),
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["gross_symbol_metric_failure_count"], 1)
        self.assertEqual(result["gross_symbol_metric_failures"][0]["symbol"], "BAD")
        self.assertFalse(
            result["gross_symbol_metric_failures_required_for_acceptance"]
        )

    def test_wrong_background_event_rate_cannot_certify(self) -> None:
        evaluation = {
            "selection_score": 0.1,
            "selection_metric_scores": [
                {"metric": "background_event_rate", "score": 4.0},
            ],
            "moment_estimates": [
                {
                    "symbol": "AAA",
                    "metric": "background_event_rate",
                    "target": 100.0,
                    "simulated_mean": 10_000.0,
                },
            ],
        }
        result = CALIBRATION.empirical_fit_summary(
            evaluation,
            maximum_score=CALIBRATION.CERTIFICATION_MAXIMUM_ROBUST_SCORE,
            maximum_metric_score=CALIBRATION.CERTIFICATION_MAXIMUM_METRIC_SCORE,
            maximum_symbol_metric_absolute_residual=(
                CALIBRATION.CERTIFICATION_GROSS_RESIDUAL_LIMIT
            ),
        )
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["gross_symbol_metric_failures"][0]["metric"],
            "background_event_rate",
        )

    def test_empirical_fit_reason_names_metric_when_aggregate_passes(self) -> None:
        summary = {
            "passed": False,
            "selection_score": 1.8,
            "maximum_allowed_score": 2.0,
            "maximum_allowed_metric_score": 3.0,
            "failing_metrics": [
                {"metric": "return_kurtosis", "score": 3.286209},
            ],
        }
        reasons = CALIBRATION.empirical_fit_failure_reasons(
            CALIBRATION.STRATIFIED_EMPIRICAL_FIT_FAILURE_SCOPE, summary,
        )
        self.assertEqual(len(reasons), 1)
        self.assertTrue(
            reasons[0].startswith(
                CALIBRATION.STRATIFIED_EMPIRICAL_FIT_FAILURE_SCOPE
            )
        )
        self.assertIn("return_kurtosis", reasons[0])
        self.assertIn("3.28621 exceeds 3", reasons[0])
        self.assertNotIn("aggregate", reasons[0])

    def test_v18_marketwide_fit_is_authoritative_when_stratified_fit_fails(
        self,
    ) -> None:
        decision = CALIBRATION.heldout_acceptance_decision(
            marketwide_validation_completed=True,
            sampled_execution_integrity_passed=True,
            sampled_coverage_passed=True,
            sampled_background_boundary_adequacy_passed=True,
            sampled_value_boundary_adequacy_passed=True,
            sampled_empirical_fit_passed=False,
            marketwide_execution_integrity_passed=True,
            marketwide_background_boundary_adequacy_passed=True,
            marketwide_value_boundary_adequacy_passed=True,
            marketwide_empirical_fit_passed=True,
        )
        self.assertTrue(decision["stratified_structural_adequacy_passed"])
        self.assertFalse(decision["stratified_empirical_fit_passed"])
        self.assertTrue(decision["marketwide_empirical_fit_passed"])
        self.assertTrue(decision["empirical_fit_passed"])
        self.assertTrue(decision["heldout_validation_passed"])

    def test_v18_stratified_structural_failure_still_blocks_certification(
        self,
    ) -> None:
        decision = CALIBRATION.heldout_acceptance_decision(
            marketwide_validation_completed=True,
            sampled_execution_integrity_passed=True,
            sampled_coverage_passed=True,
            sampled_background_boundary_adequacy_passed=True,
            sampled_value_boundary_adequacy_passed=False,
            sampled_empirical_fit_passed=True,
            marketwide_execution_integrity_passed=True,
            marketwide_background_boundary_adequacy_passed=True,
            marketwide_value_boundary_adequacy_passed=True,
            marketwide_empirical_fit_passed=True,
        )
        self.assertFalse(decision["stratified_structural_adequacy_passed"])
        self.assertTrue(decision["empirical_fit_passed"])
        self.assertFalse(decision["heldout_validation_passed"])

    def test_v18_marketwide_fit_failure_cannot_be_replaced_by_sample_fit(
        self,
    ) -> None:
        decision = CALIBRATION.heldout_acceptance_decision(
            marketwide_validation_completed=True,
            sampled_execution_integrity_passed=True,
            sampled_coverage_passed=True,
            sampled_background_boundary_adequacy_passed=True,
            sampled_value_boundary_adequacy_passed=True,
            sampled_empirical_fit_passed=True,
            marketwide_execution_integrity_passed=True,
            marketwide_background_boundary_adequacy_passed=True,
            marketwide_value_boundary_adequacy_passed=True,
            marketwide_empirical_fit_passed=False,
        )
        self.assertTrue(decision["stratified_structural_adequacy_passed"])
        self.assertFalse(decision["empirical_fit_passed"])
        self.assertFalse(decision["heldout_validation_passed"])

    def full_universe_training_evaluation(
        self,
        *,
        aggregate_score: float = 1.0,
        day_metric_scores: tuple[tuple[str, float], ...] = (
            ("2019-01-30", 1.0),
            ("2019-03-27", 1.5),
        ),
        two_sided_integrity_passed: bool = True,
        finite_boundary_adequacy_passed: bool = True,
        value_boundary_adequacy_passed: bool = True,
    ) -> dict[str, object]:
        return {
            "selection_score": aggregate_score,
            "two_sided_integrity_passed": two_sided_integrity_passed,
            "finite_boundary_adequacy_passed": finite_boundary_adequacy_passed,
            "value_boundary_adequacy_passed": value_boundary_adequacy_passed,
            "errors": [],
            "training_day_evaluations": [
                {
                    "date": training_date,
                    "evaluation": {
                        "selection_score": 1.0,
                        "selection_metric_scores": [
                            {
                                "metric": "return_kurtosis",
                                "score": metric_score,
                            },
                        ],
                        "moment_estimates": [],
                    },
                }
                for training_date, metric_score in day_metric_scores
            ],
        }

    def summarize_full_universe_training(
        self, evaluation: dict[str, object],
    ) -> dict[str, object]:
        return CALIBRATION.full_universe_training_adequacy_summary(
            evaluation,
            maximum_score=CALIBRATION.CERTIFICATION_MAXIMUM_ROBUST_SCORE,
            maximum_metric_score=(
                CALIBRATION.CERTIFICATION_MAXIMUM_METRIC_SCORE
            ),
            maximum_symbol_metric_absolute_residual=(
                CALIBRATION.CERTIFICATION_GROSS_RESIDUAL_LIMIT
            ),
        )

    def test_full_universe_training_adequacy_passes_all_gates(self) -> None:
        summary = self.summarize_full_universe_training(
            self.full_universe_training_evaluation()
        )

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["training_day_count"], 2)
        self.assertTrue(summary["aggregate_selection_score_passed"])
        self.assertTrue(summary["every_training_day_empirical_fit_passed"])
        self.assertTrue(summary["execution_integrity_passed"])
        self.assertTrue(summary["finite_boundary_adequacy_passed"])
        self.assertTrue(summary["value_boundary_adequacy_passed"])
        self.assertEqual(summary["failure_reasons"], [])

    def test_full_universe_training_rejects_per_day_metric_failure(self) -> None:
        summary = self.summarize_full_universe_training(
            self.full_universe_training_evaluation(
                aggregate_score=1.0,
                day_metric_scores=(
                    ("2019-01-30", 1.0),
                    ("2019-03-27", 3.25),
                ),
            )
        )

        self.assertFalse(summary["passed"])
        self.assertTrue(summary["aggregate_selection_score_passed"])
        self.assertFalse(summary["every_training_day_empirical_fit_passed"])
        reasons = summary["failure_reasons"]
        self.assertEqual(len(reasons), 1)
        self.assertIn("2019-03-27", reasons[0])
        self.assertIn("return_kurtosis", reasons[0])
        self.assertIn("3.25 exceeds 3", reasons[0])

    def test_full_universe_training_rejects_aggregate_failure(self) -> None:
        summary = self.summarize_full_universe_training(
            self.full_universe_training_evaluation(aggregate_score=2.01)
        )

        self.assertFalse(summary["passed"])
        self.assertFalse(summary["aggregate_selection_score_passed"])
        self.assertTrue(summary["every_training_day_empirical_fit_passed"])
        self.assertEqual(len(summary["failure_reasons"]), 1)
        self.assertIn("aggregate robust-fit score 2.01 exceeds 2", (
            summary["failure_reasons"][0]
        ))

    def test_full_universe_training_rejects_execution_and_boundaries(self) -> None:
        summary = self.summarize_full_universe_training(
            self.full_universe_training_evaluation(
                two_sided_integrity_passed=False,
                finite_boundary_adequacy_passed=False,
                value_boundary_adequacy_passed=False,
            )
        )

        self.assertFalse(summary["passed"])
        self.assertFalse(summary["execution_integrity_passed"])
        self.assertFalse(summary["finite_boundary_adequacy_passed"])
        self.assertFalse(summary["value_boundary_adequacy_passed"])
        reasons = "\n".join(summary["failure_reasons"])
        self.assertIn("incomplete or one-sided", reasons)
        self.assertIn("background flow depends materially", reasons)
        self.assertIn("value-agent orders depend materially", reasons)

    def test_full_universe_training_records_heldout_leakage_barrier(self) -> None:
        summary = self.summarize_full_universe_training(
            self.full_universe_training_evaluation()
        )

        self.assertTrue(summary["selection_parameters_frozen_before_evaluation"])
        self.assertIs(summary["development_validation_targets_opened"], False)
        for day in summary["day_summaries"]:
            self.assertIs(
                day["empirical_fit"]["heldout_used_for_parameter_selection"],
                False,
            )

    def test_run_model_timeout_records_diagnostics_and_stops_process_group(self) -> None:
        sleeper = self.root / "sleeping_simulator.py"
        sleeper.write_text(
            "#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n",
            encoding="utf-8",
        )
        sleeper.chmod(0o755)
        output_dir = self.root / "timed_out_run"
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            CALIBRATION.run_model(
                launcher=(),
                binary=sleeper,
                config=self.training,
                policy=None,
                output_dir=output_dir,
                duration=1,
                seed=1729,
                local_controls=CALIBRATION.LocalFlowCandidate(
                    hawkes_activity_scale=0.3,
                    local_mm_interval_ms=1000.0,
                    local_mm_quantity_multiplier=1.0,
                    label="timeout_test",
                ),
                shared_quote_multiplier=None,
                enable_shared_mm=False,
                enable_value_agents=False,
                timeout_seconds=0.05,
            )
        log = (output_dir / "run.log").read_text(encoding="utf-8")
        self.assertIn("TIMEOUT", log)
        self.assertIn("sleeping_simulator.py", log)
        self.assertRegex(log, r"return_code=-?\d+")

    def test_run_model_nonzero_exit_records_return_code(self) -> None:
        failing = self.root / "failing_simulator.py"
        failing.write_text(
            "#!/usr/bin/env python3\nraise SystemExit(7)\n",
            encoding="utf-8",
        )
        failing.chmod(0o755)
        output_dir = self.root / "failed_run"
        with self.assertRaisesRegex(RuntimeError, "status 7"):
            CALIBRATION.run_model(
                launcher=(),
                binary=failing,
                config=self.training,
                policy=None,
                output_dir=output_dir,
                duration=1,
                seed=1729,
                local_controls=CALIBRATION.LocalFlowCandidate(
                    hawkes_activity_scale=0.3,
                    local_mm_interval_ms=1000.0,
                    local_mm_quantity_multiplier=1.0,
                    label="return_code_test",
                ),
                shared_quote_multiplier=None,
                enable_shared_mm=False,
                enable_value_agents=False,
                timeout_seconds=5.0,
            )
        self.assertIn(
            "return_code=7",
            (output_dir / "run.log").read_text(encoding="utf-8"),
        )

    def test_run_model_zero_exit_cannot_reuse_stale_summary(self) -> None:
        no_output = self.root / "zero_exit_without_summary.py"
        no_output.write_text(
            "#!/usr/bin/env python3\nraise SystemExit(0)\n",
            encoding="utf-8",
        )
        no_output.chmod(0o755)
        output_dir = self.root / "stale_summary_run"
        output_dir.mkdir()
        summary = output_dir / "fragmented_asset_summary.csv"
        summary.write_text("stale summary must not be reused\n", encoding="utf-8")
        run_log = output_dir / "run.log"
        run_log.write_text("stale run log\n", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "without a fresh"):
            CALIBRATION.run_model(
                launcher=(),
                binary=no_output,
                config=self.training,
                policy=None,
                output_dir=output_dir,
                duration=1,
                seed=1729,
                local_controls=CALIBRATION.LocalFlowCandidate(
                    hawkes_activity_scale=0.3,
                    local_mm_interval_ms=1000.0,
                    local_mm_quantity_multiplier=1.0,
                    label="zero_exit_test",
                ),
                shared_quote_multiplier=None,
                enable_shared_mm=False,
                enable_value_agents=False,
                timeout_seconds=5.0,
            )

        self.assertFalse(summary.exists())
        fresh_log = run_log.read_text(encoding="utf-8")
        self.assertIn("return_code=0", fresh_log)
        self.assertNotIn("stale run log", fresh_log)

    def test_run_model_rejects_symlink_summary(self) -> None:
        target = self.root / "external_stale_summary.csv"
        target.write_text("external stale summary\n", encoding="utf-8")
        symlinker = self.root / "symlink_summary.py"
        symlinker.write_text(
            "#!/usr/bin/env python3\n"
            "import os, sys\n"
            "output = sys.argv[sys.argv.index('--asset-summary-csv') + 1]\n"
            f"os.symlink({str(target)!r}, output)\n",
            encoding="utf-8",
        )
        symlinker.chmod(0o755)

        with self.assertRaisesRegex(RuntimeError, "regular asset summary"):
            CALIBRATION.run_model(
                launcher=(),
                binary=symlinker,
                config=self.training,
                policy=None,
                output_dir=self.root / "symlink_summary_run",
                duration=1,
                seed=1729,
                local_controls=CALIBRATION.LocalFlowCandidate(
                    hawkes_activity_scale=0.3,
                    local_mm_interval_ms=1000.0,
                    local_mm_quantity_multiplier=1.0,
                    label="symlink_summary_test",
                ),
                shared_quote_multiplier=None,
                enable_shared_mm=False,
                enable_value_agents=False,
                timeout_seconds=5.0,
            )

    def test_candidate_and_stage_diagnostics_are_atomic_and_count_observed_survivors(
        self,
    ) -> None:
        output_root = self.root / "diagnostic_run"
        progress_path = output_root / "calibration_progress.json"
        CALIBRATION.initialize_calibration_progress(
            progress_path, overwrite=False,
        )
        candidate_dir = output_root / "local" / "stage2" / "candidate_003"
        candidate = CALIBRATION.LocalFlowCandidate(
            hawkes_activity_scale=0.3,
            local_mm_interval_ms=1000.0,
            local_mm_quantity_multiplier=1.0,
            label="candidate_003",
        )
        evaluation = {
            "fit_wsmrmse": 1.0,
            "combined_uncertainty_wsmrmse": 1.1,
            "selection_score": 0.9,
            "selection_metric_scores": [],
            "two_sided_integrity_passed": True,
            "two_sided_integrity_failures": [],
            "finite_boundary_adequacy_passed": True,
            "finite_boundary_adequacy": {"passed": True, "failures": []},
            "seed_count": 1,
            "seed_wall_seconds": [0.5],
            "summary_paths": ["/run/summary.csv"],
            "errors": [],
            "moment_estimates": [],
        }
        reference = CALIBRATION.persist_candidate_evaluation(
            candidate_dir,
            block="global_local_flow",
            stage="stage2_refinement",
            cluster_id=None,
            candidate_index=3,
            candidate=candidate,
            evaluation=evaluation,
            progress_path=progress_path,
            overwrite=False,
        )
        self.assertTrue(reference["eligible"])
        candidate_path = candidate_dir / "candidate_evaluation.json"
        candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        self.assertEqual(
            candidate_payload["artifact_role"],
            "calibration_candidate_evaluation",
        )
        self.assertTrue(candidate_payload["eligibility"]["eligible"])
        original = candidate_path.read_bytes()
        with self.assertRaisesRegex(CALIBRATION.CalibrationError, "overwrite"):
            CALIBRATION.persist_candidate_evaluation(
                candidate_dir,
                block="global_local_flow",
                stage="stage2_refinement",
                cluster_id=None,
                candidate_index=3,
                candidate=candidate,
                evaluation=evaluation,
                progress_path=progress_path,
                overwrite=False,
            )
        self.assertEqual(candidate_path.read_bytes(), original)
        self.assertFalse(any(candidate_dir.glob(".candidate_evaluation.json.*.tmp")))

        checkpoint = CALIBRATION.persist_stage_checkpoint(
            output_root / "local" / "stage2",
            block="global_local_flow",
            stage="stage2_refinement",
            cluster_id=None,
            candidate_references=(reference,),
            promoted_candidate_indices=(3,),
            configured_ranked_survivor_count=2,
            progress_path=progress_path,
            overwrite=False,
        )
        self.assertEqual(checkpoint["status"], "complete")
        self.assertEqual(checkpoint["observed_counts"], {
            "evaluated_candidates": 1,
            "eligible_candidates": 1,
            "promoted_candidates": 1,
            "configured_ranked_survivor_count": 2,
        })
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        self.assertEqual(progress["event_count"], 2)
        self.assertEqual(progress["last_event"]["kind"], "stage_checkpoint")

    def test_failure_artifact_preserves_gate_failure_diagnostics(self) -> None:
        output_root = self.root / "failed_calibration"
        progress_path = output_root / "calibration_progress.json"
        CALIBRATION.initialize_calibration_progress(
            progress_path, overwrite=False,
        )
        candidate_dir = output_root / "local" / "stage2" / "candidate_004"
        candidate = CALIBRATION.LocalFlowCandidate(
            hawkes_activity_scale=0.3,
            local_mm_interval_ms=1000.0,
            local_mm_quantity_multiplier=1.0,
            label="candidate_004",
        )
        evaluation = {
            "fit_wsmrmse": 1.0,
            "combined_uncertainty_wsmrmse": 1.1,
            "selection_score": 0.9,
            "selection_metric_scores": [],
            "two_sided_integrity_passed": True,
            "two_sided_integrity_failures": [],
            "finite_boundary_adequacy_passed": False,
            "finite_boundary_adequacy": {
                "passed": False,
                "failures": [{"metric": "boundary_event_ratio"}],
            },
            "seed_count": 1,
            "seed_wall_seconds": [0.5],
            "summary_paths": ["/run/summary.csv"],
            "errors": [],
            "moment_estimates": [],
        }
        reference = CALIBRATION.persist_candidate_evaluation(
            candidate_dir,
            block="global_local_flow",
            stage="stage2_refinement",
            cluster_id=None,
            candidate_index=4,
            candidate=candidate,
            evaluation=evaluation,
            progress_path=progress_path,
            overwrite=False,
        )
        self.assertFalse(reference["eligible"])
        self.assertEqual(
            reference["failed_predicates"],
            ["finite_boundary_adequacy_passed"],
        )
        checkpoint = CALIBRATION.persist_stage_checkpoint(
            output_root / "local" / "stage2",
            block="global_local_flow",
            stage="stage2_refinement",
            cluster_id=None,
            candidate_references=(reference,),
            promoted_candidate_indices=(),
            configured_ranked_survivor_count=2,
            progress_path=progress_path,
            overwrite=False,
        )
        self.assertEqual(checkpoint["status"], "failed_no_eligible_candidates")
        failure_path = CALIBRATION.persist_calibration_failure(
            output_root,
            RuntimeError("all local-flow stage2_refinement candidates failed"),
        )
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        self.assertEqual(failure["artifact_role"], "calibration_failure")
        self.assertEqual(failure["exception_type"], "RuntimeError")
        self.assertEqual(failure["progress_checkpoint"]["snapshot"]["status"], "failed")
        self.assertEqual(
            failure["progress_checkpoint"]["snapshot"]["last_event"]["kind"],
            "calibration_failure",
        )

    def test_multiday_loss_is_mean_of_complete_day_level_wmm_scores(self) -> None:
        """Equal training days must not be silently pooled by target-row count."""
        def evaluation(score: float, combined: float, suffix: str) -> dict[str, object]:
            return {
                "fit_wsmrmse": score,
                "combined_uncertainty_wsmrmse": combined,
                "selection_score": score,
                "selection_metric_scores": [],
                "seed_wall_seconds": [1.0, 2.0],
                "summary_paths": [f"/{suffix}/seed_1.csv", f"/{suffix}/seed_2.csv"],
                "errors": [],
                "moment_estimates": [],
            }

        first = CALIBRATION.TrainingDay(
            date="2019-01-30",
            universe_config=self.training,
            target_root=self.root / "targets_a",
            fields=CONFIG_FIELDS,
            rows=tuple(),
            universe_config_sha256="a" * 64,
        )
        second = CALIBRATION.TrainingDay(
            date="2019-03-27",
            universe_config=self.training,
            target_root=self.root / "targets_b",
            fields=CONFIG_FIELDS,
            rows=tuple(),
            universe_config_sha256="b" * 64,
        )
        pooled = CALIBRATION.aggregate_training_day_evaluations(
            ((first, evaluation(1.0, 2.0, "first")),
             (second, evaluation(3.0, 6.0, "second"))),
            seed_count=2,
        )
        self.assertEqual(
            pooled["aggregation"],
            "median_plus_mad_of_day_level_metric_balanced_huber",
        )
        self.assertEqual(pooled["training_day_count"], 2)
        self.assertEqual(pooled["seed_count"], 2)
        self.assertAlmostEqual(float(pooled["fit_wsmrmse"]), 2.0)
        self.assertAlmostEqual(float(pooled["combined_uncertainty_wsmrmse"]), 4.0)
        self.assertAlmostEqual(float(pooled["selection_score"]), 2.25)
        reports = pooled["training_day_evaluations"]
        self.assertEqual([entry["date"] for entry in reports], [
            "2019-01-30", "2019-03-27",
        ])

    def test_multiday_protocol_requires_explicit_pooled_direct_inputs(self) -> None:
        parser = CALIBRATION.build_parser()
        args = parser.parse_args([
            "--binary", "/bin/true",
            "--heldout-opening-source-config", str(self.heldout),
            "--cluster-assignments", str(self.root / "assignments.csv"),
            "--validation-sample", str(self.root / "validation.csv"),
            "--heldout-date", "2020-01-30",
            "--heldout-target-root", str(self.root / "heldout_targets"),
            "--output-dir", str(self.root / "output"),
            "--training-day", "2019-01-30", str(self.training), str(self.root / "a"),
            "--training-day", "2019-03-27", str(self.training), str(self.root / "b"),
        ])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                CALIBRATION.validate_arguments(args, parser)

    def test_pooling_provenance_requires_producer_project_root(self) -> None:
        parser = CALIBRATION.build_parser()
        args = parser.parse_args([
            "--binary", "/bin/true",
            "--training-universe-config", str(self.training),
            "--heldout-opening-source-config", str(self.heldout),
            "--cluster-assignments", str(self.root / "assignments.csv"),
            "--validation-sample", str(self.root / "validation.csv"),
            "--training-date", "2020-01-30",
            "--heldout-date", "2020-01-31",
            "--training-target-root", str(self.root / "training_targets"),
            "--heldout-target-root", str(self.root / "heldout_targets"),
            "--output-dir", str(self.root / "output"),
            "--pooling-provenance", str(self.root / "pool.json"),
        ])
        error_output = io.StringIO()
        with contextlib.redirect_stderr(error_output):
            with self.assertRaises(SystemExit):
                CALIBRATION.validate_arguments(args, parser)
        self.assertIn(
            "--pooling-provenance and --pooling-producer-project-root",
            error_output.getvalue(),
        )

    def test_cluster_policy_expands_to_entire_universe_not_per_stock_fit(self) -> None:
        assignments = self.root / "cluster_assignments.csv"
        validations = self.root / "validation_sample.csv"
        write_csv(assignments, (
            "book_id", "symbol", "cluster_id", "is_representative",
        ), [
            {"book_id": 0, "symbol": "AAA", "cluster_id": 0, "is_representative": 1},
            {"book_id": 1, "symbol": "BBB", "cluster_id": 0, "is_representative": 0},
            {"book_id": 2, "symbol": "CCC", "cluster_id": 1, "is_representative": 1},
            {"book_id": 3, "symbol": "DDD", "cluster_id": 1, "is_representative": 0},
        ])
        write_csv(validations, ("cluster_id", "symbol"), [
            {"cluster_id": 0, "symbol": "BBB"},
            {"cluster_id": 1, "symbol": "DDD"},
        ])
        layout = CALIBRATION.load_cluster_layout(
            assignments, validations, ("AAA", "BBB", "CCC", "DDD"),
        )
        selected = {
            0: CALIBRATION.Candidate(
                True, 5.0, 0.05, "threshold_5_depth_participation_0.05"
            ),
            1: CALIBRATION.Candidate(False, 0.0, 0.05, "disabled_baseline"),
        }
        policy = self.root / "cluster_value_agent_policy.csv"
        CALIBRATION.write_policy_csv(
            policy, ("AAA", "BBB", "CCC", "DDD"), layout, selected,
            policy_source="test", overwrite=False,
        )
        rows = read_csv(policy)
        self.assertEqual(list(rows[0]), list(CALIBRATION.POLICY_FIELDS))
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["symbol"] for row in rows}, {"AAA", "BBB", "CCC", "DDD"})
        by_symbol = {row["symbol"]: row for row in rows}
        self.assertEqual(by_symbol["AAA"]["value_threshold_bps"], "5")
        self.assertEqual(
            by_symbol["BBB"]["value_depth_participation"], "0.050000000000000003"
        )
        self.assertEqual(by_symbol["CCC"]["enabled"], "0")
        self.assertEqual(by_symbol["DDD"]["enabled"], "0")

    def test_candidate_grids_match_the_fixed_small_searches(self) -> None:
        local = CALIBRATION.local_flow_candidate_grid(
            (0.3,), (500.0, 1000.0, 2000.0), (0.5, 1.0, 2.0),
            (0.0, 0.25, 0.5, 1.0),
        )
        self.assertEqual(len(local), 37)
        self.assertFalse(local[0].local_mm_enabled)
        enabled = [candidate for candidate in local if candidate.local_mm_enabled]
        self.assertEqual(
            enabled[0],
            CALIBRATION.LocalFlowCandidate(
                0.3, 500.0, 0.5,
                "lambda_0.3_local_interval_500ms_local_quantity_0.5_local_improve_p_0",
                local_mm_improvement_probability=0.0,
            ),
        )
        shared = CALIBRATION.shared_quote_candidate_grid((2.0, 0.5, 1.0, 0.5))
        self.assertFalse(shared[0].enabled)
        self.assertEqual(
            [candidate.multiplier for candidate in shared], [0.0, 0.5, 1.0, 2.0],
        )
        policies = CALIBRATION.candidate_grid(
            (5.0, 8.0, 10.0, 15.0), (0.05, 0.1, 0.25, 0.5)
        )
        self.assertEqual(len(policies), 17)  # 16 enabled choices plus the baseline.
        self.assertFalse(policies[0].enabled)

    def test_local_flow_refinement_adds_unique_midpoint_candidates(self) -> None:
        original = CALIBRATION.local_flow_candidate_grid(
            (0.3,), (500.0, 1000.0, 2000.0), (0.5, 1.0, 2.0),
            (0.0, 0.25, 0.5, 1.0),
        )
        leader = next(
            candidate for candidate in original
            if candidate.hawkes_activity_scale == 0.3
            and candidate.local_mm_interval_ms == 1000.0
            and candidate.local_mm_quantity_multiplier == 1.0
            and candidate.local_mm_improvement_probability == 0.5
        )
        refined = CALIBRATION.refine_local_flow_candidates(
            (leader,), original, maximum_new_candidates=32,
        )
        self.assertGreater(len(refined), 0)
        self.assertLessEqual(len(refined), 32)
        original_keys = {
            (item.hawkes_activity_scale, item.local_mm_interval_ms,
             item.local_mm_quantity_multiplier,
             item.local_mm_improvement_probability)
            for item in original
        }
        refined_keys = {
            (item.hawkes_activity_scale, item.local_mm_interval_ms,
             item.local_mm_quantity_multiplier,
             item.local_mm_improvement_probability)
            for item in refined
        }
        self.assertEqual(len(refined_keys), len(refined))
        self.assertTrue(original_keys.isdisjoint(refined_keys))
        self.assertTrue(all(item.label.startswith("refined_") for item in refined))

    def test_rank_one_command_uses_the_actual_fragmented_policy_interface(self) -> None:
        controls = CALIBRATION.LocalFlowCandidate(
            0.3, 1000.0, 1.0, "lambda_0.3_local_interval_1000ms_local_quantity_1",
        )
        command = CALIBRATION.command_for_run(
            launcher=("srun", "--ntasks=1"),
            binary=pathlib.Path("/bin/fragmented_mpi_lob"),
            config=pathlib.Path("/tmp/subset.csv"),
            policy=pathlib.Path("/tmp/policy.csv"),
            summary=pathlib.Path("/tmp/summary.csv"),
            duration=300,
            seed=1729,
            local_controls=controls,
            shared_quote_multiplier=None,
            enable_shared_mm=False,
            enable_value_agents=True,
        )
        self.assertEqual(command[:2], ["srun", "--ntasks=1"])
        self.assertNotIn("--venues", command)
        self.assertEqual(command[command.index("--window-ms") + 1], "1000")
        self.assertIn("--disable-shared-mm", command)
        self.assertEqual(
            command[command.index("--value-agent-policy-csv") + 1],
            "/tmp/policy.csv",
        )
        self.assertEqual(
            command[command.index("--asset-summary-csv") + 1],
            "/tmp/summary.csv",
        )
        self.assertEqual(
            command[command.index("--hawkes-activity-scale") + 1], "0.3",
        )
        self.assertEqual(
            command[command.index("--local-mm-interval-ms") + 1], "1000.0",
        )
        self.assertEqual(
            command[command.index("--local-mm-quantity-multiplier") + 1], "1.0",
        )
        self.assertEqual(
            command[command.index("--local-mm-improvement-probability") + 1],
            "0.0",
        )

    def test_command_can_disable_both_later_blocks_or_enable_shared_proxy(self) -> None:
        controls = CALIBRATION.LocalFlowCandidate(
            0.4, 500.0, 2.0, "lambda_0.4_local_interval_500ms_local_quantity_2",
        )
        local_only = CALIBRATION.command_for_run(
            launcher=(),
            binary=pathlib.Path("/bin/fragmented_mpi_lob"),
            config=pathlib.Path("/tmp/subset.csv"),
            policy=None,
            summary=pathlib.Path("/tmp/summary.csv"),
            duration=300,
            seed=1729,
            local_controls=controls,
            shared_quote_multiplier=None,
            enable_shared_mm=False,
            enable_value_agents=False,
        )
        self.assertIn("--disable-shared-mm", local_only)
        self.assertIn("--disable-value-agent", local_only)
        self.assertNotIn("--value-agent-policy-csv", local_only)

        final_model = CALIBRATION.command_for_run(
            launcher=(),
            binary=pathlib.Path("/bin/fragmented_mpi_lob"),
            config=pathlib.Path("/tmp/subset.csv"),
            policy=pathlib.Path("/tmp/policy.csv"),
            summary=pathlib.Path("/tmp/summary.csv"),
            duration=300,
            seed=1729,
            local_controls=controls,
            shared_quote_multiplier=1.0,
            enable_shared_mm=True,
            enable_value_agents=True,
        )
        self.assertNotIn("--disable-shared-mm", final_model)
        self.assertEqual(
            final_model[final_model.index("--shared-quote-multiplier") + 1], "1.0",
        )
        self.assertIn("--shared-quote-relative", final_model)

    def test_noncanonical_smoke_run_is_preliminary_with_frozen_controls(self) -> None:
        """A short smoke run must never masquerade as the thesis protocol."""
        assignments = self.root / "cluster_assignments.csv"
        validations = self.root / "validation_sample.csv"
        write_csv(assignments, (
            "book_id", "symbol", "cluster_id", "is_representative",
        ), [
            {"book_id": 0, "symbol": "AAA", "cluster_id": 0, "is_representative": 1},
            {"book_id": 1, "symbol": "BBB", "cluster_id": 0, "is_representative": 0},
            {"book_id": 2, "symbol": "CCC", "cluster_id": 1, "is_representative": 1},
            {"book_id": 3, "symbol": "DDD", "cluster_id": 1, "is_representative": 0},
        ])
        write_csv(validations, ("cluster_id", "symbol"), [
            {"cluster_id": 0, "symbol": "BBB"},
            {"cluster_id": 1, "symbol": "DDD"},
        ])
        training_root = self.root / "training_targets"
        heldout_root = self.root / "heldout_targets"
        symbols = ("AAA", "BBB", "CCC", "DDD")
        for root, day in ((training_root, "2020-01-30"), (heldout_root, "2020-01-31")):
            self.write_targets(root, date=day, symbols=symbols, window=1)
            self.write_targets(root, date=day, symbols=symbols, window=2)
            self.write_targets(root, date=day, symbols=symbols, window=None)

        fake_binary = self.root / "fake_fragmented_mpi_lob.py"
        fake_binary.write_text(
            "#!/usr/bin/env python3\n"
            "import csv\n"
            "import pathlib\n"
            "import sys\n"
            "args = sys.argv[1:]\n"
            "duration = int(args[args.index('--duration-seconds') + 1])\n"
            "def value(flag):\n"
            "    return pathlib.Path(args[args.index(flag) + 1])\n"
            "config = value('--universe-config')\n"
            "summary = value('--asset-summary-csv')\n"
            "with config.open(newline='', encoding='utf-8') as source:\n"
            "    input_rows = list(csv.DictReader(source))\n"
            "metrics = ('background_event_rate', 'mean_spread_ticks', 'mean_bid_depth', 'mean_ask_depth', "
            "'mid_move_rate', 'return_variance', 'return_kurtosis', "
            "'absolute_return_acf1', 'two_sided_sample_fraction')\n"
            "fields = ('asset_id', 'symbol', 'sample_count', 'expected_sample_count', "
            "'invalid_sample_count', 'structurally_valid', 'background_event_count', "
            "'background_market_requested_quantity', 'background_cancel_requested_quantity', "
            "'removal_boundary_truncation_events', 'removal_boundary_truncated_quantity', "
            "'background_boundary_truncation_events', 'background_boundary_truncated_quantity', "
            "'value_order_count', 'value_requested_quantity', "
            "'value_boundary_truncation_events', 'value_boundary_truncated_quantity', "
            "'other_boundary_truncation_events', 'other_boundary_truncated_quantity', *metrics)\n"
            "summary.parent.mkdir(parents=True, exist_ok=True)\n"
            "with summary.open('w', newline='', encoding='utf-8') as output:\n"
            "    writer = csv.DictWriter(output, fieldnames=fields)\n"
            "    writer.writeheader()\n"
            "    for index, row in enumerate(input_rows):\n"
            "        writer.writerow({'asset_id': index, 'symbol': row['symbol'], "
            "        'sample_count': duration, 'expected_sample_count': duration, "
            "        'invalid_sample_count': 0, 'structurally_valid': 1, "
            "        'background_event_count': 100, 'background_market_requested_quantity': 50, "
            "        'background_cancel_requested_quantity': 50, "
            "        'removal_boundary_truncation_events': 0, "
            "        'removal_boundary_truncated_quantity': 0, "
            "        'background_boundary_truncation_events': 0, "
            "        'background_boundary_truncated_quantity': 0, "
            "        'value_order_count': 0, 'value_requested_quantity': 0, "
            "        'value_boundary_truncation_events': 0, 'value_boundary_truncated_quantity': 0, "
            "        'other_boundary_truncation_events': 0, 'other_boundary_truncated_quantity': 0, "
            "        **{metric: (1.0 if metric == 'two_sided_sample_fraction' else 10.0) for metric in metrics}})\n",
            encoding="utf-8",
        )
        fake_binary.chmod(0o755)
        output_dir = self.root / "calibration_output"
        parser = CALIBRATION.build_parser()
        args = parser.parse_args([
            "--binary", str(fake_binary),
            "--training-universe-config", str(self.training),
            "--heldout-opening-source-config", str(self.heldout),
            "--cluster-assignments", str(assignments),
            "--validation-sample", str(validations),
            "--training-date", "2020-01-30",
            "--heldout-date", "2020-01-31",
            "--training-target-root", str(training_root),
            "--heldout-target-root", str(heldout_root),
            "--output-dir", str(output_dir),
            "--stage1-duration", "1",
            "--stage2-duration", "2",
            "--stage3-duration", "3",
            "--session-duration", "3",
            "--stage1-top-candidates", "2",
            "--stage2-top-candidates", "1",
            "--stage1-seeds", "1729",
            "--stage2-seeds", "1729", "7919",
            "--stage3-seeds", "1729", "7919",
            "--thresholds", "5",
            "--depth-participations", "0.05",
            "--hawkes-activity-scales", "0.3",
            "--local-mm-intervals-ms", "1000",
            "--local-mm-quantity-multipliers", "1",
            "--shared-quote-multipliers", "1.0",
            "--marketwide-validation",
        ])
        CALIBRATION.validate_arguments(args, parser)
        result = CALIBRATION.run(args)
        report_path = pathlib.Path(str(result["report"]))
        handoff_path = pathlib.Path(str(result["preliminary_result"]))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        self.assertFalse(result["certified_for_case_study"])
        self.assertNotIn("handoff", result)
        self.assertTrue(report["certification"]["empirical_fit_passed"])
        self.assertEqual(
            report["certification"]["empirical_fit_acceptance_scope"],
            "full_universe_marketwide",
        )
        self.assertTrue(
            report["certification"]["stratified_empirical_fit_passed"]
        )
        self.assertEqual(
            report["certification"]
            ["stratified_empirical_fit_acceptance_role"],
            "required_reported_diagnostic_only",
        )
        self.assertTrue(
            report["certification"]["marketwide_empirical_fit_passed"]
        )
        self.assertFalse(report["certification"]["certified_for_case_study"])
        self.assertTrue(handoff["certification"]["empirical_fit_passed"])
        self.assertFalse(handoff["certification"]["certified_for_case_study"])
        self.assertFalse(
            report["certification"]["runtime_matches_certification_profile"]
        )
        self.assertFalse(
            report["observed_runtime_profile"][
                "certification_profile_enforced"
            ]
        )
        horizon = report["protocol"]["three_horizon_screen"]
        self.assertEqual(
            horizon["stage1"]
            ["value_policy_survivors_after_stage_per_cluster"],
            2,
        )
        self.assertEqual(
            horizon["stage2"]
            ["value_policy_survivors_after_stage_per_cluster"],
            2,
        )
        self.assertEqual(horizon["stage1"]["shared_quote_survivor_cap"], 2)
        self.assertEqual(
            horizon["stage1"]["shared_quote_candidates_promoted"], 2,
        )
        self.assertEqual(horizon["stage2"]["shared_quote_survivor_cap"], 1)
        self.assertEqual(
            horizon["stage2"]["shared_quote_candidates_promoted"], 1,
        )
        self.assertEqual(horizon["stage3"]["shared_quote_survivor_cap"], 1)
        self.assertEqual(
            horizon["stage3"]["shared_quote_candidates_promoted"], 1,
        )
        shared_counts = handoff["observed_survivor_counts"][
            "global_shared_quote"
        ]
        self.assertEqual(
            [
                shared_counts[stage]["promoted_candidates"]
                for stage in (
                    "stage1_screen", "stage2_refinement", "stage3_full",
                )
            ],
            [2, 1, 1],
        )
        self.assertEqual(handoff["artifact_role"], "preliminary_not_certified")
        self.assertEqual(report["schema_version"], 2)
        sampled_status = json.loads(
            (output_dir / "heldout_stratified_validation_status.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(sampled_status["schema_version"], 2)
        self.assertTrue(sampled_status["structural_adequacy_passed"])
        self.assertEqual(sampled_status["failure_reasons"], [])
        self.assertEqual(sampled_status["empirical_fit_failure_reasons"], [])
        self.assertEqual(
            sampled_status["empirical_fit_acceptance_role"],
            "required_reported_diagnostic_only",
        )
        marketwide_status = json.loads(
            (output_dir / "heldout_marketwide_validation_status.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(marketwide_status["schema_version"], 2)
        self.assertEqual(
            marketwide_status["empirical_fit_acceptance_role"],
            "authoritative_certification_gate",
        )
        self.assertEqual(
            handoff["heldout_stratified_validation"]
            ["empirical_fit_acceptance_role"],
            "required_reported_diagnostic_only",
        )
        self.assertEqual(report["protocol"]["training_date"], "2020-01-30")
        self.assertNotIn(
            "training_day_count",
            report["global_local_flow_selection"]["training_evaluation"],
        )
        self.assertEqual(
            report["global_local_flow_selection"]["controls"],
            {
                "hawkes_activity_scale": 0.3,
                "local_mm_enabled": False,
                "local_mm_improvement_probability": 0.0,
                "local_mm_interval_ms": 1000.0,
                "local_mm_quantity_multiplier": 1.0,
                "label": "lambda_0.3_local_mm_disabled_baseline",
            },
        )
        self.assertEqual(
            handoff["runtime_controls"],
            {
                "decision_window_ms": 1000.0,
                "hawkes_activity_scale": 0.3,
                "local_market_maker_enabled": False,
                "local_mm_interval_ms": 1000.0,
                "local_mm_quantity_multiplier": 1.0,
                "local_mm_improvement_probability": 0.0,
                "shared_market_maker_enabled": False,
                "shared_quote_levels": 1,
                "shared_quote_mode": "relative_to_empirical_symbol_quote_size",
                "shared_quote_multiplier": 0.0,
            },
        )
        self.assertEqual(
            handoff["calibration_report_sha256"],
            CALIBRATION.sha256_file(report_path),
        )
        self.assertTrue(pathlib.Path(handoff["value_agent_policy_csv"]).is_file())
        selected = read_csv(output_dir / "cluster_selected_policies.csv")
        self.assertEqual(len(selected), 2)
        self.assertEqual({row["shared_mm_enabled"] for row in selected}, {"0"})
        self.assertEqual({row["shared_quote_multiplier"] for row in selected}, {"0.0"})
        self.assertTrue(result["marketwide_validation"])
        self.assertTrue((
            output_dir / "heldout_marketwide_coverage_summary.json"
        ).is_file())
        self.assertEqual(
            report["heldout_marketwide_distributional_validation"]
            ["coverage_summary"]["failing_symbol_count"],
            0,
        )

    def test_finite_but_grossly_poor_heldout_fit_is_preliminary_not_certified(self) -> None:
        """Finite output and complete books alone must never certify the ABM."""
        assignments = self.root / "poor_fit_cluster_assignments.csv"
        validations = self.root / "poor_fit_validation_sample.csv"
        write_csv(assignments, (
            "book_id", "symbol", "cluster_id", "is_representative",
        ), [
            {"book_id": 0, "symbol": "AAA", "cluster_id": 0, "is_representative": 1},
            {"book_id": 1, "symbol": "BBB", "cluster_id": 0, "is_representative": 0},
            {"book_id": 2, "symbol": "CCC", "cluster_id": 1, "is_representative": 1},
            {"book_id": 3, "symbol": "DDD", "cluster_id": 1, "is_representative": 0},
        ])
        write_csv(validations, ("cluster_id", "symbol"), [
            {"cluster_id": 0, "symbol": "BBB"},
            {"cluster_id": 1, "symbol": "DDD"},
        ])
        training_root = self.root / "poor_fit_training_targets"
        heldout_root = self.root / "poor_fit_heldout_targets"
        symbols = ("AAA", "BBB", "CCC", "DDD")
        for root, day in (
            (training_root, "2020-01-30"),
            (heldout_root, "2020-01-31"),
        ):
            self.write_targets(root, date=day, symbols=symbols, window=1)
            self.write_targets(root, date=day, symbols=symbols, window=2)
            self.write_targets(root, date=day, symbols=symbols, window=None)

        fake_binary = self.root / "fake_grossly_poor_heldout_lob.py"
        fake_binary.write_text(
            "#!/usr/bin/env python3\n"
            "import csv\n"
            "import pathlib\n"
            "import sys\n"
            "args = sys.argv[1:]\n"
            "duration = int(args[args.index('--duration-seconds') + 1])\n"
            "def value(flag):\n"
            "    return pathlib.Path(args[args.index(flag) + 1])\n"
            "config = value('--universe-config')\n"
            "summary = value('--asset-summary-csv')\n"
            "heldout = config.name.startswith('heldout_')\n"
            "with config.open(newline='', encoding='utf-8') as source:\n"
            "    input_rows = list(csv.DictReader(source))\n"
            "metrics = ('background_event_rate', 'mean_spread_ticks', 'mean_bid_depth', 'mean_ask_depth', "
            "'mid_move_rate', 'return_variance', 'return_kurtosis', "
            "'absolute_return_acf1', 'two_sided_sample_fraction')\n"
            "fields = ('asset_id', 'symbol', 'sample_count', 'expected_sample_count', "
            "'invalid_sample_count', 'structurally_valid', 'background_event_count', "
            "'background_market_requested_quantity', 'background_cancel_requested_quantity', "
            "'removal_boundary_truncation_events', 'removal_boundary_truncated_quantity', "
            "'background_boundary_truncation_events', 'background_boundary_truncated_quantity', "
            "'value_order_count', 'value_requested_quantity', "
            "'value_boundary_truncation_events', 'value_boundary_truncated_quantity', "
            "'other_boundary_truncation_events', 'other_boundary_truncated_quantity', *metrics)\n"
            "summary.parent.mkdir(parents=True, exist_ok=True)\n"
            "with summary.open('w', newline='', encoding='utf-8') as output:\n"
            "    writer = csv.DictWriter(output, fieldnames=fields)\n"
            "    writer.writeheader()\n"
            "    for index, row in enumerate(input_rows):\n"
            "        metric_values = {metric: (1.0 if metric == "
            "'two_sided_sample_fraction' else (1000000000.0 if heldout else 10.0)) "
            "for metric in metrics}\n"
            "        writer.writerow({'asset_id': index, 'symbol': row['symbol'], "
            "        'sample_count': duration, 'expected_sample_count': duration, "
            "        'invalid_sample_count': 0, 'structurally_valid': 1, "
            "        'background_event_count': 100, 'background_market_requested_quantity': 50, "
            "        'background_cancel_requested_quantity': 50, "
            "        'removal_boundary_truncation_events': 0, "
            "        'removal_boundary_truncated_quantity': 0, "
            "        'background_boundary_truncation_events': 0, "
            "        'background_boundary_truncated_quantity': 0, "
            "        'value_order_count': 0, 'value_requested_quantity': 0, "
            "        'value_boundary_truncation_events': 0, 'value_boundary_truncated_quantity': 0, "
            "        'other_boundary_truncation_events': 0, 'other_boundary_truncated_quantity': 0, "
            "        **metric_values})\n",
            encoding="utf-8",
        )
        fake_binary.chmod(0o755)
        output_dir = self.root / "poor_fit_calibration_output"
        parser = CALIBRATION.build_parser()
        args = parser.parse_args([
            "--binary", str(fake_binary),
            "--training-universe-config", str(self.training),
            "--heldout-opening-source-config", str(self.heldout),
            "--cluster-assignments", str(assignments),
            "--validation-sample", str(validations),
            "--training-date", "2020-01-30",
            "--heldout-date", "2020-01-31",
            "--training-target-root", str(training_root),
            "--heldout-target-root", str(heldout_root),
            "--output-dir", str(output_dir),
            "--stage1-duration", "1",
            "--stage2-duration", "2",
            "--stage3-duration", "3",
            "--session-duration", "3",
            "--stage1-top-candidates", "2",
            "--stage2-top-candidates", "1",
            "--stage1-seeds", "1729",
            "--stage2-seeds", "1729", "7919",
            "--stage3-seeds", "1729", "7919",
            "--thresholds", "5",
            "--depth-participations", "0.05",
            "--hawkes-activity-scales", "0.3",
            "--local-mm-intervals-ms", "1000",
            "--local-mm-quantity-multipliers", "1",
            "--shared-quote-multipliers", "1.0",
            "--marketwide-validation",
        ])
        CALIBRATION.validate_arguments(args, parser)
        result = CALIBRATION.run(args)

        self.assertFalse(result["certified_for_case_study"])
        self.assertNotIn("handoff", result)
        self.assertFalse((output_dir / "calibration_handoff.json").exists())
        preliminary_path = pathlib.Path(str(result["preliminary_result"]))
        self.assertEqual(preliminary_path.name, "preliminary_calibration_result.json")
        preliminary = json.loads(preliminary_path.read_text(encoding="utf-8"))
        self.assertEqual(preliminary["artifact_role"], "preliminary_not_certified")
        self.assertFalse(preliminary["certification"]["empirical_fit_passed"])
        self.assertFalse(preliminary["certification"]["certified_for_case_study"])

        report = json.loads(pathlib.Path(str(result["report"])).read_text(encoding="utf-8"))
        self.assertTrue(report["certification"]["execution_integrity_passed"])
        self.assertTrue(report["certification"]["coverage_passed"])
        self.assertFalse(report["certification"]["empirical_fit_passed"])
        self.assertFalse(report["certification"]["certified_for_case_study"])

    def test_multiday_run_scores_each_session_and_freezes_pooled_backgrounds(self) -> None:
        """The held-out model must freeze the declared pooled inputs, not day one."""
        assignments = self.root / "cluster_assignments.csv"
        validations = self.root / "validation_sample.csv"
        write_csv(assignments, (
            "book_id", "symbol", "cluster_id", "is_representative",
        ), [
            {"book_id": 0, "symbol": "AAA", "cluster_id": 0, "is_representative": 1},
            {"book_id": 1, "symbol": "BBB", "cluster_id": 0, "is_representative": 0},
            {"book_id": 2, "symbol": "CCC", "cluster_id": 1, "is_representative": 1},
            {"book_id": 3, "symbol": "DDD", "cluster_id": 1, "is_representative": 0},
        ])
        write_csv(validations, ("cluster_id", "symbol"), [
            {"cluster_id": 0, "symbol": "BBB"},
            {"cluster_id": 1, "symbol": "DDD"},
        ])
        second_training = self.root / "training_20190327.csv"
        second_rows = config_rows()
        for row in second_rows:
            symbol = str(row["symbol"]).lower()
            data_dir = self.root / "daily_20190327" / symbol
            data_dir.mkdir(parents=True, exist_ok=True)
            rates = data_dir / "rates.csv"
            rates.write_text("event_type,configured_mu\nlimit_buy,2\n", encoding="utf-8")
            for filename in CALIBRATION.SIMULATOR_EMPIRICAL_INPUT_FILENAMES:
                (data_dir / filename).write_text("value,count\n2,1\n", encoding="utf-8")
            (data_dir / f"itch_manifest_{symbol}_20190327.json").write_text(
                '{"schema_version":1}\n', encoding="utf-8",
            )
            row["data_dir"] = str(data_dir)
            row["hawkes_rates_file"] = str(rates)
        write_csv(second_training, CONFIG_FIELDS, second_rows)
        pooled_training = self.root / "pooled_training.csv"
        pooled_rows = config_rows()
        for row in pooled_rows:
            symbol = str(row["symbol"]).lower()
            data_dir = self.root / "pooled_five_day" / symbol
            data_dir.mkdir(parents=True, exist_ok=True)
            rates = data_dir / "rates.csv"
            rates.write_text("event_type,configured_mu\nlimit_buy,3\n", encoding="utf-8")
            for filename in CALIBRATION.SIMULATOR_EMPIRICAL_INPUT_FILENAMES:
                (data_dir / filename).write_text("value,count\n3,1\n", encoding="utf-8")
            (data_dir / f"itch_manifest_{symbol}_pooled.json").write_text(
                '{"schema_version":1}\n', encoding="utf-8",
            )
            row["data_dir"] = str(data_dir)
            row["hawkes_rates_file"] = str(rates)
        write_csv(pooled_training, CONFIG_FIELDS, pooled_rows)

        first_targets = self.root / "targets_20190130"
        second_targets = self.root / "targets_20190327"
        heldout_targets = self.root / "targets_20200130"
        symbols = ("AAA", "BBB", "CCC", "DDD")
        for root, day in (
            (first_targets, "2019-01-30"),
            (second_targets, "2019-03-27"),
            (heldout_targets, "2020-01-30"),
        ):
            self.write_targets(root, date=day, symbols=symbols, window=1)
            self.write_targets(root, date=day, symbols=symbols, window=2)
            self.write_targets(root, date=day, symbols=symbols, window=None)

        fake_binary = self.root / "fake_multiday_fragmented_mpi_lob.py"
        fake_binary.write_text(
            "#!/usr/bin/env python3\n"
            "import csv\n"
            "import pathlib\n"
            "import sys\n"
            "args = sys.argv[1:]\n"
            "duration = int(args[args.index('--duration-seconds') + 1])\n"
            "def value(flag):\n"
            "    return pathlib.Path(args[args.index(flag) + 1])\n"
            "config = value('--universe-config')\n"
            "summary = value('--asset-summary-csv')\n"
            "with config.open(newline='', encoding='utf-8') as source:\n"
            "    input_rows = list(csv.DictReader(source))\n"
            "metrics = ('background_event_rate', 'mean_spread_ticks', 'mean_bid_depth', 'mean_ask_depth', "
            "'mid_move_rate', 'return_variance', 'return_kurtosis', "
            "'absolute_return_acf1', 'two_sided_sample_fraction')\n"
            "fields = ('asset_id', 'symbol', 'sample_count', 'expected_sample_count', "
            "'invalid_sample_count', 'structurally_valid', 'background_event_count', "
            "'background_market_requested_quantity', 'background_cancel_requested_quantity', "
            "'removal_boundary_truncation_events', 'removal_boundary_truncated_quantity', "
            "'background_boundary_truncation_events', 'background_boundary_truncated_quantity', "
            "'value_order_count', 'value_requested_quantity', "
            "'value_boundary_truncation_events', 'value_boundary_truncated_quantity', "
            "'other_boundary_truncation_events', 'other_boundary_truncated_quantity', *metrics)\n"
            "summary.parent.mkdir(parents=True, exist_ok=True)\n"
            "with summary.open('w', newline='', encoding='utf-8') as output:\n"
            "    writer = csv.DictWriter(output, fieldnames=fields)\n"
            "    writer.writeheader()\n"
            "    for index, row in enumerate(input_rows):\n"
            "        writer.writerow({'asset_id': index, 'symbol': row['symbol'], "
            "        'sample_count': duration, 'expected_sample_count': duration, "
            "        'invalid_sample_count': 0, 'structurally_valid': 1, "
            "        'background_event_count': 100, 'background_market_requested_quantity': 50, "
            "        'background_cancel_requested_quantity': 50, "
            "        'removal_boundary_truncation_events': 0, "
            "        'removal_boundary_truncated_quantity': 0, "
            "        'background_boundary_truncation_events': 0, "
            "        'background_boundary_truncated_quantity': 0, "
            "        'value_order_count': 0, 'value_requested_quantity': 0, "
            "        'value_boundary_truncation_events': 0, 'value_boundary_truncated_quantity': 0, "
            "        'other_boundary_truncation_events': 0, 'other_boundary_truncated_quantity': 0, "
            "        **{metric: (1.0 if metric == 'two_sided_sample_fraction' else 10.0) for metric in metrics}})\n",
            encoding="utf-8",
        )
        fake_binary.chmod(0o755)
        output_dir = self.root / "multiday_calibration_output"
        parser = CALIBRATION.build_parser()
        args = parser.parse_args([
            "--binary", str(fake_binary),
            "--heldout-opening-source-config", str(self.heldout),
            "--cluster-assignments", str(assignments),
            "--validation-sample", str(validations),
            "--heldout-date", "2020-01-30",
            "--heldout-target-root", str(heldout_targets),
            # Deliberately reverse the command order: provenance is reported
            # chronologically, but each date retains its own config/root.
            "--training-day", "2019-03-27", str(second_training), str(second_targets),
            "--training-day", "2019-01-30", str(self.training), str(first_targets),
            "--pooled-training-universe-config", str(pooled_training),
            "--output-dir", str(output_dir),
            "--stage1-duration", "1",
            "--stage2-duration", "2",
            "--stage3-duration", "3",
            "--session-duration", "3",
            "--stage1-top-candidates", "2",
            "--stage2-top-candidates", "1",
            "--stage1-seeds", "1729",
            "--stage2-seeds", "1729", "7919",
            "--stage3-seeds", "1729", "7919",
            "--thresholds", "5",
            "--depth-participations", "0.05",
            "--hawkes-activity-scales", "0.3",
            "--local-mm-intervals-ms", "1000",
            "--local-mm-quantity-multipliers", "1",
            "--shared-quote-multipliers", "1.0",
        ])
        CALIBRATION.validate_arguments(args, parser)
        result = CALIBRATION.run(args)
        report = json.loads(pathlib.Path(str(result["report"])).read_text(encoding="utf-8"))
        handoff = json.loads(pathlib.Path(str(result["preliminary_result"])).read_text(encoding="utf-8"))

        protocol = report["protocol"]
        self.assertIsNone(protocol["training_date"])
        self.assertEqual(protocol["training_dates"], ["2019-01-30", "2019-03-27"])
        self.assertEqual(protocol["training_day_count"], 2)
        self.assertEqual(
            [entry["target_root"] for entry in protocol["training_days"]],
            [str(first_targets.resolve()), str(second_targets.resolve())],
        )
        self.assertEqual(
            protocol["pooled_training_universe_config"], str(pooled_training.resolve()),
        )
        selected_evaluation = report["global_local_flow_selection"]["training_evaluation"]
        self.assertEqual(selected_evaluation["training_day_count"], 2)
        self.assertEqual(
            [entry["date"] for entry in selected_evaluation["training_day_evaluations"]],
            ["2019-01-30", "2019-03-27"],
        )
        self.assertEqual(handoff["training_universe_config"], str(pooled_training.resolve()))
        self.assertEqual(handoff["training_days"], protocol["training_days"])
        frozen_rows = read_csv(output_dir / "heldout_openings_frozen_backgrounds.csv")
        self.assertEqual(frozen_rows[0]["data_dir"], pooled_rows[0]["data_dir"])
        self.assertEqual(
            frozen_rows[0]["hawkes_rates_file"], pooled_rows[0]["hawkes_rates_file"]
        )
        self.assertEqual(frozen_rows[0]["fundamental_price_ticks"], "10050")
        self.assertTrue(list(output_dir.glob(
            "**/day_20190130/**/fragmented_asset_summary.csv"
        )))
        self.assertTrue(list(output_dir.glob(
            "**/day_20190327/**/fragmented_asset_summary.csv"
        )))


if __name__ == "__main__":
    unittest.main()
