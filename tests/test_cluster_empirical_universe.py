#!/usr/bin/env python3
"""Regression tests for empirical-universe liquidity clustering."""

from __future__ import annotations

import csv
import json
import pathlib
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import cluster_empirical_universe as clustering  # noqa: E402


CONFIG_FIELDS = (
    "book_id",
    "symbol",
    "data_dir",
    "fundamental_price_ticks",
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


class EmpiricalUniverseClusterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.data_root = self.root / "derived_data"
        self.config = self.root / "nasdaq_common_plus_qqq_20200130.csv"
        self.symbols = [
            "QQQ", "AAA", "AAB", "AAC", "AAD", "AAE",
            "AAF", "AAG", "AAH", "AAI", "AAJ", "AAK",
        ]
        rows: list[dict[str, object]] = []
        for index, symbol in enumerate(self.symbols):
            regime = index // 4
            directory_name = f"itch_20200130_{symbol.lower()}"
            directory = self.data_root / directory_name
            directory.mkdir(parents=True)
            # The three regimes are deliberately separated in all empirical
            # feature directions, with small within-regime variation.
            count = 100 * (regime + 1) + 5 * index
            with (directory / f"itch_manifest_{symbol.lower()}_20200130.json").open(
                "w", encoding="utf-8"
            ) as output:
                json.dump({
                    "session_start": "09:30:00",
                    "session_end": "16:00:00",
                    "distribution_observation_counts": {
                        event: count for event in clustering.EVENT_NAMES
                    },
                }, output)
            write_csv(
                directory / f"market_targets_{symbol.lower()}_20200130.csv",
                ("name", "target", "scale", "weight"),
                [
                    {"name": "mean_spread_ticks",
                     "target": 1.0 + 3.0 * regime + 0.05 * (index % 4),
                     "scale": 1.0, "weight": 1.0},
                    {"name": "mean_bid_depth",
                     "target": 100.0 + 300.0 * regime + index,
                     "scale": 1.0, "weight": 1.0},
                    {"name": "mean_ask_depth",
                     "target": 110.0 + 300.0 * regime + index,
                     "scale": 1.0, "weight": 1.0},
                    {"name": "return_variance",
                     "target": 1.0e-6 * (regime + 1) + 1.0e-8 * index,
                     "scale": 1.0, "weight": 1.0},
                ],
            )
            # Three-stage calibration creates these neighbouring prefix files.
            # Clustering must continue to consume full-session direct features.
            write_csv(
                directory / f"market_targets_{symbol.lower()}_20200130_window_300s.csv",
                ("name", "target", "scale", "weight"),
                [
                    {"name": "mean_spread_ticks", "target": 99.0,
                     "scale": 1.0, "weight": 1.0},
                    {"name": "mean_bid_depth", "target": 99.0,
                     "scale": 1.0, "weight": 1.0},
                    {"name": "mean_ask_depth", "target": 99.0,
                     "scale": 1.0, "weight": 1.0},
                    {"name": "return_variance", "target": 99.0,
                     "scale": 1.0, "weight": 1.0},
                ],
            )
            rows.append({
                "book_id": index,
                "symbol": symbol,
                # Use a relative configuration path to exercise --data-root.
                "data_dir": directory_name,
                "fundamental_price_ticks": 10_000 + 250 * index,
            })
        write_csv(self.config, CONFIG_FIELDS, rows)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_clustering(self, output_dir: pathlib.Path) -> None:
        result = clustering.main([
            "--universe-config", str(self.config),
            "--data-root", str(self.data_root),
            "--output-dir", str(output_dir),
            "--clusters", "3",
            "--validation-per-cluster", "2",
            "--seed", "7",
        ])
        self.assertEqual(result, 0)

    def test_assignments_and_validation_are_deterministic(self) -> None:
        first = self.root / "first"
        second = self.root / "second"
        self.run_clustering(first)
        self.run_clustering(second)

        first_assignments = (first / "cluster_assignments.csv").read_text(
            encoding="utf-8"
        )
        second_assignments = (second / "cluster_assignments.csv").read_text(
            encoding="utf-8"
        )
        self.assertEqual(first_assignments, second_assignments)
        self.assertEqual(
            (first / "validation_sample.csv").read_text(encoding="utf-8"),
            (second / "validation_sample.csv").read_text(encoding="utf-8"),
        )

        assignments = read_csv(first / "cluster_assignments.csv")
        validations = read_csv(first / "validation_sample.csv")
        self.assertEqual(len(assignments), len(self.symbols))
        self.assertEqual(
            set(assignments[0]), set(clustering.ASSIGNMENT_FIELDS),
        )
        self.assertEqual(
            {"event_rate_per_second", "mean_spread_ticks", "mean_top_depth",
             "return_variance", "opening_mid_price_ticks"},
            set(clustering.RAW_FEATURE_NAMES),
        )
        qqq = next(row for row in assignments if row["symbol"] == "QQQ")
        self.assertEqual(float(qqq["mean_spread_ticks"]), 1.0)

        cluster_ids = {int(row["cluster_id"]) for row in assignments}
        self.assertEqual(cluster_ids, {0, 1, 2})
        representatives = [row for row in assignments
                           if row["is_representative"] == "1"]
        self.assertEqual(len(representatives), 3)
        self.assertEqual({row["symbol"] for row in representatives},
                         {row["symbol"] for row in assignments
                          if row["selection_role"] == "representative"})
        self.assertFalse(
            {row["symbol"] for row in representatives}.intersection(
                row["symbol"] for row in validations
            )
        )

        expected_validation_count = sum(
            min(2, max(0, sum(int(row["cluster_id"]) == cluster_id
                              for row in assignments) - 1))
            for cluster_id in cluster_ids
        )
        self.assertEqual(len(validations), expected_validation_count)
        self.assertTrue(all(row["symbol"] in self.symbols for row in validations))

        with (first / "cluster_manifest.json").open(encoding="utf-8") as source:
            manifest = json.load(source)
        self.assertFalse(manifest["inputs"]["raw_itch_input_used"])
        self.assertEqual(manifest["clustering"]["cluster_count"], 3)
        self.assertEqual(manifest["counts"]["accepted_books"], len(self.symbols))
        self.assertEqual(manifest["counts"]["representatives"], 3)
        self.assertEqual(
            manifest["features"]["raw_feature_columns"],
            list(clustering.RAW_FEATURE_NAMES),
        )

    def test_rejects_more_clusters_than_empirical_books(self) -> None:
        result = clustering.main([
            "--universe-config", str(self.config),
            "--data-root", str(self.data_root),
            "--output-dir", str(self.root / "invalid"),
            "--clusters", "13",
        ])
        self.assertEqual(result, 1)
        self.assertFalse((self.root / "invalid").exists())

    def test_minimum_size_repair_handles_adversarial_uneven_clusters(self) -> None:
        def observations() -> list[clustering.Observation]:
            # Unconstrained three-means isolates the single 100-valued outlier
            # and the three 10-valued observations from fourteen zeros.
            values = [0.0] * 14 + [10.0] * 3 + [100.0]
            result: list[clustering.Observation] = []
            for index, value in enumerate(values):
                item = clustering.Observation(
                    book_id=index,
                    symbol=f"S{index:02d}",
                    config_data_dir="unused",
                    data_dir=self.root,
                    event_rate_per_second=1.0,
                    mean_spread_ticks=1.0,
                    mean_top_depth=1.0,
                    return_variance=1.0,
                    opening_mid_price_ticks=1.0,
                )
                item.standardized = (value, value, value, value, value)
                result.append(item)
            return result

        unconstrained = observations()
        clustering.cluster_observations(
            unconstrained, clusters=3, max_iterations=100,
            minimum_cluster_size=1,
        )
        unconstrained_counts = sorted(
            sum(item.cluster_id == cluster_id for item in unconstrained)
            for cluster_id in range(3)
        )
        self.assertLess(unconstrained_counts[0], 6)

        first = observations()
        second = observations()
        clustering.cluster_observations(
            first, clusters=3, max_iterations=100,
            minimum_cluster_size=6,
        )
        clustering.cluster_observations(
            second, clusters=3, max_iterations=100,
            minimum_cluster_size=6,
        )
        self.assertEqual(
            [item.cluster_id for item in first],
            [item.cluster_id for item in second],
        )
        self.assertEqual(
            sorted(sum(item.cluster_id == cluster_id for item in first)
                   for cluster_id in range(3)),
            [6, 6, 6],
        )

    def test_parser_keeps_study_defaults(self) -> None:
        parser = clustering.build_parser()
        arguments = parser.parse_args([
            "--universe-config", "universe.csv",
            "--output-dir", "out",
        ])
        self.assertEqual(arguments.clusters, 10)
        self.assertEqual(arguments.validation_per_cluster, 3)
        self.assertEqual(arguments.minimum_cluster_size, 1)
        self.assertEqual(arguments.seed, 20200130)


if __name__ == "__main__":
    unittest.main()
