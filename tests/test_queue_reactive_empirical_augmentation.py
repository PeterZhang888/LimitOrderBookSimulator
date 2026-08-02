#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Tests for lossless queue-reactive empirical-bundle augmentation."""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


PROJECT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

import apply_queue_reactive_empirical_augmentation as apply_module  # noqa: E402
import build_queue_reactive_empirical_augmentation as build_module  # noqa: E402
import verify_queue_reactive_empirical_bundle as verify_module  # noqa: E402


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class StateTargetConstructionTest(unittest.TestCase):
    def test_state_targets_are_exactly_bound_to_fixed_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            pooled = root / "pooled.csv"
            with pooled.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=[
                    "symbol", "target_mean_bid_depth", "target_mean_ask_depth",
                ])
                writer.writeheader()
                writer.writerows([
                    {
                        "symbol": "AAPL",
                        "target_mean_bid_depth": "20",
                        "target_mean_ask_depth": "30",
                    },
                    {
                        "symbol": "QQQ",
                        "target_mean_bid_depth": "10",
                        "target_mean_ask_depth": "15",
                    },
                ])
            target = root / "targets.csv"
            record = build_module.build_state_targets(
                pooled, ["QQQ", "AAPL"], target,
            )
            self.assertEqual(record["symbol_count"], 2)
            self.assertEqual(record["sha256"], build_module.sha256_file(target))
            with target.open(newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([row["symbol"] for row in rows], ["QQQ", "AAPL"])

    def test_state_target_universe_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            pooled = root / "pooled.csv"
            pooled.write_text(
                "symbol,target_mean_bid_depth,target_mean_ask_depth\n"
                "QQQ,10,15\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                build_module.AugmentationError, "do not equal",
            ):
                build_module.build_state_targets(
                    pooled, ["QQQ", "AAPL"], root / "targets.csv",
                )


class AugmentationApplicationTest(unittest.TestCase):
    def fixture(
        self, root: pathlib.Path, *, mismatch_counts: bool = False,
        off_grid_mark: bool = False,
    ) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
        baseline = root / "baseline"
        augmentation = root / "augmentation"
        output = root / "output"
        relative = pathlib.Path(
            "itch_20190130/empirical_data/itch_20190130_qqq"
        )
        baseline_dir = baseline / relative
        augmentation_dir = augmentation / relative
        baseline_dir.mkdir(parents=True)
        augmentation_dir.mkdir(parents=True)
        legacy_counts = {
            "limit_buy": 2,
            "limit_sell": 3,
            "market_buy": 4,
            "market_sell": 5,
            "cancel_bid": 6,
            "cancel_ask": 7,
        }
        legacy_manifest = {
            "trading_date": "2019-01-30",
            "symbol": "QQQ",
            "input_sha256": "source-hash",
            "valid_snapshots": 23400,
            "invalid_snapshots": 0,
            "distribution_observation_counts": legacy_counts,
            "placement_counts": {"inside_spread_limit_orders": 3},
            "market_target_windows": {
                "300": {
                    "duration_seconds": 300,
                    "observations": 300,
                    "file": "market_targets_qqq_20190130_window_300s.csv",
                    "values": {"mean_spread_ticks": 1.5},
                    "scales": {"mean_spread_ticks": 0.2},
                },
            },
        }
        write_json(
            baseline_dir / "itch_manifest_qqq_20190130.json",
            legacy_manifest,
        )
        (baseline_dir / "legacy.txt").write_text("unchanged\n")
        for filename in build_module.REQUIRED_QUEUE_FILES:
            path = augmentation_dir / filename
            if filename == "limit_buy_improvement_distribution.txt":
                path.write_text(
                    "improvement_ticks,improvement_price_units,count\n"
                    "1,100,1\n"
                    + ("0.01,1,1\n" if off_grid_mark else "")
                )
            elif filename == "limit_sell_improvement_distribution.txt":
                path.write_text(
                    "improvement_ticks,improvement_price_units,count\n1,100,2\n"
                )
            else:
                path.write_text("column\nvalue\n")
        queue_counts = dict(legacy_counts)
        if mismatch_counts:
            queue_counts["limit_buy"] += 1
        block = {
            "schema_version": 2,
            "training_only": True,
            "queue_policy_estimation_ready": True,
            "event_count_conservation": {
                "totals_equal": True,
                "equals_legacy_quantity_observation_counts": True,
                "by_event_type": queue_counts,
            },
            "exposure": {"exact_nanosecond_conservation": True},
            "artifacts": {
                filename: filename
                for filename in build_module.REQUIRED_QUEUE_FILES
            },
        }
        source_manifest = augmentation_dir / "source_extractor_manifest.json"
        source_manifest_value = {
            **legacy_manifest,
            "market_target_windows": {
                "300": {
                    **legacy_manifest["market_target_windows"]["300"],
                    "values": {
                        "mean_spread_ticks": 1.5,
                        "two_sided_sample_fraction": 1.0,
                    },
                    "scales": {
                        "mean_spread_ticks": 0.2,
                        "two_sided_sample_fraction": 0.005,
                    },
                },
            },
            "queue_reactive_training_artifacts": block,
        }
        write_json(source_manifest, source_manifest_value)
        sidecar = augmentation_dir / "queue_reactive_training_artifacts.json"
        write_json(sidecar, {
            "schema_version": 1,
            "trading_date": "2019-01-30",
            "symbol": "QQQ",
            "queue_reactive_training_artifacts": block,
            "legacy_distribution_observation_counts": legacy_counts,
            "legacy_placement_counts": legacy_manifest["placement_counts"],
        })
        files = []
        for path in sorted(augmentation_dir.iterdir()):
            files.append({
                "path": str(path),
                "relative_name": path.name,
                "sha256": apply_module.sha256_file(path),
            })
        record = {
            "trading_date": "2019-01-30",
            "symbol": "QQQ",
            "relative_directory": str(relative),
            "files": files,
        }
        target_file = augmentation / "queue_reactive_state_targets.csv"
        target_file.write_text(
            "symbol,target_mean_bid_depth,target_mean_ask_depth\nQQQ,10,10\n"
        )
        records = [record]
        write_json(
            augmentation / "queue_reactive_augmentation_manifest.json",
            {
                "schema_version": 1,
                "status": "complete",
                "role": "queue_reactive_sufficient_statistics_augmentation",
                "legacy_empirical_bundle_modified": False,
                "record_count": 1,
                "records": records,
                "records_sha256": apply_module.sha256_json(records),
            },
        )
        return baseline, augmentation, output

    def test_application_preserves_baseline_and_merges_only_queue_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            baseline, augmentation, output = self.fixture(
                pathlib.Path(temporary)
            )
            report = apply_module.apply(type("Args", (), {
                "baseline_root": baseline,
                "augmentation_root": augmentation,
                "output_root": output,
                "copy_files": False,
            })())
            self.assertEqual(report["record_count"], 1)
            relative = pathlib.Path(
                "itch_20190130/empirical_data/itch_20190130_qqq"
            )
            baseline_manifest = json.loads(
                (baseline / relative / "itch_manifest_qqq_20190130.json").read_text()
            )
            output_manifest = json.loads(
                (output / relative / "itch_manifest_qqq_20190130.json").read_text()
            )
            self.assertNotIn("queue_reactive_training_artifacts", baseline_manifest)
            self.assertIn("queue_reactive_training_artifacts", output_manifest)
            prefix = output_manifest["market_target_windows"]["300"]
            self.assertEqual(prefix["valid_snapshots"], 300)
            self.assertEqual(prefix["invalid_snapshots"], 0)
            self.assertFalse(
                output_manifest["prefix_snapshot_accounting_recovery"]
                ["existing_empirical_targets_replaced"]
            )
            self.assertTrue(
                (output / "queue_reactive_augmentation_provenance.json").is_file()
            )
            provenance = json.loads(
                (output / "queue_reactive_augmentation_provenance.json").read_text()
            )
            self.assertEqual(provenance["schema_version"], 3)
            self.assertEqual(
                provenance["prefix_snapshot_accounting_recovery_summary"]
                ["records_with_recovered_prefix_counts"],
                1,
            )
            self.assertEqual(
                provenance["state_targets"]["sha256"],
                apply_module.sha256_file(output / "queue_reactive_state_targets.csv"),
            )
            verified = verify_module.verify(type("Args", (), {
                "data_root": output,
                "expected_symbols": 1,
                "expected_date": ["2019-01-30"],
            })())
            self.assertEqual(verified["status"], "passed")
            self.assertEqual(verified["verified_file_count"], 8)
            self.assertEqual(
                (baseline / relative / "legacy.txt").stat().st_ino,
                (output / relative / "legacy.txt").stat().st_ino,
            )

    def test_off_grid_marks_are_preserved_and_explicitly_projected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            baseline, augmentation, output = self.fixture(
                pathlib.Path(temporary), off_grid_mark=True,
            )
            apply_module.apply(type("Args", (), {
                "baseline_root": baseline,
                "augmentation_root": augmentation,
                "output_root": output,
                "copy_files": False,
            })())
            relative = pathlib.Path(
                "itch_20190130/empirical_data/itch_20190130_qqq"
            )
            raw_file = (
                output / relative / "limit_buy_improvement_distribution.txt"
            )
            self.assertIn("0.01,1,1", raw_file.read_text())
            manifest = json.loads(
                (output / relative / "itch_manifest_qqq_20190130.json").read_text()
            )
            audit = manifest["queue_reactive_runtime_compatibility"]
            self.assertEqual(audit["raw_exact_inside_spread_mark_count"], 4)
            self.assertEqual(audit["runtime_compatible_mark_count"], 3)
            self.assertEqual(audit["excluded_off_grid_mark_count"], 1)
            verified = verify_module.verify(type("Args", (), {
                "data_root": output,
                "expected_symbols": 1,
                "expected_date": ["2019-01-30"],
            })())
            self.assertEqual(verified["status"], "passed")

    def test_bundle_verifier_rejects_tampered_queue_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            baseline, augmentation, output = self.fixture(pathlib.Path(temporary))
            apply_module.apply(type("Args", (), {
                "baseline_root": baseline,
                "augmentation_root": augmentation,
                "output_root": output,
                "copy_files": False,
            })())
            artifact = (
                output / "itch_20190130/empirical_data/itch_20190130_qqq"
                / "queue_state_counts.csv"
            )
            artifact.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(
                verify_module.VerificationError, "hash mismatch",
            ):
                verify_module.verify(type("Args", (), {
                    "data_root": output,
                    "expected_symbols": 1,
                    "expected_date": ["2019-01-30"],
                })())

    def test_application_rejects_new_legacy_event_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            baseline, augmentation, output = self.fixture(
                pathlib.Path(temporary), mismatch_counts=True,
            )
            with self.assertRaisesRegex(
                apply_module.ApplicationError, "event counts differ",
            ):
                apply_module.apply(type("Args", (), {
                    "baseline_root": baseline,
                    "augmentation_root": augmentation,
                    "output_root": output,
                    "copy_files": False,
                })())
            self.assertFalse(output.exists())


class CompletedManifestVerificationTest(unittest.TestCase):
    def test_completed_manifest_rehashes_every_retained_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            baseline, augmentation, _ = AugmentationApplicationTest().fixture(root)
            del baseline
            manifest = json.loads(
                (augmentation / "queue_reactive_augmentation_manifest.json").read_text()
            )
            manifest["cohort"] = {
                "symbol_count": 1,
                "canonical_sha256": "cohort",
            }
            manifest["state_targets"] = {"sha256": "targets"}
            manifest["extractor"] = {"sha256": "extractor"}
            original_dates = build_module.EXPECTED_DATES
            try:
                build_module.EXPECTED_DATES = ("2019-01-30",)
                verified = build_module.verify_completed_manifest(
                    augmentation,
                    manifest,
                    symbols=["QQQ"],
                    cohort_sha256="cohort",
                    extractor_sha256="extractor",
                    state_targets_sha256="targets",
                )
                self.assertIs(verified, manifest)
                retained = (
                    augmentation
                    / "itch_20190130/empirical_data/itch_20190130_qqq"
                    / "queue_state_counts.csv"
                )
                retained.write_text("changed\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    build_module.AugmentationError, "missing or changed",
                ):
                    build_module.verify_completed_manifest(
                        augmentation,
                        manifest,
                        symbols=["QQQ"],
                        cohort_sha256="cohort",
                        extractor_sha256="extractor",
                        state_targets_sha256="targets",
                    )
            finally:
                build_module.EXPECTED_DATES = original_dates


if __name__ == "__main__":
    unittest.main()
