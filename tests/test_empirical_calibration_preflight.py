#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Regression tests for the allocation-saving empirical preflight."""

from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "empirical_preflight",
    ROOT / "scripts" / "preflight_empirical_calibration_inputs.py",
)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREFLIGHT
SPEC.loader.exec_module(PREFLIGHT)
CALIBRATION = PREFLIGHT.calibration


class EmpiricalCalibrationPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.day = self.root / "itch_20190130"
        self.targets = self.day / "empirical_data" / "itch_20190130_aaa"
        self.targets.mkdir(parents=True)
        with (self.day / "nasdaq_common_plus_qqq_20190130.csv").open(
            "w", newline="", encoding="utf-8",
        ) as output:
            writer = csv.DictWriter(output, fieldnames=("book_id", "symbol"))
            writer.writeheader()
            writer.writerow({"book_id": 0, "symbol": "AAA"})
        values = {
            metric: 1.0 if metric == "two_sided_sample_fraction" else 10.0
            for metric in CALIBRATION.METRICS
            if metric != "background_event_rate"
        }
        scales = {
            metric: 0.005 if metric == "two_sided_sample_fraction" else 2.0
            for metric in values
        }
        for horizon in (300, 3600, None):
            suffix = "" if horizon is None else f"_window_{horizon}s"
            path = self.targets / f"market_targets_aaa_20190130{suffix}.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output, fieldnames=("name", "target", "scale", "weight"),
                )
                writer.writeheader()
                for metric in CALIBRATION.METRICS:
                    if metric == "background_event_rate":
                        target, scale = 6 / 23_400, 1.0e-6
                    else:
                        target, scale = values[metric], scales[metric]
                    writer.writerow({
                        "name": metric,
                        "target": target,
                        "scale": scale,
                        "weight": 1,
                    })
        self.manifest_path = self.targets / "itch_manifest_aaa_20190130.json"
        manifest = {
            "snapshot_interval_ms": 1000,
            "trading_date": "2019-01-30",
            "symbol": "AAA",
            "session_start": "09:30:00",
            "session_end": "16:00:00",
            "aggregation_duration_seconds": 23_400,
            "valid_snapshots": 23_400,
            "invalid_snapshots": 0,
            "distribution_observation_counts": {
                event: 1 for event in CALIBRATION.BACKGROUND_EVENT_NAMES
            },
            "market_values": values,
            "market_target_scales": scales,
            "market_target_windows": {},
        }
        for horizon in (300, 3600):
            manifest["market_target_windows"][str(horizon)] = {
                "file": f"market_targets_aaa_20190130_window_{horizon}s.csv",
                "duration_seconds": horizon,
                "observations": horizon,
                "valid_snapshots": horizon,
                "invalid_snapshots": 0,
                "values": values,
                "scales": scales,
            }
        self.manifest_path.write_text(
            json.dumps(manifest), encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_bundle_passes(self) -> None:
        report = PREFLIGHT.audit(
            data_root=self.root,
            symbols=("AAA",),
            dates=("2019-01-30",),
            expected_symbol_count=1,
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["checked_symbol_horizon_sets"], 3)

    def test_old_prefix_manifest_uses_exact_full_coverage_proof(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        prefix = manifest["market_target_windows"]["300"]
        prefix.pop("valid_snapshots")
        prefix.pop("invalid_snapshots")
        prefix["values"].pop("two_sided_sample_fraction")
        prefix["scales"].pop("two_sided_sample_fraction")
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report = PREFLIGHT.audit(
            data_root=self.root,
            symbols=("AAA",),
            dates=("2019-01-30",),
            expected_symbol_count=1,
        )
        self.assertEqual(report["status"], "passed")

    def test_old_prefix_manifest_fails_when_full_session_is_not_complete(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["valid_snapshots"] = 23_399
        manifest["invalid_snapshots"] = 1
        prefix = manifest["market_target_windows"]["300"]
        prefix.pop("valid_snapshots")
        prefix.pop("invalid_snapshots")
        prefix["values"].pop("two_sided_sample_fraction")
        prefix["scales"].pop("two_sided_sample_fraction")
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(
            CALIBRATION.CalibrationError,
            "lacks exact prefix valid/invalid counts or two-sided coverage",
        ):
            PREFLIGHT.audit(
                data_root=self.root,
                symbols=("AAA",),
                dates=("2019-01-30",),
                expected_symbol_count=1,
            )


if __name__ == "__main__":
    unittest.main()
