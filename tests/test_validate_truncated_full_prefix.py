#!/usr/bin/env python3
"""Tests for exact shortened/full stochastic-prefix certification."""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_truncated_full_prefix.py"


class PrefixValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_metrics(self, name: str, values: list[float]) -> pathlib.Path:
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=["time_seconds", "depth"])
            writer.writeheader()
            for index, value in enumerate(values):
                writer.writerow({"time_seconds": index, "depth": value})
        return path

    def write_raw(
        self, name: str, metrics: pathlib.Path, duration: int,
    ) -> pathlib.Path:
        path = self.root / name
        fields = [
            "risk_limit_per_asset", "seed", "shared_mm_mode", "shock_mode",
            "metrics_csv", "metrics_csv_sha256",
            "requested_duration_seconds",
            "requested_stochastic_baseline_normalization_seconds",
        ]
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "risk_limit_per_asset": 800,
                "seed": 20200130,
                "shared_mm_mode": "global",
                "shock_mode": "on",
                "metrics_csv": metrics,
                "metrics_csv_sha256": hashlib.sha256(
                    metrics.read_bytes()
                ).hexdigest(),
                "requested_duration_seconds": duration,
                "requested_stochastic_baseline_normalization_seconds": 23400,
            })
        return path

    def run_case(self, short_values: list[float], full_values: list[float]):
        short = self.write_raw(
            "short_raw.csv", self.write_metrics("short.csv", short_values),
            11702,
        )
        full = self.write_raw(
            "full_raw.csv", self.write_metrics("full.csv", full_values),
            23400,
        )
        output = self.root / "certificate.json"
        completed = subprocess.run(
            [
                "python3", str(SCRIPT), "--short-raw", str(short),
                "--full-raw", str(full), "--output", str(output),
            ],
            text=True, capture_output=True, check=False,
        )
        return completed, output

    def test_accepts_exact_prefix(self) -> None:
        completed, output = self.run_case([10.0, 9.0], [10.0, 9.0, 8.0])
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "exact_truncated_full_prefix_passed")
        self.assertEqual(payload["comparisons"][0]["matched_observations"], 2)

    def test_rejects_horizon_dependent_prefix(self) -> None:
        completed, output = self.run_case([10.0, 9.1], [10.0, 9.0, 8.0])
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(output.exists())
        self.assertIn("prefix differs", completed.stderr)


if __name__ == "__main__":
    unittest.main()
