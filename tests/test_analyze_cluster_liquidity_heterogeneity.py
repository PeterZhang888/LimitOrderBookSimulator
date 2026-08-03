#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "cluster_heterogeneity", SCRIPT_DIR / "analyze_cluster_liquidity_heterogeneity.py"
)
assert SPEC and SPEC.loader
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


class ClusterHeterogeneityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.symbols = [f"S{cluster}_{copy}" for cluster in range(10) for copy in range(2)]
        self.midpoints = {symbol: 10_000.0 for symbol in self.symbols}
        self.clusters = {
            symbol: cluster
            for cluster in range(10)
            for symbol in (f"S{cluster}_0", f"S{cluster}_1")
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_csv(self, name: str, fields: list[str], rows: list[dict[str, object]]) -> pathlib.Path:
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_post_shock_window_uses_only_declared_horizon(self) -> None:
        metrics = self.write_csv(
            "cluster_metrics.csv",
            [
                "time_seconds", "cluster_id", "non_target_asset_count",
                "mean_top_depth", "mean_spread_bps",
            ],
            [
                {
                    "time_seconds": time,
                    "cluster_id": cluster,
                    "non_target_asset_count": 2,
                    "mean_top_depth": 1000 + 10 * time + cluster,
                    "mean_spread_bps": 20 + time + cluster,
                }
                for time in (9, 11, 12, 13)
                for cluster in range(10)
            ],
        )
        row = {
            "cluster_metrics_csv": str(metrics),
            "cluster_metrics_csv_sha256": ANALYSIS.primary.sha256_file(metrics),
        }
        snapshot = ANALYSIS.cluster_post_shock_window(
            row, shock_time=10, horizon=2,
        )
        self.assertEqual(snapshot[0]["symbol_count"], 2.0)
        self.assertAlmostEqual(snapshot[0]["mean_top_depth"], 1115.0)
        self.assertAlmostEqual(snapshot[0]["mean_spread_bps"], 31.5)
        self.assertAlmostEqual(snapshot[9]["mean_spread_bps"], 40.5)

    def test_artifact_hash_is_enforced(self) -> None:
        path = self.write_csv("artifact.csv", ["x"], [{"x": 1}])
        with self.assertRaisesRegex(ANALYSIS.ClusterAnalysisError, "SHA-256"):
            ANALYSIS.require_artifact({
                "asset_summary_csv": str(path),
                "asset_summary_csv_sha256": "0" * 64,
            }, "asset_summary_csv")


if __name__ == "__main__":
    unittest.main()
