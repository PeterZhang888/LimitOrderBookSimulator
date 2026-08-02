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

    def test_snapshot_excludes_targets_and_converts_spread_to_bps(self) -> None:
        targets = self.write_csv(
            "targets.csv", ["symbol", "is_shock_target"],
            [
                {"symbol": symbol, "is_shock_target": int(symbol == "S0_0")}
                for symbol in self.symbols
            ],
        )
        summary = self.write_csv(
            "summary.csv",
            ["symbol", "mean_bid_depth", "mean_ask_depth", "mean_spread_ticks"],
            [
                {
                    "symbol": symbol,
                    "mean_bid_depth": 100 + self.clusters[symbol],
                    "mean_ask_depth": 200 + self.clusters[symbol],
                    "mean_spread_ticks": 2 + self.clusters[symbol],
                }
                for symbol in self.symbols
            ],
        )
        row = {
            "shock_targets_csv": str(targets),
            "shock_targets_csv_sha256": ANALYSIS.primary.sha256_file(targets),
            "asset_summary_csv": str(summary),
            "asset_summary_csv_sha256": ANALYSIS.primary.sha256_file(summary),
        }
        selected = ANALYSIS.target_mask(row, set(self.symbols))
        snapshot = ANALYSIS.cluster_snapshot(
            row, self.midpoints, self.clusters, selected
        )
        self.assertEqual(selected, {"S0_0"})
        self.assertEqual(snapshot[0]["symbol_count"], 1.0)
        self.assertEqual(snapshot[1]["symbol_count"], 2.0)
        self.assertAlmostEqual(snapshot[0]["mean_top_depth"], 300.0)
        self.assertAlmostEqual(snapshot[0]["mean_spread_bps"], 2.0)
        self.assertAlmostEqual(snapshot[9]["mean_spread_bps"], 11.0)

    def test_artifact_hash_is_enforced(self) -> None:
        path = self.write_csv("artifact.csv", ["x"], [{"x": 1}])
        with self.assertRaisesRegex(ANALYSIS.ClusterAnalysisError, "SHA-256"):
            ANALYSIS.require_artifact({
                "asset_summary_csv": str(path),
                "asset_summary_csv_sha256": "0" * 64,
            }, "asset_summary_csv")


if __name__ == "__main__":
    unittest.main()
