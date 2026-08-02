#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import compare_qqq_baseline as comparison  # noqa: E402


class BaselineComparisonTest(unittest.TestCase):
    def test_standardized_rmse(self) -> None:
        simulation = {name: "2" for name in comparison.TARGET_FIELDS}
        targets = {
            name: {"target": 1.0, "scale": 0.5, "weight": 1.0}
            for name in comparison.TARGET_FIELDS
        }
        rows, objective = comparison.compare(simulation, targets)
        self.assertEqual(len(rows), len(comparison.TARGET_FIELDS))
        self.assertAlmostEqual(objective, 2.0)

    def test_rejects_structurally_invalid_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "summary.csv"
            path.write_text(
                "book_id,sample_count,expected_sample_count,structurally_valid\n"
                "0,10,10,0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not structurally valid"):
                comparison.load_simulation_row(path, 0)


if __name__ == "__main__":
    unittest.main()
