#!/usr/bin/env python3
"""Regression tests for the compact-input workstation driver."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_local_multiday_calibration as workflow  # noqa: E402


class LocalMultidayInputTest(unittest.TestCase):
    def test_compact_bundle_does_not_require_launcher_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config = root / "nasdaq_common_plus_qqq_20190130.csv"
            config.write_text("book_id,symbol\n0,QQQ\n", encoding="utf-8")
            targets = root / "empirical_data"
            targets.mkdir()

            observed_config, observed_targets = workflow.completed_universe(
                root, "2019-01-30"
            )

            self.assertEqual(observed_config, config.resolve())
            self.assertEqual(observed_targets, targets.resolve())
            self.assertFalse((root / "calibration_job_metadata.json").exists())

    def test_compact_bundle_still_requires_empirical_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with self.assertRaisesRegex(
                FileNotFoundError, "nasdaq_common_plus_qqq_20190130.csv"
            ):
                workflow.completed_universe(root, "2019-01-30")


if __name__ == "__main__":
    unittest.main()
