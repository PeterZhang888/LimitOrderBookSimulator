#!/usr/bin/env python3
"""Static contract for analysis-only recovery of completed financial paths."""

from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "submit_queue_reactive_case_analysis.sh"


class QueueCaseAnalysisSubmissionContractTest(unittest.TestCase):
    def test_reuses_completed_paths_without_mpi_or_simulation_runner(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for artifact in (
            "financial_global_raw.csv",
            "financial_uncoupled_raw.csv",
            "financial_shared_off_raw.csv",
            "mechanism_preflight_raw.csv",
            "portable_queue_reactive_case.json",
        ):
            self.assertIn(artifact, source)
        self.assertIn("analyze_fragmented_shared_liquidity_case.py", source)
        self.assertIn("analyze_cluster_liquidity_heterogeneity.py", source)
        self.assertIn("validate_truncated_full_prefix.py", source)
        self.assertNotIn("run_fragmented_mpi_experiments.py", source)
        self.assertNotIn("mpirun", source)


if __name__ == "__main__":
    unittest.main()
