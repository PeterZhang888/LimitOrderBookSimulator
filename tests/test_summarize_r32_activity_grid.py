# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
from __future__ import annotations

import csv
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import summarize_r32_activity_grid as summary  # noqa: E402


class R32GridSummaryTests(unittest.TestCase):
    def make_candidate(
        self, root: pathlib.Path, scale: float, status: str, error: float,
    ) -> pathlib.Path:
        adequacy = root / "full_training_adequacy"
        pilot = adequacy / "directional_pilot"
        diagnostics = pilot / "strict_diagnostics"
        diagnostics.mkdir(parents=True)
        (adequacy / "full_universe_expansion_inputs.json").write_text(json.dumps({
            "training_refinement_warm_start": {
                "coupling_scale": {"global_coupling_scale": scale},
            },
        }))
        (pilot / "pilot_decision.json").write_text(json.dumps({
            "status": status,
            "training_only": True,
            "failures": [] if status == "passed" else ["ACF"],
            "thresholds": {"maximum_acf_absolute_error": {
                "mean": 0.035, "median": 0.04, "p90": 0.055,
            }},
        }))
        if status == "passed":
            (pilot / "directional_pilot_handoff.json").write_text("{}\n")
        with (diagnostics / "absolute_return_acf_distribution.csv").open(
            "w", newline="", encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=("statistic", "absolute_error"))
            writer.writeheader()
            for _day in range(3):
                for statistic in ("mean", "median", "p90"):
                    writer.writerow({"statistic": statistic, "absolute_error": error})
        return root

    def test_passed_candidate_is_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_candidate(pathlib.Path(directory), 0.75, "passed", 0.01)
            record = summary.audit_candidate(0.75, root)
            self.assertTrue(record["eligible_for_full_matrix"])
            self.assertEqual(record["status"], "passed")

    def test_rejected_candidate_is_preserved_but_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_candidate(pathlib.Path(directory), 1.0, "rejected", 0.08)
            record = summary.audit_candidate(1.0, root)
            self.assertFalse(record["eligible_for_full_matrix"])
            self.assertEqual(record["failure_count"], 1)

    def test_mislabeled_scale_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_candidate(pathlib.Path(directory), 1.0, "passed", 0.01)
            with self.assertRaises(summary.GridAuditError):
                summary.audit_candidate(1.25, root)


if __name__ == "__main__":
    unittest.main()
