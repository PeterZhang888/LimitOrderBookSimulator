#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Regression tests for persistent failure diagnostics and R9 packaging."""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "scripts" / "summarize_calibration_diagnostics.py"
PACKAGER = ROOT / "scripts" / "package_calibration_result.sh"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CalibrationDiagnosticSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name) / "calibration"
        self.candidate_dir = (
            self.root / "global_local_flow_calibration" / "stage2_refinement"
            / "candidate_057"
        )
        self.candidate_dir.mkdir(parents=True)
        self.candidate_path = self.candidate_dir / "candidate_evaluation.json"
        run_1729 = str(
            self.candidate_dir / "day_20190130" / "seed_1729"
            / "fragmented_asset_summary.csv"
        )
        run_7919 = str(
            self.candidate_dir / "day_20190130" / "seed_7919"
            / "fragmented_asset_summary.csv"
        )
        adequacy = {
            "passed": False,
            "thresholds": {
                "maximum_asset_event_ratio": 0.05,
                "maximum_asset_quantity_ratio": 0.05,
                "maximum_run_event_ratio": 0.01,
                "maximum_run_quantity_ratio": 0.01,
            },
            "runs": [
                {
                    "summary_path": run_1729,
                    "assets": [
                        {
                            "symbol": "AAA",
                            "boundary_truncation_events": 8,
                            "background_event_count": 100,
                            "boundary_event_ratio": 0.08,
                            "boundary_truncated_quantity": 2,
                            "background_removal_requested_quantity": 100,
                            "boundary_quantity_ratio": 0.02,
                        },
                        {
                            "symbol": "BBB",
                            "boundary_truncation_events": 0,
                            "background_event_count": 100,
                            "boundary_event_ratio": 0.0,
                            "boundary_truncated_quantity": 0,
                            "background_removal_requested_quantity": 100,
                            "boundary_quantity_ratio": 0.0,
                        },
                    ],
                    "aggregate": {
                        "boundary_truncation_events": 8,
                        "background_event_count": 200,
                        "boundary_event_ratio": 0.04,
                        "boundary_truncated_quantity": 2,
                        "background_removal_requested_quantity": 200,
                        "boundary_quantity_ratio": 0.01,
                    },
                },
                {
                    "summary_path": run_7919,
                    "assets": [
                        {
                            "symbol": "AAA",
                            "boundary_truncation_events": 1,
                            "background_event_count": 100,
                            "boundary_event_ratio": 0.01,
                            "boundary_truncated_quantity": 7,
                            "background_removal_requested_quantity": 100,
                            "boundary_quantity_ratio": 0.07,
                        }
                    ],
                    "aggregate": {
                        "boundary_truncation_events": 1,
                        "background_event_count": 100,
                        "boundary_event_ratio": 0.01,
                        "boundary_truncated_quantity": 7,
                        "background_removal_requested_quantity": 100,
                        "boundary_quantity_ratio": 0.07,
                    },
                },
            ],
            "failures": [
                {
                    "scope": "asset_seed", "summary_path": run_1729,
                    "symbol": "AAA", "metric": "boundary_event_ratio",
                    "numerator": 8, "denominator": 100, "ratio": 0.08,
                    "maximum": 0.05,
                },
                {
                    "scope": "run_aggregate", "summary_path": run_1729,
                    "metric": "boundary_event_ratio", "numerator": 8,
                    "denominator": 200, "ratio": 0.04, "maximum": 0.01,
                },
                {
                    "scope": "asset_seed", "summary_path": run_7919,
                    "symbol": "AAA", "metric": "boundary_quantity_ratio",
                    "numerator": 7, "denominator": 100, "ratio": 0.07,
                    "maximum": 0.05,
                },
                {
                    "scope": "run_aggregate", "summary_path": run_7919,
                    "metric": "boundary_quantity_ratio", "numerator": 7,
                    "denominator": 100, "ratio": 0.07, "maximum": 0.01,
                },
            ],
        }
        candidate = {
            "schema_version": 1,
            "artifact_role": "calibration_candidate_evaluation",
            "block": "local_flow",
            "stage": "stage2_refinement",
            "cluster_id": None,
            "candidate_index": 57,
            "candidate": {
                "hawkes_activity_scale": 0.3,
                "local_mm_enabled": True,
                "local_mm_interval_ms": 1000.0,
                "local_mm_quantity_multiplier": 1.0,
                "local_mm_improvement_probability": 0.1,
                "label": "candidate-57",
            },
            "eligibility": {
                "eligible": False,
                "predicates": {
                    "finite_selection_score": True,
                    "finite_fit_wsmrmse": True,
                    "two_sided_integrity_passed": True,
                    "finite_boundary_adequacy_passed": False,
                    "error_free": True,
                },
                "errors": [],
            },
            "evaluation": {
                "fit_wsmrmse": 1.25,
                "selection_score": 1.5,
                "errors": [],
                "two_sided_integrity_failures": [],
                "training_day_evaluations": [
                    {
                        "date": "2019-01-30",
                        "evaluation": {
                            "finite_boundary_adequacy": adequacy,
                        },
                    }
                ],
            },
        }
        self.candidate_path.write_text(
            json.dumps(candidate, indent=2), encoding="utf-8",
        )
        self.reference = {
            "block": "local_flow", "stage": "stage2_refinement",
            "cluster_id": None, "candidate_index": 57,
            "candidate_label": "candidate-57", "eligible": False,
            "failed_predicates": ["finite_boundary_adequacy_passed"],
            "path": str(self.candidate_path),
            "sha256": sha256(self.candidate_path),
        }
        checkpoint_path = self.candidate_dir.parent / "stage_checkpoint.json"
        checkpoint_path.write_text(json.dumps({
            "schema_version": 1,
            "artifact_role": "calibration_stage_checkpoint",
            "status": "failed_no_eligible_candidates",
            "block": "local_flow", "stage": "stage2_refinement",
            "cluster_id": None,
            "observed_counts": {
                "evaluated_candidates": 1, "eligible_candidates": 0,
                "promoted_candidates": 0,
                "configured_ranked_survivor_count": 3,
            },
            "candidate_evaluations": [self.reference],
        }, indent=2), encoding="utf-8")
        failure_event = {
            "kind": "calibration_failure", "exception_type": "CalibrationError",
            "message": "all stage2 candidates failed",
        }
        progress_payload = {
            "schema_version": 1,
            "artifact_role": "calibration_progress_checkpoint",
            "status": "failed", "event_count": 2,
            "events": [
                {"kind": "candidate_evaluation", **self.reference}, failure_event,
            ],
            "last_event": failure_event,
        }
        progress_path = self.root / "calibration_progress.json"
        progress_path.write_text(
            json.dumps(progress_payload, indent=2), encoding="utf-8",
        )
        (self.root / "calibration_failure.json").write_text(json.dumps({
            "schema_version": 1, "artifact_role": "calibration_failure",
            "status": "failed", "exception_type": "CalibrationError",
            "message": "all stage2 candidates failed",
            "progress_checkpoint": {
                "path": str(progress_path), "sha256": sha256(progress_path),
                "snapshot": progress_payload,
            },
        }, indent=2), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_emits_candidate_day_seed_and_worst_symbol_tables(self) -> None:
        output = self.root.parent / "diagnostic_summary"
        completed = subprocess.run([
            sys.executable, str(SUMMARIZER),
            "--calibration-root", str(self.root),
            "--output-dir", str(output),
        ], check=True, text=True, stdout=subprocess.PIPE)
        result = json.loads(completed.stdout)
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["boundary_failure_count"], 4)

        with (output / "candidate_summary.csv").open(newline="") as source:
            candidates = list(csv.DictReader(source))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["eligible"], "False")
        self.assertEqual(candidates[0]["asset_boundary_failure_count"], "2")
        self.assertEqual(candidates[0]["run_boundary_failure_count"], "2")
        self.assertEqual(candidates[0]["failing_seeds"], "1729;7919")

        with (output / "day_seed_boundary.csv").open(newline="") as source:
            runs = list(csv.DictReader(source))
        self.assertEqual([row["seed"] for row in runs], ["1729", "7919"])
        self.assertEqual(runs[0]["max_asset_event_symbol"], "AAA")
        self.assertEqual(runs[1]["max_asset_quantity_symbol"], "AAA")

        summary = json.loads((output / "diagnostic_summary.json").read_text())
        self.assertEqual(summary["integrity"]["verified_hashes"], 1)
        self.assertTrue(
            summary["integrity"][
                "terminal_failure_progress_checkpoint_verified"
            ]
        )
        self.assertEqual(
            summary["counts"]["failed_predicates"],
            {"finite_boundary_adequacy_passed": 1},
        )
        self.assertEqual(
            summary["counts"]["boundary_failures_by_scope"],
            {"asset_seed": 2, "run_aggregate": 2},
        )
        event_worst = summary["worst_symbols_by_metric"]["boundary_event_ratio"]
        self.assertEqual(event_worst[0]["symbol"], "AAA")
        self.assertAlmostEqual(event_worst[0]["max_ratio"], 0.08)

    def test_detects_candidate_tampering(self) -> None:
        with self.candidate_path.open("a", encoding="utf-8") as output:
            output.write("\n")
        completed = subprocess.run([
            sys.executable, str(SUMMARIZER),
            "--calibration-root", str(self.root),
            "--output-dir", str(self.root.parent / "diagnostic_summary"),
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("SHA-256 mismatch", completed.stderr)


class FailedCalibrationPackagingTest(unittest.TestCase):
    def test_packages_failed_result_and_logs_without_success_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            project = base / "project"
            result = project / "results" / "seagull" / "cluster_value_calibration_45249"
            logs = project / "slurm"
            output = base / "packages"
            result.mkdir(parents=True)
            logs.mkdir(parents=True)
            output.mkdir()
            (result / "calibration_failure.json").write_text(
                '{"status":"failed"}\n', encoding="utf-8",
            )
            (result / "diagnostic_summary.json").write_text(
                '{"artifact_role":"diagnostic"}\n', encoding="utf-8",
            )
            (logs / "lob-cluster-cal-45249.out").write_text("stdout\n")
            (logs / "lob-cluster-cal-45249.err").write_text("stderr\n")
            subprocess.run([
                "bash", str(PACKAGER), str(project), "45249", str(output),
            ], check=True, text=True, stdout=subprocess.PIPE)
            archive = output / "calibration_45249_r9_complete.tar.gz"
            self.assertTrue(archive.is_file())
            with tarfile.open(archive, "r:gz") as package:
                names = set(package.getnames())
            self.assertIn(
                "results/seagull/cluster_value_calibration_45249/"
                "calibration_failure.json", names,
            )
            self.assertIn("slurm/lob-cluster-cal-45249.out", names)
            self.assertIn("slurm/lob-cluster-cal-45249.err", names)
            checksum = archive.with_suffix(archive.suffix + ".sha256")
            self.assertTrue(checksum.is_file())
            self.assertIn(archive.name, checksum.read_text())


if __name__ == "__main__":
    unittest.main()
