from __future__ import annotations

import csv
import importlib.util
import json
import os
import pathlib
import types
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = load_module(
    "global_calibration_certification_verifier",
    SCRIPTS / "verify_global_calibration_certification.py",
)
CALIBRATION = VERIFIER.calibration


class GlobalCertificationVerifierTests(unittest.TestCase):
    def test_profile_digest_is_pinned(self) -> None:
        self.assertEqual(
            CALIBRATION.certification_profile_sha256(),
            VERIFIER.PINNED_PROFILE_SHA256,
        )
        self.assertEqual(
            CALIBRATION.certification_profile()["profile_id"],
            "development_validation_gate",
        )

    def test_verifier_is_bound_by_every_workflow_digest(self) -> None:
        pool = load_module(
            "pool_for_global_verifier_test",
            SCRIPTS / "pool_multiday_empirical_universe.py",
        )
        relatives = (
            "scripts/verify_global_calibration_certification.py",
            "tests/test_global_calibration_certification_verifier.py",
        )
        for relative in relatives:
            self.assertIn(relative, CALIBRATION.WORKFLOW_SEMANTICS_FILES)
            self.assertIn(relative, pool.WORKFLOW_SEMANTICS_FILES)
        case_text = (ROOT / "submit_real_universe_case_study.sh").read_text(
            encoding="utf-8"
        )
        for relative in relatives:
            self.assertIn(f'"{relative}"', case_text)
        self.assertEqual(
            CALIBRATION.workflow_source_semantics_sha256(ROOT),
            pool.workflow_source_semantics_sha256(ROOT),
        )

    def test_seed_matrix_rejects_missing_and_extra_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for seed in VERIFIER.HELDOUT_SEEDS:
                (root / f"seed_{seed}").mkdir()
            VERIFIER.require_exact_seed_directories(
                root, VERIFIER.HELDOUT_SEEDS, "test matrix"
            )
            (root / f"seed_{VERIFIER.HELDOUT_SEEDS[-1]}").rmdir()
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure, "seed directories differ"
            ):
                VERIFIER.require_exact_seed_directories(
                    root, VERIFIER.HELDOUT_SEEDS, "test matrix"
                )
            (root / f"seed_{VERIFIER.HELDOUT_SEEDS[-1]}").mkdir()
            (root / "seed_999").mkdir()
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure, "seed directories differ"
            ):
                VERIFIER.require_exact_seed_directories(
                    root, VERIFIER.HELDOUT_SEEDS, "test matrix"
                )
            (root / "seed_999").rmdir()
            (root / "unexpected").mkdir()
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure, "seed directories differ"
            ):
                VERIFIER.require_exact_seed_directories(
                    root, VERIFIER.HELDOUT_SEEDS, "test matrix"
                )

    def test_control_booleans_must_be_json_booleans(self) -> None:
        local_checkpoint = {
            "selected_global_local_flow": {
                "controls": {
                    "hawkes_activity_scale": 0.3,
                    "local_mm_interval_ms": 1000.0,
                    "local_mm_quantity_multiplier": 1.0,
                    "local_mm_improvement_probability": 0.0,
                    "local_mm_enabled": "false",
                    "label": "malformed",
                }
            }
        }
        with self.assertRaisesRegex(
            VERIFIER.CertificationFailure, "must be a JSON boolean"
        ):
            VERIFIER.local_controls_from_checkpoint(local_checkpoint)

        shared_checkpoint = {
            "selected_global_shared_quote": {
                "candidate": {"enabled": "false", "multiplier": 1.0}
            }
        }
        with self.assertRaisesRegex(
            VERIFIER.CertificationFailure, "must be a JSON boolean"
        ):
            VERIFIER.shared_controls_from_checkpoint(shared_checkpoint)

    def test_shared_quote_disabled_baseline_and_enabled_treatment_contract(self) -> None:
        disabled = {
            "selected_global_shared_quote": {
                "candidate": {"enabled": False, "multiplier": 0.0}
            }
        }
        self.assertEqual(
            VERIFIER.shared_controls_from_checkpoint(disabled), (False, 0.0)
        )
        enabled = {
            "selected_global_shared_quote": {
                "candidate": {"enabled": True, "multiplier": 1.0}
            }
        }
        self.assertEqual(
            VERIFIER.shared_controls_from_checkpoint(enabled), (True, 1.0)
        )
        disabled["selected_global_shared_quote"]["candidate"]["multiplier"] = 1.0
        with self.assertRaisesRegex(
            VERIFIER.CertificationFailure, "disabled.*exactly zero",
        ):
            VERIFIER.shared_controls_from_checkpoint(disabled)
        enabled["selected_global_shared_quote"]["candidate"]["multiplier"] = 0.0
        with self.assertRaisesRegex(
            VERIFIER.CertificationFailure, "enabled.*positive",
        ):
            VERIFIER.shared_controls_from_checkpoint(enabled)

    def test_marketwide_status_schema_matches_generator(self) -> None:
        self.assertEqual(
            VERIFIER.MARKETWIDE_STATUS_SCHEMA_VERSION,
            CALIBRATION.MARKETWIDE_STATUS_SCHEMA_VERSION,
        )
        self.assertEqual(CALIBRATION.MARKETWIDE_STATUS_SCHEMA_VERSION, 2)

    def test_selection_checkpoint_schema_matches_generator(self) -> None:
        self.assertEqual(CALIBRATION.SELECTION_CHECKPOINT_SCHEMA_VERSION, 2)

    def test_producer_cluster_checkpoint_retains_all_stage3_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            selected_rows = []
            evaluations = {}
            expected_per_cluster = (
                len(VERIFIER.TRAINING_DATES) * len(VERIFIER.HELDOUT_SEEDS)
            )
            for cluster_id in range(VERIFIER.CLUSTER_COUNT):
                selected_rows.append({
                    "cluster_id": cluster_id,
                    "cluster_label": f"liquidity_{cluster_id:02d}",
                    "representative_symbols": f"R{cluster_id}",
                    "validation_symbols": f"V{cluster_id}",
                    "enabled": 0,
                    "value_threshold_bps": 0.0,
                    "value_depth_participation": 0.0,
                })
                summary_paths = []
                day_evaluations = []
                for day_index, date in enumerate(VERIFIER.TRAINING_DATES):
                    day_paths = []
                    for seed_index, seed in enumerate(VERIFIER.HELDOUT_SEEDS):
                        path = root / (
                            f"cluster_{cluster_id:02d}_{date}_{seed}.csv"
                        )
                        path.write_text("symbol\nQQQ\n", encoding="utf-8")
                        day_paths.append(str(path))
                    summary_paths.extend(day_paths)
                    day_evaluations.append({
                        "date": date,
                        "summary_paths": day_paths,
                    })
                evaluations[cluster_id] = {
                    "fit_wsmrmse": 1.0,
                    "combined_uncertainty_wsmrmse": 1.0,
                    "selection_score": 1.0,
                    "selection_metric_scores": [],
                    "two_sided_integrity_passed": True,
                    "two_sided_integrity_failures": [],
                    "finite_boundary_adequacy_passed": True,
                    "finite_boundary_adequacy": {},
                    "value_boundary_adequacy_passed": True,
                    "value_boundary_adequacy": {},
                    "seed_count": expected_per_cluster,
                    "seed_wall_seconds": [1.0] * expected_per_cluster,
                    "summary_paths": summary_paths,
                    "errors": {},
                    "moment_estimates": {},
                    "training_day_count": len(VERIFIER.TRAINING_DATES),
                    "aggregation": "test",
                    "training_day_evaluations": day_evaluations,
                }
            records = CALIBRATION.checkpoint_cluster_policy_records(
                selected_rows,
                evaluations,
                expected_summary_count_per_cluster=expected_per_cluster,
            )
            observed = VERIFIER.selection_summary_paths(
                records, "selected_cluster_policies",
            )
            self.assertEqual(len(records), VERIFIER.CLUSTER_COUNT)
            self.assertEqual(
                len(observed),
                VERIFIER.CLUSTER_COUNT * expected_per_cluster,
            )
            self.assertEqual(len(set(observed)), len(observed))

    def test_verifier_rejects_a_different_executing_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            other_root = pathlib.Path(temporary).resolve()
            args = types.SimpleNamespace(project_root=str(other_root))
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure,
                "executing certification verifier is not from --project-root",
            ):
                VERIFIER.verify(args)

    def test_direct_file_and_seed_directory_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure, "symbolic link",
            ):
                VERIFIER.regular_file(link, "linked evidence")

            real_seed = root / "real_seed"
            real_seed.mkdir()
            seed_root = root / "seeds"
            seed_root.mkdir()
            (seed_root / f"seed_{VERIFIER.HELDOUT_SEEDS[0]}").symlink_to(
                real_seed, target_is_directory=True,
            )
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure, "symbolic-link entry",
            ):
                VERIFIER.require_exact_seed_directories(
                    seed_root, (VERIFIER.HELDOUT_SEEDS[0],), "linked matrix",
                )

    def test_duplicate_and_hardlinked_seed_summaries_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            summaries = self.write_summary_set(root)
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure, "distinct asset-summary path",
            ):
                VERIFIER.recompute_evaluation(
                    [summaries[0]] * 5, [1.0] * 5, ("QQQ",),
                    self.exact_targets(),
                )
            summaries[-1].unlink()
            # pathlib.Path.hardlink_to was added after the Python 3.9 runtime
            # used on Seagull. os.link has the same source/destination
            # semantics and keeps this provenance test portable.
            os.link(summaries[0], summaries[-1])
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure, "reuse one asset-summary inode",
            ):
                VERIFIER.recompute_evaluation(
                    summaries, [1.0] * 5, ("QQQ",), self.exact_targets(),
                )

    def test_exact_file_and_ordered_subset_bindings_reject_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            canonical = root / "canonical.csv"
            exact = root / "exact.csv"
            mutated = root / "mutated.csv"
            canonical.write_text("symbol,value\nA,1\nB,2\n", encoding="utf-8")
            exact.write_bytes(canonical.read_bytes())
            mutated.write_text("symbol,value\nA,1\nB,3\n", encoding="utf-8")
            VERIFIER.require_identical_file(exact, canonical, "runtime config")
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure, "not byte-identical",
            ):
                VERIFIER.require_identical_file(
                    mutated, canonical, "runtime config",
                )
            subset = root / "subset.csv"
            subset.write_text("symbol,value\nB,2\n", encoding="utf-8")
            VERIFIER.require_exact_ordered_subset(
                subset, canonical, ("B",), "stratified config",
            )
            subset.write_text("symbol,value\nB,9\n", encoding="utf-8")
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure, "exact ordered subset",
            ):
                VERIFIER.require_exact_ordered_subset(
                    subset, canonical, ("B",), "stratified config",
                )

    def test_stratified_config_uses_canonical_not_cluster_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            canonical = root / "canonical.csv"
            subset = root / "stratified.csv"
            canonical.write_text(
                "book_id,symbol,value\n0,A,1\n1,B,2\n2,C,3\n3,D,4\n",
                encoding="utf-8",
            )
            # The producer chooses validation symbols cluster-by-cluster, but
            # subset_config_rows preserves canonical full-universe row order.
            cluster_order = ("D", "B")
            canonical_order = VERIFIER.canonical_filtered_symbol_order(
                canonical, cluster_order, "producer-shaped stratified config",
            )
            self.assertEqual(canonical_order, ("B", "D"))
            subset.write_text(
                "book_id,symbol,value\n0,B,2\n1,D,4\n", encoding="utf-8",
            )
            VERIFIER.require_exact_ordered_subset(
                subset, canonical, canonical_order,
                "producer-shaped stratified config", renumber_book_id=True,
            )
            self.assertEqual(set(canonical_order), set(cluster_order))

    def test_one_centroid_derives_three_training_representatives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            assignments = root / "assignments.csv"
            validation = root / "validation.csv"
            assignment_fields = (
                "symbol", "cluster_id", "distance_to_centroid",
                "is_representative", "is_validation_sample",
            )
            assignment_rows = []
            validation_rows = []
            symbols = []
            for cluster in range(VERIFIER.CLUSTER_COUNT):
                for member in range(6):
                    symbol = f"C{cluster}S{member}"
                    symbols.append(symbol)
                    assignment_rows.append({
                        "symbol": symbol,
                        "cluster_id": str(cluster),
                        "distance_to_centroid": str(member),
                        "is_representative": "1" if member == 0 else "0",
                        "is_validation_sample": "1" if member >= 3 else "0",
                    })
                    if member >= 3:
                        validation_rows.append({
                            "symbol": symbol, "cluster_id": str(cluster),
                        })
            with assignments.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=assignment_fields)
                writer.writeheader()
                writer.writerows(assignment_rows)
            with validation.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output, fieldnames=("symbol", "cluster_id"),
                )
                writer.writeheader()
                writer.writerows(validation_rows)
            observed_validation, representatives = VERIFIER.verify_cluster_layout(
                assignments=assignments,
                validation_sample=validation,
                symbols=tuple(symbols),
            )
            self.assertEqual(len(observed_validation), 30)
            for cluster in range(VERIFIER.CLUSTER_COUNT):
                self.assertEqual(
                    representatives[cluster],
                    tuple(f"C{cluster}S{member}" for member in range(3)),
                )

    def test_launchers_require_the_standalone_pass_artifact(self) -> None:
        calibration_submit = (ROOT / "submit_cluster_value_agent_calibration.sh").read_text(
            encoding="utf-8"
        )
        case_submit = (ROOT / "submit_real_universe_case_study.sh").read_text(
            encoding="utf-8"
        )
        verifier_path = (
            '${PROJECT_DIR}/scripts/verify_global_calibration_certification.py'
        )
        self.assertIn(f'python3 "{verifier_path}"', calibration_submit)
        self.assertIn(f'python3 "{verifier_path}"', case_submit)
        self.assertIn(
            "independent_global_calibration_certification.json",
            calibration_submit,
        )
        self.assertIn(
            "stored independent certification differs from fresh re-verification",
            case_submit,
        )
        self.assertIn(
            'INPUT_SNAPSHOT_DIR="${RESULT_DIR}/input_snapshot"', case_submit,
        )
        self.assertIn(
            'UNIVERSE_CONFIG="${INPUT_SNAPSHOT_DIR}/universe_config.csv"',
            case_submit,
        )
        self.assertIn(
            'VALUE_AGENT_POLICY_CSV="${INPUT_SNAPSHOT_DIR}/value_agent_policy.csv"',
            case_submit,
        )
        self.assertIn(
            'SHOCK_CLUSTER_CSV="${INPUT_SNAPSHOT_DIR}/shock_clusters.csv"',
            case_submit,
        )
        self.assertIn("certified_sha256", case_submit)
        self.assertIn("original_path", case_submit)
        self.assertIn(
            "independent_global_calibration_certification.json",
            CALIBRATION.TERMINAL_CALIBRATION_ARTIFACT_FILENAMES,
        )

    def test_run_log_rejects_nonzero_and_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            log = root / "run.log"
            log.write_text(
                'command=["/tmp/simulator"]\n'
                "wall_seconds_external=1.0\n"
                "return_code=1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure, "did not complete successfully"
            ):
                VERIFIER.parse_run_log(log)
            log.write_text(
                'command=["/tmp/simulator"]\n'
                "wall_seconds_external=1.0\n"
                "return_code=0\n"
                "TIMEOUT\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure, "did not complete successfully"
            ):
                VERIFIER.parse_run_log(log)

    @staticmethod
    def write_summary(path: pathlib.Path, *, one_sided: bool = False,
                      boundary_failure: bool = False) -> None:
        fields = [
            "symbol", "sample_count", "expected_sample_count",
            "invalid_sample_count", "structurally_valid",
        ]
        for field in (*CALIBRATION.METRICS, *CALIBRATION.BOUNDARY_SUMMARY_FIELDS):
            if field not in fields:
                fields.append(field)
        row = {field: "0" for field in fields}
        row.update({
            "symbol": "QQQ",
            "sample_count": "23399" if one_sided else "23400",
            "expected_sample_count": "23400",
            "invalid_sample_count": "1" if one_sided else "0",
            "structurally_valid": "0" if one_sided else "1",
            "background_event_count": "10000",
            "background_event_rate": str(10000 / 23400),
            "mean_spread_ticks": "2",
            "mean_bid_depth": "100",
            "mean_ask_depth": "100",
            "mid_move_rate": "0.1",
            "return_variance": "1e-8",
            "return_kurtosis": "5",
            "absolute_return_acf1": "0.1",
            "two_sided_sample_fraction": str(23399 / 23400) if one_sided else "1",
            "background_market_requested_quantity": "10000",
            "background_cancel_requested_quantity": "10000",
        })
        if boundary_failure:
            row.update({
                "removal_boundary_truncation_events": "1000",
                "removal_boundary_truncated_quantity": "1000",
                "background_boundary_truncation_events": "1000",
                "background_boundary_truncated_quantity": "1000",
            })
        with path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

    @classmethod
    def write_summary_set(cls, root: pathlib.Path, **kwargs) -> list[pathlib.Path]:
        paths: list[pathlib.Path] = []
        for index in range(5):
            path = root / f"summary_{index}.csv"
            cls.write_summary(path, **kwargs)
            paths.append(path)
        return paths

    @staticmethod
    def exact_targets():
        values = {
            "background_event_rate": 10000 / 23400,
            "mean_spread_ticks": 2.0,
            "mean_bid_depth": 100.0,
            "mean_ask_depth": 100.0,
            "mid_move_rate": 0.1,
            "return_variance": 1.0e-8,
            "return_kurtosis": 5.0,
            "absolute_return_acf1": 0.1,
            "two_sided_sample_fraction": 1.0,
        }
        return {
            "QQQ": {
                metric: CALIBRATION.TargetMoment(target, 1.0, 1.0)
                for metric, target in values.items()
            }
        }

    def test_recomputed_raw_evidence_passes_when_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summaries = self.write_summary_set(pathlib.Path(temporary))
            evaluation = VERIFIER.recompute_evaluation(
                summaries, [1.0] * 5, ("QQQ",), self.exact_targets()
            )
            result = VERIFIER.assert_scope_passes(evaluation, "synthetic exact")
            self.assertTrue(result["empirical_fit"]["passed"])

    def test_stratified_empirical_failure_is_preserved_but_nonblocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summaries = self.write_summary_set(pathlib.Path(temporary))
            targets = self.exact_targets()
            targets["QQQ"]["mean_spread_ticks"] = CALIBRATION.TargetMoment(
                1_000.0, 1.0, 1.0,
            )
            evaluation = VERIFIER.recompute_evaluation(
                summaries, [1.0] * 5, ("QQQ",), targets,
            )
            result = VERIFIER.assert_structural_scope_passes(
                evaluation,
                CALIBRATION.STRATIFIED_EMPIRICAL_FIT_FAILURE_SCOPE,
            )
            self.assertFalse(result["empirical_fit"]["passed"])
            self.assertTrue(result["empirical_fit_failure_reasons"])
            self.assertTrue(all(
                reason.startswith(
                    CALIBRATION.STRATIFIED_EMPIRICAL_FIT_FAILURE_SCOPE
                )
                for reason in result["empirical_fit_failure_reasons"]
            ))

    def test_authoritative_marketwide_empirical_failure_is_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summaries = self.write_summary_set(pathlib.Path(temporary))
            targets = self.exact_targets()
            targets["QQQ"]["mean_spread_ticks"] = CALIBRATION.TargetMoment(
                1_000.0, 1.0, 1.0,
            )
            evaluation = VERIFIER.recompute_evaluation(
                summaries, [1.0] * 5, ("QQQ",), targets,
            )
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure,
                "failed recomputed empirical-fit gate",
            ):
                VERIFIER.assert_scope_passes(
                    evaluation, "market-wide authoritative",
                )

    def test_stratified_structural_failure_remains_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summaries = self.write_summary_set(
                pathlib.Path(temporary), boundary_failure=True,
            )
            evaluation = VERIFIER.recompute_evaluation(
                summaries, [1.0] * 5, ("QQQ",), self.exact_targets(),
            )
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure,
                "failed recomputed structural gates.*finite-boundary adequacy",
            ):
                VERIFIER.assert_structural_scope_passes(
                    evaluation, "stratified structural",
                )

    def test_claimed_pass_cannot_hide_one_sided_raw_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summaries = self.write_summary_set(
                pathlib.Path(temporary), one_sided=True,
            )
            evaluation = VERIFIER.recompute_evaluation(
                summaries, [1.0] * 5, ("QQQ",), self.exact_targets()
            )
            # A status JSON could claim every boolean is true; the verifier
            # reaches this gate from raw CSVs and rejects it regardless.
            claimed_status = {"passed": True, "execution_integrity_passed": True}
            self.assertTrue(claimed_status["passed"])
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure,
                "fixed-clock/two-sided execution integrity",
            ):
                VERIFIER.assert_scope_passes(evaluation, "adversarial one-sided")

    def test_claimed_pass_cannot_hide_boundary_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summaries = self.write_summary_set(
                pathlib.Path(temporary), boundary_failure=True,
            )
            evaluation = VERIFIER.recompute_evaluation(
                summaries, [1.0] * 5, ("QQQ",), self.exact_targets()
            )
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure,
                "background finite-boundary adequacy",
            ):
                VERIFIER.assert_scope_passes(evaluation, "adversarial boundary")

    def test_missing_handoff_cannot_be_replaced_by_preliminary_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "preliminary_calibration_result.json").write_text(
                json.dumps({"artifact_role": "preliminary_not_certified"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                VERIFIER.CertificationFailure, "handoff is missing"
            ):
                VERIFIER.json_object(
                    root / "calibration_handoff.json",
                    "certified calibration handoff",
                )


if __name__ == "__main__":
    unittest.main()
