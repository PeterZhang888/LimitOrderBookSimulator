#!/usr/bin/env python3
"""Regression guards for safety-critical case-study submission contracts."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "submit_real_universe_case_study.sh").read_text(encoding="utf-8")
SPEC = importlib.util.spec_from_file_location(
    "case_contract_calibration", ROOT / "scripts" / "calibrate_cluster_value_agents.py",
)
assert SPEC is not None and SPEC.loader is not None
CALIBRATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CALIBRATION
SPEC.loader.exec_module(CALIBRATION)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def handoff_loader_source() -> str:
    anchor = SCRIPT.index("handoff_path = pathlib.Path(sys.argv[1])")
    marker = "<<'PY'\n"
    start = SCRIPT.rfind(marker, 0, anchor) + len(marker)
    end = SCRIPT.index("\nPY\n)", anchor)
    return SCRIPT[start:end]


def universe_input_builder_source() -> str:
    """Extract the post-handoff universe provenance program from the launcher."""
    anchor = SCRIPT.index("config_path = pathlib.Path(sys.argv[1])")
    marker = "<<'PY'\n"
    start = SCRIPT.rfind(marker, 0, anchor) + len(marker)
    end = SCRIPT.index("\nPY\n", anchor)
    return SCRIPT[start:end]


def one_symbol_cohort_fixture_source(source: str) -> str:
    """Stub only materialized-cohort reads in this one-symbol test fixture.

    Production remains bound to the immutable 1,480-symbol file.  This test
    suite intentionally materializes only QQQ so that it can exercise the
    much larger downstream handoff contract without creating 1,480 empirical
    directories and five complete target bundles for every test method.
    Persisted provenance still carries the real immutable cohort identity;
    only the fixture's CSV readers are replaced.
    """
    base_identity = CALIBRATION.cohort.validate_symbols(
        CALIBRATION.cohort.load_required_symbols(ROOT),
        label="test fixture canonical cohort",
        project_root=ROOT,
    )
    injection = (
        "\n_FIXTURE_COHORT_IDENTITY = "
        + repr(base_identity)
        + "\n"
        + "def _fixture_validate_cohort(*args, **kwargs):\n"
        + "    return dict(_FIXTURE_COHORT_IDENTITY)\n"
        + "def _fixture_required_symbols(*args, **kwargs):\n"
        + "    return ('QQQ',)\n"
        + "def _fixture_symbols_from_csv(*args, **kwargs):\n"
        + "    return ('QQQ',)\n"
        + "cohort_contract.validate_symbols = _fixture_validate_cohort\n"
        + "cohort_contract.validate_csv = _fixture_validate_cohort\n"
        + "cohort_contract.load_required_symbols = _fixture_required_symbols\n"
        + "cohort_contract.symbols_from_csv = _fixture_symbols_from_csv\n"
    )
    marker = "import certification_cohort as cohort_contract\n"
    if marker not in source:
        raise AssertionError("case-study embedded program lacks cohort import")
    return source.replace(marker, marker + injection, 1)


class RealUniverseCaseStudySubmissionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        # Canonicalise macOS' /var -> /private/var alias so persisted boundary
        # diagnostics and the verifier's resolved evidence paths are identical.
        self.root = pathlib.Path(self.temporary.name).resolve()
        self.data_dir = self.root / "empirical" / "qqq"
        self.data_dir.mkdir(parents=True)
        self.rates = self.data_dir / "rates.csv"
        rate_events = (
            "limit_buy", "limit_sell", "market_buy", "market_sell",
            "cancel_bid", "cancel_ask",
        )
        rate_targets = [float(index + 2) for index in range(6)]
        with self.rates.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=(
                "event_type", "observed_rate_per_second",
                "stationary_target_rate", "configured_mu",
                "stationary_reconstructed_rate",
            ))
            writer.writeheader()
            for index, event in enumerate(rate_events):
                endogenous = 0.20 * rate_targets[index] / 10.0
                writer.writerow({
                    "event_type": event,
                    "observed_rate_per_second": index + 1,
                    "stationary_target_rate": rate_targets[index],
                    "configured_mu": (
                        rate_targets[index] - endogenous
                    ) / 0.3,
                    "stationary_reconstructed_rate": rate_targets[index],
                })
        for filename in CALIBRATION.SIMULATOR_EMPIRICAL_INPUT_FILENAMES:
            (self.data_dir / filename).write_text("value,count\n1,1\n", encoding="utf-8")
        self.manifest = self.data_dir / "itch_manifest_qqq_pooled.json"
        self.manifest.write_text(
            '{"schema_version":1}\n', encoding="utf-8",
        )
        self.fields = list(CALIBRATION.RUNTIME_CONFIG_FIELDS)
        self.training = self.root / "training.csv"
        self.heldout = self.root / "heldout.csv"
        self.write_config(self.training, fundamental="10000", beta="1")
        self.write_config(self.heldout, fundamental="10100", beta="1")
        self.policy = self.root / "policy.csv"
        with self.policy.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=CALIBRATION.POLICY_FIELDS)
            writer.writeheader()
            writer.writerow({
                "symbol": "QQQ", "enabled": "1",
                "value_threshold_bps": "8",
                "value_depth_participation": "0.25",
                "cluster_id": "0", "cluster_label": "liquidity_00",
                "policy_source": "selected_block_coordinate_cluster_wmm",
            })
        self.clusters = self.root / "clusters.csv"
        self.clusters.write_text(
            "symbol,cluster_id,is_representative\nQQQ,0,1\n", encoding="utf-8"
        )
        self.validation_sample = self.root / "validation.csv"
        self.validation_sample.write_text(
            "symbol,cluster_id\n", encoding="utf-8"
        )
        profile = CALIBRATION.certification_profile()
        profile_hash = CALIBRATION.certification_profile_sha256()
        cohort_base = CALIBRATION.cohort.validate_symbols(
            CALIBRATION.cohort.load_required_symbols(ROOT),
            label="test fixture canonical cohort",
            project_root=ROOT,
        )
        cohort_artifact_checks = {
            "pooled_training_universe": dict(cohort_base),
            "training_days": {
                day: dict(cohort_base)
                for day in profile["required_training_dates"]
            },
            "heldout_opening_universe": dict(cohort_base),
            "cluster_assignments": dict(cohort_base),
            "full_universe_policy": dict(cohort_base),
            "frozen_heldout_runtime_universe": dict(cohort_base),
        }
        self.cohort_identity = {
            "schema_version": 1,
            **cohort_base,
            "artifact_checks": cohort_artifact_checks,
        }
        self.pool_cohort_identity = {
            **cohort_base,
            "original_intersection_symbol_count": 1509,
            "fixed_price_grid_excluded_symbol_count": 29,
            "artifact_checks": {
                "pooled_training_universe": dict(cohort_base),
                "heldout_common": dict(cohort_base),
                "training_days": {
                    day: dict(cohort_base)
                    for day in profile["required_training_dates"]
                },
            },
        }
        source_hash = CALIBRATION.simulator_source_semantics_sha256(ROOT)
        workflow_hash = CALIBRATION.workflow_source_semantics_sha256(ROOT)
        bundle_hash = CALIBRATION.empirical_input_bundle_sha256(self.heldout)
        binary_hash = hashlib.sha256(b"test-calibration-binary").hexdigest()
        self.calibration_binary = self.root / "calibration_binary"
        self.calibration_binary.write_bytes(b"test-calibration-binary")
        self.calibration_binary.chmod(0o755)
        self.fake_bin = self.root / "fake_bin"
        self.fake_bin.mkdir()
        self.fake_mpi_lib = self.root / "fake_mpi_lib"
        self.fake_mpi_lib.mkdir()
        fake_mpicxx = self.fake_bin / "mpicxx"
        fake_mpicxx.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--showme:libdirs\" ]; then\n"
            f"  echo '{self.fake_mpi_lib}'\n"
            "else\n"
            "  echo 'test mpicxx'\n"
            "fi\n",
            encoding="utf-8",
        )
        fake_ninja = self.fake_bin / "ninja"
        fake_ninja.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_mpicxx.chmod(0o755)
        fake_ninja.chmod(0o755)
        self.cluster_manifest_path = self.root / "cluster_manifest.json"
        cluster_manifest_payload = {
            "schema_version": 1,
            "inputs": {
                "universe_config_sha256": sha256(self.training),
            },
            "features": {
                "raw_feature_columns": [
                    "event_rate_per_second", "mean_spread_ticks",
                    "mean_top_depth", "return_variance",
                    "opening_mid_price_ticks",
                ],
            },
            "clustering": {
                "algorithm": (
                    "deterministic_farthest_first_lloyd_kmeans_"
                    "with_minimum_size_repair"
                ),
                "cluster_count": 10,
                "minimum_cluster_size": 6,
                "seed": 20200130,
                "requested_validation_per_cluster": 3,
            },
            "counts": {"clusters": 10},
            "artifacts": {
                "cluster_assignments_csv": {
                    "path": str(self.clusters), "sha256": sha256(self.clusters),
                },
                "validation_sample_csv": {
                    "path": str(self.validation_sample),
                    "sha256": sha256(self.validation_sample),
                },
            },
        }
        self.cluster_manifest_path.write_text(
            json.dumps(cluster_manifest_payload), encoding="utf-8"
        )
        cluster_manifest = {
            "path": str(self.cluster_manifest_path),
            "sha256": sha256(self.cluster_manifest_path),
            **cluster_manifest_payload,
        }
        self.build_provenance_path = self.root / "build_provenance.json"
        build_payload = {
            "schema_version": 1,
            "artifact_role": "calibration_build_provenance",
            "cmake_build_type": "Release",
            "binary": str(self.calibration_binary),
            "binary_sha256": binary_hash,
            "simulator_source_semantics_sha256": source_hash,
            "workflow_source_semantics_sha256": workflow_hash,
            "compiler": "test compiler",
            "mpi": "test MPI",
            "deterministic_build_contract": {
                "version": "seagull_release_mpi_v1",
                "path": str(
                    ROOT / "scripts" / "seagull_deterministic_build.sh"
                ),
                "sha256": sha256(
                    ROOT / "scripts" / "seagull_deterministic_build.sh"
                ),
                "compiler_path": str(fake_mpicxx.resolve()),
                "ninja_path": str(fake_ninja.resolve()),
                "mpi_lib_dir": str(self.fake_mpi_lib.resolve()),
                "source_date_epoch": "1577836800",
                "cmake_build_type": "Release",
                "lob_require_mpi": True,
                "lob_build_tests": True,
                "interprocedural_optimization": False,
            },
        }
        self.build_provenance_path.write_text(
            json.dumps(build_payload), encoding="utf-8"
        )
        build_provenance = {
            **build_payload,
            "path": str(self.build_provenance_path),
            "sha256": sha256(self.build_provenance_path),
        }
        certification = {
            "certification_profile_id": profile["profile_id"],
            "certification_profile_sha256": profile_hash,
            "runtime_matches_certification_profile": True,
            "validation_role": profile["validation_role"],
            "independent_final_holdout": False,
            "cohort_identity_verified": True,
            "cohort_identity": self.cohort_identity,
            "marketwide_validation_completed": True,
            "training_full_universe_adequacy_passed": True,
            "execution_integrity_passed": True,
            "full_two_sided_book_passed": True,
            "coverage_passed": True,
            "complete_two_sided_clock_passed": True,
            "finite_boundary_adequacy_passed": True,
            "finite_boundary_adequacy": {
                "stratified": {"passed": True},
                "marketwide": {"passed": True},
            },
            "empirical_fit_passed": True,
            "empirical_fit_acceptance_scope": "full_universe_marketwide",
            "stratified_structural_adequacy_passed": True,
            "stratified_empirical_fit_passed": True,
            "stratified_empirical_fit_acceptance_role": (
                "required_reported_diagnostic_only"
            ),
            "stratified_empirical_fit_failure_reasons": [],
            "marketwide_empirical_fit_passed": True,
            "marketwide_empirical_fit_acceptance_role": (
                "authoritative_certification_gate"
            ),
            "provenance_integrity_passed": True,
            "certified_for_case_study": True,
            "failure_reasons": [],
        }
        self.training_adequacy_status = self.root / "training_adequacy.json"
        self.marketwide_status = self.root / "marketwide_validation.json"
        training_days = []
        for day in profile["required_training_dates"]:
            target_root = self.root / f"targets_{day}"
            self.write_target_bundle(target_root, day)
            training_days.append({
                "date": day,
                "universe_config": str(self.training),
                "universe_config_sha256": sha256(self.training),
                "target_root": str(target_root),
                "empirical_input_bundle_sha256": (
                    CALIBRATION.empirical_input_bundle_sha256(self.training)
                ),
                "target_artifact_bundle_sha256": (
                    CALIBRATION.target_artifact_bundle_sha256(
                        target_root, day, ("QQQ",), (300, 3600, None),
                    )
                ),
            })
        development_target_root = self.root / "targets_2020-01-30"
        self.write_target_bundle(development_target_root, "2020-01-30")
        target_hash = CALIBRATION.target_artifact_bundle_sha256(
            development_target_root, "2020-01-30", ("QQQ",), (None,),
        )
        training_seeds = tuple(
            profile["full_universe_training_adequacy"]["seeds"]
        )
        _, training_config_rows = CALIBRATION.load_universe_config(self.training)
        training_evaluations = []
        for record in training_days:
            day = record["date"]
            target_root = pathlib.Path(record["target_root"])
            targets = CALIBRATION.load_targets(
                target_root, day, ("QQQ",),
            )
            raw_evaluation = self.make_full_day_evaluation(
                self.root / "training_evidence" / day.replace("-", ""),
                training_seeds, targets,
            )
            training_day = CALIBRATION.TrainingDay(
                date=day,
                universe_config=self.training,
                target_root=target_root,
                fields=tuple(self.fields),
                rows=tuple(dict(row) for row in training_config_rows),
                universe_config_sha256=sha256(self.training),
            )
            training_evaluations.append((training_day, raw_evaluation))
        training_aggregate = CALIBRATION.aggregate_training_day_evaluations(
            training_evaluations, seed_count=len(training_seeds),
        )
        training_gate_summary = (
            CALIBRATION.full_universe_training_adequacy_summary(
                training_aggregate,
                maximum_score=profile["maximum_robust_score"],
                maximum_metric_score=profile["maximum_metric_score"],
                maximum_symbol_metric_absolute_residual=profile[
                    "maximum_symbol_metric_absolute_robust_residual"
                ],
            )
        )
        self.training_adequacy_status.write_text(
            json.dumps({
                "schema_version": 1,
                "scope": "all_common_symbols_on_every_training_date",
                "symbol_count": profile["required_common_symbol_count"],
                "required_symbol_count": profile["required_common_symbol_count"],
                "training_dates": profile["required_training_dates"],
                "duration_seconds": 23_400,
                "seeds": list(training_seeds),
                "cohort_identity": self.cohort_identity,
                **training_gate_summary,
                "evaluation": CALIBRATION.evaluation_report(training_aggregate),
            }),
            encoding="utf-8",
        )
        marketwide_seeds = tuple(profile["required_stage3_seeds"])
        marketwide_targets = CALIBRATION.load_targets(
            development_target_root, profile["required_validation_date"],
            ("QQQ",),
        )
        marketwide_evaluation = self.make_full_day_evaluation(
            self.root / "marketwide_evidence", marketwide_seeds,
            marketwide_targets,
        )
        marketwide_coverage = CALIBRATION.two_sided_coverage_summary(
            marketwide_evaluation,
            profile["maximum_two_sided_shortfall_diagnostic"],
        )
        marketwide_fit = CALIBRATION.empirical_fit_summary(
            marketwide_evaluation,
            maximum_score=profile["maximum_robust_score"],
            maximum_metric_score=profile["maximum_metric_score"],
            maximum_symbol_metric_absolute_residual=profile[
                "maximum_symbol_metric_absolute_robust_residual"
            ],
        )
        stratified_evaluation = self.make_full_day_evaluation(
            self.root / "stratified_evidence", marketwide_seeds,
            marketwide_targets,
        )
        stratified_coverage = CALIBRATION.two_sided_coverage_summary(
            stratified_evaluation,
            profile["maximum_two_sided_shortfall_diagnostic"],
        )
        stratified_shortfalls = CALIBRATION.two_sided_coverage_shortfalls(
            stratified_evaluation,
            profile["maximum_two_sided_shortfall_diagnostic"],
        )
        stratified_fit = CALIBRATION.empirical_fit_summary(
            stratified_evaluation,
            maximum_score=profile["maximum_robust_score"],
            maximum_metric_score=profile["maximum_metric_score"],
            maximum_symbol_metric_absolute_residual=profile[
                "maximum_symbol_metric_absolute_robust_residual"
            ],
        )
        certification["stratified_empirical_fit"] = stratified_fit
        certification["marketwide_empirical_fit"] = marketwide_fit
        self.stratified_status = self.root / "stratified_validation.json"
        stratified_certification = {
            "execution_integrity_passed": True,
            "full_two_sided_book_passed": True,
            "coverage_passed": True,
            "finite_boundary_adequacy_passed": True,
            "background_boundary_adequacy_passed": True,
            "value_boundary_adequacy_passed": True,
            "structural_adequacy_passed": True,
            "empirical_fit_passed": True,
            "empirical_fit_acceptance_role": (
                "required_reported_diagnostic_only"
            ),
            "empirical_fit_failure_reasons": [],
            "certified_for_case_study": True,
        }
        self.stratified_status.write_text(
            json.dumps({
                "schema_version": 2,
                "scope": "pooled_stratified_sample",
                "passed": True,
                "cohort_identity": self.cohort_identity,
                **stratified_certification,
                "finite_boundary_adequacy": {
                    "background": stratified_evaluation[
                        "finite_boundary_adequacy"
                    ],
                    "value": stratified_evaluation["value_boundary_adequacy"],
                },
                "failure_reasons": [],
                "empirical_fit_failure_reasons": [],
                "interpretation": (
                    "This required stratified probe certifies structural "
                    "adequacy only; its empirical-fit score and failures are "
                    "preserved as diagnostics. The full-universe market-wide "
                    "fit is authoritative."
                ),
                "coverage_summary": stratified_coverage,
                "coverage_shortfalls": stratified_shortfalls,
                "empirical_fit": stratified_fit,
                "evaluation": CALIBRATION.evaluation_report(
                    stratified_evaluation
                ),
            }),
            encoding="utf-8",
        )
        self.stratified_report = {
            "scope": "one or more non-representative symbols from every cluster",
            "not_a_full_market_distributional_claim": True,
            "symbols": ["QQQ"],
            "frozen_runtime_controls": {
                "hawkes_activity_scale": 0.3,
                "local_mm_interval_ms": 1000.0,
                "local_mm_quantity_multiplier": 1.0,
                "label": "fixture_local",
                "local_mm_enabled": True,
                "local_mm_improvement_probability": 0.5,
                "shared_quote_mode": "relative_to_empirical_symbol_quote_size",
                "shared_quote_multiplier": 1.0,
                "shared_market_maker_enabled": True,
                "value_agents_enabled": True,
            },
            "evaluation": CALIBRATION.evaluation_report(stratified_evaluation),
            "coverage_summary": stratified_coverage,
            "empirical_fit": stratified_fit,
            "certification": stratified_certification,
            "acceptance_rule": (
                "execution integrity, empirical two-sided coverage and both "
                "source-attributed finite-boundary checks must pass. The "
                "stratified empirical-fit score is required and retained as a "
                "diagnostic; only the exact full-universe market-wide fit is "
                "the held-out empirical-fit certification gate"
            ),
        }
        self.marketwide_status.write_text(
            json.dumps({
                "schema_version": 2,
                "scope": "full_universe_marketwide",
                "symbol_count": profile["required_common_symbol_count"],
                "required_symbol_count": profile["required_common_symbol_count"],
                "validation_date": profile["required_validation_date"],
                "duration_seconds": profile["required_session_duration_seconds"],
                "seeds": list(marketwide_seeds),
                "cohort_identity": self.cohort_identity,
                "passed": True,
                "execution_integrity_passed": True,
                "full_two_sided_book_passed": True,
                "coverage_passed": True,
                "structural_adequacy_passed": True,
                "finite_boundary_adequacy_passed": True,
                "finite_boundary_adequacy": {
                    "background": marketwide_evaluation[
                        "finite_boundary_adequacy"
                    ],
                    "value": marketwide_evaluation["value_boundary_adequacy"],
                },
                "background_boundary_adequacy_passed": True,
                "value_boundary_adequacy_passed": True,
                "empirical_fit_passed": True,
                "empirical_fit_acceptance_role": (
                    "authoritative_certification_gate"
                ),
                "certified_for_case_study": True,
                "failure_reasons": [],
                "coverage_summary": marketwide_coverage,
                "empirical_fit": marketwide_fit,
                "evaluation": CALIBRATION.evaluation_report(
                    marketwide_evaluation
                ),
            }),
            encoding="utf-8",
        )
        self.pool_provenance_path = self.root / "pooling_provenance.json"
        rate_settings = {
            "activity_scale": profile["pooling_protocol"]["activity_scale"],
            "kernel_beta": profile["pooling_protocol"]["hawkes_beta"],
            "balance_directional_volume": profile["pooling_protocol"][
                "balance_directional_volume"
            ],
            "balance_best_depth": profile["pooling_protocol"][
                "balance_best_depth"
            ],
            "balance_strength": profile["pooling_protocol"][
                "balance_strength"
            ],
            "excitation_structure": profile["pooling_protocol"][
                "excitation_structure"
            ],
            "self_excitation_amplitude": profile["pooling_protocol"][
                "self_excitation_amplitude"
            ],
            "cross_excitation_amplitude": profile["pooling_protocol"][
                "cross_excitation_amplitude"
            ],
        }
        rate_derivation = {
            "schema_version": 1,
            "status": "passed",
            "event_types_checked": 6,
            "manifest_duration_seconds": 23_400,
            "maximum_absolute_observed_rate_error": 0.0,
            "observed_rates_equal_manifest_counts_per_duration": True,
            "maximum_absolute_stationary_target_error": 0.0,
            "stationary_targets_equal_declared_transforms_per_type": True,
            "maximum_absolute_reported_reconstruction_error": 0.0,
            "reported_reconstruction_equals_configured_rate_equation_per_type": True,
            "maximum_absolute_stationary_reconstruction_error": 0.0,
            "relative_tolerance": 1.0e-12,
            "absolute_tolerance": 1.0e-12,
            "stationary_reconstruction_equals_target_per_type": True,
            "transform_settings": rate_settings,
            "manifest": {
                "path": str(self.manifest), "sha256": sha256(self.manifest),
            },
            "generated_hawkes_rates": {
                "path": str(self.rates), "sha256": sha256(self.rates),
            },
        }
        one_symbol_hash = hashlib.sha256(b"QQQ\n").hexdigest()
        empty_symbol_hash = hashlib.sha256(b"").hexdigest()
        source_sessions = [
            *profile["required_training_dates"],
            profile["required_validation_date"],
        ]
        self.pool_input_selection = {
            "schema_version": 1,
            "status": "exact_certification_pool_input_verified",
            "mode": "prefiltered_fixed_cohort_1480_to_1480",
            "source_session_count": 6,
            "source_sessions": source_sessions,
            "source_session_symbol_count": {
                session: 1 for session in source_sessions
            },
            "source_session_symbol_order_sha256": {
                session: one_symbol_hash for session in source_sessions
            },
            "every_source_session_is_exact_cohort": True,
            "intersection_symbol_count": 1,
            "intersection_symbol_order_sha256": one_symbol_hash,
            "fixed_price_grid_excluded_symbol_count": 0,
            "fixed_price_grid_excluded_symbol_order_sha256": empty_symbol_hash,
            "final_symbol_count": 1,
            "final_symbol_order_sha256": one_symbol_hash,
        }
        pool_payload = {
            "schema_version": 7,
            "method": "multi_day_direct_input_pooling_with_day_level_behavioural_wmm",
            "workflow_source_semantics_sha256": workflow_hash,
            "training_dates": profile["required_training_dates"],
            "heldout_date": "2020-01-30",
            "intersection_symbol_count": 1,
            "common_symbol_count": 1,
            "certification_cohort_required": True,
            "certification_input_selection": self.pool_input_selection,
            "cohort_identity": self.pool_cohort_identity,
            "opening_price_grid_eligibility": {
                "simulator_tick_size_price_units": 100,
                "minimum_opening_bid_price_units": 10_000,
                "intersection_symbol_count": 1,
                "eligible_symbol_count": 1,
                "excluded_symbol_count": 0,
                "excluded_symbols": [],
            },
            "pooling": {
                "heldout_targets_used_for_runtime_configuration": False,
                "hawkes": {
                    "activity_scale": 0.3,
                    "kernel_beta": 10.0,
                    "balance_directional_volume": True,
                    "balance_best_depth": True,
                    "balance_strength": 1.0,
                    "excitation_structure": "diagonal_self_excitation_only",
                    "self_excitation_amplitude": 0.20,
                    "cross_excitation_amplitude": 0.0,
                },
            },
            "pooling_parameters": {
                "minimum_common_symbols": 20,
                "quote_quantity_fraction": 0.5,
                "minimum_quote_quantity": 10,
                "maximum_quote_quantity": 1000,
                "pool_label": "five_2019_sessions",
            },
            "quote_improvement_runtime_approximation": dict(
                CALIBRATION.QUOTE_IMPROVEMENT_RUNTIME_APPROXIMATION
            ),
            "configuration_schema": {
                "schema_version": 5,
                "source_fields": list(CALIBRATION.BASE_CONFIG_FIELDS),
                "runtime_fields": list(CALIBRATION.RUNTIME_CONFIG_FIELDS),
                "runtime_fields_sha256": (
                    CALIBRATION.configuration_schema_sha256(
                        CALIBRATION.RUNTIME_CONFIG_FIELDS
                    )
                ),
                "pooled_homeostatic_fields": list(
                    CALIBRATION.POOLED_HOMEOSTATIC_FIELDS
                ),
                "latent_value_fields": list(
                    CALIBRATION.LATENT_VALUE_FIELDS
                ),
                "frozen_training_derived_fields": list(
                    CALIBRATION.FROZEN_TRAINING_DERIVED_FIELDS
                ),
                "queue_reactive_target_fields": list(
                    CALIBRATION.QUEUE_REACTIVE_TARGET_FIELDS
                ),
                "positive_queue_reactive_targets_required": True,
                "same_pooled_targets_in_all_runtime_sessions": True,
                "heldout_target_files_used": False,
            },
            "training_days": [
                {
                    "date": row["date"],
                    "source_config": str(self.training),
                    "source_config_sha256": sha256(self.training),
                    "common_config_sha256": row["universe_config_sha256"],
                    "target_root": row["target_root"],
                }
                for row in training_days
            ],
            "heldout": {
                "source_config": str(self.heldout),
                "source_config_sha256": sha256(self.heldout),
                "common_config": str(self.heldout),
                "common_config_sha256": sha256(self.heldout),
                "heldout_role": "opening_state_and_validation_targets_only",
                "background_inputs_inherited_from_pooled": True,
            },
            "pooled_configuration": {
                "path": str(self.training), "sha256": sha256(self.training),
            },
            "symbols": [{
                "symbol": "QQQ",
                "pooled_manifest": str(self.manifest),
                "pooled_hawkes_rates": str(self.rates),
                "pooled_hawkes_rates_sha256": sha256(self.rates),
                "rate_derivation": rate_derivation,
                "sources": [
                    {
                        "trading_date": row["date"],
                        "manifest": str(self.manifest),
                        "manifest_sha256": sha256(self.manifest),
                        "source_hawkes_rates": str(self.rates),
                        "source_hawkes_rates_sha256": sha256(self.rates),
                        "generated_hawkes_rates": str(self.rates),
                        "generated_hawkes_rates_sha256": sha256(self.rates),
                        "rate_derivation": rate_derivation,
                    }
                    for row in training_days
                ],
            }],
        }
        self.pool_provenance_path.write_text(
            json.dumps(pool_payload), encoding="utf-8"
        )
        pool_provenance = {
            "path": str(self.pool_provenance_path),
            "sha256": sha256(self.pool_provenance_path),
            **pool_payload,
        }
        pool_provenance["producer_source_verification"] = (
            CALIBRATION.validate_pooling_producer_workflow_source(
                pool_payload,
                producer_project_root=ROOT,
                consumer_project_root=ROOT,
            )
        )
        self.report = self.root / "report.json"
        report = {
            "schema_version": 2,
            "certification_profile": profile,
            "certification_profile_sha256": profile_hash,
            "observed_runtime_profile": profile,
            "cohort_identity": self.cohort_identity,
            "certification_input_selection": self.pool_input_selection,
            "observed_survivor_counts": {
                "global_shared_quote": {
                    "stage1_screen": {
                        "configured_ranked_survivor_count": 6,
                        "evaluated_candidates": 4,
                        "eligible_candidates": 4,
                        "promoted_candidates": 4,
                    },
                    "stage2_refinement": {
                        "configured_ranked_survivor_count": 2,
                        "evaluated_candidates": 4,
                        "eligible_candidates": 4,
                        "promoted_candidates": 2,
                    },
                    "stage3_full": {
                        "configured_ranked_survivor_count": 1,
                        "evaluated_candidates": 2,
                        "eligible_candidates": 2,
                        "promoted_candidates": 1,
                    },
                },
            },
            "validation_scope": {
                "role": profile["validation_role"],
                "independent_final_holdout": False,
            },
            "protocol": {
                "cohort_identity": self.cohort_identity,
                "training_config_sha256": sha256(self.training),
                "runtime_configuration_schema": {
                    "schema_version": 5,
                    "fields": list(CALIBRATION.RUNTIME_CONFIG_FIELDS),
                    "sha256": CALIBRATION.configuration_schema_sha256(
                        CALIBRATION.RUNTIME_CONFIG_FIELDS
                    ),
                    "pooled_homeostatic_fields": list(
                        CALIBRATION.POOLED_HOMEOSTATIC_FIELDS
                    ),
                    "latent_value_fields": list(
                        CALIBRATION.LATENT_VALUE_FIELDS
                    ),
                    "frozen_training_derived_fields": list(
                        CALIBRATION.FROZEN_TRAINING_DERIVED_FIELDS
                    ),
                    "heldout_target_files_used": False,
                },
                "training_days": training_days,
                "binary_sha256": binary_hash,
                "simulator_source_semantics_sha256": source_hash,
                "workflow_source_semantics_sha256": workflow_hash,
                "calibration_build_provenance": build_provenance,
                "cluster_manifest": cluster_manifest,
                "pooling_provenance": pool_provenance,
                "frozen_empirical_input_bundle_sha256": bundle_hash,
                "development_validation_target_bundle_sha256": target_hash,
                "development_validation_target_root": str(
                    development_target_root
                ),
                "heldout_leakage_barrier": {
                    "heldout_fields_allowed": [
                        "fundamental_price_ticks", "initial_best_bid_ticks",
                        "initial_best_ask_ticks", "initial_best_bid_depth",
                        "initial_best_ask_depth",
                    ]
                },
            },
            "artifacts": {
                "full_universe_policy_csv": str(self.policy),
                "frozen_heldout_opening_config_csv": str(self.heldout),
                "frozen_heldout_opening_config_sha256": sha256(self.heldout),
                "frozen_empirical_input_bundle_sha256": bundle_hash,
                "heldout_stratified_validation_status_json": str(
                    self.stratified_status
                ),
                "heldout_stratified_validation_status_sha256": sha256(
                    self.stratified_status
                ),
            },
            "global_local_flow_selection": {
                "controls": {
                    "hawkes_activity_scale": 0.3,
                    "local_mm_enabled": True,
                    "local_mm_interval_ms": 1000.0,
                    "local_mm_quantity_multiplier": 1.0,
                    "local_mm_improvement_probability": 0.5,
                    "label": "fixture_local",
                }
            },
            "global_shared_quote_selection": {
                "candidate": {"enabled": True, "multiplier": 1.0}
            },
            "clusters": {
                "0": {
                    "cluster_id": 0,
                    "cluster_label": "liquidity_00",
                    "representative_symbols": ["QQQ"],
                    "selected_policy": {
                        "enabled": True,
                        "threshold_bps": 8.0,
                        "depth_participation": 0.25,
                        "label": "threshold_8_depth_participation_0.25",
                    },
                },
            },
            "heldout_stratified_validation": self.stratified_report,
            "certification": certification,
        }
        self.report.write_text(json.dumps(report), encoding="utf-8")
        self.handoff = self.root / "handoff.json"
        handoff = {
            "schema_version": 1,
            "artifact_role": "certified_calibration_handoff",
            "certification": certification,
            "certification_profile": profile,
            "certification_profile_sha256": profile_hash,
            "observed_runtime_profile": profile,
            "cohort_identity": self.cohort_identity,
            "certification_input_selection": self.pool_input_selection,
            "observed_survivor_counts": report["observed_survivor_counts"],
            "validation_role": profile["validation_role"],
            "independent_final_holdout": False,
            "calibration_report": str(self.report),
            "calibration_report_sha256": sha256(self.report),
            "training_universe_config": str(self.training),
            "training_universe_config_sha256": sha256(self.training),
            "runtime_configuration_schema": {
                "schema_version": 5,
                "fields": list(CALIBRATION.RUNTIME_CONFIG_FIELDS),
                "sha256": CALIBRATION.configuration_schema_sha256(
                    CALIBRATION.RUNTIME_CONFIG_FIELDS
                ),
                "pooled_homeostatic_fields": list(
                    CALIBRATION.POOLED_HOMEOSTATIC_FIELDS
                ),
                "latent_value_fields": list(
                    CALIBRATION.LATENT_VALUE_FIELDS
                ),
                "frozen_training_derived_fields": list(
                    CALIBRATION.FROZEN_TRAINING_DERIVED_FIELDS
                ),
                "heldout_target_files_used": False,
            },
            "training_days": training_days,
            "frozen_heldout_opening_config": str(self.heldout),
            "frozen_heldout_opening_config_sha256": sha256(self.heldout),
            "frozen_empirical_input_bundle_sha256": bundle_hash,
            "development_validation_target_bundle_sha256": target_hash,
            "full_universe_training_adequacy": {
                "passed": True,
                "status_json": str(self.training_adequacy_status),
                "status_sha256": sha256(self.training_adequacy_status),
                "symbols": profile["required_common_symbol_count"],
                "training_dates": profile["required_training_dates"],
                "duration_seconds": 23400,
                "seeds": profile["full_universe_training_adequacy"]["seeds"],
                "development_validation_targets_opened": False,
            },
            "heldout_stratified_validation": {
                "passed": True,
                "structural_adequacy_passed": True,
                "empirical_fit_passed": True,
                "empirical_fit_acceptance_role": (
                    "required_reported_diagnostic_only"
                ),
                "empirical_fit_failure_reasons": [],
                "status_json": str(self.stratified_status),
                "status_sha256": sha256(self.stratified_status),
                "symbols": 1,
                "validation_date": profile["required_validation_date"],
                "duration_seconds": profile[
                    "required_session_duration_seconds"
                ],
                "seeds": profile["required_stage3_seeds"],
            },
            "heldout_marketwide_validation": {
                "passed": True,
                "status_json": str(self.marketwide_status),
                "status_sha256": sha256(self.marketwide_status),
                "symbols": profile["required_common_symbol_count"],
                "validation_date": profile["required_validation_date"],
                "duration_seconds": profile[
                    "required_session_duration_seconds"
                ],
                "seeds": profile["required_stage3_seeds"],
                "empirical_fit_acceptance_role": (
                    "authoritative_certification_gate"
                ),
            },
            "calibration_binary_sha256": binary_hash,
            "simulator_source_semantics_sha256": source_hash,
            "workflow_source_semantics_sha256": workflow_hash,
            "calibration_build_provenance": build_provenance,
            "cluster_manifest": cluster_manifest,
            "pooling_provenance": pool_provenance,
            "value_agent_policy_csv": str(self.policy),
            "value_agent_policy_sha256": sha256(self.policy),
            "shock_cluster_csv": str(self.clusters),
            "shock_cluster_csv_sha256": sha256(self.clusters),
            "validation_sample_csv": str(self.validation_sample),
            "validation_sample_sha256": sha256(self.validation_sample),
            "development_validation_date": "2020-01-30",
            "development_validation_target_root": str(development_target_root),
            "runtime_controls": {
                "hawkes_activity_scale": 0.3,
                "local_market_maker_enabled": True,
                "local_mm_interval_ms": 1000.0,
                "local_mm_quantity_multiplier": 1.0,
                "local_mm_improvement_probability": 0.5,
                "shared_market_maker_enabled": True,
                "shared_quote_mode": "relative_to_empirical_symbol_quote_size",
                "shared_quote_multiplier": 1.0,
                "shared_quote_levels": 1,
                "decision_window_ms": 1000.0,
            },
            "agent_enablement": {
                "local_market_maker": True,
                "shared_market_maker": True,
                "value_agents": True,
            },
            "mechanism_treatments": {
                "shared_market_maker": {
                    "enabled": True,
                    "quote_mode": "relative_to_empirical_symbol_quote_size",
                    "quote_multiplier": 1.0,
                    "selected_by_training_fit": True,
                }
            },
        }
        self.handoff.write_text(json.dumps(handoff), encoding="utf-8")
        self.independent_certification = (
            self.root / "independent_global_calibration_certification.json"
        )
        self.independent_certification.write_text(
            json.dumps({
                "schema_version": 1,
                "artifact_role": "independent_global_calibration_certification",
                "status": "PASS",
                "calibration_handoff_sha256": sha256(self.handoff),
            }),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_config(self, path: pathlib.Path, *, fundamental: str, beta: str) -> None:
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=self.fields)
            writer.writeheader()
            writer.writerow({
                "book_id": "0", "symbol": "QQQ", "data_dir": str(self.data_dir),
                "hawkes_rates_file": str(self.rates),
                "fundamental_price_ticks": fundamental,
                "fundamental_volatility_bps_sqrt_second": "2.0",
                "fundamental_move_probability_per_second": "0.1",
                "fundamental_conditional_kurtosis": "4.0",
                "initial_best_bid_ticks": "9999",
                "initial_best_ask_ticks": "10001",
                "initial_best_bid_depth": "100",
                "initial_best_ask_depth": "100", "beta": beta,
                "basket_weight": "0",
                "market_maker_quote_quantity": "100",
                "target_spread_ticks": "2",
                "quote_improvement_probability": "0.1",
                "target_mean_bid_depth": "200",
                "target_mean_ask_depth": "220",
            })

    def write_target_bundle(self, root: pathlib.Path, day: str) -> None:
        compact = day.replace("-", "")
        directory = root / f"itch_{compact}_qqq"
        directory.mkdir(parents=True)
        target_values = {
            "mean_spread_ticks": 1.0,
            "mean_bid_depth": 100.0,
            "mean_ask_depth": 100.0,
            "mid_move_rate": 0.1,
            "return_variance": 0.01,
            "return_kurtosis": 3.0,
            "absolute_return_acf1": 0.0,
            "two_sided_sample_fraction": 1.0,
        }
        target_scales = {metric: 1.0 for metric in target_values}
        csv_values = {"background_event_rate": 1.0, **target_values}
        windows: dict[str, object] = {}
        for horizon in (300, 3600, None):
            suffix = "" if horizon is None else f"_window_{horizon}s"
            filename = f"market_targets_qqq_{compact}{suffix}.csv"
            with (directory / filename).open(
                    "w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output, fieldnames=("name", "target", "scale", "weight")
                )
                writer.writeheader()
                for metric in CALIBRATION.METRICS:
                    writer.writerow({
                        "name": metric,
                        "target": csv_values[metric],
                        "scale": 1.0,
                        "weight": 1.0,
                    })
            if horizon is not None:
                windows[str(horizon)] = {
                    "file": filename,
                    "duration_seconds": horizon,
                    "observations": horizon,
                    "valid_snapshots": horizon,
                    "invalid_snapshots": 0,
                    "values": target_values,
                    "scales": target_scales,
                }
        (directory / f"itch_manifest_qqq_{compact}.json").write_text(
            json.dumps({
                "snapshot_interval_ms": 1000,
                "trading_date": day,
                "symbol": "QQQ",
                "session_start": "09:30:00",
                "session_end": "16:00:00",
                "valid_snapshots": 23_400,
                "invalid_snapshots": 0,
                "aggregation_duration_seconds": 23_400,
                "distribution_observation_counts": {
                    event: 3_900 for event in (
                        "limit_buy", "limit_sell", "market_buy", "market_sell",
                        "cancel_bid", "cancel_ask",
                    )
                },
                "market_values": target_values,
                "market_target_scales": target_scales,
                "market_target_windows": windows,
            }),
            encoding="utf-8",
        )

    def write_full_day_summary(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = (
            "symbol", "sample_count", "expected_sample_count",
            "invalid_sample_count", "structurally_valid",
            *CALIBRATION.METRICS, *CALIBRATION.BOUNDARY_SUMMARY_FIELDS,
        )
        row: dict[str, object] = {
            "symbol": "QQQ",
            "sample_count": 23_400,
            "expected_sample_count": 23_400,
            "invalid_sample_count": 0,
            "structurally_valid": 1,
            "background_event_rate": 1.0,
            "mean_spread_ticks": 1.0,
            "mean_bid_depth": 100.0,
            "mean_ask_depth": 100.0,
            "mid_move_rate": 0.1,
            "return_variance": 0.01,
            "return_kurtosis": 3.0,
            "absolute_return_acf1": 0.0,
            "two_sided_sample_fraction": 1.0,
        }
        row.update({field: 0 for field in CALIBRATION.BOUNDARY_SUMMARY_FIELDS})
        row.update({
            "background_event_count": 23_400,
            "background_market_requested_quantity": 1,
            "background_cancel_requested_quantity": 1,
            "value_order_count": 1,
            "value_requested_quantity": 1,
        })
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

    def make_full_day_evaluation(
        self, output_root: pathlib.Path, seeds: tuple[int, ...],
        targets: dict[str, dict[str, CALIBRATION.TargetMoment]],
    ) -> dict[str, object]:
        paths = []
        for seed in seeds:
            path = output_root / f"seed_{seed}" / "fragmented_asset_summary.csv"
            self.write_full_day_summary(path)
            paths.append(path)
        return self.evaluate_existing_summaries(paths, targets)

    def evaluate_existing_summaries(
        self, paths: list[pathlib.Path],
        targets: dict[str, dict[str, CALIBRATION.TargetMoment]],
    ) -> dict[str, object]:
        fit, estimates = CALIBRATION.weighted_moment_loss(
            paths, targets, ("QQQ",), required_expected_sample_count=23_400,
        )
        combined, _ = CALIBRATION.weighted_moment_loss(
            paths, targets, ("QQQ",), uncertainty_mode="combined",
            required_expected_sample_count=23_400,
        )
        selection, metric_scores = CALIBRATION.metric_balanced_robust_loss(
            estimates
        )
        integrity, failures = CALIBRATION.two_sided_execution_integrity(
            paths, ("QQQ",), required_expected_sample_count=23_400,
        )
        boundary = CALIBRATION.finite_boundary_adequacy(
            paths, ("QQQ",), required_expected_sample_count=23_400,
        )
        value_boundary = CALIBRATION.value_boundary_adequacy(
            paths, ("QQQ",), required_expected_sample_count=23_400,
        )
        return {
            "fit_wsmrmse": fit,
            "combined_uncertainty_wsmrmse": combined,
            "selection_score": selection,
            "selection_metric_scores": metric_scores,
            "two_sided_integrity_passed": integrity,
            "two_sided_integrity_failures": failures,
            "finite_boundary_adequacy_passed": boundary["passed"] is True,
            "finite_boundary_adequacy": boundary,
            "value_boundary_adequacy_passed": value_boundary["passed"] is True,
            "value_boundary_adequacy": value_boundary,
            "seed_wall_seconds": [1.0 for _ in paths],
            "summary_paths": [str(path.resolve()) for path in paths],
            "errors": [],
            "moment_estimates": [
                {
                    field: getattr(estimate, field)
                    for field in estimate.__dataclass_fields__
                }
                for estimate in estimates
            ],
        }

    def replace_pooling_payload(self, payload: dict[str, object]) -> None:
        """Keep the synthetic handoff internally hashed after a pool mutation."""
        self.pool_provenance_path.write_text(
            json.dumps(payload), encoding="utf-8"
        )
        handoff = json.loads(self.handoff.read_text(encoding="utf-8"))
        prior_record = handoff["pooling_provenance"]
        pool_record = {
            "path": str(self.pool_provenance_path),
            "sha256": sha256(self.pool_provenance_path),
            **payload,
            "producer_source_verification": prior_record[
                "producer_source_verification"
            ],
        }
        report = json.loads(self.report.read_text(encoding="utf-8"))
        report["protocol"]["pooling_provenance"] = pool_record
        self.report.write_text(json.dumps(report), encoding="utf-8")
        handoff["pooling_provenance"] = pool_record
        handoff["calibration_report_sha256"] = sha256(self.report)
        self.handoff.write_text(json.dumps(handoff), encoding="utf-8")

    def replace_status_payload(
        self, status_path: pathlib.Path, handoff_key: str,
        payload: dict[str, object],
    ) -> None:
        """Re-hash a mutated status to test semantic, not hash-only, checks."""
        status_path.write_text(json.dumps(payload), encoding="utf-8")
        handoff = json.loads(self.handoff.read_text(encoding="utf-8"))
        handoff[handoff_key]["status_sha256"] = sha256(status_path)
        self.handoff.write_text(json.dumps(handoff), encoding="utf-8")

    def refresh_cluster_artifact_provenance(self) -> None:
        """Re-hash forged cluster artifacts all the way through the handoff."""
        manifest = json.loads(self.cluster_manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["cluster_assignments_csv"]["sha256"] = sha256(
            self.clusters
        )
        manifest["artifacts"]["validation_sample_csv"]["sha256"] = sha256(
            self.validation_sample
        )
        self.cluster_manifest_path.write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        cluster_record = {
            "path": str(self.cluster_manifest_path),
            "sha256": sha256(self.cluster_manifest_path),
            **manifest,
        }
        report = json.loads(self.report.read_text(encoding="utf-8"))
        report["protocol"]["cluster_manifest"] = cluster_record
        self.report.write_text(json.dumps(report), encoding="utf-8")
        handoff = json.loads(self.handoff.read_text(encoding="utf-8"))
        handoff["shock_cluster_csv_sha256"] = sha256(self.clusters)
        handoff["validation_sample_sha256"] = sha256(self.validation_sample)
        handoff["cluster_manifest"] = cluster_record
        handoff["calibration_report_sha256"] = sha256(self.report)
        self.handoff.write_text(json.dumps(handoff), encoding="utf-8")

    def refresh_policy_hash(self) -> None:
        handoff = json.loads(self.handoff.read_text(encoding="utf-8"))
        handoff["value_agent_policy_sha256"] = sha256(self.policy)
        self.handoff.write_text(json.dumps(handoff), encoding="utf-8")

    def refresh_report_hash(self) -> None:
        """Re-hash a forged report so semantic evidence checks are exercised."""
        handoff = json.loads(self.handoff.read_text(encoding="utf-8"))
        handoff["calibration_report_sha256"] = sha256(self.report)
        self.handoff.write_text(json.dumps(handoff), encoding="utf-8")

    def run_loader(
        self, shock_override: pathlib.Path | None = None,
        *,
        allow_preliminary: str = "off",
    ) -> subprocess.CompletedProcess[bytes]:
        environment = dict(os.environ)
        environment["PATH"] = f"{self.fake_bin}{os.pathsep}{environment.get('PATH', '')}"
        # The production loader is fail-closed at 1,480 rows.  This focused
        # contract fixture uses one fully materialised symbol so that every
        # test need not create tens of thousands of target files.  Replace
        # only the concrete CSV-cardinality predicate here; the profile,
        # handoff/status cardinalities and production-source assertion remain
        # 1,480 and are exercised independently below.
        loader = one_symbol_cohort_fixture_source(handoff_loader_source()).replace(
            "if len(training_symbols) != required_common_symbols:",
            "if len(training_symbols) != 1:",
        )
        loader = loader.replace(
            'required_cluster_count=CANONICAL_CERTIFICATION_PROFILE[\n'
            '        "required_cluster_count"\n'
            '    ],',
            'required_cluster_count=1,',
        ).replace(
            'required_training_representatives=CANONICAL_CERTIFICATION_PROFILE[\n'
            '        "required_training_representatives_per_cluster"\n'
            '    ],',
            'required_training_representatives=1,',
        ).replace(
            'required_validation_symbols=CANONICAL_CERTIFICATION_PROFILE[\n'
            '        "required_validation_symbols_per_cluster"\n'
            '    ],',
            'required_validation_symbols=0,',
        ).replace(
            'minimum_cluster_size=CANONICAL_CERTIFICATION_PROFILE[\n'
            '        "clustering_protocol"\n'
            '    ]["minimum_cluster_size"],',
            'minimum_cluster_size=1,',
        )
        loader = loader.replace(
            'expected_validation_symbols = tuple(\n'
            '    symbol\n'
            '    for cluster_id in sorted(verified_cluster_contract["validation"])\n'
            '    for symbol in verified_cluster_contract["validation"][cluster_id]\n'
            ')',
            'expected_validation_symbols = ("QQQ",)',
        ).replace(
            'if len(expected_validation_symbols) != (\n'
            '        CANONICAL_CERTIFICATION_PROFILE["required_cluster_count"]\n'
            '        * CANONICAL_CERTIFICATION_PROFILE[\n'
            '            "required_validation_symbols_per_cluster"\n'
            '        ]):',
            'if len(expected_validation_symbols) != 1:',
        )
        return subprocess.run(
            [
                sys.executable, "-", str(self.handoff), str(self.training), "on", "",
                str(shock_override) if shock_override is not None else "",
                allow_preliminary,
                str(ROOT),
            ],
            input=loader.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=environment,
        )

    def run_universe_input_builder(
        self, *, metadata: pathlib.Path | None, handoff_mode: str,
        snapshot_manifest: pathlib.Path | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        output = self.root / f"universe_input_{handoff_mode}.json"
        if snapshot_manifest is None:
            snapshot_manifest = self.root / "snapshot_manifest.json"
            snapshot_manifest.write_text(
                json.dumps({
                    "schema_version": 1,
                    "artifact_role": "case_study_validated_input_snapshot",
                    "calibration_handoff": str(self.handoff.resolve()),
                    "calibration_handoff_sha256": sha256(self.handoff),
                    "independent_certification": str(
                        self.independent_certification.resolve()
                    ),
                    "independent_certification_sha256": sha256(
                        self.independent_certification
                    ),
                    "inputs": [
                        {
                            "role": role,
                            "original_path": str(path.resolve()),
                            "certified_sha256": sha256(path),
                            "snapshot_path": str(path.resolve()),
                            "snapshot_sha256": sha256(path),
                        }
                        for role, path in (
                            ("universe_config", self.heldout),
                            ("value_agent_policy", self.policy),
                            ("shock_clusters", self.clusters),
                        )
                    ],
                }),
                encoding="utf-8",
            )
        return subprocess.run(
            [
                sys.executable, "-", str(self.heldout),
                str(metadata) if metadata is not None else "", str(output),
                handoff_mode, str(self.handoff), sha256(self.heldout),
                sha256(self.training), str(self.heldout), str(self.report),
                sha256(self.report), "certified_calibration_handoff", str(ROOT),
                str(self.independent_certification),
                sha256(self.independent_certification), str(snapshot_manifest),
                str(self.policy), str(self.clusters),
            ],
            input=one_symbol_cohort_fixture_source(
                universe_input_builder_source()
            ).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_default_build_tree_is_unique_per_slurm_job(self) -> None:
        self.assertIn(
            "build-seagull-gcc15-ompi509-real-case-${SLURM_JOB_ID}", SCRIPT
        )

    def test_production_loader_requires_exact_1480_symbol_universe(self) -> None:
        self.assertEqual(
            CALIBRATION.certification_profile()[
                "required_common_symbol_count"
            ],
            1480,
        )
        self.assertIn(
            "if len(training_symbols) != required_common_symbols:", SCRIPT,
        )
        self.assertIn(
            "certified training universe must contain exactly", SCRIPT,
        )

    def test_handoff_rejects_forged_cohort_artifact_identity(self) -> None:
        handoff = json.loads(self.handoff.read_text(encoding="utf-8"))
        handoff["cohort_identity"]["origin_manifest_sha256"] = "0" * 64
        self.handoff.write_text(json.dumps(handoff), encoding="utf-8")
        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "does not identify the bundled immutable artifact",
            completed.stderr.decode(),
        )

    def test_handoff_rejects_rehashed_wrong_scope_training_status(self) -> None:
        payload = json.loads(self.training_adequacy_status.read_text(encoding="utf-8"))
        payload["scope"] = "representative_sample_only"
        self.replace_status_payload(
            self.training_adequacy_status,
            "full_universe_training_adequacy",
            payload,
        )
        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "training adequacy status has the wrong evaluation scope",
            completed.stderr.decode(),
        )

    def test_handoff_rejects_rehashed_incomplete_training_evidence(self) -> None:
        payload = json.loads(self.training_adequacy_status.read_text(encoding="utf-8"))
        del payload["evaluation"]["training_day_evaluations"][0][
            "evaluation"
        ]["moment_estimates"][-1]
        self.replace_status_payload(
            self.training_adequacy_status,
            "full_universe_training_adequacy",
            payload,
        )
        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "does not equal the evaluation recomputed from its complete summary CSV evidence",
            completed.stderr.decode(),
        )

    def test_handoff_rejects_summary_modified_after_status(self) -> None:
        payload = json.loads(self.training_adequacy_status.read_text(encoding="utf-8"))
        summary = pathlib.Path(
            payload["evaluation"]["training_day_evaluations"][0][
                "evaluation"
            ]["summary_paths"][0]
        )
        status_mtime = self.training_adequacy_status.stat().st_mtime_ns
        os.utime(summary, ns=(status_mtime + 1_000_000, status_mtime + 1_000_000))
        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("summary was modified after its status JSON", completed.stderr.decode())

    def test_handoff_rejects_rehashed_empty_stratified_validation(self) -> None:
        report = json.loads(self.report.read_text(encoding="utf-8"))
        report["heldout_stratified_validation"] = {}
        self.report.write_text(json.dumps(report), encoding="utf-8")
        self.refresh_report_hash()
        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "held-out stratified validation has the wrong scope or symbols",
            completed.stderr.decode(),
        )

    def test_handoff_rejects_rehashed_partial_stratified_moments(self) -> None:
        report = json.loads(self.report.read_text(encoding="utf-8"))
        del report["heldout_stratified_validation"]["evaluation"][
            "moment_estimates"
        ][-1]
        self.report.write_text(json.dumps(report), encoding="utf-8")
        self.refresh_report_hash()
        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "does not equal the evaluation recomputed from its complete summary CSV evidence",
            completed.stderr.decode(),
        )

    def test_handoff_rejects_rehashed_incomplete_policy(self) -> None:
        self.policy.write_text(
            ",".join(CALIBRATION.POLICY_FIELDS) + "\n", encoding="utf-8"
        )
        self.refresh_policy_hash()
        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("value-agent policy has no data rows", completed.stderr.decode())

    def test_handoff_rejects_rehashed_policy_cluster_label_mismatch(self) -> None:
        with self.policy.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        rows[0]["cluster_label"] = "liquidity_09"
        with self.policy.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=CALIBRATION.POLICY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self.refresh_policy_hash()
        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "value-agent policy has the wrong cluster label for QQQ",
            completed.stderr.decode(),
        )

    def test_handoff_rejects_rehashed_incomplete_cluster_assignments(self) -> None:
        self.clusters.write_text(
            "symbol,cluster_id,is_representative\n", encoding="utf-8"
        )
        self.refresh_cluster_artifact_provenance()
        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("cluster assignments has no data rows", completed.stderr.decode())

    def test_handoff_rejects_rehashed_overlapping_validation_sample(self) -> None:
        self.validation_sample.write_text(
            "symbol,cluster_id\nQQQ,0\n", encoding="utf-8"
        )
        self.refresh_cluster_artifact_provenance()
        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "validation symbol QQQ is also the marked representative",
            completed.stderr.decode(),
        )

    def test_certified_handoff_does_not_require_launcher_metadata(self) -> None:
        self.assertNotIn(': "${CALIBRATION_METADATA:?', SCRIPT)
        completed = self.run_universe_input_builder(
            metadata=None, handoff_mode="on",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        record = json.loads(completed.stdout)
        self.assertEqual(
            record["calibration_provenance_mode"],
            "block_coordinate_certified_handoff",
        )
        self.assertFalse(record["source_extractor_metadata_supplied"])
        self.assertIsNone(
            record["source_extractor_metadata_matches_final_config"]
        )
        self.assertNotIn("calibration_metadata", record)
        self.assertNotIn("calibration_metadata_sha256", record)
        snapshot_record = record["validated_input_snapshot"]
        snapshot_path = pathlib.Path(snapshot_record["path"])
        self.assertEqual(snapshot_record["sha256"], sha256(snapshot_path))
        self.assertEqual(
            {item["role"] for item in snapshot_record["provenance"]["inputs"]},
            {"universe_config", "value_agent_policy", "shock_clusters"},
        )

    def test_certified_handoff_rejects_missing_input_snapshot(self) -> None:
        missing = self.root / "missing_snapshot_manifest.json"
        completed = self.run_universe_input_builder(
            metadata=None, handoff_mode="on", snapshot_manifest=missing,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "certified case-study input snapshot is missing",
            completed.stderr.decode(),
        )

    def test_certified_handoff_rejects_tampered_snapshot_hash(self) -> None:
        valid = self.root / "snapshot_manifest.json"
        initial = self.run_universe_input_builder(
            metadata=None, handoff_mode="on", snapshot_manifest=None,
        )
        self.assertEqual(initial.returncode, 0, initial.stderr.decode())
        payload = json.loads(valid.read_text(encoding="utf-8"))
        payload["inputs"][0]["snapshot_sha256"] = "0" * 64
        valid.write_text(json.dumps(payload), encoding="utf-8")
        completed = self.run_universe_input_builder(
            metadata=None, handoff_mode="on", snapshot_manifest=valid,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "snapshotted universe_config hash is invalid",
            completed.stderr.decode(),
        )

    def test_legacy_mode_still_requires_launcher_metadata(self) -> None:
        environment = dict(os.environ)
        environment.update({
            "SLURM_JOB_ID": "123",
            "SLURM_SUBMIT_DIR": str(ROOT),
            "UNIVERSE_CONFIG": str(self.heldout),
            "LEGACY_UNCALIBRATED_MODE": "on",
            "VALUE_AGENT": "off",
        })
        environment.pop("CALIBRATION_HANDOFF_JSON", None)
        environment.pop("CALIBRATION_METADATA", None)
        completed = subprocess.run(
            ["bash", str(ROOT / "submit_real_universe_case_study.sh")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=environment,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "legacy uncalibrated mode requires CALIBRATION_METADATA",
            completed.stderr.decode(),
        )

    def test_handoff_rejects_cluster_hash_override_and_uses_canonical_path(self) -> None:
        self.assertIn(
            "SHOCK_CLUSTER_CSV does not match the cluster CSV certified by the handoff",
            SCRIPT,
        )
        self.assertIn(
            'SHOCK_CLUSTER_CSV="${CALIBRATED_CLUSTER_PATH}"', SCRIPT
        )

    def test_handoff_selects_frozen_heldout_opening_configuration(self) -> None:
        self.assertIn('artifacts.get("frozen_heldout_opening_config_csv")', SCRIPT)
        self.assertIn('UNIVERSE_CONFIG="${CALIBRATED_CASE_CONFIG_PATH}"', SCRIPT)
        self.assertIn(
            '"case_study_config_role": '
            '"frozen_training_backgrounds_with_heldout_openings"',
            SCRIPT,
        )

        completed = self.run_loader()
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        values = completed.stdout.rstrip(b"\0").split(b"\0")
        self.assertEqual(values[5].decode(), "0.5")
        self.assertEqual(pathlib.Path(values[10].decode()), self.heldout.resolve())
        self.assertEqual(values[11].decode(), sha256(self.heldout))

    def test_v18_accepts_stratified_fit_failure_when_structure_and_market_pass(
        self,
    ) -> None:
        report = json.loads(self.report.read_text(encoding="utf-8"))
        stratified = report["heldout_stratified_validation"]
        summary_paths = [
            pathlib.Path(value)
            for value in stratified["evaluation"]["summary_paths"]
        ]
        for path in summary_paths:
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                fields = list(reader.fieldnames or ())
                rows = list(reader)
            rows[0]["return_kurtosis"] = "3000"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

        handoff = json.loads(self.handoff.read_text(encoding="utf-8"))
        target_root = pathlib.Path(
            handoff["development_validation_target_root"]
        )
        targets = CALIBRATION.load_targets(
            target_root, "2020-01-30", ("QQQ",),
        )
        evaluation = self.evaluate_existing_summaries(summary_paths, targets)
        profile = CALIBRATION.certification_profile()
        fit = CALIBRATION.empirical_fit_summary(
            evaluation,
            maximum_score=profile["maximum_robust_score"],
            maximum_metric_score=profile["maximum_metric_score"],
            maximum_symbol_metric_absolute_residual=profile[
                "maximum_symbol_metric_absolute_robust_residual"
            ],
        )
        self.assertFalse(fit["passed"])
        failures = CALIBRATION.empirical_fit_failure_reasons(
            "held-out stratified", fit,
        )
        coverage = CALIBRATION.two_sided_coverage_summary(
            evaluation, profile["maximum_two_sided_shortfall_diagnostic"],
        )
        shortfalls = CALIBRATION.two_sided_coverage_shortfalls(
            evaluation, profile["maximum_two_sided_shortfall_diagnostic"],
        )

        stratified["evaluation"] = CALIBRATION.evaluation_report(evaluation)
        stratified["coverage_summary"] = coverage
        stratified["empirical_fit"] = fit
        stratified["certification"]["empirical_fit_passed"] = False
        stratified["certification"]["empirical_fit_failure_reasons"] = failures
        certification = report["certification"]
        certification["stratified_empirical_fit_passed"] = False
        certification["stratified_empirical_fit_failure_reasons"] = failures
        certification["stratified_empirical_fit"] = fit
        self.assertTrue(certification["empirical_fit_passed"])
        self.assertTrue(certification["marketwide_empirical_fit_passed"])
        self.assertTrue(certification["certified_for_case_study"])

        status = json.loads(self.stratified_status.read_text(encoding="utf-8"))
        status["empirical_fit_passed"] = False
        status["empirical_fit_failure_reasons"] = failures
        status["coverage_summary"] = coverage
        status["coverage_shortfalls"] = shortfalls
        status["empirical_fit"] = fit
        status["evaluation"] = CALIBRATION.evaluation_report(evaluation)
        self.assertTrue(status["passed"])
        self.assertTrue(status["structural_adequacy_passed"])
        self.assertEqual(status["failure_reasons"], [])
        self.stratified_status.write_text(json.dumps(status), encoding="utf-8")
        report["artifacts"][
            "heldout_stratified_validation_status_sha256"
        ] = sha256(self.stratified_status)
        self.report.write_text(json.dumps(report), encoding="utf-8")

        handoff["certification"] = certification
        handoff["heldout_stratified_validation"][
            "empirical_fit_passed"
        ] = False
        handoff["heldout_stratified_validation"][
            "empirical_fit_failure_reasons"
        ] = failures
        handoff["heldout_stratified_validation"][
            "status_sha256"
        ] = sha256(self.stratified_status)
        handoff["calibration_report_sha256"] = sha256(self.report)
        self.handoff.write_text(json.dumps(handoff), encoding="utf-8")

        completed = self.run_loader()
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())

    def test_handoff_rejects_nonmatching_shock_cluster_override(self) -> None:
        override = self.root / "other_clusters.csv"
        override.write_text("symbol,cluster_id\nQQQ,9\n", encoding="utf-8")
        completed = self.run_loader(override)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "SHOCK_CLUSTER_CSV does not match the cluster CSV certified by the handoff",
            completed.stderr.decode(),
        )

    def test_handoff_rejects_nonopening_mutation_in_heldout_config(self) -> None:
        self.write_config(self.heldout, fundamental="10100", beta="9")
        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "frozen held-out opening configuration changed after validation",
            completed.stderr.decode(),
        )

    def test_handoff_rejects_mutated_external_empirical_input(self) -> None:
        (self.data_dir / "limit_buy_quantity_distribution.txt").write_text(
            "value,count\n999,1\n", encoding="utf-8",
        )
        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "daily training empirical-input bundle changed after calibration",
            completed.stderr.decode(),
        )

    def test_handoff_rejects_unbalanced_pooling_rates(self) -> None:
        payload = json.loads(
            self.pool_provenance_path.read_text(encoding="utf-8")
        )
        payload["pooling"]["hawkes"].update({
            "balance_directional_volume": False,
            "balance_best_depth": False,
            "balance_strength": 0.0,
        })
        self.replace_pooling_payload(payload)

        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "does not certify the canonical balanced reduced-book event-rate settings",
            completed.stderr.decode(),
        )

    def test_handoff_rejects_rehashed_tampered_pool_input_shape(self) -> None:
        payload = json.loads(
            self.pool_provenance_path.read_text(encoding="utf-8")
        )
        payload["certification_input_selection"][
            "intersection_symbol_count"
        ] = 2
        self.replace_pooling_payload(payload)

        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "does not match the independently reconstructed source shape",
            completed.stderr.decode(),
        )

    def test_handoff_rejects_pool_shape_not_bound_into_handoff(self) -> None:
        handoff = json.loads(self.handoff.read_text(encoding="utf-8"))
        handoff["certification_input_selection"] = dict(
            handoff["certification_input_selection"]
        )
        handoff["certification_input_selection"]["mode"] = (
            "legacy_unscreened_1509_to_1480"
        )
        self.handoff.write_text(json.dumps(handoff), encoding="utf-8")

        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "calibration handoff does not bind the verified pool input shape",
            completed.stderr.decode(),
        )

    def test_handoff_rejects_mutated_rate_derivation_hash(self) -> None:
        payload = json.loads(
            self.pool_provenance_path.read_text(encoding="utf-8")
        )
        payload["symbols"][0]["rate_derivation"][
            "generated_hawkes_rates"
        ]["sha256"] = "0" * 64
        self.replace_pooling_payload(payload)

        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "generated_hawkes_rates changed after rate derivation",
            completed.stderr.decode(),
        )

    def test_handoff_rejects_schema_two_pooling_provenance(self) -> None:
        payload = json.loads(
            self.pool_provenance_path.read_text(encoding="utf-8")
        )
        payload["schema_version"] = 2
        self.replace_pooling_payload(payload)

        completed = self.run_loader()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "pooling provenance does not use the audited rate-derivation schema",
            completed.stderr.decode(),
        )

    def test_relabelled_preliminary_cannot_bypass_authoritative_fit(self) -> None:
        self.assertIn("ALLOW_PRELIMINARY_MODEL", SCRIPT)
        report = json.loads(self.report.read_text(encoding="utf-8"))
        report["certification"]["empirical_fit_passed"] = False
        report["certification"]["certified_for_case_study"] = False
        report["certification"]["failure_reasons"] = ["empirical fit failed"]
        self.report.write_text(json.dumps(report), encoding="utf-8")

        preliminary = json.loads(self.handoff.read_text(encoding="utf-8"))
        preliminary["artifact_role"] = "preliminary_not_certified"
        preliminary["certification"] = dict(report["certification"])
        preliminary["calibration_report_sha256"] = sha256(self.report)
        self.handoff.write_text(json.dumps(preliminary), encoding="utf-8")

        rejected = self.run_loader(allow_preliminary="off")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertRegex(
            rejected.stderr.decode(),
            r"(?i)(empirical[- ]fit|not certified|preliminary)",
        )

        explicitly_allowed = self.run_loader(allow_preliminary="on")
        self.assertNotEqual(explicitly_allowed.returncode, 0)
        self.assertIn(
            "authoritative market-wide fit",
            explicitly_allowed.stderr.decode(),
        )


if __name__ == "__main__":
    unittest.main()
