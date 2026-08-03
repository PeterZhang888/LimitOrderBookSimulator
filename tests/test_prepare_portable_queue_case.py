#!/usr/bin/env python3
"""Tests for the hash-bound final case-study bundle preparation."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "prepare_portable_queue_case",
    ROOT / "scripts/prepare_portable_queue_case.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def record(path: pathlib.Path) -> dict[str, str]:
    return {"path": str(path), "sha256": MODULE.sha256_file(path)}


class PortableQueueCaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name).resolve()
        self.project = self.root / "project"
        self.evidence = self.root / "evidence"
        self.pool = self.root / "pool"
        self.selection = self.root / "selection"
        self.data = self.root / "data"
        self.output = self.root / "output"
        for path in (self.project, self.evidence, self.pool, self.selection, self.data):
            path.mkdir(parents=True)
        (self.project / "SOURCE_MANIFEST.sha256").write_text(
            "fixture manifest\n", encoding="utf-8",
        )
        self.executable = self.root / "fragmented_mpi_lob"
        self.executable.write_bytes(b"deterministic executable fixture")
        self.symbols = ("QQQ", "AAPL")
        self.deployment = self.selection / "full_training_configs/deployment_config.csv"
        self.heldout = self.evidence / "development_validation/heldout_simulation_config.csv"
        self._write_runtime_configs()
        self.value = self.selection / "value_policy.csv"
        self.value.write_text(
            "symbol,enabled\nQQQ,1\nAAPL,1\n", encoding="utf-8",
        )
        self.clusters = self.selection / "liquidity_clusters/cluster_assignments.csv"
        self.clusters.parent.mkdir(parents=True)
        self.clusters.write_text(
            "symbol,cluster_id\nQQQ,0\nAAPL,0\n", encoding="utf-8",
        )
        self.background = (
            self.selection / "queue_reactive_policy/symbol_policy_mapping.csv"
        )
        cluster_root = self.selection / "queue_reactive_policy/clusters/cluster_0"
        cluster_root.mkdir(parents=True)
        policy_files = {
            "policy_file": cluster_root / "cluster_policy.csv",
            "limit_buy_improvement_file": (
                cluster_root / "limit_buy_improvement_distribution.csv"
            ),
            "limit_sell_improvement_file": (
                cluster_root / "limit_sell_improvement_distribution.csv"
            ),
        }
        for path in policy_files.values():
            path.write_text("value,count\n1,1\n", encoding="utf-8")
        self.background.parent.mkdir(parents=True, exist_ok=True)
        with self.background.open("w", newline="", encoding="utf-8") as destination:
            fields = ["symbol", "cluster_id", *policy_files]
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            for symbol in self.symbols:
                writer.writerow({
                    "symbol": symbol,
                    "cluster_id": 0,
                    **{key: f"/old/{path.name}" for key, path in policy_files.items()},
                })
        self._write_handoffs()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_runtime_configs(self) -> None:
        self.deployment.parent.mkdir(parents=True)
        self.heldout.parent.mkdir(parents=True)
        rows: list[dict[str, object]] = []
        for book_id, symbol in enumerate(self.symbols):
            data_name = f"pooled_{symbol.lower()}"
            data_dir = self.pool / "pooled_data" / data_name
            data_dir.mkdir(parents=True)
            rates = data_dir / "rates.csv"
            rates.write_text("event_type,configured_mu\nlimit_buy,1\n", encoding="utf-8")
            manifest = (
                self.data / "itch_20200130/empirical_data"
                / f"itch_20200130_{symbol.lower()}"
                / f"itch_manifest_{symbol.lower()}_20200130.json"
            )
            write_json(manifest, {"schema_version": 1, "symbol": symbol})
            rows.append({
                "book_id": book_id,
                "symbol": symbol,
                "data_dir": f"/old/{data_name}",
                "hawkes_rates_file": "/old/rates.csv",
                "fundamental_price_ticks": 10_000 + book_id,
                "fundamental_volatility_bps_sqrt_second": 1.0,
                "fundamental_move_probability_per_second": 0.1,
                "fundamental_conditional_kurtosis": 3.0,
                "initial_best_bid_ticks": 9_999 + book_id,
                "initial_best_ask_ticks": 10_001 + book_id,
                "initial_best_bid_depth": 100,
                "initial_best_ask_depth": 110,
                "beta": 1.0,
                "basket_weight": 0.0,
                "market_maker_quote_quantity": 100,
                "target_spread_ticks": 2,
                "quote_improvement_probability": 0.1,
                "target_mean_bid_depth": 100,
                "target_mean_ask_depth": 110,
                "fundamental_log_variance_persistence": 0.9,
                "fundamental_log_variance_std": 0.2,
                "fundamental_order_flow_coupling": 0.3,
            })
        for path, opening_shift in ((self.deployment, 0), (self.heldout, 10)):
            with path.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(destination, fieldnames=MODULE.RUNTIME_FIELDS)
                writer.writeheader()
                for source in rows:
                    row = dict(source)
                    for field in MODULE.HELDOUT_OPENING_FIELDS:
                        row[field] = float(row[field]) + opening_shift
                    writer.writerow(row)

    def _write_handoffs(self) -> None:
        write_json(self.pool / "pooling_provenance.json", {"schema_version": 1})
        write_json(
            self.selection / "selection/training_selection_freeze.json",
            {"schema_version": 1},
        )
        strict = self.evidence / "development_validation/strict_validation_report.json"
        write_json(strict, {"passed": True, "evaluation_role": "development_validation"})
        augmentation = (
            self.evidence / "provenance/queue_reactive_augmentation_provenance.json"
        )
        write_json(augmentation, {"schema_version": 3, "status": "passed"})
        freeze = self.evidence / "training/expanded_training_freeze.json"
        write_json(freeze, {
            "schema_version": 1,
            "status": "expanded_training_adequacy_frozen",
            "training_only": True,
            "full_universe_training_adequacy_passed": True,
            "heldout_execution_authorized": True,
            "frozen_artifacts": {
                "deployment_config": record(self.deployment),
                "value_policy": record(self.value),
                "background_policy_mapping": record(self.background),
                "cluster_map": record(self.clusters),
                "executable": record(self.executable),
            },
            "selection": {
                "local_candidate": {
                    "enabled": True,
                    "interval_ms": 1000.0,
                    "quantity_multiplier": 1.0,
                    "improvement_probability": 0.25,
                    "spread_elasticity": 0.0,
                    "max_improvement_probability": 1.0,
                },
            },
        })
        heldout_manifest = self.evidence / "development_validation/heldout_run_manifest.json"
        write_json(heldout_manifest, {
            "schema_version": 1,
            "status": "heldout_adequacy_passed",
            "validation_claimed": True,
            "evaluation_role": "development_validation",
            "gate_protocol": "marketwide-six-v2",
            "all_other_simulation_fields_frozen": True,
            "training_freeze": record(freeze),
            "strict_report": {**record(strict), "passed": True},
            "simulation_config": record(self.heldout),
        })

    def args(self, output: pathlib.Path | None = None) -> argparse.Namespace:
        return argparse.Namespace(
            project_root=self.project,
            evidence_root=self.evidence,
            pool_root=self.pool,
            selection_root=self.selection,
            data_root=self.data,
            executable=self.executable,
            output_root=output or self.output,
            allow_post_validation_shared_dealer_amendment=False,
        )

    @mock.patch.object(MODULE.cohort, "validate_symbols")
    def test_prepares_hash_bound_portable_case(self, validate: mock.Mock) -> None:
        validate.return_value = {
            "schema_version": 1,
            "status": "exact_cohort_verified",
            "symbol_count": 2,
        }
        report = MODULE.run(self.args())
        self.assertEqual(report["status"], "portable_queue_reactive_case_ready")
        manifest_path = self.output / "portable_queue_reactive_case.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 5)
        self.assertEqual(payload["symbol_count"], 2)
        protocol = payload["case_study_protocol"]
        self.assertEqual(
            protocol["profile_id"],
            "systemic_liquidity_shock_queue_reactive",
        )
        self.assertEqual(protocol["production_ranks"], 16)
        self.assertEqual(protocol["financial_risk_limits"], [800.0, 1600.0])
        self.assertEqual(protocol["reference_risk_limit"], 1600.0)
        self.assertEqual(protocol["shock_fraction"], 0.10)
        self.assertEqual(protocol["shock_reference_bid_depth_multiple"], 3.0)
        self.assertEqual(
            protocol["shock_direction_rule"],
            "inventory_adverse_at_left_limit",
        )
        self.assertTrue(protocol["shared_quote_relative"])
        self.assertEqual(protocol["shared_quote_multiplier"], 2.0)
        self.assertTrue(protocol["shared_capacity_relative"])
        self.assertEqual(
            protocol["stochastic_baseline_normalization_seconds"], 23400.0,
        )
        self.assertEqual(protocol["repetitions"], 20)
        self.assertEqual(protocol["path_count"], 200)
        self.assertEqual(
            protocol["uncoupled_capacity_control"],
            "asset_specific_equal_total_capacity",
        )
        self.assertEqual(
            protocol["primary_outcome"],
            "relative_non_target_top_depth_deterioration",
        )
        self.assertEqual(
            protocol["reporting_horizons_seconds"], [1, 5, 30, 300, 1800],
        )
        self.assertTrue(protocol["asset_level_shock_dose_equality_required"])
        self.assertTrue(protocol["shock_fill_ownership_required"])
        self.assertTrue(protocol["truncated_full_prefix_equality_required"])
        self.assertEqual(
            protocol[
                "required_pre_shock_requested_two_sided_book_fraction"
            ],
            1.0,
        )
        self.assertEqual(
            protocol[
                "minimum_pre_shock_resting_two_sided_book_fraction"
            ],
            0.95,
        )
        self.assertEqual(
            protocol["minimum_pre_shock_bbo_depth_participation"], 0.05,
        )
        self.assertEqual(
            protocol["minimum_shock_absorption_fraction"], 0.025,
        )
        self.assertEqual(protocol["shared_quote_levels"], 3)
        self.assertTrue(
            protocol[
                "state_contingent_direction_rule_identical_across_mechanisms"
            ]
        )
        self.assertTrue(
            protocol[
                "target_side_materiality_assessed_by_realized_shock_absorption"
            ]
        )
        self.assertEqual(protocol["minimum_shared_quote_scale"], 0.05)
        self.assertEqual(
            payload["shared_market_maker"]["tight_risk_limit_per_asset"],
            800.0,
        )
        self.assertEqual(
            payload["shared_market_maker"]["reference_risk_limit_per_asset"],
            1600.0,
        )
        self.assertEqual(
            payload["shared_market_maker"]["minimum_quote_scale"], 0.05
        )
        self.assertFalse(
            payload["executable_provenance"][
                "post_validation_treatment_amendment"
            ]
        )
        self.assertEqual(
            payload["executable_provenance"]["amendment_scope"], "none"
        )
        digest_payload = dict(payload)
        expected_digest = digest_payload.pop("artifact_sha256")
        self.assertEqual(MODULE.sha256_json(digest_payload), expected_digest)
        self.assertEqual(payload["runtime_configuration_schema"], MODULE.RUNTIME_SCHEMA)
        self.assertEqual(payload["empirical_target_artifacts"]["entry_count"], 2)
        self.assertEqual(payload["background_policy_artifacts"]["entry_count"], 3)
        fields, rows = MODULE.read_csv(
            self.output / "heldout_20200130_queue_reactive_case.csv"
        )
        self.assertEqual(fields, list(MODULE.RUNTIME_FIELDS))
        self.assertEqual(float(rows[0]["fundamental_price_ticks"]), 10_010.0)
        self.assertTrue(pathlib.Path(rows[0]["data_dir"]).is_dir())
        self.assertTrue(pathlib.Path(rows[0]["hawkes_rates_file"]).is_file())

    @mock.patch.object(MODULE.cohort, "validate_symbols")
    def test_rejects_executable_not_in_freeze(self, validate: mock.Mock) -> None:
        validate.return_value = {"status": "exact_cohort_verified"}
        self.executable.write_bytes(b"changed after validation")
        with self.assertRaisesRegex(
            MODULE.PreparationError, "rebuilt executable differs",
        ):
            MODULE.run(self.args(self.root / "bad-output"))

    @mock.patch.object(MODULE.cohort, "validate_symbols")
    def test_upgrades_legacy_heldout_schema_from_frozen_training_deployment(
        self, validate: mock.Mock,
    ) -> None:
        validate.return_value = {"status": "exact_cohort_verified"}
        fields, rows = MODULE.read_csv(self.heldout)
        legacy_fields = fields[:-3]
        MODULE.write_csv(self.heldout, legacy_fields, rows)
        manifest_path = self.evidence / "development_validation/heldout_run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["simulation_config"] = record(self.heldout)
        write_json(manifest_path, manifest)

        output = self.root / "legacy-output"
        MODULE.run(self.args(output))
        output_fields, output_rows = MODULE.read_csv(
            output / "heldout_20200130_queue_reactive_case.csv"
        )
        self.assertEqual(output_fields, list(MODULE.RUNTIME_FIELDS))
        for field in MODULE.RUNTIME_FIELDS[-3:]:
            self.assertEqual(output_rows[0][field], rows[0][field])
        self.assertEqual(
            float(output_rows[0]["fundamental_price_ticks"]), 10_010.0
        )

    @mock.patch.object(MODULE.cohort, "validate_symbols")
    def test_projects_hash_bound_heldout_with_extra_validation_columns(
        self, validate: mock.Mock,
    ) -> None:
        validate.return_value = {"status": "exact_cohort_verified"}
        _, rows = MODULE.read_csv(self.heldout)
        heldout_fields = [
            "symbol", "book_id", "validation_only_score",
            *MODULE.HELDOUT_OPENING_FIELDS,
            "target_spread_ticks",
        ]
        for row in rows:
            row["validation_only_score"] = "123.5"
        MODULE.write_csv(self.heldout, heldout_fields, rows)
        manifest_path = self.evidence / "development_validation/heldout_run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["simulation_config"] = record(self.heldout)
        write_json(manifest_path, manifest)

        output = self.root / "projected-output"
        MODULE.run(self.args(output))
        output_fields, output_rows = MODULE.read_csv(
            output / "heldout_20200130_queue_reactive_case.csv"
        )
        self.assertEqual(output_fields, list(MODULE.RUNTIME_FIELDS))
        self.assertNotIn("validation_only_score", output_fields)
        self.assertEqual(
            float(output_rows[0]["fundamental_price_ticks"]), 10_010.0
        )

    @mock.patch.object(MODULE.cohort, "validate_symbols")
    def test_records_explicit_shared_dealer_amendment(
        self, validate: mock.Mock,
    ) -> None:
        validate.return_value = {"status": "exact_cohort_verified"}
        validated_hash = MODULE.sha256_file(self.executable)
        self.executable.write_bytes(b"shared-dealer treatment amendment")
        args = self.args(self.root / "amended-output")
        args.allow_post_validation_shared_dealer_amendment = True
        MODULE.run(args)
        payload = json.loads((
            args.output_root / "portable_queue_reactive_case.json"
        ).read_text(encoding="utf-8"))
        provenance = payload["executable_provenance"]
        self.assertTrue(provenance["post_validation_treatment_amendment"])
        self.assertEqual(
            provenance["validated_baseline_executable_sha256"], validated_hash
        )
        self.assertEqual(
            provenance["case_executable_sha256"],
            MODULE.sha256_file(self.executable),
        )
        self.assertEqual(
            provenance["amendment_scope"],
            "shared_dealer_counterfactual_and_observation_only",
        )
        self.assertFalse(provenance["ordinary_market_validation_claim_extended"])

    @mock.patch.object(MODULE.cohort, "validate_symbols")
    def test_rejects_changed_nonopening_heldout_field(
        self, validate: mock.Mock,
    ) -> None:
        validate.return_value = {"status": "exact_cohort_verified"}
        fields, rows = MODULE.read_csv(self.heldout)
        rows[0]["target_spread_ticks"] = "3"
        MODULE.write_csv(self.heldout, fields, rows)
        # Refresh only the held-out manifest record; the frozen deployment stays fixed.
        manifest_path = self.evidence / "development_validation/heldout_run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["simulation_config"] = record(self.heldout)
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(
            MODULE.PreparationError, "changes frozen field target_spread_ticks",
        ):
            MODULE.run(self.args(self.root / "changed-output"))


if __name__ == "__main__":
    unittest.main()
