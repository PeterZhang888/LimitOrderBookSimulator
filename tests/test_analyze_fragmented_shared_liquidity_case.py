#!/usr/bin/env python3
"""Focused provenance-pairing tests for the shared-liquidity analysis."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "shared_liquidity_analysis",
    ROOT / "scripts" / "analyze_fragmented_shared_liquidity_case.py",
)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


PAIRING_METADATA = {
    "executable_sha256": "d" * 64,
    "campaign_manifest": "/tmp/universe_input.json",
    "campaign_manifest_sha256": "e" * 64,
    "input_config_sha256": "a" * 64,
    "requested_window_ms": "1000.0",
    "requested_duration_seconds": "23400",
    "requested_stochastic_baseline_normalization_seconds": "23400.0",
    "requested_shock_time_seconds": "11700.0",
    "requested_shock_fraction": "0.10",
    "requested_shock_top_depth_multiple": "0.0",
    "requested_shock_reference_bid_depth_multiple": "3.0",
    "requested_shock_inventory_adverse": "1",
    "requested_shock_target_count": "0",
    "requested_shock_target_seed": "314159",
    "requested_local_inventory_limit": "800.0",
    "requested_capacity_threshold": "0.5",
    "requested_minimum_shared_quote_scale": "0.05",
    "shock_cluster_sha256": "b" * 64,
    "requested_shared_quote_relative": "1",
    "requested_shared_capacity_relative": "1",
    "requested_shared_quote_multiplier": "2.0",
    "requested_shared_quote_levels": "3",
    "requested_local_mm_enabled": "1",
    "requested_value_agent_enabled": "1",
    "hawkes_activity_scale": "0.3",
    "local_mm_interval_ms": "1000.0",
    "local_mm_quantity_multiplier": "1.0",
    "local_mm_improvement_probability": "0.25",
    "local_mm_spread_elasticity": "0.0",
    "local_mm_max_improvement_probability": "1.0",
    "shared_quote_quantity": "200",
    "value_agent_policy_sha256": "c" * 64,
}


def raw_case(row: dict[str, str]) -> object:
    return ANALYSIS.RawCase(
        row=row,
        seed=20200130,
        risk_limit=100.0,
        shock=False,
        metrics_path=pathlib.Path("unused.csv"),
    )


class PairingMetadataTest(unittest.TestCase):
    def test_records_all_runtime_and_shock_pairing_fields(self) -> None:
        metadata = ANALYSIS.ensure_common_metadata(
            [raw_case(dict(PAIRING_METADATA)), raw_case(dict(PAIRING_METADATA))]
        )
        for field in (
            "hawkes_activity_scale",
            "local_mm_interval_ms",
            "local_mm_quantity_multiplier",
            "local_mm_improvement_probability",
            "local_mm_spread_elasticity",
            "local_mm_max_improvement_probability",
            "shared_quote_quantity",
            "value_agent_policy_sha256",
            "requested_shock_target_count",
            "requested_shock_reference_bid_depth_multiple",
            "requested_shock_inventory_adverse",
            "requested_minimum_shared_quote_scale",
        ):
            self.assertEqual(metadata[field], PAIRING_METADATA[field])

    def test_rejects_each_newly_guarded_pairing_mismatch(self) -> None:
        for field in (
            "hawkes_activity_scale",
            "local_mm_interval_ms",
            "local_mm_quantity_multiplier",
            "local_mm_improvement_probability",
            "local_mm_spread_elasticity",
            "local_mm_max_improvement_probability",
            "shared_quote_quantity",
            "value_agent_policy_sha256",
            "requested_shock_target_count",
            "requested_shock_reference_bid_depth_multiple",
            "requested_shock_inventory_adverse",
            "requested_minimum_shared_quote_scale",
        ):
            with self.subTest(field=field):
                changed = dict(PAIRING_METADATA)
                changed[field] = "different"
                with self.assertRaisesRegex(
                    ANALYSIS.AnalysisError, rf"paired cases mix {field}"
                ):
                    ANALYSIS.ensure_common_metadata(
                        [raw_case(dict(PAIRING_METADATA)), raw_case(changed)]
                    )

    def test_rejects_missing_newly_guarded_pairing_value(self) -> None:
        changed = dict(PAIRING_METADATA)
        changed["value_agent_policy_sha256"] = ""
        with self.assertRaisesRegex(
            ANALYSIS.AnalysisError,
            "paired cases have a missing value for value_agent_policy_sha256",
        ):
            ANALYSIS.ensure_common_metadata([raw_case(changed)])

    def test_disabled_value_agent_has_no_policy_hash(self) -> None:
        disabled = dict(PAIRING_METADATA)
        disabled["requested_value_agent_enabled"] = "0"
        disabled["value_agent_policy_sha256"] = ""
        metadata = ANALYSIS.ensure_common_metadata([raw_case(disabled)])
        self.assertEqual(metadata["value_agent_policy_sha256"], "")


class SharedDealerCoverageTest(unittest.TestCase):
    def series(self, requested: float = 1.0, resting: float = 1.0):
        return {
            "9": {
                "shared_requested_two_sided_asset_fraction": requested,
                "shared_two_sided_active_asset_fraction": resting,
            },
            "10": {
                "shared_requested_two_sided_asset_fraction": requested,
                "shared_two_sided_active_asset_fraction": resting,
            },
        }

    def test_accepts_complete_policy_and_resting_coverage(self) -> None:
        ANALYSIS.validate_shared_dealer_coverage(
            self.series(), label="test", shock_time_seconds=10.0,
            lookback_seconds=2.0,
        )

    def test_requires_universal_requested_coverage(self) -> None:
        fraction = 1479.0 / 1480.0
        with self.assertRaisesRegex(ANALYSIS.AnalysisError, "every book"):
            ANALYSIS.validate_shared_dealer_coverage(
                self.series(requested=fraction), label="test",
                shock_time_seconds=10.0, lookback_seconds=2.0,
            )
    def test_accepts_material_resting_coverage_after_immediate_fills(self) -> None:
        fraction = 1479.0 / 1480.0
        ANALYSIS.validate_shared_dealer_coverage(
            self.series(resting=fraction), label="test",
            shock_time_seconds=10.0, lookback_seconds=2.0,
        )

    def test_rejects_resting_coverage_below_declared_threshold(self) -> None:
        with self.assertRaisesRegex(ANALYSIS.AnalysisError, "below 0.95"):
            ANALYSIS.validate_shared_dealer_coverage(
                self.series(resting=0.949), label="test",
                shock_time_seconds=10.0, lookback_seconds=2.0,
            )


class ShockDoseManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def case(self, rows: list[dict[str, object]]) -> object:
        path = self.root / "targets.csv"
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "asset_id", "symbol", "cluster_id", "is_shock_target",
                    "shock_enabled", "requested_sell_quantity", "mask_seed",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        total = sum(int(row["requested_sell_quantity"]) for row in rows)
        return ANALYSIS.RawCase(
            row={
                "shock_targets_csv": str(path),
                "shock_targets_csv_sha256": ANALYSIS.sha256_file(path),
                "shock_requested_quantity": str(total),
            },
            seed=1,
            risk_limit=100.0,
            shock=True,
            metrics_path=pathlib.Path("unused.csv"),
        )

    def test_reads_exact_asset_level_dose_vector(self) -> None:
        case = self.case([
            {
                "asset_id": 0, "symbol": "AAA", "cluster_id": 0,
                "is_shock_target": 1, "shock_enabled": 1,
                "requested_sell_quantity": 300, "mask_seed": 7,
            },
            {
                "asset_id": 1, "symbol": "BBB", "cluster_id": 1,
                "is_shock_target": 0, "shock_enabled": 1,
                "requested_sell_quantity": 0, "mask_seed": 7,
            },
        ])
        self.assertEqual(
            ANALYSIS.read_shock_manifest(case),
            ((0, "AAA", 300), (1, "BBB", 0)),
        )

    def test_reads_and_audits_inventory_adverse_mixed_sides(self) -> None:
        path = self.root / "inventory_adverse_targets.csv"
        fieldnames = [
            "asset_id", "symbol", "cluster_id", "is_shock_target",
            "shock_enabled", "requested_quantity",
            "requested_sell_quantity", "requested_buy_quantity",
            "shock_side", "pre_shock_shared_inventory", "direction_rule",
            "mask_seed",
        ]
        rows = [
            {
                "asset_id": 0, "symbol": "AAA", "cluster_id": 0,
                "is_shock_target": 1, "shock_enabled": 1,
                "requested_quantity": 300, "requested_sell_quantity": 300,
                "requested_buy_quantity": 0, "shock_side": "sell",
                "pre_shock_shared_inventory": 12,
                "direction_rule": "inventory_adverse", "mask_seed": 7,
            },
            {
                "asset_id": 1, "symbol": "BBB", "cluster_id": 1,
                "is_shock_target": 1, "shock_enabled": 1,
                "requested_quantity": 200, "requested_sell_quantity": 0,
                "requested_buy_quantity": 200, "shock_side": "buy",
                "pre_shock_shared_inventory": -9,
                "direction_rule": "inventory_adverse", "mask_seed": 7,
            },
        ]
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        case = ANALYSIS.RawCase(
            row={
                "shock_targets_csv": str(path),
                "shock_targets_csv_sha256": ANALYSIS.sha256_file(path),
                "shock_requested_quantity": "500",
            },
            seed=1,
            risk_limit=100.0,
            shock=True,
            metrics_path=pathlib.Path("unused.csv"),
        )
        self.assertEqual(
            ANALYSIS.read_shock_manifest(case),
            ((0, "AAA", 300), (1, "BBB", 200)),
        )
        rows[1]["shock_side"] = "sell"
        rows[1]["requested_sell_quantity"] = 200
        rows[1]["requested_buy_quantity"] = 0
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        invalid = ANALYSIS.RawCase(
            row={
                "shock_targets_csv": str(path),
                "shock_targets_csv_sha256": ANALYSIS.sha256_file(path),
                "shock_requested_quantity": "500",
            },
            seed=1,
            risk_limit=100.0,
            shock=True,
            metrics_path=pathlib.Path("unused.csv"),
        )
        with self.assertRaisesRegex(
            ANALYSIS.AnalysisError, "not inventory-adverse"
        ):
            ANALYSIS.read_shock_manifest(invalid)

    def test_rejects_hash_or_target_quantity_disagreement(self) -> None:
        case = self.case([{
            "asset_id": 0, "symbol": "AAA", "cluster_id": 0,
            "is_shock_target": 1, "shock_enabled": 1,
            "requested_sell_quantity": 0, "mask_seed": 7,
        }])
        with self.assertRaisesRegex(ANALYSIS.AnalysisError, "disagree"):
            ANALYSIS.read_shock_manifest(case)
        changed = dict(case.row)
        changed["shock_targets_csv_sha256"] = "0" * 64
        bad_hash = ANALYSIS.RawCase(
            row=changed, seed=1, risk_limit=100.0, shock=True,
            metrics_path=pathlib.Path("unused.csv"),
        )
        with self.assertRaisesRegex(ANALYSIS.AnalysisError, "SHA-256"):
            ANALYSIS.read_shock_manifest(bad_hash)

    def test_rejects_duplicate_or_noncanonical_asset_identity(self) -> None:
        duplicate = self.case([
            {
                "asset_id": 0, "symbol": "AAA", "cluster_id": 0,
                "is_shock_target": 1, "shock_enabled": 1,
                "requested_sell_quantity": 100, "mask_seed": 7,
            },
            {
                "asset_id": 0, "symbol": "BBB", "cluster_id": 1,
                "is_shock_target": 0, "shock_enabled": 1,
                "requested_sell_quantity": 0, "mask_seed": 7,
            },
        ])
        with self.assertRaisesRegex(ANALYSIS.AnalysisError, "duplicate"):
            ANALYSIS.read_shock_manifest(duplicate)

        noncanonical = self.case([{
            "asset_id": 1, "symbol": "AAA", "cluster_id": 0,
            "is_shock_target": 1, "shock_enabled": 1,
            "requested_sell_quantity": 100, "mask_seed": 7,
        }])
        with self.assertRaisesRegex(ANALYSIS.AnalysisError, "canonical"):
            ANALYSIS.read_shock_manifest(noncanonical)


class UniverseCohortIdentityTest(unittest.TestCase):
    RUNTIME_FIELDS = [
        "book_id", "symbol", "data_dir", "hawkes_rates_file",
        "fundamental_price_ticks", "fundamental_volatility_bps_sqrt_second",
        "fundamental_move_probability_per_second",
        "fundamental_conditional_kurtosis", "initial_best_bid_ticks",
        "initial_best_ask_ticks", "initial_best_bid_depth",
        "initial_best_ask_depth", "beta", "basket_weight",
        "market_maker_quote_quantity", "target_spread_ticks",
        "quote_improvement_probability", "target_mean_bid_depth",
        "target_mean_ask_depth",
    ]
    EXTENDED_RUNTIME_FIELDS = [
        *RUNTIME_FIELDS,
        "fundamental_log_variance_persistence",
        "fundamental_log_variance_std",
        "fundamental_order_flow_coupling",
    ]

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name).resolve()
        self.symbols = ANALYSIS.cohort.load_required_symbols(ROOT)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_config(self, symbols: tuple[str, ...]) -> pathlib.Path:
        path = self.root / "universe.csv"
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=self.RUNTIME_FIELDS)
            writer.writeheader()
            for book_id, symbol in enumerate(symbols):
                writer.writerow({
                    "book_id": book_id,
                    "symbol": symbol,
                    "data_dir": "/unused",
                    "hawkes_rates_file": "/unused/rates.csv",
                    "fundamental_price_ticks": 10_000,
                    "fundamental_volatility_bps_sqrt_second": 1.0,
                    "fundamental_move_probability_per_second": 0.1,
                    "fundamental_conditional_kurtosis": 3.0,
                    "initial_best_bid_ticks": 9_999,
                    "initial_best_ask_ticks": 10_001,
                    "initial_best_bid_depth": 100,
                    "initial_best_ask_depth": 100,
                    "beta": 1.0,
                    "basket_weight": 0.0,
                    "market_maker_quote_quantity": 100,
                    "target_spread_ticks": 2,
                    "quote_improvement_probability": 0.1,
                    "target_mean_bid_depth": 100,
                    "target_mean_ask_depth": 100,
                })
        return path

    def write_schema_five_manifest(
        self, config: pathlib.Path, *, cohort_identity: object,
        name: str,
    ) -> tuple[pathlib.Path, dict[str, str]]:
        fields, rows = [], []
        with config.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        for row in rows:
            row["fundamental_log_variance_persistence"] = "0.9"
            row["fundamental_log_variance_std"] = "0.2"
            row["fundamental_order_flow_coupling"] = "0.3"
        with config.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(
                output, fieldnames=self.EXTENDED_RUNTIME_FIELDS,
            )
            writer.writeheader()
            writer.writerows(rows)

        executable = self.root / "fragmented_mpi_lob"
        executable.write_bytes(b"schema contract executable")
        executable_hash = ANALYSIS.sha256_file(executable)
        config_hash = ANALYSIS.sha256_file(config)
        schema_hash = hashlib.sha256(json.dumps(
            self.EXTENDED_RUNTIME_FIELDS,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        runtime_schema = {
            "schema_version": 6,
            "fields": self.EXTENDED_RUNTIME_FIELDS,
            "sha256": schema_hash,
            "pooled_homeostatic_fields": [
                "target_spread_ticks", "target_mean_bid_depth",
                "target_mean_ask_depth",
            ],
            "latent_value_fields": [
                "fundamental_volatility_bps_sqrt_second",
                "fundamental_move_probability_per_second",
                "fundamental_conditional_kurtosis",
                "fundamental_log_variance_persistence",
                "fundamental_log_variance_std",
                "fundamental_order_flow_coupling",
            ],
            "frozen_training_derived_fields": [
                "target_spread_ticks", "target_mean_bid_depth",
                "target_mean_ask_depth",
                "fundamental_volatility_bps_sqrt_second",
                "fundamental_move_probability_per_second",
                "fundamental_conditional_kurtosis",
                "fundamental_log_variance_persistence",
                "fundamental_log_variance_std",
                "fundamental_order_flow_coupling",
            ],
            "heldout_target_files_used": False,
        }
        profile = {
            "profile_id": "systemic_liquidity_shock_queue_reactive",
            "experiment": "liquidity_shock_causality",
            "duration_seconds": 23400,
            "decision_window_ms": 1000.0,
            "cadence_windows_ms": [1000.0],
            "stochastic_baseline_normalization_seconds": 23400.0,
            "shock_time_seconds": 11700.0,
            "shock_fraction": 0.10,
            "shock_top_depth_multiple": 0.0,
            "shock_reference_bid_depth_multiple": 3.0,
            "shock_direction_rule": "inventory_adverse_at_left_limit",
            "shock_target_count": 0,
            "shock_target_seed": 314159,
            "local_inventory_limit": 800.0,
            "capacity_threshold": 0.5,
            "minimum_shared_quote_scale": 0.05,
            "shared_quote_relative": True,
            "shared_capacity_relative": True,
            "shared_quote_multiplier": 2.0,
            "shared_quote_levels": 3,
            "required_pre_shock_requested_two_sided_book_fraction": 1.0,
            "minimum_pre_shock_resting_two_sided_book_fraction": 0.95,
            "mechanism_preflight_lookback_seconds": 60.0,
            "minimum_pre_shock_economic_quote_scale": 0.25,
            "maximum_pre_shock_utilization": 0.90,
            "minimum_pre_shock_bbo_depth_participation": 0.05,
            "target_side_materiality_assessed_by_realized_shock_absorption": True,
            "minimum_nonzero_inventory_asset_fraction": 0.25,
            "minimum_shock_absorption_fraction": 0.025,
            "preflight_threshold_status": (
                "fixed_after_mechanism_pilot_before_financial_paths"
            ),
            "local_mm_spread_elasticity": 0.0,
            "local_mm_max_improvement_probability": 1.0,
            "primary_outcome": "relative_non_target_top_depth_deterioration",
            "secondary_outcome": "non_target_spread_deterioration_bps",
            "reporting_horizons_seconds": [1, 5, 30, 300, 1800],
            "uncoupled_capacity_control": (
                "asset_specific_equal_total_capacity"
            ),
            "asset_level_shock_dose_equality_required": True,
            "state_contingent_direction_rule_identical_across_mechanisms": True,
            "shock_fill_ownership_required": True,
            "truncated_full_prefix_equality_required": True,
            "shared_off_treatment_isolation_required": True,
        }
        profile_hash = hashlib.sha256(json.dumps(
            profile,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")).hexdigest()
        path = self.root / name
        path.write_text(json.dumps({
            "schema_version": 5,
            "calibration_provenance_mode": (
                "queue_reactive_training_freeze_and_heldout_validation"
            ),
            "universe_config": str(config),
            "universe_config_sha256": config_hash,
            "runtime_configuration_schema": runtime_schema,
            "cohort_identity": cohort_identity,
            "book_count": 1480,
            "case_executable": str(executable),
            "case_executable_sha256": executable_hash,
            "executable_provenance": {
                "validated_baseline_executable_sha256": executable_hash,
                "case_executable_sha256": executable_hash,
                "post_validation_treatment_amendment": False,
                "amendment_scope": "none",
                "ordinary_market_calibration_parameters_changed": False,
                "ordinary_market_validation_claim_extended": False,
            },
            "case_study_protocol": profile,
            "case_study_protocol_sha256": profile_hash,
        }), encoding="utf-8")
        metadata = {
            **PAIRING_METADATA,
            "campaign_manifest": str(path.resolve()),
            "campaign_manifest_sha256": ANALYSIS.sha256_file(path),
            "input_config_sha256": config_hash,
            "executable_sha256": executable_hash,
        }
        return path, metadata

    def write_manifest(
        self, config: pathlib.Path, *, cohort_identity: object,
    ) -> tuple[pathlib.Path, dict[str, str]]:
        config_hash = ANALYSIS.sha256_file(config)
        schema_hash = hashlib.sha256(json.dumps(
            self.RUNTIME_FIELDS, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        path = self.root / "universe_input.json"
        path.write_text(json.dumps({
            "schema_version": 4,
            "calibration_provenance_mode": "block_coordinate_certified_handoff",
            "universe_config": str(config),
            "universe_config_sha256": config_hash,
            "runtime_configuration_schema": {
                "schema_version": 5,
                "fields": self.RUNTIME_FIELDS,
                "sha256": schema_hash,
                "pooled_homeostatic_fields": [
                    "target_spread_ticks", "target_mean_bid_depth",
                    "target_mean_ask_depth",
                ],
                "latent_value_fields": [
                    "fundamental_volatility_bps_sqrt_second",
                    "fundamental_move_probability_per_second",
                    "fundamental_conditional_kurtosis",
                ],
                "frozen_training_derived_fields": [
                    "target_spread_ticks", "target_mean_bid_depth",
                    "target_mean_ask_depth",
                    "fundamental_volatility_bps_sqrt_second",
                    "fundamental_move_probability_per_second",
                    "fundamental_conditional_kurtosis",
                ],
                "heldout_target_files_used": False,
            },
            "cohort_identity": cohort_identity,
            "book_count": 1480,
        }), encoding="utf-8")
        metadata = {
            "campaign_manifest": str(path),
            "campaign_manifest_sha256": ANALYSIS.sha256_file(path),
            "input_config_sha256": config_hash,
        }
        return path, metadata

    def test_analysis_rejects_missing_persisted_cohort_identity(self) -> None:
        config = self.write_config(self.symbols)
        manifest, metadata = self.write_manifest(config, cohort_identity=None)
        with self.assertRaisesRegex(
            ANALYSIS.AnalysisError, "universe-input cohort identity is not an object",
        ):
            ANALYSIS.validate_universe_input(manifest, metadata)

    def test_analysis_rejects_same_size_substituted_universe(self) -> None:
        wrong_symbols = (*self.symbols[:-1], "ZZZZZZ")
        config = self.write_config(wrong_symbols)
        identity = ANALYSIS.cohort.validate_symbols(
            self.symbols, label="canonical test cohort", project_root=ROOT,
        )
        manifest, metadata = self.write_manifest(
            config, cohort_identity=identity,
        )
        with self.assertRaisesRegex(
            ANALYSIS.AnalysisError, "not the immutable 1480-symbol cohort",
        ):
            ANALYSIS.validate_universe_input(manifest, metadata)

    def test_schema_five_accepts_canonical_producer_identity(self) -> None:
        config = self.write_config(self.symbols)
        identity = ANALYSIS.cohort.validate_symbols(
            self.symbols, label="canonical test cohort", project_root=ROOT,
        )
        manifest, metadata = self.write_schema_five_manifest(
            config, cohort_identity=identity, name="schema5.json",
        )
        payload = ANALYSIS.validate_universe_input(manifest, metadata)
        self.assertEqual(
            payload["_analysis_cohort_identity_projection"], "none",
        )

    def test_schema_five_projects_only_omitted_identity_schema_tag(self) -> None:
        config = self.write_config(self.symbols)
        identity = dict(ANALYSIS.cohort.validate_symbols(
            self.symbols, label="canonical test cohort", project_root=ROOT,
        ))
        identity.pop("schema_version")
        manifest, metadata = self.write_schema_five_manifest(
            config, cohort_identity=identity, name="legacy-schema5.json",
        )
        payload = ANALYSIS.validate_universe_input(manifest, metadata)
        self.assertEqual(
            payload["_analysis_cohort_identity_projection"],
            "schema_version_1_supplied_for_portable_schema_5",
        )


if __name__ == "__main__":
    unittest.main()
