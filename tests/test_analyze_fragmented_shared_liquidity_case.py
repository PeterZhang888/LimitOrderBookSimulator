#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
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
    "requested_shock_time_seconds": "11700.0",
    "requested_shock_fraction": "0.01",
    "requested_shock_top_depth_multiple": "1.0",
    "requested_shock_target_count": "0",
    "requested_shock_target_seed": "314159",
    "requested_local_inventory_limit": "100.0",
    "requested_capacity_threshold": "0.5",
    "shock_cluster_sha256": "b" * 64,
    "requested_shared_quote_relative": "0",
    "requested_shared_quote_multiplier": "1.0",
    "requested_shared_quote_levels": "1",
    "requested_local_mm_enabled": "1",
    "requested_value_agent_enabled": "1",
    "hawkes_activity_scale": "0.3",
    "local_mm_interval_ms": "1000.0",
    "local_mm_quantity_multiplier": "1.0",
    "local_mm_improvement_probability": "0.5",
    "local_mm_spread_elasticity": "1.0",
    "local_mm_max_improvement_probability": "0.75",
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


if __name__ == "__main__":
    unittest.main()
