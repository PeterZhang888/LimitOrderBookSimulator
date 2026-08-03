#!/usr/bin/env python3
"""Mock-simulator tests for the queue-reactive calibration driver."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import stat
import sys
import tempfile
import unittest
from unittest import mock

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import calibrate_queue_reactive_model as driver  # noqa: E402

DATES = (
    "2019-01-30", "2019-03-27", "2019-07-30", "2019-10-30", "2019-12-30",
)
SYMBOLS = ("AAA", "BBB", "CCC", "DDD")
TARGET_VALUES = {
    "background_event_rate": 1.0,
    "mean_spread_ticks": 2.0,
    "mean_bid_depth": 100.0,
    "mean_ask_depth": 100.0,
    "mid_move_rate": 0.1,
    "return_variance": 1.0e-6,
    "return_kurtosis": 3.0,
    "absolute_return_acf1": 0.1,
    "two_sided_sample_fraction": 1.0,
}


def write_csv(path: pathlib.Path, fields: tuple[str, ...],
              rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class QueueReactiveCalibrationDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.cluster_map = self.root / "cluster_assignments.csv"
        write_csv(
            self.cluster_map, ("symbol", "cluster_id"),
            [{"symbol": symbol, "cluster_id": index // 2}
             for index, symbol in enumerate(SYMBOLS)],
        )
        self.configs = {day: self.write_config(day) for day in DATES}
        self.deployment = self.write_config("2019-12-30", name="deployment.csv")
        self.background = self.write_background_mapping()
        self.candidates = self.write_candidates()
        self.executable = self.write_mock_simulator()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_short_screen_pruning_keeps_promoted_raw_evidence(self) -> None:
        cluster_root = self.root / "stage2" / "cluster_0"
        runtime = {}
        for candidate_id in ("keep", "drop"):
            candidate_root = cluster_root / candidate_id
            config = candidate_root / "configs" / "20190130.csv"
            background = candidate_root / "background" / "20190130.csv"
            policy = candidate_root / "value_policy.csv"
            for path in (config, background, policy):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(candidate_id + "\n", encoding="utf-8")
            runtime[candidate_id] = (
                {DATES[0]: config}, {DATES[0]: background}, policy,
            )
        result = cluster_root / "cluster_result.json"
        driver.write_json(result, {"status": "recorded"})

        driver.prune_unpromoted_screen_artifacts(
            cluster_root=cluster_root,
            candidate_runtime=runtime,
            retained_candidate_ids=("keep",),
            cluster_result_path=result,
        )

        self.assertTrue((cluster_root / "keep").is_dir())
        self.assertFalse((cluster_root / "drop").exists())
        manifest = json.loads(
            (cluster_root / "screen_artifact_retention.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["retained_raw_candidate_ids"], ["keep"])
        self.assertEqual(manifest["pruned_reproducible_candidate_ids"], ["drop"])
        self.assertEqual(
            manifest["scientific_records_retained_in"]["sha256"],
            driver.sha256_file(result),
        )

    def write_config(
        self,
        day: str,
        *,
        name: str | None = None,
        symbols: tuple[str, ...] = SYMBOLS,
    ) -> pathlib.Path:
        path = self.root / (name or f"config_{day.replace('-', '')}.csv")
        fields = (
            "book_id", "symbol", "data_dir", "target_data_dir",
            "hawkes_rates_file",
            "fundamental_price_ticks", "initial_best_bid_ticks",
            "initial_best_ask_ticks", "initial_best_bid_depth",
            "initial_best_ask_depth", "beta", "basket_weight",
            "market_maker_quote_quantity", "target_spread_ticks",
            "fundamental_volatility_bps_sqrt_second",
            "fundamental_move_probability_per_second",
            "fundamental_conditional_kurtosis", "target_mean_bid_depth",
            "target_mean_ask_depth",
        )
        rows: list[dict[str, object]] = []
        for index, symbol in enumerate(symbols):
            data_dir = self.root / "dummy_data" / symbol.lower()
            data_dir.mkdir(parents=True, exist_ok=True)
            target_dir = (
                self.root / "dummy_targets" / day.replace("-", "")
                / symbol.lower()
            )
            target_dir.mkdir(parents=True, exist_ok=True)
            for filename in driver.MODEL_MARK_FILES:
                (data_dir / filename).write_text("value\n1\n", encoding="utf-8")
            write_csv(
                target_dir / "market_targets_mock.csv",
                ("name", "target", "scale", "weight"),
                [
                    {
                        "name": metric,
                        "target": value,
                        "scale": 1.0,
                        "weight": 1,
                    }
                    for metric, value in TARGET_VALUES.items()
                    if metric != "background_event_rate"
                ],
            )
            (target_dir / "itch_manifest_mock.json").write_text(
                json.dumps({
                    "symbol": symbol,
                    "trading_date": day,
                    "aggregation_duration_seconds": 23_400,
                    "distribution_observation_counts": {
                        event: 3_900 for event in driver.strict.BACKGROUND_EVENTS
                    },
                }),
                encoding="utf-8",
            )
            rates = data_dir / "rates.csv"
            rates.write_text("event_type,stationary_target_rate\n", encoding="utf-8")
            rows.append({
                "book_id": index, "symbol": symbol, "data_dir": data_dir,
                "target_data_dir": target_dir,
                "hawkes_rates_file": rates,
                "fundamental_price_ticks": 1000 + index,
                "initial_best_bid_ticks": 999 + index,
                "initial_best_ask_ticks": 1001 + index,
                "initial_best_bid_depth": 100 + index,
                "initial_best_ask_depth": 110 + index,
                "beta": 1, "basket_weight": 0,
                "market_maker_quote_quantity": 50, "target_spread_ticks": 2,
                "fundamental_volatility_bps_sqrt_second": 1,
                "fundamental_move_probability_per_second": 0.1,
                "fundamental_conditional_kurtosis": 3,
                "target_mean_bid_depth": 100, "target_mean_ask_depth": 100,
            })
        write_csv(path, fields, rows)
        return path

    def write_background_mapping(
        self,
        *,
        name: str = "symbol_policy_mapping.csv",
        symbols: tuple[str, ...] = SYMBOLS,
    ) -> pathlib.Path:
        policy = self.root / "queue_policy.csv"
        buy = self.root / "buy.csv"
        sell = self.root / "sell.csv"
        policy.write_text("dummy\n", encoding="utf-8")
        buy.write_text("ticks,weight\n1,1\n", encoding="utf-8")
        sell.write_text("ticks,weight\n1,1\n", encoding="utf-8")
        path = self.root / name
        write_csv(
            path,
            ("symbol", "cluster_id", "policy_file",
             "limit_buy_improvement_file", "limit_sell_improvement_file"),
            [{
                "symbol": symbol, "cluster_id": 0 if index < 2 else 1,
                "policy_file": policy,
                "limit_buy_improvement_file": buy,
                "limit_sell_improvement_file": sell,
            } for index, symbol in enumerate(symbols)],
        )
        return path

    def write_candidates(self) -> pathlib.Path:
        path = self.root / "candidates.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "stage1": {
                "seeds": [7], "survivor_count": 1,
                "local_mm_candidates": [
                    {"id": "bad_local", "enabled": True, "interval_ms": 2000,
                     "quantity_multiplier": 1, "improvement_probability": 0},
                    {"id": "good_local", "enabled": True, "interval_ms": 1000,
                     "quantity_multiplier": 1, "improvement_probability": 0,
                     "spread_elasticity": 0.5,
                     "max_improvement_probability": 0.75,
                     "must_promote_after_short_screen": True},
                ],
            },
            "stage2": {
                "seeds": [11],
                "value_policy_candidates": [
                    {"id": "value5", "enabled": True, "threshold_bps": 5,
                     "depth_participation": 0.05,
                     "gap_elasticity": 0.5,
                     "max_depth_participation": 0.5,
                     "trigger_mode": "news_impulse",
                     "maximum_news_rechecks": 0},
                    {"id": "value10", "enabled": True, "threshold_bps": 10,
                     "depth_participation": 0.05},
                ],
                "volatility_candidates": [
                    {"id": "vol_a", "fundamental_variance_scale": 0.5,
                     "fundamental_log_variance_persistence": 0.8,
                     "fundamental_log_variance_std": 0.4,
                     "fundamental_excess_kurtosis_share": 1.0,
                     "fundamental_tail_transmission_multiplier": 4.0},
                    {"id": "vol_b", "fundamental_variance_scale": 0.75,
                     "fundamental_log_variance_persistence": 0.9,
                     "fundamental_log_variance_std": 0.6,
                     "fundamental_excess_kurtosis_share": 1.0},
                ],
                "full_day_confirmation_count": 1,
                "full_day_confirmation_seeds": [11],
                "full_day_recheck_counts": [0, 1],
            },
            "stage3": {"seeds": [13]},
            "timeout_seconds": {
                "stage1": 10, "stage2": 10, "stage3": 10, "heldout": 10,
            },
        }), encoding="utf-8")
        return path

    def write_mock_simulator(self) -> pathlib.Path:
        path = self.root / "mock_fragmented_mpi_lob.py"
        path.write_text(
            """#!/usr/bin/env python3
import argparse, csv, pathlib
p=argparse.ArgumentParser(add_help=False)
for flag in ('--duration-seconds','--seed','--universe-config','--window-ms','--asset-summary-interval-ms','--asset-summary-csv','--background-model','--background-policy-csv','--local-mm-interval-ms','--local-mm-quantity-multiplier','--local-mm-improvement-probability','--local-mm-spread-elasticity','--local-mm-max-improvement-probability','--value-agent-policy-csv'):
 p.add_argument(flag)
p.add_argument('--disable-shared-mm',action='store_true')
p.add_argument('--disable-local-mm',action='store_true')
p.add_argument('--disable-value-agent',action='store_true')
a,_=p.parse_known_args()
if not a.disable_shared_mm or a.background_model != 'queue-reactive-v1': raise SystemExit(9)
if float(a.local_mm_interval_ms) == 2000.0: raise SystemExit(3)
with open(a.universe_config,newline='') as f: rows=list(csv.DictReader(f))
policies={}
if a.value_agent_policy_csv:
 with open(a.value_agent_policy_csv,newline='') as f: policies={r['symbol']:r for r in csv.DictReader(f)}
out=[]
for i,r in enumerate(rows):
 symbol=r['symbol']; cluster=0 if symbol in ('AAA','BBB') else 1
 spread=2.; variance=1e-6; kurtosis=3.; acf=.1; move=.1
 if policies:
  threshold=float(policies[symbol]['value_threshold_bps'])
  persistence=float(r['fundamental_log_variance_persistence']); std=float(r['fundamental_log_variance_std'])
  opt_t=5. if cluster==0 else 10.; opt_p=.8 if cluster==0 else .9; opt_s=.4 if cluster==0 else .6
  error=abs(threshold-opt_t)/5+abs(persistence-opt_p)*5+abs(std-opt_s)*2
  spread=2*(1+error); variance=1e-6*(1+error); kurtosis=3*(1+error); acf=.1+min(.5,error*.05); move=.1+min(.5,error*.05)
 out.append({'asset_id':i,'symbol':symbol,'sample_count':int(a.duration_seconds),'expected_sample_count':int(a.duration_seconds),'invalid_sample_count':0,'structurally_valid':1,'background_event_rate':1.,'mean_spread_ticks':spread,'mean_bid_depth':100.,'mean_ask_depth':100.,'mid_move_rate':move,'return_variance':variance,'return_kurtosis':kurtosis,'absolute_return_acf1':acf,'two_sided_sample_fraction':1.})
path=pathlib.Path(a.asset_summary_csv); path.parent.mkdir(parents=True,exist_ok=True)
with path.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
""", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    @staticmethod
    def fake_targets(config: driver.ConfigTable, *, day: str,
                     symbols: tuple[str, ...], duration: int):
        del config, day, duration
        return {
            symbol: {
                metric: driver.legacy.TargetMoment(value, 1.0, 1.0)
                for metric, value in TARGET_VALUES.items()
            } for symbol in symbols
        }

    @staticmethod
    def fake_strict(arguments):
        arguments = list(arguments)
        output = pathlib.Path(arguments[arguments.index("--output-dir") + 1])
        report_path = output / "strict_validation_report.json"
        payload = {
            "schema_version": 1,
            "evaluation_role": arguments[
                arguments.index("--evaluation-role") + 1
            ],
            "gate_protocol": (
                arguments[arguments.index("--gate-protocol") + 1]
                if "--gate-protocol" in arguments else "strict-nine-v1"
            ),
            "passed": True,
            "mock": True,
        }
        driver.write_json(report_path, payload)
        return payload, report_path

    @staticmethod
    def failing_strict(arguments):
        arguments = list(arguments)
        output = pathlib.Path(arguments[arguments.index("--output-dir") + 1])
        report_path = output / "strict_validation_report.json"
        payload = {"schema_version": 1, "passed": False, "mock": True}
        driver.write_json(report_path, payload)
        return payload, report_path

    def training_args(self, output: pathlib.Path) -> argparse.Namespace:
        argv = [
            "train", "--executable", str(self.executable),
            "--background-policy", str(self.background),
            "--deployment-config", str(self.deployment),
            "--cluster-map", str(self.cluster_map),
            "--candidate-config", str(self.candidates),
            "--output-root", str(output),
        ]
        for day in DATES:
            argv.extend(("--training-config", f"{day}={self.configs[day]}"))
        return driver.parser().parse_args(argv)

    def run_training(self, output: pathlib.Path) -> dict[str, object]:
        with (
            mock.patch.object(driver, "load_targets", side_effect=self.fake_targets),
            mock.patch.object(
                driver, "run_strict_evaluation", side_effect=self.fake_strict,
            ),
        ):
            return driver.train(self.training_args(output))

    def run_full_universe_expansion(
        self,
        *,
        selection_freeze: pathlib.Path,
        output: pathlib.Path,
        strict_side_effect=None,
        extra_args: tuple[str, ...] = (),
    ) -> dict[str, object]:
        full_symbols = SYMBOLS + ("EEE", "FFF")
        full_cluster_map = self.root / "full_cluster_assignments.csv"
        write_csv(
            full_cluster_map,
            ("symbol", "cluster_id"),
            [
                {"symbol": symbol, "cluster_id": 0 if index < 2 else 1}
                for index, symbol in enumerate(full_symbols)
            ],
        )
        full_deployment = self.write_config(
            "2019-12-30", name="full_deployment.csv", symbols=full_symbols,
        )
        full_background = self.write_background_mapping(
            name="full_symbol_policy_mapping.csv", symbols=full_symbols,
        )
        full_configs = {
            day: self.write_config(
                day,
                name=f"full_config_{day.replace('-', '')}.csv",
                symbols=full_symbols,
            )
            for day in DATES
        }
        argv = [
            "expand-full-universe",
            "--freeze-record", str(selection_freeze),
            "--full-deployment-config", str(full_deployment),
            "--full-background-policy", str(full_background),
            "--full-cluster-map", str(full_cluster_map),
            "--output-root", str(output),
        ]
        for day in DATES:
            argv.extend(("--training-config", f"{day}={full_configs[day]}"))
        argv.extend(extra_args)
        args = driver.parser().parse_args(argv)
        if strict_side_effect is None:
            strict_side_effect = self.fake_strict
        with (
            mock.patch.object(
                driver, "load_targets", side_effect=self.fake_targets,
            ),
            mock.patch.object(
                driver,
                "run_strict_evaluation",
                side_effect=strict_side_effect,
            ),
        ):
            return driver.expand_full_universe(args)

    def test_three_stages_promote_and_freeze_deterministic_evidence(self) -> None:
        output = self.root / "training"
        result = self.run_training(output)
        freeze_path = pathlib.Path(result["training_selection_freeze"])
        self.assertEqual(
            result["training_selection_freeze_sha256"],
            driver.sha256_file(freeze_path),
        )
        freeze = json.loads(freeze_path.read_text())
        self.assertEqual(
            freeze["status"],
            "stratified_training_selection_frozen_pending_full_universe",
        )
        self.assertFalse(freeze["heldout_execution_authorized"])

        self.assertFalse(freeze["heldout_inputs_read"])
        self.assertTrue(freeze["ordinary_market_shared_mm_disabled"])
        self.assertEqual(freeze["selection"]["local_candidate"]["identifier"], "good_local")
        self.assertEqual(
            freeze["selection"]["local_candidate"]["spread_elasticity"],
            0.5,
        )
        self.assertEqual(
            freeze["selection"]["local_candidate"][
                "max_improvement_probability"
            ],
            0.75,
        )
        self.assertEqual(
            freeze["selection"]["value_by_cluster"]["0"]["identifier"],
            "value5_rechecks_0",
        )
        self.assertEqual(
            freeze["selection"]["value_by_cluster"]["0"]["gap_elasticity"],
            0.5,
        )
        self.assertEqual(
            freeze["selection"]["value_by_cluster"]["0"][
                "max_depth_participation"
            ],
            0.5,
        )
        self.assertEqual(freeze["selection"]["value_by_cluster"]["1"]["identifier"], "value10")
        self.assertEqual(
            freeze["selection"]["value_by_cluster"]["1"]["gap_elasticity"],
            0.0,
        )
        self.assertEqual(
            freeze["selection"]["value_by_cluster"]["1"][
                "max_depth_participation"
            ],
            1.0,
        )
        self.assertEqual(freeze["selection"]["volatility_by_cluster"]["0"]["identifier"], "vol_a")
        self.assertEqual(freeze["selection"]["volatility_by_cluster"]["1"]["identifier"], "vol_b")
        refinement_source = freeze["selection"][
            "training_only_global_refinement"
        ]
        self.assertFalse(refinement_source["heldout_inputs_read"])
        self.assertEqual(refinement_source["evaluation_seeds"], [11])
        cluster_zero = json.loads((
            output / "stage2/local_good_local/cluster_0/cluster_result.json"
        ).read_text())
        news_confirmation_ids = {
            row["candidate_id"]
            for row in cluster_zero["confirmation_candidates"]
            if row["screen_candidate_id"].startswith("value_value5__")
        }
        self.assertTrue(any(value.endswith("__rechecks_0")
                            for value in news_confirmation_ids))
        self.assertTrue(any(value.endswith("__rechecks_1")
                            for value in news_confirmation_ids))
        self.assertEqual(cluster_zero["full_day_recheck_counts"], [0, 1])
        self.assertEqual(
            cluster_zero["confirmation_cap_scope"],
            "expanded_full_day_variants_total",
        )
        self.assertLessEqual(
            cluster_zero["retained_confirmation_variant_count"], 40,
        )
        self.assertEqual(
            set(cluster_zero["retained_confirmation_variant_ids"]),
            {
                row["candidate_id"]
                for row in cluster_zero["confirmation_candidates"]
            },
        )
        refinement = json.loads((
            output / "stage2/local_good_local/global_refinement/"
            "global_refinement_result.json"
        ).read_text())
        self.assertEqual(refinement["beam_width"], 3)
        self.assertEqual(refinement["alternatives_per_cluster_limit"], 4)
        self.assertEqual(refinement["stage3_finalist_limit"], 3)
        self.assertFalse(refinement["heldout_inputs_read"])
        self.assertGreaterEqual(refinement["evaluated_assignment_count"], 1)
        manifest = json.loads(
            (output / "training_inputs_manifest.json").read_text()
        )
        protocol = manifest["training_only_global_refinement_protocol"]
        self.assertTrue(protocol["strict_thresholds_immutable"])
        self.assertFalse(protocol["heldout_inputs_read"])
        self.assertEqual(
            freeze["training_only_global_refinement_protocol"], protocol,
        )
        self.assertEqual(
            freeze["training_inputs_manifest"]["sha256"],
            driver.sha256_file(output / "training_inputs_manifest.json"),
        )
        local_protocol = manifest["parsed_local_candidate_protocol"]
        self.assertEqual(local_protocol["candidate_count"], 2)
        self.assertEqual(
            local_protocol["spread_adaptive_improvement"][
                "adaptive_candidate_ids"
            ],
            ["good_local"],
        )
        transitive_paths = {
            row["path"] for row in freeze["transitive_runtime_artifacts"][
                "entries"
            ]
        }
        self.assertTrue(any(
            "/global_refinement/" in path
            and path.endswith("fragmented_asset_summary.csv")
            for path in transitive_paths
        ))
        with pathlib.Path(
            freeze["frozen_artifacts"]["deployment_config"]["path"]
        ).open(newline="", encoding="utf-8") as source:
            frozen_rows = {row["symbol"]: row for row in csv.DictReader(source)}
        self.assertAlmostEqual(
            float(frozen_rows["AAA"][
                "fundamental_volatility_bps_sqrt_second"
            ]),
            0.5 ** 0.5,
        )
        self.assertAlmostEqual(
            float(frozen_rows["AAA"]["fundamental_conditional_kurtosis"]),
            9.0 * math.exp(-0.4 * 0.4),
        )
        stage1 = json.loads((output / "stage1/stage1_result.json").read_text())
        bad = next(row for row in stage1["candidates"] if row["candidate_id"] == "bad_local")
        self.assertFalse(bad["score"]["eligible"])
        self.assertEqual(stage1["promoted_candidate_ids"], ["good_local"])
        self.assertEqual(
            stage1["mandatory_long_horizon_candidate_ids"], ["good_local"],
        )
        for day_records in freeze["stage3_runs"].values():
            for record in day_records:
                command = record["command"]
                self.assertIn("--disable-shared-mm", command)
                self.assertEqual(command[command.index("--background-model") + 1], "queue-reactive-v1")
                self.assertEqual(
                    float(command[
                        command.index("--local-mm-spread-elasticity") + 1
                    ]),
                    0.5,
                )
                self.assertEqual(
                    float(command[
                        command.index(
                            "--local-mm-max-improvement-probability"
                        ) + 1
                    ]),
                    0.75,
                )
                self.assertEqual(record["command_sha256"], driver.sha256_json(command))
        strict_args = json.loads(
            (output / "strict_training_evaluator_arguments.json").read_text()
        )["arguments"]
        self.assertEqual(strict_args.count("--expected-date"), 5)
        self.assertEqual(strict_args.count("--sim-summary"), 5)

        with (output / "frozen_model/value_policy.csv").open(
            newline="", encoding="utf-8",
        ) as source:
            frozen_policy = {
                row["symbol"]: row for row in csv.DictReader(source)
            }
        self.assertEqual(
            float(frozen_policy["AAA"]["value_gap_elasticity"]), 0.5,
        )
        self.assertEqual(
            float(frozen_policy["AAA"]["value_max_depth_participation"]),
            0.5,
        )
        self.assertEqual(
            float(frozen_policy["CCC"]["value_gap_elasticity"]), 0.0,
        )
        self.assertEqual(
            float(frozen_policy["CCC"]["value_max_depth_participation"]),
            1.0,
        )

    def test_six_component_certificate_recomputes_authoritative_gate(
        self,
    ) -> None:
        evidence = self.root / "six_component_evidence"
        report_path = evidence / "strict_validation_report.json"
        score_path = evidence / "marketwide_metric_scores.csv"
        residual_path = evidence / "symbol_residuals.csv"
        driver.write_json(report_path, {
            "evaluation_role": "training_fit",
            "passed": False,
            "gate": {"gate_id": "strict_queue_reactive_fit_gate_v1"},
            "expected_dates": list(DATES),
            "date_results": [
                {"date": day, "passed": False} for day in DATES
            ],
        })
        residual_rows: list[dict[str, object]] = []
        score_rows: list[dict[str, object]] = []
        for day in DATES:
            for metric in driver.strict.METRICS:
                score_rows.append({
                    "date": day,
                    "metric": metric,
                    "score": 0.0,
                })
                for symbol in SYMBOLS:
                    residual_rows.append({
                        "date": day,
                        "symbol": symbol,
                        "metric": metric,
                        "target": 100.0,
                        "simulated_seed_mean": 100.0,
                        "robust_residual": 0.0,
                    })
        write_csv(
            score_path,
            ("date", "metric", "score"),
            score_rows,
        )
        write_csv(
            residual_path,
            (
                "date", "symbol", "metric", "target",
                "simulated_seed_mean", "robust_residual",
            ),
            residual_rows,
        )
        certificate_path = evidence / "six_component_training_certificate.json"
        driver.write_json(certificate_path, {
            "schema_version": 1,
            "status": "six_component_training_adequacy_passed",
            "passed": True,
            "protocol": {
                "classification": "retrospective_development_reanalysis",
                "primary_dimensions": [
                    "activity", "spread", "combined_top_depth",
                    "mid_move_rate", "return_variance",
                    "absolute_return_acf1",
                ],
                "marketwide_metrics_are_authoritative": True,
                "return_kurtosis_role": "diagnostic_only",
                "cluster_metric_role": "diagnostic_only",
                "acf_distribution_moment_role": "diagnostic_only",
                "structural_book_checks_remain_mandatory": True,
            },
            "date_results": [
                {"date": day, "passed": True} for day in DATES
            ],
            "source_evidence": {
                "strict_report": str(report_path),
                "strict_report_sha256": driver.sha256_file(report_path),
                "marketwide_scores": str(score_path),
                "marketwide_scores_sha256": driver.sha256_file(score_path),
            },
        })

        verified = driver.verified_six_component_certificate(
            certificate_path
        )
        self.assertEqual(len(verified["authoritative_reanalysis"]), 5)
        self.assertTrue(all(
            record["passed"]
            for record in verified["authoritative_reanalysis"]
        ))

        residual_rows[0]["robust_residual"] = 1.0
        write_csv(
            residual_path,
            (
                "date", "symbol", "metric", "target",
                "simulated_seed_mean", "robust_residual",
            ),
            residual_rows,
        )
        with self.assertRaisesRegex(
            driver.CalibrationDriverError, "disagrees",
        ):
            driver.verified_six_component_certificate(certificate_path)

    def test_interface_prevents_training_leakage_and_records_launcher_ranks(self) -> None:
        args = self.training_args(self.root / "training_contract")
        self.assertEqual(args.background_policy.resolve(), self.background.resolve())
        self.assertFalse(hasattr(args, "heldout_date"))
        launcher = driver.parse_launcher("mpirun -np 2")
        self.assertEqual(driver.launcher_rank_count(launcher), 2)
        self.assertEqual(driver.launcher_rank_count(()), 1)
        with self.assertRaises(driver.CalibrationDriverError):
            driver.parse_launcher("srun --ntasks=2 -n 4")

    def test_parallel_run_matrix_preserves_order_and_mpi_rank_metadata(self) -> None:
        local = driver.LocalCandidate(
            identifier="local", enabled=True, interval_ms=1000.0,
            quantity_multiplier=1.0, improvement_probability=0.0,
        )
        observed_ranks: list[int] = []

        def fake_execute(**kwargs):
            observed_ranks.append(kwargs["mpi_ranks"])
            return {
                "success": True,
                "summary_path": str(kwargs["run_dir"] / "fragmented_asset_summary.csv"),
                "command": list(kwargs["command"]),
            }

        selected_days = {day: self.configs[day] for day in DATES[:2]}
        with mock.patch.object(driver, "execute_run", side_effect=fake_execute):
            runs = driver.run_candidate_matrix(
                executable=self.executable,
                launcher=driver.parse_launcher("mpirun -np 2"),
                configs=selected_days,
                background_policies={day: self.background for day in selected_days},
                value_policy=None,
                local=local,
                duration=300,
                seeds=(7, 11),
                output_root=self.root / "parallel_matrix",
                timeout_seconds=10,
                run_workers=4,
            )
        self.assertEqual(list(runs), list(DATES[:2]))
        for day in DATES[:2]:
            self.assertEqual(
                [record["base_seed"] for record in runs[day]], [7, 11]
            )
        self.assertEqual(sorted(observed_ranks), [2, 2, 2, 2])

    def test_session_seed_is_date_specific_and_candidate_independent(self) -> None:
        first = driver.session_model_seed(1729, "2019-01-30")
        self.assertEqual(
            first, driver.session_model_seed(1729, "2019-01-30")
        )
        self.assertNotEqual(
            first, driver.session_model_seed(1729, "2019-03-27")
        )
        self.assertNotEqual(
            first, driver.session_model_seed(7919, "2019-01-30")
        )
        self.assertGreater(first, 0)
        self.assertLess(first, 1 << 63)
        with self.assertRaises(driver.CalibrationDriverError):
            driver.session_model_seed(1729, "2019-1-30")

    def test_full_day_confirmation_preserves_super_unit_news_regime(self) -> None:
        def record(identifier: str, *, enabled: bool, participation: float,
                   scale: float) -> dict[str, object]:
            return {
                "candidate_id": identifier,
                "value_candidate": {
                    "enabled": enabled,
                    "depth_participation": participation,
                },
                "volatility_candidate": {"variance_scale": scale},
            }

        eligible = [
            record("screen_leader", enabled=True, participation=0.02,
                   scale=0.10),
            record("value_off", enabled=False, participation=0.02,
                   scale=0.0),
            record("middle", enabled=True, participation=0.05, scale=0.50),
            record("unit", enabled=True, participation=0.10, scale=0.75),
            record("super_unit", enabled=True, participation=0.20,
                   scale=1.50),
        ]
        promoted = driver.diverse_confirmation_candidates(
            eligible, global_count=1,
        )
        promoted_ids = {
            str(candidate["candidate_id"]) for candidate in promoted
        }
        self.assertIn("super_unit", promoted_ids)

    def test_confirmation_frontier_preserves_mode_specific_tail_candidate(
        self,
    ) -> None:
        def record(identifier: str, *, mode: str, aggregate: float,
                   variance_score: float) -> dict[str, object]:
            return {
                "candidate_id": identifier,
                "value_candidate": {
                    "enabled": True,
                    "trigger_mode": mode,
                    "depth_participation": 0.05,
                },
                "volatility_candidate": {"variance_scale": 1.0},
                "score": {
                    "eligible": True,
                    "aggregate_score": aggregate,
                    "day_results": [{
                        "date": day,
                        "metric_scores": [{
                            "metric": metric,
                            "score": (
                                variance_score
                                if metric == "return_variance" else 1.0
                            ),
                        } for metric in driver.STAGE2_METRICS],
                    } for day in DATES],
                },
            }

        candidates = [
            record("news_smooth", mode="news_impulse", aggregate=0.5,
                   variance_score=4.0),
            record("news_variance_tail", mode="news_impulse", aggregate=1.0,
                   variance_score=0.5),
            record("periodic_smooth", mode="periodic_gap", aggregate=0.6,
                   variance_score=1.0),
        ]
        selected = driver.diverse_confirmation_candidates(
            candidates, global_count=1, candidate_cap=24,
        )
        identifiers = {str(row["candidate_id"]) for row in selected}
        self.assertIn("news_variance_tail", identifiers)
        self.assertIn("periodic_smooth", identifiers)

    def test_mandatory_full_day_joint_candidate_precedes_short_screen_leader(
        self,
    ) -> None:
        def record(identifier: str) -> dict[str, object]:
            return {
                "candidate_id": identifier,
                "value_candidate": {
                    "enabled": True,
                    "trigger_mode": "news_impulse",
                    "depth_participation": 0.05,
                },
                "volatility_candidate": {"variance_scale": 1.0},
                "score": {
                    "eligible": True,
                    "aggregate_score": 1.0,
                    "day_results": [],
                },
            }

        selected = driver.diverse_confirmation_candidates(
            [record("smooth"), record("long_horizon_anchor")],
            global_count=1,
            candidate_cap=2,
            mandatory_candidate_ids=("long_horizon_anchor",),
        )
        self.assertEqual(selected[0]["candidate_id"], "long_horizon_anchor")
        with self.assertRaisesRegex(
            driver.CalibrationDriverError, "not structurally eligible",
        ):
            driver.diverse_confirmation_candidates(
                [record("smooth")],
                global_count=1,
                mandatory_candidate_ids=("missing",),
            )

    def test_cluster_confirmation_gate_requires_every_metric_on_every_date(
        self,
    ) -> None:
        day_results = []
        for day in DATES:
            day_results.append({
                "date": day,
                "metric_scores": [
                    {"metric": metric, "score": 1.0}
                    for metric in driver.STAGE2_METRICS
                ],
            })
        record = {
            "score": {"eligible": True, "day_results": day_results},
        }
        audit = driver.cluster_confirmation_gate_audit(record)
        self.assertTrue(audit["passed"])
        self.assertEqual(
            audit["maximum_metric_score"],
            driver.strict.MAX_CLUSTER_METRIC_SCORE,
        )

        # One failure on one training date must disqualify the alternative;
        # the search must not average it away across dates.
        failed = json.loads(json.dumps(record))
        metric = driver.strict.CLUSTER_GATE_METRICS[0]
        failed_row = next(
            row for row in failed["score"]["day_results"][2]["metric_scores"]
            if row["metric"] == metric
        )
        failed_row["score"] = driver.strict.MAX_CLUSTER_METRIC_SCORE + 0.01
        failed_audit = driver.cluster_confirmation_gate_audit(failed)
        self.assertFalse(failed_audit["passed"])
        self.assertIn(DATES[2], failed_audit["failure_reasons"][0])

    def test_full_day_rechecks_expand_news_only_after_screen(self) -> None:
        news = driver.ValueCandidate(
            "news", True, 5.0, 0.05, "news_impulse", 0, 0.5, 0.75,
        )
        variants = driver.full_day_value_variants(news, (4, 0, 2))
        self.assertEqual(
            [candidate.identifier for candidate in variants],
            ["news_rechecks_0", "news_rechecks_2", "news_rechecks_4"],
        )
        self.assertEqual(
            [candidate.maximum_news_rechecks for candidate in variants],
            [0, 2, 4],
        )
        self.assertEqual(
            [candidate.gap_elasticity for candidate in variants],
            [0.5, 0.5, 0.5],
        )
        self.assertEqual(
            [candidate.max_depth_participation for candidate in variants],
            [0.75, 0.75, 0.75],
        )
        periodic = driver.ValueCandidate(
            "periodic", True, 5.0, 0.05, "periodic_gap", 0,
        )
        self.assertEqual(
            driver.full_day_value_variants(periodic, (0, 2, 4)),
            (periodic,),
        )

    def test_expanded_variant_cap_preserves_bases_and_recheck_curve(self) -> None:
        screened = []
        for index in range(3):
            screened.append({
                "candidate_id": f"base_{index}",
                "value_candidate": driver.ValueCandidate(
                    f"news_{index}", True, 5.0 + index, 0.05,
                    "news_impulse", 0,
                ).__dict__,
            })
        generated, retained = driver.plan_full_day_confirmation_variants(
            screened, recheck_counts=(0, 1, 2), total_cap=5,
        )
        self.assertEqual(len(generated), 9)
        self.assertEqual(len(retained), 5)
        self.assertEqual(
            [item[0] for item in retained[:3]],
            [
                "base_0__rechecks_0",
                "base_0__rechecks_1",
                "base_0__rechecks_2",
            ],
        )
        self.assertEqual(
            {str(item[1]["candidate_id"]) for item in retained},
            {"base_0", "base_1", "base_2"},
        )
        self.assertEqual(
            {item[2].maximum_news_rechecks for item in retained},
            {0, 1, 2},
        )

        realistic_screened = []
        for index in range(12):
            realistic_screened.append({
                "candidate_id": f"frontier_{index}",
                "value_candidate": driver.ValueCandidate(
                    f"frontier_news_{index}", True, 5.0 + index, 0.05,
                    "news_impulse", 0,
                ).__dict__,
            })
        _, realistic_retained = (
            driver.plan_full_day_confirmation_variants(
                realistic_screened,
                recheck_counts=(0, 1, 2, 4, 8),
                total_cap=40,
            )
        )
        self.assertEqual(len(realistic_retained), 40)
        self.assertTrue(any(
            str(item[1]["candidate_id"]) != "frontier_0"
            and item[2].maximum_news_rechecks > 0
            for item in realistic_retained
        ))

    def test_global_refinement_protocol_defaults_and_hard_caps(self) -> None:
        protocol = driver.load_candidate_protocol(self.candidates)
        self.assertEqual(protocol.global_alternatives_per_cluster, 4)
        self.assertEqual(protocol.global_beam_width, 3)
        self.assertEqual(protocol.global_stage3_finalist_count, 3)
        self.assertEqual(protocol.full_day_recheck_counts, (0, 1))
        self.assertEqual(protocol.full_day_confirmation_candidate_cap, 40)

        default_payload = json.loads(self.candidates.read_text())
        del default_payload["stage2"]["full_day_recheck_counts"]
        defaults_path = self.root / "protocol_defaults.json"
        defaults_path.write_text(json.dumps(default_payload), encoding="utf-8")
        defaults = driver.load_candidate_protocol(defaults_path)
        self.assertEqual(defaults.full_day_recheck_counts, (0,))

        payload = json.loads(self.candidates.read_text())
        payload["stage2"]["global_refinement_beam_width"] = 4
        invalid = self.root / "invalid_beam_width.json"
        invalid.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(
            driver.CalibrationDriverError, "beam_width must not exceed 3",
        ):
            driver.load_candidate_protocol(invalid)

        cap_payload = json.loads(self.candidates.read_text())
        cap_payload["stage2"]["full_day_confirmation_candidate_cap"] = 49
        invalid_cap = self.root / "invalid_confirmation_cap.json"
        invalid_cap.write_text(json.dumps(cap_payload), encoding="utf-8")
        with self.assertRaisesRegex(
            driver.CalibrationDriverError,
            "confirmation_candidate_cap must not exceed 48",
        ):
            driver.load_candidate_protocol(invalid_cap)

    def test_gap_adaptive_value_controls_default_and_fail_closed(self) -> None:
        protocol = driver.load_candidate_protocol(self.candidates)
        by_id = {candidate.identifier: candidate for candidate in protocol.value}
        self.assertEqual(by_id["value5"].gap_elasticity, 0.5)
        self.assertEqual(by_id["value5"].max_depth_participation, 0.5)

        # This fixture intentionally omits both new fields.  Zero elasticity
        # bypasses scaling exactly, so the nonbinding default cap of one
        # retains the legacy constant-size policy bit-for-bit.
        self.assertEqual(by_id["value10"].gap_elasticity, 0.0)
        self.assertEqual(by_id["value10"].max_depth_participation, 1.0)
        legacy = driver.ValueCandidate(
            "legacy", True, 10.0, 0.2, "news_impulse", 0,
        )
        self.assertEqual(legacy.gap_elasticity, 0.0)
        self.assertEqual(legacy.max_depth_participation, 1.0)

        invalid_cases = (
            ("gap_elasticity", -0.1, "invalid value controls"),
            ("max_depth_participation", 0.01, "invalid value controls"),
            ("max_depth_participation", 1.01, "invalid value controls"),
        )
        for index, (field, value, message) in enumerate(invalid_cases):
            with self.subTest(field=field, value=value):
                payload = json.loads(self.candidates.read_text())
                payload["stage2"]["value_policy_candidates"][0].update({
                    "enabled": True,
                    field: value,
                })
                path = self.root / f"invalid_gap_control_{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(
                    driver.CalibrationDriverError, message,
                ):
                    driver.load_candidate_protocol(path)

        zero_threshold_payload = json.loads(self.candidates.read_text())
        zero_threshold_candidate = zero_threshold_payload["stage2"][
            "value_policy_candidates"
        ][0]
        zero_threshold_candidate["threshold_bps"] = 0.0
        zero_threshold_path = self.root / "gap_with_zero_threshold.json"
        zero_threshold_path.write_text(
            json.dumps(zero_threshold_payload), encoding="utf-8",
        )
        with self.assertRaisesRegex(
            driver.CalibrationDriverError, "invalid value controls",
        ):
            driver.load_candidate_protocol(zero_threshold_path)

        disabled_payload = json.loads(self.candidates.read_text())
        disabled_candidate = disabled_payload["stage2"][
            "value_policy_candidates"
        ][0]
        disabled_candidate["enabled"] = False
        disabled_candidate["gap_elasticity"] = 0.5
        disabled_path = self.root / "disabled_gap_elasticity.json"
        disabled_path.write_text(
            json.dumps(disabled_payload), encoding="utf-8",
        )
        with self.assertRaisesRegex(
            driver.CalibrationDriverError,
            "disabled policy must use zero gap elasticity",
        ):
            driver.load_candidate_protocol(disabled_path)

    def test_tail_transmission_controls_are_bounded_and_identifiable(self) -> None:
        protocol = driver.load_candidate_protocol(self.candidates)
        by_id = {
            candidate.identifier: candidate
            for candidate in protocol.volatility
        }
        self.assertEqual(by_id["vol_a"].tail_transmission_multiplier, 4.0)
        self.assertEqual(by_id["vol_b"].tail_transmission_multiplier, 1.0)

        for index, value in enumerate((0.99, 8.01)):
            payload = json.loads(self.candidates.read_text())
            payload["stage2"]["volatility_candidates"][0][
                "fundamental_tail_transmission_multiplier"
            ] = value
            path = self.root / f"invalid_tail_multiplier_{index}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                driver.CalibrationDriverError, "invalid volatility controls",
            ):
                driver.load_candidate_protocol(path)

        ambiguous = json.loads(self.candidates.read_text())
        ambiguous_candidate = ambiguous["stage2"][
            "volatility_candidates"
        ][0]
        ambiguous_candidate["fundamental_excess_kurtosis_share"] = 0.5
        ambiguous_candidate["fundamental_tail_transmission_multiplier"] = 4.0
        ambiguous_path = self.root / "ambiguous_tail_allocation.json"
        ambiguous_path.write_text(json.dumps(ambiguous), encoding="utf-8")
        with self.assertRaisesRegex(
            driver.CalibrationDriverError,
            "requires the complete empirical excess-kurtosis allocation",
        ):
            driver.load_candidate_protocol(ambiguous_path)

    def test_spread_adaptive_local_controls_default_and_fail_closed(self) -> None:
        protocol = driver.load_candidate_protocol(self.candidates)
        by_id = {candidate.identifier: candidate for candidate in protocol.local}
        self.assertEqual(by_id["good_local"].spread_elasticity, 0.5)
        self.assertEqual(
            by_id["good_local"].max_improvement_probability, 0.75,
        )
        self.assertTrue(
            by_id["good_local"].must_promote_after_short_screen
        )

        # The untouched control omits both fields.  Zero elasticity bypasses
        # spread scaling exactly and the unit cap is nonbinding, retaining the
        # legacy constant improvement probability.
        self.assertEqual(by_id["bad_local"].spread_elasticity, 0.0)
        self.assertEqual(
            by_id["bad_local"].max_improvement_probability, 1.0,
        )
        legacy = driver.LocalCandidate(
            "legacy", True, 1000.0, 1.0, 0.25,
        )
        self.assertEqual(legacy.spread_elasticity, 0.0)
        self.assertEqual(legacy.max_improvement_probability, 1.0)
        self.assertFalse(legacy.must_promote_after_short_screen)

        invalid_cases = (
            ("spread_elasticity", -0.1),
            ("max_improvement_probability", -0.1),
            ("max_improvement_probability", 1.01),
        )
        for index, (field, value) in enumerate(invalid_cases):
            with self.subTest(field=field, value=value):
                payload = json.loads(self.candidates.read_text())
                payload["stage1"]["local_mm_candidates"][1][field] = value
                path = self.root / f"invalid_local_spread_{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(
                    driver.CalibrationDriverError,
                    "invalid numerical controls",
                ):
                    driver.load_candidate_protocol(path)

        below_base = json.loads(self.candidates.read_text())
        below_base["stage1"]["local_mm_candidates"][1][
            "improvement_probability"
        ] = 0.8
        below_base_path = self.root / "local_cap_below_base.json"
        below_base_path.write_text(json.dumps(below_base), encoding="utf-8")
        with self.assertRaisesRegex(
            driver.CalibrationDriverError, "invalid numerical controls",
        ):
            driver.load_candidate_protocol(below_base_path)

        disabled = json.loads(self.candidates.read_text())
        disabled_candidate = disabled["stage1"]["local_mm_candidates"][1]
        disabled_candidate["enabled"] = False
        disabled_path = self.root / "disabled_local_spread.json"
        disabled_path.write_text(json.dumps(disabled), encoding="utf-8")
        with self.assertRaisesRegex(
            driver.CalibrationDriverError,
            "disabled local-MM policy must use zero spread elasticity",
        ):
            driver.load_candidate_protocol(disabled_path)

        non_boolean = json.loads(self.candidates.read_text())
        non_boolean["stage1"]["local_mm_candidates"][1][
            "must_promote_after_short_screen"
        ] = 1
        non_boolean_path = self.root / "non_boolean_mandatory.json"
        non_boolean_path.write_text(json.dumps(non_boolean), encoding="utf-8")
        with self.assertRaisesRegex(
            driver.CalibrationDriverError, "must be boolean",
        ):
            driver.load_candidate_protocol(non_boolean_path)

    def test_repository_protocol_adds_only_predeclared_local_spread_variants(
        self,
    ) -> None:
        payload = json.loads((
            PROJECT_ROOT / "config/queue_reactive_calibration_candidates_v1.json"
        ).read_text())
        candidates = payload["stage1"]["local_mm_candidates"]
        adaptive = [
            candidate for candidate in candidates
            if float(candidate.get("spread_elasticity", 0.0)) > 0.0
        ]
        self.assertEqual(len(candidates), 13)
        self.assertEqual(len(adaptive), 5)
        self.assertEqual(
            {
                (
                    float(candidate["interval_ms"]),
                    float(candidate["improvement_probability"]),
                    float(candidate["spread_elasticity"]),
                    float(candidate["max_improvement_probability"]),
                )
                for candidate in adaptive
            },
            {
                (2000.0, 0.25, 0.4, 0.55),
                (2000.0, 0.25, 0.5, 0.75),
                (2000.0, 0.25, 0.5, 1.0),
                (2000.0, 0.25, 1.0, 0.75),
                (2000.0, 0.25, 1.0, 1.0),
            },
        )
        self.assertEqual(
            [
                candidate["id"] for candidate in candidates
                if candidate.get("must_promote_after_short_screen") is True
            ],
            ["local_slow_spread_eta040_cap055"],
        )
        self.assertEqual(
            payload["stage1"]["fixed_local_candidate_id"],
            "local_reference",
        )
        protocol = driver.load_candidate_protocol(
            PROJECT_ROOT / "config/queue_reactive_calibration_candidates_v1.json"
        )
        self.assertEqual(
            [candidate.identifier for candidate in protocol.local],
            ["local_reference"],
        )
        self.assertEqual(protocol.stage1_survivors, 1)

    def test_repository_protocol_adds_only_predeclared_gap_variants(self) -> None:
        payload = json.loads((
            PROJECT_ROOT / "config/queue_reactive_calibration_candidates_v1.json"
        ).read_text())
        candidates = payload["stage2"]["value_policy_candidates"]
        adaptive = [
            candidate for candidate in candidates
            if float(candidate.get("gap_elasticity", 0.0)) > 0.0
        ]
        protocol = driver.load_candidate_protocol(
            PROJECT_ROOT / "config/queue_reactive_calibration_candidates_v1.json"
        )
        self.assertEqual(
            protocol.mandatory_full_day_joint_candidates,
            (
                "value_value_20bps_20pct__vol_latent_scale_400_tail025",
                "value_value_5bps_20pct_eta100_cap075__vol_"
                "latent_scale_200_phi080_tail100_x400",
                "value_value_5bps_20pct_eta100_cap075__vol_"
                "latent_scale_200_phi080_tail100_x300",
                "value_value_20bps_20pct__vol_"
                "latent_scale_050_phi080_tail100",
            ),
        )
        self.assertEqual(len(candidates), 20)
        self.assertEqual(len(adaptive), 7)
        self.assertEqual(
            {
                (
                    float(candidate["threshold_bps"]),
                    float(candidate["depth_participation"]),
                    float(candidate["gap_elasticity"]),
                    float(candidate["max_depth_participation"]),
                )
                for candidate in adaptive
            },
            {
                (20.0, 0.02, 0.5, 0.10),
                (20.0, 0.02, 1.0, 0.20),
                (10.0, 0.05, 0.5, 0.10),
                (10.0, 0.05, 1.0, 0.20),
                (5.0, 0.05, 0.5, 0.10),
                (5.0, 0.05, 1.0, 0.20),
                (5.0, 0.20, 1.0, 0.75),
            },
        )
        volatility = payload["stage2"]["volatility_candidates"]
        amplified = [
            candidate for candidate in volatility
            if float(candidate.get(
                "fundamental_tail_transmission_multiplier", 1.0
            )) > 1.0
        ]
        self.assertEqual(len(amplified), 8)
        self.assertEqual(
            {
                (
                    float(candidate["fundamental_variance_scale"]),
                    float(candidate["fundamental_tail_transmission_multiplier"]),
                )
                for candidate in amplified
            },
            {
                (1.0, 2.0), (1.0, 3.0), (1.0, 4.0), (1.0, 8.0),
                (2.0, 2.0), (2.0, 3.0), (2.0, 4.0), (2.0, 8.0),
            },
        )

    def test_cluster_labels_must_match_before_any_simulator_run(self) -> None:
        with self.background.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
            fields = tuple(reader.fieldnames or ())
        rows[0]["cluster_id"] = "different_cluster"
        write_csv(self.background, fields, rows)
        with mock.patch.object(driver, "execute_run") as execute:
            with self.assertRaisesRegex(
                driver.CalibrationDriverError, "cluster assignments disagree",
            ):
                driver.train(self.training_args(self.root / "cluster_mismatch"))
        execute.assert_not_called()

    def test_all_stage_targets_are_preflighted_before_simulation(self) -> None:
        observed: list[tuple[str, int]] = []

        def fail_on_prefix(config, *, day, symbols, duration):
            del config, symbols
            observed.append((day, duration))
            if day == DATES[0] and duration == driver.STAGE2_DURATION:
                raise driver.CalibrationDriverError("missing 3600s prefix targets")
            return self.fake_targets(
                self.deployment, day=day, symbols=SYMBOLS, duration=duration,
            )

        with (
            mock.patch.object(driver, "load_targets", side_effect=fail_on_prefix),
            mock.patch.object(driver, "execute_run") as execute,
        ):
            with self.assertRaisesRegex(
                driver.CalibrationDriverError, "missing 3600s prefix targets",
            ):
                driver.train(self.training_args(self.root / "target_preflight"))
        self.assertIn((DATES[0], driver.STAGE1_DURATION), observed)
        self.assertIn((DATES[0], driver.STAGE2_DURATION), observed)
        execute.assert_not_called()

    def test_failed_small_panel_gate_writes_only_unauthorized_selection(self) -> None:
        output = self.root / "strict_failure"
        with (
            mock.patch.object(driver, "load_targets", side_effect=self.fake_targets),
            mock.patch.object(
                driver, "run_strict_evaluation", side_effect=self.failing_strict,
            ),
        ):
            result = driver.train(self.training_args(output))
        freeze = json.loads(pathlib.Path(
            result["training_selection_freeze"]
        ).read_text())
        self.assertEqual(
            freeze["status"],
            "stratified_training_selection_frozen_pending_full_universe",
        )
        self.assertFalse(freeze["small_panel_strict_training_gate_passed"])
        self.assertFalse(freeze["heldout_execution_authorized"])
        self.assertFalse((output / "expanded_training_freeze.json").exists())

    def test_heldout_copies_only_opening_fields_and_checks_frozen_hashes(self) -> None:
        training_output = self.root / "training_for_heldout"
        result = self.run_training(training_output)
        selection_freeze = pathlib.Path(result["training_selection_freeze"])
        with self.assertRaisesRegex(
            driver.CalibrationDriverError, "does not authorize",
        ):
            driver.heldout(driver.parser().parse_args([
                "heldout", "--freeze-record", str(selection_freeze),
                "--heldout-date", "2020-01-30",
                "--heldout-opening-config", str(self.deployment),
                "--heldout-target-config", str(self.deployment),
                "--heldout-seed", "16",
                "--output-root", str(self.root / "blocked_heldout"),
            ]))
        expansion_output = self.root / "full_universe_training"
        expansion = self.run_full_universe_expansion(
            selection_freeze=selection_freeze,
            output=expansion_output,
        )
        freeze_path = pathlib.Path(expansion["expanded_training_freeze"])
        opening = self.write_config(
            "2020-01-30",
            name="heldout_opening.csv",
            symbols=SYMBOLS + ("EEE", "FFF"),
        )
        with opening.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            row["fundamental_price_ticks"] = str(float(row["fundamental_price_ticks"]) + 500)
            row["data_dir"] = str(self.root / "FORBIDDEN_HELDOUT_FLOW")
            row["target_data_dir"] = str(
                self.root / "FORBIDDEN_HELDOUT_FLOW" / row["symbol"].lower()
            )
        write_csv(opening, tuple(rows[0]), rows)
        output = self.root / "heldout"
        args = driver.parser().parse_args([
            "heldout", "--freeze-record", str(freeze_path),
            "--heldout-date", "2020-01-30",
            "--heldout-opening-config", str(opening),
            "--heldout-target-config", str(opening),
            "--heldout-seed", "17", "--output-root", str(output),
        ])
        with mock.patch.object(
            driver, "run_strict_evaluation", side_effect=self.fake_strict,
        ):
            heldout_result = driver.heldout(args)
        self.assertEqual(heldout_result["status"], "heldout_adequacy_passed")
        merged_text = (output / "heldout_simulation_config.csv").read_text()
        merged = list(csv.DictReader(merged_text.splitlines()))
        with (expansion_output / "frozen_full_universe_model/deployment_config.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            frozen = list(csv.DictReader(source))
        self.assertEqual(
            [row["fundamental_price_ticks"] for row in merged],
            [row["fundamental_price_ticks"] for row in rows],
        )
        self.assertEqual(
            [row["data_dir"] for row in merged],
            [row["data_dir"] for row in frozen],
        )
        self.assertTrue(all(
            "FORBIDDEN_HELDOUT_FLOW" in row["target_data_dir"] for row in merged
        ))
        failed_output = self.root / "heldout_strict_failure"
        failed_args = driver.parser().parse_args([
            "heldout", "--freeze-record", str(freeze_path),
            "--heldout-date", "2020-01-30",
            "--heldout-opening-config", str(opening),
            "--heldout-target-config", str(opening),
            "--heldout-seed", "18", "--output-root", str(failed_output),
        ])
        with mock.patch.object(
            driver, "run_strict_evaluation", side_effect=self.failing_strict,
        ):
            with self.assertRaisesRegex(
                driver.CalibrationDriverError, "adequacy gate failed",
            ):
                driver.heldout(failed_args)
        failed_manifest = json.loads(
            (failed_output / "heldout_run_manifest.json").read_text()
        )
        self.assertEqual(failed_manifest["status"], "heldout_adequacy_failed")
        mark_path = pathlib.Path(frozen[0]["data_dir"]) / driver.MODEL_MARK_FILES[0]
        mark_path.write_text(mark_path.read_text() + "# mutation\n")
        tamper_args = driver.parser().parse_args([
            "heldout", "--freeze-record", str(freeze_path),
            "--heldout-date", "2020-01-30",
            "--heldout-opening-config", str(opening),
            "--heldout-target-config", str(opening),
            "--heldout-seed", "19", "--output-root", str(self.root / "tamper"),
        ])
        with self.assertRaisesRegex(driver.CalibrationDriverError, "hash-mismatched"):
            driver.heldout(tamper_args)

    def test_failed_full_universe_gate_writes_no_authorizing_freeze(self) -> None:
        training = self.run_training(self.root / "training_for_failed_expand")
        selection_freeze = pathlib.Path(
            training["training_selection_freeze"]
        )
        expansion_output = self.root / "failed_full_universe_training"
        with self.assertRaisesRegex(
            driver.CalibrationDriverError, "no expanded freeze",
        ):
            self.run_full_universe_expansion(
                selection_freeze=selection_freeze,
                output=expansion_output,
                strict_side_effect=self.failing_strict,
            )
        self.assertFalse(
            (expansion_output / "expanded_training_freeze.json").exists()
        )
        adequacy = json.loads((
            expansion_output / "full_universe_training_adequacy.json"
        ).read_text())
        self.assertFalse(adequacy["passed"])
        self.assertEqual(
            adequacy["status"],
            "strict_training_gate_failed_no_expanded_freeze",
        )

    def test_six_component_expansion_uses_versioned_pass_flags(self) -> None:
        training = self.run_training(self.root / "training_for_six_expand")
        selection_freeze = pathlib.Path(
            training["training_selection_freeze"]
        )
        certificate = self.root / "six_component_certificate.json"
        certificate.write_text("{}\n", encoding="utf-8")
        verified_certificate = {
            "path": str(certificate),
            "sha256": driver.sha256_file(certificate),
            "verified_source_evidence": {},
            "symbol_residuals": {
                "path": str(certificate),
                "sha256": driver.sha256_file(certificate),
            },
            "authoritative_reanalysis": [
                {"date": day, "passed": True} for day in DATES
            ],
            "classification": (
                "retrospective_development_protocol_revision"
            ),
        }
        output = self.root / "six_component_full_universe_training"
        with mock.patch.object(
            driver,
            "verified_six_component_certificate",
            return_value=verified_certificate,
        ):
            result = self.run_full_universe_expansion(
                selection_freeze=selection_freeze,
                output=output,
                extra_args=(
                    "--gate-protocol", "marketwide-six-v2",
                    "--six-component-protocol-certificate", str(certificate),
                    "--max-refinement-iterations", "0",
                ),
            )
        freeze = json.loads(pathlib.Path(
            result["expanded_training_freeze"]
        ).read_text())
        self.assertTrue(freeze["training_adequacy_gate_passed"])
        self.assertFalse(freeze["strict_training_gate_passed"])
        self.assertTrue(
            freeze["marketwide_six_component_training_gate_passed"]
        )
        self.assertEqual(
            freeze["training_adequacy_protocol"], "marketwide-six-v2",
        )
        self.assertIn(
            "six_component_training_evaluation",
            freeze["strict_training_report"]["path"],
        )

    def test_full_universe_refinement_must_pass_before_authorizing_freeze(self) -> None:
        training = self.run_training(self.root / "training_for_refinement")
        selection_freeze = pathlib.Path(
            training["training_selection_freeze"]
        )
        calls = 0

        def fail_then_pass(arguments):
            nonlocal calls
            calls += 1
            arguments = list(arguments)
            output = pathlib.Path(
                arguments[arguments.index("--output-dir") + 1]
            )
            if calls == 1:
                residual_rows = []
                for cluster in ("0", "1"):
                    for metric, target, simulated in (
                        ("return_variance", 0.8, 1.0),
                        ("return_kurtosis", 4.0, 3.0),
                        ("absolute_return_acf1", 0.14, 0.08),
                    ):
                        residual_rows.append({
                            "cluster_id": cluster,
                            "metric": metric,
                            "target": target,
                            "simulated_seed_mean": simulated,
                        })
                write_csv(
                    output / "symbol_residuals.csv",
                    (
                        "cluster_id", "metric", "target",
                        "simulated_seed_mean",
                    ),
                    residual_rows,
                )
                report = {"schema_version": 1, "passed": False, "mock": True}
            else:
                report = {"schema_version": 1, "passed": True, "mock": True}
            report_path = output / "strict_validation_report.json"
            driver.write_json(report_path, report)
            return report, report_path

        output = self.root / "refined_full_universe_training"
        result = self.run_full_universe_expansion(
            selection_freeze=selection_freeze,
            output=output,
            strict_side_effect=fail_then_pass,
        )
        self.assertEqual(calls, 2)
        freeze = json.loads(pathlib.Path(
            result["expanded_training_freeze"]
        ).read_text())
        self.assertTrue(freeze["full_universe_training_adequacy_passed"])
        self.assertTrue(
            freeze["selection_parameters_changed_during_2019_training"]
        )
        refinement = freeze["selection"]["full_universe_refinement"]
        self.assertTrue(refinement["performed"])
        self.assertEqual(refinement["completed_updates"], 1)
        self.assertFalse(refinement["heldout_inputs_read"])
        self.assertIn(
            "full_universe_refinement/iteration_1",
            freeze["frozen_artifacts"]["deployment_config"]["path"],
        )

    def test_full_universe_refinement_is_cluster_specific_and_bounded(self) -> None:
        residuals = self.root / "training_symbol_residuals.csv"
        rows = []
        for cluster, values in {
            "0": {
                "return_variance": (0.5, 1.0),
                "return_kurtosis": (8.0, 4.0),
                "absolute_return_acf1": (0.15, 0.05),
            },
            "1": {
                "return_variance": (1.0, 1.0),
                "return_kurtosis": (4.0, 4.0),
                "absolute_return_acf1": (0.05, 0.10),
            },
        }.items():
            for metric, (target, simulated) in values.items():
                rows.append({
                    "cluster_id": cluster,
                    "metric": metric,
                    "target": target,
                    "simulated_seed_mean": simulated,
                })
        write_csv(
            residuals,
            ("cluster_id", "metric", "target", "simulated_seed_mean"),
            rows,
        )
        baseline = {
            cluster: driver.VolatilityCandidate(
                identifier=f"baseline_{cluster}",
                variance_scale=2.0,
                persistence=0.8,
                std=0.15,
                excess_kurtosis_share=1.0,
                tail_transmission_multiplier=2.0,
            )
            for cluster in ("0", "1")
        }
        refined, evidence = driver.refine_full_universe_volatility(
            symbol_residuals=residuals,
            current=baseline,
            cluster_ids={"0", "1"},
            iteration=1,
        )
        self.assertLess(refined["0"].variance_scale, 2.0)
        self.assertGreater(refined["0"].tail_transmission_multiplier, 2.0)
        self.assertGreater(refined["0"].std, 0.15)
        self.assertEqual(refined["0"].persistence, 0.95)
        self.assertEqual(refined["1"].std, 0.0)
        self.assertEqual(refined["1"].persistence, 0.0)
        self.assertTrue(evidence["training_only"])
        self.assertFalse(evidence["heldout_inputs_read"])

    def test_full_universe_refinement_rejects_incomplete_evidence(self) -> None:
        residuals = self.root / "incomplete_symbol_residuals.csv"
        write_csv(
            residuals,
            ("cluster_id", "metric", "target"),
            ({"cluster_id": "0", "metric": "return_variance", "target": 1},),
        )
        current = {
            "0": driver.VolatilityCandidate(
                "baseline", 1.0, 0.8, 0.15, 1.0, 1.0,
            )
        }
        with self.assertRaisesRegex(
            driver.CalibrationDriverError, "lacks columns",
        ):
            driver.refine_full_universe_volatility(
                symbol_residuals=residuals,
                current=current,
                cluster_ids={"0"},
                iteration=1,
            )


if __name__ == "__main__":
    unittest.main()
