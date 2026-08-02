#!/usr/bin/env python3
"""Deterministic tests for the standalone strict fit evaluator."""

from __future__ import annotations

import csv
import json
import pathlib
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_strict_model_validation as validation  # noqa: E402


SYMBOLS = ("AAA", "BBB", "CCC", "DDD")
TARGETS = {
    "mean_spread_ticks": 2.0,
    "mean_bid_depth": 100.0,
    "mean_ask_depth": 100.0,
    "mid_move_rate": 0.1,
    "return_variance": 1.0e-6,
    "return_kurtosis": 3.0,
    "absolute_return_acf1": 0.1,
    "two_sided_sample_fraction": 1.0,
}


def write_csv(
    path: pathlib.Path,
    fields: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class StrictValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.cluster_map = self.root / "cluster_assignments.csv"
        write_csv(
            self.cluster_map,
            ("symbol", "cluster_id"),
            [
                {"symbol": symbol, "cluster_id": index // 2}
                for index, symbol in enumerate(SYMBOLS)
            ],
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_day(
        self, day: str, *, legacy_coverage_omission: bool = False,
    ) -> pathlib.Path:
        compact = day.replace("-", "")
        config = self.root / f"config_{compact}.csv"
        config_rows = []
        for symbol in SYMBOLS:
            data_dir = self.root / f"itch_{compact}_{symbol.lower()}"
            target = data_dir / f"market_targets_{symbol.lower()}_{compact}.csv"
            target_values = dict(TARGETS)
            if legacy_coverage_omission:
                target_values.pop("two_sided_sample_fraction")
            write_csv(
                target,
                ("name", "target", "scale", "weight"),
                [
                    {
                        "name": metric,
                        "target": value,
                        "scale": 0.1,
                        "weight": 1,
                    }
                    for metric, value in target_values.items()
                ],
            )
            authoritative_values = dict(target_values)
            authoritative_scales = {
                metric: 0.1 for metric in target_values
            }
            manifest = {
                "symbol": symbol,
                "trading_date": day,
                "aggregation_duration_seconds": 60,
                "valid_snapshots": 60,
                "invalid_snapshots": 0,
                "market_values": authoritative_values,
                "market_target_scales": authoritative_scales,
                "distribution_observation_counts": {
                    event: 10 for event in validation.BACKGROUND_EVENTS
                },
            }
            (data_dir / f"itch_manifest_{symbol.lower()}_{compact}.json").write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )
            config_rows.append({"symbol": symbol, "data_dir": data_dir})
        write_csv(config, ("symbol", "data_dir"), config_rows)
        return config

    def test_legacy_coverage_omission_uses_exact_snapshot_counters(self) -> None:
        day = "2020-01-30"
        output = self.root / "legacy_coverage"
        args = [
            "--evaluation-role", "development_validation",
            "--cluster-map", str(self.cluster_map),
            "--expected-cluster-count", "2",
            "--output-dir", str(output),
            "--expected-date", day,
        ]
        config = self.write_day(day, legacy_coverage_omission=True)
        args.extend(("--target-config", f"{day}={config}"))
        for seed in (7, 11):
            summary = self.write_summary(day, seed)
            args.extend(("--sim-summary", f"{day}:{seed}={summary}"))
        self.assertEqual(validation.main(args), 0)
        report = json.loads(
            (output / "strict_validation_report.json").read_text()
        )
        self.assertTrue(report["passed"])

    def test_target_manifest_date_must_match_evaluation_date(self) -> None:
        day = "2020-01-30"
        args = self.arguments(
            role="development_validation", days=(day,),
            output=self.root / "wrong_target_date",
        )
        target_index = args.index("--target-config") + 1
        config = pathlib.Path(args[target_index].split("=", 1)[1])
        with config.open(newline="", encoding="utf-8") as handle:
            first = next(csv.DictReader(handle))
        manifest = next(pathlib.Path(first["data_dir"]).glob("itch_manifest_*.json"))
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["trading_date"] = "2019-12-30"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(validation.main(args), 2)

    def write_summary(
        self,
        day: str,
        seed: int,
        *,
        changes: dict[tuple[str, str], float] | None = None,
    ) -> pathlib.Path:
        changes = changes or {}
        compact = day.replace("-", "")
        path = self.root / f"summary_{compact}_{seed}.csv"
        rows: list[dict[str, object]] = []
        for index, symbol in enumerate(SYMBOLS):
            metrics = {
                "background_event_rate": 1.0,
                **TARGETS,
            }
            for metric in validation.METRICS:
                metrics[metric] = changes.get((symbol, metric), metrics[metric])
            rows.append({
                "asset_id": index,
                "symbol": symbol,
                "sample_count": 60,
                "expected_sample_count": 60,
                "invalid_sample_count": 0,
                "structurally_valid": 1,
                **metrics,
            })
        write_csv(
            path,
            (
                "asset_id", "symbol", "sample_count", "expected_sample_count",
                "invalid_sample_count", "structurally_valid", *validation.METRICS,
            ),
            rows,
        )
        return path

    def arguments(
        self,
        *,
        role: str,
        days: tuple[str, ...],
        output: pathlib.Path,
        changes: dict[tuple[str, str], float] | None = None,
    ) -> list[str]:
        result = [
            "--evaluation-role", role,
            "--cluster-map", str(self.cluster_map),
            "--expected-cluster-count", "2",
            "--output-dir", str(output),
        ]
        for day in days:
            config = self.write_day(day)
            result.extend(("--expected-date", day))
            result.extend(("--target-config", f"{day}={config}"))
            for seed in (7, 11):
                summary = self.write_summary(
                    day, seed, changes=changes,
                )
                result.extend(("--sim-summary", f"{day}:{seed}={summary}"))
        return result

    def test_training_passes_each_date_and_outputs_are_deterministic(self) -> None:
        days = ("2019-01-30", "2019-03-27")
        output_one = self.root / "one"
        args = self.arguments(
            role="training_fit", days=days, output=output_one,
        )
        self.assertEqual(validation.main(args), 0)
        report_one = json.loads(
            (output_one / "strict_validation_report.json").read_text()
        )
        self.assertTrue(report_one["passed"])
        self.assertTrue(report_one["all_dates_passed_separately"])
        self.assertEqual(
            [row["date"] for row in report_one["date_results"]], list(days)
        )
        self.assertTrue(all(row["passed"] for row in report_one["date_results"]))

        output_two = self.root / "two"
        second_args = [
            str(output_two) if value == str(output_one) else value for value in args
        ]
        self.assertEqual(validation.main(second_args), 0)
        for filename in report_one["diagnostic_files"].values():
            self.assertEqual(
                (output_one / filename).read_bytes(),
                (output_two / filename).read_bytes(),
            )
        self.assertEqual(
            (output_one / "strict_validation_report.json").read_bytes(),
            (output_two / "strict_validation_report.json").read_bytes(),
        )

    def test_marketwide_six_protocol_keeps_secondary_failures_diagnostic(self) -> None:
        day = "2020-01-30"
        changes = {
            (symbol, "return_kurtosis"): 30.0 for symbol in SYMBOLS
        }
        changes.update({
            (symbol, "absolute_return_acf1"): 0.13 for symbol in SYMBOLS
        })
        output = self.root / "six_component_diagnostics"
        args = self.arguments(
            role="development_validation", days=(day,),
            output=output, changes=changes,
        )
        args.extend((
            "--gate-protocol", validation.MARKETWIDE_SIX_GATE,
        ))
        self.assertEqual(validation.main(args), 0)
        report = json.loads(
            (output / "strict_validation_report.json").read_text()
        )
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["gate"]["primary_metric_set"],
            list(validation.SIX_COMPONENT_METRICS),
        )
        warnings = report["date_results"][0]["diagnostic_warnings"]
        self.assertTrue(any("return_kurtosis" in value for value in warnings))
        self.assertTrue(any("ACF" in value for value in warnings))
        with (output / "marketwide_metric_scores.csv").open(
            newline="", encoding="utf-8",
        ) as source:
            rows = list(csv.DictReader(source))
        combined = [
            row for row in rows
            if row["metric"] == validation.COMBINED_DEPTH_METRIC
        ]
        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0]["gate_role"], "primary")
        self.assertEqual(combined[0]["contributes_to_pass"], "True")

    def test_marketwide_six_protocol_still_rejects_primary_failure(self) -> None:
        day = "2020-01-30"
        changes = {
            (symbol, "mean_spread_ticks"): 200.0 for symbol in SYMBOLS
        }
        output = self.root / "six_component_primary_failure"
        args = self.arguments(
            role="development_validation", days=(day,),
            output=output, changes=changes,
        )
        args.extend((
            "--gate-protocol", validation.MARKETWIDE_SIX_GATE,
        ))
        self.assertEqual(validation.main(args), 1)
        report = json.loads(
            (output / "strict_validation_report.json").read_text()
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any(
            "mean_spread_ticks" in value
            for value in report["date_results"][0]["failure_reasons"]
        ))

    def test_cluster_and_symbol_gates_fail_without_relaxing_thresholds(self) -> None:
        day = "2020-01-30"
        changes = {("AAA", "mean_spread_ticks"): 200.0}
        output = self.root / "failure"
        args = self.arguments(
            role="development_validation", days=(day,), output=output,
            changes=changes,
        )
        self.assertEqual(validation.main(args), 1)
        report = json.loads(
            (output / "strict_validation_report.json").read_text()
        )
        self.assertFalse(report["passed"])
        reasons = report["date_results"][0]["failure_reasons"]
        self.assertTrue(any("cluster 0 mean_spread_ticks" in reason for reason in reasons))
        self.assertTrue(any("gross-failure fraction" in reason for reason in reasons))
        self.assertEqual(
            report["gate"]["each_cluster_metric_score_maximum"], 3.0
        )

    def test_acf_distribution_gate_is_independent_of_robust_score(self) -> None:
        day = "2020-01-30"
        changes = {
            (symbol, "absolute_return_acf1"): 0.125 for symbol in SYMBOLS
        }
        output = self.root / "acf_failure"
        args = self.arguments(
            role="development_validation", days=(day,), output=output,
            changes=changes,
        )
        self.assertEqual(validation.main(args), 1)
        report = json.loads(
            (output / "strict_validation_report.json").read_text()
        )
        result = report["date_results"][0]
        self.assertLess(result["marketwide_robust_score"], 1.5)
        self.assertTrue(any("ACF mean error" in reason for reason in result["failure_reasons"]))

    def test_development_pass_never_claims_certification(self) -> None:
        day = "2020-01-30"
        output = self.root / "development"
        args = self.arguments(
            role="development_validation", days=(day,), output=output,
        )
        self.assertEqual(validation.main(args), 0)
        report = json.loads(
            (output / "strict_validation_report.json").read_text()
        )
        self.assertEqual(
            report["result_label"],
            "development_validation_adequate_not_certification",
        )
        self.assertFalse(report["certification_claimed"])
        self.assertNotIn("certified", json.dumps(report).lower())

    def test_final_holdout_requires_a_frozen_protocol_record(self) -> None:
        day = "2021-01-29"
        output = self.root / "final"
        args = self.arguments(
            role="untouched_final_holdout", days=(day,), output=output,
        )
        self.assertEqual(validation.main(args), 2)
        freeze = self.root / "protocol_freeze.json"
        freeze.write_text('{"model":"frozen-before-holdout"}\n', encoding="utf-8")
        args.extend(("--protocol-freeze-record", str(freeze)))
        self.assertEqual(validation.main(args), 0)
        report = json.loads(
            (output / "strict_validation_report.json").read_text()
        )
        self.assertEqual(
            report["protocol_freeze_record"]["sha256"], validation.sha256(freeze)
        )
        self.assertFalse(report["certification_claimed"])

    def test_schema2_duration_fallback_requires_exact_agreement(self) -> None:
        manifest = {
            "session_start": "09:30:00",
            "session_end": "16:00:00",
            "queue_reactive_training_artifacts": {
                "exposure": {"expected_session_seconds": 23_400},
            },
            "market_target_windows": {
                "23400": {"duration_seconds": 23_400},
            },
        }
        self.assertEqual(
            validation.aggregation_duration_seconds(
                manifest, manifest_path=self.root / "manifest.json",
            ),
            23_400,
        )

    def test_schema2_duration_fallback_rejects_any_mismatch(self) -> None:
        manifest = {
            "session_start": "09:30:00",
            "session_end": "16:00:00",
            "queue_reactive_training_artifacts": {
                "exposure": {"expected_session_seconds": 23_399},
            },
        }
        with self.assertRaisesRegex(
            validation.EvaluationError, "inconsistent full-session durations",
        ):
            validation.aggregation_duration_seconds(
                manifest, manifest_path=self.root / "manifest.json",
            )


if __name__ == "__main__":
    unittest.main()
