#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Focused tests for the training-only queue-reactive policy fitter."""

from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "fit_queue_reactive_hawkes_policy.py"
SPEC = importlib.util.spec_from_file_location("queue_policy_fitter", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
fitter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fitter
SPEC.loader.exec_module(fitter)


DATES = (
    "2019-01-30",
    "2019-03-27",
    "2019-05-30",
    "2019-07-30",
    "2019-10-30",
)
SYMBOLS = ("AAA", "BBB", "CCC")


def write_csv(path: pathlib.Path, fields: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(fields)
        writer.writerows(rows)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


class Fixture:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.training_roots: list[tuple[str, pathlib.Path]] = []
        self.config = root / "pooled_config.csv"
        self.assignments = root / "cluster_assignments.csv"
        self._write_configuration()
        for date_index, date in enumerate(DATES):
            extraction_root = root / f"training_{date.replace('-', '')}"
            self.training_roots.append((date, extraction_root))
            for symbol_index, symbol in enumerate(SYMBOLS):
                self._write_symbol_artifacts(
                    extraction_root, date, symbol, date_index + symbol_index
                )

    def _write_configuration(self) -> None:
        config_rows: list[list[object]] = []
        for book_id, symbol in enumerate(SYMBOLS):
            rates = self.root / f"rates_{symbol.lower()}.csv"
            write_csv(
                rates,
                ["event_type", "stationary_target_rate"],
                [[event_type, 1.0 + book_id * 0.1] for event_type in fitter.EVENT_TYPES],
            )
            config_rows.append([book_id, symbol, "unused", rates])
        write_csv(
            self.config,
            ["book_id", "symbol", "data_dir", "hawkes_rates_file"],
            config_rows,
        )
        write_csv(
            self.assignments,
            ["symbol", "cluster_id"],
            [["AAA", 0], ["BBB", 1], ["CCC", 0]],
        )

    def _write_symbol_artifacts(
        self,
        extraction_root: pathlib.Path,
        date: str,
        symbol: str,
        perturbation: int,
    ) -> None:
        compact = date.replace("-", "")
        directory = extraction_root / "empirical_data" / (
            f"itch_{compact}_{symbol.lower()}"
        )
        intraday_rows: list[list[object]] = []
        state_rows: list[list[object]] = []
        exposure_rows: list[list[object]] = []
        event_totals = {event_type: 0 for event_type in fitter.EVENT_TYPES}
        for half_hour in range(fitter.HALF_HOUR_BINS):
            if half_hour % 2:
                state = ("wider", "buy_high", "low", "high")
            else:
                state = ("one_tick", "sell_high", "high", "low")
            exposure_rows.append([half_hour, *state, 1800.0])
            for event_index, event_type in enumerate(fitter.EVENT_TYPES):
                count = 4 + half_hour + event_index + perturbation
                event_totals[event_type] += count
                intraday_rows.append([half_hour, event_type, count])
                state_rows.append([half_hour, event_type, *state, count])
        write_csv(
            directory / "intraday_event_counts.csv",
            ["half_hour_bin", "event_type", "count"],
            intraday_rows,
        )
        write_csv(
            directory / "queue_state_counts.csv",
            [
                "half_hour_bin", "event_type", "spread_bin",
                "queue_imbalance_bin", "bid_depth_ratio_bin",
                "ask_depth_ratio_bin", "count",
            ],
            state_rows,
        )
        write_csv(
            directory / "queue_state_exposure.csv",
            [
                "half_hour_bin", "spread_bin", "queue_imbalance_bin",
                "bid_depth_ratio_bin", "ask_depth_ratio_bin", "exposure_seconds",
            ],
            exposure_rows,
        )
        lag_rows: list[list[object]] = []
        allowed_order = {
            (source, target): edge_index
            for edge_index, (target, source) in enumerate(
                sorted(fitter.topology_ratios())
            )
        }
        for lag in fitter.LAG_SECONDS:
            for source in fitter.EVENT_TYPES:
                for target in fitter.EVENT_TYPES:
                    if lag == 0 and source == target:
                        correlation = 1.0
                    elif (source, target) in allowed_order and lag in (1, 2):
                        correlation = (
                            0.08 + 0.008 * allowed_order[(source, target)]
                            + 0.005 * perturbation
                        )
                    elif (
                        (source, target) in allowed_order
                        and lag in (5, 10, 20, 30)
                    ):
                        correlation = (
                            0.02 + 0.002 * allowed_order[(source, target)]
                            + 0.001 * perturbation
                        )
                    elif (
                        source == "market_buy"
                        and target == "market_sell"
                        and lag != 0
                    ):
                        # A deliberately large disallowed correlation verifies
                        # that the frozen sparse topology is enforced.
                        correlation = 0.90
                    else:
                        correlation = 0.0
                    lag_rows.append([
                        source, target, lag, 23400 - lag,
                        1.0, 1.0, 1.0, 1.0, correlation, correlation,
                    ])
        write_csv(
            directory / "event_count_lag_moments.csv",
            [
                "source_event_type", "target_event_type", "lag_seconds",
                "paired_bins", "source_mean_count", "target_mean_count",
                "source_variance", "target_variance", "covariance",
                "correlation",
            ],
            lag_rows,
        )
        for side, distance in (("limit_buy", 100), ("limit_sell", 200)):
            write_csv(
                directory / f"{side}_improvement_distribution.txt",
                ["improvement_ticks", "improvement_price_units", "count"],
                [[distance / 100, distance, 10 + perturbation]],
            )
        manifest = {
            "trading_date": date,
            "symbol": symbol,
            "queue_reactive_training_artifacts": {
                "schema_version": 2,
                "training_only": True,
                "queue_policy_estimation_ready": True,
                "pre_event_state_definition": {
                    "equal_timestamp_messages_share_one_left_limit_state": True,
                    "zero_duration_intermediate_states_are_not_used_as_covariates": True,
                },
                "artifacts": {
                    "event_count_lag_moments": "event_count_lag_moments.csv"
                },
                "artifact_row_counts": {
                    "event_count_lag_moments": len(fitter.LAG_SECONDS) * 6 * 6
                },
                "event_count_conservation": {
                    "totals_equal": True,
                    "equals_legacy_quantity_observation_counts": True,
                    "by_event_type": event_totals,
                },
                "exposure": {
                    "exact_nanosecond_conservation": True,
                    "expected_session_seconds": 23400.0,
                },
                "lag_moment_definition": {
                    "count_bin_seconds": 1,
                    "lags_seconds": list(fitter.LAG_SECONDS),
                    "direction": "source count at t versus target count at t+lag",
                },
            },
        }
        (directory / f"itch_manifest_{symbol.lower()}_{compact}.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )

    def argv(self, output: pathlib.Path) -> list[str]:
        values: list[str] = []
        for date, root in self.training_roots:
            values.extend(["--training-root", f"{date}={root}"])
        values.extend([
            "--cluster-assignments", str(self.assignments),
            "--pooled-config", str(self.config),
            "--output-root", str(output),
            "--forbid-date", "2020-01-30",
        ])
        return values


class QueuePolicyFitterTest(unittest.TestCase):
    def test_fit_is_normalized_stable_deterministic_and_training_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = Fixture(root)
            first = root / "output_a"
            second = root / "output_b"
            self.assertEqual(fitter.main(fixture.argv(first)), 0)
            self.assertEqual(fitter.main(fixture.argv(second)), 0)

            manifest = json.loads(
                (first / "training_policy_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["training_dates"], sorted(DATES))
            self.assertFalse(manifest["heldout_inputs_read"])
            self.assertEqual(manifest["forbidden_dates"], ["2020-01-30"])

            for cluster_id in ("0", "1"):
                directory = first / "clusters" / f"cluster_{cluster_id}"
                intraday = read_csv(directory / "intraday_factors.csv")
                self.assertEqual(len(intraday), fitter.HALF_HOUR_BINS)
                for event_type in fitter.EVENT_TYPES:
                    mean = sum(float(row[event_type]) for row in intraday) / len(intraday)
                    self.assertAlmostEqual(mean, 1.0, places=13)

                policy = json.loads((directory / "policy.json").read_text())
                self.assertLess(policy["hawkes"]["spectral_radius"], 0.75)
                self.assertLess(
                    policy["hawkes"]["maximum_integrated_row_sum"], 0.75
                )
                self.assertLess(
                    policy["hawkes"]["maximum_integrated_column_sum"], 0.75
                )
                self.assertTrue(policy["training_only"])
                state_diagnostics = policy["state_response"]["diagnostics"]
                self.assertEqual(policy["state_response"]["estimator_version"], 2)
                self.assertEqual(
                    state_diagnostics["offset_scope"],
                    "symbol_specific_frozen_training_stationary_target_rates",
                )
                self.assertFalse(
                    state_diagnostics["deployment_cluster_mean_used_as_offset"]
                )
                self.assertEqual(
                    state_diagnostics["offset_source_count"],
                    len(DATES) * policy["estimation_member_count"],
                )
                self.assertEqual(
                    state_diagnostics["offset_symbol_count"],
                    policy["estimation_member_count"],
                )
                proxy = policy["hawkes"]["lag_moment_proxy"]
                self.assertFalse(proxy["identifiable_multivariate_hawkes_estimate"])
                self.assertEqual(proxy["estimator_version"], 2)
                self.assertGreater(
                    proxy["fast"]["pre_feasibility_max_edge_strength"], 0.0
                )
                self.assertGreater(
                    proxy["slow"]["pre_feasibility_max_edge_strength"], 0.0
                )
                self.assertEqual(len(proxy["fast"]["edge_estimates"]), 12)
                self.assertTrue(all(
                    "post_feasibility_integrated_branching" in edge
                    for edge in proxy["fast"]["edge_estimates"]
                ))
                long_rows = read_csv(directory / "cluster_policy.csv")
                self.assertTrue(any(row["kind"] == "fast_alpha" for row in long_rows))
                self.assertFalse(any(row["kind"] == "stationary_target" for row in long_rows))
                self.assertTrue(
                    any(row["kind"] == "diagnostic_cluster_target" for row in long_rows)
                )
                metadata = {
                    row["target"]: row["value"]
                    for row in long_rows if row["kind"] == "meta"
                }
                self.assertEqual(metadata["intraday_origin_ns"], "0")
                self.assertEqual(
                    metadata["intraday_bin_width_ns"], "1800000000000"
                )
                self.assertEqual(metadata["fast_beta"], "1.0")
                self.assertEqual(metadata["slow_beta"], "0.1")
                self.assertEqual(metadata["state_log_multiplier_bound"], "4.0")
                self.assertLess(float(metadata["maximum_integrated_row_sum"]), 0.75)
                self.assertLess(
                    float(metadata["maximum_integrated_column_sum"]), 0.75
                )
                self.assertEqual(
                    metadata["matrix_orientation"], "response_rows_trigger_columns"
                )
                self.assertEqual(
                    metadata["stationary_target_scope"],
                    "descriptive_cluster_member_mean",
                )
                improvement = read_csv(
                    directory / "limit_buy_improvement_distribution.csv"
                )
                self.assertEqual(
                    list(improvement[0]),
                    ["improvement_ticks", "improvement_price_units", "count"],
                )

            mapping = read_csv(first / "symbol_policy_mapping.csv")
            self.assertEqual(
                list(mapping[0]),
                [
                    "symbol", "cluster_id", "policy_file",
                    "limit_buy_improvement_file", "limit_sell_improvement_file",
                ],
            )
            self.assertTrue(all(not pathlib.Path(row["policy_file"]).is_absolute() for row in mapping))
            estimation_mapping = read_csv(
                first / "estimation_symbol_policy_mapping.csv"
            )
            self.assertEqual(
                {row["symbol"] for row in estimation_mapping}, set(SYMBOLS)
            )
            estimation_clusters = read_csv(
                first / "estimation_cluster_assignments.csv"
            )
            self.assertEqual(
                {row["symbol"] for row in estimation_clusters}, set(SYMBOLS)
            )
            self.assertTrue(
                manifest["certification"][
                    "state_response_uses_symbol_specific_frozen_offsets"
                ]
            )

            first_files = {
                path.relative_to(first): path.read_bytes()
                for path in first.rglob("*") if path.is_file()
            }
            second_files = {
                path.relative_to(second): path.read_bytes()
                for path in second.rglob("*") if path.is_file()
            }
            self.assertEqual(first_files, second_files)

    def test_edge_specific_proxy_preserves_heterogeneity_and_sparse_topology(
        self,
    ) -> None:
        aggregate = fitter.empty_accumulator()
        allowed = set(fitter.topology_ratios())
        for lag in fitter.LAG_SECONDS:
            for source in fitter.EVENT_TYPES:
                for target in fitter.EVENT_TYPES:
                    correlation = 0.0
                    if lag != 0 and (target, source) in allowed:
                        correlation = 0.12
                    if (
                        lag != 0
                        and target == "limit_buy"
                        and source == "limit_buy"
                    ):
                        correlation = 0.30
                    if (
                        lag != 0
                        and target == "limit_sell"
                        and source == "limit_sell"
                    ):
                        correlation = 0.06
                    if (
                        lag != 0
                        and target == "cancel_bid"
                        and source == "limit_buy"
                    ):
                        correlation = -0.40
                    if (
                        lag != 0
                        and target == "market_sell"
                        and source == "market_buy"
                    ):
                        correlation = 0.95
                    aggregate.lag_moments[(source, target, lag)].add(
                        23400 - lag,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        correlation,
                    )

        args = SimpleNamespace(
            lag_shrinkage_symbol_days=25.0,
            fast_correlation_gain=0.75,
            fast_branching_cap=0.20,
            slow_correlation_gain=0.25,
            slow_branching_cap=0.08,
        )
        fast, slow, audit = fitter.data_informed_branching_topology(
            aggregate, args
        )
        index = fitter.EVENT_INDEX
        self.assertGreater(
            fast[index["limit_buy"]][index["limit_buy"]],
            fast[index["limit_sell"]][index["limit_sell"]],
        )
        self.assertGreater(
            slow[index["limit_buy"]][index["limit_buy"]],
            slow[index["limit_sell"]][index["limit_sell"]],
        )
        self.assertEqual(
            fast[index["cancel_bid"]][index["limit_buy"]], 0.0
        )
        self.assertEqual(
            slow[index["cancel_bid"]][index["limit_buy"]], 0.0
        )
        self.assertEqual(
            fast[index["market_sell"]][index["market_buy"]], 0.0
        )
        self.assertEqual(
            slow[index["market_sell"]][index["market_buy"]], 0.0
        )
        self.assertEqual(audit["estimator_version"], 2)
        self.assertEqual(
            audit["fast"]["pre_feasibility_nonzero_edge_count"], 11
        )
        self.assertEqual(len(audit["fast"]["edge_estimates"]), 12)

    def test_global_stability_scale_preserves_every_nonzero_edge_ratio(self) -> None:
        fast = fitter.zero_matrix()
        slow = fitter.zero_matrix()
        fast[0][0] = 0.60
        fast[0][1] = 0.30
        fast[2][2] = 0.20
        slow[1][1] = 0.40
        slow[4][3] = 0.25
        original_fast = [row[:] for row in fast]
        original_slow = [row[:] for row in slow]

        (
            scaled_fast,
            scaled_slow,
            scale,
            radius,
            maximum_row_sum,
            maximum_column_sum,
        ) = fitter.scaled_branching_for_targets(
            [[1.0] * len(fitter.EVENT_TYPES)],
            0.20,
            fast,
            slow,
        )
        self.assertGreater(scale, 0.0)
        self.assertLess(scale, 1.0)
        for original, scaled in zip(original_fast, scaled_fast):
            for before, after in zip(original, scaled):
                self.assertAlmostEqual(after, scale * before, places=14)
        for original, scaled in zip(original_slow, scaled_slow):
            for before, after in zip(original, scaled):
                self.assertAlmostEqual(after, scale * before, places=14)
        self.assertAlmostEqual(
            scaled_fast[0][0] / scaled_fast[0][1], 2.0, places=14
        )
        self.assertLess(radius, 0.20)
        self.assertLess(maximum_row_sum, 0.20)
        self.assertLess(maximum_column_sum, 0.20)

        integrated = fitter.add_matrices(scaled_fast, scaled_slow)
        immigration = fitter.immigration_for_targets(
            [1.0] * len(fitter.EVENT_TYPES), integrated, 0.30
        )
        reconstructed = [
            0.30 * baseline + endogenous
            for baseline, endogenous in zip(
                immigration,
                fitter.matvec(integrated, [1.0] * len(fitter.EVENT_TYPES)),
            )
        ]
        for observed in reconstructed:
            self.assertAlmostEqual(observed, 1.0, places=13)

    def test_shifted_spectral_radius_handles_period_two_sparse_matrix(self) -> None:
        matrix = fitter.zero_matrix()
        matrix[0][1] = 0.20
        matrix[1][0] = 0.80
        self.assertAlmostEqual(fitter.spectral_radius(matrix), 0.40, places=12)

    def test_state_fit_uses_each_symbol_specific_rate_offset(self) -> None:
        first_state = (0, "one_tick", "sell_high", "high", "low")
        second_state = (0, "wider", "buy_high", "low", "high")
        first_rates = (100.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        second_rates = (1.0, 100.0, 1.0, 1.0, 1.0, 1.0)

        def source(
            symbol: str,
            state: tuple[int, str, str, str, str],
            rates: tuple[float, ...],
        ) -> fitter.StateFitSource:
            counts = {
                (state[0], event_type, *state[1:]): int(rate)
                for event_type, rate in zip(fitter.EVENT_TYPES, rates)
            }
            return fitter.StateFitSource(
                date="2019-01-30",
                symbol=symbol,
                counts=counts,
                exposure={state: 1800.0},
                base_rates=rates,
            )

        # The two symbols occupy different queue states and have opposite event
        # mixtures, but each mixture exactly equals its own frozen target-rate
        # offset.  A pooled cluster-mean offset would spuriously estimate a
        # queue-state effect; symbol-specific offsets leave every slope at zero.
        coefficients, diagnostics = fitter.fit_state_coefficient_matrix(
            [
                source("AAA", first_state, first_rates),
                source("BBB", second_state, second_rates),
            ],
            [[1.0] * len(fitter.EVENT_TYPES) for _ in range(fitter.HALF_HOUR_BINS)],
            penalty=25.0,
            bound=1.5,
        )
        self.assertLess(
            max(abs(value) for row in coefficients for value in row),
            1.0e-10,
        )
        self.assertEqual(diagnostics["offset_source_count"], 2)
        self.assertEqual(diagnostics["offset_symbol_count"], 2)
        self.assertFalse(diagnostics["deployment_cluster_mean_used_as_offset"])

    def test_nonfinite_kernel_decay_is_rejected_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = Fixture(root)
            for option in ("--fast-beta", "--slow-beta"):
                for value in ("nan", "inf"):
                    with self.subTest(option=option, value=value):
                        output = root / (
                            option.removeprefix("--").replace("-", "_")
                            + "_" + value
                        )
                        self.assertEqual(
                            fitter.main(fixture.argv(output) + [option, value]),
                            1,
                        )
                        self.assertFalse(output.exists())

    def test_forbidden_date_is_rejected_before_output_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = Fixture(root)
            argv = fixture.argv(root / "forbidden_output")
            first_date_index = argv.index("--training-root") + 1
            argv[first_date_index] = argv[first_date_index].replace(
                "2019-01-30=", "2020-01-30="
            )
            self.assertEqual(fitter.main(argv), 1)
            self.assertFalse((root / "forbidden_output").exists())

    def test_explicit_stratified_subset_fits_clusters_and_maps_full_universe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = Fixture(root)
            subset = root / "fit_symbols.txt"
            subset.write_text("AAA\nBBB\n", encoding="utf-8")
            output = root / "subset_output"
            argv = fixture.argv(output) + ["--fit-symbols-file", str(subset)]
            self.assertEqual(fitter.main(argv), 0)
            manifest = json.loads(
                (output / "training_policy_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["symbol_count"], 3)
            self.assertEqual(manifest["fitting_symbol_count"], 2)
            mapping = read_csv(output / "symbol_policy_mapping.csv")
            self.assertEqual({row["symbol"] for row in mapping}, set(SYMBOLS))
            estimation_mapping = read_csv(
                output / "estimation_symbol_policy_mapping.csv"
            )
            self.assertEqual(
                {row["symbol"] for row in estimation_mapping}, {"AAA", "BBB"}
            )
            estimation_clusters = read_csv(
                output / "estimation_cluster_assignments.csv"
            )
            self.assertEqual(
                {row["symbol"] for row in estimation_clusters}, {"AAA", "BBB"}
            )
            cluster_zero = json.loads(
                (output / "clusters" / "cluster_0" / "policy.json").read_text()
            )
            cluster_one = json.loads(
                (output / "clusters" / "cluster_1" / "policy.json").read_text()
            )
            self.assertEqual(cluster_zero["member_count"], 2)
            self.assertEqual(cluster_zero["estimation_members"], ["AAA"])
            self.assertLess(
                cluster_zero["hawkes"]["lag_moment_proxy"]["fast"]
                    ["pre_feasibility_mean_edge_strength"],
                cluster_one["hawkes"]["lag_moment_proxy"]["fast"]
                    ["pre_feasibility_mean_edge_strength"],
            )

    def test_full_fit_can_emit_a_separate_selection_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = Fixture(root)
            selection = root / "selection_symbols.txt"
            selection.write_text("AAA\nBBB\n", encoding="utf-8")
            output = root / "full_fit_selection_output"
            argv = fixture.argv(output) + [
                "--selection-symbols-file", str(selection),
            ]
            self.assertEqual(fitter.main(argv), 0)
            manifest = json.loads((
                output / "training_policy_manifest.json"
            ).read_text(encoding="utf-8"))
            self.assertEqual(manifest["symbol_count"], 3)
            self.assertEqual(manifest["fitting_symbol_count"], 3)
            self.assertEqual(manifest["selection_symbol_count"], 2)
            self.assertEqual(
                manifest["estimation_mapping_role"],
                "behavioural_selection_subset",
            )
            mapping = read_csv(output / "symbol_policy_mapping.csv")
            self.assertEqual({row["symbol"] for row in mapping}, set(SYMBOLS))
            selection_mapping = read_csv(
                output / "estimation_symbol_policy_mapping.csv"
            )
            self.assertEqual(
                {row["symbol"] for row in selection_mapping},
                {"AAA", "BBB"},
            )
            cluster_zero = json.loads((
                output / "clusters" / "cluster_0" / "policy.json"
            ).read_text())
            self.assertEqual(
                cluster_zero["estimation_members"], ["AAA", "CCC"],
            )

    def test_missing_exposure_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = Fixture(root)
            date, training_root = fixture.training_roots[0]
            exposure = (
                training_root / "empirical_data"
                / f"itch_{date.replace('-', '')}_aaa" / "queue_state_exposure.csv"
            )
            exposure.unlink()
            self.assertEqual(fitter.main(fixture.argv(root / "missing_output")), 1)

    def test_missing_lag_moments_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = Fixture(root)
            date, training_root = fixture.training_roots[0]
            lag_path = (
                training_root / "empirical_data"
                / f"itch_{date.replace('-', '')}_aaa"
                / "event_count_lag_moments.csv"
            )
            lag_path.unlink()
            self.assertEqual(fitter.main(fixture.argv(root / "missing_lag_output")), 1)

    def test_event_count_conservation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = Fixture(root)
            date, training_root = fixture.training_roots[0]
            counts = (
                training_root / "empirical_data"
                / f"itch_{date.replace('-', '')}_aaa" / "queue_state_counts.csv"
            )
            rows = read_csv(counts)
            rows[0]["count"] = str(int(rows[0]["count"]) + 1)
            write_csv(counts, list(rows[0]), [[row[key] for key in rows[0]] for row in rows])
            self.assertEqual(fitter.main(fixture.argv(root / "bad_counts_output")), 1)

    def test_subtick_or_inconsistent_improvement_marks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = Fixture(root)
            date, training_root = fixture.training_roots[0]
            improvement = (
                training_root / "empirical_data"
                / f"itch_{date.replace('-', '')}_aaa"
                / "limit_buy_improvement_distribution.txt"
            )
            write_csv(
                improvement,
                ["improvement_ticks", "improvement_price_units", "count"],
                [[1, 99, 10]],
            )
            self.assertEqual(
                fitter.main(fixture.argv(root / "bad_improvement_output")), 1
            )

    def test_consistent_off_grid_improvement_is_audited_and_projected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "improvements.csv"
            write_csv(
                path,
                ["improvement_ticks", "improvement_price_units", "count"],
                [[1, 100, 7], [0.01, 1, 3]],
            )
            loaded = fitter.load_improvements(path)
            self.assertEqual(loaded.distribution, {100: 7})
            self.assertEqual(loaded.raw_count, 10)
            self.assertEqual(loaded.runtime_compatible_count, 7)
            self.assertEqual(loaded.excluded_off_grid_count, 3)


if __name__ == "__main__":
    unittest.main()
