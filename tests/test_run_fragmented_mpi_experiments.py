#!/usr/bin/env python3
"""Command-construction tests for cluster-policy fragmented MPI campaigns."""

from __future__ import annotations

import csv
import importlib.util
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fragmented_runner",
    ROOT / "scripts" / "run_fragmented_mpi_experiments.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class FragmentedRunnerPolicyTest(unittest.TestCase):
    @staticmethod
    def option_value(command: list[str], option: str) -> str:
        return command[command.index(option) + 1]

    @staticmethod
    def write_background_bundle(root: pathlib.Path) -> pathlib.Path:
        policy = root / "cluster_policy.csv"
        buy = root / "buy_improvement.csv"
        sell = root / "sell_improvement.csv"
        policy.write_text("kind,target,source,bin,value\n", encoding="utf-8")
        for path in (buy, sell):
            path.write_text(
                "improvement_ticks,improvement_price_units,count\n1,100,1\n",
                encoding="utf-8",
            )
        mapping = root / "background_mapping.csv"
        mapping.write_text(
            "symbol,cluster_id,policy_file,limit_buy_improvement_file,"
            "limit_sell_improvement_file\n"
            "QQQ,0,cluster_policy.csv,buy_improvement.csv,sell_improvement.csv\n",
            encoding="utf-8",
        )
        return mapping

    def test_queue_background_is_fail_closed_forwarded_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            executable = root / "fragmented_mpi_lob"
            executable.write_text("binary", encoding="utf-8")
            config = root / "universe.csv"
            config.write_text("book_id,symbol\n0,QQQ\n", encoding="utf-8")
            mapping = self.write_background_bundle(root)
            output = root / "raw.csv"
            commands: list[list[str]] = []

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                return subprocess.CompletedProcess(
                    command, 0,
                    stdout=(
                        "fragmented_mpi_lob ranks=1 assets=1 lobs=1 "
                        "wall_seconds=0.1 processed_orders=1 state_hash=0x1 "
                        "risk_limit_per_asset=100 "
                        "background_model=queue-reactive-v1 "
                        f"background_policy={mapping.resolve()}\n"
                    ),
                    stderr="",
                )

            argv = [
                "runner", "--executable", str(executable),
                "--universe-config", str(config), "--output", str(output),
                "--ranks", "1", "--risk-limits", "100",
                "--shared-mm-modes", "off", "--shock-modes", "off",
                "--duration-seconds", "60", "--repetitions", "1",
                "--background-model", "queue-reactive-v1",
                "--background-policy-csv", str(mapping),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                RUNNER, "run_mpi_command", side_effect=fake_run,
            ):
                self.assertEqual(RUNNER.main(), 0)

            self.assertEqual(len(commands), 1)
            command = commands[0]
            executable_index = command.index(str(executable.resolve()))
            self.assertGreater(command.index("--background-model"), executable_index)
            self.assertEqual(
                self.option_value(command, "--background-model"),
                "queue-reactive-v1",
            )
            self.assertEqual(
                self.option_value(command, "--background-policy-csv"),
                str(mapping.resolve()),
            )
            with output.open(newline="", encoding="utf-8") as source:
                row = next(csv.DictReader(source))
            self.assertEqual(row["background_model"], "queue-reactive-v1")
            self.assertEqual(
                row["background_policy_sha256"], RUNNER.sha256_file(mapping)
            )
            digest, count = RUNNER.background_artifact_manifest(mapping.resolve())
            self.assertEqual(row["background_artifacts_sha256"], digest)
            self.assertEqual(row["background_artifact_count"], str(count))
            with output.with_name("raw_summary.csv").open(
                newline="", encoding="utf-8",
            ) as source:
                summary = next(csv.DictReader(source))
            self.assertEqual(summary["background_model"], "queue-reactive-v1")
            self.assertEqual(
                summary["background_policy_sha256"],
                RUNNER.sha256_file(mapping),
            )
            self.assertEqual(summary["background_artifacts_sha256"], digest)
            self.assertEqual(summary["background_artifact_count"], str(count))

            # A referenced policy mutation changes the transitive identity even
            # though the mapping CSV itself is unchanged.
            before = digest
            (root / "cluster_policy.csv").write_text(
                "kind,target,source,bin,value\n# changed\n", encoding="utf-8"
            )
            after, _ = RUNNER.background_artifact_manifest(mapping.resolve())
            self.assertNotEqual(before, after)

    def test_background_model_and_policy_must_be_paired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            executable = root / "fragmented_mpi_lob"
            executable.write_text("binary", encoding="utf-8")
            config = root / "universe.csv"
            config.write_text("book_id,symbol\n0,QQQ\n", encoding="utf-8")
            mapping = self.write_background_bundle(root)
            base = [
                "runner", "--executable", str(executable),
                "--universe-config", str(config),
                "--output", str(root / "raw.csv"),
            ]
            for extra in (
                ["--background-model", "queue-reactive-v1"],
                ["--background-policy-csv", str(mapping)],
            ):
                with self.subTest(extra=extra), mock.patch.object(
                    sys, "argv", [*base, *extra],
                ):
                    with self.assertRaisesRegex(SystemExit, "requires"):
                        RUNNER.main()

    def test_policy_and_asset_summary_are_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            executable = root / "fragmented_mpi_lob"
            executable.write_text("", encoding="utf-8")
            config = root / "universe.csv"
            config.write_text("book_id,symbol\n0,QQQ\n", encoding="utf-8")
            policy = root / "policy.csv"
            policy.write_text(
                "symbol,enabled,value_threshold_bps,value_order_quantity\n"
                "QQQ,on,8,50\n",
                encoding="utf-8",
            )
            output = root / "raw.csv"
            summary = root / "summary.csv"
            summaries = root / "asset_summaries"
            commands: list[list[str]] = []

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                for option in (
                    "--metrics-csv", "--shock-targets-csv", "--asset-summary-csv",
                ):
                    if option in command:
                        artifact = pathlib.Path(self.option_value(command, option))
                        artifact.parent.mkdir(parents=True, exist_ok=True)
                        artifact.write_text("header\n", encoding="utf-8")
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=(
                        "fragmented_mpi_lob ranks=1 assets=1 lobs=1 "
                        "wall_seconds=0.1 processed_orders=1 state_hash=0x1 "
                        "risk_limit_per_asset=100\n"
                    ),
                    stderr="",
                )

            argv = [
                "runner",
                "--executable", str(executable),
                "--universe-config", str(config),
                "--output", str(output),
                "--summary", str(summary),
                "--ranks", "1",
                "--risk-limits", "100",
                "--shared-mm-modes", "off",
                "--shock-modes", "off",
                "--duration-seconds", "60",
                "--shock-time-seconds", "30",
                "--repetitions", "1",
                "--value-agent-policy-csv", str(policy),
                "--asset-summary-dir", str(summaries),
                "--asset-summary-interval-ms", "60000",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                RUNNER, "run_mpi_command", side_effect=fake_run
            ):
                self.assertEqual(RUNNER.main(), 0)

            self.assertEqual(len(commands), 1)
            command = commands[0]
            self.assertIn("--value-agent-policy-csv", command)
            self.assertIn(str(policy.resolve()), command)
            self.assertIn("--asset-summary-csv", command)
            self.assertIn("--asset-summary-interval-ms", command)
            self.assertIn("60000.0", command)
            self.assertNotIn("--venues", command)
            self.assertEqual(
                self.option_value(command, "--hawkes-activity-scale"), "0.3"
            )
            self.assertEqual(
                self.option_value(command, "--local-mm-interval-ms"), "1000.0"
            )
            self.assertEqual(
                self.option_value(command, "--local-mm-quantity-multiplier"), "1.0"
            )
            self.assertEqual(
                self.option_value(command, "--local-mm-improvement-probability"),
                "0.0",
            )
            self.assertEqual(
                self.option_value(command, "--local-mm-spread-elasticity"),
                "0.0",
            )
            self.assertEqual(
                self.option_value(
                    command, "--local-mm-max-improvement-probability"
                ),
                "1.0",
            )
            self.assertEqual(
                self.option_value(command, "--shared-quote-quantity"), "200"
            )
            with output.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["value_agent_policy_sha256"], RUNNER.sha256_file(policy))
            self.assertTrue(rows[0]["asset_summary_csv"].endswith(".csv"))
            self.assertEqual(rows[0]["hawkes_activity_scale"], "0.3")
            self.assertEqual(rows[0]["local_mm_interval_ms"], "1000.0")
            self.assertEqual(rows[0]["local_mm_quantity_multiplier"], "1.0")
            self.assertEqual(rows[0]["local_mm_improvement_probability"], "0.0")
            self.assertEqual(rows[0]["local_mm_spread_elasticity"], "0.0")
            self.assertEqual(
                rows[0]["local_mm_max_improvement_probability"], "1.0"
            )
            self.assertEqual(rows[0]["shared_quote_quantity"], "200")

    def test_control_matrix_is_forwarded_and_never_pooled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            executable = root / "fragmented_mpi_lob"
            executable.write_text("", encoding="utf-8")
            config = root / "base.csv"
            config.write_text("book_id,symbol\n0,QQQ\n", encoding="utf-8")
            output = root / "raw.csv"
            summary = root / "summary.csv"
            metrics = root / "metrics"
            commands: list[list[str]] = []

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                for option in (
                    "--metrics-csv", "--shock-targets-csv", "--asset-summary-csv",
                ):
                    if option in command:
                        artifact = pathlib.Path(self.option_value(command, option))
                        artifact.parent.mkdir(parents=True, exist_ok=True)
                        artifact.write_text("header\n", encoding="utf-8")
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=(
                        "fragmented_mpi_lob ranks=1 assets=1 lobs=1 "
                        "wall_seconds=0.1 processed_orders=1 state_hash=0x1 "
                        "risk_limit_per_asset=100\n"
                    ),
                    stderr="",
                )

            argv = [
                "runner",
                "--executable", str(executable),
                "--base-config", str(config),
                "--output", str(output),
                "--summary", str(summary),
                "--metrics-dir", str(metrics),
                "--assets", "1",
                "--ranks", "1",
                "--risk-limits", "100",
                "--shared-mm-modes", "on",
                "--shock-modes", "off",
                "--duration-seconds", "60",
                "--repetitions", "1",
                "--hawkes-activity-scales", "0.2,0.4",
                "--local-mm-intervals-ms", "500,1000",
                "--local-mm-quantity-multipliers", "0.5,1",
                "--local-mm-improvement-probabilities", "0,0.5",
                "--local-mm-spread-elasticities", "0,1",
                "--local-mm-max-improvement-probabilities", "0.5,1",
                "--shared-quote-quantities", "100,400",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                RUNNER, "run_mpi_command", side_effect=fake_run
            ):
                self.assertEqual(RUNNER.main(), 0)

            self.assertEqual(len(commands), 128)
            command_scenarios = {
                (
                    self.option_value(command, "--hawkes-activity-scale"),
                    self.option_value(command, "--local-mm-interval-ms"),
                    self.option_value(command, "--local-mm-quantity-multiplier"),
                    self.option_value(command, "--local-mm-improvement-probability"),
                    self.option_value(command, "--local-mm-spread-elasticity"),
                    self.option_value(
                        command, "--local-mm-max-improvement-probability"
                    ),
                    self.option_value(command, "--shared-quote-quantity"),
                )
                for command in commands
            }
            self.assertEqual(len(command_scenarios), 128)

            with output.open(newline="", encoding="utf-8") as source:
                raw_rows = list(csv.DictReader(source))
            self.assertEqual(len(raw_rows), 128)
            raw_scenarios = {
                (
                    row["hawkes_activity_scale"],
                    row["local_mm_interval_ms"],
                    row["local_mm_quantity_multiplier"],
                    row["local_mm_improvement_probability"],
                    row["local_mm_spread_elasticity"],
                    row["local_mm_max_improvement_probability"],
                    row["shared_quote_quantity"],
                )
                for row in raw_rows
            }
            self.assertEqual(raw_scenarios, command_scenarios)
            self.assertEqual(len({row["control_scenario"] for row in raw_rows}), 128)
            self.assertEqual(len({row["metrics_csv"] for row in raw_rows}), 128)

            with summary.open(newline="", encoding="utf-8") as source:
                summary_rows = list(csv.DictReader(source))
            self.assertEqual(len(summary_rows), 128)
            summary_scenarios = {
                (
                    row["hawkes_activity_scale"],
                    row["local_mm_interval_ms"],
                    row["local_mm_quantity_multiplier"],
                    row["local_mm_improvement_probability"],
                    row["local_mm_spread_elasticity"],
                    row["local_mm_max_improvement_probability"],
                    row["shared_quote_quantity"],
                )
                for row in summary_rows
            }
            self.assertEqual(summary_scenarios, command_scenarios)

    def test_relative_shared_quotes_reject_ignored_fixed_quantity_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            executable = root / "fragmented_mpi_lob"
            executable.write_text("", encoding="utf-8")
            config = root / "base.csv"
            config.write_text("book_id,symbol\n0,QQQ\n", encoding="utf-8")
            argv = [
                "runner",
                "--executable", str(executable),
                "--base-config", str(config),
                "--output", str(root / "raw.csv"),
                "--assets", "1",
                "--ranks", "1",
                "--duration-seconds", "60",
                "--shared-quote-relative",
                "--shared-quote-quantities", "100,200",
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(
                    SystemExit, "cannot be combined with multiple"
                ):
                    RUNNER.main()

    def test_disabled_shared_mm_rejects_quote_quantity_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            executable = root / "fragmented_mpi_lob"
            executable.write_text("", encoding="utf-8")
            config = root / "base.csv"
            config.write_text("book_id,symbol\n0,QQQ\n", encoding="utf-8")
            argv = [
                "runner",
                "--executable", str(executable),
                "--base-config", str(config),
                "--output", str(root / "raw.csv"),
                "--assets", "1",
                "--ranks", "1",
                "--duration-seconds", "60",
                "--shared-mm-modes", "on,off",
                "--shared-quote-quantities", "100,200",
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(
                    SystemExit, "shared-mm=off ignores quote quantity"
                ):
                    RUNNER.main()

    def test_timeout_kills_and_reaps_launcher_process_group(self) -> None:
        process = mock.Mock()
        process.pid = 4321
        process.returncode = -signal.SIGKILL
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(
                ["mpirun", "-np", "2", "simulator"], 1.0,
                output="partial stdout", stderr="partial stderr",
            ),
            subprocess.TimeoutExpired(["mpirun"], 5.0),
            ("final stdout", "final stderr"),
        ]
        with mock.patch.object(
            RUNNER.subprocess, "Popen", return_value=process
        ) as popen, mock.patch.object(RUNNER.os, "killpg") as killpg:
            with self.assertRaises(RUNNER.MpiRunTimeout) as raised:
                RUNNER.run_mpi_command(
                    ["mpirun", "-np", "2", "simulator"],
                    environment={"PATH": os.environ.get("PATH", "")},
                    timeout_seconds=1.0,
                )
        self.assertEqual(raised.exception.stdout, "final stdout")
        self.assertEqual(raised.exception.stderr, "final stderr")
        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(4321, signal.SIGTERM),
                mock.call(4321, signal.SIGKILL),
            ],
        )
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_timeout_checkpoint_and_resume_only_missing_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            executable = root / "fragmented_mpi_lob"
            executable.write_text("binary", encoding="utf-8")
            config = root / "base.csv"
            config.write_text("book_id,symbol\n0,QQQ\n", encoding="utf-8")
            output = root / "raw.csv"
            summary = root / "summary.csv"
            commands: list[list[str]] = []

            def result(command: list[str]) -> subprocess.CompletedProcess[str]:
                rank = command[command.index("-np") + 1]
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=(
                        f"fragmented_mpi_lob ranks={rank} assets=1 lobs=1 "
                        "wall_seconds=0.1 processed_orders=1 state_hash=0x1 "
                        "risk_limit_per_asset=100\n"
                    ),
                    stderr="",
                )

            def first_attempt(
                command: list[str], **_: object,
            ) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                rank = command[command.index("-np") + 1]
                if rank == "2":
                    raise RUNNER.MpiRunTimeout(command, 1.0, "", "timed out")
                return result(command)

            argv = [
                "runner",
                "--executable", str(executable),
                "--base-config", str(config),
                "--output", str(output),
                "--summary", str(summary),
                "--assets", "1",
                "--ranks", "1,2",
                "--risk-limits", "100",
                "--shared-mm-modes", "off",
                "--shock-modes", "off",
                "--duration-seconds", "60",
                "--repetitions", "1",
                "--run-timeout-seconds", "1",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                RUNNER, "run_mpi_command", side_effect=first_attempt
            ):
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    RUNNER.main()
            self.assertTrue(output.is_file())
            self.assertFalse(summary.exists())
            with output.open(newline="", encoding="utf-8") as source:
                checkpoint = list(csv.DictReader(source))
            self.assertEqual([row["ranks"] for row in checkpoint], ["1"])
            self.assertTrue(checkpoint[0]["campaign_sha256"])
            self.assertTrue(checkpoint[0]["run_key"])

            resumed_commands: list[list[str]] = []

            def resume_attempt(
                command: list[str], **_: object,
            ) -> subprocess.CompletedProcess[str]:
                resumed_commands.append(command)
                return result(command)

            with mock.patch.object(sys, "argv", [*argv, "--resume"]), \
                    mock.patch.object(
                        RUNNER, "run_mpi_command", side_effect=resume_attempt
                    ):
                self.assertEqual(RUNNER.main(), 0)
            self.assertEqual(len(resumed_commands), 1)
            self.assertEqual(
                resumed_commands[0][resumed_commands[0].index("-np") + 1], "2"
            )
            with output.open(newline="", encoding="utf-8") as source:
                completed_rows = list(csv.DictReader(source))
            self.assertEqual([row["ranks"] for row in completed_rows], ["1", "2"])
            self.assertEqual(len({row["run_key"] for row in completed_rows}), 2)
            self.assertTrue(summary.is_file())

            with mock.patch.object(sys, "argv", [*argv, "--resume"]), \
                    mock.patch.object(RUNNER, "run_mpi_command") as no_run:
                self.assertEqual(RUNNER.main(), 0)
            no_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
