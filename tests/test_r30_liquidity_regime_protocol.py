from __future__ import annotations

import csv
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_queue_reactive_model as driver  # noqa: E402


class LiquidityRegimeProtocolTests(unittest.TestCase):
    def test_slurm_launcher_separates_heldout_runtime_and_targets(self) -> None:
        launcher = (
            ROOT / "submit_queue_reactive_full_validation_hpc.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'HELDOUT_CONFIG="${POOL_ROOT}/heldout_common.csv"', launcher
        )
        self.assertIn(
            'POOLING_PROVENANCE="${POOL_ROOT}/pooling_provenance.json"',
            launcher,
        )
        self.assertIn("scripts/prepare_heldout_target_config.py", launcher)
        self.assertIn("--pooling-provenance", launcher)
        self.assertIn("--expected-date 2020-01-30", launcher)
        self.assertIn(
            '--heldout-target-config "${HELDOUT_TARGET_CONFIG}"', launcher
        )
        self.assertNotIn(
            '--heldout-target-config "${HELDOUT_CONFIG}"', launcher
        )

    def test_hash_bound_training_seed(self) -> None:
        candidates, record = driver.load_training_refinement_seed(
            ROOT / "config/r30_r28_training_refinement_seed.json",
            cluster_ids={str(value) for value in range(10)},
        )
        self.assertEqual(len(candidates), 10)
        self.assertEqual(candidates["2"].order_flow_coupling, 0.963)
        self.assertEqual(candidates["6"].order_flow_coupling, 0.0)
        self.assertTrue(record["training_only"])
        self.assertFalse(record["heldout_inputs_read"])
        self.assertEqual(len(record["verified_structural_repair_sources"]), 8)

    def test_successful_checkpoint_is_reused_only_after_hash_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory) / "run"
            summary = run_dir / "fragmented_asset_summary.csv"
            command = [
                sys.executable,
                "-c",
                (
                    "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("
                    "'asset_id,symbol\\n0,X\\n')"
                ),
                str(summary),
            ]
            first = driver.execute_run(
                command=command,
                run_dir=run_dir,
                timeout_seconds=30,
                mpi_ranks=1,
            )
            resumed = driver.execute_run(
                command=command,
                run_dir=run_dir,
                timeout_seconds=30,
                mpi_ranks=1,
            )
            self.assertTrue(first["success"])
            self.assertTrue(resumed["resumed_from_verified_checkpoint"])

    def test_acf_residual_updates_stochastic_baseline_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            residuals = pathlib.Path(directory) / "symbol_residuals.csv"
            with residuals.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=(
                        "cluster_id", "metric", "target",
                        "simulated_seed_mean",
                    ),
                )
                writer.writeheader()
                writer.writerows((
                    {"cluster_id": "0", "metric": "return_variance",
                     "target": 1.0, "simulated_seed_mean": 1.0},
                    {"cluster_id": "0", "metric": "return_kurtosis",
                     "target": 3.0, "simulated_seed_mean": 3.0},
                    {"cluster_id": "0", "metric": "absolute_return_acf1",
                     "target": 0.10, "simulated_seed_mean": 0.05},
                ))
            current = {
                "0": driver.VolatilityCandidate(
                    identifier="initial",
                    variance_scale=1.0,
                    persistence=0.95,
                    std=0.25,
                    excess_kurtosis_share=1.0,
                )
            }
            refined, audit = driver.refine_full_universe_volatility(
                symbol_residuals=residuals,
                current=current,
                cluster_ids={"0"},
                iteration=1,
            )
            self.assertAlmostEqual(refined["0"].order_flow_coupling, 0.35)
            self.assertEqual(
                audit["method"],
                "session_recentred_cluster_activity_update_v1",
            )

    def test_global_training_scale_is_bounded_and_recorded(self) -> None:
        source = {
            "0": driver.VolatilityCandidate(
                identifier="seed",
                variance_scale=1.0,
                persistence=0.95,
                std=0.25,
                excess_kurtosis_share=1.0,
                order_flow_coupling=0.5,
            ),
            "1": driver.VolatilityCandidate(
                identifier="zero",
                variance_scale=1.0,
                persistence=0.0,
                std=0.0,
                excess_kurtosis_share=1.0,
                order_flow_coupling=0.0,
            ),
        }
        scaled, record = driver.scale_training_coupling(source, 1.25)
        self.assertEqual(scaled["0"].order_flow_coupling, 0.625)
        self.assertEqual(scaled["1"].order_flow_coupling, 0.0)
        self.assertEqual(record["global_coupling_scale"], 1.25)
        self.assertEqual(record["projection_bounds"], [0.0, 2.5])
        self.assertTrue(record["training_only"])


if __name__ == "__main__":
    unittest.main()
