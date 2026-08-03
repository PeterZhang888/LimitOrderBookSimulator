import csv
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "validate_shared_dealer_preflight.py"
SPEC = importlib.util.spec_from_file_location("shared_dealer_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREFLIGHT
SPEC.loader.exec_module(PREFLIGHT)


class SharedDealerPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_metrics(
        self, name: str, *, scale: float, active: float,
        resting_depth: float, two_sided_active: float | None = None,
        requested_two_sided: float = 1.0,
    ) -> pathlib.Path:
        if two_sided_active is None:
            two_sided_active = active
        path = self.root / name
        fields = [
            "time_seconds", "shared_quote_scale", "shared_requested_quote_depth",
            "shared_risk_reducing_requested_quote_depth",
            "shared_risk_increasing_requested_quote_depth",
            "shared_resting_quote_depth",
            "shared_risk_reducing_resting_quote_depth",
            "shared_risk_increasing_resting_quote_depth",
            "shared_active_asset_fraction",
            "shared_two_sided_active_asset_fraction",
            "shared_utilization", "shocked_bid_top_depth",
            "shocked_shared_bid_resting_depth",
            "shocked_shared_bid_participation",
            "shared_nonzero_inventory_asset_fraction",
            "mean_absolute_shared_inventory",
            "mean_absolute_shocked_shared_inventory",
            "shared_requested_active_asset_fraction",
            "shared_requested_two_sided_asset_fraction",
            "shared_best_bid_depth", "shared_best_ask_depth",
            "shared_at_best_bid_asset_fraction",
            "shared_at_best_ask_asset_fraction",
            "shared_bbo_depth_participation",
            "shared_gross_exposure",
        ]
        with path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "time_seconds": 9, "shared_quote_scale": scale,
                "shared_requested_quote_depth": 100,
                "shared_risk_reducing_requested_quote_depth": 25,
                "shared_risk_increasing_requested_quote_depth": 75,
                "shared_resting_quote_depth": resting_depth,
                "shared_risk_reducing_resting_quote_depth": min(
                    20.0, resting_depth
                ),
                "shared_risk_increasing_resting_quote_depth": max(
                    0.0, resting_depth - 20.0
                ),
                "shared_active_asset_fraction": active,
                "shared_two_sided_active_asset_fraction": two_sided_active,
                "shared_utilization": 0.6,
                "shocked_bid_top_depth": 100,
                "shocked_shared_bid_resting_depth": 20,
                "shocked_shared_bid_participation": 0.2,
                "shared_nonzero_inventory_asset_fraction": 0.8,
                "mean_absolute_shared_inventory": 250,
                "mean_absolute_shocked_shared_inventory": 275,
                "shared_requested_active_asset_fraction": 1.0,
                "shared_requested_two_sided_asset_fraction": requested_two_sided,
                "shared_best_bid_depth": 20,
                "shared_best_ask_depth": 20,
                "shared_at_best_bid_asset_fraction": active,
                "shared_at_best_ask_asset_fraction": active,
                "shared_bbo_depth_participation": 0.2,
                "shared_gross_exposure": 600,
            })
            # Boundary observations are left limits: the t=10 row is written
            # immediately before the shock stamped t=10 is processed.
            writer.writerow({
                "time_seconds": 10, "shared_quote_scale": scale,
                "shared_requested_quote_depth": 100,
                "shared_risk_reducing_requested_quote_depth": 25,
                "shared_risk_increasing_requested_quote_depth": 75,
                "shared_resting_quote_depth": resting_depth,
                "shared_risk_reducing_resting_quote_depth": min(
                    20.0, resting_depth
                ),
                "shared_risk_increasing_resting_quote_depth": max(
                    0.0, resting_depth - 20.0
                ),
                "shared_active_asset_fraction": active,
                "shared_two_sided_active_asset_fraction": two_sided_active,
                "shared_utilization": 0.6,
                "shocked_bid_top_depth": 100,
                "shocked_shared_bid_resting_depth": 20,
                "shocked_shared_bid_participation": 0.2,
                "shared_nonzero_inventory_asset_fraction": 0.8,
                "mean_absolute_shared_inventory": 250,
                "mean_absolute_shocked_shared_inventory": 275,
                "shared_requested_active_asset_fraction": 1.0,
                "shared_requested_two_sided_asset_fraction": requested_two_sided,
                "shared_best_bid_depth": 20,
                "shared_best_ask_depth": 20,
                "shared_at_best_bid_asset_fraction": active,
                "shared_at_best_ask_asset_fraction": active,
                "shared_bbo_depth_participation": 0.2,
                "shared_gross_exposure": 600,
            })
            is_shock = name == "shock.csv"
            writer.writerow({
                "time_seconds": 11,
                "shared_quote_scale": max(0.0, scale - 0.01) if is_shock else scale,
                "shared_requested_quote_depth": 100,
                "shared_risk_reducing_requested_quote_depth": 25,
                "shared_risk_increasing_requested_quote_depth": 75,
                "shared_resting_quote_depth": resting_depth,
                "shared_risk_reducing_resting_quote_depth": min(20.0, resting_depth),
                "shared_risk_increasing_resting_quote_depth": max(0.0, resting_depth - 20.0),
                "shared_active_asset_fraction": active,
                "shared_two_sided_active_asset_fraction": two_sided_active,
                "shared_utilization": 0.61 if is_shock else 0.6,
                "shocked_bid_top_depth": 100,
                "shocked_shared_bid_resting_depth": 20,
                "shocked_shared_bid_participation": 0.2,
                "shared_nonzero_inventory_asset_fraction": 0.8,
                "mean_absolute_shared_inventory": 250,
                "mean_absolute_shocked_shared_inventory": 275,
                "shared_requested_active_asset_fraction": 1.0,
                "shared_requested_two_sided_asset_fraction": requested_two_sided,
                "shared_best_bid_depth": 20,
                "shared_best_ask_depth": 20,
                "shared_at_best_bid_asset_fraction": active,
                "shared_at_best_ask_asset_fraction": active,
                "shared_bbo_depth_participation": 0.2,
                "shared_gross_exposure": 610 if is_shock else 600,
            })
        return path

    def run_case(
        self, *, scale: float, absorbed: int, resting_depth: float = 90.0,
        two_sided_active: float = 1.0,
        requested_two_sided: float = 1.0,
        minimum_resting_two_sided: float = 1.0,
    ) -> subprocess.CompletedProcess[str]:
        control = self.write_metrics(
            "control.csv", scale=scale, active=1.0,
            resting_depth=resting_depth,
            two_sided_active=two_sided_active,
            requested_two_sided=requested_two_sided,
        )
        shock = self.write_metrics(
            "shock.csv", scale=scale, active=1.0,
            resting_depth=resting_depth,
            two_sided_active=two_sided_active,
            requested_two_sided=requested_two_sided,
        )
        raw = self.root / "raw.csv"
        fields = [
            "shared_mm_mode", "risk_limit_per_asset", "shock_mode",
            "metrics_csv", "shock_executed_quantity", "shock_shared_mm_quantity",
            "shock_local_mm_quantity", "shock_value_agent_quantity",
            "shock_background_quantity", "shock_other_quantity",
        ]
        with raw.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "shared_mm_mode": "global", "risk_limit_per_asset": 100,
                "shock_mode": "off", "metrics_csv": control,
                "shock_executed_quantity": 0, "shock_shared_mm_quantity": 0,
                "shock_local_mm_quantity": 0, "shock_value_agent_quantity": 0,
                "shock_background_quantity": 0, "shock_other_quantity": 0,
            })
            writer.writerow({
                "shared_mm_mode": "global", "risk_limit_per_asset": 100,
                "shock_mode": "on", "metrics_csv": shock,
                "shock_executed_quantity": 50, "shock_shared_mm_quantity": absorbed,
                "shock_local_mm_quantity": 10,
                "shock_value_agent_quantity": 5,
                "shock_background_quantity": 15,
                "shock_other_quantity": 20 - absorbed,
            })
        return subprocess.run(
            [
                "python3", str(SCRIPT), "--raw", str(raw),
                "--shock-time-seconds", "10", "--output", str(self.root / "out.json"),
                "--lookback-seconds", "2",
                "--minimum-resting-two-sided-active-asset-fraction",
                str(minimum_resting_two_sided),
            ],
            text=True, capture_output=True, check=False,
        )

    def test_passes_active_absorbing_dealer(self) -> None:
        completed = self.run_case(scale=0.4, absorbed=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((self.root / "out.json").read_text())
        self.assertEqual(payload["status"], "passed")

    def test_rejects_dealer_that_is_present_but_economically_throttled(self) -> None:
        completed = self.run_case(scale=0.1155296705, absorbed=20)
        self.assertNotEqual(completed.returncode, 0)
        payload = json.loads((self.root / "out.json").read_text())
        self.assertTrue(any(
            "economic activity gate" in item for item in payload["failures"]
        ))

    def test_rejects_dealer_at_numerical_floor(self) -> None:
        completed = self.run_case(scale=0.05, absorbed=20)
        self.assertNotEqual(completed.returncode, 0)
        payload = json.loads((self.root / "out.json").read_text())
        self.assertTrue(any(
            "not above the numerical floor" in item
            for item in payload["failures"]
        ))

    def test_rejects_inactive_dealer(self) -> None:
        completed = self.run_case(scale=0.0, absorbed=0)
        self.assertNotEqual(completed.returncode, 0)
        payload = json.loads((self.root / "out.json").read_text())
        self.assertEqual(payload["status"], "failed")
        self.assertTrue(any("quote scale" in item for item in payload["failures"]))

    def test_rejects_requested_but_not_actually_resting_quotes(self) -> None:
        completed = self.run_case(scale=0.4, absorbed=20, resting_depth=0.0)
        self.assertNotEqual(completed.returncode, 0)
        payload = json.loads((self.root / "out.json").read_text())
        self.assertTrue(any(
            "no standing shared quote" in item for item in payload["failures"]
        ))

    def test_rejects_one_sided_presence_in_any_book(self) -> None:
        completed = self.run_case(
            scale=0.4, absorbed=20,
            requested_two_sided=1479.0 / 1480.0,
        )
        self.assertNotEqual(completed.returncode, 0)
        payload = json.loads((self.root / "out.json").read_text())
        self.assertTrue(any(
            "two-sided" in item
            for item in payload["failures"]
        ))

    def test_accepts_material_resting_coverage_with_universal_intent(self) -> None:
        completed = self.run_case(
            scale=0.4,
            absorbed=20,
            two_sided_active=0.96,
            requested_two_sided=1.0,
            minimum_resting_two_sided=0.95,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_audits_inventory_adverse_target_directions(self) -> None:
        path = self.root / "targets.csv"
        fields = [
            "asset_id", "is_shock_target", "shock_enabled",
            "requested_quantity", "requested_sell_quantity",
            "requested_buy_quantity", "shock_side",
            "pre_shock_shared_inventory", "direction_rule",
        ]
        rows = [
            {
                "asset_id": 0, "is_shock_target": 1, "shock_enabled": 1,
                "requested_quantity": 300, "requested_sell_quantity": 300,
                "requested_buy_quantity": 0, "shock_side": "sell",
                "pre_shock_shared_inventory": 4,
                "direction_rule": "inventory_adverse",
            },
            {
                "asset_id": 1, "is_shock_target": 1, "shock_enabled": 1,
                "requested_quantity": 200, "requested_sell_quantity": 0,
                "requested_buy_quantity": 200, "shock_side": "buy",
                "pre_shock_shared_inventory": -3,
                "direction_rule": "inventory_adverse",
            },
        ]
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        run = {
            "requested_shock_inventory_adverse": "1",
            "shock_targets_csv": str(path),
            "shock_targets_csv_sha256": PREFLIGHT.sha256_file(path),
            "shock_requested_quantity": "500",
        }
        self.assertEqual(
            PREFLIGHT.audit_inventory_adverse_targets(
                run, source=self.root / "raw.csv",
            ),
            {
                "target_count": 2,
                "buy_target_count": 1,
                "sell_target_count": 1,
                "requested_quantity": 500,
            },
        )
        rows[1]["shock_side"] = "sell"
        rows[1]["requested_buy_quantity"] = 0
        rows[1]["requested_sell_quantity"] = 200
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        run["shock_targets_csv_sha256"] = PREFLIGHT.sha256_file(path)
        with self.assertRaisesRegex(
            PREFLIGHT.PreflightError, "not inventory-adverse"
        ):
            PREFLIGHT.audit_inventory_adverse_targets(
                run, source=self.root / "raw.csv",
            )


if __name__ == "__main__":
    unittest.main()
