#!/usr/bin/env python3
"""Independently recompute inventory-stress scientific and execution results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from typing import Iterable

import numpy as np
import pandas as pd


T_CRITICAL_95_DF19 = 2.093024054


def describe(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary input is empty or non-finite")
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if array.size > 1 else 0.0
    se = sd / math.sqrt(array.size)
    half_width = T_CRITICAL_95_DF19 * se if array.size == 20 else 1.96 * se
    return {
        "n": int(array.size),
        "mean": mean,
        "median": float(np.median(array)),
        "sd": sd,
        "se": se,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "cohen_dz": mean / sd if sd > 0.0 else 0.0,
    }


def verify_completion_hashes(root: pathlib.Path) -> dict[str, int]:
    manifest_path = root / "case_job_completion.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = 0
    bytes_checked = 0
    original_marker = f"/results/seagull/{root.name}/"
    for record in payload["hash_bound_artifacts"]:
        original = str(record["path"])
        if original_marker not in original:
            raise ValueError(f"unexpected artifact path: {original}")
        relative = original.split(original_marker, 1)[1]
        local = root / relative
        if not local.is_file():
            raise FileNotFoundError(local)
        data = local.read_bytes()
        if len(data) != int(record["bytes"]):
            raise ValueError(f"size mismatch: {local}")
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"hash mismatch: {local}")
        checked += 1
        bytes_checked += len(data)
    return {"checked_files": checked, "checked_bytes": bytes_checked}


def raw_execution_audit(root: pathlib.Path) -> dict[str, object]:
    expected = {
        "financial_global_raw.csv": 80,
        "financial_uncoupled_raw.csv": 80,
        "financial_shared_off_raw.csv": 40,
        "mechanism_preflight_raw.csv": 4,
        "rank_equivalence_raw.csv": 2,
    }
    result: dict[str, object] = {}
    all_rows: list[pd.DataFrame] = []
    for filename, expected_rows in expected.items():
        frame = pd.read_csv(root / filename)
        if len(frame) != expected_rows:
            raise ValueError(f"{filename}: {len(frame)} != {expected_rows}")
        if frame["state_hash"].isna().any():
            raise ValueError(f"{filename}: missing state hashes")
        result[filename] = {
            "rows": int(len(frame)),
            "wall_seconds_sum": float(frame["wall_seconds"].sum()),
            "wall_seconds_median": float(frame["wall_seconds"].median()),
            "communication_fraction_median": float(
                frame["communication_fraction"].median()
            ),
            "compute_imbalance_median": float(frame["compute_imbalance"].median()),
            "orders_per_wall_second_median": float(
                (frame["processed_orders"] / frame["wall_seconds"]).median()
            ),
        }
        all_rows.append(frame.assign(source=filename))
    combined = pd.concat(all_rows, ignore_index=True)
    rank = pd.read_csv(root / "rank_equivalence_raw.csv").sort_values("ranks")
    if rank["state_hash"].nunique() != 1:
        raise ValueError("rank equivalence hashes differ")
    rank_one = float(rank.loc[rank["ranks"] == 1, "wall_seconds"].iloc[0])
    rank_16 = float(rank.loc[rank["ranks"] == 16, "wall_seconds"].iloc[0])
    result["all_simulator_runs"] = {
        "count": int(len(combined)),
        "wall_seconds_sum": float(combined["wall_seconds"].sum()),
    }
    result["rank_equivalence"] = {
        "state_hash": str(rank["state_hash"].iloc[0]),
        "rank_1_wall_seconds": rank_one,
        "rank_16_wall_seconds": rank_16,
        "speedup": rank_one / rank_16,
        "parallel_efficiency": rank_one / rank_16 / 16.0,
    }
    return result


def endpoint_audit(root: pathlib.Path) -> dict[str, object]:
    frame = pd.read_csv(root / "financial_analysis/paired_liquidity_effects_by_seed.csv")
    if len(frame) != 40:
        raise ValueError("expected 40 seed-capacity endpoint rows")
    outcomes = {
        "depth_units": "mean_depth_difference_in_differences",
        "relative_depth_fraction": "mean_relative_depth_difference_in_differences",
        "spread_bps": "mean_spread_difference_in_differences_bps",
        "affected_fraction": "mean_affected_fraction_difference_in_differences",
        "absorption_fraction": "shared_mm_absorption_fraction",
        "immediate_exposure_delta": "immediate_shock_minus_control_gross_exposure",
        "immediate_utilization_delta": "immediate_shock_minus_control_utilization",
        "immediate_quote_scale_delta": "immediate_shock_minus_control_quote_scale",
        "minimum_quote_scale": "minimum_shared_quote_scale_on_shock",
        "minimum_two_sided_fraction": "minimum_two_sided_book_fraction_on_shock",
    }
    result: dict[str, object] = {}
    for risk, group in frame.groupby("risk_limit_per_asset", sort=True):
        entry: dict[str, object] = {}
        for label, column in outcomes.items():
            entry[label] = describe(group[column])
        entry["absorption_below_preflight_threshold_count"] = int(
            (group["shared_mm_absorption_fraction"] < 0.025).sum()
        )
        entry["quote_scale_at_floor_count"] = int(
            (group["minimum_shared_quote_scale_on_shock"] <= 0.050000001).sum()
        )
        entry["positive_depth_effect_seed_count"] = int(
            (group["mean_depth_difference_in_differences"] > 0.0).sum()
        )
        entry["positive_spread_effect_seed_count"] = int(
            (group["mean_spread_difference_in_differences_bps"] > 0.0).sum()
        )
        result[str(int(risk))] = entry

    indexed = frame.set_index(["seed", "risk_limit_per_asset"])
    paired: dict[str, object] = {}
    for label, column in outcomes.items():
        difference = (
            indexed.xs(800, level="risk_limit_per_asset")[column]
            - indexed.xs(1600, level="risk_limit_per_asset")[column]
        )
        paired[f"800_minus_1600_{label}"] = describe(difference)
    result["capacity_contrast"] = paired

    checkpoints: dict[str, object] = {}
    for second in (1, 5, 30, 300, 1800):
        for metric, prefix in (
            ("depth", "depth_did_at_"),
            ("spread", "spread_did_bps_at_"),
            ("affected", "affected_fraction_did_at_"),
        ):
            column = f"{prefix}{second}s"
            for risk, group in frame.groupby("risk_limit_per_asset", sort=True):
                checkpoints[f"risk{int(risk)}_{metric}_{second}s"] = describe(
                    group[column]
                )
    result["checkpoints"] = checkpoints
    return result


def time_path_audit(root: pathlib.Path) -> dict[str, object]:
    frame = pd.read_csv(root / "financial_analysis/paired_liquidity_effects_by_time.csv")
    frame["elapsed"] = frame["time_seconds"] - 11700.0
    result: dict[str, object] = {}
    for risk, group in frame.groupby("risk_limit_per_asset", sort=True):
        means = group.groupby("elapsed", as_index=False).mean(numeric_only=True)
        entry: dict[str, object] = {}
        for label, column in (
            ("depth", "depth_difference_in_differences"),
            ("relative_depth", "relative_depth_difference_in_differences"),
            ("spread", "spread_difference_in_differences_bps"),
            ("affected", "affected_fraction_difference_in_differences"),
        ):
            maximum = means.loc[means[column].idxmax()]
            minimum = means.loc[means[column].idxmin()]
            entry[f"mean_path_{label}_maximum"] = {
                "value": float(maximum[column]),
                "seconds_after_shock": int(maximum["elapsed"]),
            }
            entry[f"mean_path_{label}_minimum"] = {
                "value": float(minimum[column]),
                "seconds_after_shock": int(minimum["elapsed"]),
            }
        windows = ((1, 30), (31, 300), (301, 1800))
        for lower, upper in windows:
            selection = group[(group["elapsed"] >= lower) & (group["elapsed"] <= upper)]
            entry[f"window_{lower}_{upper}"] = {
                "depth_mean": float(selection["depth_difference_in_differences"].mean()),
                "relative_depth_mean": float(
                    selection["relative_depth_difference_in_differences"].mean()
                ),
                "spread_mean_bps": float(
                    selection["spread_difference_in_differences_bps"].mean()
                ),
                "affected_fraction_mean": float(
                    selection["affected_fraction_difference_in_differences"].mean()
                ),
                "quote_scale_mean": float(
                    selection["global_shock_shared_quote_scale"].mean()
                ),
                "utilization_mean": float(
                    selection["global_shock_shared_utilization"].mean()
                ),
            }
        result[str(int(risk))] = entry
    return result


def cluster_audit(root: pathlib.Path) -> dict[str, object]:
    frame = pd.read_csv(
        root / "financial_cluster_analysis/cluster_liquidity_effects_by_seed.csv"
    )
    assignments = pd.read_csv(root / "portable_case/cluster_assignments.csv")
    counts = assignments.groupby("cluster_id").size().to_dict()
    result: dict[str, object] = {"cluster_symbol_counts": {str(k): int(v) for k, v in counts.items()}}
    rows: list[dict[str, object]] = []
    for (risk, cluster), group in frame.groupby(
        ["risk_limit_per_asset", "cluster_id"], sort=True
    ):
        depth = describe(group["depth_difference_in_differences_percent_of_baseline"])
        spread = describe(group["spread_difference_in_differences_bps"])
        rows.append({
            "risk_limit_per_asset": int(risk),
            "cluster_id": int(cluster),
            "symbol_count": int(counts[int(cluster)]),
            "non_target_symbol_count": int(group["non_target_symbol_count"].iloc[0]),
            "baseline_top_depth": float(group["baseline_uncoupled_top_depth"].mean()),
            "baseline_spread_bps": float(group["baseline_uncoupled_spread_bps"].mean()),
            "depth_effect_percent": depth,
            "spread_effect_bps": spread,
            "depth_ci_excludes_zero": bool(
                depth["ci95_lower"] > 0.0 or depth["ci95_upper"] < 0.0
            ),
            "spread_ci_excludes_zero": bool(
                spread["ci95_lower"] > 0.0 or spread["ci95_upper"] < 0.0
            ),
        })
    result["rows"] = rows
    result["depth_ci_excludes_zero_count"] = sum(
        bool(row["depth_ci_excludes_zero"]) for row in rows
    )
    result["spread_ci_excludes_zero_count"] = sum(
        bool(row["spread_ci_excludes_zero"]) for row in rows
    )
    return result


def direction_audit(root: pathlib.Path) -> dict[str, object]:
    raw = pd.read_csv(root / "financial_global_raw.csv")
    shock = raw[raw["shock_mode"] == "on"].copy()
    rows: list[dict[str, object]] = []
    for record in shock.itertuples(index=False):
        local = root / "financial_targets/global" / pathlib.Path(
            record.shock_targets_csv
        ).name
        targets = pd.read_csv(local)
        targets = targets[targets["is_shock_target"] == 1]
        if len(targets) != 148:
            raise ValueError(f"{local}: expected 148 targets")
        if set(targets["direction_rule"]) != {"inventory_adverse"}:
            raise ValueError(f"{local}: wrong direction rule")
        valid = np.where(
            targets["pre_shock_shared_inventory"].to_numpy() < 0,
            targets["shock_side"].to_numpy() == "buy",
            targets["shock_side"].to_numpy() == "sell",
        )
        if not bool(np.all(valid)):
            raise ValueError(f"{local}: non-adverse direction")
        rows.append({
            "risk_limit_per_asset": int(record.risk_limit_per_asset),
            "seed": int(record.seed),
            "buy_targets": int((targets["shock_side"] == "buy").sum()),
            "sell_targets": int((targets["shock_side"] == "sell").sum()),
            "requested_quantity": int(targets["requested_quantity"].sum()),
        })
    frame = pd.DataFrame(rows)
    return {
        "audited_manifests": len(rows),
        "by_capacity": {
            str(int(risk)): {
                "buy_targets": describe(group["buy_targets"]),
                "sell_targets": describe(group["sell_targets"]),
                "requested_quantity_unique": sorted(
                    int(value) for value in group["requested_quantity"].unique()
                ),
            }
            for risk, group in frame.groupby("risk_limit_per_asset", sort=True)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = {
        "completion_hashes": verify_completion_hashes(root),
        "execution": raw_execution_audit(root),
        "endpoints": endpoint_audit(root),
        "time_paths": time_path_audit(root),
        "clusters": cluster_audit(root),
        "shock_directions": direction_audit(root),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"analysis={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
