#!/usr/bin/env python3
"""Derive plotting tables from an inventory-stress result directory."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


T95_19 = 2.093
SHOCK_TIME_SECONDS = 11700
CHECKPOINTS = (1, 2, 3, 4, 5, 10, 15, 20, 30, 300, 1800)
EARLY_SECONDS = (2, 3, 4, 5, 10)


def ci(values: pd.Series) -> tuple[float, float, float, int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    count = len(clean)
    mean = float(clean.mean())
    if count < 2:
        return mean, math.nan, math.nan, count
    half_width = T95_19 * float(clean.std(ddof=1)) / math.sqrt(count)
    return mean, mean - half_width, mean + half_width, count


def summarize(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group, subframe in frame.groupby(group_columns, sort=True):
        key = group if isinstance(group, tuple) else (group,)
        row: dict[str, float | int | str] = dict(zip(group_columns, key))
        for column in value_columns:
            mean, lower, upper, count = ci(subframe[column])
            row[f"{column}_mean"] = mean
            row[f"{column}_lower"] = lower
            row[f"{column}_upper"] = upper
            row[f"{column}_n"] = count
        rows.append(row)
    return pd.DataFrame(rows)


def load_path_map(root: Path, mode: str) -> dict[tuple[int, int, str], tuple[Path, Path]]:
    raw = pd.read_csv(root / f"financial_{mode}_raw.csv")
    result: dict[tuple[int, int, str], tuple[Path, Path]] = {}
    for row in raw.itertuples(index=False):
        key = (int(row.risk_limit_per_asset), int(row.seed), str(row.shock_mode))
        metrics = root / "financial_metrics" / mode / Path(row.metrics_csv).name
        clusters = root / "financial_cluster_metrics" / mode / Path(row.cluster_metrics_csv).name
        result[key] = (metrics, clusters)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = args.result_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paired_time = pd.read_csv(
        root / "financial_analysis" / "paired_liquidity_effects_by_time.csv"
    )
    paired_time["relative_depth_effect_percent"] = (
        100.0 * paired_time["relative_depth_difference_in_differences"]
    )
    paired_time["post_shock_seconds"] = (
        paired_time["time_seconds"] - SHOCK_TIME_SECONDS
    )
    time_summary = summarize(
        paired_time,
        ["risk_limit_per_asset", "post_shock_seconds"],
        [
            "relative_depth_effect_percent",
            "depth_difference_in_differences",
            "spread_difference_in_differences_bps",
        ],
    )
    time_summary.to_csv(output_dir / "marketwide_time_summary.csv", index=False)

    path_map = load_path_map(root, "global")
    raw_rows: list[pd.DataFrame] = []
    seeds = list(range(20200130, 20200150))
    for risk in (800, 1600):
        for seed in seeds:
            control = pd.read_csv(path_map[(risk, seed, "off")][0])
            shock = pd.read_csv(path_map[(risk, seed, "on")][0])
            merged = control.merge(
                shock,
                on="time_seconds",
                suffixes=("_control", "_shock"),
                validate="one_to_one",
            )
            merged["post_shock_seconds"] = (
                merged["time_seconds"] - SHOCK_TIME_SECONDS
            )
            merged = merged[
                merged["post_shock_seconds"].isin(CHECKPOINTS)
            ].copy()
            merged["risk_limit_per_asset"] = risk
            merged["seed"] = seed
            merged["gross_exposure_delta"] = (
                merged["shared_gross_exposure_shock"]
                - merged["shared_gross_exposure_control"]
            )
            merged["quote_scale_delta"] = (
                merged["shared_quote_scale_shock"]
                - merged["shared_quote_scale_control"]
            )
            merged["requested_depth_reduction"] = (
                merged["unshocked_shared_requested_quote_depth_control"]
                - merged["unshocked_shared_requested_quote_depth_shock"]
            )
            raw_rows.append(
                merged[
                    [
                        "risk_limit_per_asset",
                        "seed",
                        "post_shock_seconds",
                        "gross_exposure_delta",
                        "quote_scale_delta",
                        "requested_depth_reduction",
                    ]
                ]
            )
    mechanism = pd.concat(raw_rows, ignore_index=True)
    mechanism.to_csv(output_dir / "mechanism_checkpoint_by_seed.csv", index=False)
    mechanism_summary = summarize(
        mechanism,
        ["risk_limit_per_asset", "post_shock_seconds"],
        ["gross_exposure_delta", "quote_scale_delta", "requested_depth_reduction"],
    )
    mechanism_summary.to_csv(
        output_dir / "mechanism_checkpoint_summary.csv", index=False
    )

    cluster_rows: list[pd.DataFrame] = []
    for risk in (800, 1600):
        for seed in seeds:
            control = pd.read_csv(path_map[(risk, seed, "off")][1])
            shock = pd.read_csv(path_map[(risk, seed, "on")][1])
            merged = control.merge(
                shock,
                on=["time_seconds", "cluster_id"],
                suffixes=("_control", "_shock"),
                validate="one_to_one",
            )
            merged["post_shock_seconds"] = (
                merged["time_seconds"] - SHOCK_TIME_SECONDS
            )
            merged = merged[
                merged["post_shock_seconds"].isin(EARLY_SECONDS)
            ].copy()
            merged["relative_depth_effect_percent"] = 100.0 * (
                merged["mean_top_depth_control"]
                - merged["mean_top_depth_shock"]
            ) / merged["mean_top_depth_control"]
            merged["spread_effect_bps"] = (
                merged["mean_spread_bps_shock"]
                - merged["mean_spread_bps_control"]
            )
            per_seed = (
                merged.groupby("cluster_id", as_index=False)[
                    ["relative_depth_effect_percent", "spread_effect_bps"]
                ]
                .mean()
                .assign(risk_limit_per_asset=risk, seed=seed)
            )
            cluster_rows.append(per_seed)
    early_clusters = pd.concat(cluster_rows, ignore_index=True)
    early_clusters.to_csv(output_dir / "cluster_early_response_by_seed.csv", index=False)
    early_cluster_summary = summarize(
        early_clusters,
        ["risk_limit_per_asset", "cluster_id"],
        ["relative_depth_effect_percent", "spread_effect_bps"],
    )
    early_cluster_summary.to_csv(
        output_dir / "cluster_early_response_summary.csv", index=False
    )

    endpoint_clusters = pd.read_csv(
        root
        / "financial_cluster_analysis"
        / "cluster_liquidity_effect_summary.csv"
    )
    endpoint_clusters.to_csv(
        output_dir / "cluster_1800s_effect_summary.csv", index=False
    )

    print(f"Wrote derived tables to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
