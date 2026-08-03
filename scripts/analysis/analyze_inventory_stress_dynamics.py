#!/usr/bin/env python3
"""Compute time-resolved financial and cluster inventory-stress diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SHOCK_TIME = 11700
POST_HORIZON = 1800
T95 = 2.093
WINDOWS = (
    (2, 5, "02--05 s"),
    (6, 10, "06--10 s"),
    (11, 20, "11--20 s"),
    (21, 30, "21--30 s"),
    (31, 60, "31--60 s"),
    (61, 120, "61--120 s"),
    (121, 300, "121--300 s"),
    (301, 600, "301--600 s"),
    (601, 1800, "601--1,800 s"),
)


def summarize_values(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n = len(clean)
    mean = float(clean.mean())
    if n < 2:
        return {"n": n, "mean": mean, "lower": math.nan, "upper": math.nan}
    half = T95 * float(clean.std(ddof=1)) / math.sqrt(n)
    return {"n": n, "mean": mean, "lower": mean - half, "upper": mean + half}


def summarize_frame(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group, subframe in frame.groupby(group_columns, sort=True):
        key = group if isinstance(group, tuple) else (group,)
        row: dict[str, float | int | str] = dict(zip(group_columns, key))
        for value_column in value_columns:
            stats = summarize_values(subframe[value_column])
            for statistic, value in stats.items():
                row[f"{value_column}_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def load_path_map(root: Path) -> dict[tuple[int, int, str], tuple[Path, Path]]:
    raw = pd.read_csv(root / "financial_global_raw.csv")
    paths: dict[tuple[int, int, str], tuple[Path, Path]] = {}
    for row in raw.itertuples(index=False):
        key = (int(row.risk_limit_per_asset), int(row.seed), str(row.shock_mode))
        metrics = root / "financial_metrics" / "global" / Path(row.metrics_csv).name
        clusters = (
            root
            / "financial_cluster_metrics"
            / "global"
            / Path(row.cluster_metrics_csv).name
        )
        paths[key] = (metrics, clusters)
    if len(paths) != 80:
        raise RuntimeError(f"expected 80 global paths, observed {len(paths)}")
    return paths


def keep_post_shock(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[
        (frame["time_seconds"] >= SHOCK_TIME + 1)
        & (frame["time_seconds"] <= SHOCK_TIME + POST_HORIZON)
    ].copy()
    result["post_shock_seconds"] = result["time_seconds"] - SHOCK_TIME
    return result


def first_sustained_nonsignificant(
    summary: pd.DataFrame,
    onset: int,
    direction: str,
    consecutive: int = 5,
) -> int | None:
    ordered = summary.sort_values("post_shock_seconds").reset_index(drop=True)
    if direction == "positive":
        significant = ordered["lower"] > 0.0
    elif direction == "negative":
        significant = ordered["upper"] < 0.0
    else:
        raise ValueError(direction)
    times = ordered["post_shock_seconds"].astype(int).to_numpy()
    significant_values = significant.to_numpy()
    for index, time_value in enumerate(times):
        if time_value <= onset or index + consecutive > len(times):
            continue
        expected = np.arange(time_value, time_value + consecutive)
        if np.array_equal(times[index : index + consecutive], expected) and not np.any(
            significant_values[index : index + consecutive]
        ):
            return int(time_value)
    return None


def recovery_record(
    summary: pd.DataFrame,
    direction: str,
    onset_limit: int = 30,
) -> dict[str, float | int | None]:
    ordered = summary.sort_values("post_shock_seconds").copy()
    if direction == "positive":
        significant = ordered["lower"] > 0.0
    else:
        significant = ordered["upper"] < 0.0
    initial = ordered[(ordered["post_shock_seconds"] <= onset_limit) & significant]
    if initial.empty:
        return {
            "onset_seconds": None,
            "initial_episode_last_significant_seconds": None,
            "recovery_seconds": None,
            "significant_seconds_1_30": 0,
        }
    onset = int(initial["post_shock_seconds"].iloc[0])
    recovery = first_sustained_nonsignificant(ordered, onset, direction)
    before_recovery = ordered[significant]
    if recovery is not None:
        before_recovery = before_recovery[
            before_recovery["post_shock_seconds"] < recovery
        ]
    last_significant = int(before_recovery["post_shock_seconds"].max())
    return {
        "onset_seconds": onset,
        "initial_episode_last_significant_seconds": last_significant,
        "recovery_seconds": recovery,
        "significant_seconds_1_30": int(
            significant[ordered["post_shock_seconds"] <= 30].sum()
        ),
    }


def first_fractional_decay(
    summary: pd.DataFrame,
    value_column: str,
    initial_second: int,
    fraction: float,
    consecutive: int = 3,
) -> int | None:
    ordered = summary.sort_values("post_shock_seconds").reset_index(drop=True)
    initial = float(
        ordered.loc[
            ordered["post_shock_seconds"] == initial_second, value_column
        ].iloc[0]
    )
    threshold = abs(initial) * fraction
    times = ordered["post_shock_seconds"].astype(int).to_numpy()
    below = np.abs(ordered[value_column].to_numpy()) <= threshold
    for index, time_value in enumerate(times):
        if time_value <= initial_second or index + consecutive > len(times):
            continue
        if not np.array_equal(
            times[index : index + consecutive],
            np.arange(time_value, time_value + consecutive),
        ):
            continue
        if np.all(below[index : index + consecutive]):
            return int(time_value)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = args.result_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = load_path_map(root)
    seeds = sorted({key[1] for key in paths})

    market_rows: list[pd.DataFrame] = []
    cluster_rows: list[pd.DataFrame] = []
    for risk in (800, 1600):
        for seed in seeds:
            metrics_by_mode: dict[str, pd.DataFrame] = {}
            clusters_by_mode: dict[str, pd.DataFrame] = {}
            for mode in ("off", "on"):
                metric_path, cluster_path = paths[(risk, seed, mode)]
                metrics_by_mode[mode] = keep_post_shock(
                    pd.read_csv(
                        metric_path,
                        usecols=[
                            "time_seconds",
                            "unshocked_mean_top_depth",
                            "unshocked_mean_spread_bps",
                            "shared_gross_exposure",
                            "shared_utilization",
                            "shared_quote_scale",
                            "unshocked_shared_requested_quote_depth",
                            "unshocked_shared_resting_quote_depth",
                        ],
                    )
                )
                clusters_by_mode[mode] = keep_post_shock(
                    pd.read_csv(
                        cluster_path,
                        usecols=[
                            "time_seconds",
                            "cluster_id",
                            "non_target_asset_count",
                            "mean_top_depth",
                            "mean_spread_bps",
                            "affected_asset_fraction",
                            "shared_requested_quote_depth",
                            "shared_resting_quote_depth",
                        ],
                    )
                )

            market = metrics_by_mode["off"].merge(
                metrics_by_mode["on"],
                on=["time_seconds", "post_shock_seconds"],
                suffixes=("_control", "_shock"),
                validate="one_to_one",
            )
            market["risk_limit_per_asset"] = risk
            market["seed"] = seed
            market["relative_depth_deterioration_percent"] = 100.0 * (
                market["unshocked_mean_top_depth_control"]
                - market["unshocked_mean_top_depth_shock"]
            ) / market["unshocked_mean_top_depth_control"]
            market["spread_change_bps"] = (
                market["unshocked_mean_spread_bps_shock"]
                - market["unshocked_mean_spread_bps_control"]
            )
            market["exposure_change"] = (
                market["shared_gross_exposure_shock"]
                - market["shared_gross_exposure_control"]
            )
            market["utilization_change"] = (
                market["shared_utilization_shock"]
                - market["shared_utilization_control"]
            )
            market["quote_scale_change"] = (
                market["shared_quote_scale_shock"]
                - market["shared_quote_scale_control"]
            )
            market["requested_depth_reduction"] = (
                market["unshocked_shared_requested_quote_depth_control"]
                - market["unshocked_shared_requested_quote_depth_shock"]
            )
            market["resting_depth_reduction"] = (
                market["unshocked_shared_resting_quote_depth_control"]
                - market["unshocked_shared_resting_quote_depth_shock"]
            )
            market_rows.append(
                market[
                    [
                        "risk_limit_per_asset",
                        "seed",
                        "post_shock_seconds",
                        "relative_depth_deterioration_percent",
                        "spread_change_bps",
                        "exposure_change",
                        "utilization_change",
                        "quote_scale_change",
                        "requested_depth_reduction",
                        "resting_depth_reduction",
                    ]
                ]
            )

            clusters = clusters_by_mode["off"].merge(
                clusters_by_mode["on"],
                on=["time_seconds", "post_shock_seconds", "cluster_id"],
                suffixes=("_control", "_shock"),
                validate="one_to_one",
            )
            if not np.array_equal(
                clusters["non_target_asset_count_control"],
                clusters["non_target_asset_count_shock"],
            ):
                raise RuntimeError("cluster non-target cohorts differ")
            clusters["risk_limit_per_asset"] = risk
            clusters["seed"] = seed
            clusters["relative_depth_deterioration_percent"] = 100.0 * (
                clusters["mean_top_depth_control"]
                - clusters["mean_top_depth_shock"]
            ) / clusters["mean_top_depth_control"]
            clusters["depth_deterioration_units"] = (
                clusters["mean_top_depth_control"]
                - clusters["mean_top_depth_shock"]
            )
            clusters["spread_change_bps"] = (
                clusters["mean_spread_bps_shock"]
                - clusters["mean_spread_bps_control"]
            )
            clusters["affected_fraction_change"] = (
                clusters["affected_asset_fraction_shock"]
                - clusters["affected_asset_fraction_control"]
            )
            clusters["requested_depth_reduction"] = (
                clusters["shared_requested_quote_depth_control"]
                - clusters["shared_requested_quote_depth_shock"]
            )
            clusters["resting_depth_reduction"] = (
                clusters["shared_resting_quote_depth_control"]
                - clusters["shared_resting_quote_depth_shock"]
            )
            cluster_rows.append(
                clusters[
                    [
                        "risk_limit_per_asset",
                        "seed",
                        "post_shock_seconds",
                        "cluster_id",
                        "non_target_asset_count_control",
                        "relative_depth_deterioration_percent",
                        "depth_deterioration_units",
                        "spread_change_bps",
                        "affected_fraction_change",
                        "requested_depth_reduction",
                        "resting_depth_reduction",
                    ]
                ].rename(
                    columns={
                        "non_target_asset_count_control": "non_target_asset_count"
                    }
                )
            )

    market_by_seed = pd.concat(market_rows, ignore_index=True)
    clusters_by_seed = pd.concat(cluster_rows, ignore_index=True)
    market_by_seed.to_csv(output / "market_time_by_seed.csv", index=False)
    clusters_by_seed.to_csv(output / "cluster_time_by_seed.csv", index=False)

    market_values = [
        "relative_depth_deterioration_percent",
        "spread_change_bps",
        "exposure_change",
        "utilization_change",
        "quote_scale_change",
        "requested_depth_reduction",
        "resting_depth_reduction",
    ]
    cluster_values = [
        "relative_depth_deterioration_percent",
        "depth_deterioration_units",
        "spread_change_bps",
        "affected_fraction_change",
        "requested_depth_reduction",
        "resting_depth_reduction",
    ]
    market_time = summarize_frame(
        market_by_seed,
        ["risk_limit_per_asset", "post_shock_seconds"],
        market_values,
    )
    cluster_time = summarize_frame(
        clusters_by_seed,
        ["risk_limit_per_asset", "cluster_id", "post_shock_seconds"],
        cluster_values,
    )
    market_time.to_csv(output / "market_time_summary.csv", index=False)
    cluster_time.to_csv(output / "cluster_time_summary.csv", index=False)

    def add_window(frame: pd.DataFrame) -> pd.DataFrame:
        assigned = frame.copy()
        assigned["window"] = ""
        for start, end, label in WINDOWS:
            assigned.loc[
                assigned["post_shock_seconds"].between(start, end), "window"
            ] = label
        return assigned[assigned["window"] != ""]

    market_window_seed = (
        add_window(market_by_seed)
        .groupby(["risk_limit_per_asset", "seed", "window"], as_index=False)[
            market_values
        ]
        .mean()
    )
    cluster_window_seed = (
        add_window(clusters_by_seed)
        .groupby(
            ["risk_limit_per_asset", "seed", "cluster_id", "window"],
            as_index=False,
        )[cluster_values]
        .mean()
    )
    window_order = {label: index for index, (_, _, label) in enumerate(WINDOWS)}
    market_windows = summarize_frame(
        market_window_seed,
        ["risk_limit_per_asset", "window"],
        market_values,
    )
    cluster_windows = summarize_frame(
        cluster_window_seed,
        ["risk_limit_per_asset", "cluster_id", "window"],
        cluster_values,
    )
    market_windows["window_order"] = market_windows["window"].map(window_order)
    cluster_windows["window_order"] = cluster_windows["window"].map(window_order)
    market_windows.sort_values(
        ["risk_limit_per_asset", "window_order"]
    ).to_csv(output / "market_window_summary.csv", index=False)
    cluster_windows.sort_values(
        ["risk_limit_per_asset", "cluster_id", "window_order"]
    ).to_csv(output / "cluster_window_summary.csv", index=False)

    market_recovery: list[dict[str, float | int | str | None]] = []
    for risk in (800, 1600):
        subset = market_time[market_time["risk_limit_per_asset"] == risk]
        for metric, direction in (
            ("relative_depth_deterioration_percent", "positive"),
            ("requested_depth_reduction", "positive"),
            ("resting_depth_reduction", "positive"),
            ("exposure_change", "positive"),
            ("quote_scale_change", "negative"),
        ):
            metric_summary = subset[
                [
                    "post_shock_seconds",
                    f"{metric}_mean",
                    f"{metric}_lower",
                    f"{metric}_upper",
                ]
            ].rename(
                columns={
                    f"{metric}_mean": "mean",
                    f"{metric}_lower": "lower",
                    f"{metric}_upper": "upper",
                }
            )
            record: dict[str, float | int | str | None] = {
                "risk_limit_per_asset": risk,
                "metric": metric,
            }
            record.update(recovery_record(metric_summary, direction))
            initial_second = 1 if metric != "requested_depth_reduction" else 2
            record["mean_half_decay_seconds"] = first_fractional_decay(
                metric_summary, "mean", initial_second, 0.5
            )
            record["mean_tenth_decay_seconds"] = first_fractional_decay(
                metric_summary, "mean", initial_second, 0.1
            )
            market_recovery.append(record)
    pd.DataFrame(market_recovery).to_csv(
        output / "market_recovery_summary.csv", index=False
    )

    cluster_recovery: list[dict[str, float | int | None]] = []
    for (risk, cluster), subset in cluster_time.groupby(
        ["risk_limit_per_asset", "cluster_id"], sort=True
    ):
        metric = "relative_depth_deterioration_percent"
        metric_summary = subset[
            [
                "post_shock_seconds",
                f"{metric}_mean",
                f"{metric}_lower",
                f"{metric}_upper",
            ]
        ].rename(
            columns={
                f"{metric}_mean": "mean",
                f"{metric}_lower": "lower",
                f"{metric}_upper": "upper",
            }
        )
        record: dict[str, float | int | None] = {
            "risk_limit_per_asset": int(risk),
            "cluster_id": int(cluster),
        }
        record.update(recovery_record(metric_summary, "positive"))
        early = metric_summary[metric_summary["post_shock_seconds"].between(2, 10)]
        peak_row = early.loc[early["mean"].idxmax()]
        record["early_peak_percent"] = float(peak_row["mean"])
        record["early_peak_seconds"] = int(peak_row["post_shock_seconds"])
        cluster_recovery.append(record)
    cluster_recovery_frame = pd.DataFrame(cluster_recovery)

    baseline = pd.read_csv(
        root
        / "financial_cluster_analysis"
        / "cluster_liquidity_effect_summary.csv"
    )
    baseline = baseline[baseline["risk_limit_per_asset"] == 800][
        [
            "cluster_id",
            "non_target_symbol_count",
            "mean_baseline_uncoupled_top_depth",
            "mean_baseline_uncoupled_spread_bps",
        ]
    ].drop_duplicates("cluster_id").rename(
        columns={
            "mean_baseline_uncoupled_top_depth": "baseline_mean_top_depth",
            "mean_baseline_uncoupled_spread_bps": "baseline_mean_spread_bps",
        }
    )
    counts = (
        pd.read_csv(root / "portable_case" / "cluster_assignments.csv")
        .groupby("cluster_id")
        .size()
        .rename("symbol_count")
        .reset_index()
    )
    baseline = baseline.merge(counts, on="cluster_id", validate="one_to_one")
    cluster_recovery_frame = cluster_recovery_frame.merge(
        baseline, on="cluster_id", how="left", validate="many_to_one"
    )
    cluster_recovery_frame.to_csv(
        output / "cluster_recovery_summary.csv", index=False
    )

    audit = {
        "schema_version": 1,
        "shock_time_seconds": SHOCK_TIME,
        "post_shock_horizon_seconds": POST_HORIZON,
        "seeds": seeds,
        "paired_seed_count": len(seeds),
        "recovery_definition": (
            "first second after initial significance at which the paired 95% "
            "confidence interval includes zero for five consecutive seconds"
        ),
        "confidence_interval": (
            "mean plus/minus t_0.975,19 times the paired-seed standard error; "
            "t critical value 2.093"
        ),
        "market_rows": len(market_by_seed),
        "cluster_rows": len(clusters_by_seed),
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
