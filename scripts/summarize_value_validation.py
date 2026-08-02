#!/usr/bin/env python3
"""Create thesis-ready tables from chronological weighted-moment validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics


SYMBOLS = ("QQQ", "AAPL", "MSFT", "AMZN")
METRICS = (
    "mean_spread_ticks",
    "mean_bid_depth",
    "mean_ask_depth",
    "mid_move_rate",
    "return_variance",
    "return_kurtosis",
    "absolute_return_acf1",
)


def compact(date: str) -> str:
    return date.replace("-", "")


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def target_table(root: pathlib.Path, date: str,
                 symbol: str) -> dict[str, tuple[float, float, float]]:
    token = compact(date)
    path = root / f"itch_{token}_{symbol.lower()}" / (
        f"market_targets_{symbol.lower()}_{token}.csv"
    )
    return {
        row["name"]: (
            float(row["target"]), float(row["scale"]),
            float(row.get("weight", "1.0")),
        )
        for row in read_csv(path)
    }


def finite_mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("validation summaries contain a non-finite metric")
    return statistics.fmean(values)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def format_optional(value: float | None) -> str:
    return f"{value:.6g}" if value is not None and math.isfinite(value) else "n/a"


def summary_paths_for_split(
    report: dict[str, object], result_root: pathlib.Path, split: str,
    seeds: list[int],
) -> list[pathlib.Path]:
    """Read modern report paths, with a compatibility fallback for old runs."""
    key = "selected_training_evaluation" if split == "training" else "heldout_evaluation"
    evaluation = report.get(key)
    if isinstance(evaluation, dict):
        paths = evaluation.get("summary_paths")
        if isinstance(paths, list) and paths:
            return [pathlib.Path(str(path)).resolve() for path in paths]
    directory = "coupled_training" if split == "training" else "heldout_validation"
    return [
        result_root / directory / f"seed_{seed}" / "sequential_multi_asset_summary.csv"
        for seed in seeds
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()

    report_path = pathlib.Path(args.report).resolve()
    with report_path.open() as source:
        report = json.load(source)
    protocol = report["protocol"]
    seeds = [int(value) for value in protocol["seeds"]]
    target_root = pathlib.Path(args.target_root).resolve()
    result_root = pathlib.Path(args.result_root).resolve()

    split_specs = (
        ("training", protocol["training_date"]),
        ("heldout", protocol["heldout_date"]),
    )
    detail_rows: list[dict[str, object]] = []
    symbol_wsmrmse: dict[tuple[str, str], float] = {}
    split_wsmrmse: dict[str, float] = {}
    split_combined_wsmrmse: dict[str, float | None] = {}
    structural_counts: dict[str, tuple[int, int]] = {}
    split_wall_times: dict[str, list[float]] = {}
    for split_name, date in split_specs:
        summaries: list[dict[str, dict[str, str]]] = []
        structural = 0
        total = 0
        summary_paths = summary_paths_for_split(
            report, result_root, split_name, seeds,
        )
        for summary_path in summary_paths:
            by_symbol = {row["symbol"]: row for row in read_csv(summary_path)}
            if set(by_symbol) != set(SYMBOLS):
                raise ValueError(f"unexpected symbols in {summary_path}")
            summaries.append(by_symbol)
            for symbol in SYMBOLS:
                total += 1
                structural += by_symbol[symbol]["structurally_valid"] == "1"
                row = by_symbol[symbol]
                if ("sample_count" in row and "expected_sample_count" in row
                        and row["sample_count"] != row["expected_sample_count"]):
                    raise ValueError(
                        f"incomplete fixed-clock samples for {symbol} in {summary_path}"
                    )
        structural_counts[split_name] = (structural, total)
        if structural != total:
            raise ValueError(f"structurally invalid book in {split_name} validation")

        weighted_squared = 0.0
        combined_weighted_squared = 0.0
        total_weight = 0.0
        has_mc_se = len(summaries) >= 2
        wall_times: list[float] = []
        for by_symbol in summaries:
            for symbol in SYMBOLS:
                value = by_symbol[symbol].get("wall_seconds")
                if value is not None:
                    parsed = float(value)
                    if math.isfinite(parsed):
                        wall_times.append(parsed)
        split_wall_times[split_name] = wall_times

        for symbol in SYMBOLS:
            targets = target_table(target_root, date, symbol)
            symbol_weighted_squared = 0.0
            symbol_weight = 0.0
            for metric in METRICS:
                target, empirical_scale, weight = targets[metric]
                if (not math.isfinite(empirical_scale) or empirical_scale <= 0.0
                        or not math.isfinite(weight) or weight <= 0.0):
                    raise ValueError(f"invalid target scale for {date} {symbol} {metric}")
                observed = [float(summary[symbol][metric]) for summary in summaries]
                mean_observed = finite_mean(observed)
                simulated_sample_sd = sample_sd(observed)
                simulated_mc_se = (
                    simulated_sample_sd / math.sqrt(len(observed))
                    if has_mc_se else None
                )
                combined_scale = (
                    math.hypot(empirical_scale, simulated_mc_se)
                    if simulated_mc_se is not None else None
                )
                empirical_z = (mean_observed - target) / empirical_scale
                combined_z = (
                    (mean_observed - target) / combined_scale
                    if combined_scale is not None else None
                )
                contribution = weight * empirical_z * empirical_z
                combined_contribution = (
                    weight * combined_z * combined_z
                    if combined_z is not None else 0.0
                )
                weighted_squared += contribution
                combined_weighted_squared += combined_contribution
                total_weight += weight
                symbol_weighted_squared += contribution
                symbol_weight += weight
                detail_rows.append({
                    "split": split_name,
                    "date": date,
                    "symbol": symbol,
                    "metric": metric,
                    "empirical_target": target,
                    "empirical_scale": empirical_scale,
                    "importance_weight": weight,
                    "simulated_mean": mean_observed,
                    "simulation_sample_sd": simulated_sample_sd,
                    "simulation_mc_se": simulated_mc_se,
                    "combined_scale": combined_scale,
                    "empirical_standardized_error": empirical_z,
                    "combined_standardized_error": combined_z,
                    "weighted_squared_error": contribution,
                    "seed_count": len(summaries),
                })
            symbol_wsmrmse[(split_name, symbol)] = math.sqrt(
                symbol_weighted_squared / symbol_weight
            )
        split_wsmrmse[split_name] = math.sqrt(weighted_squared / total_weight)
        split_combined_wsmrmse[split_name] = (
            math.sqrt(combined_weighted_squared / total_weight)
            if has_mc_se else None
        )

    output_csv = pathlib.Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    parameters = report["selected_parameters"]
    coupling_mode = protocol.get("coupling_mode", "legacy_unspecified")
    training_time = (
        finite_mean(split_wall_times["training"])
        if split_wall_times["training"] else math.nan
    )
    heldout_time = (
        finite_mean(split_wall_times["heldout"])
        if split_wall_times["heldout"] else math.nan
    )
    lines = [
        "# Chronological weighted-moment validation",
        "",
        f"Training session: {protocol['training_date']}; held-out session: "
        f"{protocol['heldout_date']}. Selection used held-out targets: "
        f"{str(protocol['selection_uses_heldout_targets']).lower()}.",
        "",
        f"Coupling mode used consistently for selection and validation: "
        f"`{coupling_mode}`.",
        "",
        "Selected parameters: "
        f"threshold {parameters['threshold_bps']} bps, response step "
        f"{parameters['response_step_bps']} bps, base quantity "
        f"{parameters['base_order_quantity']}, fundamental volatility "
        f"{parameters['volatility_bps_sqrt_second']} bps/√s.",
        "",
        "| Split | Fit WSMRMSE | Combined-uncertainty WSMRMSE | Structurally valid books | "
        "Mean external wall time per full-day seed (s) |",
        "|---|---:|---:|---:|---:|",
        (f"| Training | {split_wsmrmse['training']:.6g} | "
         f"{format_optional(split_combined_wsmrmse['training'])} | "
         f"{structural_counts['training'][0]}/{structural_counts['training'][1]} | "
         f"{training_time:.6g} |"),
        (f"| Held out | {split_wsmrmse['heldout']:.6g} | "
         f"{format_optional(split_combined_wsmrmse['heldout'])} | "
         f"{structural_counts['heldout'][0]}/{structural_counts['heldout'][1]} | "
         f"{heldout_time:.6g} |"),
        "",
        "| Symbol | Training fit WSMRMSE | Held-out fit WSMRMSE |",
        "|---|---:|---:|",
    ]
    for symbol in SYMBOLS:
        lines.append(
            f"| {symbol} | {symbol_wsmrmse[('training', symbol)]:.6g} | "
            f"{symbol_wsmrmse[('heldout', symbol)]:.6g} |"
        )
    lines.extend([
        "",
        "Fit WSMRMSE is the predeclared diagonal weighted standardized "
        "simulated-moment loss, using empirical delete-block-jackknife scales. "
        "The combined-uncertainty score adds the simulated Monte-Carlo standard "
        "error only as a reporting diagnostic; it is not used to select a noisier "
        "candidate. No residual cap or post-hoc acceptance threshold is used. "
        "Structural validity is a correctness gate, not evidence of empirical fit.",
        "",
        f"Detailed metric table: `{output_csv.name}`.",
        "",
    ])
    output_markdown = pathlib.Path(args.output_markdown).resolve()
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines))
    print(output_markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
