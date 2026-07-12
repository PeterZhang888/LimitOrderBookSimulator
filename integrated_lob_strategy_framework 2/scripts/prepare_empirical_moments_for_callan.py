#!/usr/bin/env python3
"""
Prepare the empirical target file consumed by the C++ MPI simulator.

Accepted input formats:
  1) moment,value
  2) moment_name,target
  3) moment_name,target,weight,scale_floor

Output format:
  moment_name,target,weight,scale_floor

This script is useful when the empirical ITCH analysis was done locally in a
Jupyter notebook and must be converted into the canonical calibration target
file before uploading to Callan.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ALIASES = {
    "mean_return": "return_mean",
    "var_return": "return_variance",
    "variance_return": "return_variance",
    "skew_return": "return_skewness",
    "skewness_return": "return_skewness",
    "kurt_return": "return_kurtosis",
    "kurtosis_return": "return_kurtosis",
    "acf_return_lag1": "return_acf_lag1",
    "acf_abs_return_lag1": "abs_return_acf_lag1",
    "acf_abs_return_lag2": "abs_return_acf_lag2",
    "acf_abs_return_lag5": "abs_return_acf_lag5",
    "acf_abs_return_lag10": "abs_return_acf_lag10",
    "acf_abs_return_lag25": "abs_return_acf_lag25",
    "acf_abs_return_lag50": "abs_return_acf_lag50",
    "acf_abs_return_lag100": "abs_return_acf_lag100",
    "diffusion_10s": "price_diffusion_10s",
    "p90_spread": "spread_p90",
    "p95_spread": "spread_p95",
}

DEFAULTS = {
    "return_mean": (0.5, 1e-6),
    "return_variance": (3.0, 1e-8),
    "return_skewness": (0.5, 0.05),
    "return_kurtosis": (1.5, 1.0),
    "return_acf_lag1": (2.0, 0.01),
    "abs_return_acf_lag1": (1.5, 0.02),
    "abs_return_acf_lag2": (1.0, 0.02),
    "abs_return_acf_lag5": (1.0, 0.02),
    "abs_return_acf_lag10": (1.0, 0.02),
    "abs_return_acf_lag25": (0.75, 0.02),
    "abs_return_acf_lag50": (0.75, 0.02),
    "abs_return_acf_lag100": (0.5, 0.02),
    "mean_spread": (2.5, 100.0),
    "spread_p90": (1.5, 100.0),
    "spread_p95": (1.0, 100.0),
    "volume_volatility_corr": (1.0, 0.05),
    "price_diffusion_10s": (3.0, 1000.0),
    "mean_best_bid_depth": (2.0, 100.0),
    "mean_best_ask_depth": (2.0, 100.0),
    "mid_move_rate": (3.0, 0.02),
    "zero_mid_change_ratio": (2.0, 0.02),
    "market_order_best_removal_rate": (2.0, 0.002),
    "cancel_best_removal_rate": (2.0, 0.002),
    "limit_order_rate": (1.0, 1.0),
    "limit_buy_rate": (0.5, 0.5),
    "limit_sell_rate": (0.5, 0.5),
    "market_order_rate": (1.0, 0.5),
    "market_buy_rate": (0.5, 0.25),
    "market_sell_rate": (0.5, 0.25),
    "cancel_rate": (1.0, 1.0),
    "cancel_bid_rate": (0.5, 0.5),
    "cancel_ask_rate": (0.5, 0.5),
}

ORDER = list(DEFAULTS.keys())


def canonical(name: str) -> str:
    key = str(name).strip().lower()
    return ALIASES.get(key, key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Local empirical moments CSV from Jupyter")
    parser.add_argument("--output", default="empirical_moments.csv", help="Canonical file to upload to Callan")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    lower = {c.lower(): c for c in df.columns}

    name_col = lower.get("moment_name") or lower.get("moment") or lower.get("name") or lower.get("metric")
    target_col = lower.get("target") or lower.get("value") or lower.get("empirical") or lower.get("empirical_value")
    if name_col is None or target_col is None:
        raise ValueError("Input must have moment/value or moment_name/target columns")

    weight_col = lower.get("weight")
    floor_col = lower.get("scale_floor") or lower.get("floor")

    rows_by_name = {}
    for _, row in df.iterrows():
        name = canonical(row[name_col])
        target = float(row[target_col])
        weight, floor = DEFAULTS.get(name, (1.0, 1e-6))
        if weight_col is not None and pd.notna(row[weight_col]):
            weight = float(row[weight_col])
        if floor_col is not None and pd.notna(row[floor_col]):
            floor = float(row[floor_col])
        rows_by_name[name] = {
            "moment_name": name,
            "target": target,
            "weight": weight,
            "scale_floor": floor,
        }

    ordered_rows = []
    for name in ORDER:
        if name in rows_by_name:
            ordered_rows.append(rows_by_name[name])

    # Preserve any additional moments after the standard list.
    for name, row in rows_by_name.items():
        if name not in ORDER:
            ordered_rows.append(row)

    if not ordered_rows:
        raise ValueError("No valid empirical moment rows found")

    out = pd.DataFrame(ordered_rows)
    out.to_csv(args.output, index=False)
    print(f"Wrote canonical empirical target file: {args.output}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
