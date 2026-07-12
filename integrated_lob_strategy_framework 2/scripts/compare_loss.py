#!/usr/bin/env python3
"""Compute the same weighted relative loss in Python for a simulated moment CSV."""

from __future__ import annotations

import argparse
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", required=True, help="CSV with columns moment_name,value OR a calibration_results.csv row")
    parser.add_argument("--targets", default="empirical_moments.csv")
    parser.add_argument("--row", type=int, default=0, help="Row index if --sim is calibration_results.csv")
    parser.add_argument("--output", default="python_loss_detail.csv")
    args = parser.parse_args()

    targets = pd.read_csv(args.targets)
    sim_df = pd.read_csv(args.sim)

    if {"moment_name", "value"}.issubset(sim_df.columns):
        sim_map = dict(zip(sim_df["moment_name"], sim_df["value"]))
    else:
        row = sim_df.iloc[args.row]
        sim_map = {col: row[col] for col in sim_df.columns}

    rows = []
    weighted = 0.0
    total_weight = 0.0
    for _, t in targets.iterrows():
        name = str(t["moment_name"])
        target = float(t["target"])
        weight = float(t.get("weight", 1.0))
        floor = float(t.get("scale_floor", 1e-6))
        sim = float(sim_map.get(name, 0.0))
        denom = max(abs(target), floor, 1e-12)
        rel = (sim - target) / denom
        term = weight * rel * rel
        weighted += term
        total_weight += weight
        rows.append({
            "moment_name": name,
            "simulated": sim,
            "target": target,
            "weight": weight,
            "scale_floor": floor,
            "relative_error": rel,
            "weighted_loss_term": term,
        })
    total_loss = weighted / total_weight if total_weight > 0 else weighted
    out = pd.DataFrame(rows)
    out.loc[len(out)] = ["TOTAL_LOSS", total_loss, None, None, None, None, None]
    out.to_csv(args.output, index=False)
    print(f"total_loss={total_loss:.12g}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
