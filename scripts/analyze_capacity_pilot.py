#!/usr/bin/env python3
"""Classify candidate global capacities from paired pilot paths."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def f(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--shock-time-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = read_csv(args.raw)
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in raw:
        if row.get("shared_mm_mode") != "global":
            raise ValueError("capacity pilot must contain global mode only")
        key = (row["risk_limit_per_asset"], row["shock_mode"])
        if key in indexed:
            raise ValueError(f"duplicate pilot case {key}")
        indexed[key] = row

    output: list[dict[str, object]] = []
    for risk in sorted({key[0] for key in indexed}, key=float):
        pairs: dict[str, list[dict[str, str]]] = {}
        for mode in ("off", "on"):
            row = indexed.get((risk, mode))
            if row is None:
                raise ValueError(f"missing {mode} path for capacity {risk}")
            metrics = read_csv(Path(row["metrics_csv"]))
            pairs[mode] = metrics
        shock_metrics = pairs["on"]
        control_metrics = pairs["off"]
        pre = [r for r in shock_metrics if f(r, "time_seconds") < args.shock_time_seconds]
        post = [r for r in shock_metrics if f(r, "time_seconds") >= args.shock_time_seconds]
        control_phi = [f(r, "shared_quote_scale") for r in control_metrics]
        pre_phi = [f(r, "shared_quote_scale") for r in pre]
        post_phi = [f(r, "shared_quote_scale") for r in post]
        if not pre_phi or not post_phi:
            raise ValueError(f"pilot capacity {risk} lacks pre/post observations")
        pre_binding = min(pre_phi) < 1.0 - 1.0e-12
        post_min = min(post_phi)
        control_binding = min(control_phi) < 1.0 - 1.0e-12
        if not pre_binding and not control_binding and post_min >= 1.0 - 1.0e-12:
            regime = "non_binding"
        elif not pre_binding and not control_binding and post_min < 0.5:
            regime = "tight_shock_activated"
        elif not pre_binding and not control_binding and post_min < 1.0:
            regime = "moderate_shock_activated"
        else:
            regime = "already_binding_before_shock"
        requested = f(indexed[(risk, "on")], "shock_requested_quantity")
        executed = f(indexed[(risk, "on")], "shock_executed_quantity")
        absorbed = f(indexed[(risk, "on")], "shock_shared_mm_quantity")
        output.append({
            "global_risk_limit_per_asset": risk,
            "regime": regime,
            "pre_shock_mean_phi": sum(pre_phi) / len(pre_phi),
            "pre_shock_min_phi": min(pre_phi),
            "post_shock_min_phi": post_min,
            "control_min_phi": min(control_phi),
            "post_windows_phi_below_1": sum(value < 1.0 - 1.0e-12 for value in post_phi),
            "post_windows_phi_below_0_5": sum(value < 0.5 for value in post_phi),
            "shock_requested_quantity": requested,
            "shock_executed_quantity": executed,
            "shared_mm_absorption_fraction": absorbed / executed if executed else 0.0,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"capacity_diagnostics={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
