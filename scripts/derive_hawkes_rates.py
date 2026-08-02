#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Derive Hawkes immigration rates from an ITCH extraction manifest.

The extractor reports the observed regular-session rate of each simulator event
bucket.  For the fixed exponential Hawkes kernel used by the C++ model,

    lambda_bar = activity_scale * mu + (alpha / beta) * lambda_bar.

This script solves that equation for ``mu`` so the configured process has the
observed rates as its stationary mean.  It is a deterministic first-stage
calibration, not a replacement for fitting the full excitation matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Sequence


EVENT_NAMES = (
    "limit_buy",
    "limit_sell",
    "market_buy",
    "market_sell",
    "cancel_bid",
    "cancel_ask",
)

SELF_EXCITATION_AMPLITUDE = 0.20
EXCITATION_STRUCTURE = "diagonal_self_excitation_only"


def excitation_settings() -> dict[str, float | str]:
    """Return the fixed excitation structure consumed by Python and C++."""
    return {
        "excitation_structure": EXCITATION_STRUCTURE,
        "self_excitation_amplitude": SELF_EXCITATION_AMPLITUDE,
        "cross_excitation_amplitude": 0.0,
    }


def clock_seconds(value: str) -> int:
    hour, minute, second = (int(piece) for piece in value.split(":"))
    return (hour * 60 + minute) * 60 + second


def default_alpha() -> list[list[float]]:
    alpha = [[0.0 for _ in EVENT_NAMES] for _ in EVENT_NAMES]
    for index in range(len(EVENT_NAMES)):
        alpha[index][index] = SELF_EXCITATION_AMPLITUDE
    return alpha


def derive(observed_rates: Sequence[float],
           activity_scale: float,
           beta: float,
           alpha: Sequence[Sequence[float]]) -> list[float]:
    if len(observed_rates) != len(EVENT_NAMES):
        raise ValueError("six observed event rates are required")
    if activity_scale <= 0.0 or beta <= 0.0:
        raise ValueError("activity scale and beta must be positive")
    result: list[float] = []
    for row in range(len(EVENT_NAMES)):
        endogenous = sum(
            alpha[row][column] * observed_rates[column] / beta
            for column in range(len(EVENT_NAMES))
        )
        baseline = (observed_rates[row] - endogenous) / activity_scale
        # A nonnegative linear Hawkes process cannot reproduce a target whose
        # endogenous contribution already exceeds that target.  Clipping a
        # negative baseline to zero silently changes the stationary mean and
        # caused the ACNB schema-3 pooling failure.  Permit only negligible
        # floating-point undershoot; otherwise fail at the derivation site.
        tolerance = 1.0e-15 * max(
            1.0, abs(observed_rates[row]), abs(endogenous)
        )
        if baseline < -tolerance:
            raise ValueError(
                "stationary target is infeasible with a nonnegative Hawkes "
                f"baseline for {EVENT_NAMES[row]}: target="
                f"{observed_rates[row]:.17g}, endogenous="
                f"{endogenous:.17g}, implied_mu={baseline:.17g}"
            )
        result.append(max(0.0, baseline))
    return result


def weighted_distribution(path: pathlib.Path, value_column: str) -> tuple[float, float]:
    """Return weighted mean and mass at zero from an extractor distribution."""
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    total = sum(float(row["count"]) for row in rows)
    if total <= 0.0:
        raise ValueError(f"empty distribution: {path}")
    weighted_sum = sum(
        float(row[value_column]) * float(row["count"]) for row in rows
    )
    zero = sum(
        float(row["count"]) for row in rows if float(row[value_column]) == 0.0
    )
    return weighted_sum / total, zero / total


def balance_directional_volume(data_dir: pathlib.Path,
                               rates: Sequence[float]) -> list[float]:
    """Remove drift created by sampling buy/sell marginals independently.

    ITCH side flows are correlated in the real market.  Once the simulator
    samples the six marginals independently, even a small difference in mean
    mark size can create an implausible one-way price process.  For each
    buy/sell pair this constraint preserves the pair's total event rate while
    making expected submitted quantity per second equal on the two sides.
    """
    adjusted = list(rates)
    quantity_files = (
        "limit_buy_quantity_distribution.txt",
        "limit_sell_quantity_distribution.txt",
        "market_buy_quantity_distribution.txt",
        "market_sell_quantity_distribution.txt",
        "cancel_bid_quantity_distribution.txt",
        "cancel_ask_quantity_distribution.txt",
    )
    means = [
        weighted_distribution(data_dir / filename, "quantity")[0]
        for filename in quantity_files
    ]
    for left, right in ((0, 1), (2, 3), (4, 5)):
        total_rate = rates[left] + rates[right]
        total_mean = means[left] + means[right]
        if total_rate <= 0.0 or total_mean <= 0.0:
            adjusted[left] = 0.0
            adjusted[right] = 0.0
            continue
        adjusted[left] = total_rate * means[right] / total_mean
        adjusted[right] = total_rate * means[left] / total_mean
    return adjusted


def balance_best_depth(data_dir: pathlib.Path,
                       rates: Sequence[float]) -> list[float]:
    """Match expected additions/removals at the best bid and ask.

    Independent cancellation marks do not retain ITCH order references.  This
    moment correction raises/lower cancellation intensity so expected quantity
    added at distance zero equals cancellations plus aggressive executions.
    """
    adjusted = list(rates)
    limit_buy_mean, _ = weighted_distribution(
        data_dir / "limit_buy_quantity_distribution.txt", "quantity")
    limit_sell_mean, _ = weighted_distribution(
        data_dir / "limit_sell_quantity_distribution.txt", "quantity")
    market_buy_mean, _ = weighted_distribution(
        data_dir / "market_buy_quantity_distribution.txt", "quantity")
    market_sell_mean, _ = weighted_distribution(
        data_dir / "market_sell_quantity_distribution.txt", "quantity")
    cancel_bid_mean, _ = weighted_distribution(
        data_dir / "cancel_bid_quantity_distribution.txt", "quantity")
    cancel_ask_mean, _ = weighted_distribution(
        data_dir / "cancel_ask_quantity_distribution.txt", "quantity")
    _, limit_buy_zero = weighted_distribution(
        data_dir / "limit_buy_distance_distribution.txt", "distance_ticks")
    _, limit_sell_zero = weighted_distribution(
        data_dir / "limit_sell_distance_distribution.txt", "distance_ticks")
    _, cancel_bid_zero = weighted_distribution(
        data_dir / "cancel_bid_distance_distribution.txt", "distance_ticks")
    _, cancel_ask_zero = weighted_distribution(
        data_dir / "cancel_ask_distance_distribution.txt", "distance_ticks")

    bid_addition = rates[0] * limit_buy_zero * limit_buy_mean
    bid_market_removal = rates[3] * market_sell_mean
    ask_addition = rates[1] * limit_sell_zero * limit_sell_mean
    ask_market_removal = rates[2] * market_buy_mean
    adjusted[4] = max(
        0.0,
        (bid_addition - bid_market_removal) / (cancel_bid_zero * cancel_bid_mean),
    )
    adjusted[5] = max(
        0.0,
        (ask_addition - ask_market_removal) / (cancel_ask_zero * cancel_ask_mean),
    )
    return adjusted


def run(args: argparse.Namespace) -> list[dict[str, float | str]]:
    manifest_path = pathlib.Path(args.manifest).resolve()
    with manifest_path.open() as source:
        manifest = json.load(source)

    # A normal extractor manifest represents exactly one intraday interval and
    # therefore derives its duration from HH:MM:SS bounds.  The multi-day
    # pooling utility keeps those ordinary session bounds for readability but
    # records the *sum* of its source-session durations explicitly.  Preferring
    # that audited field prevents a five-day pooled event count from being
    # mistaken for one day of intensity.
    if "aggregation_duration_seconds" in manifest:
        try:
            duration_seconds = float(manifest["aggregation_duration_seconds"])
        except (TypeError, ValueError) as error:
            raise ValueError("manifest aggregation_duration_seconds is invalid") from error
        if not duration_seconds.is_integer():
            raise ValueError("manifest aggregation_duration_seconds must be integral")
        duration_seconds = int(duration_seconds)
    else:
        duration_seconds = clock_seconds(manifest["session_end"]) - clock_seconds(
            manifest["session_start"]
        )
    if duration_seconds <= 0:
        raise ValueError("manifest contains an invalid session interval")
    counts = manifest["distribution_observation_counts"]
    observed = [float(counts[name]) / duration_seconds for name in EVENT_NAMES]
    if not 0.0 <= args.balance_strength <= 5.0:
        raise ValueError("--balance-strength must be between 0 and 5")
    base_target = balance_directional_volume(
        manifest_path.parent, observed
    ) if args.balance_directional_volume else list(observed)
    if args.balance_best_depth:
        fully_balanced = balance_best_depth(manifest_path.parent, base_target)
        stationary_target = [
            original + args.balance_strength * (balanced - original)
            for original, balanced in zip(base_target, fully_balanced)
        ]
    else:
        stationary_target = base_target
    alpha = default_alpha()
    configured_mu = derive(stationary_target, args.activity_scale, args.beta, alpha)

    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(EVENT_NAMES):
        endogenous = sum(
            alpha[index][column] * stationary_target[column] / args.beta
            for column in range(len(EVENT_NAMES))
        )
        rows.append({
            "event_type": name,
            "observed_count": int(counts[name]),
            "observed_rate_per_second": observed[index],
            "stationary_target_rate": stationary_target[index],
            "configured_mu": configured_mu[index],
            "stationary_reconstructed_rate": (
                args.activity_scale * configured_mu[index] + endogenous
            ),
        })

    output_path = pathlib.Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="ITCH extraction manifest JSON")
    parser.add_argument("--output", required=True, help="output Hawkes-rate CSV")
    parser.add_argument("--activity-scale", type=float, default=0.30)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument(
        "--balance-directional-volume",
        action="store_true",
        help="equalise expected buy/sell mark volume while preserving pair event rates",
    )
    parser.add_argument(
        "--balance-best-depth",
        action="store_true",
        help="moment-correct cancel rates for stable expected best-quote depth",
    )
    parser.add_argument(
        "--balance-strength",
        type=float,
        default=1.0,
        help="cancel-rate correction: observed=0, moment-balanced=1, stronger up to 5",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = run(args)
    print(json.dumps({
        "output": str(pathlib.Path(args.output).resolve()),
        "event_rates": {
            str(row["event_type"]): row["observed_rate_per_second"] for row in rows
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
