#!/usr/bin/env python3
"""Build a four-book simulator CSV without leaking held-out targets.

Opening BBO/fundamental values may come from a held-out day, while background
distributions, Hawkes rates, market-maker size, and target spreads remain frozen
from the calibration day.  This separation is the key guard against refitting
nuisance parameters on the validation data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from typing import Iterable


SYMBOLS = ("QQQ", "AAPL", "MSFT", "AMZN")
FIELDNAMES = (
    "book_id", "symbol", "data_dir", "hawkes_rates_file",
    "fundamental_price_ticks", "initial_best_bid_ticks",
    "initial_best_ask_ticks", "initial_best_bid_depth",
    "initial_best_ask_depth", "beta", "basket_weight",
    "market_maker_quote_quantity", "target_spread_ticks",
    "quote_improvement_probability",
)


def compact(date: str) -> str:
    value = date.replace("-", "")
    if len(value) != 8 or not value.isdigit():
        raise ValueError(f"invalid ISO date: {date}")
    return value


def read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def portable_path(path: pathlib.Path) -> str:
    """Prefer a project-relative path without breaking external data roots."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(pathlib.Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def weighted_median(rows: Iterable[dict[str, str]]) -> int:
    values = sorted(
        (int(row["quantity"]), int(row["count"])) for row in rows
        if int(row["quantity"]) > 0 and int(row["count"]) > 0
    )
    total = sum(count for _, count in values)
    if total <= 0:
        raise ValueError("quantity distribution is empty")
    cumulative = 0
    for value, count in values:
        cumulative += count
        if 2 * cumulative >= total:
            return value
    raise AssertionError("unreachable weighted median")


def read_weights(path: pathlib.Path, calibration_date: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for row in read_rows(path):
        symbol = row["symbol"].upper()
        if symbol in SYMBOLS[1:]:
            if symbol in weights:
                raise ValueError(f"weight file contains duplicate symbol {symbol}")
            field = ("raw_qqq_portfolio_weight"
                     if "raw_qqq_portfolio_weight" in row
                     else "raw_nasdaq100_weight")
            value = float(row[field])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"invalid positive basket weight for {symbol}")
            for date_field in ("source_as_of", "filing_date"):
                if row.get(date_field):
                    source_date = compact(row[date_field])
                    if source_date > calibration_date:
                        raise ValueError(
                            f"{date_field} for {symbol} occurs after calibration "
                            f"date: {row[date_field]}"
                        )
            weights[symbol] = value
    missing = set(SYMBOLS[1:]).difference(weights)
    if missing:
        raise ValueError("weight file is missing: " + ", ".join(sorted(missing)))
    return weights


def quote_improvement_probability(calibration_dir: pathlib.Path,
                                  symbol: str,
                                  calibration_date: str,
                                  mean_spread: float) -> float:
    """Use the identifiable aggregate distance-zero split.

    The compact extractor records one aggregate inside-spread count, while its
    buy/sell distance histograms retain separate zero counts. The runtime
    scalar is therefore ``inside / (buy_zero + sell_zero)``. The descriptive
    ``inside / eligible`` rate cannot be applied independently to both sides.
    Manifests produced before placement counts were added retain a transparent
    spread-based fallback so old non-certified fixtures remain usable.
    """
    manifest_path = calibration_dir / (
        f"itch_manifest_{symbol.lower()}_{calibration_date}.json"
    )
    if manifest_path.exists():
        with manifest_path.open() as source:
            manifest = json.load(source)
        placement = manifest.get("placement_counts", {})
        has_placement_counts = (
            "improvement_eligible_limit_orders" in placement
            and "inside_spread_limit_orders" in placement
        )
        eligible = int(placement.get("improvement_eligible_limit_orders", 0))
        inside = int(placement.get("inside_spread_limit_orders", 0))
        if has_placement_counts:
            if inside < 0 or eligible < 0 or inside > eligible:
                raise ValueError(f"invalid placement counts in {manifest_path}")
            zero_count = 0
            for side in ("buy", "sell"):
                distance_path = (
                    calibration_dir
                    / f"limit_{side}_distance_distribution.txt"
                )
                distance_rows = read_rows(distance_path)
                for row in distance_rows:
                    distance = int(row["distance_ticks"])
                    count = int(row["count"])
                    if distance < 0 or count <= 0:
                        raise ValueError(
                            f"invalid distance distribution in {distance_path}"
                        )
                    if distance == 0:
                        zero_count += count
            if inside > zero_count:
                raise ValueError(
                    f"inside count exceeds combined distance-zero count in "
                    f"{manifest_path}"
                )
            return inside / zero_count if zero_count else 0.0
    return min(0.05, 0.05 / max(1.0, mean_spread))


def build(args: argparse.Namespace) -> list[dict[str, object]]:
    data_root = pathlib.Path(args.data_root)
    opening_date = compact(args.opening_date)
    calibration_date = compact(args.calibration_date)
    opening_path = data_root / (
        f"itch_{opening_date}_basket/opening_bbo_{opening_date}.csv"
    )
    openings = {row["symbol"].upper(): row for row in read_rows(opening_path)}
    missing = set(SYMBOLS).difference(openings)
    if missing:
        raise ValueError("opening BBO is missing: " + ", ".join(sorted(missing)))
    weights = read_weights(pathlib.Path(args.weights_file), calibration_date)
    qqq_midpoint = float(openings["QQQ"]["mid_price_ticks"])
    if qqq_midpoint <= 0.0:
        raise ValueError("QQQ opening midpoint must be positive")

    rows: list[dict[str, object]] = []
    for book_id, symbol in enumerate(SYMBOLS):
        lower = symbol.lower()
        calibration_dir = data_root / f"itch_{calibration_date}_{lower}"
        target_rows = {
            row["name"]: float(row["target"])
            for row in read_rows(
                calibration_dir
                / f"market_targets_{lower}_{calibration_date}.csv"
            )
        }
        mean_spread = target_rows["mean_spread_ticks"]
        spread = max(1, round(mean_spread))
        median_buy = weighted_median(read_rows(
            calibration_dir / "limit_buy_quantity_distribution.txt"
        ))
        median_sell = weighted_median(read_rows(
            calibration_dir / "limit_sell_quantity_distribution.txt"
        ))
        quote_quantity = max(10, min(1_000, round(0.5 * (median_buy + median_sell))))
        opening = openings[symbol]
        rows.append({
            "book_id": book_id,
            "symbol": symbol,
            "data_dir": portable_path(calibration_dir),
            "hawkes_rates_file": portable_path(
                calibration_dir
                / f"hawkes_rates_{lower}_balanced_{calibration_date}.csv"
            ),
            "fundamental_price_ticks": opening["mid_price_ticks"],
            "initial_best_bid_ticks": opening["best_bid_ticks"],
            "initial_best_ask_ticks": opening["best_ask_ticks"],
            "initial_best_bid_depth": opening["best_bid_depth"],
            "initial_best_ask_depth": opening["best_ask_depth"],
            # Common-factor exposure is measured in QQQ-share equivalents.
            # Price-ratio beta makes a hedge approximately dollar neutral at
            # the opening state without using any later held-out prices.
            "beta": float(opening["mid_price_ticks"]) / qqq_midpoint,
            "basket_weight": 0.0 if symbol == "QQQ" else weights[symbol],
            "market_maker_quote_quantity": quote_quantity,
            "target_spread_ticks": spread,
            "quote_improvement_probability": quote_improvement_probability(
                calibration_dir, symbol, calibration_date, mean_spread,
            ),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--opening-date", required=True)
    parser.add_argument("--calibration-date", required=True)
    parser.add_argument(
        "--weights-file",
        default="config/qqq_reduced_basket_weights_20190930.csv",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = build(args)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
