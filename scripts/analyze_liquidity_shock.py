#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Measure causal liquidity propagation in paired control/shock state traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from collections import defaultdict


STATE_FIELDS = (
    "best_bid_ticks", "best_ask_ticks", "best_bid_depth", "best_ask_depth",
    "mid_price_ticks",
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trace(path: pathlib.Path) -> dict[int, list[dict[str, str]]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    required = {"book_id", "symbol", "exchange_time_ns", *STATE_FIELDS}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} is empty or lacks required state fields")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["book_id"])].append(row)
    for book_rows in grouped.values():
        book_rows.sort(key=lambda row: int(row["exchange_time_ns"]))
    return dict(grouped)


def within_recovery(control: dict[str, str], shock: dict[str, str],
                    args: argparse.Namespace) -> bool:
    spread_control = int(control["best_ask_ticks"]) - int(control["best_bid_ticks"])
    spread_shock = int(shock["best_ask_ticks"]) - int(shock["best_bid_ticks"])
    mid_difference = abs(float(shock["mid_price_ticks"])
                         - float(control["mid_price_ticks"])) / args.tick_size
    depth_control = int(control["best_bid_depth"]) + int(control["best_ask_depth"])
    depth_shock = int(shock["best_bid_depth"]) + int(shock["best_ask_depth"])
    depth_limit = max(args.depth_absolute_tolerance,
                      args.depth_relative_tolerance * max(1, depth_control))
    return (
        abs(spread_shock - spread_control) / args.tick_size
            <= args.spread_tolerance_ticks
        and mid_difference <= args.mid_tolerance_ticks
        and abs(depth_shock - depth_control) <= depth_limit
    )


def analyze_book(control_rows: list[dict[str, str]],
                 shock_rows: list[dict[str, str]],
                 args: argparse.Namespace) -> dict[str, object]:
    control_by_time = {int(row["exchange_time_ns"]): row for row in control_rows}
    shock_by_time = {int(row["exchange_time_ns"]): row for row in shock_rows}
    if set(control_by_time) != set(shock_by_time):
        raise ValueError("paired traces do not have identical sample times")
    symbol = control_rows[0]["symbol"]
    if any(row["symbol"] != symbol for row in control_rows + shock_rows):
        raise ValueError("symbol changes inside one book trace")

    times = sorted(control_by_time)
    for timestamp in times:
        if timestamp >= args.shock_time_ns:
            break
        control = control_by_time[timestamp]
        shock = shock_by_time[timestamp]
        if any(control[field] != shock[field] for field in STATE_FIELDS):
            raise ValueError(
                f"paired paths for {symbol} differ before the shock at {timestamp}"
            )

    post_times = [timestamp for timestamp in times
                  if timestamp >= args.shock_time_ns]
    response_times = [
        timestamp for timestamp in post_times
        if any(control_by_time[timestamp][field] != shock_by_time[timestamp][field]
               for field in STATE_FIELDS)
    ]
    first_response = response_times[0] if response_times else None

    peak_depth_loss = 0
    peak_depth_loss_fraction = 0.0
    peak_spread_widening = 0.0
    peak_mid_displacement = 0.0
    for timestamp in post_times:
        control = control_by_time[timestamp]
        shock = shock_by_time[timestamp]
        control_depth = (int(control["best_bid_depth"])
                         + int(control["best_ask_depth"]))
        shock_depth = (int(shock["best_bid_depth"])
                       + int(shock["best_ask_depth"]))
        depth_loss = max(0, control_depth - shock_depth)
        peak_depth_loss = max(peak_depth_loss, depth_loss)
        peak_depth_loss_fraction = max(
            peak_depth_loss_fraction,
            depth_loss / max(1, control_depth),
        )
        control_spread = int(control["best_ask_ticks"]) - int(control["best_bid_ticks"])
        shock_spread = int(shock["best_ask_ticks"]) - int(shock["best_bid_ticks"])
        peak_spread_widening = max(
            peak_spread_widening,
            max(0, shock_spread - control_spread) / args.tick_size,
        )
        peak_mid_displacement = max(
            peak_mid_displacement,
            abs(float(shock["mid_price_ticks"])
                - float(control["mid_price_ticks"])) / args.tick_size,
        )

    recovery_time = None
    if first_response is not None:
        start_index = post_times.index(first_response)
        window = args.recovery_window_samples
        for index in range(start_index, len(post_times) - window + 1):
            candidate = post_times[index:index + window]
            if all(within_recovery(control_by_time[timestamp],
                                   shock_by_time[timestamp], args)
                   for timestamp in candidate):
                recovery_time = candidate[0]
                break

    return {
        "book_id": int(control_rows[0]["book_id"]),
        "symbol": symbol,
        "affected": first_response is not None,
        "first_response_time_ns": first_response,
        "propagation_delay_seconds": (
            (first_response - args.shock_time_ns) / 1e9
            if first_response is not None else None
        ),
        "peak_best_depth_loss": peak_depth_loss,
        "peak_best_depth_loss_fraction": peak_depth_loss_fraction,
        "peak_spread_widening_ticks": peak_spread_widening,
        "peak_absolute_mid_displacement_ticks": peak_mid_displacement,
        "recovered": recovery_time is not None,
        "recovery_time_ns": recovery_time,
        "recovery_seconds_after_shock": (
            (recovery_time - args.shock_time_ns) / 1e9
            if recovery_time is not None else None
        ),
    }


def analyze(args: argparse.Namespace) -> dict[str, object]:
    control_path = pathlib.Path(args.control_trace).resolve()
    shock_path = pathlib.Path(args.shock_trace).resolve()
    control = load_trace(control_path)
    shock = load_trace(shock_path)
    if set(control) != set(shock):
        raise ValueError("control and shock traces contain different books")
    books = [analyze_book(control[book_id], shock[book_id], args)
             for book_id in sorted(control)]
    affected = [book for book in books if book["affected"]]
    cross_asset = [book for book in affected if book["book_id"] != args.shock_book]
    recovered = [book for book in affected if book["recovered"]]
    control_index = {
        book_id: {int(row["exchange_time_ns"]): row for row in rows}
        for book_id, rows in control.items()
    }
    shock_index = {
        book_id: {int(row["exchange_time_ns"]): row for row in rows}
        for book_id, rows in shock.items()
    }
    common_times = set.intersection(*(
        set(control_index[book_id]) for book_id in sorted(control)
    ))
    peak_aggregate_depth_loss = 0
    for timestamp in common_times:
        if timestamp < args.shock_time_ns:
            continue
        aggregate_loss = 0
        for book_id in sorted(control):
            control_row = control_index[book_id][timestamp]
            shock_row = shock_index[book_id][timestamp]
            control_depth = (int(control_row["best_bid_depth"])
                             + int(control_row["best_ask_depth"]))
            shock_depth = (int(shock_row["best_bid_depth"])
                           + int(shock_row["best_ask_depth"]))
            aggregate_loss += max(0, control_depth - shock_depth)
        peak_aggregate_depth_loss = max(peak_aggregate_depth_loss, aggregate_loss)
    return {
        "protocol": {
            "control_trace": str(control_path),
            "control_trace_sha256": sha256_file(control_path),
            "shock_trace": str(shock_path),
            "shock_trace_sha256": sha256_file(shock_path),
            "shock_time_ns": args.shock_time_ns,
            "shock_book": args.shock_book,
            "tick_size": args.tick_size,
            "recovery_window_samples": args.recovery_window_samples,
            "recovery_tolerances": {
                "spread_ticks": args.spread_tolerance_ticks,
                "mid_ticks": args.mid_tolerance_ticks,
                "depth_relative": args.depth_relative_tolerance,
                "depth_absolute": args.depth_absolute_tolerance,
            },
        },
        "system": {
            "book_count": len(books),
            "affected_book_count": len(affected),
            "cross_asset_affected_book_count": len(cross_asset),
            "first_cross_asset_response_seconds": min(
                (float(book["propagation_delay_seconds"]) for book in cross_asset),
                default=None,
            ),
            "all_affected_books_recovered": len(recovered) == len(affected),
            "system_recovery_seconds": max(
                (float(book["recovery_seconds_after_shock"]) for book in recovered),
                default=None,
            ) if len(recovered) == len(affected) else None,
            "peak_aggregate_best_depth_loss": peak_aggregate_depth_loss,
        },
        "books": books,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-trace", required=True)
    parser.add_argument("--shock-trace", required=True)
    parser.add_argument("--shock-time-ns", type=int, required=True)
    parser.add_argument("--shock-book", type=int, required=True)
    parser.add_argument("--tick-size", type=int, default=100)
    parser.add_argument("--spread-tolerance-ticks", type=float, default=1.0)
    parser.add_argument("--mid-tolerance-ticks", type=float, default=1.0)
    parser.add_argument("--depth-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--depth-absolute-tolerance", type=int, default=100)
    parser.add_argument("--recovery-window-samples", type=int, default=30)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    if args.shock_time_ns < 0 or args.tick_size <= 0:
        parser.error("shock time must be non-negative and tick size positive")
    if args.recovery_window_samples <= 0:
        parser.error("recovery window must be positive")

    report = analyze(args)
    output_json = pathlib.Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    books = report["books"]
    assert isinstance(books, list)
    output_csv = pathlib.Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(books[0]))
        writer.writeheader()
        writer.writerows(books)
    print(json.dumps({"system": report["system"], "output": str(output_json)},
                     sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
