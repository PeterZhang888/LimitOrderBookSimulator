#!/usr/bin/env python3
"""Stream one NASDAQ ITCH 5.0 symbol into simulator calibration inputs.

The ITCH file is a sequence of two-byte big-endian message lengths followed by
the message body.  This extractor never expands the full gzip file to disk.  It
uses Stock Locate to ignore unrelated symbols after the directory message and
reconstructs the selected visible order book from A/F/E/C/X/D/U messages.

Outputs are intentionally explicit about what is observed and what is inferred:

* six weighted quantity distributions used by the C++ simulator;
* four quote-relative distance distributions;
* fixed-clock market targets compatible with the calibration loader;
* a JSON summary/provenance manifest with message and data-quality counters.

Executions against resting asks are classified as aggressive buys; executions
against resting bids are classified as aggressive sells.  Non-cross trade
messages (P) are counted for diagnostics but are not inserted into the visible
LOB event buckets because they do not identify a resting displayed order.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import gzip
import heapq
import io
import json
import math
import os
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import BinaryIO, DefaultDict, Dict, Iterable, Optional, Tuple


NANOSECONDS_PER_SECOND = 1_000_000_000
ITCH_PRICE_UNITS_PER_CENT = 100


def parse_clock_ns(value: str) -> int:
    pieces = value.split(":")
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("time must be HH:MM:SS")
    try:
        hour, minute, second = (int(piece) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("time must be HH:MM:SS") from exc
    if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
        raise argparse.ArgumentTypeError("time must be a valid HH:MM:SS")
    return ((hour * 60 + minute) * 60 + second) * NANOSECONDS_PER_SECOND


def timestamp_ns(message: bytes) -> int:
    return int.from_bytes(message[5:11], "big", signed=False)


def stock_text(field: bytes) -> str:
    return field.decode("ascii", errors="replace").strip()


def safe_log_return(current: float, previous: float) -> Optional[float]:
    if current <= 0.0 or previous <= 0.0:
        return None
    value = math.log(current / previous)
    return value if math.isfinite(value) else None


@dataclass
class Order:
    side: str
    price: int
    remaining: int


class VisibleBook:
    def __init__(self) -> None:
        self.orders: Dict[int, Order] = {}
        self.bid_depth: DefaultDict[int, int] = collections.defaultdict(int)
        self.ask_depth: DefaultDict[int, int] = collections.defaultdict(int)
        self.bid_heap: list[int] = []
        self.ask_heap: list[int] = []

    def _levels(self, side: str) -> DefaultDict[int, int]:
        return self.bid_depth if side == "B" else self.ask_depth

    def add(self, reference: int, side: str, shares: int, price: int) -> None:
        if shares <= 0 or price <= 0 or side not in ("B", "S"):
            return
        if reference in self.orders:
            self.delete(reference)
        self.orders[reference] = Order(side=side, price=price, remaining=shares)
        levels = self._levels(side)
        levels[price] += shares
        if side == "B":
            heapq.heappush(self.bid_heap, -price)
        else:
            heapq.heappush(self.ask_heap, price)

    def reduce(self, reference: int, shares: int) -> Tuple[Optional[Order], int]:
        order = self.orders.get(reference)
        if order is None or shares <= 0:
            return order, 0
        removed = min(shares, order.remaining)
        levels = self._levels(order.side)
        levels[order.price] -= removed
        if levels[order.price] <= 0:
            levels.pop(order.price, None)
        order.remaining -= removed
        if order.remaining <= 0:
            self.orders.pop(reference, None)
        return order, removed

    def delete(self, reference: int) -> Tuple[Optional[Order], int]:
        order = self.orders.get(reference)
        if order is None:
            return None, 0
        remaining = order.remaining
        self.reduce(reference, remaining)
        return order, remaining

    def replace(self, old_reference: int, new_reference: int, shares: int, price: int) -> Optional[Order]:
        old, _ = self.delete(old_reference)
        if old is not None:
            self.add(new_reference, old.side, shares, price)
        return old

    def best_bid(self) -> int:
        while self.bid_heap and self.bid_depth.get(-self.bid_heap[0], 0) <= 0:
            heapq.heappop(self.bid_heap)
        return -self.bid_heap[0] if self.bid_heap else 0

    def best_ask(self) -> int:
        while self.ask_heap and self.ask_depth.get(self.ask_heap[0], 0) <= 0:
            heapq.heappop(self.ask_heap)
        return self.ask_heap[0] if self.ask_heap else 0

    def best_depth(self, side: str) -> int:
        price = self.best_bid() if side == "B" else self.best_ask()
        return self._levels(side).get(price, 0) if price > 0 else 0

    def distance_ticks(self, side: str, price: int) -> int:
        if side == "B":
            best = self.best_bid()
            return max(0, (best - price) // ITCH_PRICE_UNITS_PER_CENT) if best > 0 else 0
        best = self.best_ask()
        return max(0, (price - best) // ITCH_PRICE_UNITS_PER_CENT) if best > 0 else 0


class FixedClockSummary:
    def __init__(self) -> None:
        self.snapshots = 0
        self.invalid_snapshots = 0
        self.spread_sum = 0.0
        self.bid_depth_sum = 0.0
        self.ask_depth_sum = 0.0
        self.mid_moves = 0
        self.previous_mid: Optional[float] = None
        self.returns: list[float] = []
        # One regular-clock observation is small, while retaining the series
        # permits a deterministic delete-block jackknife for target scales.
        self.observations: list[tuple[float, float, float, float]] = []
        # Unlike ``observations``, this series retains invalid/one-sided clock
        # ticks as None.  Calibration windows are defined in elapsed market
        # time, so silently taking the first N *valid* observations would use
        # a longer empirical horizon whenever a real book is one-sided.
        self.clock_observations: list[
            Optional[tuple[float, float, float, float]]
        ] = []

    def observe(self, book: VisibleBook) -> None:
        bid = book.best_bid()
        ask = book.best_ask()
        if bid <= 0 or ask <= bid:
            self.invalid_snapshots += 1
            self.clock_observations.append(None)
            return
        mid = 0.5 * (bid + ask)
        self.snapshots += 1
        self.spread_sum += (ask - bid) / ITCH_PRICE_UNITS_PER_CENT
        self.bid_depth_sum += book.best_depth("B")
        self.ask_depth_sum += book.best_depth("S")
        observation = (
            (ask - bid) / ITCH_PRICE_UNITS_PER_CENT,
            float(book.best_depth("B")),
            float(book.best_depth("S")),
            mid,
        )
        self.observations.append(observation)
        self.clock_observations.append(observation)
        if self.previous_mid is not None:
            if mid != self.previous_mid:
                self.mid_moves += 1
            value = safe_log_return(mid, self.previous_mid)
            if value is not None:
                self.returns.append(value)
        self.previous_mid = mid

    def values(self) -> Dict[str, float]:
        return self._values_from_blocks([self._clock_series()])

    def _clock_series(
        self,
    ) -> list[Optional[tuple[float, float, float, float]]]:
        # The fallback retains compatibility with small programmatic fixtures
        # that predate explicit clock-validity recording.
        return (
            list(self.clock_observations)
            if self.clock_observations else list(self.observations)
        )

    @staticmethod
    def _values_from_blocks(
        blocks: list[list[Optional[tuple[float, float, float, float]]]],
    ) -> Dict[str, float]:
        clock_count = sum(len(block) for block in blocks)
        observations = [
            item for block in blocks for item in block if item is not None
        ]
        if not observations:
            result = {name: 0.0 for name in MARKET_TARGET_NAMES}
            result["two_sided_sample_fraction"] = 0.0
            return result
        returns: list[float] = []
        mid_moves = 0
        adjacent_pairs = 0
        for block in blocks:
            previous: Optional[tuple[float, float, float, float]] = None
            for observation in block:
                if observation is None:
                    previous = None
                    continue
                if previous is None:
                    previous = observation
                    continue
                current = observation[3]
                previous_mid = previous[3]
                adjacent_pairs += 1
                mid_moves += current != previous_mid
                value = safe_log_return(current, previous_mid)
                if value is not None:
                    returns.append(value)
                previous = observation
        result = {
            "mean_spread_ticks": statistics.fmean(item[0] for item in observations),
            "mean_bid_depth": statistics.fmean(item[1] for item in observations),
            "mean_ask_depth": statistics.fmean(item[2] for item in observations),
            "mid_move_rate": mid_moves / adjacent_pairs if adjacent_pairs else 0.0,
            "return_variance": 0.0,
            "return_kurtosis": 0.0,
            "absolute_return_acf1": 0.0,
            "two_sided_sample_fraction": (
                len(observations) / clock_count if clock_count else 0.0
            ),
        }
        if not returns:
            return result
        mean_return = statistics.fmean(returns)
        second_raw = statistics.fmean(value * value for value in returns)
        variance = max(0.0, second_raw - mean_return * mean_return)
        result["return_variance"] = variance
        if variance > 0.0:
            fourth_raw = statistics.fmean(value ** 4 for value in returns)
            result["return_kurtosis"] = fourth_raw / (variance * variance)
        absolute = [abs(value) for value in returns]
        if len(absolute) > 1:
            mean_abs = statistics.fmean(absolute)
            variance_abs = statistics.fmean(
                (value - mean_abs) ** 2 for value in absolute
            )
            if variance_abs > 0.0:
                cross = statistics.fmean(
                    absolute[index] * absolute[index - 1]
                    for index in range(1, len(absolute))
                )
                result["absolute_return_acf1"] = (
                    cross - mean_abs * mean_abs
                ) / variance_abs
        return result

    @classmethod
    def _target_scales_for_observations(
        cls,
        observations: list[Optional[tuple[float, float, float, float]]],
        *,
        block_observations: int = 300,
        minimum_blocks: int = 8,
    ) -> tuple[Dict[str, float], str, int]:
        """Return target scales for one contiguous regular-clock prefix.

        The short prefix used by the first calibration screen cannot support a
        stable delete-block jackknife.  It is intentionally labelled as a
        provisional screen scale; later stages use enough observations for the
        same block-jackknife construction as the full-session target.
        """
        if block_observations <= 1:
            raise ValueError("block_observations must exceed one")
        blocks = [
            observations[start:start + block_observations]
            for start in range(0, len(observations), block_observations)
            if len(observations[start:start + block_observations]) >= 2
        ]
        values = cls._values_from_blocks([observations])
        if len(blocks) < minimum_blocks:
            return (
                {name: provisional_scale(name, values[name])
                 for name in MARKET_TARGET_NAMES},
                "provisional_10pct_with_metric_floors_short_window",
                len(blocks),
            )
        delete_estimates = {name: [] for name in MARKET_TARGET_NAMES}
        for omitted in range(len(blocks)):
            estimate = cls._values_from_blocks(
                [block for index, block in enumerate(blocks) if index != omitted]
            )
            for name in MARKET_TARGET_NAMES:
                delete_estimates[name].append(estimate[name])
        scales: Dict[str, float] = {}
        block_count = len(blocks)
        for name, estimates in delete_estimates.items():
            center = statistics.fmean(estimates)
            jackknife_se = math.sqrt(
                (block_count - 1) / block_count
                * sum((value - center) ** 2 for value in estimates)
            )
            scales[name] = max(metric_scale_floor(name), jackknife_se)
        return scales, "delete_block_jackknife_300_observations_with_floors", block_count

    def window_values_and_scales(
        self,
        observations: int,
        *,
        block_observations: int = 300,
        minimum_blocks: int = 8,
    ) -> tuple[Dict[str, float], Dict[str, float], str, int]:
        """Return moments and scales for the first ``observations`` samples.

        A calibration horizon must be matched against the *same* empirical
        prefix, rather than against a full-day target with a shorter simulated
        path.  Requiring a complete prefix catches accidental partial ITCH
        extractions before expensive simulation begins.
        """
        if observations <= 0:
            raise ValueError("window observations must be positive")
        clock_series = self._clock_series()
        if len(clock_series) < observations:
            raise ValueError(
                "requested target window has fewer fixed-clock observations "
                f"than required ({len(clock_series)} < {observations})"
            )
        prefix = clock_series[:observations]
        values = self._values_from_blocks([prefix])
        scales, method, blocks = self._target_scales_for_observations(
            prefix,
            block_observations=block_observations,
            minimum_blocks=minimum_blocks,
        )
        return values, scales, method, blocks

    def target_scales(self, block_observations: int = 300,
                      minimum_blocks: int = 8) -> tuple[Dict[str, float], str, int]:
        """Estimate target uncertainty with a deterministic delete-block jackknife.

        Five-minute blocks at the default one-second clock preserve local
        dependence. Metric-specific floors prevent degenerate scales. Very
        short test fixtures transparently retain the legacy provisional rule.
        """
        scales, method, blocks = self._target_scales_for_observations(
            self._clock_series(),
            block_observations=block_observations,
            minimum_blocks=minimum_blocks,
        )
        if method == "provisional_10pct_with_metric_floors_short_window":
            # Preserve the established artifact label for legacy short fixtures
            # and one-symbol extraction tests.
            method = "provisional_10pct_with_metric_floors_short_fixture"
        return scales, method, blocks


def increment_positive(counter: DefaultDict[int, int], value: int) -> None:
    if value > 0:
        counter[value] += 1


def increment_nonnegative(counter: DefaultDict[int, int], value: int) -> None:
    if value >= 0:
        counter[value] += 1


def write_distribution(path: pathlib.Path, value_name: str, values: Dict[int, int]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([value_name, "count"])
        for value, count in sorted(values.items()):
            writer.writerow([value, count])


MARKET_TARGET_NAMES = (
    "mean_spread_ticks",
    "mean_bid_depth",
    "mean_ask_depth",
    "mid_move_rate",
    "return_variance",
    "return_kurtosis",
    "absolute_return_acf1",
    "two_sided_sample_fraction",
)

# Predeclared equal importance weights for the four-asset diagonal weighted
# moment match.  The principal weighting comes from each metric's empirical
# delete-block-jackknife scale, which standardises units and sampling
# uncertainty.  Keeping the optional importance weights equal avoids a
# post-hoc preference for any stylised fact while still allowing a future,
# preregistered study to change them consistently across train and holdout.
MARKET_TARGET_WEIGHTS = {name: 1.0 for name in MARKET_TARGET_NAMES}


def metric_scale_floor(name: str) -> float:
    floors = {
        "mean_spread_ticks": 0.25,
        "mean_bid_depth": 100.0,
        "mean_ask_depth": 100.0,
        "mid_move_rate": 0.01,
        "return_variance": 1.0e-12,
        "return_kurtosis": 0.5,
        "absolute_return_acf1": 0.02,
        # A half-percentage-point scale makes 100% empirical coverage a
        # meaningful liquid-book target without allowing one rare stochastic
        # snapshot to make the entire candidate loss undefined.
        "two_sided_sample_fraction": 0.005,
    }
    return floors[name]


def provisional_scale(name: str, target: float) -> float:
    return max(metric_scale_floor(name), 0.10 * abs(target))


def write_market_targets(path: pathlib.Path, values: Dict[str, float],
                         scales: Optional[Dict[str, float]] = None) -> None:
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["name", "target", "scale", "weight"])
        for name in MARKET_TARGET_NAMES:
            target = values[name]
            scale = scales[name] if scales is not None else provisional_scale(name, target)
            writer.writerow([
                name, f"{target:.17g}", f"{scale:.17g}",
                f"{MARKET_TARGET_WEIGHTS[name]:.17g}",
            ])


def read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise EOFError(f"truncated ITCH message: expected {size} bytes, received {len(data)}")
    return data


def extract(args: argparse.Namespace) -> Dict[str, object]:
    input_path = pathlib.Path(args.input).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    symbol = args.symbol.upper()
    start_ns = parse_clock_ns(args.start)
    end_ns = parse_clock_ns(args.end)
    if end_ns <= start_ns:
        raise ValueError("session end must be after session start")
    snapshot_interval_ns = args.snapshot_ms * 1_000_000

    book = VisibleBook()
    summary = FixedClockSummary()
    locate: Optional[int] = None
    # Match SimulationRecorder: observe at +1 interval, not at the session's
    # initial instant.  With the default interval this produces 23,400 regular-
    # session observations for a 6.5-hour day.
    next_snapshot_ns = start_ns + snapshot_interval_ns

    quantity: Dict[str, DefaultDict[int, int]] = {
        name: collections.defaultdict(int)
        for name in ("limit_buy", "limit_sell", "market_buy", "market_sell", "cancel_bid", "cancel_ask")
    }
    distance: Dict[str, DefaultDict[int, int]] = {
        name: collections.defaultdict(int)
        for name in ("limit_buy", "limit_sell", "cancel_bid", "cancel_ask")
    }
    message_counts: DefaultDict[str, int] = collections.defaultdict(int)
    selected_counts: DefaultDict[str, int] = collections.defaultdict(int)
    quality: DefaultDict[str, int] = collections.defaultdict(int)
    placement: DefaultDict[str, int] = collections.defaultdict(int)
    match_observations: Dict[int, Tuple[str, int]] = {}

    total_messages = 0
    started = time.monotonic()
    last_progress = started
    compressed_size = input_path.stat().st_size
    input_sha256 = getattr(args, "input_sha256", "").strip().lower()
    if input_sha256:
        if len(input_sha256) != 64 or any(character not in "0123456789abcdef" for character in input_sha256):
            raise ValueError("--input-sha256 must contain exactly 64 hexadecimal characters")

    def in_session(event_time_ns: int) -> bool:
        return start_ns <= event_time_ns < end_ns

    def emit_snapshots_until(event_time_ns: int) -> None:
        nonlocal next_snapshot_ns
        limit = min(event_time_ns, end_ns)
        while next_snapshot_ns <= limit:
            summary.observe(book)
            next_snapshot_ns += snapshot_interval_ns

    def observe_add(side: str, shares: int, price: int) -> None:
        bucket = "limit_buy" if side == "B" else "limit_sell"
        increment_positive(quantity[bucket], shares)
        increment_nonnegative(distance[bucket], book.distance_ticks(side, price))
        best_bid = book.best_bid()
        best_ask = book.best_ask()
        if best_bid > 0 and best_ask - best_bid >= 2 * ITCH_PRICE_UNITS_PER_CENT:
            placement["improvement_eligible_limit_orders"] += 1
            if best_bid < price < best_ask:
                placement["inside_spread_limit_orders"] += 1

    def observe_cancel(order: Order, shares: int) -> None:
        bucket = "cancel_bid" if order.side == "B" else "cancel_ask"
        increment_positive(quantity[bucket], shares)
        increment_nonnegative(distance[bucket], book.distance_ticks(order.side, order.price))

    raw = input_path.open("rb")
    try:
        magic = raw.read(2)
        raw.seek(0)
        decoded: BinaryIO = gzip.GzipFile(fileobj=raw, mode="rb") if magic == b"\x1f\x8b" else raw
        stream = io.BufferedReader(decoded, buffer_size=8 * 1024 * 1024)
        while True:
            length_bytes = stream.read(2)
            if not length_bytes:
                break
            if len(length_bytes) != 2:
                raise EOFError("truncated two-byte ITCH record length")
            length = int.from_bytes(length_bytes, "big", signed=False)
            if length == 0:
                quality["binaryfile_terminator_records"] += 1
                break
            message = read_exact(stream, length)
            total_messages += 1
            kind = chr(message[0])
            message_counts[kind] += 1

            if kind == "R" and len(message) == 39:
                directory_symbol = stock_text(message[11:19])
                if directory_symbol == symbol:
                    locate = int.from_bytes(message[1:3], "big", signed=False)
                    selected_counts["stock_directory"] += 1
                continue

            if locate is None or len(message) < 11:
                continue
            message_locate = int.from_bytes(message[1:3], "big", signed=False)
            if message_locate != locate:
                continue

            event_time_ns = timestamp_ns(message)
            emit_snapshots_until(event_time_ns)
            selected_counts[kind] += 1
            record = in_session(event_time_ns)

            if (kind == "A" and len(message) == 36) or (kind == "F" and len(message) == 40):
                reference = int.from_bytes(message[11:19], "big", signed=False)
                side = chr(message[19])
                shares = int.from_bytes(message[20:24], "big", signed=False)
                message_symbol = stock_text(message[24:32])
                price = int.from_bytes(message[32:36], "big", signed=False)
                if message_symbol != symbol:
                    quality["locate_symbol_mismatch"] += 1
                    continue
                if record:
                    observe_add(side, shares, price)
                book.add(reference, side, shares, price)

            elif (kind == "E" and len(message) == 31) or (kind == "C" and len(message) == 36):
                reference = int.from_bytes(message[11:19], "big", signed=False)
                executed = int.from_bytes(message[19:23], "big", signed=False)
                match_number = int.from_bytes(message[23:31], "big", signed=False)
                order = book.orders.get(reference)
                if order is None:
                    quality["execution_missing_order"] += 1
                    continue
                printable = kind == "E" or chr(message[31]) == "Y"
                if record and printable:
                    bucket = "market_sell" if order.side == "B" else "market_buy"
                    increment_positive(quantity[bucket], executed)
                    match_observations[match_number] = (bucket, executed)
                elif record and kind == "C":
                    quality["non_printable_c_executions_excluded_from_flow"] += 1
                _, removed = book.reduce(reference, executed)
                if removed != executed:
                    quality["execution_quantity_clamped"] += 1

            elif kind == "X" and len(message) == 23:
                reference = int.from_bytes(message[11:19], "big", signed=False)
                cancelled = int.from_bytes(message[19:23], "big", signed=False)
                order = book.orders.get(reference)
                if order is None:
                    quality["cancel_missing_order"] += 1
                    continue
                if record:
                    observe_cancel(order, min(cancelled, order.remaining))
                _, removed = book.reduce(reference, cancelled)
                if removed != cancelled:
                    quality["cancel_quantity_clamped"] += 1

            elif kind == "D" and len(message) == 19:
                reference = int.from_bytes(message[11:19], "big", signed=False)
                order = book.orders.get(reference)
                if order is None:
                    quality["delete_missing_order"] += 1
                    continue
                if record:
                    observe_cancel(order, order.remaining)
                book.delete(reference)

            elif kind == "U" and len(message) == 35:
                old_reference = int.from_bytes(message[11:19], "big", signed=False)
                new_reference = int.from_bytes(message[19:27], "big", signed=False)
                shares = int.from_bytes(message[27:31], "big", signed=False)
                price = int.from_bytes(message[31:35], "big", signed=False)
                old = book.orders.get(old_reference)
                if old is None:
                    quality["replace_missing_order"] += 1
                    continue
                side = old.side
                if record:
                    observe_cancel(old, old.remaining)
                book.delete(old_reference)
                if record:
                    observe_add(side, shares, price)
                book.add(new_reference, side, shares, price)

            elif kind == "P" and len(message) == 44:
                selected_counts["non_cross_trade_excluded"] += 1

            elif kind == "B" and len(message) == 19:
                match_number = int.from_bytes(message[11:19], "big", signed=False)
                observation = match_observations.pop(match_number, None)
                if observation is not None:
                    bucket, executed = observation
                    quantity[bucket][executed] -= 1
                    if quantity[bucket][executed] <= 0:
                        quantity[bucket].pop(executed, None)
                    quality["broken_trade_flow_observations_removed"] += 1

            now = time.monotonic()
            if args.progress_seconds > 0 and now - last_progress >= args.progress_seconds:
                compressed = raw.tell()
                percentage = 100.0 * compressed / compressed_size if compressed_size else 0.0
                elapsed = now - started
                print(
                    f"ITCH scan: {percentage:6.2f}% compressed, {total_messages:,} messages, "
                    f"{elapsed:,.1f}s elapsed",
                    file=sys.stderr,
                    flush=True,
                )
                last_progress = now

        emit_snapshots_until(end_ns)
    finally:
        raw.close()

    if locate is None:
        raise RuntimeError(f"symbol {symbol!r} was not found in Stock Directory messages")

    filenames = {
        "limit_buy": "limit_buy_quantity_distribution.txt",
        "limit_sell": "limit_sell_quantity_distribution.txt",
        "market_buy": "market_buy_quantity_distribution.txt",
        "market_sell": "market_sell_quantity_distribution.txt",
        "cancel_bid": "cancel_bid_quantity_distribution.txt",
        "cancel_ask": "cancel_ask_quantity_distribution.txt",
    }
    for bucket, filename in filenames.items():
        write_distribution(output_dir / filename, "quantity", quantity[bucket])

    distance_filenames = {
        "limit_buy": "limit_buy_distance_distribution.txt",
        "limit_sell": "limit_sell_distance_distribution.txt",
        "cancel_bid": "cancel_bid_distance_distribution.txt",
        "cancel_ask": "cancel_ask_distance_distribution.txt",
    }
    for bucket, filename in distance_filenames.items():
        write_distribution(output_dir / filename, "distance_ticks", distance[bucket])

    market_values = summary.values()
    market_scales, scale_method, scale_blocks = summary.target_scales()
    market_target_path = output_dir / f"market_targets_{symbol.lower()}_{args.date.replace('-', '')}.csv"
    write_market_targets(market_target_path, market_values, market_scales)
    window_targets: Dict[str, object] = {}
    for seconds in getattr(args, "target_window_seconds", []):
        if seconds <= 0:
            raise ValueError("--target-window-seconds values must be positive")
        numerator = seconds * 1000
        if numerator % args.snapshot_ms != 0:
            raise ValueError(
                "target window must contain an integer number of fixed-clock observations"
            )
        observations = numerator // args.snapshot_ms
        values, scales, method, blocks = summary.window_values_and_scales(observations)
        filename = (
            f"market_targets_{symbol.lower()}_{args.date.replace('-', '')}_"
            f"window_{seconds}s.csv"
        )
        write_market_targets(output_dir / filename, values, scales)
        window_targets[str(seconds)] = {
            "file": filename,
            "duration_seconds": seconds,
            "observations": observations,
            "valid_snapshots": sum(
                item is not None
                for item in summary._clock_series()[:observations]
            ),
            "invalid_snapshots": sum(
                item is None
                for item in summary._clock_series()[:observations]
            ),
            "two_sided_sample_fraction": values[
                "two_sided_sample_fraction"
            ],
            "scale_method": method,
            "scale_blocks": blocks,
            "values": values,
            "scales": scales,
        }

    elapsed = time.monotonic() - started
    manifest: Dict[str, object] = {
        "format": "NASDAQ TotalView-ITCH 5.0",
        # Keep the artifact relocatable and avoid leaking a workstation path.
        "input_path": input_path.name,
        "input_size_bytes": compressed_size,
        "input_sha256": input_sha256 or None,
        "input_mtime_utc": dt.datetime.fromtimestamp(
            input_path.stat().st_mtime, tz=dt.timezone.utc
        ).isoformat(),
        "trading_date": args.date,
        "symbol": symbol,
        "stock_locate": locate,
        "session_start": args.start,
        "session_end": args.end,
        "snapshot_interval_ms": args.snapshot_ms,
        "price_encoding": "ITCH integer units of 0.0001 USD",
        "tick_size_price_units": ITCH_PRICE_UNITS_PER_CENT,
        "aggressor_inference": "resting ask execution -> aggressive buy; resting bid execution -> aggressive sell",
        "replace_handling": "old remaining quantity recorded as cancellation; replacement recorded as new limit order",
        "delete_handling": "remaining displayed quantity recorded as cancellation",
        "non_cross_trade_handling": "P messages excluded from visible LOB event buckets",
        "cross_trade_handling": "Q cross messages excluded from continuous visible LOB targets",
        "non_printable_execution_handling": "C printable=N mutates displayed depth but is excluded from aggressive-flow distributions",
        "broken_trade_handling": "B reverses a previously recorded E/C aggressive-flow observation when its match is known",
        "total_messages": total_messages,
        "message_counts": dict(sorted(message_counts.items())),
        "selected_symbol_counts": dict(sorted(selected_counts.items())),
        "data_quality_counts": dict(sorted(quality.items())),
        "placement_counts": dict(sorted(placement.items())),
        "distribution_observation_counts": {
            name: int(sum(counter.values())) for name, counter in quantity.items()
        },
        "valid_snapshots": summary.snapshots,
        "invalid_snapshots": summary.invalid_snapshots,
        "market_values": market_values,
        "market_target_scale_method": scale_method,
        "market_target_scale_blocks": scale_blocks,
        "market_target_scales": market_scales,
        "market_target_windows": window_targets,
        "elapsed_wall_seconds": elapsed,
        "extractor": "scripts/extract_itch50_symbol.py",
    }
    manifest_path = output_dir / f"itch_manifest_{symbol.lower()}_{args.date.replace('-', '')}.json"
    with manifest_path.open("w") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="path to .gz or raw ITCH 5.0 file")
    parser.add_argument(
        "--input-sha256",
        default="",
        help="optional precomputed SHA-256 recorded in the provenance manifest",
    )
    parser.add_argument("--symbol", default="QQQ", help="NASDAQ stock symbol (default: QQQ)")
    parser.add_argument("--date", default="2020-01-30", help="trading date recorded in the manifest")
    parser.add_argument("--start", default="09:30:00", help="regular-session start, HH:MM:SS")
    parser.add_argument("--end", default="16:00:00", help="regular-session end, HH:MM:SS")
    parser.add_argument(
        "--snapshot-ms",
        type=int,
        default=1000,
        help="fixed market-state sampling interval (default: 1000, matching the C++ recorder)",
    )
    parser.add_argument(
        "--target-window-seconds",
        type=int,
        nargs="*",
        default=[],
        help=("write matched empirical-prefix target CSVs for these simulated "
              "durations"),
    )
    parser.add_argument("--output-dir", required=True, help="directory for distributions, targets and manifest")
    parser.add_argument("--progress-seconds", type=float, default=15.0, help="stderr progress interval; 0 disables")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.snapshot_ms <= 0:
        parser.error("--snapshot-ms must be positive")
    try:
        manifest = extract(args)
    except Exception as exc:
        print(f"ITCH extraction failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "symbol": manifest["symbol"],
        "stock_locate": manifest["stock_locate"],
        "total_messages": manifest["total_messages"],
        "valid_snapshots": manifest["valid_snapshots"],
        "elapsed_wall_seconds": manifest["elapsed_wall_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
