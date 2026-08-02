#!/usr/bin/env python3
"""Extract several ITCH 5.0 books in one streaming pass.

This is the production extractor for the coupled-LOB experiment.  It preserves
the single-symbol extractor's event classification and fixed-clock targets, but
routes messages by Stock Locate so QQQ, AAPL, MSFT, and AMZN are reconstructed
without decompressing and scanning the multi-gigabyte source four times.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import pathlib
import resource
import sys
import time
from dataclasses import dataclass, field
from typing import BinaryIO, DefaultDict, Dict, Optional, Tuple

from extract_itch50_symbol import (
    FixedClockSummary,
    Order,
    VisibleBook,
    increment_nonnegative,
    increment_positive,
    parse_clock_ns,
    read_exact,
    stock_text,
    timestamp_ns,
    write_distribution,
    write_market_targets,
)


QUANTITY_BUCKETS = (
    "limit_buy", "limit_sell", "market_buy", "market_sell",
    "cancel_bid", "cancel_ask",
)
DISTANCE_BUCKETS = ("limit_buy", "limit_sell", "cancel_bid", "cancel_ask")
NANOSECONDS_PER_SECOND = 1_000_000_000
TICK_SIZE_PRICE_UNITS = 100
HALF_HOUR_SECONDS = 30 * 60
HALF_HOUR_NS = HALF_HOUR_SECONDS * NANOSECONDS_PER_SECOND
HALF_HOUR_BIN_COUNT = 13
SPREAD_BINS = ("one_tick", "wider", "unavailable")
QUEUE_IMBALANCE_BINS = (
    "sell_very_high",
    "sell_high",
    "balanced",
    "buy_high",
    "buy_very_high",
    "unavailable",
)
DEPTH_RATIO_BINS = ("low", "typical", "high", "unavailable")
OPENING_BBO_FIELDS = (
    "symbol",
    "clock",
    "best_bid_ticks",
    "best_ask_ticks",
    "best_bid_depth",
    "best_ask_depth",
    "mid_price_ticks",
)


@dataclass(frozen=True)
class StateTargets:
    mean_bid_depth: float
    mean_ask_depth: float


@dataclass(frozen=True)
class StateTargetsSource:
    filename: str
    sha256: str


@dataclass(frozen=True)
class TrainingEventObservation:
    time_key: Tuple[int, str]
    state_key: Tuple[int, str, str, str, str, str]
    second_key: Tuple[int, str]


@dataclass(frozen=True)
class FlowObservation:
    bucket: str
    quantity: int
    training_event: Optional[TrainingEventObservation]


def format_clock_ns(value_ns: int) -> str:
    seconds = value_ns // NANOSECONDS_PER_SECOND
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def decrement_counter(counter: DefaultDict[object, int], key: object,
                      description: str) -> None:
    if counter.get(key, 0) <= 0:
        raise RuntimeError(f"cannot reverse absent {description}: {key!r}")
    counter[key] -= 1
    if counter[key] == 0:
        counter.pop(key, None)


def queue_imbalance_bin(bid_depth: int, ask_depth: int) -> str:
    total = bid_depth + ask_depth
    if bid_depth <= 0 or ask_depth <= 0 or total <= 0:
        return "unavailable"
    imbalance = (bid_depth - ask_depth) / total
    if imbalance < -0.6:
        return "sell_very_high"
    if imbalance < -0.2:
        return "sell_high"
    if imbalance < 0.2:
        return "balanced"
    if imbalance < 0.6:
        return "buy_high"
    return "buy_very_high"


def depth_ratio_bin(depth: int, target: float) -> str:
    if depth <= 0:
        return "unavailable"
    ratio = depth / target
    if ratio < 0.5:
        return "low"
    if ratio < 1.5:
        return "typical"
    return "high"


@dataclass
class QueueReactiveTrainingSummary:
    """Streaming sufficient statistics for queue-reactive estimation.

    The object retains counters and state-duration totals only.  Individual
    events are never stored.  The one per-match token used for printable
    executions is the minimum information needed to honour later broken-trade
    messages exactly, matching the legacy quantity-distribution semantics.
    """

    targets: StateTargets
    source: StateTargetsSource
    start_ns: int
    end_ns: int
    exposure_cursor_ns: int = field(init=False)
    # ITCH permits several messages for one symbol to carry the same exchange
    # timestamp.  A continuous-time point-process covariate is the predictable
    # left limit S(t-), so every modelled event in such a timestamp batch must
    # use the book state before the first message in the batch.  Recomputing
    # the state after each zero-duration message would create event states with
    # no corresponding exposure and would let file order leak into the fitted
    # queue response.
    event_state_time_ns: Optional[int] = field(init=False, default=None)
    event_state_bins: Optional[Tuple[str, str, str, str]] = field(
        init=False, default=None
    )
    improvement_price_units: Dict[str, DefaultDict[int, int]] = field(
        default_factory=lambda: {
            "limit_buy": collections.defaultdict(int),
            "limit_sell": collections.defaultdict(int),
        }
    )
    event_time_counts: DefaultDict[Tuple[int, str], int] = field(
        default_factory=lambda: collections.defaultdict(int)
    )
    state_event_counts: DefaultDict[
        Tuple[int, str, str, str, str, str], int
    ] = field(default_factory=lambda: collections.defaultdict(int))
    state_exposure_ns: DefaultDict[
        Tuple[int, str, str, str, str], int
    ] = field(default_factory=lambda: collections.defaultdict(int))
    # One-second count bins are sufficient statistics for auditable lagged
    # count correlations.  They are not an event stream and contain no order
    # identifiers, timestamps below one second, prices or quantities.
    second_event_counts: DefaultDict[Tuple[int, str], int] = field(
        default_factory=lambda: collections.defaultdict(int)
    )

    def __post_init__(self) -> None:
        self.exposure_cursor_ns = self.start_ns

    def half_hour_bin(self, event_time_ns: int) -> int:
        if not self.start_ns <= event_time_ns < self.end_ns:
            raise ValueError("training event lies outside the extraction session")
        index = (event_time_ns - self.start_ns) // HALF_HOUR_NS
        if not 0 <= index < HALF_HOUR_BIN_COUNT:
            raise ValueError("training event lies outside the 13 half-hour bins")
        return int(index)

    def state_bins(self, book: VisibleBook) -> Tuple[str, str, str, str]:
        bid = book.best_bid()
        ask = book.best_ask()
        bid_depth = book.best_depth("B")
        ask_depth = book.best_depth("S")
        spread = ask - bid
        if (
            bid <= 0
            or ask <= bid
            or bid_depth <= 0
            or ask_depth <= 0
            or spread < TICK_SIZE_PRICE_UNITS
        ):
            return ("unavailable",) * 4
        spread_bin = (
            "one_tick" if spread == TICK_SIZE_PRICE_UNITS else "wider"
        )
        return (
            spread_bin,
            queue_imbalance_bin(bid_depth, ask_depth),
            depth_ratio_bin(bid_depth, self.targets.mean_bid_depth),
            depth_ratio_bin(ask_depth, self.targets.mean_ask_depth),
        )

    def advance_exposure(self, book: VisibleBook, event_time_ns: int) -> None:
        clipped = min(max(event_time_ns, self.start_ns), self.end_ns)
        if clipped < self.exposure_cursor_ns:
            raise RuntimeError("selected-symbol ITCH timestamps are not monotone")
        while self.exposure_cursor_ns < clipped:
            index = (self.exposure_cursor_ns - self.start_ns) // HALF_HOUR_NS
            if not 0 <= index < HALF_HOUR_BIN_COUNT:
                raise RuntimeError("exposure lies outside the 13 half-hour bins")
            boundary = min(
                clipped,
                self.start_ns + (index + 1) * HALF_HOUR_NS,
            )
            state_key = (int(index), *self.state_bins(book))
            self.state_exposure_ns[state_key] += (
                boundary - self.exposure_cursor_ns
            )
            self.exposure_cursor_ns = boundary
        if (
            self.start_ns <= event_time_ns < self.end_ns
            and self.event_state_time_ns != event_time_ns
        ):
            self.event_state_time_ns = event_time_ns
            self.event_state_bins = self.state_bins(book)

    def observe_event(self, event_type: str,
                      event_time_ns: int,
                      book: VisibleBook) -> TrainingEventObservation:
        if event_type not in QUANTITY_BUCKETS:
            raise ValueError(f"unknown queue-reactive event type: {event_type}")
        index = self.half_hour_bin(event_time_ns)
        time_key = (index, event_type)
        # ``advance_exposure`` is called before every selected-symbol message.
        # Require that call and use its timestamp-batch left limit rather than
        # the potentially mutated book passed by a later tied message.
        if (
            self.event_state_time_ns != event_time_ns
            or self.event_state_bins is None
        ):
            raise RuntimeError(
                "queue-reactive event observation lacks a timestamp-batch "
                "left-limit state"
            )
        state_key = (index, event_type, *self.event_state_bins)
        second_index = (event_time_ns - self.start_ns) // NANOSECONDS_PER_SECOND
        if not 0 <= second_index < (self.end_ns - self.start_ns) // NANOSECONDS_PER_SECOND:
            raise ValueError("training event lies outside one-second count bins")
        second_key = (int(second_index), event_type)
        self.event_time_counts[time_key] += 1
        self.state_event_counts[state_key] += 1
        self.second_event_counts[second_key] += 1
        return TrainingEventObservation(time_key, state_key, second_key)

    def reverse_event(self, observation: TrainingEventObservation) -> None:
        decrement_counter(
            self.event_time_counts, observation.time_key,
            "half-hour event observation",
        )
        decrement_counter(
            self.state_event_counts, observation.state_key,
            "pre-event state observation",
        )
        decrement_counter(
            self.second_event_counts, observation.second_key,
            "one-second event observation",
        )

    def observe_inside_spread(self, side: str, price: int,
                              book: VisibleBook) -> None:
        best_bid = book.best_bid()
        best_ask = book.best_ask()
        if not best_bid < price < best_ask:
            return
        if side == "B":
            bucket = "limit_buy"
            improvement = price - best_bid
        else:
            bucket = "limit_sell"
            improvement = best_ask - price
        if improvement > 0:
            self.improvement_price_units[bucket][improvement] += 1


@dataclass
class SymbolState:
    symbol: str
    next_snapshot_ns: int
    book: VisibleBook = field(default_factory=VisibleBook)
    summary: FixedClockSummary = field(default_factory=FixedClockSummary)
    locate: Optional[int] = None
    quantity: Dict[str, DefaultDict[int, int]] = field(default_factory=lambda: {
        name: collections.defaultdict(int) for name in QUANTITY_BUCKETS
    })
    distance: Dict[str, DefaultDict[int, int]] = field(default_factory=lambda: {
        name: collections.defaultdict(int) for name in DISTANCE_BUCKETS
    })
    selected_counts: DefaultDict[str, int] = field(
        default_factory=lambda: collections.defaultdict(int)
    )
    quality: DefaultDict[str, int] = field(
        default_factory=lambda: collections.defaultdict(int)
    )
    placement: DefaultDict[str, int] = field(
        default_factory=lambda: collections.defaultdict(int)
    )
    match_observations: Dict[int, FlowObservation] = field(default_factory=dict)
    opening: Optional[dict[str, object]] = None
    training: Optional[QueueReactiveTrainingSummary] = None

    def emit_snapshots_until(self, event_time_ns: int, end_ns: int,
                             snapshot_interval_ns: int) -> None:
        limit = min(event_time_ns, end_ns)
        while self.next_snapshot_ns <= limit:
            self.summary.observe(self.book)
            self.next_snapshot_ns += snapshot_interval_ns

    def capture_opening(self, clock: str) -> None:
        if self.opening is not None:
            return
        bid = self.book.best_bid()
        ask = self.book.best_ask()
        self.opening = {
            "symbol": self.symbol,
            "clock": clock,
            "best_bid_ticks": bid,
            "best_ask_ticks": ask,
            "best_bid_depth": self.book.best_depth("B"),
            "best_ask_depth": self.book.best_depth("S"),
            "mid_price_ticks": 0.5 * (bid + ask) if bid > 0 and ask > bid else 0.0,
        }

    def observe_add(self, side: str, shares: int, price: int,
                    event_time_ns: int) -> None:
        bucket = "limit_buy" if side == "B" else "limit_sell"
        increment_positive(self.quantity[bucket], shares)
        increment_nonnegative(
            self.distance[bucket], self.book.distance_ticks(side, price)
        )
        best_bid = self.book.best_bid()
        best_ask = self.book.best_ask()
        tick = 100
        if best_bid > 0 and best_ask - best_bid >= 2 * tick:
            self.placement["improvement_eligible_limit_orders"] += 1
            if best_bid < price < best_ask:
                self.placement["inside_spread_limit_orders"] += 1
        if self.training is not None and shares > 0:
            self.training.observe_inside_spread(side, price, self.book)
            self.training.observe_event(bucket, event_time_ns, self.book)

    def observe_cancel(self, order: Order, shares: int,
                       event_time_ns: int) -> None:
        bucket = "cancel_bid" if order.side == "B" else "cancel_ask"
        increment_positive(self.quantity[bucket], shares)
        increment_nonnegative(
            self.distance[bucket],
            self.book.distance_ticks(order.side, order.price),
        )
        if self.training is not None and shares > 0:
            self.training.observe_event(bucket, event_time_ns, self.book)


def validate_sha256(value: str) -> str:
    digest = value.strip().lower()
    if digest and (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("--input-sha256 must contain exactly 64 hexadecimal characters")
    return digest


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_state_targets(path_value: str,
                       symbols: list[str]) -> tuple[
                           dict[str, StateTargets], StateTargetsSource
                       ]:
    path = pathlib.Path(path_value).resolve()
    required = {
        "symbol",
        "target_mean_bid_depth",
        "target_mean_ask_depth",
    }
    targets: dict[str, StateTargets] = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or [])
        missing_fields = sorted(required - fields)
        if missing_fields:
            raise ValueError(
                "--state-targets-csv lacks required columns: "
                + ", ".join(missing_fields)
            )
        for row_number, row in enumerate(reader, start=2):
            symbol = str(row["symbol"]).strip().upper()
            if not symbol:
                raise ValueError(
                    f"--state-targets-csv row {row_number} has an empty symbol"
                )
            if symbol in targets:
                raise ValueError(
                    f"--state-targets-csv contains duplicate symbol {symbol}"
                )
            try:
                bid = float(row["target_mean_bid_depth"])
                ask = float(row["target_mean_ask_depth"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"--state-targets-csv row {row_number} has non-numeric depths"
                ) from error
            if not math.isfinite(bid) or bid <= 0.0:
                raise ValueError(
                    f"--state-targets-csv {symbol} bid-depth target must be "
                    "finite and positive"
                )
            if not math.isfinite(ask) or ask <= 0.0:
                raise ValueError(
                    f"--state-targets-csv {symbol} ask-depth target must be "
                    "finite and positive"
                )
            targets[symbol] = StateTargets(bid, ask)
    missing_symbols = sorted(set(symbols) - targets.keys())
    if missing_symbols:
        raise ValueError(
            "--state-targets-csv lacks selected symbols: "
            + ", ".join(missing_symbols)
        )
    return (
        {symbol: targets[symbol] for symbol in symbols},
        StateTargetsSource(path.name, sha256_file(path)),
    )


def apply_message(state: SymbolState, message: bytes, kind: str,
                  record: bool, event_time_ns: int) -> None:
    book = state.book
    if (kind == "A" and len(message) == 36) or (
        kind == "F" and len(message) == 40
    ):
        reference = int.from_bytes(message[11:19], "big", signed=False)
        side = chr(message[19])
        shares = int.from_bytes(message[20:24], "big", signed=False)
        message_symbol = stock_text(message[24:32])
        price = int.from_bytes(message[32:36], "big", signed=False)
        if message_symbol != state.symbol:
            state.quality["locate_symbol_mismatch"] += 1
            return
        if record:
            state.observe_add(side, shares, price, event_time_ns)
        book.add(reference, side, shares, price)
        return

    if (kind == "E" and len(message) == 31) or (
        kind == "C" and len(message) == 36
    ):
        reference = int.from_bytes(message[11:19], "big", signed=False)
        executed = int.from_bytes(message[19:23], "big", signed=False)
        match_number = int.from_bytes(message[23:31], "big", signed=False)
        order = book.orders.get(reference)
        if order is None:
            state.quality["execution_missing_order"] += 1
            return
        printable = kind == "E" or chr(message[31]) == "Y"
        if record and printable:
            bucket = "market_sell" if order.side == "B" else "market_buy"
            increment_positive(state.quantity[bucket], executed)
            training_event = None
            if state.training is not None and executed > 0:
                training_event = state.training.observe_event(
                    bucket, event_time_ns, book
                )
            state.match_observations[match_number] = FlowObservation(
                bucket, executed, training_event
            )
        elif record and kind == "C":
            state.quality["non_printable_c_executions_excluded_from_flow"] += 1
        _, removed = book.reduce(reference, executed)
        if removed != executed:
            state.quality["execution_quantity_clamped"] += 1
        return

    if kind == "X" and len(message) == 23:
        reference = int.from_bytes(message[11:19], "big", signed=False)
        cancelled = int.from_bytes(message[19:23], "big", signed=False)
        order = book.orders.get(reference)
        if order is None:
            state.quality["cancel_missing_order"] += 1
            return
        if record:
            state.observe_cancel(
                order, min(cancelled, order.remaining), event_time_ns
            )
        _, removed = book.reduce(reference, cancelled)
        if removed != cancelled:
            state.quality["cancel_quantity_clamped"] += 1
        return

    if kind == "D" and len(message) == 19:
        reference = int.from_bytes(message[11:19], "big", signed=False)
        order = book.orders.get(reference)
        if order is None:
            state.quality["delete_missing_order"] += 1
            return
        if record:
            state.observe_cancel(order, order.remaining, event_time_ns)
        book.delete(reference)
        return

    if kind == "U" and len(message) == 35:
        old_reference = int.from_bytes(message[11:19], "big", signed=False)
        new_reference = int.from_bytes(message[19:27], "big", signed=False)
        shares = int.from_bytes(message[27:31], "big", signed=False)
        price = int.from_bytes(message[31:35], "big", signed=False)
        old = book.orders.get(old_reference)
        if old is None:
            state.quality["replace_missing_order"] += 1
            return
        side = old.side
        if record:
            state.observe_cancel(old, old.remaining, event_time_ns)
        book.delete(old_reference)
        if record:
            state.observe_add(side, shares, price, event_time_ns)
        book.add(new_reference, side, shares, price)
        return

    if kind == "P" and len(message) == 44:
        state.selected_counts["non_cross_trade_excluded"] += 1
        return

    if kind == "B" and len(message) == 19:
        match_number = int.from_bytes(message[11:19], "big", signed=False)
        observation = state.match_observations.pop(match_number, None)
        if observation is not None:
            # Retain the legacy reversal operation exactly.  ITCH executions
            # have positive quantities, but the direct decrement also matches
            # historical artifacts if a malformed zero-sized fixture appears.
            quantity_counter = state.quantity[observation.bucket]
            quantity_counter[observation.quantity] -= 1
            if quantity_counter[observation.quantity] <= 0:
                quantity_counter.pop(observation.quantity, None)
            if (
                state.training is not None
                and observation.training_event is not None
            ):
                state.training.reverse_event(observation.training_event)
            state.quality["broken_trade_flow_observations_removed"] += 1


def half_hour_bounds(start_ns: int, index: int) -> tuple[str, str]:
    lower = start_ns + index * HALF_HOUR_NS
    upper = lower + HALF_HOUR_NS
    return format_clock_ns(lower), format_clock_ns(upper)


def write_queue_reactive_training_artifacts(
    state: SymbolState,
    output_dir: pathlib.Path,
) -> dict[str, object]:
    training = state.training
    if training is None:
        raise ValueError("queue-reactive artifacts require state targets")

    improvement_filenames = {
        "limit_buy": "limit_buy_improvement_distribution.txt",
        "limit_sell": "limit_sell_improvement_distribution.txt",
    }
    improvement_rows: dict[str, int] = {}
    for event_type, filename in improvement_filenames.items():
        row_count = 0
        with (output_dir / filename).open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow([
                "improvement_ticks",
                "improvement_price_units",
                "count",
            ])
            for price_units, count in sorted(
                training.improvement_price_units[event_type].items()
            ):
                writer.writerow([
                    f"{price_units / TICK_SIZE_PRICE_UNITS:.17g}",
                    price_units,
                    count,
                ])
                row_count += 1
        improvement_rows[event_type] = row_count

    time_filename = "intraday_event_counts.csv"
    with (output_dir / time_filename).open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "half_hour_bin",
            "bin_start",
            "bin_end",
            "event_type",
            "count",
        ])
        for index in range(HALF_HOUR_BIN_COUNT):
            lower, upper = half_hour_bounds(training.start_ns, index)
            for event_type in QUANTITY_BUCKETS:
                writer.writerow([
                    index,
                    lower,
                    upper,
                    event_type,
                    training.event_time_counts.get((index, event_type), 0),
                ])

    state_count_filename = "queue_state_counts.csv"
    state_count_rows = 0
    with (output_dir / state_count_filename).open(
        "w", newline=""
    ) as output:
        writer = csv.writer(output)
        writer.writerow([
            "half_hour_bin",
            "bin_start",
            "bin_end",
            "event_type",
            "spread_bin",
            "queue_imbalance_bin",
            "bid_depth_ratio_bin",
            "ask_depth_ratio_bin",
            "count",
        ])
        for key, count in sorted(training.state_event_counts.items()):
            index, event_type, spread, imbalance, bid_ratio, ask_ratio = key
            lower, upper = half_hour_bounds(training.start_ns, index)
            writer.writerow([
                index,
                lower,
                upper,
                event_type,
                spread,
                imbalance,
                bid_ratio,
                ask_ratio,
                count,
            ])
            state_count_rows += 1

    exposure_filename = "queue_state_exposure.csv"
    exposure_rows = 0
    with (output_dir / exposure_filename).open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "half_hour_bin",
            "bin_start",
            "bin_end",
            "spread_bin",
            "queue_imbalance_bin",
            "bid_depth_ratio_bin",
            "ask_depth_ratio_bin",
            "exposure_seconds",
        ])
        for key, exposure_ns in sorted(training.state_exposure_ns.items()):
            index, spread, imbalance, bid_ratio, ask_ratio = key
            lower, upper = half_hour_bounds(training.start_ns, index)
            writer.writerow([
                index,
                lower,
                upper,
                spread,
                imbalance,
                bid_ratio,
                ask_ratio,
                f"{exposure_ns / NANOSECONDS_PER_SECOND:.9f}",
            ])
            exposure_rows += 1

    lag_filename = "event_count_lag_moments.csv"
    lag_rows = 0
    session_seconds = (training.end_ns - training.start_ns) // (
        NANOSECONDS_PER_SECOND
    )
    if session_seconds <= 1:
        raise RuntimeError(
            "queue-reactive lag moments require more than one session second"
        )
    requested_lags = (0, 1, 2, 5, 10, 20, 30)
    lag_seconds = [lag for lag in requested_lags if lag < session_seconds]
    sparse_series = {
        event_type: {
            second: count
            for (second, observed_type), count
            in training.second_event_counts.items()
            if observed_type == event_type
        }
        for event_type in QUANTITY_BUCKETS
    }
    with (output_dir / lag_filename).open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "source_event_type",
            "target_event_type",
            "lag_seconds",
            "paired_bins",
            "source_mean_count",
            "target_mean_count",
            "source_variance",
            "target_variance",
            "covariance",
            "correlation",
        ])
        for lag in lag_seconds:
            paired = session_seconds - lag
            source_stats = {}
            target_stats = {}
            for event_type in QUANTITY_BUCKETS:
                series = sparse_series[event_type]
                source_values = [
                    value for second, value in series.items()
                    if second < paired
                ]
                target_values = [
                    value for second, value in series.items()
                    if second >= lag
                ]
                source_stats[event_type] = (
                    math.fsum(source_values),
                    math.fsum(value * value for value in source_values),
                )
                target_stats[event_type] = (
                    math.fsum(target_values),
                    math.fsum(value * value for value in target_values),
                )
            for source_type in QUANTITY_BUCKETS:
                source = sparse_series[source_type]
                source_sum, source_square_sum = source_stats[source_type]
                source_mean = source_sum / paired
                source_variance = max(
                    0.0, source_square_sum / paired - source_mean ** 2
                )
                for target_type in QUANTITY_BUCKETS:
                    target = sparse_series[target_type]
                    target_sum, target_square_sum = target_stats[target_type]
                    target_mean = target_sum / paired
                    target_variance = max(
                        0.0, target_square_sum / paired - target_mean ** 2
                    )
                    cross_sum = math.fsum(
                        source_value * target.get(second + lag, 0)
                        for second, source_value in source.items()
                        if second < paired
                    )
                    covariance = cross_sum / paired - (
                        source_mean * target_mean
                    )
                    denominator = math.sqrt(
                        source_variance * target_variance
                    )
                    correlation = (
                        covariance / denominator if denominator > 0.0 else 0.0
                    )
                    writer.writerow([
                        source_type,
                        target_type,
                        lag,
                        paired,
                        f"{source_mean:.17g}",
                        f"{target_mean:.17g}",
                        f"{source_variance:.17g}",
                        f"{target_variance:.17g}",
                        f"{covariance:.17g}",
                        f"{correlation:.17g}",
                    ])
                    lag_rows += 1

    expected_exposure_ns = training.end_ns - training.start_ns
    total_exposure_ns = sum(training.state_exposure_ns.values())
    if total_exposure_ns != expected_exposure_ns:
        raise RuntimeError(
            "queue-state exposure does not cover the extraction session: "
            f"{total_exposure_ns} != {expected_exposure_ns} ns"
        )
    valid_exposure_ns = sum(
        exposure_ns
        for (_, spread, _, _, _), exposure_ns
        in training.state_exposure_ns.items()
        if spread != "unavailable"
    )
    unavailable_exposure_ns = total_exposure_ns - valid_exposure_ns

    distribution_counts = {
        event_type: int(sum(state.quantity[event_type].values()))
        for event_type in QUANTITY_BUCKETS
    }
    training_counts = {
        event_type: sum(
            count
            for (index, observed_type), count
            in training.event_time_counts.items()
            if observed_type == event_type
        )
        for event_type in QUANTITY_BUCKETS
    }
    if training_counts != distribution_counts:
        raise RuntimeError(
            "queue-reactive event counts disagree with the legacy quantity "
            f"observations: {training_counts!r} != {distribution_counts!r}"
        )
    total_time_events = sum(training.event_time_counts.values())
    total_state_events = sum(training.state_event_counts.values())
    if total_time_events != total_state_events:
        raise RuntimeError(
            "half-hour and pre-event-state count totals disagree: "
            f"{total_time_events} != {total_state_events}"
        )

    return {
        "schema_version": 2,
        "training_only": True,
        "queue_policy_estimation_ready": valid_exposure_ns > 0,
        "streaming_sufficient_statistics_only": True,
        "event_stream_retained": False,
        "state_targets": {
            "source_filename": training.source.filename,
            "source_sha256": training.source.sha256,
            "target_mean_bid_depth": training.targets.mean_bid_depth,
            "target_mean_ask_depth": training.targets.mean_ask_depth,
        },
        "artifacts": {
            "limit_buy_improvement_distribution": (
                improvement_filenames["limit_buy"]
            ),
            "limit_sell_improvement_distribution": (
                improvement_filenames["limit_sell"]
            ),
            "intraday_event_counts": time_filename,
            "queue_state_counts": state_count_filename,
            "queue_state_exposure": exposure_filename,
            "event_count_lag_moments": lag_filename,
        },
        "artifact_row_counts": {
            "limit_buy_improvement_distribution": (
                improvement_rows["limit_buy"]
            ),
            "limit_sell_improvement_distribution": (
                improvement_rows["limit_sell"]
            ),
            "intraday_event_counts": (
                HALF_HOUR_BIN_COUNT * len(QUANTITY_BUCKETS)
            ),
            "queue_state_counts": state_count_rows,
            "queue_state_exposure": exposure_rows,
            "event_count_lag_moments": lag_rows,
        },
        "event_types": list(QUANTITY_BUCKETS),
        "half_hour_bins": {
            "count": HALF_HOUR_BIN_COUNT,
            "width_seconds": HALF_HOUR_SECONDS,
            "anchor": format_clock_ns(training.start_ns),
            "right_open": True,
        },
        "pre_event_state_definition": {
            "state_is_sampled_before_book_mutation": True,
            "predictable_state": "left limit S(t-) before the first selected-symbol message at timestamp t",
            "equal_timestamp_messages_share_one_left_limit_state": True,
            "zero_duration_intermediate_states_are_not_used_as_covariates": True,
            "spread_bins": {
                "one_tick": "best_ask-best_bid == 100 ITCH price units",
                "wider": "best_ask-best_bid > 100 ITCH price units",
                "unavailable": "book is not valid two-sided or spread is below one tick",
            },
            "queue_imbalance": "(best_bid_depth-best_ask_depth)/(best_bid_depth+best_ask_depth)",
            "queue_imbalance_bins": [
                {"name": "sell_very_high", "interval": "[-1,-0.6)"},
                {"name": "sell_high", "interval": "[-0.6,-0.2)"},
                {"name": "balanced", "interval": "[-0.2,0.2)"},
                {"name": "buy_high", "interval": "[0.2,0.6)"},
                {"name": "buy_very_high", "interval": "[0.6,1]"},
                {"name": "unavailable", "interval": None},
            ],
            "depth_ratio": "best-side depth / side-specific training target mean depth",
            "depth_ratio_bins": [
                {"name": "low", "interval": "(0,0.5)"},
                {"name": "typical", "interval": "[0.5,1.5)"},
                {"name": "high", "interval": "[1.5,infinity)"},
                {"name": "unavailable", "interval": None},
            ],
        },
        "inside_spread_improvement_definition": {
            "bid": "new bid price minus pre-event best bid",
            "ask": "pre-event best ask minus new ask price",
            "tick_size_price_units": TICK_SIZE_PRICE_UNITS,
            "off_grid_values_are_retained_exactly_in_price_units": True,
        },
        "exposure": {
            "expected_session_seconds": (
                expected_exposure_ns / NANOSECONDS_PER_SECOND
            ),
            "total_seconds": total_exposure_ns / NANOSECONDS_PER_SECOND,
            "valid_two_sided_seconds": (
                valid_exposure_ns / NANOSECONDS_PER_SECOND
            ),
            "unavailable_seconds": (
                unavailable_exposure_ns / NANOSECONDS_PER_SECOND
            ),
            "exact_nanosecond_conservation": True,
        },
        "event_count_conservation": {
            "by_event_type": training_counts,
            "equals_legacy_quantity_observation_counts": True,
            "half_hour_total": total_time_events,
            "pre_event_state_total": total_state_events,
            "totals_equal": True,
        },
        "broken_trade_reversal": (
            "B removes the matched printable execution from legacy quantity, "
            "half-hour, pre-event-state and one-second counters; exposure is unchanged"
        ),
        "lag_moment_definition": {
            "count_bin_seconds": 1,
            "lags_seconds": lag_seconds,
            "direction": "source count at t versus target count at t+lag",
            "normalization": "population covariance and correlation",
            "individual_event_stream_retained": False,
        },
    }


def write_state(state: SymbolState, args: argparse.Namespace,
                input_path: pathlib.Path, compressed_size: int,
                input_sha256: str, total_messages: int,
                message_counts: DefaultDict[str, int], elapsed: float) -> dict[str, object]:
    date_compact = args.date.replace("-", "")
    output_dir = pathlib.Path(args.output_root).resolve() / (
        f"itch_{date_compact}_{state.symbol.lower()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    quantity_filenames = {
        name: f"{name}_quantity_distribution.txt" for name in QUANTITY_BUCKETS
    }
    for bucket, filename in quantity_filenames.items():
        write_distribution(output_dir / filename, "quantity", state.quantity[bucket])
    for bucket in DISTANCE_BUCKETS:
        write_distribution(
            output_dir / f"{bucket}_distance_distribution.txt",
            "distance_ticks",
            state.distance[bucket],
        )

    market_values = state.summary.values()
    market_scales, scale_method, scale_blocks = state.summary.target_scales()
    write_market_targets(
        output_dir / f"market_targets_{state.symbol.lower()}_{date_compact}.csv",
        market_values,
        market_scales,
    )
    window_targets: dict[str, dict[str, object]] = {}
    for seconds in getattr(args, "target_window_seconds", []):
        if seconds <= 0:
            raise ValueError("--target-window-seconds values must be positive")
        numerator = seconds * 1000
        if numerator % args.snapshot_ms != 0:
            raise ValueError(
                "target window must contain an integer number of fixed-clock "
                "observations"
            )
        observations = numerator // args.snapshot_ms
        values, scales, method, blocks = state.summary.window_values_and_scales(
            observations
        )
        filename = (
            f"market_targets_{state.symbol.lower()}_{date_compact}_"
            f"window_{seconds}s.csv"
        )
        write_market_targets(output_dir / filename, values, scales)
        window_targets[str(seconds)] = {
            "file": filename,
            "duration_seconds": seconds,
            "observations": observations,
            "scale_method": method,
            "scale_blocks": blocks,
            "values": values,
            "scales": scales,
        }
    manifest: dict[str, object] = {
        "format": "NASDAQ TotalView-ITCH 5.0",
        "input_path": input_path.name,
        "input_size_bytes": compressed_size,
        "input_sha256": input_sha256 or None,
        "input_mtime_utc": dt.datetime.fromtimestamp(
            input_path.stat().st_mtime, tz=dt.timezone.utc
        ).isoformat(),
        "trading_date": args.date,
        "symbol": state.symbol,
        "stock_locate": state.locate,
        "session_start": args.start,
        "session_end": args.end,
        "aggregation_duration_seconds": (
            parse_clock_ns(args.end) - parse_clock_ns(args.start)
        ) / NANOSECONDS_PER_SECOND,
        "snapshot_interval_ms": args.snapshot_ms,
        "price_encoding": "ITCH integer units of 0.0001 USD",
        "tick_size_price_units": 100,
        "aggressor_inference": "resting ask execution -> aggressive buy; resting bid execution -> aggressive sell",
        "replace_handling": "old remaining quantity recorded as cancellation; replacement recorded as new limit order",
        "delete_handling": "remaining displayed quantity recorded as cancellation",
        "non_cross_trade_handling": "P messages excluded from visible LOB event buckets",
        "cross_trade_handling": "Q cross messages excluded from continuous visible LOB targets",
        "non_printable_execution_handling": "C printable=N mutates displayed depth but is excluded from aggressive-flow distributions",
        "broken_trade_handling": "B reverses a previously recorded E/C aggressive-flow observation when its match is known",
        "total_messages": total_messages,
        "message_counts": dict(sorted(message_counts.items())),
        "selected_symbol_counts": dict(sorted(state.selected_counts.items())),
        "data_quality_counts": dict(sorted(state.quality.items())),
        "placement_counts": dict(sorted(state.placement.items())),
        "distribution_observation_counts": {
            name: int(sum(counter.values()))
            for name, counter in state.quantity.items()
        },
        "valid_snapshots": state.summary.snapshots,
        "invalid_snapshots": state.summary.invalid_snapshots,
        "market_values": market_values,
        "market_target_scale_method": scale_method,
        "market_target_scale_blocks": scale_blocks,
        "market_target_scales": market_scales,
        "market_target_windows": window_targets,
        "elapsed_wall_seconds": elapsed,
        "extractor": "scripts/extract_itch50_symbols.py",
        "single_pass_symbol_count": len(args.symbols),
    }
    if state.training is not None:
        manifest["queue_reactive_training_artifacts"] = (
            write_queue_reactive_training_artifacts(state, output_dir)
        )
    manifest_path = output_dir / (
        f"itch_manifest_{state.symbol.lower()}_{date_compact}.json"
    )
    with manifest_path.open("w") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    return manifest


def extract_many(args: argparse.Namespace) -> list[dict[str, object]]:
    input_path = pathlib.Path(args.input).resolve()
    symbols = [symbol.upper() for symbol in args.symbols]
    if len(symbols) != len(set(symbols)):
        raise ValueError("--symbols must not contain duplicates")
    start_ns = parse_clock_ns(args.start)
    end_ns = parse_clock_ns(args.end)
    if end_ns <= start_ns:
        raise ValueError("session end must be after session start")
    if args.snapshot_ms <= 0:
        raise ValueError("--snapshot-ms must be positive")
    snapshot_interval_ns = args.snapshot_ms * 1_000_000
    state_targets_value = getattr(args, "state_targets_csv", None)
    state_targets: Optional[dict[str, StateTargets]] = None
    state_targets_source: Optional[StateTargetsSource] = None
    if state_targets_value:
        if end_ns - start_ns > HALF_HOUR_BIN_COUNT * HALF_HOUR_NS:
            raise ValueError(
                "queue-reactive extraction cannot exceed the 13 half-hour "
                "training bins"
            )
        state_targets, state_targets_source = load_state_targets(
            str(state_targets_value), symbols
        )
    states = {
        symbol: SymbolState(symbol, start_ns + snapshot_interval_ns)
        for symbol in symbols
    }
    if state_targets is not None:
        assert state_targets_source is not None
        for symbol, state in states.items():
            state.training = QueueReactiveTrainingSummary(
                state_targets[symbol], state_targets_source, start_ns, end_ns
            )
    locate_to_state: dict[int, SymbolState] = {}
    input_sha256 = validate_sha256(getattr(args, "input_sha256", ""))
    compressed_size = input_path.stat().st_size
    message_counts: DefaultDict[str, int] = collections.defaultdict(int)
    total_messages = 0
    started = time.monotonic()
    last_progress = started

    raw = input_path.open("rb")
    try:
        magic = raw.read(2)
        raw.seek(0)
        decoded: BinaryIO = (
            gzip.GzipFile(fileobj=raw, mode="rb") if magic == b"\x1f\x8b" else raw
        )
        stream = io.BufferedReader(decoded, buffer_size=8 * 1024 * 1024)
        while True:
            length_bytes = stream.read(2)
            if not length_bytes:
                break
            if len(length_bytes) != 2:
                raise EOFError("truncated two-byte ITCH record length")
            length = int.from_bytes(length_bytes, "big", signed=False)
            if length == 0:
                for state in states.values():
                    state.quality["binaryfile_terminator_records"] += 1
                break
            message = read_exact(stream, length)
            total_messages += 1
            kind = chr(message[0])
            message_counts[kind] += 1

            if kind == "R" and len(message) == 39:
                symbol = stock_text(message[11:19])
                state = states.get(symbol)
                if state is not None:
                    locate = int.from_bytes(message[1:3], "big", signed=False)
                    if state.locate is not None and state.locate != locate:
                        raise RuntimeError(f"symbol {symbol} has conflicting Stock Locate values")
                    state.locate = locate
                    locate_to_state[locate] = state
                    state.selected_counts["stock_directory"] += 1
                continue

            if len(message) < 11:
                continue
            locate = int.from_bytes(message[1:3], "big", signed=False)
            state = locate_to_state.get(locate)
            if state is None:
                continue
            event_time_ns = timestamp_ns(message)
            state.emit_snapshots_until(event_time_ns, end_ns, snapshot_interval_ns)
            if state.training is not None:
                state.training.advance_exposure(state.book, event_time_ns)
            if event_time_ns >= start_ns:
                state.capture_opening(args.start)
            state.selected_counts[kind] += 1
            apply_message(
                state,
                message,
                kind,
                start_ns <= event_time_ns < end_ns,
                event_time_ns,
            )

            now = time.monotonic()
            if args.progress_seconds > 0 and now - last_progress >= args.progress_seconds:
                compressed = raw.tell()
                percentage = 100.0 * compressed / compressed_size if compressed_size else 0.0
                print(
                    f"ITCH scan: {percentage:6.2f}% compressed, "
                    f"{total_messages:,} messages, {now - started:,.1f}s elapsed",
                    file=sys.stderr,
                    flush=True,
                )
                last_progress = now
    finally:
        raw.close()

    missing = [symbol for symbol, state in states.items() if state.locate is None]
    if missing:
        raise RuntimeError(
            "symbols absent from Stock Directory: " + ", ".join(sorted(missing))
        )
    eligible_states: list[SymbolState] = []
    exclusions: list[dict[str, str]] = []
    for state in states.values():
        state.emit_snapshots_until(end_ns, end_ns, snapshot_interval_ns)
        if state.training is not None:
            state.training.advance_exposure(state.book, end_ns)
        state.capture_opening(args.start)
        assert state.opening is not None
        if (
            int(state.opening["best_bid_ticks"]) <= 0
            or int(state.opening["best_ask_ticks"])
                <= int(state.opening["best_bid_ticks"])
        ):
            if not getattr(args, "skip_invalid_openings", False):
                raise RuntimeError(f"{state.symbol} is not two-sided at {args.start}")
            exclusions.append({
                "symbol": state.symbol,
                "reason": f"not_two_sided_at_{args.start}",
            })
            continue
        eligible_states.append(state)

    if not eligible_states and not getattr(args, "skip_invalid_openings", False):
        raise RuntimeError("no selected symbol is two-sided at the requested start")

    elapsed = time.monotonic() - started
    eligible_symbols = {state.symbol for state in eligible_states}
    manifests = [
        write_state(
            states[symbol], args, input_path, compressed_size,
            input_sha256, total_messages, message_counts, elapsed,
        )
        for symbol in symbols
        if symbol in eligible_symbols
    ]

    date_compact = args.date.replace("-", "")
    opening_path = pathlib.Path(args.output_root).resolve() / (
        f"itch_{date_compact}_basket/opening_bbo_{date_compact}.csv"
    )
    opening_path.parent.mkdir(parents=True, exist_ok=True)
    openings = [state.opening for state in eligible_states]
    with opening_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=OPENING_BBO_FIELDS)
        writer.writeheader()
        writer.writerows(openings)
    if exclusions:
        exclusions_path = pathlib.Path(args.output_root).resolve() / (
            f"itch_{date_compact}_exclusions.csv"
        )
        with exclusions_path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["symbol", "reason"])
            writer.writeheader()
            writer.writerows(exclusions)
    return manifests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-sha256", default="")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument(
        "--symbols-file",
        help="newline-delimited symbol list; mutually exclusive with --symbols",
    )
    parser.add_argument("--date", required=True)
    parser.add_argument("--start", default="09:30:00")
    parser.add_argument("--end", default="16:00:00")
    parser.add_argument("--snapshot-ms", type=int, default=1000)
    parser.add_argument(
        "--target-window-seconds",
        type=int,
        nargs="*",
        default=[],
        help=("write matched empirical-prefix target CSVs for these simulated "
              "durations; intended for the three-stage calibration workflow"),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--state-targets-csv",
        help=(
            "optional training-only CSV with symbol, "
            "target_mean_bid_depth and target_mean_ask_depth; enables "
            "queue-state event counts and matching state exposure artifacts"
        ),
    )
    parser.add_argument("--progress-seconds", type=float, default=15.0)
    parser.add_argument(
        "--skip-invalid-openings",
        action="store_true",
        help="write valid symbols and record non-two-sided starts as exclusions",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if (args.symbols is None) == (args.symbols_file is None):
            raise ValueError("specify exactly one of --symbols or --symbols-file")
        if args.symbols_file is not None:
            with pathlib.Path(args.symbols_file).open(encoding="utf-8") as source:
                args.symbols = [line.strip() for line in source if line.strip()]
        assert args.symbols is not None
        manifests = extract_many(args)
    except Exception as exc:
        print(f"ITCH multi-symbol extraction failed: {exc}", file=sys.stderr)
        return 1
    peak_rss_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB while macOS reports bytes.  Normalise the diagnostic
    # so local and Seagull batch-size decisions use the same unit.
    peak_rss_bytes = (
        peak_rss_raw if sys.platform == "darwin" else peak_rss_raw * 1024
    )
    print(json.dumps({
        "symbols": [manifest["symbol"] for manifest in manifests],
        "total_messages": (
            manifests[0]["total_messages"] if manifests else None
        ),
        "valid_snapshots": {
            manifest["symbol"]: manifest["valid_snapshots"]
            for manifest in manifests
        },
        "elapsed_wall_seconds": (
            manifests[0]["elapsed_wall_seconds"] if manifests else None
        ),
        "peak_rss_bytes": peak_rss_bytes,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
