#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Build an auditable, real-symbol multi-asset configuration from ITCH outputs.

The program deliberately does *not* manufacture a second venue or replicate a
small set of templates.  It consumes the per-symbol directories produced by
``extract_itch50_symbols.py`` for one ITCH date, filters unusable instruments
with recorded reasons, derives a Hawkes-rate file for every accepted symbol,
and writes one contiguous ``MultiAssetBookConfig`` CSV.

The intended invocation is from the project root, for example::

    python3 scripts/build_itch_universe_config.py \
        --data-root data --trading-date 2020-01-30 \
        --catalog data/itch_20200130_basket/opening_bbo_20200130.csv \
        --output config/nasdaq_universe_20200130.csv \
        --provenance config/nasdaq_universe_20200130.provenance.json

``--catalog`` need contain only a ``symbol`` column.  It can itself be the
opening-BBO file; otherwise pass ``--opening-bbo`` separately.  Accepted
symbols are sorted deterministically with QQQ at book id zero.  The script
does not claim that every Stock Directory entry is an eligible common stock:
ETFs, warrants, inactive instruments, and symbols with incomplete visible-book
data are retained in the provenance report as rejected candidates.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import derive_hawkes_rates as hawkes  # noqa: E402


QUANTITY_EVENTS = tuple(hawkes.EVENT_NAMES)
DISTANCE_EVENTS = (
    "limit_buy",
    "limit_sell",
    "cancel_bid",
    "cancel_ask",
)
OPENING_COLUMNS = (
    "symbol",
    "best_bid_ticks",
    "best_ask_ticks",
    "best_bid_depth",
    "best_ask_depth",
    "mid_price_ticks",
)
CONFIG_FIELDS = (
    "book_id",
    "symbol",
    "data_dir",
    "hawkes_rates_file",
    "fundamental_price_ticks",
    "initial_best_bid_ticks",
    "initial_best_ask_ticks",
    "initial_best_bid_depth",
    "initial_best_ask_depth",
    "beta",
    "basket_weight",
    "market_maker_quote_quantity",
    "target_spread_ticks",
    "quote_improvement_probability",
)
MAX_INT32 = 2_147_483_647
SCHEMA_VERSION = 1


class UniverseBuildError(ValueError):
    """Raised when the requested universe cannot form a valid C++ config."""


@dataclass(frozen=True)
class OpeningBbo:
    bid_ticks: int
    ask_ticks: int
    bid_depth: int
    ask_depth: int
    mid_ticks: float


@dataclass(frozen=True)
class DistributionStats:
    total_count: int
    weighted_median: int | None
    mean: float
    zero_count: int
    zero_fraction: float


@dataclass
class PreparedSymbol:
    symbol: str
    data_dir: pathlib.Path
    manifest_path: pathlib.Path
    manifest: dict[str, Any]
    opening: OpeningBbo
    quantity_stats: dict[str, DistributionStats]
    target_spread_ticks: int
    quote_improvement_probability: float
    market_maker_quote_quantity: int
    rate_path: pathlib.Path
    hawkes_rows: list[dict[str, float | str]] | None = None
    rate_derivation: dict[str, object] | None = None


def compact_date(value: str) -> str:
    """Return YYYYMMDD and reject ambiguous date strings."""
    text = value.replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise UniverseBuildError("--trading-date must be YYYY-MM-DD")
    try:
        dt.date(int(text[:4]), int(text[4:6]), int(text[6:]))
    except ValueError as error:
        raise UniverseBuildError(f"invalid --trading-date: {value}") from error
    return text


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_headers(fieldnames: Sequence[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise UniverseBuildError("CSV has no header row")
    result: dict[str, str] = {}
    for fieldname in fieldnames:
        key = fieldname.strip().lower()
        if not key:
            continue
        if key in result:
            raise UniverseBuildError(f"duplicate CSV column after normalisation: {key}")
        result[key] = fieldname
    return result


def read_csv(path: pathlib.Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    try:
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            headers = canonical_headers(reader.fieldnames)
            rows = [dict(row) for row in reader]
    except OSError as error:
        raise UniverseBuildError(f"cannot read CSV {path}: {error}") from error
    return headers, rows


def get_field(row: Mapping[str, str], headers: Mapping[str, str], name: str) -> str:
    column = headers.get(name)
    return "" if column is None else str(row.get(column, "") or "").strip()


def normalise_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not symbol:
        raise ValueError("empty_symbol")
    if "/" in symbol or "\\" in symbol or any(character.isspace() for character in symbol):
        raise ValueError("unsafe_symbol")
    return symbol


def first_unique_symbols(rows: Iterable[Mapping[str, str]],
                         headers: Mapping[str, str],
                         source_name: str) -> tuple[list[str], set[str], list[str]]:
    """Return candidates, duplicate symbols, and row-level input issues."""
    if "symbol" not in headers:
        raise UniverseBuildError(f"{source_name} must contain a symbol column")
    symbols: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    issues: list[str] = []
    for number, row in enumerate(rows, start=2):
        try:
            symbol = normalise_symbol(get_field(row, headers, "symbol"))
        except ValueError as error:
            issues.append(f"{source_name}:row_{number}:{error}")
            continue
        if symbol in seen:
            duplicates.add(symbol)
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols, duplicates, issues


def read_openings(path: pathlib.Path) -> tuple[dict[str, Mapping[str, str]], set[str], list[str]]:
    headers, rows = read_csv(path)
    missing = [name for name in OPENING_COLUMNS if name not in headers]
    if missing:
        raise UniverseBuildError(
            f"opening BBO file {path} is missing columns: {', '.join(missing)}"
        )
    openings: dict[str, Mapping[str, str]] = {}
    duplicates: set[str] = set()
    issues: list[str] = []
    for number, row in enumerate(rows, start=2):
        try:
            symbol = normalise_symbol(get_field(row, headers, "symbol"))
        except ValueError as error:
            issues.append(f"opening_bbo:row_{number}:{error}")
            continue
        if symbol in openings:
            duplicates.add(symbol)
            continue
        # Use canonical keys internally so later validation is independent of
        # capitalisation in a third-party candidate file.
        openings[symbol] = {
            name: get_field(row, headers, name) for name in OPENING_COLUMNS
        }
    return openings, duplicates, issues


def parse_finite_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid_{name}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"nonfinite_{name}")
    return parsed


def parse_positive_int(value: str, name: str) -> int:
    parsed = parse_finite_float(value, name)
    if parsed <= 0.0:
        raise ValueError(f"nonpositive_{name}")
    if not parsed.is_integer():
        raise ValueError(f"nonintegral_{name}")
    integer = int(parsed)
    if integer > MAX_INT32:
        raise ValueError(f"out_of_range_{name}")
    return integer


def validate_opening(row: Mapping[str, str]) -> tuple[OpeningBbo | None, list[str]]:
    reasons: list[str] = []
    values: dict[str, int | float] = {}
    for name in (
        "best_bid_ticks",
        "best_ask_ticks",
        "best_bid_depth",
        "best_ask_depth",
    ):
        try:
            values[name] = parse_positive_int(str(row.get(name, "")), name)
        except ValueError as error:
            reasons.append(str(error))
    try:
        mid = parse_finite_float(str(row.get("mid_price_ticks", "")), "mid_price_ticks")
        if mid <= 0.0:
            reasons.append("nonpositive_mid_price_ticks")
    except ValueError as error:
        mid = math.nan
        reasons.append(str(error))

    if reasons:
        return None, reasons
    bid = int(values["best_bid_ticks"])
    ask = int(values["best_ask_ticks"])
    if ask <= bid:
        reasons.append("nonpositive_or_crossed_opening_spread")
    if not bid <= mid <= ask:
        reasons.append("mid_price_outside_opening_bbo")
    if reasons:
        return None, reasons
    return OpeningBbo(
        bid_ticks=bid,
        ask_ticks=ask,
        bid_depth=int(values["best_bid_depth"]),
        ask_depth=int(values["best_ask_depth"]),
        mid_ticks=mid,
    ), []


def parse_distribution_int(value: str, name: str, *, minimum: int) -> int:
    parsed = parse_finite_float(value, name)
    if not parsed.is_integer():
        raise ValueError(f"nonintegral_{name}")
    integer = int(parsed)
    if integer < minimum:
        raise ValueError(f"invalid_{name}")
    return integer


def distribution_stats(path: pathlib.Path, value_column: str, *, minimum_value: int) -> DistributionStats:
    """Validate an empirical distribution and calculate the marks needed here."""
    headers, rows = read_csv(path)
    if value_column not in headers or "count" not in headers:
        raise ValueError(f"missing_required_columns:{path.name}")
    observations: list[tuple[int, int]] = []
    for number, row in enumerate(rows, start=2):
        try:
            value = parse_distribution_int(
                get_field(row, headers, value_column), value_column,
                minimum=minimum_value,
            )
            count = parse_distribution_int(
                get_field(row, headers, "count"), "count", minimum=1,
            )
        except ValueError as error:
            raise ValueError(f"invalid_row_{number}:{error}") from error
        observations.append((value, count))
    if not observations:
        raise ValueError("empty_file")
    total = sum(count for _, count in observations)
    if total <= 0:
        raise ValueError("zero_total_count")
    weighted_sum = sum(value * count for value, count in observations)
    ordered = sorted(observations)
    cumulative = 0
    median: int | None = None
    for value, count in ordered:
        cumulative += count
        if 2 * cumulative >= total:
            median = value
            break
    return DistributionStats(
        total_count=total,
        weighted_median=median,
        mean=weighted_sum / total,
        zero_count=sum(count for value, count in observations if value == 0),
        zero_fraction=sum(count for value, count in observations if value == 0) / total,
    )


def parse_manifest_count(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid_{name}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid_{name}") from error
    if not math.isfinite(parsed) or parsed <= 0.0 or not parsed.is_integer():
        raise ValueError(f"invalid_{name}")
    return int(parsed)


def parse_nonnegative_manifest_count(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid_{name}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid_{name}") from error
    if not math.isfinite(parsed) or parsed < 0.0 or not parsed.is_integer():
        raise ValueError(f"invalid_{name}")
    return int(parsed)


def load_manifest(path: pathlib.Path, symbol: str, trading_date: str) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    try:
        with path.open() as source:
            manifest = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"invalid_manifest:{error}"]
    if not isinstance(manifest, dict):
        return None, ["manifest_is_not_an_object"]
    if str(manifest.get("symbol", "")).strip().upper() != symbol:
        reasons.append("manifest_symbol_mismatch")
    if str(manifest.get("trading_date", "")).strip() != trading_date:
        reasons.append("manifest_trading_date_mismatch")
    try:
        session_seconds = (
            hawkes.clock_seconds(str(manifest["session_end"]))
            - hawkes.clock_seconds(str(manifest["session_start"]))
        )
        if session_seconds <= 0:
            reasons.append("invalid_manifest_session_interval")
    except (KeyError, ValueError):
        reasons.append("invalid_manifest_session_interval")

    counts = manifest.get("distribution_observation_counts")
    if not isinstance(counts, dict):
        reasons.append("missing_distribution_observation_counts")
    else:
        for event in QUANTITY_EVENTS:
            try:
                parse_manifest_count(counts.get(event), f"observed_count_{event}")
            except ValueError as error:
                reasons.append(str(error))
    placement = manifest.get("placement_counts")
    if not isinstance(placement, dict):
        reasons.append("missing_placement_counts")
    else:
        try:
            eligible = parse_nonnegative_manifest_count(
                placement.get("improvement_eligible_limit_orders"),
                "improvement_eligible_limit_orders",
            )
            inside = parse_nonnegative_manifest_count(
                placement.get("inside_spread_limit_orders"),
                "inside_spread_limit_orders",
            )
            if inside > eligible:
                reasons.append("inside_spread_orders_exceed_eligible_orders")
        except ValueError as error:
            reasons.append(str(error))
    return manifest, reasons


def read_target_spread(path: pathlib.Path) -> tuple[int | None, list[str]]:
    try:
        headers, rows = read_csv(path)
    except UniverseBuildError as error:
        return None, [f"invalid_market_targets:{error}"]
    if "name" not in headers or "target" not in headers:
        return None, ["market_targets_missing_name_or_target"]
    values = [
        get_field(row, headers, "target")
        for row in rows
        if get_field(row, headers, "name") == "mean_spread_ticks"
    ]
    if len(values) != 1:
        return None, ["market_targets_missing_or_duplicate_mean_spread_ticks"]
    try:
        spread = parse_finite_float(values[0], "mean_spread_ticks")
    except ValueError as error:
        return None, [str(error)]
    if spread <= 0.0:
        return None, ["nonpositive_mean_spread_ticks"]
    rounded = int(math.floor(spread + 0.5))
    if rounded <= 0 or rounded > MAX_INT32:
        return None, ["out_of_range_target_spread_ticks"]
    return rounded, []


def quote_improvement_probability(
    manifest: Mapping[str, Any],
    buy_distances: DistributionStats,
    sell_distances: DistributionStats,
) -> float:
    """Return the aggregate zero-distance split identifiable from compact ITCH.

    Inside-spread and at-best additions are both encoded at distance zero.
    The compact manifest retains one aggregate inside count, not a buy/sell
    allocation. Dividing by the combined side-zero count therefore gives the
    unique maximum-symmetry runtime scalar supported by these artifacts.
    """
    placement = manifest["placement_counts"]
    eligible = parse_nonnegative_manifest_count(
        placement["improvement_eligible_limit_orders"],
        "improvement_eligible_limit_orders",
    )
    inside = parse_nonnegative_manifest_count(
        placement["inside_spread_limit_orders"],
        "inside_spread_limit_orders",
    )
    if inside > eligible:
        raise ValueError("inside_spread_orders_exceed_eligible_orders")
    combined_zero = buy_distances.zero_count + sell_distances.zero_count
    if inside > combined_zero:
        raise ValueError("inside_spread_orders_exceed_combined_zero_distance_orders")
    if combined_zero == 0:
        if inside != 0:
            raise ValueError("positive_inside_count_without_zero_distance_orders")
        return 0.0
    probability = inside / combined_zero
    if not 0.0 <= probability <= 1.0:
        raise ValueError("invalid_quote_improvement_probability")
    return probability


def quote_quantity(limit_buy: DistributionStats,
                   limit_sell: DistributionStats,
                   fraction: float,
                   minimum: int,
                   maximum: int) -> int:
    if limit_buy.weighted_median is None or limit_sell.weighted_median is None:
        raise ValueError("limit_quantity_distribution_has_no_median")
    value = fraction * 0.5 * (
        limit_buy.weighted_median + limit_sell.weighted_median
    )
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("invalid_market_maker_quote_quantity")
    rounded = int(math.floor(value + 0.5))
    return max(minimum, min(maximum, rounded))


def expected_paths(data_root: pathlib.Path, compact: str, symbol: str) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    directory = data_root / f"itch_{compact}_{symbol.lower()}"
    manifest = directory / f"itch_manifest_{symbol.lower()}_{compact}.json"
    targets = directory / f"market_targets_{symbol.lower()}_{compact}.csv"
    return directory, manifest, targets


def rate_output_path(directory: pathlib.Path, compact: str, symbol: str, label: str) -> pathlib.Path:
    return directory / f"hawkes_rates_{symbol.lower()}_{label}_{compact}.csv"


def validate_symbol(symbol: str,
                    opening_rows: Mapping[str, Mapping[str, str]],
                    duplicate_catalog_symbols: set[str],
                    duplicate_opening_symbols: set[str],
                    data_root: pathlib.Path,
                    compact: str,
                    trading_date: str,
                    args: argparse.Namespace) -> tuple[PreparedSymbol | None, list[str]]:
    """Validate all inputs required by one simulated book without side effects."""
    reasons: list[str] = []
    if symbol in duplicate_catalog_symbols:
        reasons.append("duplicate_symbol_in_catalog")
    if symbol in duplicate_opening_symbols:
        reasons.append("duplicate_symbol_in_opening_bbo")
    opening: OpeningBbo | None = None
    raw_opening = opening_rows.get(symbol)
    if raw_opening is None:
        reasons.append("missing_opening_bbo")
    else:
        opening, opening_reasons = validate_opening(raw_opening)
        reasons.extend(opening_reasons)

    directory, manifest_path, target_path = expected_paths(data_root, compact, symbol)
    manifest: dict[str, Any] | None = None
    if not directory.is_dir():
        reasons.append("missing_symbol_extractor_directory")
    if not manifest_path.is_file():
        reasons.append("missing_manifest")
    else:
        manifest, manifest_reasons = load_manifest(
            manifest_path, symbol, trading_date
        )
        reasons.extend(manifest_reasons)

    quantity: dict[str, DistributionStats] = {}
    distances: dict[str, DistributionStats] = {}
    if directory.is_dir():
        for event in QUANTITY_EVENTS:
            path = directory / f"{event}_quantity_distribution.txt"
            if not path.is_file():
                reasons.append(f"missing_quantity_distribution:{event}")
                continue
            try:
                quantity[event] = distribution_stats(
                    path, "quantity", minimum_value=1
                )
            except (UniverseBuildError, ValueError) as error:
                reasons.append(f"invalid_quantity_distribution:{event}:{error}")
        for event in DISTANCE_EVENTS:
            path = directory / f"{event}_distance_distribution.txt"
            if not path.is_file():
                reasons.append(f"missing_distance_distribution:{event}")
                continue
            try:
                distances[event] = distribution_stats(
                    path, "distance_ticks", minimum_value=0
                )
            except (UniverseBuildError, ValueError) as error:
                reasons.append(f"invalid_distance_distribution:{event}:{error}")

    # The extractor writes these manifest counts from the same distributions.
    # Checking equality catches a mixed or partly overwritten symbol directory
    # before it becomes a supposedly real calibrated book.
    if manifest is not None and isinstance(
        manifest.get("distribution_observation_counts"), dict
    ):
        manifest_counts = manifest["distribution_observation_counts"]
        for event, stats in quantity.items():
            try:
                expected_count = parse_manifest_count(
                    manifest_counts.get(event), f"observed_count_{event}"
                )
            except ValueError:
                # The malformed-count reason was already added by load_manifest.
                continue
            if stats.total_count != expected_count:
                reasons.append(f"manifest_distribution_count_mismatch:{event}")
        for event, stats in distances.items():
            try:
                expected_count = parse_manifest_count(
                    manifest_counts.get(event), f"observed_count_{event}"
                )
            except ValueError:
                continue
            if stats.total_count != expected_count:
                reasons.append(
                    f"manifest_distance_distribution_count_mismatch:{event}"
                )

    target_spread: int | None = None
    if not target_path.is_file():
        reasons.append("missing_market_targets")
    else:
        target_spread, target_reasons = read_target_spread(target_path)
        reasons.extend(target_reasons)

    if args.balance_best_depth:
        for event in ("cancel_bid", "cancel_ask"):
            stats = distances.get(event)
            if stats is not None and stats.zero_fraction <= 0.0:
                reasons.append(f"missing_zero_distance_mass:{event}")

    if manifest is not None:
        try:
            probability = quote_improvement_probability(
                manifest, distances["limit_buy"], distances["limit_sell"]
            )
        except (KeyError, ValueError) as error:
            probability = math.nan
            reasons.append(f"invalid_quote_improvement_probability:{error}")
    else:
        probability = math.nan

    try:
        maker_quantity = quote_quantity(
            quantity["limit_buy"], quantity["limit_sell"],
            args.quote_quantity_fraction,
            args.minimum_quote_quantity,
            args.maximum_quote_quantity,
        )
    except (KeyError, ValueError) as error:
        maker_quantity = 0
        reasons.append(f"invalid_market_maker_quote_quantity:{error}")

    if reasons or opening is None or manifest is None or target_spread is None:
        return None, sorted(set(reasons))
    return PreparedSymbol(
        symbol=symbol,
        data_dir=directory,
        manifest_path=manifest_path,
        manifest=manifest,
        opening=opening,
        quantity_stats=quantity,
        target_spread_ticks=target_spread,
        quote_improvement_probability=probability,
        market_maker_quote_quantity=maker_quantity,
        rate_path=rate_output_path(
            directory, compact, symbol, args.rate_label
        ),
    ), []


def expected_rate_derivation(
    manifest_path: pathlib.Path, *, balance_directional_volume: bool,
    balance_best_depth: bool, balance_strength: float,
) -> tuple[int, list[float], list[float]]:
    """Recompute observed and transformed targets independently of the CSV."""
    try:
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise UniverseBuildError(
            f"cannot read rate-derivation manifest {manifest_path}: {error}"
        ) from error
    try:
        if "aggregation_duration_seconds" in manifest:
            duration_value = parse_finite_float(
                str(manifest["aggregation_duration_seconds"]),
                "aggregation_duration_seconds",
            )
            if not duration_value.is_integer():
                raise ValueError("nonintegral aggregation_duration_seconds")
            duration = int(duration_value)
        else:
            start = dt.time.fromisoformat(str(manifest["session_start"]))
            end = dt.time.fromisoformat(str(manifest["session_end"]))
            start_seconds = 3600 * start.hour + 60 * start.minute + start.second
            end_seconds = 3600 * end.hour + 60 * end.minute + end.second
            duration = end_seconds - start_seconds
        counts = manifest["distribution_observation_counts"]
    except (KeyError, TypeError, ValueError) as error:
        raise UniverseBuildError(
            f"invalid rate-derivation clock/counts in {manifest_path}"
        ) from error
    if duration <= 0 or not isinstance(counts, dict):
        raise UniverseBuildError(
            f"invalid rate-derivation clock/counts in {manifest_path}"
        )
    observed = [
        parse_manifest_count(counts.get(event), f"observed_count_{event}")
        / duration
        for event in QUANTITY_EVENTS
    ]
    directory = manifest_path.parent
    quantity_means = [
        distribution_stats(
            directory / f"{event}_quantity_distribution.txt",
            "quantity", minimum_value=1,
        ).mean
        for event in QUANTITY_EVENTS
    ]
    directional = list(observed)
    if balance_directional_volume:
        for left, right in ((0, 1), (2, 3), (4, 5)):
            total_rate = observed[left] + observed[right]
            total_mean = quantity_means[left] + quantity_means[right]
            if total_rate <= 0.0 or total_mean <= 0.0:
                directional[left] = 0.0
                directional[right] = 0.0
            else:
                directional[left] = total_rate * quantity_means[right] / total_mean
                directional[right] = total_rate * quantity_means[left] / total_mean
    if not balance_best_depth:
        return duration, observed, directional
    distance_zero = {
        event: distribution_stats(
            directory / f"{event}_distance_distribution.txt",
            "distance_ticks", minimum_value=0,
        ).zero_fraction
        for event in DISTANCE_EVENTS
    }
    bid_denominator = distance_zero["cancel_bid"] * quantity_means[4]
    ask_denominator = distance_zero["cancel_ask"] * quantity_means[5]
    if bid_denominator <= 0.0 or ask_denominator <= 0.0:
        raise UniverseBuildError(
            f"rate-derivation best-depth transform has zero cancellation "
            f"support below {directory}"
        )
    balanced = list(directional)
    balanced[4] = max(
        0.0,
        (
            directional[0] * distance_zero["limit_buy"] * quantity_means[0]
            - directional[3] * quantity_means[3]
        ) / bid_denominator,
    )
    balanced[5] = max(
        0.0,
        (
            directional[1] * distance_zero["limit_sell"] * quantity_means[1]
            - directional[2] * quantity_means[2]
        ) / ask_denominator,
    )
    targets = [
        original + balance_strength * (adjusted - original)
        for original, adjusted in zip(directional, balanced)
    ]
    return duration, observed, targets


def validate_generated_rates(
    path: pathlib.Path, *, label: str, manifest_path: pathlib.Path,
    activity_scale: float,
    kernel_beta: float, balance_directional_volume: bool,
    balance_best_depth: bool, balance_strength: float,
) -> dict[str, object]:
    """Validate the published columns and stationary Hawkes reconstruction."""
    headers, rows = read_csv(path)
    required = {
        "event_type", "observed_rate_per_second", "stationary_target_rate",
        "configured_mu", "stationary_reconstructed_rate",
    }
    missing_fields = sorted(required.difference(headers))
    if missing_fields:
        raise UniverseBuildError(
            "generated Hawkes file is missing: " + ", ".join(missing_fields)
        )
    event_names = [get_field(row, headers, "event_type") for row in rows]
    if event_names != list(QUANTITY_EVENTS):
        raise UniverseBuildError(
            f"{label} generated Hawkes events have the wrong order: {event_names}"
        )
    duration, expected_observed, expected_targets = expected_rate_derivation(
        manifest_path,
        balance_directional_volume=balance_directional_volume,
        balance_best_depth=balance_best_depth,
        balance_strength=balance_strength,
    )
    maximum_observed_error = 0.0
    maximum_target_error = 0.0
    maximum_reconstruction_error = 0.0
    maximum_reported_reconstruction_error = 0.0
    if activity_scale <= 0.0 or kernel_beta <= 0.0:
        raise UniverseBuildError(f"{label} has invalid Hawkes inversion settings")
    for index, (row, event) in enumerate(zip(rows, event_names)):
        values = {
            name: parse_finite_float(get_field(row, headers, name), name)
            for name in required if name != "event_type"
        }
        if min(values.values()) < 0.0:
            raise UniverseBuildError(
                f"{label} has a negative generated Hawkes rate for {event}"
            )
        target = values["stationary_target_rate"]
        reconstructed = values["stationary_reconstructed_rate"]
        observed = values["observed_rate_per_second"]
        configured_mu = values["configured_mu"]
        observed_error = abs(observed - expected_observed[index])
        target_error = abs(target - expected_targets[index])
        alpha = hawkes.default_alpha()
        endogenous = sum(
            alpha[index][column] * float(
                rows[column][headers["stationary_target_rate"]]
            ) / kernel_beta
            for column in range(len(QUANTITY_EVENTS))
        )
        computed_reconstruction = activity_scale * configured_mu + endogenous
        reported_reconstruction_error = abs(
            reconstructed - computed_reconstruction
        )
        reconstruction_error = abs(computed_reconstruction - target)
        maximum_observed_error = max(maximum_observed_error, observed_error)
        maximum_target_error = max(maximum_target_error, target_error)
        maximum_reconstruction_error = max(
            maximum_reconstruction_error, reconstruction_error
        )
        maximum_reported_reconstruction_error = max(
            maximum_reported_reconstruction_error,
            reported_reconstruction_error,
        )
        if not math.isclose(
                observed, expected_observed[index],
                rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise UniverseBuildError(
                f"{label} observed rate disagrees with manifest count/duration "
                f"for {event}"
            )
        if not math.isclose(
                target, expected_targets[index],
                rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise UniverseBuildError(
                f"{label} stationary target disagrees with the declared "
                f"reduced-book transforms for {event}"
            )
        if not math.isclose(
                reconstructed, computed_reconstruction,
                rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise UniverseBuildError(
                f"{label} reported stationary reconstruction disagrees with "
                f"configured_mu for {event}"
            )
        if not math.isclose(
                computed_reconstruction, target,
                rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise UniverseBuildError(
                f"{label} Hawkes inversion cannot reconstruct the stationary "
                f"target for {event}: target={target:.17g}, "
                f"reconstructed={reconstructed:.17g}"
            )
    return {
        "schema_version": 1,
        "status": "passed",
        "event_types_checked": len(rows),
        "manifest_duration_seconds": duration,
        "maximum_absolute_observed_rate_error": maximum_observed_error,
        "observed_rates_equal_manifest_counts_per_duration": True,
        "maximum_absolute_stationary_target_error": maximum_target_error,
        "stationary_targets_equal_declared_transforms_per_type": True,
        "maximum_absolute_reported_reconstruction_error": (
            maximum_reported_reconstruction_error
        ),
        "reported_reconstruction_equals_configured_rate_equation_per_type": True,
        "maximum_absolute_stationary_reconstruction_error": (
            maximum_reconstruction_error
        ),
        "relative_tolerance": 1.0e-12,
        "absolute_tolerance": 1.0e-12,
        "stationary_reconstruction_equals_target_per_type": True,
        "transform_settings": {
            "activity_scale": activity_scale,
            "kernel_beta": kernel_beta,
            "balance_directional_volume": balance_directional_volume,
            "balance_best_depth": balance_best_depth,
            "balance_strength": balance_strength,
            **hawkes.excitation_settings(),
        },
    }


def derive_rates(prepared: PreparedSymbol, args: argparse.Namespace) -> list[dict[str, float | str]]:
    """Call the established first-stage calibration in process and atomically publish it."""
    output = prepared.rate_path
    if output.exists() and not args.overwrite:
        raise UniverseBuildError(f"Hawkes-rate output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    # derive_hawkes_rates opens the output itself, so remove the placeholder.
    temporary.unlink()
    try:
        rows = hawkes.run(argparse.Namespace(
            manifest=str(prepared.manifest_path),
            output=str(temporary),
            activity_scale=args.activity_scale,
            beta=args.hawkes_beta,
            balance_directional_volume=args.balance_directional_volume,
            balance_best_depth=args.balance_best_depth,
            balance_strength=args.balance_strength,
        ))
        rate_audit = validate_generated_rates(
            temporary, label=prepared.symbol,
            manifest_path=prepared.manifest_path,
            activity_scale=args.activity_scale,
            kernel_beta=args.hawkes_beta,
            balance_directional_volume=args.balance_directional_volume,
            balance_best_depth=args.balance_best_depth,
            balance_strength=args.balance_strength,
        )
        os.replace(temporary, output)
        prepared.rate_derivation = {
            **rate_audit,
            "manifest": {
                "path": str(prepared.manifest_path.resolve()),
                "sha256": sha256_file(prepared.manifest_path),
            },
            "generated_hawkes_rates": {
                "path": str(output.resolve()),
                "sha256": sha256_file(output),
            },
        }
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return rows


def portable_path(path: pathlib.Path,
                  path_root: pathlib.Path,
                  absolute_paths: bool = False) -> str:
    """Use project-relative paths where possible, unless absolute paths are requested."""
    resolved = path.resolve()
    if absolute_paths:
        return str(resolved)
    try:
        return str(resolved.relative_to(path_root.resolve()))
    except ValueError:
        return str(resolved)


def atomic_csv(path: pathlib.Path,
               fieldnames: Sequence[str],
               rows: Sequence[Mapping[str, object]],
               overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise UniverseBuildError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: pathlib.Path, value: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise UniverseBuildError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def qqq_first(symbols: Iterable[str]) -> list[str]:
    return sorted(symbols, key=lambda symbol: (symbol != "QQQ", symbol))


def build_provenance_base(args: argparse.Namespace,
                          catalog_path: pathlib.Path | None,
                          opening_path: pathlib.Path,
                          data_root: pathlib.Path,
                          compact: str,
                          input_issues: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "builder": {
            "script": str(pathlib.Path(__file__).resolve()),
            "script_sha256": sha256_file(pathlib.Path(__file__).resolve()),
        },
        "inputs": {
            "trading_date": args.trading_date,
            "trading_date_compact": compact,
            "data_root": str(data_root.resolve()),
            "catalog": (
                {"path": str(catalog_path.resolve()), "sha256": sha256_file(catalog_path)}
                if catalog_path is not None else None
            ),
            "opening_bbo": {
                "path": str(opening_path.resolve()),
                "sha256": sha256_file(opening_path),
            },
            "input_issues": list(input_issues),
        },
        "hawkes_derivation": {
            "implementation": "scripts/derive_hawkes_rates.py::run",
            "activity_scale": args.activity_scale,
            "kernel_beta": args.hawkes_beta,
            "balance_directional_volume": args.balance_directional_volume,
            "balance_best_depth": args.balance_best_depth,
            "balance_strength": args.balance_strength,
            "rate_label": args.rate_label,
        },
        "configuration_policy": {
            "ordering": "QQQ first, then lexicographic symbol order",
            "basket_weight": 0.0,
            "beta": "opening_mid_price_ticks / cross_sectional_median_opening_mid_price_ticks",
            "target_spread_ticks": "rounded empirical mean_spread_ticks",
            "quote_improvement_probability": (
                "aggregate inside-spread additions divided by the combined "
                "buy/sell distance-zero addition count; one maximum-symmetry "
                "zero split because compact ITCH artifacts do not identify "
                "the side/state joint allocation"
            ),
            "market_maker_quote_quantity": {
                "source": "0.5 * (weighted median limit-buy mark + weighted median limit-sell mark)",
                "fraction": args.quote_quantity_fraction,
                "minimum": args.minimum_quote_quantity,
                "maximum": args.maximum_quote_quantity,
            },
        },
    }


def rate_row_summary(rows: Sequence[Mapping[str, float | str]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        event = str(row["event_type"])
        result[event] = {
            "observed_rate_per_second": float(row["observed_rate_per_second"]),
            "stationary_target_rate": float(row["stationary_target_rate"]),
            "configured_mu": float(row["configured_mu"]),
        }
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Build outputs and return a compact machine-readable summary."""
    compact = compact_date(args.trading_date)
    data_root = pathlib.Path(args.data_root).resolve()
    catalog_path = pathlib.Path(args.catalog).resolve() if args.catalog else None
    opening_path = pathlib.Path(args.opening_bbo).resolve() if args.opening_bbo else catalog_path
    if opening_path is None:
        raise UniverseBuildError("provide --catalog, --opening-bbo, or both")
    if not data_root.is_dir():
        raise UniverseBuildError(f"--data-root is not a directory: {data_root}")

    input_issues: list[str] = []
    if catalog_path is not None:
        catalog_headers, catalog_rows = read_csv(catalog_path)
        catalog_symbols, catalog_duplicates, catalog_issues = first_unique_symbols(
            catalog_rows, catalog_headers, "catalog"
        )
        input_issues.extend(catalog_issues)
    else:
        catalog_symbols = []
        catalog_duplicates = set()

    openings, opening_duplicates, opening_issues = read_openings(opening_path)
    input_issues.extend(opening_issues)
    if catalog_path is None:
        candidate_symbols = list(openings)
        catalog_duplicates = set()
    else:
        candidate_symbols = catalog_symbols
    if not candidate_symbols:
        raise UniverseBuildError("candidate catalog contains no valid symbols")

    provenance = build_provenance_base(
        args, catalog_path, opening_path, data_root, compact, input_issues
    )
    rejected: list[dict[str, Any]] = []
    prepared_symbols: list[PreparedSymbol] = []
    for symbol in qqq_first(candidate_symbols):
        prepared, reasons = validate_symbol(
            symbol,
            openings,
            catalog_duplicates,
            opening_duplicates,
            data_root,
            compact,
            args.trading_date,
            args,
        )
        if prepared is None:
            rejected.append({"symbol": symbol, "reasons": reasons})
        else:
            prepared_symbols.append(prepared)

    # Run the repository's established calibration routine only after static
    # validation.  A single numerical failure is isolated to that symbol and
    # retained in the provenance rather than aborting the entire universe.
    accepted: list[PreparedSymbol] = []
    for prepared in prepared_symbols:
        try:
            prepared.hawkes_rows = derive_rates(prepared, args)
        except Exception as error:
            rejected.append({
                "symbol": prepared.symbol,
                "reasons": [f"hawkes_derivation_failed:{type(error).__name__}:{error}"],
            })
            continue
        accepted.append(prepared)

    accepted.sort(key=lambda item: (item.symbol != "QQQ", item.symbol))
    accepted_symbols = {item.symbol for item in accepted}
    provenance["candidate_count"] = len(candidate_symbols)
    provenance["accepted_count"] = len(accepted)
    provenance["rejected_count"] = len(rejected)
    provenance["rejected_symbols"] = sorted(rejected, key=lambda item: item["symbol"])

    provenance_path = pathlib.Path(args.provenance).resolve()
    if "QQQ" not in accepted_symbols:
        provenance["status"] = "failed"
        provenance["failure"] = "QQQ was not accepted; a QQQ-first configuration cannot be written"
        atomic_json(provenance_path, provenance, args.overwrite)
        raise UniverseBuildError(
            f"QQQ was not accepted; inspect provenance at {provenance_path}"
        )
    if not accepted:
        provenance["status"] = "failed"
        provenance["failure"] = "no symbols passed validation"
        atomic_json(provenance_path, provenance, args.overwrite)
        raise UniverseBuildError("no symbols passed validation")

    median_opening_price = float(statistics.median(
        item.opening.mid_ticks for item in accepted
    ))
    if not math.isfinite(median_opening_price) or median_opening_price <= 0.0:
        raise UniverseBuildError("invalid cross-sectional median opening price")
    path_root = pathlib.Path(args.path_root).resolve()
    config_rows: list[dict[str, object]] = []
    accepted_provenance: list[dict[str, Any]] = []
    for book_id, item in enumerate(accepted):
        beta = item.opening.mid_ticks / median_opening_price
        if not math.isfinite(beta) or beta <= 0.0:
            raise UniverseBuildError(f"invalid price-ratio beta for {item.symbol}")
        config_rows.append({
            "book_id": book_id,
            "symbol": item.symbol,
            "data_dir": portable_path(
                item.data_dir, path_root, getattr(args, "absolute_paths", False)
            ),
            "hawkes_rates_file": portable_path(
                item.rate_path, path_root, getattr(args, "absolute_paths", False)
            ),
            "fundamental_price_ticks": item.opening.mid_ticks,
            "initial_best_bid_ticks": item.opening.bid_ticks,
            "initial_best_ask_ticks": item.opening.ask_ticks,
            "initial_best_bid_depth": item.opening.bid_depth,
            "initial_best_ask_depth": item.opening.ask_depth,
            "beta": beta,
            "basket_weight": 0.0,
            "market_maker_quote_quantity": item.market_maker_quote_quantity,
            "target_spread_ticks": item.target_spread_ticks,
            "quote_improvement_probability": item.quote_improvement_probability,
        })
        accepted_provenance.append({
            "book_id": book_id,
            "symbol": item.symbol,
            "data_dir": str(item.data_dir.resolve()),
            "manifest": {
                "path": str(item.manifest_path.resolve()),
                "sha256": sha256_file(item.manifest_path),
                "source_input_path": item.manifest.get("input_path"),
                "source_input_sha256": item.manifest.get("input_sha256"),
                "stock_locate": item.manifest.get("stock_locate"),
            },
            "opening_bbo": {
                "best_bid_ticks": item.opening.bid_ticks,
                "best_ask_ticks": item.opening.ask_ticks,
                "best_bid_depth": item.opening.bid_depth,
                "best_ask_depth": item.opening.ask_depth,
                "mid_price_ticks": item.opening.mid_ticks,
            },
            "beta": beta,
            "basket_weight": 0.0,
            "market_maker_quote_quantity": item.market_maker_quote_quantity,
            "target_spread_ticks": item.target_spread_ticks,
            "quote_improvement_probability": item.quote_improvement_probability,
            "hawkes_rates_file": str(item.rate_path.resolve()),
            "hawkes_rates_sha256": sha256_file(item.rate_path),
            "hawkes_event_rates": rate_row_summary(item.hawkes_rows or []),
            "rate_derivation": item.rate_derivation,
        })

    output_path = pathlib.Path(args.output).resolve()
    atomic_csv(output_path, CONFIG_FIELDS, config_rows, args.overwrite)
    provenance.update({
        "status": "complete",
        "cross_sectional_median_opening_price_ticks": median_opening_price,
        "configuration": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "path_root": str(path_root),
            "absolute_paths": bool(getattr(args, "absolute_paths", False)),
            "fieldnames": list(CONFIG_FIELDS),
        },
        "accepted_symbols": accepted_provenance,
    })
    atomic_json(provenance_path, provenance, args.overwrite)
    return {
        "output": str(output_path),
        "provenance": str(provenance_path),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "cross_sectional_median_opening_price_ticks": median_opening_price,
        "qqq_book_id": 0,
    }


def write_fixture_csv(path: pathlib.Path,
                      fields: Sequence[str],
                      rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_self_test_symbol(root: pathlib.Path, compact: str, symbol: str) -> None:
    """Create a minimal valid extractor fixture for ``--self-test``."""
    directory = root / f"itch_{compact}_{symbol.lower()}"
    directory.mkdir(parents=True, exist_ok=True)
    for index, event in enumerate(QUANTITY_EVENTS, start=1):
        quantities = (
            (1, 2) if event in {"market_buy", "market_sell"}
            else (10 + index, 20 + index)
        )
        write_fixture_csv(
            directory / f"{event}_quantity_distribution.txt",
            ["quantity", "count"],
            [
                {"quantity": quantities[0], "count": 5},
                {"quantity": quantities[1], "count": 5},
            ],
        )
    for event in DISTANCE_EVENTS:
        write_fixture_csv(
            directory / f"{event}_distance_distribution.txt",
            ["distance_ticks", "count"],
            [{"distance_ticks": 0, "count": 8}, {"distance_ticks": 1, "count": 2}],
        )
    write_fixture_csv(
        directory / f"market_targets_{symbol.lower()}_{compact}.csv",
        ["name", "target", "scale", "weight"],
        [{"name": "mean_spread_ticks", "target": 2.0, "scale": 1.0, "weight": 1.0}],
    )
    manifest = {
        "symbol": symbol,
        "trading_date": "2020-01-30",
        "session_start": "09:30:00",
        "session_end": "16:00:00",
        "distribution_observation_counts": {
            event: 10 for event in QUANTITY_EVENTS
        },
        "placement_counts": {
            "improvement_eligible_limit_orders": 10,
            "inside_spread_limit_orders": 2,
        },
    }
    with (directory / f"itch_manifest_{symbol.lower()}_{compact}.json").open("w") as output:
        json.dump(manifest, output)


def run_self_test() -> None:
    """A compact end-to-end regression test requiring no real ITCH file."""
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        data_root = root / "data"
        compact = "20200130"
        write_self_test_symbol(data_root, compact, "QQQ")
        write_self_test_symbol(data_root, compact, "AAPL")
        write_self_test_symbol(data_root, compact, "EMPTY")
        (data_root / f"itch_{compact}_empty" / "market_buy_quantity_distribution.txt").unlink()
        # BAD has no extractor directory; EMPTY verifies the required six
        # quantity distributions are not silently treated as zero activity.
        catalog = root / "catalog.csv"
        write_fixture_csv(catalog, ["symbol"], [
            {"symbol": "AAPL"}, {"symbol": "BAD"}, {"symbol": "EMPTY"},
            {"symbol": "QQQ"},
        ])
        opening = root / "opening.csv"
        write_fixture_csv(opening, list(OPENING_COLUMNS), [
            {"symbol": "QQQ", "best_bid_ticks": 1999, "best_ask_ticks": 2001,
             "best_bid_depth": 100, "best_ask_depth": 100, "mid_price_ticks": 2000.0},
            {"symbol": "AAPL", "best_bid_ticks": 2999, "best_ask_ticks": 3001,
             "best_bid_depth": 100, "best_ask_depth": 100, "mid_price_ticks": 3000.0},
            {"symbol": "BAD", "best_bid_ticks": 99, "best_ask_ticks": 101,
             "best_bid_depth": 100, "best_ask_depth": 100, "mid_price_ticks": 100.0},
            {"symbol": "EMPTY", "best_bid_ticks": 499, "best_ask_ticks": 501,
             "best_bid_depth": 100, "best_ask_depth": 100, "mid_price_ticks": 500.0},
        ])
        result = build(argparse.Namespace(
            data_root=str(data_root), trading_date="2020-01-30",
            catalog=str(catalog), opening_bbo=str(opening),
            output=str(root / "config.csv"), provenance=str(root / "provenance.json"),
            path_root=str(root), activity_scale=0.30, hawkes_beta=10.0,
            balance_directional_volume=True, balance_best_depth=True,
            balance_strength=1.0, rate_label="universe_balanced",
            quote_quantity_fraction=0.5, minimum_quote_quantity=10,
            maximum_quote_quantity=1000, overwrite=False,
        ))
        if result["accepted_count"] != 2 or result["rejected_count"] != 2:
            raise AssertionError(f"unexpected self-test result: {result}")
        _, config_rows = read_csv(root / "config.csv")
        if [row["symbol"] for row in config_rows] != ["QQQ", "AAPL"]:
            raise AssertionError("QQQ-first deterministic ordering failed")
        if any(float(row["basket_weight"]) != 0.0 for row in config_rows):
            raise AssertionError("basket weights must be zero")
        if abs(float(config_rows[0]["beta"]) - 0.8) > 1e-12:
            raise AssertionError("cross-sectional price-ratio beta failed")
        if abs(float(config_rows[0]["quote_improvement_probability"]) - 0.125) > 1e-12:
            raise AssertionError("aggregate quote-improvement zero split failed")
        with (root / "provenance.json").open() as source:
            provenance = json.load(source)
        for record in provenance["accepted_symbols"]:
            audit = record.get("rate_derivation")
            if not isinstance(audit, dict) or audit.get("status") != "passed":
                raise AssertionError("rate derivation audit was not recorded")
            if audit.get("transform_settings") != {
                "activity_scale": 0.3,
                "kernel_beta": 10.0,
                "balance_directional_volume": True,
                "balance_best_depth": True,
                "balance_strength": 1.0,
                **hawkes.excitation_settings(),
            }:
                raise AssertionError("balanced rate settings were not recorded")
            for artifact in ("manifest", "generated_hawkes_rates"):
                artifact_record = audit.get(artifact)
                if (not isinstance(artifact_record, dict)
                        or sha256_file(pathlib.Path(artifact_record["path"]))
                            != artifact_record.get("sha256")):
                    raise AssertionError(
                        f"rate derivation {artifact} hash was not recorded"
                    )
        rejected = provenance["rejected_symbols"]
        rejected_by_symbol = {row["symbol"]: set(row["reasons"]) for row in rejected}
        if set(rejected_by_symbol) != {"BAD", "EMPTY"}:
            raise AssertionError("explicit rejection provenance failed")
        required_reasons = {
            "missing_symbol_extractor_directory",
            "missing_manifest",
            "missing_market_targets",
        }
        if not required_reasons.issubset(rejected_by_symbol["BAD"]):
            raise AssertionError("explicit rejection provenance failed")
        if "missing_quantity_distribution:market_buy" not in rejected_by_symbol["EMPTY"]:
            raise AssertionError("explicit rejection provenance failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="root containing itch_YYYYMMDD_symbol directories")
    parser.add_argument("--trading-date", help="ITCH trading date in YYYY-MM-DD form")
    parser.add_argument(
        "--catalog",
        help="candidate CSV with a symbol column; omit to use every opening-BBO row",
    )
    parser.add_argument(
        "--opening-bbo",
        help="CSV with symbol and calibrated opening BBO columns; defaults to --catalog",
    )
    parser.add_argument("--output", help="output MultiAssetBookConfig CSV")
    parser.add_argument("--provenance", help="output JSON audit/provenance file")
    parser.add_argument(
        "--path-root",
        default=str(pathlib.Path.cwd()),
        help="make config data paths relative to this directory when possible",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help=(
            "write absolute data_dir and hawkes_rates_file paths; use this for "
            "a Slurm result directory that will be invoked from another cwd"
        ),
    )
    parser.add_argument("--activity-scale", type=float, default=0.30)
    parser.add_argument("--hawkes-beta", type=float, default=10.0)
    parser.add_argument("--balance-strength", type=float, default=1.0)
    parser.add_argument(
        "--balance-directional-volume", dest="balance_directional_volume",
        action="store_true",
        help="apply the reduced-book directional-volume rate transform",
    )
    parser.add_argument(
        "--no-balance-directional-volume", dest="balance_directional_volume",
        action="store_false", help="disable the reduced-book directional-volume correction",
    )
    parser.add_argument(
        "--balance-best-depth", dest="balance_best_depth", action="store_true",
        help="apply the reduced-book best-depth cancellation-rate transform",
    )
    parser.add_argument(
        "--no-balance-best-depth", dest="balance_best_depth", action="store_false",
        help="disable the reduced-book best-depth cancellation correction",
    )
    parser.set_defaults(balance_directional_volume=True, balance_best_depth=True)
    parser.add_argument(
        "--rate-label", default="universe_balanced",
        help="label embedded in generated per-symbol Hawkes-rate filenames",
    )
    parser.add_argument("--quote-quantity-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-quote-quantity", type=int, default=10)
    parser.add_argument("--maximum-quote-quantity", type=int, default=1000)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="allow replacement of generated Hawkes files, config, and provenance",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="run the compact end-to-end fixture test and exit",
    )
    return parser


def validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    required = ("data_root", "trading_date", "output", "provenance")
    missing = [f"--{name.replace('_', '-')}" for name in required if not getattr(args, name)]
    if missing:
        parser.error("required unless --self-test: " + ", ".join(missing))
    if not args.catalog and not args.opening_bbo:
        parser.error("provide --catalog, --opening-bbo, or both")
    if not math.isfinite(args.activity_scale) or args.activity_scale <= 0.0:
        parser.error("--activity-scale must be finite and positive")
    if not math.isfinite(args.hawkes_beta) or args.hawkes_beta <= 0.0:
        parser.error("--hawkes-beta must be finite and positive")
    if not math.isfinite(args.balance_strength) or not 0.0 <= args.balance_strength <= 5.0:
        parser.error("--balance-strength must be finite and between 0 and 5")
    if not math.isfinite(args.quote_quantity_fraction) or args.quote_quantity_fraction <= 0.0:
        parser.error("--quote-quantity-fraction must be finite and positive")
    if not 1 <= args.minimum_quote_quantity <= args.maximum_quote_quantity <= MAX_INT32:
        parser.error("quote-quantity bounds must satisfy 1 <= minimum <= maximum <= INT32_MAX")
    if not args.rate_label or any(character in args.rate_label for character in "/\\"):
        parser.error("--rate-label must be a nonempty filename component")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        print("build_itch_universe_config self-test: PASS")
        return 0
    validate_arguments(args, parser)
    try:
        summary = build(args)
    except UniverseBuildError as error:
        print(f"ITCH universe configuration failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
