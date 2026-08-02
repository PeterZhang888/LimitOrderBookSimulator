#!/usr/bin/env python3
"""Fit training-only queue-reactive Hawkes policies by liquidity cluster.

The fitter consumes sufficient statistics emitted by
``extract_itch50_symbols.py --state-targets-csv``.  It never reads a held-out
market-target file.  Event-time seasonality and pre-event queue response are
estimated separately; marginal rates remain frozen at the rates in the pooled
training configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


EVENT_TYPES = (
    "limit_buy",
    "limit_sell",
    "market_buy",
    "market_sell",
    "cancel_bid",
    "cancel_ask",
)
EVENT_INDEX = {name: index for index, name in enumerate(EVENT_TYPES)}
HALF_HOUR_BINS = 13
LAG_SECONDS = (0, 1, 2, 5, 10, 20, 30)
FEATURE_NAMES = (
    "log_spread_ratio",
    "log_bid_depth_ratio",
    "log_ask_depth_ratio",
    "queue_imbalance",
)
SPREAD_VALUE = {"one_tick": 0.0, "wider": math.log(2.0)}
IMBALANCE_VALUE = {
    "sell_very_high": -0.8,
    "sell_high": -0.4,
    "balanced": 0.0,
    "buy_high": 0.4,
    "buy_very_high": 0.8,
}
DEPTH_VALUE = {
    "low": math.log(0.35),
    "typical": 0.0,
    "high": math.log(2.5),
}


class PolicyFitError(RuntimeError):
    """Raised when a policy cannot be estimated without silent repair."""


@dataclass(frozen=True)
class TrainingRoot:
    date: str
    root: pathlib.Path


@dataclass(frozen=True)
class StateFitSource:
    date: str
    symbol: str
    counts: dict[tuple[int, str, str, str, str, str], int]
    exposure: dict[tuple[int, str, str, str, str], float]
    base_rates: tuple[float, ...]


@dataclass(frozen=True)
class ImprovementLoadResult:
    distribution: Counter[int]
    raw_count: int
    runtime_compatible_count: int
    excluded_off_grid_count: int


@dataclass
class ClusterAccumulator:
    members: list[str]
    estimation_members: list[str]
    intraday_counts: list[list[int]]
    state_fit_sources: list[StateFitSource]
    improvements: dict[str, Counter[int]]
    improvement_projection: dict[str, Counter[str]]
    lag_moments: dict[tuple[str, str, int], "LagMomentAggregate"]
    sources: list[dict[str, object]]


@dataclass
class LagMomentAggregate:
    paired_bins: int = 0
    weighted_source_mean: float = 0.0
    weighted_target_mean: float = 0.0
    weighted_source_variance: float = 0.0
    weighted_target_variance: float = 0.0
    weighted_covariance: float = 0.0
    contributions: int = 0

    def add(
        self,
        paired_bins: int,
        source_mean: float,
        target_mean: float,
        source_variance: float,
        target_variance: float,
        covariance: float,
    ) -> None:
        self.paired_bins += paired_bins
        self.weighted_source_mean += paired_bins * source_mean
        self.weighted_target_mean += paired_bins * target_mean
        self.weighted_source_variance += paired_bins * source_variance
        self.weighted_target_variance += paired_bins * target_variance
        self.weighted_covariance += paired_bins * covariance
        self.contributions += 1

    def summary(self) -> dict[str, float | int]:
        if self.paired_bins <= 0 or self.contributions <= 0:
            raise PolicyFitError("empty lag-moment aggregate")
        source_variance = self.weighted_source_variance / self.paired_bins
        target_variance = self.weighted_target_variance / self.paired_bins
        covariance = self.weighted_covariance / self.paired_bins
        denominator = math.sqrt(source_variance * target_variance)
        correlation = covariance / denominator if denominator > 0.0 else 0.0
        return {
            "paired_bins": self.paired_bins,
            "contributions": self.contributions,
            "source_mean_count": self.weighted_source_mean / self.paired_bins,
            "target_mean_count": self.weighted_target_mean / self.paired_bins,
            "source_variance": source_variance,
            "target_variance": target_variance,
            "covariance": covariance,
            "correlation": max(-1.0, min(1.0, correlation)),
        }


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    except OSError as error:
        raise PolicyFitError(f"cannot read required artifact {path}: {error}") from error
    if not rows:
        raise PolicyFitError(f"required artifact is empty: {path}")
    return rows


def require_columns(
    rows: Sequence[Mapping[str, str]], required: Iterable[str], path: pathlib.Path
) -> None:
    observed = set(rows[0])
    missing = sorted(set(required) - observed)
    if missing:
        raise PolicyFitError(f"{path} lacks required columns: {', '.join(missing)}")


def parse_nonnegative_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise PolicyFitError(f"{label} is not an integer: {value!r}") from error
    if parsed < 0:
        raise PolicyFitError(f"{label} must be nonnegative")
    return parsed


def parse_positive_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise PolicyFitError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise PolicyFitError(f"{label} must be finite and positive")
    return parsed


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_training_root(value: str) -> TrainingRoot:
    if "=" not in value:
        raise argparse.ArgumentTypeError("training roots must be DATE=PATH")
    date, raw_root = value.split("=", 1)
    date = date.strip()
    if len(date) != 10 or date[4] != "-" or date[7] != "-":
        raise argparse.ArgumentTypeError(f"invalid ISO training date: {date!r}")
    root = pathlib.Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise argparse.ArgumentTypeError(f"training root is not a directory: {root}")
    return TrainingRoot(date, root)


def symbol_artifact_dir(root: TrainingRoot, symbol: str) -> pathlib.Path:
    compact = root.date.replace("-", "")
    dirname = f"itch_{compact}_{symbol.lower()}"
    candidates = (
        root.root / dirname,
        root.root / "empirical_data" / dirname,
    )
    found = [path for path in candidates if path.is_dir()]
    if len(found) != 1:
        raise PolicyFitError(
            f"expected exactly one artifact directory for {symbol} on {root.date}; "
            f"checked {', '.join(str(path) for path in candidates)}"
        )
    return found[0]


def load_symbol_map(path: pathlib.Path, value_column: str) -> dict[str, str]:
    rows = read_csv(path)
    require_columns(rows, ("symbol", value_column), path)
    result: dict[str, str] = {}
    for row in rows:
        symbol = row["symbol"].strip().upper()
        if not symbol or symbol in result:
            raise PolicyFitError(f"blank or duplicate symbol in {path}: {symbol!r}")
        result[symbol] = row[value_column].strip()
    return result


def load_pooled_config(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    except OSError as error:
        raise PolicyFitError(f"cannot read pooled config {path}: {error}") from error
    if not rows:
        raise PolicyFitError(f"pooled config is empty: {path}")
    require_columns(rows, ("book_id", "symbol", "hawkes_rates_file"), path)
    seen: set[str] = set()
    for expected_id, row in enumerate(rows):
        if parse_nonnegative_int(row["book_id"], "book_id") != expected_id:
            raise PolicyFitError("pooled config book_id values must be contiguous from zero")
        symbol = row["symbol"].strip().upper()
        if not symbol or symbol in seen:
            raise PolicyFitError(f"blank or duplicate pooled symbol: {symbol!r}")
        seen.add(symbol)
        row["symbol"] = symbol
    return fieldnames, rows


def load_target_rates(path: pathlib.Path) -> list[float]:
    rows = read_csv(path)
    require_columns(rows, ("event_type", "stationary_target_rate"), path)
    by_type: dict[str, float] = {}
    for row in rows:
        event_type = row["event_type"]
        if event_type not in EVENT_INDEX or event_type in by_type:
            raise PolicyFitError(f"unexpected or duplicate event type in {path}: {event_type!r}")
        try:
            value = float(row["stationary_target_rate"])
        except ValueError as error:
            raise PolicyFitError(f"invalid target rate in {path}") from error
        if not math.isfinite(value) or value < 0.0:
            raise PolicyFitError(f"target rates must be finite and nonnegative in {path}")
        by_type[event_type] = value
    if set(by_type) != set(EVENT_TYPES):
        raise PolicyFitError(f"{path} does not contain exactly the six event types")
    return [by_type[name] for name in EVENT_TYPES]


def load_intraday(path: pathlib.Path) -> tuple[list[list[int]], dict[str, int]]:
    rows = read_csv(path)
    require_columns(rows, ("half_hour_bin", "event_type", "count"), path)
    result = [[0 for _ in EVENT_TYPES] for _ in range(HALF_HOUR_BINS)]
    seen: set[tuple[int, str]] = set()
    totals = {name: 0 for name in EVENT_TYPES}
    for row in rows:
        index = parse_nonnegative_int(row["half_hour_bin"], "half_hour_bin")
        event_type = row["event_type"]
        if index >= HALF_HOUR_BINS or event_type not in EVENT_INDEX:
            raise PolicyFitError(f"invalid intraday cell in {path}")
        key = (index, event_type)
        if key in seen:
            raise PolicyFitError(f"duplicate intraday cell {key!r} in {path}")
        seen.add(key)
        count = parse_nonnegative_int(row["count"], "intraday count")
        result[index][EVENT_INDEX[event_type]] = count
        totals[event_type] += count
    expected = {(i, event_type) for i in range(HALF_HOUR_BINS) for event_type in EVENT_TYPES}
    if seen != expected:
        raise PolicyFitError(f"{path} must contain exactly 13 x 6 intraday cells")
    return result, totals


def load_state_exposure(
    path: pathlib.Path,
) -> dict[tuple[int, str, str, str, str], float]:
    rows = read_csv(path)
    columns = (
        "half_hour_bin", "spread_bin", "queue_imbalance_bin",
        "bid_depth_ratio_bin", "ask_depth_ratio_bin", "exposure_seconds",
    )
    require_columns(rows, columns, path)
    result: dict[tuple[int, str, str, str, str], float] = {}
    for row in rows:
        index = parse_nonnegative_int(row["half_hour_bin"], "half_hour_bin")
        if index >= HALF_HOUR_BINS:
            raise PolicyFitError(f"invalid exposure half-hour bin in {path}")
        key = (
            index, row["spread_bin"], row["queue_imbalance_bin"],
            row["bid_depth_ratio_bin"], row["ask_depth_ratio_bin"],
        )
        if key in result:
            raise PolicyFitError(f"duplicate exposure cell {key!r} in {path}")
        result[key] = parse_positive_float(row["exposure_seconds"], "exposure_seconds")
    return result


def load_state_counts(
    path: pathlib.Path,
) -> tuple[dict[tuple[int, str, str, str, str, str], int], dict[str, int]]:
    rows = read_csv(path)
    columns = (
        "half_hour_bin", "event_type", "spread_bin", "queue_imbalance_bin",
        "bid_depth_ratio_bin", "ask_depth_ratio_bin", "count",
    )
    require_columns(rows, columns, path)
    result: dict[tuple[int, str, str, str, str, str], int] = {}
    totals = {name: 0 for name in EVENT_TYPES}
    for row in rows:
        index = parse_nonnegative_int(row["half_hour_bin"], "half_hour_bin")
        event_type = row["event_type"]
        if index >= HALF_HOUR_BINS or event_type not in EVENT_INDEX:
            raise PolicyFitError(f"invalid state-count cell in {path}")
        key = (
            index, event_type, row["spread_bin"], row["queue_imbalance_bin"],
            row["bid_depth_ratio_bin"], row["ask_depth_ratio_bin"],
        )
        if key in result:
            raise PolicyFitError(f"duplicate state-count cell {key!r} in {path}")
        count = parse_nonnegative_int(row["count"], "state count")
        result[key] = count
        totals[event_type] += count
    return result, totals


def load_improvements(path: pathlib.Path) -> ImprovementLoadResult:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = set(reader.fieldnames or [])
            rows = list(reader)
    except OSError as error:
        raise PolicyFitError(f"cannot read required artifact {path}: {error}") from error
    missing = {"improvement_ticks", "improvement_price_units", "count"} - fields
    if missing:
        raise PolicyFitError(f"{path} lacks required columns: {', '.join(sorted(missing))}")
    result: Counter[int] = Counter()
    raw_count = 0
    compatible_count = 0
    for row in rows:
        distance = parse_nonnegative_int(
            row["improvement_price_units"], "improvement_price_units"
        )
        ticks = finite_float(row["improvement_ticks"], "improvement_ticks")
        count = parse_nonnegative_int(row["count"], "improvement count")
        if distance <= 0 or count <= 0:
            raise PolicyFitError(f"improvement distances and counts must be positive in {path}")
        if abs(ticks - distance / 100.0) > 1.0e-12:
            raise PolicyFitError(
                f"{path} contains inconsistent improvement units and ticks"
            )
        raw_count += count
        if distance % 100 == 0:
            result[distance] += count
            compatible_count += count
    return ImprovementLoadResult(
        distribution=result,
        raw_count=raw_count,
        runtime_compatible_count=compatible_count,
        excluded_off_grid_count=raw_count - compatible_count,
    )


def finite_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise PolicyFitError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(parsed):
        raise PolicyFitError(f"{label} must be finite")
    return parsed


def load_lag_moments(path: pathlib.Path) -> dict[tuple[str, str, int], dict[str, float | int]]:
    rows = read_csv(path)
    required = (
        "source_event_type", "target_event_type", "lag_seconds", "paired_bins",
        "source_mean_count", "target_mean_count", "source_variance",
        "target_variance", "covariance", "correlation",
    )
    require_columns(rows, required, path)
    result: dict[tuple[str, str, int], dict[str, float | int]] = {}
    for row in rows:
        source = row["source_event_type"]
        target = row["target_event_type"]
        lag = parse_nonnegative_int(row["lag_seconds"], "lag_seconds")
        if source not in EVENT_INDEX or target not in EVENT_INDEX or lag not in LAG_SECONDS:
            raise PolicyFitError(f"invalid lag-moment identity in {path}")
        key = (source, target, lag)
        if key in result:
            raise PolicyFitError(f"duplicate lag-moment cell {key!r} in {path}")
        paired = parse_nonnegative_int(row["paired_bins"], "paired_bins")
        if paired != 23400 - lag:
            raise PolicyFitError(
                f"lag moment {key!r} has paired_bins={paired}, expected {23400-lag}"
            )
        values = {
            name: finite_float(row[name], name)
            for name in (
                "source_mean_count", "target_mean_count", "source_variance",
                "target_variance", "covariance", "correlation",
            )
        }
        if values["source_mean_count"] < 0.0 or values["target_mean_count"] < 0.0:
            raise PolicyFitError(f"negative lag-count mean in {path}")
        if values["source_variance"] < 0.0 or values["target_variance"] < 0.0:
            raise PolicyFitError(f"negative lag-count variance in {path}")
        if abs(values["correlation"]) > 1.0 + 1.0e-9:
            raise PolicyFitError(f"lag correlation outside [-1,1] in {path}")
        denominator = math.sqrt(
            values["source_variance"] * values["target_variance"]
        )
        reconstructed = values["covariance"] / denominator if denominator > 0.0 else 0.0
        if abs(reconstructed - values["correlation"]) > 1.0e-9:
            raise PolicyFitError(f"lag covariance/correlation mismatch in {path}: {key!r}")
        result[key] = {"paired_bins": paired, **values}
    expected = {
        (source, target, lag)
        for lag in LAG_SECONDS
        for source in EVENT_TYPES
        for target in EVENT_TYPES
    }
    if set(result) != expected:
        raise PolicyFitError(f"{path} must contain exactly 7 x 6 x 6 lag cells")
    return result


def verify_extractor_manifest(
    artifact_dir: pathlib.Path, date: str, symbol: str
) -> tuple[pathlib.Path, dict[str, object]]:
    compact = date.replace("-", "")
    path = artifact_dir / f"itch_manifest_{symbol.lower()}_{compact}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PolicyFitError(f"cannot read extractor manifest {path}: {error}") from error
    if payload.get("trading_date") != date or payload.get("symbol") != symbol:
        raise PolicyFitError(f"extractor manifest identity mismatch: {path}")
    training = payload.get("queue_reactive_training_artifacts")
    if not isinstance(training, dict):
        raise PolicyFitError(f"extractor manifest lacks queue-reactive training audit: {path}")
    conservation = training.get("event_count_conservation")
    exposure = training.get("exposure")
    expected_seconds = (
        exposure.get("expected_session_seconds")
        if isinstance(exposure, dict) else None
    )
    artifacts = training.get("artifacts")
    artifact_rows = training.get("artifact_row_counts")
    lag_definition = training.get("lag_moment_definition")
    if (
        training.get("schema_version") != 2
        or training.get("training_only") is not True
        or training.get("queue_policy_estimation_ready") is not True
        or not isinstance(training.get("pre_event_state_definition"), dict)
        or training["pre_event_state_definition"].get(
            "equal_timestamp_messages_share_one_left_limit_state"
        ) is not True
        or training["pre_event_state_definition"].get(
            "zero_duration_intermediate_states_are_not_used_as_covariates"
        ) is not True
        or not isinstance(conservation, dict)
        or conservation.get("totals_equal") is not True
        or conservation.get("equals_legacy_quantity_observation_counts") is not True
        or not isinstance(exposure, dict)
        or exposure.get("exact_nanosecond_conservation") is not True
        or not isinstance(expected_seconds, (int, float))
        or abs(float(expected_seconds) - 23400.0) > 1.0e-9
        or not isinstance(artifacts, dict)
        or artifacts.get("event_count_lag_moments") != "event_count_lag_moments.csv"
        or not isinstance(artifact_rows, dict)
        or artifact_rows.get("event_count_lag_moments") != len(LAG_SECONDS) * 6 * 6
        or not isinstance(lag_definition, dict)
        or lag_definition.get("count_bin_seconds") != 1
        or lag_definition.get("lags_seconds") != list(LAG_SECONDS)
        or lag_definition.get("direction")
        != "source count at t versus target count at t+lag"
    ):
        raise PolicyFitError(f"extractor manifest did not certify training artifacts: {path}")
    return path, training


def empty_accumulator() -> ClusterAccumulator:
    return ClusterAccumulator(
        [],
        [],
        [[0 for _ in EVENT_TYPES] for _ in range(HALF_HOUR_BINS)],
        [],
        {"limit_buy": Counter(), "limit_sell": Counter()},
        {"limit_buy": Counter(), "limit_sell": Counter()},
        defaultdict(LagMomentAggregate),
        [],
    )


def normalized_intraday_factors(
    counts: Sequence[Sequence[int]], shrinkage: float
) -> list[list[float]]:
    if not math.isfinite(shrinkage) or shrinkage < 0.0:
        raise PolicyFitError("intraday shrinkage must be finite and nonnegative")
    result = [[1.0 for _ in EVENT_TYPES] for _ in range(HALF_HOUR_BINS)]
    for event_index, event_type in enumerate(EVENT_TYPES):
        values = [row[event_index] for row in counts]
        mean_count = sum(values) / HALF_HOUR_BINS
        if mean_count <= 0.0:
            raise PolicyFitError(f"cluster has no training events for {event_type}")
        factors = [
            (value + shrinkage * mean_count) / ((1.0 + shrinkage) * mean_count)
            for value in values
        ]
        normalization = sum(factors) / HALF_HOUR_BINS
        for index, factor in enumerate(factors):
            result[index][event_index] = factor / normalization
    return result


def state_features(key: tuple[int, str, str, str, str]) -> list[float] | None:
    _, spread, imbalance, bid_ratio, ask_ratio = key
    if "unavailable" in (spread, imbalance, bid_ratio, ask_ratio):
        return None
    try:
        return [
            SPREAD_VALUE[spread],
            DEPTH_VALUE[bid_ratio],
            DEPTH_VALUE[ask_ratio],
            IMBALANCE_VALUE[imbalance],
        ]
    except KeyError as error:
        raise PolicyFitError(f"unknown queue-state bin: {error.args[0]!r}") from error


def solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    augmented = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-12:
            raise PolicyFitError("penalized queue-state system is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        for item in range(column, size + 1):
            augmented[column][item] /= scale
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for item in range(column, size + 1):
                augmented[row][item] -= factor * augmented[column][item]
    return [augmented[index][size] for index in range(size)]


def fit_state_coefficient_matrix(
    sources: Sequence[StateFitSource],
    intraday: Sequence[Sequence[float]],
    penalty: float,
    bound: float,
) -> tuple[list[list[float]], dict[str, object]]:
    if penalty <= 0.0 or not math.isfinite(penalty):
        raise PolicyFitError("state coefficient penalty must be finite and positive")
    if bound <= 0.0 or not math.isfinite(bound):
        raise PolicyFitError("state coefficient bound must be finite and positive")
    if not sources:
        raise PolicyFitError("conditional queue-state fitting has no symbol-day sources")
    observations: list[tuple[list[float], int, list[int], tuple[float, ...]]] = []
    valid_counts = [0 for _ in EVENT_TYPES]
    valid_exposure = 0.0
    for source in sources:
        if len(source.base_rates) != len(EVENT_TYPES) or any(
            not math.isfinite(value) or value <= 0.0
            for value in source.base_rates
        ):
            raise PolicyFitError(
                "conditional queue-state fitting requires positive symbol-specific "
                f"base rates for all types: {source.symbol} on {source.date}"
            )
        for state_key, seconds in sorted(source.exposure.items()):
            features = state_features(state_key)
            if features is None:
                continue
            half_hour = state_key[0]
            observed = [
                source.counts.get(
                    (half_hour, event_type, *state_key[1:]), 0
                )
                for event_type in EVENT_TYPES
            ]
            observations.append(
                (features, half_hour, observed, source.base_rates)
            )
            for event_index, value in enumerate(observed):
                valid_counts[event_index] += value
            valid_exposure += seconds

        # A count in a state absent from this symbol-day's exposure cannot be
        # assigned a finite conditional probability.  Keep the check local to
        # the source: pooling first would let another symbol's exposure mask a
        # malformed artifact.
        for count_key, count in source.counts.items():
            if count <= 0:
                continue
            state_key = (count_key[0], *count_key[2:])
            if (
                state_features(state_key) is not None
                and state_key not in source.exposure
            ):
                raise PolicyFitError(
                    f"{source.symbol} on {source.date}: {count_key[1]} has "
                    f"{count} events in a state with no exposure: {state_key!r}"
                )
    total_valid_count = sum(valid_counts)
    if (not observations or valid_exposure <= 0.0 or total_valid_count <= 0
            or any(value <= 0 for value in valid_counts)):
        raise PolicyFitError(
            "insufficient valid exposure/events for conditional queue-state fit"
        )

    type_count = len(EVENT_TYPES)
    feature_count = len(FEATURE_NAMES)
    dimension = type_count * feature_count
    beta = [[0.0 for _ in FEATURE_NAMES] for _ in EVENT_TYPES]
    converged = False
    iterations = 0
    for iterations in range(1, 101):
        gradient = [0.0] * dimension
        information = [[0.0] * dimension for _ in range(dimension)]
        for features, half_hour, observed, base_rates in observations:
            total = sum(observed)
            if total <= 0:
                continue
            logits = []
            for event_index in range(type_count):
                factor = intraday[half_hour][event_index]
                if not math.isfinite(factor) or factor <= 0.0:
                    raise PolicyFitError(
                        "conditional queue-state fitting requires positive "
                        "intraday factors"
                    )
                logits.append(
                    math.log(base_rates[event_index] * factor)
                    + sum(
                        beta[event_index][feature] * features[feature]
                        for feature in range(feature_count)
                    )
                )
            maximum = max(logits)
            weights = [math.exp(value - maximum) for value in logits]
            weight_sum = sum(weights)
            probabilities = [value / weight_sum for value in weights]
            for target in range(type_count):
                for feature in range(feature_count):
                    row = target * feature_count + feature
                    gradient[row] += (
                        observed[target] - total * probabilities[target]
                    ) * features[feature]
                    for source in range(type_count):
                        covariance = probabilities[target] * (
                            (1.0 if target == source else 0.0)
                            - probabilities[source]
                        )
                        for other_feature in range(feature_count):
                            column = source * feature_count + other_feature
                            information[row][column] += (
                                total * covariance * features[feature]
                                * features[other_feature]
                            )
        for event_index in range(type_count):
            for feature in range(feature_count):
                index = event_index * feature_count + feature
                gradient[index] -= penalty * beta[event_index][feature]
                information[index][index] += penalty
        step = solve_linear(information, gradient)
        # Deterministic damping protects sparse cells. Centering removes the
        # common per-feature shift that cancels under runtime normalization.
        max_step = max(abs(value) for value in step)
        damping = min(1.0, 1.0 / max_step) if max_step > 1.0 else 1.0
        for event_index in range(type_count):
            for feature in range(feature_count):
                beta[event_index][feature] += damping * step[
                    event_index * feature_count + feature
                ]
        for feature in range(feature_count):
            mean = sum(beta[event][feature] for event in range(type_count)) / type_count
            for event in range(type_count):
                beta[event][feature] -= mean
        if max(abs(damping * value) for value in step) < 1.0e-9:
            converged = True
            break
    if not converged:
        raise PolicyFitError("penalized conditional-multinomial IRLS did not converge")
    unbounded = [row[:] for row in beta]
    bounded = [
        [max(-bound, min(bound, value)) for value in row]
        for row in unbounded
    ]
    return bounded, {
        "estimator": (
            "L2-penalized conditional multinomial IRLS matching the runtime "
            "hazard-preserving type normalization"
        ),
        "valid_event_count": total_valid_count,
        "valid_event_count_by_type": dict(zip(EVENT_TYPES, valid_counts)),
        "valid_exposure_seconds": valid_exposure,
        "offset_scope": "symbol_specific_frozen_training_stationary_target_rates",
        "offset_source_count": len(sources),
        "offset_symbol_count": len({source.symbol for source in sources}),
        "deployment_cluster_mean_used_as_offset": False,
        "irls_iterations": iterations,
        "converged": True,
        "unbounded_coefficients": {
            event_type: dict(zip(FEATURE_NAMES, row))
            for event_type, row in zip(EVENT_TYPES, unbounded)
        },
        "coefficient_clipped": any(
            original != clipped
            for original_row, clipped_row in zip(unbounded, bounded)
            for original, clipped in zip(original_row, clipped_row)
        ),
    }


def zero_matrix() -> list[list[float]]:
    return [[0.0 for _ in EVENT_TYPES] for _ in EVENT_TYPES]


def topology_ratios() -> dict[tuple[str, str], tuple[float, float]]:
    """Map (target, source) to fast/slow edge regularization ratios."""
    result = {(event_type, event_type): (1.0, 1.0) for event_type in EVENT_TYPES}
    for target, source in (
        ("limit_buy", "market_sell"),
        ("limit_buy", "cancel_bid"),
        ("limit_sell", "market_buy"),
        ("limit_sell", "cancel_ask"),
    ):
        result[(target, source)] = (2.0 / 3.0, 0.75)
    for target, source in (
        ("cancel_bid", "limit_buy"),
        ("cancel_ask", "limit_sell"),
    ):
        result[(target, source)] = (0.5, 0.5)
    return result


def edge_branching_proxy_for_lags(
    aggregate: ClusterAccumulator,
    lags: Sequence[int],
    shrinkage_symbol_days: float,
    gain: float,
    cap: float,
    topology_ratio_index: int,
) -> tuple[dict[tuple[str, str], float], dict[str, object]]:
    """Estimate one integrated branching weight per allowed sparse edge.

    The one-second count correlations are only moment proxies: they do not
    identify a multivariate Hawkes likelihood.  Estimating each *allowed*
    edge separately nevertheless preserves empirical response heterogeneity
    that was discarded by the previous single-strength template.  Shrinkage
    is toward zero and the predeclared topology ratio remains a conservative
    regularizer.  A later, single global scale enforces nonnegative
    immigration and the common stability bounds without changing edge ratios.
    """
    if shrinkage_symbol_days < 0.0 or not math.isfinite(shrinkage_symbol_days):
        raise PolicyFitError("lag-correlation shrinkage must be finite and nonnegative")
    if gain <= 0.0 or cap <= 0.0 or not math.isfinite(gain) or not math.isfinite(cap):
        raise PolicyFitError("lag proxy gain/cap must be finite and positive")
    if topology_ratio_index not in (0, 1):
        raise PolicyFitError("topology-ratio index must identify fast or slow weights")

    ratios = topology_ratios()
    strengths: dict[tuple[str, str], float] = {}
    edge_estimates: list[dict[str, object]] = []
    cell_count = 0
    for target, source in sorted(ratios):
        cells: list[dict[str, object]] = []
        weighted_positive = 0.0
        total_weight = 0
        contribution_total = 0
        for lag in lags:
            key = (source, target, lag)
            moment = aggregate.lag_moments.get(key)
            if moment is None:
                raise PolicyFitError(f"missing pooled lag-moment cell {key!r}")
            summary = moment.summary()
            correlation = float(summary["correlation"])
            weight = int(summary["paired_bins"])
            weighted_positive += weight * max(0.0, correlation)
            total_weight += weight
            contribution_total += int(summary["contributions"])
            cell_count += 1
            cells.append({
                "source_event_type": source,
                "target_event_type": target,
                "lag_seconds": lag,
                **summary,
                "positive_part": max(0.0, correlation),
            })
        if total_weight <= 0 or not cells:
            raise PolicyFitError(
                f"lag proxy edge {(target, source)!r} has no estimable cells"
            )
        raw_positive_mean = weighted_positive / total_weight
        average_contributions = contribution_total / len(cells)
        shrinkage_factor = average_contributions / (
            average_contributions + shrinkage_symbol_days
        )
        shrunk_positive_mean = raw_positive_mean * shrinkage_factor
        uncapped_base_strength = gain * shrunk_positive_mean
        capped_base_strength = min(cap, uncapped_base_strength)
        topology_ratio = ratios[(target, source)][topology_ratio_index]
        strength = topology_ratio * capped_base_strength
        strengths[(target, source)] = strength
        edge_estimates.append({
            "target_event_type": target,
            "source_event_type": source,
            "topology_ratio": topology_ratio,
            "raw_paired_bin_weighted_positive_correlation_mean": raw_positive_mean,
            "average_symbol_day_contributions_per_cell": average_contributions,
            "shrinkage_factor": shrinkage_factor,
            "shrunk_positive_correlation_mean": shrunk_positive_mean,
            "uncapped_base_strength": uncapped_base_strength,
            "base_strength_before_topology_ratio": capped_base_strength,
            "pre_feasibility_integrated_branching": strength,
            "cap_active": uncapped_base_strength > cap,
            "raw_cells": cells,
        })

    nonzero_strengths = [value for value in strengths.values() if value > 0.0]
    return strengths, {
        "lags_seconds": list(lags),
        "allowed_edge_count": len(ratios),
        "cell_count": cell_count,
        "shrinkage_equivalent_symbol_days": shrinkage_symbol_days,
        "mapping_gain": gain,
        "pre_feasibility_cap": cap,
        "pre_feasibility_nonzero_edge_count": len(nonzero_strengths),
        "pre_feasibility_mean_edge_strength": (
            sum(strengths.values()) / len(strengths)
        ),
        "pre_feasibility_max_edge_strength": max(strengths.values()),
        "formula": (
            "for each allowed edge: r_plus=sum_l(paired_bins_l*max(0,rho_l))/"
            "sum_l(paired_bins_l); shrink=nbar/(nbar+tau); "
            "B_edge=topology_ratio*min(cap,gain*r_plus*shrink)"
        ),
        "edge_estimates": edge_estimates,
    }


def data_informed_branching_topology(
    aggregate: ClusterAccumulator,
    args: argparse.Namespace,
) -> tuple[list[list[float]], list[list[float]], dict[str, object]]:
    """Return MoM-proxy integrated fast/slow branching templates.

    One-second correlations do not identify a marked multivariate Hawkes
    likelihood.  They tune one nonnegative strength for each edge of a
    predeclared sparse topology, with fixed fast and slow decay rates.  This
    restricted method-of-moments interpretation is recorded in the audit.
    """
    fast_strengths, fast_audit = edge_branching_proxy_for_lags(
        aggregate,
        (1, 2),
        args.lag_shrinkage_symbol_days,
        args.fast_correlation_gain,
        args.fast_branching_cap,
        0,
    )
    slow_strengths, slow_audit = edge_branching_proxy_for_lags(
        aggregate,
        (5, 10, 20, 30),
        args.lag_shrinkage_symbol_days,
        args.slow_correlation_gain,
        args.slow_branching_cap,
        1,
    )
    if sum(fast_strengths.values()) + sum(slow_strengths.values()) <= 1.0e-12:
        raise PolicyFitError(
            "all allowed-edge fast and slow lag-correlation strengths are zero"
        )
    fast = zero_matrix()
    slow = zero_matrix()
    for target, source in topology_ratios():
        fast[EVENT_INDEX[target]][EVENT_INDEX[source]] = fast_strengths[(target, source)]
        slow[EVENT_INDEX[target]][EVENT_INDEX[source]] = slow_strengths[(target, source)]
    return fast, slow, {
        "estimator_version": 2,
        "estimator_class": (
            "edge-specific sparse method-of-moments proxy from pooled "
            "one-second count correlations"
        ),
        "identifiable_multivariate_hawkes_estimate": False,
        "reason": (
            "lag correlations do not identify 72 exponential-kernel amplitudes; "
            "they tune only the allowed edge strengths on a frozen sparse topology "
            "with fixed fast and slow decay rates"
        ),
        "limitations": [
            "one-second aggregation discards sub-second excitation",
            "positive lag correlation is associational and does not establish causality",
            "slow correlations may retain residual intraday or latent-state dependence",
            "the proxy does not identify edge-specific decay rates",
        ],
        "correlation_pooling": (
            "within-symbol-day centered covariance and variance are pooled by paired bins; "
            "only the positive correlation part enters the proxy"
        ),
        "pooled_correlation_formula": (
            "rho=(sum_j n_j cov_j/sum_j n_j)/sqrt((sum_j n_j var_source_j/"
            "sum_j n_j)*(sum_j n_j var_target_j/sum_j n_j))"
        ),
        "topology_orientation": "target/response rows, source/trigger columns",
        "allowed_edges": [
            {"target_event_type": target, "source_event_type": source}
            for target, source in sorted(topology_ratios())
        ],
        "fast": fast_audit,
        "slow": slow_audit,
    }


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def add_matrices(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]], scale: float = 1.0
) -> list[list[float]]:
    return [
        [scale * (a + b) for a, b in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def spectral_radius(matrix: Sequence[Sequence[float]]) -> float:
    """Return the Perron root of a nonnegative matrix deterministically.

    Iterating on ``I + matrix`` avoids the period-two oscillation of ordinary
    power iteration for bipartite sparse matrices.  The Perron root shifts by
    exactly one, and the Rayleigh quotient mirrors the production C++ audit.
    """
    if not matrix:
        return 0.0
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise PolicyFitError("branching matrix must be square")
    if any(
        (not math.isfinite(value)) or value < 0.0
        for row in matrix for value in row
    ):
        raise PolicyFitError("branching matrix must be finite and nonnegative")
    vector = [1.0] * len(matrix)
    previous = -1.0
    estimate = 0.0
    for iteration in range(20_000):
        product = [
            vector[row]
            + sum(matrix[row][column] * vector[column] for column in range(size))
            for row in range(size)
        ]
        norm = max(product)
        if not math.isfinite(norm) or norm <= 0.0:
            raise PolicyFitError("branching matrix has invalid spectral radius")
        vector = [value / norm for value in product]
        shifted_product = [
            vector[row]
            + sum(matrix[row][column] * vector[column] for column in range(size))
            for row in range(size)
        ]
        denominator = sum(value * value for value in vector)
        estimate = (
            sum(a * b for a, b in zip(vector, shifted_product)) / denominator
            - 1.0
        )
        if (
            iteration > 64
            and abs(estimate - previous)
            <= 1.0e-14 * max(1.0, abs(estimate))
        ):
            break
        previous = estimate
    return max(0.0, estimate)


def scaled_branching_for_targets(
    target_vectors: Sequence[Sequence[float]],
    maximum_radius: float,
    fast: Sequence[Sequence[float]],
    slow: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[list[float]], float, float, float, float]:
    integrated = add_matrices(fast, slow)
    scale = 1.0
    for targets in target_vectors:
        excitation = matvec(integrated, targets)
        for target, incoming in zip(targets, excitation):
            if incoming > 0.0:
                scale = min(scale, 0.95 * target / incoming)
    template_radius = spectral_radius(integrated)
    if template_radius > 0.0:
        scale = min(scale, 0.999 * maximum_radius / template_radius)
    template_maximum_row_sum = max(sum(row) for row in integrated)
    if template_maximum_row_sum > 0.0:
        scale = min(
            scale,
            0.999 * maximum_radius / template_maximum_row_sum,
        )
    # Runtime state response may relabel the accepted event type before its
    # excitation column is applied.  Consequently the relevant worst-case
    # offspring bound is also the induced one-norm (maximum trigger-column
    # sum), not only the response-row sum used for the latent linear model.
    # Fit and runtime must certify the same matrix rather than relying on the
    # C++ loader to reject an otherwise completed policy fit.
    template_maximum_column_sum = max(
        sum(integrated[row][column] for row in range(len(integrated)))
        for column in range(len(integrated))
    )
    if template_maximum_column_sum > 0.0:
        scale = min(
            scale,
            0.999 * maximum_radius / template_maximum_column_sum,
        )
    if not math.isfinite(scale) or scale <= 1.0e-12:
        raise PolicyFitError("cannot derive a positive feasible branching scale")
    fast = [[scale * value for value in row] for row in fast]
    slow = [[scale * value for value in row] for row in slow]
    radius = spectral_radius(add_matrices(fast, slow))
    integrated = add_matrices(fast, slow)
    maximum_row_sum = max(sum(row) for row in integrated)
    maximum_column_sum = max(
        sum(integrated[row][column] for row in range(len(integrated)))
        for column in range(len(integrated))
    )
    if radius >= maximum_radius or radius >= 1.0:
        raise PolicyFitError(f"branching matrix is unstable: spectral radius {radius}")
    if maximum_row_sum >= maximum_radius:
        raise PolicyFitError(
            "branching matrix violates the sufficient row-sum stability gate: "
            f"{maximum_row_sum}"
        )
    if maximum_column_sum >= maximum_radius:
        raise PolicyFitError(
            "branching matrix violates the state-responsive column-sum "
            f"stability gate: {maximum_column_sum}"
        )
    return (
        fast, slow, scale, radius, maximum_row_sum, maximum_column_sum,
    )


def immigration_for_targets(
    targets: Sequence[float], integrated: Sequence[Sequence[float]], activity_scale: float
) -> list[float]:
    incoming = matvec(integrated, targets)
    mu = [(target - excitation) / activity_scale for target, excitation in zip(targets, incoming)]
    if any((not math.isfinite(value)) or value < -1.0e-12 for value in mu):
        raise PolicyFitError("frozen target rates imply negative Hawkes immigration")
    mu = [max(0.0, value) for value in mu]
    reconstructed = [
        activity_scale * value + excitation for value, excitation in zip(mu, incoming)
    ]
    if any(abs(a - b) > 1.0e-10 * max(1.0, abs(b)) for a, b in zip(reconstructed, targets)):
        raise PolicyFitError("stationary reconstruction does not equal frozen target rates")
    return mu


def write_matrix_csv(path: pathlib.Path, matrix: Sequence[Sequence[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["response_event_type", *EVENT_TYPES])
        for event_type, row in zip(EVENT_TYPES, matrix):
            writer.writerow([event_type, *(f"{value:.17g}" for value in row)])


def write_intraday_csv(path: pathlib.Path, factors: Sequence[Sequence[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["half_hour_bin", *EVENT_TYPES])
        for index, row in enumerate(factors):
            writer.writerow([index, *(f"{value:.17g}" for value in row)])


def write_state_csv(path: pathlib.Path, coefficients: Sequence[Sequence[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["event_type", *FEATURE_NAMES])
        for event_type, row in zip(EVENT_TYPES, coefficients):
            writer.writerow([event_type, *(f"{value:.17g}" for value in row)])


def write_improvement_csv(path: pathlib.Path, counts: Counter[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["improvement_ticks", "improvement_price_units", "count"])
        for distance, count in sorted(counts.items()):
            writer.writerow([f"{distance / 100.0:.17g}", distance, count])


def write_long_form_policy_csv(
    path: pathlib.Path,
    cluster_id: str,
    mean_targets: Sequence[float],
    fast: Sequence[Sequence[float]],
    slow: Sequence[Sequence[float]],
    fast_beta: float,
    slow_beta: float,
    coefficients: Sequence[Sequence[float]],
    factors: Sequence[Sequence[float]],
    activity_scale: float,
    spectral_radius_value: float,
    maximum_integrated_row_sum: float,
    maximum_integrated_column_sum: float,
    state_log_multiplier_bound: float,
) -> None:
    """Write the dependency-free runtime policy interchange format.

    Paths in the separate symbol mapping are relative to its parent output
    root.  Matrix rows are responding types and columns are triggering types.
    ``fast_alpha`` and ``slow_alpha`` are kernel amplitudes (integrated
    branching multiplied by beta), matching the C++ runtime convention.
    Cluster stationary targets are descriptive member means; per-symbol
    targets remain frozen in each symbol's original Hawkes rate file.
    """
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["kind", "target", "source", "bin", "value"])
        metadata = (
            ("schema_version", 1),
            ("cluster_id", cluster_id),
            ("activity_scale", activity_scale),
            ("fast_beta", fast_beta),
            ("slow_beta", slow_beta),
            ("intraday_origin_ns", 0),
            ("intraday_bin_width_ns", 1_800_000_000_000),
            ("state_log_multiplier_bound", state_log_multiplier_bound),
            ("spectral_radius", spectral_radius_value),
            ("maximum_integrated_row_sum", maximum_integrated_row_sum),
            ("maximum_integrated_column_sum", maximum_integrated_column_sum),
            ("matrix_orientation", "response_rows_trigger_columns"),
            ("stationary_target_scope", "descriptive_cluster_member_mean"),
        )
        for target, value in metadata:
            writer.writerow(["meta", target, "", "", value])
        for event_type, value in zip(EVENT_TYPES, mean_targets):
            writer.writerow([
                "diagnostic_cluster_target", event_type, "", "", f"{value:.17g}"
            ])
        for target_index, target in enumerate(EVENT_TYPES):
            for source_index, source in enumerate(EVENT_TYPES):
                writer.writerow([
                    "fast_alpha", target, source, "",
                    f"{fast[target_index][source_index] * fast_beta:.17g}",
                ])
                writer.writerow([
                    "slow_alpha", target, source, "",
                    f"{slow[target_index][source_index] * slow_beta:.17g}",
                ])
        for target, row in zip(EVENT_TYPES, coefficients):
            for source, value in zip(FEATURE_NAMES, row):
                writer.writerow(["state_coefficient", target, source, "", f"{value:.17g}"])
        for bin_index, row in enumerate(factors):
            for target, value in zip(EVENT_TYPES, row):
                writer.writerow(["intraday_factor", target, "", bin_index, f"{value:.17g}"])


def fit(args: argparse.Namespace) -> dict[str, object]:
    training_roots: list[TrainingRoot] = args.training_root
    if len(training_roots) != args.expected_training_days:
        raise PolicyFitError(
            f"expected exactly {args.expected_training_days} training roots, "
            f"observed {len(training_roots)}"
        )
    dates = [item.date for item in training_roots]
    if len(set(dates)) != len(dates):
        raise PolicyFitError("training dates must be unique")
    forbidden = set(args.forbid_date)
    leaked = sorted(set(dates) & forbidden)
    if leaked:
        raise PolicyFitError(f"forbidden held-out date supplied as training input: {leaked}")

    output_root = pathlib.Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise PolicyFitError(f"output root must be absent or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    config_path = pathlib.Path(args.pooled_config).expanduser().resolve()
    cluster_path = pathlib.Path(args.cluster_assignments).expanduser().resolve()
    fieldnames, config_rows = load_pooled_config(config_path)
    cluster_map = load_symbol_map(cluster_path, "cluster_id")
    for symbol, cluster_id in cluster_map.items():
        if not cluster_id.isdigit():
            raise PolicyFitError(
                f"cluster_id must be a nonnegative integer for runtime loading: "
                f"{symbol}={cluster_id!r}"
            )
    symbols = [row["symbol"] for row in config_rows]
    if not set(symbols).issubset(cluster_map):
        missing = sorted(set(symbols) - set(cluster_map))
        raise PolicyFitError(
            f"cluster assignments do not cover the pooled config: {missing[:5]}"
        )
    def symbol_file(path_text: str, *, option: str) -> set[str]:
        source = pathlib.Path(path_text).expanduser().resolve()
        selected = {
            line.strip().upper()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if not selected:
            raise PolicyFitError(f"{option} contains no symbols")
        unknown = sorted(selected - set(symbols))
        if unknown:
            raise PolicyFitError(
                f"{option} contains symbols absent from pooled config: "
                f"{unknown[:5]}"
            )
        return selected

    fitting_symbols = set(symbols)
    if args.fit_symbols_file:
        fitting_symbols = symbol_file(
            args.fit_symbols_file, option="--fit-symbols-file",
        )
    selection_symbols = set(fitting_symbols)
    if args.selection_symbols_file:
        selection_symbols = symbol_file(
            args.selection_symbols_file,
            option="--selection-symbols-file",
        )

    clusters: dict[str, ClusterAccumulator] = defaultdict(empty_accumulator)
    target_rates: dict[str, list[float]] = {}
    config_by_symbol = {row["symbol"]: row for row in config_rows}
    for symbol in symbols:
        raw_rates = pathlib.Path(config_by_symbol[symbol]["hawkes_rates_file"])
        if not raw_rates.is_absolute():
            raw_rates = (config_path.parent / raw_rates).resolve()
        target_rates[symbol] = load_target_rates(raw_rates)
        clusters[cluster_map[symbol]].members.append(symbol)
        if symbol in fitting_symbols:
            clusters[cluster_map[symbol]].estimation_members.append(symbol)

    uncovered_clusters = sorted(
        cluster_id
        for cluster_id, cluster in clusters.items()
        if not cluster.estimation_members
    )
    if uncovered_clusters:
        raise PolicyFitError(
            "fit-symbol list has no training representative for clusters: "
            + ", ".join(uncovered_clusters)
        )
    uncovered_selection_clusters = sorted(
        cluster_id for cluster_id in clusters
        if not any(
            cluster_map[symbol] == cluster_id
            for symbol in selection_symbols
        )
    )
    if uncovered_selection_clusters:
        raise PolicyFitError(
            "selection-symbol list has no representative for clusters: "
            + ", ".join(uncovered_selection_clusters)
        )

    for root in sorted(training_roots, key=lambda item: item.date):
        for symbol in sorted(fitting_symbols):
            cluster = clusters[cluster_map[symbol]]
            artifact_dir = symbol_artifact_dir(root, symbol)
            manifest_path, training_audit = verify_extractor_manifest(
                artifact_dir, root.date, symbol
            )
            intraday_path = artifact_dir / "intraday_event_counts.csv"
            counts_path = artifact_dir / "queue_state_counts.csv"
            exposure_path = artifact_dir / "queue_state_exposure.csv"
            lag_path = artifact_dir / "event_count_lag_moments.csv"
            intraday, time_totals = load_intraday(intraday_path)
            state_counts, state_totals = load_state_counts(counts_path)
            exposure = load_state_exposure(exposure_path)
            lag_moments = load_lag_moments(lag_path)
            if time_totals != state_totals:
                raise PolicyFitError(
                    f"event-count conservation failed for {symbol} on {root.date}: "
                    f"{time_totals!r} != {state_totals!r}"
                )
            audit_counts = training_audit["event_count_conservation"]["by_event_type"]
            if time_totals != audit_counts:
                raise PolicyFitError(
                    f"CSV/manifest count audit failed for {symbol} on {root.date}"
                )
            if abs(sum(exposure.values()) - 23400.0) > 1.0e-6:
                raise PolicyFitError(
                    f"queue-state exposure is not one full 23,400-second session "
                    f"for {symbol} on {root.date}"
                )
            for index in range(HALF_HOUR_BINS):
                for event_index in range(len(EVENT_TYPES)):
                    cluster.intraday_counts[index][event_index] += intraday[index][event_index]
            cluster.state_fit_sources.append(StateFitSource(
                date=root.date,
                symbol=symbol,
                counts=state_counts,
                exposure=exposure,
                base_rates=tuple(target_rates[symbol]),
            ))
            source_improvement_audit: dict[str, dict[str, object]] = {}
            for side in ("limit_buy", "limit_sell"):
                path = artifact_dir / f"{side}_improvement_distribution.txt"
                loaded = load_improvements(path)
                cluster.improvements[side].update(loaded.distribution)
                projection = cluster.improvement_projection[side]
                projection["raw_exact_count"] += loaded.raw_count
                projection["runtime_compatible_count"] += (
                    loaded.runtime_compatible_count
                )
                projection["excluded_off_grid_count"] += (
                    loaded.excluded_off_grid_count
                )
                source_improvement_audit[side] = {
                    "source_file": str(path),
                    "source_sha256": sha256_file(path),
                    "raw_exact_count": loaded.raw_count,
                    "runtime_compatible_count": loaded.runtime_compatible_count,
                    "excluded_off_grid_count": loaded.excluded_off_grid_count,
                }
            for key, moment in lag_moments.items():
                cluster.lag_moments[key].add(
                    int(moment["paired_bins"]),
                    float(moment["source_mean_count"]),
                    float(moment["target_mean_count"]),
                    float(moment["source_variance"]),
                    float(moment["target_variance"]),
                    float(moment["covariance"]),
                )
            cluster.sources.append({
                "date": root.date,
                "symbol": symbol,
                "artifact_directory": str(artifact_dir),
                "intraday_sha256": sha256_file(intraday_path),
                "state_counts_sha256": sha256_file(counts_path),
                "state_exposure_sha256": sha256_file(exposure_path),
                "extractor_manifest_sha256": sha256_file(manifest_path),
                "lag_moments_sha256": sha256_file(lag_path),
                "inside_spread_mark_projection": source_improvement_audit,
            })

    activity_scale = args.activity_scale
    if not math.isfinite(activity_scale) or activity_scale <= 0.0:
        raise PolicyFitError("activity scale must be finite and positive")
    cluster_payloads: dict[str, dict[str, object]] = {}
    symbol_payloads: dict[str, dict[str, object]] = {}
    for cluster_id in sorted(
        clusters,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    ):
        cluster = clusters[cluster_id]
        factors = normalized_intraday_factors(cluster.intraday_counts, args.intraday_shrinkage)
        vectors = [target_rates[symbol] for symbol in cluster.members]
        mean_targets = [
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(len(EVENT_TYPES))
        ]
        coefficients, diagnostics = fit_state_coefficient_matrix(
            cluster.state_fit_sources,
            factors,
            args.state_penalty,
            args.state_coefficient_bound,
        )
        for side in ("limit_buy", "limit_sell"):
            if not cluster.improvements[side]:
                raise PolicyFitError(
                    f"cluster {cluster_id} has no pooled {side} inside-spread marks"
                )
        fast_template, slow_template, lag_proxy_audit = (
            data_informed_branching_topology(cluster, args)
        )
        (
            fast,
            slow,
            branching_scale,
            radius,
            maximum_row_sum,
            maximum_column_sum,
        ) = (
            scaled_branching_for_targets(
            vectors,
            args.maximum_spectral_radius,
            fast_template,
            slow_template,
            )
        )
        lag_proxy_audit["feasibility_and_stability_rescale"] = branching_scale
        for band in ("fast", "slow"):
            band_audit = lag_proxy_audit[band]
            if not isinstance(band_audit, dict):
                raise PolicyFitError("internal lag-proxy band audit is invalid")
            edge_estimates = band_audit["edge_estimates"]
            if not isinstance(edge_estimates, list) or not edge_estimates:
                raise PolicyFitError("internal lag-proxy edge audit is invalid")
            post_strengths: list[float] = []
            for edge in edge_estimates:
                if not isinstance(edge, dict):
                    raise PolicyFitError("internal lag-proxy edge record is invalid")
                post_strength = (
                    float(edge["pre_feasibility_integrated_branching"])
                    * branching_scale
                )
                edge["post_feasibility_integrated_branching"] = post_strength
                post_strengths.append(post_strength)
            band_audit["post_feasibility_mean_edge_strength"] = (
                sum(post_strengths) / len(post_strengths)
            )
            band_audit["post_feasibility_max_edge_strength"] = max(post_strengths)
        integrated = add_matrices(fast, slow)

        directory = output_root / "clusters" / f"cluster_{cluster_id}"
        directory.mkdir(parents=True)
        write_intraday_csv(directory / "intraday_factors.csv", factors)
        write_state_csv(directory / "state_coefficients.csv", coefficients)
        write_matrix_csv(directory / "fast_branching_matrix.csv", fast)
        write_matrix_csv(directory / "slow_branching_matrix.csv", slow)
        write_improvement_csv(
            directory / "limit_buy_improvement_distribution.csv",
            cluster.improvements["limit_buy"],
        )
        write_improvement_csv(
            directory / "limit_sell_improvement_distribution.csv",
            cluster.improvements["limit_sell"],
        )
        write_long_form_policy_csv(
            directory / "cluster_policy.csv",
            cluster_id,
            mean_targets,
            fast,
            slow,
            args.fast_beta,
            args.slow_beta,
            coefficients,
            factors,
            activity_scale,
            radius,
            maximum_row_sum,
            maximum_column_sum,
            args.state_log_multiplier_bound,
        )
        payload: dict[str, object] = {
            "schema_version": 1,
            "model": "queue-reactive-hawkes-v2-edge-specific-proxy",
            "training_only": True,
            "cluster_id": cluster_id,
            "member_count": len(cluster.members),
            "members": sorted(cluster.members),
            "estimation_member_count": len(cluster.estimation_members),
            "estimation_members": sorted(cluster.estimation_members),
            "event_types": list(EVENT_TYPES),
            "intraday": {
                "bin_count": HALF_HOUR_BINS,
                "bin_width_seconds": 1800,
                "runtime_origin_ns": 0,
                "clock_mapping": "extractor half_hour_bin maps directly to simulator elapsed-session time",
                "normalization": "arithmetic mean one separately for each event type",
                "shrinkage_toward_one": args.intraday_shrinkage,
                "file": f"clusters/cluster_{cluster_id}/intraday_factors.csv",
            },
            "state_response": {
                "estimator_version": 2,
                "features": list(FEATURE_NAMES),
                "estimator": (
                    "deterministic L2-penalized conditional multinomial IRLS "
                    "with cluster intraday factors and symbol-specific frozen "
                    "training-rate offsets"
                ),
                "penalty": args.state_penalty,
                "coefficient_bound": args.state_coefficient_bound,
                "runtime_log_multiplier_bound": args.state_log_multiplier_bound,
                "file": f"clusters/cluster_{cluster_id}/state_coefficients.csv",
                "diagnostics": diagnostics,
            },
            "hawkes": {
                "matrix_orientation": "response rows, trigger columns",
                "activity_scale": activity_scale,
                "fast_beta_per_second": args.fast_beta,
                "slow_beta_per_second": args.slow_beta,
                "fast_integrated_branching_file": f"clusters/cluster_{cluster_id}/fast_branching_matrix.csv",
                "slow_integrated_branching_file": f"clusters/cluster_{cluster_id}/slow_branching_matrix.csv",
                "branching_scale": branching_scale,
                "lag_moment_proxy": lag_proxy_audit,
                "spectral_radius": radius,
                "maximum_integrated_row_sum": maximum_row_sum,
                "maximum_integrated_column_sum": maximum_column_sum,
                "maximum_spectral_radius": args.maximum_spectral_radius,
                "topology": "self excitation plus same-side market/cancel-to-limit replenishment and limit-to-cancel response",
                "runtime_long_form_file": f"clusters/cluster_{cluster_id}/cluster_policy.csv",
            },
            "improvement_marks": {
                "limit_buy_file": f"clusters/cluster_{cluster_id}/limit_buy_improvement_distribution.csv",
                "limit_sell_file": f"clusters/cluster_{cluster_id}/limit_sell_improvement_distribution.csv",
                "price_unit_is_ITCH_1e-4_dollars": True,
                "runtime_price_grid_units": 100,
                "projection": (
                    "retain positive exact ITCH price improvements divisible by "
                    "100 price units; preserve and hash all raw source rows"
                ),
                "raw_artifacts_modified": False,
                "audit": {
                    side: dict(sorted(counts.items()))
                    for side, counts in sorted(
                        cluster.improvement_projection.items()
                    )
                },
            },
            "source_count": len(cluster.sources),
            "sources": sorted(cluster.sources, key=lambda item: (str(item["date"]), str(item["symbol"]))),
        }
        write_json(directory / "policy.json", payload)
        cluster_payloads[cluster_id] = payload

        for symbol in sorted(cluster.members):
            targets = target_rates[symbol]
            mu = immigration_for_targets(targets, integrated, activity_scale)
            symbol_payload = {
                "schema_version": 1,
                "model": "queue-reactive-hawkes-v2-edge-specific-proxy",
                "training_only": True,
                "symbol": symbol,
                "cluster_id": cluster_id,
                "cluster_policy_file": f"clusters/cluster_{cluster_id}/policy.json",
                "event_types": list(EVENT_TYPES),
                "stationary_target_rates": dict(zip(EVENT_TYPES, targets)),
                "immigration_rates_before_activity_scale": dict(zip(EVENT_TYPES, mu)),
                "stationary_reconstruction_passed": True,
            }
            symbol_path = output_root / "symbols" / f"{symbol.lower()}_policy.json"
            write_json(symbol_path, symbol_payload)
            symbol_payloads[symbol] = symbol_payload

    def write_mapping(mapping_path: pathlib.Path, mapping_symbols: Sequence[str]) -> None:
        with mapping_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([
                "symbol", "cluster_id", "policy_file",
                "limit_buy_improvement_file", "limit_sell_improvement_file",
            ])
            for symbol in mapping_symbols:
                cluster_id = cluster_map[symbol]
                prefix = f"clusters/cluster_{cluster_id}"
                writer.writerow([
                    symbol,
                    cluster_id,
                    f"{prefix}/cluster_policy.csv",
                    f"{prefix}/limit_buy_improvement_distribution.csv",
                    f"{prefix}/limit_sell_improvement_distribution.csv",
                ])

    mapping_path = output_root / "symbol_policy_mapping.csv"
    write_mapping(mapping_path, sorted(symbols))
    estimation_mapping_path = output_root / "estimation_symbol_policy_mapping.csv"
    write_mapping(estimation_mapping_path, sorted(selection_symbols))
    estimation_clusters_path = output_root / "estimation_cluster_assignments.csv"
    with estimation_clusters_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["symbol", "cluster_id"])
        for symbol in sorted(selection_symbols):
            writer.writerow([symbol, cluster_map[symbol]])

    expanded_fields = list(fieldnames)
    for name in ("cluster_id", "background_model", "background_policy_file"):
        if name not in expanded_fields:
            expanded_fields.append(name)
    expanded_path = output_root / "expanded_training_config.csv"
    with expanded_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=expanded_fields, extrasaction="ignore")
        writer.writeheader()
        for original in config_rows:
            row = dict(original)
            symbol = row["symbol"]
            row["cluster_id"] = cluster_map[symbol]
            row["background_model"] = "queue-reactive-hawkes-v2-edge-specific-proxy"
            row["background_policy_file"] = f"symbols/{symbol.lower()}_policy.json"
            writer.writerow(row)

    manifest = {
        "schema_version": 1,
        "model": "queue-reactive-hawkes-v2-edge-specific-proxy",
        "status": "training_policy_fit_complete",
        "workflow_source": {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": sha256_file(pathlib.Path(__file__).resolve()),
        },
        "training_only": True,
        "training_dates": sorted(dates),
        "forbidden_dates": sorted(forbidden),
        "heldout_inputs_read": False,
        "symbol_count": len(symbols),
        "fitting_symbol_count": len(fitting_symbols),
        "selection_symbol_count": len(selection_symbols),
        "cluster_count": len(clusters),
        "pooled_config_sha256": sha256_file(config_path),
        "cluster_assignments_sha256": sha256_file(cluster_path),
        "expanded_training_config": "expanded_training_config.csv",
        "cluster_policy_files": {
            cluster_id: f"clusters/cluster_{cluster_id}/policy.json"
            for cluster_id in sorted(cluster_payloads)
        },
        "symbol_policy_pattern": "symbols/<lowercase-symbol>_policy.json",
        "symbol_policy_mapping": "symbol_policy_mapping.csv",
        "estimation_symbol_policy_mapping": (
            "estimation_symbol_policy_mapping.csv"
        ),
        "estimation_cluster_assignments": "estimation_cluster_assignments.csv",
        "estimation_mapping_role": "behavioural_selection_subset",
        "mapping_path_semantics": "all mapping paths are relative to the output root",
        "certification": {
            "intraday_typewise_means_equal_one": True,
            "count_conservation_checked_per_symbol_day_type": True,
            "state_counts_require_observed_exposure": True,
            "state_response_uses_symbol_specific_frozen_offsets": True,
            "stationary_targets_reconstructed_for_every_symbol": True,
            "all_cluster_spectral_radii_below_limit": True,
            "all_cluster_integrated_row_sums_below_limit": True,
            "all_cluster_integrated_column_sums_below_limit": True,
            "lag_moment_artifacts_complete": True,
            "branching_strengths_are_training_data_informed_mom_proxies": True,
            "off_grid_improvement_marks_are_explicitly_audited_and_excluded": True,
            "raw_improvement_artifacts_are_unmodified": True,
        },
    }
    write_json(output_root / "training_policy_manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--training-root", action="append", type=parse_training_root, required=True,
        help="training extraction root as ISO-DATE=PATH; repeat once per training day",
    )
    result.add_argument("--cluster-assignments", required=True)
    result.add_argument("--pooled-config", required=True)
    result.add_argument(
        "--fit-symbols-file",
        help="optional newline-delimited training subset; it must represent every cluster",
    )
    result.add_argument(
        "--selection-symbols-file",
        help=(
            "optional newline-delimited behavioural-selection subset; "
            "cluster policy fitting still uses --fit-symbols-file, or every "
            "training symbol when that option is omitted"
        ),
    )
    result.add_argument("--output-root", required=True)
    result.add_argument("--forbid-date", action="append", default=[])
    result.add_argument("--expected-training-days", type=int, default=5)
    result.add_argument("--intraday-shrinkage", type=float, default=1.0)
    result.add_argument("--state-penalty", type=float, default=25.0)
    result.add_argument("--state-coefficient-bound", type=float, default=1.5)
    result.add_argument("--state-log-multiplier-bound", type=float, default=4.0)
    result.add_argument("--lag-shrinkage-symbol-days", type=float, default=25.0)
    result.add_argument("--fast-correlation-gain", type=float, default=0.75)
    result.add_argument("--slow-correlation-gain", type=float, default=0.25)
    result.add_argument("--fast-branching-cap", type=float, default=0.20)
    result.add_argument("--slow-branching-cap", type=float, default=0.08)
    result.add_argument("--maximum-spectral-radius", type=float, default=0.75)
    result.add_argument("--activity-scale", type=float, default=0.30)
    result.add_argument(
        "--fast-beta", type=float, default=1.0,
        help="fast exponential decay per second; default matches the 1--2 s proxy band",
    )
    result.add_argument(
        "--slow-beta", type=float, default=0.1,
        help="slow exponential decay per second; default matches the 5--30 s proxy band",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.expected_training_days <= 0:
            raise PolicyFitError("--expected-training-days must be positive")
        if not 0.0 < args.maximum_spectral_radius < 0.75 + 1.0e-15:
            raise PolicyFitError("--maximum-spectral-radius must lie in (0,0.75]")
        if (
            not math.isfinite(args.fast_beta)
            or not math.isfinite(args.slow_beta)
            or args.fast_beta <= 0.0
            or args.slow_beta <= 0.0
        ):
            raise PolicyFitError("kernel decay rates must be finite and positive")
        if (
            args.state_log_multiplier_bound <= 0.0
            or not math.isfinite(args.state_log_multiplier_bound)
        ):
            raise PolicyFitError("--state-log-multiplier-bound must be finite and positive")
        manifest = fit(args)
    except (PolicyFitError, OSError) as error:
        print(f"queue-reactive policy fitting failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
