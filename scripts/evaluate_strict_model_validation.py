#!/usr/bin/env python3
"""Versioned fit evaluation for the fragmented-LOB simulator.

The evaluator consumes empirical per-symbol target configurations, a frozen
liquidity-cluster assignment and one or more per-asset simulator summaries for
each date/seed.  It does not run the simulator and it never changes model
parameters.  The same gates are applied independently to every supplied date.

Three evidential roles are deliberately distinct:

``training_fit``
    In-sample adequacy.  Every expected training date must pass separately.
``development_validation``
    Out-of-sample development evidence that may inform a later model revision.
    A pass is *not* called certification or final validation.
``untouched_final_holdout``
    Final evidence only when a non-empty, pre-existing protocol-freeze record
    is supplied.  The record is hashed into the report; the evaluator cannot
    prove that the date was never inspected, so that claim remains a research
    governance responsibility.

``strict-nine-v1`` preserves the original development gate.  The explicitly
labelled ``marketwide-six-v2`` protocol is a retrospective development
revision: activity, spread, combined top depth, mid-move rate, return
variance and absolute-return persistence determine adequacy; kurtosis,
side-specific depth, cluster scores and ACF distribution moments remain
visible diagnostics.  Malformed or structurally incomplete inputs always fail
closed under both protocols.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
import sys
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 1

METRICS = (
    "background_event_rate",
    "mean_spread_ticks",
    "mean_bid_depth",
    "mean_ask_depth",
    "mid_move_rate",
    "return_variance",
    "return_kurtosis",
    "absolute_return_acf1",
    "two_sided_sample_fraction",
)
COMBINED_DEPTH_METRIC = "combined_top_depth"
REPORTED_METRICS = (*METRICS, COMBINED_DEPTH_METRIC)
SIX_COMPONENT_METRICS = (
    "background_event_rate",
    "mean_spread_ticks",
    COMBINED_DEPTH_METRIC,
    "mid_move_rate",
    "return_variance",
    "absolute_return_acf1",
)
POSITIVE_LOG_RATIO_METRICS = frozenset({
    "background_event_rate",
    "mean_spread_ticks",
    "mean_bid_depth",
    "mean_ask_depth",
    COMBINED_DEPTH_METRIC,
    "return_variance",
    "return_kurtosis",
})
CLUSTER_GATE_METRICS = (
    "mean_spread_ticks",
    "return_variance",
    "return_kurtosis",
)
BACKGROUND_EVENTS = (
    "limit_buy", "limit_sell", "market_buy", "market_sell",
    "cancel_bid", "cancel_ask",
)

# Immutable gate v1.  These are constants, not CLI options, so an observed
# result cannot be turned into a pass by rerunning with looser thresholds.
STRICT_NINE_GATE = "strict-nine-v1"
MARKETWIDE_SIX_GATE = "marketwide-six-v2"
GATE_PROTOCOLS = (STRICT_NINE_GATE, MARKETWIDE_SIX_GATE)
GATE_IDS = {
    STRICT_NINE_GATE: "strict_queue_reactive_fit_gate_v1",
    MARKETWIDE_SIX_GATE: (
        "retrospective_marketwide_six_component_adequacy_gate_v2"
    ),
}
MAX_MARKETWIDE_ROBUST_SCORE = 1.5
MAX_MARKETWIDE_METRIC_SCORE = 2.5
MAX_CLUSTER_METRIC_SCORE = 3.0
GROSS_RESIDUAL_LIMIT = 6.0
MIN_PER_METRIC_SYMBOL_FRACTION_WITHIN_LIMIT = 0.95
MAX_SYMBOL_ANY_GROSS_FAILURE_FRACTION = 0.10
MAX_ACF_MEAN_ABSOLUTE_ERROR = 0.02
MAX_ACF_MEDIAN_ABSOLUTE_ERROR = 0.02
MAX_ACF_P90_ABSOLUTE_ERROR = 0.03
ROBUST_LOG_RATIO_UNIT = math.log(1.5)
ROBUST_MID_MOVE_LOG_ODDS_UNIT = math.log(2.0)
ROBUST_ACF_FISHER_UNIT = 0.25
ROBUST_COVERAGE_UNIT = 0.01
ROBUST_PROBABILITY_EPSILON = 1.0e-6
HUBER_DELTA = 2.0

ROLES = (
    "training_fit",
    "development_validation",
    "untouched_final_holdout",
)
ROLE_LABELS = {
    "training_fit": "training_fit_adequate",
    "development_validation": (
        "development_validation_adequate_not_certification"
    ),
    "untouched_final_holdout": "untouched_final_holdout_adequate",
}


class EvaluationError(RuntimeError):
    """Raised when an input violates the strict evaluation contract."""


@dataclass(frozen=True)
class Target:
    value: float
    scale: float


@dataclass(frozen=True)
class SummarySpec:
    day: str
    seed: int
    path: pathlib.Path


def finite_float(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"{label} is not numeric") from error
    if not math.isfinite(result):
        raise EvaluationError(f"{label} is not finite")
    return result


def clock_seconds(value: object, *, label: str) -> float:
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) != 3:
        raise EvaluationError(f"{label} must be HH:MM:SS[.fraction]")
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    except ValueError as error:
        raise EvaluationError(f"{label} must be HH:MM:SS[.fraction]") from error
    if hours < 0 or hours > 23 or minutes < 0 or minutes > 59 \
            or not math.isfinite(seconds) or seconds < 0.0 or seconds >= 60.0:
        raise EvaluationError(f"{label} is outside the valid clock range")
    return 3600.0 * hours + 60.0 * minutes + seconds


def aggregation_duration_seconds(
    manifest: Mapping[str, object], *, manifest_path: pathlib.Path,
) -> float:
    """Resolve and cross-check the full-session observation duration.

    New extractor manifests carry an explicit top-level duration.  Schema-2
    queue artifacts created during the R21 repair predate that compatibility
    field but contain three independent exact sources: session clocks, the
    queue-exposure audit, and the full-window target record.  Accept those
    artifacts only when every available source agrees; never guess a default.
    """
    candidates: dict[str, float] = {}
    explicit = manifest.get("aggregation_duration_seconds")
    if explicit is not None:
        candidates["aggregation_duration_seconds"] = finite_float(
            explicit,
            label=f"{manifest_path}:aggregation_duration_seconds",
        )

    start = manifest.get("session_start")
    end = manifest.get("session_end")
    if (start is None) != (end is None):
        raise EvaluationError(
            f"{manifest_path} must provide both session_start and session_end"
        )
    if start is not None:
        candidates["session_clock_difference"] = (
            clock_seconds(end, label=f"{manifest_path}:session_end")
            - clock_seconds(start, label=f"{manifest_path}:session_start")
        )

    queue = manifest.get("queue_reactive_training_artifacts")
    if isinstance(queue, Mapping):
        exposure = queue.get("exposure")
        if isinstance(exposure, Mapping) and exposure.get(
            "expected_session_seconds"
        ) is not None:
            candidates["queue_expected_session_seconds"] = finite_float(
                exposure.get("expected_session_seconds"),
                label=(
                    f"{manifest_path}:queue_reactive_training_artifacts:"
                    "exposure:expected_session_seconds"
                ),
            )

    if not candidates:
        raise EvaluationError(
            f"{manifest_path} has no auditable full-session duration"
        )
    for source, duration in candidates.items():
        if duration <= 0.0:
            raise EvaluationError(
                f"{manifest_path}:{source} must be positive"
            )
    reference = next(iter(candidates.values()))
    for source, duration in candidates.items():
        if abs(duration - reference) > 1.0e-9:
            raise EvaluationError(
                f"{manifest_path} has inconsistent full-session durations: "
                f"{candidates!r}"
            )

    windows = manifest.get("market_target_windows")
    if isinstance(windows, Mapping) and float(reference).is_integer():
        record = windows.get(str(int(reference)))
        if record is not None:
            if not isinstance(record, Mapping):
                raise EvaluationError(
                    f"{manifest_path}:market_target_windows full-session "
                    "record is malformed"
                )
            window_duration = finite_float(
                record.get("duration_seconds"),
                label=f"{manifest_path}:market_target_windows:duration_seconds",
            )
            if abs(window_duration - reference) > 1.0e-9:
                raise EvaluationError(
                    f"{manifest_path} full-window duration disagrees with session"
                )
    return reference


def exact_integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise EvaluationError(f"{label} is not an integer")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"{label} is not an integer") from error
    if str(value).strip() not in {str(parsed), f"{parsed}.0"}:
        raise EvaluationError(f"{label} is not an exact integer")
    if parsed < minimum:
        raise EvaluationError(f"{label} must be at least {minimum}")
    return parsed


def normalized_date(value: str, *, label: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise EvaluationError(f"{label} must be YYYY-MM-DD: {value!r}") from error
    return parsed.isoformat()


def normalized_symbol(value: object, *, label: str) -> str:
    symbol = str(value).strip().upper()
    if not symbol or any(character.isspace() for character in symbol):
        raise EvaluationError(f"{label} is not a valid non-empty symbol")
    return symbol


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise EvaluationError("percentile requires at least one observation")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def clipped_logit(value: float) -> float:
    probability = min(
        1.0 - ROBUST_PROBABILITY_EPSILON,
        max(ROBUST_PROBABILITY_EPSILON, value),
    )
    return math.log(probability / (1.0 - probability))


def robust_residual(metric: str, simulated: float, target: float) -> float:
    """Dimensionless residual used by the existing model-selection protocol."""
    if metric in POSITIVE_LOG_RATIO_METRICS:
        floor = 1.0e-12 if metric == "return_variance" else 1.0e-9
        return math.log(max(simulated, floor) / max(target, floor)) / (
            ROBUST_LOG_RATIO_UNIT
        )
    if metric == "mid_move_rate":
        return (clipped_logit(simulated) - clipped_logit(target)) / (
            ROBUST_MID_MOVE_LOG_ODDS_UNIT
        )
    if metric == "absolute_return_acf1":
        simulated_clipped = min(
            1.0 - ROBUST_PROBABILITY_EPSILON,
            max(-1.0 + ROBUST_PROBABILITY_EPSILON, simulated),
        )
        target_clipped = min(
            1.0 - ROBUST_PROBABILITY_EPSILON,
            max(-1.0 + ROBUST_PROBABILITY_EPSILON, target),
        )
        return (
            math.atanh(simulated_clipped) - math.atanh(target_clipped)
        ) / ROBUST_ACF_FISHER_UNIT
    if metric == "two_sided_sample_fraction":
        return (simulated - target) / ROBUST_COVERAGE_UNIT
    raise EvaluationError(f"no robust transform declared for {metric}")


def huber_loss(residual: float) -> float:
    absolute = abs(residual)
    if absolute <= HUBER_DELTA:
        return 0.5 * residual * residual
    return HUBER_DELTA * (absolute - 0.5 * HUBER_DELTA)


def residual_score(residuals: Sequence[float]) -> float:
    if not residuals:
        raise EvaluationError("a robust score cannot be computed from zero residuals")
    return math.sqrt(2.0 * statistics.fmean(huber_loss(x) for x in residuals))


def parse_key_path(value: str, *, option: str) -> tuple[str, pathlib.Path]:
    if "=" not in value:
        raise EvaluationError(f"{option} requires DATE=PATH, observed {value!r}")
    raw_day, raw_path = value.split("=", 1)
    day = normalized_date(raw_day, label=option)
    path = pathlib.Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise EvaluationError(f"{option} file does not exist: {path}")
    return day, path


def parse_summary_spec(value: str) -> SummarySpec:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise EvaluationError(
            "--sim-summary requires DATE:SEED=PATH, "
            f"observed {value!r}"
        )
    left, raw_path = value.split("=", 1)
    raw_day, raw_seed = left.rsplit(":", 1)
    day = normalized_date(raw_day, label="--sim-summary date")
    seed = exact_integer(raw_seed, label="--sim-summary seed", minimum=0)
    path = pathlib.Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise EvaluationError(f"--sim-summary file does not exist: {path}")
    return SummarySpec(day=day, seed=seed, path=path)


def read_cluster_map(path: pathlib.Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if "symbol" not in fields or "cluster_id" not in fields:
            raise EvaluationError(
                f"cluster map {path} requires symbol and cluster_id columns"
            )
        result: dict[str, str] = {}
        for line, row in enumerate(reader, start=2):
            symbol = normalized_symbol(row["symbol"], label=f"{path}:{line}:symbol")
            cluster = str(row["cluster_id"]).strip()
            if not cluster:
                raise EvaluationError(f"{path}:{line}:cluster_id is empty")
            if symbol in result:
                raise EvaluationError(f"duplicate cluster assignment for {symbol}")
            result[symbol] = cluster
    if not result:
        raise EvaluationError(f"cluster map is empty: {path}")
    return result


def resolve_row_path(config: pathlib.Path, value: object, *, label: str) -> pathlib.Path:
    raw = str(value).strip()
    if not raw:
        raise EvaluationError(f"{label} is empty")
    candidate = pathlib.Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = config.parent / candidate
    return candidate.resolve()


def unique_artifact(
    data_dir: pathlib.Path,
    *,
    glob_pattern: str,
    label: str,
    excluded_fragment: str | None = None,
) -> pathlib.Path:
    matches = sorted(
        path for path in data_dir.glob(glob_pattern)
        if excluded_fragment is None or excluded_fragment not in path.name
    )
    if len(matches) != 1:
        raise EvaluationError(
            f"{label} in {data_dir} must resolve to exactly one file; "
            f"observed {[path.name for path in matches]}"
        )
    return matches[0]


def validate_target_value(metric: str, value: float, *, label: str) -> None:
    if metric in POSITIVE_LOG_RATIO_METRICS and value < 0.0:
        raise EvaluationError(f"{label} must be nonnegative")
    if metric in {"mid_move_rate", "two_sided_sample_fraction"} and not (
        0.0 <= value <= 1.0
    ):
        raise EvaluationError(f"{label} must lie in [0, 1]")
    if metric == "absolute_return_acf1" and not (-1.0 < value < 1.0):
        raise EvaluationError(f"{label} must lie strictly between -1 and 1")


def exact_full_session_coverage(
    manifest: Mapping[str, object], *, manifest_path: pathlib.Path,
) -> Target:
    """Derive the coverage moment from exact extractor counters.

    Some already-extracted ITCH artifacts predate the explicit coverage row.
    They remain losslessly usable when their valid and invalid counters cover
    the complete declared aggregation horizon.  Missing or inconsistent
    counters are rejected rather than guessed.
    """
    duration = aggregation_duration_seconds(
        manifest, manifest_path=manifest_path,
    )
    valid = exact_integer(
        manifest.get("valid_snapshots"),
        label=f"{manifest_path}:valid_snapshots",
        minimum=0,
    )
    invalid = exact_integer(
        manifest.get("invalid_snapshots"),
        label=f"{manifest_path}:invalid_snapshots",
        minimum=0,
    )
    if valid + invalid != duration:
        raise EvaluationError(
            f"manifest {manifest_path} accounts for {valid + invalid} "
            f"snapshots, expected {duration}"
        )
    value = valid / duration
    scale = max(0.005, math.sqrt(value * (1.0 - value) / duration))
    return Target(value=value, scale=scale)


def load_target_artifacts(
    config_path: pathlib.Path,
    *,
    expected_date: str | None = None,
) -> tuple[dict[str, dict[str, Target]], list[dict[str, str]]]:
    """Load target moments by following each simulator config's data_dir."""
    with config_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if "symbol" not in fields or not fields.intersection({
            "target_data_dir", "empirical_data_dir", "data_dir",
        }):
            raise EvaluationError(
                f"target config {config_path} requires symbol and one of "
                "target_data_dir, empirical_data_dir or data_dir"
            )
        rows = list(reader)
    if not rows:
        raise EvaluationError(f"target config is empty: {config_path}")

    result: dict[str, dict[str, Target]] = {}
    provenance: list[dict[str, str]] = []
    for line, row in enumerate(rows, start=2):
        symbol = normalized_symbol(
            row["symbol"], label=f"{config_path}:{line}:symbol"
        )
        if symbol in result:
            raise EvaluationError(f"duplicate target symbol {symbol} in {config_path}")
        directory_columns = [
            column for column in (
                "target_data_dir", "empirical_data_dir", "data_dir",
            )
            if str(row.get(column, "")).strip()
        ]
        if not directory_columns:
            raise EvaluationError(
                f"{config_path}:{line} has no empirical target directory"
            )
        directory_column = directory_columns[0]
        data_dir = resolve_row_path(
            config_path,
            row[directory_column],
            label=f"{config_path}:{line}:{directory_column}",
        )
        if not data_dir.is_dir():
            raise EvaluationError(f"target data_dir is not a directory: {data_dir}")
        target_path = unique_artifact(
            data_dir,
            glob_pattern="market_targets_*.csv",
            excluded_fragment="_window_",
            label=f"full-session target for {symbol}",
        )
        manifest_path = unique_artifact(
            data_dir,
            glob_pattern="itch_manifest_*.json",
            label=f"manifest for {symbol}",
        )
        with target_path.open(newline="", encoding="utf-8") as handle:
            target_reader = csv.DictReader(handle)
            if not {"name", "target", "scale"}.issubset(
                set(target_reader.fieldnames or ())
            ):
                raise EvaluationError(
                    f"target file {target_path} requires name,target,scale"
                )
            values: dict[str, Target] = {}
            for target_line, target_row in enumerate(target_reader, start=2):
                metric = str(target_row["name"]).strip()
                if metric not in METRICS or metric == "background_event_rate":
                    continue
                if metric in values:
                    raise EvaluationError(
                        f"duplicate target {metric} in {target_path}:{target_line}"
                    )
                value = finite_float(
                    target_row["target"], label=f"{target_path}:{metric}:target"
                )
                scale = finite_float(
                    target_row["scale"], label=f"{target_path}:{metric}:scale"
                )
                if scale <= 0.0:
                    raise EvaluationError(f"{target_path}:{metric}:scale must be positive")
                if str(target_row.get("weight", "1")).strip():
                    weight = finite_float(
                        target_row.get("weight", "1"),
                        label=f"{target_path}:{metric}:weight",
                    )
                    if weight != 1.0:
                        raise EvaluationError(
                            f"{target_path}:{metric}:weight must equal one; "
                            "strict scoring gives every metric equal weight"
                        )
                validate_target_value(
                    metric, value, label=f"{target_path}:{metric}:target"
                )
                values[metric] = Target(value=value, scale=scale)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_symbol = normalized_symbol(
            manifest.get("symbol", ""), label=f"{manifest_path}:symbol"
        )
        if manifest_symbol != symbol:
            raise EvaluationError(
                f"manifest {manifest_path} names {manifest_symbol}, expected {symbol}"
            )
        if expected_date is not None:
            declared_date = manifest.get("trading_date")
            if not isinstance(declared_date, str):
                raise EvaluationError(
                    f"manifest {manifest_path} lacks trading_date; cannot prove "
                    f"that it supplies targets for {expected_date}"
                )
            try:
                normalized_manifest_date = normalized_date(
                    declared_date, label=f"{manifest_path}:trading_date"
                )
            except EvaluationError as error:
                raise EvaluationError(
                    f"manifest {manifest_path} does not identify a single-date "
                    f"target for {expected_date}: {declared_date!r}"
                ) from error
            if normalized_manifest_date != expected_date:
                raise EvaluationError(
                    f"manifest {manifest_path} targets {normalized_manifest_date}, "
                    f"expected {expected_date}"
                )
        authoritative_values = manifest.get("market_values")
        authoritative_scales = manifest.get("market_target_scales")
        if authoritative_values is not None or authoritative_scales is not None:
            if not isinstance(authoritative_values, Mapping) or not isinstance(
                authoritative_scales, Mapping
            ):
                raise EvaluationError(
                    f"manifest {manifest_path} has incomplete authoritative target maps"
                )
            authoritative_values = dict(authoritative_values)
            authoritative_scales = dict(authoritative_scales)
            coverage_metric = "two_sided_sample_fraction"
            has_coverage_value = coverage_metric in authoritative_values
            has_coverage_scale = coverage_metric in authoritative_scales
            if has_coverage_value != has_coverage_scale:
                raise EvaluationError(
                    f"manifest {manifest_path} has incomplete authoritative "
                    f"{coverage_metric} target or scale"
                )
            if not has_coverage_value:
                coverage = exact_full_session_coverage(
                    manifest, manifest_path=manifest_path,
                )
                authoritative_values[coverage_metric] = coverage.value
                authoritative_scales[coverage_metric] = coverage.scale
                if coverage_metric not in values:
                    values[coverage_metric] = coverage
            for metric, target in values.items():
                manifest_value = finite_float(
                    authoritative_values.get(metric),
                    label=f"{manifest_path}:market_values:{metric}",
                )
                manifest_scale = finite_float(
                    authoritative_scales.get(metric),
                    label=f"{manifest_path}:market_target_scales:{metric}",
                )
                if manifest_value != target.value or manifest_scale != target.scale:
                    raise EvaluationError(
                        f"target CSV and manifest disagree for {symbol}:{metric}"
                    )
        counts = manifest.get("distribution_observation_counts")
        if not isinstance(counts, Mapping):
            raise EvaluationError(
                f"manifest {manifest_path} lacks distribution_observation_counts"
            )
        event_count = sum(
            exact_integer(
                counts.get(event),
                label=f"{manifest_path}:{event} event count",
                minimum=0,
            )
            for event in BACKGROUND_EVENTS
        )
        duration = aggregation_duration_seconds(
            manifest, manifest_path=manifest_path,
        )
        background_target = event_count / duration
        background_scale = max(
            1.0e-6,
            math.sqrt(event_count) / duration,
            0.05 * background_target,
        )
        values["background_event_rate"] = Target(
            value=background_target, scale=background_scale
        )
        missing = sorted(set(METRICS).difference(values))
        if missing:
            raise EvaluationError(
                f"target artifacts for {symbol} lack metrics: {missing}"
            )
        result[symbol] = values
        provenance.append({
            "symbol": symbol,
            "trading_date": expected_date or str(
                manifest.get("trading_date", "")
            ),
            "target_file": str(target_path),
            "target_sha256": sha256(target_path),
            "manifest_file": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
        })
    return result, provenance


def load_summary(path: pathlib.Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {
            "symbol", "sample_count", "expected_sample_count",
            "invalid_sample_count", "structurally_valid", *METRICS,
        }
        missing = sorted(required.difference(fields))
        if missing:
            raise EvaluationError(f"summary {path} lacks columns: {missing}")
        result: dict[str, dict[str, float]] = {}
        for line, row in enumerate(reader, start=2):
            symbol = normalized_symbol(row["symbol"], label=f"{path}:{line}:symbol")
            if symbol in result:
                raise EvaluationError(f"duplicate summary symbol {symbol} in {path}")
            samples = exact_integer(
                row["sample_count"], label=f"{path}:{symbol}:sample_count", minimum=1
            )
            expected = exact_integer(
                row["expected_sample_count"],
                label=f"{path}:{symbol}:expected_sample_count",
                minimum=1,
            )
            invalid = exact_integer(
                row["invalid_sample_count"],
                label=f"{path}:{symbol}:invalid_sample_count",
                minimum=0,
            )
            structural = exact_integer(
                row["structurally_valid"],
                label=f"{path}:{symbol}:structurally_valid",
                minimum=0,
            )
            if samples != expected or invalid != 0 or structural != 1:
                raise EvaluationError(
                    f"summary {path} has incomplete/invalid fixed-clock state for "
                    f"{symbol}: samples={samples}, expected={expected}, "
                    f"invalid={invalid}, structurally_valid={structural}"
                )
            metrics = {
                metric: finite_float(
                    row[metric], label=f"{path}:{symbol}:{metric}"
                )
                for metric in METRICS
            }
            for metric, value in metrics.items():
                validate_target_value(
                    metric, value, label=f"{path}:{symbol}:{metric}"
                )
            result[symbol] = metrics
    if not result:
        raise EvaluationError(f"summary is empty: {path}")
    return result


def evaluate_date(
    *,
    role: str,
    gate_protocol: str,
    day: str,
    targets: Mapping[str, Mapping[str, Target]],
    summaries: Sequence[tuple[int, pathlib.Path, Mapping[str, Mapping[str, float]]]],
    clusters: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    symbols = tuple(sorted(targets))
    expected_symbols = set(symbols)
    if set(clusters) != expected_symbols:
        missing = sorted(expected_symbols.difference(clusters))
        extra = sorted(set(clusters).difference(expected_symbols))
        raise EvaluationError(
            f"cluster universe differs from {day} target universe; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    if not summaries:
        raise EvaluationError(f"date {day} has no simulator summaries")
    for seed, path, summary in summaries:
        if set(summary) != expected_symbols:
            missing = sorted(expected_symbols.difference(summary))
            extra = sorted(set(summary).difference(expected_symbols))
            raise EvaluationError(
                f"summary universe differs for {day} seed {seed} ({path}); "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

    gate_metric_sequence = (
        METRICS
        if gate_protocol == STRICT_NINE_GATE
        else SIX_COMPONENT_METRICS
    )
    gate_metrics = frozenset(gate_metric_sequence)
    symbol_rows: list[dict[str, object]] = []
    residuals_by_metric: dict[str, list[float]] = {
        metric: [] for metric in REPORTED_METRICS
    }
    residuals_by_cluster_metric: dict[tuple[str, str], list[float]] = {}
    target_by_metric: dict[str, list[float]] = {
        metric: [] for metric in REPORTED_METRICS
    }
    simulated_by_metric: dict[str, list[float]] = {
        metric: [] for metric in REPORTED_METRICS
    }
    for symbol in symbols:
        cluster = clusters[symbol]
        for metric in METRICS:
            target = targets[symbol][metric].value
            simulated = statistics.fmean(
                summary[symbol][metric] for _, _, summary in summaries
            )
            residual = robust_residual(metric, simulated, target)
            within = abs(residual) <= GROSS_RESIDUAL_LIMIT + 1.0e-12
            row = {
                "role": role,
                "date": day,
                "symbol": symbol,
                "cluster_id": cluster,
                "metric": metric,
                "target": target,
                "simulated_seed_mean": simulated,
                "robust_residual": residual,
                "absolute_robust_residual": abs(residual),
                "within_gross_residual_limit": within,
                "seed_count": len(summaries),
            }
            symbol_rows.append(row)
            residuals_by_metric[metric].append(residual)
            residuals_by_cluster_metric.setdefault((cluster, metric), []).append(
                residual
            )
            target_by_metric[metric].append(target)
            simulated_by_metric[metric].append(simulated)

        combined_target = (
            targets[symbol]["mean_bid_depth"].value
            + targets[symbol]["mean_ask_depth"].value
        )
        combined_simulated = statistics.fmean(
            summary[symbol]["mean_bid_depth"]
            + summary[symbol]["mean_ask_depth"]
            for _, _, summary in summaries
        )
        combined_residual = robust_residual(
            COMBINED_DEPTH_METRIC,
            combined_simulated,
            combined_target,
        )
        combined_within = (
            abs(combined_residual) <= GROSS_RESIDUAL_LIMIT + 1.0e-12
        )
        symbol_rows.append({
            "role": role,
            "date": day,
            "symbol": symbol,
            "cluster_id": cluster,
            "metric": COMBINED_DEPTH_METRIC,
            "target": combined_target,
            "simulated_seed_mean": combined_simulated,
            "robust_residual": combined_residual,
            "absolute_robust_residual": abs(combined_residual),
            "within_gross_residual_limit": combined_within,
            "seed_count": len(summaries),
        })
        residuals_by_metric[COMBINED_DEPTH_METRIC].append(combined_residual)
        residuals_by_cluster_metric.setdefault(
            (cluster, COMBINED_DEPTH_METRIC), []
        ).append(combined_residual)
        target_by_metric[COMBINED_DEPTH_METRIC].append(combined_target)
        simulated_by_metric[COMBINED_DEPTH_METRIC].append(combined_simulated)

    metric_rows: list[dict[str, object]] = []
    failure_reasons: list[str] = []
    diagnostic_warnings: list[str] = []
    metric_losses: dict[str, float] = {}
    for metric in REPORTED_METRICS:
        values = residuals_by_metric[metric]
        score = residual_score(values)
        mean_loss = statistics.fmean(huber_loss(value) for value in values)
        metric_losses[metric] = mean_loss
        within_fraction = sum(
            abs(value) <= GROSS_RESIDUAL_LIMIT + 1.0e-12 for value in values
        ) / len(values)
        score_pass = score <= MAX_MARKETWIDE_METRIC_SCORE + 1.0e-12
        coverage_pass = (
            within_fraction
            >= MIN_PER_METRIC_SYMBOL_FRACTION_WITHIN_LIMIT - 1.0e-12
        )
        contributes = metric in gate_metrics
        issue_sink = failure_reasons if contributes else diagnostic_warnings
        if not score_pass:
            issue_sink.append(
                f"{metric} market-wide score {score:.6g} exceeds "
                f"{MAX_MARKETWIDE_METRIC_SCORE:g}"
            )
        if not coverage_pass:
            issue_sink.append(
                f"{metric} symbol coverage {within_fraction:.6g} is below "
                f"{MIN_PER_METRIC_SYMBOL_FRACTION_WITHIN_LIMIT:g}"
            )
        if contributes:
            gate_role = "primary"
        elif metric == "two_sided_sample_fraction":
            gate_role = "structural_diagnostic"
        else:
            gate_role = "diagnostic"
        metric_rows.append({
            "role": role,
            "date": day,
            "metric": metric,
            "gate_role": gate_role,
            "contributes_to_pass": contributes,
            "score": score,
            "maximum_score": MAX_MARKETWIDE_METRIC_SCORE,
            "score_passed": score_pass,
            "symbol_count": len(values),
            "fraction_within_residual_limit": within_fraction,
            "minimum_fraction_within_residual_limit": (
                MIN_PER_METRIC_SYMBOL_FRACTION_WITHIN_LIMIT
            ),
            "coverage_passed": coverage_pass,
            "target_mean": statistics.fmean(target_by_metric[metric]),
            "simulated_mean": statistics.fmean(simulated_by_metric[metric]),
            "target_median": percentile(target_by_metric[metric], 0.5),
            "simulated_median": percentile(simulated_by_metric[metric], 0.5),
            "target_p90": percentile(target_by_metric[metric], 0.9),
            "simulated_p90": percentile(simulated_by_metric[metric], 0.9),
        })

    marketwide_score = math.sqrt(2.0 * statistics.fmean(
        metric_losses[metric] for metric in gate_metric_sequence
    ))
    marketwide_pass = (
        marketwide_score <= MAX_MARKETWIDE_ROBUST_SCORE + 1.0e-12
    )
    if not marketwide_pass:
        failure_reasons.append(
            f"market-wide robust score {marketwide_score:.6g} exceeds "
            f"{MAX_MARKETWIDE_ROBUST_SCORE:g}"
        )

    cluster_rows: list[dict[str, object]] = []
    for cluster in sorted(set(clusters.values())):
        for metric in CLUSTER_GATE_METRICS:
            values = residuals_by_cluster_metric[(cluster, metric)]
            score = residual_score(values)
            passed = score <= MAX_CLUSTER_METRIC_SCORE + 1.0e-12
            if not passed:
                sink = (
                    failure_reasons
                    if gate_protocol == STRICT_NINE_GATE
                    else diagnostic_warnings
                )
                sink.append(
                    f"cluster {cluster} {metric} score {score:.6g} exceeds "
                    f"{MAX_CLUSTER_METRIC_SCORE:g}"
                )
            cluster_rows.append({
                "role": role,
                "date": day,
                "cluster_id": cluster,
                "metric": metric,
                "symbol_count": len(values),
                "score": score,
                "maximum_score": MAX_CLUSTER_METRIC_SCORE,
                "passed": passed,
                "gate_role": (
                    "primary"
                    if gate_protocol == STRICT_NINE_GATE
                    else "diagnostic"
                ),
            })

    failed_symbols = {
        str(row["symbol"])
        for row in symbol_rows
        if row["metric"] in gate_metrics
        and row["within_gross_residual_limit"] is False
    }
    gross_failure_fraction = len(failed_symbols) / len(symbols)
    gross_failure_pass = (
        gross_failure_fraction
        <= MAX_SYMBOL_ANY_GROSS_FAILURE_FRACTION + 1.0e-12
    )
    if not gross_failure_pass:
        failure_reasons.append(
            f"any-metric gross-failure fraction {gross_failure_fraction:.6g} "
            f"exceeds {MAX_SYMBOL_ANY_GROSS_FAILURE_FRACTION:g}"
        )

    acf_target = target_by_metric["absolute_return_acf1"]
    acf_simulated = simulated_by_metric["absolute_return_acf1"]
    acf_rows: list[dict[str, object]] = []
    for statistic_name, probability, limit in (
        ("mean", None, MAX_ACF_MEAN_ABSOLUTE_ERROR),
        ("median", 0.5, MAX_ACF_MEDIAN_ABSOLUTE_ERROR),
        ("p90", 0.9, MAX_ACF_P90_ABSOLUTE_ERROR),
    ):
        if probability is None:
            empirical_stat = statistics.fmean(acf_target)
            simulated_stat = statistics.fmean(acf_simulated)
        else:
            empirical_stat = percentile(acf_target, probability)
            simulated_stat = percentile(acf_simulated, probability)
        absolute_error = abs(simulated_stat - empirical_stat)
        passed = absolute_error <= limit + 1.0e-12
        if not passed:
            sink = (
                failure_reasons
                if gate_protocol == STRICT_NINE_GATE
                else diagnostic_warnings
            )
            sink.append(
                f"absolute-return ACF {statistic_name} error "
                f"{absolute_error:.6g} exceeds {limit:g}"
            )
        acf_rows.append({
            "role": role,
            "date": day,
            "statistic": statistic_name,
            "empirical": empirical_stat,
            "simulated": simulated_stat,
            "absolute_error": absolute_error,
            "maximum_absolute_error": limit,
            "passed": passed,
        })

    passed = not failure_reasons
    report = {
        "date": day,
        "gate_protocol": gate_protocol,
        "passed": passed,
        "symbol_count": len(symbols),
        "cluster_count": len(set(clusters.values())),
        "seeds": [seed for seed, _, _ in summaries],
        "summary_files": [
            {"seed": seed, "path": str(path), "sha256": sha256(path)}
            for seed, path, _ in summaries
        ],
        "marketwide_robust_score": marketwide_score,
        "marketwide_robust_score_passed": marketwide_pass,
        "gross_failure_symbol_count": len(failed_symbols),
        "gross_failure_symbol_fraction": gross_failure_fraction,
        "gross_failure_fraction_passed": gross_failure_pass,
        "failure_reasons": failure_reasons,
        "diagnostic_warnings": diagnostic_warnings,
        "primary_metrics": list(gate_metric_sequence),
    }
    diagnostics = {
        "symbol_residuals": symbol_rows,
        "metric_scores": metric_rows,
        "cluster_metric_scores": cluster_rows,
        "acf_distribution": acf_rows,
    }
    return report, diagnostics


def write_csv(
    path: pathlib.Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def gate_definition(gate_protocol: str) -> dict[str, object]:
    if gate_protocol not in GATE_PROTOCOLS:
        raise EvaluationError(f"unknown gate protocol: {gate_protocol}")
    primary_metrics = (
        list(METRICS)
        if gate_protocol == STRICT_NINE_GATE
        else list(SIX_COMPONENT_METRICS)
    )
    return {
        "gate_id": GATE_IDS[gate_protocol],
        "gate_protocol": gate_protocol,
        "reported_metric_set": list(REPORTED_METRICS),
        "primary_metric_set": primary_metrics,
        "combined_depth_definition": (
            "per-symbol bid-plus-ask top depth before robust scoring"
        ),
        "cluster_gate_metrics": (
            list(CLUSTER_GATE_METRICS)
            if gate_protocol == STRICT_NINE_GATE else []
        ),
        "diagnostic_metrics": (
            [] if gate_protocol == STRICT_NINE_GATE else [
                "mean_bid_depth",
                "mean_ask_depth",
                "return_kurtosis",
                "two_sided_sample_fraction",
                "cluster_metric_scores",
                "absolute_return_acf_distribution_moments",
            ]
        ),
        "structural_contract": (
            "every symbol/seed must have the complete expected fixed-clock "
            "sample count, zero invalid samples and structurally_valid=1"
        ),
        "marketwide_robust_score_maximum": MAX_MARKETWIDE_ROBUST_SCORE,
        "each_marketwide_metric_score_maximum": MAX_MARKETWIDE_METRIC_SCORE,
        "each_cluster_metric_score_maximum": MAX_CLUSTER_METRIC_SCORE,
        "robust_residual_absolute_limit": GROSS_RESIDUAL_LIMIT,
        "per_metric_symbol_fraction_within_limit_minimum": (
            MIN_PER_METRIC_SYMBOL_FRACTION_WITHIN_LIMIT
        ),
        "symbol_any_gross_failure_fraction_maximum": (
            MAX_SYMBOL_ANY_GROSS_FAILURE_FRACTION
        ),
        "absolute_return_acf_mean_absolute_error_maximum": (
            MAX_ACF_MEAN_ABSOLUTE_ERROR
        ),
        "absolute_return_acf_median_absolute_error_maximum": (
            MAX_ACF_MEDIAN_ABSOLUTE_ERROR
        ),
        "absolute_return_acf_p90_absolute_error_maximum": (
            MAX_ACF_P90_ABSOLUTE_ERROR
        ),
        "huber_delta": HUBER_DELTA,
        "all_dates_must_pass_separately": True,
        "thresholds_are_command_line_overridable": False,
        "protocol_revision_classification": (
            "original_predeclared_gate"
            if gate_protocol == STRICT_NINE_GATE
            else "retrospective_development_protocol_revision"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-role", choices=ROLES, required=True)
    parser.add_argument(
        "--gate-protocol", choices=GATE_PROTOCOLS,
        default=STRICT_NINE_GATE,
        help=(
            "versioned adequacy protocol; strict-nine-v1 remains the "
            "backward-compatible default"
        ),
    )
    parser.add_argument(
        "--target-config", action="append", required=True, metavar="DATE=PATH",
        help="repeat once per empirical date",
    )
    parser.add_argument(
        "--sim-summary", action="append", required=True,
        metavar="DATE:SEED=PATH", help="repeat once per date and seed",
    )
    parser.add_argument("--cluster-map", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--expected-date", action="append", default=[], metavar="YYYY-MM-DD",
        help=(
            "predeclared required dates; training_fit requires at least two and "
            "every listed date must be supplied and pass"
        ),
    )
    parser.add_argument(
        "--expected-cluster-count", type=int, default=10,
        help="exact cluster count required by the frozen mapping (default: 10)",
    )
    parser.add_argument(
        "--protocol-freeze-record", type=pathlib.Path,
        help="required only for untouched_final_holdout; hashed into the report",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    role = str(args.evaluation_role)
    gate_protocol = str(args.gate_protocol)
    if args.expected_cluster_count <= 0:
        raise EvaluationError("--expected-cluster-count must be positive")
    expected_dates = tuple(
        normalized_date(value, label="--expected-date")
        for value in args.expected_date
    )
    if len(set(expected_dates)) != len(expected_dates):
        raise EvaluationError("--expected-date contains duplicates")
    if role == "training_fit" and len(expected_dates) < 2:
        raise EvaluationError(
            "training_fit requires at least two explicit --expected-date values; "
            "every training date is gated separately"
        )
    if not expected_dates:
        raise EvaluationError("at least one --expected-date is required")

    freeze_record: dict[str, object] | None = None
    if role == "untouched_final_holdout":
        if args.protocol_freeze_record is None:
            raise EvaluationError(
                "untouched_final_holdout requires --protocol-freeze-record"
            )
        freeze_path = args.protocol_freeze_record.expanduser().resolve()
        if not freeze_path.is_file() or freeze_path.stat().st_size == 0:
            raise EvaluationError(
                f"protocol freeze record is missing or empty: {freeze_path}"
            )
        freeze_record = {
            "path": str(freeze_path),
            "sha256": sha256(freeze_path),
            "semantic_claim": (
                "caller asserts this record predates inspection of the final "
                "holdout; the evaluator records but cannot prove that fact"
            ),
        }
    elif args.protocol_freeze_record is not None:
        raise EvaluationError(
            "--protocol-freeze-record is only valid for untouched_final_holdout"
        )

    config_specs = [
        parse_key_path(value, option="--target-config")
        for value in args.target_config
    ]
    target_configs: dict[str, pathlib.Path] = {}
    for day, path in config_specs:
        if day in target_configs:
            raise EvaluationError(f"duplicate target config date {day}")
        target_configs[day] = path
    summaries_by_date: dict[str, list[SummarySpec]] = {}
    seen_summary_keys: set[tuple[str, int]] = set()
    for raw in args.sim_summary:
        spec = parse_summary_spec(raw)
        key = (spec.day, spec.seed)
        if key in seen_summary_keys:
            raise EvaluationError(
                f"duplicate simulator summary date/seed {spec.day}:{spec.seed}"
            )
        seen_summary_keys.add(key)
        summaries_by_date.setdefault(spec.day, []).append(spec)

    expected_set = set(expected_dates)
    if set(target_configs) != expected_set:
        raise EvaluationError(
            "target-config dates must equal expected dates; "
            f"expected={sorted(expected_set)}, observed={sorted(target_configs)}"
        )
    if set(summaries_by_date) != expected_set:
        raise EvaluationError(
            "sim-summary dates must equal expected dates; "
            f"expected={sorted(expected_set)}, observed={sorted(summaries_by_date)}"
        )

    cluster_path = args.cluster_map.expanduser().resolve()
    if not cluster_path.is_file():
        raise EvaluationError(f"cluster map does not exist: {cluster_path}")
    clusters = read_cluster_map(cluster_path)
    observed_cluster_count = len(set(clusters.values()))
    if observed_cluster_count != args.expected_cluster_count:
        raise EvaluationError(
            f"cluster map has {observed_cluster_count} clusters; expected "
            f"{args.expected_cluster_count}"
        )

    date_reports: list[dict[str, object]] = []
    target_provenance: list[dict[str, object]] = []
    diagnostics: dict[str, list[dict[str, object]]] = {
        "symbol_residuals": [],
        "metric_scores": [],
        "cluster_metric_scores": [],
        "acf_distribution": [],
    }
    for day in expected_dates:
        config_path = target_configs[day]
        targets, target_files = load_target_artifacts(
            config_path, expected_date=day,
        )
        target_provenance.append({
            "date": day,
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "artifacts": target_files,
        })
        loaded_summaries = []
        for spec in sorted(summaries_by_date[day], key=lambda value: value.seed):
            loaded_summaries.append((spec.seed, spec.path, load_summary(spec.path)))
        report, date_diagnostics = evaluate_date(
            role=role,
            gate_protocol=gate_protocol,
            day=day,
            targets=targets,
            summaries=loaded_summaries,
            clusters=clusters,
        )
        date_reports.append(report)
        for name, rows in date_diagnostics.items():
            diagnostics[name].extend(rows)

    passed = all(bool(report["passed"]) for report in date_reports)
    if role == "development_validation":
        interpretation = (
            "development validation passed the predeclared adequacy gate; this "
            "is not certification and may inform model revision"
            if passed else
            "development validation failed the predeclared adequacy gate; this "
            "is not certification and the failures must remain visible"
        )
    elif role == "training_fit":
        interpretation = (
            "every training date passed separately"
            if passed else "at least one training date failed"
        )
    else:
        interpretation = (
            "the declared untouched final holdout passed"
            if passed else "the declared untouched final holdout failed"
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": gate_definition(gate_protocol),
        "gate_protocol": gate_protocol,
        "evaluation_role": role,
        "result_label": ROLE_LABELS[role] if passed else f"{role}_failed",
        "passed": passed,
        "certification_claimed": False,
        "interpretation": interpretation,
        "expected_dates": list(expected_dates),
        "all_dates_passed_separately": passed,
        "date_results": date_reports,
        "cluster_mapping": {
            "path": str(cluster_path),
            "sha256": sha256(cluster_path),
            "symbol_count": len(clusters),
            "cluster_count": observed_cluster_count,
        },
        "target_provenance": target_provenance,
        "protocol_freeze_record": freeze_record,
        "diagnostic_files": {
            "date_gate_results_csv": "date_gate_results.csv",
            "marketwide_metric_scores_csv": "marketwide_metric_scores.csv",
            "cluster_metric_scores_csv": "cluster_metric_scores.csv",
            "symbol_residuals_csv": "symbol_residuals.csv",
            "absolute_return_acf_distribution_csv": (
                "absolute_return_acf_distribution.csv"
            ),
        },
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "date_gate_results.csv",
        [{
            **row,
            "failure_reasons": json.dumps(
                row["failure_reasons"], separators=(",", ":")
            ),
            "diagnostic_warnings": json.dumps(
                row["diagnostic_warnings"], separators=(",", ":")
            ),
        } for row in date_reports],
        (
            "date", "gate_protocol", "passed", "symbol_count",
            "cluster_count", "seeds",
            "marketwide_robust_score", "marketwide_robust_score_passed",
            "gross_failure_symbol_count", "gross_failure_symbol_fraction",
            "gross_failure_fraction_passed", "failure_reasons",
            "diagnostic_warnings",
        ),
    )
    write_csv(
        output_dir / "marketwide_metric_scores.csv",
        diagnostics["metric_scores"],
        (
            "role", "date", "metric", "gate_role",
            "contributes_to_pass", "score", "maximum_score",
            "score_passed", "symbol_count", "fraction_within_residual_limit",
            "minimum_fraction_within_residual_limit", "coverage_passed",
            "target_mean", "simulated_mean", "target_median",
            "simulated_median", "target_p90", "simulated_p90",
        ),
    )
    write_csv(
        output_dir / "cluster_metric_scores.csv",
        diagnostics["cluster_metric_scores"],
        (
            "role", "date", "cluster_id", "metric", "symbol_count", "score",
            "maximum_score", "passed", "gate_role",
        ),
    )
    write_csv(
        output_dir / "symbol_residuals.csv",
        diagnostics["symbol_residuals"],
        (
            "role", "date", "symbol", "cluster_id", "metric", "target",
            "simulated_seed_mean", "robust_residual",
            "absolute_robust_residual", "within_gross_residual_limit",
            "seed_count",
        ),
    )
    write_csv(
        output_dir / "absolute_return_acf_distribution.csv",
        diagnostics["acf_distribution"],
        (
            "role", "date", "statistic", "empirical", "simulated",
            "absolute_error", "maximum_absolute_error", "passed",
        ),
    )
    report_path = output_dir / "strict_validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (EvaluationError, OSError, json.JSONDecodeError) as error:
        print(f"strict validation input error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "evaluation_role": report["evaluation_role"],
        "gate_protocol": report["gate_protocol"],
        "passed": report["passed"],
        "result_label": report["result_label"],
        "output": str(args.output_dir.expanduser().resolve()),
    }, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
