#!/usr/bin/env python3
"""Create deterministic liquidity clusters from an empirical-universe config.

This utility consumes the *derived* per-symbol calibration artifacts already
referenced by an all-universe ``MultiAssetBookConfig`` CSV.  It never reads a
raw ITCH archive and does not alter the simulator configuration.  The five
direct empirical features are:

* ``event_rate_per_second``: the six visible-flow counts in the extractor
  manifest divided by the regular-session duration;
* ``mean_spread_ticks``;
* ``mean_top_depth``: mean of empirical bid and ask top-of-book depth;
* ``return_variance``: variance of one-second log returns; and
* ``opening_mid_price_ticks`` from the empirical configuration.

Positive heavy-tailed variables are log-transformed, then standardized across
the accepted universe.  A dependency-free deterministic farthest-first
initialised Lloyd k-means implementation creates ten clusters by default.  A
deterministic minimum-size repair can prevent undersized clusters when the
downstream protocol needs disjoint training and validation samples.  A
cluster representative is the empirical book nearest its final centroid.  A
seeded, stratified validation sample is selected independently from the
representative in every cluster.

Example::

    python3 scripts/cluster_empirical_universe.py \
      --universe-config /path/nasdaq_common_plus_qqq_20200130.csv \
      --data-root /path/assembled_data \
      --output-dir results/liquidity_clusters

The output directory contains ``cluster_assignments.csv``,
``validation_sample.csv`` and ``cluster_manifest.json``.  The first CSV also
marks the ten representative symbols, which are the intended small set for
cluster-level behavioural calibration.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


CONFIG_REQUIRED_FIELDS = (
    "book_id",
    "symbol",
    "data_dir",
    "fundamental_price_ticks",
)
EVENT_NAMES = (
    "limit_buy",
    "limit_sell",
    "market_buy",
    "market_sell",
    "cancel_bid",
    "cancel_ask",
)
TARGET_NAMES = (
    "mean_spread_ticks",
    "mean_bid_depth",
    "mean_ask_depth",
    "return_variance",
)
RAW_FEATURE_NAMES = (
    "event_rate_per_second",
    "mean_spread_ticks",
    "mean_top_depth",
    "return_variance",
    "opening_mid_price_ticks",
)
TRANSFORMED_FEATURE_NAMES = (
    "log_event_rate_per_second",
    "log_mean_spread_ticks",
    "log_mean_top_depth",
    "log_return_variance",
    "log_opening_mid_price_ticks",
)
STANDARDIZED_FEATURE_NAMES = (
    "z_event_rate_per_second",
    "z_mean_spread_ticks",
    "z_mean_top_depth",
    "z_return_variance",
    "z_opening_mid_price_ticks",
)
ASSIGNMENT_FIELDS = (
    "book_id",
    "symbol",
    "data_dir",
    "cluster_id",
    "cluster_label",
    "liquidity_score",
    *RAW_FEATURE_NAMES,
    *TRANSFORMED_FEATURE_NAMES,
    *STANDARDIZED_FEATURE_NAMES,
    "distance_to_centroid",
    "is_representative",
    "is_validation_sample",
    "selection_role",
)
VALIDATION_FIELDS = (
    "cluster_id",
    "cluster_label",
    "book_id",
    "symbol",
    "data_dir",
    *RAW_FEATURE_NAMES,
    "distance_to_centroid",
)


class ClusterError(ValueError):
    """Raised for invalid empirical clustering inputs."""


@dataclass
class Observation:
    book_id: int
    symbol: str
    config_data_dir: str
    data_dir: pathlib.Path
    event_rate_per_second: float
    mean_spread_ticks: float
    mean_top_depth: float
    return_variance: float
    opening_mid_price_ticks: float
    transformed: tuple[float, ...] = field(default_factory=tuple)
    standardized: tuple[float, ...] = field(default_factory=tuple)
    cluster_id: int = -1
    liquidity_score: float = math.nan
    distance_to_centroid: float = math.nan
    is_representative: bool = False
    is_validation_sample: bool = False

    def raw_features(self) -> tuple[float, ...]:
        return (
            self.event_rate_per_second,
            self.mean_spread_ticks,
            self.mean_top_depth,
            self.return_variance,
            self.opening_mid_price_ticks,
        )


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not symbol:
        raise ClusterError("empty symbol")
    if any(character.isspace() for character in symbol) or "/" in symbol or "\\" in symbol:
        raise ClusterError(f"unsafe symbol: {value!r}")
    return symbol


def finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ClusterError(f"invalid {label}: {value!r}") from error
    if not math.isfinite(result):
        raise ClusterError(f"non-finite {label}: {value!r}")
    return result


def nonnegative_int(value: object, label: str) -> int:
    number = finite_float(value, label)
    if number < 0.0 or not number.is_integer():
        raise ClusterError(f"{label} must be a non-negative integer: {value!r}")
    return int(number)


def nonnegative_book_id(value: object, label: str) -> int:
    return nonnegative_int(value, label)


def positive_float(value: object, label: str) -> float:
    result = finite_float(value, label)
    if result <= 0.0:
        raise ClusterError(f"{label} must be positive: {value!r}")
    return result


def read_csv(path: pathlib.Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = tuple(reader.fieldnames or ())
            rows = [dict(row) for row in reader]
    except OSError as error:
        raise ClusterError(f"cannot read CSV {path}: {error}") from error
    if not fields:
        raise ClusterError(f"CSV has no header: {path}")
    return fields, rows


def require_columns(path: pathlib.Path,
                    fields: Iterable[str],
                    required: Iterable[str]) -> None:
    available = {field.strip() for field in fields}
    missing = sorted(set(required).difference(available))
    if missing:
        raise ClusterError(f"{path} is missing columns: {', '.join(missing)}")


def parse_clock_seconds(value: object, label: str) -> int:
    try:
        hour, minute, second = (int(piece) for piece in str(value).split(":"))
    except (TypeError, ValueError) as error:
        raise ClusterError(f"invalid {label}: {value!r}") from error
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ClusterError(f"invalid {label}: {value!r}")
    return 3600 * hour + 60 * minute + second


def resolve_data_dir(raw: str,
                     config_path: pathlib.Path,
                     data_root: pathlib.Path | None) -> pathlib.Path:
    """Resolve config paths while permitting a relocated extracted-data root."""
    configured = pathlib.Path(raw)
    candidates: list[pathlib.Path] = []
    if configured.is_absolute():
        candidates.append(configured)
    else:
        # An all-universe config usually carries absolute paths.  The remaining
        # cases support checked-in relative configs and a copied data directory.
        if data_root is not None:
            candidates.append(data_root / configured.name)
            candidates.append(data_root / configured)
        candidates.append(config_path.parent.parent / configured)
        candidates.append(pathlib.Path.cwd() / configured)

    seen: set[pathlib.Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir():
            return resolved
    rendered = ", ".join(str(path) for path in candidates)
    raise ClusterError(
        f"cannot resolve data_dir {raw!r} from {config_path}; checked: {rendered}"
    )


def manifest_path_for(directory: pathlib.Path, symbol: str) -> pathlib.Path:
    candidates = sorted(directory.glob(
        f"itch_manifest_{glob.escape(symbol.lower())}_*.json"
    ))
    if len(candidates) != 1:
        raise ClusterError(
            f"{directory} needs exactly one manifest for {symbol}; found {len(candidates)}"
        )
    return candidates[0]


def targets_path_for(directory: pathlib.Path, symbol: str) -> pathlib.Path:
    # The three-stage behavioural workflow may add short-prefix target CSVs
    # beside the full-session artifact.  Liquidity clustering intentionally
    # uses only the full-session direct empirical features.
    candidates = sorted(
        path for path in directory.glob(
            f"market_targets_{glob.escape(symbol.lower())}_*.csv"
        )
        if "_window_" not in path.name
    )
    if len(candidates) != 1:
        raise ClusterError(
            f"{directory} needs exactly one market-target CSV for {symbol}; found {len(candidates)}"
        )
    return candidates[0]


def load_event_rate(manifest_path: pathlib.Path) -> float:
    try:
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ClusterError(f"cannot read manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ClusterError(f"manifest is not an object: {manifest_path}")
    try:
        # A normal manifest covers one session.  A pooled empirical template
        # preserves those session bounds but records the total source duration
        # separately, so its aggregated event counts are not misread as a
        # single-day rate.
        if "aggregation_duration_seconds" in manifest:
            duration_value = finite_float(
                manifest["aggregation_duration_seconds"],
                f"{manifest_path}:aggregation_duration_seconds",
            )
            if not duration_value.is_integer():
                raise ClusterError(
                    f"{manifest_path}:aggregation_duration_seconds must be integral"
                )
            duration = int(duration_value)
        else:
            duration = (
                parse_clock_seconds(manifest["session_end"], "session_end")
                - parse_clock_seconds(manifest["session_start"], "session_start")
            )
    except KeyError as error:
        raise ClusterError(f"manifest lacks session boundary: {manifest_path}") from error
    if duration <= 0:
        raise ClusterError(f"manifest has non-positive session duration: {manifest_path}")
    counts = manifest.get("distribution_observation_counts")
    if not isinstance(counts, dict):
        raise ClusterError(f"manifest lacks distribution_observation_counts: {manifest_path}")
    total = 0
    for event in EVENT_NAMES:
        if event not in counts:
            raise ClusterError(f"manifest lacks {event} count: {manifest_path}")
        total += nonnegative_int(counts[event], f"{manifest_path}:{event}")
    if total <= 0:
        raise ClusterError(f"manifest has zero selected event count: {manifest_path}")
    return total / duration


def load_target_features(target_path: pathlib.Path) -> tuple[float, float, float]:
    fields, rows = read_csv(target_path)
    require_columns(target_path, fields, ("name", "target"))
    values: dict[str, float] = {}
    for line_number, row in enumerate(rows, start=2):
        name = row.get("name", "").strip()
        if name not in TARGET_NAMES:
            continue
        if name in values:
            raise ClusterError(f"duplicate target {name} in {target_path}:{line_number}")
        values[name] = finite_float(row.get("target", ""), f"{target_path}:{name}")
    missing = sorted(set(TARGET_NAMES).difference(values))
    if missing:
        raise ClusterError(f"{target_path} is missing targets: {', '.join(missing)}")
    spread = positive_float(values["mean_spread_ticks"], f"{target_path}:mean_spread_ticks")
    bid_depth = positive_float(values["mean_bid_depth"], f"{target_path}:mean_bid_depth")
    ask_depth = positive_float(values["mean_ask_depth"], f"{target_path}:mean_ask_depth")
    variance = finite_float(values["return_variance"], f"{target_path}:return_variance")
    if variance < 0.0:
        raise ClusterError(f"{target_path}:return_variance must be non-negative")
    return spread, 0.5 * (bid_depth + ask_depth), variance


def load_observations(config_path: pathlib.Path,
                      data_root: pathlib.Path | None) -> list[Observation]:
    fields, rows = read_csv(config_path)
    require_columns(config_path, fields, CONFIG_REQUIRED_FIELDS)
    observations: list[Observation] = []
    seen_symbols: set[str] = set()
    seen_book_ids: set[int] = set()
    for line_number, row in enumerate(rows, start=2):
        symbol = normalise_symbol(row.get("symbol", ""))
        if symbol in seen_symbols:
            raise ClusterError(f"duplicate symbol {symbol} in {config_path}:{line_number}")
        book_id = nonnegative_book_id(row.get("book_id", ""), f"{config_path}:{line_number}:book_id")
        if book_id in seen_book_ids:
            raise ClusterError(f"duplicate book_id {book_id} in {config_path}:{line_number}")
        data_text = row.get("data_dir", "").strip()
        if not data_text:
            raise ClusterError(f"empty data_dir in {config_path}:{line_number}")
        data_dir = resolve_data_dir(data_text, config_path, data_root)
        rate = load_event_rate(manifest_path_for(data_dir, symbol))
        spread, top_depth, variance = load_target_features(
            targets_path_for(data_dir, symbol)
        )
        price = positive_float(
            row.get("fundamental_price_ticks", ""),
            f"{config_path}:{line_number}:fundamental_price_ticks",
        )
        observations.append(Observation(
            book_id=book_id,
            symbol=symbol,
            config_data_dir=data_text,
            data_dir=data_dir,
            event_rate_per_second=rate,
            mean_spread_ticks=spread,
            mean_top_depth=top_depth,
            return_variance=variance,
            opening_mid_price_ticks=price,
        ))
        seen_symbols.add(symbol)
        seen_book_ids.add(book_id)
    if not observations:
        raise ClusterError(f"universe configuration is empty: {config_path}")
    return sorted(observations, key=lambda item: item.symbol)


def transform_and_standardize(observations: list[Observation]) -> dict[str, Any]:
    """Apply transparent log transforms and population z-standardization."""
    positive_variances = [item.return_variance for item in observations
                          if item.return_variance > 0.0]
    variance_floor = min(positive_variances) / 10.0 if positive_variances else 1e-16
    variance_floor = max(variance_floor, 1e-300)
    transformed: list[tuple[float, ...]] = []
    for item in observations:
        values = (
            math.log1p(item.event_rate_per_second),
            math.log1p(item.mean_spread_ticks),
            math.log1p(item.mean_top_depth),
            math.log(max(item.return_variance, variance_floor)),
            math.log(item.opening_mid_price_ticks),
        )
        if not all(math.isfinite(value) for value in values):
            raise ClusterError(f"non-finite transformed feature for {item.symbol}")
        transformed.append(values)
        item.transformed = values
    columns = list(zip(*transformed))
    means = [statistics.fmean(column) for column in columns]
    standard_deviations = [math.sqrt(statistics.fmean(
        (value - mean) ** 2 for value in column
    )) for column, mean in zip(columns, means)]
    for item in observations:
        item.standardized = tuple(
            0.0 if deviation <= 1e-15 else (value - mean) / deviation
            for value, mean, deviation in zip(
                item.transformed, means, standard_deviations
            )
        )
    return {
        "return_variance_floor": variance_floor,
        "transformed_feature_means": dict(zip(TRANSFORMED_FEATURE_NAMES, means)),
        "transformed_feature_population_standard_deviations": dict(
            zip(TRANSFORMED_FEATURE_NAMES, standard_deviations)
        ),
    }


def squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def mean_vector(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        raise ClusterError("cannot calculate a centroid from no vectors")
    return tuple(statistics.fmean(vector[index] for vector in vectors)
                 for index in range(len(vectors[0])))


def initial_centroids(observations: Sequence[Observation], clusters: int) -> list[tuple[float, ...]]:
    """Deterministic centre-first/farthest-first initialisation."""
    global_centroid = mean_vector([item.standardized for item in observations])
    first = min(
        range(len(observations)),
        key=lambda index: (squared_distance(observations[index].standardized, global_centroid),
                           observations[index].symbol),
    )
    chosen = [first]
    while len(chosen) < clusters:
        candidates = [index for index in range(len(observations)) if index not in chosen]
        if not candidates:
            raise ClusterError("not enough observations for requested cluster count")
        next_index = min(
            candidates,
            key=lambda index: (
                -min(squared_distance(
                    observations[index].standardized,
                    observations[chosen_index].standardized,
                ) for chosen_index in chosen),
                observations[index].symbol,
            ),
        )
        chosen.append(next_index)
    return [observations[index].standardized for index in chosen]


def assign_points(observations: Sequence[Observation],
                  centroids: Sequence[Sequence[float]]) -> list[int]:
    return [
        min(
            range(len(centroids)),
            key=lambda cluster_id: (
                squared_distance(item.standardized, centroids[cluster_id]),
                cluster_id,
            ),
        )
        for item in observations
    ]


def repair_empty_clusters(assignments: list[int],
                          observations: Sequence[Observation],
                          centroids: Sequence[Sequence[float]]) -> None:
    """Move deterministic boundary observations into empty clusters."""
    counts = [assignments.count(cluster_id) for cluster_id in range(len(centroids))]
    for empty_cluster in range(len(centroids)):
        if counts[empty_cluster] > 0:
            continue
        candidates = [
            index for index, assigned in enumerate(assignments)
            if counts[assigned] > 1
        ]
        if not candidates:
            raise ClusterError("unable to repair an empty cluster")
        selected = min(
            candidates,
            key=lambda index: (
                -squared_distance(
                    observations[index].standardized,
                    centroids[assignments[index]],
                ),
                observations[index].symbol,
            ),
        )
        source_cluster = assignments[selected]
        assignments[selected] = empty_cluster
        counts[source_cluster] -= 1
        counts[empty_cluster] += 1


def repair_minimum_cluster_sizes(
    assignments: list[int],
    observations: Sequence[Observation],
    centroids: Sequence[Sequence[float]],
    minimum_cluster_size: int,
) -> None:
    """Deterministically enforce a lower cluster-size bound after Lloyd convergence.

    The unconstrained optimum may isolate an extreme symbol in a tiny cluster.
    Certification needs disjoint training and development-validation samples,
    so the canonical run requires at least six members per cluster.  Each move
    is the least increase in squared-distance objective among donor clusters
    that remain above the lower bound; symbol and cluster identifiers provide
    deterministic tie breaks.
    """
    cluster_count = len(centroids)
    if minimum_cluster_size < 1:
        raise ClusterError("--minimum-cluster-size must be positive")
    if len(observations) < cluster_count * minimum_cluster_size:
        raise ClusterError(
            "accepted universe is too small for the requested cluster count and "
            f"minimum size: {len(observations)} < "
            f"{cluster_count}*{minimum_cluster_size}"
        )
    counts = [assignments.count(cluster_id) for cluster_id in range(cluster_count)]
    while min(counts) < minimum_cluster_size:
        destination = next(
            cluster_id for cluster_id, count in enumerate(counts)
            if count < minimum_cluster_size
        )
        candidates = [
            index for index, source in enumerate(assignments)
            if source != destination and counts[source] > minimum_cluster_size
        ]
        if not candidates:
            raise ClusterError("unable to enforce the minimum cluster size")
        selected = min(
            candidates,
            key=lambda index: (
                squared_distance(
                    observations[index].standardized, centroids[destination],
                ) - squared_distance(
                    observations[index].standardized, centroids[assignments[index]],
                ),
                squared_distance(
                    observations[index].standardized, centroids[destination],
                ),
                observations[index].symbol,
                assignments[index],
            ),
        )
        source = assignments[selected]
        assignments[selected] = destination
        counts[source] -= 1
        counts[destination] += 1


def cluster_observations(observations: list[Observation],
                         clusters: int,
                         max_iterations: int,
                         minimum_cluster_size: int = 1,
                         ) -> tuple[list[tuple[float, ...]], int]:
    if clusters < 1:
        raise ClusterError("--clusters must be positive")
    if clusters > len(observations):
        raise ClusterError(
            f"--clusters={clusters} exceeds accepted universe size {len(observations)}"
        )
    if max_iterations < 1:
        raise ClusterError("--max-iterations must be positive")
    if minimum_cluster_size < 1:
        raise ClusterError("--minimum-cluster-size must be positive")
    if len(observations) < clusters * minimum_cluster_size:
        raise ClusterError(
            "accepted universe is too small for the requested cluster count and "
            f"minimum size: {len(observations)} < {clusters}*{minimum_cluster_size}"
        )
    centroids = initial_centroids(observations, clusters)
    previous_assignments: list[int] | None = None
    for iteration in range(1, max_iterations + 1):
        assignments = assign_points(observations, centroids)
        repair_empty_clusters(assignments, observations, centroids)
        updated = [
            mean_vector([
                observations[index].standardized
                for index, assignment in enumerate(assignments)
                if assignment == cluster_id
            ])
            for cluster_id in range(clusters)
        ]
        if assignments == previous_assignments:
            centroids = updated
            break
        centroids = updated
        previous_assignments = assignments
    else:
        raise ClusterError(
            f"k-means did not converge after {max_iterations} iterations"
        )
    repair_minimum_cluster_sizes(
        assignments, observations, centroids, minimum_cluster_size,
    )
    centroids = [
        mean_vector([
            observations[index].standardized
            for index, assignment in enumerate(assignments)
            if assignment == cluster_id
        ])
        for cluster_id in range(clusters)
    ]
    for item, assignment in zip(observations, assignments):
        item.cluster_id = assignment
    return centroids, iteration


def remap_clusters_by_liquidity(observations: list[Observation],
                                centroids: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    """Give cluster identifiers an interpretable least-to-most-liquidity order."""
    # Price remains part of clustering but is intentionally omitted from this
    # descriptive ranking.  Lower spread, lower volatility, higher event rate
    # and higher displayed depth imply a larger score.
    scores = [center[0] - center[1] + center[2] - center[3]
              for center in centroids]
    old_order = sorted(
        range(len(centroids)),
        key=lambda old: (scores[old], tuple(centroids[old]), old),
    )
    mapping = {old: new for new, old in enumerate(old_order)}
    remapped = [centroids[old] for old in old_order]
    remapped_scores = [scores[old] for old in old_order]
    for item in observations:
        item.cluster_id = mapping[item.cluster_id]
        item.liquidity_score = remapped_scores[item.cluster_id]
        item.distance_to_centroid = math.sqrt(squared_distance(
            item.standardized, remapped[item.cluster_id]
        ))
    return remapped


def selection_hash(seed: int, cluster_id: int, symbol: str) -> str:
    payload = f"{seed}:{cluster_id}:{symbol}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_representatives_and_validation(observations: list[Observation],
                                          clusters: int,
                                          validation_per_cluster: int,
                                          seed: int) -> dict[int, dict[str, list[str] | str]]:
    if validation_per_cluster < 0:
        raise ClusterError("--validation-per-cluster must be non-negative")
    result: dict[int, dict[str, list[str] | str]] = {}
    for cluster_id in range(clusters):
        members = sorted(
            (item for item in observations if item.cluster_id == cluster_id),
            key=lambda item: (item.distance_to_centroid, item.symbol),
        )
        if not members:
            raise ClusterError(f"cluster {cluster_id} unexpectedly has no members")
        representative = members[0]
        representative.is_representative = True
        candidates = sorted(
            members[1:],
            key=lambda item: (selection_hash(seed, cluster_id, item.symbol), item.symbol),
        )
        if len(candidates) < validation_per_cluster:
            raise ClusterError(
                f"cluster {cluster_id} has only {len(candidates)} non-representative "
                f"members but {validation_per_cluster} validation symbols were requested"
            )
        validation = candidates[:validation_per_cluster]
        for item in validation:
            item.is_validation_sample = True
        result[cluster_id] = {
            "representative_symbol": representative.symbol,
            "validation_symbols": [item.symbol for item in validation],
        }
    return result


def format_number(value: float) -> str:
    return format(value, ".17g")


def assignment_row(item: Observation) -> dict[str, object]:
    raw = item.raw_features()
    row: dict[str, object] = {
        "book_id": item.book_id,
        "symbol": item.symbol,
        "data_dir": str(item.data_dir),
        "cluster_id": item.cluster_id,
        "cluster_label": f"liquidity_{item.cluster_id:02d}",
        "liquidity_score": format_number(item.liquidity_score),
    }
    row.update({name: format_number(value) for name, value in zip(RAW_FEATURE_NAMES, raw)})
    row.update({name: format_number(value) for name, value in zip(
        TRANSFORMED_FEATURE_NAMES, item.transformed
    )})
    row.update({name: format_number(value) for name, value in zip(
        STANDARDIZED_FEATURE_NAMES, item.standardized
    )})
    row["distance_to_centroid"] = format_number(item.distance_to_centroid)
    row["is_representative"] = int(item.is_representative)
    row["is_validation_sample"] = int(item.is_validation_sample)
    row["selection_role"] = (
        "representative" if item.is_representative else
        "validation" if item.is_validation_sample else
        "none"
    )
    return row


def validation_row(item: Observation) -> dict[str, object]:
    row = assignment_row(item)
    return {field: row[field] for field in VALIDATION_FIELDS}


def atomic_csv(path: pathlib.Path,
               fieldnames: Sequence[str],
               rows: Iterable[Mapping[str, object]],
               overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ClusterError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: pathlib.Path, value: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ClusterError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = pathlib.Path(args.universe_config).expanduser().resolve()
    if not config_path.is_file():
        raise ClusterError(f"--universe-config is not a file: {config_path}")
    data_root = (
        pathlib.Path(args.data_root).expanduser().resolve()
        if args.data_root else None
    )
    if data_root is not None and not data_root.is_dir():
        raise ClusterError(f"--data-root is not a directory: {data_root}")
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    if args.seed < 0:
        raise ClusterError("--seed must be non-negative")

    observations = load_observations(config_path, data_root)
    if args.clusters > len(observations):
        raise ClusterError(
            f"--clusters={args.clusters} exceeds accepted universe size {len(observations)}"
        )
    transform_metadata = transform_and_standardize(observations)
    centroids, iterations = cluster_observations(
        observations, args.clusters, args.max_iterations,
        args.minimum_cluster_size,
    )
    centroids = remap_clusters_by_liquidity(observations, centroids)
    selected = select_representatives_and_validation(
        observations, args.clusters, args.validation_per_cluster, args.seed
    )

    assignments_path = output_dir / "cluster_assignments.csv"
    validation_path = output_dir / "validation_sample.csv"
    manifest_path = output_dir / "cluster_manifest.json"
    output_paths = (assignments_path, validation_path, manifest_path)
    if not args.overwrite:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            raise ClusterError(
                "refusing to overwrite existing output(s): "
                + ", ".join(str(path) for path in existing)
            )

    ordered = sorted(observations, key=lambda item: item.book_id)
    assignment_rows = [assignment_row(item) for item in ordered]
    validation_rows = [validation_row(item) for item in ordered
                       if item.is_validation_sample]
    atomic_csv(assignments_path, ASSIGNMENT_FIELDS, assignment_rows, args.overwrite)
    atomic_csv(validation_path, VALIDATION_FIELDS, validation_rows, args.overwrite)

    cluster_records = []
    for cluster_id in range(args.clusters):
        members = [item for item in observations if item.cluster_id == cluster_id]
        centroid = centroids[cluster_id]
        cluster_records.append({
            "cluster_id": cluster_id,
            "cluster_label": f"liquidity_{cluster_id:02d}",
            "liquidity_score": observations[next(
                index for index, item in enumerate(observations)
                if item.cluster_id == cluster_id
            )].liquidity_score,
            "size": len(members),
            "representative_symbol": selected[cluster_id]["representative_symbol"],
            "validation_symbols": selected[cluster_id]["validation_symbols"],
            "centroid_standardized": dict(zip(
                STANDARDIZED_FEATURE_NAMES, centroid
            )),
        })

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "inputs": {
            "universe_config": str(config_path),
            "universe_config_sha256": sha256_file(config_path),
            "data_root": str(data_root) if data_root is not None else None,
            "raw_itch_input_used": False,
        },
        "features": {
            "raw": {
                "event_rate_per_second": (
                    "sum of six manifest distribution_observation_counts "
                    "divided by manifest regular-session seconds"
                ),
                "mean_spread_ticks": "market_targets mean_spread_ticks target",
                "mean_top_depth": (
                    "mean of market_targets mean_bid_depth and mean_ask_depth"
                ),
                "return_variance": "market_targets return_variance target",
                "opening_mid_price_ticks": (
                    "universe-config fundamental_price_ticks (empirical opening midpoint)"
                ),
            },
            "raw_feature_columns": list(RAW_FEATURE_NAMES),
            "transformed_feature_columns": list(TRANSFORMED_FEATURE_NAMES),
            "standardized_feature_columns": list(STANDARDIZED_FEATURE_NAMES),
            "transformation": {
                "event_rate_per_second": "log1p",
                "mean_spread_ticks": "log1p",
                "mean_top_depth": "log1p",
                "return_variance": "log(max(value, return_variance_floor))",
                "opening_mid_price_ticks": "log",
            },
            "standardization": "cross-sectional population z-score; zero-variance feature maps to zero",
            **transform_metadata,
        },
        "clustering": {
            "algorithm": (
                "deterministic_farthest_first_lloyd_kmeans_"
                "with_minimum_size_repair"
            ),
            "cluster_count": args.clusters,
            "minimum_cluster_size": args.minimum_cluster_size,
            "max_iterations": args.max_iterations,
            "converged_iterations": iterations,
            "cluster_identifier_order": (
                "ascending descriptive liquidity score: z(event rate) - z(spread) "
                "+ z(top depth) - z(return variance)"
            ),
            "validation_selection": (
                "seeded SHA-256 ordering within each cluster after removing its centroid-nearest representative"
            ),
            "seed": args.seed,
            "requested_validation_per_cluster": args.validation_per_cluster,
        },
        "counts": {
            "accepted_books": len(observations),
            "clusters": args.clusters,
            "representatives": sum(item.is_representative for item in observations),
            "validation_samples": len(validation_rows),
        },
        "clusters": cluster_records,
        "artifacts": {
            "cluster_assignments_csv": {
                "path": str(assignments_path),
                "sha256": sha256_file(assignments_path),
            },
            "validation_sample_csv": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
            },
        },
    }
    atomic_json(manifest_path, manifest, args.overwrite)
    return {
        "cluster_assignments": str(assignments_path),
        "validation_sample": str(validation_path),
        "manifest": str(manifest_path),
        "accepted_books": len(observations),
        "clusters": args.clusters,
        "validation_samples": len(validation_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--universe-config", required=True,
        help="all-universe MultiAssetBookConfig CSV produced by build_itch_universe_config.py",
    )
    parser.add_argument(
        "--data-root",
        help=(
            "optional root containing derived itch_YYYYMMDD_SYMBOL directories; "
            "needed when config data_dir paths are relative or relocated"
        ),
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="directory for cluster_assignments.csv, validation_sample.csv and cluster_manifest.json",
    )
    parser.add_argument(
        "--clusters", type=int, default=10,
        help="number of deterministic liquidity clusters (default: 10)",
    )
    parser.add_argument(
        "--validation-per-cluster", type=int, default=3,
        help="validation symbols sampled per cluster after its representative (default: 3)",
    )
    parser.add_argument(
        "--minimum-cluster-size", type=int, default=1,
        help=(
            "deterministic post-Lloyd lower bound on cluster membership; the "
            "certified 3-training/3-validation workflow uses 6 (default: 1)"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=20200130,
        help="non-negative seed for deterministic validation sampling (default: 20200130)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=100,
        help="maximum Lloyd k-means iterations (default: 100)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing artifacts in --output-dir",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except ClusterError as error:
        print(f"empirical-universe clustering failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
