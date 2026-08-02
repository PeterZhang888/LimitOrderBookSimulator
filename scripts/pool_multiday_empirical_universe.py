#!/usr/bin/env python3
"""Create one auditable multi-day empirical training template for the LOB ABM.

The fragmented-LOB model needs one direct-input directory per symbol at run
time: six Hawkes immigration rates, empirical order-size and price-distance
marks, a local quote-size proxy, and a quote-improvement probability.  A
single ITCH day is too noisy to call a general calibration template.  This
tool therefore pools *direct* inputs from several complete, chronologically
earlier ITCH sessions while retaining day-specific event flows, marks and
openings for behavioural-policy fitting.  The spread and queue-depth feedback
anchors are estimated once from all training sessions and frozen identically
in every runtime configuration.

It deliberately does not manufacture an ``average opening book``.  The pooled
configuration contains an actual reference opening from the latest training
day only to satisfy the executable schema.  The calibration driver replaces
all five opening-state fields with the truly held-out day's observed opening
before validation.  Candidate behavioural policies are evaluated against each
individual training day, not against the pooled target CSV written here.

Pooling rules
-------------
* Histogram counts are summed across the selected days.  Thus every observed
  ITCH order mark has equal weight.
* Hawkes event rates use total observed events divided by total observed
  seconds.  For equal-duration full days this equals the arithmetic mean of
  daily event rates.
* Full-session target means are arithmetic means across days.  Their reported
  scale includes both daily target uncertainty and between-day variation.  The
  artifact is used for liquidity clustering/provenance, not to replace the
  day-level WMM training objective.
* ``target_spread_ticks``, ``target_mean_bid_depth`` and
  ``target_mean_ask_depth`` use those pooled training means in every daily,
  development-validation and final runtime.  Held-out target files never
  supply a runtime field.
* The local quote quantity and quote-improvement probability are recomputed
  from the pooled marks and pooled placement counts.

The input configuration rows must have the standard one-book-per-symbol schema
written by ``build_itch_universe_config.py``.  The output contains:

* ``pooled_training_universe.csv`` -- direct pooled inputs, for frozen
  held-out validation and final case studies;
* ``training_days/<date>/universe_common.csv`` -- per-day, common-symbol
  configurations for the multi-day behavioural objective;
* ``heldout_common.csv`` -- pooled training runtime inputs with only the five
  observed held-out opening fields substituted;
* ``pooled_data/<symbol>/`` -- generated mark distributions, manifest, target,
  and Hawkes-rate file per symbol; and
* ``pooling_provenance.json`` -- complete source and aggregation provenance.

Example
-------
\n
    python3 scripts/pool_multiday_empirical_universe.py \\
      --training-day 2019-01-30 /data/jan/config.csv \\
      --training-day 2019-03-27 /data/mar/config.csv \\
      --training-day 2019-07-30 /data/jul/config.csv \\
      --training-day 2019-10-30 /data/oct/config.csv \\
      --training-day 2019-12-30 /data/dec/config.csv \\
      --heldout-date 2020-01-30 --heldout-config /data/test/config.csv \\
      --output-root results/multiday_2019_training
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import derive_hawkes_rates as hawkes  # noqa: E402
import certification_cohort as cohort  # noqa: E402


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
# These state targets are optional to the generic C++ CSV reader so legacy
# smoke configurations remain executable, but they are mandatory in every
# configuration produced by the certified five-day workflow.  They are
# estimated once from training data and then frozen; the held-out target files
# are never consulted when constructing a runtime configuration.
QUEUE_REACTIVE_TARGET_FIELDS = (
    "target_mean_bid_depth",
    "target_mean_ask_depth",
)
LATENT_VALUE_FIELDS = (
    "fundamental_volatility_bps_sqrt_second",
    "fundamental_move_probability_per_second",
    "fundamental_conditional_kurtosis",
)
RUNTIME_CONFIG_FIELDS = (
    CONFIG_FIELDS + LATENT_VALUE_FIELDS + QUEUE_REACTIVE_TARGET_FIELDS
)
POOLED_HOMEOSTATIC_FIELDS = (
    "target_spread_ticks",
    *QUEUE_REACTIVE_TARGET_FIELDS,
)
FROZEN_TRAINING_DERIVED_FIELDS = (
    *POOLED_HOMEOSTATIC_FIELDS,
    *LATENT_VALUE_FIELDS,
)
OPENING_FIELDS = (
    "fundamental_price_ticks",
    "initial_best_bid_ticks",
    "initial_best_ask_ticks",
    "initial_best_bid_depth",
    "initial_best_ask_depth",
)
QUANTITY_EVENTS = tuple(hawkes.EVENT_NAMES)
DISTANCE_EVENTS = (
    "limit_buy",
    "limit_sell",
    "cancel_bid",
    "cancel_ask",
)
PLACEMENT_FIELDS = (
    "improvement_eligible_limit_orders",
    "inside_spread_limit_orders",
)

# The compact extractor identifies the aggregate count of inside-spread limit
# additions, but it does not retain their buy/sell allocation or their joint
# distribution with the pre-add book state. Every inside-spread addition is
# encoded at distance zero, as is an at-best addition. The only always-
# identifiable split is therefore the fraction of *combined* buy/sell
# distance-zero additions that were inside the spread. The runtime applies
# this one maximum-symmetry split to either side after sampling a zero-distance
# mark and only when a geometric inside price exists (spread >= two ticks).
# ``inside / eligible`` remains useful descriptive evidence, but is not the
# runtime probability and must never be divided by each side's marginal zero
# mass.
QUOTE_IMPROVEMENT_COMPATIBILITY = {
    "schema_version": 2,
    "status": "preflight_passed",
    "descriptive_empirical_rate": (
        "inside_spread_limit_orders / improvement_eligible_limit_orders; "
        "eligible means pre-add displayed spread >= two simulator ticks"
    ),
    "runtime_estimand": (
        "inside_spread_limit_orders / "
        "(limit_buy_distance_zero_count + limit_sell_distance_zero_count)"
    ),
    "runtime_mapping": (
        "apply one aggregate maximum-symmetry split after a buy or sell "
        "distance-zero mark is sampled and only when the simulated spread "
        "is at least two ticks"
    ),
    "exact_joint_mark_calibration": False,
    "side_and_state_joint_counts_identified": False,
    "reduced_form_maximum_symmetry_zero_split": True,
    "probability_clamping_permitted": False,
    "compatibility_criterion": (
        "inside_spread_limit_orders <= "
        "limit_buy_distance_zero_count + limit_sell_distance_zero_count"
    ),
    "absolute_tolerance": 1.0e-12,
    "scope": (
        "every training-day and pooled symbol configuration; held-out "
        "runtime backgrounds are frozen from pooled training inputs"
    ),
}

# ITCH prices are integer multiples of USD 0.0001.  The fragmented simulator
# currently uses one fixed USD 0.01 price grid for every book, so its empirical
# universe must begin each fitted/validated session in the one-cent quoting
# regime.  Under Rule 612, sub-dollar securities may quote in USD 0.0001
# increments and therefore require a different simulator representation.
DEFAULT_SIMULATOR_TICK_SIZE_PRICE_UNITS = 100
DEFAULT_MINIMUM_OPENING_BID_PRICE_UNITS = 10_000


class PoolingError(ValueError):
    """Raised when inputs cannot form an auditable pooled empirical template."""


@dataclass(frozen=True)
class DayConfig:
    trading_date: str
    path: pathlib.Path
    fields: tuple[str, ...]
    rows_by_symbol: Mapping[str, Mapping[str, str]]


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration_schema_sha256(fields: Sequence[str]) -> str:
    """Hash an ordered CSV schema independently of platform line endings."""
    encoded = json.dumps(
        list(fields), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


WORKFLOW_SEMANTICS_FILES = (
    "submit_five_day_pooled_training.sh",
    "submit_cluster_value_agent_calibration.sh",
    "submit_real_universe_case_study.sh",
    "scripts/derive_hawkes_rates.py",
    "scripts/pool_multiday_empirical_universe.py",
    "scripts/cluster_empirical_universe.py",
    "scripts/intersect_empirical_universe_configs.py",
    "scripts/calibrate_cluster_value_agents.py",
    "scripts/run_fragmented_mpi_experiments.py",
    "scripts/analyze_capacity_pilot.py",
    "scripts/analyze_fragmented_shared_liquidity_case.py",
    "scripts/seagull_deterministic_build.sh",
    "scripts/certification_cohort.py",
    "scripts/verify_global_calibration_certification.py",
    "tests/test_global_calibration_certification_verifier.py",
    "config/certification_symbols_1480.txt",
    "config/certification_symbols_1480_origin.json",
)


def workflow_source_semantics_sha256(project_root: pathlib.Path) -> str:
    root = project_root.resolve()
    paths = [root / relative for relative in WORKFLOW_SEMANTICS_FILES]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise PoolingError(
            "incomplete workflow source tree; missing "
            + ", ".join(str(path) for path in missing)
        )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def parse_iso_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise PoolingError(f"{label} must be ISO YYYY-MM-DD: {value!r}") from error


def parse_clock_seconds(value: object, label: str) -> int:
    try:
        hour, minute, second = (int(piece) for piece in str(value).split(":"))
    except (TypeError, ValueError) as error:
        raise PoolingError(f"invalid {label}: {value!r}") from error
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise PoolingError(f"invalid {label}: {value!r}")
    return 3600 * hour + 60 * minute + second


def finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PoolingError(f"invalid {label}: {value!r}") from error
    if not math.isfinite(result):
        raise PoolingError(f"non-finite {label}: {value!r}")
    return result


def positive_int(value: object, label: str) -> int:
    number = finite_float(value, label)
    if number <= 0.0 or not number.is_integer():
        raise PoolingError(f"{label} must be a positive integer: {value!r}")
    return int(number)


def nonnegative_int(value: object, label: str) -> int:
    number = finite_float(value, label)
    if number < 0.0 or not number.is_integer():
        raise PoolingError(f"{label} must be a non-negative integer: {value!r}")
    return int(number)


def normalise_symbol(value: object) -> str:
    symbol = str(value or "").strip().upper()
    if not symbol or any(character.isspace() for character in symbol):
        raise PoolingError(f"invalid symbol: {value!r}")
    if "/" in symbol or "\\" in symbol:
        raise PoolingError(f"unsafe symbol: {value!r}")
    return symbol


def read_csv(path: pathlib.Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = tuple(reader.fieldnames or ())
            rows = [dict(row) for row in reader]
    except OSError as error:
        raise PoolingError(f"cannot read CSV {path}: {error}") from error
    if not fields:
        raise PoolingError(f"CSV has no header: {path}")
    return fields, rows


def load_config(trading_date: str, raw_path: str) -> DayConfig:
    path = pathlib.Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise PoolingError(f"training configuration is not a file: {path}")
    fields, rows = read_csv(path)
    missing = sorted(set(CONFIG_FIELDS).difference(fields))
    if missing:
        raise PoolingError(f"{path} is missing config fields: {', '.join(missing)}")
    rows_by_symbol: dict[str, Mapping[str, str]] = {}
    seen_ids: set[int] = set()
    for line_number, row in enumerate(rows, start=2):
        symbol = normalise_symbol(row.get("symbol", ""))
        if symbol in rows_by_symbol:
            raise PoolingError(f"duplicate symbol {symbol} in {path}:{line_number}")
        book_id = nonnegative_int(row.get("book_id", ""), f"{path}:{line_number}:book_id")
        if book_id in seen_ids:
            raise PoolingError(f"duplicate book_id {book_id} in {path}:{line_number}")
        seen_ids.add(book_id)
        rows_by_symbol[symbol] = row
    if not rows_by_symbol:
        raise PoolingError(f"configuration is empty: {path}")
    if sorted(seen_ids) != list(range(len(seen_ids))):
        raise PoolingError(f"book_id values in {path} must be contiguous from zero")
    return DayConfig(trading_date, path, fields, rows_by_symbol)


def resolve_path(raw: str, config_path: pathlib.Path, *, directory: bool) -> pathlib.Path:
    candidate = pathlib.Path(raw).expanduser()
    if candidate.is_absolute():
        # Extracted universe configurations intentionally record absolute paths
        # so a run cannot silently pick up the wrong empirical directory.  The
        # compact six-session bundle is nevertheless expected to move from the
        # workstation to Seagull.  If the recorded path no longer exists,
        # preserve the suffix beginning at ``empirical_data`` and resolve it
        # beside the transferred configuration.  The basename alternatives
        # retain compatibility with older extractor layouts.
        candidates = [candidate]
        parts = candidate.parts
        if "empirical_data" in parts:
            empirical_index = len(parts) - 1 - parts[::-1].index("empirical_data")
            candidates.append(config_path.parent.joinpath(*parts[empirical_index:]))
        candidates.extend([
            config_path.parent / "empirical_data" / candidate.name,
            config_path.parent / candidate.name,
        ])
    else:
        candidates = [
            config_path.parent / candidate,
            pathlib.Path.cwd() / candidate,
        ]
    # Do not report/check the same fallback twice for shallow layouts.
    candidates = list(dict.fromkeys(candidates))
    for path in candidates:
        resolved = path.resolve()
        if (resolved.is_dir() if directory else resolved.is_file()):
            return resolved
    kind = "directory" if directory else "file"
    rendered = ", ".join(str(path) for path in candidates)
    raise PoolingError(f"cannot resolve {kind} {raw!r} from {config_path}; checked {rendered}")


def source_data_dir(day: DayConfig, symbol: str) -> pathlib.Path:
    return resolve_path(str(day.rows_by_symbol[symbol]["data_dir"]), day.path, directory=True)


def source_rate_path(day: DayConfig, symbol: str) -> pathlib.Path:
    return resolve_path(
        str(day.rows_by_symbol[symbol]["hawkes_rates_file"]),
        day.path,
        directory=False,
    )


def canonical_daily_rate_path(
    output_root: pathlib.Path, trading_date: str, symbol: str,
) -> pathlib.Path:
    compact = trading_date.replace("-", "")
    return (
        output_root / "training_days" / trading_date / "hawkes_rates"
        / f"hawkes_rates_{symbol.lower()}_balanced_{compact}.csv"
    )


def rate_manifest_inputs(
    manifest_path: pathlib.Path,
) -> tuple[dict[str, Any], float, list[float]]:
    """Read the rate clock/counts without trusting the generated rate CSV."""
    try:
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise PoolingError(
            f"cannot read rate-derivation manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise PoolingError(
            f"rate-derivation manifest is not an object: {manifest_path}"
        )
    if "aggregation_duration_seconds" in manifest:
        duration = finite_float(
            manifest.get("aggregation_duration_seconds"),
            f"{manifest_path}:aggregation_duration_seconds",
        )
    else:
        duration = float(
            parse_clock_seconds(
                manifest.get("session_end"), f"{manifest_path}:session_end"
            )
            - parse_clock_seconds(
                manifest.get("session_start"), f"{manifest_path}:session_start"
            )
        )
    if duration <= 0.0 or not duration.is_integer():
        raise PoolingError(
            f"rate-derivation manifest has invalid duration: {manifest_path}"
        )
    counts = manifest.get("distribution_observation_counts")
    if not isinstance(counts, dict):
        raise PoolingError(
            f"rate-derivation manifest lacks event counts: {manifest_path}"
        )
    expected_observed: list[float] = []
    for event in hawkes.EVENT_NAMES:
        count = finite_float(
            counts.get(event), f"{manifest_path}:{event}:observed_count"
        )
        if count < 0.0 or not count.is_integer():
            raise PoolingError(
                f"rate-derivation manifest has invalid count for {event}: "
                f"{manifest_path}"
            )
        expected_observed.append(count / duration)
    return manifest, duration, expected_observed


def rate_distribution_moment(
    path: pathlib.Path, value_column: str,
) -> tuple[float, float]:
    """Return a histogram mean and zero fraction for an independent audit."""
    values = load_distribution(path, value_column)
    total = sum(values.values())
    if not math.isfinite(total) or total <= 0.0:
        raise PoolingError(f"invalid rate-derivation distribution: {path}")
    mean = sum(value * weight for value, weight in values.items()) / total
    return mean, values.get(0, 0.0) / total


def expected_stationary_targets(
    manifest_path: pathlib.Path, observed_rates: Sequence[float], *,
    balance_directional_volume: bool, balance_best_depth: bool,
    balance_strength: float,
) -> list[float]:
    """Independently apply the declared reduced-book moment transforms."""
    directory = manifest_path.parent
    quantity_means = [
        rate_distribution_moment(
            directory / f"{event}_quantity_distribution.txt", "quantity"
        )[0]
        for event in hawkes.EVENT_NAMES
    ]
    directional = list(observed_rates)
    if balance_directional_volume:
        for left, right in ((0, 1), (2, 3), (4, 5)):
            total_rate = observed_rates[left] + observed_rates[right]
            total_mean = quantity_means[left] + quantity_means[right]
            if total_rate <= 0.0 or total_mean <= 0.0:
                directional[left] = 0.0
                directional[right] = 0.0
            else:
                directional[left] = (
                    total_rate * quantity_means[right] / total_mean
                )
                directional[right] = (
                    total_rate * quantity_means[left] / total_mean
                )
    if not balance_best_depth:
        return directional

    limit_buy_zero = rate_distribution_moment(
        directory / "limit_buy_distance_distribution.txt", "distance_ticks"
    )[1]
    limit_sell_zero = rate_distribution_moment(
        directory / "limit_sell_distance_distribution.txt", "distance_ticks"
    )[1]
    cancel_bid_zero = rate_distribution_moment(
        directory / "cancel_bid_distance_distribution.txt", "distance_ticks"
    )[1]
    cancel_ask_zero = rate_distribution_moment(
        directory / "cancel_ask_distance_distribution.txt", "distance_ticks"
    )[1]
    bid_cancel_denominator = cancel_bid_zero * quantity_means[4]
    ask_cancel_denominator = cancel_ask_zero * quantity_means[5]
    if bid_cancel_denominator <= 0.0 or ask_cancel_denominator <= 0.0:
        raise PoolingError(
            f"rate-derivation best-depth transform has zero cancellation "
            f"support below {directory}"
        )
    fully_balanced = list(directional)
    fully_balanced[4] = max(
        0.0,
        (
            directional[0] * limit_buy_zero * quantity_means[0]
            - directional[3] * quantity_means[3]
        ) / bid_cancel_denominator,
    )
    fully_balanced[5] = max(
        0.0,
        (
            directional[1] * limit_sell_zero * quantity_means[1]
            - directional[2] * quantity_means[2]
        ) / ask_cancel_denominator,
    )
    return [
        original + balance_strength * (balanced - original)
        for original, balanced in zip(directional, fully_balanced)
    ]


def validate_rate_derivation(
    rows: Sequence[Mapping[str, object]], *, label: str,
    manifest_path: pathlib.Path,
    activity_scale: float, kernel_beta: float,
    balance_directional_volume: bool, balance_best_depth: bool,
    balance_strength: float,
) -> dict[str, object]:
    """Audit that the configured Hawkes rates reconstruct their fitted targets."""
    observed_names = [str(row.get("event_type", "")) for row in rows]
    if observed_names != list(hawkes.EVENT_NAMES):
        raise PoolingError(
            f"{label} Hawkes rows have the wrong event order: {observed_names}"
        )
    _manifest, duration, expected_observed = rate_manifest_inputs(manifest_path)
    expected_targets = expected_stationary_targets(
        manifest_path, expected_observed,
        balance_directional_volume=balance_directional_volume,
        balance_best_depth=balance_best_depth,
        balance_strength=balance_strength,
    )
    maximum_observed_error = 0.0
    maximum_target_error = 0.0
    maximum_reconstruction_error = 0.0
    maximum_reported_reconstruction_error = 0.0
    if activity_scale <= 0.0 or kernel_beta <= 0.0:
        raise PoolingError(f"{label} has invalid Hawkes inversion settings")
    for index, row in enumerate(rows):
        event = str(row["event_type"])
        observed = finite_float(
            row.get("observed_rate_per_second"), f"{label}:{event}:observed"
        )
        target = finite_float(
            row.get("stationary_target_rate"), f"{label}:{event}:target"
        )
        configured_mu = finite_float(
            row.get("configured_mu"), f"{label}:{event}:configured_mu"
        )
        reconstructed = finite_float(
            row.get("stationary_reconstructed_rate"),
            f"{label}:{event}:reconstructed",
        )
        if min(observed, target, configured_mu, reconstructed) < 0.0:
            raise PoolingError(
                f"{label} has a negative Hawkes rate for {event}"
            )
        observed_error = abs(observed - expected_observed[index])
        target_error = abs(target - expected_targets[index])
        alpha = hawkes.default_alpha()
        endogenous = sum(
            alpha[index][column] * finite_float(
                rows[column].get("stationary_target_rate"),
                f"{label}:{hawkes.EVENT_NAMES[column]}:target",
            ) / kernel_beta
            for column in range(len(hawkes.EVENT_NAMES))
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
            raise PoolingError(
                f"{label} observed rate disagrees with manifest count/duration "
                f"for {event}: generated={observed:.17g}, "
                f"expected={expected_observed[index]:.17g}"
            )
        if not math.isclose(
                target, expected_targets[index],
                rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise PoolingError(
                f"{label} stationary target disagrees with the declared "
                f"reduced-book transforms for {event}: "
                f"generated={target:.17g}, expected={expected_targets[index]:.17g}"
            )
        if not math.isclose(
                reconstructed, computed_reconstruction,
                rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise PoolingError(
                f"{label} reported stationary reconstruction disagrees with "
                f"configured_mu for {event}: "
                f"reported={reconstructed:.17g}, "
                f"computed={computed_reconstruction:.17g}"
            )
        if not math.isclose(
                computed_reconstruction, target,
                rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise PoolingError(
                f"{label} Hawkes inversion cannot reconstruct the stationary "
                f"target for {event}: target={target:.17g}, "
                f"reconstructed={computed_reconstruction:.17g}, "
                f"observed={observed:.17g}"
            )
    return {
        "schema_version": 1,
        "status": "passed",
        "event_types_checked": len(rows),
        "manifest_duration_seconds": int(duration),
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


def rate_derivation_artifacts(
    audit: Mapping[str, object], *, manifest_path: pathlib.Path,
    generated_path: pathlib.Path,
) -> dict[str, object]:
    """Bind a successful numerical audit to the exact manifest and rate file."""
    return {
        **audit,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "generated_hawkes_rates": {
            "path": str(generated_path.resolve()),
            "sha256": sha256_file(generated_path),
        },
    }


def write_daily_balanced_rates(
    *, day: DayConfig, symbol: str, output_root: pathlib.Path,
    args: argparse.Namespace,
) -> tuple[pathlib.Path, dict[str, object]]:
    """Regenerate one daily rate file without mutating compact source data."""
    directory = source_data_dir(day, symbol)
    manifest_path, _ = source_manifest(directory, symbol)
    source_path = source_rate_path(day, symbol)
    output = canonical_daily_rate_path(output_root, day.trading_date, symbol)
    if output.exists() and not args.overwrite:
        raise PoolingError(f"refusing to overwrite generated daily rates: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        hawkes.run(argparse.Namespace(
            manifest=str(manifest_path), output=str(temporary),
            activity_scale=args.activity_scale, beta=args.hawkes_beta,
            balance_directional_volume=args.balance_directional_volume,
            balance_best_depth=args.balance_best_depth,
            balance_strength=args.balance_strength,
        ))
        _, generated_rows = read_csv(temporary)
        rate_audit = validate_rate_derivation(
            generated_rows, label=f"{symbol} {day.trading_date}",
            manifest_path=manifest_path,
            activity_scale=args.activity_scale,
            kernel_beta=args.hawkes_beta,
            balance_directional_volume=args.balance_directional_volume,
            balance_best_depth=args.balance_best_depth,
            balance_strength=args.balance_strength,
        )
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    rate_derivation = rate_derivation_artifacts(
        rate_audit, manifest_path=manifest_path, generated_path=output
    )
    return output.resolve(), {
        "source_hawkes_rates": str(source_path),
        "source_hawkes_rates_sha256": sha256_file(source_path),
        "generated_hawkes_rates": str(output.resolve()),
        "generated_hawkes_rates_sha256": sha256_file(output),
        "rate_derivation": rate_derivation,
    }


def find_single(directory: pathlib.Path, pattern: str, label: str) -> pathlib.Path:
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1:
        raise PoolingError(f"{directory} needs exactly one {label}; found {len(paths)}")
    return paths[0]


def source_manifest(directory: pathlib.Path, symbol: str) -> tuple[pathlib.Path, dict[str, Any]]:
    path = find_single(directory, f"itch_manifest_{symbol.lower()}_*.json", "manifest")
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise PoolingError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise PoolingError(f"manifest is not an object: {path}")
    return path, value


def source_target(directory: pathlib.Path, symbol: str) -> pathlib.Path:
    candidates = [
        path for path in directory.glob(f"market_targets_{symbol.lower()}_*.csv")
        if "_window_" not in path.name
    ]
    if len(candidates) != 1:
        raise PoolingError(
            f"{directory} needs exactly one full-session target for {symbol}; found {len(candidates)}"
        )
    return candidates[0]


def snapshot_coverage_target(manifest: Mapping[str, Any], *, path: pathlib.Path) -> dict[str, float | str]:
    """Build the empirical fixed-clock two-sidedness moment from provenance."""
    try:
        valid = int(manifest["valid_snapshots"])
        invalid = int(manifest["invalid_snapshots"])
    except (KeyError, TypeError, ValueError) as error:
        raise PoolingError(
            f"manifest lacks valid/invalid fixed-clock snapshot counts: {path}"
        ) from error
    total = valid + invalid
    if valid < 0 or invalid < 0 or total <= 0:
        raise PoolingError(f"manifest has invalid snapshot accounting: {path}")
    fraction = valid / total
    binomial_se = math.sqrt(fraction * (1.0 - fraction) / total)
    return {
        "name": "two_sided_sample_fraction",
        "target": fraction,
        "scale": max(0.005, binomial_se),
        "weight": 1.0,
    }


def load_distribution(path: pathlib.Path, value_column: str) -> dict[int, float]:
    fields, rows = read_csv(path)
    if value_column not in fields:
        raise PoolingError(f"{path} lacks {value_column} column")
    weight_column = next(
        (name for name in ("count", "frequency", "weight", "probability", "prob", "mass")
         if name in fields),
        None,
    )
    result: dict[int, float] = {}
    for line_number, row in enumerate(rows, start=2):
        raw_value = finite_float(row.get(value_column, ""), f"{path}:{line_number}:{value_column}")
        if not raw_value.is_integer() or raw_value < 0.0:
            raise PoolingError(f"invalid non-negative integer mark in {path}:{line_number}")
        weight = 1.0 if weight_column is None else finite_float(
            row.get(weight_column, ""), f"{path}:{line_number}:{weight_column}"
        )
        if weight < 0.0:
            raise PoolingError(f"negative mark weight in {path}:{line_number}")
        if weight > 0.0:
            value = int(raw_value)
            result[value] = result.get(value, 0.0) + weight
    if not result or not math.isfinite(sum(result.values())):
        raise PoolingError(f"empty or invalid empirical distribution: {path}")
    return result


def quote_improvement_compatibility_record(
    *,
    eligible_count: object,
    inside_count: object,
    buy_distances: Mapping[int, float],
    sell_distances: Mapping[int, float],
    label: str,
    configured_probability: object | None = None,
) -> dict[str, object]:
    """Derive the one zero-distance split identifiable from compact marks."""
    eligible = finite_float(eligible_count, f"{label}:eligible_count")
    inside = finite_float(inside_count, f"{label}:inside_count")
    if (eligible < 0.0 or inside < 0.0 or not eligible.is_integer()
            or not inside.is_integer()):
        raise PoolingError(
            f"{label} quote-improvement counts must be non-negative integers"
        )
    if inside > eligible:
        raise PoolingError(f"{label} inside-spread count exceeds eligible count")
    buy_total = sum(buy_distances.values())
    sell_total = sum(sell_distances.values())
    if (not math.isfinite(buy_total) or buy_total <= 0.0
            or not math.isfinite(sell_total) or sell_total <= 0.0):
        raise PoolingError(f"{label} has an empty limit-distance histogram")
    buy_zero_count = buy_distances.get(0, 0.0)
    sell_zero_count = sell_distances.get(0, 0.0)
    combined_zero_count = buy_zero_count + sell_zero_count
    tolerance = float(QUOTE_IMPROVEMENT_COMPATIBILITY["absolute_tolerance"])
    if inside > combined_zero_count + tolerance:
        raise PoolingError(
            f"{label} inside-spread count={inside:.17g} exceeds the combined "
            f"distance-zero count (buy={buy_zero_count:.17g}, "
            f"sell={sell_zero_count:.17g}); compact inputs are inconsistent"
        )
    if combined_zero_count <= tolerance:
        if inside > tolerance:
            raise PoolingError(
                f"{label} has positive inside-spread count but no "
                "distance-zero additions"
            )
        zero_split = 0.0
    else:
        zero_split = inside / combined_zero_count
    if zero_split < 0.0 or zero_split > 1.0:
        raise PoolingError(f"{label} has an invalid aggregate zero split")
    descriptive_rate = inside / eligible if eligible > 0.0 else 0.0

    configured = None
    source_semantics = None
    if configured_probability is not None:
        configured = finite_float(
            configured_probability,
            f"{label}:configured quote_improvement_probability",
        )
        if configured < 0.0 or configured > 1.0:
            raise PoolingError(
                f"{label} configured quote_improvement_probability is outside [0,1]"
            )
        if math.isclose(configured, zero_split, rel_tol=1.0e-12, abs_tol=1.0e-12):
            source_semantics = "aggregate_zero_split_v2"
        elif math.isclose(
                configured, descriptive_rate,
                rel_tol=1.0e-12, abs_tol=1.0e-12):
            source_semantics = "legacy_eligible_rate_v1_migrated"
        else:
            raise PoolingError(
                f"{label} configured quote_improvement_probability="
                f"{configured:.17g} matches neither the aggregate zero split "
                f"({zero_split:.17g}) nor the legacy eligible rate "
                f"({descriptive_rate:.17g})"
            )
    return {
        "status": "passed",
        "quote_improvement_probability": zero_split,
        "runtime_zero_split_probability": zero_split,
        "descriptive_eligible_improvement_rate": descriptive_rate,
        "improvement_eligible_limit_orders": int(eligible),
        "inside_spread_limit_orders": int(inside),
        "limit_buy_distance_count": buy_total,
        "limit_sell_distance_count": sell_total,
        "limit_buy_distance_zero_count": buy_zero_count,
        "limit_sell_distance_zero_count": sell_zero_count,
        "combined_distance_zero_count": combined_zero_count,
        "limit_buy_distance_zero_mass": buy_zero_count / buy_total,
        "limit_sell_distance_zero_mass": sell_zero_count / sell_total,
        "configured_input_probability": configured,
        "configured_input_semantics": source_semantics,
        "side_allocation": "proportional_to_observed_side_zero_counts",
        "absolute_tolerance": tolerance,
        "probability_clamped": False,
    }


def source_config_quote_improvement_compatibility(
    day: DayConfig, symbol: str
) -> dict[str, object]:
    """Audit one unpooled runtime row against its compact empirical marks."""
    directory = source_data_dir(day, symbol)
    manifest_path, manifest = source_manifest(directory, symbol)
    if str(manifest.get("trading_date", "")) != day.trading_date:
        raise PoolingError(
            f"manifest date disagrees with {day.trading_date}: {manifest_path}"
        )
    placements = placement_counts(manifest, path=manifest_path)
    eligible = placements["improvement_eligible_limit_orders"]
    inside = placements["inside_spread_limit_orders"]
    if inside > eligible:
        raise PoolingError(
            f"inside-spread count exceeds eligible count for "
            f"{symbol} {day.trading_date}"
        )
    counts = manifest_counts(manifest, path=manifest_path)
    buy_distances = load_distribution(
        directory / "limit_buy_distance_distribution.txt", "distance_ticks"
    )
    sell_distances = load_distribution(
        directory / "limit_sell_distance_distribution.txt", "distance_ticks"
    )
    for event, histogram in (
        ("limit_buy", buy_distances), ("limit_sell", sell_distances)
    ):
        observed = sum(histogram.values())
        if not math.isclose(
                observed, counts[event], rel_tol=1.0e-9, abs_tol=1.0e-6):
            raise PoolingError(
                f"{symbol} {day.trading_date} {event} distance count "
                f"{observed:.17g} disagrees with manifest count {counts[event]}"
            )
    compatibility = quote_improvement_compatibility_record(
        eligible_count=eligible,
        inside_count=inside,
        buy_distances=buy_distances,
        sell_distances=sell_distances,
        label=f"{symbol} {day.trading_date}",
        configured_probability=(
            day.rows_by_symbol[symbol].get("quote_improvement_probability", "")
        ),
    )
    return {
        "symbol": symbol,
        "date": day.trading_date,
        "config": str(day.path),
        "config_sha256": sha256_file(day.path),
        "data_dir": str(directory),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "placement_counts": placements,
        **compatibility,
    }


def write_distribution(path: pathlib.Path, value_column: str, values: Mapping[int, float]) -> None:
    if not values:
        raise PoolingError(f"refusing to write empty pooled distribution: {path}")
    rows = [
        {value_column: value, "count": f"{weight:.17g}"}
        for value, weight in sorted(values.items())
        if math.isfinite(weight) and weight > 0.0
    ]
    atomic_csv(path, (value_column, "count"), rows, overwrite=True)


def weighted_median(values: Mapping[int, float]) -> int:
    total = sum(values.values())
    if not math.isfinite(total) or total <= 0.0:
        raise PoolingError("cannot calculate weighted median of empty distribution")
    threshold = 0.5 * total
    cumulative = 0.0
    for value, weight in sorted(values.items()):
        cumulative += weight
        if cumulative >= threshold:
            return value
    raise AssertionError("weighted median traversal did not terminate")


def load_targets(path: pathlib.Path) -> list[dict[str, float | str]]:
    fields, rows = read_csv(path)
    required = {"name", "target", "scale", "weight"}
    missing = sorted(required.difference(fields))
    if missing:
        raise PoolingError(f"{path} is missing target columns: {', '.join(missing)}")
    result: list[dict[str, float | str]] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        name = str(row.get("name", "") or "").strip()
        if not name or name in seen:
            raise PoolingError(f"invalid or duplicate target name in {path}:{line_number}")
        target = finite_float(row.get("target", ""), f"{path}:{line_number}:target")
        scale = finite_float(row.get("scale", ""), f"{path}:{line_number}:scale")
        weight = finite_float(row.get("weight", ""), f"{path}:{line_number}:weight")
        if scale <= 0.0 or weight <= 0.0:
            raise PoolingError(f"non-positive target scale or weight in {path}:{line_number}")
        result.append({"name": name, "target": target, "scale": scale, "weight": weight})
        seen.add(name)
    if not result:
        raise PoolingError(f"target CSV is empty: {path}")
    return result


def aggregate_targets(day_targets: Sequence[Sequence[Mapping[str, float | str]]]) -> list[dict[str, str]]:
    if not day_targets:
        raise PoolingError("no targets to aggregate")
    names = [str(row["name"]) for row in day_targets[0]]
    if len(names) != len(set(names)):
        raise PoolingError("duplicate target names")
    rows_by_day = [{str(row["name"]): row for row in rows} for rows in day_targets]
    for rows in rows_by_day[1:]:
        if set(rows) != set(names):
            raise PoolingError("training target names differ across days")
    result: list[dict[str, str]] = []
    day_count = len(rows_by_day)
    for name in names:
        entries = [rows[name] for rows in rows_by_day]
        values = [float(entry["target"]) for entry in entries]
        scales = [float(entry["scale"]) for entry in entries]
        weights = [float(entry["weight"]) for entry in entries]
        if max(weights) - min(weights) > 1.0e-12 * max(1.0, max(abs(weight) for weight in weights)):
            raise PoolingError(f"WMM weight for {name} differs across training days")
        mean_target = statistics.fmean(values)
        # The target is a mean over days.  The first component carries each
        # day's empirical/jackknife uncertainty; the second admits genuine
        # between-day variation rather than treating five sessions as copies.
        within_variance = sum(scale * scale for scale in scales) / (day_count * day_count)
        between_variance = (
            statistics.variance(values) / day_count if day_count > 1 else 0.0
        )
        scale = math.sqrt(max(0.0, within_variance + between_variance))
        if not math.isfinite(scale) or scale <= 0.0:
            scale = max(1.0e-12, statistics.fmean(scales) / math.sqrt(day_count))
        if name == "two_sided_sample_fraction":
            scale = max(0.005, scale)
        result.append({
            "name": name,
            "target": f"{mean_target:.17g}",
            "scale": f"{scale:.17g}",
            "weight": f"{statistics.fmean(weights):.17g}",
        })
    return result


def atomic_csv(path: pathlib.Path,
               fieldnames: Sequence[str],
               rows: Sequence[Mapping[str, object]],
               *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise PoolingError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: pathlib.Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise PoolingError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def output_label(days: Sequence[DayConfig]) -> str:
    return "_".join(day.trading_date.replace("-", "") for day in days)


def common_symbols(days: Sequence[DayConfig], heldout: DayConfig) -> list[str]:
    common = set(heldout.rows_by_symbol)
    for day in days:
        common.intersection_update(day.rows_by_symbol)
    if "QQQ" not in common:
        raise PoolingError("QQQ is not common to every training and held-out configuration")
    return sorted(common, key=lambda symbol: (symbol != "QQQ", symbol))


def select_opening_price_grid_compatible_symbols(
    days: Sequence[DayConfig],
    heldout: DayConfig,
    symbols: Sequence[str],
    *,
    simulator_tick_size_price_units: int,
    minimum_opening_bid_price_units: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Apply the simulator's declared opening-price domain before fitting.

    Only BBO values already admitted as session-start model inputs are read;
    no held-out event rate, order mark, intraday moment, or outcome enters this
    screen.  Requiring every opening BBO to lie on the simulator's fixed grid
    prevents a sub-dollar/sub-penny ITCH book from being silently rounded into
    a different market model.
    """
    sessions = [*days, heldout]
    eligible: list[str] = []
    excluded: list[dict[str, Any]] = []
    for symbol in symbols:
        issues: list[dict[str, Any]] = []
        minimum_observed_bid: int | None = None
        for session in sessions:
            row = session.rows_by_symbol[symbol]
            bid = positive_int(
                row["initial_best_bid_ticks"],
                f"{session.trading_date}:{symbol}:initial_best_bid_ticks",
            )
            ask = positive_int(
                row["initial_best_ask_ticks"],
                f"{session.trading_date}:{symbol}:initial_best_ask_ticks",
            )
            minimum_observed_bid = (
                bid if minimum_observed_bid is None
                else min(minimum_observed_bid, bid)
            )
            if ask <= bid:
                issues.append({
                    "date": session.trading_date,
                    "reason": "opening_ask_not_above_bid",
                    "bid_price_units": bid,
                    "ask_price_units": ask,
                })
                continue
            if bid < minimum_opening_bid_price_units:
                issues.append({
                    "date": session.trading_date,
                    "reason": "opening_bid_below_model_price_regime",
                    "bid_price_units": bid,
                    "minimum_bid_price_units": minimum_opening_bid_price_units,
                })
            if (bid % simulator_tick_size_price_units != 0
                    or ask % simulator_tick_size_price_units != 0):
                issues.append({
                    "date": session.trading_date,
                    "reason": "opening_bbo_off_simulator_price_grid",
                    "bid_price_units": bid,
                    "ask_price_units": ask,
                    "simulator_tick_size_price_units": (
                        simulator_tick_size_price_units
                    ),
                })
        if issues:
            excluded.append({
                "symbol": symbol,
                "minimum_observed_opening_bid_price_units": minimum_observed_bid,
                "issues": issues,
            })
        else:
            eligible.append(symbol)

    if "QQQ" not in eligible:
        qqq = next(
            (entry for entry in excluded if entry["symbol"] == "QQQ"), None
        )
        raise PoolingError(
            "QQQ fails the declared opening price-grid eligibility rule: "
            f"{qqq['issues'] if qqq is not None else 'unknown reason'}"
        )
    return eligible, excluded


def subset_rows(day: DayConfig, symbols: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for book_id, symbol in enumerate(symbols):
        row = dict(day.rows_by_symbol[symbol])
        row["book_id"] = str(book_id)
        # Emit canonical paths for the machine performing the pooling.  This
        # makes every common-day configuration produced on Seagull directly
        # executable even when the source CSV was created on macOS.
        row["data_dir"] = str(source_data_dir(day, symbol))
        row["hawkes_rates_file"] = str(source_rate_path(day, symbol))
        rows.append(row)
    return rows


def inject_pooled_homeostatic_targets(
    rows: Sequence[Mapping[str, str]],
    pooled_rows_by_symbol: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    """Freeze the same five-day state targets into every runtime session.

    Daily event clocks, marks, opening states and the other direct inputs stay
    session-specific.  Only the quantities used by the homeostatic spread and
    queue-depth mechanisms come from the pooled training estimator.  In
    particular, no value from a held-out target CSV is accepted here.
    """
    result: list[dict[str, str]] = []
    for row in rows:
        symbol = normalise_symbol(row.get("symbol", ""))
        pooled = pooled_rows_by_symbol.get(symbol)
        if pooled is None:
            raise PoolingError(f"no pooled homeostatic targets for {symbol}")
        copied = dict(row)
        for field in FROZEN_TRAINING_DERIVED_FIELDS:
            value = finite_float(
                pooled.get(field, ""), f"{symbol}:{field} pooled runtime target"
            )
            if value <= 0.0:
                raise PoolingError(
                    f"{symbol} pooled runtime target {field} must be positive"
                )
            copied[field] = str(pooled[field])
        result.append(copied)
    return result


def freeze_pooled_backgrounds_with_heldout_openings(
    pooled_rows_by_symbol: Mapping[str, Mapping[str, str]],
    heldout: DayConfig,
    symbols: Sequence[str],
) -> list[dict[str, str]]:
    """Construct a leakage-safe held-out runtime configuration.

    Every non-opening field comes from the pooled training template. The
    held-out configuration contributes only the five session-start opening
    fields. In particular, its event rates, mark files and legacy/new quote-
    improvement scalar are never instantiated by validation.
    """
    result: list[dict[str, str]] = []
    for book_id, symbol in enumerate(symbols):
        pooled = pooled_rows_by_symbol.get(symbol)
        if pooled is None:
            raise PoolingError(f"no pooled training row for held-out {symbol}")
        opening_source = heldout.rows_by_symbol[symbol]
        row = dict(pooled)
        row["book_id"] = str(book_id)
        for field in OPENING_FIELDS:
            row[field] = str(opening_source[field])
        result.append(row)
    return result


def require_same_schema(days: Sequence[DayConfig], heldout: DayConfig) -> tuple[str, ...]:
    fields = days[0].fields
    if tuple(fields) != tuple(CONFIG_FIELDS):
        # Configs generated by the repository carry exactly this schema.  A
        # strict check stops an innocuous-looking extra column from silently
        # becoming a different frozen model at validation time.
        raise PoolingError("training configuration header does not match standard empirical schema")
    for day in [*days[1:], heldout]:
        if day.fields != fields:
            raise PoolingError("all training and held-out configurations must have identical headers")
    return fields


def manifest_counts(manifest: Mapping[str, Any], *, path: pathlib.Path) -> dict[str, int]:
    raw = manifest.get("distribution_observation_counts")
    if not isinstance(raw, Mapping):
        raise PoolingError(f"manifest lacks distribution_observation_counts: {path}")
    result: dict[str, int] = {}
    for event in QUANTITY_EVENTS:
        number = finite_float(raw.get(event), f"{path}:{event}")
        if number < 0.0 or not number.is_integer():
            raise PoolingError(f"invalid manifest event count {event}: {path}")
        result[event] = int(number)
    return result


def placement_counts(manifest: Mapping[str, Any], *, path: pathlib.Path) -> dict[str, int]:
    raw = manifest.get("placement_counts")
    if not isinstance(raw, Mapping):
        raise PoolingError(f"manifest lacks placement_counts: {path}")
    result: dict[str, int] = {}
    for name in PLACEMENT_FIELDS:
        number = finite_float(raw.get(name), f"{path}:{name}")
        if number < 0.0 or not number.is_integer():
            raise PoolingError(f"invalid placement count {name}: {path}")
        result[name] = int(number)
    return result


def merge_histograms(histograms: Iterable[Mapping[int, float]]) -> dict[int, float]:
    merged: dict[int, float] = {}
    for histogram in histograms:
        for value, weight in histogram.items():
            merged[value] = merged.get(value, 0.0) + weight
    if not merged:
        raise PoolingError("cannot merge zero histograms")
    return merged


def round_nearest(value: float, *, minimum: int, maximum: int, label: str) -> int:
    if not math.isfinite(value):
        raise PoolingError(f"non-finite {label}")
    result = int(math.floor(value + 0.5))
    return max(minimum, min(maximum, result))


def pooled_symbol(
    *,
    symbol: str,
    days: Sequence[DayConfig],
    output_root: pathlib.Path,
    label: str,
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, Any]]:
    source_records: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []
    source_targets: list[list[dict[str, float | str]]] = []
    quantities: dict[str, list[dict[int, float]]] = {event: [] for event in QUANTITY_EVENTS}
    distances: dict[str, list[dict[int, float]]] = {event: [] for event in DISTANCE_EVENTS}
    session_starts: set[str] = set()
    session_ends: set[str] = set()
    total_duration = 0

    for day in days:
        directory = source_data_dir(day, symbol)
        manifest_path, manifest = source_manifest(directory, symbol)
        _generated_daily_rate, daily_rate_provenance = write_daily_balanced_rates(
            day=day, symbol=symbol, output_root=output_root, args=args,
        )
        manifest_date = str(manifest.get("trading_date", ""))
        if manifest_date != day.trading_date:
            raise PoolingError(
                f"manifest date {manifest_date!r} does not match declared training date "
                f"{day.trading_date!r}: {manifest_path}"
            )
        start = str(manifest.get("session_start", ""))
        end = str(manifest.get("session_end", ""))
        duration = parse_clock_seconds(end, "session_end") - parse_clock_seconds(start, "session_start")
        if duration <= 0:
            raise PoolingError(f"manifest has invalid session bounds: {manifest_path}")
        session_starts.add(start)
        session_ends.add(end)
        total_duration += duration
        counts = manifest_counts(manifest, path=manifest_path)
        placements = placement_counts(manifest, path=manifest_path)
        for event in QUANTITY_EVENTS:
            histogram = load_distribution(directory / f"{event}_quantity_distribution.txt", "quantity")
            observed_mass = sum(histogram.values())
            if not math.isclose(observed_mass, counts[event], rel_tol=1.0e-9, abs_tol=1.0e-6):
                raise PoolingError(
                    f"distribution count mismatches manifest for {symbol} {day.trading_date} {event}"
                )
            quantities[event].append(histogram)
        daily_distances: dict[str, dict[int, float]] = {}
        for event in DISTANCE_EVENTS:
            histogram = load_distribution(
                directory / f"{event}_distance_distribution.txt", "distance_ticks"
            )
            observed_mass = sum(histogram.values())
            if not math.isclose(
                    observed_mass, counts[event],
                    rel_tol=1.0e-9, abs_tol=1.0e-6):
                raise PoolingError(
                    f"distance distribution count mismatches manifest for "
                    f"{symbol} {day.trading_date} {event}: "
                    f"observed={observed_mass:.17g}, expected={counts[event]}"
                )
            daily_distances[event] = histogram
            distances[event].append(histogram)
        eligible = placements["improvement_eligible_limit_orders"]
        inside = placements["inside_spread_limit_orders"]
        if inside > eligible:
            raise PoolingError(
                f"inside-spread count exceeds eligible count for "
                f"{symbol} {day.trading_date}"
            )
        improvement_compatibility = quote_improvement_compatibility_record(
            eligible_count=eligible,
            inside_count=inside,
            buy_distances=daily_distances["limit_buy"],
            sell_distances=daily_distances["limit_sell"],
            label=f"{symbol} {day.trading_date}",
            configured_probability=(
                day.rows_by_symbol[symbol].get(
                    "quote_improvement_probability", ""
                )
            ),
        )
        target_path = source_target(directory, symbol)
        daily_targets = load_targets(target_path)
        daily_target_names = {str(row["name"]) for row in daily_targets}
        if "two_sided_sample_fraction" not in daily_target_names:
            daily_targets.append(snapshot_coverage_target(manifest, path=manifest_path))
        source_targets.append(daily_targets)
        source_manifests.append(manifest)
        coverage = snapshot_coverage_target(manifest, path=manifest_path)
        source_records.append({
            "trading_date": day.trading_date,
            "config": str(day.path),
            "config_sha256": sha256_file(day.path),
            "data_dir": str(directory),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "target": str(target_path),
            "target_sha256": sha256_file(target_path),
            "session_duration_seconds": duration,
            "valid_snapshots": int(manifest["valid_snapshots"]),
            "invalid_snapshots": int(manifest["invalid_snapshots"]),
            "two_sided_sample_fraction": coverage["target"],
            "distribution_observation_counts": counts,
            "placement_counts": placements,
            "quote_improvement_compatibility": improvement_compatibility,
            **daily_rate_provenance,
        })

    if len(session_starts) != 1 or len(session_ends) != 1:
        raise PoolingError(f"training sessions have incompatible bounds for {symbol}")
    session_start = next(iter(session_starts))
    session_end = next(iter(session_ends))
    destination = output_root / "pooled_data" / f"pooled_{label}_{symbol.lower()}"
    if destination.exists() and not args.overwrite:
        raise PoolingError(f"refusing to overwrite existing pooled data directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    pooled_quantities = {event: merge_histograms(values) for event, values in quantities.items()}
    pooled_distances = {event: merge_histograms(values) for event, values in distances.items()}
    for event, values in pooled_quantities.items():
        write_distribution(destination / f"{event}_quantity_distribution.txt", "quantity", values)
    for event, values in pooled_distances.items():
        write_distribution(destination / f"{event}_distance_distribution.txt", "distance_ticks", values)

    total_event_counts = {
        event: sum(record["distribution_observation_counts"][event] for record in source_records)
        for event in QUANTITY_EVENTS
    }
    total_placements = {
        field: sum(record["placement_counts"][field] for record in source_records)
        for field in PLACEMENT_FIELDS
    }
    eligible = total_placements["improvement_eligible_limit_orders"]
    inside = total_placements["inside_spread_limit_orders"]
    if inside > eligible:
        raise PoolingError(f"pooled inside-spread count exceeds eligible count for {symbol}")
    pooled_improvement_compatibility = quote_improvement_compatibility_record(
        eligible_count=eligible,
        inside_count=inside,
        buy_distances=pooled_distances["limit_buy"],
        sell_distances=pooled_distances["limit_sell"],
        label=f"{symbol} pooled training template",
    )
    quote_improvement = float(
        pooled_improvement_compatibility["runtime_zero_split_probability"]
    )
    target_rows = aggregate_targets(source_targets)
    target_name = f"market_targets_{symbol.lower()}_pooled_{label}.csv"
    target_path = destination / target_name
    atomic_csv(target_path, ("name", "target", "scale", "weight"), target_rows, overwrite=True)

    manifest_path = destination / f"itch_manifest_{symbol.lower()}_pooled_{label}.json"
    manifest: dict[str, Any] = {
        "format": "pooled NASDAQ TotalView-ITCH empirical training template",
        "schema_version": 1,
        "symbol": symbol,
        "trading_date": f"pooled:{label}",
        "session_start": session_start,
        "session_end": session_end,
        "aggregation_duration_seconds": total_duration,
        "valid_snapshots": sum(
            int(record["valid_snapshots"]) for record in source_records
        ),
        "invalid_snapshots": sum(
            int(record["invalid_snapshots"]) for record in source_records
        ),
        "distribution_observation_counts": total_event_counts,
        "placement_counts": total_placements,
        "quote_improvement_compatibility": pooled_improvement_compatibility,
        "aggregation": {
            "method": "pooled_observations_over_complete_training_sessions",
            "opening_state": "latest_training_session_reference_only; heldout opening replaces it before validation",
            "marks": "histogram counts summed across days",
            "event_rates": "total event count divided by total observed seconds",
            "target_means": "arithmetic mean of daily full-session targets",
            "target_scales": "daily empirical uncertainty plus between-day variation",
            "source_sessions": source_records,
        },
    }
    atomic_json(manifest_path, manifest, overwrite=True)

    rate_path = destination / f"hawkes_rates_{symbol.lower()}_pooled_{label}.csv"
    temporary_rate = destination / f".{rate_path.name}.tmp"
    try:
        hawkes.run(argparse.Namespace(
            manifest=str(manifest_path), output=str(temporary_rate),
            activity_scale=args.activity_scale, beta=args.hawkes_beta,
            balance_directional_volume=args.balance_directional_volume,
            balance_best_depth=args.balance_best_depth,
            balance_strength=args.balance_strength,
        ))
        _, pooled_generated_rows = read_csv(temporary_rate)
        pooled_rate_audit = validate_rate_derivation(
            pooled_generated_rows,
            label=f"{symbol} pooled training template",
            manifest_path=manifest_path,
            activity_scale=args.activity_scale,
            kernel_beta=args.hawkes_beta,
            balance_directional_volume=args.balance_directional_volume,
            balance_best_depth=args.balance_best_depth,
            balance_strength=args.balance_strength,
        )
        os.replace(temporary_rate, rate_path)
    except Exception:
        temporary_rate.unlink(missing_ok=True)
        raise
    pooled_rate_derivation = rate_derivation_artifacts(
        pooled_rate_audit, manifest_path=manifest_path,
        generated_path=rate_path,
    )

    buy_median = weighted_median(pooled_quantities["limit_buy"])
    sell_median = weighted_median(pooled_quantities["limit_sell"])
    quote_quantity = round_nearest(
        args.quote_quantity_fraction * 0.5 * (buy_median + sell_median),
        minimum=args.minimum_quote_quantity,
        maximum=args.maximum_quote_quantity,
        label=f"{symbol} market maker quote quantity",
    )
    target_by_name = {row["name"]: row for row in target_rows}
    if "mean_spread_ticks" not in target_by_name:
        raise PoolingError(f"pooled target for {symbol} lacks mean_spread_ticks")
    if "return_variance" not in target_by_name:
        raise PoolingError(f"pooled target for {symbol} lacks return_variance")
    if "mid_move_rate" not in target_by_name:
        raise PoolingError(f"pooled target for {symbol} lacks mid_move_rate")
    if "return_kurtosis" not in target_by_name:
        raise PoolingError(f"pooled target for {symbol} lacks return_kurtosis")
    pooled_return_variance = finite_float(
        target_by_name["return_variance"]["target"],
        f"{symbol}:return_variance pooled target",
    )
    if pooled_return_variance < 0.0:
        raise PoolingError(
            f"pooled return_variance target for {symbol} must be non-negative"
        )
    fundamental_volatility = 10_000.0 * math.sqrt(pooled_return_variance)
    fundamental_move_probability = finite_float(
        target_by_name["mid_move_rate"]["target"],
        f"{symbol}:mid_move_rate pooled target",
    )
    if not 0.0 <= fundamental_move_probability <= 1.0:
        raise PoolingError(
            f"pooled mid_move_rate for {symbol} must lie in [0, 1]"
        )
    pooled_return_kurtosis = finite_float(
        target_by_name["return_kurtosis"]["target"],
        f"{symbol}:return_kurtosis pooled target",
    )
    fundamental_conditional_kurtosis = (
        pooled_return_kurtosis * fundamental_move_probability
    )
    # A fixed-clock return that is zero outside a price move has conditional
    # kurtosis K*p.  Its mathematical lower bound is one.  Refuse incoherent
    # training moments instead of silently clamping the empirical estimand.
    if (not math.isfinite(fundamental_conditional_kurtosis)
            or fundamental_conditional_kurtosis < 1.0):
        raise PoolingError(
            f"pooled return moments for {symbol} imply invalid conditional "
            f"kurtosis {fundamental_conditional_kurtosis:.17g}; require "
            "return_kurtosis * mid_move_rate >= 1"
        )
    target_spread = round_nearest(
        finite_float(target_by_name["mean_spread_ticks"]["target"], "mean_spread_ticks"),
        minimum=1, maximum=2_147_483_647, label=f"{symbol} target spread",
    )
    pooled_depth_targets: dict[str, float] = {}
    for side, metric in (("bid", "mean_bid_depth"), ("ask", "mean_ask_depth")):
        if metric not in target_by_name:
            raise PoolingError(f"pooled target for {symbol} lacks {metric}")
        value = finite_float(
            target_by_name[metric]["target"], f"{symbol}:{metric} pooled target"
        )
        if value <= 0.0:
            raise PoolingError(
                f"pooled {metric} target for {symbol} must be positive"
            )
        pooled_depth_targets[side] = value
    # Opening is intentionally *not* averaged.  The latest complete training
    # session supplies a valid executable placeholder; freeze-and-replace
    # swaps it for the held-out observed opening before validation.
    latest_row = days[-1].rows_by_symbol[symbol]
    opening_reference = {
        field: str(latest_row[field]) for field in OPENING_FIELDS
    }
    metadata = {
        "symbol": symbol,
        "pooled_data_dir": str(destination),
        "pooled_manifest": str(manifest_path),
        "pooled_target": str(target_path),
        "pooled_hawkes_rates": str(rate_path),
        "pooled_hawkes_rates_sha256": sha256_file(rate_path),
        "rate_derivation": pooled_rate_derivation,
        "pooled_source_duration_seconds": total_duration,
        "latest_training_opening_reference_date": days[-1].trading_date,
        "market_maker_quote_quantity": quote_quantity,
        "target_spread_ticks": target_spread,
        "target_mean_bid_depth": pooled_depth_targets["bid"],
        "target_mean_ask_depth": pooled_depth_targets["ask"],
        "fundamental_volatility_bps_sqrt_second": fundamental_volatility,
        "fundamental_move_probability_per_second": (
            fundamental_move_probability
        ),
        "fundamental_conditional_kurtosis": (
            fundamental_conditional_kurtosis
        ),
        "quote_improvement_probability": quote_improvement,
        "quote_improvement_compatibility": pooled_improvement_compatibility,
        "sources": source_records,
    }
    config_row = {
        "book_id": "",
        "symbol": symbol,
        "data_dir": str(destination.resolve()),
        "hawkes_rates_file": str(rate_path.resolve()),
        **opening_reference,
        "beta": "",  # filled after the cross-sectional pooled-price pass
        "basket_weight": str(latest_row["basket_weight"]),
        "market_maker_quote_quantity": str(quote_quantity),
        "target_spread_ticks": str(target_spread),
        "quote_improvement_probability": f"{quote_improvement:.17g}",
        "fundamental_volatility_bps_sqrt_second": (
            f"{fundamental_volatility:.17g}"
        ),
        "fundamental_move_probability_per_second": (
            f"{fundamental_move_probability:.17g}"
        ),
        "fundamental_conditional_kurtosis": (
            f"{fundamental_conditional_kurtosis:.17g}"
        ),
        "target_mean_bid_depth": f"{pooled_depth_targets['bid']:.17g}",
        "target_mean_ask_depth": f"{pooled_depth_targets['ask']:.17g}",
    }
    return config_row, metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.training_day) < 2:
        raise PoolingError("at least two --training-day entries are required for multi-day pooling")
    parsed_training: list[tuple[date, str, str]] = []
    seen_dates: set[date] = set()
    for raw_date, raw_config in args.training_day:
        parsed = parse_iso_date(raw_date, "--training-day date")
        if parsed in seen_dates:
            raise PoolingError(f"duplicate training date: {raw_date}")
        seen_dates.add(parsed)
        parsed_training.append((parsed, raw_date, raw_config))
    parsed_training.sort(key=lambda item: item[0])
    heldout_date = parse_iso_date(args.heldout_date, "--heldout-date")
    if any(day >= heldout_date for day, _, _ in parsed_training):
        raise PoolingError("every training date must precede the held-out date")
    days = [load_config(raw_date, config) for _, raw_date, config in parsed_training]
    heldout = load_config(args.heldout_date, args.heldout_config)
    target_roots: dict[str, pathlib.Path] = {}
    for raw_date, raw_root in args.training_target_root:
        parse_iso_date(raw_date, "--training-target-root date")
        if raw_date in target_roots:
            raise PoolingError(f"duplicate --training-target-root date: {raw_date}")
        root = pathlib.Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise PoolingError(f"training target root is not a directory: {root}")
        target_roots[raw_date] = root
    if target_roots and set(target_roots) != {day.trading_date for day in days}:
        raise PoolingError(
            "--training-target-root entries must cover exactly the declared training dates"
        )
    heldout_target_root: pathlib.Path | None = None
    if args.heldout_target_root:
        heldout_target_root = pathlib.Path(args.heldout_target_root).expanduser().resolve()
        if not heldout_target_root.is_dir():
            raise PoolingError(f"held-out target root is not a directory: {heldout_target_root}")
    fields = require_same_schema(days, heldout)
    intersection_symbols = common_symbols(days, heldout)
    symbols, price_grid_exclusions = select_opening_price_grid_compatible_symbols(
        days,
        heldout,
        intersection_symbols,
        simulator_tick_size_price_units=(
            args.simulator_tick_size_price_units
        ),
        minimum_opening_bid_price_units=(
            args.minimum_opening_bid_price_units
        ),
    )
    if len(symbols) < args.minimum_symbols:
        raise PoolingError(
            f"only {len(symbols)} price-grid-compatible common symbols remain "
            f"from {len(intersection_symbols)}; "
            f"--minimum-symbols={args.minimum_symbols}"
        )
    cohort_identity: dict[str, object] | None = None
    certification_input_selection: dict[str, object] | None = None
    if args.require_certification_cohort:
        try:
            cohort_identity = cohort.validate_symbols(
                symbols,
                label="pooled price-grid-compatible common universe",
                project_root=SCRIPT_DIR.parent,
            )
            certification_input_selection = (
                cohort.certification_pool_input_selection(
                    source_sessions={
                        **{
                            day.trading_date: tuple(day.rows_by_symbol)
                            for day in days
                        },
                        heldout.trading_date: tuple(heldout.rows_by_symbol),
                    },
                    excluded_symbols=(
                        entry["symbol"] for entry in price_grid_exclusions
                    ),
                    final_symbols=symbols,
                    project_root=SCRIPT_DIR.parent,
                )
            )
        except cohort.CohortIdentityError as error:
            raise PoolingError(str(error)) from error

    # Audit every training symbol-day before creating any output. Reporting
    # the complete set (with bounded display) avoids a slow one-error-per-job
    # repair cycle on the cluster. The held-out marks are deliberately not
    # audited or instantiated: heldout_common.csv is built from pooled
    # training backgrounds plus only the observed held-out opening fields.
    source_improvement_checks: dict[tuple[str, str], dict[str, object]] = {}
    improvement_errors: list[str] = []
    for day in days:
        for symbol in symbols:
            try:
                source_improvement_checks[(day.trading_date, symbol)] = (
                    source_config_quote_improvement_compatibility(day, symbol)
                )
            except PoolingError as error:
                improvement_errors.append(str(error))
    if improvement_errors:
        maximum_reported = 100
        shown = improvement_errors[:maximum_reported]
        remainder = len(improvement_errors) - len(shown)
        suffix = (
            f"\n... {remainder} additional incompatibilities omitted"
            if remainder > 0 else ""
        )
        raise PoolingError(
            f"quote-improvement preflight found {len(improvement_errors)} "
            "training symbol-day incompatibilities:\n- "
            + "\n- ".join(shown) + suffix
        )

    output_root = pathlib.Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise PoolingError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    label = args.label or output_label(days)
    if any(character in label for character in "/\\") or not label:
        raise PoolingError("--label must be a nonempty filename component")

    pooled_rows: list[dict[str, str]] = []
    pooled_symbol_metadata: list[dict[str, Any]] = []
    for book_id, symbol in enumerate(symbols):
        row, metadata = pooled_symbol(
            symbol=symbol, days=days, output_root=output_root,
            label=label, args=args,
        )
        row["book_id"] = str(book_id)
        pooled_rows.append(row)
        pooled_symbol_metadata.append(metadata)

    # pooled beta is a cross-sectional risk proxy based on the median of the
    # five observed training-day opening midpoints.  It is deliberately frozen
    # at validation; it is not an estimate using future held-out flow.
    pooled_fundamentals = [
        statistics.median(
            finite_float(day.rows_by_symbol[row["symbol"]]["fundamental_price_ticks"],
                         f"{day.trading_date}:{row['symbol']}:fundamental")
            for day in days
        )
        for row in pooled_rows
    ]
    cross_sectional_median = statistics.median(pooled_fundamentals)
    if not math.isfinite(cross_sectional_median) or cross_sectional_median <= 0.0:
        raise PoolingError("invalid pooled cross-sectional median price")
    for row, fundamental in zip(pooled_rows, pooled_fundamentals):
        row["beta"] = f"{fundamental / cross_sectional_median:.17g}"

    pooled_rows_by_symbol = {row["symbol"]: row for row in pooled_rows}

    # Produce reindexed, common-symbol daily configs only after the pooled
    # state targets exist.  Event clocks, marks and openings remain daily, but
    # target spread and queue-depth anchors are the same five-day estimates in
    # every training run.  This prevents day-specific target oracles.
    training_common: list[dict[str, Any]] = []
    for day in days:
        path = output_root / "training_days" / day.trading_date / "universe_common.csv"
        runtime_rows = inject_pooled_homeostatic_targets(
            subset_rows(day, symbols), pooled_rows_by_symbol
        )
        for row in runtime_rows:
            check = source_improvement_checks[
                (day.trading_date, row["symbol"])
            ]
            row["quote_improvement_probability"] = (
                f"{float(check['runtime_zero_split_probability']):.17g}"
            )
            generated_rate = canonical_daily_rate_path(
                output_root, day.trading_date, row["symbol"]
            ).resolve()
            if not generated_rate.is_file():
                raise PoolingError(
                    f"missing generated daily Hawkes rates: {generated_rate}"
                )
            row["hawkes_rates_file"] = str(generated_rate)
        atomic_csv(
            path, RUNTIME_CONFIG_FIELDS, runtime_rows, overwrite=args.overwrite
        )
        training_common.append({
            "date": day.trading_date,
            "source_config": str(day.path),
            "source_config_sha256": sha256_file(day.path),
            "common_config": str(path.resolve()),
            "common_config_sha256": sha256_file(path),
            "target_root": (
                str(target_roots[day.trading_date])
                if day.trading_date in target_roots else None
            ),
        })
    heldout_common_path = output_root / "heldout_common.csv"
    heldout_runtime_rows = freeze_pooled_backgrounds_with_heldout_openings(
        pooled_rows_by_symbol, heldout, symbols
    )
    atomic_csv(
        heldout_common_path, RUNTIME_CONFIG_FIELDS, heldout_runtime_rows,
        overwrite=args.overwrite,
    )

    pooled_config_path = output_root / "pooled_training_universe.csv"
    atomic_csv(
        pooled_config_path, RUNTIME_CONFIG_FIELDS, pooled_rows,
        overwrite=args.overwrite,
    )
    if args.require_certification_cohort:
        assert cohort_identity is not None
        cohort_artifacts = {
            "pooled_training_universe": cohort.validate_csv(
                pooled_config_path,
                label="pooled training universe",
                project_root=SCRIPT_DIR.parent,
            ),
            "heldout_common": cohort.validate_csv(
                heldout_common_path,
                label="frozen held-out opening universe",
                project_root=SCRIPT_DIR.parent,
            ),
            "training_days": {
                entry["date"]: cohort.validate_csv(
                    pathlib.Path(str(entry["common_config"])),
                    label=f"training universe {entry['date']}",
                    project_root=SCRIPT_DIR.parent,
                )
                for entry in training_common
            },
        }
    else:
        cohort_artifacts = None
    provenance_path = output_root / "pooling_provenance.json"
    provenance: dict[str, Any] = {
        "schema_version": 7,
        "method": "multi_day_direct_input_pooling_with_day_level_behavioural_wmm",
        "workflow_source_semantics_sha256": workflow_source_semantics_sha256(
            SCRIPT_DIR.parent
        ),
        "training_dates": [day.trading_date for day in days],
        "heldout_date": args.heldout_date,
        "intersection_symbol_count": len(intersection_symbols),
        "common_symbol_count": len(symbols),
        "certification_cohort_required": bool(
            args.require_certification_cohort
        ),
        "certification_input_selection": certification_input_selection,
        "cohort_identity": (
            {
                **cohort_identity,
                # These two values describe the frozen cohort's historical
                # R16 origin.  The observed input shape of this pool is kept
                # separately in certification_input_selection so a fresh
                # already-prefiltered extraction is never misreported as a
                # new 1,509-to-1,480 screen.
                "original_intersection_symbol_count": 1_509,
                "fixed_price_grid_excluded_symbol_count": 29,
                "artifact_checks": cohort_artifacts,
            }
            if cohort_identity is not None else None
        ),
        "opening_price_grid_eligibility": {
            "method": (
                "all training and held-out session-start BBOs must be on the "
                "simulator's fixed price grid and the opening bid must be in "
                "the one-cent quoting regime"
            ),
            "information_used": (
                "configuration opening bid/ask only; no held-out event rates, "
                "marks, intraday moments, or outcomes"
            ),
            "itch_price_unit_usd": 0.0001,
            "simulator_tick_size_price_units": (
                args.simulator_tick_size_price_units
            ),
            "simulator_tick_size_usd": (
                args.simulator_tick_size_price_units * 0.0001
            ),
            "minimum_opening_bid_price_units": (
                args.minimum_opening_bid_price_units
            ),
            "minimum_opening_bid_usd": (
                args.minimum_opening_bid_price_units * 0.0001
            ),
            "sessions_checked": [
                *[day.trading_date for day in days], heldout.trading_date
            ],
            "intersection_symbol_count": len(intersection_symbols),
            "eligible_symbol_count": len(symbols),
            "excluded_symbol_count": len(price_grid_exclusions),
            "excluded_symbols": price_grid_exclusions,
        },
        "qqq_book_id": 0,
        "pooling": {
            "histograms": "sum raw observed histogram weights across dates",
            "event_rates": (
                "observed event-type counts / total observed seconds followed "
                "by the recorded reduced-book directional-volume and "
                "best-depth transforms, then audited stationary Hawkes inversion"
            ),
            "target_means": "arithmetic mean of daily full-session moments",
            "target_scales": "daily uncertainty plus between-day variation",
            "opening_state": "latest training-day reference only; held-out opening replaces it before validation",
            "runtime_state_targets": (
                "one five-day pooled target_spread_ticks and one five-day pooled "
                "bid/ask mean-depth pair per symbol, plus a sparse latent-value "
                "process whose unconditional volatility, one-second move "
                "probability and conditional kurtosis come from pooled training "
                "moments; frozen identically "
                "into all training, development-validation and final runtime configs"
            ),
            "heldout_targets_used_for_runtime_configuration": False,
            "beta": "median training fundamental price / cross-sectional median",
            "cross_sectional_median_training_price_ticks": cross_sectional_median,
            "hawkes": {
                "activity_scale": args.activity_scale,
                "kernel_beta": args.hawkes_beta,
                "balance_directional_volume": args.balance_directional_volume,
                "balance_best_depth": args.balance_best_depth,
                "balance_strength": args.balance_strength,
                **hawkes.excitation_settings(),
            },
        },
        "pooling_parameters": {
            "minimum_common_symbols": args.minimum_symbols,
            "quote_quantity_fraction": args.quote_quantity_fraction,
            "minimum_quote_quantity": args.minimum_quote_quantity,
            "maximum_quote_quantity": args.maximum_quote_quantity,
            "pool_label": label,
        },
        "quote_improvement_runtime_approximation": dict(
            QUOTE_IMPROVEMENT_COMPATIBILITY
        ),
        "configuration_schema": {
            "schema_version": 5,
            "source_fields": list(CONFIG_FIELDS),
            "runtime_fields": list(RUNTIME_CONFIG_FIELDS),
            "runtime_fields_sha256": configuration_schema_sha256(
                RUNTIME_CONFIG_FIELDS
            ),
            "pooled_homeostatic_fields": list(POOLED_HOMEOSTATIC_FIELDS),
            "latent_value_fields": list(LATENT_VALUE_FIELDS),
            "frozen_training_derived_fields": list(
                FROZEN_TRAINING_DERIVED_FIELDS
            ),
            "queue_reactive_target_fields": list(
                QUEUE_REACTIVE_TARGET_FIELDS
            ),
            "positive_queue_reactive_targets_required": True,
            "same_pooled_targets_in_all_runtime_sessions": True,
            "heldout_target_files_used": False,
        },
        "training_days": training_common,
        "heldout": {
            "source_config": str(heldout.path),
            "source_config_sha256": sha256_file(heldout.path),
            "common_config": str(heldout_common_path.resolve()),
            "common_config_sha256": sha256_file(heldout_common_path),
            "target_root": str(heldout_target_root) if heldout_target_root is not None else None,
            "heldout_role": "opening_state_and_validation_targets_only",
            "opening_fields_copied_from_heldout": list(OPENING_FIELDS),
            "background_inputs_inherited_from_pooled": True,
            "quote_improvement_compatibility": {
                "status": "frozen_from_pooled_training",
                "symbol_count": len(symbols),
                "heldout_mark_inputs_instantiated": False,
                "runtime_probability_source": "pooled_training_universe.csv",
            },
        },
        "pooled_configuration": {
            "path": str(pooled_config_path.resolve()),
            "sha256": sha256_file(pooled_config_path),
        },
        "symbols": pooled_symbol_metadata,
    }
    atomic_json(provenance_path, provenance, overwrite=args.overwrite)
    return {
        "pooled_training_universe": str(pooled_config_path),
        "heldout_common": str(heldout_common_path),
        "provenance": str(provenance_path),
        "training_common_configs": [entry["common_config"] for entry in training_common],
        "intersection_symbol_count": len(intersection_symbols),
        "common_symbol_count": len(symbols),
        "price_grid_excluded_symbol_count": len(price_grid_exclusions),
        "certification_input_selection": certification_input_selection,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-day", nargs=2, action="append", metavar=("DATE", "CONFIG"),
        required=True,
        help="repeat for each chronologically earlier ITCH training session",
    )
    parser.add_argument(
        "--training-target-root", nargs=2, action="append", default=[],
        metavar=("DATE", "ROOT"),
        help=(
            "optional repeatable empirical-data root for each training date; when "
            "supplied for every day it is carried into pooling provenance for the "
            "multi-day behavioural-calibration driver"
        ),
    )
    parser.add_argument("--heldout-date", required=True)
    parser.add_argument("--heldout-config", required=True)
    parser.add_argument(
        "--heldout-target-root",
        help="optional held-out empirical-data root recorded in pooling provenance",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--minimum-symbols", type=int, default=20)
    parser.add_argument(
        "--require-certification-cohort",
        action="store_true",
        help=(
            "fail unless the post-grid balanced panel is the bundled, ordered "
            "1,480-symbol development-validation cohort"
        ),
    )
    parser.add_argument(
        "--simulator-tick-size-price-units",
        type=int,
        default=DEFAULT_SIMULATOR_TICK_SIZE_PRICE_UNITS,
        help=(
            "fixed simulator price increment in ITCH USD-0.0001 units "
            "(default: 100 = USD 0.01)"
        ),
    )
    parser.add_argument(
        "--minimum-opening-bid-price-units",
        type=int,
        default=DEFAULT_MINIMUM_OPENING_BID_PRICE_UNITS,
        help=(
            "minimum opening bid on every training and held-out session in "
            "ITCH USD-0.0001 units (default: 10000 = USD 1.00)"
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
        action="store_false",
    )
    parser.add_argument(
        "--balance-best-depth", dest="balance_best_depth", action="store_true",
        help="apply the reduced-book best-depth cancellation-rate transform",
    )
    parser.add_argument(
        "--no-balance-best-depth", dest="balance_best_depth", action="store_false",
    )
    # Compact independent mark sampling drops side dependence and exact order
    # references.  The canonical reduced-book mapping restores its two
    # predeclared moment constraints before the stationary Hawkes inversion.
    parser.set_defaults(balance_directional_volume=True, balance_best_depth=True)
    parser.add_argument("--quote-quantity-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-quote-quantity", type=int, default=10)
    parser.add_argument("--maximum-quote-quantity", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.minimum_symbols <= 0:
        parser.error("--minimum-symbols must be positive")
    if args.simulator_tick_size_price_units <= 0:
        parser.error("--simulator-tick-size-price-units must be positive")
    if args.minimum_opening_bid_price_units <= 0:
        parser.error("--minimum-opening-bid-price-units must be positive")
    if (args.minimum_opening_bid_price_units
            % args.simulator_tick_size_price_units != 0):
        parser.error(
            "--minimum-opening-bid-price-units must be a multiple of "
            "--simulator-tick-size-price-units"
        )
    if not math.isfinite(args.activity_scale) or args.activity_scale <= 0.0:
        parser.error("--activity-scale must be finite and positive")
    if not math.isfinite(args.hawkes_beta) or args.hawkes_beta <= 0.0:
        parser.error("--hawkes-beta must be finite and positive")
    if not math.isfinite(args.balance_strength) or not 0.0 <= args.balance_strength <= 5.0:
        parser.error("--balance-strength must be finite and between 0 and 5")
    if (not math.isfinite(args.quote_quantity_fraction)
            or args.quote_quantity_fraction <= 0.0):
        parser.error("--quote-quantity-fraction must be finite and positive")
    if not 1 <= args.minimum_quote_quantity <= args.maximum_quote_quantity <= 2_147_483_647:
        parser.error("invalid quote-quantity bounds")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(args, parser)
    try:
        result = run(args)
    except PoolingError as error:
        print(f"multi-day pooling failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
