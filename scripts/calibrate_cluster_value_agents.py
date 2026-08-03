#!/usr/bin/env python3
"""Calibrate a compact fragmented-LOB model with a block-coordinate protocol.

This is the behavioural-calibration counterpart to the direct ITCH extractor.
It deliberately *does not* fit a separate strategic agent to every symbol.
Instead, an externally prepared ``cluster_assignments.csv`` partitions the
empirical universe into liquidity clusters and this script selects one small
policy

``(enabled, value_threshold_bps, value_depth_participation)``

per cluster.  The two numerical controls are exactly the controls used by
``fragmented_mpi_lob``'s coarse-grained value agent.  Per-symbol event rates,
mark distributions, opening books, local-market-maker inputs and other
empirical book parameters are direct inputs from ITCH and are never part of
this behavioural search.  The value agent observes a rank-independent latent
fundamental whose per-symbol volatility is derived from the five training
sessions' pooled one-second return variance; the held-out return series is
never used to construct that process.

Every block uses the same matching empirical horizons:

* 300 seconds: inexpensive structural screen against the first 300 seconds of
  the training session;
* 3,600 seconds: multi-seed refinement against the matching prefix; and
* 23,400 seconds: full-session selection against the full-session target.

The blocks are deliberately separated so that the search is statistically
auditable and does not become a prohibitively large Cartesian product:

1. A global local-flow block keeps the Hawkes activity scale fixed at the
   value used when the ITCH rates were inverted, and first tests a nested
   local-MM-off baseline before enabled refresh/quantity controls.  Shared and
   value agents are disabled and the objective is restricted to spread,
   top-depth and book-integrity moments.  Mid-price changes are not treated as
   an order-arrival proxy.
2. With that triple frozen, the existing cluster-level value-policy search
   selects ``(enabled, value_threshold_bps, value_depth_participation)`` with the
   shared supplier disabled.
3. With local flow and cluster policies frozen, a nested shared-MM-off baseline
   is compared with symbol-relative shared-quote multipliers.

After all cluster policies are selected, the script creates a full-universe
policy CSV and validates it without refitting on a stratified sample from every
cluster.  In single-day mode the validation configuration copies that training
session's direct inputs; with multiple training sessions it copies an explicit
pooled direct-input configuration.  In both cases it replaces only the
held-out opening midpoint and BBO/depth fields.  By default, the resulting
pooled sample distribution is labelled exactly that: it is not claimed to be a
full-market distribution.  ``--marketwide-validation`` adds an explicit
full-universe held-out run and is the only mode labelled a market-wide
distributional validation.  Under the immutable certification profile the
stratified execution, two-sided coverage and source-attributed boundary checks
remain required, while its empirical-fit score is a required reported
diagnostic.  The exact full-universe market-wide empirical fit is the
authoritative held-out fit gate.  No held-out result is used for selection.

The tool accepts a direct rank-one executable invocation by default.  On a
cluster, pass a complete one-rank launcher prefix, for example::

    --launcher 'mpirun --bind-to core --map-by slot -np 1'

No MPI work is launched by this Python process itself; the optional launcher
is simply prepended to each executable command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import pathlib
import shlex
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import certification_cohort as cohort  # noqa: E402


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

# Candidate selection deliberately does not use the raw empirical-standard-error
# WMM statistic below.  That statistic remains valuable as a goodness-of-fit
# diagnostic, but with millions of ITCH observations a single precisely
# estimated (and structurally misspecified) moment can dominate every other
# stylised fact.  The training selector therefore compares dimensionless,
# economically interpretable residuals and gives every metric one equal vote.
POSITIVE_LOG_RATIO_METRICS = frozenset({
    "background_event_rate",
    "mean_spread_ticks",
    "mean_bid_depth",
    "mean_ask_depth",
    "return_variance",
    "return_kurtosis",
})
ROBUST_LOG_RATIO_UNIT = math.log(1.5)
ROBUST_MID_MOVE_LOG_ODDS_UNIT = math.log(2.0)
ROBUST_ACF_FISHER_UNIT = 0.25
ROBUST_COVERAGE_UNIT = 0.01
ROBUST_PROBABILITY_EPSILON = 1.0e-6
DEFAULT_ROBUST_HUBER_DELTA = 2.0
DEFAULT_DAY_STABILITY_PENALTY = 0.25

HAWKES_EXCITATION_SETTINGS: dict[str, float | str] = {
    "excitation_structure": "diagonal_self_excitation_only",
    "self_excitation_amplitude": 0.20,
    "cross_excitation_amplitude": 0.0,
}

# This is a named, immutable case-study gate.  These values are deliberately
# not command-line options: a looser post-hoc threshold must not be able to
# turn a preliminary fit into a certified artifact.  A future protocol must
# use a new gate identifier and update the downstream verifier explicitly.
CERTIFICATION_GATE_ID = "development_validation_gate"
CERTIFICATION_MAXIMUM_ROBUST_SCORE = 2.0
CERTIFICATION_MAXIMUM_METRIC_SCORE = 3.0
CERTIFICATION_GROSS_RESIDUAL_LIMIT = 6.0
CERTIFICATION_MAXIMUM_TWO_SIDED_SHORTFALL = 0.01
VALIDATION_ROLE = "development_validation_after_protocol_revision"
INDEPENDENT_FINAL_HOLDOUT = False
FIXED_HAWKES_ACTIVITY_SCALE = 0.30
CERTIFICATION_SESSION_START = "09:30:00"
CERTIFICATION_SESSION_END = "16:00:00"
CERTIFICATION_SESSION_DURATION_SECONDS = 23_400
CERTIFICATION_SNAPSHOT_INTERVAL_MS = 1_000
CERTIFICATION_STAGE3_SEEDS = (1729, 7919, 1103, 6599, 2027)
# Independent post-selection adequacy seeds are deliberately distinct from
# candidate-selection seeds.  The exact values are part of the immutable
# protocol so changing the acceptance scope cannot change the evidence.
CERTIFICATION_TRAINING_ADEQUACY_SEEDS = (
    3424815697, 1799108475, 2301941028, 3637917665, 3007455382,
)
CERTIFICATION_TRAINING_DATES = (
    "2019-01-30", "2019-03-27", "2019-07-30", "2019-10-30", "2019-12-30",
)
CERTIFICATION_VALIDATION_DATE = "2020-01-30"
CERTIFICATION_COMMON_SYMBOL_COUNT = 1480
CERTIFICATION_CLUSTER_COUNT = 10
CERTIFICATION_TRAINING_REPRESENTATIVES_PER_CLUSTER = 3
CERTIFICATION_VALIDATION_SYMBOLS_PER_CLUSTER = 3
CERTIFICATION_STAGE1_DURATION_SECONDS = 300
CERTIFICATION_STAGE2_DURATION_SECONDS = 3_600
CERTIFICATION_STAGE1_SEEDS = (1729,)
CERTIFICATION_STAGE2_SEEDS = (1729, 7919)
CERTIFICATION_STAGE1_SURVIVORS = 6
CERTIFICATION_STAGE1_REFINEMENT_CANDIDATES = 32
CERTIFICATION_STAGE2_SURVIVORS = 2
LOCAL_FLOW_STAGE1_PROMOTION = "all_structurally_eligible"
LOCAL_FLOW_STAGE2_PROMOTION = "all_structurally_eligible"
LOCAL_FLOW_STAGE3_SELECTION = (
    "best_training_fit_among_structurally_eligible"
)
VALUE_POLICY_STAGE1_PROMOTION = (
    "all_structurally_eligible_threshold_depth_policies_plus_disabled_baseline"
)
VALUE_POLICY_STAGE2_PROMOTION = VALUE_POLICY_STAGE1_PROMOTION
CERTIFICATION_VALUE_THRESHOLDS_BPS = (5.0, 8.0, 10.0, 15.0, 25.0, 40.0)
STRATIFIED_EMPIRICAL_FIT_ACCEPTANCE_ROLE = "required_reported_diagnostic_only"
STRATIFIED_EMPIRICAL_FIT_FAILURE_SCOPE = "held-out stratified"
MARKETWIDE_EMPIRICAL_FIT_ACCEPTANCE_ROLE = (
    "authoritative_certification_gate"
)
MARKETWIDE_STATUS_SCHEMA_VERSION = 2
SELECTION_CHECKPOINT_SCHEMA_VERSION = 2
CERTIFICATION_VALUE_DEPTH_PARTICIPATIONS = (0.05, 0.1, 0.25, 0.5)
CERTIFICATION_VALUE_POLICIES_PER_DEPTH = len(
    CERTIFICATION_VALUE_THRESHOLDS_BPS
)
CERTIFICATION_VALUE_FULL_DAY_CANDIDATES = (
    1
    + len(CERTIFICATION_VALUE_THRESHOLDS_BPS)
    * len(CERTIFICATION_VALUE_DEPTH_PARTICIPATIONS)
)
CERTIFICATION_LOCAL_MM_INTERVALS_MS = (500.0, 1000.0, 2000.0)
CERTIFICATION_LOCAL_MM_QUANTITY_MULTIPLIERS = (0.5, 1.0, 2.0)
CERTIFICATION_LOCAL_MM_IMPROVEMENT_PROBABILITIES = (0.0, 0.25, 0.5, 1.0)
CERTIFICATION_SHARED_QUOTE_MULTIPLIERS = (0.5, 1.0, 2.0)
CERTIFICATION_SHARED_QUOTE_CANDIDATE_COUNT = (
    1 + len(CERTIFICATION_SHARED_QUOTE_MULTIPLIERS)
)
CERTIFICATION_SHARED_QUOTE_STAGE1_SURVIVOR_CAP = (
    CERTIFICATION_STAGE1_SURVIVORS
)
CERTIFICATION_SHARED_QUOTE_STAGE1_PROMOTED_COUNT = min(
    CERTIFICATION_SHARED_QUOTE_CANDIDATE_COUNT,
    CERTIFICATION_SHARED_QUOTE_STAGE1_SURVIVOR_CAP,
)
CERTIFICATION_SHARED_QUOTE_STAGE2_SURVIVOR_CAP = (
    CERTIFICATION_STAGE2_SURVIVORS
)
CERTIFICATION_SHARED_QUOTE_STAGE2_PROMOTED_COUNT = min(
    CERTIFICATION_SHARED_QUOTE_STAGE1_PROMOTED_COUNT,
    CERTIFICATION_SHARED_QUOTE_STAGE2_SURVIVOR_CAP,
)
CERTIFICATION_SHARED_QUOTE_STAGE3_SURVIVOR_CAP = 1
CERTIFICATION_SHARED_QUOTE_STAGE3_PROMOTED_COUNT = min(
    CERTIFICATION_SHARED_QUOTE_STAGE2_PROMOTED_COUNT,
    CERTIFICATION_SHARED_QUOTE_STAGE3_SURVIVOR_CAP,
)
CERTIFICATION_SHARED_TREATMENT_MULTIPLIER = 1.0
CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_EVENT_RATIO = 0.05
CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_QUANTITY_RATIO = 0.05
CERTIFICATION_MAXIMUM_RUN_BOUNDARY_EVENT_RATIO = 0.01
CERTIFICATION_MAXIMUM_RUN_BOUNDARY_QUANTITY_RATIO = 0.01
BOUNDARY_SUMMARY_FIELDS = (
    "background_event_count",
    "background_market_requested_quantity",
    "background_cancel_requested_quantity",
    "removal_boundary_truncation_events",
    "removal_boundary_truncated_quantity",
    "background_boundary_truncation_events",
    "background_boundary_truncated_quantity",
    "value_order_count",
    "value_requested_quantity",
    "value_boundary_truncation_events",
    "value_boundary_truncated_quantity",
    "other_boundary_truncation_events",
    "other_boundary_truncated_quantity",
)
BOUNDARY_ACTION_SUMMARY_FIELDS = (
    "market_boundary_truncation_events",
    "market_boundary_truncated_quantity",
    "cancel_boundary_truncation_events",
    "cancel_boundary_truncated_quantity",
)
BOUNDARY_SOURCE_SUMMARY_FIELDS = (
    "background_boundary_truncation_events",
    "background_boundary_truncated_quantity",
    "value_boundary_truncation_events",
    "value_boundary_truncated_quantity",
    "other_boundary_truncation_events",
    "other_boundary_truncated_quantity",
)
QUOTE_IMPROVEMENT_RUNTIME_APPROXIMATION = {
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

SIMULATOR_EMPIRICAL_INPUT_FILENAMES = (
    "limit_buy_quantity_distribution.txt",
    "limit_sell_quantity_distribution.txt",
    "market_buy_quantity_distribution.txt",
    "market_sell_quantity_distribution.txt",
    "cancel_bid_quantity_distribution.txt",
    "cancel_ask_quantity_distribution.txt",
    "limit_buy_distance_distribution.txt",
    "limit_sell_distance_distribution.txt",
    "cancel_bid_distance_distribution.txt",
    "cancel_ask_distance_distribution.txt",
)

BACKGROUND_EVENT_NAMES = (
    "limit_buy", "limit_sell", "market_buy", "market_sell",
    "cancel_bid", "cancel_ask",
)

# These files jointly define the executable calibration/case-study workflow.
# C++ source is hashed separately so the report can distinguish simulator
# semantics from orchestration, calibration and analysis semantics.
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

# Hawkes rate files already invert ITCH event rates at activity scale 0.30.
# Re-selecting that scale against mid-price movement would confound activity
# with price response.  Block 1 therefore freezes it and calibrates only the
# local liquidity supplier against book-state moments.
LOCAL_FLOW_METRICS = (
    "background_event_rate",
    "mean_spread_ticks",
    "mean_bid_depth",
    "mean_ask_depth",
    "two_sided_sample_fraction",
)

# The two-candidate structural preflight asks a narrower question than local
# policy selection: can the repaired background/book mechanics sustain an
# empirically plausible displayed queue before the expensive grid is started?
# Spread is deliberately excluded because the local market maker is the model
# component calibrated to repair spread; requiring a background-only control
# to match it would reject the intended decomposition rather than a structural
# defect.
STRUCTURAL_PREFLIGHT_DEPTH_METRICS = (
    "mean_bid_depth",
    "mean_ask_depth",
)

DECISION_WINDOW_MS = 1000.0

BASE_CONFIG_FIELDS = (
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
    BASE_CONFIG_FIELDS + LATENT_VALUE_FIELDS + QUEUE_REACTIVE_TARGET_FIELDS
)
CONFIG_REQUIRED_FIELDS = RUNTIME_CONFIG_FIELDS
POOLED_HOMEOSTATIC_FIELDS = (
    "target_spread_ticks",
    *QUEUE_REACTIVE_TARGET_FIELDS,
)
FROZEN_TRAINING_DERIVED_FIELDS = (
    *POOLED_HOMEOSTATIC_FIELDS,
    *LATENT_VALUE_FIELDS,
)
RUNTIME_CONFIG_SCHEMA_VERSION = 5
POOLING_PROVENANCE_SCHEMA_VERSION = 7

# These files are terminal statements about one completed attempt.  In
# particular, ``calibration_handoff.json`` grants downstream authority to run
# the case study.  An overwrite attempt must revoke that authority before any
# new validation can fail, while retaining non-terminal checkpoints and run
# directories for diagnosis.
TERMINAL_CALIBRATION_ARTIFACT_FILENAMES = (
    "calibration_handoff.json",
    "independent_global_calibration_certification.json",
    "preliminary_calibration_result.json",
    "cluster_value_agent_calibration_report.json",
    "calibration_failure.json",
)

# These are observations at the new day's opening.  Every other configuration
# field is frozen: changing e.g. Hawkes rates or mark distributions would turn
# held-out validation into a second direct calibration.
HELDOUT_OPENING_FIELDS = (
    "fundamental_price_ticks",
    "initial_best_bid_ticks",
    "initial_best_ask_ticks",
    "initial_best_bid_depth",
    "initial_best_ask_depth",
)

POLICY_FIELDS = (
    "symbol",
    "enabled",
    "value_threshold_bps",
    "value_depth_participation",
    "cluster_id",
    "cluster_label",
    "policy_source",
)

DETAIL_FIELDS = (
    "phase",
    "cluster_id",
    "cluster_label",
    "candidate_index",
    "candidate_label",
    "enabled",
    "value_threshold_bps",
    "value_depth_participation",
    "hawkes_activity_scale",
    "local_mm_enabled",
    "local_mm_interval_ms",
    "local_mm_quantity_multiplier",
    "local_mm_improvement_probability",
    "shared_mm_enabled",
    "shared_quote_multiplier",
    "fit_wsmrmse",
    "combined_uncertainty_wsmrmse",
    "selection_score",
    "seed_count",
    "training_day_count",
    "aggregation",
    "errors",
)

VALIDATION_DETAIL_FIELDS = (
    "phase",
    "scope",
    "cluster_id",
    "cluster_label",
    "symbol",
    "metric",
    "target",
    "empirical_scale",
    "weight",
    "simulated_mean",
    "simulated_sample_sd",
    "simulated_mean_se",
    "combined_scale",
    "empirical_standardized_residual",
    "combined_uncertainty_residual",
    "objective_residual",
    "weighted_squared_residual",
    "seed_count",
)

DISTRIBUTION_FIELDS = (
    "scope",
    "metric",
    "symbol_count",
    "target_mean",
    "simulated_mean",
    "target_median",
    "simulated_median",
    "target_p10",
    "simulated_p10",
    "target_p90",
    "simulated_p90",
    "mean_difference",
    "median_difference",
)


class CalibrationError(ValueError):
    """Raised for an invalid or leakage-prone calibration protocol."""


@dataclass(frozen=True)
class TrainingDay:
    """One complete empirical session used only for policy selection.

    The direct inputs remain session-specific while candidates are being
    scored.  This is important: a five-day calibration is not implemented by
    averaging prices, depths, or file paths into a fictitious ``average day``.
    A separately prepared pooled direct-input configuration is used only after
    selection, when the frozen model is initialised for the held-out session.
    """

    date: str
    universe_config: pathlib.Path
    target_root: pathlib.Path
    fields: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    universe_config_sha256: str

    @property
    def identifier(self) -> str:
        return f"day_{compact_date(self.date)}"


@dataclass(frozen=True)
class TargetMoment:
    """One empirical moment and its declared diagonal WMM scale and weight."""

    target: float
    empirical_scale: float
    weight: float


@dataclass(frozen=True)
class MomentEstimate:
    """A seed-averaged simulated estimate used by the WMM objective."""

    symbol: str
    metric: str
    target: float
    empirical_scale: float
    weight: float
    simulated_mean: float
    simulated_sample_sd: float
    simulated_mean_se: float
    combined_scale: float
    empirical_standardized_residual: float
    combined_uncertainty_residual: float
    objective_residual: float
    weighted_squared_residual: float
    seed_count: int


@dataclass(frozen=True)
class Candidate:
    """A candidate cluster-wide policy for the actual fragmented value agent."""

    enabled: bool
    threshold_bps: float
    depth_participation: float
    label: str


@dataclass(frozen=True)
class LocalFlowCandidate:
    """Global local-flow controls accepted by ``fragmented_mpi_lob``."""

    hawkes_activity_scale: float
    local_mm_interval_ms: float
    local_mm_quantity_multiplier: float
    label: str
    local_mm_enabled: bool = True
    local_mm_improvement_probability: float = 0.0


@dataclass(frozen=True)
class SharedQuoteCandidate:
    """A nested shared-liquidity baseline or symbol-relative treatment."""

    enabled: bool
    multiplier: float
    label: str


@dataclass(frozen=True)
class ClusterLayout:
    """Validated membership, representative, and held-out sample metadata."""

    by_symbol: Mapping[str, int]
    representatives: Mapping[int, tuple[str, ...]]
    validation_symbols: Mapping[int, tuple[str, ...]]

    @property
    def cluster_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.representatives))


def compact_date(value: str) -> str:
    """Return an eight-digit date after requiring a real ISO calendar date."""
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise CalibrationError(f"invalid ISO date: {value!r}") from error
    return parsed.strftime("%Y%m%d")


def normalise_symbol(value: object, *, label: str) -> str:
    symbol = str(value).strip().upper()
    if not symbol:
        raise CalibrationError(f"empty symbol in {label}")
    if any(character.isspace() for character in symbol) or "," in symbol:
        raise CalibrationError(f"unsafe symbol {value!r} in {label}")
    return symbol


def parse_cluster_id(value: object, *, label: str) -> int:
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"invalid cluster_id {value!r} in {label}") from error
    if result < 0:
        raise CalibrationError(f"negative cluster_id in {label}: {result}")
    return result


def parse_bool(value: object, *, label: str) -> bool:
    rendered = str(value).strip().lower()
    if rendered in {"1", "true", "yes", "on"}:
        return True
    if rendered in {"0", "false", "no", "off"}:
        return False
    raise CalibrationError(f"invalid boolean {value!r} in {label}")


def finite_float(value: object, *, label: str) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"invalid floating-point value for {label}: {value!r}") from error
    if not math.isfinite(result):
        raise CalibrationError(f"non-finite floating-point value for {label}: {value!r}")
    return result


def csv_table(path: pathlib.Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Read a non-empty CSV table, retaining header order for config rewriting."""
    if not path.is_file():
        raise CalibrationError(f"missing CSV file: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise CalibrationError(f"CSV file has no header: {path}")
        fields = tuple(field.strip() for field in reader.fieldnames)
        if any(not field for field in fields) or len(set(fields)) != len(fields):
            raise CalibrationError(f"CSV has an invalid or duplicate header: {path}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if row is None:
                continue
            if None in row:
                raise CalibrationError(f"too many columns in {path}:{line_number}")
            rows.append({field: (row.get(field) or "").strip() for field in fields})
    if not rows:
        raise CalibrationError(f"CSV file has no data rows: {path}")
    return fields, rows


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration_schema_sha256(fields: Sequence[str]) -> str:
    encoded = json.dumps(
        list(fields), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def certification_profile() -> dict[str, object]:
    """Return the immutable eligibility gate committed with this source."""
    return {
        "profile_id": CERTIFICATION_GATE_ID,
        "certification_profile_enforced": True,
        "validation_role": VALIDATION_ROLE,
        "independent_final_holdout": INDEPENDENT_FINAL_HOLDOUT,
        "required_session_duration_seconds": CERTIFICATION_SESSION_DURATION_SECONDS,
        "required_training_dates": list(CERTIFICATION_TRAINING_DATES),
        "required_validation_date": CERTIFICATION_VALIDATION_DATE,
        "required_common_symbol_count": CERTIFICATION_COMMON_SYMBOL_COUNT,
        "required_common_symbol_order_sha256": (
            cohort.REQUIRED_SYMBOL_ORDER_SHA256
        ),
        "cohort_identity": {
            "cohort_file": cohort.COHORT_RELATIVE_PATH.as_posix(),
            "cohort_symbol_count": cohort.REQUIRED_SYMBOL_COUNT,
            "cohort_symbol_order_sha256": (
                cohort.REQUIRED_SYMBOL_ORDER_SHA256
            ),
            "canonical_order": "QQQ_first_then_lexicographic",
            "origin_manifest": (
                cohort.ORIGIN_MANIFEST_RELATIVE_PATH.as_posix()
            ),
            "selection_role": "development_validation_balanced_panel",
            "heldout_availability_conditioned": True,
            "heldout_target_values_used": False,
            "independent_final_holdout": False,
            "original_intersection_symbol_count": 1509,
            "fixed_price_grid_excluded_symbol_count": 29,
            "final_symbol_count": CERTIFICATION_COMMON_SYMBOL_COUNT,
            "pooled_training_universe_csv_sha256": (
                cohort.POOLED_TRAINING_CSV_SHA256
            ),
            "pooling_provenance_sha256": (
                cohort.POOLING_PROVENANCE_SHA256
            ),
            "interpretation": (
                "fixed development-validation balanced panel conditioned on "
                "symbol availability and opening-price-grid compatibility on "
                "2020-01-30; no held-out target value entered cohort selection"
            ),
        },
        "required_cluster_count": CERTIFICATION_CLUSTER_COUNT,
        "empirical_target_session": {
            "session_start": CERTIFICATION_SESSION_START,
            "session_end": CERTIFICATION_SESSION_END,
            "duration_seconds": CERTIFICATION_SESSION_DURATION_SECONDS,
            "snapshot_interval_ms": CERTIFICATION_SNAPSHOT_INTERVAL_MS,
            "full_session_observations": (
                CERTIFICATION_SESSION_DURATION_SECONDS
            ),
        },
        "required_training_representatives_per_cluster": (
            CERTIFICATION_TRAINING_REPRESENTATIVES_PER_CLUSTER
        ),
        "required_validation_symbols_per_cluster": (
            CERTIFICATION_VALIDATION_SYMBOLS_PER_CLUSTER
        ),
        "stage1_duration_seconds": CERTIFICATION_STAGE1_DURATION_SECONDS,
        "stage2_duration_seconds": CERTIFICATION_STAGE2_DURATION_SECONDS,
        "asset_summary_interval_ms": int(DECISION_WINDOW_MS),
        "required_stage1_seeds": list(CERTIFICATION_STAGE1_SEEDS),
        "required_stage2_seeds": list(CERTIFICATION_STAGE2_SEEDS),
        "required_stage3_seeds": list(CERTIFICATION_STAGE3_SEEDS),
        "shared_quote_candidate_count": (
            CERTIFICATION_SHARED_QUOTE_CANDIDATE_COUNT
        ),
        "shared_quote_stage1_survivor_cap": (
            CERTIFICATION_SHARED_QUOTE_STAGE1_SURVIVOR_CAP
        ),
        "shared_quote_stage1_promoted_candidates": (
            CERTIFICATION_SHARED_QUOTE_STAGE1_PROMOTED_COUNT
        ),
        "local_flow_stage1_refinement_leaders": (
            CERTIFICATION_STAGE1_SURVIVORS
        ),
        "stage1_refinement_candidates": (
            CERTIFICATION_STAGE1_REFINEMENT_CANDIDATES
        ),
        "shared_quote_stage2_survivor_cap": (
            CERTIFICATION_SHARED_QUOTE_STAGE2_SURVIVOR_CAP
        ),
        "shared_quote_stage2_promoted_candidates": (
            CERTIFICATION_SHARED_QUOTE_STAGE2_PROMOTED_COUNT
        ),
        "shared_quote_stage3_survivor_cap": (
            CERTIFICATION_SHARED_QUOTE_STAGE3_SURVIVOR_CAP
        ),
        "shared_quote_stage3_promoted_candidates": (
            CERTIFICATION_SHARED_QUOTE_STAGE3_PROMOTED_COUNT
        ),
        "local_flow_stage1_promotion": LOCAL_FLOW_STAGE1_PROMOTION,
        "local_flow_stage2_promotion": LOCAL_FLOW_STAGE2_PROMOTION,
        "local_flow_stage3_selection": LOCAL_FLOW_STAGE3_SELECTION,
        "value_policy_stage1_promotion": VALUE_POLICY_STAGE1_PROMOTION,
        "value_policy_stage2_promotion": VALUE_POLICY_STAGE2_PROMOTION,
        "value_policy_stage1_survivors_per_depth": (
            CERTIFICATION_VALUE_POLICIES_PER_DEPTH
        ),
        "value_policy_stage2_survivors_per_depth": (
            CERTIFICATION_VALUE_POLICIES_PER_DEPTH
        ),
        "value_policy_stage3_candidates_per_cluster": (
            CERTIFICATION_VALUE_FULL_DAY_CANDIDATES
        ),
        "structural_preflight": {
            "required_candidate_roles": [
                "background_only",
                "enabled_local_mm_reference",
            ],
            "duration_seconds": CERTIFICATION_STAGE2_DURATION_SECONDS,
            "seeds": list(CERTIFICATION_STAGE2_SEEDS),
            "empirical_admissibility_metrics": list(
                STRUCTURAL_PREFLIGHT_DEPTH_METRICS
            ),
            "maximum_robust_score": CERTIFICATION_MAXIMUM_ROBUST_SCORE,
            "maximum_metric_score": CERTIFICATION_MAXIMUM_METRIC_SCORE,
            "maximum_symbol_metric_absolute_robust_residual": (
                CERTIFICATION_GROSS_RESIDUAL_LIMIT
            ),
            "gross_symbol_metric_failures_role": (
                "diagnostic_only_during_structural_preflight"
            ),
            "zero_gross_symbol_metric_failures_required": False,
            "strict_gross_symbol_gate_retained_for_development_validation": False,
            "two_sided_integrity_required": True,
            "finite_boundary_adequacy_required": True,
            "both_candidates_must_pass": True,
            "spread_excluded_because_local_mm_is_spread_repair": True,
            "training_targets_only": True,
        },
        "value_thresholds_bps": list(CERTIFICATION_VALUE_THRESHOLDS_BPS),
        "value_depth_participations": list(
            CERTIFICATION_VALUE_DEPTH_PARTICIPATIONS
        ),
        "hawkes_activity_scales": [FIXED_HAWKES_ACTIVITY_SCALE],
        "local_mm_intervals_ms": list(CERTIFICATION_LOCAL_MM_INTERVALS_MS),
        "local_mm_quantity_multipliers": list(
            CERTIFICATION_LOCAL_MM_QUANTITY_MULTIPLIERS
        ),
        "local_mm_improvement_probabilities": list(
            CERTIFICATION_LOCAL_MM_IMPROVEMENT_PROBABILITIES
        ),
        "shared_quote_multipliers": list(CERTIFICATION_SHARED_QUOTE_MULTIPLIERS),
        "shared_treatment_multiplier": CERTIFICATION_SHARED_TREATMENT_MULTIPLIER,
        "background_event_rate_acceptance_required": True,
        "build_provenance_required": True,
        "workflow_source_semantics_required": True,
        "clustering_protocol": {
            "algorithm": (
                "deterministic_farthest_first_lloyd_kmeans_"
                "with_minimum_size_repair"
            ),
            "seed": 20200130,
            "minimum_cluster_size": 6,
            "features": [
                "event_rate_per_second", "mean_spread_ticks", "mean_top_depth",
                "return_variance", "opening_mid_price_ticks",
            ],
        },
        "pooling_protocol": {
            "activity_scale": 0.3,
            "hawkes_beta": 10.0,
            "balance_directional_volume": True,
            "balance_best_depth": True,
            "balance_strength": 1.0,
            **HAWKES_EXCITATION_SETTINGS,
            "simulator_tick_size_price_units": 100,
            "minimum_opening_bid_price_units": 10000,
            "minimum_common_symbols": 20,
            "quote_quantity_fraction": 0.5,
            "minimum_quote_quantity": 10,
            "maximum_quote_quantity": 1000,
            "pool_label": "five_2019_sessions",
            "runtime_configuration_schema_version": (
                RUNTIME_CONFIG_SCHEMA_VERSION
            ),
            "runtime_configuration_schema_sha256": (
                configuration_schema_sha256(RUNTIME_CONFIG_FIELDS)
            ),
            "pooled_homeostatic_fields": list(POOLED_HOMEOSTATIC_FIELDS),
            "latent_value_fields": list(LATENT_VALUE_FIELDS),
            "frozen_training_derived_fields": list(
                FROZEN_TRAINING_DERIVED_FIELDS
            ),
            "heldout_target_files_used_for_runtime_configuration": False,
        },
        "marketwide_validation_required": True,
        "heldout_validation_acceptance_protocol": {
            "authoritative_empirical_fit_scope": (
                "full_universe_marketwide"
            ),
            "stratified": {
                "required": True,
                "scope": "three_nonrepresentative_symbols_per_cluster",
                "required_symbol_count": (
                    CERTIFICATION_CLUSTER_COUNT
                    * CERTIFICATION_VALIDATION_SYMBOLS_PER_CLUSTER
                ),
                "execution_integrity_required": True,
                "two_sided_clock_required": True,
                "empirical_coverage_required": True,
                "background_boundary_adequacy_required": True,
                "value_boundary_adequacy_required": True,
                "empirical_fit_computation_required": True,
                "empirical_fit_acceptance_role": (
                    STRATIFIED_EMPIRICAL_FIT_ACCEPTANCE_ROLE
                ),
            },
            "marketwide": {
                "required": True,
                "scope": "all_1480_common_symbols",
                "required_symbol_count": CERTIFICATION_COMMON_SYMBOL_COUNT,
                "execution_integrity_required": True,
                "two_sided_clock_required": True,
                "background_boundary_adequacy_required": True,
                "value_boundary_adequacy_required": True,
                "empirical_fit_computation_required": True,
                "empirical_fit_acceptance_role": (
                    MARKETWIDE_EMPIRICAL_FIT_ACCEPTANCE_ROLE
                ),
                "maximum_robust_score": (
                    CERTIFICATION_MAXIMUM_ROBUST_SCORE
                ),
                "maximum_metric_score": (
                    CERTIFICATION_MAXIMUM_METRIC_SCORE
                ),
                "maximum_symbol_metric_absolute_robust_residual": (
                    CERTIFICATION_GROSS_RESIDUAL_LIMIT
                ),
            },
            "heldout_information_used_for_selection": False,
            "thresholds_changed_from_v17": False,
            "seeds_changed_from_v17": False,
        },
        "model_semantics": {
            "local_market_maker": (
                "owned_queue_and_spread_reactive_one_tick_limit_quotes"
            ),
            "value_agent": (
                "contrarian_market_order_protected_at_perceived_fundamental_"
                "and_sized_as_a_cluster_calibrated_fraction_of_displayed_"
                "opposite_side_depth_"
                "against_rank_independent_sparse_"
                "training_moment_latent_value"
            ),
            "finite_book_reserve": "final_displayed_share_not_owner_zero_share",
        },
        "nested_policy_selection": {
            "disabled_baseline_promoted_through_stage2": True,
            "each_depth_participation_stratum_promoted_through_stage2": True,
            "all_threshold_depth_policies_promoted_through_stage2": True,
            "complete_grid_eligibility_required_at_each_stage": True,
            "full_day_selection": "best_training_fit_among_eligible_candidates",
            "heldout_information_used_for_selection": False,
        },
        "full_universe_training_adequacy": {
            "required_before_development_validation": True,
            "scope": "all_common_symbols_on_every_training_date",
            "required_training_dates": list(CERTIFICATION_TRAINING_DATES),
            "session_duration_seconds": (
                CERTIFICATION_SESSION_DURATION_SECONDS
            ),
            "seeds": list(CERTIFICATION_TRAINING_ADEQUACY_SEEDS),
            "seed_derivation": "predeclared_sha256_derived_seed_set",
            "seed_set_inherited_from_profile_id": CERTIFICATION_GATE_ID,
            "every_training_day_must_pass": True,
            "maximum_aggregate_robust_score": (
                CERTIFICATION_MAXIMUM_ROBUST_SCORE
            ),
            "maximum_day_robust_score": CERTIFICATION_MAXIMUM_ROBUST_SCORE,
            "maximum_day_metric_score": CERTIFICATION_MAXIMUM_METRIC_SCORE,
            "two_sided_integrity_required": True,
            "finite_boundary_adequacy_required": True,
            "development_validation_targets_opened": False,
        },
        "gross_symbol_metric_failures_role": (
            "diagnostic_outliers_under_cluster_level_calibration"
        ),
        "gross_symbol_metric_failures_required_for_acceptance": False,
        "simulated_two_sided_fraction_required": 1.0,
        "maximum_robust_score": CERTIFICATION_MAXIMUM_ROBUST_SCORE,
        "maximum_metric_score": CERTIFICATION_MAXIMUM_METRIC_SCORE,
        "maximum_symbol_metric_absolute_robust_residual": (
            CERTIFICATION_GROSS_RESIDUAL_LIMIT
        ),
        "maximum_two_sided_shortfall_diagnostic": (
            CERTIFICATION_MAXIMUM_TWO_SIDED_SHORTFALL
        ),
        "finite_boundary_adequacy": {
            "model_adequacy_gate": True,
            "source_attribution_required": True,
            "background_gate_scope": (
                "per_symbol_pooled_across_predeclared_seeds_and_"
                "market_aggregate_pooled_across_symbols_and_seeds"
            ),
            "maximum_asset_event_ratio": (
                CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_EVENT_RATIO
            ),
            "maximum_asset_quantity_ratio": (
                CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_QUANTITY_RATIO
            ),
            "maximum_run_event_ratio": (
                CERTIFICATION_MAXIMUM_RUN_BOUNDARY_EVENT_RATIO
            ),
            "maximum_run_quantity_ratio": (
                CERTIFICATION_MAXIMUM_RUN_BOUNDARY_QUANTITY_RATIO
            ),
            "event_ratio": (
                "background_boundary_truncation_events / "
                "background_event_count"
            ),
            "quantity_ratio": (
                "background_boundary_truncated_quantity / "
                "(background_market_requested_quantity + "
                "background_cancel_requested_quantity)"
            ),
            "value_event_ratio": (
                "value_boundary_truncation_events / value_order_count"
            ),
            "value_quantity_ratio": (
                "value_boundary_truncated_quantity / value_requested_quantity"
            ),
            "development_validation_sources_required": [
                "background", "value",
            ],
            "per_seed_ratios_role": "diagnostic_only",
            "zero_denominator_rule": (
                "passes only when the corresponding numerator is zero"
            ),
        },
        "cluster_training_finite_boundary_adequacy": {
            "scope": (
                "value_agent_source_only; per_symbol_date_and_cluster_"
                "candidate_pooled_across_predeclared_stage_seeds"
            ),
            "reason": (
                "the background and local-flow model is frozen by block one; "
                "block two must not divide value-agent boundary events by "
                "background denominators or reclassify a three-symbol cluster "
                "as a market-wide aggregate"
            ),
            "maximum_symbol_date_event_ratio": (
                CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_EVENT_RATIO
            ),
            "maximum_symbol_date_quantity_ratio": (
                CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_QUANTITY_RATIO
            ),
            "maximum_cluster_candidate_event_ratio": (
                CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_EVENT_RATIO
            ),
            "maximum_cluster_candidate_quantity_ratio": (
                CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_QUANTITY_RATIO
            ),
            "per_seed_ratios_role": "diagnostic_only",
            "background_boundary_role": "diagnostic_frozen_block_one",
            "development_validation_requires_background_and_value_gates": True,
        },
        "stochastic_stream_identity": "stable_hash_of_symbol_not_subset_book_id",
    }


def certification_profile_sha256() -> str:
    encoded = json.dumps(
        certification_profile(), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_profile_with_acceptance_thresholds(
    gate_profile: Mapping[str, object],
    *,
    maximum_robust_score: float,
    maximum_metric_score: float,
    maximum_two_sided_shortfall: float,
) -> dict[str, object]:
    """Record the acceptance thresholds actually used by one invocation.

    The immutable profile is the comparison target, but a diagnostic run may
    deliberately use different thresholds.  Copying the immutable values into
    ``observed_runtime_profile`` would mask that drift and could make a
    noncanonical run appear eligible for certification.  The nested training-
    adequacy thresholds must be replaced as well as the top-level held-out
    thresholds because both gates consume the command-line values.
    """
    profile = dict(gate_profile)
    raw_training_gate = gate_profile.get("full_universe_training_adequacy")
    if not isinstance(raw_training_gate, Mapping):
        raise CalibrationError(
            "certification profile lacks full-universe training adequacy"
        )
    training_gate = dict(raw_training_gate)
    training_gate.update({
        "maximum_aggregate_robust_score": maximum_robust_score,
        "maximum_day_robust_score": maximum_robust_score,
        "maximum_day_metric_score": maximum_metric_score,
    })
    profile.update({
        "maximum_robust_score": maximum_robust_score,
        "maximum_metric_score": maximum_metric_score,
        "maximum_two_sided_shortfall_diagnostic": (
            maximum_two_sided_shortfall
        ),
        "full_universe_training_adequacy": training_gate,
    })
    return profile


def empirical_input_bundle_sha256(config_path: pathlib.Path) -> str:
    """Hash every external empirical file consumed by the simulator.

    Hashing only the universe CSV is insufficient because its rows point to
    mutable Hawkes-rate and empirical-mark files.  The digest is based on
    logical book/symbol/role identifiers plus bytes, so it is independent of
    the absolute directory prefix while still detecting any content change.
    """
    fields, rows = csv_table(config_path)
    required = {"book_id", "symbol", "data_dir", "hawkes_rates_file"}
    missing = sorted(required.difference(fields))
    if missing:
        raise CalibrationError(
            f"cannot hash empirical bundle for {config_path}; missing {missing}"
        )
    digest = hashlib.sha256()

    def add_file(identity: str, path: pathlib.Path) -> None:
        if not path.is_file():
            raise CalibrationError(f"empirical input is not a regular file: {path}")
        name = identity.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)

    for row in sorted(rows, key=lambda item: int(item["book_id"])):
        book_id = int(row["book_id"])
        symbol = normalise_symbol(row["symbol"], label=f"{config_path}:symbol")
        data_dir = pathlib.Path(row["data_dir"]).expanduser()
        rates = pathlib.Path(row["hawkes_rates_file"]).expanduser()
        if not data_dir.is_absolute() or not rates.is_absolute():
            raise CalibrationError(
                f"empirical input paths must be absolute for {symbol} in {config_path}"
            )
        data_dir = data_dir.resolve()
        rates = rates.resolve()
        add_file(f"{book_id}:{symbol}:hawkes_rates", rates)
        for filename in SIMULATOR_EMPIRICAL_INPUT_FILENAMES:
            add_file(f"{book_id}:{symbol}:mark:{filename}", data_dir / filename)
        manifests = sorted(data_dir.glob("itch_manifest_*.json"))
        if len(manifests) != 1:
            raise CalibrationError(
                f"{data_dir} needs exactly one ITCH manifest for provenance; "
                f"found {len(manifests)}"
            )
        add_file(f"{book_id}:{symbol}:manifest", manifests[0])
    return digest.hexdigest()


def simulator_source_semantics_sha256(project_root: pathlib.Path) -> str:
    """Hash the deterministic C++ simulator source semantics, not build paths."""
    root = project_root.resolve()
    files = [root / "CMakeLists.txt"]
    files.extend(sorted((root / "include").rglob("*.hpp")))
    files.extend(sorted((root / "src").rglob("*.cpp")))
    if not files or any(not path.is_file() for path in files):
        raise CalibrationError(f"incomplete simulator source tree below {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def workflow_source_semantics_sha256(project_root: pathlib.Path) -> str:
    """Hash every Python/shell file that defines the certified workflow."""
    root = project_root.resolve()
    files = [root / relative for relative in WORKFLOW_SEMANTICS_FILES]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise CalibrationError(
            "incomplete workflow source tree; missing "
            + ", ".join(str(path) for path in missing)
        )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def validate_pooling_producer_workflow_source(
    payload: Mapping[str, object],
    *,
    producer_project_root: pathlib.Path,
    consumer_project_root: pathlib.Path,
) -> dict[str, object]:
    """Bind a reused pool to its producer tree without conflating revisions.

    A calibration-only protocol revision legitimately changes the current
    workflow hash even when the immutable empirical pool is reused byte for
    byte.  The pool's recorded hash must therefore be checked against the
    exact source tree that produced it.  The current consumer tree remains
    independently bound by ``validate_build_provenance``.
    """
    producer_root = producer_project_root.expanduser().resolve()
    consumer_root = consumer_project_root.expanduser().resolve()
    if not producer_root.is_dir():
        raise CalibrationError(
            "pooling producer project root is not a directory: "
            f"{producer_root}"
        )
    recorded = payload.get("workflow_source_semantics_sha256")
    if (not isinstance(recorded, str) or len(recorded) != 64
            or any(character not in "0123456789abcdefABCDEF"
                   for character in recorded)):
        raise CalibrationError(
            "pooling provenance has no valid producer workflow SHA-256"
        )
    observed = workflow_source_semantics_sha256(producer_root)
    if observed.lower() != recorded.lower():
        raise CalibrationError(
            "pooling provenance workflow hash does not match the declared "
            f"producer source tree: recorded={recorded.lower()} "
            f"observed={observed.lower()} root={producer_root}"
        )
    consumer = workflow_source_semantics_sha256(consumer_root)
    return {
        "schema_version": 1,
        "status": "producer_source_verified",
        "producer_project_root": str(producer_root),
        "recorded_producer_workflow_source_semantics_sha256": recorded.lower(),
        "observed_producer_workflow_source_semantics_sha256": observed.lower(),
        "consumer_project_root": str(consumer_root),
        "consumer_workflow_source_semantics_sha256": consumer,
        "producer_and_consumer_workflow_semantics_identical": (
            observed == consumer
        ),
    }


def json_object(path: pathlib.Path, *, label: str) -> dict[str, object]:
    if not path.is_file():
        raise CalibrationError(f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(f"cannot parse {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CalibrationError(f"{label} is not a JSON object: {path}")
    return value


def validate_build_provenance(
    path: pathlib.Path,
    *,
    binary: pathlib.Path,
    project_root: pathlib.Path,
) -> dict[str, object]:
    """Bind the executable used for fitting to the source tree that built it."""
    payload = json_object(path, label="calibration build provenance")
    if payload.get("schema_version") != 1:
        raise CalibrationError("unsupported calibration build-provenance schema")
    if payload.get("artifact_role") != "calibration_build_provenance":
        raise CalibrationError("calibration build provenance has the wrong role")
    try:
        recorded_binary = pathlib.Path(str(payload["binary"])).expanduser().resolve()
    except KeyError as error:
        raise CalibrationError("calibration build provenance lacks binary") from error
    if recorded_binary != binary:
        raise CalibrationError(
            "calibration build provenance names a different executable"
        )
    binary_hash = sha256_file(binary)
    if payload.get("binary_sha256") != binary_hash:
        raise CalibrationError(
            "calibration executable SHA-256 disagrees with build provenance"
        )
    source_hash = simulator_source_semantics_sha256(project_root)
    if payload.get("simulator_source_semantics_sha256") != source_hash:
        raise CalibrationError(
            "calibration build provenance does not match current C++ source semantics"
        )
    workflow_hash = workflow_source_semantics_sha256(project_root)
    if payload.get("workflow_source_semantics_sha256") != workflow_hash:
        raise CalibrationError(
            "calibration build provenance does not match current workflow semantics"
        )
    if payload.get("cmake_build_type") != "Release":
        raise CalibrationError("certified calibration requires a Release build")
    contract = payload.get("deterministic_build_contract")
    expected_contract_path = (
        project_root / "scripts" / "seagull_deterministic_build.sh"
    ).resolve()
    if not isinstance(contract, Mapping):
        raise CalibrationError(
            "calibration build provenance lacks deterministic build contract"
        )
    expected_contract_values = {
        "version": "seagull_release_mpi_v1",
        "path": str(expected_contract_path),
        "sha256": sha256_file(expected_contract_path),
        "source_date_epoch": "1577836800",
        "cmake_build_type": "Release",
        "lob_require_mpi": True,
        "lob_build_tests": True,
        "interprocedural_optimization": False,
    }
    for key, expected in expected_contract_values.items():
        if contract.get(key) != expected:
            raise CalibrationError(
                f"calibration deterministic build contract disagrees for {key}"
            )
    for key in ("compiler_path", "ninja_path", "mpi_lib_dir"):
        value = contract.get(key)
        if not isinstance(value, str) or not pathlib.Path(value).is_absolute():
            raise CalibrationError(
                f"calibration deterministic build contract has invalid {key}"
            )
    return {
        **payload,
        "path": str(path),
        "sha256": sha256_file(path),
    }


def validate_cluster_manifest(
    path: pathlib.Path,
    *,
    assignments_path: pathlib.Path,
    validation_path: pathlib.Path,
    universe_config_path: pathlib.Path,
) -> dict[str, object]:
    """Verify clustering method, inputs and both materialised CSV artifacts."""
    payload = json_object(path, label="cluster manifest")
    if payload.get("schema_version") != 1:
        raise CalibrationError("unsupported cluster-manifest schema")
    inputs = payload.get("inputs")
    clustering = payload.get("clustering")
    features = payload.get("features")
    artifacts = payload.get("artifacts")
    counts = payload.get("counts")
    if not all(isinstance(value, Mapping) for value in (
            inputs, clustering, features, artifacts, counts)):
        raise CalibrationError("cluster manifest is missing required objects")
    assert isinstance(inputs, Mapping)
    assert isinstance(clustering, Mapping)
    assert isinstance(features, Mapping)
    assert isinstance(artifacts, Mapping)
    assert isinstance(counts, Mapping)
    if inputs.get("universe_config_sha256") != sha256_file(universe_config_path):
        raise CalibrationError("cluster manifest was built from a different universe")
    if clustering.get("algorithm") != (
            "deterministic_farthest_first_lloyd_kmeans_"
            "with_minimum_size_repair"):
        raise CalibrationError("cluster manifest uses an unsupported algorithm")
    if clustering.get("cluster_count") != CERTIFICATION_CLUSTER_COUNT:
        raise CalibrationError("cluster manifest does not contain ten clusters")
    if clustering.get("seed") != 20200130:
        raise CalibrationError("cluster manifest does not use seed 20200130")
    if clustering.get("minimum_cluster_size") != 6:
        raise CalibrationError(
            "cluster manifest does not guarantee three disjoint training and "
            "three development-validation symbols per cluster"
        )
    if (clustering.get("requested_validation_per_cluster")
            != CERTIFICATION_VALIDATION_SYMBOLS_PER_CLUSTER):
        raise CalibrationError(
            "cluster manifest does not request three validation symbols per cluster"
        )
    expected_raw = [
        "event_rate_per_second", "mean_spread_ticks", "mean_top_depth",
        "return_variance", "opening_mid_price_ticks",
    ]
    if features.get("raw_feature_columns") != expected_raw:
        raise CalibrationError("cluster manifest uses an unsupported feature vector")
    if counts.get("clusters") != CERTIFICATION_CLUSTER_COUNT:
        raise CalibrationError("cluster manifest count disagrees with its method")
    for key, expected_path in (
        ("cluster_assignments_csv", assignments_path),
        ("validation_sample_csv", validation_path),
    ):
        record = artifacts.get(key)
        if not isinstance(record, Mapping):
            raise CalibrationError(f"cluster manifest lacks {key}")
        recorded_path = pathlib.Path(str(record.get("path", ""))).expanduser().resolve()
        if recorded_path != expected_path:
            raise CalibrationError(f"cluster manifest {key} path disagrees")
        if record.get("sha256") != sha256_file(expected_path):
            raise CalibrationError(f"cluster manifest {key} hash disagrees")
    return {"path": str(path), "sha256": sha256_file(path), **payload}


def validate_pooling_symbol_coverage(
    *,
    common_symbol_count: object,
    symbol_records: object,
    pooled_rows: Sequence[Mapping[str, str]],
    training_days: Sequence[TrainingDay],
) -> tuple[str, ...]:
    """Require one provenance record for every pooled/training symbol.

    File hashes establish artifact identity, but do not by themselves prove
    that the JSON's per-symbol audit list is unique and complete.  This check
    closes that gap before any individual rate artifact is trusted.
    """
    if (isinstance(common_symbol_count, bool)
            or not isinstance(common_symbol_count, int)
            or common_symbol_count <= 0):
        raise CalibrationError(
            "pooling provenance has an invalid common-symbol count"
        )

    pooled_symbols = tuple(
        normalise_symbol(
            row.get("symbol", ""), label="pooled runtime configuration",
        )
        for row in pooled_rows
    )
    pooled_symbol_set = set(pooled_symbols)
    if len(pooled_symbol_set) != len(pooled_symbols):
        raise CalibrationError(
            "pooled runtime configuration contains duplicate symbols"
        )
    if common_symbol_count != len(pooled_symbols):
        raise CalibrationError(
            "pooling provenance common-symbol count differs from the pooled "
            "runtime configuration"
        )

    for day in training_days:
        training_symbols = tuple(
            normalise_symbol(
                row.get("symbol", ""),
                label=f"training runtime configuration {day.date}",
            )
            for row in day.rows
        )
        training_symbol_set = set(training_symbols)
        if len(training_symbol_set) != len(training_symbols):
            raise CalibrationError(
                f"training config {day.date} contains duplicate symbols"
            )
        if training_symbol_set != pooled_symbol_set:
            missing = sorted(pooled_symbol_set - training_symbol_set)
            extra = sorted(training_symbol_set - pooled_symbol_set)
            raise CalibrationError(
                f"training config {day.date} symbols differ from the pooled "
                f"runtime configuration (missing={missing}, extra={extra})"
            )

    if not isinstance(symbol_records, list):
        raise CalibrationError(
            "pooling provenance lacks pooled-symbol compatibility records"
        )
    provenance_symbols: list[str] = []
    seen_provenance_symbols: set[str] = set()
    for record in symbol_records:
        if not isinstance(record, Mapping):
            raise CalibrationError("malformed pooled-symbol provenance record")
        symbol = normalise_symbol(
            record.get("symbol", ""), label="pooled-symbol provenance",
        )
        if symbol in seen_provenance_symbols:
            raise CalibrationError(
                f"duplicate pooled-symbol provenance record for {symbol}"
            )
        seen_provenance_symbols.add(symbol)
        provenance_symbols.append(symbol)

    if len(provenance_symbols) != common_symbol_count:
        raise CalibrationError(
            "pooling provenance lacks pooled-symbol compatibility records"
        )
    if seen_provenance_symbols != pooled_symbol_set:
        missing = sorted(pooled_symbol_set - seen_provenance_symbols)
        extra = sorted(seen_provenance_symbols - pooled_symbol_set)
        raise CalibrationError(
            "pooling provenance symbol records differ from the pooled runtime "
            f"configuration (missing={missing}, extra={extra})"
        )
    return tuple(provenance_symbols)


def validate_pooling_provenance(
    path: pathlib.Path,
    *,
    training_days: Sequence[TrainingDay],
    pooled_config_path: pathlib.Path,
    heldout_config_path: pathlib.Path,
    heldout_target_root: pathlib.Path,
    producer_project_root: pathlib.Path,
    project_root: pathlib.Path,
) -> dict[str, object]:
    """Verify the exact direct-input pooling protocol used for certification."""
    payload = json_object(path, label="pooling provenance")
    if payload.get("schema_version") != POOLING_PROVENANCE_SCHEMA_VERSION:
        raise CalibrationError(
            "unsupported pooling-provenance schema; regenerate the five-day "
            "pool so queue-reactive depth targets are frozen without held-out "
            "leakage"
        )
    if payload.get("method") != (
            "multi_day_direct_input_pooling_with_day_level_behavioural_wmm"):
        raise CalibrationError("pooling provenance uses an unsupported method")
    if payload.get("training_dates") != list(CERTIFICATION_TRAINING_DATES):
        raise CalibrationError("pooling provenance has noncanonical training dates")
    if payload.get("heldout_date") != CERTIFICATION_VALIDATION_DATE:
        raise CalibrationError("pooling provenance has a noncanonical validation date")
    producer_source_verification = validate_pooling_producer_workflow_source(
        payload,
        producer_project_root=producer_project_root,
        consumer_project_root=project_root,
    )
    pooling = payload.get("pooling")
    opening = payload.get("opening_price_grid_eligibility")
    parameters = payload.get("pooling_parameters")
    config_schema = payload.get("configuration_schema")
    if not isinstance(pooling, Mapping) or not isinstance(opening, Mapping):
        raise CalibrationError("pooling provenance lacks method settings")
    if not isinstance(parameters, Mapping):
        raise CalibrationError("pooling provenance lacks immutable pooling parameters")
    expected_schema = {
        "schema_version": RUNTIME_CONFIG_SCHEMA_VERSION,
        "source_fields": list(BASE_CONFIG_FIELDS),
        "runtime_fields": list(RUNTIME_CONFIG_FIELDS),
        "runtime_fields_sha256": configuration_schema_sha256(
            RUNTIME_CONFIG_FIELDS
        ),
        "pooled_homeostatic_fields": list(POOLED_HOMEOSTATIC_FIELDS),
        "latent_value_fields": list(LATENT_VALUE_FIELDS),
        "frozen_training_derived_fields": list(
            FROZEN_TRAINING_DERIVED_FIELDS
        ),
        "queue_reactive_target_fields": list(QUEUE_REACTIVE_TARGET_FIELDS),
        "positive_queue_reactive_targets_required": True,
        "same_pooled_targets_in_all_runtime_sessions": True,
        "heldout_target_files_used": False,
    }
    if not isinstance(config_schema, Mapping) or dict(config_schema) != expected_schema:
        raise CalibrationError(
            "pooling provenance has an unsupported runtime configuration schema"
        )
    if pooling.get("heldout_targets_used_for_runtime_configuration") is not False:
        raise CalibrationError(
            "pooling provenance does not prove the held-out target/runtime barrier"
        )
    if payload.get("quote_improvement_runtime_approximation") != (
            QUOTE_IMPROVEMENT_RUNTIME_APPROXIMATION):
        raise CalibrationError(
            "pooling provenance lacks the fail-closed quote-improvement "
            "compatibility preflight"
        )
    hawkes = pooling.get("hawkes")
    if not isinstance(hawkes, Mapping) or dict(hawkes) != {
        "activity_scale": 0.3,
        "kernel_beta": 10.0,
        "balance_directional_volume": True,
        "balance_best_depth": True,
        "balance_strength": 1.0,
        **HAWKES_EXCITATION_SETTINGS,
    }:
        raise CalibrationError("pooling provenance has noncanonical Hawkes settings")
    if opening.get("simulator_tick_size_price_units") != 100:
        raise CalibrationError("pooling provenance has a noncanonical tick size")
    if opening.get("minimum_opening_bid_price_units") != 10000:
        raise CalibrationError("pooling provenance has a noncanonical price screen")
    expected_parameters = {
        "minimum_common_symbols": 20,
        "quote_quantity_fraction": 0.5,
        "minimum_quote_quantity": 10,
        "maximum_quote_quantity": 1000,
        "pool_label": "five_2019_sessions",
    }
    if dict(parameters) != expected_parameters:
        raise CalibrationError("pooling provenance has noncanonical pooling parameters")
    pooled = payload.get("pooled_configuration")
    heldout = payload.get("heldout")
    records = payload.get("training_days")
    if (not isinstance(pooled, Mapping) or not isinstance(heldout, Mapping)
            or not isinstance(records, list)):
        raise CalibrationError("pooling provenance lacks input/output records")
    if pathlib.Path(str(pooled.get("path", ""))).expanduser().resolve() != pooled_config_path:
        raise CalibrationError("pooling provenance names a different pooled universe")
    if pooled.get("sha256") != sha256_file(pooled_config_path):
        raise CalibrationError("pooled-universe hash disagrees with pooling provenance")
    if pathlib.Path(str(heldout.get("common_config", ""))).expanduser().resolve() \
            != heldout_config_path:
        raise CalibrationError("pooling provenance names a different validation config")
    if heldout.get("common_config_sha256") != sha256_file(heldout_config_path):
        raise CalibrationError("validation-config hash disagrees with pooling provenance")
    if pathlib.Path(str(heldout.get("target_root", ""))).expanduser().resolve() \
            != heldout_target_root:
        raise CalibrationError("pooling provenance names a different validation target root")
    by_date = {
        str(record.get("date")): record for record in records
        if isinstance(record, Mapping)
    }
    if set(by_date) != set(CERTIFICATION_TRAINING_DATES):
        raise CalibrationError("pooling provenance has malformed training-day records")
    for day in training_days:
        record = by_date[day.date]
        if pathlib.Path(str(record.get("common_config", ""))).expanduser().resolve() \
                != day.universe_config:
            raise CalibrationError(
                f"pooling provenance names a different config for {day.date}"
            )
        if record.get("common_config_sha256") != day.universe_config_sha256:
            raise CalibrationError(
                f"training config hash disagrees with pooling provenance for {day.date}"
            )
        if pathlib.Path(str(record.get("target_root", ""))).expanduser().resolve() \
                != day.target_root:
            raise CalibrationError(
                f"pooling provenance names a different target root for {day.date}"
            )
    common_symbol_count = payload.get("common_symbol_count")
    symbol_records = payload.get("symbols")
    _, pooled_rows = load_universe_config(pooled_config_path)
    validate_pooling_symbol_coverage(
        common_symbol_count=common_symbol_count,
        symbol_records=symbol_records,
        pooled_rows=pooled_rows,
        training_days=training_days,
    )
    try:
        source_sessions: dict[str, tuple[str, ...]] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise CalibrationError(
                    "pooling source-training record is malformed"
                )
            source_date = str(record.get("date", ""))
            source_path = pathlib.Path(
                str(record.get("source_config", ""))
            ).expanduser().resolve()
            if (not source_path.is_file()
                    or record.get("source_config_sha256")
                        != sha256_file(source_path)):
                raise CalibrationError(
                    f"pooling source config changed for {source_date}"
                )
            source_sessions[source_date] = cohort.symbols_from_csv(
                source_path,
                label=f"pooling source universe {source_date}",
            )
        heldout_source_path = pathlib.Path(
            str(heldout.get("source_config", ""))
        ).expanduser().resolve()
        if (not heldout_source_path.is_file()
                or heldout.get("source_config_sha256")
                    != sha256_file(heldout_source_path)):
            raise CalibrationError("pooling held-out source config changed")
        source_sessions[CERTIFICATION_VALIDATION_DATE] = (
            cohort.symbols_from_csv(
                heldout_source_path,
                label=(
                    "pooling source universe "
                    f"{CERTIFICATION_VALIDATION_DATE}"
                ),
            )
        )
        raw_exclusions = opening.get("excluded_symbols")
        if not isinstance(raw_exclusions, list) or any(
                not isinstance(entry, Mapping) for entry in raw_exclusions):
            raise CalibrationError(
                "pooling provenance has malformed fixed-grid exclusions"
            )
        expected_input_selection = (
            cohort.certification_pool_input_selection(
                source_sessions=source_sessions,
                excluded_symbols=(
                    entry.get("symbol", "") for entry in raw_exclusions
                    if isinstance(entry, Mapping)
                ),
                final_symbols=(row["symbol"] for row in pooled_rows),
                project_root=project_root,
            )
        )
        cohort.require_pool_input_selection_record(
            payload.get("certification_input_selection"),
            expected=expected_input_selection,
            label="pooling provenance certification_input_selection",
        )
        expected_cohort_artifacts = {
            "pooled_training_universe": cohort.validate_csv(
                pooled_config_path,
                label="pooled training universe",
                project_root=project_root,
            ),
            "heldout_common": cohort.validate_csv(
                heldout_config_path,
                label="frozen held-out opening universe",
                project_root=project_root,
            ),
            "training_days": {
                day.date: cohort.validate_symbols(
                    (row["symbol"] for row in day.rows),
                    label=f"training universe {day.date}",
                    project_root=project_root,
                )
                for day in training_days
            },
        }
        expected_cohort_identity = {
            **cohort.validate_csv(
                pooled_config_path,
                label="pooled training universe",
                project_root=project_root,
            ),
            "original_intersection_symbol_count": 1_509,
            "fixed_price_grid_excluded_symbol_count": 29,
            "artifact_checks": expected_cohort_artifacts,
        }
    except cohort.CohortIdentityError as error:
        raise CalibrationError(str(error)) from error
    if payload.get("certification_cohort_required") is not True:
        raise CalibrationError(
            "pooling provenance did not require the immutable certification cohort"
        )
    if payload.get("cohort_identity") != expected_cohort_identity:
        raise CalibrationError(
            "pooling provenance cohort identity disagrees with the exact "
            "pooled, daily-training, or held-out configuration cohort"
        )
    assert isinstance(common_symbol_count, int)
    assert isinstance(symbol_records, list)
    heldout_compatibility = heldout.get("quote_improvement_compatibility")
    if (heldout.get("heldout_role")
                != "opening_state_and_validation_targets_only"
            or heldout.get("opening_fields_copied_from_heldout")
                != list(HELDOUT_OPENING_FIELDS)
            or heldout.get("background_inputs_inherited_from_pooled") is not True
            or not isinstance(heldout_compatibility, Mapping)
            or heldout_compatibility.get("status")
                != "frozen_from_pooled_training"
            or heldout_compatibility.get("symbol_count") != common_symbol_count
            or heldout_compatibility.get("heldout_mark_inputs_instantiated")
                is not False
            or heldout_compatibility.get("runtime_probability_source")
                != "pooled_training_universe.csv"):
        raise CalibrationError(
            "pooling provenance does not certify that held-out runtime "
            "backgrounds inherit the checked pooled template"
        )
    training_rows_by_date = {
        day.date: {row["symbol"]: row for row in day.rows}
        for day in training_days
    }

    expected_hawkes = {
        "activity_scale": 0.3,
        "kernel_beta": 10.0,
        "balance_directional_volume": True,
        "balance_best_depth": True,
        "balance_strength": 1.0,
        **HAWKES_EXCITATION_SETTINGS,
    }

    def verified_rate_artifact(
        value: object, *, label: str,
    ) -> tuple[pathlib.Path, str]:
        if not isinstance(value, Mapping):
            raise CalibrationError(f"{label} is not an artifact record")
        artifact_path = pathlib.Path(
            str(value.get("path", ""))
        ).expanduser().resolve()
        recorded_hash = value.get("sha256")
        if (not artifact_path.is_file()
                or not isinstance(recorded_hash, str)
                or recorded_hash != sha256_file(artifact_path)):
            raise CalibrationError(f"{label} changed after rate derivation")
        return artifact_path, recorded_hash

    def verify_rate_derivation(
        value: object,
        *,
        label: str,
        expected_manifest: pathlib.Path | None,
        expected_manifest_sha256: object,
        expected_generated: pathlib.Path,
        expected_generated_sha256: object,
    ) -> None:
        if not isinstance(value, Mapping):
            raise CalibrationError(f"{label} lacks rate_derivation")

        def nonnegative(field: str) -> float:
            number = finite_float(
                value.get(field), label=f"{label}.rate_derivation.{field}",
            )
            if number < 0.0:
                raise CalibrationError(
                    f"{label}.rate_derivation.{field} must be nonnegative"
                )
            return number

        manifest_duration = value.get("manifest_duration_seconds")
        if (value.get("schema_version") != 1
                or value.get("status") != "passed"
                or value.get("event_types_checked")
                    != len(BACKGROUND_EVENT_NAMES)
                or isinstance(manifest_duration, bool)
                or not isinstance(manifest_duration, int)
                or manifest_duration <= 0
                or value.get("relative_tolerance") != 1.0e-12
                or value.get("absolute_tolerance") != 1.0e-12
                or value.get(
                    "stationary_reconstruction_equals_target_per_type"
                ) is not True
                or value.get(
                    "observed_rates_equal_manifest_counts_per_duration"
                ) is not True
                or value.get(
                    "stationary_targets_equal_declared_transforms_per_type"
                ) is not True
                or value.get(
                    "reported_reconstruction_equals_configured_rate_equation_per_type"
                ) is not True
                or value.get("transform_settings") != expected_hawkes):
            raise CalibrationError(
                f"{label} has an incomplete stationary-rate derivation audit"
            )
        reconstruction_error = nonnegative(
            "maximum_absolute_stationary_reconstruction_error"
        )
        observed_error = nonnegative("maximum_absolute_observed_rate_error")
        target_error = nonnegative(
            "maximum_absolute_stationary_target_error"
        )
        reported_error = nonnegative(
            "maximum_absolute_reported_reconstruction_error"
        )
        if observed_error > 1.0e-12 or target_error > 1.0e-12:
            raise CalibrationError(
                f"{label} independent rate-derivation audit exceeds tolerance"
            )
        manifest_path, manifest_hash = verified_rate_artifact(
            value.get("manifest"), label=f"{label}.rate_derivation.manifest",
        )
        generated_path, generated_hash = verified_rate_artifact(
            value.get("generated_hawkes_rates"),
            label=f"{label}.rate_derivation.generated_hawkes_rates",
        )
        if (expected_manifest is not None and manifest_path != expected_manifest):
            raise CalibrationError(
                f"{label} rate_derivation names a different manifest"
            )
        if (expected_manifest_sha256 is not None
                and manifest_hash != expected_manifest_sha256):
            raise CalibrationError(
                f"{label} rate_derivation has a conflicting manifest hash"
            )
        if (generated_path != expected_generated
                or generated_hash != expected_generated_sha256):
            raise CalibrationError(
                f"{label} generated rate path/hash disagrees with its audit"
            )

        try:
            with generated_path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
        except OSError as error:
            raise CalibrationError(
                f"cannot read {label} generated Hawkes rates: {error}"
            ) from error
        if [row.get("event_type") for row in rows] != list(
                BACKGROUND_EVENT_NAMES):
            raise CalibrationError(
                f"{label} generated Hawkes rates have the wrong event order"
            )
        target_rates: list[float] = []
        for row in rows:
            target = finite_float(
                row.get("stationary_target_rate"),
                label=(
                    f"{label}.{row.get('event_type')}.stationary_target_rate"
                ),
            )
            if target < 0.0:
                raise CalibrationError(
                    f"{label} has a negative stationary target rate"
                )
            target_rates.append(target)
        observed_reconstruction_error = 0.0
        observed_reported_error = 0.0
        for index, row in enumerate(rows):
            event = str(row["event_type"])
            configured_mu = finite_float(
                row.get("configured_mu"),
                label=f"{label}.{event}.configured_mu",
            )
            reconstructed = finite_float(
                row.get("stationary_reconstructed_rate"),
                label=f"{label}.{event}.stationary_reconstructed_rate",
            )
            if configured_mu < 0.0 or reconstructed < 0.0:
                raise CalibrationError(
                    f"{label} generated Hawkes rates must be nonnegative"
                )
            target = target_rates[index]
            if expected_hawkes["cross_excitation_amplitude"] != 0.0:
                raise CalibrationError(
                    "certified workflow requires zero cross excitation"
                )
            endogenous = (
                expected_hawkes["self_excitation_amplitude"] * target
                / expected_hawkes["kernel_beta"]
            )
            computed = (
                expected_hawkes["activity_scale"] * configured_mu + endogenous
            )
            observed_reconstruction_error = max(
                observed_reconstruction_error, abs(computed - target),
            )
            observed_reported_error = max(
                observed_reported_error, abs(reconstructed - computed),
            )
            if (not math.isclose(
                    reconstructed, computed, rel_tol=1.0e-12,
                    abs_tol=1.0e-12)
                    or not math.isclose(
                        computed, target, rel_tol=1.0e-12,
                        abs_tol=1.0e-12)):
                raise CalibrationError(
                    f"{label} generated Hawkes rates fail reconstruction for "
                    f"{event}"
                )
        if not math.isclose(
                reconstruction_error, observed_reconstruction_error,
                rel_tol=1.0e-12, abs_tol=1.0e-15):
            raise CalibrationError(
                f"{label} recorded reconstruction error disagrees with its CSV"
            )
        if not math.isclose(
                reported_error, observed_reported_error,
                rel_tol=1.0e-12, abs_tol=1.0e-15):
            raise CalibrationError(
                f"{label} reported-equation error disagrees with its CSV"
            )

    for record in symbol_records:
        if not isinstance(record, Mapping):
            raise CalibrationError("malformed pooled-symbol provenance record")
        symbol = normalise_symbol(
            str(record.get("symbol", "")), label="pooled-symbol provenance"
        )
        pooled_check = record.get("quote_improvement_compatibility")
        sources = record.get("sources")
        pooled_manifest_path = pathlib.Path(
            str(record.get("pooled_manifest", ""))
        ).expanduser().resolve()
        pooled_rate_path = pathlib.Path(
            str(record.get("pooled_hawkes_rates", ""))
        ).expanduser().resolve()
        pooled_rate_hash = record.get("pooled_hawkes_rates_sha256")
        if (not isinstance(pooled_check, Mapping)
                or pooled_check.get("status") != "passed"
                or pooled_check.get("probability_clamped") is not False
                or pooled_check.get("side_allocation")
                    != "proportional_to_observed_side_zero_counts"
                or not pooled_manifest_path.is_file()
                or not pooled_rate_path.is_file()
                or pooled_rate_hash != sha256_file(pooled_rate_path)
                or not isinstance(sources, list)
                or len(sources) != len(CERTIFICATION_TRAINING_DATES)
                or any(
                    not isinstance(source, Mapping)
                    or not isinstance(
                        source.get("quote_improvement_compatibility"), Mapping
                    )
                    or source["quote_improvement_compatibility"].get("status")
                        != "passed"
                    or source["quote_improvement_compatibility"].get(
                        "probability_clamped"
                    ) is not False
                    or source["quote_improvement_compatibility"].get(
                        "side_allocation"
                    ) != "proportional_to_observed_side_zero_counts"
                    for source in sources
                )):
            raise CalibrationError(
                "pooling provenance has an incomplete quote-improvement "
                "or rate-derivation compatibility record"
            )
        verify_rate_derivation(
            record.get("rate_derivation"),
            label=f"pooled symbol {symbol}",
            expected_manifest=pooled_manifest_path,
            expected_manifest_sha256=None,
            expected_generated=pooled_rate_path,
            expected_generated_sha256=pooled_rate_hash,
        )
        sources_by_date = {
            str(source.get("trading_date")): source
            for source in sources if isinstance(source, Mapping)
        }
        if set(sources_by_date) != set(CERTIFICATION_TRAINING_DATES):
            raise CalibrationError(
                f"pooling provenance has malformed daily-rate records for {symbol}"
            )
        for date, source in sources_by_date.items():
            manifest = pathlib.Path(
                str(source.get("manifest", ""))
            ).expanduser().resolve()
            generated = pathlib.Path(
                str(source.get("generated_hawkes_rates", ""))
            ).expanduser().resolve()
            source_rate = pathlib.Path(
                str(source.get("source_hawkes_rates", ""))
            ).expanduser().resolve()
            runtime_row = training_rows_by_date[date].get(symbol)
            if runtime_row is None:
                raise CalibrationError(
                    f"training config {date} lacks provenance symbol {symbol}"
                )
            configured = pathlib.Path(
                runtime_row["hawkes_rates_file"]
            ).expanduser().resolve()
            generated_hash = source.get("generated_hawkes_rates_sha256")
            manifest_hash = source.get("manifest_sha256")
            if (not manifest.is_file()
                    or manifest_hash != sha256_file(manifest)
                    or not generated.is_file()
                    or generated != configured
                    or generated_hash != sha256_file(generated)
                    or not source_rate.is_file()
                    or source.get("source_hawkes_rates_sha256")
                        != sha256_file(source_rate)):
                raise CalibrationError(
                    f"pooling provenance cannot verify daily rates for "
                    f"{symbol} {date}"
                )
            verify_rate_derivation(
                source.get("rate_derivation"),
                label=f"daily symbol {symbol} {date}",
                expected_manifest=manifest,
                expected_manifest_sha256=manifest_hash,
                expected_generated=generated,
                expected_generated_sha256=generated_hash,
            )
    result = {
        **payload,
        "path": str(path),
        "sha256": sha256_file(path),
        "producer_source_verification": producer_source_verification,
    }
    return result


def atomic_csv(path: pathlib.Path,
               fields: Sequence[str],
               rows: Iterable[Mapping[str, object]],
               *, overwrite: bool) -> None:
    """Write a CSV atomically and never silently replace a final artifact."""
    if path.exists() and not overwrite:
        raise CalibrationError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True,
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(fields), extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: pathlib.Path,
                payload: Mapping[str, Any],
                *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise CalibrationError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True,
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(json_safe(payload), output, indent=2, sort_keys=True,
                      allow_nan=False)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def json_safe(value: Any) -> Any:
    """Replace non-finite diagnostics with null before strict JSON serialization."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_universe_config(path: pathlib.Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Read and validate the exact CSV shape accepted by fragmented_mpi_lob."""
    fields, rows = csv_table(path)
    missing = sorted(set(CONFIG_REQUIRED_FIELDS).difference(fields))
    if missing:
        raise CalibrationError(
            f"universe configuration {path} is missing required columns: {', '.join(missing)}"
        )
    symbols: set[str] = set()
    identifiers: set[int] = set()
    for line_number, row in enumerate(rows, start=2):
        symbol = normalise_symbol(row["symbol"], label=f"{path}:{line_number}")
        row["symbol"] = symbol
        try:
            book_id = int(row["book_id"])
        except ValueError as error:
            raise CalibrationError(f"invalid book_id in {path}:{line_number}") from error
        if book_id < 0 or book_id in identifiers:
            raise CalibrationError(f"duplicate/negative book_id in {path}:{line_number}")
        if symbol in symbols:
            raise CalibrationError(f"duplicate symbol {symbol} in {path}")
        for field in FROZEN_TRAINING_DERIVED_FIELDS:
            target = finite_float(
                row[field], label=f"{path}:{line_number}:{field}"
            )
            if (field in LATENT_VALUE_FIELDS
                    and field != "fundamental_conditional_kurtosis"
                    and target < 0.0):
                raise CalibrationError(
                    f"{field} must be non-negative in {path}:{line_number}"
                )
            if (field == "fundamental_move_probability_per_second"
                    and target > 1.0):
                raise CalibrationError(
                    f"{field} must not exceed one in {path}:{line_number}"
                )
            if (field == "fundamental_conditional_kurtosis"
                    and target < 1.0):
                raise CalibrationError(
                    f"{field} must be at least one in {path}:{line_number}"
                )
            if field not in LATENT_VALUE_FIELDS and target <= 0.0:
                raise CalibrationError(
                    f"{field} must be positive in {path}:{line_number}"
                )
        symbols.add(symbol)
        identifiers.add(book_id)
    expected_ids = set(range(len(rows)))
    if identifiers != expected_ids:
        raise CalibrationError(
            f"book_id values in {path} must be contiguous from zero"
        )
    rows.sort(key=lambda row: int(row["book_id"]))
    return fields, rows


def validate_frozen_homeostatic_targets(
    training_days: Sequence[TrainingDay],
    pooled_rows: Sequence[Mapping[str, str]],
    heldout_rows: Sequence[Mapping[str, str]],
) -> None:
    """Require one pooled state-target vector in every runtime config.

    The five empirical training sessions still retain their own Hawkes clocks,
    marks and opening states.  The homeostatic spread/depth anchors are model
    parameters, however, so allowing one value per day would leak each day's
    full-session outcomes into its run.  Exact string equality is intentional:
    the pooler materialises the same canonical values in every CSV.
    """
    pooled_by_symbol = {row["symbol"]: row for row in pooled_rows}
    if len(pooled_by_symbol) != len(pooled_rows):
        raise CalibrationError("pooled universe has duplicate symbols")
    sessions: list[tuple[str, Sequence[Mapping[str, str]]]] = [
        (f"training day {day.date}", day.rows) for day in training_days
    ]
    sessions.append(("development-validation opening source", heldout_rows))
    for label, rows in sessions:
        by_symbol = {row["symbol"]: row for row in rows}
        if set(by_symbol) != set(pooled_by_symbol):
            raise CalibrationError(
                f"{label} symbols differ from pooled training universe"
            )
        for symbol, pooled in pooled_by_symbol.items():
            observed = by_symbol[symbol]
            for field in FROZEN_TRAINING_DERIVED_FIELDS:
                if observed.get(field) != pooled.get(field):
                    raise CalibrationError(
                        f"{label} uses a non-pooled {field} for {symbol}; "
                        "all runtime sessions must use the same five-day "
                        "training estimate"
                    )


def align_config_rows_to_symbols(rows: Sequence[Mapping[str, str]],
                                 symbols: Sequence[str],
                                 *, label: str) -> list[dict[str, str]]:
    """Return a canonical symbol order and contiguous IDs for a daily config.

    Daily ITCH extraction jobs can legitimately emit a different original
    ``book_id`` order.  The simulator's random streams and cross-book state
    are nevertheless easier to audit if every training-day run uses the same
    logical symbol-to-ID order.  This function changes no empirical input; it
    only canonicalises the transport identifier in a temporary simulator CSV.
    """
    ordered_symbols = tuple(normalise_symbol(symbol, label=f"{label}:symbols")
                            for symbol in symbols)
    if not ordered_symbols or len(set(ordered_symbols)) != len(ordered_symbols):
        raise CalibrationError(f"{label} has an invalid canonical symbol order")
    by_symbol = {row["symbol"]: row for row in rows}
    expected = set(ordered_symbols)
    if set(by_symbol) != expected:
        missing = sorted(expected.difference(by_symbol))
        extra = sorted(set(by_symbol).difference(expected))
        raise CalibrationError(
            f"{label} symbols do not match the training universe; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    aligned: list[dict[str, str]] = []
    for book_id, symbol in enumerate(ordered_symbols):
        copied = dict(by_symbol[symbol])
        copied["book_id"] = str(book_id)
        aligned.append(copied)
    return aligned


def load_training_days(args: argparse.Namespace) -> tuple[TrainingDay, ...]:
    """Load one legacy or many explicit training sessions consistently.

    ``--training-day`` inputs are sorted chronologically for reproducible
    reporting.  Each session must carry the same universe schema and symbols;
    only its session-specific direct ITCH inputs and target root may differ.
    """
    if args.training_day:
        raw_days = [tuple(entry) for entry in args.training_day]
    else:
        # Argument validation guarantees these legacy values are all present.
        raw_days = [(args.training_date,
                     args.training_universe_config,
                     args.training_target_root)]
    loaded: list[TrainingDay] = []
    canonical_fields: tuple[str, ...] | None = None
    canonical_symbols: tuple[str, ...] | None = None
    for raw_date, raw_config, raw_target_root in sorted(raw_days, key=lambda item: item[0]):
        config_path = pathlib.Path(str(raw_config)).expanduser().resolve()
        target_root = pathlib.Path(str(raw_target_root)).expanduser().resolve()
        fields, rows = load_universe_config(config_path)
        if canonical_fields is None:
            canonical_fields = fields
            canonical_symbols = tuple(row["symbol"] for row in rows)
        else:
            if fields != canonical_fields:
                raise CalibrationError(
                    "all training universe configurations must have identical headers "
                    "in the same order"
                )
            assert canonical_symbols is not None
            rows = align_config_rows_to_symbols(
                rows, canonical_symbols, label=f"training day {raw_date}",
            )
        loaded.append(TrainingDay(
            date=str(raw_date),
            universe_config=config_path,
            target_root=target_root,
            fields=fields,
            rows=tuple(dict(row) for row in rows),
            universe_config_sha256=sha256_file(config_path),
        ))
    if not loaded:
        raise CalibrationError("at least one training day is required")
    return tuple(loaded)


def load_pooled_training_config(
    args: argparse.Namespace,
    training_days: Sequence[TrainingDay],
) -> tuple[pathlib.Path, tuple[str, ...], list[dict[str, str]], str]:
    """Load the frozen direct inputs used to initialise held-out validation.

    In multi-day mode the caller must explicitly provide this configuration.
    It can point at pooled Hawkes/mark artifacts prepared outside this policy
    search, but it must have the same schema and symbol universe as every
    candidate-evaluation day.  This prevents an accidental first-day fallback
    being described as a pooled five-day model.
    """
    if len(training_days) > 1 and not args.pooled_training_universe_config:
        raise CalibrationError(
            "--pooled-training-universe-config is required with multiple training days; "
            "provide the explicitly pooled direct-input universe used for held-out validation"
        )
    chosen = (
        args.pooled_training_universe_config
        if args.pooled_training_universe_config
        else str(training_days[0].universe_config)
    )
    pooled_path = pathlib.Path(chosen).expanduser().resolve()
    fields, rows = load_universe_config(pooled_path)
    first = training_days[0]
    if fields != first.fields:
        raise CalibrationError(
            "pooled training universe configuration must have the same headers as "
            "the daily training configurations"
        )
    aligned_rows = align_config_rows_to_symbols(
        rows, tuple(row["symbol"] for row in first.rows),
        label="pooled training universe configuration",
    )
    return pooled_path, fields, aligned_rows, sha256_file(pooled_path)


def write_training_subset_configs(
    output_root: pathlib.Path,
    training_days: Sequence[TrainingDay],
    symbols: Sequence[str],
    *,
    filename: str,
    overwrite: bool,
) -> dict[str, pathlib.Path]:
    """Materialise canonical day-specific simulator configs for one subset."""
    result: dict[str, pathlib.Path] = {}
    for training_day in training_days:
        # Retain the established artifact path in legacy single-day runs.
        path = (
            output_root / filename if len(training_days) == 1
            else output_root / "training_days" / training_day.identifier / filename
        )
        write_config_csv(
            path, training_day.fields,
            subset_config_rows(training_day.rows, symbols),
            overwrite=overwrite,
        )
        result[training_day.date] = path
    return result


def training_day_provenance(training_day: TrainingDay) -> dict[str, object]:
    """Return stable, non-target-opening provenance for report artifacts."""
    return {
        "date": training_day.date,
        "universe_config": str(training_day.universe_config),
        "universe_config_sha256": training_day.universe_config_sha256,
        "target_root": str(training_day.target_root),
    }


def merge_frozen_heldout_config(
    training_fields: Sequence[str],
    training_rows: Sequence[Mapping[str, str]],
    heldout_fields: Sequence[str],
    heldout_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Copy training backgrounds and substitute only held-out opening state.

    The held-out source is intentionally checked rather than trusted.  A
    changed data directory, Hawkes file, local quote input, beta, or any future
    non-opening field is a leakage/refit and is rejected before simulation.
    """
    if tuple(training_fields) != tuple(heldout_fields):
        raise CalibrationError(
            "training and held-out universe configurations must have identical "
            "headers in the same order"
        )
    training_by_symbol = {row["symbol"]: row for row in training_rows}
    heldout_by_symbol = {row["symbol"]: row for row in heldout_rows}
    if set(training_by_symbol) != set(heldout_by_symbol):
        raise CalibrationError(
            "training and held-out universe configurations must contain exactly "
            "the same symbols"
        )
    opening = set(HELDOUT_OPENING_FIELDS)
    merged: list[dict[str, str]] = []
    for training_row in training_rows:
        symbol = training_row["symbol"]
        heldout_row = heldout_by_symbol[symbol]
        for field in training_fields:
            if field in opening:
                continue
            if training_row[field] != heldout_row[field]:
                raise CalibrationError(
                    f"held-out configuration refits {symbol} field {field}; "
                    "only opening midpoint/BBO/depth fields may change"
                )
        result = dict(training_row)
        for field in HELDOUT_OPENING_FIELDS:
            result[field] = heldout_row[field]
        merged.append(result)
    return merged


def freeze_training_backgrounds_with_heldout_openings(
    training_fields: Sequence[str],
    training_rows: Sequence[Mapping[str, str]],
    opening_source_fields: Sequence[str],
    opening_source_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Copy only a later day's opening state onto frozen training inputs.

    ``opening_source_rows`` may be a normal, fully re-extracted held-out
    universe configuration.  Its Hawkes files, data directories, quote inputs
    and other direct calibrations are deliberately ignored: accepting those
    fields would turn the validation path into a second model fit.  This
    function is consequently separate from :func:`merge_frozen_heldout_config`,
    which remains the strict audit for an already-frozen input configuration.
    """
    missing = sorted(set(HELDOUT_OPENING_FIELDS).difference(opening_source_fields))
    if missing:
        raise CalibrationError(
            "held-out opening source is missing fields: " + ", ".join(missing)
        )
    training_by_symbol = {row["symbol"]: row for row in training_rows}
    source_by_symbol = {row["symbol"]: row for row in opening_source_rows}
    if set(training_by_symbol) != set(source_by_symbol):
        raise CalibrationError(
            "training and held-out opening-source configurations must contain "
            "exactly the same symbols"
        )
    result: list[dict[str, str]] = []
    for training_row in training_rows:
        source_row = source_by_symbol[training_row["symbol"]]
        merged = dict(training_row)
        for field in HELDOUT_OPENING_FIELDS:
            merged[field] = source_row[field]
        result.append(merged)
    return result


def subset_config_rows(rows: Sequence[Mapping[str, str]],
                       symbols: Iterable[str]) -> list[dict[str, str]]:
    """Select symbols in original universe order and reset C++-required IDs."""
    selected_symbols = {normalise_symbol(symbol, label="subset") for symbol in symbols}
    if not selected_symbols:
        raise CalibrationError("cannot make a simulator configuration with zero symbols")
    available = {row["symbol"] for row in rows}
    unknown = sorted(selected_symbols.difference(available))
    if unknown:
        raise CalibrationError(f"subset includes symbols absent from configuration: {unknown}")
    result: list[dict[str, str]] = []
    for row in rows:
        if row["symbol"] not in selected_symbols:
            continue
        copy = dict(row)
        copy["book_id"] = str(len(result))
        result.append(copy)
    if len(result) != len(selected_symbols):
        raise CalibrationError("subset selection unexpectedly lost a symbol")
    return result


def write_config_csv(path: pathlib.Path,
                     fields: Sequence[str],
                     rows: Sequence[Mapping[str, str]],
                     *, overwrite: bool) -> None:
    atomic_csv(path, fields, rows, overwrite=overwrite)


def load_cluster_layout(assignments_path: pathlib.Path,
                        validation_path: pathlib.Path,
                        symbols: Iterable[str]) -> ClusterLayout:
    """Validate the precomputed ten-cluster layout against the full universe."""
    assignment_fields, assignment_rows = csv_table(assignments_path)
    required_assignment = {"symbol", "cluster_id", "is_representative"}
    missing_assignment = sorted(required_assignment.difference(assignment_fields))
    if missing_assignment:
        raise CalibrationError(
            f"cluster assignments missing required columns: {', '.join(missing_assignment)}"
        )
    expected_symbols = {normalise_symbol(symbol, label="universe") for symbol in symbols}
    membership: dict[str, int] = {}
    marked_representatives: dict[int, list[str]] = {}
    member_order: dict[int, list[tuple[float, str]]] = {}
    for line_number, row in enumerate(assignment_rows, start=2):
        symbol = normalise_symbol(row["symbol"], label=f"{assignments_path}:{line_number}")
        cluster_id = parse_cluster_id(
            row["cluster_id"], label=f"{assignments_path}:{line_number}"
        )
        if symbol in membership:
            raise CalibrationError(f"duplicate cluster assignment for {symbol}")
        membership[symbol] = cluster_id
        try:
            distance = float(row.get("distance_to_centroid", "nan"))
        except ValueError:
            distance = math.nan
        if not math.isfinite(distance):
            distance = float(line_number)
        member_order.setdefault(cluster_id, []).append((distance, symbol))
        if parse_bool(row["is_representative"],
                      label=f"{assignments_path}:{line_number}:is_representative"):
            marked_representatives.setdefault(cluster_id, []).append(symbol)
    if set(membership) != expected_symbols:
        missing = sorted(expected_symbols.difference(membership))
        extra = sorted(set(membership).difference(expected_symbols))
        raise CalibrationError(
            "cluster assignments must cover the universe exactly; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    cluster_ids = set(membership.values())
    if set(marked_representatives) != cluster_ids:
        missing = sorted(cluster_ids.difference(marked_representatives))
        raise CalibrationError(f"clusters without a representative: {missing}")
    for cluster_id, selected in marked_representatives.items():
        if len(selected) != 1:
            raise CalibrationError(
                f"cluster {cluster_id} needs exactly one representative; found {selected}"
            )

    validation_fields, validation_rows = csv_table(validation_path)
    required_validation = {"symbol", "cluster_id"}
    missing_validation = sorted(required_validation.difference(validation_fields))
    if missing_validation:
        raise CalibrationError(
            f"validation sample missing required columns: {', '.join(missing_validation)}"
        )
    validation_by_cluster: dict[int, list[str]] = {cluster: [] for cluster in cluster_ids}
    validation_seen: set[str] = set()
    for line_number, row in enumerate(validation_rows, start=2):
        symbol = normalise_symbol(row["symbol"], label=f"{validation_path}:{line_number}")
        cluster_id = parse_cluster_id(
            row["cluster_id"], label=f"{validation_path}:{line_number}"
        )
        if symbol not in membership:
            raise CalibrationError(f"validation symbol {symbol} is not in assignments")
        if membership[symbol] != cluster_id:
            raise CalibrationError(
                f"validation symbol {symbol} has cluster {cluster_id}, but assignments "
                f"place it in {membership[symbol]}"
            )
        if symbol in validation_seen:
            raise CalibrationError(f"duplicate validation symbol: {symbol}")
        if symbol in marked_representatives[cluster_id]:
            raise CalibrationError(
                f"validation symbol {symbol} is also cluster {cluster_id}'s representative"
            )
        validation_by_cluster[cluster_id].append(symbol)
        validation_seen.add(symbol)
    absent = sorted(cluster for cluster, selected in validation_by_cluster.items() if not selected)
    if absent:
        raise CalibrationError(
            "stratified validation requires at least one non-representative symbol "
            f"from every cluster; missing clusters: {absent}"
        )
    normalized_representatives: dict[int, tuple[str, ...]] = {}
    for cluster_id in sorted(cluster_ids):
        validation_set = set(validation_by_cluster[cluster_id])
        ordered_candidates = [
            symbol for _, symbol in sorted(member_order[cluster_id])
            if symbol not in validation_set
        ]
        marked = marked_representatives[cluster_id][0]
        ordered_candidates = [marked] + [
            symbol for symbol in ordered_candidates if symbol != marked
        ]
        selected = tuple(
            ordered_candidates[:CERTIFICATION_TRAINING_REPRESENTATIVES_PER_CLUSTER]
        )
        if not selected:
            raise CalibrationError(
                f"cluster {cluster_id} has no training representative after the "
                "validation sample is held aside"
            )
        normalized_representatives[cluster_id] = selected
    return ClusterLayout(
        by_symbol=membership,
        representatives={key: tuple(value) for key, value in normalized_representatives.items()},
        validation_symbols={
            key: tuple(sorted(value)) for key, value in validation_by_cluster.items()
        },
    )


def candidate_grid(thresholds: Sequence[float],
                   depth_participations: Sequence[float]) -> list[Candidate]:
    """Make a deterministic policy grid including the no-value-agent baseline."""
    candidates = [Candidate(False, 0.0, 0.25, "disabled_baseline")]
    seen: set[tuple[float, float]] = set()
    for threshold, participation in itertools.product(
            sorted(thresholds), sorted(depth_participations)):
        if not math.isfinite(threshold) or threshold < 0.0:
            raise CalibrationError("all --thresholds values must be finite and non-negative")
        if (not math.isfinite(participation)
                or participation <= 0.0 or participation > 1.0):
            raise CalibrationError(
                "all --depth-participations values must be in (0, 1]"
            )
        key = (threshold, participation)
        if key in seen:
            continue
        seen.add(key)
        label = f"threshold_{threshold:g}_depth_participation_{participation:g}"
        candidates.append(Candidate(True, threshold, participation, label))
    if len(candidates) == 1:
        raise CalibrationError(
            "at least one enabled threshold/participation candidate is required"
        )
    return candidates


def local_flow_candidate_grid(
    hawkes_activity_scales: Sequence[float],
    local_mm_intervals_ms: Sequence[float],
    local_mm_quantity_multipliers: Sequence[float],
    local_mm_improvement_probabilities: Sequence[float],
) -> list[LocalFlowCandidate]:
    """Build nested local-MM-off baselines plus the enabled control grid.

    These controls are global because the fragmented executable has one model
    cadence for the complete market realisation.  The activity scale is not a
    behavioural search dimension: each rate file was analytically inverted at
    0.30, so changing it here would undo that direct event-rate calibration.
    """
    activities = sorted(set(hawkes_activity_scales))
    intervals = sorted(set(local_mm_intervals_ms))
    quantities = sorted(set(local_mm_quantity_multipliers))
    improvement_probabilities = sorted(set(local_mm_improvement_probabilities))
    if not activities or any(not math.isfinite(value) or value <= 0.0
                             for value in activities):
        raise CalibrationError(
            "all --hawkes-activity-scales values must be finite and positive"
        )
    if (len(activities) != 1 or not math.isclose(
            activities[0], FIXED_HAWKES_ACTIVITY_SCALE,
            rel_tol=0.0, abs_tol=1.0e-12)):
        raise CalibrationError(
            "--hawkes-activity-scales must be exactly 0.30: Hawkes immigration "
            "rates were inverted at that fixed scale, so it is not a behavioural "
            "search parameter"
        )
    if not intervals or any(not math.isfinite(value) or value <= 0.0
                            for value in intervals):
        raise CalibrationError(
            "all --local-mm-intervals-ms values must be finite and positive"
        )
    if not quantities or any(not math.isfinite(value) or value <= 0.0
                             for value in quantities):
        raise CalibrationError(
            "all --local-mm-quantity-multipliers values must be finite and positive"
        )
    if (not improvement_probabilities
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0
                   for value in improvement_probabilities)):
        raise CalibrationError(
            "all --local-mm-improvement-probabilities values must be finite "
            "probabilities in [0, 1]"
        )

    candidates: list[LocalFlowCandidate] = [
        LocalFlowCandidate(
            hawkes_activity_scale=activity,
            local_mm_enabled=False,
            # These positive placeholders are ignored by --disable-local-mm.
            local_mm_interval_ms=intervals[0],
            local_mm_quantity_multiplier=quantities[0],
            local_mm_improvement_probability=improvement_probabilities[0],
            label=f"lambda_{activity:g}_local_mm_disabled_baseline",
        )
        for activity in activities
    ]
    seen: set[tuple[bool, float, float, float, float]] = {
        (item.local_mm_enabled, item.hawkes_activity_scale,
         item.local_mm_interval_ms, item.local_mm_quantity_multiplier,
         item.local_mm_improvement_probability)
        for item in candidates
    }
    for activity, interval, quantity, improvement_probability in itertools.product(
        activities, intervals, quantities, improvement_probabilities,
    ):
        key = (True, activity, interval, quantity, improvement_probability)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(LocalFlowCandidate(
            hawkes_activity_scale=activity,
            local_mm_enabled=True,
            local_mm_interval_ms=interval,
            local_mm_quantity_multiplier=quantity,
            local_mm_improvement_probability=improvement_probability,
            label=(
                f"lambda_{activity:g}_local_interval_{interval:g}ms_"
                f"local_quantity_{quantity:g}_"
                f"local_improve_p_{improvement_probability:g}"
            ),
        ))
    if not candidates:
        raise CalibrationError("local-flow candidate grid is empty")
    return candidates


def refine_local_flow_candidates(
    leaders: Sequence[LocalFlowCandidate],
    original_grid: Sequence[LocalFlowCandidate],
    maximum_new_candidates: int,
) -> list[LocalFlowCandidate]:
    """Generate deterministic midpoint candidates around leading grid points.

    Refinement explores gaps between adjacent local-MM interval, quantity and
    price-improvement-probability levels.  The Hawkes scale axis has one fixed
    value and is never refined.
    """
    if maximum_new_candidates < 0:
        raise CalibrationError("maximum refinement candidate count cannot be negative")
    if maximum_new_candidates == 0 or not leaders:
        return []
    enabled_grid = [item for item in original_grid if item.local_mm_enabled]
    enabled_leaders = [item for item in leaders if item.local_mm_enabled]
    if not enabled_grid or not enabled_leaders:
        return []
    activity_axis = sorted({item.hawkes_activity_scale for item in enabled_grid})
    interval_axis = sorted({item.local_mm_interval_ms for item in enabled_grid})
    quantity_axis = sorted({item.local_mm_quantity_multiplier for item in enabled_grid})
    improvement_axis = sorted({
        item.local_mm_improvement_probability for item in enabled_grid
    })
    existing = {
        (item.local_mm_enabled, item.hawkes_activity_scale, item.local_mm_interval_ms,
         item.local_mm_quantity_multiplier, item.local_mm_improvement_probability)
        for item in original_grid
    }

    def local_values(value: float, axis: Sequence[float]) -> tuple[float, ...]:
        values = {value}
        lower = [candidate for candidate in axis if candidate < value]
        upper = [candidate for candidate in axis if candidate > value]
        if lower:
            values.add(math.sqrt(lower[-1] * value))
        if upper:
            values.add(math.sqrt(value * upper[0]))
        return tuple(sorted(values))

    def probability_values(value: float, axis: Sequence[float]) -> tuple[float, ...]:
        values = {value}
        lower = [candidate for candidate in axis if candidate < value]
        upper = [candidate for candidate in axis if candidate > value]
        if lower:
            values.add((lower[-1] + value) / 2.0)
        if upper:
            values.add((value + upper[0]) / 2.0)
        return tuple(sorted(values))

    result: list[LocalFlowCandidate] = []
    for leader in enabled_leaders:
        candidate_coordinates = itertools.product(
            local_values(leader.hawkes_activity_scale, activity_axis),
            local_values(leader.local_mm_interval_ms, interval_axis),
            local_values(leader.local_mm_quantity_multiplier, quantity_axis),
            probability_values(
                leader.local_mm_improvement_probability, improvement_axis,
            ),
        )
        for activity, interval, quantity, improvement_probability in candidate_coordinates:
            key = (True, activity, interval, quantity, improvement_probability)
            if key in existing:
                continue
            existing.add(key)
            result.append(LocalFlowCandidate(
                hawkes_activity_scale=activity,
                local_mm_enabled=True,
                local_mm_interval_ms=interval,
                local_mm_quantity_multiplier=quantity,
                local_mm_improvement_probability=improvement_probability,
                label=(
                    f"refined_lambda_{activity:.9g}_local_interval_{interval:.9g}ms_"
                    f"local_quantity_{quantity:.9g}_"
                    f"local_improve_p_{improvement_probability:.9g}"
                ),
            ))
            if len(result) >= maximum_new_candidates:
                return result
    return result


def shared_quote_candidate_grid(
    multipliers: Sequence[float],
) -> list[SharedQuoteCandidate]:
    """Build an off baseline plus symbol-relative shared-MM multipliers."""
    candidates = [SharedQuoteCandidate(
        enabled=False, multiplier=0.0, label="shared_mm_disabled_baseline",
    )]
    seen: set[float] = set()
    for multiplier in sorted(multipliers):
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise CalibrationError(
                "all --shared-quote-multipliers values must be finite and positive"
            )
        if multiplier in seen:
            continue
        seen.add(multiplier)
        candidates.append(SharedQuoteCandidate(
            enabled=True,
            multiplier=multiplier,
            label=f"shared_quote_relative_multiplier_{multiplier:g}",
        ))
    if len(candidates) == 1:
        raise CalibrationError("at least one enabled shared-quote multiplier is required")
    return candidates


def policy_rows_for_symbols(symbols: Iterable[str],
                            layout: ClusterLayout,
                            selected: Mapping[int, Candidate],
                            *, policy_source: str) -> list[dict[str, object]]:
    """Expand cluster policy choices into the C++ parser's aligned CSV rows."""
    result: list[dict[str, object]] = []
    for symbol in symbols:
        normalized = normalise_symbol(symbol, label="policy")
        try:
            cluster_id = layout.by_symbol[normalized]
            candidate = selected[cluster_id]
        except KeyError as error:
            raise CalibrationError(
                f"no cluster policy selected for {normalized}") from error
        result.append({
            "symbol": normalized,
            "enabled": int(candidate.enabled),
            "value_threshold_bps": format(candidate.threshold_bps, ".17g"),
            "value_depth_participation": format(
                candidate.depth_participation, ".17g"
            ),
            "cluster_id": cluster_id,
            "cluster_label": f"liquidity_{cluster_id:02d}",
            "policy_source": policy_source,
        })
    if not result:
        raise CalibrationError("cannot write a value-agent policy with no rows")
    return result


def write_policy_csv(path: pathlib.Path,
                     symbols: Iterable[str],
                     layout: ClusterLayout,
                     selected: Mapping[int, Candidate],
                     *, policy_source: str,
                     overwrite: bool) -> None:
    atomic_csv(
        path,
        POLICY_FIELDS,
        policy_rows_for_symbols(symbols, layout, selected, policy_source=policy_source),
        overwrite=overwrite,
    )


def target_candidates(root: pathlib.Path,
                      compact: str,
                      symbol: str,
                      window_seconds: int | None) -> list[pathlib.Path]:
    lower = symbol.lower()
    suffix = "" if window_seconds is None else f"_window_{window_seconds}s"
    filename = f"market_targets_{lower}_{compact}{suffix}.csv"
    candidates = (
        root / filename,
        root / f"itch_{compact}_{lower}" / filename,
    )
    # resolve() also deduplicates a target root that itself is one of the
    # standard per-symbol extracted directories.
    found: list[pathlib.Path] = []
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in found:
                found.append(resolved)
    return found


def target_snapshot_accounting(
    manifest: Mapping[str, object],
    *,
    manifest_path: pathlib.Path,
    window_seconds: int | None,
) -> tuple[int, int]:
    """Return exact one-second valid/invalid counts for a target horizon.

    The multi-symbol extractor versions already in the six-day compact
    archive persist the prefix observation count and exact coverage value,
    while the single-symbol extractor also persists the two integer counts.
    Accept both lossless representations so this stricter audit does not
    require re-extracting ITCH, but reject any non-integral or incomplete
    accounting.
    """
    expected = (
        CERTIFICATION_SESSION_DURATION_SECONDS
        if window_seconds is None else window_seconds
    )
    source: Mapping[str, object] = manifest
    if window_seconds is not None:
        windows = manifest.get("market_target_windows")
        metadata = (
            windows.get(str(window_seconds))
            if isinstance(windows, Mapping) else None
        )
        if not isinstance(metadata, Mapping):
            raise CalibrationError(
                f"extractor manifest {manifest_path} does not record "
                f"{window_seconds}-second target accounting"
            )
        source = metadata

    valid_value = source.get("valid_snapshots")
    invalid_value = source.get("invalid_snapshots")
    if valid_value is not None or invalid_value is not None:
        if (isinstance(valid_value, bool) or isinstance(invalid_value, bool)
                or not isinstance(valid_value, int)
                or not isinstance(invalid_value, int)):
            raise CalibrationError(
                f"extractor manifest {manifest_path} has non-integer "
                "valid/invalid snapshot accounting"
            )
        valid = valid_value
        invalid = invalid_value
    elif window_seconds is not None:
        values = source.get("values")
        coverage_value = (
            values.get("two_sided_sample_fraction")
            if isinstance(values, Mapping) else None
        )
        # The compact six-day archive was produced by an extractor revision
        # that omitted prefix coverage from otherwise complete window
        # records.  If (and only if) the authoritative full-session counters
        # prove that every one-second snapshot was valid, every prefix is
        # necessarily fully covered as well.  This is an exact logical
        # implication, not an imputation.  Any session containing an invalid
        # snapshot still fails closed because its location within the prefix
        # is unknown.
        if coverage_value is None:
            full_valid = manifest.get("valid_snapshots")
            full_invalid = manifest.get("invalid_snapshots")
            if (
                not isinstance(full_valid, bool)
                and not isinstance(full_invalid, bool)
                and isinstance(full_valid, int)
                and isinstance(full_invalid, int)
                and full_valid == CERTIFICATION_SESSION_DURATION_SECONDS
                and full_invalid == 0
            ):
                valid = expected
                invalid = 0
                return valid, invalid
        if isinstance(coverage_value, bool):
            coverage_value = None
        try:
            coverage = float(coverage_value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise CalibrationError(
                f"extractor manifest {manifest_path} lacks exact prefix "
                "valid/invalid counts or two-sided coverage"
            ) from error
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise CalibrationError(
                f"extractor manifest {manifest_path} has invalid prefix "
                "two-sided coverage"
            )
        raw_valid = coverage * expected
        valid = int(round(raw_valid))
        if not math.isclose(
            raw_valid, float(valid), rel_tol=0.0, abs_tol=1.0e-8,
        ):
            raise CalibrationError(
                f"extractor manifest {manifest_path} prefix coverage cannot "
                "be reconciled to an integer snapshot count"
            )
        invalid = expected - valid
    else:
        raise CalibrationError(
            f"extractor manifest {manifest_path} lacks full-session "
            "valid/invalid snapshot accounting"
        )

    if valid < 0 or invalid < 0 or valid + invalid != expected:
        raise CalibrationError(
            f"extractor manifest {manifest_path} accounts for "
            f"{valid + invalid} observations; expected exactly {expected}"
        )
    return valid, invalid


def validate_target_manifest(target_path: pathlib.Path,
                             symbol: str,
                             compact: str,
                             window_seconds: int | None) -> Mapping[str, object]:
    """Require the one-second provenance matching calibration summaries.

    The fragmented calibration command samples every simulated book on a
    one-second fixed clock.  A target produced at 60 seconds is a different
    statistic even if it happens to have a matching file name, so accepting it
    would make the WMM objective scientifically invalid.
    """
    manifest_path = target_path.parent / f"itch_manifest_{symbol.lower()}_{compact}.json"
    if not manifest_path.is_file():
        raise CalibrationError(
            f"target {target_path} has no extractor manifest {manifest_path}; "
            "re-extract it at a one-second clock"
        )
    try:
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(f"cannot read extractor manifest {manifest_path}") from error
    interval_value = manifest.get("snapshot_interval_ms")
    if (isinstance(interval_value, bool)
            or not isinstance(interval_value, (int, float))):
        raise CalibrationError(
            f"extractor manifest {manifest_path} has no valid snapshot_interval_ms"
        )
    interval_numeric = float(interval_value)
    if (not math.isfinite(interval_numeric)
            or not interval_numeric.is_integer()):
        raise CalibrationError(
            f"extractor manifest {manifest_path} has no valid snapshot_interval_ms"
        )
    interval_ms = int(interval_numeric)
    if interval_ms != 1000:
        raise CalibrationError(
            f"target {target_path} uses snapshot_interval_ms={interval_ms}; "
            "cluster behavioural calibration requires 1000 ms"
        )
    expected_date = f"{compact[0:4]}-{compact[4:6]}-{compact[6:8]}"
    if manifest.get("trading_date") != expected_date:
        raise CalibrationError(
            f"extractor manifest {manifest_path} has trading_date="
            f"{manifest.get('trading_date')!r}; expected {expected_date!r}"
        )
    if str(manifest.get("symbol", "")).strip().upper() != symbol:
        raise CalibrationError(
            f"extractor manifest {manifest_path} does not identify symbol {symbol}"
        )
    if (manifest.get("session_start") != CERTIFICATION_SESSION_START
            or manifest.get("session_end") != CERTIFICATION_SESSION_END):
        raise CalibrationError(
            f"extractor manifest {manifest_path} must cover the canonical "
            f"{CERTIFICATION_SESSION_START}-{CERTIFICATION_SESSION_END} session"
        )
    aggregation_duration = manifest.get("aggregation_duration_seconds")
    if aggregation_duration is not None:
        if isinstance(aggregation_duration, bool):
            raise CalibrationError(
                f"extractor manifest {manifest_path} has invalid "
                "aggregation_duration_seconds"
            )
        try:
            aggregation_seconds = float(aggregation_duration)
        except (TypeError, ValueError) as error:
            raise CalibrationError(
                f"extractor manifest {manifest_path} has invalid "
                "aggregation_duration_seconds"
            ) from error
        if (not math.isfinite(aggregation_seconds)
                or not aggregation_seconds.is_integer()
                or int(aggregation_seconds)
                    != CERTIFICATION_SESSION_DURATION_SECONDS):
            raise CalibrationError(
                f"extractor manifest {manifest_path} has "
                f"aggregation_duration_seconds={aggregation_duration!r}; "
                f"expected {CERTIFICATION_SESSION_DURATION_SECONDS}"
            )
    if window_seconds is None:
        target_snapshot_accounting(
            manifest, manifest_path=manifest_path, window_seconds=None,
        )
        return manifest
    windows = manifest.get("market_target_windows")
    if not isinstance(windows, Mapping):
        raise CalibrationError(
            f"extractor manifest {manifest_path} does not record target-window provenance"
        )
    metadata = windows.get(str(window_seconds))
    if not isinstance(metadata, Mapping):
        raise CalibrationError(
            f"extractor manifest {manifest_path} does not record {window_seconds}-second targets"
        )
    if metadata.get("file") != target_path.name:
        raise CalibrationError(
            f"extractor manifest {manifest_path} does not identify {target_path.name} "
            f"as its {window_seconds}-second target"
        )
    if metadata.get("duration_seconds") != window_seconds:
        raise CalibrationError(
            f"extractor manifest {manifest_path} has "
            f"duration_seconds={metadata.get('duration_seconds')!r} for a "
            f"{window_seconds}-second target"
        )
    if metadata.get("observations") != window_seconds:
        raise CalibrationError(
            f"extractor manifest {manifest_path} has {metadata.get('observations')!r} "
            f"observations for a {window_seconds}-second target; expected one per second"
        )
    target_snapshot_accounting(
        manifest, manifest_path=manifest_path,
        window_seconds=window_seconds,
    )
    return manifest


def target_artifact_bundle_sha256(
    root: pathlib.Path,
    day: str,
    symbols: Sequence[str],
    horizons: Sequence[int | None],
) -> str:
    """Bind every target CSV and extractor manifest used by a phase."""
    compact = compact_date(day)
    digest = hashlib.sha256()
    manifests_seen: set[pathlib.Path] = set()

    def add(identity: str, path: pathlib.Path) -> None:
        content = path.read_bytes()
        name = identity.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)

    ordered_symbols = sorted(
        normalise_symbol(value, label="target bundle symbol") for value in symbols
    )
    for symbol in ordered_symbols:
        for horizon in horizons:
            candidates = target_candidates(root, compact, symbol, horizon)
            if len(candidates) != 1:
                raise CalibrationError(
                    f"target bundle needs exactly one {day}/{symbol}/{horizon} "
                    f"target; found {len(candidates)}"
                )
            target = candidates[0]
            validate_target_manifest(target, symbol, compact, horizon)
            role = "full_session" if horizon is None else f"window_{horizon}s"
            add(f"{day}:{symbol}:{role}:target", target)
            manifest = target.parent / f"itch_manifest_{symbol.lower()}_{compact}.json"
            if manifest not in manifests_seen:
                add(f"{day}:{symbol}:manifest", manifest)
                manifests_seen.add(manifest)
    return digest.hexdigest()


def empirical_two_sided_target(
    manifest: Mapping[str, object],
    *,
    manifest_path: pathlib.Path,
    window_seconds: int | None,
) -> TargetMoment:
    """Recover coverage from extractor provenance for legacy target CSVs.

    New extraction artifacts store the moment directly in every target CSV.
    This fallback keeps the already completed multi-day ITCH extraction usable
    by reading its certified valid/invalid snapshot counts instead of forcing a
    second scan of the multi-gigabyte source files.
    """
    valid, invalid = target_snapshot_accounting(
        manifest, manifest_path=manifest_path,
        window_seconds=window_seconds,
    )
    total = valid + invalid
    target = valid / total
    binomial_se = math.sqrt(target * (1.0 - target) / total)
    return TargetMoment(target=target, empirical_scale=max(0.005, binomial_se), weight=1.0)


def empirical_background_event_rate_target(
    manifest: Mapping[str, object], *, manifest_path: pathlib.Path,
) -> TargetMoment:
    """Derive the six-bucket visible-flow rate from extractor provenance.

    The extractor does not duplicate this direct statistic in every horizon
    target CSV.  Hawkes inputs are stationary rate inputs, so the same
    complete-session observed rate is intentionally used at all three
    simulation horizons.  The robust selector uses a log-ratio residual; the
    scale below is retained only for the conventional WMM diagnostic.
    """
    counts = manifest.get("distribution_observation_counts")
    if not isinstance(counts, Mapping):
        raise CalibrationError(
            f"extractor manifest {manifest_path} lacks "
            "distribution_observation_counts"
        )
    total = 0
    for event in BACKGROUND_EVENT_NAMES:
        try:
            count_value = counts[event]
        except KeyError as error:
            raise CalibrationError(
                f"extractor manifest {manifest_path} has no valid {event} count"
            ) from error
        if (isinstance(count_value, bool)
                or not isinstance(count_value, (int, float))):
            raise CalibrationError(
                f"extractor manifest {manifest_path} has no valid {event} count"
            )
        if isinstance(count_value, int):
            count = count_value
        else:
            if (not math.isfinite(count_value)
                    or not count_value.is_integer()):
                raise CalibrationError(
                    f"extractor manifest {manifest_path} has no valid "
                    f"{event} count"
                )
            count = int(count_value)
        if count < 0:
            raise CalibrationError(
                f"extractor manifest {manifest_path} has negative {event} count"
            )
        total += count
    if total <= 0:
        raise CalibrationError(
            f"extractor manifest {manifest_path} has zero visible-flow events"
        )
    duration_value = manifest.get("aggregation_duration_seconds")
    if duration_value is not None:
        try:
            duration_float = float(duration_value)
        except (TypeError, ValueError) as error:
            raise CalibrationError(
                f"extractor manifest {manifest_path} has invalid "
                "aggregation_duration_seconds"
            ) from error
        if (not math.isfinite(duration_float) or duration_float <= 0.0
                or not duration_float.is_integer()):
            raise CalibrationError(
                f"extractor manifest {manifest_path} has invalid "
                "aggregation_duration_seconds"
            )
        duration = int(duration_float)
    else:
        def clock_seconds(field: str) -> int:
            value = manifest.get(field)
            if not isinstance(value, str):
                raise CalibrationError(
                    f"extractor manifest {manifest_path} lacks {field}"
                )
            try:
                pieces = [int(piece) for piece in value.split(":")]
            except ValueError as error:
                raise CalibrationError(
                    f"extractor manifest {manifest_path} has invalid {field}"
                ) from error
            if (len(pieces) != 3 or not 0 <= pieces[0] <= 23
                    or not 0 <= pieces[1] <= 59 or not 0 <= pieces[2] <= 59):
                raise CalibrationError(
                    f"extractor manifest {manifest_path} has invalid {field}"
                )
            return (pieces[0] * 60 + pieces[1]) * 60 + pieces[2]

        duration = clock_seconds("session_end") - clock_seconds("session_start")
        if duration <= 0:
            raise CalibrationError(
                f"extractor manifest {manifest_path} has non-positive session duration"
            )
    target = total / duration
    poisson_se = math.sqrt(total) / duration
    diagnostic_scale = max(1.0e-6, poisson_se, 0.05 * target)
    return TargetMoment(
        target=target, empirical_scale=diagnostic_scale, weight=1.0,
    )


def authoritative_manifest_target_maps(
    manifest: Mapping[str, object],
    *,
    manifest_path: pathlib.Path,
    window_seconds: int | None,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Return the extractor's authoritative target and scale mappings."""
    if window_seconds is None:
        raw_values = manifest.get("market_values")
        raw_scales = manifest.get("market_target_scales")
    else:
        windows = manifest.get("market_target_windows")
        metadata = (
            windows.get(str(window_seconds))
            if isinstance(windows, Mapping) else None
        )
        if not isinstance(metadata, Mapping):
            raise CalibrationError(
                f"extractor manifest {manifest_path} lacks the "
                f"{window_seconds}-second target record"
            )
        raw_values = metadata.get("values")
        raw_scales = metadata.get("scales")
    if not isinstance(raw_values, Mapping) or not isinstance(raw_scales, Mapping):
        raise CalibrationError(
            f"extractor manifest {manifest_path} lacks authoritative target "
            "values or scales"
        )
    values = dict(raw_values)
    scales = dict(raw_scales)
    metric = "two_sided_sample_fraction"
    has_value = metric in values
    has_scale = metric in scales
    if has_value != has_scale:
        raise CalibrationError(
            f"extractor manifest {manifest_path} has incomplete authoritative "
            f"{metric} target or scale"
        )
    if not has_value:
        coverage = empirical_two_sided_target(
            manifest,
            manifest_path=manifest_path,
            window_seconds=window_seconds,
        )
        values[metric] = coverage.target
        scales[metric] = coverage.empirical_scale
    return values, scales


def strict_manifest_float(value: object, *, label: str) -> float:
    """Parse a JSON numeric value without bool/string coercion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{label} is not a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationError(f"{label} is not a finite JSON number")
    return result


def load_targets(root: pathlib.Path,
                 day: str,
                 symbols: Iterable[str],
                 *, window_seconds: int | None = None) -> dict[str, dict[str, TargetMoment]]:
    """Load strict full-session or horizon-matched target artifacts by symbol."""
    if window_seconds is not None and window_seconds <= 0:
        raise CalibrationError("target window duration must be positive")
    if not root.is_dir():
        raise CalibrationError(f"target root is not a directory: {root}")
    compact = compact_date(day)
    result: dict[str, dict[str, TargetMoment]] = {}
    for raw_symbol in symbols:
        symbol = normalise_symbol(raw_symbol, label="target symbols")
        found = target_candidates(root, compact, symbol, window_seconds)
        description = "full-session" if window_seconds is None else f"{window_seconds}-second"
        if not found:
            suffix = "" if window_seconds is None else f"_window_{window_seconds}s"
            expected = root / f"itch_{compact}_{symbol.lower()}" / (
                f"market_targets_{symbol.lower()}_{compact}{suffix}.csv"
            )
            raise FileNotFoundError(
                f"missing {description} target for {symbol}: expected {expected}; "
                "short stages require an extractor-generated matched-prefix target"
            )
        if len(found) != 1:
            raise CalibrationError(
                f"ambiguous {description} targets for {symbol}: {', '.join(map(str, found))}"
            )
        manifest_path = (
            found[0].parent
            / f"itch_manifest_{symbol.lower()}_{compact}.json"
        )
        manifest = validate_target_manifest(found[0], symbol, compact, window_seconds)
        fields, rows = csv_table(found[0])
        required = {"name", "target", "scale"}
        missing = sorted(required.difference(fields))
        if missing:
            raise CalibrationError(f"target file {found[0]} missing columns: {missing}")
        values: dict[str, TargetMoment] = {}
        for line_number, row in enumerate(rows, start=2):
            metric = row["name"]
            if metric not in METRICS:
                continue
            if metric in values:
                raise CalibrationError(f"duplicate target metric {metric} in {found[0]}:{line_number}")
            target = finite_float(row["target"], label=f"{found[0]}:{metric}:target")
            scale = finite_float(row["scale"], label=f"{found[0]}:{metric}:scale")
            weight = finite_float(
                row.get("weight", "1"), label=f"{found[0]}:{metric}:weight"
            )
            if scale <= 0.0 or weight <= 0.0:
                raise CalibrationError(
                    f"target scale and weight must be positive in {found[0]} for {metric}"
                )
            # The extractor used by the certified protocol predeclares equal
            # importance weights.  The manifest authoritatively binds target
            # values and scales, but does not carry a second weight map; do
            # not therefore let an edited CSV silently alter the WMM
            # diagnostic or its deterministic tie-breaker.
            if weight != 1.0:
                raise CalibrationError(
                    f"certified extractor target weight must equal 1 in "
                    f"{found[0]} for {metric}; observed {weight:.17g}"
                )
            values[metric] = TargetMoment(target, scale, weight)
        manifest_values, manifest_scales = authoritative_manifest_target_maps(
            manifest,
            manifest_path=manifest_path,
            window_seconds=window_seconds,
        )
        extractor_metrics = tuple(
            metric for metric in METRICS if metric != "background_event_rate"
        )
        for metric in extractor_metrics:
            if metric not in manifest_values or metric not in manifest_scales:
                raise CalibrationError(
                    f"extractor manifest {manifest_path} lacks authoritative "
                    f"{metric} target or scale"
                )
            manifest_target = strict_manifest_float(
                manifest_values[metric],
                label=f"extractor manifest {manifest_path} {metric} target",
            )
            manifest_scale = strict_manifest_float(
                manifest_scales[metric],
                label=f"extractor manifest {manifest_path} {metric} scale",
            )
            if manifest_scale <= 0.0:
                raise CalibrationError(
                    f"extractor manifest {manifest_path} has non-positive "
                    f"{metric} scale"
                )
            if metric not in values:
                if metric == "two_sided_sample_fraction":
                    # Earlier extractor target CSVs omitted this row while
                    # persisting exact clock coverage in the manifest.
                    values[metric] = TargetMoment(
                        manifest_target, manifest_scale, 1.0,
                    )
                    continue
                raise CalibrationError(
                    f"target file {found[0]} missing required metric {metric}"
                )
            csv_moment = values[metric]
            if (not math.isclose(
                    csv_moment.target, manifest_target,
                    rel_tol=1.0e-12, abs_tol=1.0e-15)
                    or not math.isclose(
                        csv_moment.empirical_scale, manifest_scale,
                        rel_tol=1.0e-12, abs_tol=1.0e-15)):
                raise CalibrationError(
                    f"target file {found[0]} {metric} target or scale "
                    f"disagrees with extractor manifest {manifest_path}"
                )
            values[metric] = TargetMoment(
                manifest_target, manifest_scale, csv_moment.weight,
            )

        manifest_coverage = empirical_two_sided_target(
            manifest,
            manifest_path=manifest_path,
            window_seconds=window_seconds,
        )
        if not math.isclose(
            values["two_sided_sample_fraction"].target,
            manifest_coverage.target,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise CalibrationError(
                f"target file {found[0]} two_sided_sample_fraction "
                f"disagrees with exact snapshot accounting in {manifest_path}"
            )

        manifest_event_rate = empirical_background_event_rate_target(
            manifest, manifest_path=manifest_path,
        )
        if "background_event_rate" in values:
            csv_event_rate = values["background_event_rate"]
            if not math.isclose(
                csv_event_rate.target, manifest_event_rate.target,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise CalibrationError(
                    f"target file {found[0]} background_event_rate "
                    f"disagrees with event counts in {manifest_path}"
                )
            values["background_event_rate"] = TargetMoment(
                manifest_event_rate.target,
                csv_event_rate.empirical_scale,
                csv_event_rate.weight,
            )
        else:
            values["background_event_rate"] = manifest_event_rate
        missing_metrics = sorted(set(METRICS).difference(values))
        if missing_metrics:
            raise CalibrationError(
                f"target file {found[0]} missing required metrics: {missing_metrics}"
            )
        result[symbol] = values
    if not result:
        raise CalibrationError("no symbols supplied for target loading")
    return result


def summary_rows(path: pathlib.Path,
                 symbols: Iterable[str],
                 *,
                 required_expected_sample_count: int | None = None,
                 ) -> dict[str, dict[str, str]]:
    """Load a fragmented per-asset summary and verify fixed-clock accounting."""
    expected = {normalise_symbol(symbol, label="summary symbols") for symbol in symbols}
    fields, rows = csv_table(path)
    required = {
        "symbol", "sample_count", "expected_sample_count",
        "invalid_sample_count", "structurally_valid", *METRICS,
        *BOUNDARY_SUMMARY_FIELDS,
    }
    missing = sorted(required.difference(fields))
    if missing:
        raise CalibrationError(f"asset summary {path} missing columns: {missing}")
    result: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        symbol = normalise_symbol(row["symbol"], label=f"{path}:{line_number}")
        if symbol in result:
            raise CalibrationError(f"duplicate summary row for {symbol} in {path}")
        try:
            valid_samples = int(row["sample_count"])
            invalid_samples = int(row["invalid_sample_count"])
            expected_samples = int(row["expected_sample_count"])
        except ValueError as error:
            raise CalibrationError(
                f"asset summary {path} has non-integer sample accounting for {symbol}"
            ) from error
        if valid_samples < 0 or invalid_samples < 0 or expected_samples <= 0:
            raise CalibrationError(
                f"asset summary {path} has invalid sample counts for {symbol}: "
                f"valid={valid_samples} invalid={invalid_samples} "
                f"expected={expected_samples}"
            )
        if (required_expected_sample_count is not None
                and expected_samples != required_expected_sample_count):
            raise CalibrationError(
                f"asset summary {path} has the wrong fixed-clock horizon for "
                f"{symbol}: expected_sample_count={expected_samples}, requested="
                f"{required_expected_sample_count}"
            )
        if valid_samples + invalid_samples != expected_samples:
            raise CalibrationError(
                f"asset summary {path} has incomplete fixed-clock accounting for "
                f"{symbol}: valid={valid_samples} invalid={invalid_samples} "
                f"expected={expected_samples}"
            )
        coverage = finite_float(
            row["two_sided_sample_fraction"],
            label=f"{path}:{symbol}:two_sided_sample_fraction",
        )
        expected_coverage = valid_samples / expected_samples
        if not 0.0 <= coverage <= 1.0:
            raise CalibrationError(
                f"asset summary {path} has out-of-range two-sided coverage for "
                f"{symbol}: {coverage}"
            )
        if not math.isclose(coverage, expected_coverage, rel_tol=0.0, abs_tol=1.0e-12):
            raise CalibrationError(
                f"asset summary {path} has inconsistent two-sided coverage for "
                f"{symbol}: fraction={coverage:.17g} but "
                f"sample_count/expected_sample_count={expected_coverage:.17g}"
            )
        if row["structurally_valid"] not in {"0", "1"}:
            raise CalibrationError(
                f"asset summary {path} has invalid structurally_valid flag for "
                f"{symbol}: {row['structurally_valid']!r}"
            )
        expected_structurally_valid = invalid_samples == 0 and valid_samples == expected_samples
        reported_structurally_valid = row["structurally_valid"] == "1"
        if reported_structurally_valid != expected_structurally_valid:
            raise CalibrationError(
                f"asset summary {path} has an inconsistent structurally_valid flag "
                f"for {symbol}: reported={int(reported_structurally_valid)} but "
                f"invalid_sample_count={invalid_samples} and "
                f"sample_count/expected_sample_count={valid_samples}/{expected_samples} "
                f"imply {int(expected_structurally_valid)}"
            )
        for field in BOUNDARY_SUMMARY_FIELDS:
            try:
                value = int(row[field])
            except ValueError as error:
                raise CalibrationError(
                    f"asset summary {path} has non-integer {field} for {symbol}"
                ) from error
            if value < 0:
                raise CalibrationError(
                    f"asset summary {path} has negative {field} for {symbol}"
                )
        action_fields_present = set(BOUNDARY_ACTION_SUMMARY_FIELDS).intersection(fields)
        if action_fields_present and action_fields_present != set(
                BOUNDARY_ACTION_SUMMARY_FIELDS):
            missing_action_fields = sorted(
                set(BOUNDARY_ACTION_SUMMARY_FIELDS).difference(fields)
            )
            raise CalibrationError(
                f"asset summary {path} has an incomplete action-specific "
                f"boundary schema; missing={missing_action_fields}"
            )
        if action_fields_present:
            action_counts: dict[str, int] = {}
            for field in BOUNDARY_ACTION_SUMMARY_FIELDS:
                try:
                    action_counts[field] = int(row[field])
                except ValueError as error:
                    raise CalibrationError(
                        f"asset summary {path} has non-integer {field} for "
                        f"{symbol}"
                    ) from error
                if action_counts[field] < 0:
                    raise CalibrationError(
                        f"asset summary {path} has negative {field} for {symbol}"
                    )
            if int(row["removal_boundary_truncation_events"]) != (
                    action_counts["market_boundary_truncation_events"]
                    + action_counts["cancel_boundary_truncation_events"]):
                raise CalibrationError(
                    f"asset summary {path} has inconsistent action-specific "
                    f"boundary-event accounting for {symbol}"
                )
            if int(row["removal_boundary_truncated_quantity"]) != (
                    action_counts["market_boundary_truncated_quantity"]
                    + action_counts["cancel_boundary_truncated_quantity"]):
                raise CalibrationError(
                    f"asset summary {path} has inconsistent action-specific "
                    f"boundary-quantity accounting for {symbol}"
                )
        source_counts = {
            field: int(row[field]) for field in BOUNDARY_SOURCE_SUMMARY_FIELDS
        }
        if int(row["removal_boundary_truncation_events"]) != (
                source_counts["background_boundary_truncation_events"]
                + source_counts["value_boundary_truncation_events"]
                + source_counts["other_boundary_truncation_events"]):
            raise CalibrationError(
                f"asset summary {path} has inconsistent source-specific "
                f"boundary-event accounting for {symbol}"
            )
        if int(row["removal_boundary_truncated_quantity"]) != (
                source_counts["background_boundary_truncated_quantity"]
                + source_counts["value_boundary_truncated_quantity"]
                + source_counts["other_boundary_truncated_quantity"]):
            raise CalibrationError(
                f"asset summary {path} has inconsistent source-specific "
                f"boundary-quantity accounting for {symbol}"
            )
        if source_counts["background_boundary_truncation_events"] > int(
                row["background_event_count"]):
            raise CalibrationError(
                f"asset summary {path} reports more background boundary events "
                f"than generated background events for {symbol}"
            )
        if source_counts["value_boundary_truncation_events"] > int(
                row["value_order_count"]):
            raise CalibrationError(
                f"asset summary {path} reports more value boundary events than "
                f"submitted value orders for {symbol}"
            )
        if source_counts["value_boundary_truncated_quantity"] > int(
                row["value_requested_quantity"]):
            raise CalibrationError(
                f"asset summary {path} reports more value boundary quantity than "
                f"requested value quantity for {symbol}"
            )
        result[symbol] = row
    if set(result) != expected:
        missing_symbols = sorted(expected.difference(result))
        extra_symbols = sorted(set(result).difference(expected))
        raise CalibrationError(
            f"asset summary {path} symbols do not match requested config; "
            f"missing={missing_symbols} extra={extra_symbols}"
        )
    return result


def two_sided_execution_integrity(
    summary_paths: Sequence[pathlib.Path],
    symbols: Sequence[str],
    *,
    required_expected_sample_count: int | None = None,
) -> tuple[bool, list[dict[str, object]]]:
    """Require every fixed-clock observation to contain both book sides."""
    failures: list[dict[str, object]] = []
    for path in summary_paths:
        rows = summary_rows(
            path, symbols,
            required_expected_sample_count=required_expected_sample_count,
        )
        for symbol, row in rows.items():
            valid = int(row["sample_count"])
            invalid = int(row["invalid_sample_count"])
            expected = int(row["expected_sample_count"])
            fraction = finite_float(
                row["two_sided_sample_fraction"],
                label=f"{path}:{symbol}:two_sided_sample_fraction",
            )
            if (invalid != 0 or valid != expected
                    or not math.isclose(fraction, 1.0, rel_tol=0.0, abs_tol=1.0e-12)):
                failures.append({
                    "summary_path": str(path),
                    "symbol": symbol,
                    "sample_count": valid,
                    "invalid_sample_count": invalid,
                    "expected_sample_count": expected,
                    "two_sided_sample_fraction": fraction,
                })
    return not failures, failures


def _finite_boundary_ratio(
    numerator: int, denominator: int
) -> tuple[float | None, bool]:
    """Return (ratio, denominator-valid); 0/0 is the only valid zero case."""
    if denominator == 0:
        return (0.0, True) if numerator == 0 else (None, False)
    return numerator / denominator, True


def _source_boundary_adequacy(
    summary_paths: Sequence[pathlib.Path],
    symbols: Sequence[str],
    *,
    source: str,
    maximum_symbol_event_ratio: float,
    maximum_symbol_quantity_ratio: float,
    maximum_aggregate_event_ratio: float,
    maximum_aggregate_quantity_ratio: float,
    required_expected_sample_count: int | None = None,
) -> dict[str, object]:
    """Gate one removal source using predeclared-seed pooled estimands.

    A single stochastic path is not an independent model.  Ratios are
    therefore pooled by symbol over the complete predeclared seed set before
    applying the asset limit, and over symbols and seeds before applying the
    aggregate limit.  Per-seed exceedances remain explicit diagnostics.  Most
    importantly, numerators and denominators always refer to the same source.
    """
    if source not in {"background", "value"}:
        raise CalibrationError(f"unsupported boundary source: {source}")
    if source == "background":
        event_numerator_field = "background_boundary_truncation_events"
        quantity_numerator_field = "background_boundary_truncated_quantity"
        event_denominator_fields = ("background_event_count",)
        quantity_denominator_fields = (
            "background_market_requested_quantity",
            "background_cancel_requested_quantity",
        )
    else:
        event_numerator_field = "value_boundary_truncation_events"
        quantity_numerator_field = "value_boundary_truncated_quantity"
        event_denominator_fields = ("value_order_count",)
        quantity_denominator_fields = ("value_requested_quantity",)

    thresholds = {
        "maximum_symbol_seed_pool_event_ratio": maximum_symbol_event_ratio,
        "maximum_symbol_seed_pool_quantity_ratio": maximum_symbol_quantity_ratio,
        "maximum_aggregate_seed_pool_event_ratio": maximum_aggregate_event_ratio,
        "maximum_aggregate_seed_pool_quantity_ratio": maximum_aggregate_quantity_ratio,
    }
    runs: list[dict[str, object]] = []
    diagnostic_per_seed_failures: list[dict[str, object]] = []
    symbol_totals: dict[str, dict[str, int]] = {
        symbol: {"event_numerator": 0, "event_denominator": 0,
                 "quantity_numerator": 0, "quantity_denominator": 0}
        for symbol in sorted(symbols)
    }
    for path in summary_paths:
        rows = summary_rows(
            path, symbols,
            required_expected_sample_count=required_expected_sample_count,
        )
        assets: list[dict[str, object]] = []
        run_totals = {"event_numerator": 0, "event_denominator": 0,
                      "quantity_numerator": 0, "quantity_denominator": 0}
        for symbol in sorted(rows):
            row = rows[symbol]
            event_numerator = int(row[event_numerator_field])
            event_denominator = sum(int(row[field]) for field in event_denominator_fields)
            quantity_numerator = int(row[quantity_numerator_field])
            quantity_denominator = sum(
                int(row[field]) for field in quantity_denominator_fields
            )
            event_ratio, event_valid = _finite_boundary_ratio(
                event_numerator, event_denominator
            )
            quantity_ratio, quantity_valid = _finite_boundary_ratio(
                quantity_numerator, quantity_denominator
            )
            record = {
                "symbol": symbol,
                "source": source,
                "boundary_truncation_events": event_numerator,
                "source_event_count": event_denominator,
                "boundary_event_ratio": event_ratio,
                "boundary_truncated_quantity": quantity_numerator,
                "source_requested_quantity": quantity_denominator,
                "boundary_quantity_ratio": quantity_ratio,
            }
            if source == "background":
                record["background_event_count"] = event_denominator
                record["background_removal_requested_quantity"] = quantity_denominator
            else:
                record["value_order_count"] = event_denominator
                record["value_requested_quantity"] = quantity_denominator
            assets.append(record)
            for metric, ratio, valid, threshold, numerator, denominator in (
                ("boundary_event_ratio", event_ratio, event_valid,
                 maximum_symbol_event_ratio, event_numerator, event_denominator),
                ("boundary_quantity_ratio", quantity_ratio, quantity_valid,
                 maximum_symbol_quantity_ratio, quantity_numerator,
                 quantity_denominator),
            ):
                if not valid or ratio is None or ratio > threshold + 1.0e-15:
                    diagnostic_per_seed_failures.append({
                        "scope": "asset_seed_diagnostic",
                        "summary_path": str(path), "symbol": symbol,
                        "source": source, "metric": metric,
                        "numerator": numerator, "denominator": denominator,
                        "ratio": ratio, "maximum": threshold,
                    })
            totals = symbol_totals[symbol]
            for label, amount in (
                ("event_numerator", event_numerator),
                ("event_denominator", event_denominator),
                ("quantity_numerator", quantity_numerator),
                ("quantity_denominator", quantity_denominator),
            ):
                totals[label] += amount
                run_totals[label] += amount
        run_event_ratio, _ = _finite_boundary_ratio(
            run_totals["event_numerator"], run_totals["event_denominator"]
        )
        run_quantity_ratio, _ = _finite_boundary_ratio(
            run_totals["quantity_numerator"], run_totals["quantity_denominator"]
        )
        aggregate = {
            "source": source,
            "boundary_truncation_events": run_totals["event_numerator"],
            "source_event_count": run_totals["event_denominator"],
            "boundary_event_ratio": run_event_ratio,
            "boundary_truncated_quantity": run_totals["quantity_numerator"],
            "source_requested_quantity": run_totals["quantity_denominator"],
            "boundary_quantity_ratio": run_quantity_ratio,
        }
        if source == "background":
            aggregate["background_event_count"] = run_totals["event_denominator"]
            aggregate["background_removal_requested_quantity"] = (
                run_totals["quantity_denominator"]
            )
        else:
            aggregate["value_order_count"] = run_totals["event_denominator"]
            aggregate["value_requested_quantity"] = run_totals["quantity_denominator"]
        runs.append({"summary_path": str(path), "assets": assets,
                     "aggregate": aggregate})

    failures: list[dict[str, object]] = []
    symbol_pooled: list[dict[str, object]] = []
    aggregate_totals = {"event_numerator": 0, "event_denominator": 0,
                        "quantity_numerator": 0, "quantity_denominator": 0}
    for symbol, totals in symbol_totals.items():
        event_ratio, event_valid = _finite_boundary_ratio(
            totals["event_numerator"], totals["event_denominator"]
        )
        quantity_ratio, quantity_valid = _finite_boundary_ratio(
            totals["quantity_numerator"], totals["quantity_denominator"]
        )
        record = {"symbol": symbol, "source": source,
                  "boundary_truncation_events": totals["event_numerator"],
                  "source_event_count": totals["event_denominator"],
                  "boundary_event_ratio": event_ratio,
                  "boundary_truncated_quantity": totals["quantity_numerator"],
                  "source_requested_quantity": totals["quantity_denominator"],
                  "boundary_quantity_ratio": quantity_ratio}
        symbol_pooled.append(record)
        for metric, ratio, valid, threshold, numerator, denominator in (
            ("boundary_event_ratio", event_ratio, event_valid,
             maximum_symbol_event_ratio, totals["event_numerator"],
             totals["event_denominator"]),
            ("boundary_quantity_ratio", quantity_ratio, quantity_valid,
             maximum_symbol_quantity_ratio, totals["quantity_numerator"],
             totals["quantity_denominator"]),
        ):
            if not valid or ratio is None or ratio > threshold + 1.0e-15:
                failures.append({"scope": "symbol_seed_pool", "symbol": symbol,
                                 "source": source, "metric": metric,
                                 "numerator": numerator, "denominator": denominator,
                                 "ratio": ratio, "maximum": threshold})
        for label in aggregate_totals:
            aggregate_totals[label] += totals[label]

    aggregate_event_ratio, aggregate_event_valid = _finite_boundary_ratio(
        aggregate_totals["event_numerator"], aggregate_totals["event_denominator"]
    )
    aggregate_quantity_ratio, aggregate_quantity_valid = _finite_boundary_ratio(
        aggregate_totals["quantity_numerator"],
        aggregate_totals["quantity_denominator"],
    )
    aggregate_pooled = {
        "source": source,
        "boundary_truncation_events": aggregate_totals["event_numerator"],
        "source_event_count": aggregate_totals["event_denominator"],
        "boundary_event_ratio": aggregate_event_ratio,
        "boundary_truncated_quantity": aggregate_totals["quantity_numerator"],
        "source_requested_quantity": aggregate_totals["quantity_denominator"],
        "boundary_quantity_ratio": aggregate_quantity_ratio,
        "run_count": len(runs),
    }
    for metric, ratio, valid, threshold, numerator, denominator in (
        ("boundary_event_ratio", aggregate_event_ratio, aggregate_event_valid,
         maximum_aggregate_event_ratio, aggregate_totals["event_numerator"],
         aggregate_totals["event_denominator"]),
        ("boundary_quantity_ratio", aggregate_quantity_ratio,
         aggregate_quantity_valid, maximum_aggregate_quantity_ratio,
         aggregate_totals["quantity_numerator"],
         aggregate_totals["quantity_denominator"]),
    ):
        if not valid or ratio is None or ratio > threshold + 1.0e-15:
            failures.append({"scope": "aggregate_seed_pool", "source": source,
                             "metric": metric, "numerator": numerator,
                             "denominator": denominator, "ratio": ratio,
                             "maximum": threshold})
    return {
        "schema_version": 2,
        "source": source,
        "passed": not failures,
        "scope": "pooled_across_predeclared_seeds",
        "thresholds": thresholds,
        "per_seed_ratios_role": "diagnostic_only",
        "zero_denominator_rule": (
            "passes only when the corresponding numerator is zero"
        ),
        "runs": runs,
        "diagnostic_per_seed_failures": diagnostic_per_seed_failures,
        "symbol_pooled": symbol_pooled,
        "aggregate_pooled": aggregate_pooled,
        "failures": failures,
    }


def finite_boundary_adequacy(
    summary_paths: Sequence[pathlib.Path],
    symbols: Sequence[str],
    *,
    required_expected_sample_count: int | None = None,
) -> dict[str, object]:
    """Predeclared finite-boundary gate for background order flow only."""
    return _source_boundary_adequacy(
        summary_paths, symbols, source="background",
        maximum_symbol_event_ratio=CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_EVENT_RATIO,
        maximum_symbol_quantity_ratio=(
            CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_QUANTITY_RATIO
        ),
        maximum_aggregate_event_ratio=CERTIFICATION_MAXIMUM_RUN_BOUNDARY_EVENT_RATIO,
        maximum_aggregate_quantity_ratio=(
            CERTIFICATION_MAXIMUM_RUN_BOUNDARY_QUANTITY_RATIO
        ),
        required_expected_sample_count=required_expected_sample_count,
    )


def value_boundary_adequacy(
    summary_paths: Sequence[pathlib.Path],
    symbols: Sequence[str],
    *,
    required_expected_sample_count: int | None = None,
) -> dict[str, object]:
    """Predeclared finite-boundary gate attributable to value-agent orders."""
    return _source_boundary_adequacy(
        summary_paths, symbols, source="value",
        maximum_symbol_event_ratio=CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_EVENT_RATIO,
        maximum_symbol_quantity_ratio=(
            CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_QUANTITY_RATIO
        ),
        maximum_aggregate_event_ratio=CERTIFICATION_MAXIMUM_RUN_BOUNDARY_EVENT_RATIO,
        maximum_aggregate_quantity_ratio=(
            CERTIFICATION_MAXIMUM_RUN_BOUNDARY_QUANTITY_RATIO
        ),
        required_expected_sample_count=required_expected_sample_count,
    )


def weighted_moment_loss(
    summary_paths: Sequence[pathlib.Path],
    targets: Mapping[str, Mapping[str, TargetMoment]],
    symbols: Iterable[str],
    *,
    uncertainty_mode: str = "empirical",
    metrics: Sequence[str] = METRICS,
    required_expected_sample_count: int | None = None,
) -> tuple[float, list[MomentEstimate]]:
    """Compute diagonal weighted standardized WMM across symbols and seeds."""
    ordered_symbols = tuple(normalise_symbol(symbol, label="WMM symbols") for symbol in symbols)
    if not ordered_symbols:
        raise CalibrationError("weighted moment matching requires at least one symbol")
    if not summary_paths:
        raise CalibrationError("weighted moment matching requires at least one seed summary")
    if uncertainty_mode not in {"empirical", "combined"}:
        raise CalibrationError("uncertainty_mode must be 'empirical' or 'combined'")
    ordered_metrics = tuple(metrics)
    if not ordered_metrics or any(metric not in METRICS for metric in ordered_metrics):
        raise CalibrationError("WMM metrics must be a non-empty subset of METRICS")
    summaries = [
        summary_rows(
            path, ordered_symbols,
            required_expected_sample_count=required_expected_sample_count,
        )
        for path in summary_paths
    ]

    total_weight = 0.0
    weighted_squared = 0.0
    estimates: list[MomentEstimate] = []
    for symbol in ordered_symbols:
        if symbol not in targets:
            raise CalibrationError(f"missing targets for summary symbol {symbol}")
        for metric in ordered_metrics:
            try:
                target = targets[symbol][metric]
            except KeyError as error:
                raise CalibrationError(f"missing target {metric} for {symbol}") from error
            values = [
                finite_float(
                    summary[symbol][metric],
                    label=f"{path}:{symbol}:{metric}",
                )
                for path, summary in zip(summary_paths, summaries)
            ]
            simulated_mean = statistics.fmean(values)
            simulated_sample_sd = statistics.stdev(values) if len(values) > 1 else 0.0
            simulated_mean_se = simulated_sample_sd / math.sqrt(len(values))
            combined_scale = math.hypot(target.empirical_scale, simulated_mean_se)
            empirical_standardized = (
                (simulated_mean - target.target) / target.empirical_scale
            )
            combined_uncertainty = (
                (simulated_mean - target.target) / combined_scale
            )
            objective_residual = (
                empirical_standardized if uncertainty_mode == "empirical"
                else combined_uncertainty
            )
            contribution = target.weight * objective_residual * objective_residual
            if not all(math.isfinite(value) for value in (
                simulated_mean, simulated_sample_sd, simulated_mean_se,
                combined_scale, empirical_standardized, combined_uncertainty,
                objective_residual, contribution,
            )):
                raise CalibrationError(
                    f"non-finite WMM calculation for {symbol}:{metric}"
                )
            estimates.append(MomentEstimate(
                symbol=symbol,
                metric=metric,
                target=target.target,
                empirical_scale=target.empirical_scale,
                weight=target.weight,
                simulated_mean=simulated_mean,
                simulated_sample_sd=simulated_sample_sd,
                simulated_mean_se=simulated_mean_se,
                combined_scale=combined_scale,
                empirical_standardized_residual=empirical_standardized,
                combined_uncertainty_residual=combined_uncertainty,
                objective_residual=objective_residual,
                weighted_squared_residual=contribution,
                seed_count=len(values),
            ))
            total_weight += target.weight
            weighted_squared += contribution
    if total_weight <= 0.0:
        raise CalibrationError("total WMM target weight is not positive")
    return math.sqrt(weighted_squared / total_weight), estimates


def _estimate_value(estimate: MomentEstimate | Mapping[str, object], name: str) -> object:
    return getattr(estimate, name) if isinstance(estimate, MomentEstimate) else estimate[name]


def _clipped_logit(value: float) -> float:
    probability = min(
        1.0 - ROBUST_PROBABILITY_EPSILON,
        max(ROBUST_PROBABILITY_EPSILON, value),
    )
    return math.log(probability / (1.0 - probability))


def _robust_moment_residual(metric: str, simulated: float, target: float) -> float:
    """Return a dimensionless training residual with an economic interpretation."""
    if metric in POSITIVE_LOG_RATIO_METRICS:
        floor = 1.0e-12 if metric == "return_variance" else 1.0e-9
        return (
            math.log(max(simulated, floor) / max(target, floor))
            / ROBUST_LOG_RATIO_UNIT
        )
    if metric == "mid_move_rate":
        return (
            _clipped_logit(simulated) - _clipped_logit(target)
        ) / ROBUST_MID_MOVE_LOG_ODDS_UNIT
    if metric == "absolute_return_acf1":
        simulated_clipped = min(1.0 - ROBUST_PROBABILITY_EPSILON,
                                max(-1.0 + ROBUST_PROBABILITY_EPSILON, simulated))
        target_clipped = min(1.0 - ROBUST_PROBABILITY_EPSILON,
                             max(-1.0 + ROBUST_PROBABILITY_EPSILON, target))
        return (
            math.atanh(simulated_clipped) - math.atanh(target_clipped)
        ) / ROBUST_ACF_FISHER_UNIT
    if metric == "two_sided_sample_fraction":
        return (simulated - target) / ROBUST_COVERAGE_UNIT
    raise CalibrationError(f"no robust residual transform declared for metric {metric}")


def _huber_loss(residual: float, delta: float) -> float:
    absolute = abs(residual)
    if absolute <= delta:
        return 0.5 * residual * residual
    return delta * (absolute - 0.5 * delta)


def metric_balanced_robust_loss(
    estimates: Sequence[MomentEstimate | Mapping[str, object]],
    *,
    metrics: Sequence[str] = METRICS,
    huber_delta: float = DEFAULT_ROBUST_HUBER_DELTA,
) -> tuple[float, list[dict[str, object]]]:
    """Score moments with equal metric weight and bounded-influence Huber loss.

    This score is used only for training candidate selection and the frozen
    held-out fit gate.  The conventional empirical-SE WMM is retained beside
    it as an uncapped diagnostic.  Averaging first within each metric prevents
    the magnitude or symbol count of return variance from silently determining
    the entire model.
    """
    ordered_metrics = tuple(metrics)
    if not ordered_metrics or any(metric not in METRICS for metric in ordered_metrics):
        raise CalibrationError("robust metrics must be a non-empty subset of METRICS")
    if not math.isfinite(huber_delta) or huber_delta <= 0.0:
        raise CalibrationError("Huber delta must be finite and positive")
    residuals: dict[str, list[float]] = {metric: [] for metric in ordered_metrics}
    for estimate in estimates:
        metric = str(_estimate_value(estimate, "metric"))
        if metric not in residuals:
            continue
        simulated = finite_float(
            _estimate_value(estimate, "simulated_mean"),
            label=f"robust simulated {metric}",
        )
        target = finite_float(
            _estimate_value(estimate, "target"),
            label=f"robust target {metric}",
        )
        residual = _robust_moment_residual(metric, simulated, target)
        if not math.isfinite(residual):
            raise CalibrationError(f"non-finite robust residual for {metric}")
        residuals[metric].append(residual)

    missing = [metric for metric, values in residuals.items() if not values]
    if missing:
        raise CalibrationError(f"robust score lacks estimates for metrics: {missing}")
    details: list[dict[str, object]] = []
    metric_losses: list[float] = []
    for metric in ordered_metrics:
        values = residuals[metric]
        absolute_values = [abs(value) for value in values]
        mean_loss = statistics.fmean(_huber_loss(value, huber_delta) for value in values)
        metric_score = math.sqrt(2.0 * mean_loss)
        metric_losses.append(mean_loss)
        details.append({
            "metric": metric,
            "symbol_count": len(values),
            "mean_huber_loss": mean_loss,
            "score": metric_score,
            "median_absolute_residual": percentile(
                absolute_values, 0.5,
            ),
            "p90_absolute_residual": percentile(absolute_values, 0.9),
            "maximum_absolute_residual": max(absolute_values),
            "fraction_within_gross_residual_limit": (
                sum(value <= CERTIFICATION_GROSS_RESIDUAL_LIMIT
                    for value in absolute_values) / len(absolute_values)
            ),
        })
    return math.sqrt(2.0 * statistics.fmean(metric_losses)), details


def evaluation_selection_score(evaluation: Mapping[str, object]) -> float:
    """Return the centralized, training-only candidate ordering score."""
    return float(evaluation.get("selection_score", math.inf))


def candidate_selection_key(item: Mapping[str, object]) -> tuple[float, float, int]:
    """Order candidates once, consistently, without held-out information."""
    evaluation = item["evaluation"]
    if not isinstance(evaluation, Mapping):
        raise CalibrationError("candidate evaluation is not a mapping")
    return (
        evaluation_selection_score(evaluation),
        float(evaluation["fit_wsmrmse"]),
        int(item["candidate_index"]),
    )


def candidate_is_eligible(item: Mapping[str, object]) -> bool:
    evaluation = item.get("evaluation")
    return (
        isinstance(evaluation, Mapping)
        and math.isfinite(evaluation_selection_score(evaluation))
        and math.isfinite(float(evaluation["fit_wsmrmse"]))
        and evaluation.get("two_sided_integrity_passed") is True
        and evaluation.get("finite_boundary_adequacy_passed") is True
        and evaluation.get("value_boundary_adequacy_passed", True) is True
        and (
            "structural_depth_fit" not in evaluation
            or (
                isinstance(evaluation.get("structural_depth_fit"), Mapping)
                and evaluation["structural_depth_fit"].get("passed") is True
            )
        )
        and not evaluation.get("errors")
    )


def ranked_survivor_trajectory(
    initial_candidate_count: int,
    survivor_caps: Sequence[int],
) -> tuple[int, ...]:
    """Return the promoted count after each sequential ranked-stage cap."""
    if initial_candidate_count <= 0:
        raise CalibrationError("ranked trajectory needs at least one candidate")
    remaining = initial_candidate_count
    promoted: list[int] = []
    for cap in survivor_caps:
        if cap <= 0:
            raise CalibrationError("ranked trajectory caps must be positive")
        remaining = min(remaining, cap)
        promoted.append(remaining)
    return tuple(promoted)


def ranked_policy_stage_survivors(
    stage_name: str,
    eligible: Sequence[Mapping[str, object]],
    survivor_count: int,
    required_depth_participations: Sequence[float] | None = None,
    required_thresholds_bps: Sequence[float] | None = None,
) -> list[Mapping[str, object]]:
    """Promote the complete eligible value grid through short-horizon stages.

    The disabled policy is part of the scientific question, not merely one
    more grid point.  Every configured threshold/depth combination and the
    disabled policy must reach the full-day stage, so a five-minute or one-hour
    prefix cannot choose either behavioural dimension.  Full-day selection
    remains purely training based and may select baseline or treatment.
    """
    if survivor_count <= 0:
        raise CalibrationError("ranked policy survivor count must be positive")
    ranked = sorted(eligible, key=candidate_selection_key)
    if stage_name not in {
        "stage1_screen", "stage2_refinement", "stage3_full",
    }:
        raise CalibrationError(f"unknown ranked policy stage: {stage_name}")
    value_candidates = [
        item for item in ranked
        if isinstance(item.get("candidate"), Candidate)
    ]
    if value_candidates and len(value_candidates) != len(ranked):
        raise CalibrationError(
            "ranked policy stage mixes value and non-value candidates"
        )
    if value_candidates:
        if (required_depth_participations is None
                or required_thresholds_bps is None):
            raise CalibrationError(
                f"{stage_name} requires the complete configured value-policy "
                "threshold and depth grid"
            )
        configured_depths = tuple(sorted(set(
            float(value) for value in required_depth_participations
        )))
        configured_thresholds = tuple(sorted(set(
            float(value) for value in required_thresholds_bps
        )))
        if not configured_depths or not configured_thresholds:
            raise CalibrationError(
                f"{stage_name} has an empty configured value-policy grid"
            )

        expected_grid = {
            (threshold, depth)
            for threshold in configured_thresholds
            for depth in configured_depths
        }
        observed_grid: dict[tuple[float, float], Mapping[str, object]] = {}
        unexpected: list[tuple[float, float]] = []
        for item in ranked:
            candidate = item["candidate"]
            assert isinstance(candidate, Candidate)
            if not candidate.enabled:
                continue
            threshold_matches = [
                threshold for threshold in configured_thresholds
                if math.isclose(
                    candidate.threshold_bps, threshold,
                    rel_tol=0.0, abs_tol=1.0e-15,
                )
            ]
            depth_matches = [
                depth for depth in configured_depths
                if math.isclose(
                    candidate.depth_participation, depth,
                    rel_tol=0.0, abs_tol=1.0e-15,
                )
            ]
            if len(threshold_matches) != 1 or len(depth_matches) != 1:
                unexpected.append((
                    candidate.threshold_bps, candidate.depth_participation,
                ))
                continue
            key = (threshold_matches[0], depth_matches[0])
            if key in observed_grid:
                raise CalibrationError(
                    f"{stage_name} has duplicate eligible value-policy "
                    f"candidate for threshold/depth {key}"
                )
            observed_grid[key] = item
        missing = sorted(expected_grid - set(observed_grid))
        if unexpected or missing:
            raise CalibrationError(
                f"{stage_name} eligible value-policy grid is incomplete; "
                f"missing={missing}, unexpected={sorted(unexpected)}"
            )
        # Eligibility is structural/integrity screening only at the short
        # horizons. Fit scores remain diagnostic and cannot prune the grid.
        # Stage 3 first verifies that the same complete grid survived its
        # full-day checks, then uses the declared training fit to select one.
        promoted = (
            list(ranked[:1])
            if stage_name == "stage3_full" else list(ranked)
        )
    else:
        promoted = list(ranked[:min(survivor_count, len(ranked))])
    baselines = [
        item for item in ranked
        if getattr(item.get("candidate"), "enabled", None) is False
    ]
    baseline = baselines[0] if len(baselines) == 1 else None
    if value_candidates and len(baselines) != 1:
        raise CalibrationError(
            f"{stage_name} requires exactly one eligible disabled value-policy "
            f"baseline; observed {len(baselines)}"
        )
    if (stage_name != "stage3_full"
            and baseline is not None and baseline not in promoted):
        if value_candidates:
            promoted.append(baseline)
        elif promoted:
            promoted[-1] = baseline
        else:
            promoted.append(baseline)
        promoted.sort(key=candidate_selection_key)
    return promoted


def select_local_flow_stage_survivors(
    stage_name: str,
    evaluated: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Apply the integrity-first local-flow promotion protocol.

    Stage-1 and Stage-2 fit scores order diagnostics only: every candidate
    satisfying the existing structural/integrity predicate advances.  The
    full-day stage is the first point at which fit selects a single winner.
    Sorting here makes the policy independent of caller insertion order.
    """
    eligible = sorted(
        (item for item in evaluated if candidate_is_eligible(item)),
        key=candidate_selection_key,
    )
    if stage_name in {"stage1_screen", "stage2_refinement"}:
        return eligible
    if stage_name == "stage3_full":
        return eligible[:1]
    raise CalibrationError(f"unknown local-flow calibration stage: {stage_name}")


def command_for_run(*,
                    launcher: Sequence[str],
                    binary: pathlib.Path,
                    config: pathlib.Path,
                    policy: pathlib.Path | None,
                    summary: pathlib.Path,
                    duration: int,
                    seed: int,
                    local_controls: LocalFlowCandidate,
                    shared_quote_multiplier: float | None,
                    enable_shared_mm: bool,
                    enable_value_agents: bool) -> list[str]:
    """Build the exact one-rank fragmented simulator command."""
    command = [
        *launcher,
        str(binary),
        "--duration-seconds", str(duration),
        "--seed", str(seed),
        "--universe-config", str(config),
        "--window-ms", format(DECISION_WINDOW_MS, ".17g"),
        "--asset-summary-interval-ms", format(DECISION_WINDOW_MS, ".17g"),
        "--hawkes-activity-scale", str(local_controls.hawkes_activity_scale),
        "--local-mm-interval-ms", str(local_controls.local_mm_interval_ms),
        "--local-mm-quantity-multiplier",
        str(local_controls.local_mm_quantity_multiplier),
        "--local-mm-improvement-probability",
        str(local_controls.local_mm_improvement_probability),
        "--asset-summary-csv", str(summary),
    ]
    if not local_controls.local_mm_enabled:
        command.append("--disable-local-mm")
    if enable_shared_mm:
        if shared_quote_multiplier is None or shared_quote_multiplier <= 0.0:
            raise CalibrationError(
                "a positive relative shared quote multiplier is required when shared MM is enabled"
            )
        command.extend([
            "--shared-quote-relative",
            "--shared-quote-multiplier", str(shared_quote_multiplier),
            # Ordinary quote supply is calibrated with phi fixed at one.
            # Capacity scenarios are applied only after Q_S is frozen.
            "--uncoupled-shared-mm",
        ])
    else:
        command.append("--disable-shared-mm")
    if not enable_value_agents:
        command.append("--disable-value-agent")
    elif policy is not None:
        command.extend(["--value-agent-policy-csv", str(policy)])
    else:
        raise CalibrationError("a policy CSV is required when value agents are enabled")
    return command


def run_model(*,
              launcher: Sequence[str],
              binary: pathlib.Path,
              config: pathlib.Path,
              policy: pathlib.Path | None,
              output_dir: pathlib.Path,
              duration: int,
              seed: int,
              local_controls: LocalFlowCandidate,
              shared_quote_multiplier: float | None,
              enable_shared_mm: bool,
              enable_value_agents: bool,
              timeout_seconds: float) -> tuple[pathlib.Path, float]:
    """Run one deterministic replicate and retain full command diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "fragmented_asset_summary.csv"
    run_log = output_dir / "run.log"

    # A rerun may intentionally reuse a diagnostic directory.  Reusing any
    # simulator output from the preceding attempt would, however, allow a
    # launcher that exits zero without starting the simulator to be scored as
    # a fresh run.  Revoke those artifacts before launch.  Directories at the
    # terminal-file paths are rejected rather than removed recursively.
    for stale_path in (summary, run_log):
        try:
            stale_mode = stale_path.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(stale_mode):
            raise RuntimeError(
                f"run artifact path is a directory: {stale_path}"
            )
        stale_path.unlink()

    command = command_for_run(
        launcher=launcher, binary=binary, config=config, policy=policy,
        summary=summary, duration=duration, seed=seed,
        local_controls=local_controls,
        shared_quote_multiplier=shared_quote_multiplier,
        enable_shared_mm=enable_shared_mm,
        enable_value_agents=enable_value_agents,
    )
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        captured_output, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        elapsed = time.monotonic() - started
        # Terminate the complete mpirun process group.  Killing only the
        # launcher can leave simulator children consuming an allocation.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            trailing_output, _ = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            trailing_output, _ = process.communicate()
        timeout_output = error.output or ""
        if isinstance(timeout_output, bytes):
            timeout_output = timeout_output.decode("utf-8", errors="replace")
        captured_output = timeout_output + (trailing_output or "")
        with run_log.open("w", encoding="utf-8") as output:
            output.write("command=" + json.dumps(command) + "\n")
            output.write(f"wall_seconds_external={elapsed:.9f}\n")
            output.write(f"return_code={process.returncode}\n")
            output.write("TIMEOUT\n")
            output.write(captured_output)
        raise RuntimeError(
            f"simulator timed out after {timeout_seconds} seconds; see {run_log}"
        ) from error
    elapsed = time.monotonic() - started
    with run_log.open("w", encoding="utf-8") as output:
        output.write("command=" + json.dumps(command) + "\n")
        output.write(f"wall_seconds_external={elapsed:.9f}\n")
        output.write(f"return_code={process.returncode}\n")
        output.write(captured_output or "")
    if process.returncode != 0:
        raise RuntimeError(
            f"simulator returned status {process.returncode}; see {run_log}"
        )
    try:
        summary_stat = summary.lstat()
    except FileNotFoundError:
        summary_stat = None
    if (summary_stat is None
            or not stat.S_ISREG(summary_stat.st_mode)
            or summary_stat.st_size <= 0):
        raise RuntimeError(
            "simulator completed without a fresh, non-empty regular asset "
            f"summary {summary}; see {run_log}"
        )
    return summary, elapsed


def evaluate_policy(
    *,
    launcher: Sequence[str],
    binary: pathlib.Path,
    config: pathlib.Path,
    policy: pathlib.Path | None,
    symbols: Sequence[str],
    output_dir: pathlib.Path,
    duration: int,
    seeds: Sequence[int],
    targets: Mapping[str, Mapping[str, TargetMoment]],
    local_controls: LocalFlowCandidate,
    shared_quote_multiplier: float | None,
    enable_shared_mm: bool,
    enable_value_agents: bool,
    metrics: Sequence[str] = METRICS,
    timeout_seconds: float,
) -> dict[str, object]:
    """Run all common-random-number seeds and score their joint seed mean."""
    sample_numerator = duration * 1000
    if sample_numerator % int(DECISION_WINDOW_MS) != 0:
        raise CalibrationError(
            "duration does not contain an integer number of fixed-clock samples"
        )
    required_expected_sample_count = (
        sample_numerator // int(DECISION_WINDOW_MS)
    )
    summaries: list[pathlib.Path] = []
    elapsed: list[float | None] = []
    errors: list[str] = []
    for seed in seeds:
        try:
            summary, seconds = run_model(
                launcher=launcher, binary=binary, config=config, policy=policy,
                output_dir=output_dir / f"seed_{seed}", duration=duration,
                seed=seed, local_controls=local_controls,
                shared_quote_multiplier=shared_quote_multiplier,
                enable_shared_mm=enable_shared_mm,
                enable_value_agents=enable_value_agents,
                timeout_seconds=timeout_seconds,
            )
            summaries.append(summary)
            elapsed.append(seconds)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            elapsed.append(None)
            errors.append(str(error))
    if errors or len(summaries) != len(seeds):
        return {
            "fit_wsmrmse": math.inf,
            "combined_uncertainty_wsmrmse": math.inf,
            "selection_score": math.inf,
            "selection_metric_scores": [],
            "asset_summary_interval_ms": DECISION_WINDOW_MS,
            "required_expected_sample_count": required_expected_sample_count,
            "two_sided_integrity_passed": False,
            "two_sided_integrity_failures": [],
            "finite_boundary_adequacy_passed": False,
            "finite_boundary_adequacy": None,
            "value_boundary_adequacy_passed": False,
            "value_boundary_adequacy": None,
            "seed_wall_seconds": elapsed,
            "summary_paths": [str(path) for path in summaries],
            "errors": errors,
            "moment_estimates": [],
        }
    try:
        fit, estimates = weighted_moment_loss(
            summaries, targets, symbols, metrics=metrics,
            required_expected_sample_count=required_expected_sample_count,
        )
        combined, _ = weighted_moment_loss(
            summaries, targets, symbols, uncertainty_mode="combined", metrics=metrics,
            required_expected_sample_count=required_expected_sample_count,
        )
        selection_score, selection_metric_scores = metric_balanced_robust_loss(
            estimates, metrics=metrics,
        )
        two_sided_passed, two_sided_failures = two_sided_execution_integrity(
            summaries, symbols,
            required_expected_sample_count=required_expected_sample_count,
        )
        boundary_adequacy = finite_boundary_adequacy(
            summaries, symbols,
            required_expected_sample_count=required_expected_sample_count,
        )
        value_adequacy = value_boundary_adequacy(
            summaries, symbols,
            required_expected_sample_count=required_expected_sample_count,
        )
    except (CalibrationError, OSError, ValueError) as error:
        return {
            "fit_wsmrmse": math.inf,
            "combined_uncertainty_wsmrmse": math.inf,
            "selection_score": math.inf,
            "selection_metric_scores": [],
            "asset_summary_interval_ms": DECISION_WINDOW_MS,
            "required_expected_sample_count": required_expected_sample_count,
            "two_sided_integrity_passed": False,
            "two_sided_integrity_failures": [],
            "finite_boundary_adequacy_passed": False,
            "finite_boundary_adequacy": None,
            "value_boundary_adequacy_passed": False,
            "value_boundary_adequacy": None,
            "seed_wall_seconds": elapsed,
            "summary_paths": [str(path) for path in summaries],
            "errors": [str(error)],
            "moment_estimates": [],
        }
    return {
        "fit_wsmrmse": fit,
        "combined_uncertainty_wsmrmse": combined,
        "selection_score": selection_score,
        "selection_metric_scores": selection_metric_scores,
        "asset_summary_interval_ms": DECISION_WINDOW_MS,
        "required_expected_sample_count": required_expected_sample_count,
        "two_sided_integrity_passed": two_sided_passed,
        "two_sided_integrity_failures": two_sided_failures,
        "finite_boundary_adequacy_passed": bool(
            boundary_adequacy["passed"]
        ),
        "finite_boundary_adequacy": boundary_adequacy,
        "value_boundary_adequacy_passed": bool(value_adequacy["passed"]),
        "value_boundary_adequacy": value_adequacy,
        "seed_wall_seconds": elapsed,
        "summary_paths": [str(path) for path in summaries],
        "errors": [],
        "moment_estimates": [asdict(estimate) for estimate in estimates],
    }


def aggregate_training_day_evaluations(
    evaluations: Sequence[tuple[TrainingDay, Mapping[str, object]]],
    *,
    seed_count: int,
) -> dict[str, object]:
    """Robustly aggregate complete, metric-balanced day-level losses.

    Every day first receives its own metric-balanced robust score.  Candidate
    selection minimises the median daily score plus 0.25 times its median
    absolute deviation.  Raw diagonal WMM scores remain arithmetic diagnostics
    only.  Pooling residuals or concatenating target tables here would instead
    let days with different scales, symbols, or weights change the intended
    day-level contribution.
    """
    if not evaluations:
        raise CalibrationError("cannot aggregate zero training-day evaluations")
    if seed_count <= 0:
        raise CalibrationError("training-day aggregation requires a positive seed count")

    day_reports: list[dict[str, object]] = []
    errors: list[str] = []
    summary_paths: list[str] = []
    seed_wall_seconds: list[float | None] = []
    fitted: list[float] = []
    combined: list[float] = []
    selection_scores: list[float] = []
    two_sided_integrity_passed = True
    two_sided_integrity_failures: list[object] = []
    finite_boundary_adequacy_passed = True
    finite_boundary_adequacy_reports: list[object] = []
    value_boundary_adequacy_passed = True
    value_boundary_adequacy_reports: list[object] = []
    for training_day, evaluation in evaluations:
        fit = float(evaluation["fit_wsmrmse"])
        combined_fit = float(evaluation["combined_uncertainty_wsmrmse"])
        selection_score = evaluation_selection_score(evaluation)
        day_two_sided = evaluation.get("two_sided_integrity_passed") is True
        if not day_two_sided:
            two_sided_integrity_passed = False
            two_sided_integrity_failures.extend(
                evaluation.get("two_sided_integrity_failures", [])  # type: ignore[arg-type]
            )
        if evaluation.get("finite_boundary_adequacy_passed") is not True:
            finite_boundary_adequacy_passed = False
        finite_boundary_adequacy_reports.append({
            "date": training_day.date,
            "adequacy": evaluation.get("finite_boundary_adequacy"),
        })
        if evaluation.get("value_boundary_adequacy_passed") is not True:
            value_boundary_adequacy_passed = False
        value_boundary_adequacy_reports.append({
            "date": training_day.date,
            "adequacy": evaluation.get("value_boundary_adequacy"),
        })
        summary_paths.extend(str(path) for path in evaluation["summary_paths"])
        seed_wall_seconds.extend(evaluation["seed_wall_seconds"])  # type: ignore[arg-type]
        day_errors = [str(error) for error in evaluation["errors"]]
        if day_errors:
            errors.extend(f"{training_day.date}: {error}" for error in day_errors)
        if (not math.isfinite(fit) or not math.isfinite(combined_fit)
                or not math.isfinite(selection_score)):
            if not day_errors:
                errors.append(
                    f"{training_day.date}: non-finite day-level weighted moment loss"
                )
        else:
            fitted.append(fit)
            combined.append(combined_fit)
            selection_scores.append(selection_score)
        day_reports.append({
            "date": training_day.date,
            "evaluation": evaluation_report(evaluation),
        })
    if errors or len(fitted) != len(evaluations):
        return {
            "fit_wsmrmse": math.inf,
            "combined_uncertainty_wsmrmse": math.inf,
            "selection_score": math.inf,
            "selection_metric_scores": [],
            "two_sided_integrity_passed": False,
            "two_sided_integrity_failures": two_sided_integrity_failures,
            "finite_boundary_adequacy_passed": False,
            "finite_boundary_adequacy": finite_boundary_adequacy_reports,
            "value_boundary_adequacy_passed": False,
            "value_boundary_adequacy": value_boundary_adequacy_reports,
            "seed_count": seed_count,
            "training_day_count": len(evaluations),
            "aggregation": "median_plus_mad_of_day_level_metric_balanced_huber",
            "seed_wall_seconds": seed_wall_seconds,
            "summary_paths": summary_paths,
            "errors": errors,
            # Per-day estimates are retained inside ``training_day_evaluations``
            # rather than flattened without their day identity.
            "moment_estimates": [],
            "training_day_evaluations": day_reports,
        }
    median_selection = percentile(selection_scores, 0.5)
    selection_mad = percentile(
        [abs(score - median_selection) for score in selection_scores], 0.5,
    )
    return {
        "fit_wsmrmse": statistics.fmean(fitted),
        "combined_uncertainty_wsmrmse": statistics.fmean(combined),
        "selection_score": (
            median_selection + DEFAULT_DAY_STABILITY_PENALTY * selection_mad
        ),
        "selection_score_median": median_selection,
        "selection_score_mad": selection_mad,
        # Metric detail remains inside each dated evaluation; flattening it
        # would lose the explicitly robust day aggregation.
        "selection_metric_scores": [],
        "two_sided_integrity_passed": two_sided_integrity_passed,
        "two_sided_integrity_failures": two_sided_integrity_failures,
        "finite_boundary_adequacy_passed": finite_boundary_adequacy_passed,
        "finite_boundary_adequacy": finite_boundary_adequacy_reports,
        "value_boundary_adequacy_passed": value_boundary_adequacy_passed,
        "value_boundary_adequacy": value_boundary_adequacy_reports,
        "seed_count": seed_count,
        "training_day_count": len(evaluations),
        "aggregation": "median_plus_mad_of_day_level_metric_balanced_huber",
        "seed_wall_seconds": seed_wall_seconds,
        "summary_paths": summary_paths,
        "errors": [],
        "moment_estimates": [],
        "training_day_evaluations": day_reports,
    }


def cluster_training_boundary_adequacy(
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Gate only value-agent boundary contacts for cluster-policy fitting.

    Background-flow stability is calibrated and certified in Block 1.  The
    policy block must therefore measure whether *value orders* depend on the
    finite reserve, using value-order counts and requested value quantity as
    its denominators.  Seeds are pooled within each symbol/day and then over
    the complete candidate.  A three-symbol representative subset is not a
    market-wide aggregate, so the transparent 5% symbol limit is used for the
    final candidate pool; the 1% market-wide limit remains a validation gate.
    """
    base: dict[str, object] = {
        "schema_version": 2,
        "source": "value",
        "passed": False,
        "scope": (
            "pooled_across_training_dates_and_stage_seeds_for_"
            "three_symbol_cluster_candidate"
        ),
        "background_gate_role": "separate_block1_diagnostic",
        "per_seed_ratio_role": "diagnostic_only",
        "symbol_day_seed_pool_gate_retained": True,
        "development_validation_gate_unchanged": True,
        "thresholds": {
            "maximum_symbol_day_seed_pool_event_ratio": (
                CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_EVENT_RATIO
            ),
            "maximum_symbol_day_seed_pool_quantity_ratio": (
                CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_QUANTITY_RATIO
            ),
            "maximum_candidate_pool_event_ratio": (
                CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_EVENT_RATIO
            ),
            "maximum_candidate_pool_quantity_ratio": (
                CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_QUANTITY_RATIO
            ),
        },
        "pooled_aggregate": None,
        "symbol_day_seed_pool_failures": [],
        "pooled_aggregate_failures": [],
        "diagnostic_per_seed_failures": [],
        "diagnostic_day_aggregate_failures": [],
        "error": None,
    }
    try:
        raw_evidence = evaluation.get("value_boundary_adequacy")
        if isinstance(raw_evidence, Mapping):
            raw_reports: Sequence[object] = ({
                "date": "single_training_day",
                "adequacy": raw_evidence,
            },)
        elif (isinstance(raw_evidence, Sequence)
                and not isinstance(raw_evidence, (str, bytes))):
            raw_reports = raw_evidence
        else:
            raw_reports = ()
        if not raw_reports:
            raise CalibrationError(
                "cluster training value-boundary evidence is absent or malformed"
            )

        event_numerator = 0
        event_denominator = 0
        quantity_numerator = 0
        quantity_denominator = 0
        symbol_day_failures: list[dict[str, object]] = []
        diagnostic_seed_failures: list[dict[str, object]] = []
        diagnostic_day_failures: list[dict[str, object]] = []
        run_count = 0

        for raw_day in raw_reports:
            if not isinstance(raw_day, Mapping):
                raise CalibrationError(
                    "cluster training value-boundary day record is malformed"
                )
            training_date = str(raw_day.get("date", ""))
            adequacy = raw_day.get("adequacy")
            if not isinstance(adequacy, Mapping):
                raise CalibrationError(
                    f"cluster training value-boundary report is absent for {training_date}"
                )
            raw_failures = adequacy.get("failures", [])
            raw_seed_failures = adequacy.get("diagnostic_per_seed_failures", [])
            aggregate = adequacy.get("aggregate_pooled")
            if (not isinstance(raw_failures, Sequence)
                    or isinstance(raw_failures, (str, bytes))
                    or not isinstance(raw_seed_failures, Sequence)
                    or isinstance(raw_seed_failures, (str, bytes))
                    or not isinstance(aggregate, Mapping)):
                raise CalibrationError(
                    f"cluster training value-boundary report is malformed for {training_date}"
                )
            for raw_failure in raw_failures:
                if not isinstance(raw_failure, Mapping):
                    raise CalibrationError(
                        "cluster training value-boundary failure record is malformed"
                    )
                failure = {"training_date": training_date, **dict(raw_failure)}
                if failure.get("scope") == "symbol_seed_pool":
                    symbol_day_failures.append(failure)
                elif failure.get("scope") == "aggregate_seed_pool":
                    diagnostic_day_failures.append(failure)
                else:
                    raise CalibrationError(
                        "cluster training value-boundary failure has unknown scope"
                    )
            for raw_failure in raw_seed_failures:
                if not isinstance(raw_failure, Mapping):
                    raise CalibrationError(
                        "cluster training per-seed diagnostic is malformed"
                    )
                diagnostic_seed_failures.append({
                    "training_date": training_date, **dict(raw_failure),
                })
            event_numerator += int(aggregate["boundary_truncation_events"])
            event_denominator += int(aggregate["source_event_count"])
            quantity_numerator += int(aggregate["boundary_truncated_quantity"])
            quantity_denominator += int(aggregate["source_requested_quantity"])
            run_count += int(aggregate.get("run_count", 0))

        event_ratio, event_denominator_valid = _finite_boundary_ratio(
            event_numerator, event_denominator,
        )
        quantity_ratio, quantity_denominator_valid = _finite_boundary_ratio(
            quantity_numerator, quantity_denominator,
        )
        pooled = {
            "run_count": run_count,
            "source": "value",
            "boundary_truncation_events": event_numerator,
            "value_order_count": event_denominator,
            "boundary_event_ratio": event_ratio,
            "boundary_truncated_quantity": quantity_numerator,
            "value_requested_quantity": quantity_denominator,
            "boundary_quantity_ratio": quantity_ratio,
        }
        pooled_failures: list[dict[str, object]] = []
        for metric, ratio, valid, maximum, numerator, denominator in (
            (
                "boundary_event_ratio", event_ratio, event_denominator_valid,
                CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_EVENT_RATIO,
                event_numerator, event_denominator,
            ),
            (
                "boundary_quantity_ratio", quantity_ratio,
                quantity_denominator_valid,
                CERTIFICATION_MAXIMUM_ASSET_BOUNDARY_QUANTITY_RATIO,
                quantity_numerator, quantity_denominator,
            ),
        ):
            if (not valid or ratio is None
                    or ratio > maximum + 1.0e-15):
                pooled_failures.append({
                    "scope": "candidate_training_aggregate",
                    "metric": metric,
                    "numerator": numerator,
                    "denominator": denominator,
                    "ratio": ratio,
                    "maximum": maximum,
                })
        base.update({
            "passed": (
                run_count > 0
                and not symbol_day_failures
                and not pooled_failures
            ),
            "pooled_aggregate": pooled,
            "symbol_day_seed_pool_failures": symbol_day_failures,
            "pooled_aggregate_failures": pooled_failures,
            "diagnostic_per_seed_failures": diagnostic_seed_failures,
            "diagnostic_day_aggregate_failures": diagnostic_day_failures,
        })
    except (CalibrationError, KeyError, TypeError, ValueError) as error:
        base["error"] = str(error)
    return base


def evaluate_policy_across_training_days(
    *,
    launcher: Sequence[str],
    binary: pathlib.Path,
    training_days: Sequence[TrainingDay],
    configs_by_date: Mapping[str, pathlib.Path],
    policy: pathlib.Path | None,
    symbols: Sequence[str],
    output_dir: pathlib.Path,
    duration: int,
    seeds: Sequence[int],
    targets_by_date: Mapping[str, Mapping[str, Mapping[str, TargetMoment]]],
    local_controls: LocalFlowCandidate,
    shared_quote_multiplier: float | None,
    enable_shared_mm: bool,
    enable_value_agents: bool,
    metrics: Sequence[str] = METRICS,
    timeout_seconds: float,
) -> dict[str, object]:
    """Evaluate a fixed candidate on every training day before selection."""
    if not training_days:
        raise CalibrationError("candidate evaluation requires at least one training day")
    if len(training_days) == 1:
        # Exact legacy result shape and candidate/seed directory layout.
        training_day = training_days[0]
        try:
            config = configs_by_date[training_day.date]
            targets = targets_by_date[training_day.date]
        except KeyError as error:
            raise CalibrationError(
                f"missing configuration or targets for training day {training_day.date}"
            ) from error
        return evaluate_policy(
            launcher=launcher,
            binary=binary,
            config=config,
            policy=policy,
            symbols=symbols,
            output_dir=output_dir,
            duration=duration,
            seeds=seeds,
            targets=targets,
            local_controls=local_controls,
            shared_quote_multiplier=shared_quote_multiplier,
            enable_shared_mm=enable_shared_mm,
            enable_value_agents=enable_value_agents,
            metrics=metrics,
            timeout_seconds=timeout_seconds,
        )
    day_evaluations: list[tuple[TrainingDay, Mapping[str, object]]] = []
    for training_day in training_days:
        try:
            config = configs_by_date[training_day.date]
            targets = targets_by_date[training_day.date]
        except KeyError as error:
            raise CalibrationError(
                f"missing configuration or targets for training day {training_day.date}"
            ) from error
        evaluation = evaluate_policy(
            launcher=launcher,
            binary=binary,
            config=config,
            policy=policy,
            symbols=symbols,
            output_dir=output_dir / training_day.identifier,
            duration=duration,
            seeds=seeds,
            targets=targets,
            local_controls=local_controls,
            shared_quote_multiplier=shared_quote_multiplier,
            enable_shared_mm=enable_shared_mm,
            enable_value_agents=enable_value_agents,
            metrics=metrics,
            timeout_seconds=timeout_seconds,
        )
        day_evaluations.append((training_day, evaluation))
    return aggregate_training_day_evaluations(
        day_evaluations, seed_count=len(seeds),
    )


def evaluation_report(evaluation: Mapping[str, object]) -> dict[str, object]:
    """Compact JSON-safe view of an evaluation result."""
    result: dict[str, object] = {
        "fit_wsmrmse": float(evaluation["fit_wsmrmse"]),
        "combined_uncertainty_wsmrmse": float(
            evaluation["combined_uncertainty_wsmrmse"]
        ),
        "selection_score": evaluation_selection_score(evaluation),
        "selection_metric_scores": evaluation.get("selection_metric_scores", []),
        "two_sided_integrity_passed": (
            evaluation.get("two_sided_integrity_passed") is True
        ),
        "two_sided_integrity_failures": evaluation.get(
            "two_sided_integrity_failures", [],
        ),
        "finite_boundary_adequacy_passed": (
            evaluation.get("finite_boundary_adequacy_passed") is True
        ),
        "finite_boundary_adequacy": evaluation.get(
            "finite_boundary_adequacy"
        ),
        "value_boundary_adequacy_passed": (
            evaluation.get("value_boundary_adequacy_passed", True) is True
        ),
        "value_boundary_adequacy": evaluation.get(
            "value_boundary_adequacy"
        ),
        "seed_count": int(evaluation.get(
            "seed_count", len(evaluation["summary_paths"]),
        )),
        "seed_wall_seconds": evaluation["seed_wall_seconds"],
        "summary_paths": evaluation["summary_paths"],
        "errors": evaluation["errors"],
        "moment_estimates": evaluation["moment_estimates"],
    }
    if "training_day_evaluations" in evaluation:
        result.update({
            "training_day_count": int(evaluation["training_day_count"]),
            "aggregation": str(evaluation["aggregation"]),
            "training_day_evaluations": evaluation["training_day_evaluations"],
        })
        if "selection_score_median" in evaluation:
            result.update({
                "selection_score_median": float(evaluation["selection_score_median"]),
                "selection_score_mad": float(evaluation["selection_score_mad"]),
            })
    if "structural_depth_fit" in evaluation:
        result["structural_depth_fit"] = evaluation["structural_depth_fit"]
    if "background_finite_boundary_adequacy" in evaluation:
        result["background_finite_boundary_adequacy"] = evaluation[
            "background_finite_boundary_adequacy"
        ]
        result["background_finite_boundary_adequacy_passed"] = (
            evaluation.get("background_finite_boundary_adequacy_passed") is True
        )
    return result


def checkpoint_cluster_policy_records(
    selected_policy_rows: Sequence[Mapping[str, object]],
    selected_training_evaluations: Mapping[int, Mapping[str, object]],
    *,
    expected_summary_count_per_cluster: int,
) -> list[dict[str, object]]:
    """Bind every selected cluster policy to its Stage-3 training evidence.

    ``cluster_selected_policies.csv`` intentionally remains a compact table of
    scalar controls.  The pre-validation JSON checkpoint has a different
    evidentiary role: an independent verifier must be able to trace each
    selected policy back to the full-session summaries used to select it.
    Keeping this construction in one tested helper prevents the producer and
    verifier schemas from silently diverging after an expensive validation
    campaign.
    """
    if expected_summary_count_per_cluster <= 0:
        raise CalibrationError(
            "selected cluster policies require a positive Stage-3 summary count"
        )
    records: list[dict[str, object]] = []
    observed_cluster_ids: set[int] = set()
    for index, row in enumerate(selected_policy_rows):
        try:
            cluster_id = int(row["cluster_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise CalibrationError(
                f"selected cluster policy row {index} has an invalid cluster_id"
            ) from error
        if cluster_id in observed_cluster_ids:
            raise CalibrationError(
                f"selected cluster policy rows repeat cluster {cluster_id}"
            )
        observed_cluster_ids.add(cluster_id)
        try:
            training_evaluation = selected_training_evaluations[cluster_id]
        except KeyError as error:
            raise CalibrationError(
                f"selected cluster {cluster_id} has no Stage-3 training evaluation"
            ) from error
        report = evaluation_report(training_evaluation)
        raw_paths = report.get("summary_paths")
        if not isinstance(raw_paths, Sequence) or isinstance(
            raw_paths, (str, bytes, bytearray)
        ):
            raise CalibrationError(
                f"selected cluster {cluster_id} has no Stage-3 summary paths"
            )
        paths = [pathlib.Path(str(path)).expanduser() for path in raw_paths]
        if len(paths) != expected_summary_count_per_cluster:
            raise CalibrationError(
                f"selected cluster {cluster_id} Stage-3 summary count differs "
                f"from the declared protocol: expected="
                f"{expected_summary_count_per_cluster} observed={len(paths)}"
            )
        canonical_paths = [path.resolve(strict=True) for path in paths]
        if len(set(canonical_paths)) != len(canonical_paths):
            raise CalibrationError(
                f"selected cluster {cluster_id} reuses a Stage-3 summary path"
            )
        if any("heldout" in path.as_posix().lower() for path in canonical_paths):
            raise CalibrationError(
                f"selected cluster {cluster_id} references held-out evidence"
            )
        records.append({**dict(row), "training_evaluation": report})
    if set(selected_training_evaluations) != observed_cluster_ids:
        missing = sorted(set(selected_training_evaluations) - observed_cluster_ids)
        extra = sorted(observed_cluster_ids - set(selected_training_evaluations))
        raise CalibrationError(
            "selected cluster policy/evaluation keys differ: "
            f"missing_rows={missing} missing_evaluations={extra}"
        )
    return records


def candidate_eligibility_diagnostics(
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Expose every fail-closed eligibility predicate without changing it."""
    fit = float(evaluation["fit_wsmrmse"])
    selection_score = evaluation_selection_score(evaluation)
    errors = [str(error) for error in evaluation.get("errors", [])]
    predicates = {
        "finite_selection_score": math.isfinite(selection_score),
        "finite_fit_wsmrmse": math.isfinite(fit),
        "two_sided_integrity_passed": (
            evaluation.get("two_sided_integrity_passed") is True
        ),
        "finite_boundary_adequacy_passed": (
            evaluation.get("finite_boundary_adequacy_passed") is True
        ),
        "value_boundary_adequacy_passed": (
            evaluation.get("value_boundary_adequacy_passed", True) is True
        ),
        "error_free": not errors,
    }
    if "structural_depth_fit" in evaluation:
        structural_depth_fit = evaluation.get("structural_depth_fit")
        predicates["structural_depth_fit_passed"] = (
            isinstance(structural_depth_fit, Mapping)
            and structural_depth_fit.get("passed") is True
        )
    return {
        "eligible": all(predicates.values()),
        "predicates": predicates,
        "errors": errors,
    }


def invalidate_terminal_calibration_artifacts(
    output_root: pathlib.Path,
    *,
    overwrite: bool,
) -> tuple[pathlib.Path, ...]:
    """Revoke terminal results before beginning an overwrite attempt.

    The certified handoff is removed first.  Therefore even if a later path is
    malformed or cannot be removed, a failed rerun cannot leave an older
    handoff looking authoritative.  Directories and diagnostic checkpoints are
    never removed recursively.
    """
    if not overwrite:
        return ()
    removed: list[pathlib.Path] = []
    for filename in TERMINAL_CALIBRATION_ARTIFACT_FILENAMES:
        path = output_root / filename
        if path.is_symlink() or path.is_file():
            path.unlink()
            removed.append(path)
        elif path.exists():
            raise CalibrationError(
                "refusing overwrite because terminal calibration artifact is "
                f"not a regular file: {path}"
            )
    return tuple(removed)


def initialize_calibration_progress(
    path: pathlib.Path,
    *,
    overwrite: bool,
) -> dict[str, object]:
    """Create the mutable, atomic audit trail before expensive simulation."""
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_role": "calibration_progress_checkpoint",
        "status": "running",
        "event_count": 0,
        "events": [],
    }
    atomic_json(path, payload, overwrite=overwrite)
    return payload


def append_calibration_progress(
    path: pathlib.Path,
    event: Mapping[str, object],
    *,
    status: str = "running",
) -> dict[str, object]:
    """Append one compact event and atomically replace the live checkpoint."""
    if path.is_file():
        payload = json_object(path, label="calibration progress checkpoint")
    else:
        payload = {
            "schema_version": 1,
            "artifact_role": "calibration_progress_checkpoint",
            "status": "running",
            "event_count": 0,
            "events": [],
        }
    if (payload.get("schema_version") != 1
            or payload.get("artifact_role")
                != "calibration_progress_checkpoint"):
        raise CalibrationError("invalid calibration progress checkpoint")
    events = payload.get("events")
    if not isinstance(events, list):
        raise CalibrationError("calibration progress checkpoint has invalid events")
    compact_event = dict(event)
    events.append(compact_event)
    payload.update({
        "status": status,
        "event_count": len(events),
        "last_event": compact_event,
    })
    atomic_json(path, payload, overwrite=True)
    return payload


def persist_candidate_evaluation(
    candidate_dir: pathlib.Path,
    *,
    block: str,
    stage: str,
    cluster_id: int | None,
    candidate_index: int,
    candidate: Candidate | LocalFlowCandidate | SharedQuoteCandidate,
    evaluation: Mapping[str, object],
    progress_path: pathlib.Path,
    overwrite: bool,
) -> dict[str, object]:
    """Atomically retain a complete candidate result before stage selection."""
    eligibility = candidate_eligibility_diagnostics(evaluation)
    path = candidate_dir / "candidate_evaluation.json"
    payload = {
        "schema_version": 1,
        "artifact_role": "calibration_candidate_evaluation",
        "block": block,
        "stage": stage,
        "cluster_id": cluster_id,
        "candidate_index": candidate_index,
        "candidate": asdict(candidate),
        "eligibility": eligibility,
        "evaluation": evaluation_report(evaluation),
    }
    atomic_json(path, payload, overwrite=overwrite)
    reference = {
        "block": block,
        "stage": stage,
        "cluster_id": cluster_id,
        "candidate_index": candidate_index,
        "candidate_label": candidate.label,
        "eligible": eligibility["eligible"],
        "failed_predicates": [
            name for name, passed in eligibility["predicates"].items()
            if not passed
        ],
        "path": str(path),
        "sha256": sha256_file(path),
    }
    append_calibration_progress(
        progress_path,
        {"kind": "candidate_evaluation", **reference},
    )
    return reference


def persist_stage_checkpoint(
    stage_root: pathlib.Path,
    *,
    block: str,
    stage: str,
    cluster_id: int | None,
    candidate_references: Sequence[Mapping[str, object]],
    promoted_candidate_indices: Sequence[int],
    configured_ranked_survivor_count: int,
    progress_path: pathlib.Path,
    overwrite: bool,
) -> dict[str, object]:
    """Persist observed eligibility/promotion counts before advancing or aborting."""
    eligible_indices = [
        int(reference["candidate_index"])
        for reference in candidate_references
        if reference.get("eligible") is True
    ]
    promoted_indices = [int(index) for index in promoted_candidate_indices]
    counts = {
        "evaluated_candidates": len(candidate_references),
        "eligible_candidates": len(eligible_indices),
        "promoted_candidates": len(promoted_indices),
        "configured_ranked_survivor_count": configured_ranked_survivor_count,
    }
    status = "complete" if promoted_indices else "failed_no_eligible_candidates"
    path = stage_root / "stage_checkpoint.json"
    payload = {
        "schema_version": 1,
        "artifact_role": "calibration_stage_checkpoint",
        "status": status,
        "block": block,
        "stage": stage,
        "cluster_id": cluster_id,
        "observed_counts": counts,
        "eligible_candidate_indices": eligible_indices,
        "promoted_candidate_indices": promoted_indices,
        "candidate_evaluations": list(candidate_references),
    }
    atomic_json(path, payload, overwrite=overwrite)
    reference = {
        "kind": "stage_checkpoint",
        "block": block,
        "stage": stage,
        "cluster_id": cluster_id,
        "status": status,
        "observed_counts": counts,
        "path": str(path),
        "sha256": sha256_file(path),
    }
    append_calibration_progress(
        progress_path,
        reference,
        status="running" if promoted_indices else "failed",
    )
    return payload


def persist_calibration_failure(
    output_root: pathlib.Path,
    error: BaseException,
) -> pathlib.Path:
    """Write the terminal failure record without masking the original error."""
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "calibration_progress.json"
    progress: Mapping[str, object] | None = None
    if progress_path.is_file():
        try:
            progress = append_calibration_progress(
                progress_path,
                {
                    "kind": "calibration_failure",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                },
                status="failed",
            )
        except (CalibrationError, OSError, ValueError, json.JSONDecodeError):
            progress = None
    failure_path = output_root / "calibration_failure.json"
    atomic_json(failure_path, {
        "schema_version": 1,
        "artifact_role": "calibration_failure",
        "status": "failed",
        "exception_type": type(error).__name__,
        "message": str(error),
        "progress_checkpoint": (
            {
                "path": str(progress_path),
                "sha256": sha256_file(progress_path),
                "snapshot": progress,
            }
            if progress is not None and progress_path.is_file() else None
        ),
    }, overwrite=True)
    return failure_path


def candidate_policy_for_cluster(cluster_id: int,
                                 candidate: Candidate) -> dict[int, Candidate]:
    return {cluster_id: candidate}


def stage_detail_rows(stage: str,
                      cluster_id: int,
                      candidate_index: int,
                      candidate: Candidate | LocalFlowCandidate | SharedQuoteCandidate,
                      evaluation: Mapping[str, object],
                      *,
                      local_controls: LocalFlowCandidate | None = None,
                      shared_quote: SharedQuoteCandidate | None = None) -> dict[str, object]:
    """Render one auditable row for any block-coordinate candidate."""
    value_candidate = candidate if isinstance(candidate, Candidate) else None
    local_candidate = (
        candidate if isinstance(candidate, LocalFlowCandidate) else local_controls
    )
    shared_candidate = (
        candidate if isinstance(candidate, SharedQuoteCandidate) else shared_quote
    )
    return {
        "phase": stage,
        "cluster_id": cluster_id,
        "cluster_label": (
            f"liquidity_{cluster_id:02d}" if cluster_id >= 0 else "global_representatives"
        ),
        "candidate_index": candidate_index,
        "candidate_label": candidate.label,
        "enabled": int(value_candidate.enabled) if value_candidate is not None else "",
        "value_threshold_bps": (
            format(value_candidate.threshold_bps, ".17g")
            if value_candidate is not None else ""
        ),
        "value_depth_participation": (
            value_candidate.depth_participation
            if value_candidate is not None else ""
        ),
        "hawkes_activity_scale": (
            local_candidate.hawkes_activity_scale if local_candidate is not None else ""
        ),
        "local_mm_enabled": (
            int(local_candidate.local_mm_enabled) if local_candidate is not None else ""
        ),
        "local_mm_interval_ms": (
            local_candidate.local_mm_interval_ms if local_candidate is not None else ""
        ),
        "local_mm_quantity_multiplier": (
            local_candidate.local_mm_quantity_multiplier
            if local_candidate is not None else ""
        ),
        "local_mm_improvement_probability": (
            local_candidate.local_mm_improvement_probability
            if local_candidate is not None else ""
        ),
        "shared_mm_enabled": (
            int(shared_candidate.enabled) if shared_candidate is not None else ""
        ),
        "shared_quote_multiplier": (
            shared_candidate.multiplier if shared_candidate is not None else ""
        ),
        "fit_wsmrmse": float(evaluation["fit_wsmrmse"]),
        "combined_uncertainty_wsmrmse": float(
            evaluation["combined_uncertainty_wsmrmse"]
        ),
        "selection_score": evaluation_selection_score(evaluation),
        "seed_count": int(evaluation.get(
            "seed_count", len(evaluation["summary_paths"]),
        )),
        "training_day_count": int(evaluation.get("training_day_count", 1)),
        "aggregation": str(evaluation.get(
            "aggregation", "single_day_wsmrmse",
        )),
        "errors": json.dumps(evaluation["errors"], sort_keys=True),
    }


def percentile(values: Sequence[float], probability: float) -> float:
    """Deterministic linear percentile without a NumPy dependency."""
    if not values:
        raise CalibrationError("cannot calculate a percentile of no values")
    if not 0.0 <= probability <= 1.0:
        raise CalibrationError("percentile probability must lie in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution_rows(estimates: Sequence[Mapping[str, object]],
                      *, scope: str) -> list[dict[str, object]]:
    """Summarise empirical versus simulated cross-sectional moment distributions."""
    grouped: dict[str, list[Mapping[str, object]]] = {metric: [] for metric in METRICS}
    for estimate in estimates:
        grouped[str(estimate["metric"])].append(estimate)
    result: list[dict[str, object]] = []
    for metric in METRICS:
        rows = grouped[metric]
        if not rows:
            continue
        targets = [float(row["target"]) for row in rows]
        simulations = [float(row["simulated_mean"]) for row in rows]
        if not all(math.isfinite(value) for value in [*targets, *simulations]):
            continue
        target_mean = statistics.fmean(targets)
        simulated_mean = statistics.fmean(simulations)
        target_median = percentile(targets, 0.5)
        simulated_median = percentile(simulations, 0.5)
        result.append({
            "scope": scope,
            "metric": metric,
            "symbol_count": len(rows),
            "target_mean": target_mean,
            "simulated_mean": simulated_mean,
            "target_median": target_median,
            "simulated_median": simulated_median,
            "target_p10": percentile(targets, 0.1),
            "simulated_p10": percentile(simulations, 0.1),
            "target_p90": percentile(targets, 0.9),
            "simulated_p90": percentile(simulations, 0.9),
            "mean_difference": simulated_mean - target_mean,
            "median_difference": simulated_median - target_median,
        })
    return result


def two_sided_coverage_shortfalls(
    evaluation: Mapping[str, object],
    maximum_shortfall: float,
) -> list[dict[str, object]]:
    """Return held-out books whose simulated coverage misses the real target."""
    failures: list[dict[str, object]] = []
    for raw in evaluation.get("moment_estimates", []):  # type: ignore[assignment]
        estimate = dict(raw)
        if estimate.get("metric") != "two_sided_sample_fraction":
            continue
        target = float(estimate["target"])
        simulated = float(estimate["simulated_mean"])
        shortfall = target - simulated
        if shortfall > maximum_shortfall + 1.0e-12:
            failures.append({
                "symbol": str(estimate["symbol"]),
                "empirical_two_sided_fraction": target,
                "simulated_two_sided_fraction": simulated,
                "shortfall": shortfall,
                "maximum_allowed_shortfall": maximum_shortfall,
            })
    return failures


def two_sided_coverage_summary(
    evaluation: Mapping[str, object],
    maximum_shortfall: float,
) -> dict[str, object]:
    """Summarise held-out coverage without imposing a family-wise gate."""
    rows: list[dict[str, float | str]] = []
    for raw in evaluation.get("moment_estimates", []):  # type: ignore[assignment]
        estimate = dict(raw)
        if estimate.get("metric") != "two_sided_sample_fraction":
            continue
        target = finite_float(
            estimate["target"], label="coverage summary empirical target",
        )
        simulated = finite_float(
            estimate["simulated_mean"], label="coverage summary simulated mean",
        )
        rows.append({
            "symbol": str(estimate["symbol"]),
            "target": target,
            "simulated": simulated,
            "shortfall": target - simulated,
        })
    if not rows:
        raise CalibrationError(
            "held-out evaluation contains no two_sided_sample_fraction estimates"
        )
    shortfalls = [float(row["shortfall"]) for row in rows]
    failures = [
        row for row in rows
        if float(row["shortfall"]) > maximum_shortfall + 1.0e-12
    ]
    return {
        "symbol_count": len(rows),
        "maximum_allowed_shortfall": maximum_shortfall,
        "within_tolerance_count": len(rows) - len(failures),
        "within_tolerance_fraction": (len(rows) - len(failures)) / len(rows),
        "failing_symbol_count": len(failures),
        "empirical_mean": statistics.fmean(float(row["target"]) for row in rows),
        "simulated_mean": statistics.fmean(float(row["simulated"]) for row in rows),
        "mean_shortfall": statistics.fmean(shortfalls),
        "median_shortfall": percentile(shortfalls, 0.5),
        "p90_shortfall": percentile(shortfalls, 0.9),
        "maximum_observed_shortfall": max(shortfalls),
        "largest_shortfalls": sorted(
            failures, key=lambda row: (-float(row["shortfall"]), str(row["symbol"])),
        )[:20],
    }


def empirical_fit_summary(
    evaluation: Mapping[str, object],
    *,
    maximum_score: float,
    maximum_metric_score: float,
    maximum_symbol_metric_absolute_residual: float,
) -> dict[str, object]:
    """Apply the fixed development-admissibility gate to frozen parameters."""
    score = evaluation_selection_score(evaluation)
    metric_rows = [dict(row) for row in evaluation.get(
        "selection_metric_scores", [],
    )]
    metric_failures = [
        row for row in metric_rows
        if float(row.get("score", math.inf)) > maximum_metric_score
    ]
    symbol_metric_failures: list[dict[str, object]] = []
    for raw in evaluation.get("moment_estimates", []):  # type: ignore[assignment]
        estimate = dict(raw)
        metric = str(estimate.get("metric", ""))
        if metric == "two_sided_sample_fraction" or metric not in METRICS:
            continue
        simulated = finite_float(
            estimate.get("simulated_mean"), label=f"gross-fit simulated {metric}",
        )
        target = finite_float(
            estimate.get("target"), label=f"gross-fit target {metric}",
        )
        residual = _robust_moment_residual(metric, simulated, target)
        if abs(residual) > maximum_symbol_metric_absolute_residual:
            symbol_metric_failures.append({
                "symbol": str(estimate.get("symbol", "")),
                "metric": metric,
                "robust_residual": residual,
                "absolute_robust_residual": abs(residual),
                "target": target,
                "simulated_mean": simulated,
                "maximum_allowed_absolute_robust_residual": (
                    maximum_symbol_metric_absolute_residual
                ),
            })
    # The thesis calibrates one behavioural policy per liquidity cluster and
    # validates a stratified sample plus market-wide distributions.  Requiring
    # every one of 1,480 individual books to remain inside an additional hard
    # residual band silently changes that estimand into per-symbol calibration.
    # Retain the complete outlier list for model criticism, while acceptance
    # remains governed by the aggregate robust score and every per-metric
    # distributional score.
    passed = (
        math.isfinite(score)
        and score <= maximum_score
        and bool(metric_rows)
        and not metric_failures
    )
    return {
        "passed": passed,
        "selection_score": score,
        "maximum_allowed_score": maximum_score,
        "maximum_allowed_metric_score": maximum_metric_score,
        "maximum_allowed_symbol_metric_absolute_robust_residual": (
            maximum_symbol_metric_absolute_residual
        ),
        "metric_scores": metric_rows,
        "failing_metrics": metric_failures,
        "gross_symbol_metric_failure_count": len(symbol_metric_failures),
        "gross_symbol_metric_failures": symbol_metric_failures[:100],
        "gross_symbol_metric_failures_role": (
            "diagnostic_outliers_under_cluster_level_calibration"
        ),
        "gross_symbol_metric_failures_required_for_acceptance": False,
        "selection_parameters_frozen_before_evaluation": True,
        "heldout_used_for_parameter_selection": False,
    }


def empirical_fit_failure_reasons(
    scope: str,
    summary: Mapping[str, object],
) -> list[str]:
    """Explain the exact robust-fit predicates that failed.

    The aggregate and per-metric limits are independent.  Reporting only the
    aggregate score is misleading when it passes but one metric does not,
    which is exactly what happened in the development return-kurtosis diagnostic.
    """
    if summary.get("passed") is True:
        return []
    reasons: list[str] = []
    score = float(summary.get("selection_score", math.inf))
    maximum_score = finite_float(
        summary.get("maximum_allowed_score"),
        label=f"{scope} maximum selection score",
    )
    if not math.isfinite(score) or score > maximum_score:
        reasons.append(
            f"{scope} aggregate robust-fit score {score:.6g} exceeds "
            f"{maximum_score:g}"
        )
    failing_metrics = summary.get("failing_metrics", [])
    if not isinstance(failing_metrics, Sequence) or isinstance(
        failing_metrics, (str, bytes)
    ):
        raise CalibrationError(f"{scope} failing_metrics is not a sequence")
    maximum_metric_score = finite_float(
        summary.get("maximum_allowed_metric_score"),
        label=f"{scope} maximum metric score",
    )
    for raw in failing_metrics:
        if not isinstance(raw, Mapping):
            raise CalibrationError(f"{scope} failing metric is not a mapping")
        metric = str(raw.get("metric", "unknown_metric"))
        metric_score = float(raw.get("score", math.inf))
        reasons.append(
            f"{scope} per-metric robust-fit score for {metric} "
            f"{metric_score:.6g} exceeds {maximum_metric_score:g}"
        )
    if not reasons:
        reasons.append(
            f"{scope} empirical-fit gate failed without a classified predicate; "
            "inspect the persisted empirical_fit object"
        )
    return reasons


def heldout_acceptance_decision(
    *,
    marketwide_validation_completed: bool,
    sampled_execution_integrity_passed: bool,
    sampled_coverage_passed: bool,
    sampled_background_boundary_adequacy_passed: bool,
    sampled_value_boundary_adequacy_passed: bool,
    sampled_empirical_fit_passed: bool,
    marketwide_execution_integrity_passed: bool,
    marketwide_background_boundary_adequacy_passed: bool,
    marketwide_value_boundary_adequacy_passed: bool,
    marketwide_empirical_fit_passed: bool,
) -> dict[str, bool]:
    """Combine the predeclared held-out evidence without scope substitution.

    The 30-symbol stratified run is deliberately retained as a required
    structural probe: execution, fixed-clock two-sided coverage and both
    source-attributed finite-boundary checks must pass.  Its empirical-fit
    score is still computed and reported, but it is a sampling diagnostic and
    cannot veto an acceptable fit over the exact 1,480-symbol universe.  The
    full-market empirical fit is the sole held-out fit certification gate.
    """
    sampled_structural_adequacy_passed = (
        sampled_execution_integrity_passed
        and sampled_coverage_passed
        and sampled_background_boundary_adequacy_passed
        and sampled_value_boundary_adequacy_passed
    )
    execution_integrity_passed = (
        sampled_execution_integrity_passed
        and marketwide_execution_integrity_passed
    )
    finite_boundary_adequacy_passed = (
        sampled_background_boundary_adequacy_passed
        and sampled_value_boundary_adequacy_passed
        and marketwide_background_boundary_adequacy_passed
        and marketwide_value_boundary_adequacy_passed
    )
    empirical_fit_passed = marketwide_empirical_fit_passed
    heldout_validation_passed = (
        marketwide_validation_completed
        and sampled_structural_adequacy_passed
        and marketwide_execution_integrity_passed
        and marketwide_background_boundary_adequacy_passed
        and marketwide_value_boundary_adequacy_passed
        and empirical_fit_passed
    )
    return {
        "stratified_structural_adequacy_passed": (
            sampled_structural_adequacy_passed
        ),
        "stratified_empirical_fit_passed": sampled_empirical_fit_passed,
        "marketwide_empirical_fit_passed": marketwide_empirical_fit_passed,
        "execution_integrity_passed": execution_integrity_passed,
        "coverage_passed": sampled_coverage_passed,
        "finite_boundary_adequacy_passed": finite_boundary_adequacy_passed,
        "empirical_fit_passed": empirical_fit_passed,
        "heldout_validation_passed": heldout_validation_passed,
    }


def full_universe_training_adequacy_summary(
    evaluation: Mapping[str, object],
    *,
    maximum_score: float,
    maximum_metric_score: float,
    maximum_symbol_metric_absolute_residual: float,
) -> dict[str, object]:
    """Gate the frozen policy on every symbol of every training session.

    Representative books are sufficient for bounded candidate search, but
    they cannot establish that a cluster-shared policy is adequate for the
    complete training universe.  This post-selection check therefore runs the
    frozen policy over all training books before any development-validation
    target is opened.  Each dated session must pass independently; a good day
    cannot hide a bad one inside the robust selection aggregate.
    """
    raw_days = evaluation.get("training_day_evaluations", [])
    if not isinstance(raw_days, Sequence) or isinstance(raw_days, (str, bytes)):
        raise CalibrationError(
            "full-universe training evaluation lacks dated evaluations"
        )
    if not raw_days:
        # ``evaluate_policy_across_training_days`` intentionally preserves the
        # legacy direct result shape for a one-day diagnostic workflow.
        raw_days = ({
            "date": "single_training_session",
            "evaluation": evaluation,
        },)
    day_summaries: list[dict[str, object]] = []
    failure_reasons: list[str] = []
    for raw_day in raw_days:
        if not isinstance(raw_day, Mapping):
            raise CalibrationError("training-day evaluation is not a mapping")
        date = str(raw_day.get("date", "unknown_date"))
        day_evaluation = raw_day.get("evaluation")
        if not isinstance(day_evaluation, Mapping):
            raise CalibrationError(
                f"full-universe training evaluation for {date} is absent"
            )
        fit = empirical_fit_summary(
            day_evaluation,
            maximum_score=maximum_score,
            maximum_metric_score=maximum_metric_score,
            maximum_symbol_metric_absolute_residual=(
                maximum_symbol_metric_absolute_residual
            ),
        )
        day_summaries.append({"date": date, "empirical_fit": fit})
        failure_reasons.extend(
            empirical_fit_failure_reasons(
                f"full-universe training {date}", fit,
            )
        )
    aggregate_score = evaluation_selection_score(evaluation)
    aggregate_score_passed = (
        math.isfinite(aggregate_score) and aggregate_score <= maximum_score
    )
    if not aggregate_score_passed:
        failure_reasons.append(
            "full-universe multi-day training aggregate robust-fit score "
            f"{aggregate_score:.6g} exceeds {maximum_score:g}"
        )
    execution_integrity_passed = (
        evaluation.get("two_sided_integrity_passed") is True
    )
    finite_boundary_adequacy_passed = (
        evaluation.get("finite_boundary_adequacy_passed") is True
    )
    value_boundary_adequacy_passed = (
        evaluation.get("value_boundary_adequacy_passed") is True
    )
    if not execution_integrity_passed:
        failure_reasons.append(
            "full-universe training contains incomplete or one-sided "
            "fixed-clock observations"
        )
    if not finite_boundary_adequacy_passed:
        failure_reasons.append(
            "full-universe training background flow depends materially on "
            "the finite-book reflection boundary"
        )
    if not value_boundary_adequacy_passed:
        failure_reasons.append(
            "full-universe training value-agent orders depend materially on "
            "the finite-book reflection boundary"
        )
    all_days_passed = bool(day_summaries) and all(
        bool(row["empirical_fit"]["passed"])  # type: ignore[index]
        for row in day_summaries
    )
    passed = (
        aggregate_score_passed
        and all_days_passed
        and execution_integrity_passed
        and finite_boundary_adequacy_passed
        and value_boundary_adequacy_passed
        and not evaluation.get("errors")
    )
    return {
        "passed": passed,
        "selection_parameters_frozen_before_evaluation": True,
        "development_validation_targets_opened": False,
        "training_day_count": len(day_summaries),
        "aggregate_selection_score": aggregate_score,
        "maximum_allowed_aggregate_score": maximum_score,
        "aggregate_selection_score_passed": aggregate_score_passed,
        "every_training_day_empirical_fit_passed": all_days_passed,
        "execution_integrity_passed": execution_integrity_passed,
        "finite_boundary_adequacy_passed": finite_boundary_adequacy_passed,
        "value_boundary_adequacy_passed": value_boundary_adequacy_passed,
        "day_summaries": day_summaries,
        "failure_reasons": failure_reasons,
    }


def structural_depth_fit_summary(
    evaluation: Mapping[str, object],
    *,
    require_zero_gross_symbol_failures: bool = True,
) -> dict[str, object]:
    """Apply robust systemic-fit thresholds to preflight depth only.

    The background-only preflight is not expected to reproduce spread: spread
    repair is the local market maker's calibrated role.  It must, however,
    reproduce bid and ask queue depth at the aggregate and metric levels.  A
    single uncalibrated reference policy is not required to have zero
    stock/date outliers before the parameter search begins.  Those outliers
    remain fully reported.  The optional strict flag is retained for targeted
    mechanism tests, but the cluster-level certification estimand uses the
    aggregate and per-metric distributional gates.
    """
    base: dict[str, object] = {
        "schema_version": 1,
        "passed": False,
        "metrics": list(STRUCTURAL_PREFLIGHT_DEPTH_METRICS),
        "robust_score": None,
        "maximum_allowed_robust_score": CERTIFICATION_MAXIMUM_ROBUST_SCORE,
        "maximum_allowed_metric_score": CERTIFICATION_MAXIMUM_METRIC_SCORE,
        "maximum_allowed_symbol_metric_absolute_robust_residual": (
            CERTIFICATION_GROSS_RESIDUAL_LIMIT
        ),
        "metric_scores": [],
        "failing_metrics": [],
        "gross_symbol_metric_failure_count": 0,
        "gross_symbol_metric_failures": [],
        "require_zero_gross_symbol_failures": (
            require_zero_gross_symbol_failures
        ),
        "gross_symbol_metric_failures_role": (
            "eligibility_gate" if require_zero_gross_symbol_failures
            else "diagnostic_only_during_structural_preflight"
        ),
        "aggregate_fit_passed": False,
        "training_targets_only": True,
        "spread_excluded_because_local_mm_is_spread_repair": True,
        "error": None,
    }
    try:
        raw_estimates = evaluation.get("moment_estimates", [])
        if not isinstance(raw_estimates, Sequence) or isinstance(
                raw_estimates, (str, bytes)):
            raise CalibrationError(
                "structural depth preflight lacks a moment-estimate sequence"
            )
        estimates: list[dict[str, object]] = [
            dict(raw) for raw in raw_estimates
            if isinstance(raw, Mapping)
            and str(raw.get("metric", ""))
                in STRUCTURAL_PREFLIGHT_DEPTH_METRICS
        ]
        # Multi-day aggregation intentionally retains moments inside dated
        # child evaluations.  Flatten only these two depth metrics here while
        # preserving the date on every diagnostic row.
        if not estimates:
            day_evaluations = evaluation.get("training_day_evaluations", [])
            if not isinstance(day_evaluations, Sequence) or isinstance(
                    day_evaluations, (str, bytes)):
                raise CalibrationError(
                    "structural depth preflight has malformed day evaluations"
                )
            for day_record in day_evaluations:
                if not isinstance(day_record, Mapping):
                    continue
                day_evaluation = day_record.get("evaluation")
                if not isinstance(day_evaluation, Mapping):
                    continue
                day_moments = day_evaluation.get("moment_estimates", [])
                if not isinstance(day_moments, Sequence) or isinstance(
                        day_moments, (str, bytes)):
                    continue
                for raw in day_moments:
                    if (not isinstance(raw, Mapping)
                            or str(raw.get("metric", ""))
                                not in STRUCTURAL_PREFLIGHT_DEPTH_METRICS):
                        continue
                    estimate = dict(raw)
                    estimate["training_date"] = str(
                        day_record.get("date", "")
                    )
                    estimates.append(estimate)
        score, metric_rows = metric_balanced_robust_loss(
            estimates,
            metrics=STRUCTURAL_PREFLIGHT_DEPTH_METRICS,
        )
        metric_failures = [
            dict(row) for row in metric_rows
            if float(row.get("score", math.inf))
                > CERTIFICATION_MAXIMUM_METRIC_SCORE
        ]
        symbol_metric_failures: list[dict[str, object]] = []
        for estimate in estimates:
            metric = str(estimate["metric"])
            simulated = finite_float(
                estimate.get("simulated_mean"),
                label=f"structural-depth simulated {metric}",
            )
            target = finite_float(
                estimate.get("target"),
                label=f"structural-depth target {metric}",
            )
            residual = _robust_moment_residual(metric, simulated, target)
            if abs(residual) > CERTIFICATION_GROSS_RESIDUAL_LIMIT:
                symbol_metric_failures.append({
                    "symbol": str(estimate.get("symbol", "")),
                    "training_date": str(estimate.get("training_date", "")),
                    "metric": metric,
                    "robust_residual": residual,
                    "absolute_robust_residual": abs(residual),
                    "target": target,
                    "simulated_mean": simulated,
                    "maximum_allowed_absolute_robust_residual": (
                        CERTIFICATION_GROSS_RESIDUAL_LIMIT
                    ),
                })
        aggregate_fit_passed = (
                math.isfinite(score)
                and score <= CERTIFICATION_MAXIMUM_ROBUST_SCORE
                and bool(metric_rows)
                and not metric_failures
        )
        base.update({
            "passed": (
                aggregate_fit_passed
                and (
                    not require_zero_gross_symbol_failures
                    or not symbol_metric_failures
                )
            ),
            "aggregate_fit_passed": aggregate_fit_passed,
            "robust_score": score,
            "metric_scores": [dict(row) for row in metric_rows],
            "failing_metrics": metric_failures,
            "gross_symbol_metric_failure_count": len(symbol_metric_failures),
            "gross_symbol_metric_failures": symbol_metric_failures[:100],
        })
    except (CalibrationError, KeyError, TypeError, ValueError) as error:
        base["error"] = str(error)
    return base


def evaluation_failure_message(
    scope: str,
    evaluation: Mapping[str, object],
) -> str:
    """Return a concise error that points to the exact failed run diagnostics."""
    errors = [str(error) for error in evaluation.get("errors", [])]
    summaries = [str(path) for path in evaluation.get("summary_paths", [])]
    if errors:
        rendered = "; ".join(errors[:3])
        if len(errors) > 3:
            rendered += f"; plus {len(errors) - 3} more error(s)"
        return f"{scope} failed: {rendered}"
    return (
        f"{scope} produced a non-finite WMM score without a parser error; "
        f"summary_paths={summaries}"
    )


def validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if bool(args.pooling_provenance) != bool(
            args.pooling_producer_project_root):
        parser.error(
            "--pooling-provenance and --pooling-producer-project-root "
            "must be supplied together"
        )
    if args.require_certification_profile:
        missing_provenance = [
            option for option, value in (
                ("--build-provenance", args.build_provenance),
                ("--cluster-manifest", args.cluster_manifest),
                ("--pooling-provenance", args.pooling_provenance),
                ("--pooling-producer-project-root",
                 args.pooling_producer_project_root),
            ) if not value
        ]
        if missing_provenance:
            parser.error(
                "--require-certification-profile also requires "
                + ", ".join(missing_provenance)
            )
    try:
        heldout_day = date.fromisoformat(args.heldout_date)
    except ValueError:
        parser.error("--heldout-date must be an ISO YYYY-MM-DD date")

    legacy_values = (
        args.training_date,
        args.training_universe_config,
        args.training_target_root,
    )
    if args.training_day:
        if any(value is not None for value in legacy_values):
            parser.error(
                "use either legacy --training-date/--training-universe-config/"
                "--training-target-root or repeat --training-day, not both"
            )
        training_dates: list[date] = []
        for entry in args.training_day:
            try:
                training_dates.append(date.fromisoformat(entry[0]))
            except (IndexError, TypeError, ValueError):
                parser.error(
                    "every --training-day DATE UNIVERSE_CONFIG TARGET_ROOT entry "
                    "needs an ISO YYYY-MM-DD DATE"
                )
        if len(set(training_dates)) != len(training_dates):
            parser.error("--training-day dates must be unique")
        if len(training_dates) > 1 and not args.pooled_training_universe_config:
            parser.error(
                "multiple --training-day entries require "
                "--pooled-training-universe-config for frozen held-out validation"
            )
    else:
        if any(value is None for value in legacy_values):
            parser.error(
                "legacy mode requires --training-date, --training-universe-config, "
                "and --training-target-root; alternatively use repeat --training-day"
            )
        try:
            training_dates = [date.fromisoformat(args.training_date)]
        except (TypeError, ValueError):
            parser.error("--training-date must be an ISO YYYY-MM-DD date")
    if any(training_day >= heldout_day for training_day in training_dates):
        parser.error("every training date must be earlier than the held-out date")
    if min(args.stage1_duration, args.stage2_duration, args.stage3_duration) <= 0:
        parser.error("stage durations must be positive")
    if not args.stage1_duration < args.stage2_duration < args.stage3_duration:
        parser.error("require stage1-duration < stage2-duration < stage3-duration")
    if args.stage3_duration != args.session_duration:
        parser.error(
            "stage3-duration must equal session-duration so stage 3 and held-out "
            "validation use full-session targets"
        )
    if args.stage1_top_candidates <= 0 or args.stage2_top_candidates <= 0:
        parser.error("ranked-policy survivor counts must be positive")
    if args.stage1_refinement_candidates < 0:
        parser.error("--stage1-refinement-candidates cannot be negative")
    if (not math.isfinite(args.maximum_two_sided_coverage_shortfall)
            or args.maximum_two_sided_coverage_shortfall < 0.0):
        parser.error(
            "--maximum-two-sided-coverage-shortfall must be finite and "
            "non-negative"
        )
    seed_lists = (args.stage1_seeds, args.stage2_seeds, args.stage3_seeds)
    if any(not values or len(set(values)) != len(values) for values in seed_lists):
        parser.error("each stage needs a non-empty list of unique seeds")
    if len(args.stage2_seeds) < 2 or len(args.stage3_seeds) < 2:
        parser.error("stage 2, stage 3, and held-out validation require at least two seeds")
    if args.timeout_seconds <= 0.0 or not math.isfinite(args.timeout_seconds):
        parser.error("--timeout-seconds must be finite and positive")
    for option, values in (
        ("--hawkes-activity-scales", args.hawkes_activity_scales),
        ("--local-mm-intervals-ms", args.local_mm_intervals_ms),
        ("--local-mm-quantity-multipliers", args.local_mm_quantity_multipliers),
    ):
        if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
            parser.error(f"{option} must contain finite positive values")
    if (len(args.hawkes_activity_scales) != 1 or not math.isclose(
            args.hawkes_activity_scales[0], FIXED_HAWKES_ACTIVITY_SCALE,
            rel_tol=0.0, abs_tol=1.0e-12)):
        parser.error(
            "--hawkes-activity-scales must be exactly 0.30 because the ITCH "
            "rate inversion fixes this direct-input scale"
        )
    if (not args.local_mm_improvement_probabilities
            or any(not math.isfinite(probability)
                   or not 0.0 <= probability <= 1.0
                   for probability in args.local_mm_improvement_probabilities)):
        parser.error(
            "--local-mm-improvement-probabilities must contain finite values "
            "in [0, 1]"
        )
    if (not args.shared_quote_multipliers
            or any(not math.isfinite(multiplier) or multiplier <= 0.0
                   for multiplier in args.shared_quote_multipliers)):
        parser.error("--shared-quote-multipliers must contain finite positive values")
    if (not math.isfinite(args.shared_treatment_multiplier)
            or args.shared_treatment_multiplier <= 0.0):
        parser.error("--shared-treatment-multiplier must be finite and positive")
    for option, value in (
        ("--maximum-heldout-robust-score", args.maximum_heldout_robust_score),
        ("--maximum-heldout-metric-score", args.maximum_heldout_metric_score),
    ):
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{option} must be finite and positive")
    if args.require_certification_profile:
        immutable_thresholds = (
            (
                "--maximum-two-sided-coverage-shortfall",
                args.maximum_two_sided_coverage_shortfall,
                CERTIFICATION_MAXIMUM_TWO_SIDED_SHORTFALL,
            ),
            (
                "--maximum-heldout-robust-score",
                args.maximum_heldout_robust_score,
                CERTIFICATION_MAXIMUM_ROBUST_SCORE,
            ),
            (
                "--maximum-heldout-metric-score",
                args.maximum_heldout_metric_score,
                CERTIFICATION_MAXIMUM_METRIC_SCORE,
            ),
        )
        for option, observed, expected in immutable_thresholds:
            if not math.isclose(
                    observed, expected, rel_tol=0.0, abs_tol=1.0e-12):
                parser.error(
                    f"{option} is immutable under {CERTIFICATION_GATE_ID} "
                    f"(expected {expected:g})"
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--binary", required=True,
                        help="rank-one fragmented_mpi_lob executable")
    parser.add_argument(
        "--build-provenance",
        help=("JSON emitted after the Release build; mandatory for the "
              "certification profile and binds --binary to source semantics"),
    )
    parser.add_argument("--training-universe-config",
                        help=("legacy single-session all-symbol empirical configuration; "
                              "use with --training-date and --training-target-root"))
    heldout_input = parser.add_mutually_exclusive_group(required=True)
    heldout_input.add_argument(
        "--heldout-universe-config",
        help=("already-frozen held-out configuration: it must match the training "
              "configuration except for the five opening-state fields"),
    )
    heldout_input.add_argument(
        "--heldout-opening-source-config",
        help=("normal held-out ITCH configuration used only as an opening-state "
              "source; all non-opening training inputs are frozen automatically"),
    )
    parser.add_argument("--cluster-assignments", required=True,
                        help="cluster_assignments.csv from cluster_empirical_universe.py")
    parser.add_argument("--validation-sample", required=True,
                        help="validation_sample.csv from cluster_empirical_universe.py")
    parser.add_argument(
        "--cluster-manifest",
        help=("cluster_manifest.json binding features, algorithm, seed, assignments "
              "and validation sample; mandatory for certification"),
    )
    parser.add_argument(
        "--pooling-provenance",
        help=("pooling_provenance.json for the five-session direct-input pool; "
              "mandatory for certification"),
    )
    parser.add_argument(
        "--pooling-producer-project-root",
        help=("source-tree root whose workflow hash produced "
              "--pooling-provenance; mandatory whenever pooling provenance "
              "is supplied and independently verified from the current "
              "calibration source tree"),
    )
    parser.add_argument("--training-date", help="legacy ISO training date")
    parser.add_argument("--heldout-date", required=True, help="ISO held-out date")
    parser.add_argument("--training-target-root",
                        help="legacy root containing training target directories")
    parser.add_argument(
        "--training-day", action="append", nargs=3,
        metavar=("DATE", "UNIVERSE_CONFIG", "TARGET_ROOT"),
        help=("repeat for each chronologically earlier training session. Candidate "
              "scores use the median plus 0.25 MAD of metric-balanced daily losses; "
              "cannot be combined with legacy training arguments"),
    )
    parser.add_argument(
        "--pooled-training-universe-config",
        help=("explicitly prepared pooled direct-input configuration used only for "
              "the frozen held-out model. Required when multiple --training-day "
              "entries are supplied; it is not used to score candidates."),
    )
    parser.add_argument("--heldout-target-root", required=True,
                        help="root containing held-out target directories")
    parser.add_argument("--output-dir", required=True,
                        help="new calibration result directory")
    parser.add_argument(
        "--launcher", default="",
        help=("optional complete one-rank launcher prefix, e.g. "
              "'mpirun --bind-to core --map-by slot -np 1'; default invokes binary directly"),
    )
    parser.add_argument("--stage1-duration", type=int, default=300)
    parser.add_argument("--stage2-duration", type=int, default=3_600)
    parser.add_argument("--stage3-duration", type=int, default=23_400)
    parser.add_argument("--session-duration", type=int, default=23_400)
    parser.add_argument(
        "--stage1-top-candidates", type=int, default=6,
        help=(
            "Stage-1 survivor count for ranked policy blocks and number of "
            "local-flow grid leaders used to construct midpoint refinements; "
            "does not cap structurally eligible local-flow promotion"
        ),
    )
    parser.add_argument(
        "--stage1-refinement-candidates", type=int, default=32,
        help=(
            "maximum deterministic midpoint candidates evaluated around the "
            "leading eligible Stage-1 grid points; zero disables refinement"
        ),
    )
    parser.add_argument(
        "--stage2-top-candidates", type=int, default=2,
        help=(
            "Stage-2 survivor count for ranked policy blocks; does not cap "
            "structurally eligible local-flow promotion to the full-day stage"
        ),
    )
    parser.add_argument("--stage1-seeds", type=int, nargs="+", default=[1729])
    parser.add_argument("--stage2-seeds", type=int, nargs="+", default=[1729, 7919])
    parser.add_argument(
        "--stage3-seeds", type=int, nargs="+",
        default=[1729, 7919, 1103, 6599, 2027],
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[5.0, 8.0, 10.0, 15.0, 25.0, 40.0],
        help=(
            "per-cluster value-agent threshold candidates in basis points; the "
            "disabled baseline is added separately"
        ),
    )
    parser.add_argument(
        "--depth-participations", type=float, nargs="+",
        default=list(CERTIFICATION_VALUE_DEPTH_PARTICIPATIONS),
        help=(
            "per-cluster fraction of contemporaneous displayed opposite-side "
            "depth submitted by a value order protected at the fundamental"
        ),
    )
    parser.add_argument(
        "--hawkes-activity-scales", type=float, nargs="+",
        default=[FIXED_HAWKES_ACTIVITY_SCALE],
        help=(
            "fixed Hawkes activity scale; must be exactly 0.30 because the "
            "per-symbol ITCH rate files were inverted at that value"
        ),
    )
    parser.add_argument(
        "--local-mm-intervals-ms", type=float, nargs="+",
        default=[500.0, 1000.0, 2000.0],
        help="global stage-1 local market-maker refresh candidates in milliseconds",
    )
    parser.add_argument(
        "--local-mm-quantity-multipliers", type=float, nargs="+",
        default=[0.5, 1.0, 2.0],
        help="global stage-1 multiplier of each book's ITCH local quote proxy",
    )
    parser.add_argument(
        "--local-mm-improvement-probabilities", type=float, nargs="+",
        default=[0.0, 0.25, 0.5, 1.0],
        help=(
            "global stage-1 probability that a local-MM refresh improves the "
            "current same-side BBO by one tick"
        ),
    )
    parser.add_argument(
        "--shared-quote-multipliers", type=float, nargs="+",
        default=[0.5, 1.0, 2.0],
        help=(
            "enabled stage-3 shared-MM multipliers of each symbol's empirical "
            "quote-size proxy; an off baseline is added automatically"
        ),
    )
    parser.add_argument(
        "--shared-treatment-multiplier", type=float, default=1.0,
        help=(
            "explicit nonzero shared-MM mechanism treatment retained for the "
            "case study if the nested off baseline wins; it is then a scenario "
            "parameter, not a calibrated estimate"
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--maximum-two-sided-coverage-shortfall", type=float,
        default=CERTIFICATION_MAXIMUM_TWO_SIDED_SHORTFALL,
        help=(
            "fixed protocol diagnostic: simulated coverage may not fall more than this "
            "amount below each preselected symbol's empirical two-sided fraction; "
            "the full-universe run reports the same statistic distributionally "
            "(default: 0.01)"
        ),
    )
    parser.add_argument(
        "--maximum-heldout-robust-score", type=float,
        default=CERTIFICATION_MAXIMUM_ROBUST_SCORE,
        help=(
            "immutable metric-balanced Huber threshold in the named "
            "development-validation gate"
        ),
    )
    parser.add_argument(
        "--maximum-heldout-metric-score", type=float,
        default=CERTIFICATION_MAXIMUM_METRIC_SCORE,
        help=(
            "maximum allowed held-out score for any individual stylised fact; "
            "prevents an acceptable average from hiding one gross mismatch"
        ),
    )
    parser.add_argument(
        "--marketwide-validation", action="store_true",
        help=("after stratified validation, run the frozen held-out full universe and "
              "write a true market-wide distributional validation; computationally expensive"),
    )
    parser.add_argument(
        "--require-certification-profile", action="store_true",
        help=("fail before model execution unless dates, cluster/sample sizes, "
              "horizons, seeds and candidate grids exactly match the committed profile"),
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing same-named artifact in --output-dir")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the complete leakage-safe cluster calibration protocol."""
    output_root = pathlib.Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    invalidate_terminal_calibration_artifacts(
        output_root, overwrite=args.overwrite,
    )
    progress_path = output_root / "calibration_progress.json"
    initialize_calibration_progress(progress_path, overwrite=args.overwrite)

    project_root = pathlib.Path(__file__).resolve().parents[1]
    binary = pathlib.Path(args.binary).expanduser().resolve()
    if not binary.is_file():
        raise CalibrationError(f"--binary is not a file: {binary}")
    simulator_semantics_sha256 = simulator_source_semantics_sha256(project_root)
    workflow_semantics_sha256 = workflow_source_semantics_sha256(project_root)
    build_provenance = (
        validate_build_provenance(
            pathlib.Path(args.build_provenance).expanduser().resolve(),
            binary=binary,
            project_root=project_root,
        )
        if args.build_provenance else None
    )
    heldout_input_arg = (
        args.heldout_universe_config
        if args.heldout_universe_config is not None
        else args.heldout_opening_source_config
    )
    assert heldout_input_arg is not None
    heldout_config_path = pathlib.Path(heldout_input_arg).expanduser().resolve()
    assignments_path = pathlib.Path(args.cluster_assignments).expanduser().resolve()
    validation_path = pathlib.Path(args.validation_sample).expanduser().resolve()
    heldout_target_root = pathlib.Path(args.heldout_target_root).expanduser().resolve()
    launcher = tuple(shlex.split(args.launcher))

    training_days = load_training_days(args)
    (
        pooled_training_config_path,
        pooled_training_fields,
        pooled_training_rows,
        pooled_training_config_sha256,
    ) = load_pooled_training_config(args, training_days)
    heldout_fields, heldout_rows = load_universe_config(heldout_config_path)
    validate_frozen_homeostatic_targets(
        training_days, pooled_training_rows, heldout_rows
    )
    if args.require_certification_profile:
        expected_fields = tuple(RUNTIME_CONFIG_FIELDS)
        named_schemas = [
            (f"training day {day.date}", day.fields)
            for day in training_days
        ]
        named_schemas.extend([
            ("pooled training universe", pooled_training_fields),
            ("development-validation opening source", heldout_fields),
        ])
        for label, observed_fields in named_schemas:
            if tuple(observed_fields) != expected_fields:
                raise CalibrationError(
                    f"{label} does not use certified runtime configuration "
                    f"schema version {RUNTIME_CONFIG_SCHEMA_VERSION}"
                )
    if args.heldout_universe_config is not None:
        heldout_mode = "already_frozen_heldout_config"
        heldout_opening_rows = merge_frozen_heldout_config(
            pooled_training_fields, pooled_training_rows, heldout_fields, heldout_rows,
        )
    else:
        heldout_mode = "raw_heldout_opening_source"
        heldout_opening_rows = freeze_training_backgrounds_with_heldout_openings(
            pooled_training_fields, pooled_training_rows, heldout_fields, heldout_rows,
        )
    all_symbols = tuple(row["symbol"] for row in training_days[0].rows)
    required_symbol_count = (
        CERTIFICATION_COMMON_SYMBOL_COUNT
        if args.require_certification_profile else len(all_symbols)
    )
    layout = load_cluster_layout(assignments_path, validation_path, all_symbols)
    cluster_manifest = (
        validate_cluster_manifest(
            pathlib.Path(args.cluster_manifest).expanduser().resolve(),
            assignments_path=assignments_path,
            validation_path=validation_path,
            universe_config_path=pooled_training_config_path,
        )
        if args.cluster_manifest else None
    )
    pooling_producer_project_root = (
        pathlib.Path(args.pooling_producer_project_root).expanduser().resolve()
        if args.pooling_producer_project_root else None
    )
    pooling_provenance = (
        validate_pooling_provenance(
            pathlib.Path(args.pooling_provenance).expanduser().resolve(),
            training_days=training_days,
            pooled_config_path=pooled_training_config_path,
            heldout_config_path=heldout_config_path,
            heldout_target_root=heldout_target_root,
            producer_project_root=pooling_producer_project_root,  # type: ignore[arg-type]
            project_root=project_root,
        )
        if args.pooling_provenance else None
    )
    certification_input_selection = (
        pooling_provenance.get("certification_input_selection")
        if isinstance(pooling_provenance, Mapping) else None
    )
    cohort_identity: dict[str, object] | None = None
    if args.require_certification_profile:
        try:
            artifact_checks: dict[str, object] = {
                "pooled_training_universe": cohort.validate_csv(
                    pooled_training_config_path,
                    label="pooled training universe",
                    project_root=project_root,
                ),
                "training_days": {
                    day.date: cohort.validate_csv(
                        day.universe_config,
                        label=f"training universe {day.date}",
                        project_root=project_root,
                    )
                    for day in training_days
                },
                "heldout_opening_universe": cohort.validate_csv(
                    heldout_config_path,
                    label="held-out opening universe",
                    project_root=project_root,
                ),
                "cluster_assignments": cohort.validate_csv(
                    assignments_path,
                    label="cluster assignments",
                    project_root=project_root,
                ),
            }
            cohort_identity = {
                "schema_version": 1,
                **cohort.validate_csv(
                    pooled_training_config_path,
                    label="pooled training universe",
                    project_root=project_root,
                ),
                "artifact_checks": artifact_checks,
            }
        except cohort.CohortIdentityError as error:
            raise CalibrationError(str(error)) from error
    runtime_configuration_schema = {
        "schema_version": RUNTIME_CONFIG_SCHEMA_VERSION,
        "fields": list(RUNTIME_CONFIG_FIELDS),
        "sha256": configuration_schema_sha256(RUNTIME_CONFIG_FIELDS),
        "pooled_homeostatic_fields": list(POOLED_HOMEOSTATIC_FIELDS),
        "latent_value_fields": list(LATENT_VALUE_FIELDS),
        "frozen_training_derived_fields": list(
            FROZEN_TRAINING_DERIVED_FIELDS
        ),
        "heldout_target_files_used": False,
    }
    runtime_shared_quote_candidate_count = (
        1 + len(set(args.shared_quote_multipliers))
    )
    (
        runtime_shared_quote_stage1_promoted_count,
        runtime_shared_quote_stage2_promoted_count,
        runtime_shared_quote_stage3_promoted_count,
    ) = ranked_survivor_trajectory(
        runtime_shared_quote_candidate_count,
        (
            args.stage1_top_candidates,
            args.stage2_top_candidates,
            1,
        ),
    )
    if args.require_certification_profile:
        early_checks = {
            "training_dates": [day.date for day in training_days]
                == list(CERTIFICATION_TRAINING_DATES),
            "validation_date": args.heldout_date == CERTIFICATION_VALIDATION_DATE,
            "common_symbol_count": (
                len(all_symbols) == CERTIFICATION_COMMON_SYMBOL_COUNT
            ),
            "cluster_count": len(layout.cluster_ids) == CERTIFICATION_CLUSTER_COUNT,
            "training_representatives": all(
                len(layout.representatives[cluster])
                == CERTIFICATION_TRAINING_REPRESENTATIVES_PER_CLUSTER
                for cluster in layout.cluster_ids
            ),
            "validation_symbols": all(
                len(layout.validation_symbols[cluster])
                == CERTIFICATION_VALIDATION_SYMBOLS_PER_CLUSTER
                for cluster in layout.cluster_ids
            ),
            "stage_durations": (
                args.stage1_duration == CERTIFICATION_STAGE1_DURATION_SECONDS
                and args.stage2_duration == CERTIFICATION_STAGE2_DURATION_SECONDS
                and args.stage3_duration == CERTIFICATION_SESSION_DURATION_SECONDS
                and args.session_duration == CERTIFICATION_SESSION_DURATION_SECONDS
            ),
            "stage_seeds": (
                tuple(args.stage1_seeds) == CERTIFICATION_STAGE1_SEEDS
                and tuple(args.stage2_seeds) == CERTIFICATION_STAGE2_SEEDS
                and tuple(args.stage3_seeds) == CERTIFICATION_STAGE3_SEEDS
            ),
            "promotion_and_survivors": (
                args.stage1_top_candidates == CERTIFICATION_STAGE1_SURVIVORS
                and args.stage1_refinement_candidates
                    == CERTIFICATION_STAGE1_REFINEMENT_CANDIDATES
                and args.stage2_top_candidates == CERTIFICATION_STAGE2_SURVIVORS
            ),
            "candidate_grids": (
                tuple(sorted(set(args.thresholds)))
                    == CERTIFICATION_VALUE_THRESHOLDS_BPS
                and tuple(sorted(set(args.depth_participations)))
                    == CERTIFICATION_VALUE_DEPTH_PARTICIPATIONS
                and tuple(sorted(set(args.local_mm_intervals_ms)))
                    == CERTIFICATION_LOCAL_MM_INTERVALS_MS
                and tuple(sorted(set(args.local_mm_quantity_multipliers)))
                    == CERTIFICATION_LOCAL_MM_QUANTITY_MULTIPLIERS
                and tuple(sorted(set(args.local_mm_improvement_probabilities)))
                    == CERTIFICATION_LOCAL_MM_IMPROVEMENT_PROBABILITIES
                and tuple(sorted(set(args.shared_quote_multipliers)))
                    == CERTIFICATION_SHARED_QUOTE_MULTIPLIERS
            ),
            "maximum_heldout_robust_score": (
                args.maximum_heldout_robust_score
                == CERTIFICATION_MAXIMUM_ROBUST_SCORE
            ),
            "maximum_heldout_metric_score": (
                args.maximum_heldout_metric_score
                == CERTIFICATION_MAXIMUM_METRIC_SCORE
            ),
            "maximum_two_sided_coverage_shortfall": (
                args.maximum_two_sided_coverage_shortfall
                == CERTIFICATION_MAXIMUM_TWO_SIDED_SHORTFALL
            ),
            "marketwide_validation": bool(args.marketwide_validation),
            "build_provenance": build_provenance is not None,
            "cluster_manifest": cluster_manifest is not None,
            "pooling_provenance": pooling_provenance is not None,
            "exact_common_symbol_cohort": (
                cohort_identity is not None
                and cohort_identity.get("symbol_order_sha256")
                    == cohort.REQUIRED_SYMBOL_ORDER_SHA256
            ),
        }
        failed = [name for name, passed in early_checks.items() if not passed]
        if failed:
            raise CalibrationError(
                "--require-certification-profile failed before simulation: "
                + ", ".join(failed)
            )
    value_candidates = candidate_grid(
        args.thresholds, args.depth_participations,
    )
    local_candidates = local_flow_candidate_grid(
        args.hawkes_activity_scales,
        args.local_mm_intervals_ms,
        args.local_mm_quantity_multipliers,
        args.local_mm_improvement_probabilities,
    )
    original_local_grid = tuple(local_candidates)
    shared_candidates = shared_quote_candidate_grid(args.shared_quote_multipliers)
    if len(shared_candidates) != runtime_shared_quote_candidate_count:
        raise CalibrationError(
            "shared-quote candidate grid count disagrees with its declared "
            "runtime profile"
        )
    representative_symbols = tuple(
        symbol
        for cluster_id in layout.cluster_ids
        for symbol in layout.representatives[cluster_id]
    )
    training_input_provenance = [
        {
            **training_day_provenance(training_day),
            "empirical_input_bundle_sha256": empirical_input_bundle_sha256(
                training_day.universe_config
            ),
            "target_artifact_bundle_sha256": target_artifact_bundle_sha256(
                training_day.target_root,
                training_day.date,
                all_symbols,
                (args.stage1_duration, args.stage2_duration, None),
            ),
        }
        for training_day in training_days
    ]

    # All training target files are opened before their corresponding stage;
    # held-out target files remain unopened until all policy selections finish.
    training_targets_by_stage: dict[
        str, dict[str, Mapping[str, Mapping[str, TargetMoment]]]
    ] = {
        "stage1_screen": {
            training_day.date: load_targets(
                training_day.target_root, training_day.date,
                representative_symbols, window_seconds=args.stage1_duration,
            )
            for training_day in training_days
        },
        "stage2_refinement": {
            training_day.date: load_targets(
                training_day.target_root, training_day.date,
                representative_symbols, window_seconds=args.stage2_duration,
            )
            for training_day in training_days
        },
        "stage3_full": {
            training_day.date: load_targets(
                training_day.target_root, training_day.date,
                representative_symbols,
            )
            for training_day in training_days
        },
    }
    stage_definitions = (
        ("stage1_screen", args.stage1_duration, tuple(args.stage1_seeds),
         args.stage1_top_candidates),
        ("stage2_refinement", args.stage2_duration, tuple(args.stage2_seeds),
         args.stage2_top_candidates),
        ("stage3_full", args.stage3_duration, tuple(args.stage3_seeds), 1),
    )

    # Block 1: select a single market-wide local-flow triple on a joint
    # representative market.  The fragmented executable exposes these values
    # globally, so selecting them independently by cluster would not describe
    # an executable model.  Shared and value agents are disabled here to avoid
    # fitting later strategic mechanisms into background activity.
    calibration_details: list[dict[str, object]] = []
    local_root = output_root / "global_local_flow_calibration"
    representative_config_paths = write_training_subset_configs(
        local_root, training_days, representative_symbols,
        filename="training_representative_config.csv",
        overwrite=args.overwrite,
    )
    local_current: list[tuple[int, LocalFlowCandidate]] = list(enumerate(local_candidates))
    local_stage_reports: dict[str, list[dict[str, object]]] = {}
    local_stage_promotion_counts: dict[str, int] = {}
    local_final_evaluations: dict[int, Mapping[str, object]] = {}
    observed_survivor_counts: dict[str, Any] = {
        "global_local_flow": {},
        "cluster_value_policy": {},
        "global_shared_quote": {},
    }

    # Before evaluating the full grid, require the repaired reduced-book
    # mechanics to survive the complete Stage-2 horizon for both the pure
    # empirical background and one ordinary enabled local-revision control.
    # This is a structural gate, not a ranking shortcut: both candidates must
    # satisfy two-sided integrity, finite-boundary adequacy, and a depth-only
    # empirical-fit gate under the immutable robust thresholds.  Spread is
    # excluded because the local market maker is the later spread-repair
    # component.  An invalid queue structure is therefore stopped after two
    # candidates instead of after the complete candidate grid.
    indexed_local_candidates = list(enumerate(local_candidates))
    disabled_preflight = next(
        item for item in indexed_local_candidates if not item[1].local_mm_enabled
    )
    enabled_candidates = [
        item for item in indexed_local_candidates if item[1].local_mm_enabled
    ]
    if not enabled_candidates:
        raise CalibrationError("local-flow grid has no enabled preflight control")
    enabled_preflight = min(
        enabled_candidates,
        key=lambda item: (
            abs(math.log(item[1].local_mm_interval_ms / 1000.0)),
            abs(math.log(item[1].local_mm_quantity_multiplier)),
            abs(item[1].local_mm_improvement_probability - 0.5),
            item[0],
        ),
    )
    preflight_references: list[dict[str, object]] = []
    preflight_evaluated: list[dict[str, object]] = []
    for candidate_index, candidate in (disabled_preflight, enabled_preflight):
        candidate_dir = (
            local_root / "structural_preflight"
            / f"candidate_{candidate_index:03d}"
        )
        evaluation = evaluate_policy_across_training_days(
            launcher=launcher, binary=binary, training_days=training_days,
            configs_by_date=representative_config_paths,
            policy=None, symbols=representative_symbols,
            output_dir=candidate_dir, duration=args.stage2_duration,
            seeds=tuple(args.stage2_seeds),
            targets_by_date=training_targets_by_stage["stage2_refinement"],
            local_controls=candidate, shared_quote_multiplier=None,
            enable_shared_mm=False, enable_value_agents=False,
            metrics=LOCAL_FLOW_METRICS,
            timeout_seconds=args.timeout_seconds,
        )
        evaluation["structural_depth_fit"] = structural_depth_fit_summary(
            evaluation,
            require_zero_gross_symbol_failures=False,
        )
        item = {
            "candidate_index": candidate_index,
            "candidate": candidate,
            "evaluation": evaluation,
        }
        preflight_evaluated.append(item)
        preflight_references.append(persist_candidate_evaluation(
            candidate_dir,
            block="global_local_flow",
            stage="structural_preflight",
            cluster_id=None,
            candidate_index=candidate_index,
            candidate=candidate,
            evaluation=evaluation,
            progress_path=progress_path,
            overwrite=args.overwrite,
        ))
        calibration_details.append(stage_detail_rows(
            "local_structural_preflight", -1, candidate_index,
            candidate, evaluation, local_controls=candidate,
        ))
    preflight_promoted = [
        int(item["candidate_index"])
        for item in preflight_evaluated if candidate_is_eligible(item)
    ]
    preflight_checkpoint = persist_stage_checkpoint(
        local_root / "structural_preflight",
        block="global_local_flow",
        stage="structural_preflight",
        cluster_id=None,
        candidate_references=preflight_references,
        promoted_candidate_indices=preflight_promoted,
        configured_ranked_survivor_count=2,
        progress_path=progress_path,
        overwrite=args.overwrite,
    )
    observed_survivor_counts["global_local_flow"]["structural_preflight"] = (
        preflight_checkpoint["observed_counts"]
    )
    if len(preflight_promoted) != len(preflight_evaluated):
        raise RuntimeError(
            "local-flow structural preflight failed its depth, two-sided, or "
            "finite-boundary gate; the full candidate grid was not started; "
            f"diagnostics: {local_root / 'structural_preflight'}"
        )

    for stage_name, duration, seeds, _ranked_block_survivor_count in stage_definitions:
        evaluated: list[dict[str, object]] = []
        candidate_references: list[dict[str, object]] = []
        for candidate_index, candidate in local_current:
            candidate_dir = local_root / stage_name / f"candidate_{candidate_index:03d}"
            evaluation = evaluate_policy_across_training_days(
                launcher=launcher, binary=binary, training_days=training_days,
                configs_by_date=representative_config_paths,
                policy=None, symbols=representative_symbols,
                output_dir=candidate_dir, duration=duration, seeds=seeds,
                targets_by_date=training_targets_by_stage[stage_name],
                local_controls=candidate, shared_quote_multiplier=None,
                enable_shared_mm=False, enable_value_agents=False,
                metrics=LOCAL_FLOW_METRICS,
                timeout_seconds=args.timeout_seconds,
            )
            evaluated.append({
                "candidate_index": candidate_index,
                "candidate": candidate,
                "evaluation": evaluation,
            })
            candidate_references.append(persist_candidate_evaluation(
                candidate_dir,
                block="global_local_flow",
                stage=stage_name,
                cluster_id=None,
                candidate_index=candidate_index,
                candidate=candidate,
                evaluation=evaluation,
                progress_path=progress_path,
                overwrite=args.overwrite,
            ))
            calibration_details.append(stage_detail_rows(
                f"local_{stage_name}", -1, candidate_index, candidate, evaluation,
                local_controls=candidate,
            ))
            if stage_name == "stage3_full":
                local_final_evaluations[candidate_index] = evaluation
        evaluated.sort(key=candidate_selection_key)
        if stage_name == "stage1_screen" and args.stage1_refinement_candidates > 0:
            initial_eligible = [
                item for item in evaluated
                if candidate_is_eligible(item)
            ]
            if not initial_eligible:
                stage_checkpoint = persist_stage_checkpoint(
                    local_root / stage_name,
                    block="global_local_flow",
                    stage=stage_name,
                    cluster_id=None,
                    candidate_references=candidate_references,
                    promoted_candidate_indices=(),
                    configured_ranked_survivor_count=(
                        _ranked_block_survivor_count
                    ),
                    progress_path=progress_path,
                    overwrite=args.overwrite,
                )
                observed_survivor_counts["global_local_flow"][stage_name] = (
                    stage_checkpoint["observed_counts"]
                )
                raise RuntimeError("all local-flow stage1_screen candidates failed")
            local_stage_reports["stage1_initial_grid"] = [
                {
                    "candidate_index": item["candidate_index"],
                    "candidate": asdict(item["candidate"]),
                    "evaluation": evaluation_report(item["evaluation"]),
                }
                for item in evaluated
            ]
            refined_candidates = refine_local_flow_candidates(
                [
                    item["candidate"]  # type: ignore[misc]
                    for item in initial_eligible[:args.stage1_top_candidates]
                ],
                original_local_grid,
                args.stage1_refinement_candidates,
            )
            refined_reports: list[dict[str, object]] = []
            for candidate in refined_candidates:
                candidate_index = len(local_candidates)
                local_candidates.append(candidate)
                candidate_dir = (
                    local_root / "stage1_refinement"
                    / f"candidate_{candidate_index:03d}"
                )
                evaluation = evaluate_policy_across_training_days(
                    launcher=launcher, binary=binary,
                    training_days=training_days,
                    configs_by_date=representative_config_paths,
                    policy=None, symbols=representative_symbols,
                    output_dir=candidate_dir,
                    duration=args.stage1_duration,
                    seeds=tuple(args.stage1_seeds),
                    targets_by_date=training_targets_by_stage["stage1_screen"],
                    local_controls=candidate,
                    shared_quote_multiplier=None,
                    enable_shared_mm=False,
                    enable_value_agents=False,
                    metrics=LOCAL_FLOW_METRICS,
                    timeout_seconds=args.timeout_seconds,
                )
                item = {
                    "candidate_index": candidate_index,
                    "candidate": candidate,
                    "evaluation": evaluation,
                }
                evaluated.append(item)
                candidate_references.append(persist_candidate_evaluation(
                    candidate_dir,
                    block="global_local_flow",
                    stage="stage1_refinement",
                    cluster_id=None,
                    candidate_index=candidate_index,
                    candidate=candidate,
                    evaluation=evaluation,
                    progress_path=progress_path,
                    overwrite=args.overwrite,
                ))
                refined_reports.append({
                    "candidate_index": candidate_index,
                    "candidate": asdict(candidate),
                    "evaluation": evaluation_report(evaluation),
                })
                calibration_details.append(stage_detail_rows(
                    "local_stage1_refinement", -1, candidate_index,
                    candidate, evaluation, local_controls=candidate,
                ))
            local_stage_reports["stage1_refinement"] = refined_reports
            evaluated.sort(key=candidate_selection_key)
        local_survivors = select_local_flow_stage_survivors(
            stage_name, evaluated,
        )
        stage_checkpoint = persist_stage_checkpoint(
            local_root / stage_name,
            block="global_local_flow",
            stage=stage_name,
            cluster_id=None,
            candidate_references=candidate_references,
            promoted_candidate_indices=[
                int(item["candidate_index"]) for item in local_survivors
            ],
            configured_ranked_survivor_count=_ranked_block_survivor_count,
            progress_path=progress_path,
            overwrite=args.overwrite,
        )
        observed_survivor_counts["global_local_flow"][stage_name] = (
            stage_checkpoint["observed_counts"]
        )
        if not local_survivors:
            raise RuntimeError(f"all local-flow {stage_name} candidates failed")
        local_stage_reports[stage_name] = [
            {
                "candidate_index": item["candidate_index"],
                "candidate": asdict(item["candidate"]),
                "evaluation": evaluation_report(item["evaluation"]),
            }
            for item in evaluated
        ]
        local_current = [
            (int(item["candidate_index"]), item["candidate"])
            for item in local_survivors
        ]
        local_stage_promotion_counts[stage_name] = len(local_current)
    selected_local_index, selected_local_controls = local_current[0]
    selected_local_evaluation = local_final_evaluations[selected_local_index]

    # Block 2: with the market-wide local-flow triple frozen, select one
    # compact value policy per liquidity cluster exactly as before.  The
    # shared supplier remains disabled, which keeps the cluster policy fit
    # separate from the global counterfactual liquidity proxy in block 3.
    selected_policies: dict[int, Candidate] = {}
    selected_training_evaluations: dict[int, Mapping[str, object]] = {}
    cluster_reports: dict[str, object] = {}
    for cluster_id in layout.cluster_ids:
        observed_survivor_counts["cluster_value_policy"][str(cluster_id)] = {}
        representatives = layout.representatives[cluster_id]
        cluster_root = output_root / "training_calibration" / f"cluster_{cluster_id:02d}"
        config_paths_by_date = write_training_subset_configs(
            cluster_root, training_days, representatives,
            filename="training_representative_config.csv",
            overwrite=args.overwrite,
        )
        current: list[tuple[int, Candidate]] = list(enumerate(value_candidates))
        per_stage: dict[str, list[dict[str, object]]] = {}
        final_evaluations: dict[int, Mapping[str, object]] = {}
        for stage_name, duration, seeds, survivor_count in stage_definitions:
            evaluated: list[dict[str, object]] = []
            candidate_references: list[dict[str, object]] = []
            for candidate_index, candidate in current:
                candidate_dir = cluster_root / stage_name / f"candidate_{candidate_index:03d}"
                policy_path = candidate_dir / "value_agent_policy.csv"
                write_policy_csv(
                    policy_path, representatives, layout,
                    candidate_policy_for_cluster(cluster_id, candidate),
                    policy_source=f"{stage_name}_cluster_{cluster_id}",
                    overwrite=args.overwrite,
                )
                evaluation = evaluate_policy_across_training_days(
                    launcher=launcher, binary=binary, training_days=training_days,
                    configs_by_date=config_paths_by_date,
                    policy=policy_path, symbols=representatives,
                    output_dir=candidate_dir, duration=duration, seeds=seeds,
                    targets_by_date=training_targets_by_stage[stage_name],
                    local_controls=selected_local_controls,
                    shared_quote_multiplier=None,
                    enable_shared_mm=False,
                    enable_value_agents=True,
                    timeout_seconds=args.timeout_seconds,
                )
                # Block 1 already certifies the background generator.  This
                # block gates boundary dependence attributable specifically to
                # the candidate value policy.  Background contacts remain in
                # the report as diagnostics and cannot be misattributed to the
                # policy being selected.
                evaluation["background_finite_boundary_adequacy"] = (
                    evaluation.get("finite_boundary_adequacy")
                )
                evaluation["background_finite_boundary_adequacy_passed"] = (
                    evaluation.get("finite_boundary_adequacy_passed") is True
                )
                cluster_boundary = cluster_training_boundary_adequacy(
                    evaluation
                )
                evaluation["finite_boundary_adequacy"] = cluster_boundary
                evaluation["finite_boundary_adequacy_passed"] = (
                    cluster_boundary["passed"] is True
                )
                evaluation["value_boundary_adequacy"] = cluster_boundary
                evaluation["value_boundary_adequacy_passed"] = (
                    cluster_boundary["passed"] is True
                )
                evaluated.append({
                    "candidate_index": candidate_index,
                    "candidate": candidate,
                    "evaluation": evaluation,
                })
                candidate_references.append(persist_candidate_evaluation(
                    candidate_dir,
                    block="cluster_value_policy",
                    stage=stage_name,
                    cluster_id=cluster_id,
                    candidate_index=candidate_index,
                    candidate=candidate,
                    evaluation=evaluation,
                    progress_path=progress_path,
                    overwrite=args.overwrite,
                ))
                calibration_details.append(stage_detail_rows(
                    f"value_{stage_name}", cluster_id, candidate_index, candidate,
                    evaluation, local_controls=selected_local_controls,
                ))
                if stage_name == "stage3_full":
                    final_evaluations[candidate_index] = evaluation
            evaluated.sort(key=candidate_selection_key)
            eligible = [
                item for item in evaluated
                if candidate_is_eligible(item)
            ]
            promoted = ranked_policy_stage_survivors(
                stage_name,
                eligible,
                survivor_count,
                required_depth_participations=args.depth_participations,
                required_thresholds_bps=args.thresholds,
            )
            configured_value_promotion_count = (
                1
                if stage_name == "stage3_full" else
                1
                + len(set(args.depth_participations))
                * len(set(args.thresholds))
            )
            stage_checkpoint = persist_stage_checkpoint(
                cluster_root / stage_name,
                block="cluster_value_policy",
                stage=stage_name,
                cluster_id=cluster_id,
                candidate_references=candidate_references,
                promoted_candidate_indices=[
                    int(item["candidate_index"]) for item in promoted
                ],
                configured_ranked_survivor_count=(
                    configured_value_promotion_count
                ),
                progress_path=progress_path,
                overwrite=args.overwrite,
            )
            observed_survivor_counts["cluster_value_policy"][str(cluster_id)][
                stage_name
            ] = stage_checkpoint["observed_counts"]
            if not eligible:
                raise RuntimeError(
                    f"all {stage_name} candidates failed for liquidity cluster {cluster_id}"
                )
            per_stage[stage_name] = [
                {
                    "candidate_index": item["candidate_index"],
                    "candidate": asdict(item["candidate"]),
                    "evaluation": evaluation_report(item["evaluation"]),
                }
                for item in evaluated
            ]
            current = [
                (int(item["candidate_index"]), item["candidate"])
                for item in promoted
            ]
        selected_index, selected_candidate = current[0]
        selected_evaluation = final_evaluations[selected_index]
        selected_policies[cluster_id] = selected_candidate
        selected_training_evaluations[cluster_id] = selected_evaluation
        cluster_reports[str(cluster_id)] = {
            "cluster_id": cluster_id,
            "cluster_label": f"liquidity_{cluster_id:02d}",
            "representative_symbols": list(representatives),
            "selected_candidate_index": selected_index,
            "selected_policy": asdict(selected_candidate),
            "selected_training_evaluation": evaluation_report(selected_evaluation),
            "frozen_global_local_controls": asdict(selected_local_controls),
            "stages": per_stage,
        }

    # Block 3: select one global shared-liquidity quote-size proxy using the
    # complete joint representative market and the now-frozen cluster policies.
    representative_policy_path = (
        output_root / "global_shared_quote_calibration" / "representative_policy.csv"
    )
    write_policy_csv(
        representative_policy_path, representative_symbols, layout, selected_policies,
        policy_source="selected_block2_cluster_value_wmm", overwrite=args.overwrite,
    )
    shared_root = output_root / "global_shared_quote_calibration"
    shared_current: list[tuple[int, SharedQuoteCandidate]] = list(enumerate(shared_candidates))
    shared_stage_reports: dict[str, list[dict[str, object]]] = {}
    shared_final_evaluations: dict[int, Mapping[str, object]] = {}
    expected_shared_quote_promotions = {
        "stage1_screen": runtime_shared_quote_stage1_promoted_count,
        "stage2_refinement": runtime_shared_quote_stage2_promoted_count,
        "stage3_full": runtime_shared_quote_stage3_promoted_count,
    }
    for stage_name, duration, seeds, survivor_count in stage_definitions:
        evaluated: list[dict[str, object]] = []
        candidate_references: list[dict[str, object]] = []
        for candidate_index, candidate in shared_current:
            candidate_dir = shared_root / stage_name / f"candidate_{candidate_index:03d}"
            evaluation = evaluate_policy_across_training_days(
                launcher=launcher, binary=binary, training_days=training_days,
                configs_by_date=representative_config_paths,
                policy=representative_policy_path, symbols=representative_symbols,
                output_dir=candidate_dir, duration=duration, seeds=seeds,
                targets_by_date=training_targets_by_stage[stage_name],
                local_controls=selected_local_controls,
                shared_quote_multiplier=(
                    candidate.multiplier if candidate.enabled else None
                ),
                enable_shared_mm=candidate.enabled, enable_value_agents=True,
                timeout_seconds=args.timeout_seconds,
            )
            evaluated.append({
                "candidate_index": candidate_index,
                "candidate": candidate,
                "evaluation": evaluation,
            })
            candidate_references.append(persist_candidate_evaluation(
                candidate_dir,
                block="global_shared_quote",
                stage=stage_name,
                cluster_id=None,
                candidate_index=candidate_index,
                candidate=candidate,
                evaluation=evaluation,
                progress_path=progress_path,
                overwrite=args.overwrite,
            ))
            calibration_details.append(stage_detail_rows(
                f"shared_{stage_name}", -1, candidate_index, candidate, evaluation,
                local_controls=selected_local_controls, shared_quote=candidate,
            ))
            if stage_name == "stage3_full":
                shared_final_evaluations[candidate_index] = evaluation
        evaluated.sort(key=candidate_selection_key)
        eligible = [
            item for item in evaluated
            if candidate_is_eligible(item)
        ]
        promoted = ranked_policy_stage_survivors(
            stage_name, eligible, survivor_count,
        )
        stage_checkpoint = persist_stage_checkpoint(
            shared_root / stage_name,
            block="global_shared_quote",
            stage=stage_name,
            cluster_id=None,
            candidate_references=candidate_references,
            promoted_candidate_indices=[
                int(item["candidate_index"]) for item in promoted
            ],
            configured_ranked_survivor_count=survivor_count,
            progress_path=progress_path,
            overwrite=args.overwrite,
        )
        observed_survivor_counts["global_shared_quote"][stage_name] = (
            stage_checkpoint["observed_counts"]
        )
        observed_promoted_count = int(
            stage_checkpoint["observed_counts"]["promoted_candidates"]
        )
        expected_promoted_count = expected_shared_quote_promotions[stage_name]
        if observed_promoted_count != expected_promoted_count:
            raise RuntimeError(
                "global shared-quote promotion trajectory failed closed at "
                f"{stage_name}: expected {expected_promoted_count} promoted "
                f"candidate(s), observed {observed_promoted_count}; diagnostics: "
                f"{shared_root / stage_name / 'stage_checkpoint.json'}"
            )
        if not eligible:
            raise RuntimeError(f"all shared-quote {stage_name} candidates failed")
        shared_stage_reports[stage_name] = [
            {
                "candidate_index": item["candidate_index"],
                "candidate": asdict(item["candidate"]),
                "evaluation": evaluation_report(item["evaluation"]),
            }
            for item in evaluated
        ]
        shared_current = [
            (int(item["candidate_index"]), item["candidate"])
            for item in promoted
        ]
    selected_shared_index, selected_shared_quote = shared_current[0]
    selected_shared_evaluation = shared_final_evaluations[selected_shared_index]

    # Make the full policy artifact first.  It is directly consumable by the
    # fragmented executable and contains every universe symbol exactly once.
    full_policy_path = output_root / "cluster_value_agent_policy.csv"
    write_policy_csv(
        full_policy_path, all_symbols, layout, selected_policies,
        policy_source="selected_block_coordinate_cluster_wmm", overwrite=args.overwrite,
    )
    frozen_heldout_config_path = output_root / "heldout_openings_frozen_backgrounds.csv"
    write_config_csv(
        frozen_heldout_config_path, pooled_training_fields, heldout_opening_rows,
        overwrite=args.overwrite,
    )
    if args.require_certification_profile:
        assert cohort_identity is not None
        try:
            artifact_checks = cohort_identity["artifact_checks"]
            assert isinstance(artifact_checks, dict)
            artifact_checks["full_universe_policy"] = cohort.validate_csv(
                full_policy_path,
                label="full-universe value-agent policy",
                project_root=project_root,
            )
            artifact_checks["frozen_heldout_runtime_universe"] = (
                cohort.validate_csv(
                    frozen_heldout_config_path,
                    label="frozen held-out runtime universe",
                    project_root=project_root,
                )
            )
        except cohort.CohortIdentityError as error:
            raise CalibrationError(str(error)) from error
    frozen_heldout_config_sha256 = sha256_file(frozen_heldout_config_path)
    frozen_empirical_input_bundle_sha256 = empirical_input_bundle_sha256(
        frozen_heldout_config_path
    )
    gate_profile = certification_profile()
    gate_profile_sha256 = certification_profile_sha256()
    runtime_profile = runtime_profile_with_acceptance_thresholds(
        gate_profile,
        maximum_robust_score=args.maximum_heldout_robust_score,
        maximum_metric_score=args.maximum_heldout_metric_score,
        maximum_two_sided_shortfall=(
            args.maximum_two_sided_coverage_shortfall
        ),
    )
    runtime_profile.update({
        "certification_profile_enforced": bool(
            args.require_certification_profile
        ),
        "required_session_duration_seconds": args.session_duration,
        "required_training_dates": [day.date for day in training_days],
        "required_validation_date": args.heldout_date,
        "required_common_symbol_count": len(all_symbols),
        "required_cluster_count": len(layout.cluster_ids),
        "required_training_representatives_per_cluster": (
            next(iter({len(layout.representatives[cluster])
                       for cluster in layout.cluster_ids}))
            if len({len(layout.representatives[cluster])
                    for cluster in layout.cluster_ids}) == 1 else
            sorted({len(layout.representatives[cluster])
                    for cluster in layout.cluster_ids})
        ),
        "required_validation_symbols_per_cluster": (
            next(iter({len(layout.validation_symbols[cluster])
                       for cluster in layout.cluster_ids}))
            if len({len(layout.validation_symbols[cluster])
                    for cluster in layout.cluster_ids}) == 1 else
            sorted({len(layout.validation_symbols[cluster])
                    for cluster in layout.cluster_ids})
        ),
        "stage1_duration_seconds": args.stage1_duration,
        "stage2_duration_seconds": args.stage2_duration,
        "required_stage1_seeds": list(args.stage1_seeds),
        "required_stage2_seeds": list(args.stage2_seeds),
        "required_stage3_seeds": list(args.stage3_seeds),
        "shared_quote_candidate_count": (
            runtime_shared_quote_candidate_count
        ),
        "shared_quote_stage1_survivor_cap": args.stage1_top_candidates,
        "shared_quote_stage1_promoted_candidates": (
            runtime_shared_quote_stage1_promoted_count
        ),
        "local_flow_stage1_refinement_leaders": args.stage1_top_candidates,
        "stage1_refinement_candidates": args.stage1_refinement_candidates,
        "shared_quote_stage2_survivor_cap": args.stage2_top_candidates,
        "shared_quote_stage2_promoted_candidates": (
            runtime_shared_quote_stage2_promoted_count
        ),
        "shared_quote_stage3_survivor_cap": 1,
        "shared_quote_stage3_promoted_candidates": (
            runtime_shared_quote_stage3_promoted_count
        ),
        "local_flow_stage1_promotion": LOCAL_FLOW_STAGE1_PROMOTION,
        "local_flow_stage2_promotion": LOCAL_FLOW_STAGE2_PROMOTION,
        "local_flow_stage3_selection": LOCAL_FLOW_STAGE3_SELECTION,
        "value_policy_stage1_promotion": VALUE_POLICY_STAGE1_PROMOTION,
        "value_policy_stage2_promotion": VALUE_POLICY_STAGE2_PROMOTION,
        "value_policy_stage1_survivors_per_depth": len(
            set(args.thresholds)
        ),
        "value_policy_stage2_survivors_per_depth": len(
            set(args.thresholds)
        ),
        "value_policy_stage3_candidates_per_cluster": (
            1
            + len(set(args.thresholds))
            * len(set(args.depth_participations))
        ),
        "value_thresholds_bps": sorted(set(args.thresholds)),
        "value_depth_participations": sorted(
            set(args.depth_participations)
        ),
        "hawkes_activity_scales": sorted(set(args.hawkes_activity_scales)),
        "local_mm_intervals_ms": sorted(set(args.local_mm_intervals_ms)),
        "local_mm_quantity_multipliers": sorted(
            set(args.local_mm_quantity_multipliers)
        ),
        "local_mm_improvement_probabilities": sorted(
            set(args.local_mm_improvement_probabilities)
        ),
        "shared_quote_multipliers": sorted(set(args.shared_quote_multipliers)),
        "shared_treatment_multiplier": args.shared_treatment_multiplier,
        "marketwide_validation_required": bool(args.marketwide_validation),
    })
    certification_runtime_profile_matched = runtime_profile == gate_profile

    # Persist the expensive three-block selection before validation begins.
    # This checkpoint is deliberately not a certified handoff: downstream
    # case-study jobs still require calibration_handoff.json, which is emitted
    # only after the declared held-out checks complete.
    calibration_detail_path = output_root / "cluster_calibration_detail.csv"
    atomic_csv(calibration_detail_path, DETAIL_FIELDS, calibration_details,
               overwrite=args.overwrite)
    selected_policy_rows = [
        {
            "cluster_id": cluster_id,
            "cluster_label": f"liquidity_{cluster_id:02d}",
            "representative_symbols": ";".join(layout.representatives[cluster_id]),
            "validation_symbols": ";".join(layout.validation_symbols[cluster_id]),
            "enabled": int(selected_policies[cluster_id].enabled),
            "value_threshold_bps": selected_policies[cluster_id].threshold_bps,
            "value_depth_participation": (
                selected_policies[cluster_id].depth_participation
            ),
            "training_fit_wsmrmse": float(
                selected_training_evaluations[cluster_id]["fit_wsmrmse"]
            ),
            "training_selection_score": evaluation_selection_score(
                selected_training_evaluations[cluster_id]
            ),
            "training_day_count": len(training_days),
            "training_aggregation": (
                "median_plus_mad_of_day_level_metric_balanced_huber"
                if len(training_days) > 1 else "single_day_metric_balanced_huber"
            ),
            "hawkes_activity_scale": selected_local_controls.hawkes_activity_scale,
            "local_mm_enabled": int(selected_local_controls.local_mm_enabled),
            "local_mm_interval_ms": selected_local_controls.local_mm_interval_ms,
            "local_mm_quantity_multiplier": (
                selected_local_controls.local_mm_quantity_multiplier
            ),
            "local_mm_improvement_probability": (
                selected_local_controls.local_mm_improvement_probability
            ),
            "shared_mm_enabled": int(selected_shared_quote.enabled),
            "shared_quote_multiplier": selected_shared_quote.multiplier,
        }
        for cluster_id in layout.cluster_ids
    ]
    selection_path = output_root / "cluster_selected_policies.csv"
    atomic_csv(
        selection_path,
        (
            "cluster_id", "cluster_label", "representative_symbols", "validation_symbols",
            "enabled", "value_threshold_bps", "value_depth_participation",
            "training_fit_wsmrmse", "training_selection_score",
            "training_day_count", "training_aggregation",
            "hawkes_activity_scale", "local_mm_enabled", "local_mm_interval_ms",
            "local_mm_quantity_multiplier", "local_mm_improvement_probability",
            "shared_mm_enabled",
            "shared_quote_multiplier",
        ),
        selected_policy_rows,
        overwrite=args.overwrite,
    )
    checkpoint_selected_policy_records = checkpoint_cluster_policy_records(
        selected_policy_rows,
        selected_training_evaluations,
        expected_summary_count_per_cluster=(
            len(training_days) * len(tuple(args.stage3_seeds))
        ),
    )
    checkpoint_path = output_root / "calibration_selection_checkpoint.json"
    atomic_json(checkpoint_path, {
        "schema_version": SELECTION_CHECKPOINT_SCHEMA_VERSION,
        "certified_for_case_study": False,
        "status": "selection_complete_validation_pending",
        "training_dates": [training_day.date for training_day in training_days],
        "training_input_provenance": training_input_provenance,
        "heldout_date": args.heldout_date,
        "validation_role": VALIDATION_ROLE,
        "independent_final_holdout": INDEPENDENT_FINAL_HOLDOUT,
        "certification_profile": gate_profile,
        "certification_profile_sha256": gate_profile_sha256,
        "observed_runtime_profile": runtime_profile,
        "observed_survivor_counts": observed_survivor_counts,
        "runtime_matches_certification_profile": (
            certification_runtime_profile_matched
        ),
        "simulator_source_semantics_sha256": simulator_semantics_sha256,
        "workflow_source_semantics_sha256": workflow_semantics_sha256,
        "calibration_build_provenance": build_provenance,
        "cluster_manifest_provenance": cluster_manifest,
        "pooling_provenance": pooling_provenance,
        "cohort_identity": cohort_identity,
        "selected_global_local_flow": {
            "candidate_index": selected_local_index,
            "controls": asdict(selected_local_controls),
            "training_evaluation": evaluation_report(selected_local_evaluation),
        },
        "selected_cluster_policies": checkpoint_selected_policy_records,
        "selected_global_shared_quote": {
            "candidate_index": selected_shared_index,
            "candidate": asdict(selected_shared_quote),
            "training_evaluation": evaluation_report(selected_shared_evaluation),
        },
        "artifacts": {
            "calibration_progress_checkpoint_json": str(progress_path),
            "full_universe_policy_csv": str(full_policy_path),
            "cluster_selected_policies_csv": str(selection_path),
            "cluster_calibration_detail_csv": str(calibration_detail_path),
            "frozen_heldout_opening_config_csv": str(frozen_heldout_config_path),
            "frozen_heldout_opening_config_sha256": frozen_heldout_config_sha256,
            "frozen_empirical_input_bundle_sha256": (
                frozen_empirical_input_bundle_sha256
            ),
        },
        "warning": (
            "This is an audit checkpoint only. calibration_handoff.json is absent "
            "until held-out validation completes."
        ),
    }, overwrite=args.overwrite)

    # Representative books keep the grid search tractable, but certification
    # also requires evidence that the selected cluster-shared policy is
    # adequate for every training book.  Run the frozen policy over the full
    # common universe on all five training dates before opening any 2020
    # development-validation target.  Failure stops here and cannot feed the
    # later date back into parameter selection.
    training_adequacy_root = output_root / "full_universe_training_adequacy"
    training_adequacy_configs = write_training_subset_configs(
        training_adequacy_root,
        training_days,
        all_symbols,
        filename="full_universe_training_config.csv",
        overwrite=args.overwrite,
    )
    training_adequacy_targets = {
        training_day.date: load_targets(
            training_day.target_root, training_day.date, all_symbols,
        )
        for training_day in training_days
    }
    training_adequacy_evaluation = evaluate_policy_across_training_days(
        launcher=launcher,
        binary=binary,
        training_days=training_days,
        configs_by_date=training_adequacy_configs,
        policy=full_policy_path,
        symbols=all_symbols,
        output_dir=training_adequacy_root / "runs",
        duration=args.stage3_duration,
        seeds=CERTIFICATION_TRAINING_ADEQUACY_SEEDS,
        targets_by_date=training_adequacy_targets,
        local_controls=selected_local_controls,
        shared_quote_multiplier=(
            selected_shared_quote.multiplier
            if selected_shared_quote.enabled else None
        ),
        enable_shared_mm=selected_shared_quote.enabled,
        enable_value_agents=True,
        timeout_seconds=args.timeout_seconds,
    )
    training_adequacy_status_path = (
        output_root / "full_universe_training_adequacy_status.json"
    )
    if not math.isfinite(
        float(training_adequacy_evaluation["fit_wsmrmse"])
    ):
        message = evaluation_failure_message(
            "full-universe training adequacy", training_adequacy_evaluation,
        )
        atomic_json(training_adequacy_status_path, {
            "schema_version": 1,
            "scope": "all_common_symbols_on_every_training_date",
            "symbol_count": len(all_symbols),
            "required_symbol_count": required_symbol_count,
            "training_dates": [day.date for day in training_days],
            "duration_seconds": args.stage3_duration,
            "seeds": list(CERTIFICATION_TRAINING_ADEQUACY_SEEDS),
            "cohort_identity": cohort_identity,
            "passed": False,
            "development_validation_targets_opened": False,
            "reason": message,
            "evaluation": evaluation_report(training_adequacy_evaluation),
        }, overwrite=args.overwrite)
        raise RuntimeError(message)
    training_adequacy = full_universe_training_adequacy_summary(
        training_adequacy_evaluation,
        maximum_score=args.maximum_heldout_robust_score,
        maximum_metric_score=args.maximum_heldout_metric_score,
        maximum_symbol_metric_absolute_residual=(
            CERTIFICATION_GROSS_RESIDUAL_LIMIT
        ),
    )
    training_adequacy_distribution_path = (
        output_root / "full_universe_training_distribution_by_day.csv"
    )
    training_distribution_rows: list[dict[str, object]] = []
    training_distribution_days = training_adequacy_evaluation.get(
        "training_day_evaluations", []
    )
    if not training_distribution_days:
        training_distribution_days = ({
            "date": training_days[0].date,
            "evaluation": evaluation_report(training_adequacy_evaluation),
        },)
    for raw_day in training_distribution_days:
        if not isinstance(raw_day, Mapping):
            raise CalibrationError(
                "full-universe training distribution has an invalid day record"
            )
        date = str(raw_day.get("date", "unknown_date"))
        day_evaluation = raw_day.get("evaluation")
        if not isinstance(day_evaluation, Mapping):
            raise CalibrationError(
                f"full-universe training distribution lacks {date} evaluation"
            )
        training_distribution_rows.extend(
            distribution_rows(
                day_evaluation.get("moment_estimates", []),
                scope=f"full_universe_training_{date}",
            )
        )
    atomic_csv(
        training_adequacy_distribution_path,
        DISTRIBUTION_FIELDS,
        training_distribution_rows,
        overwrite=args.overwrite,
    )
    atomic_json(training_adequacy_status_path, {
        "schema_version": 1,
        "scope": "all_common_symbols_on_every_training_date",
        "symbol_count": len(all_symbols),
        "required_symbol_count": required_symbol_count,
        "training_dates": [day.date for day in training_days],
        "duration_seconds": args.stage3_duration,
        "seeds": list(CERTIFICATION_TRAINING_ADEQUACY_SEEDS),
        "cohort_identity": cohort_identity,
        **training_adequacy,
        "evaluation": evaluation_report(training_adequacy_evaluation),
        "distribution_by_day_csv": str(training_adequacy_distribution_path),
        "distribution_by_day_sha256": sha256_file(
            training_adequacy_distribution_path
        ),
    }, overwrite=args.overwrite)
    if training_adequacy["passed"] is not True:
        reasons = training_adequacy.get("failure_reasons", [])
        reason_text = "; ".join(str(reason) for reason in reasons)
        raise RuntimeError(
            "full-universe training adequacy failed before development "
            f"validation: {reason_text}"
        )

    validation_symbols = tuple(
        symbol for cluster in layout.cluster_ids
        for symbol in layout.validation_symbols[cluster]
    )
    development_validation_target_bundle_sha256 = target_artifact_bundle_sha256(
        heldout_target_root, args.heldout_date, all_symbols, (None,),
    )
    heldout_validation_targets = load_targets(
        heldout_target_root, args.heldout_date, validation_symbols,
    )
    sampled_config_path = output_root / "heldout_stratified_validation_config.csv"
    write_config_csv(
        sampled_config_path, pooled_training_fields,
        subset_config_rows(heldout_opening_rows, validation_symbols),
        overwrite=args.overwrite,
    )
    sampled_policy_path = output_root / "heldout_stratified_validation_policy.csv"
    write_policy_csv(
        sampled_policy_path, validation_symbols, layout, selected_policies,
        policy_source="selected_block_coordinate_cluster_wmm", overwrite=args.overwrite,
    )
    sampled_evaluation = evaluate_policy(
        launcher=launcher, binary=binary, config=sampled_config_path,
        policy=sampled_policy_path, symbols=validation_symbols,
        output_dir=output_root / "heldout_stratified_validation",
        duration=args.stage3_duration, seeds=tuple(args.stage3_seeds),
        targets=heldout_validation_targets,
        local_controls=selected_local_controls,
        shared_quote_multiplier=(
            selected_shared_quote.multiplier if selected_shared_quote.enabled else None
        ),
        enable_shared_mm=selected_shared_quote.enabled,
        enable_value_agents=True,
        timeout_seconds=args.timeout_seconds,
    )
    sampled_status_path = output_root / "heldout_stratified_validation_status.json"
    if not math.isfinite(float(sampled_evaluation["fit_wsmrmse"])):
        message = evaluation_failure_message(
            "held-out stratified validation", sampled_evaluation,
        )
        atomic_json(sampled_status_path, {
            "schema_version": 1,
            "scope": "pooled_stratified_sample",
            "cohort_identity": cohort_identity,
            "passed": False,
            "reason": message,
            "evaluation": evaluation_report(sampled_evaluation),
        }, overwrite=args.overwrite)
        raise RuntimeError(message)
    sampled_coverage_shortfalls = two_sided_coverage_shortfalls(
        sampled_evaluation, args.maximum_two_sided_coverage_shortfall,
    )
    sampled_coverage_summary = two_sided_coverage_summary(
        sampled_evaluation, args.maximum_two_sided_coverage_shortfall,
    )
    sampled_empirical_fit = empirical_fit_summary(
        sampled_evaluation,
        maximum_score=args.maximum_heldout_robust_score,
        maximum_metric_score=args.maximum_heldout_metric_score,
        maximum_symbol_metric_absolute_residual=(
            CERTIFICATION_GROSS_RESIDUAL_LIMIT
        ),
    )
    sampled_full_two_sided_passed = (
        sampled_evaluation.get("two_sided_integrity_passed") is True
    )
    sampled_background_boundary_adequacy_passed = (
        sampled_evaluation.get("finite_boundary_adequacy_passed") is True
    )
    sampled_value_boundary_adequacy_passed = (
        sampled_evaluation.get("value_boundary_adequacy_passed") is True
    )
    sampled_boundary_adequacy_passed = (
        sampled_background_boundary_adequacy_passed
        and sampled_value_boundary_adequacy_passed
    )
    sampled_coverage_passed = (
        not sampled_coverage_shortfalls and sampled_full_two_sided_passed
    )
    sampled_structural_adequacy_passed = (
        sampled_full_two_sided_passed
        and sampled_boundary_adequacy_passed
        and sampled_coverage_passed
    )
    sampled_empirical_fit_failure_reasons = (
        empirical_fit_failure_reasons(
            STRATIFIED_EMPIRICAL_FIT_FAILURE_SCOPE, sampled_empirical_fit,
        )
        if not sampled_empirical_fit["passed"] else []
    )
    sampled_failure_reasons: list[str] = []
    if not sampled_full_two_sided_passed:
        sampled_failure_reasons.append(
            "held-out stratified run contains one-sided fixed-clock observations; "
            "certification requires invalid_sample_count=0 and simulated "
            "two_sided_sample_fraction=1 for every symbol and seed"
        )
    if sampled_coverage_shortfalls:
        examples = ", ".join(
            f"{row['symbol']}={float(row['simulated_two_sided_fraction']):.4f}"
            f"<{float(row['empirical_two_sided_fraction']):.4f}"
            for row in sampled_coverage_shortfalls[:5]
        )
        message = (
            "held-out stratified two-sided coverage misses the empirical target "
            f"by more than {args.maximum_two_sided_coverage_shortfall:g}: {examples}"
        )
        sampled_failure_reasons.append(message)
    if not sampled_boundary_adequacy_passed:
        sampled_failure_reasons.append(
            "held-out stratified run depends materially on the finite-book "
            "reflection boundary for background flow or value-agent orders"
        )
    atomic_json(sampled_status_path, {
        "schema_version": 2,
        "scope": "pooled_stratified_sample",
        "cohort_identity": cohort_identity,
        "passed": sampled_structural_adequacy_passed,
        "structural_adequacy_passed": sampled_structural_adequacy_passed,
        "execution_integrity_passed": sampled_full_two_sided_passed,
        "full_two_sided_book_passed": sampled_full_two_sided_passed,
        "coverage_passed": sampled_coverage_passed,
        "finite_boundary_adequacy_passed": sampled_boundary_adequacy_passed,
        "finite_boundary_adequacy": {
            "background": sampled_evaluation.get("finite_boundary_adequacy"),
            "value": sampled_evaluation.get("value_boundary_adequacy"),
        },
        "background_boundary_adequacy_passed": (
            sampled_background_boundary_adequacy_passed
        ),
        "value_boundary_adequacy_passed": sampled_value_boundary_adequacy_passed,
        "empirical_fit_passed": bool(sampled_empirical_fit["passed"]),
        "empirical_fit_acceptance_role": (
            STRATIFIED_EMPIRICAL_FIT_ACCEPTANCE_ROLE
        ),
        "certified_for_case_study": sampled_structural_adequacy_passed,
        "failure_reasons": sampled_failure_reasons,
        "empirical_fit_failure_reasons": (
            sampled_empirical_fit_failure_reasons
        ),
        "interpretation": (
            "This required stratified probe certifies structural adequacy only; "
            "its empirical-fit score and failures are preserved as diagnostics. "
            "The full-universe market-wide fit is authoritative."
        ),
        "coverage_summary": sampled_coverage_summary,
        "coverage_shortfalls": sampled_coverage_shortfalls,
        "empirical_fit": sampled_empirical_fit,
        "evaluation": evaluation_report(sampled_evaluation),
    }, overwrite=args.overwrite)

    validation_details: list[dict[str, object]] = []
    for estimate in sampled_evaluation["moment_estimates"]:
        row = dict(estimate)
        symbol = str(row["symbol"])
        cluster_id = layout.by_symbol[symbol]
        validation_details.append({
            "phase": "heldout_validation",
            "scope": "pooled_stratified_sample",
            "cluster_id": cluster_id,
            "cluster_label": f"liquidity_{cluster_id:02d}",
            **row,
        })
    detail_path = output_root / "heldout_validation_detail.csv"
    atomic_csv(detail_path, VALIDATION_DETAIL_FIELDS, validation_details,
               overwrite=args.overwrite)
    sampled_distribution_path = output_root / "heldout_pooled_stratified_distribution.csv"
    atomic_csv(
        sampled_distribution_path, DISTRIBUTION_FIELDS,
        distribution_rows(sampled_evaluation["moment_estimates"],
                          scope="pooled_stratified_sample_not_full_market"),
        overwrite=args.overwrite,
    )

    marketwide_report: Mapping[str, object] | None = None
    marketwide_status_path: pathlib.Path | None = None
    marketwide_distribution_path: pathlib.Path | None = None
    marketwide_coverage_summary_path: pathlib.Path | None = None
    marketwide_coverage_summary: Mapping[str, object] | None = None
    marketwide_empirical_fit: Mapping[str, object] | None = None
    marketwide_validation_completed = False
    marketwide_execution_integrity_passed = False
    marketwide_empirical_fit_passed = False
    marketwide_boundary_adequacy_passed = False
    marketwide_background_boundary_adequacy_passed = False
    marketwide_value_boundary_adequacy_passed = False
    marketwide_failure_reasons: list[str] = []
    if args.marketwide_validation:
        # This optional run is the only validation result described as
        # market-wide: every frozen held-out empirical book is simulated and
        # then reduced to distributional summaries rather than 2,000 tables.
        marketwide_targets = load_targets(
            heldout_target_root, args.heldout_date, all_symbols,
        )
        marketwide_evaluation = evaluate_policy(
            launcher=launcher, binary=binary, config=frozen_heldout_config_path,
            policy=full_policy_path, symbols=all_symbols,
            output_dir=output_root / "heldout_marketwide_validation",
            duration=args.stage3_duration, seeds=tuple(args.stage3_seeds),
            targets=marketwide_targets,
            local_controls=selected_local_controls,
            shared_quote_multiplier=(
                selected_shared_quote.multiplier if selected_shared_quote.enabled else None
            ),
            enable_shared_mm=selected_shared_quote.enabled,
            enable_value_agents=True,
            timeout_seconds=args.timeout_seconds,
        )
        marketwide_status_path = (
            output_root / "heldout_marketwide_validation_status.json"
        )
        if not math.isfinite(float(marketwide_evaluation["fit_wsmrmse"])):
            message = evaluation_failure_message(
                "held-out market-wide validation", marketwide_evaluation,
            )
            atomic_json(marketwide_status_path, {
                "schema_version": 1,
                "scope": "full_universe_marketwide",
                "symbol_count": len(all_symbols),
                "required_symbol_count": required_symbol_count,
                "validation_date": args.heldout_date,
                "duration_seconds": args.stage3_duration,
                "seeds": list(args.stage3_seeds),
                "cohort_identity": cohort_identity,
                "passed": False,
                "reason": message,
                "evaluation": evaluation_report(marketwide_evaluation),
            }, overwrite=args.overwrite)
            raise RuntimeError(message)
        marketwide_coverage_summary = two_sided_coverage_summary(
            marketwide_evaluation, args.maximum_two_sided_coverage_shortfall,
        )
        marketwide_empirical_fit = empirical_fit_summary(
            marketwide_evaluation,
            maximum_score=args.maximum_heldout_robust_score,
            maximum_metric_score=args.maximum_heldout_metric_score,
            maximum_symbol_metric_absolute_residual=(
                CERTIFICATION_GROSS_RESIDUAL_LIMIT
            ),
        )
        marketwide_empirical_fit_passed = bool(marketwide_empirical_fit["passed"])
        marketwide_execution_integrity_passed = (
            marketwide_evaluation.get("two_sided_integrity_passed") is True
        )
        marketwide_background_boundary_adequacy_passed = (
            marketwide_evaluation.get("finite_boundary_adequacy_passed") is True
        )
        marketwide_value_boundary_adequacy_passed = (
            marketwide_evaluation.get("value_boundary_adequacy_passed") is True
        )
        marketwide_boundary_adequacy_passed = (
            marketwide_background_boundary_adequacy_passed
            and marketwide_value_boundary_adequacy_passed
        )
        if not marketwide_execution_integrity_passed:
            marketwide_failure_reasons.append(
                "held-out market-wide run contains incomplete or one-sided "
                "fixed-clock observations"
            )
        if not marketwide_boundary_adequacy_passed:
            marketwide_failure_reasons.append(
                "held-out market-wide run depends materially on a finite-book "
                "reflection boundary"
            )
        if not marketwide_empirical_fit_passed:
            marketwide_failure_reasons.extend(
                empirical_fit_failure_reasons(
                    "held-out market-wide", marketwide_empirical_fit,
                )
            )
        marketwide_coverage_summary_path = (
            output_root / "heldout_marketwide_coverage_summary.json"
        )
        atomic_json(marketwide_coverage_summary_path, {
            "schema_version": 1,
            "scope": "full_universe_marketwide",
            "acceptance_role": (
                "distributional diagnostic; no family-wise all-symbol hard gate"
            ),
            **marketwide_coverage_summary,
        }, overwrite=args.overwrite)
        marketwide_distribution_path = output_root / "heldout_marketwide_distribution.csv"
        atomic_csv(
            marketwide_distribution_path, DISTRIBUTION_FIELDS,
            distribution_rows(marketwide_evaluation["moment_estimates"],
                              scope="full_universe_marketwide"),
            overwrite=args.overwrite,
        )
        marketwide_report = evaluation_report(marketwide_evaluation)
        marketwide_validation_completed = True
        marketwide_structural_adequacy_passed = (
            marketwide_execution_integrity_passed
            and marketwide_boundary_adequacy_passed
        )
        atomic_json(marketwide_status_path, {
            "schema_version": MARKETWIDE_STATUS_SCHEMA_VERSION,
            "scope": "full_universe_marketwide",
            "symbol_count": len(all_symbols),
            "required_symbol_count": required_symbol_count,
            "validation_date": args.heldout_date,
            "duration_seconds": args.stage3_duration,
            "seeds": list(args.stage3_seeds),
            "cohort_identity": cohort_identity,
            "passed": (
                marketwide_structural_adequacy_passed
                and marketwide_empirical_fit_passed
            ),
            "structural_adequacy_passed": (
                marketwide_structural_adequacy_passed
            ),
            "execution_integrity_passed": marketwide_execution_integrity_passed,
            "full_two_sided_book_passed": marketwide_execution_integrity_passed,
            "coverage_passed": marketwide_execution_integrity_passed,
            "finite_boundary_adequacy_passed": (
                marketwide_boundary_adequacy_passed
            ),
            "finite_boundary_adequacy": {
                "background": marketwide_evaluation.get(
                    "finite_boundary_adequacy"
                ),
                "value": marketwide_evaluation.get(
                    "value_boundary_adequacy"
                ),
            },
            "background_boundary_adequacy_passed": (
                marketwide_background_boundary_adequacy_passed
            ),
            "value_boundary_adequacy_passed": (
                marketwide_value_boundary_adequacy_passed
            ),
            "empirical_fit_passed": marketwide_empirical_fit_passed,
            "empirical_fit_acceptance_role": (
                MARKETWIDE_EMPIRICAL_FIT_ACCEPTANCE_ROLE
            ),
            "certified_for_case_study": (
                marketwide_structural_adequacy_passed
                and marketwide_empirical_fit_passed
            ),
            "failure_reasons": marketwide_failure_reasons,
            "interpretation": (
                "symbol-level coverage shortfalls are distributional diagnostics; "
                "the metric-balanced empirical-fit gate remains mandatory"
            ),
            "coverage_summary": marketwide_coverage_summary,
            "empirical_fit": marketwide_empirical_fit,
            "evaluation": marketwide_report,
        }, overwrite=args.overwrite)

    heldout_decision = heldout_acceptance_decision(
        marketwide_validation_completed=marketwide_validation_completed,
        sampled_execution_integrity_passed=(
            math.isfinite(float(sampled_evaluation["fit_wsmrmse"]))
            and sampled_full_two_sided_passed
        ),
        sampled_coverage_passed=sampled_coverage_passed,
        sampled_background_boundary_adequacy_passed=(
            sampled_background_boundary_adequacy_passed
        ),
        sampled_value_boundary_adequacy_passed=(
            sampled_value_boundary_adequacy_passed
        ),
        sampled_empirical_fit_passed=bool(sampled_empirical_fit["passed"]),
        marketwide_execution_integrity_passed=(
            marketwide_execution_integrity_passed
        ),
        marketwide_background_boundary_adequacy_passed=(
            marketwide_background_boundary_adequacy_passed
        ),
        marketwide_value_boundary_adequacy_passed=(
            marketwide_value_boundary_adequacy_passed
        ),
        marketwide_empirical_fit_passed=marketwide_empirical_fit_passed,
    )
    execution_integrity_passed = heldout_decision[
        "execution_integrity_passed"
    ]
    coverage_passed = heldout_decision["coverage_passed"]
    finite_boundary_adequacy_passed = heldout_decision[
        "finite_boundary_adequacy_passed"
    ]
    empirical_fit_passed = heldout_decision["empirical_fit_passed"]
    provenance_integrity_passed = all(
        value is not None
        for value in (
            build_provenance, cluster_manifest, pooling_provenance,
            cohort_identity,
        )
    )
    training_full_universe_adequacy_passed = (
        training_adequacy.get("passed") is True
    )
    certified_for_case_study = (
        certification_runtime_profile_matched
        and training_full_universe_adequacy_passed
        and heldout_decision["heldout_validation_passed"]
        and provenance_integrity_passed
    )
    certification_failure_reasons: list[str] = []
    if not certification_runtime_profile_matched:
        certification_failure_reasons.append(
            "runtime/search design does not exactly match the immutable "
            f"{CERTIFICATION_GATE_ID} profile; compare observed_runtime_profile "
            "with certification_profile in the report"
        )
    if not marketwide_validation_completed:
        certification_failure_reasons.append(
            "full-universe development validation was not completed"
        )
    if not training_full_universe_adequacy_passed:
        certification_failure_reasons.append(
            "full-universe training adequacy did not pass before development "
            "validation"
        )
    if not execution_integrity_passed:
        certification_failure_reasons.append(
            "fixed-clock execution integrity failed"
        )
    if not coverage_passed:
        certification_failure_reasons.append(
            "held-out stratified empirical two-sided coverage failed"
        )
    if not empirical_fit_passed:
        if isinstance(marketwide_empirical_fit, Mapping):
            certification_failure_reasons.extend(
                empirical_fit_failure_reasons(
                    "held-out market-wide", marketwide_empirical_fit,
                )
            )
    if not finite_boundary_adequacy_passed:
        certification_failure_reasons.append(
            "finite-book boundary adequacy gate failed"
        )
    if not provenance_integrity_passed:
        certification_failure_reasons.append(
            "build, clustering, or pooling provenance is absent"
        )
    certification = {
        "certification_profile_id": CERTIFICATION_GATE_ID,
        "certification_profile_sha256": gate_profile_sha256,
        "runtime_matches_certification_profile": certification_runtime_profile_matched,
        "validation_role": VALIDATION_ROLE,
        "independent_final_holdout": INDEPENDENT_FINAL_HOLDOUT,
        "cohort_identity_verified": cohort_identity is not None,
        "cohort_identity": cohort_identity,
        "certification_input_selection": certification_input_selection,
        "marketwide_validation_completed": marketwide_validation_completed,
        "training_full_universe_adequacy_passed": (
            training_full_universe_adequacy_passed
        ),
        "execution_integrity_passed": execution_integrity_passed,
        "full_two_sided_book_passed": (
            sampled_full_two_sided_passed and marketwide_execution_integrity_passed
        ),
        # With the hard 100% simulated two-sided requirement this is an alias,
        # not a statistically independent empirical test.  The empirical
        # coverage distribution remains in the report as a diagnostic.
        "coverage_passed": coverage_passed,
        "complete_two_sided_clock_passed": coverage_passed,
        "finite_boundary_adequacy_passed": finite_boundary_adequacy_passed,
        "finite_boundary_adequacy": {
            "stratified": {
                "background": sampled_evaluation.get(
                    "finite_boundary_adequacy"
                ),
                "value": sampled_evaluation.get("value_boundary_adequacy"),
            },
            "marketwide": (
                {
                    "background": marketwide_report.get(
                        "finite_boundary_adequacy"
                    ),
                    "value": marketwide_report.get(
                        "value_boundary_adequacy"
                    ),
                }
                if isinstance(marketwide_report, Mapping) else None
            ),
        },
        "background_boundary_adequacy_passed": (
            sampled_background_boundary_adequacy_passed
            and marketwide_background_boundary_adequacy_passed
        ),
        "value_boundary_adequacy_passed": (
            sampled_value_boundary_adequacy_passed
            and marketwide_value_boundary_adequacy_passed
        ),
        "empirical_fit_passed": empirical_fit_passed,
        "empirical_fit_acceptance_scope": "full_universe_marketwide",
        "stratified_structural_adequacy_passed": heldout_decision[
            "stratified_structural_adequacy_passed"
        ],
        "stratified_empirical_fit_passed": heldout_decision[
            "stratified_empirical_fit_passed"
        ],
        "stratified_empirical_fit_acceptance_role": (
            STRATIFIED_EMPIRICAL_FIT_ACCEPTANCE_ROLE
        ),
        "stratified_empirical_fit_failure_reasons": (
            sampled_empirical_fit_failure_reasons
        ),
        "marketwide_empirical_fit_passed": heldout_decision[
            "marketwide_empirical_fit_passed"
        ],
        "marketwide_empirical_fit_acceptance_role": (
            MARKETWIDE_EMPIRICAL_FIT_ACCEPTANCE_ROLE
        ),
        "provenance_integrity_passed": provenance_integrity_passed,
        "certified_for_case_study": certified_for_case_study,
        "failure_reasons": certification_failure_reasons,
        "heldout_parameters_frozen": True,
        "heldout_used_for_parameter_selection": False,
        "training_full_universe_adequacy": training_adequacy,
        "stratified_empirical_fit": sampled_empirical_fit,
        "marketwide_empirical_fit": marketwide_empirical_fit,
    }

    report_path = output_root / "cluster_value_agent_calibration_report.json"
    handoff_path = output_root / "calibration_handoff.json"
    preliminary_path = output_root / "preliminary_calibration_result.json"
    report: dict[str, object] = {
        "schema_version": 2,
        "certification": certification,
        "certification_profile": gate_profile,
        "certification_profile_sha256": gate_profile_sha256,
        "observed_runtime_profile": runtime_profile,
        "observed_survivor_counts": observed_survivor_counts,
        "cohort_identity": cohort_identity,
        "certification_input_selection": certification_input_selection,
        "validation_scope": {
            "role": VALIDATION_ROLE,
            "independent_final_holdout": INDEPENDENT_FINAL_HOLDOUT,
            "interpretation": (
                "Chronologically later baseline temporal-transfer and model-"
                "admissibility check. Residuals from this date informed the "
                "protocol repair, so it is not an untouched final test."
            ),
            "nonclaims": [
                "not a formal cross-sectional distribution test",
                "not validation of shock contagion",
                "not evidence that the counterfactual shock occurred in ITCH",
            ],
        },
        "protocol": {
            "name": "three_block_coordinate_metric_balanced_robust_matching",
            # ``training_date`` is retained for report-schema compatibility.
            # The plural provenance below is authoritative for multi-session
            # fitting.
            "training_date": training_days[0].date if len(training_days) == 1 else None,
            "training_dates": [training_day.date for training_day in training_days],
            "training_day_count": len(training_days),
            "training_days": training_input_provenance,
            "training_selection_aggregation": {
                "method": "median_plus_0.25_mad_of_day_metric_balanced_huber_scores",
                "reason": (
                    "each metric first receives equal weight within a day; the "
                    "median protects selection from one unusual session and the "
                    "MAD term penalises unstable candidates"
                ),
            },
            "pooled_training_universe_config": str(pooled_training_config_path),
            "pooled_training_universe_config_sha256": pooled_training_config_sha256,
            "runtime_configuration_schema": runtime_configuration_schema,
            "heldout_date": args.heldout_date,
            "validation_role": VALIDATION_ROLE,
            "independent_final_holdout": INDEPENDENT_FINAL_HOLDOUT,
            "three_horizon_screen": {
                "stage1": {
                    "duration_seconds": args.stage1_duration,
                    "purpose": "short structural and empirical-coverage screen",
                    "target": "matching first-session prefix",
                    "seeds": args.stage1_seeds,
                    "local_flow_promotion_rule": LOCAL_FLOW_STAGE1_PROMOTION,
                    "local_flow_candidates_promoted": (
                        local_stage_promotion_counts["stage1_screen"]
                    ),
                    "shared_quote_survivor_cap": args.stage1_top_candidates,
                    "shared_quote_candidates_promoted": (
                        runtime_shared_quote_stage1_promoted_count
                    ),
                    "value_policy_promotion_rule": (
                        VALUE_POLICY_STAGE1_PROMOTION
                    ),
                    "value_policy_survivors_after_stage_per_cluster": (
                        len(value_candidates)
                    ),
                    "local_flow_refinement_leaders": (
                        args.stage1_top_candidates
                    ),
                    "initial_grid_candidates": len(original_local_grid),
                    "maximum_midpoint_refinement_candidates": (
                        args.stage1_refinement_candidates
                    ),
                },
                "stage2": {
                    "duration_seconds": args.stage2_duration,
                    "purpose": "multi-seed intermediate refinement",
                    "target": "matching first-session prefix",
                    "seeds": args.stage2_seeds,
                    "local_flow_promotion_rule": LOCAL_FLOW_STAGE2_PROMOTION,
                    "local_flow_candidates_promoted": (
                        local_stage_promotion_counts["stage2_refinement"]
                    ),
                    "shared_quote_survivor_cap": args.stage2_top_candidates,
                    "shared_quote_candidates_promoted": (
                        runtime_shared_quote_stage2_promoted_count
                    ),
                    "value_policy_promotion_rule": (
                        VALUE_POLICY_STAGE2_PROMOTION
                    ),
                    "value_policy_survivors_after_stage_per_cluster": (
                        len(value_candidates)
                    ),
                },
                "stage3": {
                    "duration_seconds": args.stage3_duration,
                    "purpose": "full-session selection",
                    "target": "full-session target",
                    "seeds": args.stage3_seeds,
                    "local_flow_selection_rule": LOCAL_FLOW_STAGE3_SELECTION,
                    "local_flow_candidates_selected": (
                        local_stage_promotion_counts["stage3_full"]
                    ),
                    "shared_quote_survivor_cap": 1,
                    "shared_quote_candidates_promoted": (
                        runtime_shared_quote_stage3_promoted_count
                    ),
                    "value_policy_survivors_after_stage_per_cluster": 1,
                },
            },
            "block_coordinate_protocol": {
                "block1_global_local_flow": {
                    "parameters": [
                        "local_mm_enabled",
                        "local_mm_interval_ms",
                        "local_mm_quantity_multiplier",
                        "local_mm_improvement_probability",
                    ],
                    "fixed_direct_input": {
                        "hawkes_activity_scale": FIXED_HAWKES_ACTIVITY_SCALE,
                        "reason": (
                            "per-symbol Hawkes immigration rates were analytically "
                            "inverted from ITCH event counts at scale 0.30"
                        ),
                    },
                    "candidate_count": len(local_candidates),
                    "initial_grid_candidate_count": len(original_local_grid),
                    "refined_candidate_count": (
                        len(local_candidates) - len(original_local_grid)
                    ),
                    "objective_metrics": list(LOCAL_FLOW_METRICS),
                    "shared_market_maker": "disabled",
                    "value_agents": "disabled",
                    "scope": "joint representative market",
                    "selection": {
                        "candidate_index": selected_local_index,
                        "controls": asdict(selected_local_controls),
                        "training_evaluation": evaluation_report(
                            selected_local_evaluation
                        ),
                        "stages": local_stage_reports,
                    },
                },
                "block2_cluster_value_policy": {
                    "parameters": [
                        "enabled", "value_threshold_bps",
                        "value_depth_participation",
                    ],
                    "candidate_count_per_cluster": len(value_candidates),
                    "disabled_baseline_included": True,
                    "frozen_block1_controls": asdict(selected_local_controls),
                    "shared_market_maker": "disabled",
                    "scope": (
                        "candidate selection on training representatives, then "
                        "mandatory adequacy on every symbol and training date"
                    ),
                },
                "block3_global_shared_quote_proxy": {
                    "parameter": "shared_quote_multiplier_relative_to_empirical_symbol_size",
                    "candidate_count": len(shared_candidates),
                    "frozen_block1_controls": asdict(selected_local_controls),
                    "frozen_block2_policies": "one selected policy per cluster",
                    "shared_market_maker": (
                        "nested disabled baseline or enabled in uncoupled mode with "
                        "global quote scale fixed at one"
                    ),
                    "value_agents": "enabled with selected policy CSV",
                    "scope": "joint representative market",
                    "interpretation": (
                        "a nested off baseline plus relative mechanism intensities; "
                        "agent/background attribution and any dealer identity are "
                        "not identified by anonymous ITCH"
                    ),
                    "selection": {
                        "candidate_index": selected_shared_index,
                        "candidate": asdict(selected_shared_quote),
                        "training_evaluation": evaluation_report(
                            selected_shared_evaluation
                        ),
                        "stages": shared_stage_reports,
                    },
                },
            },
            "simulation": {
                "ranks": 1,
                "books_per_asset": 1,
                "decision_window_ms": DECISION_WINDOW_MS,
                "asset_summary_interval_ms": DECISION_WINDOW_MS,
                "final_validation_shared_market_maker": (
                    "enabled" if selected_shared_quote.enabled else "disabled"
                ),
                "final_validation_global_capacity": (
                    "uncoupled; phi fixed at one"
                    if selected_shared_quote.enabled else "not applicable; shared MM off"
                ),
                "final_validation_value_agents": "enabled with selected policies",
                "launcher": list(launcher),
            },
            "direct_book_calibration": (
                "candidate evaluation uses each training day's per-symbol ITCH event "
                "rates, empirical mark distributions, opening book inputs and local "
                "quoting inputs directly. The state-feedback anchors "
                "target_spread_ticks, target_mean_bid_depth and "
                "target_mean_ask_depth are the same five-day pooled training "
                "estimates in every training and validation runtime, rather than "
                "day-specific outcome oracles. These direct inputs are not behavioural "
                "WMM parameters. The latent fundamental volatility is likewise "
                "computed once as 10,000 times the square root of pooled training "
                "one-second return variance and remains frozen at validation. "
                "Held-out validation freezes the separately prepared "
                "pooled direct-input configuration named above. Empirical "
                "Background-order quote improvement uses the aggregate "
                "training zero-distance split I/(Z_buy+Z_sell) recorded in "
                "those per-symbol inputs; I/E is descriptive only. The local "
                "maker has a separate "
                "calibrated improvement probability, while the shared maker "
                "quotes passively at the current BBO with symbol-relative size."
            ),
            "heldout_leakage_barrier": {
                "backgrounds": "exact pooled training direct-input configuration values",
                "heldout_fields_allowed": list(HELDOUT_OPENING_FIELDS),
                "heldout_input_mode": heldout_mode,
                "heldout_targets_opened_after_selection": True,
                "heldout_targets_used_for_runtime_configuration": False,
                "pooled_homeostatic_fields_frozen": list(
                    POOLED_HOMEOSTATIC_FIELDS
                ),
            },
            # Kept for the existing certified case-study handoff loader: this
            # is the exact direct-input universe used after selection.
            "training_config_sha256": pooled_training_config_sha256,
            "heldout_config_sha256": sha256_file(heldout_config_path),
            "cluster_assignments_sha256": sha256_file(assignments_path),
            "validation_sample_sha256": sha256_file(validation_path),
            "cluster_manifest": cluster_manifest,
            "pooling_provenance": pooling_provenance,
            "cohort_identity": cohort_identity,
            "binary_sha256": sha256_file(binary),
            "calibration_build_provenance": build_provenance,
            "simulator_source_semantics_sha256": simulator_semantics_sha256,
            "workflow_source_semantics_sha256": workflow_semantics_sha256,
            "frozen_empirical_input_bundle_sha256": (
                frozen_empirical_input_bundle_sha256
            ),
            "development_validation_target_bundle_sha256": (
                development_validation_target_bundle_sha256
            ),
            "development_validation_target_root": str(heldout_target_root),
        },
        "moment_matching": {
            "name": "metric_balanced_robust_training_selection_with_raw_wmm_diagnostics",
            "day_level_selection_formula": (
                "sqrt(sum(weight * ((seed_mean_simulated - empirical_target) / "
                "empirical_scale)^2) / sum(weight))"
            ),
            "selection_formula": (
                "positive moments use log ratios; mid-move uses clipped log odds; "
                "absolute-return ACF uses a clipped Fisher transform; coverage uses "
                "a one-percentage-point difference scale. Huber losses are averaged "
                "within metric, then equally across metrics. Training days use "
                "median + 0.25 MAD"
            ),
            "selection_uses": "training sessions only; held-out parameters remain frozen",
            "selection_huber_delta": DEFAULT_ROBUST_HUBER_DELTA,
            "combined_uncertainty_diagnostic": (
                "sqrt(sum(weight * ((seed_mean_simulated - empirical_target) / "
                "hypot(empirical_scale, simulation_mc_se))^2) / sum(weight))"
            ),
            "all_model_moments": list(METRICS),
            "block1_local_flow_moments": list(LOCAL_FLOW_METRICS),
            "two_sided_coverage": {
                "role": "empirical calibration moment and held-out acceptance check",
                "structural_validity_definition": (
                    "complete fixed-clock accounting plus invalid_sample_count=0 and "
                    "simulated two_sided_sample_fraction=1 for every symbol and seed"
                ),
                "maximum_development_validation_shortfall": (
                    CERTIFICATION_MAXIMUM_TWO_SIDED_SHORTFALL
                ),
                "interpretation": (
                    "100 percent remains the target for a symbol-day whose ITCH "
                    "book is two-sided at every fixed-clock observation"
                ),
            },
        },
        "candidate_grids": {
            "global_local_flow": [asdict(candidate) for candidate in local_candidates],
            "cluster_value_policy": [asdict(candidate) for candidate in value_candidates],
            "global_shared_quote_proxy": [
                asdict(candidate) for candidate in shared_candidates
            ],
        },
        "global_local_flow_selection": {
            "candidate_index": selected_local_index,
            "controls": asdict(selected_local_controls),
            "training_evaluation": evaluation_report(selected_local_evaluation),
            "stages": local_stage_reports,
        },
        "clusters": cluster_reports,
        "global_shared_quote_selection": {
            "candidate_index": selected_shared_index,
            "candidate": asdict(selected_shared_quote),
            "training_evaluation": evaluation_report(selected_shared_evaluation),
            "stages": shared_stage_reports,
        },
        "full_universe_training_adequacy": {
            "scope": "all_common_symbols_on_every_training_date",
            "symbols": len(all_symbols),
            "dates": [day.date for day in training_days],
            "duration_seconds": args.stage3_duration,
            "seeds": list(CERTIFICATION_TRAINING_ADEQUACY_SEEDS),
            "status": training_adequacy,
            "status_json": str(training_adequacy_status_path),
            "status_sha256": sha256_file(training_adequacy_status_path),
            "distribution_by_day_csv": str(
                training_adequacy_distribution_path
            ),
            "distribution_by_day_sha256": sha256_file(
                training_adequacy_distribution_path
            ),
            "development_validation_targets_opened": False,
        },
        "heldout_stratified_validation": {
            "scope": "one or more non-representative symbols from every cluster",
            "not_a_full_market_distributional_claim": True,
            "symbols": list(validation_symbols),
            "frozen_runtime_controls": {
                **asdict(selected_local_controls),
                "shared_quote_mode": "relative_to_empirical_symbol_quote_size",
                "shared_quote_multiplier": selected_shared_quote.multiplier,
                "shared_market_maker_enabled": selected_shared_quote.enabled,
                "value_agents_enabled": True,
            },
            "evaluation": evaluation_report(sampled_evaluation),
            "coverage_summary": sampled_coverage_summary,
            "empirical_fit": sampled_empirical_fit,
            "certification": {
                "execution_integrity_passed": sampled_full_two_sided_passed,
                "full_two_sided_book_passed": sampled_full_two_sided_passed,
                "coverage_passed": sampled_coverage_passed,
                "finite_boundary_adequacy_passed": (
                    sampled_boundary_adequacy_passed
                ),
                "background_boundary_adequacy_passed": (
                    sampled_background_boundary_adequacy_passed
                ),
                "value_boundary_adequacy_passed": (
                    sampled_value_boundary_adequacy_passed
                ),
                "structural_adequacy_passed": (
                    sampled_structural_adequacy_passed
                ),
                "empirical_fit_passed": bool(sampled_empirical_fit["passed"]),
                "empirical_fit_acceptance_role": (
                    STRATIFIED_EMPIRICAL_FIT_ACCEPTANCE_ROLE
                ),
                "empirical_fit_failure_reasons": (
                    sampled_empirical_fit_failure_reasons
                ),
                "certified_for_case_study": (
                    sampled_structural_adequacy_passed
                ),
            },
            "acceptance_rule": (
                "execution integrity, empirical two-sided coverage and both "
                "source-attributed finite-boundary checks must pass. The "
                "stratified empirical-fit score is required and retained as a "
                "diagnostic; only the exact full-universe market-wide fit is "
                "the held-out empirical-fit certification gate"
            ),
        },
        "heldout_marketwide_distributional_validation": {
            "enabled": bool(args.marketwide_validation),
            "scope": (
                "full held-out universe, aggregate distributions only"
                if args.marketwide_validation else
                "not run; pooled sample output must not be called market-wide"
            ),
            "evaluation": marketwide_report,
            "coverage_summary": marketwide_coverage_summary,
            "empirical_fit": marketwide_empirical_fit,
            "acceptance_role": (
                "authoritative full-universe temporal-transfer/admissibility "
                "and held-out empirical-fit certification gate; cross-sectional "
                "summaries are descriptive and symbol-level gross residuals "
                "are retained as model-criticism diagnostics"
                if args.marketwide_validation else "not run"
            ),
        },
        "artifacts": {
            "calibration_progress_checkpoint_json": str(progress_path),
            "cluster_assignments_csv": str(assignments_path),
            "cluster_assignments_sha256": sha256_file(assignments_path),
            "validation_sample_csv": str(validation_path),
            "validation_sample_sha256": sha256_file(validation_path),
            "cluster_manifest_json": (
                cluster_manifest["path"] if cluster_manifest else None
            ),
            "cluster_manifest_sha256": (
                cluster_manifest["sha256"] if cluster_manifest else None
            ),
            "full_universe_policy_csv": str(full_policy_path),
            "representative_policy_csv": str(representative_policy_path),
            "frozen_heldout_opening_config_csv": str(frozen_heldout_config_path),
            "frozen_heldout_opening_config_sha256": frozen_heldout_config_sha256,
            "frozen_empirical_input_bundle_sha256": (
                frozen_empirical_input_bundle_sha256
            ),
            "cluster_selected_policies_csv": str(selection_path),
            "cluster_calibration_detail_csv": str(calibration_detail_path),
            "calibration_selection_checkpoint_json": str(checkpoint_path),
            "full_universe_training_adequacy_status_json": (
                str(training_adequacy_status_path)
            ),
            "full_universe_training_adequacy_status_sha256": (
                sha256_file(training_adequacy_status_path)
            ),
            "full_universe_training_distribution_by_day_csv": (
                str(training_adequacy_distribution_path)
            ),
            "full_universe_training_distribution_by_day_sha256": (
                sha256_file(training_adequacy_distribution_path)
            ),
            "heldout_stratified_validation_status_json": str(sampled_status_path),
            "heldout_stratified_validation_status_sha256": sha256_file(
                sampled_status_path
            ),
            "heldout_validation_detail_csv": str(detail_path),
            "heldout_pooled_stratified_distribution_csv": str(sampled_distribution_path),
            "heldout_marketwide_distribution_csv": (
                str(marketwide_distribution_path) if marketwide_distribution_path else None
            ),
            "heldout_marketwide_coverage_summary_json": (
                str(marketwide_coverage_summary_path)
                if marketwide_coverage_summary_path else None
            ),
            "calibration_handoff_json": (
                str(handoff_path) if certified_for_case_study else None
            ),
            "preliminary_calibration_result_json": (
                None if certified_for_case_study else str(preliminary_path)
            ),
        },
    }
    atomic_json(report_path, report, overwrite=args.overwrite)
    shared_treatment_multiplier = (
        selected_shared_quote.multiplier
        if selected_shared_quote.enabled else args.shared_treatment_multiplier
    )
    provenance_record: dict[str, object] = {
        "schema_version": 1,
        "generated_by": "calibrate_cluster_value_agents.py",
        "certification": certification,
        "certification_profile": gate_profile,
        "certification_profile_sha256": gate_profile_sha256,
        "observed_runtime_profile": runtime_profile,
        "observed_survivor_counts": observed_survivor_counts,
        "cohort_identity": cohort_identity,
        "certification_input_selection": certification_input_selection,
        "calibration_progress_checkpoint": str(progress_path),
        "validation_role": VALIDATION_ROLE,
        "independent_final_holdout": INDEPENDENT_FINAL_HOLDOUT,
        "calibration_report": str(report_path),
        "calibration_report_sha256": sha256_file(report_path),
        # Existing case-study scripts consume these names and intentionally
        # certify the pooled direct-input configuration used after selection.
        "training_universe_config": str(pooled_training_config_path),
        "training_universe_config_sha256": pooled_training_config_sha256,
        "pooled_training_universe_config": str(pooled_training_config_path),
        "pooled_training_universe_config_sha256": pooled_training_config_sha256,
        "runtime_configuration_schema": runtime_configuration_schema,
        "frozen_heldout_opening_config": str(frozen_heldout_config_path),
        "frozen_heldout_opening_config_sha256": frozen_heldout_config_sha256,
        "frozen_empirical_input_bundle_sha256": (
            frozen_empirical_input_bundle_sha256
        ),
        "calibration_binary_sha256": sha256_file(binary),
        "simulator_source_semantics_sha256": simulator_semantics_sha256,
        "workflow_source_semantics_sha256": workflow_semantics_sha256,
        "calibration_build_provenance": build_provenance,
        "cluster_manifest": cluster_manifest,
        "pooling_provenance": pooling_provenance,
        "training_days": training_input_provenance,
        "development_validation_date": args.heldout_date,
        "development_validation_target_root": str(heldout_target_root),
        "development_validation_target_bundle_sha256": (
            development_validation_target_bundle_sha256
        ),
        "training_selection_aggregation": (
            "median_plus_0.25_mad_of_day_metric_balanced_huber_scores"
        ),
        "full_universe_training_adequacy": {
            "passed": training_full_universe_adequacy_passed,
            "status_json": str(training_adequacy_status_path),
            "status_sha256": sha256_file(training_adequacy_status_path),
            "distribution_by_day_csv": str(
                training_adequacy_distribution_path
            ),
            "distribution_by_day_sha256": sha256_file(
                training_adequacy_distribution_path
            ),
            "symbols": len(all_symbols),
            "training_dates": [day.date for day in training_days],
            "duration_seconds": args.stage3_duration,
            "seeds": list(CERTIFICATION_TRAINING_ADEQUACY_SEEDS),
            "development_validation_targets_opened": False,
        },
        "heldout_stratified_validation": {
            "passed": sampled_structural_adequacy_passed,
            "structural_adequacy_passed": (
                sampled_structural_adequacy_passed
            ),
            "empirical_fit_passed": bool(sampled_empirical_fit["passed"]),
            "empirical_fit_acceptance_role": (
                STRATIFIED_EMPIRICAL_FIT_ACCEPTANCE_ROLE
            ),
            "empirical_fit_failure_reasons": (
                sampled_empirical_fit_failure_reasons
            ),
            "status_json": str(sampled_status_path),
            "status_sha256": sha256_file(sampled_status_path),
            "symbols": len(validation_symbols),
            "validation_date": args.heldout_date,
            "duration_seconds": args.stage3_duration,
            "seeds": list(args.stage3_seeds),
        },
        "heldout_marketwide_validation": {
            "passed": (
                marketwide_validation_completed
                and marketwide_execution_integrity_passed
                and marketwide_boundary_adequacy_passed
                and marketwide_empirical_fit_passed
            ),
            "status_json": (
                str(marketwide_status_path)
                if marketwide_status_path is not None else None
            ),
            "status_sha256": (
                sha256_file(marketwide_status_path)
                if marketwide_status_path is not None else None
            ),
            "symbols": len(all_symbols),
            "validation_date": args.heldout_date,
            "duration_seconds": args.stage3_duration,
            "seeds": list(args.stage3_seeds),
            "empirical_fit_acceptance_role": (
                MARKETWIDE_EMPIRICAL_FIT_ACCEPTANCE_ROLE
            ),
        },
        "value_agent_policy_csv": str(full_policy_path),
        "value_agent_policy_sha256": sha256_file(full_policy_path),
        "shock_cluster_csv": str(assignments_path),
        "shock_cluster_csv_sha256": sha256_file(assignments_path),
        "validation_sample_csv": str(validation_path),
        "validation_sample_sha256": sha256_file(validation_path),
        "runtime_controls": {
            "hawkes_activity_scale": selected_local_controls.hawkes_activity_scale,
            "local_market_maker_enabled": selected_local_controls.local_mm_enabled,
            "local_mm_interval_ms": selected_local_controls.local_mm_interval_ms,
            "local_mm_quantity_multiplier": (
                selected_local_controls.local_mm_quantity_multiplier
            ),
            "local_mm_improvement_probability": (
                selected_local_controls.local_mm_improvement_probability
            ),
            "shared_market_maker_enabled": selected_shared_quote.enabled,
            "shared_quote_mode": "relative_to_empirical_symbol_quote_size",
            "shared_quote_multiplier": selected_shared_quote.multiplier,
            "shared_quote_levels": 1,
            "decision_window_ms": DECISION_WINDOW_MS,
        },
        "agent_enablement": {
            "local_market_maker": selected_local_controls.local_mm_enabled,
            "shared_market_maker": selected_shared_quote.enabled,
            "value_agents": True,
        },
        "mechanism_treatments": {
            "shared_market_maker": {
                "enabled": True,
                "quote_mode": "relative_to_empirical_symbol_quote_size",
                "quote_multiplier": shared_treatment_multiplier,
                "selected_by_training_fit": selected_shared_quote.enabled,
                "interpretation": (
                    "selected model control" if selected_shared_quote.enabled else
                    "explicit nonzero case-study scenario; not calibrated"
                ),
            },
        },
        "claim_boundary": (
            "Anonymous ITCH identifies aggregate background/book inputs, including "
            "the aggregate background-order distance-zero split, but does not "
            "identify its side/state allocation or which participant supplied "
            "the orders. The local-MM improvement probability "
            "is therefore selected separately, and local-MM/shared-MM off baselines "
            "are nested explicitly. "
            "Any nonzero shared-MM treatment is a mechanism scenario, not an estimate "
            "of an identifiable real dealer."
        ),
    }
    result: dict[str, object] = {
        "report": str(report_path),
        "certified_for_case_study": certified_for_case_study,
        "full_universe_policy": str(full_policy_path),
        "clusters": len(layout.cluster_ids),
        "training_days": len(training_days),
        "pooled_training_universe_config": str(pooled_training_config_path),
        "stratified_validation_symbols": len(validation_symbols),
        "marketwide_validation": bool(args.marketwide_validation),
        "calibration_progress": str(progress_path),
    }
    if certified_for_case_study:
        provenance_record["artifact_role"] = "certified_calibration_handoff"
        atomic_json(handoff_path, provenance_record, overwrite=args.overwrite)
        if args.overwrite and preliminary_path.exists():
            preliminary_path.unlink()
        result["handoff"] = str(handoff_path)
    else:
        provenance_record["artifact_role"] = "preliminary_not_certified"
        provenance_record["warning"] = (
            "Development-validation certification was not granted: "
            + "; ".join(certification_failure_reasons)
            + ". This artifact is not a normal calibration handoff and requires "
            "ALLOW_PRELIMINARY_MODEL=on downstream."
        )
        atomic_json(preliminary_path, provenance_record, overwrite=args.overwrite)
        if args.overwrite and handoff_path.exists():
            handoff_path.unlink()
        result["preliminary_result"] = str(preliminary_path)
    append_calibration_progress(
        progress_path,
        {
            "kind": "calibration_complete",
            "certified_for_case_study": certified_for_case_study,
            "calibration_report": str(report_path),
        },
        status="complete",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(args, parser)
    try:
        result = run(args)
    except (CalibrationError, FileNotFoundError, RuntimeError, OSError) as error:
        failure_note = ""
        try:
            failure_path = persist_calibration_failure(
                pathlib.Path(args.output_dir).expanduser().resolve(), error,
            )
            failure_note = f"; diagnostics: {failure_path}"
        except (CalibrationError, OSError, ValueError) as diagnostic_error:
            failure_note = (
                "; additionally failed to persist calibration_failure.json: "
                f"{diagnostic_error}"
            )
        print(
            f"cluster value-agent calibration failed: {error}{failure_note}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(json_safe(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
