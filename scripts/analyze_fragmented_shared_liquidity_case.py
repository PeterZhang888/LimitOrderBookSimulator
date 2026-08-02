#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Analyse paired global-capacity, uncoupled, and shared-MM-off stress paths.

Continuous non-target depth and spread are the primary endpoints.  The main
contrast is (global shock-control) minus (uncoupled shock-control), which keeps
ordinary shared quotes and local inventory skew in both mechanisms.  The
shared-MM-off pair is retained only as a secondary negative control.
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
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import certification_cohort as cohort  # noqa: E402


REQUIRED_METRIC_FIELDS = frozenset(
    {
        "time_seconds",
        "affected_unshocked_fraction",
        "two_sided_book_fraction",
        "mean_spread_bps",
        "mean_top_depth",
        "unshocked_mean_spread_bps",
        "unshocked_mean_top_depth",
        "shared_gross_exposure",
        "shared_utilization",
        "shared_quote_scale",
        "unshocked_shared_requested_quote_depth",
        "mean_shocked_shared_inventory",
        "value_agent_order_count",
        "value_agent_requested_quantity",
    }
)


class AnalysisError(ValueError):
    """Raised for a missing, mixed, or non-paired experimental input."""


@dataclass(frozen=True)
class RawCase:
    row: Mapping[str, str]
    seed: int
    risk_limit: float
    shock: bool
    metrics_path: pathlib.Path


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: str, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"invalid {field}: {value!r}") from error
    if not math.isfinite(result):
        raise AnalysisError(f"non-finite {field}: {value!r}")
    return result


def positive_int(value: str, field: str) -> int:
    result = finite_float(value, field)
    if result <= 0.0 or not result.is_integer():
        raise AnalysisError(f"{field} must be a positive integer: {value!r}")
    return int(result)


def canonical_risk(value: float) -> str:
    return format(value, ".12g")


def read_raw(path: pathlib.Path, expected_mm_mode: str, rank: int | None) -> list[RawCase]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
    except OSError as error:
        raise AnalysisError(f"cannot read raw result CSV {path}: {error}") from error
    if not rows:
        raise AnalysisError(f"raw result CSV is empty: {path}")

    required = {
        "shared_mm_mode", "shock_mode", "seed", "ranks",
        "risk_limit_per_asset", "metrics_csv", "input_mode",
        "metrics_csv_sha256", "executable", "executable_sha256",
        "campaign_manifest", "campaign_manifest_sha256",
        "input_config_sha256", "requested_window_ms",
        "requested_shock_time_seconds", "requested_shock_fraction",
        "requested_shock_top_depth_multiple",
        "requested_shock_target_count",
        "requested_shock_target_seed", "requested_local_inventory_limit",
        "requested_capacity_threshold", "shock_cluster_sha256",
        "requested_shared_quote_relative", "requested_shared_quote_multiplier",
        "requested_shared_quote_levels", "requested_local_mm_enabled",
        "requested_value_agent_enabled", "hawkes_activity_scale",
        "local_mm_interval_ms", "local_mm_quantity_multiplier",
        "local_mm_improvement_probability",
        "local_mm_spread_elasticity",
        "local_mm_max_improvement_probability",
        "shared_quote_quantity", "value_agent_policy_sha256",
    }
    missing = required.difference(rows[0])
    if missing:
        raise AnalysisError(
            f"{path} was produced by an older runner and is missing: "
            + ", ".join(sorted(missing))
        )

    selected: list[RawCase] = []
    for number, row in enumerate(rows, start=2):
        if row.get("shared_mm_mode") != expected_mm_mode:
            raise AnalysisError(
                f"{path}:{number} has shared_mm_mode={row.get('shared_mm_mode')!r}; "
                f"expected {expected_mm_mode!r}"
            )
        if row.get("input_mode") != "empirical_universe":
            raise AnalysisError(
                f"{path}:{number} is not an empirical-universe result"
            )
        case_rank = positive_int(row.get("ranks", ""), f"{path}:{number}:ranks")
        if rank is not None and case_rank != rank:
            continue
        shock_text = row.get("shock_mode", "")
        if shock_text not in {"on", "off"}:
            raise AnalysisError(f"{path}:{number} has invalid shock_mode={shock_text!r}")
        metrics_text = row.get("metrics_csv", "").strip()
        if not metrics_text:
            raise AnalysisError(
                f"{path}:{number} lacks metrics_csv; rerun with --metrics-dir"
            )
        metrics_path = pathlib.Path(metrics_text)
        if not metrics_path.is_file():
            raise AnalysisError(
                f"{path}:{number} refers to missing metrics CSV: {metrics_path}"
            )
        recorded_metrics_hash = row.get("metrics_csv_sha256", "")
        if sha256_file(metrics_path) != recorded_metrics_hash:
            raise AnalysisError(
                f"{path}:{number} metrics CSV SHA-256 mismatch: {metrics_path}"
            )
        executable = pathlib.Path(row.get("executable", ""))
        if not executable.is_file():
            raise AnalysisError(
                f"{path}:{number} refers to a missing executable: {executable}"
            )
        if sha256_file(executable) != row.get("executable_sha256", ""):
            raise AnalysisError(
                f"{path}:{number} executable SHA-256 mismatch: {executable}"
            )
        campaign_manifest = pathlib.Path(row.get("campaign_manifest", ""))
        if not campaign_manifest.is_file():
            raise AnalysisError(
                f"{path}:{number} refers to a missing campaign manifest"
            )
        if sha256_file(campaign_manifest) != row.get(
                "campaign_manifest_sha256", ""):
            raise AnalysisError(
                f"{path}:{number} campaign-manifest SHA-256 mismatch"
            )
        selected.append(
            RawCase(
                row=row,
                seed=positive_int(row.get("seed", ""), f"{path}:{number}:seed"),
                risk_limit=finite_float(
                    row.get("risk_limit_per_asset", ""),
                    f"{path}:{number}:risk_limit_per_asset",
                ),
                shock=shock_text == "on",
                metrics_path=metrics_path,
            )
        )
    if not selected:
        qualifier = f" at ranks={rank}" if rank is not None else ""
        raise AnalysisError(f"no usable rows found in {path}{qualifier}")
    return selected


def ensure_common_metadata(cases: Iterable[RawCase]) -> dict[str, str]:
    keys = (
        "executable_sha256",
        "campaign_manifest",
        "campaign_manifest_sha256",
        "input_config_sha256",
        "requested_window_ms",
        "requested_shock_time_seconds",
        "requested_shock_fraction",
        "requested_shock_top_depth_multiple",
        "requested_shock_target_count",
        "requested_shock_target_seed",
        "requested_local_inventory_limit",
        "requested_capacity_threshold",
        "shock_cluster_sha256",
        "requested_shared_quote_relative",
        "requested_shared_quote_multiplier",
        "requested_shared_quote_levels",
        "requested_local_mm_enabled",
        "requested_value_agent_enabled",
        "hawkes_activity_scale",
        "local_mm_interval_ms",
        "local_mm_quantity_multiplier",
        "local_mm_improvement_probability",
        "local_mm_spread_elasticity",
        "local_mm_max_improvement_probability",
        "shared_quote_quantity",
        "value_agent_policy_sha256",
    )
    rows = list(cases)
    metadata: dict[str, str] = {}
    for key in keys:
        values = {case.row.get(key, "") for case in rows}
        if "" in values:
            # An absent policy hash is meaningful only for a uniformly disabled
            # value-agent ablation.  Enabled paths must identify the exact
            # policy artifact used by every paired run.
            if (key == "value_agent_policy_sha256" and values == {""}
                    and metadata.get("requested_value_agent_enabled") == "0"):
                metadata[key] = ""
                continue
            raise AnalysisError(f"paired cases have a missing value for {key}")
        if len(values) != 1:
            raise AnalysisError(
                f"paired cases mix {key}: {', '.join(sorted(values))}"
            )
        metadata[key] = values.pop()
    return metadata


def read_metrics(path: pathlib.Path) -> dict[str, dict[str, float]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            missing = REQUIRED_METRIC_FIELDS.difference(reader.fieldnames or ())
            if missing:
                raise AnalysisError(
                    f"metrics CSV {path} is missing: {', '.join(sorted(missing))}"
                )
            result: dict[str, dict[str, float]] = {}
            for line_number, row in enumerate(reader, start=2):
                time_text = row.get("time_seconds", "")
                time_value = finite_float(time_text, f"{path}:{line_number}:time_seconds")
                key = format(time_value, ".12g")
                if key in result:
                    raise AnalysisError(f"duplicate decision time {time_text!r} in {path}")
                values = {
                    field: finite_float(row.get(field, ""), f"{path}:{line_number}:{field}")
                    for field in REQUIRED_METRIC_FIELDS
                }
                result[key] = values
    except OSError as error:
        raise AnalysisError(f"cannot read metrics CSV {path}: {error}") from error
    if not result:
        raise AnalysisError(f"metrics CSV is empty: {path}")
    return result


def atomic_csv(path: pathlib.Path,
               fieldnames: Sequence[str],
               rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: pathlib.Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def unique_cases(cases: Iterable[RawCase], label: str) -> dict[tuple[str, int, bool], RawCase]:
    result: dict[tuple[str, int, bool], RawCase] = {}
    for case in cases:
        key = (canonical_risk(case.risk_limit), case.seed, case.shock)
        if key in result:
            raise AnalysisError(
                f"duplicate {label} case for risk={key[0]}, seed={case.seed}, shock={key[2]}"
            )
        result[key] = case
    return result


def unique_off_cases(cases: Iterable[RawCase]) -> dict[tuple[int, bool], RawCase]:
    result: dict[tuple[int, bool], RawCase] = {}
    for case in cases:
        key = (case.seed, case.shock)
        if key in result:
            raise AnalysisError(
                "shared-MM-off raw results contain more than one risk setting for "
                f"seed={case.seed}, shock={case.shock}; submit that negative control once"
            )
        result[key] = case
    return result


def mean(values: Sequence[float], label: str) -> float:
    if not values:
        raise AnalysisError(f"cannot calculate {label} from an empty series")
    return statistics.fmean(values)


def sample_standard_deviation(values: Sequence[float]) -> float:
    """Return the across-seed sample standard deviation (zero for one seed)."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def percentile(values: Sequence[float], probability: float, label: str) -> float:
    """Linearly interpolated empirical percentile across paired seed effects."""
    if not values:
        raise AnalysisError(f"cannot calculate {label} from an empty series")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_analysis(global_cases: list[RawCase],
                   uncoupled_cases: list[RawCase],
                   off_cases: list[RawCase],
                   shock_time_seconds: float,
                   horizon_seconds: float) -> tuple[
                       list[dict[str, object]],
                       list[dict[str, object]],
                       list[dict[str, object]],
                   ]:
    global_index = unique_cases(global_cases, "globally constrained shared MM")
    uncoupled_index = unique_off_cases(uncoupled_cases)
    off_index = unique_off_cases(off_cases)
    risks = sorted({canonical_risk(case.risk_limit) for case in global_cases}, key=float)
    seeds = sorted({case.seed for case in global_cases})
    uncoupled_seeds = {case.seed for case in uncoupled_cases}
    off_seeds = {case.seed for case in off_cases}
    if set(seeds) != off_seeds or set(seeds) != uncoupled_seeds:
        raise AnalysisError(
            "path seeds do not match across global, uncoupled, and off modes"
        )

    metrics_cache: dict[pathlib.Path, dict[str, dict[str, float]]] = {}
    def metrics(case: RawCase) -> dict[str, dict[str, float]]:
        if case.metrics_path not in metrics_cache:
            metrics_cache[case.metrics_path] = read_metrics(case.metrics_path)
        return metrics_cache[case.metrics_path]

    per_time: list[dict[str, object]] = []
    per_seed: list[dict[str, object]] = []
    for risk in risks:
        for seed in seeds:
            required_global = {
                shock: global_index.get((risk, seed, shock)) for shock in (False, True)
            }
            required_uncoupled = {
                shock: uncoupled_index.get((seed, shock)) for shock in (False, True)
            }
            required_off = {
                shock: off_index.get((seed, shock)) for shock in (False, True)
            }
            if any(case is None for case in required_global.values()):
                raise AnalysisError(f"missing global shock/control pair for risk={risk}, seed={seed}")
            if any(case is None for case in required_uncoupled.values()):
                raise AnalysisError(f"missing uncoupled shock/control pair for seed={seed}")
            if any(case is None for case in required_off.values()):
                raise AnalysisError(f"missing shared-MM-off shock/control pair for seed={seed}")
            global_control, global_shock = required_global[False], required_global[True]
            uncoupled_control, uncoupled_shock = (
                required_uncoupled[False], required_uncoupled[True]
            )
            off_control, off_shock = required_off[False], required_off[True]
            assert (global_control and global_shock and uncoupled_control
                    and uncoupled_shock and off_control and off_shock)

            series = [metrics(case) for case in (
                global_control, global_shock, uncoupled_control,
                uncoupled_shock, off_control, off_shock,
            )]
            reference_times = set(series[0])
            if any(set(candidate) != reference_times for candidate in series[1:]):
                raise AnalysisError(
                    f"metrics decision times differ within paired cases for risk={risk}, seed={seed}"
                )
            for control_index, shock_index, label in (
                (0, 1, "global"), (2, 3, "uncoupled"), (4, 5, "off"),
            ):
                for time_key in reference_times:
                    if float(time_key) > shock_time_seconds:
                        continue
                    for field in REQUIRED_METRIC_FIELDS:
                        left = series[control_index][time_key][field]
                        right = series[shock_index][time_key][field]
                        if not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12):
                            raise AnalysisError(
                                f"{label} shock/control paths differ before t_s "
                                f"for risk={risk}, seed={seed}, time={time_key}, field={field}"
                            )
            post_times = sorted(
                (float(value) for value in reference_times
                 if shock_time_seconds < float(value) <= shock_time_seconds + horizon_seconds)
            )
            if not post_times:
                raise AnalysisError(
                    f"no decision times fall in the requested post-shock horizon for risk={risk}, seed={seed}"
                )
            depth_did_values: list[float] = []
            spread_did_values: list[float] = []
            affected_did_values: list[float] = []
            global_depth_values: list[float] = []
            global_spread_values: list[float] = []
            off_delta_values: list[float] = []
            global_two_sided_shock: list[float] = []
            global_quote_scale_shock: list[float] = []
            for time_value in post_times:
                time_key = format(time_value, ".12g")
                (global_control_values, global_shock_values,
                 uncoupled_control_values, uncoupled_shock_values,
                 off_control_values, off_shock_values) = (
                    candidate[time_key] for candidate in series
                )
                global_depth = (
                    global_control_values["unshocked_mean_top_depth"]
                    - global_shock_values["unshocked_mean_top_depth"]
                )
                uncoupled_depth = (
                    uncoupled_control_values["unshocked_mean_top_depth"]
                    - uncoupled_shock_values["unshocked_mean_top_depth"]
                )
                global_spread = (
                    global_shock_values["unshocked_mean_spread_bps"]
                    - global_control_values["unshocked_mean_spread_bps"]
                )
                uncoupled_spread = (
                    uncoupled_shock_values["unshocked_mean_spread_bps"]
                    - uncoupled_control_values["unshocked_mean_spread_bps"]
                )
                global_affected = (
                    global_shock_values["affected_unshocked_fraction"]
                    - global_control_values["affected_unshocked_fraction"]
                )
                uncoupled_affected = (
                    uncoupled_shock_values["affected_unshocked_fraction"]
                    - uncoupled_control_values["affected_unshocked_fraction"]
                )
                off_delta = off_shock_values["affected_unshocked_fraction"] - (
                    off_control_values["affected_unshocked_fraction"]
                )
                depth_did = global_depth - uncoupled_depth
                spread_did = global_spread - uncoupled_spread
                affected_did = global_affected - uncoupled_affected
                depth_did_values.append(depth_did)
                spread_did_values.append(spread_did)
                affected_did_values.append(affected_did)
                global_depth_values.append(global_depth)
                global_spread_values.append(global_spread)
                off_delta_values.append(off_delta)
                global_two_sided_shock.append(
                    global_shock_values["two_sided_book_fraction"]
                )
                global_quote_scale_shock.append(global_shock_values["shared_quote_scale"])
                per_time.append({
                    "risk_limit_per_asset": risk,
                    "seed": seed,
                    "time_seconds": f"{time_value:.9f}",
                    "global_depth_deterioration": global_depth,
                    "uncoupled_depth_deterioration": uncoupled_depth,
                    "depth_difference_in_differences": depth_did,
                    "global_spread_deterioration_bps": global_spread,
                    "uncoupled_spread_deterioration_bps": uncoupled_spread,
                    "spread_difference_in_differences_bps": spread_did,
                    "affected_fraction_difference_in_differences": affected_did,
                    "shared_off_affected_shock_minus_control": off_delta,
                    "global_shock_two_sided_book_fraction": global_shock_values[
                        "two_sided_book_fraction"],
                    "global_shock_shared_utilization": global_shock_values[
                        "shared_utilization"],
                    "global_shock_shared_quote_scale": global_shock_values[
                        "shared_quote_scale"],
                    "global_shock_shared_gross_exposure": global_shock_values[
                        "shared_gross_exposure"],
                    "global_shock_unshocked_shared_requested_quote_depth": (
                        global_shock_values[
                            "unshocked_shared_requested_quote_depth"
                        ]
                    ),
                    "global_shock_mean_shocked_shared_inventory": (
                        global_shock_values["mean_shocked_shared_inventory"]
                    ),
                    "target_shared_inventory_shock_minus_control": (
                        global_shock_values["mean_shocked_shared_inventory"]
                        - global_control_values["mean_shocked_shared_inventory"]
                    ),
                    "global_shock_value_agent_order_count": global_shock_values[
                        "value_agent_order_count"],
                    "global_shock_value_agent_requested_quantity": (
                        global_shock_values["value_agent_requested_quantity"]
                    ),
                    "global_shock_unshocked_mean_top_depth": global_shock_values[
                        "unshocked_mean_top_depth"],
                    "global_shock_unshocked_mean_spread_bps": global_shock_values[
                        "unshocked_mean_spread_bps"],
                })
            requested = finite_float(
                global_shock.row.get("shock_requested_quantity", ""),
                "shock_requested_quantity",
            )
            executed = finite_float(
                global_shock.row.get("shock_executed_quantity", ""),
                "shock_executed_quantity",
            )
            shared_absorbed = finite_float(
                global_shock.row.get("shock_shared_mm_quantity", ""),
                "shock_shared_mm_quantity",
            )
            per_seed.append({
                "risk_limit_per_asset": risk,
                "seed": seed,
                "post_shock_horizon_seconds": horizon_seconds,
                "post_shock_observations": len(post_times),
                "mean_depth_difference_in_differences": mean(depth_did_values, "mean depth DID"),
                "peak_depth_difference_in_differences": max(depth_did_values),
                "integrated_depth_difference_in_differences": sum(depth_did_values)
                    * horizon_seconds / len(depth_did_values),
                "mean_spread_difference_in_differences_bps": mean(
                    spread_did_values, "mean spread DID"),
                "peak_spread_difference_in_differences_bps": max(spread_did_values),
                "integrated_spread_difference_in_differences_bps_seconds": (
                    sum(spread_did_values) * horizon_seconds / len(spread_did_values)
                ),
                "mean_affected_fraction_difference_in_differences": mean(
                    affected_did_values, "mean affected DID"),
                "mean_global_depth_deterioration": mean(global_depth_values, "global depth"),
                "mean_global_spread_deterioration_bps": mean(global_spread_values, "global spread"),
                "mean_shared_mm_off_shock_minus_control": mean(off_delta_values, "mean off delta"),
                "minimum_two_sided_book_fraction_on_shock": min(global_two_sided_shock),
                "minimum_shared_quote_scale_on_shock": min(global_quote_scale_shock),
                "shock_requested_quantity": requested,
                "shock_executed_quantity": executed,
                "shock_shared_mm_quantity": shared_absorbed,
                "shock_execution_fraction": executed / requested if requested > 0.0 else 0.0,
                "shared_mm_absorption_fraction": (
                    shared_absorbed / executed if executed > 0.0 else 0.0
                ),
            })

    summaries: list[dict[str, object]] = []
    by_risk: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in per_seed:
        by_risk[str(row["risk_limit_per_asset"])].append(row)
    for risk in risks:
        rows = by_risk[risk]
        depth_effects = [
            float(row["mean_depth_difference_in_differences"]) for row in rows
        ]
        spread_effects = [
            float(row["mean_spread_difference_in_differences_bps"]) for row in rows
        ]
        affected_effects = [
            float(row["mean_affected_fraction_difference_in_differences"])
            for row in rows
        ]
        summaries.append({
            "risk_limit_per_asset": risk,
            "seed_count": len(rows),
            "mean_depth_difference_in_differences": mean(depth_effects, "summary depth DID"),
            "sample_sd_depth_difference_in_differences": sample_standard_deviation(
                depth_effects
            ),
            "percentile_2p5_depth_difference_in_differences": percentile(
                depth_effects, 0.025, "depth DID percentile"
            ),
            "percentile_97p5_depth_difference_in_differences": percentile(
                depth_effects, 0.975, "depth DID percentile"
            ),
            "minimum_seed_mean_depth_difference_in_differences": min(
                depth_effects
            ),
            "maximum_seed_mean_depth_difference_in_differences": max(
                depth_effects
            ),
            "mean_spread_difference_in_differences_bps": mean(
                spread_effects, "summary spread DID"
            ),
            "sample_sd_spread_difference_in_differences_bps": (
                sample_standard_deviation(spread_effects)
            ),
            "percentile_2p5_spread_difference_in_differences_bps": percentile(
                spread_effects, 0.025, "spread DID percentile"
            ),
            "percentile_97p5_spread_difference_in_differences_bps": percentile(
                spread_effects, 0.975, "spread DID percentile"
            ),
            "mean_affected_fraction_difference_in_differences": mean(
                affected_effects, "summary affected-fraction DID"
            ),
            "sample_sd_affected_fraction_difference_in_differences": (
                sample_standard_deviation(affected_effects)
            ),
            "mean_peak_depth_difference_in_differences": mean(
                [float(row["peak_depth_difference_in_differences"]) for row in rows], "summary peak DID"
            ),
            "mean_minimum_two_sided_book_fraction_on_shock": mean(
                [float(row["minimum_two_sided_book_fraction_on_shock"]) for row in rows],
                "summary two-sided fraction",
            ),
            "mean_minimum_shared_quote_scale_on_shock": mean(
                [float(row["minimum_shared_quote_scale_on_shock"]) for row in rows],
                "summary quote scale",
            ),
            "mean_shared_mm_absorption_fraction": mean(
                [float(row["shared_mm_absorption_fraction"]) for row in rows],
                "summary absorption fraction",
            ),
        })
    return per_time, per_seed, summaries


def validate_universe_input(
    path: pathlib.Path,
    metadata: Mapping[str, str],
) -> dict[str, object]:
    """Verify the certified campaign manifest against every paired raw row."""
    if not path.is_file():
        raise AnalysisError(f"universe-input manifest is missing: {path}")
    observed_hash = sha256_file(path)
    if metadata.get("campaign_manifest_sha256") != observed_hash:
        raise AnalysisError(
            "raw rows were not produced from the supplied universe-input manifest"
        )
    recorded_path = pathlib.Path(metadata.get("campaign_manifest", "")).resolve()
    if recorded_path != path:
        raise AnalysisError(
            "raw rows refer to a different campaign manifest path"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot parse universe-input manifest: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 4:
        raise AnalysisError("unsupported universe-input manifest schema")
    provenance_mode = payload.get("calibration_provenance_mode")
    if provenance_mode not in {
            "block_coordinate_certified_handoff",
            "queue_reactive_training_freeze_and_heldout_validation",
    }:
        raise AnalysisError(
            "financial analysis requires a certified calibration handoff or "
            "a passed queue-reactive held-out validation handoff"
        )
    config_path = pathlib.Path(str(payload.get("universe_config", ""))).resolve()
    if not config_path.is_file():
        raise AnalysisError("universe-input configuration is missing")
    config_hash = sha256_file(config_path)
    if (payload.get("universe_config_sha256") != config_hash
            or metadata.get("input_config_sha256") != config_hash):
        raise AnalysisError("universe configuration hash disagrees across artifacts")
    expected_runtime_fields = [
        "book_id", "symbol", "data_dir", "hawkes_rates_file",
        "fundamental_price_ticks", "fundamental_volatility_bps_sqrt_second",
        "fundamental_move_probability_per_second",
        "fundamental_conditional_kurtosis",
        "initial_best_bid_ticks",
        "initial_best_ask_ticks", "initial_best_bid_depth",
        "initial_best_ask_depth", "beta", "basket_weight",
        "market_maker_quote_quantity", "target_spread_ticks",
        "quote_improvement_probability", "target_mean_bid_depth",
        "target_mean_ask_depth",
    ]
    schema_hash = hashlib.sha256(json.dumps(
        expected_runtime_fields, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    expected_runtime_schema = {
        "schema_version": 5,
        "fields": expected_runtime_fields,
        "sha256": schema_hash,
        "pooled_homeostatic_fields": [
            "target_spread_ticks", "target_mean_bid_depth",
            "target_mean_ask_depth",
        ],
        "latent_value_fields": [
            "fundamental_volatility_bps_sqrt_second",
            "fundamental_move_probability_per_second",
            "fundamental_conditional_kurtosis",
        ],
        "frozen_training_derived_fields": [
            "target_spread_ticks", "target_mean_bid_depth",
            "target_mean_ask_depth",
            "fundamental_volatility_bps_sqrt_second",
            "fundamental_move_probability_per_second",
            "fundamental_conditional_kurtosis",
        ],
        "heldout_target_files_used": False,
    }
    if payload.get("runtime_configuration_schema") != expected_runtime_schema:
        raise AnalysisError(
            "universe-input manifest lacks the certified runtime schema"
        )
    try:
        with config_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            config_fields = list(reader.fieldnames or [])
            config_rows = list(reader)
    except OSError as error:
        raise AnalysisError(f"cannot inspect universe configuration: {error}") from error
    if config_fields != expected_runtime_fields or not config_rows:
        raise AnalysisError("universe configuration has an unsupported runtime schema")
    project_root = pathlib.Path(__file__).resolve().parents[1]
    try:
        observed_cohort = cohort.validate_symbols(
            (row.get("symbol", "") for row in config_rows),
            label="case-study universe input",
            project_root=project_root,
        )
        persisted_cohort = cohort.require_identity_record(
            payload.get("cohort_identity"),
            label="universe-input cohort identity",
        )
    except cohort.CohortIdentityError as error:
        raise AnalysisError(str(error)) from error
    if (payload.get("book_count") != cohort.REQUIRED_SYMBOL_COUNT
            or persisted_cohort.get("symbol_order_sha256")
                != observed_cohort.get("symbol_order_sha256")):
        raise AnalysisError(
            "universe-input manifest does not bind the exact certification cohort"
        )
    for line_number, row in enumerate(config_rows, start=2):
        for field in (
                "target_spread_ticks", "target_mean_bid_depth",
                "target_mean_ask_depth"):
            try:
                target = float(row.get(field, ""))
            except ValueError as error:
                raise AnalysisError(
                    f"nonnumeric {field} in universe configuration line {line_number}"
                ) from error
            if not math.isfinite(target) or target <= 0.0:
                raise AnalysisError(
                    f"nonpositive {field} in universe configuration line {line_number}"
                )
        try:
            latent_volatility = float(row.get(
                "fundamental_volatility_bps_sqrt_second", ""
            ))
        except ValueError as error:
            raise AnalysisError(
                f"nonnumeric latent volatility in universe configuration line {line_number}"
            ) from error
        if not math.isfinite(latent_volatility) or latent_volatility < 0.0:
            raise AnalysisError(
                f"negative latent volatility in universe configuration line {line_number}"
            )
        try:
            latent_move_probability = float(row.get(
                "fundamental_move_probability_per_second", ""
            ))
        except ValueError as error:
            raise AnalysisError(
                "nonnumeric latent move probability in universe configuration "
                f"line {line_number}"
            ) from error
        if (not math.isfinite(latent_move_probability)
                or not 0.0 <= latent_move_probability <= 1.0):
            raise AnalysisError(
                "out-of-range latent move probability in universe configuration "
                f"line {line_number}"
            )
        try:
            latent_conditional_kurtosis = float(row.get(
                "fundamental_conditional_kurtosis", ""
            ))
        except ValueError as error:
            raise AnalysisError(
                "nonnumeric latent conditional kurtosis in universe configuration "
                f"line {line_number}"
            ) from error
        if (not math.isfinite(latent_conditional_kurtosis)
                or latent_conditional_kurtosis < 1.0):
            raise AnalysisError(
                "invalid latent conditional kurtosis in universe configuration "
                f"line {line_number}"
            )
    executable_path = pathlib.Path(str(payload.get("case_executable", ""))).resolve()
    if not executable_path.is_file():
        raise AnalysisError("case-study executable in universe-input is missing")
    executable_hash = sha256_file(executable_path)
    if (payload.get("case_executable_sha256") != executable_hash
            or metadata.get("executable_sha256") != executable_hash):
        raise AnalysisError("case executable hash disagrees across artifacts")
    profile = payload.get("case_study_protocol")
    if not isinstance(profile, dict) or profile.get("profile_id") != (
            "systemic_liquidity_case_v1"):
        raise AnalysisError("universe-input lacks the canonical case-study profile")
    encoded = json.dumps(
        profile, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    if payload.get("case_study_protocol_sha256") != hashlib.sha256(encoded).hexdigest():
        raise AnalysisError("case-study protocol hash is invalid")
    expected_metadata = {
        "requested_shock_time_seconds": profile.get("shock_time_seconds"),
        "requested_shock_fraction": profile.get("shock_fraction"),
        "requested_shock_top_depth_multiple": profile.get(
            "shock_top_depth_multiple"
        ),
        "requested_shock_target_count": profile.get("shock_target_count"),
        "requested_shock_target_seed": profile.get("shock_target_seed"),
        "requested_local_inventory_limit": profile.get("local_inventory_limit"),
        "requested_capacity_threshold": profile.get("capacity_threshold"),
        "local_mm_spread_elasticity": profile.get(
            "local_mm_spread_elasticity"
        ),
        "local_mm_max_improvement_probability": profile.get(
            "local_mm_max_improvement_probability"
        ),
    }
    for field, expected in expected_metadata.items():
        try:
            observed = float(metadata[field])
            expected_float = float(expected)
        except (KeyError, TypeError, ValueError) as error:
            raise AnalysisError(
                f"invalid canonical profile comparison for {field}"
            ) from error
        if not math.isclose(observed, expected_float, rel_tol=0.0, abs_tol=1.0e-12):
            raise AnalysisError(
                f"raw scenario {field} differs from the campaign manifest"
            )
    try:
        observed_window = float(metadata["requested_window_ms"])
        allowed_windows = (
            [float(value) for value in profile.get("cadence_windows_ms", [])]
            if profile.get("experiment") in {"cadence", "all"}
            else [float(profile["decision_window_ms"])]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisError("campaign manifest has invalid cadence metadata") from error
    if not any(math.isclose(
            observed_window, value, rel_tol=0.0, abs_tol=1.0e-12
    ) for value in allowed_windows):
        raise AnalysisError(
            "raw decision-window cadence is not allowed by the campaign manifest"
        )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-raw", type=pathlib.Path, required=True)
    parser.add_argument("--uncoupled-raw", type=pathlib.Path, required=True)
    parser.add_argument("--shared-off-raw", type=pathlib.Path, required=True)
    parser.add_argument(
        "--universe-input", type=pathlib.Path, required=True,
        help="universe_input.json recorded by the case-study submission",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--shock-time-seconds", type=float, required=True)
    parser.add_argument("--horizon-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--rank", type=int,
        help="select one rank count when a raw file contains more than one",
    )
    args = parser.parse_args(argv)
    if (not math.isfinite(args.shock_time_seconds)
            or args.shock_time_seconds < 0.0):
        raise SystemExit("--shock-time-seconds must be finite and non-negative")
    if not math.isfinite(args.horizon_seconds) or args.horizon_seconds <= 0.0:
        raise SystemExit("--horizon-seconds must be finite and positive")
    if args.rank is not None and args.rank <= 0:
        raise SystemExit("--rank must be positive")

    try:
        global_path = args.global_raw.resolve()
        uncoupled_path = args.uncoupled_raw.resolve()
        off_path = args.shared_off_raw.resolve()
        global_cases = read_raw(global_path, "global", args.rank)
        uncoupled_cases = read_raw(uncoupled_path, "uncoupled", args.rank)
        off_cases = read_raw(off_path, "off", args.rank)
        metadata = ensure_common_metadata(
            [*global_cases, *uncoupled_cases, *off_cases]
        )
        universe_input_path = args.universe_input.resolve()
        universe_input = validate_universe_input(universe_input_path, metadata)
        profile = universe_input["case_study_protocol"]
        assert isinstance(profile, dict)
        if args.rank is None or args.rank != int(profile["science_ranks"]):
            raise AnalysisError(
                "analysis rank does not match the canonical campaign profile"
            )
        global_risks = sorted({case.risk_limit for case in global_cases})
        science_risks = sorted(float(value) for value in profile["science_risk_limits"])
        reference_risk = float(profile["reference_risk_limit"])
        permitted_risk_sets = [science_risks]
        if profile.get("experiment") in {"cadence", "all"}:
            permitted_risk_sets.append([reference_risk])
        if not any(
            len(global_risks) == len(expected)
            and all(math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
                    for left, right in zip(global_risks, expected))
            for expected in permitted_risk_sets
        ):
            raise AnalysisError(
                "global risk-limit cases do not match the canonical campaign profile"
            )
        if any(not math.isclose(
                case.risk_limit, reference_risk, rel_tol=0.0, abs_tol=1.0e-12
        ) for case in [*uncoupled_cases, *off_cases]):
            raise AnalysisError(
                "uncoupled/off controls do not use the canonical reference risk limit"
            )
        expected_seeds = {
            int(profile["base_seed"]) + offset
            for offset in range(int(profile["repetitions"]))
        }
        if {case.seed for case in global_cases} != expected_seeds:
            raise AnalysisError(
                "paired path seeds do not match the canonical campaign profile"
            )
        recorded_shock_time = finite_float(
            metadata["requested_shock_time_seconds"], "requested_shock_time_seconds"
        )
        if not math.isclose(recorded_shock_time, args.shock_time_seconds, abs_tol=1e-9):
            raise AnalysisError(
                "--shock-time-seconds does not match the recorded scenario metadata"
            )
        per_time, per_seed, summaries = build_analysis(
            global_cases, uncoupled_cases, off_cases,
            args.shock_time_seconds, args.horizon_seconds
        )
        output_dir = args.output_dir.resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise AnalysisError(
                f"refusing to mix results into non-empty output directory: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_csv(
            output_dir / "paired_liquidity_effects_by_time.csv",
            list(per_time[0].keys()), per_time,
        )
        atomic_csv(
            output_dir / "paired_liquidity_effects_by_seed.csv",
            list(per_seed[0].keys()), per_seed,
        )
        atomic_csv(
            output_dir / "paired_liquidity_effect_summary.csv",
            list(summaries[0].keys()), summaries,
        )
        manifest = {
            "schema_version": 3,
            "analysis": "paired_post_shock_difference_in_differences",
            "primary_outcomes": [
                "unshocked_mean_top_depth", "unshocked_mean_spread_bps"
            ],
            "formula": (
                "(global_shock - global_control) - "
                "(uncoupled_shock - uncoupled_control)"
            ),
            "shock_time_seconds": args.shock_time_seconds,
            "post_shock_horizon_seconds": args.horizon_seconds,
            "rank_filter": args.rank,
            "global_raw": {
                "path": str(global_path), "sha256": sha256_file(global_path)
            },
            "uncoupled_raw": {
                "path": str(uncoupled_path), "sha256": sha256_file(uncoupled_path)
            },
            "shared_off_raw": {"path": str(off_path), "sha256": sha256_file(off_path)},
            "scenario_metadata": metadata,
            "universe_input": {
                "path": str(universe_input_path),
                "sha256": sha256_file(universe_input_path),
                "case_study_protocol_sha256": universe_input[
                    "case_study_protocol_sha256"
                ],
                "case_executable_sha256": universe_input[
                    "case_executable_sha256"
                ],
            },
            "seed_count": len({case.seed for case in global_cases}),
            "risk_limits": sorted(
                {canonical_risk(case.risk_limit) for case in global_cases}, key=float
            ),
            "outputs": {
                "time_series": "paired_liquidity_effects_by_time.csv",
                "seed_effects": "paired_liquidity_effects_by_seed.csv",
                "summary": "paired_liquidity_effect_summary.csv",
            },
        }
        atomic_json(output_dir / "analysis_manifest.json", manifest)
        print(json.dumps({"output_dir": str(output_dir), "summary": summaries}, sort_keys=True))
    except AnalysisError as error:
        print(f"shared-liquidity analysis failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
