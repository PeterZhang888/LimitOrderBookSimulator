#!/usr/bin/env python3
"""Deterministic two-level calibration for the queue-reactive LOB model.

This driver calibrates one ordinary-market realisation at a time.  Independent
selection realisations may execute concurrently, while full-universe
realisations may use the exact whole-book MPI decomposition.  Shared market
making is disabled in every command.  Selection reads only five explicitly
named training sessions:

1. 300-second local-flow screen (activity, spread and depth);
2. 1,800-second per-cluster value/volatility block coordinate step;
3. 23,400-second finalist comparison across every training date and seed.

The ``train`` subcommand writes a non-authorizing panel-selection freeze and
has no held-out arguments.  ``expand-full-universe`` applies that selection to
the complete 2019 universe, performs a bounded cluster-level moment refinement
when the panel parameters do not generalise, and then applies the immutable
strict gate.  Only a passed expanded freeze can authorize ``heldout``, which
copies the 2020 opening state but no held-out flow parameters.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import pathlib
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping, Sequence


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import calibrate_cluster_value_agents as legacy  # noqa: E402
import evaluate_strict_model_validation as strict  # noqa: E402


SCHEMA_VERSION = 1
TRAINING_DAY_COUNT = 5
STAGE1_DURATION = 300
STAGE2_DURATION = 3_600
STAGE3_DURATION = 23_400
WINDOW_MS = 1_000
STAGE1_METRICS = (
    "background_event_rate",
    "mean_spread_ticks",
    "mean_bid_depth",
    "mean_ask_depth",
    "two_sided_sample_fraction",
)
STAGE2_METRICS = (
    "mean_spread_ticks",
    "mean_bid_depth",
    "mean_ask_depth",
    "mid_move_rate",
    "return_variance",
    "return_kurtosis",
    "absolute_return_acf1",
    "two_sided_sample_fraction",
)
STAGE3_METRICS = strict.METRICS
OPENING_FIELDS = (
    "fundamental_price_ticks",
    "initial_best_bid_ticks",
    "initial_best_ask_ticks",
    "initial_best_bid_depth",
    "initial_best_ask_depth",
)
VOLATILITY_FIELDS = (
    "fundamental_log_variance_persistence",
    "fundamental_log_variance_std",
    "fundamental_order_flow_coupling",
)
TARGET_DIRECTORY_FIELD = "target_data_dir"
MODEL_MARK_FILES = (
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
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class CalibrationDriverError(RuntimeError):
    """Input or protocol error that must fail closed."""


@dataclass(frozen=True)
class DatedPath:
    day: str
    path: pathlib.Path


@dataclass(frozen=True)
class LocalCandidate:
    identifier: str
    enabled: bool
    interval_ms: float
    quantity_multiplier: float
    improvement_probability: float
    spread_elasticity: float = 0.0
    max_improvement_probability: float = 1.0
    must_promote_after_short_screen: bool = False


@dataclass(frozen=True)
class ValueCandidate:
    identifier: str
    enabled: bool
    threshold_bps: float
    depth_participation: float
    trigger_mode: str
    maximum_news_rechecks: int
    gap_elasticity: float = 0.0
    max_depth_participation: float = 1.0


@dataclass(frozen=True)
class VolatilityCandidate:
    identifier: str
    variance_scale: float
    persistence: float
    std: float
    excess_kurtosis_share: float
    # A bounded transmission correction for fourth-moment attenuation between
    # latent fundamental news and realized LOB returns.  It is deliberately
    # separate from ``excess_kurtosis_share``: the latter remains a genuine
    # allocation in [0,1], while this multiplier defaults to the exact legacy
    # value one and leaves the innovation variance unchanged.
    tail_transmission_multiplier: float = 1.0
    # Direct loading theta of a persistent, session-recentred Hawkes
    # immigration baseline.  ``std`` remains the latent volatility-state
    # control; it is not multiplied into this activity loading.  The historical
    # field name is retained in CSV and checkpoints to keep old evidence
    # readable.
    order_flow_coupling: float = 0.0


@dataclass(frozen=True)
class CandidateProtocol:
    local: tuple[LocalCandidate, ...]
    value: tuple[ValueCandidate, ...]
    volatility: tuple[VolatilityCandidate, ...]
    stage1_seeds: tuple[int, ...]
    stage2_seeds: tuple[int, ...]
    stage2_confirmation_seeds: tuple[int, ...]
    stage3_seeds: tuple[int, ...]
    stage1_survivors: int
    stage2_confirmation_count: int
    full_day_confirmation_candidate_cap: int
    full_day_recheck_counts: tuple[int, ...]
    mandatory_full_day_joint_candidates: tuple[str, ...]
    global_alternatives_per_cluster: int
    global_beam_width: int
    global_stage3_finalist_count: int
    timeout_seconds: Mapping[str, float]


@dataclass(frozen=True)
class ConfigTable:
    path: pathlib.Path
    fields: tuple[str, ...]
    rows: tuple[Mapping[str, str], ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(str(row["symbol"]) for row in self.rows)


def finite_float(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise CalibrationDriverError(f"{label} is not numeric") from error
    if not math.isfinite(result):
        raise CalibrationDriverError(f"{label} is not finite")
    return result


def exact_integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise CalibrationDriverError(f"{label} is not an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise CalibrationDriverError(f"{label} is not an integer") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise CalibrationDriverError(f"{label} is not an exact integer")
    result = int(numeric)
    if result < minimum:
        raise CalibrationDriverError(f"{label} must be at least {minimum}")
    return result


def normalized_date(value: str, *, label: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as error:
        raise CalibrationDriverError(
            f"{label} must be a real ISO date: {value!r}"
        ) from error


def normalized_symbol(value: object, *, label: str) -> str:
    symbol = str(value).strip().upper()
    if not symbol or any(character.isspace() for character in symbol):
        raise CalibrationDriverError(f"{label} is not a valid symbol")
    return symbol


def safe_identifier(value: object, *, label: str) -> str:
    result = str(value).strip()
    if not SAFE_ID.fullmatch(result):
        raise CalibrationDriverError(
            f"{label} must match {SAFE_ID.pattern!r}: {result!r}"
        )
    return result


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def verified_six_component_certificate(
    path: pathlib.Path,
) -> dict[str, object]:
    """Verify the visible retrospective protocol-revision certificate.

    The certificate never substitutes for evaluation.  It binds the original
    failed nine-metric report and metric table so the revised six-component
    gate cannot erase or overwrite the evidence that motivated the revision.
    """
    certificate_path = path.expanduser().resolve()
    try:
        payload = json.loads(certificate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationDriverError(
            f"cannot read six-component certificate {certificate_path}: {error}"
        ) from error
    expected_dimensions = [
        "activity",
        "spread",
        "combined_top_depth",
        "mid_move_rate",
        "return_variance",
        "absolute_return_acf1",
    ]
    protocol = payload.get("protocol")
    if (
        payload.get("schema_version") != 1
        or payload.get("status")
            != "six_component_training_adequacy_passed"
        or payload.get("passed") is not True
        or not isinstance(protocol, Mapping)
        or protocol.get("classification")
            != "retrospective_development_reanalysis"
        or protocol.get("primary_dimensions") != expected_dimensions
        or protocol.get("marketwide_metrics_are_authoritative") is not True
        or protocol.get("return_kurtosis_role") != "diagnostic_only"
        or protocol.get("cluster_metric_role") != "diagnostic_only"
        or protocol.get("acf_distribution_moment_role") != "diagnostic_only"
        or protocol.get("structural_book_checks_remain_mandatory") is not True
    ):
        raise CalibrationDriverError(
            "six-component certificate has an invalid protocol contract"
        )
    dates = payload.get("date_results")
    if not isinstance(dates, list) or len(dates) != TRAINING_DAY_COUNT \
            or any(not isinstance(record, Mapping) or record.get("passed") is not True
                   for record in dates):
        raise CalibrationDriverError(
            "six-component certificate does not pass every training date"
        )
    evidence = payload.get("source_evidence")
    if not isinstance(evidence, Mapping):
        raise CalibrationDriverError(
            "six-component certificate lacks source evidence"
        )
    verified_sources: dict[str, dict[str, str]] = {}
    for key in ("strict_report", "marketwide_scores"):
        source_path = pathlib.Path(str(evidence.get(key, ""))).resolve()
        expected_hash = str(evidence.get(f"{key}_sha256", ""))
        if (
            not source_path.is_file()
            or not expected_hash
            or sha256_file(source_path) != expected_hash
        ):
            raise CalibrationDriverError(
                f"six-component source evidence is missing or hash-mismatched: "
                f"{source_path}"
            )
        verified_sources[key] = {
            "path": str(source_path),
            "sha256": expected_hash,
        }
    original_report = json.loads(
        pathlib.Path(verified_sources["strict_report"]["path"])
        .read_text(encoding="utf-8")
    )
    if (
        original_report.get("evaluation_role") != "training_fit"
        or original_report.get("passed") is not False
        or original_report.get("gate", {}).get("gate_id")
            != "strict_queue_reactive_fit_gate_v1"
    ):
        raise CalibrationDriverError(
            "certificate must retain the visible failed strict-nine-v1 report"
        )
    strict_path = pathlib.Path(verified_sources["strict_report"]["path"])
    residual_path = strict_path.parent / "symbol_residuals.csv"
    if not residual_path.is_file():
        raise CalibrationDriverError(
            f"six-component source residuals are missing: {residual_path}"
        )
    with residual_path.open(newline="", encoding="utf-8") as source:
        residual_rows = list(csv.DictReader(source))
    with pathlib.Path(
        verified_sources["marketwide_scores"]["path"]
    ).open(newline="", encoding="utf-8") as source:
        score_rows = list(csv.DictReader(source))
    expected_dates = {
        str(record.get("date")) for record in dates
        if isinstance(record, Mapping)
    }
    certificate_dates = sorted(expected_dates)
    if (
        sorted(str(day) for day in original_report.get("expected_dates", []))
            != certificate_dates
        or sorted(
            str(record.get("date"))
            for record in original_report.get("date_results", [])
            if isinstance(record, Mapping)
        ) != certificate_dates
    ):
        raise CalibrationDriverError(
            "six-component certificate dates disagree with the bound "
            "strict-nine-v1 report"
        )
    raw_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    by_symbol: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in residual_rows:
        day = str(row.get("date", ""))
        metric = str(row.get("metric", ""))
        symbol = str(row.get("symbol", ""))
        if day not in expected_dates or metric not in strict.METRICS or not symbol:
            raise CalibrationDriverError(
                "six-component residual evidence contains an unexpected row"
            )
        raw_by_key.setdefault((day, metric), []).append(row)
        by_symbol.setdefault((day, symbol), {})[metric] = row
    published_scores = {
        (str(row["date"]), str(row["metric"])): row
        for row in score_rows
    }
    for key, rows in raw_by_key.items():
        residuals = [
            finite_float(row["robust_residual"], label=f"{key} residual")
            for row in rows
        ]
        recomputed = strict.residual_score(residuals)
        published = published_scores.get(key)
        if published is None or abs(
            recomputed
            - finite_float(published["score"], label=f"{key} published score")
        ) > 1.0e-12:
            raise CalibrationDriverError(
                f"six-component residual evidence disagrees with {key} score"
            )

    authoritative_results: list[dict[str, object]] = []
    for day in sorted(expected_dates):
        symbols = sorted(
            symbol for candidate_day, symbol in by_symbol
            if candidate_day == day
        )
        if not symbols:
            raise CalibrationDriverError(
                f"six-component residual evidence lacks {day}"
            )
        residuals_by_metric: dict[str, list[float]] = {
            metric: [] for metric in strict.SIX_COMPONENT_METRICS
        }
        for symbol in symbols:
            metrics = by_symbol[(day, symbol)]
            if set(metrics) != set(strict.METRICS):
                raise CalibrationDriverError(
                    f"six-component residual evidence is incomplete for "
                    f"{day}/{symbol}"
                )
            for metric in strict.SIX_COMPONENT_METRICS:
                if metric == strict.COMBINED_DEPTH_METRIC:
                    bid = metrics["mean_bid_depth"]
                    ask = metrics["mean_ask_depth"]
                    target = finite_float(
                        bid["target"], label=f"{day}/{symbol} bid target"
                    ) + finite_float(
                        ask["target"], label=f"{day}/{symbol} ask target"
                    )
                    simulated = finite_float(
                        bid["simulated_seed_mean"],
                        label=f"{day}/{symbol} bid simulated",
                    ) + finite_float(
                        ask["simulated_seed_mean"],
                        label=f"{day}/{symbol} ask simulated",
                    )
                    residual = strict.robust_residual(
                        metric, simulated, target
                    )
                else:
                    residual = finite_float(
                        metrics[metric]["robust_residual"],
                        label=f"{day}/{symbol}/{metric} residual",
                    )
                residuals_by_metric[metric].append(residual)
        component_scores = {
            metric: strict.residual_score(values)
            for metric, values in residuals_by_metric.items()
        }
        component_coverage = {
            metric: sum(
                abs(value) <= strict.GROSS_RESIDUAL_LIMIT + 1.0e-12
                for value in values
            ) / len(values)
            for metric, values in residuals_by_metric.items()
        }
        robust_score = math.sqrt(2.0 * statistics.fmean(
            statistics.fmean(strict.huber_loss(value) for value in values)
            for values in residuals_by_metric.values()
        ))
        gross_fraction = sum(
            any(
                abs(residuals_by_metric[metric][index])
                    > strict.GROSS_RESIDUAL_LIMIT + 1.0e-12
                for metric in strict.SIX_COMPONENT_METRICS
            )
            for index in range(len(symbols))
        ) / len(symbols)
        passed = (
            robust_score <= strict.MAX_MARKETWIDE_ROBUST_SCORE + 1.0e-12
            and all(
                score <= strict.MAX_MARKETWIDE_METRIC_SCORE + 1.0e-12
                for score in component_scores.values()
            )
            and all(
                coverage
                    >= strict.MIN_PER_METRIC_SYMBOL_FRACTION_WITHIN_LIMIT
                        - 1.0e-12
                for coverage in component_coverage.values()
            )
            and gross_fraction
                <= strict.MAX_SYMBOL_ANY_GROSS_FAILURE_FRACTION + 1.0e-12
        )
        if not passed:
            raise CalibrationDriverError(
                f"authoritative six-component reanalysis failed for {day}"
            )
        authoritative_results.append({
            "date": day,
            "passed": True,
            "marketwide_robust_score": robust_score,
            "component_scores": component_scores,
            "component_coverage": component_coverage,
            "gross_failure_symbol_fraction": gross_fraction,
        })
    return {
        "path": str(certificate_path),
        "sha256": sha256_file(certificate_path),
        "verified_source_evidence": verified_sources,
        "symbol_residuals": {
            "path": str(residual_path),
            "sha256": sha256_file(residual_path),
        },
        "authoritative_reanalysis": authoritative_results,
        "classification": "retrospective_development_protocol_revision",
    }


def transitive_runtime_artifacts(
    *,
    configs: Sequence[pathlib.Path],
    background_policies: Sequence[pathlib.Path],
    value_policies: Sequence[pathlib.Path],
    executables: Sequence[pathlib.Path],
    summaries: Sequence[pathlib.Path],
    selection_records: Sequence[pathlib.Path],
    workflow_sources: Sequence[pathlib.Path],
    candidate_config: pathlib.Path,
    cluster_map: pathlib.Path,
) -> dict[str, object]:
    """Hash every direct and transitive file consumed during selection."""
    roles_by_path: dict[pathlib.Path, set[str]] = {}

    def add(path: pathlib.Path, role: str) -> None:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise CalibrationDriverError(
                f"transitive runtime artifact is missing: {resolved} ({role})"
            )
        roles_by_path.setdefault(resolved, set()).add(role)

    add(candidate_config, "candidate_protocol")
    add(cluster_map, "cluster_assignment")
    for source in sorted(set(path.resolve() for path in workflow_sources)):
        add(source, "calibration_workflow_source")
    for executable in sorted(set(path.resolve() for path in executables)):
        add(executable, "simulator_executable")
    for summary in sorted(set(path.resolve() for path in summaries)):
        add(summary, "calibration_asset_summary")
    for record in sorted(set(path.resolve() for path in selection_records)):
        add(record, "calibration_selection_record")
    for value_policy in sorted(set(path.resolve() for path in value_policies)):
        add(value_policy, "value_policy")
    for config_path in sorted(set(path.resolve() for path in configs)):
        add(config_path, "runtime_config")
        table = read_config(config_path)
        for row in table.rows:
            symbol = str(row["symbol"])
            add(
                pathlib.Path(str(row["hawkes_rates_file"])),
                f"{symbol}:stationary_hawkes_rates",
            )
            data_dir = pathlib.Path(str(row["data_dir"]))
            for filename in MODEL_MARK_FILES:
                add(data_dir / filename, f"{symbol}:empirical_mark:{filename}")
            target_dir = str(row.get(TARGET_DIRECTORY_FIELD, "")).strip()
            if target_dir:
                target_root = pathlib.Path(target_dir)
                target_paths = sorted(target_root.glob("market_targets_*.csv"))
                manifest_paths = sorted(target_root.glob("itch_manifest_*.json"))
                if not target_paths or not manifest_paths:
                    raise CalibrationDriverError(
                        f"target_data_dir lacks targets/manifests: {target_root}"
                    )
                for target_path in target_paths:
                    add(target_path, f"{symbol}:empirical_calibration_target")
                for manifest_path in manifest_paths:
                    add(manifest_path, f"{symbol}:empirical_target_manifest")
    for background_policy in sorted(
        set(path.resolve() for path in background_policies)
    ):
        add(background_policy, "queue_policy_mapping")
        policy_manifest = background_policy.parent / "training_policy_manifest.json"
        if policy_manifest.is_file():
            add(policy_manifest, "queue_policy_fit_manifest")
            try:
                fit_manifest = json.loads(
                    policy_manifest.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                raise CalibrationDriverError(
                    f"cannot read queue-policy fit manifest: {error}"
                ) from error
            workflow_record = fit_manifest.get("workflow_source")
            if isinstance(workflow_record, Mapping):
                source_path = pathlib.Path(
                    str(workflow_record.get("path", ""))
                ).expanduser().resolve()
                expected_source_hash = str(
                    workflow_record.get("sha256", "")
                )
                if (
                    not source_path.is_file()
                    or sha256_file(source_path) != expected_source_hash
                ):
                    raise CalibrationDriverError(
                        "queue-policy fitter source is missing or hash-mismatched"
                    )
                add(source_path, "queue_policy_fit_workflow_source")
            cluster_policy_files = fit_manifest.get("cluster_policy_files", {})
            if isinstance(cluster_policy_files, Mapping):
                for cluster_id, relative_path in cluster_policy_files.items():
                    policy_audit = pathlib.Path(str(relative_path))
                    if not policy_audit.is_absolute():
                        policy_audit = policy_manifest.parent / policy_audit
                    add(
                        policy_audit,
                        f"queue_policy_cluster_{cluster_id}_fit_audit",
                    )
        _, mapping_rows = read_background_mapping(background_policy)
        for row in mapping_rows:
            symbol = str(row["symbol"])
            add(pathlib.Path(row["policy_file"]), f"{symbol}:queue_cluster_policy")
            add(
                pathlib.Path(row["limit_buy_improvement_file"]),
                f"{symbol}:limit_buy_improvement_marks",
            )
            add(
                pathlib.Path(row["limit_sell_improvement_file"]),
                f"{symbol}:limit_sell_improvement_marks",
            )
    entries = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "roles": sorted(roles),
        }
        for path, roles in sorted(roles_by_path.items(), key=lambda item: str(item[0]))
    ]
    return {
        "schema_version": 1,
        "entry_count": len(entries),
        "entries": entries,
        "manifest_sha256": sha256_json(entries),
        "scope": "all simulator commands, summaries, model inputs and target artifacts consumed during three-stage selection",
    }


def command_artifact_paths(*payloads: object) -> dict[str, set[pathlib.Path]]:
    """Recover all file arguments from nested deterministic run records."""
    result = {
        "configs": set(),
        "background_policies": set(),
        "value_policies": set(),
        "executables": set(),
        "summaries": set(),
    }

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            command = value.get("command")
            if isinstance(command, list) and "--duration-seconds" in command:
                tokens = [str(token) for token in command]
                duration_index = tokens.index("--duration-seconds")
                if duration_index == 0:
                    raise CalibrationDriverError("run command lacks executable")
                result["executables"].add(
                    pathlib.Path(tokens[duration_index - 1]).resolve()
                )
                for flag, key in (
                    ("--universe-config", "configs"),
                    ("--background-policy-csv", "background_policies"),
                    ("--value-agent-policy-csv", "value_policies"),
                    ("--asset-summary-csv", "summaries"),
                ):
                    if flag in tokens:
                        # A structurally failed invocation can exit before it
                        # creates the requested summary.  Its run-result JSON
                        # remains selection evidence, but a nonexistent output
                        # was not consumed and cannot be frozen as an input.
                        if flag == "--asset-summary-csv" and value.get(
                            "success"
                        ) is not True:
                            continue
                        result[key].add(
                            pathlib.Path(tokens[tokens.index(flag) + 1]).resolve()
                        )
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    for payload in payloads:
        visit(payload)
    return result


def verify_transitive_runtime_artifacts(manifest: Mapping[str, object]) -> None:
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise CalibrationDriverError("freeze lacks a non-empty transitive artifact list")
    if manifest.get("manifest_sha256") != sha256_json(entries):
        raise CalibrationDriverError("transitive artifact manifest digest is invalid")
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise CalibrationDriverError("invalid transitive artifact record")
        path = pathlib.Path(str(raw.get("path", ""))).resolve()
        if not path.is_file() or sha256_file(path) != raw.get("sha256"):
            raise CalibrationDriverError(
                f"transitive runtime artifact is missing or hash-mismatched: {path}"
            )


def write_json(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prune_unpromoted_screen_artifacts(
    *,
    cluster_root: pathlib.Path,
    candidate_runtime: Mapping[
        str,
        tuple[
            dict[str, pathlib.Path],
            dict[str, pathlib.Path],
            pathlib.Path,
        ],
    ],
    retained_candidate_ids: Sequence[str],
    cluster_result_path: pathlib.Path,
) -> None:
    """Prune reproducible short-screen files after their scores are recorded.

    ``cluster_result.json`` already contains every candidate command digest,
    per-day score and structural diagnostic.  Keeping a second filesystem copy
    of every generated CSV and successful 1,800-second summary creates tens of
    thousands of redundant files.  Raw artifacts remain for every candidate
    promoted to full-day confirmation; only unpromoted short-screen artifacts
    are removed, and only after the cluster result is atomically written.
    """
    retained = set(retained_candidate_ids)
    unknown = sorted(retained.difference(candidate_runtime))
    if unknown:
        raise CalibrationDriverError(
            "screen-artifact retention refers to unknown candidates: "
            + ", ".join(unknown[:10])
        )
    resolved_cluster_root = cluster_root.resolve()
    removed: list[str] = []
    for candidate_id, (_, _, policy_path) in sorted(candidate_runtime.items()):
        if candidate_id in retained:
            continue
        candidate_root = policy_path.parent.resolve()
        if (
            candidate_root.parent != resolved_cluster_root
            or candidate_root.name != candidate_id
        ):
            raise CalibrationDriverError(
                "refusing to prune a screen artifact outside its cluster root"
            )
        if candidate_root.exists():
            shutil.rmtree(candidate_root)
        removed.append(candidate_id)
    write_json(cluster_root / "screen_artifact_retention.json", {
        "schema_version": SCHEMA_VERSION,
        "status": "unpromoted_short_screen_artifacts_pruned",
        "scientific_records_retained_in": {
            "path": str(cluster_result_path),
            "sha256": sha256_file(cluster_result_path),
        },
        "candidate_count": len(candidate_runtime),
        "retained_raw_candidate_ids": sorted(retained),
        "pruned_reproducible_candidate_ids": removed,
        "pruned_count": len(removed),
        "retention_rule": "retain_every_full_day_promoted_screen_candidate",
    })


def write_csv(
    path: pathlib.Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_dated_path(value: str, *, option: str) -> DatedPath:
    if "=" not in value:
        raise CalibrationDriverError(f"{option} requires DATE=PATH")
    raw_day, raw_path = value.split("=", 1)
    day = normalized_date(raw_day, label=option)
    path = pathlib.Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise CalibrationDriverError(f"{option} file does not exist: {path}")
    return DatedPath(day, path)


def dated_mapping(
    raw_values: Sequence[str], *, option: str, exact_count: int | None = None,
) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    for raw in raw_values:
        item = parse_dated_path(raw, option=option)
        if item.day in result:
            raise CalibrationDriverError(f"duplicate {option} date {item.day}")
        result[item.day] = item.path
    if exact_count is not None and len(result) != exact_count:
        raise CalibrationDriverError(
            f"{option} requires exactly {exact_count} dates; observed {len(result)}"
        )
    return result


def read_config(path: pathlib.Path) -> ConfigTable:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    required = {
        "book_id", "symbol", "data_dir", "hawkes_rates_file", *OPENING_FIELDS,
    }
    missing = sorted(required.difference(fields))
    if missing:
        raise CalibrationDriverError(f"config {path} lacks columns: {missing}")
    if not rows:
        raise CalibrationDriverError(f"config is empty: {path}")
    seen: set[str] = set()
    path_fields = ("data_dir", "hawkes_rates_file", "target_data_dir")
    for index, row in enumerate(rows):
        book_id = exact_integer(
            row["book_id"], label=f"{path}:{index + 2}:book_id", minimum=0
        )
        if book_id != index:
            raise CalibrationDriverError(
                f"config {path} book_id values must be contiguous from zero"
            )
        symbol = normalized_symbol(
            row["symbol"], label=f"{path}:{index + 2}:symbol"
        )
        if symbol in seen:
            raise CalibrationDriverError(f"duplicate symbol {symbol} in {path}")
        seen.add(symbol)
        row["symbol"] = symbol
        for field in path_fields:
            raw_value = str(row.get(field, "")).strip()
            if not raw_value:
                continue
            resolved = pathlib.Path(raw_value).expanduser()
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            row[field] = str(resolved.resolve())
    return ConfigTable(path.resolve(), fields, tuple(rows))


def read_cluster_map(path: pathlib.Path) -> dict[str, str]:
    return strict.read_cluster_map(path)


def load_candidate_protocol(path: pathlib.Path) -> CandidateProtocol:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationDriverError(f"cannot read candidate config {path}: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise CalibrationDriverError("candidate config requires schema_version 1")

    def mapping(name: str) -> Mapping[str, object]:
        value = payload.get(name)
        if not isinstance(value, Mapping):
            raise CalibrationDriverError(f"candidate config lacks object {name}")
        return value

    stage1 = mapping("stage1")
    stage2 = mapping("stage2")
    stage3 = mapping("stage3")

    def seeds(block: Mapping[str, object], name: str) -> tuple[int, ...]:
        values = block.get("seeds")
        if not isinstance(values, list) or not values:
            raise CalibrationDriverError(f"{name}.seeds must be a non-empty list")
        result = tuple(
            exact_integer(value, label=f"{name}.seeds", minimum=0)
            for value in values
        )
        if len(set(result)) != len(result):
            raise CalibrationDriverError(f"{name}.seeds contains duplicates")
        return result

    local_rows = stage1.get("local_mm_candidates")
    value_rows = stage2.get("value_policy_candidates")
    volatility_rows = stage2.get("volatility_candidates")
    if not isinstance(local_rows, list) or not local_rows:
        raise CalibrationDriverError("stage1.local_mm_candidates must be non-empty")
    if not isinstance(value_rows, list) or not value_rows:
        raise CalibrationDriverError("stage2.value_policy_candidates must be non-empty")
    if not isinstance(volatility_rows, list) or not volatility_rows:
        raise CalibrationDriverError("stage2.volatility_candidates must be non-empty")

    local: list[LocalCandidate] = []
    value: list[ValueCandidate] = []
    volatility: list[VolatilityCandidate] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(local_rows):
        if not isinstance(raw, Mapping):
            raise CalibrationDriverError("local-MM candidate must be an object")
        identifier = safe_identifier(raw.get("id"), label=f"local candidate {index} id")
        if identifier in identifiers:
            raise CalibrationDriverError(f"duplicate candidate id {identifier}")
        identifiers.add(identifier)
        enabled = raw.get("enabled")
        if not isinstance(enabled, bool):
            raise CalibrationDriverError(f"{identifier}.enabled must be boolean")
        interval = finite_float(raw.get("interval_ms"), label=f"{identifier}.interval_ms")
        quantity = finite_float(
            raw.get("quantity_multiplier"), label=f"{identifier}.quantity_multiplier"
        )
        improvement = finite_float(
            raw.get("improvement_probability"),
            label=f"{identifier}.improvement_probability",
        )
        spread_elasticity = finite_float(
            raw.get("spread_elasticity", 0.0),
            label=f"{identifier}.spread_elasticity",
        )
        max_improvement_probability = finite_float(
            raw.get("max_improvement_probability", 1.0),
            label=f"{identifier}.max_improvement_probability",
        )
        must_promote = raw.get("must_promote_after_short_screen", False)
        if not isinstance(must_promote, bool):
            raise CalibrationDriverError(
                f"{identifier}.must_promote_after_short_screen must be boolean"
            )
        if (interval <= 0.0 or quantity <= 0.0
                or spread_elasticity < 0.0
                or not 0.0 <= improvement <= max_improvement_probability <= 1.0):
            raise CalibrationDriverError(f"invalid numerical controls for {identifier}")
        if not enabled and spread_elasticity != 0.0:
            raise CalibrationDriverError(
                f"{identifier} disabled local-MM policy must use zero spread elasticity"
            )
        if not enabled and must_promote:
            raise CalibrationDriverError(
                f"{identifier} disabled local-MM policy cannot be mandatory"
            )
        local.append(LocalCandidate(
            identifier, enabled, interval, quantity, improvement,
            spread_elasticity, max_improvement_probability, must_promote,
        ))

    fixed_local_id = stage1.get("fixed_local_candidate_id")
    if fixed_local_id is not None:
        fixed_local_id = safe_identifier(
            fixed_local_id, label="stage1.fixed_local_candidate_id",
        )
        fixed_matches = [
            candidate for candidate in local
            if candidate.identifier == fixed_local_id
        ]
        if len(fixed_matches) != 1:
            raise CalibrationDriverError(
                "stage1.fixed_local_candidate_id does not identify exactly "
                "one declared local-MM candidate"
            )
        local = fixed_matches

    for index, raw in enumerate(value_rows):
        if not isinstance(raw, Mapping):
            raise CalibrationDriverError("value candidate must be an object")
        identifier = safe_identifier(raw.get("id"), label=f"value candidate {index} id")
        if identifier in identifiers:
            raise CalibrationDriverError(f"duplicate candidate id {identifier}")
        identifiers.add(identifier)
        enabled = raw.get("enabled")
        if not isinstance(enabled, bool):
            raise CalibrationDriverError(f"{identifier}.enabled must be boolean")
        threshold = finite_float(
            raw.get("threshold_bps"), label=f"{identifier}.threshold_bps"
        )
        participation = finite_float(
            raw.get("depth_participation"),
            label=f"{identifier}.depth_participation",
        )
        gap_elasticity = finite_float(
            raw.get("gap_elasticity", 0.0),
            label=f"{identifier}.gap_elasticity",
        )
        max_depth_participation = finite_float(
            raw.get("max_depth_participation", 1.0),
            label=f"{identifier}.max_depth_participation",
        )
        trigger_mode = raw.get("trigger_mode", "periodic_gap")
        if not isinstance(trigger_mode, str) or trigger_mode not in {
            "periodic_gap", "news_impulse",
        }:
            raise CalibrationDriverError(
                f"{identifier}.trigger_mode must be periodic_gap or news_impulse"
            )
        maximum_news_rechecks = exact_integer(
            raw.get("maximum_news_rechecks", 0),
            label=f"{identifier}.maximum_news_rechecks",
            minimum=0,
        )
        if maximum_news_rechecks > 16:
            raise CalibrationDriverError(
                f"{identifier}.maximum_news_rechecks must not exceed 16"
            )
        if trigger_mode == "periodic_gap" and maximum_news_rechecks != 0:
            raise CalibrationDriverError(
                f"{identifier} periodic_gap cannot have news rechecks"
            )
        if not enabled and maximum_news_rechecks != 0:
            raise CalibrationDriverError(
                f"{identifier} disabled policy cannot have news rechecks"
            )
        if (threshold < 0.0 or not 0.0 < participation <= 1.0
                or gap_elasticity < 0.0
                or not participation <= max_depth_participation <= 1.0
                or (gap_elasticity > 0.0 and threshold <= 0.0)):
            raise CalibrationDriverError(f"invalid value controls for {identifier}")
        if not enabled and gap_elasticity != 0.0:
            raise CalibrationDriverError(
                f"{identifier} disabled policy must use zero gap elasticity"
            )
        value.append(ValueCandidate(
            identifier, enabled, threshold, participation, trigger_mode,
            maximum_news_rechecks, gap_elasticity,
            max_depth_participation,
        ))

    for index, raw in enumerate(volatility_rows):
        if not isinstance(raw, Mapping):
            raise CalibrationDriverError("volatility candidate must be an object")
        identifier = safe_identifier(
            raw.get("id"), label=f"volatility candidate {index} id"
        )
        if identifier in identifiers:
            raise CalibrationDriverError(f"duplicate candidate id {identifier}")
        identifiers.add(identifier)
        persistence = finite_float(
            raw.get("fundamental_log_variance_persistence"),
            label=f"{identifier}.fundamental_log_variance_persistence",
        )
        std = finite_float(
            raw.get("fundamental_log_variance_std"),
            label=f"{identifier}.fundamental_log_variance_std",
        )
        variance_scale = finite_float(
            raw.get("fundamental_variance_scale"),
            label=f"{identifier}.fundamental_variance_scale",
        )
        excess_kurtosis_share = finite_float(
            raw.get("fundamental_excess_kurtosis_share"),
            label=f"{identifier}.fundamental_excess_kurtosis_share",
        )
        tail_transmission_multiplier = finite_float(
            raw.get("fundamental_tail_transmission_multiplier", 1.0),
            label=(
                f"{identifier}.fundamental_tail_transmission_multiplier"
            ),
        )
        order_flow_coupling = finite_float(
            raw.get("fundamental_order_flow_coupling", 0.0),
            label=f"{identifier}.fundamental_order_flow_coupling",
        )
        if (not 0.0 <= persistence < 1.0 or std < 0.0
                or not 0.0 <= variance_scale <= 4.0
                or not 0.0 <= excess_kurtosis_share <= 1.0
                or not 1.0 <= tail_transmission_multiplier <= 8.0
                or not 0.0 <= order_flow_coupling <= 2.5):
            raise CalibrationDriverError(f"invalid volatility controls for {identifier}")
        if (tail_transmission_multiplier > 1.0 + 1.0e-12
                and excess_kurtosis_share < 1.0 - 1.0e-12):
            raise CalibrationDriverError(
                f"{identifier} tail transmission above one requires the "
                "complete empirical excess-kurtosis allocation"
            )
        volatility.append(VolatilityCandidate(
            identifier, variance_scale, persistence, std,
            excess_kurtosis_share, tail_transmission_multiplier,
            order_flow_coupling,
        ))

    survivor_count = exact_integer(
        stage1.get("survivor_count"), label="stage1.survivor_count", minimum=1
    )
    if survivor_count > len(local):
        raise CalibrationDriverError(
            "stage1.survivor_count exceeds the local-MM candidate count"
        )
    confirmation_count = exact_integer(
        stage2.get("full_day_confirmation_count"),
        label="stage2.full_day_confirmation_count",
        minimum=1,
    )
    if confirmation_count > len(value) * len(volatility):
        raise CalibrationDriverError(
            "stage2.full_day_confirmation_count exceeds the joint candidate count"
        )
    raw_recheck_counts = stage2.get("full_day_recheck_counts", [0])
    if not isinstance(raw_recheck_counts, list) or not raw_recheck_counts:
        raise CalibrationDriverError(
            "stage2.full_day_recheck_counts must be a non-empty list"
        )
    full_day_recheck_counts = tuple(
        exact_integer(
            value,
            label="stage2.full_day_recheck_counts",
            minimum=0,
        )
        for value in raw_recheck_counts
    )
    if len(set(full_day_recheck_counts)) != len(full_day_recheck_counts):
        raise CalibrationDriverError(
            "stage2.full_day_recheck_counts contains duplicates"
        )
    if any(value > 16 for value in full_day_recheck_counts):
        raise CalibrationDriverError(
            "stage2.full_day_recheck_counts values must not exceed 16"
        )
    full_day_recheck_counts = tuple(sorted(full_day_recheck_counts))
    raw_mandatory_joint = stage2.get(
        "mandatory_full_day_joint_candidates", []
    )
    if not isinstance(raw_mandatory_joint, list):
        raise CalibrationDriverError(
            "stage2.mandatory_full_day_joint_candidates must be a list"
        )
    mandatory_full_day_joint_candidates: list[str] = []
    value_ids = {candidate.identifier for candidate in value}
    volatility_ids = {candidate.identifier for candidate in volatility}
    for index, raw in enumerate(raw_mandatory_joint):
        if not isinstance(raw, Mapping):
            raise CalibrationDriverError(
                "mandatory full-day joint candidate must be an object"
            )
        value_id = safe_identifier(
            raw.get("value_candidate_id"),
            label=f"mandatory joint candidate {index} value id",
        )
        volatility_id = safe_identifier(
            raw.get("volatility_candidate_id"),
            label=f"mandatory joint candidate {index} volatility id",
        )
        if value_id not in value_ids or volatility_id not in volatility_ids:
            raise CalibrationDriverError(
                "mandatory full-day joint candidate references an unknown "
                f"value/volatility id: {value_id}/{volatility_id}"
            )
        joint_id = f"value_{value_id}__vol_{volatility_id}"
        if joint_id in mandatory_full_day_joint_candidates:
            raise CalibrationDriverError(
                f"duplicate mandatory full-day joint candidate {joint_id}"
            )
        mandatory_full_day_joint_candidates.append(joint_id)
    full_day_confirmation_candidate_cap = exact_integer(
        stage2.get("full_day_confirmation_candidate_cap", 40),
        label="stage2.full_day_confirmation_candidate_cap",
        minimum=1,
    )
    if full_day_confirmation_candidate_cap > 48:
        raise CalibrationDriverError(
            "stage2.full_day_confirmation_candidate_cap must not exceed 48"
        )
    minimum_variant_capacity = (
        confirmation_count + len(full_day_recheck_counts) - 1
    )
    if full_day_confirmation_candidate_cap < minimum_variant_capacity:
        raise CalibrationDriverError(
            "stage2.full_day_confirmation_candidate_cap must accommodate "
            "every global leader plus every configured recheck count; require "
            f"at least {minimum_variant_capacity}"
        )
    full_day_confirmation_base_cap = min(
        9,
        full_day_confirmation_candidate_cap
        - len(full_day_recheck_counts) + 1,
    )
    if confirmation_count > full_day_confirmation_base_cap:
        raise CalibrationDriverError(
            "stage2.full_day_confirmation_count exceeds the bounded base "
            f"frontier capacity {full_day_confirmation_base_cap}"
        )
    if len(mandatory_full_day_joint_candidates) > full_day_confirmation_base_cap:
        raise CalibrationDriverError(
            "mandatory full-day joint candidates exceed the bounded base "
            "frontier capacity"
        )
    global_alternatives_per_cluster = exact_integer(
        stage2.get("global_refinement_alternatives_per_cluster", 4),
        label="stage2.global_refinement_alternatives_per_cluster",
        minimum=1,
    )
    if global_alternatives_per_cluster > 4:
        raise CalibrationDriverError(
            "stage2.global_refinement_alternatives_per_cluster must not exceed 4"
        )
    global_beam_width = exact_integer(
        stage2.get("global_refinement_beam_width", 3),
        label="stage2.global_refinement_beam_width",
        minimum=1,
    )
    if global_beam_width > 3:
        raise CalibrationDriverError(
            "stage2.global_refinement_beam_width must not exceed 3"
        )
    global_stage3_finalist_count = exact_integer(
        stage3.get("global_refinement_finalist_count", 3),
        label="stage3.global_refinement_finalist_count",
        minimum=1,
    )
    if global_stage3_finalist_count > 3:
        raise CalibrationDriverError(
            "stage3.global_refinement_finalist_count must not exceed 3"
        )
    timeout_block = payload.get("timeout_seconds", {})
    if not isinstance(timeout_block, Mapping):
        raise CalibrationDriverError("timeout_seconds must be an object")
    timeout_seconds: dict[str, float] = {}
    for name, default in (("stage1", 600.0), ("stage2", 1_800.0), ("stage3", 14_400.0), ("heldout", 14_400.0)):
        timeout_value = finite_float(
            timeout_block.get(name, default), label=f"timeout_seconds.{name}"
        )
        if timeout_value <= 0.0:
            raise CalibrationDriverError(f"timeout_seconds.{name} must be positive")
        timeout_seconds[name] = timeout_value
    return CandidateProtocol(
        local=tuple(local),
        value=tuple(value),
        volatility=tuple(volatility),
        stage1_seeds=seeds(stage1, "stage1"),
        stage2_seeds=seeds(stage2, "stage2"),
        stage2_confirmation_seeds=seeds(
            {
                "seeds": stage2.get("full_day_confirmation_seeds"),
            },
            "stage2.full_day_confirmation",
        ),
        stage3_seeds=seeds(stage3, "stage3"),
        stage1_survivors=survivor_count,
        stage2_confirmation_count=confirmation_count,
        full_day_confirmation_candidate_cap=(
            full_day_confirmation_candidate_cap
        ),
        full_day_recheck_counts=full_day_recheck_counts,
        mandatory_full_day_joint_candidates=tuple(
            mandatory_full_day_joint_candidates
        ),
        global_alternatives_per_cluster=global_alternatives_per_cluster,
        global_beam_width=global_beam_width,
        global_stage3_finalist_count=global_stage3_finalist_count,
        timeout_seconds=timeout_seconds,
    )


def read_background_mapping(path: pathlib.Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    required = {
        "symbol", "cluster_id", "policy_file",
        "limit_buy_improvement_file", "limit_sell_improvement_file",
    }
    missing = sorted(required.difference(fields))
    if missing or not rows:
        raise CalibrationDriverError(
            f"background policy mapping {path} is empty or lacks {missing}"
        )
    seen: set[str] = set()
    for row in rows:
        symbol = normalized_symbol(row["symbol"], label=f"{path}:symbol")
        if symbol in seen:
            raise CalibrationDriverError(f"duplicate mapping symbol {symbol}")
        seen.add(symbol)
        row["symbol"] = symbol
        for field in (
            "policy_file", "limit_buy_improvement_file",
            "limit_sell_improvement_file",
        ):
            resolved = pathlib.Path(row[field]).expanduser()
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            resolved = resolved.resolve()
            if not resolved.is_file():
                raise CalibrationDriverError(
                    f"background policy artifact does not exist: {resolved}"
                )
            row[field] = str(resolved)
    return fields, rows


def validate_universes(
    configs: Mapping[str, ConfigTable],
    background_mapping: pathlib.Path,
    clusters: Mapping[str, str],
) -> None:
    cluster_symbols = set(clusters)
    _, mapping_rows = read_background_mapping(background_mapping)
    if {row["symbol"] for row in mapping_rows} != cluster_symbols:
        raise CalibrationDriverError(
            "pooled background-policy universe differs from cluster map"
        )
    mapping_clusters = {
        row["symbol"]: str(row["cluster_id"]).strip()
        for row in mapping_rows
    }
    cluster_mismatches = sorted(
        symbol for symbol in cluster_symbols
        if mapping_clusters[symbol] != clusters[symbol]
    )
    if cluster_mismatches:
        preview = ", ".join(cluster_mismatches[:10])
        suffix = "" if len(cluster_mismatches) <= 10 else ", ..."
        raise CalibrationDriverError(
            "background-policy and calibration cluster assignments disagree "
            f"for {len(cluster_mismatches)} symbols: {preview}{suffix}"
        )
    for day, config in configs.items():
        if set(config.symbols) != cluster_symbols:
            raise CalibrationDriverError(
                f"training config {day} universe differs from cluster map"
            )


def extended_fields(fields: Sequence[str]) -> tuple[str, ...]:
    result = list(fields)
    for field in VOLATILITY_FIELDS:
        if field not in result:
            result.append(field)
    return tuple(result)


def prepared_fields(fields: Sequence[str]) -> tuple[str, ...]:
    result = list(fields)
    if TARGET_DIRECTORY_FIELD not in result:
        result.append(TARGET_DIRECTORY_FIELD)
    return tuple(result)


def prepare_dated_config(
    path: pathlib.Path,
    *,
    frozen_base: ConfigTable,
    dated_observations: ConfigTable,
) -> ConfigTable:
    """Copy only opening state and target provenance into a frozen model config."""
    if set(frozen_base.symbols) != set(dated_observations.symbols):
        raise CalibrationDriverError(
            "dated opening/target universe differs from frozen deployment universe"
        )
    dated_by_symbol = {
        str(row["symbol"]): row for row in dated_observations.rows
    }
    rows: list[dict[str, object]] = []
    for base_row in frozen_base.rows:
        symbol = str(base_row["symbol"])
        dated_row = dated_by_symbol[symbol]
        merged: dict[str, object] = dict(base_row)
        for field in OPENING_FIELDS:
            merged[field] = dated_row[field]
        target_directory = (
            str(dated_row.get(TARGET_DIRECTORY_FIELD, "")).strip()
            or str(dated_row["data_dir"])
        )
        merged[TARGET_DIRECTORY_FIELD] = target_directory
        rows.append(merged)
    write_csv(path, prepared_fields(frozen_base.fields), rows)
    return read_config(path)


def prepare_heldout_config(
    path: pathlib.Path,
    *,
    frozen_base: ConfigTable,
    opening_observations: ConfigTable,
    target_observations: ConfigTable,
) -> ConfigTable:
    """Merge held-out opening state and target location, never model inputs."""
    expected = set(frozen_base.symbols)
    if set(opening_observations.symbols) != expected:
        raise CalibrationDriverError("held-out opening universe differs from frozen universe")
    if set(target_observations.symbols) != expected:
        raise CalibrationDriverError("held-out target universe differs from frozen universe")
    opening_by_symbol = {
        str(row["symbol"]): row for row in opening_observations.rows
    }
    target_by_symbol = {
        str(row["symbol"]): row for row in target_observations.rows
    }
    rows: list[dict[str, object]] = []
    for base_row in frozen_base.rows:
        symbol = str(base_row["symbol"])
        merged: dict[str, object] = dict(base_row)
        for field in OPENING_FIELDS:
            merged[field] = opening_by_symbol[symbol][field]
        target_row = target_by_symbol[symbol]
        merged[TARGET_DIRECTORY_FIELD] = (
            str(target_row.get(TARGET_DIRECTORY_FIELD, "")).strip()
            or str(target_row["data_dir"])
        )
        rows.append(merged)
    write_csv(path, prepared_fields(frozen_base.fields), rows)
    return read_config(path)


def write_derived_config(
    path: pathlib.Path,
    source: ConfigTable,
    *,
    clusters: Mapping[str, str],
    volatility_by_cluster: Mapping[str, VolatilityCandidate] | None,
    only_cluster: str | None = None,
) -> tuple[str, ...]:
    rows: list[dict[str, object]] = []
    for source_row in source.rows:
        symbol = str(source_row["symbol"])
        cluster = clusters[symbol]
        if only_cluster is not None and cluster != only_cluster:
            continue
        row: dict[str, object] = dict(source_row)
        row["book_id"] = len(rows)
        candidate = (
            volatility_by_cluster.get(cluster)
            if volatility_by_cluster is not None else None
        )
        row["fundamental_log_variance_persistence"] = (
            format(candidate.persistence, ".17g") if candidate is not None else "0"
        )
        row["fundamental_log_variance_std"] = (
            format(candidate.std, ".17g") if candidate is not None else "0"
        )
        row["fundamental_order_flow_coupling"] = (
            format(candidate.order_flow_coupling, ".17g")
            if candidate is not None else "0"
        )
        if candidate is not None:
            # The latent reference value is unobserved and reaches the book
            # only through thresholded, partial value demand.  Scale the
            # empirical one-second volatility anchor rather than treating it
            # as an upper bound.  Values below one allocate less variation to
            # news; values above one compensate for incomplete transmission.
            source_volatility = finite_float(
                source_row["fundamental_volatility_bps_sqrt_second"],
                label=(
                    f"{source.path}:{symbol}:"
                    "fundamental_volatility_bps_sqrt_second"
                ),
            )
            row["fundamental_volatility_bps_sqrt_second"] = format(
                source_volatility * math.sqrt(candidate.variance_scale),
                ".17g",
            )

            # The log-volatility multiplier is normalized in second moment,
            # but its fourth/second-squared moment is exp(std^2).  Remove that
            # known factor from the innovation kurtosis so a persistence
            # candidate does not silently inflate a moment already allocated
            # from the training data.  The excess-kurtosis share is a separate
            # explicit allocation between latent news and microstructure.
            source_kurtosis = finite_float(
                source_row["fundamental_conditional_kurtosis"],
                label=f"{source.path}:{symbol}:fundamental_conditional_kurtosis",
            )
            desired_kurtosis = 1.0 + (
                candidate.excess_kurtosis_share
                * candidate.tail_transmission_multiplier
                * (source_kurtosis - 1.0)
            )
            innovation_kurtosis = desired_kurtosis * math.exp(
                -candidate.std * candidate.std
            )
            if innovation_kurtosis < 1.0:
                raise CalibrationDriverError(
                    f"volatility candidate {candidate.identifier} gives "
                    f"infeasible conditional kurtosis {innovation_kurtosis:.17g} "
                    f"for {symbol}; require at least one"
                )
            row["fundamental_conditional_kurtosis"] = format(
                innovation_kurtosis, ".17g"
            )
        rows.append(row)
    if not rows:
        raise CalibrationDriverError(f"derived config {path} has no rows")
    fields = extended_fields(source.fields)
    write_csv(path, fields, rows)
    return tuple(str(row["symbol"]) for row in rows)


def write_value_policy(
    path: pathlib.Path,
    symbols: Sequence[str],
    clusters: Mapping[str, str],
    selected: Mapping[str, ValueCandidate],
) -> None:
    rows = []
    for symbol in symbols:
        cluster = clusters[symbol]
        candidate = selected[cluster]
        rows.append({
            "symbol": symbol,
            "enabled": int(candidate.enabled),
            "value_threshold_bps": format(candidate.threshold_bps, ".17g"),
            "value_depth_participation": format(
                candidate.depth_participation, ".17g"
            ),
            "value_gap_elasticity": format(
                candidate.gap_elasticity, ".17g"
            ),
            "value_max_depth_participation": format(
                candidate.max_depth_participation, ".17g"
            ),
            "value_trigger_mode": candidate.trigger_mode,
            "value_maximum_news_rechecks": candidate.maximum_news_rechecks,
            "cluster_id": cluster,
            "candidate_id": candidate.identifier,
        })
    write_csv(
        path,
        (
            "symbol", "enabled", "value_threshold_bps",
            "value_depth_participation", "value_gap_elasticity",
            "value_max_depth_participation", "value_trigger_mode",
            "value_maximum_news_rechecks",
            "cluster_id", "candidate_id",
        ),
        rows,
    )


def write_subset_background_mapping(
    path: pathlib.Path,
    source_path: pathlib.Path,
    symbols: Sequence[str],
) -> None:
    fields, source_rows = read_background_mapping(source_path)
    wanted = set(symbols)
    rows = [row for row in source_rows if row["symbol"] in wanted]
    if {row["symbol"] for row in rows} != wanted:
        raise CalibrationDriverError("cannot construct complete subset mapping")
    rows.sort(key=lambda row: symbols.index(row["symbol"]))
    write_csv(path, fields, rows)


def target_root(row: Mapping[str, str]) -> pathlib.Path:
    raw = str(row.get("target_data_dir", "")).strip() or str(row["data_dir"])
    return pathlib.Path(raw).resolve()


def load_targets(
    config: ConfigTable,
    *,
    day: str,
    symbols: Sequence[str],
    duration: int,
) -> dict[str, Mapping[str, legacy.TargetMoment]]:
    by_symbol = {str(row["symbol"]): row for row in config.rows}
    result: dict[str, Mapping[str, legacy.TargetMoment]] = {}
    window = None if duration == STAGE3_DURATION else duration
    for symbol in symbols:
        if symbol not in by_symbol:
            raise CalibrationDriverError(f"target config lacks symbol {symbol}")
        try:
            result[symbol] = legacy.load_targets(
                target_root(by_symbol[symbol]), day, (symbol,),
                window_seconds=window,
            )[symbol]
        except (OSError, ValueError, legacy.CalibrationError) as error:
            raise CalibrationDriverError(
                f"cannot load {duration}s targets for {day}/{symbol}: {error}"
            ) from error
    return result


def preflight_training_targets(
    configs: Mapping[str, ConfigTable],
) -> list[dict[str, object]]:
    """Load every stage target before the first simulator invocation."""
    records: list[dict[str, object]] = []
    for day, config in sorted(configs.items()):
        symbols = tuple(config.symbols)
        try:
            strict_targets, _ = strict.load_target_artifacts(
                config.path, expected_date=day,
            )
        except (OSError, ValueError, strict.EvaluationError) as error:
            raise CalibrationDriverError(
                f"strict full-session target preflight failed for {day}: {error}"
            ) from error
        if set(strict_targets) != set(symbols):
            raise CalibrationDriverError(
                f"strict full-session target preflight returned an incomplete "
                f"universe for {day}"
            )
        for duration in (STAGE1_DURATION, STAGE2_DURATION, STAGE3_DURATION):
            targets = load_targets(
                config, day=day, symbols=symbols, duration=duration,
            )
            if set(targets) != set(symbols):
                raise CalibrationDriverError(
                    f"target preflight returned an incomplete universe for "
                    f"{day} at {duration}s"
                )
            records.append({
                "date": day,
                "duration_seconds": duration,
                "symbol_count": len(symbols),
                "status": "loaded_before_simulation",
                "strict_evaluator_compatible": True,
            })
    return records


def launcher_rank_count(tokens: Sequence[str]) -> int:
    """Return the explicit MPI/Slurm rank count in a launcher prefix.

    An empty launcher denotes a direct singleton execution.  A non-empty
    launcher must state a single, internally consistent rank count.  This
    avoids silently inheriting an allocation-wide task count from Slurm.
    """
    if not tokens:
        return 1
    observed_ranks: list[int] = []
    for index, token in enumerate(tokens):
        if token in {"-np", "-n", "--ntasks"}:
            if index + 1 >= len(tokens):
                raise CalibrationDriverError(f"launcher {token} lacks a value")
            observed_ranks.append(exact_integer(
                tokens[index + 1], label=f"launcher {token}", minimum=1
            ))
        elif token.startswith("--ntasks="):
            observed_ranks.append(exact_integer(
                token.split("=", 1)[1], label="launcher --ntasks", minimum=1
            ))
    if not observed_ranks:
        raise CalibrationDriverError(
            "a non-empty launcher must explicitly request its rank count "
            "with -np, -n or --ntasks"
        )
    if len(set(observed_ranks)) != 1:
        raise CalibrationDriverError(
            f"launcher contains inconsistent rank counts: {observed_ranks}"
        )
    return observed_ranks[0]


def parse_launcher(value: str) -> tuple[str, ...]:
    tokens = tuple(shlex.split(value)) if value.strip() else ()
    launcher_rank_count(tokens)
    return tokens


def command_for_run(
    *,
    launcher: Sequence[str],
    executable: pathlib.Path,
    config: pathlib.Path,
    background_policy: pathlib.Path,
    value_policy: pathlib.Path | None,
    summary: pathlib.Path,
    duration: int,
    seed: int,
    local: LocalCandidate,
) -> list[str]:
    command = [
        *launcher,
        str(executable),
        "--duration-seconds", str(duration),
        "--seed", str(seed),
        "--universe-config", str(config),
        "--window-ms", str(WINDOW_MS),
        # The value policy has its own model clock.  Keep it explicit in every
        # calibration command so changing the MPI communication window cannot
        # silently change the number of behavioural decisions.
        "--value-agent-interval-ms", str(WINDOW_MS),
        "--asset-summary-interval-ms", str(WINDOW_MS),
        "--asset-summary-csv", str(summary),
        "--background-model", "queue-reactive-v1",
        "--background-policy-csv", str(background_policy),
        "--local-mm-interval-ms", format(local.interval_ms, ".17g"),
        "--local-mm-quantity-multiplier", format(
            local.quantity_multiplier, ".17g"
        ),
        "--local-mm-improvement-probability", format(
            local.improvement_probability, ".17g"
        ),
        "--local-mm-spread-elasticity", format(
            local.spread_elasticity, ".17g"
        ),
        "--local-mm-max-improvement-probability", format(
            local.max_improvement_probability, ".17g"
        ),
        "--disable-shared-mm",
    ]
    if not local.enabled:
        command.append("--disable-local-mm")
    if value_policy is None:
        command.append("--disable-value-agent")
    else:
        command.extend(("--value-agent-policy-csv", str(value_policy)))
    return command


def execute_run(
    *, command: Sequence[str], run_dir: pathlib.Path, timeout_seconds: float,
    mpi_ranks: int,
) -> dict[str, object]:
    summary = run_dir / "fragmented_asset_summary.csv"
    if str(summary) not in command:
        raise CalibrationDriverError("run command does not target its run directory")
    command_list = list(command)
    command_record = {
        "command": command_list,
        "command_sha256": sha256_json(command_list),
        "mpi_ranks": mpi_ranks,
        "one_rank_contract": mpi_ranks == 1,
        "shared_mm_disabled": "--disable-shared-mm" in command_list,
        "queue_reactive_model": (
            "--background-model" in command_list
            and command_list[command_list.index("--background-model") + 1]
            == "queue-reactive-v1"
        ),
    }
    result_path = run_dir / "run_result.json"
    if run_dir.exists():
        if not result_path.is_file():
            raise CalibrationDriverError(
                f"incomplete checkpoint directory exists: {run_dir}; preserve "
                "it for diagnosis or remove only that exact directory"
            )
        try:
            previous = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CalibrationDriverError(
                f"cannot read checkpoint {result_path}: {error}"
            ) from error
        if previous.get("command_sha256") != command_record["command_sha256"]:
            raise CalibrationDriverError(
                f"checkpoint command differs from requested run: {run_dir}"
            )
        if previous.get("success") is not True or not summary.is_file() \
                or summary.stat().st_size == 0:
            raise CalibrationDriverError(
                f"checkpoint is not a successful complete run: {run_dir}"
            )
        observed_hash = sha256_file(summary)
        if previous.get("summary_sha256") != observed_hash:
            raise CalibrationDriverError(
                f"checkpoint summary hash mismatch: {summary}"
            )
        return {**previous, "resumed_from_verified_checkpoint": True}

    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "command.json", command_record)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command_list,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        output = completed.stdout or ""
        return_code: int | None = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return_code = None
        timed_out = True
    elapsed = time.monotonic() - started
    (run_dir / "simulator.log").write_text(output, encoding="utf-8")
    success = (
        not timed_out
        and return_code == 0
        and summary.is_file()
        and summary.stat().st_size > 0
    )
    result = {
        **command_record,
        "return_code": return_code,
        "timed_out": timed_out,
        "wall_seconds": elapsed,
        "summary_path": str(summary),
        "summary_sha256": sha256_file(summary) if success else None,
        "success": success,
    }
    write_json(result_path, result)
    return result


def score_runs(
    *,
    run_records_by_day: Mapping[str, Sequence[Mapping[str, object]]],
    configs: Mapping[str, ConfigTable],
    symbols_by_day: Mapping[str, Sequence[str]],
    duration: int,
    metrics: Sequence[str],
) -> dict[str, object]:
    day_results: list[dict[str, object]] = []
    structural_failures: list[str] = []
    for day in sorted(configs):
        records = list(run_records_by_day.get(day, ()))
        if not records or any(record.get("success") is not True for record in records):
            structural_failures.append(f"{day}: simulator execution failed")
            continue
        symbols = tuple(symbols_by_day[day])
        try:
            targets = load_targets(
                configs[day], day=day, symbols=symbols, duration=duration
            )
            summaries = [
                strict.load_summary(pathlib.Path(str(record["summary_path"])))
                for record in records
            ]
            expected = set(symbols)
            for summary in summaries:
                if set(summary) != expected:
                    raise CalibrationDriverError(
                        f"{day}: summary universe differs from candidate universe"
                    )
            metric_rows = []
            metric_losses = []
            for metric in metrics:
                residuals = []
                target_values = []
                simulated_values = []
                for symbol in symbols:
                    simulated = statistics.fmean(
                        summary[symbol][metric] for summary in summaries
                    )
                    target = targets[symbol][metric].target
                    residuals.append(strict.robust_residual(
                        metric, simulated, target
                    ))
                    target_values.append(target)
                    simulated_values.append(simulated)
                score = strict.residual_score(residuals)
                metric_losses.append(
                    statistics.fmean(strict.huber_loss(value) for value in residuals)
                )
                metric_rows.append({
                    "metric": metric,
                    "score": score,
                    "symbol_count": len(symbols),
                    "gross_failure_count": sum(
                        abs(value) > strict.GROSS_RESIDUAL_LIMIT + 1.0e-12
                        for value in residuals
                    ),
                    "target_mean": statistics.fmean(target_values),
                    "simulated_mean": statistics.fmean(simulated_values),
                    "target_median": strict.percentile(target_values, 0.5),
                    "simulated_median": strict.percentile(
                        simulated_values, 0.5
                    ),
                    "target_p90": strict.percentile(target_values, 0.9),
                    "simulated_p90": strict.percentile(
                        simulated_values, 0.9
                    ),
                })
            day_score = math.sqrt(2.0 * statistics.fmean(metric_losses))
            day_results.append({
                "date": day,
                "score": day_score,
                "metric_scores": metric_rows,
                "seed_count": len(records),
            })
        except (OSError, ValueError, legacy.CalibrationError, strict.EvaluationError, CalibrationDriverError) as error:
            structural_failures.append(f"{day}: {error}")
    if structural_failures or len(day_results) != len(configs):
        return {
            "eligible": False,
            "aggregate_score": None,
            "day_results": day_results,
            "structural_failures": structural_failures,
        }
    day_scores = [float(row["score"]) for row in day_results]
    median = statistics.median(day_scores)
    mad = statistics.median(abs(value - median) for value in day_scores)
    return {
        "eligible": True,
        "aggregate_score": median + 0.25 * mad,
        "aggregation": "median_daily_robust_score_plus_0.25_MAD",
        "median_daily_score": median,
        "daily_score_MAD": mad,
        "day_results": day_results,
        "structural_failures": [],
    }


def candidate_sort_key(record: Mapping[str, object]) -> tuple[int, float, str]:
    score = record.get("score", {}).get("aggregate_score") if isinstance(
        record.get("score"), Mapping
    ) else None
    return (
        0 if record.get("score", {}).get("eligible") is True else 1,  # type: ignore[union-attr]
        float(score) if score is not None else math.inf,
        str(record["candidate_id"]),
    )


def full_day_confirmation_sort_key(
    record: Mapping[str, object],
) -> tuple[int, int, float, float, float, str]:
    """Rank full-day cluster confirmations using the immutable gate geometry.

    A three-symbol cluster is not itself the authoritative 30-symbol strict
    evaluation.  Nevertheless, selecting it solely by a smooth mean loss can
    hide one severe tail or ACF miss.  Count the same kinds of exceedance first,
    then use normalized magnitude, worst-date loss and the existing aggregate
    score as deterministic tie breakers.  The final assembled panel is still
    required to pass the unchanged strict evaluator before any freeze.
    """
    score = record.get("score")
    if not isinstance(score, Mapping) or score.get("eligible") is not True:
        return (1, sys.maxsize, math.inf, math.inf, math.inf,
                str(record.get("candidate_id", "")))
    violations = 0
    normalized_excesses: list[float] = []
    worst_day = 0.0
    for day in score.get("day_results", []):
        if not isinstance(day, Mapping):
            continue
        day_score = float(day["score"])
        worst_day = max(worst_day, day_score)
        if day_score > strict.MAX_MARKETWIDE_ROBUST_SCORE:
            violations += 1
            normalized_excesses.append(
                day_score / strict.MAX_MARKETWIDE_ROBUST_SCORE
            )
        for metric in day.get("metric_scores", []):
            if not isinstance(metric, Mapping):
                continue
            metric_name = str(metric["metric"])
            metric_score = float(metric["score"])
            if metric_score > strict.MAX_MARKETWIDE_METRIC_SCORE:
                violations += 1
                normalized_excesses.append(
                    metric_score / strict.MAX_MARKETWIDE_METRIC_SCORE
                )
            if (metric_name in strict.CLUSTER_GATE_METRICS
                    and metric_score > strict.MAX_CLUSTER_METRIC_SCORE):
                violations += 1
                normalized_excesses.append(
                    metric_score / strict.MAX_CLUSTER_METRIC_SCORE
                )
            violations += int(metric.get("gross_failure_count", 0))
            if metric_name == "absolute_return_acf1":
                for suffix, limit in (
                    ("mean", strict.MAX_ACF_MEAN_ABSOLUTE_ERROR),
                    ("median", strict.MAX_ACF_MEDIAN_ABSOLUTE_ERROR),
                    ("p90", strict.MAX_ACF_P90_ABSOLUTE_ERROR),
                ):
                    error = abs(
                        float(metric[f"simulated_{suffix}"])
                        - float(metric[f"target_{suffix}"])
                    )
                    if error > limit:
                        violations += 1
                        normalized_excesses.append(error / limit)
    return (
        0,
        violations,
        max(normalized_excesses, default=0.0),
        worst_day,
        float(score["aggregate_score"]),
        str(record["candidate_id"]),
    )


def diverse_confirmation_candidates(
    eligible: Sequence[dict[str, object]],
    *,
    global_count: int,
    candidate_cap: int = 24,
    mandatory_candidate_ids: Sequence[str] = (),
) -> list[dict[str, object]]:
    """Retain smooth leaders plus scientifically distinct full-day anchors.

    An 1,800-second screen can favor an active value policy that later builds
    excessive absolute-return persistence.  It can also return several
    behaviorally identical value-off/volatility pairs.  Full-day confirmation
    therefore takes the global leaders, a metric frontier within each trigger
    mode, and representatives of participation and latent-variance regimes.
    This broadens evidence; it neither changes a score nor relaxes the final
    strict gate.  The explicit cap keeps the evidence matrix bounded.
    """
    if candidate_cap < global_count or candidate_cap > 48:
        raise CalibrationDriverError(
            "full-day confirmation candidate cap must be between the global "
            "leader count and 48"
        )
    selected: list[dict[str, object]] = []
    identifiers: set[str] = set()

    def add(record: dict[str, object] | None) -> None:
        if record is None:
            return
        identifier = str(record["candidate_id"])
        if identifier not in identifiers and len(selected) < candidate_cap:
            selected.append(record)
            identifiers.add(identifier)

    by_id = {str(record["candidate_id"]): record for record in eligible}
    missing_mandatory = sorted(
        set(mandatory_candidate_ids).difference(by_id)
    )
    if missing_mandatory:
        raise CalibrationDriverError(
            "mandatory full-day joint candidates are not structurally "
            f"eligible after the short screen: {missing_mandatory}"
        )
    # Mandatory long-horizon anchors come first so the variant planner keeps
    # their complete, predeclared recheck curve.  This does not alter their
    # score; it prevents a short sample with few tail events from silently
    # deleting the structural candidate before full-day evaluation.
    for identifier in mandatory_candidate_ids:
        add(by_id[str(identifier)])

    for record in eligible[:global_count]:
        add(record)

    def first_matching(predicate: object) -> dict[str, object] | None:
        for record in eligible:
            value = record.get("value_candidate")
            volatility = record.get("volatility_candidate")
            if not isinstance(value, Mapping) or not isinstance(
                volatility, Mapping
            ):
                continue
            if predicate(value, volatility):  # type: ignore[operator]
                return record
        return None

    def trigger_mode(record: Mapping[str, object]) -> str:
        value = record.get("value_candidate")
        if not isinstance(value, Mapping) or value.get("enabled") is not True:
            return "disabled"
        return str(value.get("trigger_mode", "periodic_gap"))

    def worst_date_metric_score(
        record: Mapping[str, object], metric_name: str,
    ) -> float:
        score = record.get("score")
        if not isinstance(score, Mapping):
            return math.inf
        observed: list[float] = []
        for day in score.get("day_results", []):
            if not isinstance(day, Mapping):
                continue
            for metric in day.get("metric_scores", []):
                if (isinstance(metric, Mapping)
                        and str(metric.get("metric")) == metric_name):
                    observed.append(float(metric["score"]))
                    break
        return max(observed) if len(observed) == TRAINING_DAY_COUNT else math.inf

    # Stochastic-volatility persistence and innovation tails are separately
    # identifiable in principle: the former controls clustering of absolute
    # returns, while the latter controls conditional fourth moments.  A short
    # screen can rank these regimes poorly because it contains few tail
    # events.  Preserve one smooth leader from each predeclared decomposition
    # so full-day evidence, rather than a 30-minute accident, decides between
    # them.  Missing fields default to the legacy iid regime for compatibility
    # with older protocol fixtures.  These regime anchors precede metric
    # anchors because a short sample contains too few tail events for a stable
    # metric ranking.
    volatility_regimes = (
        lambda volatility: (
            float(volatility.get("std", 0.0)) <= 1.0e-12
            and float(volatility.get("excess_kurtosis_share", 1.0)) < 0.99
        ),
        lambda volatility: (
            float(volatility.get("persistence", 0.0)) > 0.0
            and float(volatility.get("std", 0.0)) > 0.0
            and float(volatility.get("excess_kurtosis_share", 1.0)) < 0.99
        ),
        lambda volatility: (
            float(volatility.get("persistence", 0.0)) > 0.0
            and float(volatility.get("std", 0.0)) > 0.0
            and float(volatility.get("excess_kurtosis_share", 1.0)) >= 0.99
        ),
    )
    for regime in volatility_regimes:
        add(first_matching(
            lambda value, volatility, regime=regime: (
                value.get("enabled") is True and regime(volatility)
            )
        ))

    # A smooth aggregate can hide a tail failure.  Preserve the best
    # worst-date candidate for ACF, kurtosis and variance first, followed by
    # the remaining strict cluster metric.  Full-day evidence still applies
    # the original immutable gates to every date.
    modes = sorted({trigger_mode(record) for record in eligible})
    frontier_metrics = (
        "absolute_return_acf1",
        "return_kurtosis",
        "return_variance",
        "mean_spread_ticks",
    )
    for mode in modes:
        in_mode = [record for record in eligible if trigger_mode(record) == mode]
        add(min(in_mode, key=candidate_sort_key) if in_mode else None)
        for metric_name in frontier_metrics:
            add(min(
                in_mode,
                key=lambda record, metric_name=metric_name: (
                    worst_date_metric_score(record, metric_name),
                    candidate_sort_key(record),
                ),
            ) if in_mode else None)

    # One canonical no-value path is sufficient: latent fundamentals cannot
    # move the LOB when the value policy is disabled.
    add(first_matching(
        lambda value, volatility: (
            value.get("enabled") is False
            and abs(float(volatility["variance_scale"])) <= 1.0e-12
        )
    ))

    # Preserve all four one-shot participation regimes explicitly.  A
    # news-impulse policy acts far less often than a periodic-gap policy, so
    # its training grid uses 2--20% rather than repeatedly submitting tiny
    # fractions of the same displayed queue.
    for participation in (0.02, 0.05, 0.10, 0.20):
        add(first_matching(
            lambda value, volatility, participation=participation: (
                value.get("enabled") is True
                and abs(float(value["depth_participation"])
                        - participation) <= 1.0e-12
            )
        ))

    # Preserve low, intermediate, unit-scale and super-unit latent-news
    # regimes.  The reference value reaches the book only through
    # thresholded value demand, so a scale above one is a distinct and
    # scientifically relevant transmission regime; it must not be hidden in
    # the same anchor bucket as a scale of 0.75 or 1.0.
    variance_bands = (
        lambda scale: scale <= 0.25 + 1.0e-12,
        lambda scale: 0.25 + 1.0e-12 < scale < 0.75 - 1.0e-12,
        lambda scale: 0.75 - 1.0e-12 <= scale <= 1.0 + 1.0e-12,
        lambda scale: scale > 1.0 + 1.0e-12,
    )
    for band in variance_bands:
        add(first_matching(
            lambda value, volatility, band=band: (
                value.get("enabled") is True
                and band(float(volatility["variance_scale"]))
            )
        ))
    return selected


def full_day_value_variants(
    candidate: ValueCandidate,
    recheck_counts: Sequence[int],
) -> tuple[ValueCandidate, ...]:
    """Expand news rechecks only after the short stage-2 screen.

    Periodic and disabled policies cannot use the bounded news-follow-up
    mechanism and therefore retain an exact zero count.  Candidate identifiers
    are deterministic and unique across the expanded confirmation set.
    """
    if not candidate.enabled or candidate.trigger_mode != "news_impulse":
        if candidate.maximum_news_rechecks != 0:
            raise CalibrationDriverError(
                f"{candidate.identifier} cannot use news rechecks"
            )
        return (candidate,)
    variants = []
    for count in sorted(set(recheck_counts)):
        if count < 0 or count > 16:
            raise CalibrationDriverError(
                "full-day news recheck count must be between 0 and 16"
            )
        variants.append(ValueCandidate(
            identifier=f"{candidate.identifier}_r{count}",
            enabled=candidate.enabled,
            threshold_bps=candidate.threshold_bps,
            depth_participation=candidate.depth_participation,
            trigger_mode=candidate.trigger_mode,
            maximum_news_rechecks=count,
            gap_elasticity=candidate.gap_elasticity,
            max_depth_participation=candidate.max_depth_participation,
        ))
    if not variants:
        raise CalibrationDriverError(
            "full-day news recheck expansion produced no variants"
        )
    return tuple(variants)


def plan_full_day_confirmation_variants(
    screened: Sequence[dict[str, object]],
    *,
    recheck_counts: Sequence[int],
    total_cap: int,
) -> tuple[
    list[tuple[str, dict[str, object], ValueCandidate]],
    list[tuple[str, dict[str, object], ValueCandidate]],
]:
    """Generate and deterministically cap post-screen recheck variants.

    The cap covers expanded variants, not base candidates.  All variants of
    the smooth-leading base are retained first.  If that base is not a news
    policy, all variants of the first news base are also retained so every
    configured recheck count is represented.  Every remaining base then gets
    its first variant before later variants are added in stable round-robin
    order.  The caller records both generated and retained identifiers.
    """
    if not screened:
        raise CalibrationDriverError(
            "cannot plan full-day confirmation without screened candidates"
        )
    variants_by_base: list[
        list[tuple[str, dict[str, object], ValueCandidate]]
    ] = []
    generated: list[tuple[str, dict[str, object], ValueCandidate]] = []
    for record in screened:
        base_id = str(record["candidate_id"])
        candidate = ValueCandidate(
            **record["value_candidate"]  # type: ignore[arg-type]
        )
        base_variants = []
        for variant in full_day_value_variants(candidate, recheck_counts):
            identifier = (
                f"{base_id}__rechecks_{variant.maximum_news_rechecks}"
            )
            item = (identifier, record, variant)
            base_variants.append(item)
            generated.append(item)
        variants_by_base.append(base_variants)
    generated_ids = [item[0] for item in generated]
    if len(set(generated_ids)) != len(generated_ids):
        raise CalibrationDriverError(
            "full-day confirmation variant identifiers are not unique"
        )
    if total_cap <= 0 or total_cap > 48:
        raise CalibrationDriverError(
            "full-day confirmation expanded-variant cap must be 1--48"
        )

    retained: list[tuple[str, dict[str, object], ValueCandidate]] = []
    retained_ids: set[str] = set()

    def add(item: tuple[str, dict[str, object], ValueCandidate]) -> None:
        if item[0] not in retained_ids and len(retained) < total_cap:
            retained.append(item)
            retained_ids.add(item[0])

    # Preserve the complete follow-up response curve of the smooth leader.
    for item in variants_by_base[0]:
        add(item)

    # A disabled/periodic leader has no news curve.  Preserve all recheck
    # counts for the first news base before filling secondary choices.
    if len(variants_by_base[0]) == 1:
        for base_variants in variants_by_base[1:]:
            if len(base_variants) > 1:
                for item in base_variants:
                    add(item)
                break

    # Every scientifically selected frontier base receives one confirmation.
    for base_variants in variants_by_base:
        add(base_variants[0])

    # Fill remaining capacity by recheck round, then stable base order.
    maximum_variant_count = max(len(values) for values in variants_by_base)
    for variant_index in range(1, maximum_variant_count):
        for base_variants in variants_by_base:
            if variant_index < len(base_variants):
                add(base_variants[variant_index])

    if len(retained) < len(variants_by_base):
        raise CalibrationDriverError(
            "full-day confirmation cap cannot retain every screened base"
        )
    news_counts = {
        item[2].maximum_news_rechecks
        for item in retained
        if item[2].enabled and item[2].trigger_mode == "news_impulse"
    }
    has_news_base = any(
        item[2].enabled and item[2].trigger_mode == "news_impulse"
        for item in generated
    )
    expected_news_counts = set(recheck_counts) if has_news_base else set()
    if not expected_news_counts.issubset(news_counts):
        raise CalibrationDriverError(
            "full-day confirmation cap cannot represent every news recheck count"
        )
    return generated, retained


def cluster_confirmation_gate_audit(
    record: Mapping[str, object],
) -> dict[str, object]:
    """Audit the immutable strict cluster-metric gate on every training date.

    The authoritative strict evaluator operates on the assembled panel.  This
    precondition uses only the subset of that evaluator which is meaningful
    for an isolated liquidity cluster: each metric named by
    ``CLUSTER_GATE_METRICS`` must remain below the unchanged cluster threshold
    on all five training dates.  No held-out artifact is read and no threshold
    is estimated from the candidates.
    """
    score = record.get("score")
    audit: dict[str, object] = {
        "passed": False,
        "required_training_date_count": TRAINING_DAY_COUNT,
        "required_metrics": list(strict.CLUSTER_GATE_METRICS),
        "maximum_metric_score": strict.MAX_CLUSTER_METRIC_SCORE,
        "threshold_source": "evaluate_strict_model_validation.py",
        "date_results": [],
        "failure_reasons": [],
    }
    failures: list[str] = audit["failure_reasons"]  # type: ignore[assignment]
    if not isinstance(score, Mapping) or score.get("eligible") is not True:
        failures.append("candidate is not structurally eligible")
        return audit
    day_rows = score.get("day_results")
    if not isinstance(day_rows, list) or len(day_rows) != TRAINING_DAY_COUNT:
        failures.append(
            "candidate does not contain exactly five full-day training results"
        )
        return audit
    seen_dates: set[str] = set()
    date_audits: list[dict[str, object]] = []
    for day_row in day_rows:
        if not isinstance(day_row, Mapping):
            failures.append("malformed full-day training result")
            continue
        day = str(day_row.get("date", ""))
        if not day or day in seen_dates:
            failures.append(f"missing or duplicate training date {day!r}")
            continue
        seen_dates.add(day)
        metric_rows = day_row.get("metric_scores")
        if not isinstance(metric_rows, list):
            failures.append(f"{day}: metric rows are absent")
            continue
        by_name = {
            str(row.get("metric")): row
            for row in metric_rows
            if isinstance(row, Mapping)
        }
        metric_audits: list[dict[str, object]] = []
        for metric in strict.CLUSTER_GATE_METRICS:
            row = by_name.get(metric)
            if row is None:
                failures.append(f"{day}: missing cluster-gate metric {metric}")
                continue
            metric_score = finite_float(
                row.get("score"), label=f"{day}:{metric}:score"
            )
            passed = (
                metric_score
                <= strict.MAX_CLUSTER_METRIC_SCORE + 1.0e-12
            )
            metric_audits.append({
                "metric": metric,
                "score": metric_score,
                "maximum_score": strict.MAX_CLUSTER_METRIC_SCORE,
                "passed": passed,
            })
            if not passed:
                failures.append(
                    f"{day}: {metric} score {metric_score:.17g} exceeds "
                    f"{strict.MAX_CLUSTER_METRIC_SCORE:.17g}"
                )
        date_audits.append({
            "date": day,
            "metric_results": metric_audits,
            "passed": (
                len(metric_audits) == len(strict.CLUSTER_GATE_METRICS)
                and all(bool(row["passed"]) for row in metric_audits)
            ),
        })
    audit["date_results"] = sorted(
        date_audits, key=lambda row: str(row["date"])
    )
    audit["passed"] = (
        not failures
        and len(seen_dates) == TRAINING_DAY_COUNT
        and all(bool(row["passed"]) for row in date_audits)
    )
    return audit


def global_assignment_identifier(
    local_id: str,
    cluster_ids: Sequence[str],
    assignment: Mapping[str, Mapping[str, object]],
) -> str:
    """Return a deterministic, path-safe identifier for a cluster assignment."""
    ordered = [
        [cluster, str(assignment[cluster]["candidate_id"])]
        for cluster in cluster_ids
    ]
    return f"{local_id}__global_{sha256_json(ordered)[:16]}"


def session_model_seed(base_seed: int, day: str) -> int:
    """Derive an independent, reproducible stream for one trading session.

    Common random numbers remain exact across competing candidates within a
    date because they receive the same derived seed.  Distinct empirical days
    no longer replay an identical symbol-level fundamental path merely because
    the protocol uses the same logical replicate index.
    """
    try:
        parsed_day = date.fromisoformat(day)
    except ValueError as error:
        raise CalibrationDriverError(
            f"invalid ISO trading date for session seed: {day}"
        ) from error
    if parsed_day.isoformat() != day or base_seed < 0:
        raise CalibrationDriverError(
            f"invalid session-seed inputs: base_seed={base_seed}, day={day}"
        )
    payload = f"queue-reactive-session-seed-v1\0{base_seed}\0{day}".encode(
        "ascii"
    )
    # Keep the value in the positive signed 63-bit range accepted uniformly
    # by the C++ and Python command-line parsers.
    derived = int.from_bytes(
        hashlib.sha256(payload).digest()[:8], "big"
    ) & ((1 << 63) - 1)
    return derived if derived != 0 else 1


def ensure_clean_output(path: pathlib.Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise CalibrationDriverError(f"output root is not a directory: {path}")
        if any(path.iterdir()):
            raise CalibrationDriverError(
                f"output root must not exist or must be empty: {path}"
            )
    else:
        path.mkdir(parents=True)


def prepare_output(path: pathlib.Path, *, resume: bool) -> None:
    """Create a new output root or reopen one containing verified checkpoints."""
    if not resume:
        ensure_clean_output(path)
        return
    if path.exists() and not path.is_dir():
        raise CalibrationDriverError(f"output root is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def run_strict_evaluation(arguments: Sequence[str]) -> tuple[dict[str, object], pathlib.Path]:
    """Run the immutable strict evaluator in-process and return its hashed report."""
    namespace = strict.build_parser().parse_args(list(arguments))
    report = strict.run(namespace)
    report_path = namespace.output_dir.expanduser().resolve() / (
        "strict_validation_report.json"
    )
    if not report_path.is_file():
        raise CalibrationDriverError(
            "strict evaluator returned without strict_validation_report.json"
        )
    return report, report_path


def run_candidate_matrix(
    *,
    executable: pathlib.Path,
    launcher: Sequence[str],
    configs: Mapping[str, pathlib.Path],
    background_policies: Mapping[str, pathlib.Path],
    value_policy: pathlib.Path | None,
    local: LocalCandidate,
    duration: int,
    seeds: Sequence[int],
    output_root: pathlib.Path,
    timeout_seconds: float,
    run_workers: int = 1,
) -> dict[str, list[dict[str, object]]]:
    if run_workers <= 0:
        raise CalibrationDriverError("run_workers must be positive")
    mpi_ranks = launcher_rank_count(launcher)
    tasks: list[tuple[str, int, int, pathlib.Path, list[str]]] = []
    for day in sorted(configs):
        for base_seed in seeds:
            seed = session_model_seed(base_seed, day)
            run_dir = (
                output_root / f"day_{day.replace('-', '')}"
                / f"base_seed_{base_seed}"
            )
            summary = run_dir / "fragmented_asset_summary.csv"
            command = command_for_run(
                launcher=launcher,
                executable=executable,
                config=configs[day],
                background_policy=background_policies[day],
                value_policy=value_policy,
                summary=summary,
                duration=duration,
                seed=seed,
                local=local,
            )
            tasks.append((day, base_seed, seed, run_dir, command))

    def run_one(
        task: tuple[str, int, int, pathlib.Path, list[str]],
    ) -> tuple[str, int, dict[str, object]]:
        day, base_seed, seed, run_dir, command = task
        record = execute_run(
            command=command, run_dir=run_dir,
            timeout_seconds=timeout_seconds, mpi_ranks=mpi_ranks,
        )
        # Each task owns its record and directory, so this update is safe even
        # when independent simulator processes execute concurrently.
        record["base_seed"] = base_seed
        record["session_seed"] = seed
        record["session_date"] = day
        record["session_seed_derivation"] = (
            "sha256(queue-reactive-session-seed-v1,base_seed,date)"
        )
        return day, base_seed, record

    if run_workers == 1:
        completed = [run_one(task) for task in tasks]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=run_workers,
            thread_name_prefix="lob-calibration-run",
        ) as executor:
            # map preserves task order, making manifests independent of OS
            # process completion order.
            completed = list(executor.map(run_one, tasks))

    result: dict[str, list[dict[str, object]]] = {
        day: [] for day in sorted(configs)
    }
    for day, _base_seed, record in completed:
        result[day].append(record)
    return result


STATE_HASH_PATTERN = re.compile(r"\bstate_hash=(0x[0-9a-fA-F]+)\b")


def run_state_hash(record: Mapping[str, object]) -> str:
    summary_path = pathlib.Path(str(record["summary_path"]))
    log_path = summary_path.parent / "simulator.log"
    if not log_path.is_file():
        raise CalibrationDriverError(f"simulator log is missing: {log_path}")
    matches = STATE_HASH_PATTERN.findall(
        log_path.read_text(encoding="utf-8", errors="replace")
    )
    if len(matches) != 1:
        raise CalibrationDriverError(
            f"expected one terminal state hash in {log_path}; observed {len(matches)}"
        )
    return matches[0].lower()


def verify_rank_equivalence(
    *,
    executable: pathlib.Path,
    reference_launcher: Sequence[str],
    production_launcher: Sequence[str],
    config: pathlib.Path,
    background_policy: pathlib.Path,
    value_policy: pathlib.Path,
    local: LocalCandidate,
    day: str,
    base_seed: int,
    duration: int,
    output_root: pathlib.Path,
    timeout_seconds: float,
) -> pathlib.Path:
    """Fail closed unless one-rank and production-rank results are identical."""
    reference_ranks = launcher_rank_count(reference_launcher)
    production_ranks = launcher_rank_count(production_launcher)
    if reference_ranks != 1 or production_ranks <= 1:
        raise CalibrationDriverError(
            "rank-equivalence preflight requires a one-rank reference and a "
            "multi-rank production launcher"
        )
    common = {
        "executable": executable,
        "configs": {day: config},
        "background_policies": {day: background_policy},
        "value_policy": value_policy,
        "local": local,
        "duration": duration,
        "seeds": (base_seed,),
        "timeout_seconds": timeout_seconds,
        "run_workers": 1,
    }
    reference = run_candidate_matrix(
        launcher=reference_launcher,
        output_root=output_root / "rank_1",
        **common,
    )[day][0]
    production = run_candidate_matrix(
        launcher=production_launcher,
        output_root=output_root / f"rank_{production_ranks}",
        **common,
    )[day][0]
    if reference.get("success") is not True or production.get("success") is not True:
        raise CalibrationDriverError(
            "rank-equivalence preflight simulation failed; production execution "
            "is not authorized"
        )
    reference_summary = pathlib.Path(str(reference["summary_path"]))
    production_summary = pathlib.Path(str(production["summary_path"]))
    reference_hash = run_state_hash(reference)
    production_hash = run_state_hash(production)
    summary_equal = reference_summary.read_bytes() == production_summary.read_bytes()
    state_equal = reference_hash == production_hash
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "status": "rank_equivalence_passed" if summary_equal and state_equal else "rank_equivalence_failed",
        "reference_ranks": reference_ranks,
        "production_ranks": production_ranks,
        "duration_seconds": duration,
        "date": day,
        "base_seed": base_seed,
        "executable": {"path": str(executable), "sha256": sha256_file(executable)},
        "configuration": {"path": str(config), "sha256": sha256_file(config)},
        "background_policy": {
            "path": str(background_policy), "sha256": sha256_file(background_policy),
        },
        "value_policy": {"path": str(value_policy), "sha256": sha256_file(value_policy)},
        "reference_state_hash": reference_hash,
        "production_state_hash": production_hash,
        "reference_summary_sha256": sha256_file(reference_summary),
        "production_summary_sha256": sha256_file(production_summary),
        "summary_bytes_equal": summary_equal,
        "terminal_state_hash_equal": state_equal,
        "reference_run": reference,
        "production_run": production,
    }
    evidence_path = output_root / "rank_equivalence.json"
    write_json(evidence_path, evidence)
    if not summary_equal or not state_equal:
        raise CalibrationDriverError(
            "one-rank and production-rank preflight outputs differ; see "
            f"{evidence_path}"
        )
    return evidence_path


def train(args: argparse.Namespace) -> dict[str, object]:
    output_root = args.output_root.expanduser().resolve()
    ensure_clean_output(output_root)
    executable = args.executable.expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise CalibrationDriverError(f"executable is missing or not executable: {executable}")
    candidate_path = args.candidate_config.expanduser().resolve()
    cluster_path = args.cluster_map.expanduser().resolve()
    deployment_path = args.deployment_config.expanduser().resolve()
    background_policy = args.background_policy.expanduser().resolve()
    for path, label in (
        (candidate_path, "candidate config"),
        (cluster_path, "cluster map"),
        (deployment_path, "deployment config"),
        (background_policy, "pooled training-only background policy"),
    ):
        if not path.is_file():
            raise CalibrationDriverError(f"{label} does not exist: {path}")
    launcher = parse_launcher(args.launcher)
    if launcher_rank_count(launcher) != 1:
        raise CalibrationDriverError(
            "the 30-symbol parameter-selection phase requires one rank per "
            "independent run; use --run-workers to execute those runs in parallel"
        )
    if args.run_workers <= 0:
        raise CalibrationDriverError("--run-workers must be positive")
    protocol = load_candidate_protocol(candidate_path)
    clusters = read_cluster_map(cluster_path)
    training_paths = dated_mapping(
        args.training_config, option="--training-config", exact_count=TRAINING_DAY_COUNT
    )
    dated_observation_configs = {
        day: read_config(path) for day, path in training_paths.items()
    }
    validate_universes(dated_observation_configs, background_policy, clusters)
    deployment_config = read_config(deployment_path)
    if set(deployment_config.symbols) != set(clusters):
        raise CalibrationDriverError("deployment config universe differs from cluster map")
    _, deployment_mapping_rows = read_background_mapping(background_policy)
    if {row["symbol"] for row in deployment_mapping_rows} != set(clusters):
        raise CalibrationDriverError(
            "deployment background-policy universe differs from cluster map"
        )
    prepared_root = output_root / "prepared_training_configs"
    configs = {
        day: prepare_dated_config(
            prepared_root / f"{day}.csv",
            frozen_base=deployment_config,
            dated_observations=dated_observation_configs[day],
        )
        for day in sorted(dated_observation_configs)
    }
    target_preflight = preflight_training_targets(configs)
    background_paths = {day: background_policy for day in configs}

    inputs_manifest = {
        "schema_version": SCHEMA_VERSION,
        "training_dates": sorted(configs),
        "training_configs": {
            day: {
                "opening_and_target_source_path": str(
                    dated_observation_configs[day].path
                ),
                "opening_and_target_source_sha256": sha256_file(
                    dated_observation_configs[day].path
                ),
                "prepared_runtime_path": str(config.path),
                "prepared_runtime_sha256": sha256_file(config.path),
                "only_opening_fields_and_target_data_dir_vary": True,
            }
            for day, config in sorted(configs.items())
        },
        "pooled_training_background_policy": {
            "path": str(background_policy),
            "sha256": sha256_file(background_policy),
            "reused_unchanged_across_training_dates": True,
        },
        "candidate_config": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
        "parsed_local_candidate_protocol": {
            "candidate_count": len(protocol.local),
            "candidates": [candidate.__dict__ for candidate in protocol.local],
            "spread_adaptive_improvement": {
                "base_field": "improvement_probability",
                "elasticity_field": "spread_elasticity",
                "cap_field": "max_improvement_probability",
                "reference_spread_field": "target_spread_ticks",
                "zero_elasticity_is_exact_legacy_mode": True,
                "default_max_improvement_probability": 1.0,
                "adaptive_candidate_ids": [
                    candidate.identifier for candidate in protocol.local
                    if candidate.spread_elasticity > 0.0
                ],
            },
        },
        "parsed_value_candidate_protocol": {
            "candidate_count": len(protocol.value),
            "candidates": [candidate.__dict__ for candidate in protocol.value],
            "gap_adaptive_sizing": {
                "base_field": "depth_participation",
                "elasticity_field": "gap_elasticity",
                "cap_field": "max_depth_participation",
                "zero_elasticity_is_exact_legacy_mode": True,
                "default_max_depth_participation": 1.0,
                "adaptive_candidate_ids": [
                    candidate.identifier for candidate in protocol.value
                    if candidate.gap_elasticity > 0.0
                ],
            },
        },
        "cluster_map": {"path": str(cluster_path), "sha256": sha256_file(cluster_path)},
        "deployment_config": {"path": str(deployment_path), "sha256": sha256_file(deployment_path)},
        "executable": {"path": str(executable), "sha256": sha256_file(executable)},
        "launcher": list(launcher),
        "selection_execution": {
            "mpi_ranks_per_run": 1,
            "maximum_concurrent_runs": args.run_workers,
            "parallelism": "independent_run_task_parallelism",
        },
        "heldout_inputs_read": False,
        "target_preflight": target_preflight,
        "training_only_global_refinement_protocol": {
            "full_day_confirmation_candidate_cap": (
                protocol.full_day_confirmation_candidate_cap
            ),
            "confirmation_cap_scope": "expanded_full_day_variants_total",
            "screened_base_candidate_cap": min(
                9,
                protocol.full_day_confirmation_candidate_cap
                - len(protocol.full_day_recheck_counts) + 1,
            ),
            "mandatory_full_day_joint_candidate_ids": list(
                protocol.mandatory_full_day_joint_candidates
            ),
            "full_day_recheck_counts": list(
                protocol.full_day_recheck_counts
            ),
            "rechecks_expanded_after_short_screen_only": True,
            "alternatives_per_cluster_limit": (
                protocol.global_alternatives_per_cluster
            ),
            "beam_width": protocol.global_beam_width,
            "stage3_finalist_limit": (
                protocol.global_stage3_finalist_count
            ),
            "search_seeds": list(protocol.stage2_confirmation_seeds),
            "finalist_seeds": list(protocol.stage3_seeds),
            "full_day_duration_seconds": STAGE3_DURATION,
            "strict_cluster_gate_metrics": list(
                strict.CLUSTER_GATE_METRICS
            ),
            "strict_cluster_metric_score_maximum": (
                strict.MAX_CLUSTER_METRIC_SCORE
            ),
            "strict_thresholds_immutable": True,
            "heldout_inputs_read": False,
        },
    }
    write_json(output_root / "training_inputs_manifest.json", inputs_manifest)

    stage1_configs: dict[str, pathlib.Path] = {}
    for day, source in configs.items():
        path = output_root / "stage1" / "configs" / f"{day}.csv"
        write_derived_config(
            path, source, clusters=clusters, volatility_by_cluster=None
        )
        stage1_configs[day] = path
    stage1_records = []
    for local in protocol.local:
        candidate_root = output_root / "stage1" / "candidates" / local.identifier
        runs = run_candidate_matrix(
            executable=executable,
            launcher=launcher,
            configs=stage1_configs,
            background_policies=background_paths,
            value_policy=None,
            local=local,
            duration=STAGE1_DURATION,
            seeds=protocol.stage1_seeds,
            output_root=candidate_root,
            timeout_seconds=protocol.timeout_seconds["stage1"],
            run_workers=args.run_workers,
        )
        score = score_runs(
            run_records_by_day=runs,
            configs=configs,
            symbols_by_day={day: config.symbols for day, config in configs.items()},
            duration=STAGE1_DURATION,
            metrics=STAGE1_METRICS,
        )
        stage1_records.append({
            "candidate_id": local.identifier,
            "candidate": local.__dict__,
            "score": score,
            "runs": runs,
        })
    stage1_records.sort(key=candidate_sort_key)
    eligible_stage1 = [
        record for record in stage1_records
        if record["score"]["eligible"] is True  # type: ignore[index]
    ]
    if len(eligible_stage1) < protocol.stage1_survivors:
        write_json(output_root / "stage1" / "stage1_result.json", {
            "status": "failed", "candidates": stage1_records,
        })
        raise CalibrationDriverError(
            "too few structurally eligible stage-1 candidates for promotion"
        )
    mandatory_ids = {
        candidate.identifier for candidate in protocol.local
        if candidate.must_promote_after_short_screen
    }
    eligible_ids = {str(record["candidate_id"]) for record in eligible_stage1}
    missing_mandatory = sorted(mandatory_ids.difference(eligible_ids))
    if missing_mandatory:
        raise CalibrationDriverError(
            "mandatory long-horizon local-MM candidates failed the structural "
            f"short screen: {missing_mandatory}"
        )
    if len(mandatory_ids) > protocol.stage1_survivors:
        raise CalibrationDriverError(
            "mandatory long-horizon local-MM candidates exceed the stage-1 "
            "survivor count"
        )
    promoted = [
        record for record in eligible_stage1
        if str(record["candidate_id"]) in mandatory_ids
    ]
    promoted.extend(
        record for record in eligible_stage1
        if str(record["candidate_id"]) not in mandatory_ids
    )
    promoted = promoted[:protocol.stage1_survivors]
    promoted.sort(key=candidate_sort_key)
    write_json(output_root / "stage1" / "stage1_result.json", {
        "schema_version": SCHEMA_VERSION,
        "duration_seconds": STAGE1_DURATION,
        "metrics": list(STAGE1_METRICS),
        "promotion_count": protocol.stage1_survivors,
        "mandatory_long_horizon_candidate_ids": sorted(mandatory_ids),
        "promoted_candidate_ids": [record["candidate_id"] for record in promoted],
        "candidates": stage1_records,
    })

    local_by_id = {candidate.identifier: candidate for candidate in protocol.local}
    cluster_ids = tuple(sorted(set(clusters.values())))
    finalist_records: list[dict[str, object]] = []
    global_refinement_records: list[dict[str, object]] = []
    exhausted_local_records: list[dict[str, object]] = []
    for promoted_record in promoted:
        local_id = str(promoted_record["candidate_id"])
        local = local_by_id[local_id]
        alternatives_by_cluster: dict[str, list[dict[str, object]]] = {}
        stage2_cluster_results = []
        local_failure: dict[str, object] | None = None
        for cluster in cluster_ids:
            cluster_symbols = tuple(
                symbol for symbol in sorted(clusters) if clusters[symbol] == cluster
            )
            candidate_records = []
            candidate_runtime: dict[
                str,
                tuple[
                    dict[str, pathlib.Path],
                    dict[str, pathlib.Path],
                    pathlib.Path,
                ],
            ] = {}
            for value_candidate in protocol.value:
                for volatility_candidate in protocol.volatility:
                    joint_id = (
                        f"value_{value_candidate.identifier}__vol_"
                        f"{volatility_candidate.identifier}"
                    )
                    root = (
                        output_root / "stage2" / f"local_{local_id}"
                        / f"cluster_{cluster}" / joint_id
                    )
                    derived_configs: dict[str, pathlib.Path] = {}
                    derived_backgrounds: dict[str, pathlib.Path] = {}
                    for day, source in configs.items():
                        config_path = root / "configs" / f"{day}.csv"
                        symbols = write_derived_config(
                            config_path,
                            source,
                            clusters=clusters,
                            volatility_by_cluster={cluster: volatility_candidate},
                            only_cluster=cluster,
                        )
                        if symbols != cluster_symbols:
                            raise CalibrationDriverError(
                                "cluster config symbol order is not deterministic"
                            )
                        mapping_path = root / "background" / f"{day}.csv"
                        write_subset_background_mapping(
                            mapping_path, background_paths[day], symbols
                        )
                        derived_configs[day] = config_path
                        derived_backgrounds[day] = mapping_path
                    policy_path = root / "value_policy.csv"
                    write_value_policy(
                        policy_path,
                        cluster_symbols,
                        clusters,
                        {cluster: value_candidate},
                    )
                    runs = run_candidate_matrix(
                        executable=executable,
                        launcher=launcher,
                        configs=derived_configs,
                        background_policies=derived_backgrounds,
                        value_policy=policy_path,
                        local=local,
                        duration=STAGE2_DURATION,
                        seeds=protocol.stage2_seeds,
                        output_root=root / "runs",
                        timeout_seconds=protocol.timeout_seconds["stage2"],
                        run_workers=args.run_workers,
                    )
                    score = score_runs(
                        run_records_by_day=runs,
                        configs=configs,
                        symbols_by_day={day: cluster_symbols for day in configs},
                        duration=STAGE2_DURATION,
                        metrics=STAGE2_METRICS,
                    )
                    candidate_runtime[joint_id] = (
                        derived_configs, derived_backgrounds, policy_path
                    )
                    candidate_records.append({
                        "candidate_id": joint_id,
                        "value_candidate": value_candidate.__dict__,
                        "volatility_candidate": volatility_candidate.__dict__,
                        "score": score,
                        "runs": runs,
                    })
            candidate_records.sort(key=candidate_sort_key)
            eligible = [
                record for record in candidate_records
                if record["score"]["eligible"] is True  # type: ignore[index]
            ]
            if not eligible:
                cluster_root = (
                    output_root / "stage2" / f"local_{local_id}"
                    / f"cluster_{cluster}"
                )
                cluster_result_path = cluster_root / "cluster_result.json"
                write_json(
                    cluster_result_path,
                    {"status": "failed", "candidates": candidate_records},
                )
                prune_unpromoted_screen_artifacts(
                    cluster_root=cluster_root,
                    candidate_runtime=candidate_runtime,
                    retained_candidate_ids=(),
                    cluster_result_path=cluster_result_path,
                )
                local_failure = {
                    "cluster_id": cluster,
                    "reason": (
                        "no structurally eligible stage-2 candidate"
                    ),
                }
                break
            screened = diverse_confirmation_candidates(
                eligible,
                global_count=protocol.stage2_confirmation_count,
                candidate_cap=min(
                    9,
                    protocol.full_day_confirmation_candidate_cap
                    - len(protocol.full_day_recheck_counts) + 1,
                ),
                mandatory_candidate_ids=(
                    protocol.mandatory_full_day_joint_candidates
                ),
            )
            generated_variant_plan, retained_variant_plan = (
                plan_full_day_confirmation_variants(
                    screened,
                    recheck_counts=protocol.full_day_recheck_counts,
                    total_cap=(
                        protocol.full_day_confirmation_candidate_cap
                    ),
                )
            )
            confirmation_records = []
            confirmation_root = (
                output_root / "stage2" / f"local_{local_id}"
                / f"cluster_{cluster}" / "full_day_confirmation"
            )
            for confirmation_id, screened_record, value_variant in (
                retained_variant_plan
            ):
                joint_id = str(screened_record["candidate_id"])
                derived_configs, derived_backgrounds, _ = (
                    candidate_runtime[joint_id]
                )
                variant_root = confirmation_root / confirmation_id
                policy_path = variant_root / "value_policy.csv"
                write_value_policy(
                    policy_path,
                    cluster_symbols,
                    clusters,
                    {cluster: value_variant},
                )
                confirmation_runs = run_candidate_matrix(
                    executable=executable,
                    launcher=launcher,
                    configs=derived_configs,
                    background_policies=derived_backgrounds,
                    value_policy=policy_path,
                    local=local,
                    duration=STAGE3_DURATION,
                    seeds=protocol.stage2_confirmation_seeds,
                    output_root=variant_root / "runs",
                    timeout_seconds=protocol.timeout_seconds["stage3"],
                    run_workers=args.run_workers,
                )
                confirmation_score = score_runs(
                    run_records_by_day=confirmation_runs,
                    configs=configs,
                    symbols_by_day={day: cluster_symbols for day in configs},
                    duration=STAGE3_DURATION,
                    metrics=STAGE2_METRICS,
                )
                confirmation_records.append({
                    "candidate_id": confirmation_id,
                    "screen_candidate_id": joint_id,
                    "value_candidate": value_variant.__dict__,
                    "volatility_candidate": screened_record[
                        "volatility_candidate"
                    ],
                    "screen_score": screened_record["score"],
                    "score": confirmation_score,
                    "runs": confirmation_runs,
                    "full_day_recheck_expansion": {
                        "performed_after_short_screen": True,
                        "maximum_news_rechecks": (
                            value_variant.maximum_news_rechecks
                        ),
                        "configured_counts": list(
                            protocol.full_day_recheck_counts
                        ),
                    },
                })
            confirmation_records.sort(key=full_day_confirmation_sort_key)
            confirmed_eligible = [
                record for record in confirmation_records
                if record["score"]["eligible"] is True  # type: ignore[index]
            ]
            if not confirmed_eligible:
                cluster_root = (
                    output_root / "stage2" / f"local_{local_id}"
                    / f"cluster_{cluster}"
                )
                cluster_result_path = cluster_root / "cluster_result.json"
                write_json(
                    cluster_result_path,
                    {
                        "status": "full_day_confirmation_failed",
                        "screen_candidates": candidate_records,
                        "confirmation_candidates": confirmation_records,
                    },
                )
                prune_unpromoted_screen_artifacts(
                    cluster_root=cluster_root,
                    candidate_runtime=candidate_runtime,
                    retained_candidate_ids=tuple(
                        str(record["candidate_id"]) for record in screened
                    ),
                    cluster_result_path=cluster_result_path,
                )
                local_failure = {
                    "cluster_id": cluster,
                    "reason": (
                        "no structurally eligible full-day confirmation"
                    ),
                }
                break
            for record in confirmation_records:
                record["strict_cluster_metric_gate"] = (
                    cluster_confirmation_gate_audit(record)
                )
            strict_cluster_eligible = [
                record for record in confirmation_records
                if record["strict_cluster_metric_gate"]["passed"] is True  # type: ignore[index]
            ]
            # Three stratified symbols per liquidity cluster are enough for a
            # bounded search panel, but not for a stable upper-tail adequacy
            # decision.  In particular, one stock can determine a cluster's
            # p90 variance or kurtosis.  Preserve the immutable gate as a
            # diagnostic and prefer its passing candidates whenever they
            # exist.  If none pass, retain the structurally eligible robust
            # frontier provisionally.  This provisional selection can never
            # authorize held-out execution: only the separate 1,480-symbol
            # five-date expansion gate can upgrade it below.
            retained_pool = (
                strict_cluster_eligible
                if strict_cluster_eligible else confirmed_eligible
            )
            retained = retained_pool[
                :protocol.global_alternatives_per_cluster
            ]
            alternatives_by_cluster[cluster] = retained
            winner = retained[0]
            strict_cluster_passed = bool(strict_cluster_eligible)
            cluster_result = {
                "cluster_id": cluster,
                "selected_candidate_id": winner["candidate_id"],
                "status": (
                    "strict_cluster_metric_alternatives_retained"
                    if strict_cluster_passed else
                    "provisional_robust_alternatives_retained_pending_"
                    "full_universe_adequacy"
                ),
                "selection_basis": (
                    "immutable_small_panel_cluster_gate"
                    if strict_cluster_passed else
                    "structurally_eligible_full_day_robust_score"
                ),
                "small_panel_cluster_gate_passed": strict_cluster_passed,
                "small_panel_symbol_count": len(cluster_symbols),
                "full_universe_adequacy_required_before_heldout": True,
                "screen_duration_seconds": STAGE2_DURATION,
                "confirmation_duration_seconds": STAGE3_DURATION,
                "screen_promoted_candidate_ids": [
                    record["candidate_id"] for record in screened
                ],
                "full_day_confirmation_candidate_cap": (
                    protocol.full_day_confirmation_candidate_cap
                ),
                "confirmation_cap_scope": (
                    "expanded_full_day_variants_total"
                ),
                "screened_base_candidate_cap": min(
                    9,
                    protocol.full_day_confirmation_candidate_cap
                    - len(protocol.full_day_recheck_counts) + 1,
                ),
                "generated_confirmation_variant_count": len(
                    generated_variant_plan
                ),
                "generated_confirmation_variant_ids": [
                    item[0] for item in generated_variant_plan
                ],
                "retained_confirmation_variant_count": len(
                    retained_variant_plan
                ),
                "retained_confirmation_variant_ids": [
                    item[0] for item in retained_variant_plan
                ],
                "full_day_recheck_counts": list(
                    protocol.full_day_recheck_counts
                ),
                "rechecks_expanded_after_short_screen_only": True,
                "retained_alternative_limit": (
                    protocol.global_alternatives_per_cluster
                ),
                "retained_alternative_ids": [
                    record["candidate_id"] for record in retained
                ],
                "strict_cluster_metric_gate": {
                    "required_training_date_count": TRAINING_DAY_COUNT,
                    "required_metrics": list(strict.CLUSTER_GATE_METRICS),
                    "maximum_metric_score": strict.MAX_CLUSTER_METRIC_SCORE,
                    "thresholds_immutable": True,
                },
                "candidates": candidate_records,
                "confirmation_candidates": confirmation_records,
            }
            cluster_root = (
                output_root / "stage2" / f"local_{local_id}"
                / f"cluster_{cluster}"
            )
            cluster_result_path = cluster_root / "cluster_result.json"
            write_json(cluster_result_path, cluster_result)
            prune_unpromoted_screen_artifacts(
                cluster_root=cluster_root,
                candidate_runtime=candidate_runtime,
                retained_candidate_ids=tuple(
                    str(record["candidate_id"]) for record in screened
                ),
                cluster_result_path=cluster_result_path,
            )
            # Finalists bind the complete cluster evidence by path and hash
            # instead of embedding the 40--50 MB candidate table a second time.
            stage2_cluster_results.append({
                key: value for key, value in cluster_result.items()
                if key not in {"candidates", "confirmation_candidates"}
            } | {
                "screen_candidate_count": len(candidate_records),
                "confirmation_candidate_count": len(confirmation_records),
                "cluster_result_artifact": {
                    "path": str(cluster_result_path),
                    "sha256": sha256_file(cluster_result_path),
                },
                "screen_artifact_retention": {
                    "path": str(cluster_root / "screen_artifact_retention.json"),
                    "sha256": sha256_file(
                        cluster_root / "screen_artifact_retention.json"
                    ),
                },
            })

        # Stage 1 promotes multiple local-liquidity regimes precisely because
        # a short screen cannot establish full-day feasibility.  Exhausting
        # one promoted regime is evidence about that regime, not authority to
        # abort the declared search before trying the others.
        if local_failure is not None:
            exhausted_record = {
                "local_candidate_id": local_id,
                "status": "promoted_local_regime_exhausted",
                "failure": local_failure,
                "completed_cluster_count": len(stage2_cluster_results),
                "completed_cluster_ids": [
                    result["cluster_id"] for result in stage2_cluster_results
                ],
                "heldout_inputs_read": False,
                "strict_thresholds_immutable": True,
            }
            exhausted_local_records.append(exhausted_record)
            write_json(
                output_root / "stage2" / f"local_{local_id}"
                / "local_search_result.json",
                exhausted_record,
            )
            continue

        # Clusterwise optima need not form the best marketwide model.  Search
        # a bounded Cartesian neighbourhood using full-day, full-panel runs.
        # The beam is deterministic and never uses held-out observations.
        alternative_lookup: dict[str, dict[str, dict[str, object]]] = {
            cluster: {
                str(record["candidate_id"]): record
                for record in alternatives_by_cluster[cluster]
            }
            for cluster in cluster_ids
        }
        assignment_cache: dict[
            tuple[str, ...], dict[str, object]
        ] = {}

        def evaluate_global_assignment(
            assignment_key: tuple[str, ...],
        ) -> dict[str, object]:
            cached = assignment_cache.get(assignment_key)
            if cached is not None:
                return cached
            if len(assignment_key) != len(cluster_ids):
                raise CalibrationDriverError(
                    "global-refinement assignment length differs from "
                    "the frozen cluster order"
                )
            assignment = {
                cluster: alternative_lookup[cluster][candidate_id]
                for cluster, candidate_id in zip(cluster_ids, assignment_key)
            }
            assignment_id = global_assignment_identifier(
                local_id, cluster_ids, assignment
            )
            assignment_root = (
                output_root / "stage2" / f"local_{local_id}"
                / "global_refinement" / assignment_id
            )
            selected_value = {
                cluster: ValueCandidate(
                    **assignment[cluster]["value_candidate"]  # type: ignore[arg-type]
                )
                for cluster in cluster_ids
            }
            selected_volatility = {
                cluster: VolatilityCandidate(
                    **assignment[cluster]["volatility_candidate"]  # type: ignore[arg-type]
                )
                for cluster in cluster_ids
            }
            assignment_configs: dict[str, pathlib.Path] = {}
            for day, source in configs.items():
                path = assignment_root / "configs" / f"{day}.csv"
                write_derived_config(
                    path,
                    source,
                    clusters=clusters,
                    volatility_by_cluster=selected_volatility,
                )
                assignment_configs[day] = path
            assignment_policy = assignment_root / "value_policy.csv"
            write_value_policy(
                assignment_policy,
                tuple(sorted(clusters)),
                clusters,
                selected_value,
            )
            assignment_runs = run_candidate_matrix(
                executable=executable,
                launcher=launcher,
                configs=assignment_configs,
                background_policies=background_paths,
                value_policy=assignment_policy,
                local=local,
                duration=STAGE3_DURATION,
                seeds=protocol.stage2_confirmation_seeds,
                output_root=assignment_root / "runs",
                timeout_seconds=protocol.timeout_seconds["stage3"],
                run_workers=args.run_workers,
            )
            assignment_score = score_runs(
                run_records_by_day=assignment_runs,
                configs=configs,
                symbols_by_day={
                    day: tuple(sorted(clusters)) for day in configs
                },
                duration=STAGE3_DURATION,
                metrics=STAGE3_METRICS,
            )
            record: dict[str, object] = {
                "candidate_id": assignment_id,
                "assignment_key": list(assignment_key),
                "cluster_candidate_ids": {
                    cluster: assignment_key[index]
                    for index, cluster in enumerate(cluster_ids)
                },
                "selected_value_by_cluster": {
                    cluster: selected_value[cluster].__dict__
                    for cluster in cluster_ids
                },
                "selected_volatility_by_cluster": {
                    cluster: selected_volatility[cluster].__dict__
                    for cluster in cluster_ids
                },
                "configs": {
                    day: str(path)
                    for day, path in assignment_configs.items()
                },
                "value_policy": str(assignment_policy),
                "score": assignment_score,
                "runs": assignment_runs,
                "evaluation_seeds": list(
                    protocol.stage2_confirmation_seeds
                ),
                "heldout_inputs_read": False,
            }
            assignment_cache[assignment_key] = record
            write_json(assignment_root / "assignment_result.json", record)
            return record

        initial_key = tuple(
            str(alternatives_by_cluster[cluster][0]["candidate_id"])
            for cluster in cluster_ids
        )
        beam_keys = [initial_key]
        beam_trace: list[dict[str, object]] = []
        evaluate_global_assignment(initial_key)
        for cluster_index, cluster in enumerate(cluster_ids):
            expanded_keys = {
                key[:cluster_index] + (str(alternative["candidate_id"]),)
                + key[cluster_index + 1:]
                for key in beam_keys
                for alternative in alternatives_by_cluster[cluster]
            }
            expanded_records = [
                evaluate_global_assignment(key)
                for key in sorted(expanded_keys)
            ]
            expanded_records.sort(key=full_day_confirmation_sort_key)
            eligible_records = [
                record for record in expanded_records
                if record["score"]["eligible"] is True  # type: ignore[index]
            ]
            if not eligible_records:
                raise CalibrationDriverError(
                    "global refinement produced no structurally eligible "
                    f"assignment after cluster {cluster}"
                )
            retained_records = eligible_records[:protocol.global_beam_width]
            beam_keys = [
                tuple(str(value) for value in record["assignment_key"])  # type: ignore[arg-type]
                for record in retained_records
            ]
            beam_trace.append({
                "coordinate_index": cluster_index,
                "cluster_id": cluster,
                "expanded_assignment_ids": [
                    record["candidate_id"] for record in expanded_records
                ],
                "retained_assignment_ids": [
                    record["candidate_id"] for record in retained_records
                ],
            })

        beam_records = [assignment_cache[key] for key in beam_keys]
        beam_records.sort(key=full_day_confirmation_sort_key)
        global_refinement_records.extend(
            assignment_cache[key] for key in sorted(assignment_cache)
        )
        global_root = (
            output_root / "stage2" / f"local_{local_id}"
            / "global_refinement"
        )
        write_json(global_root / "global_refinement_result.json", {
            "schema_version": SCHEMA_VERSION,
            "status": "training_only_global_refinement_completed",
            "heldout_inputs_read": False,
            "cluster_order": list(cluster_ids),
            "alternatives_per_cluster_limit": (
                protocol.global_alternatives_per_cluster
            ),
            "beam_width": protocol.global_beam_width,
            "stage3_finalist_limit": (
                protocol.global_stage3_finalist_count
            ),
            "evaluation_duration_seconds": STAGE3_DURATION,
            "evaluation_seeds": list(protocol.stage2_confirmation_seeds),
            "strict_thresholds_immutable": True,
            "beam_trace": beam_trace,
            "final_beam_assignment_ids": [
                record["candidate_id"] for record in beam_records
            ],
            "evaluated_assignment_count": len(assignment_cache),
        })

        # Re-evaluate the best distinct marketwide combinations with the
        # stage-3 seed set.  These fresh runs, not the search runs, are passed
        # to the unchanged strict evaluator below.
        for search_record in beam_records[
            :protocol.global_stage3_finalist_count
        ]:
            finalist_id = str(search_record["candidate_id"])
            selected_value = {
                cluster: ValueCandidate(**payload)  # type: ignore[arg-type]
                for cluster, payload in search_record[
                    "selected_value_by_cluster"
                ].items()  # type: ignore[union-attr]
            }
            selected_volatility = {
                cluster: VolatilityCandidate(**payload)  # type: ignore[arg-type]
                for cluster, payload in search_record[
                    "selected_volatility_by_cluster"
                ].items()  # type: ignore[union-attr]
            }
            finalist_root = output_root / "stage3" / f"finalist_{finalist_id}"
            finalist_configs: dict[str, pathlib.Path] = {}
            for day, source in configs.items():
                path = finalist_root / "configs" / f"{day}.csv"
                write_derived_config(
                    path,
                    source,
                    clusters=clusters,
                    volatility_by_cluster=selected_volatility,
                )
                finalist_configs[day] = path
            value_policy = finalist_root / "value_policy.csv"
            write_value_policy(
                value_policy,
                tuple(sorted(clusters)),
                clusters,
                selected_value,
            )
            runs = run_candidate_matrix(
                executable=executable,
                launcher=launcher,
                configs=finalist_configs,
                background_policies=background_paths,
                value_policy=value_policy,
                local=local,
                duration=STAGE3_DURATION,
                seeds=protocol.stage3_seeds,
                output_root=finalist_root / "runs",
                timeout_seconds=protocol.timeout_seconds["stage3"],
                run_workers=args.run_workers,
            )
            score = score_runs(
                run_records_by_day=runs,
                configs=configs,
                symbols_by_day={
                    day: tuple(sorted(clusters)) for day in configs
                },
                duration=STAGE3_DURATION,
                metrics=STAGE3_METRICS,
            )
            finalist_record = {
                "candidate_id": finalist_id,
                "local_candidate": local.__dict__,
                "selected_value_by_cluster": {
                    cluster: selected_value[cluster].__dict__
                    for cluster in cluster_ids
                },
                "selected_volatility_by_cluster": {
                    cluster: selected_volatility[cluster].__dict__
                    for cluster in cluster_ids
                },
                "stage2_cluster_results": stage2_cluster_results,
                "global_refinement_source": {
                    "candidate_id": search_record["candidate_id"],
                    "score": search_record["score"],
                    "evaluation_seeds": search_record["evaluation_seeds"],
                    "heldout_inputs_read": False,
                },
                "stage3_configs": {
                    day: str(path)
                    for day, path in finalist_configs.items()
                },
                "value_policy": str(value_policy),
                "score": score,
                "runs": runs,
            }
            finalist_records.append(finalist_record)
            write_json(
                finalist_root / "finalist_result.json", finalist_record
            )

    finalist_records.sort(key=candidate_sort_key)
    eligible_finalists = [
        record for record in finalist_records
        if record["score"]["eligible"] is True  # type: ignore[index]
    ]
    if not eligible_finalists:
        write_json(output_root / "stage3" / "stage3_result.json", {
            "status": "failed",
            "finalists": finalist_records,
            "exhausted_promoted_local_regimes": exhausted_local_records,
        })
        raise CalibrationDriverError(
            "no structurally eligible full-day finalist after exhausting "
            "every promoted local-liquidity regime"
        )

    # The smooth stage-3 loss is useful for search, but it is not freeze
    # authority.  Evaluate every structurally eligible full-day finalist with
    # the immutable strict gate.  Otherwise the smooth winner could fail while
    # another already-simulated finalist passes, producing a false negative.
    # This cannot loosen the gate: every candidate still has to pass every date
    # separately before it may be selected.
    strictly_passing_finalists: list[tuple[
        dict[str, object], list[str], dict[str, object], pathlib.Path,
    ]] = []
    for finalist in eligible_finalists:
        finalist_id = str(finalist["candidate_id"])
        finalist_root = output_root / "stage3" / f"finalist_{finalist_id}"
        strict_arguments = [
            "--evaluation-role", "training_fit",
            "--cluster-map", str(cluster_path),
            "--expected-cluster-count", str(len(cluster_ids)),
            "--output-dir", str(finalist_root / "strict_training_evaluation"),
        ]
        for day in sorted(configs):
            strict_arguments.extend(("--expected-date", day))
            strict_arguments.extend(
                ("--target-config", f"{day}={configs[day].path}")
            )
            for record in finalist["runs"][day]:  # type: ignore[index]
                command = record["command"]
                seed = command[command.index("--seed") + 1]
                strict_arguments.extend((
                    "--sim-summary",
                    f"{day}:{seed}={record['summary_path']}",
                ))
        write_json(finalist_root / "strict_training_evaluator_arguments.json", {
            "script": str(SCRIPT_DIR / "evaluate_strict_model_validation.py"),
            "arguments": strict_arguments,
            "automatic_execution_required_before_freeze": True,
        })
        strict_report, strict_report_path = run_strict_evaluation(
            strict_arguments
        )
        finalist["strict_training_evaluation"] = {
            "passed": strict_report.get("passed") is True,
            "path": str(strict_report_path),
            "sha256": sha256_file(strict_report_path),
        }
        finalist["strict_training_evaluator_arguments"] = strict_arguments
        write_json(finalist_root / "finalist_result.json", finalist)
        if strict_report.get("passed") is True:
            strictly_passing_finalists.append((
                finalist, strict_arguments, strict_report, strict_report_path,
            ))

    # The stratified 30-symbol stage selects and freezes parameters, but it no
    # longer grants adequacy authority.  Prefer a finalist which passed the
    # unchanged small-panel gate; otherwise freeze the best structurally
    # eligible robust finalist and record the small-panel miss explicitly.
    # In both cases only ``expand-full-universe`` may authorize 2020 use.
    if strictly_passing_finalists:
        strictly_passing_finalists.sort(
            key=lambda item: candidate_sort_key(item[0])
        )
        (winner, strict_arguments, strict_training_report,
         strict_training_report_path) = strictly_passing_finalists[0]
        small_panel_strict_passed = True
    else:
        winner = eligible_finalists[0]
        strict_arguments = list(
            winner["strict_training_evaluator_arguments"]  # type: ignore[arg-type]
        )
        evaluation = winner["strict_training_evaluation"]
        if not isinstance(evaluation, Mapping):
            raise CalibrationDriverError(
                "diagnostic finalist lacks strict-evaluation provenance"
            )
        strict_training_report_path = pathlib.Path(
            str(evaluation["path"])
        ).resolve()
        strict_training_report = json.loads(
            strict_training_report_path.read_text(encoding="utf-8")
        )
        small_panel_strict_passed = False
    write_json(output_root / "stage3" / "stage3_result.json", {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "small_panel_strict_finalist_selected_pending_full_universe"
            if small_panel_strict_passed else
            "small_panel_robust_finalist_selected_pending_full_universe"
        ),
        "duration_seconds": STAGE3_DURATION,
        "metrics": list(STAGE3_METRICS),
        "selected_finalist_id": winner["candidate_id"],
        "small_panel_strict_training_gate_passed": (
            small_panel_strict_passed
        ),
        "full_universe_training_adequacy_required_before_heldout": True,
        "finalists": finalist_records,
    })

    selected_value = {
        cluster: ValueCandidate(**payload)  # type: ignore[arg-type]
        for cluster, payload in winner["selected_value_by_cluster"].items()  # type: ignore[union-attr]
    }
    selected_volatility = {
        cluster: VolatilityCandidate(**payload)  # type: ignore[arg-type]
        for cluster, payload in winner["selected_volatility_by_cluster"].items()  # type: ignore[union-attr]
    }
    frozen_root = output_root / "frozen_model"
    frozen_config = frozen_root / "deployment_config.csv"
    write_derived_config(
        frozen_config,
        deployment_config,
        clusters=clusters,
        volatility_by_cluster=selected_volatility,
    )
    frozen_value_policy = frozen_root / "value_policy.csv"
    write_value_policy(
        frozen_value_policy, tuple(sorted(clusters)), clusters, selected_value
    )
    frozen_background = frozen_root / "background_policy_mapping.csv"
    write_subset_background_mapping(
        frozen_background, background_policy, tuple(sorted(clusters))
    )

    write_json(output_root / "strict_training_evaluator_arguments.json", {
        "script": str(SCRIPT_DIR / "evaluate_strict_model_validation.py"),
        "arguments": strict_arguments,
        "automatic_execution_required_before_freeze": True,
        "selected_finalist_id": winner["candidate_id"],
    })

    command_artifacts = command_artifact_paths(
        stage1_records, global_refinement_records, finalist_records,
    )
    command_artifacts["configs"].add(frozen_config)
    command_artifacts["background_policies"].add(frozen_background)
    command_artifacts["value_policies"].add(frozen_value_policy)
    command_artifacts["executables"].add(executable)
    transitive_manifest = transitive_runtime_artifacts(
        configs=tuple(command_artifacts["configs"]),
        background_policies=tuple(command_artifacts["background_policies"]),
        value_policies=tuple(command_artifacts["value_policies"]),
        executables=tuple(command_artifacts["executables"]),
        summaries=tuple(command_artifacts["summaries"]),
        selection_records=tuple(
            path for stage in ("stage1", "stage2", "stage3")
            for path in (output_root / stage).glob("**/*_result.json")
        ) + tuple(
            (output_root / "stage2").glob(
                "**/screen_artifact_retention.json"
            )
        ) + (
            output_root / "training_inputs_manifest.json",
            output_root / "strict_training_evaluator_arguments.json",
            strict_training_report_path,
        ),
        workflow_sources=(
            pathlib.Path(__file__),
            pathlib.Path(legacy.__file__),
            pathlib.Path(legacy.cohort.__file__),
            pathlib.Path(strict.__file__),
        ),
        candidate_config=candidate_path,
        cluster_map=cluster_path,
    )

    freeze = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "stratified_training_selection_frozen_pending_full_universe"
        ),
        "training_only": True,
        "heldout_inputs_read": False,
        "frozen_before_any_heldout_run": True,
        "heldout_execution_authorized": False,
        "full_universe_training_adequacy_required": True,
        "small_panel_strict_training_gate_passed": (
            small_panel_strict_passed
        ),
        "stratified_selection_symbol_count": len(clusters),
        "training_dates": sorted(configs),
        "stage_durations_seconds": [STAGE1_DURATION, STAGE2_DURATION, STAGE3_DURATION],
        "ordinary_market_shared_mm_disabled": True,
        "one_rank_execution": True,
        "execution": {
            "mpi_ranks_per_run": 1,
            "maximum_concurrent_runs": args.run_workers,
            "parallelism": "independent_run_task_parallelism",
        },
        "training_inputs_manifest": {
            "path": str(output_root / "training_inputs_manifest.json"),
            "sha256": sha256_file(
                output_root / "training_inputs_manifest.json"
            ),
        },
        "training_only_global_refinement_protocol": inputs_manifest[
            "training_only_global_refinement_protocol"
        ],
        "selection": {
            "local_candidate": winner["local_candidate"],
            "value_by_cluster": winner["selected_value_by_cluster"],
            "volatility_by_cluster": winner["selected_volatility_by_cluster"],
            "stage3_score": winner["score"],
            "training_only_global_refinement": winner[
                "global_refinement_source"
            ],
        },
        "frozen_artifacts": {
            "deployment_config": {"path": str(frozen_config), "sha256": sha256_file(frozen_config)},
            "value_policy": {"path": str(frozen_value_policy), "sha256": sha256_file(frozen_value_policy)},
            "background_policy_mapping": {"path": str(frozen_background), "sha256": sha256_file(frozen_background)},
            "cluster_map": {"path": str(cluster_path), "sha256": sha256_file(cluster_path)},
            "candidate_config": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "executable": {"path": str(executable), "sha256": sha256_file(executable)},
        },
        "stage3_runs": winner["runs"],
        "transitive_runtime_artifacts": transitive_manifest,
        "strict_training_evaluator_arguments": str(
            output_root / "strict_training_evaluator_arguments.json"
        ),
        "strict_training_gate_passed": small_panel_strict_passed,
        "small_panel_strict_training_report": {
            "path": str(strict_training_report_path),
            "sha256": sha256_file(strict_training_report_path),
        },
        "validation_claimed": False,
        "certification_claimed": False,
    }
    freeze_path = output_root / "training_selection_freeze.json"
    write_json(freeze_path, freeze)
    return {
        "status": freeze["status"],
        "training_selection_freeze": str(freeze_path),
        "training_selection_freeze_sha256": sha256_file(freeze_path),
        "selected_finalist_id": winner["candidate_id"],
        "heldout_execution_authorized": False,
        "validation_claimed": False,
    }


def verified_artifact(record: Mapping[str, object], label: str) -> pathlib.Path:
    path = pathlib.Path(str(record.get("path", ""))).resolve()
    expected = str(record.get("sha256", ""))
    if not path.is_file() or sha256_file(path) != expected:
        raise CalibrationDriverError(f"frozen {label} is missing or hash-mismatched")
    return path


def verified_training_freeze(
    freeze_path: pathlib.Path,
    *,
    allowed_statuses: Sequence[str],
) -> dict[str, object]:
    """Read a freeze only after every file it binds still matches its hash."""
    if not freeze_path.is_file():
        raise CalibrationDriverError(
            f"freeze record does not exist: {freeze_path}"
        )
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationDriverError(
            f"cannot read freeze record {freeze_path}: {error}"
        ) from error
    if not isinstance(freeze, dict):
        raise CalibrationDriverError("freeze record is not a JSON object")
    if (
        freeze.get("status") not in set(allowed_statuses)
        or freeze.get("heldout_inputs_read") is not False
        or freeze.get("frozen_before_any_heldout_run") is not True
    ):
        raise CalibrationDriverError(
            "freeze record does not authorize this post-freeze operation"
        )
    transitive_manifest = freeze.get("transitive_runtime_artifacts")
    if not isinstance(transitive_manifest, Mapping):
        raise CalibrationDriverError(
            "freeze record lacks transitive runtime artifacts"
        )
    verify_transitive_runtime_artifacts(transitive_manifest)
    artifacts = freeze.get("frozen_artifacts")
    if not isinstance(artifacts, Mapping):
        raise CalibrationDriverError("freeze record lacks frozen_artifacts")
    for key, label in (
        ("deployment_config", "deployment config"),
        ("value_policy", "value policy"),
        ("background_policy_mapping", "background policy"),
        ("cluster_map", "cluster map"),
        ("candidate_config", "candidate config"),
        ("executable", "executable"),
    ):
        record = artifacts.get(key)
        if not isinstance(record, Mapping):
            raise CalibrationDriverError(f"freeze lacks {label} provenance")
        verified_artifact(record, label)
    return freeze


def mapping_fingerprints(path: pathlib.Path) -> dict[str, dict[str, str]]:
    """Bind mapping semantics by content, independently of path relocation."""
    _, rows = read_background_mapping(path)
    return {
        str(row["symbol"]): {
            "cluster_id": str(row["cluster_id"]).strip(),
            "policy_sha256": sha256_file(pathlib.Path(row["policy_file"])),
            "limit_buy_improvement_sha256": sha256_file(
                pathlib.Path(row["limit_buy_improvement_file"])
            ),
            "limit_sell_improvement_sha256": sha256_file(
                pathlib.Path(row["limit_sell_improvement_file"])
            ),
        }
        for row in rows
    }


def verify_estimation_subset(
    *,
    freeze: Mapping[str, object],
    full_deployment: ConfigTable,
    full_clusters: Mapping[str, str],
    full_background: pathlib.Path,
) -> tuple[int, int]:
    """Prove that expansion applies the frozen fit to a consistent superset."""
    artifacts = freeze["frozen_artifacts"]
    if not isinstance(artifacts, Mapping):
        raise CalibrationDriverError("freeze lacks frozen artifacts")
    fit_cluster_path = verified_artifact(
        artifacts["cluster_map"], "estimation cluster map",  # type: ignore[arg-type]
    )
    fit_background_path = verified_artifact(
        artifacts["background_policy_mapping"],
        "estimation background mapping",  # type: ignore[arg-type]
    )
    fit_clusters = read_cluster_map(fit_cluster_path)
    fit_symbols = set(fit_clusters)
    full_symbols = set(full_deployment.symbols)
    if not fit_symbols < full_symbols:
        raise CalibrationDriverError(
            "full-universe expansion must be a strict superset of the frozen "
            "estimation universe"
        )
    if set(full_clusters) != full_symbols:
        raise CalibrationDriverError(
            "full deployment universe differs from full cluster map"
        )
    cluster_mismatches = sorted(
        symbol for symbol in fit_symbols
        if full_clusters.get(symbol) != fit_clusters[symbol]
    )
    if cluster_mismatches:
        raise CalibrationDriverError(
            "full cluster map changes frozen estimation assignments: "
            + ", ".join(cluster_mismatches[:10])
        )

    manifest_record = freeze.get("training_inputs_manifest")
    if not isinstance(manifest_record, Mapping):
        raise CalibrationDriverError(
            "freeze lacks its training-input manifest provenance"
        )
    manifest_path = verified_artifact(
        manifest_record, "training-input manifest"
    )
    try:
        training_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationDriverError(
            f"cannot read training-input manifest: {error}"
        ) from error
    deployment_record = training_manifest.get("deployment_config")
    if not isinstance(deployment_record, Mapping):
        raise CalibrationDriverError(
            "training-input manifest lacks estimation deployment config"
        )
    fit_deployment_path = verified_artifact(
        deployment_record, "estimation source deployment config"
    )
    fit_deployment = read_config(fit_deployment_path)
    if set(fit_deployment.symbols) != fit_symbols:
        raise CalibrationDriverError(
            "estimation source deployment universe differs from frozen map"
        )
    full_by_symbol = {
        str(row["symbol"]): row for row in full_deployment.rows
    }
    shared_fields = sorted(
        set(fit_deployment.fields).intersection(full_deployment.fields)
        .difference({"book_id", TARGET_DIRECTORY_FIELD})
    )
    for fit_row in fit_deployment.rows:
        symbol = str(fit_row["symbol"])
        full_row = full_by_symbol[symbol]
        mismatched = [
            field for field in shared_fields
            if str(fit_row.get(field, "")) != str(full_row.get(field, ""))
        ]
        if mismatched:
            raise CalibrationDriverError(
                f"full deployment changes frozen estimation inputs for "
                f"{symbol}: {mismatched[:10]}"
            )

    fit_mapping = mapping_fingerprints(fit_background_path)
    full_mapping = mapping_fingerprints(full_background)
    if set(full_mapping) != full_symbols:
        raise CalibrationDriverError(
            "full background-policy universe differs from full deployment"
        )
    mapping_mismatches = sorted(
        symbol for symbol in fit_symbols
        if full_mapping.get(symbol) != fit_mapping[symbol]
    )
    if mapping_mismatches:
        raise CalibrationDriverError(
            "full background mapping changes frozen estimation policies: "
            + ", ".join(mapping_mismatches[:10])
        )
    return len(fit_symbols), len(full_symbols)


def bounded(value: float, lower: float, upper: float) -> float:
    """Return a finite value projected onto a declared numerical interval."""
    if not math.isfinite(value):
        raise CalibrationDriverError("cannot bound a non-finite value")
    if not lower <= upper:
        raise CalibrationDriverError("invalid numerical projection interval")
    return min(upper, max(lower, value))


def latent_absolute_return_acf(
    persistence: float,
    stationary_std: float,
) -> float:
    """Analytic lag-one |return| ACF contributed by log volatility.

    The expression uses a unit-variance Gaussian innovation as a transparent
    moment bridge.  It is not treated as a fit result: the next simulator
    iteration measures the realised LOB response and corrects any discrepancy.
    """
    if stationary_std <= 0.0:
        return 0.0
    if not 0.0 <= persistence < 1.0:
        raise CalibrationDriverError("invalid log-volatility persistence")
    squared_std = stationary_std * stationary_std
    mean_absolute_squared = 2.0 / math.pi
    numerator = mean_absolute_squared * (
        math.exp(0.25 * squared_std * (persistence - 1.0))
        - math.exp(-0.25 * squared_std)
    )
    denominator = 1.0 - mean_absolute_squared * math.exp(
        -0.25 * squared_std
    )
    if denominator <= 0.0:
        raise CalibrationDriverError("invalid latent ACF denominator")
    return numerator / denominator


def invert_latent_absolute_return_acf(
    target: float,
    *,
    persistence: float,
    maximum_std: float = 0.75,
) -> float:
    """Invert ``latent_absolute_return_acf`` by deterministic bisection."""
    if target <= 0.0:
        return 0.0
    attainable = latent_absolute_return_acf(persistence, maximum_std)
    if target >= attainable:
        return maximum_std
    lower = 0.0
    upper = maximum_std
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if latent_absolute_return_acf(persistence, midpoint) < target:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def geometric_median(values: Sequence[float], *, label: str) -> float:
    positive = [value for value in values if math.isfinite(value) and value > 0.0]
    if not positive:
        raise CalibrationDriverError(f"{label} has no finite positive values")
    return math.exp(statistics.median(math.log(value) for value in positive))


def refine_full_universe_volatility(
    *,
    symbol_residuals: pathlib.Path,
    current: Mapping[str, VolatilityCandidate],
    cluster_ids: set[str],
    iteration: int,
) -> tuple[dict[str, VolatilityCandidate], dict[str, object]]:
    """One bounded simulated-moment update using 2019 training residuals.

    Variance scale and tail transmission receive multiplicative robust updates.
    The persistent stochastic-baseline loading receives the primary ACF
    update.  This changes the accepted-event clock through active and quiet
    intervals while preserving immigration in expectation. Every update is
    bounded, recorded and re-evaluated by the simulator; no validation target
    is read.
    """
    if not symbol_residuals.is_file():
        raise CalibrationDriverError(
            "strict evaluator did not write symbol_residuals.csv; cannot "
            "perform full-universe training refinement"
        )
    with symbol_residuals.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        required = {
            "cluster_id", "metric", "target", "simulated_seed_mean",
        }
        missing = sorted(required.difference(fields))
        if missing:
            raise CalibrationDriverError(
                f"symbol residual table lacks columns: {missing}"
            )
        rows = list(reader)
    if not rows:
        raise CalibrationDriverError("symbol residual table is empty")

    refined: dict[str, VolatilityCandidate] = {}
    records: list[dict[str, object]] = []
    for cluster in sorted(cluster_ids, key=lambda value: (len(value), value)):
        if cluster not in current:
            raise CalibrationDriverError(
                f"current volatility map lacks cluster {cluster}"
            )
        by_metric: dict[str, list[tuple[float, float]]] = {}
        for row in rows:
            if str(row["cluster_id"]) != cluster:
                continue
            metric = str(row["metric"])
            target = finite_float(
                row["target"], label=f"cluster {cluster} {metric} target",
            )
            simulated = finite_float(
                row["simulated_seed_mean"],
                label=f"cluster {cluster} {metric} simulated value",
            )
            by_metric.setdefault(metric, []).append((target, simulated))
        for metric in (
            "return_variance", "return_kurtosis", "absolute_return_acf1",
        ):
            if not by_metric.get(metric):
                raise CalibrationDriverError(
                    f"cluster {cluster} lacks {metric} residuals"
                )

        variance_ratio = geometric_median(
            [target / simulated for target, simulated
             in by_metric["return_variance"] if simulated > 0.0],
            label=f"cluster {cluster} return-variance ratio",
        )
        kurtosis_ratio = geometric_median(
            [target / simulated for target, simulated
             in by_metric["return_kurtosis"] if simulated > 0.0],
            label=f"cluster {cluster} return-kurtosis ratio",
        )
        acf_error = statistics.fmean(
            target - simulated
            for target, simulated in by_metric["absolute_return_acf1"]
        )

        previous = current[cluster]
        # Damping prevents a noisy one-seed or finite-sample residual from
        # becoming a disproportionate structural parameter change.
        variance_factor = bounded(variance_ratio, 0.40, 2.50) ** 0.85
        kurtosis_factor = bounded(kurtosis_ratio, 0.40, 2.50) ** 0.70
        variance_scale = bounded(
            previous.variance_scale * variance_factor, 0.05, 8.0,
        )
        tail_multiplier = bounded(
            previous.tail_transmission_multiplier * kurtosis_factor,
            0.25, 24.0,
        )

        current_latent_acf = latent_absolute_return_acf(
            previous.persistence, previous.std,
        )
        target_latent_acf = bounded(
            current_latent_acf + 0.20 * acf_error, 0.0, 0.20,
        )
        persistence = 0.0 if target_latent_acf == 0.0 else 0.95
        stationary_std = invert_latent_absolute_return_acf(
            target_latent_acf,
            persistence=persistence if persistence > 0.0 else 0.95,
        )
        order_flow_coupling = bounded(
            previous.order_flow_coupling + 7.0 * acf_error,
            0.0,
            2.5,
        )
        if order_flow_coupling > 0.0:
            persistence = max(persistence, 0.95)
            stationary_std = max(stationary_std, 0.25)
        candidate = VolatilityCandidate(
            identifier=f"full_refinement_{iteration}_cluster_{cluster}",
            variance_scale=variance_scale,
            persistence=persistence,
            std=stationary_std,
            excess_kurtosis_share=previous.excess_kurtosis_share,
            tail_transmission_multiplier=tail_multiplier,
            order_flow_coupling=order_flow_coupling,
        )
        refined[cluster] = candidate
        records.append({
            "cluster_id": cluster,
            "source_candidate": previous.__dict__,
            "variance_target_to_simulated_geometric_median": variance_ratio,
            "kurtosis_target_to_simulated_geometric_median": kurtosis_ratio,
            "absolute_return_acf_target_minus_simulated_mean": acf_error,
            "variance_update_factor_after_damping": variance_factor,
            "kurtosis_update_factor_after_damping": kurtosis_factor,
            "target_latent_absolute_return_acf": target_latent_acf,
            "order_flow_coupling_update_gain": 7.0,
            "order_flow_coupling_after_projection": order_flow_coupling,
            "refined_candidate": candidate.__dict__,
        })
    return refined, {
        "iteration": iteration,
        "training_only": True,
        "heldout_inputs_read": False,
        "method": "session_recentred_cluster_activity_update_v1",
        "records": records,
    }


def load_training_refinement_seed(
    path: pathlib.Path,
    *,
    cluster_ids: set[str],
) -> tuple[dict[str, VolatilityCandidate], dict[str, object]]:
    """Load a hash-bound seed derived only from prior training residuals.

    This is a warm start, never a passed fit.  Both referenced R28 evidence
    files must be present and hash-identical before any simulator is launched.
    """
    manifest_path = path.expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationDriverError(
            f"cannot read training refinement seed {manifest_path}: {error}"
        ) from error
    if payload.get("schema_version") != 1 \
            or payload.get("status") != "training_only_refinement_seed" \
            or payload.get("training_only") is not True \
            or payload.get("heldout_inputs_read") is not False:
        raise CalibrationDriverError(
            "training refinement seed has an invalid evidential role"
        )
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise CalibrationDriverError("training refinement seed lacks source evidence")
    verified_sources: dict[str, dict[str, str]] = {}
    for key in ("strict_report", "symbol_residuals"):
        raw_path = pathlib.Path(str(source.get(key, "")))
        source_path = raw_path if raw_path.is_absolute() \
            else (manifest_path.parent / raw_path).resolve()
        expected_hash = str(source.get(f"{key}_sha256", ""))
        if not source_path.is_file() or not expected_hash \
                or sha256_file(source_path) != expected_hash:
            raise CalibrationDriverError(
                f"training refinement seed evidence is missing or "
                f"hash-mismatched: {source_path}"
            )
        verified_sources[key] = {
            "path": str(source_path),
            "sha256": expected_hash,
        }
    strict_report = json.loads(
        pathlib.Path(verified_sources["strict_report"]["path"])
        .read_text(encoding="utf-8")
    )
    if strict_report.get("evaluation_role") != "training_fit" \
            or strict_report.get("passed") is not False:
        raise CalibrationDriverError(
            "warm-start strict report must be a visible failed training fit"
        )

    verified_repair_sources: dict[str, dict[str, str]] = {}
    repair = payload.get("structural_repair_evidence")
    if repair is not None:
        if not isinstance(repair, Mapping) \
                or repair.get("heldout_inputs_read") is not False:
            raise CalibrationDriverError(
                "structural repair evidence has an invalid evidential role"
            )
        for key in (
            "pilot_decision", "strict_report", "acf_distribution",
            "symbol_residuals", "same_seed_comparison",
        ):
            raw_path = pathlib.Path(str(repair.get(key, "")))
            repair_path = raw_path if raw_path.is_absolute() \
                else (manifest_path.parent / raw_path).resolve()
            expected_hash = str(repair.get(f"{key}_sha256", ""))
            if not repair_path.is_file() or not expected_hash \
                    or sha256_file(repair_path) != expected_hash:
                raise CalibrationDriverError(
                    "structural repair evidence is missing or "
                    f"hash-mismatched: {repair_path}"
                )
            verified_repair_sources[key] = {
                "path": str(repair_path), "sha256": expected_hash,
            }
        summaries = repair.get("r28_same_seed_summaries")
        expected_summary_dates = {
            "2019-01-30", "2019-03-27", "2019-10-30",
        }
        if not isinstance(summaries, Mapping) \
                or set(map(str, summaries)) != expected_summary_dates:
            raise CalibrationDriverError(
                "structural repair evidence lacks the exact R28 same-seed "
                "summary set"
            )
        for day in sorted(expected_summary_dates):
            summary = summaries.get(day)
            if not isinstance(summary, Mapping):
                raise CalibrationDriverError(
                    f"R28 same-seed summary is malformed for {day}"
                )
            raw_path = pathlib.Path(str(summary.get("path", "")))
            summary_path = raw_path if raw_path.is_absolute() \
                else (manifest_path.parent / raw_path).resolve()
            expected_hash = str(summary.get("sha256", ""))
            if not summary_path.is_file() or not expected_hash \
                    or sha256_file(summary_path) != expected_hash:
                raise CalibrationDriverError(
                    "R28 same-seed evidence is missing or hash-mismatched: "
                    f"{summary_path}"
                )
            verified_repair_sources[f"r28_summary_{day}"] = {
                "path": str(summary_path), "sha256": expected_hash,
            }
        pilot = json.loads(
            pathlib.Path(verified_repair_sources["pilot_decision"]["path"])
            .read_text(encoding="utf-8")
        )
        repair_report = json.loads(
            pathlib.Path(verified_repair_sources["strict_report"]["path"])
            .read_text(encoding="utf-8")
        )
        if pilot.get("training_only") is not True \
                or pilot.get("status") != "rejected" \
                or repair_report.get("evaluation_role") != "training_fit" \
                or repair_report.get("passed") is not False:
            raise CalibrationDriverError(
                "structural repair evidence must document a rejected "
                "training-only pilot"
            )

    raw_candidates = payload.get("volatility_by_cluster")
    if not isinstance(raw_candidates, Mapping) \
            or set(map(str, raw_candidates)) != cluster_ids:
        raise CalibrationDriverError(
            "training refinement seed does not cover the frozen clusters"
        )
    candidates: dict[str, VolatilityCandidate] = {}
    for cluster in sorted(cluster_ids, key=lambda value: (len(value), value)):
        raw = raw_candidates.get(cluster)
        if not isinstance(raw, Mapping):
            raise CalibrationDriverError(
                f"training refinement seed cluster {cluster} is malformed"
            )
        candidate = VolatilityCandidate(**raw)  # type: ignore[arg-type]
        numeric = (
            candidate.variance_scale, candidate.persistence, candidate.std,
            candidate.excess_kurtosis_share,
            candidate.tail_transmission_multiplier,
            candidate.order_flow_coupling,
        )
        if not all(math.isfinite(value) for value in numeric) \
                or not 0.05 <= candidate.variance_scale <= 8.0 \
                or not 0.0 <= candidate.persistence < 1.0 \
                or not 0.0 <= candidate.std <= 0.75 \
                or not 0.0 <= candidate.excess_kurtosis_share <= 1.0 \
                or not 0.25 <= candidate.tail_transmission_multiplier <= 24.0 \
                or not 0.0 <= candidate.order_flow_coupling <= 2.5 \
                or (candidate.order_flow_coupling > 0.0
                    and candidate.std <= 0.0):
            raise CalibrationDriverError(
                f"training refinement seed cluster {cluster} is out of bounds"
            )
        candidates[cluster] = candidate
    return candidates, {
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "verified_training_sources": verified_sources,
        "verified_structural_repair_sources": verified_repair_sources,
        "training_only": True,
        "heldout_inputs_read": False,
    }


def scale_training_coupling(
    candidates: Mapping[str, VolatilityCandidate],
    scale: float,
) -> tuple[dict[str, VolatilityCandidate], dict[str, object]]:
    """Apply one recorded global shrinkage/expansion to a training seed.

    This is deliberately a single scalar rather than ten independently tuned
    cluster parameters. It permits a cheap training-only sensitivity rerun
    without opening a 10-dimensional search or reading held-out targets.
    """
    if not math.isfinite(scale) or not 0.5 <= scale <= 1.25:
        raise CalibrationDriverError(
            "--initial-coupling-scale must be finite and in [0.5,1.25]"
        )
    scaled: dict[str, VolatilityCandidate] = {}
    records: list[dict[str, object]] = []
    for cluster, previous in sorted(
        candidates.items(), key=lambda item: (len(item[0]), item[0]),
    ):
        coupling = bounded(previous.order_flow_coupling * scale, 0.0, 2.5)
        scaled[cluster] = VolatilityCandidate(
            identifier=f"{previous.identifier}_scale_{scale:g}",
            variance_scale=previous.variance_scale,
            persistence=previous.persistence,
            std=previous.std,
            excess_kurtosis_share=previous.excess_kurtosis_share,
            tail_transmission_multiplier=previous.tail_transmission_multiplier,
            order_flow_coupling=coupling,
        )
        records.append({
            "cluster_id": cluster,
            "unscaled_coupling": previous.order_flow_coupling,
            "scaled_coupling": coupling,
        })
    return scaled, {
        "training_only": True,
        "heldout_inputs_read": False,
        "global_coupling_scale": scale,
        "projection_bounds": [0.0, 2.5],
        "records": records,
    }


def expand_full_universe(args: argparse.Namespace) -> dict[str, object]:
    """Expand a panel fit and calibrate it on the larger 2019 universe.

    This phase has deliberately no held-out arguments.  A bounded number of
    cluster-level simulated-moment updates may use only 2019 residuals.  The
    final parameters must pass the same immutable strict evaluator on every
    full-universe training session before an expanded freeze can authorize
    development validation.
    """
    output_root = args.output_root.expanduser().resolve()
    prepare_output(output_root, resume=args.resume)
    freeze_path = args.freeze_record.expanduser().resolve()
    freeze = verified_training_freeze(
        freeze_path,
        allowed_statuses=(
            "stratified_training_selection_frozen_pending_full_universe",
        ),
    )
    if freeze.get("training_only") is not True:
        raise CalibrationDriverError(
            "only a training-only selection freeze may be expanded"
        )
    artifacts = freeze["frozen_artifacts"]
    if not isinstance(artifacts, Mapping):
        raise CalibrationDriverError("freeze lacks frozen artifacts")
    candidate_path = verified_artifact(
        artifacts["candidate_config"], "candidate config",  # type: ignore[arg-type]
    )
    selection_executable = verified_artifact(
        artifacts["executable"], "executable",  # type: ignore[arg-type]
    )
    executable = selection_executable
    if args.executable is not None:
        executable = args.executable.expanduser().resolve()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise CalibrationDriverError(
                f"replacement runtime executable is missing or not executable: "
                f"{executable}"
            )
    protocol = load_candidate_protocol(candidate_path)
    launcher = parse_launcher(args.launcher)
    production_ranks = launcher_rank_count(launcher)
    gate_protocol = str(args.gate_protocol)
    protocol_revision_certificate: dict[str, object] | None = None
    if gate_protocol == strict.MARKETWIDE_SIX_GATE:
        if args.six_component_protocol_certificate is None:
            raise CalibrationDriverError(
                "marketwide-six-v2 requires the visible retrospective "
                "six-component protocol certificate"
            )
        protocol_revision_certificate = verified_six_component_certificate(
            args.six_component_protocol_certificate
        )
    elif args.six_component_protocol_certificate is not None:
        raise CalibrationDriverError(
            "--six-component-protocol-certificate is valid only with "
            "marketwide-six-v2"
        )
    if args.run_workers <= 0:
        raise CalibrationDriverError("--run-workers must be positive")
    if not 1 <= args.rank_equivalence_duration <= STAGE3_DURATION:
        raise CalibrationDriverError(
            f"--rank-equivalence-duration must be in [1,{STAGE3_DURATION}]"
        )

    full_deployment_path = args.full_deployment_config.expanduser().resolve()
    full_cluster_path = args.full_cluster_map.expanduser().resolve()
    full_background_path = args.full_background_policy.expanduser().resolve()
    for path, label in (
        (full_deployment_path, "full deployment config"),
        (full_cluster_path, "full cluster map"),
        (full_background_path, "full background mapping"),
    ):
        if not path.is_file():
            raise CalibrationDriverError(f"{label} does not exist: {path}")
    full_deployment = read_config(full_deployment_path)
    full_clusters = read_cluster_map(full_cluster_path)
    fit_count, full_count = verify_estimation_subset(
        freeze=freeze,
        full_deployment=full_deployment,
        full_clusters=full_clusters,
        full_background=full_background_path,
    )
    training_paths = dated_mapping(
        args.training_config,
        option="--training-config",
        exact_count=TRAINING_DAY_COUNT,
    )
    if sorted(training_paths) != sorted(freeze.get("training_dates", [])):
        raise CalibrationDriverError(
            "full-universe training dates differ from the frozen selection dates"
        )
    dated_observations = {
        day: read_config(path) for day, path in training_paths.items()
    }
    validate_universes(
        dated_observations, full_background_path, full_clusters,
    )

    selection = freeze.get("selection")
    if not isinstance(selection, Mapping):
        raise CalibrationDriverError("freeze lacks selected model parameters")
    local_payload = selection.get("local_candidate")
    value_payload = selection.get("value_by_cluster")
    volatility_payload = selection.get("volatility_by_cluster")
    if not isinstance(local_payload, Mapping) \
            or not isinstance(value_payload, Mapping) \
            or not isinstance(volatility_payload, Mapping):
        raise CalibrationDriverError("freeze selection payload is malformed")
    local = LocalCandidate(**local_payload)  # type: ignore[arg-type]
    selected_value = {
        str(cluster): ValueCandidate(**payload)  # type: ignore[arg-type]
        for cluster, payload in value_payload.items()
    }
    selected_volatility = {
        str(cluster): VolatilityCandidate(**payload)  # type: ignore[arg-type]
        for cluster, payload in volatility_payload.items()
    }
    cluster_ids = set(full_clusters.values())
    if set(selected_value) != cluster_ids or set(selected_volatility) != cluster_ids:
        raise CalibrationDriverError(
            "frozen cluster policies do not cover the full-universe cluster map"
        )

    if not 0 <= args.max_refinement_iterations <= 4:
        raise CalibrationDriverError(
            "--max-refinement-iterations must be between zero and four"
        )
    if not 0 <= args.minimum_refinement_iterations \
            <= args.max_refinement_iterations:
        raise CalibrationDriverError(
            "--minimum-refinement-iterations must be between zero and "
            "--max-refinement-iterations"
        )

    if args.directional_pilot_only and not args.directional_pilot:
        raise CalibrationDriverError(
            "--directional-pilot-only requires --directional-pilot"
        )

    frozen_root = output_root / "frozen_full_universe_model"
    expanded_value = frozen_root / "value_policy.csv"
    write_value_policy(
        expanded_value, full_deployment.symbols,
        full_clusters, selected_value,
    )
    expanded_background = frozen_root / "background_policy_mapping.csv"
    write_subset_background_mapping(
        expanded_background, full_background_path, full_deployment.symbols,
    )
    expanded_clusters = frozen_root / "cluster_map.csv"
    write_csv(
        expanded_clusters, ("symbol", "cluster_id"),
        ({"symbol": symbol, "cluster_id": full_clusters[symbol]}
         for symbol in full_deployment.symbols),
    )

    target_preflight: list[dict[str, object]] | None = None
    inputs_manifest_path = output_root / "full_universe_expansion_inputs.json"
    expansion_inputs = {
        "schema_version": SCHEMA_VERSION,
        "status": "full_universe_training_adequacy_pending",
        "source_training_freeze": {
            "path": str(freeze_path), "sha256": sha256_file(freeze_path),
        },
        "selection_parameters_changed": False,
        "refinement_protocol": {
            "enabled": args.max_refinement_iterations > 0,
            "maximum_updates": args.max_refinement_iterations,
            "method": "session_recentred_cluster_activity_update_v1",
            "moments": [
                "return_variance", "return_kurtosis",
                "absolute_return_acf1",
            ],
            "training_only": True,
            "adequacy_protocol_unchanged_within_continuation": True,
        },
        "training_adequacy_protocol": gate_protocol,
        "protocol_revision_certificate": protocol_revision_certificate,
        "minimum_refinement_iterations": args.minimum_refinement_iterations,
        "estimation_symbol_count": fit_count,
        "full_universe_symbol_count": full_count,
        "runtime_executable": {
            "path": str(executable),
            "sha256": sha256_file(executable),
            "replaces_selection_executable": executable != selection_executable,
            "selection_executable": {
                "path": str(selection_executable),
                "sha256": sha256_file(selection_executable),
            },
        },
        "training_dates": sorted(dated_observations),
        "training_seeds": list(protocol.stage3_seeds),
        "duration_seconds": STAGE3_DURATION,
        "one_rank_execution": production_ranks == 1,
        "execution": {
            "mpi_ranks_per_run": production_ranks,
            "maximum_concurrent_runs": args.run_workers,
            "parallelism": "whole_book_mpi",
            "rank_equivalence_required": production_ranks > 1,
            "rank_equivalence": None,
        },
        "heldout_inputs_read": False,
        "development_validation_targets_opened": False,
        "target_preflight": None,
        "full_deployment_config": {
            "path": str(full_deployment_path),
            "sha256": sha256_file(full_deployment_path),
        },
        "full_background_policy": {
            "path": str(full_background_path),
            "sha256": sha256_file(full_background_path),
        },
        "full_cluster_map": {
            "path": str(full_cluster_path),
            "sha256": sha256_file(full_cluster_path),
        },
    }
    rank_equivalence_path: pathlib.Path | None = None
    directional_pilot_record: dict[str, object] | None = None
    current_volatility = dict(selected_volatility)
    warm_start_record: dict[str, object] | None = None
    if args.initial_refinement_manifest is not None:
        current_volatility, warm_start_record = load_training_refinement_seed(
            args.initial_refinement_manifest,
            cluster_ids=cluster_ids,
        )
        current_volatility, scale_record = scale_training_coupling(
            current_volatility, args.initial_coupling_scale,
        )
        warm_start_record["coupling_scale"] = scale_record
        expansion_inputs["training_refinement_warm_start"] = warm_start_record
        expansion_inputs["selection_parameters_changed"] = True
    elif args.initial_coupling_scale != 1.0:
        raise CalibrationDriverError(
            "--initial-coupling-scale requires --initial-refinement-manifest"
        )
    refinement_records: list[dict[str, object]] = []
    iteration_runtime: list[dict[str, object]] = []
    evidence_paths: list[pathlib.Path] = []
    all_iteration_runs: list[dict[str, list[dict[str, object]]]] = []
    strict_passed = False
    strict_report: Mapping[str, object] = {}
    strict_report_path = output_root / "strict_report_not_written.json"
    strict_record = output_root / "strict_arguments_not_written.json"
    runs: dict[str, list[dict[str, object]]] = {}
    configs: dict[str, ConfigTable] = {}
    expanded_config = frozen_root / "deployment_config.csv"

    for iteration in range(args.max_refinement_iterations + 1):
        iteration_root = (
            frozen_root if iteration == 0 else
            output_root / "full_universe_refinement" / f"iteration_{iteration}"
        )
        expanded_config = iteration_root / "deployment_config.csv"
        write_derived_config(
            expanded_config, full_deployment,
            clusters=full_clusters,
            volatility_by_cluster=current_volatility,
        )
        expanded_table = read_config(expanded_config)
        prepared_root = iteration_root / "prepared_training_configs"
        configs = {
            day: prepare_dated_config(
                prepared_root / f"{day}.csv",
                frozen_base=expanded_table,
                dated_observations=dated_observations[day],
            )
            for day in sorted(dated_observations)
        }
        if target_preflight is None:
            target_preflight = preflight_training_targets(configs)
        if iteration == 0 and production_ranks > 1:
            if not args.reference_launcher.strip():
                raise CalibrationDriverError(
                    "multi-rank full-universe execution requires an explicit "
                    "--reference-launcher requesting one rank"
                )
            reference_launcher = parse_launcher(args.reference_launcher)
            pilot_day = sorted(configs)[0]
            rank_equivalence_path = verify_rank_equivalence(
                executable=executable,
                reference_launcher=reference_launcher,
                production_launcher=launcher,
                config=configs[pilot_day].path,
                background_policy=expanded_background,
                value_policy=expanded_value,
                local=local,
                day=pilot_day,
                base_seed=protocol.stage3_seeds[0],
                duration=args.rank_equivalence_duration,
                output_root=output_root / "rank_equivalence_preflight",
                timeout_seconds=protocol.timeout_seconds["stage3"],
            )
            expansion_inputs["execution"]["rank_equivalence"] = {  # type: ignore[index]
                "path": str(rank_equivalence_path),
                "sha256": sha256_file(rank_equivalence_path),
            }

        if iteration == 0 and args.directional_pilot:
            # Three historically difficult 2019 dates and one predeclared
            # seed provide a cheap mechanism check. This is not an adequacy
            # pass; it only prevents launching 25 full runs when the new
            # stochastic-baseline channel plainly failed to move the ACF target.
            preferred = ("2019-01-30", "2019-03-27", "2019-10-30")
            pilot_days = tuple(day for day in preferred if day in configs)
            if len(pilot_days) != len(preferred):
                raise CalibrationDriverError(
                    "directional pilot dates are absent from training configs"
                )
            pilot_runs = run_candidate_matrix(
                executable=executable,
                launcher=launcher,
                configs={day: configs[day].path for day in pilot_days},
                background_policies={day: expanded_background for day in pilot_days},
                value_policy=expanded_value,
                local=local,
                duration=STAGE3_DURATION,
                seeds=(protocol.stage3_seeds[0],),
                output_root=output_root / "directional_pilot" / "runs",
                timeout_seconds=protocol.timeout_seconds["stage3"],
                run_workers=args.run_workers,
            )
            if any(
                record["success"] is not True
                for records in pilot_runs.values() for record in records
            ):
                raise CalibrationDriverError(
                    "directional pilot simulator failure; full matrix was not started"
                )
            pilot_output = output_root / "directional_pilot" / "strict_diagnostics"
            pilot_arguments = [
                "--evaluation-role", "training_fit",
                "--gate-protocol", gate_protocol,
                "--cluster-map", str(expanded_clusters),
                "--expected-cluster-count", str(len(cluster_ids)),
                "--output-dir", str(pilot_output),
            ]
            for day in pilot_days:
                pilot_arguments.extend(("--expected-date", day))
                pilot_arguments.extend(
                    ("--target-config", f"{day}={configs[day].path}")
                )
                record = pilot_runs[day][0]
                command = record["command"]
                seed = command[command.index("--seed") + 1]
                pilot_arguments.extend((
                    "--sim-summary", f"{day}:{seed}={record['summary_path']}",
                ))
            pilot_report, pilot_report_path = run_strict_evaluation(
                pilot_arguments
            )
            date_records = pilot_report.get("date_results")
            if not isinstance(date_records, list):
                raise CalibrationDriverError(
                    "directional pilot evaluator returned no date records"
                )
            pilot_failures: list[str] = []
            for record in date_records:
                if not isinstance(record, Mapping):
                    raise CalibrationDriverError(
                        "directional pilot date record is malformed"
                    )
                day = str(record.get("date"))
                robust = finite_float(
                    record.get("marketwide_robust_score"),
                    label=f"directional pilot {day} robust score",
                )
                gross = finite_float(
                    record.get("gross_failure_symbol_fraction"),
                    label=f"directional pilot {day} gross failure fraction",
                )
                if robust > 1.8:
                    pilot_failures.append(
                        f"{day}: robust score {robust:.6g} exceeds 1.8"
                    )
                if gross > 0.15:
                    pilot_failures.append(
                        f"{day}: gross-failure fraction {gross:.6g} exceeds 0.15"
                    )
            acf_path = pilot_output / "absolute_return_acf_distribution.csv"
            with acf_path.open(newline="", encoding="utf-8") as source:
                acf_rows = list(csv.DictReader(source))
            limits = {"mean": 0.035, "median": 0.040, "p90": 0.055}
            expected_acf_rows = len(pilot_days) * len(limits)
            if len(acf_rows) != expected_acf_rows:
                raise CalibrationDriverError(
                    "directional pilot ACF diagnostics are incomplete"
                )
            for row in acf_rows:
                statistic = str(row.get("statistic"))
                if statistic not in limits:
                    raise CalibrationDriverError(
                        "directional pilot contains an unknown ACF statistic"
                    )
                error = finite_float(
                    row.get("absolute_error"),
                    label="directional pilot ACF absolute error",
                )
                if error > limits[statistic]:
                    pilot_failures.append(
                        f"{row.get('date')}:{statistic} ACF error "
                        f"{error:.6g} exceeds {limits[statistic]:.6g}"
                    )
            directional_pilot_record = {
                "status": "passed" if not pilot_failures else "rejected",
                "authorizes_strict_gate": False,
                "training_only": True,
                "dates": list(pilot_days),
                "base_seed": protocol.stage3_seeds[0],
                "thresholds": {
                    "maximum_marketwide_robust_score": 1.8,
                    "maximum_gross_failure_fraction": 0.15,
                    "maximum_acf_absolute_error": limits,
                },
                "failures": pilot_failures,
                "strict_report": {
                    "path": str(pilot_report_path),
                    "sha256": sha256_file(pilot_report_path),
                    "strict_passed": pilot_report.get("passed") is True,
                },
                "runs": pilot_runs,
            }
            write_json(
                output_root / "directional_pilot" / "pilot_decision.json",
                directional_pilot_record,
            )
            expansion_inputs["directional_pilot"] = directional_pilot_record
            if pilot_failures:
                write_json(inputs_manifest_path, expansion_inputs)
                raise CalibrationDriverError(
                    "directional 2019 pilot rejected the stochastic-baseline "
                    "mechanism; the 25-run strict matrix was not started"
                )
            if args.directional_pilot_only:
                handoff = output_root / "directional_pilot" \
                    / "directional_pilot_handoff.json"
                write_json(handoff, {
                    "schema_version": SCHEMA_VERSION,
                    "status": "directional_pilot_passed_full_matrix_not_run",
                    "training_only": True,
                    "heldout_inputs_read": False,
                    "authorizes_heldout": False,
                    "authorizes_full_training_matrix": True,
                    "pilot_decision": {
                        "path": str(
                            output_root / "directional_pilot"
                            / "pilot_decision.json"
                        ),
                        "sha256": sha256_file(
                            output_root / "directional_pilot"
                            / "pilot_decision.json"
                        ),
                    },
                })
                expansion_inputs["status"] = (
                    "directional_pilot_passed_full_matrix_not_run"
                )
                write_json(inputs_manifest_path, expansion_inputs)
                return {
                    "status": "directional_pilot_passed_full_matrix_not_run",
                    "handoff": str(handoff),
                    "full_training_matrix_run": False,
                    "heldout_inputs_read": False,
                }

        run_root = (
            output_root / "full_universe_training_runs"
            if iteration == 0 else iteration_root / "training_runs"
        )
        runs = run_candidate_matrix(
            executable=executable,
            launcher=launcher,
            configs={day: config.path for day, config in configs.items()},
            background_policies={day: expanded_background for day in configs},
            value_policy=expanded_value,
            local=local,
            duration=STAGE3_DURATION,
            seeds=protocol.stage3_seeds,
            output_root=run_root,
            timeout_seconds=protocol.timeout_seconds["stage3"],
            run_workers=args.run_workers,
        )
        all_iteration_runs.append(runs)
        if any(
            record["success"] is not True
            for day_runs in runs.values() for record in day_runs
        ):
            write_json(output_root / "full_universe_training_adequacy.json", {
                **expansion_inputs,
                "status": "simulation_failed_no_expanded_freeze",
                "passed": False,
                "failed_iteration": iteration,
                "runs": runs,
            })
            raise CalibrationDriverError(
                "at least one full-universe 2019 training run failed; no "
                "expanded freeze was written"
            )

        evaluation_directory = (
            "strict_full_training_evaluation"
            if gate_protocol == strict.STRICT_NINE_GATE
            else "six_component_training_evaluation"
        )
        strict_output = (
            output_root / evaluation_directory
            if iteration == 0 else iteration_root / evaluation_directory
        )
        strict_arguments = [
            "--evaluation-role", "training_fit",
            "--gate-protocol", gate_protocol,
            "--cluster-map", str(expanded_clusters),
            "--expected-cluster-count", str(len(cluster_ids)),
            "--output-dir", str(strict_output),
        ]
        for day in sorted(configs):
            strict_arguments.extend(("--expected-date", day))
            strict_arguments.extend(
                ("--target-config", f"{day}={configs[day].path}")
            )
            for record in runs[day]:
                command = record["command"]
                seed = command[command.index("--seed") + 1]
                strict_arguments.extend((
                    "--sim-summary", f"{day}:{seed}={record['summary_path']}",
                ))
        strict_record = iteration_root / (
            "strict_full_training_arguments.json"
            if gate_protocol == strict.STRICT_NINE_GATE
            else "six_component_training_arguments.json"
        )
        write_json(strict_record, {
            "script": str(SCRIPT_DIR / "evaluate_strict_model_validation.py"),
            "arguments": strict_arguments,
            "heldout_inputs_read": False,
            "refinement_iteration": iteration,
            "gate_protocol": gate_protocol,
        })
        evidence_paths.append(strict_record)
        strict_report, strict_report_path = run_strict_evaluation(
            strict_arguments
        )
        evidence_paths.append(strict_report_path)
        strict_passed = strict_report.get("passed") is True
        iteration_runtime.append({
            "iteration": iteration,
            "deployment_config": {
                "path": str(expanded_config),
                "sha256": sha256_file(expanded_config),
            },
            "strict_report": {
                "path": str(strict_report_path),
                "sha256": sha256_file(strict_report_path),
                "passed": strict_passed,
            },
            "volatility_by_cluster": {
                cluster: candidate.__dict__
                for cluster, candidate in sorted(current_volatility.items())
            },
        })
        if strict_passed and iteration >= args.minimum_refinement_iterations:
            break
        if iteration == args.max_refinement_iterations:
            break
        residual_path = strict_output / "symbol_residuals.csv"
        try:
            current_volatility, refinement = refine_full_universe_volatility(
                symbol_residuals=residual_path,
                current=current_volatility,
                cluster_ids=cluster_ids,
                iteration=iteration + 1,
            )
        except CalibrationDriverError as error:
            expansion_inputs["target_preflight"] = target_preflight
            expansion_inputs["refinement_iterations"] = iteration_runtime
            write_json(inputs_manifest_path, expansion_inputs)
            write_json(output_root / "full_universe_training_adequacy.json", {
                **expansion_inputs,
                "status": "strict_training_gate_failed_no_expanded_freeze",
                "passed": False,
                "strict_report": {
                    "path": str(strict_report_path),
                    "sha256": sha256_file(strict_report_path),
                },
                "runs": runs,
                "refinement_error": str(error),
            })
            raise CalibrationDriverError(
                f"{error}; no expanded freeze was written and held-out "
                "execution remains unauthorized"
            ) from error
        refinement_path = (
            output_root / "full_universe_refinement"
            / f"iteration_{iteration + 1}" / "moment_update.json"
        )
        write_json(refinement_path, refinement)
        evidence_paths.append(refinement_path)
        refinement_records.append({
            "path": str(refinement_path),
            "sha256": sha256_file(refinement_path),
            **refinement,
        })

    if target_preflight is None:
        raise CalibrationDriverError("full-universe target preflight was skipped")
    expansion_inputs["target_preflight"] = target_preflight
    expansion_inputs["selection_parameters_changed"] = bool(refinement_records)
    expansion_inputs["completed_refinement_updates"] = len(refinement_records)
    expansion_inputs["refinement_iterations"] = iteration_runtime
    write_json(inputs_manifest_path, expansion_inputs)

    adequacy_path = output_root / "full_universe_training_adequacy.json"
    write_json(adequacy_path, {
        **expansion_inputs,
        "status": (
            "full_universe_training_adequacy_passed"
            if strict_passed else
            "strict_training_gate_failed_no_expanded_freeze"
        ),
        "passed": strict_passed,
        "strict_report": {
            "path": str(strict_report_path),
            "sha256": sha256_file(strict_report_path),
        },
        "refinement_updates": refinement_records,
        "iteration_runtime": iteration_runtime,
        "runs": runs,
    })
    if not strict_passed:
        raise CalibrationDriverError(
            "full-universe 2019 strict adequacy failed; no expanded freeze was "
            "written and held-out execution remains unauthorized"
        )

    command_artifacts = command_artifact_paths(runs)
    for iteration_runs in all_iteration_runs[:-1]:
        previous_artifacts = command_artifact_paths(iteration_runs)
        for role, paths in previous_artifacts.items():
            command_artifacts[role].update(paths)
    command_artifacts["configs"].add(expanded_config)
    command_artifacts["background_policies"].add(expanded_background)
    command_artifacts["value_policies"].add(expanded_value)
    command_artifacts["executables"].add(executable)
    if rank_equivalence_path is not None:
        rank_payload = json.loads(
            rank_equivalence_path.read_text(encoding="utf-8")
        )
        for key in ("reference_run", "production_run"):
            run_payload = rank_payload.get(key)
            if not isinstance(run_payload, Mapping):
                raise CalibrationDriverError(
                    "rank-equivalence evidence lacks a run record"
                )
            command_artifacts["summaries"].add(
                pathlib.Path(str(run_payload["summary_path"])).resolve()
            )
    protocol_revision_paths: tuple[pathlib.Path, ...] = ()
    if protocol_revision_certificate is not None:
        source_records = protocol_revision_certificate[
            "verified_source_evidence"
        ]
        residual_record = protocol_revision_certificate["symbol_residuals"]
        if not isinstance(source_records, Mapping) \
                or not isinstance(residual_record, Mapping):
            raise CalibrationDriverError(
                "verified protocol revision evidence is malformed"
            )
        protocol_revision_paths = (
            pathlib.Path(str(protocol_revision_certificate["path"])),
            *(pathlib.Path(str(record["path"]))
              for record in source_records.values()
              if isinstance(record, Mapping)),
            pathlib.Path(str(residual_record["path"])),
        )
    transitive_manifest = transitive_runtime_artifacts(
        configs=tuple(command_artifacts["configs"]),
        background_policies=tuple(command_artifacts["background_policies"]),
        value_policies=tuple(command_artifacts["value_policies"]),
        executables=tuple(command_artifacts["executables"]),
        summaries=tuple(command_artifacts["summaries"]),
        selection_records=(
            adequacy_path,
            strict_report_path,
            strict_record,
            *evidence_paths,
            *protocol_revision_paths,
            *((rank_equivalence_path,) if rank_equivalence_path else ()),
        ),
        workflow_sources=(
            pathlib.Path(__file__),
            pathlib.Path(legacy.__file__),
            pathlib.Path(legacy.cohort.__file__),
            pathlib.Path(strict.__file__),
        ),
        candidate_config=candidate_path,
        cluster_map=expanded_clusters,
    )
    expanded_selection = dict(selection)
    expanded_selection["volatility_by_cluster"] = {
        cluster: candidate.__dict__
        for cluster, candidate in sorted(current_volatility.items())
    }
    expanded_selection["full_universe_refinement"] = {
        "performed": bool(refinement_records),
        "completed_updates": len(refinement_records),
        "method": "bounded_cluster_simulated_moment_update_v1",
        "training_only": True,
        "heldout_inputs_read": False,
        "adequacy_protocol_unchanged_within_continuation": True,
        "training_adequacy_protocol": gate_protocol,
        "evidence": refinement_records,
    }
    expanded_freeze = {
        "schema_version": SCHEMA_VERSION,
        "status": "expanded_training_adequacy_frozen",
        "training_only": True,
        "heldout_inputs_read": False,
        "frozen_before_any_heldout_run": True,
        "development_validation_targets_opened": False,
        "allowed_heldout_role": "development_validation",
        "heldout_execution_authorized": True,
        "training_adequacy_protocol": gate_protocol,
        "protocol_revision_certificate": protocol_revision_certificate,
        "training_dates": sorted(configs),
        "full_universe_training_adequacy_passed": True,
        "ordinary_market_shared_mm_disabled": True,
        "one_rank_execution": production_ranks == 1,
        "execution": expansion_inputs["execution"],
        "estimation_symbol_count": fit_count,
        "deployment_symbol_count": full_count,
        "source_training_freeze": {
            "path": str(freeze_path), "sha256": sha256_file(freeze_path),
        },
        "training_inputs_manifest": {
            "path": str(inputs_manifest_path),
            "sha256": sha256_file(inputs_manifest_path),
        },
        "selection": expanded_selection,
        "selection_parameters_changed_during_2019_training": bool(
            refinement_records
        ),
        "frozen_artifacts": {
            "deployment_config": {
                "path": str(expanded_config),
                "sha256": sha256_file(expanded_config),
            },
            "value_policy": {
                "path": str(expanded_value),
                "sha256": sha256_file(expanded_value),
            },
            "background_policy_mapping": {
                "path": str(expanded_background),
                "sha256": sha256_file(expanded_background),
            },
            "cluster_map": {
                "path": str(expanded_clusters),
                "sha256": sha256_file(expanded_clusters),
            },
            "candidate_config": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "executable": {
                "path": str(executable),
                "sha256": sha256_file(executable),
            },
        },
        "stage3_runs": runs,
        "training_adequacy_gate_passed": True,
        "strict_training_gate_passed": (
            gate_protocol == strict.STRICT_NINE_GATE
        ),
        "marketwide_six_component_training_gate_passed": (
            gate_protocol == strict.MARKETWIDE_SIX_GATE
        ),
        "strict_training_report": {
            "path": str(strict_report_path),
            "sha256": sha256_file(strict_report_path),
        },
        "full_universe_training_adequacy": {
            "path": str(adequacy_path),
            "sha256": sha256_file(adequacy_path),
        },
        "rank_equivalence": (
            {"path": str(rank_equivalence_path),
             "sha256": sha256_file(rank_equivalence_path)}
            if rank_equivalence_path is not None else None
        ),
        "transitive_runtime_artifacts": transitive_manifest,
        "validation_claimed": False,
        "certification_claimed": False,
    }
    expanded_freeze_path = output_root / "expanded_training_freeze.json"
    write_json(expanded_freeze_path, expanded_freeze)
    return {
        "status": expanded_freeze["status"],
        "expanded_training_freeze": str(expanded_freeze_path),
        "expanded_training_freeze_sha256": sha256_file(
            expanded_freeze_path
        ),
        "validation_claimed": False,
        "certification_claimed": False,
    }


def heldout(args: argparse.Namespace) -> dict[str, object]:
    output_root = args.output_root.expanduser().resolve()
    prepare_output(output_root, resume=args.resume)
    freeze_path = args.freeze_record.expanduser().resolve()
    freeze = verified_training_freeze(
        freeze_path,
        allowed_statuses=("expanded_training_adequacy_frozen",),
    )
    if (
        freeze.get("full_universe_training_adequacy_passed") is not True
        or freeze.get("heldout_execution_authorized") is not True
        or freeze.get("allowed_heldout_role") != "development_validation"
    ):
        raise CalibrationDriverError(
            "only a passed full-universe 2019 adequacy freeze authorizes "
            "development validation"
        )
    gate_protocol = str(
        freeze.get("training_adequacy_protocol", strict.STRICT_NINE_GATE)
    )
    if gate_protocol not in strict.GATE_PROTOCOLS:
        raise CalibrationDriverError(
            f"frozen training adequacy protocol is unknown: {gate_protocol}"
        )
    if gate_protocol == strict.MARKETWIDE_SIX_GATE:
        certificate = freeze.get("protocol_revision_certificate")
        if not isinstance(certificate, Mapping):
            raise CalibrationDriverError(
                "six-component training freeze lacks its protocol revision "
                "certificate"
            )
        verified_artifact(certificate, "six-component protocol certificate")
    if args.heldout_role != "development_validation":
        raise CalibrationDriverError(
            "the 2020 date is development validation, not an untouched final "
            "holdout"
        )
    artifacts = freeze.get("frozen_artifacts")
    if not isinstance(artifacts, Mapping):
        raise CalibrationDriverError("freeze record lacks frozen_artifacts")
    frozen_config_path = verified_artifact(
        artifacts["deployment_config"], "deployment config"  # type: ignore[arg-type]
    )
    value_policy = verified_artifact(
        artifacts["value_policy"], "value policy"  # type: ignore[arg-type]
    )
    background_policy = verified_artifact(
        artifacts["background_policy_mapping"], "background policy"  # type: ignore[arg-type]
    )
    cluster_path = verified_artifact(
        artifacts["cluster_map"], "cluster map"  # type: ignore[arg-type]
    )
    executable = verified_artifact(
        artifacts["executable"], "executable"  # type: ignore[arg-type]
    )
    candidate_path = verified_artifact(
        artifacts["candidate_config"], "candidate config"  # type: ignore[arg-type]
    )
    protocol = load_candidate_protocol(candidate_path)
    local_payload = freeze["selection"]["local_candidate"]
    local = LocalCandidate(**local_payload)
    clusters = read_cluster_map(cluster_path)
    frozen_config = read_config(frozen_config_path)

    heldout_day = normalized_date(args.heldout_date, label="--heldout-date")
    if heldout_day in set(freeze.get("training_dates", [])):
        raise CalibrationDriverError("held-out date is one of the frozen training dates")
    opening_config_path = args.heldout_opening_config.expanduser().resolve()
    target_config_path = args.heldout_target_config.expanduser().resolve()
    if not opening_config_path.is_file() or not target_config_path.is_file():
        raise CalibrationDriverError("held-out opening/target config is missing")
    opening_config = read_config(opening_config_path)
    target_config = read_config(target_config_path)
    heldout_config_path = output_root / "heldout_simulation_config.csv"
    prepared_heldout = prepare_heldout_config(
        heldout_config_path,
        frozen_base=frozen_config,
        opening_observations=opening_config,
        target_observations=target_config,
    )

    launcher = parse_launcher(args.launcher)
    production_ranks = launcher_rank_count(launcher)
    if args.run_workers <= 0:
        raise CalibrationDriverError("--run-workers must be positive")
    execution = freeze.get("execution")
    if isinstance(execution, Mapping):
        frozen_ranks = exact_integer(
            execution.get("mpi_ranks_per_run"),
            label="frozen mpi_ranks_per_run", minimum=1,
        )
    else:
        frozen_ranks = 1 if freeze.get("one_rank_execution") is True else 0
    if production_ranks != frozen_ranks:
        raise CalibrationDriverError(
            f"held-out launcher requests {production_ranks} ranks but the "
            f"passed training freeze authorizes {frozen_ranks}"
        )
    rank_equivalence_record: dict[str, object] | None = None
    if production_ranks > 1:
        rank_record = freeze.get("rank_equivalence")
        if not isinstance(rank_record, Mapping):
            raise CalibrationDriverError(
                "multi-rank held-out execution lacks frozen rank-equivalence evidence"
            )
        rank_path = verified_artifact(
            rank_record, "rank-equivalence evidence"
        )
        try:
            rank_equivalence_record = json.loads(
                rank_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise CalibrationDriverError(
                f"cannot read rank-equivalence evidence: {error}"
            ) from error
        if (
            rank_equivalence_record.get("status") != "rank_equivalence_passed"
            or rank_equivalence_record.get("reference_ranks") != 1
            or rank_equivalence_record.get("production_ranks") != production_ranks
            or rank_equivalence_record.get("summary_bytes_equal") is not True
            or rank_equivalence_record.get("terminal_state_hash_equal") is not True
            or rank_equivalence_record.get("executable", {}).get("sha256")
                != sha256_file(executable)
        ):
            raise CalibrationDriverError(
                "frozen rank-equivalence evidence does not authorize this launcher"
            )
    if sha256_file(executable) != artifacts["executable"]["sha256"]:  # type: ignore[index]
        raise CalibrationDriverError("executable changed after freeze verification")
    seeds = tuple(
        exact_integer(value, label="--heldout-seed", minimum=0)
        for value in args.heldout_seed
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise CalibrationDriverError("held-out seeds must be non-empty and unique")
    runs = run_candidate_matrix(
        executable=executable,
        launcher=launcher,
        configs={heldout_day: prepared_heldout.path},
        background_policies={heldout_day: background_policy},
        value_policy=value_policy,
        local=local,
        duration=STAGE3_DURATION,
        seeds=seeds,
        output_root=output_root / "runs",
        timeout_seconds=protocol.timeout_seconds["heldout"],
        run_workers=args.run_workers,
    )
    if any(record["success"] is not True for record in runs[heldout_day]):
        raise CalibrationDriverError("at least one held-out simulation failed")

    strict_arguments = [
        "--evaluation-role", args.heldout_role,
        "--gate-protocol", gate_protocol,
        "--expected-date", heldout_day,
        "--target-config", f"{heldout_day}={prepared_heldout.path}",
        "--cluster-map", str(cluster_path),
        "--expected-cluster-count", str(len(set(clusters.values()))),
        "--output-dir", str(output_root / "strict_evaluation"),
    ]
    for record in runs[heldout_day]:
        command = record["command"]
        seed = command[command.index("--seed") + 1]
        strict_arguments.extend((
            "--sim-summary", f"{heldout_day}:{seed}={record['summary_path']}"
        ))
    if args.heldout_role == "untouched_final_holdout":
        strict_arguments.extend(("--protocol-freeze-record", str(freeze_path)))
    strict_record = output_root / "strict_evaluator_arguments.json"
    write_json(strict_record, {
        "script": str(SCRIPT_DIR / "evaluate_strict_model_validation.py"),
        "arguments": strict_arguments,
        "automatic_execution_required": True,
    })
    strict_report, strict_report_path = run_strict_evaluation(strict_arguments)
    strict_passed = strict_report.get("passed") is True
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "heldout_adequacy_passed"
            if strict_passed else "heldout_adequacy_failed"
        ),
        "evaluation_role": args.heldout_role,
        "gate_protocol": gate_protocol,
        "heldout_date": heldout_day,
        "training_freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
        "heldout_opening_config": {"path": str(opening_config_path), "sha256": sha256_file(opening_config_path)},
        "heldout_target_config": {"path": str(target_config_path), "sha256": sha256_file(target_config_path)},
        "simulation_config": {"path": str(heldout_config_path), "sha256": sha256_file(heldout_config_path)},
        "opening_fields_copied": list(OPENING_FIELDS),
        "target_data_dir_copied_from_heldout_target_config": True,
        "all_other_simulation_fields_frozen": True,
        "execution": {
            "mpi_ranks_per_run": production_ranks,
            "maximum_concurrent_runs": args.run_workers,
            "parallelism": "whole_book_mpi",
            "rank_equivalence_verified": production_ranks == 1
                or rank_equivalence_record is not None,
        },
        "runs": runs[heldout_day],
        "strict_evaluator_arguments": str(strict_record),
        "strict_report": {
            "path": str(strict_report_path),
            "sha256": sha256_file(strict_report_path),
            "passed": strict_passed,
        },
        "validation_claimed": strict_passed,
        "certification_claimed": False,
    }
    manifest_path = output_root / "heldout_run_manifest.json"
    write_json(manifest_path, manifest)
    if not strict_passed:
        raise CalibrationDriverError(
            "held-out strict adequacy gate failed; diagnostics were preserved"
        )
    return {
        "status": manifest["status"],
        "manifest": str(manifest_path),
        "strict_evaluator_arguments": str(strict_record),
        "strict_report": str(strict_report_path),
        "validation_claimed": True,
        "certification_claimed": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="subcommand", required=True)
    training = subparsers.add_parser("train", help="run three training-only stages")
    training.add_argument("--executable", type=pathlib.Path, required=True)
    training.add_argument(
        "--launcher", default="",
        help="shell-quoted one-rank prefix for each independent selection run",
    )
    training.add_argument(
        "--run-workers", type=int, default=1,
        help="maximum concurrent independent selection runs",
    )
    training.add_argument(
        "--training-config", action="append", required=True, metavar="DATE=PATH"
    )
    training.add_argument(
        "--background-policy", type=pathlib.Path, required=True,
        help=(
            "one pooled training-only symbol_policy_mapping.csv, reused "
            "unchanged across all five dates"
        ),
    )
    training.add_argument("--deployment-config", type=pathlib.Path, required=True)
    training.add_argument("--cluster-map", type=pathlib.Path, required=True)
    training.add_argument("--candidate-config", type=pathlib.Path, required=True)
    training.add_argument("--output-root", type=pathlib.Path, required=True)

    expansion = subparsers.add_parser(
        "expand-full-universe",
        help=(
            "apply a frozen stratified-panel selection unchanged to the "
            "complete 2019 training universe"
        ),
    )
    expansion.add_argument("--freeze-record", type=pathlib.Path, required=True)
    expansion.add_argument(
        "--training-config", action="append", required=True,
        metavar="DATE=PATH",
    )
    expansion.add_argument(
        "--full-deployment-config", type=pathlib.Path, required=True,
    )
    expansion.add_argument(
        "--executable", type=pathlib.Path,
        help=(
            "audited replacement runtime; omitted uses the executable hashed "
            "by the phase-1 selection freeze"
        ),
    )
    expansion.add_argument(
        "--full-background-policy", type=pathlib.Path, required=True,
    )
    expansion.add_argument("--full-cluster-map", type=pathlib.Path, required=True)
    expansion.add_argument(
        "--launcher", default="",
        help="explicit production launcher, including its MPI rank count",
    )
    expansion.add_argument(
        "--reference-launcher", default="",
        help="explicit one-rank launcher used by the rank-equivalence preflight",
    )
    expansion.add_argument(
        "--rank-equivalence-duration", type=int, default=300,
        help="seconds simulated by both one-rank and production-rank preflight runs",
    )
    expansion.add_argument(
        "--run-workers", type=int, default=1,
        help="maximum concurrent full-universe MPI realisations",
    )
    expansion.add_argument(
        "--max-refinement-iterations", type=int, default=2,
        help=(
            "maximum bounded 2019-only cluster moment updates after an "
            "unsuccessful full-universe gate (default: 2)"
        ),
    )
    expansion.add_argument(
        "--minimum-refinement-iterations", type=int, default=0,
        help=(
            "minimum previously selected 2019-only update to reproduce before "
            "acceptance; used by hash-verified continuation (default: 0)"
        ),
    )
    expansion.add_argument(
        "--gate-protocol", choices=strict.GATE_PROTOCOLS,
        default=strict.STRICT_NINE_GATE,
        help="versioned full-universe training adequacy protocol",
    )
    expansion.add_argument(
        "--six-component-protocol-certificate", type=pathlib.Path,
        help=(
            "required with marketwide-six-v2; binds the visible original "
            "strict-nine failure and retrospective revision"
        ),
    )
    expansion.add_argument(
        "--initial-refinement-manifest", type=pathlib.Path,
        help=(
            "hash-bound training-only warm start; this never bypasses the "
            "full strict adequacy gate"
        ),
    )
    expansion.add_argument(
        "--initial-coupling-scale", type=float, default=1.0,
        help=(
            "single training-only scale applied to every nonzero activity "
            "loading in the hash-bound warm start (default: 1; bounded to "
            "[0.5,1.25])"
        ),
    )
    expansion.add_argument(
        "--resume", action="store_true",
        help="reuse only hash-verified successful run checkpoints",
    )
    expansion.add_argument(
        "--directional-pilot", action="store_true",
        help=(
            "run a three-date, one-seed full-universe mechanism check before "
            "the 25-run strict matrix"
        ),
    )
    expansion.add_argument(
        "--directional-pilot-only", action="store_true",
        help=(
            "stop after a passed directional pilot; write a training-only "
            "handoff and do not launch the 25-run strict matrix"
        ),
    )
    expansion.add_argument("--output-root", type=pathlib.Path, required=True)

    validation = subparsers.add_parser(
        "heldout",
        help=(
            "run development validation after a passed full-universe "
            "training-adequacy freeze exists"
        ),
    )
    validation.add_argument("--freeze-record", type=pathlib.Path, required=True)
    validation.add_argument("--heldout-date", required=True)
    validation.add_argument(
        "--heldout-opening-config", type=pathlib.Path, required=True
    )
    validation.add_argument(
        "--heldout-target-config", type=pathlib.Path, required=True
    )
    validation.add_argument(
        "--heldout-seed", action="append", required=True, metavar="INTEGER"
    )
    validation.add_argument(
        "--heldout-role",
        choices=("development_validation", "untouched_final_holdout"),
        default="development_validation",
    )
    validation.add_argument(
        "--launcher", default="",
        help="production launcher frozen by full-universe training adequacy",
    )
    validation.add_argument(
        "--run-workers", type=int, default=1,
        help="maximum concurrent held-out MPI realisations",
    )
    validation.add_argument(
        "--resume", action="store_true",
        help="reuse only hash-verified successful run checkpoints",
    )
    validation.add_argument("--output-root", type=pathlib.Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.subcommand == "train":
            report = train(args)
        elif args.subcommand == "expand-full-universe":
            report = expand_full_universe(args)
        else:
            report = heldout(args)
    except (
        CalibrationDriverError,
        OSError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as error:
        print(f"queue-reactive calibration driver failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
