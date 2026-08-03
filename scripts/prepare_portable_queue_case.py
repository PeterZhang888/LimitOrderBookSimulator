#!/usr/bin/env python3
"""Create a verified runtime bundle for the final queue-reactive case study.

The calibration and validation manifests contain hash-bound absolute paths from
their original cluster jobs. This utility verifies those handoffs, reconstructs
the held-out runtime configuration without using held-out target statistics,
rebinds external paths to the supplied pool and policy roots, and writes the
four files consumed by the MPI case-study launcher.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import shutil
import sys
from collections.abc import Iterable, Mapping, Sequence

import certification_cohort as cohort


class PreparationError(RuntimeError):
    """The requested case bundle is incomplete or not hash-consistent."""


RUNTIME_FIELDS = (
    "book_id",
    "symbol",
    "data_dir",
    "hawkes_rates_file",
    "fundamental_price_ticks",
    "fundamental_volatility_bps_sqrt_second",
    "fundamental_move_probability_per_second",
    "fundamental_conditional_kurtosis",
    "initial_best_bid_ticks",
    "initial_best_ask_ticks",
    "initial_best_bid_depth",
    "initial_best_ask_depth",
    "beta",
    "basket_weight",
    "market_maker_quote_quantity",
    "target_spread_ticks",
    "quote_improvement_probability",
    "target_mean_bid_depth",
    "target_mean_ask_depth",
    "fundamental_log_variance_persistence",
    "fundamental_log_variance_std",
    "fundamental_order_flow_coupling",
)

HELDOUT_OPENING_FIELDS = (
    "fundamental_price_ticks",
    "initial_best_bid_ticks",
    "initial_best_ask_ticks",
    "initial_best_bid_depth",
    "initial_best_ask_depth",
)

RUNTIME_SCHEMA = {
    "schema_version": 6,
    "fields": list(RUNTIME_FIELDS),
    "sha256": hashlib.sha256(json.dumps(
        list(RUNTIME_FIELDS), ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest(),
    "pooled_homeostatic_fields": [
        "target_spread_ticks", "target_mean_bid_depth", "target_mean_ask_depth",
    ],
    "latent_value_fields": [
        "fundamental_volatility_bps_sqrt_second",
        "fundamental_move_probability_per_second",
        "fundamental_conditional_kurtosis",
        "fundamental_log_variance_persistence",
        "fundamental_log_variance_std",
        "fundamental_order_flow_coupling",
    ],
    "frozen_training_derived_fields": [
        "target_spread_ticks", "target_mean_bid_depth", "target_mean_ask_depth",
        "fundamental_volatility_bps_sqrt_second",
        "fundamental_move_probability_per_second",
        "fundamental_conditional_kurtosis",
        "fundamental_log_variance_persistence",
        "fundamental_log_variance_std",
        "fundamental_order_flow_coupling",
    ],
    "heldout_target_files_used": False,
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: pathlib.Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise PreparationError(f"{label} must be a JSON object: {path}")
    return value


def read_csv(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except OSError as error:
        raise PreparationError(f"cannot read CSV {path}: {error}") from error
    if not fields or not rows:
        raise PreparationError(f"CSV is empty: {path}")
    return fields, rows


def write_csv(
    path: pathlib.Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(fields), extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def required_path(path: pathlib.Path, *, directory: bool, label: str) -> pathlib.Path:
    path = path.expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise PreparationError(f"required {label} {kind} is missing: {path}")
    return path


def canonical_symbol(value: str, *, label: str) -> str:
    symbol = value.strip().upper()
    if not symbol or any(character.isspace() for character in symbol):
        raise PreparationError(f"invalid symbol in {label}: {value!r}")
    return symbol


def rows_by_symbol(
    path: pathlib.Path,
    *,
    required_fields: set[str],
) -> tuple[list[str], list[str], dict[str, dict[str, str]]]:
    fields, rows = read_csv(path)
    missing = sorted(required_fields.difference(fields))
    if missing:
        raise PreparationError(f"{path} lacks columns: {', '.join(missing)}")
    order: list[str] = []
    result: dict[str, dict[str, str]] = {}
    for index, source in enumerate(rows):
        symbol = canonical_symbol(source.get("symbol", ""), label=str(path))
        if symbol in result:
            raise PreparationError(f"duplicate symbol in {path}: {symbol}")
        if "book_id" in fields and int(source["book_id"]) != index:
            raise PreparationError(f"book_id is not contiguous in {path}")
        row = dict(source)
        row["symbol"] = symbol
        order.append(symbol)
        result[symbol] = row
    return fields, order, result


def digest_from_record(value: object, *, label: str) -> str:
    if not isinstance(value, Mapping):
        raise PreparationError(f"{label} must be a path/hash object")
    digest = value.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise PreparationError(f"{label} lacks a SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as error:
        raise PreparationError(f"{label} digest is not hexadecimal") from error
    return digest.lower()


def locate_record(
    value: object,
    *,
    label: str,
    preferred: Sequence[pathlib.Path],
    search_roots: Sequence[pathlib.Path],
) -> pathlib.Path:
    expected = digest_from_record(value, label=label)
    assert isinstance(value, Mapping)
    recorded = pathlib.Path(str(value.get("path", ""))).expanduser()
    candidates: list[pathlib.Path] = list(preferred)
    if recorded.is_absolute():
        candidates.append(recorded)
    basename = recorded.name
    if basename:
        for root in search_roots:
            candidates.extend(root.rglob(basename))
    observed: set[pathlib.Path] = set()
    for raw in candidates:
        path = raw.expanduser()
        if not path.is_absolute():
            path = pathlib.Path.cwd() / path
        try:
            path = path.resolve()
        except OSError:
            continue
        if path in observed or not path.is_file():
            continue
        observed.add(path)
        if sha256_file(path) == expected:
            return path
    raise PreparationError(
        f"cannot locate hash-matching {label}; expected {expected}"
    )


def verify_handoffs(
    evidence_root: pathlib.Path,
    selection_root: pathlib.Path,
    pool_root: pathlib.Path,
    data_root: pathlib.Path,
) -> dict[str, object]:
    heldout_path = required_path(
        evidence_root / "development_validation/heldout_run_manifest.json",
        directory=False,
        label="development-validation manifest",
    )
    freeze_path = required_path(
        evidence_root / "training/expanded_training_freeze.json",
        directory=False,
        label="expanded training freeze",
    )
    augmentation_path = required_path(
        evidence_root / "provenance/queue_reactive_augmentation_provenance.json",
        directory=False,
        label="augmentation provenance",
    )
    heldout = read_json(heldout_path, label="development-validation manifest")
    freeze = read_json(freeze_path, label="expanded training freeze")
    augmentation = read_json(augmentation_path, label="augmentation provenance")
    if (
        heldout.get("schema_version") != 1
        or heldout.get("status") != "heldout_adequacy_passed"
        or heldout.get("validation_claimed") is not True
        or heldout.get("evaluation_role") != "development_validation"
    ):
        raise PreparationError("development-validation handoff is not a PASS")
    if (
        freeze.get("schema_version") != 1
        or freeze.get("status") != "expanded_training_adequacy_frozen"
        or freeze.get("training_only") is not True
        or freeze.get("full_universe_training_adequacy_passed") is not True
        or freeze.get("heldout_execution_authorized") is not True
    ):
        raise PreparationError("expanded training freeze is not authorized")
    expected_freeze = digest_from_record(
        heldout.get("training_freeze"), label="held-out training freeze",
    )
    if sha256_file(freeze_path) != expected_freeze:
        raise PreparationError("held-out manifest and training freeze disagree")
    strict_report = locate_record(
        heldout.get("strict_report"),
        label="strict validation report",
        preferred=(evidence_root / "development_validation/strict_validation_report.json",),
        search_roots=(evidence_root,),
    )
    strict_payload = read_json(strict_report, label="strict validation report")
    if strict_payload.get("passed") is not True:
        raise PreparationError("strict validation report is not a PASS")
    heldout_simulation = locate_record(
        heldout.get("simulation_config"),
        label="held-out simulation configuration",
        preferred=(evidence_root / "development_validation/heldout_simulation_config.csv",),
        search_roots=(evidence_root, pool_root),
    )
    if heldout.get("all_other_simulation_fields_frozen") is not True:
        raise PreparationError("held-out manifest does not freeze non-opening fields")
    if not augmentation:
        raise PreparationError("augmentation provenance is empty")

    pooling_path = required_path(
        pool_root / "pooling_provenance.json",
        directory=False,
        label="pooling provenance",
    )
    selection_path = required_path(
        selection_root / "selection/training_selection_freeze.json",
        directory=False,
        label="selection freeze",
    )
    read_json(pooling_path, label="pooling provenance")
    read_json(selection_path, label="selection freeze")
    required_path(data_root, directory=True, label="empirical data root")
    return {
        "heldout_manifest": heldout,
        "heldout_manifest_path": heldout_path,
        "expanded_freeze": freeze,
        "expanded_freeze_path": freeze_path,
        "augmentation_path": augmentation_path,
        "strict_report_path": strict_report,
        "heldout_simulation_path": heldout_simulation,
        "pooling_path": pooling_path,
        "selection_path": selection_path,
    }


def rebind_pool_paths(row: dict[str, str], pool_root: pathlib.Path) -> None:
    raw_dir = pathlib.Path(row["data_dir"])
    candidates = (
        raw_dir,
        pool_root / "pooled_data" / raw_dir.name,
    )
    data_dir = next((path.resolve() for path in candidates if path.is_dir()), None)
    if data_dir is None:
        raise PreparationError(
            f"cannot rebind pooled data directory for {row['symbol']}: {raw_dir.name}"
        )
    rate_name = pathlib.Path(row["hawkes_rates_file"]).name
    rate_candidates = (
        pathlib.Path(row["hawkes_rates_file"]),
        data_dir / rate_name,
    )
    rate_path = next((path.resolve() for path in rate_candidates if path.is_file()), None)
    if rate_path is None:
        raise PreparationError(
            f"cannot rebind Hawkes rates for {row['symbol']}: {rate_name}"
        )
    row["data_dir"] = str(data_dir)
    row["hawkes_rates_file"] = str(rate_path)


def build_case_config(
    heldout_path: pathlib.Path,
    deployment_path: pathlib.Path,
    pool_root: pathlib.Path,
    output_path: pathlib.Path,
) -> list[str]:
    # The validation handoff is hash-bound but has evolved across releases.
    # Only identifiers and opening state are allowed to enter the case from
    # that file; the canonical runtime policy comes from frozen training.
    heldout_required = {"book_id", "symbol", *HELDOUT_OPENING_FIELDS}
    deployment_required = set(RUNTIME_FIELDS)
    heldout_fields, heldout_order, heldout = rows_by_symbol(
        heldout_path, required_fields=heldout_required,
    )
    deployment_fields, deployment_order, deployment = rows_by_symbol(
        deployment_path, required_fields=deployment_required,
    )
    if heldout_order != deployment_order:
        raise PreparationError("held-out and deployment symbol order differs")
    rows: list[dict[str, object]] = []
    # Compare every frozen field actually present in the held-out handoff.  A
    # legacy handoff omits the later latent-regime fields, so those values come
    # only from the already-frozen training deployment.
    frozen_fields = (
        set(heldout_fields).intersection(RUNTIME_FIELDS)
        .difference(HELDOUT_OPENING_FIELDS)
    )
    frozen_fields.difference_update({"data_dir", "hawkes_rates_file"})
    for book_id, symbol in enumerate(heldout_order):
        for field in frozen_fields:
            if heldout[symbol][field] != deployment[symbol][field]:
                raise PreparationError(
                    f"held-out configuration changes frozen field {field} for {symbol}"
                )
        row = dict(deployment[symbol])
        for field in HELDOUT_OPENING_FIELDS:
            row[field] = heldout[symbol][field]
        row["book_id"] = str(book_id)
        rebind_pool_paths(row, pool_root)
        rows.append(row)
    write_csv(output_path, RUNTIME_FIELDS, rows)
    return heldout_order


def build_background_mapping(
    source_path: pathlib.Path,
    policy_root: pathlib.Path,
    output_path: pathlib.Path,
) -> tuple[set[str], set[pathlib.Path]]:
    fields, rows = read_csv(source_path)
    required = {
        "symbol", "cluster_id", "policy_file",
        "limit_buy_improvement_file", "limit_sell_improvement_file",
    }
    missing = sorted(required.difference(fields))
    if missing:
        raise PreparationError(
            f"background policy mapping lacks columns: {', '.join(missing)}"
        )
    symbols: set[str] = set()
    artifacts: set[pathlib.Path] = set()
    for row in rows:
        symbol = canonical_symbol(row["symbol"], label=str(source_path))
        if symbol in symbols:
            raise PreparationError(f"duplicate background policy symbol: {symbol}")
        symbols.add(symbol)
        cluster = int(row["cluster_id"])
        cluster_root = policy_root / "clusters" / f"cluster_{cluster}"
        for field in (
            "policy_file", "limit_buy_improvement_file",
            "limit_sell_improvement_file",
        ):
            raw = pathlib.Path(row[field])
            candidates = (raw, cluster_root / raw.name)
            rebound = next(
                (path.resolve() for path in candidates if path.is_file()), None,
            )
            if rebound is None:
                raise PreparationError(
                    f"cannot rebind {field} for cluster {cluster}: {raw.name}"
                )
            row[field] = str(rebound)
            artifacts.add(rebound)
        row["symbol"] = symbol
    write_csv(output_path, fields, rows)
    return symbols, artifacts


def verify_symbol_file(
    path: pathlib.Path,
    *,
    required_columns: set[str],
) -> tuple[list[str], list[dict[str, str]], set[str]]:
    fields, rows = read_csv(path)
    missing = sorted(required_columns.difference(fields))
    if missing:
        raise PreparationError(f"{path} lacks columns: {', '.join(missing)}")
    symbols: set[str] = set()
    for row in rows:
        symbol = canonical_symbol(row["symbol"], label=str(path))
        if symbol in symbols:
            raise PreparationError(f"duplicate symbol in {path}: {symbol}")
        row["symbol"] = symbol
        symbols.add(symbol)
    return fields, rows, symbols


def artifact_manifest(paths: Iterable[pathlib.Path]) -> dict[str, object]:
    entries = [
        {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
        for path in sorted(set(paths), key=lambda value: str(value.resolve()))
    ]
    return {
        "entries": entries,
        "entry_count": len(entries),
        "manifest_sha256": sha256_json(entries),
    }


def locate_target_manifest(data_root: pathlib.Path, symbol: str) -> pathlib.Path:
    lower = symbol.lower()
    basename = f"itch_manifest_{lower}_20200130.json"
    preferred = (
        data_root / "itch_20200130" / "empirical_data"
        / f"itch_20200130_{lower}" / basename
    )
    if preferred.is_file():
        return preferred.resolve()
    matches = sorted(data_root.rglob(basename))
    if len(matches) != 1:
        raise PreparationError(
            f"expected one held-out empirical manifest for {symbol}; "
            f"observed {len(matches)}"
        )
    return matches[0].resolve()


def empirical_artifact_manifest(
    symbols: Sequence[str],
    case_path: pathlib.Path,
    data_root: pathlib.Path,
) -> dict[str, object]:
    _, order, rows = rows_by_symbol(
        case_path, required_fields={"symbol", "hawkes_rates_file"},
    )
    if list(symbols) != order:
        raise PreparationError("case configuration order changed during preparation")
    entries: list[dict[str, object]] = []
    for symbol in symbols:
        rates = required_path(
            pathlib.Path(rows[symbol]["hawkes_rates_file"]),
            directory=False,
            label=f"{symbol} pooled Hawkes rates",
        )
        manifest = locate_target_manifest(data_root, symbol)
        entries.append({
            "symbol": symbol,
            "hawkes_rates": str(rates),
            "hawkes_rates_sha256": sha256_file(rates),
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
        })
    return {
        "entries": entries,
        "entry_count": len(entries),
        "manifest_sha256": sha256_json(entries),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    project_root = required_path(args.project_root, directory=True, label="project root")
    evidence_root = required_path(args.evidence_root, directory=True, label="evidence root")
    pool_root = required_path(args.pool_root, directory=True, label="pool root")
    selection_root = required_path(
        args.selection_root, directory=True, label="selection root",
    )
    data_root = required_path(args.data_root, directory=True, label="data root")
    executable = required_path(args.executable, directory=False, label="executable")
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and (
        not output_root.is_dir() or any(output_root.iterdir())
    ):
        raise PreparationError(f"output root must be absent or empty: {output_root}")

    evidence = verify_handoffs(
        evidence_root, selection_root, pool_root, data_root,
    )
    freeze = evidence["expanded_freeze"]
    assert isinstance(freeze, dict)
    frozen = freeze.get("frozen_artifacts")
    if not isinstance(frozen, Mapping):
        raise PreparationError("expanded training freeze lacks frozen_artifacts")
    search_roots = (selection_root, evidence_root, pool_root)
    deployment_path = locate_record(
        frozen.get("deployment_config"),
        label="deployment config",
        preferred=(selection_root / "full_training_configs/deployment_config.csv",),
        search_roots=search_roots,
    )
    value_path = locate_record(
        frozen.get("value_policy"),
        label="value policy",
        preferred=(selection_root / "value_policy.csv",),
        search_roots=search_roots,
    )
    background_path = locate_record(
        frozen.get("background_policy_mapping"),
        label="background policy mapping",
        preferred=(selection_root / "queue_reactive_policy/symbol_policy_mapping.csv",),
        search_roots=search_roots,
    )
    cluster_path = locate_record(
        frozen.get("cluster_map"),
        label="cluster map",
        preferred=(selection_root / "liquidity_clusters/cluster_assignments.csv",),
        search_roots=search_roots,
    )
    validated_baseline_executable_sha256 = digest_from_record(
        frozen.get("executable"), label="frozen executable",
    )
    case_executable_sha256 = sha256_file(executable)
    post_validation_treatment_amendment = (
        case_executable_sha256 != validated_baseline_executable_sha256
    )
    allow_treatment_amendment = bool(getattr(
        args, "allow_post_validation_shared_dealer_amendment", False,
    ))
    if post_validation_treatment_amendment and not allow_treatment_amendment:
        raise PreparationError(
            "rebuilt executable differs from the frozen validated executable"
        )

    source_manifest = required_path(
        project_root / "SOURCE_MANIFEST.sha256",
        directory=False,
        label="source manifest",
    )
    heldout_path = evidence["heldout_simulation_path"]
    assert isinstance(heldout_path, pathlib.Path)
    output_root.mkdir(parents=True, exist_ok=True)
    case_path = output_root / "heldout_20200130_queue_reactive_case.csv"
    output_background = output_root / "background_policy_mapping.csv"
    output_value = output_root / "value_policy.csv"
    output_clusters = output_root / "cluster_assignments.csv"

    symbols = build_case_config(
        heldout_path, deployment_path, pool_root, case_path,
    )
    background_symbols, background_artifacts = build_background_mapping(
        background_path,
        selection_root / "queue_reactive_policy",
        output_background,
    )
    value_fields, value_rows, value_symbols = verify_symbol_file(
        value_path, required_columns={"symbol", "enabled"},
    )
    cluster_fields, cluster_rows, cluster_symbols = verify_symbol_file(
        cluster_path, required_columns={"symbol", "cluster_id"},
    )
    expected_symbols = set(symbols)
    for label, observed in (
        ("background policy", background_symbols),
        ("value policy", value_symbols),
        ("cluster map", cluster_symbols),
    ):
        if observed != expected_symbols:
            raise PreparationError(f"{label} symbol set differs from case cohort")
    cohort_identity = cohort.validate_symbols(
        symbols, label="portable case-study cohort", project_root=project_root,
    )
    shutil.copy2(value_path, output_value)
    shutil.copy2(cluster_path, output_clusters)
    if sha256_file(output_value) != sha256_file(value_path):
        raise PreparationError("value-policy copy is not byte-identical")
    if sha256_file(output_clusters) != sha256_file(cluster_path):
        raise PreparationError("cluster-map copy is not byte-identical")

    local = freeze.get("selection", {})
    if not isinstance(local, Mapping):
        raise PreparationError("training freeze lacks selected policy")
    local_candidate = local.get("local_candidate")
    if not isinstance(local_candidate, Mapping):
        raise PreparationError("training freeze lacks local-MM candidate")
    expected_local = {
        "enabled": True,
        "interval_ms": 1000.0,
        "quantity_multiplier": 1.0,
        "improvement_probability": 0.25,
        "spread_elasticity": 0.0,
        "max_improvement_probability": 1.0,
    }
    if local_candidate.get("enabled") is not True:
        raise PreparationError("frozen local-MM candidate is not enabled")
    for field, expected in expected_local.items():
        if field == "enabled":
            continue
        try:
            observed = float(local_candidate.get(field))
        except (TypeError, ValueError) as error:
            raise PreparationError(f"frozen local-MM {field} is not numeric") from error
        if observed != expected:
            raise PreparationError(
                f"frozen local-MM {field}={observed} differs from "
                f"the case-study protocol value {expected}"
            )
    protocol = {
        "profile_id": "systemic_liquidity_shock_queue_reactive",
        "experiment": "liquidity_shock_causality",
        "duration_seconds": 23400,
        "decision_window_ms": 1000.0,
        "stochastic_baseline_normalization_seconds": 23400.0,
        "cadence_windows_ms": [1000.0],
        "shock_time_seconds": 11700.0,
        "shock_fraction": 0.10,
        "shock_target_count": 0,
        "shock_target_seed": 314159,
        "shock_top_depth_multiple": 0.0,
        "shock_reference_bid_depth_multiple": 3.0,
        "shock_direction_rule": "inventory_adverse_at_left_limit",
        "post_shock_horizon_seconds": 1800.0,
        "production_ranks": 16,
        "financial_risk_limits": [800.0, 1600.0],
        "reference_risk_limit": 1600.0,
        "local_inventory_limit": 800.0,
        "capacity_threshold": 0.5,
        "minimum_shared_quote_scale": 0.05,
        "shared_quote_relative": True,
        "shared_quote_multiplier": 2.0,
        "shared_capacity_relative": True,
        "shared_quote_levels": 3,
        "required_pre_shock_requested_two_sided_book_fraction": 1.0,
        "minimum_pre_shock_resting_two_sided_book_fraction": 0.95,
        "mechanism_preflight_lookback_seconds": 60.0,
        "minimum_pre_shock_economic_quote_scale": 0.25,
        "maximum_pre_shock_utilization": 0.90,
        "minimum_pre_shock_bbo_depth_participation": 0.05,
        "target_side_materiality_assessed_by_realized_shock_absorption": True,
        "minimum_nonzero_inventory_asset_fraction": 0.25,
        "minimum_shock_absorption_fraction": 0.025,
        "preflight_threshold_status": (
            "fixed_after_mechanism_pilot_before_financial_paths"
        ),
        "local_mm_spread_elasticity": 0.0,
        "local_mm_max_improvement_probability": 1.0,
        "repetitions": 20,
        "path_count": 200,
        "base_seed": 20200130,
        "primary_contrast": (
            "global_minus_uncoupled_paired_difference_in_differences"
        ),
        "primary_outcome": "relative_non_target_top_depth_deterioration",
        "secondary_outcome": "non_target_spread_deterioration_bps",
        "reporting_horizons_seconds": [1, 5, 30, 300, 1800],
        "uncoupled_capacity_control": "asset_specific_equal_total_capacity",
        "asset_level_shock_dose_equality_required": True,
        "state_contingent_direction_rule_identical_across_mechanisms": True,
        "shock_fill_ownership_required": True,
        "truncated_full_prefix_equality_required": True,
        "shared_off_treatment_isolation_required": True,
        "computational_and_financial_outputs_share_one_frozen_model": True,
        "shared_dealer_mechanism_preflight_required": True,
    }
    runtime = {
        "case_config": case_path,
        "background_policy_mapping": output_background,
        "value_policy": output_value,
        "cluster_map": output_clusters,
        "executable": executable,
        "source_manifest": source_manifest,
    }
    background_manifest = artifact_manifest(background_artifacts)
    empirical_manifest = empirical_artifact_manifest(
        symbols, case_path, data_root,
    )
    heldout_manifest = evidence["heldout_manifest"]
    assert isinstance(heldout_manifest, Mapping)
    payload: dict[str, object] = {
        "schema_version": 5,
        "status": "portable_queue_reactive_case_ready",
        "calibration_provenance_mode": (
            "queue_reactive_training_freeze_and_heldout_validation"
        ),
        "model_role": "marketwide_development_model",
        "financial_claim_scope": "within_model_counterfactual",
        "heldout_date": "2020-01-30",
        "symbol_count": len(symbols),
        "book_count": len(symbols),
        "venue_count": 1,
        "parameters_reestimated": False,
        "path_only_rebinding": True,
        "gate_protocol": str(heldout_manifest.get(
            "gate_protocol", "marketwide-six-v2",
        )),
        "decision_window_ms": 1000.0,
        "hawkes_activity_scale": 0.3,
        "universe_config": str(case_path),
        "universe_config_sha256": sha256_file(case_path),
        "runtime_configuration_schema": RUNTIME_SCHEMA,
        "cohort_identity": cohort_identity,
        "case_executable": str(executable),
        "case_executable_sha256": case_executable_sha256,
        "executable_provenance": {
            "validated_baseline_executable_sha256": (
                validated_baseline_executable_sha256
            ),
            "case_executable_sha256": case_executable_sha256,
            "post_validation_treatment_amendment": (
                post_validation_treatment_amendment
            ),
            "amendment_scope": (
                "shared_dealer_counterfactual_and_observation_only"
                if post_validation_treatment_amendment else "none"
            ),
            "ordinary_market_calibration_parameters_changed": False,
            "ordinary_market_validation_claim_extended": False,
            "interpretation": (
                "ordinary-market adequacy remains bound to the validated "
                "baseline executable; the case executable is used only for "
                "the shared-dealer counterfactual and added observations"
            ),
        },
        "case_study_protocol": protocol,
        "case_study_protocol_sha256": sha256_json(protocol),
        "local_market_maker": {
            "enabled": True,
            "interval_ms": float(local_candidate.get("interval_ms", 0.0)),
            "quantity_multiplier": float(
                local_candidate.get("quantity_multiplier", 0.0)
            ),
            "improvement_probability": float(
                local_candidate.get("improvement_probability", 0.0)
            ),
            "spread_elasticity": float(
                local_candidate.get("spread_elasticity", 0.0)
            ),
            "max_improvement_probability": float(
                local_candidate.get("max_improvement_probability", 1.0)
            ),
        },
        "shared_market_maker": {
            "role": "post_validation_counterfactual_treatment",
            "relative_quote_multiplier": 2.0,
            "quote_levels": 1,
            "capacity_threshold": 0.5,
            "minimum_quote_scale": 0.05,
            "tight_risk_limit_per_asset": 800.0,
            "reference_risk_limit_per_asset": 1600.0,
        },
        "background_policy_artifacts": background_manifest,
        "empirical_target_artifacts": empirical_manifest,
        "runtime_artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in runtime.items()
        },
        "validation_evidence": {
            "validation_claimed": True,
            "certification_claimed": False,
            "heldout_manifest_sha256": sha256_file(
                evidence["heldout_manifest_path"],  # type: ignore[arg-type]
            ),
            "expanded_training_freeze_sha256": sha256_file(
                evidence["expanded_freeze_path"],  # type: ignore[arg-type]
            ),
            "strict_validation_report_sha256": sha256_file(
                evidence["strict_report_path"],  # type: ignore[arg-type]
            ),
            "augmentation_provenance_sha256": sha256_file(
                evidence["augmentation_path"],  # type: ignore[arg-type]
            ),
            "heldout_simulation_config_sha256": sha256_file(heldout_path),
        },
        "seagull_sources": {
            "pool_root": str(pool_root),
            "selection_root": str(selection_root),
            "data_root": str(data_root),
            "pooling_provenance_sha256": sha256_file(
                evidence["pooling_path"],  # type: ignore[arg-type]
            ),
            "selection_freeze_sha256": sha256_file(
                evidence["selection_path"],  # type: ignore[arg-type]
            ),
        },
    }
    payload["artifact_sha256"] = sha256_json(payload)
    manifest_path = output_root / "portable_queue_reactive_case.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "status": payload["status"],
        "symbol_count": len(symbols),
        "output_manifest": str(manifest_path),
        "output_manifest_sha256": sha256_file(manifest_path),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project-root", type=pathlib.Path, required=True)
    result.add_argument("--evidence-root", type=pathlib.Path, required=True)
    result.add_argument("--pool-root", type=pathlib.Path, required=True)
    result.add_argument("--selection-root", type=pathlib.Path, required=True)
    result.add_argument("--data-root", type=pathlib.Path, required=True)
    result.add_argument("--executable", type=pathlib.Path, required=True)
    result.add_argument("--output-root", type=pathlib.Path, required=True)
    result.add_argument(
        "--allow-post-validation-shared-dealer-amendment",
        action="store_true",
        help=(
            "allow a hash-different executable only as an explicitly recorded "
            "post-validation shared-dealer counterfactual/observation amendment"
        ),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(parser().parse_args(argv))
    except (PreparationError, OSError, ValueError) as error:
        print(f"case-bundle preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
