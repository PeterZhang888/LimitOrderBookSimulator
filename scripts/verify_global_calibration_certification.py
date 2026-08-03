#!/usr/bin/env python3
"""Independently certify one complete calibration output.

This program is deliberately a *read-only* verifier.  It does not infer
success from ``passed`` booleans in the calibration report.  Instead it
reopens the immutable cohort, empirical targets, all 35 full-session asset
summaries and their run logs, recomputes the robust-fit, fixed-clock and
finite-boundary gates, and checks that every persisted status agrees with the
recomputed evidence.

The only successful input is a real ``calibration_handoff.json`` produced by
the immutable ``development_validation_gate`` profile.  A preliminary
result, a smoke/pilot output, a count-only 1,480-symbol approximation, or a
partially copied result cannot pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import re
import stat
import sys
from dataclasses import asdict
from typing import Iterable, Mapping, Sequence


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import calibrate_cluster_value_agents as calibration  # noqa: E402
import certification_cohort as cohort  # noqa: E402


PINNED_PROFILE_ID = "development_validation_gate"
PINNED_PROFILE_SHA256 = (
    "a6d6b7b4c673a174251c8a2f7d80de9a0f5dcceb17391c58ca07ea7d416892ab"
)
PINNED_COHORT_SHA256 = (
    "2f57f37762772d9523fb9916fe2376a9578e337d20971fe39aa44d578f5691d3"
)
TRAINING_DATES = (
    "2019-01-30", "2019-03-27", "2019-07-30", "2019-10-30",
    "2019-12-30",
)
VALIDATION_DATE = "2020-01-30"
TRAINING_SEEDS = (
    3424815697, 1799108475, 2301941028, 3637917665, 3007455382,
)
HELDOUT_SEEDS = (1729, 7919, 1103, 6599, 2027)
SESSION_SECONDS = 23_400
SNAPSHOT_INTERVAL_MS = 1_000
SYMBOL_COUNT = 1_480
CLUSTER_COUNT = 10
VALIDATION_SYMBOLS_PER_CLUSTER = 3
MAXIMUM_ROBUST_SCORE = 2.0
MAXIMUM_METRIC_SCORE = 3.0
GROSS_RESIDUAL_DIAGNOSTIC = 6.0
MAXIMUM_TWO_SIDED_SHORTFALL = 0.01
STRATIFIED_EMPIRICAL_FIT_ACCEPTANCE_ROLE = (
    "required_reported_diagnostic_only"
)
MARKETWIDE_EMPIRICAL_FIT_ACCEPTANCE_ROLE = (
    "authoritative_certification_gate"
)
MARKETWIDE_STATUS_SCHEMA_VERSION = 2


class CertificationFailure(RuntimeError):
    """Raised whenever one mandatory item cannot be certified."""


def fail(message: str) -> None:
    raise CertificationFailure(message)


def canonical_json_sha256(value: object) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: pathlib.Path, label: str) -> pathlib.Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        fail(f"{label} is missing: {path}")
        raise AssertionError from error
    if stat.S_ISLNK(status.st_mode):
        fail(f"{label} must not be a symbolic link: {path}")
    if not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
        fail(f"{label} is not a non-empty regular file: {path}")
    return path.resolve(strict=True)


def real_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    """Return one existing directory without following a leaf symlink."""
    path = path.expanduser()
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        fail(f"{label} directory is missing: {path}")
        raise AssertionError from error
    if stat.S_ISLNK(status.st_mode):
        fail(f"{label} directory must not be a symbolic link: {path}")
    if not stat.S_ISDIR(status.st_mode):
        fail(f"{label} is not a directory: {path}")
    return path.resolve(strict=True)


def require_beneath(
    path: pathlib.Path, root: pathlib.Path, label: str,
) -> pathlib.Path:
    """Require a resolved artifact to remain within its declared result root."""
    resolved = path.resolve(strict=True)
    root = root.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        fail(f"{label} escapes the calibration result root: {resolved}")
    return resolved


def require_identical_file(
    observed: pathlib.Path,
    canonical: pathlib.Path,
    label: str,
) -> str:
    """Bind a generated runtime input to the exact canonical input bytes."""
    observed = regular_file(observed, f"{label} observed file")
    canonical = regular_file(canonical, f"{label} canonical file")
    observed_hash = sha256_file(observed)
    canonical_hash = sha256_file(canonical)
    if observed_hash != canonical_hash:
        fail(
            f"{label} is not byte-identical to its canonical input: "
            f"observed={observed_hash} canonical={canonical_hash}"
        )
    return observed_hash


def require_exact_ordered_subset(
    observed: pathlib.Path,
    canonical: pathlib.Path,
    symbols: Sequence[str],
    label: str,
    *,
    renumber_book_id: bool = False,
) -> None:
    """Require an extracted CSV to be the exact ordered canonical row subset."""
    observed_fields, observed_rows = csv_rows(observed, f"{label} observed")
    canonical_fields, canonical_rows = csv_rows(canonical, f"{label} canonical")
    if observed_fields != canonical_fields:
        fail(f"{label} changes the canonical CSV schema")
    by_symbol = {row.get("symbol", ""): row for row in canonical_rows}
    if len(by_symbol) != len(canonical_rows):
        fail(f"{label} canonical CSV has duplicate symbols")
    try:
        expected_rows = [dict(by_symbol[symbol]) for symbol in symbols]
    except KeyError as error:
        fail(f"{label} requests a symbol absent from the canonical CSV: {error.args[0]}")
        raise AssertionError from error
    if renumber_book_id:
        if "book_id" not in observed_fields:
            fail(f"{label} cannot apply canonical book-id renumbering")
        for book_id, row in enumerate(expected_rows):
            row["book_id"] = str(book_id)
    if observed_rows != expected_rows:
        fail(f"{label} is not the exact ordered subset of the canonical CSV")


def canonical_filtered_symbol_order(
    canonical_config: pathlib.Path,
    selected_symbols: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    """Return selected symbols in filtered full-configuration order."""
    selected = tuple(selected_symbols)
    if len(set(selected)) != len(selected):
        fail(f"{label} contains duplicate selected symbols")
    canonical_symbols = symbols_from_csv(canonical_config, f"{label} source")
    selected_set = set(selected)
    filtered = tuple(
        symbol for symbol in canonical_symbols if symbol in selected_set
    )
    if len(filtered) != len(selected) or set(filtered) != selected_set:
        fail(f"{label} does not select an exact subset of the frozen source")
    return filtered


def json_object(path: pathlib.Path, label: str) -> dict[str, object]:
    path = regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"cannot parse {label} {path}: {error}")
        raise AssertionError from error
    if not isinstance(value, dict):
        fail(f"{label} is not a JSON object: {path}")
    return value


def mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        fail(f"{label} is not an object")
    return value


def sequence(value: object, label: str) -> Sequence[object]:
    if (not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))):
        fail(f"{label} is not an array")
    return value


def require_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        fail(f"{label} disagrees with the immutable certification contract")


def require_hash(path: pathlib.Path, expected: object, label: str) -> str:
    if (not isinstance(expected, str) or len(expected) != 64
            or any(ch not in "0123456789abcdef" for ch in expected)):
        fail(f"{label} does not contain a lowercase SHA-256")
    observed = sha256_file(regular_file(path, label))
    if observed != expected:
        fail(f"{label} SHA-256 mismatch: expected={expected} observed={observed}")
    return observed


def path_from_record(value: object, label: str) -> pathlib.Path:
    if not isinstance(value, str) or not value:
        fail(f"{label} is not a path")
    return regular_file(pathlib.Path(value), label)


def csv_rows(path: pathlib.Path, label: str) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    path = regular_file(path, label)
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                fail(f"{label} has no CSV header")
            fields = tuple(reader.fieldnames)
            rows: list[dict[str, str]] = []
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    fail(f"{label}:{line_number} has too many columns")
                rows.append({field: (row.get(field) or "").strip() for field in fields})
    except OSError as error:
        fail(f"cannot read {label}: {error}")
    if not rows:
        fail(f"{label} has no rows")
    return fields, rows


def symbols_from_csv(path: pathlib.Path, label: str) -> tuple[str, ...]:
    fields, rows = csv_rows(path, label)
    if "symbol" not in fields:
        fail(f"{label} has no symbol column")
    values = tuple(row["symbol"] for row in rows)
    if any(not value or value != value.upper() or value != value.strip() for value in values):
        fail(f"{label} contains a noncanonical symbol")
    if len(values) != len(set(values)):
        fail(f"{label} contains duplicate symbols")
    return values


def exact_cohort_csv(
    path: pathlib.Path,
    *,
    label: str,
    project_root: pathlib.Path,
    required_symbols: Sequence[str],
) -> dict[str, object]:
    observed = symbols_from_csv(path, label)
    if tuple(observed) != tuple(required_symbols):
        fail(f"{label} is not the exact ordered 1,480-symbol cohort")
    try:
        return cohort.validate_csv(path, label=label, project_root=project_root)
    except cohort.CohortIdentityError as error:
        fail(str(error))
        raise AssertionError from error


def require_exact_seed_directories(
    root: pathlib.Path,
    seeds: Sequence[int],
    label: str,
    *,
    calibration_root: pathlib.Path | None = None,
) -> pathlib.Path:
    root = real_directory(root, label)
    if calibration_root is not None:
        require_beneath(root, calibration_root, label)
    observed: list[str] = []
    for child in root.iterdir():
        try:
            child_status = child.lstat()
        except FileNotFoundError as error:
            fail(f"{label} changed while its seed directories were inspected")
            raise AssertionError from error
        if stat.S_ISLNK(child_status.st_mode):
            fail(f"{label} contains a symbolic-link entry: {child}")
        if stat.S_ISDIR(child_status.st_mode):
            observed.append(child.name)
    observed.sort()
    expected = sorted(f"seed_{seed}" for seed in seeds)
    if observed != expected:
        fail(f"{label} seed directories differ: expected={expected} observed={observed}")
    return root


def option_value(command: Sequence[str], name: str) -> str | None:
    positions = [index for index, value in enumerate(command) if value == name]
    if len(positions) > 1:
        fail(f"run command repeats {name}")
    if not positions:
        return None
    index = positions[0]
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        fail(f"run command has no value for {name}")
    return command[index + 1]


def parse_run_log(path: pathlib.Path) -> tuple[list[str], float, str]:
    path = regular_file(path, "run log")
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    if len(lines) < 3 or not lines[0].startswith("command="):
        fail(f"run log lacks a command header: {path}")
    try:
        command = json.loads(lines[0][len("command="):])
    except json.JSONDecodeError as error:
        fail(f"run log command is invalid JSON in {path}: {error}")
        raise AssertionError from error
    if (not isinstance(command, list) or not command
            or not all(isinstance(value, str) for value in command)):
        fail(f"run log command is not a string array: {path}")
    if lines[2] != "return_code=0" or any(line == "TIMEOUT" for line in lines):
        fail(f"run did not complete successfully: {path}")
    if not lines[1].startswith("wall_seconds_external="):
        fail(f"run log lacks external wall time: {path}")
    try:
        wall = float(lines[1].split("=", 1)[1])
    except ValueError as error:
        fail(f"run log has invalid wall time: {path}")
        raise AssertionError from error
    if not math.isfinite(wall) or wall <= 0.0:
        fail(f"run log has non-positive wall time: {path}")
    state_hashes = re.findall(
        r"\bstate_hash=(0x[0-9a-fA-F]+)\b", "\n".join(lines[3:]),
    )
    if len(state_hashes) != 1:
        fail(f"run log does not contain exactly one simulator state hash: {path}")
    return list(command), wall, state_hashes[0].lower()


def command_binary_index(command: Sequence[str], binary: pathlib.Path) -> int:
    matches: list[int] = []
    expected_hash = sha256_file(binary)
    for index, value in enumerate(command):
        candidate = pathlib.Path(value).expanduser()
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        if resolved == binary or sha256_file(resolved) == expected_hash:
            matches.append(index)
    if len(matches) != 1:
        fail("run command does not contain exactly one calibration executable")
    return matches[0]


def verify_run(
    *,
    run_dir: pathlib.Path,
    seed: int,
    symbols: Sequence[str],
    config: pathlib.Path,
    policy: pathlib.Path,
    binary: pathlib.Path,
    local_controls: calibration.LocalFlowCandidate,
    shared_enabled: bool,
    shared_multiplier: float,
    calibration_root: pathlib.Path,
) -> tuple[pathlib.Path, float, dict[str, object]]:
    run_dir = real_directory(run_dir, "run evidence")
    require_beneath(run_dir, calibration_root, "run evidence")
    summary = regular_file(run_dir / "fragmented_asset_summary.csv", "asset summary")
    run_log = regular_file(run_dir / "run.log", "run log")
    require_beneath(summary, run_dir, "asset summary")
    require_beneath(run_log, run_dir, "run log")
    command, wall, state_hash = parse_run_log(run_log)
    binary_index = command_binary_index(command, binary)
    actual_binary = pathlib.Path(command[binary_index]).expanduser().resolve()
    launcher = command[:binary_index]
    expected = calibration.command_for_run(
        launcher=launcher,
        binary=actual_binary,
        config=config,
        policy=policy,
        summary=summary,
        duration=SESSION_SECONDS,
        seed=seed,
        local_controls=local_controls,
        shared_quote_multiplier=(shared_multiplier if shared_enabled else None),
        enable_shared_mm=shared_enabled,
        enable_value_agents=True,
    )
    if command != expected:
        fail(f"run command differs from the frozen full-day policy: {run_dir / 'run.log'}")
    if option_value(command, "--duration-seconds") != str(SESSION_SECONDS):
        fail(f"run is not a {SESSION_SECONDS}-second session: {run_dir}")
    if option_value(command, "--seed") != str(seed):
        fail(f"run uses the wrong seed: {run_dir}")
    if "--shock" in command:
        fail(f"validation run unexpectedly enables a shock: {run_dir}")
    observed_symbols = symbols_from_csv(summary, "asset summary")
    if tuple(observed_symbols) != tuple(symbols):
        fail(f"asset summary symbol order differs from its run configuration: {summary}")
    newest_input_mtime = max(
        regular_file(path, "run input").stat().st_mtime_ns
        for path in (binary, config, policy)
    )
    if summary.stat().st_mtime_ns < newest_input_mtime:
        fail(f"asset summary predates an executable/configuration input: {summary}")
    if run_log.stat().st_mtime_ns < summary.stat().st_mtime_ns:
        fail(f"run log predates its asset summary: {run_log}")
    evidence = {
        "run_dir": str(run_dir),
        "seed": seed,
        "state_hash": state_hash,
        "summary_path": str(summary),
        "summary_sha256": sha256_file(summary),
        "run_log_path": str(run_log),
        "run_log_sha256": sha256_file(run_log),
        "config_sha256": sha256_file(config),
        "policy_sha256": sha256_file(policy),
        "binary_sha256": sha256_file(binary),
    }
    return summary, wall, evidence


def require_unique_run_evidence(
    records: Sequence[Mapping[str, object]], expected_count: int,
) -> None:
    """Reject reused paths, hard links, logs, or simulator state hashes."""
    if len(records) != expected_count:
        fail(
            "independent certification did not collect the required number of "
            f"run-evidence records: expected={expected_count} observed={len(records)}"
        )
    for key in ("summary_path", "run_log_path"):
        paths = [regular_file(pathlib.Path(str(record[key])), key) for record in records]
        if len(set(paths)) != len(paths):
            fail(f"independent certification reuses a {key}")
        inode_identities = {(path.stat().st_dev, path.stat().st_ino) for path in paths}
        if len(inode_identities) != len(paths):
            fail(f"independent certification reuses a {key} inode")
    state_hashes = [str(record.get("state_hash", "")) for record in records]
    if any(not value.startswith("0x") for value in state_hashes):
        fail("independent certification contains an invalid simulator state hash")
    if len(set(state_hashes)) != len(state_hashes):
        fail("independent certification reuses a simulator state hash")


def recompute_evaluation(
    summary_paths: Sequence[pathlib.Path],
    walls: Sequence[float],
    symbols: Sequence[str],
    targets: Mapping[str, Mapping[str, calibration.TargetMoment]],
) -> dict[str, object]:
    if len(summary_paths) != 5 or len(walls) != 5:
        fail("each full-session evaluation must contain exactly five seeds")
    canonical_summaries = [
        regular_file(path, "full-session asset summary")
        for path in summary_paths
    ]
    if len(set(canonical_summaries)) != len(canonical_summaries):
        fail("each full-session seed must use a distinct asset-summary path")
    inode_identities = {
        (path.stat().st_dev, path.stat().st_ino) for path in canonical_summaries
    }
    if len(inode_identities) != len(canonical_summaries):
        fail("full-session seeds reuse one asset-summary inode")
    try:
        fit, estimates = calibration.weighted_moment_loss(
            canonical_summaries, targets, symbols,
            required_expected_sample_count=SESSION_SECONDS,
        )
        combined, _ = calibration.weighted_moment_loss(
            canonical_summaries, targets, symbols,
            uncertainty_mode="combined",
            required_expected_sample_count=SESSION_SECONDS,
        )
        selection_score, metric_scores = calibration.metric_balanced_robust_loss(
            estimates, metrics=calibration.METRICS,
        )
        two_sided, two_sided_failures = calibration.two_sided_execution_integrity(
            canonical_summaries, symbols,
            required_expected_sample_count=SESSION_SECONDS,
        )
        background = calibration.finite_boundary_adequacy(
            canonical_summaries, symbols,
            required_expected_sample_count=SESSION_SECONDS,
        )
        value = calibration.value_boundary_adequacy(
            canonical_summaries, symbols,
            required_expected_sample_count=SESSION_SECONDS,
        )
    except (OSError, ValueError, calibration.CalibrationError) as error:
        fail(f"cannot recompute full-session evidence: {error}")
        raise AssertionError from error
    return {
        "fit_wsmrmse": fit,
        "combined_uncertainty_wsmrmse": combined,
        "selection_score": selection_score,
        "selection_metric_scores": metric_scores,
        "asset_summary_interval_ms": SNAPSHOT_INTERVAL_MS,
        "required_expected_sample_count": SESSION_SECONDS,
        "two_sided_integrity_passed": two_sided,
        "two_sided_integrity_failures": two_sided_failures,
        "finite_boundary_adequacy_passed": background.get("passed") is True,
        "finite_boundary_adequacy": background,
        "value_boundary_adequacy_passed": value.get("passed") is True,
        "value_boundary_adequacy": value,
        "seed_count": len(summary_paths),
        "seed_wall_seconds": list(walls),
        "summary_paths": [str(path) for path in canonical_summaries],
        "errors": [],
        "moment_estimates": [asdict(estimate) for estimate in estimates],
    }


def assert_structural_scope_passes(
    evaluation: Mapping[str, object], label: str,
) -> dict[str, object]:
    """Require execution, coverage and boundary adequacy for one scope.

    The 30-symbol stratified panel is deliberately a structural sentinel.  Its
    empirical-fit score is still recomputed from the raw summaries and
    returned as model-criticism evidence, but it is not a second statistical
    acceptance gate alongside the authoritative 1,480-symbol evaluation.
    """
    fit = calibration.empirical_fit_summary(
        evaluation,
        maximum_score=MAXIMUM_ROBUST_SCORE,
        maximum_metric_score=MAXIMUM_METRIC_SCORE,
        maximum_symbol_metric_absolute_residual=GROSS_RESIDUAL_DIAGNOSTIC,
    )
    coverage = calibration.two_sided_coverage_summary(
        evaluation, MAXIMUM_TWO_SIDED_SHORTFALL,
    )
    shortfalls = calibration.two_sided_coverage_shortfalls(
        evaluation, MAXIMUM_TWO_SIDED_SHORTFALL,
    )
    failures: list[str] = []
    if evaluation.get("two_sided_integrity_passed") is not True:
        failures.append("fixed-clock/two-sided execution integrity")
    if evaluation.get("finite_boundary_adequacy_passed") is not True:
        failures.append("background finite-boundary adequacy")
    if evaluation.get("value_boundary_adequacy_passed") is not True:
        failures.append("value-order finite-boundary adequacy")
    if shortfalls:
        failures.append("empirical two-sided coverage shortfall")
    if failures:
        fail(f"{label} failed recomputed structural gates: " + "; ".join(failures))
    return {
        "empirical_fit": fit,
        "empirical_fit_failure_reasons": (
            [] if fit.get("passed") is True
            else calibration.empirical_fit_failure_reasons(label, fit)
        ),
        "coverage": coverage,
    }


def assert_scope_passes(evaluation: Mapping[str, object], label: str) -> dict[str, object]:
    """Require both structural and empirical adequacy for an authoritative scope."""
    result = assert_structural_scope_passes(evaluation, label)
    if result["empirical_fit"].get("passed") is not True:
        fail(
            f"{label} failed recomputed empirical-fit gate: "
            + "; ".join(
                str(reason)
                for reason in result["empirical_fit_failure_reasons"]
            )
        )
    return result


def require_status_evaluation(
    status_path: pathlib.Path,
    *,
    label: str,
    metadata: Mapping[str, object],
    expected_claims: Mapping[str, object],
    recomputed: Mapping[str, object],
    cohort_identity: Mapping[str, object],
) -> None:
    status = json_object(status_path, label)
    for key, expected in metadata.items():
        require_equal(status.get(key), expected, f"{label}.{key}")
    for key, expected in expected_claims.items():
        require_equal(status.get(key), expected, f"{label}.{key}")
    require_equal(status.get("cohort_identity"), cohort_identity, f"{label}.cohort_identity")
    persisted_evaluation = mapping(status.get("evaluation"), f"{label}.evaluation")
    # Wall time is non-scientific and is rounded in run.log.  Remove it at
    # every nesting level while independently recomputing every scientific
    # field, including all dated training-day evaluations.
    comparable = calibration.evaluation_report(recomputed)

    def without_wall_times(value: object) -> object:
        if isinstance(value, Mapping):
            return {
                key: without_wall_times(child)
                for key, child in value.items()
                if key != "seed_wall_seconds"
            }
        if isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)):
            return [without_wall_times(child) for child in value]
        return value

    if without_wall_times(persisted_evaluation) != without_wall_times(comparable):
        fail(f"{label}.evaluation disagrees with the reopened CSV evidence")


def selection_summary_paths(value: object, label: str) -> list[pathlib.Path]:
    """Collect distinct summary paths in one selected training-only record."""
    result: list[pathlib.Path] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key == "summary_paths":
                    for raw in sequence(child, f"{label}.summary_paths"):
                        if not isinstance(raw, str):
                            fail(f"{label}.summary_paths contains a non-path")
                        result.append(pathlib.Path(raw).expanduser())
                else:
                    visit(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child)

    visit(value)
    # Aggregated evaluations intentionally repeat each path once in the
    # top-level list and once in their per-day record.  Collapse that exact
    # representational duplication while preserving first-seen order; later
    # checks still reject reuse between logically independent selections.
    return list(dict.fromkeys(result))


def local_controls_from_checkpoint(checkpoint: Mapping[str, object]) -> calibration.LocalFlowCandidate:
    selected = mapping(
        checkpoint.get("selected_global_local_flow"),
        "selection checkpoint selected_global_local_flow",
    )
    controls = mapping(selected.get("controls"), "selected local-flow controls")
    enabled_value = controls.get("local_mm_enabled")
    if type(enabled_value) is not bool:
        fail("selection checkpoint local_mm_enabled must be a JSON boolean")
    try:
        return calibration.LocalFlowCandidate(
            hawkes_activity_scale=float(controls["hawkes_activity_scale"]),
            local_mm_interval_ms=float(controls["local_mm_interval_ms"]),
            local_mm_quantity_multiplier=float(
                controls["local_mm_quantity_multiplier"]
            ),
            local_mm_improvement_probability=float(
                controls["local_mm_improvement_probability"]
            ),
            local_mm_enabled=enabled_value,
            label=str(controls["label"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        fail(f"selection checkpoint has malformed local-flow controls: {error}")
        raise AssertionError from error


def shared_controls_from_checkpoint(checkpoint: Mapping[str, object]) -> tuple[bool, float]:
    selected = mapping(
        checkpoint.get("selected_global_shared_quote"),
        "selection checkpoint selected_global_shared_quote",
    )
    candidate = mapping(selected.get("candidate"), "selected shared quote")
    enabled_value = candidate.get("enabled")
    if type(enabled_value) is not bool:
        fail("selection checkpoint shared-quote enabled must be a JSON boolean")
    try:
        enabled = enabled_value
        multiplier = float(candidate["multiplier"])
    except (KeyError, TypeError, ValueError) as error:
        fail(f"selection checkpoint has malformed shared-quote controls: {error}")
        raise AssertionError from error
    if not math.isfinite(multiplier):
        fail("selection checkpoint has a non-finite shared-quote multiplier")
    if enabled and multiplier <= 0.0:
        fail("enabled shared-quote candidate must have a positive multiplier")
    if not enabled and multiplier != 0.0:
        fail("disabled shared-quote baseline must have multiplier exactly zero")
    return enabled, multiplier


def verify_cluster_layout(
    *,
    assignments: pathlib.Path,
    validation_sample: pathlib.Path,
    symbols: Sequence[str],
) -> tuple[tuple[str, ...], Mapping[int, tuple[str, ...]]]:
    _, assignment_rows = csv_rows(assignments, "cluster assignments")
    if tuple(row.get("symbol", "") for row in assignment_rows) != tuple(symbols):
        fail("cluster assignments do not contain the exact ordered cohort")
    by_symbol = {row["symbol"]: row for row in assignment_rows}
    cluster_ids = sorted({int(row["cluster_id"]) for row in assignment_rows})
    if cluster_ids != list(range(CLUSTER_COUNT)):
        fail("cluster assignments do not contain exactly clusters 0..9")
    marked_centroids: dict[int, set[str]] = {cluster: set() for cluster in cluster_ids}
    assigned_validation: dict[int, list[str]] = {cluster: [] for cluster in cluster_ids}
    for row in assignment_rows:
        cluster = int(row["cluster_id"])
        if row.get("is_representative") == "1":
            marked_centroids[cluster].add(row["symbol"])
        elif row.get("is_representative") != "0":
            fail("cluster assignment has an invalid representative flag")
        if row.get("is_validation_sample") == "1":
            assigned_validation[cluster].append(row["symbol"])
        elif row.get("is_validation_sample") != "0":
            fail("cluster assignment has an invalid validation flag")
    if any(len(values) != 1 for values in marked_centroids.values()):
        fail("each cluster must contain exactly one marked centroid")
    if any(len(values) != VALIDATION_SYMBOLS_PER_CLUSTER
           for values in assigned_validation.values()):
        fail("each cluster must contain exactly three validation symbols")
    _, validation_rows = csv_rows(validation_sample, "validation sample")
    validation_by_cluster: dict[int, list[str]] = {
        cluster: [] for cluster in cluster_ids
    }
    for row in validation_rows:
        symbol = row.get("symbol", "")
        if symbol not in by_symbol:
            fail(f"validation sample contains unknown symbol {symbol!r}")
        cluster = int(row["cluster_id"])
        if int(by_symbol[symbol]["cluster_id"]) != cluster:
            fail(f"validation sample cluster disagrees for {symbol}")
        validation_by_cluster[cluster].append(symbol)
    if validation_by_cluster != assigned_validation:
        fail("validation_sample.csv disagrees with cluster assignment flags")
    try:
        layout = calibration.load_cluster_layout(
            assignments, validation_sample, symbols,
        )
    except calibration.CalibrationError as error:
        fail(f"cluster layout cannot be reproduced by the production helper: {error}")
        raise AssertionError from error
    if tuple(layout.cluster_ids) != tuple(cluster_ids):
        fail("production cluster layout has noncanonical cluster identifiers")
    if any(len(layout.representatives[cluster]) != 3 for cluster in cluster_ids):
        fail("production layout does not derive exactly three training representatives")
    for cluster in cluster_ids:
        if set(layout.representatives[cluster]).intersection(
                validation_by_cluster[cluster]):
            fail(f"cluster {cluster} reuses a training representative for validation")
        if tuple(layout.validation_symbols[cluster]) != tuple(
                sorted(validation_by_cluster[cluster])):
            fail(f"cluster {cluster} validation sample is not reproducible")
    validation_symbols = tuple(
        symbol
        for cluster in cluster_ids
        for symbol in layout.validation_symbols[cluster]
    )
    return validation_symbols, layout.representatives


def verify(args: argparse.Namespace) -> dict[str, object]:
    project_root = real_directory(
        pathlib.Path(args.project_root), "project root",
    )
    if SCRIPT_DIR.parent.resolve(strict=True) != project_root:
        fail(
            "the executing certification verifier is not from --project-root"
        )
    calibration_root = real_directory(
        pathlib.Path(args.calibration_root), "calibration root",
    )
    binary = regular_file(pathlib.Path(args.binary), "calibration executable")
    build_provenance_path = regular_file(
        pathlib.Path(args.build_provenance), "calibration build provenance",
    )
    handoff_path = calibration_root / "calibration_handoff.json"
    handoff = json_object(handoff_path, "certified calibration handoff")
    if (calibration_root / "preliminary_calibration_result.json").exists():
        fail("a preliminary result coexists with the purported certified handoff")
    if (calibration_root / "calibration_failure.json").exists():
        fail("a calibration failure artifact coexists with the handoff")
    require_equal(handoff.get("schema_version"), 1, "handoff.schema_version")
    require_equal(
        handoff.get("artifact_role"), "certified_calibration_handoff",
        "handoff.artifact_role",
    )

    profile = calibration.certification_profile()
    profile_sha = canonical_json_sha256(profile)
    require_equal(profile.get("profile_id"), PINNED_PROFILE_ID, "source profile ID")
    require_equal(profile_sha, PINNED_PROFILE_SHA256, "source profile SHA-256")
    require_equal(
        calibration.certification_profile_sha256(), PINNED_PROFILE_SHA256,
        "calibration module profile SHA-256",
    )
    require_equal(handoff.get("certification_profile"), profile, "handoff profile")
    require_equal(
        handoff.get("observed_runtime_profile"), profile,
        "handoff observed runtime profile",
    )
    require_equal(
        handoff.get("certification_profile_sha256"), PINNED_PROFILE_SHA256,
        "handoff profile SHA-256",
    )

    required_symbols = cohort.load_required_symbols(project_root)
    require_equal(len(required_symbols), SYMBOL_COUNT, "cohort symbol count")
    require_equal(
        cohort.symbol_order_sha256(required_symbols, label="certification cohort"),
        PINNED_COHORT_SHA256,
        "cohort symbol-order SHA-256",
    )
    handoff_cohort = mapping(handoff.get("cohort_identity"), "handoff cohort identity")
    try:
        cohort.require_identity_record(handoff_cohort, label="handoff cohort identity")
    except cohort.CohortIdentityError as error:
        fail(str(error))

    try:
        validated_build = calibration.validate_build_provenance(
            build_provenance_path, binary=binary, project_root=project_root,
        )
    except calibration.CalibrationError as error:
        fail(str(error))
        raise AssertionError from error
    require_equal(
        handoff.get("calibration_build_provenance"), validated_build,
        "handoff build provenance",
    )
    binary_sha = sha256_file(binary)
    require_equal(
        handoff.get("calibration_binary_sha256"), binary_sha,
        "handoff executable SHA-256",
    )
    source_sha = calibration.simulator_source_semantics_sha256(project_root)
    workflow_sha = calibration.workflow_source_semantics_sha256(project_root)
    require_equal(
        handoff.get("simulator_source_semantics_sha256"), source_sha,
        "handoff simulator source semantics",
    )
    require_equal(
        handoff.get("workflow_source_semantics_sha256"), workflow_sha,
        "handoff workflow source semantics",
    )

    pooled_config = path_from_record(
        handoff.get("pooled_training_universe_config"), "pooled training universe",
    )
    require_hash(
        pooled_config, handoff.get("pooled_training_universe_config_sha256"),
        "pooled training universe",
    )
    pooled_identity = exact_cohort_csv(
        pooled_config, label="pooled training universe",
        project_root=project_root, required_symbols=required_symbols,
    )

    training_records = sequence(handoff.get("training_days"), "handoff training days")
    if len(training_records) != len(TRAINING_DATES):
        fail("handoff does not contain exactly five training-day records")
    training_days: list[calibration.TrainingDay] = []
    for expected_date, raw in zip(TRAINING_DATES, training_records, strict=True):
        record = mapping(raw, f"training day {expected_date}")
        require_equal(record.get("date"), expected_date, f"training day {expected_date} date")
        config = path_from_record(record.get("universe_config"), f"{expected_date} universe")
        require_hash(config, record.get("universe_config_sha256"), f"{expected_date} universe")
        exact_cohort_csv(
            config, label=f"training universe {expected_date}",
            project_root=project_root, required_symbols=required_symbols,
        )
        fields, rows = calibration.load_universe_config(config)
        target_root = pathlib.Path(str(record.get("target_root", ""))).expanduser().resolve()
        if not target_root.is_dir():
            fail(f"training target root is missing for {expected_date}: {target_root}")
        target_digest = calibration.target_artifact_bundle_sha256(
            target_root, expected_date, required_symbols, (300, 3_600, None),
        )
        require_equal(
            record.get("target_artifact_bundle_sha256"), target_digest,
            f"{expected_date} target artifact bundle",
        )
        input_digest = calibration.empirical_input_bundle_sha256(config)
        require_equal(
            record.get("empirical_input_bundle_sha256"), input_digest,
            f"{expected_date} empirical input bundle",
        )
        training_days.append(calibration.TrainingDay(
            date=expected_date, universe_config=config, target_root=target_root,
            fields=fields, rows=tuple(rows), universe_config_sha256=sha256_file(config),
        ))

    frozen_config = path_from_record(
        handoff.get("frozen_heldout_opening_config"), "frozen held-out config",
    )
    require_hash(
        frozen_config, handoff.get("frozen_heldout_opening_config_sha256"),
        "frozen held-out config",
    )
    exact_cohort_csv(
        frozen_config, label="frozen held-out runtime universe",
        project_root=project_root, required_symbols=required_symbols,
    )
    require_equal(
        handoff.get("frozen_empirical_input_bundle_sha256"),
        calibration.empirical_input_bundle_sha256(frozen_config),
        "frozen held-out empirical input bundle",
    )

    policy = path_from_record(handoff.get("value_agent_policy_csv"), "value policy")
    require_hash(policy, handoff.get("value_agent_policy_sha256"), "value policy")
    policy_fields, policy_rows = csv_rows(policy, "full-universe value policy")
    require_equal(
        policy_fields, tuple(calibration.POLICY_FIELDS),
        "full-universe value policy schema",
    )
    exact_cohort_csv(
        policy, label="full-universe value policy", project_root=project_root,
        required_symbols=required_symbols,
    )
    assignments = path_from_record(handoff.get("shock_cluster_csv"), "cluster assignments")
    require_hash(assignments, handoff.get("shock_cluster_csv_sha256"), "cluster assignments")
    exact_cohort_csv(
        assignments, label="cluster assignments", project_root=project_root,
        required_symbols=required_symbols,
    )
    validation_sample = path_from_record(
        handoff.get("validation_sample_csv"), "validation sample",
    )
    require_hash(
        validation_sample, handoff.get("validation_sample_sha256"),
        "validation sample",
    )
    validation_symbols, training_representatives = verify_cluster_layout(
        assignments=assignments, validation_sample=validation_sample,
        symbols=required_symbols,
    )
    if len(validation_symbols) != CLUSTER_COUNT * VALIDATION_SYMBOLS_PER_CLUSTER:
        fail("stratified held-out sample does not contain exactly 30 symbols")

    cluster_manifest_record = mapping(
        handoff.get("cluster_manifest"), "handoff cluster manifest",
    )
    cluster_manifest_path = path_from_record(
        cluster_manifest_record.get("path"), "cluster manifest",
    )
    require_hash(
        cluster_manifest_path, cluster_manifest_record.get("sha256"),
        "cluster manifest",
    )
    try:
        observed_manifest = calibration.validate_cluster_manifest(
            cluster_manifest_path, assignments_path=assignments,
            validation_path=validation_sample, universe_config_path=pooled_config,
        )
    except calibration.CalibrationError as error:
        fail(str(error))
        raise AssertionError from error
    require_equal(cluster_manifest_record, observed_manifest, "cluster manifest provenance")

    # The checkpoint is written before any development-validation target is
    # opened.  Its selected evaluations must contain training paths only.
    checkpoint_path = calibration_root / "calibration_selection_checkpoint.json"
    checkpoint = json_object(checkpoint_path, "selection checkpoint")
    require_equal(
        checkpoint.get("schema_version"),
        calibration.SELECTION_CHECKPOINT_SCHEMA_VERSION,
        "selection checkpoint schema",
    )
    require_equal(
        checkpoint.get("status"), "selection_complete_validation_pending",
        "selection checkpoint status",
    )
    require_equal(checkpoint.get("certified_for_case_study"), False, "selection checkpoint authority")
    require_equal(checkpoint.get("certification_profile"), profile, "selection profile")
    require_equal(checkpoint.get("observed_runtime_profile"), profile, "selection runtime profile")
    require_equal(
        checkpoint.get("certification_profile_sha256"), PINNED_PROFILE_SHA256,
        "selection profile SHA-256",
    )
    require_equal(checkpoint.get("cohort_identity"), handoff_cohort, "selection cohort identity")
    require_equal(checkpoint.get("training_dates"), list(TRAINING_DATES), "selection dates")
    require_equal(checkpoint.get("heldout_date"), VALIDATION_DATE, "selection held-out date")
    if "development_validation_target_root" in checkpoint:
        fail("selection checkpoint contains a development-validation target root")
    selected_cluster_policies = sequence(
        checkpoint.get("selected_cluster_policies"),
        "selection checkpoint selected_cluster_policies",
    )
    if len(selected_cluster_policies) != CLUSTER_COUNT:
        fail("selection checkpoint does not contain exactly ten cluster policies")
    observed_representatives: dict[int, tuple[str, ...]] = {}
    selected_policy_by_cluster: dict[int, Mapping[str, object]] = {}
    for index, value in enumerate(selected_cluster_policies):
        record = mapping(value, f"selected cluster policy {index}")
        try:
            cluster_id = int(record.get("cluster_id"))
        except (TypeError, ValueError) as error:
            fail(f"selected cluster policy {index} has an invalid cluster_id")
            raise AssertionError from error
        representative_text = record.get("representative_symbols")
        if not isinstance(representative_text, str):
            fail(f"selected cluster policy {cluster_id} lacks representative_symbols")
        observed_representatives[cluster_id] = tuple(
            symbol for symbol in representative_text.split(";") if symbol
        )
        selected_policy_by_cluster[cluster_id] = record
    require_equal(
        observed_representatives,
        dict(training_representatives),
        "selection checkpoint training representatives",
    )
    _, assignment_rows_for_policy = csv_rows(
        assignments, "cluster assignments for policy binding",
    )
    membership_for_policy = {
        row["symbol"]: int(row["cluster_id"])
        for row in assignment_rows_for_policy
    }
    if set(row["symbol"] for row in policy_rows) != set(membership_for_policy):
        fail("full-universe policy does not cover the cluster assignment universe")
    for row in policy_rows:
        symbol = row["symbol"]
        cluster_id = membership_for_policy[symbol]
        if int(row["cluster_id"]) != cluster_id:
            fail(f"full-universe policy changes cluster membership for {symbol}")
        selected_record = selected_policy_by_cluster[cluster_id]
        try:
            observed_enabled = calibration.parse_bool(
                row["enabled"], label=f"policy:{symbol}:enabled",
            )
            selected_enabled = calibration.parse_bool(
                selected_record.get("enabled"),
                label=f"selected cluster {cluster_id}:enabled",
            )
            observed_threshold = calibration.finite_float(
                row["value_threshold_bps"], label=f"policy:{symbol}:threshold",
            )
            selected_threshold = calibration.finite_float(
                selected_record.get("value_threshold_bps"),
                label=f"selected cluster {cluster_id}:threshold",
            )
            observed_depth = calibration.finite_float(
                row["value_depth_participation"],
                label=f"policy:{symbol}:depth participation",
            )
            selected_depth = calibration.finite_float(
                selected_record.get("value_depth_participation"),
                label=f"selected cluster {cluster_id}:depth participation",
            )
        except calibration.CalibrationError as error:
            fail(f"cannot bind selected policy for {symbol}: {error}")
            raise AssertionError from error
        if (observed_enabled is not selected_enabled
                or observed_threshold != selected_threshold
                or observed_depth != selected_depth):
            fail(
                f"full-universe policy row for {symbol} disagrees with the "
                f"selected cluster-{cluster_id} candidate"
            )
    selection_expected_summary_counts = {
        "selected_global_local_flow": (
            len(TRAINING_DATES) * len(calibration.CERTIFICATION_STAGE3_SEEDS)
        ),
        "selected_cluster_policies": (
            CLUSTER_COUNT * len(TRAINING_DATES)
            * len(calibration.CERTIFICATION_STAGE3_SEEDS)
        ),
        "selected_global_shared_quote": (
            len(TRAINING_DATES) * len(calibration.CERTIFICATION_STAGE3_SEEDS)
        ),
    }
    all_selection_paths: list[pathlib.Path] = []
    for selected_key, expected_count in selection_expected_summary_counts.items():
        raw_paths = selection_summary_paths(
            checkpoint.get(selected_key), selected_key,
        )
        paths = [
            require_beneath(
                regular_file(path, f"{selected_key} training summary"),
                calibration_root,
                f"{selected_key} training summary",
            )
            for path in raw_paths
        ]
        if not paths:
            fail(f"{selected_key} contains no training summary evidence")
        if len(paths) != expected_count:
            fail(
                f"{selected_key} training summary count differs from the "
                f"declared protocol: expected={expected_count} "
                f"observed={len(paths)}"
            )
        if any("heldout" in path.as_posix().lower() for path in paths):
            fail(f"{selected_key} contains held-out summary evidence")
        if len(set(paths)) != len(paths):
            fail(f"{selected_key} reuses a training summary path")
        inode_identities = {
            (path.stat().st_dev, path.stat().st_ino) for path in paths
        }
        if len(inode_identities) != len(paths):
            fail(f"{selected_key} reuses a training summary inode")
        all_selection_paths.extend(paths)
    if len(set(all_selection_paths)) != len(all_selection_paths):
        fail("independent selection blocks reuse a training summary path")
    selection_inode_identities = {
        (path.stat().st_dev, path.stat().st_ino)
        for path in all_selection_paths
    }
    if len(selection_inode_identities) != len(all_selection_paths):
        fail("independent selection blocks reuse a training summary inode")

    local_controls = local_controls_from_checkpoint(checkpoint)
    shared_enabled, shared_multiplier = shared_controls_from_checkpoint(checkpoint)
    runtime_controls = mapping(handoff.get("runtime_controls"), "handoff runtime controls")
    expected_runtime_controls = {
        "hawkes_activity_scale": local_controls.hawkes_activity_scale,
        "local_market_maker_enabled": local_controls.local_mm_enabled,
        "local_mm_interval_ms": local_controls.local_mm_interval_ms,
        "local_mm_quantity_multiplier": local_controls.local_mm_quantity_multiplier,
        "local_mm_improvement_probability": local_controls.local_mm_improvement_probability,
        "shared_market_maker_enabled": shared_enabled,
        "shared_quote_mode": "relative_to_empirical_symbol_quote_size",
        "shared_quote_multiplier": shared_multiplier,
        "shared_quote_levels": 1,
        "decision_window_ms": 1000.0,
    }
    require_equal(runtime_controls, expected_runtime_controls, "handoff runtime controls")

    # Pool provenance and the frozen held-out configuration are reopened.  In
    # particular, only the five opening fields may come from 2020-01-30.
    pool_record = mapping(handoff.get("pooling_provenance"), "handoff pooling provenance")
    pool_path = path_from_record(pool_record.get("path"), "pooling provenance")
    require_hash(pool_path, pool_record.get("sha256"), "pooling provenance")
    persisted_pool = json_object(pool_path, "persisted pooling provenance")
    embedded_producer = mapping(
        pool_record.get("producer_source_verification"),
        "pooling producer-source verification",
    )
    producer_root = pathlib.Path(
        str(embedded_producer.get("producer_project_root", ""))
    ).expanduser().resolve()
    try:
        observed_producer = calibration.validate_pooling_producer_workflow_source(
            persisted_pool,
            producer_project_root=producer_root,
            consumer_project_root=project_root,
        )
    except calibration.CalibrationError as error:
        fail(str(error))
        raise AssertionError from error
    require_equal(
        embedded_producer, observed_producer,
        "pooling producer-source verification",
    )
    require_equal(
        {
            **persisted_pool,
            "path": str(pool_path),
            "sha256": sha256_file(pool_path),
            "producer_source_verification": observed_producer,
        },
        pool_record,
        "embedded pooling provenance",
    )
    require_equal(persisted_pool.get("schema_version"), 7, "pooling schema version")
    require_equal(persisted_pool.get("training_dates"), list(TRAINING_DATES), "pooling dates")
    require_equal(persisted_pool.get("heldout_date"), VALIDATION_DATE, "pooling validation date")
    require_equal(
        persisted_pool.get("certification_cohort_required"), True,
        "pooling immutable cohort requirement",
    )
    pooling_settings = mapping(persisted_pool.get("pooling"), "pooling settings")
    require_equal(
        pooling_settings.get("heldout_targets_used_for_runtime_configuration"), False,
        "pooling held-out target barrier",
    )
    heldout_pool = mapping(persisted_pool.get("heldout"), "pooling held-out record")
    require_equal(
        heldout_pool.get("heldout_role"),
        "opening_state_and_validation_targets_only", "held-out pooling role",
    )
    require_equal(
        heldout_pool.get("background_inputs_inherited_from_pooled"), True,
        "held-out background inheritance",
    )
    raw_heldout_config = path_from_record(
        heldout_pool.get("common_config"), "pool held-out opening config",
    )
    pool_training_records = sequence(
        persisted_pool.get("training_days"), "pooling training days",
    )
    if len(pool_training_records) != len(TRAINING_DATES):
        fail("pooling provenance lacks exactly five training sessions")
    source_sessions: dict[str, tuple[str, ...]] = {}
    for expected_date, raw_record in zip(
            TRAINING_DATES, pool_training_records, strict=True):
        source_record = mapping(raw_record, f"pooling source {expected_date}")
        require_equal(
            source_record.get("date"), expected_date,
            f"pooling source date {expected_date}",
        )
        source_path = path_from_record(
            source_record.get("source_config"),
            f"pooling source universe {expected_date}",
        )
        require_hash(
            source_path, source_record.get("source_config_sha256"),
            f"pooling source universe {expected_date}",
        )
        source_sessions[expected_date] = cohort.symbols_from_csv(
            source_path, label=f"pooling source universe {expected_date}",
        )
    heldout_source = path_from_record(
        heldout_pool.get("source_config"), "pooling held-out source universe",
    )
    require_hash(
        heldout_source, heldout_pool.get("source_config_sha256"),
        "pooling held-out source universe",
    )
    source_sessions[VALIDATION_DATE] = cohort.symbols_from_csv(
        heldout_source, label="pooling held-out source universe",
    )
    opening_grid = mapping(
        persisted_pool.get("opening_price_grid_eligibility"),
        "pooling opening-price-grid evidence",
    )
    raw_exclusions = sequence(
        opening_grid.get("excluded_symbols"), "pooling fixed-grid exclusions",
    )
    excluded_symbols: list[str] = []
    for raw_exclusion in raw_exclusions:
        exclusion = mapping(raw_exclusion, "pooling fixed-grid exclusion")
        symbol = exclusion.get("symbol")
        if not isinstance(symbol, str):
            fail("pooling fixed-grid exclusion has no symbol")
        excluded_symbols.append(symbol)
    try:
        observed_input_selection = cohort.certification_pool_input_selection(
            source_sessions=source_sessions,
            excluded_symbols=excluded_symbols,
            final_symbols=required_symbols,
            project_root=project_root,
        )
        cohort.require_pool_input_selection_record(
            persisted_pool.get("certification_input_selection"),
            expected=observed_input_selection,
            label="pooling provenance certification_input_selection",
        )
    except cohort.CohortIdentityError as error:
        fail(str(error))
        raise AssertionError from error
    require_equal(
        handoff.get("certification_input_selection"), observed_input_selection,
        "handoff certification input selection",
    )
    expected_pool_cohort_identity = {
        **pooled_identity,
        "original_intersection_symbol_count": 1509,
        "fixed_price_grid_excluded_symbol_count": 29,
        "artifact_checks": {
            "pooled_training_universe": pooled_identity,
            "heldout_common": cohort.validate_csv(
                raw_heldout_config,
                label="pooling frozen held-out opening universe",
                project_root=project_root,
            ),
            "training_days": {
                day.date: cohort.validate_csv(
                    day.universe_config,
                    label=f"pooling training universe {day.date}",
                    project_root=project_root,
                )
                for day in training_days
            },
        },
    }
    require_equal(
        persisted_pool.get("cohort_identity"), expected_pool_cohort_identity,
        "pooling cohort identity",
    )
    expected_handoff_cohort_identity = {
        "schema_version": 1,
        **pooled_identity,
        "artifact_checks": {
            "pooled_training_universe": pooled_identity,
            "training_days": {
                day.date: cohort.validate_csv(
                    day.universe_config,
                    label=f"calibration training universe {day.date}",
                    project_root=project_root,
                )
                for day in training_days
            },
            "heldout_opening_universe": cohort.validate_csv(
                raw_heldout_config,
                label="calibration held-out opening universe",
                project_root=project_root,
            ),
            "cluster_assignments": cohort.validate_csv(
                assignments, label="certified cluster assignments",
                project_root=project_root,
            ),
            "full_universe_policy": cohort.validate_csv(
                policy, label="certified full-universe policy",
                project_root=project_root,
            ),
            "frozen_heldout_runtime_universe": cohort.validate_csv(
                frozen_config,
                label="certified frozen held-out runtime universe",
                project_root=project_root,
            ),
        },
    }
    require_equal(
        handoff_cohort, expected_handoff_cohort_identity,
        "handoff cohort artifact identity",
    )
    pooled_fields, pooled_rows = calibration.load_universe_config(pooled_config)
    heldout_fields, heldout_rows = calibration.load_universe_config(raw_heldout_config)
    frozen_fields, frozen_rows = calibration.load_universe_config(frozen_config)
    expected_frozen = calibration.merge_frozen_heldout_config(
        pooled_fields, pooled_rows, heldout_fields, heldout_rows,
    )
    if frozen_fields != pooled_fields or frozen_rows != expected_frozen:
        fail("frozen held-out runtime configuration changes more than opening state")

    heldout_target_root = pathlib.Path(
        str(handoff.get("development_validation_target_root", ""))
    ).expanduser().resolve()
    if not heldout_target_root.is_dir():
        fail(f"development-validation target root is missing: {heldout_target_root}")
    heldout_target_digest = calibration.target_artifact_bundle_sha256(
        heldout_target_root, VALIDATION_DATE, required_symbols, (None,),
    )
    require_equal(
        handoff.get("development_validation_target_bundle_sha256"),
        heldout_target_digest, "development-validation target bundle",
    )
    try:
        fully_validated_pool = calibration.validate_pooling_provenance(
            pool_path,
            training_days=training_days,
            pooled_config_path=pooled_config,
            heldout_config_path=raw_heldout_config,
            heldout_target_root=heldout_target_root,
            producer_project_root=producer_root,
            project_root=project_root,
        )
    except calibration.CalibrationError as error:
        fail(f"pooling provenance could not be independently reopened: {error}")
        raise AssertionError from error
    require_equal(
        fully_validated_pool, pool_record,
        "fully reopened pooling provenance",
    )

    # Reopen exactly five seeds on every one of the five training dates.
    training_evaluations: list[tuple[calibration.TrainingDay, Mapping[str, object]]] = []
    training_runtime_config_sha256_by_date: dict[str, str] = {}
    run_evidence: list[dict[str, object]] = []
    training_root = calibration_root / "full_universe_training_adequacy"
    training_runs_root = real_directory(
        training_root / "runs", "full-universe training runs",
    )
    require_beneath(
        training_runs_root, calibration_root, "full-universe training runs",
    )
    observed_day_dirs = sorted(
        path.name for path in training_runs_root.iterdir()
        if not path.is_symlink() and path.is_dir()
    )
    if any(path.is_symlink() for path in training_runs_root.iterdir()):
        fail("full-universe training runs contain a symbolic-link entry")
    expected_day_dirs = sorted(f"day_{date.replace('-', '')}" for date in TRAINING_DATES)
    if observed_day_dirs != expected_day_dirs:
        fail("full-universe training run directories differ from the five fixed dates")
    for day in training_days:
        identifier = f"day_{day.date.replace('-', '')}"
        config = regular_file(
            training_root / "training_days" / identifier
            / "full_universe_training_config.csv",
            f"full-universe training config {day.date}",
        )
        exact_cohort_csv(
            config, label=f"full-universe training config {day.date}",
            project_root=project_root, required_symbols=required_symbols,
        )
        training_runtime_config_sha256_by_date[day.date] = require_identical_file(
            config, day.universe_config,
            f"full-universe training config {day.date}",
        )
        run_root = training_root / "runs" / identifier
        require_exact_seed_directories(
            run_root, TRAINING_SEEDS, f"training {day.date}",
            calibration_root=calibration_root,
        )
        summaries: list[pathlib.Path] = []
        walls: list[float] = []
        for seed in TRAINING_SEEDS:
            summary, wall, evidence = verify_run(
                run_dir=run_root / f"seed_{seed}", seed=seed,
                symbols=required_symbols, config=config, policy=policy,
                binary=binary, local_controls=local_controls,
                shared_enabled=shared_enabled, shared_multiplier=shared_multiplier,
                calibration_root=calibration_root,
            )
            summaries.append(summary)
            walls.append(wall)
            evidence.update({"scope": "training", "date": day.date})
            run_evidence.append(evidence)
        targets = calibration.load_targets(
            day.target_root, day.date, required_symbols,
        )
        evaluation = recompute_evaluation(summaries, walls, required_symbols, targets)
        assert_scope_passes(evaluation, f"full-universe training {day.date}")
        training_evaluations.append((day, evaluation))

    aggregate_training = calibration.aggregate_training_day_evaluations(
        training_evaluations, seed_count=len(TRAINING_SEEDS),
    )
    training_adequacy = calibration.full_universe_training_adequacy_summary(
        aggregate_training,
        maximum_score=MAXIMUM_ROBUST_SCORE,
        maximum_metric_score=MAXIMUM_METRIC_SCORE,
        maximum_symbol_metric_absolute_residual=GROSS_RESIDUAL_DIAGNOSTIC,
    )
    if training_adequacy.get("passed") is not True:
        fail("recomputed five-day full-universe training adequacy failed: "
             + "; ".join(str(value) for value in training_adequacy.get("failure_reasons", [])))
    training_status_path = calibration_root / "full_universe_training_adequacy_status.json"
    training_record = mapping(
        handoff.get("full_universe_training_adequacy"),
        "handoff full-universe training adequacy",
    )
    require_hash(
        training_status_path, training_record.get("status_sha256"),
        "full-universe training status",
    )
    require_status_evaluation(
        training_status_path,
        label="full-universe training status",
        metadata={
            "schema_version": 1,
            "scope": "all_common_symbols_on_every_training_date",
            "symbol_count": SYMBOL_COUNT,
            "required_symbol_count": SYMBOL_COUNT,
            "training_dates": list(TRAINING_DATES),
            "duration_seconds": SESSION_SECONDS,
            "seeds": list(TRAINING_SEEDS),
            "development_validation_targets_opened": False,
        },
        expected_claims=training_adequacy,
        recomputed=aggregate_training,
        cohort_identity=handoff_cohort,
    )

    # The stratified run must be 3 non-representative symbols from each of 10
    # clusters, with the same frozen controls and five fixed seeds.
    stratified_config = regular_file(
        calibration_root / "heldout_stratified_validation_config.csv",
        "stratified held-out config",
    )
    stratified_config_symbols = canonical_filtered_symbol_order(
        frozen_config, validation_symbols, "stratified held-out config",
    )
    if symbols_from_csv(
            stratified_config, "stratified held-out config",
    ) != stratified_config_symbols:
        fail(
            "stratified held-out config does not contain the declared cluster "
            "sample in canonical full-universe order"
        )
    require_exact_ordered_subset(
        stratified_config, frozen_config, stratified_config_symbols,
        "stratified held-out config",
        renumber_book_id=True,
    )
    stratified_policy = regular_file(
        calibration_root / "heldout_stratified_validation_policy.csv",
        "stratified held-out policy",
    )
    if symbols_from_csv(stratified_policy, "stratified held-out policy") != validation_symbols:
        fail("stratified held-out policy does not match the declared cluster sample")
    stratified_policy_fields, stratified_policy_rows = csv_rows(
        stratified_policy, "stratified held-out policy",
    )
    policy_by_symbol = {row.get("symbol", ""): row for row in policy_rows}
    if (stratified_policy_fields != policy_fields
            or stratified_policy_rows != [
                policy_by_symbol[symbol] for symbol in validation_symbols
            ]):
        fail(
            "stratified held-out policy is not the exact ordered subset of "
            "the frozen full-universe policy"
        )
    stratified_root = calibration_root / "heldout_stratified_validation"
    require_exact_seed_directories(
        stratified_root, HELDOUT_SEEDS, "stratified held-out",
        calibration_root=calibration_root,
    )
    stratified_summaries: list[pathlib.Path] = []
    stratified_walls: list[float] = []
    for seed in HELDOUT_SEEDS:
        summary, wall, evidence = verify_run(
            run_dir=stratified_root / f"seed_{seed}", seed=seed,
            symbols=stratified_config_symbols, config=stratified_config,
            policy=stratified_policy, binary=binary, local_controls=local_controls,
            shared_enabled=shared_enabled, shared_multiplier=shared_multiplier,
            calibration_root=calibration_root,
        )
        stratified_summaries.append(summary)
        stratified_walls.append(wall)
        evidence.update({"scope": "stratified", "date": VALIDATION_DATE})
        run_evidence.append(evidence)
    stratified_targets = calibration.load_targets(
        heldout_target_root, VALIDATION_DATE, validation_symbols,
    )
    stratified_evaluation = recompute_evaluation(
        stratified_summaries, stratified_walls, validation_symbols,
        stratified_targets,
    )
    stratified_gate = assert_structural_scope_passes(
        stratified_evaluation,
        calibration.STRATIFIED_EMPIRICAL_FIT_FAILURE_SCOPE,
    )
    stratified_status_path = calibration_root / "heldout_stratified_validation_status.json"
    stratified_record = mapping(
        handoff.get("heldout_stratified_validation"),
        "handoff stratified validation",
    )
    require_equal(
        path_from_record(
            stratified_record.get("status_json"),
            "handoff stratified validation status",
        ),
        stratified_status_path.resolve(),
        "handoff stratified validation status path",
    )
    require_hash(
        stratified_status_path, stratified_record.get("status_sha256"),
        "stratified validation status",
    )
    stratified_empirical_fit_passed = (
        stratified_gate["empirical_fit"].get("passed") is True
    )
    require_equal(stratified_record.get("passed"), True, "handoff stratified passed")
    require_equal(
        stratified_record.get("structural_adequacy_passed"), True,
        "handoff stratified structural adequacy",
    )
    require_equal(
        stratified_record.get("empirical_fit_passed"),
        stratified_empirical_fit_passed,
        "handoff stratified empirical fit",
    )
    require_equal(
        stratified_record.get("empirical_fit_acceptance_role"),
        STRATIFIED_EMPIRICAL_FIT_ACCEPTANCE_ROLE,
        "handoff stratified empirical-fit role",
    )
    require_equal(
        stratified_record.get("empirical_fit_failure_reasons"),
        stratified_gate["empirical_fit_failure_reasons"],
        "handoff stratified empirical-fit diagnostics",
    )
    require_equal(
        stratified_record.get("symbols"), len(validation_symbols),
        "handoff stratified symbol count",
    )
    require_equal(
        stratified_record.get("validation_date"), VALIDATION_DATE,
        "handoff stratified validation date",
    )
    require_equal(
        stratified_record.get("duration_seconds"), SESSION_SECONDS,
        "handoff stratified duration",
    )
    require_equal(
        stratified_record.get("seeds"), list(HELDOUT_SEEDS),
        "handoff stratified seeds",
    )
    require_status_evaluation(
        stratified_status_path,
        label="held-out stratified status",
        metadata={"schema_version": 2, "scope": "pooled_stratified_sample"},
        expected_claims={
            "passed": True,
            "structural_adequacy_passed": True,
            "execution_integrity_passed": True,
            "full_two_sided_book_passed": True,
            "coverage_passed": True,
            "finite_boundary_adequacy_passed": True,
            "finite_boundary_adequacy": {
                "background": stratified_evaluation["finite_boundary_adequacy"],
                "value": stratified_evaluation["value_boundary_adequacy"],
            },
            "background_boundary_adequacy_passed": True,
            "value_boundary_adequacy_passed": True,
            "empirical_fit_passed": (
                stratified_empirical_fit_passed
            ),
            "empirical_fit_acceptance_role": (
                STRATIFIED_EMPIRICAL_FIT_ACCEPTANCE_ROLE
            ),
            "certified_for_case_study": True,
            "failure_reasons": [],
            "empirical_fit_failure_reasons": stratified_gate[
                "empirical_fit_failure_reasons"
            ],
            "coverage_summary": stratified_gate["coverage"],
            "coverage_shortfalls": [],
            "empirical_fit": stratified_gate["empirical_fit"],
        },
        recomputed=stratified_evaluation,
        cohort_identity=handoff_cohort,
    )

    # The market-wide gate is distinct from the 30-symbol stratified gate and
    # must contain all 1,480 symbols for all five fixed seeds.
    marketwide_root = calibration_root / "heldout_marketwide_validation"
    require_exact_seed_directories(
        marketwide_root, HELDOUT_SEEDS, "market-wide held-out",
        calibration_root=calibration_root,
    )
    marketwide_summaries: list[pathlib.Path] = []
    marketwide_walls: list[float] = []
    for seed in HELDOUT_SEEDS:
        summary, wall, evidence = verify_run(
            run_dir=marketwide_root / f"seed_{seed}", seed=seed,
            symbols=required_symbols, config=frozen_config, policy=policy,
            binary=binary, local_controls=local_controls,
            shared_enabled=shared_enabled, shared_multiplier=shared_multiplier,
            calibration_root=calibration_root,
        )
        marketwide_summaries.append(summary)
        marketwide_walls.append(wall)
        evidence.update({"scope": "marketwide", "date": VALIDATION_DATE})
        run_evidence.append(evidence)
    marketwide_targets = calibration.load_targets(
        heldout_target_root, VALIDATION_DATE, required_symbols,
    )
    marketwide_evaluation = recompute_evaluation(
        marketwide_summaries, marketwide_walls, required_symbols,
        marketwide_targets,
    )
    marketwide_gate = assert_scope_passes(
        marketwide_evaluation, "held-out market-wide validation",
    )
    marketwide_status_path = calibration_root / "heldout_marketwide_validation_status.json"
    marketwide_record = mapping(
        handoff.get("heldout_marketwide_validation"),
        "handoff market-wide validation",
    )
    require_hash(
        marketwide_status_path, marketwide_record.get("status_sha256"),
        "market-wide validation status",
    )
    require_equal(
        path_from_record(
            marketwide_record.get("status_json"),
            "handoff market-wide validation status",
        ),
        marketwide_status_path.resolve(),
        "handoff market-wide validation status path",
    )
    require_equal(marketwide_record.get("passed"), True, "handoff market-wide passed")
    require_equal(
        marketwide_record.get("symbols"), len(required_symbols),
        "handoff market-wide symbol count",
    )
    require_equal(
        marketwide_record.get("validation_date"), VALIDATION_DATE,
        "handoff market-wide validation date",
    )
    require_equal(
        marketwide_record.get("duration_seconds"), SESSION_SECONDS,
        "handoff market-wide duration",
    )
    require_equal(
        marketwide_record.get("seeds"), list(HELDOUT_SEEDS),
        "handoff market-wide seeds",
    )
    require_equal(
        marketwide_record.get("empirical_fit_acceptance_role"),
        MARKETWIDE_EMPIRICAL_FIT_ACCEPTANCE_ROLE,
        "handoff market-wide empirical-fit role",
    )
    require_status_evaluation(
        marketwide_status_path,
        label="held-out market-wide status",
        metadata={
            "schema_version": MARKETWIDE_STATUS_SCHEMA_VERSION,
            "scope": "full_universe_marketwide",
            "symbol_count": SYMBOL_COUNT,
            "required_symbol_count": SYMBOL_COUNT,
            "validation_date": VALIDATION_DATE,
            "duration_seconds": SESSION_SECONDS,
            "seeds": list(HELDOUT_SEEDS),
        },
        expected_claims={
            "passed": True,
            "execution_integrity_passed": True,
            "full_two_sided_book_passed": True,
            "coverage_passed": True,
            "finite_boundary_adequacy_passed": True,
            "finite_boundary_adequacy": {
                "background": marketwide_evaluation["finite_boundary_adequacy"],
                "value": marketwide_evaluation["value_boundary_adequacy"],
            },
            "background_boundary_adequacy_passed": True,
            "value_boundary_adequacy_passed": True,
            "empirical_fit_passed": True,
            "empirical_fit_acceptance_role": (
                MARKETWIDE_EMPIRICAL_FIT_ACCEPTANCE_ROLE
            ),
            "certified_for_case_study": True,
            "failure_reasons": [],
            "coverage_summary": marketwide_gate["coverage"],
            "empirical_fit": marketwide_gate["empirical_fit"],
        },
        recomputed=marketwide_evaluation,
        cohort_identity=handoff_cohort,
    )

    # Timestamp ordering is supplementary provenance: the checkpoint and all
    # full-universe training evidence must predate the first held-out output.
    first_heldout_mtime = min(
        path.stat().st_mtime_ns
        for path in (*stratified_summaries, *marketwide_summaries)
    )
    if checkpoint_path.stat().st_mtime_ns > first_heldout_mtime:
        fail("selection checkpoint was written after held-out execution began")
    if max(
        pathlib.Path(path).stat().st_mtime_ns
        for evaluation in (item[1] for item in training_evaluations)
        for path in evaluation["summary_paths"]  # type: ignore[index]
    ) > first_heldout_mtime:
        fail("full-universe training adequacy was not completed before held-out execution")

    require_unique_run_evidence(
        run_evidence,
        len(TRAINING_DATES) * len(TRAINING_SEEDS) + 2 * len(HELDOUT_SEEDS),
    )

    report_path = path_from_record(handoff.get("calibration_report"), "calibration report")
    require_hash(report_path, handoff.get("calibration_report_sha256"), "calibration report")

    # Persisted claims are checked *after* independent recomputation.  They
    # cannot make a failed raw evaluation pass.
    certification_record = mapping(handoff.get("certification"), "handoff certification")
    required_true_claims = (
        "runtime_matches_certification_profile",
        "training_full_universe_adequacy_passed",
        "execution_integrity_passed",
        "full_two_sided_book_passed",
        "complete_two_sided_clock_passed",
        "coverage_passed",
        "finite_boundary_adequacy_passed",
        "background_boundary_adequacy_passed",
        "value_boundary_adequacy_passed",
        "empirical_fit_passed",
        "marketwide_validation_completed",
        "stratified_structural_adequacy_passed",
        "provenance_integrity_passed",
        "cohort_identity_verified",
        "certified_for_case_study",
    )
    for key in required_true_claims:
        require_equal(certification_record.get(key), True, f"handoff certification.{key}")
    require_equal(
        certification_record.get("empirical_fit_acceptance_scope"),
        "full_universe_marketwide",
        "handoff certification empirical-fit acceptance scope",
    )
    require_equal(
        certification_record.get("stratified_empirical_fit_passed"),
        stratified_empirical_fit_passed,
        "handoff certification.stratified_empirical_fit_passed",
    )
    require_equal(
        certification_record.get("stratified_empirical_fit_acceptance_role"),
        STRATIFIED_EMPIRICAL_FIT_ACCEPTANCE_ROLE,
        "handoff certification.stratified_empirical_fit_acceptance_role",
    )
    require_equal(
        certification_record.get("stratified_empirical_fit_failure_reasons"),
        stratified_gate["empirical_fit_failure_reasons"],
        "handoff certification.stratified_empirical_fit_failure_reasons",
    )
    require_equal(
        certification_record.get("marketwide_empirical_fit_passed"), True,
        "handoff certification.marketwide_empirical_fit_passed",
    )
    require_equal(
        certification_record.get("marketwide_empirical_fit_acceptance_role"),
        MARKETWIDE_EMPIRICAL_FIT_ACCEPTANCE_ROLE,
        "handoff certification.marketwide_empirical_fit_acceptance_role",
    )
    require_equal(
        certification_record.get("certification_profile_id"), PINNED_PROFILE_ID,
        "handoff certification profile ID",
    )
    require_equal(
        certification_record.get("certification_profile_sha256"),
        PINNED_PROFILE_SHA256, "handoff certification profile SHA-256",
    )
    require_equal(certification_record.get("failure_reasons"), [], "certification failures")

    return {
        "schema_version": 1,
        "artifact_role": "independent_global_calibration_certification",
        "status": "PASS",
        "profile_id": PINNED_PROFILE_ID,
        "profile_sha256": PINNED_PROFILE_SHA256,
        "cohort_symbol_count": SYMBOL_COUNT,
        "cohort_symbol_order_sha256": PINNED_COHORT_SHA256,
        "training_dates": list(TRAINING_DATES),
        "training_seeds_per_date": len(TRAINING_SEEDS),
        "training_full_day_runs": len(TRAINING_DATES) * len(TRAINING_SEEDS),
        "training_runtime_config_sha256_by_date": (
            training_runtime_config_sha256_by_date
        ),
        "stratified_symbols": len(validation_symbols),
        "stratified_full_day_runs": len(HELDOUT_SEEDS),
        "marketwide_symbols": len(required_symbols),
        "marketwide_full_day_runs": len(HELDOUT_SEEDS),
        "session_duration_seconds": SESSION_SECONDS,
        "recomputed_training_aggregate_score": training_adequacy[
            "aggregate_selection_score"
        ],
        "recomputed_stratified_score": stratified_gate["empirical_fit"][
            "selection_score"
        ],
        "recomputed_stratified_empirical_fit_passed": (
            stratified_empirical_fit_passed
        ),
        "stratified_empirical_fit_acceptance_role": (
            STRATIFIED_EMPIRICAL_FIT_ACCEPTANCE_ROLE
        ),
        "recomputed_marketwide_score": marketwide_gate["empirical_fit"][
            "selection_score"
        ],
        "marketwide_empirical_fit_acceptance_role": (
            MARKETWIDE_EMPIRICAL_FIT_ACCEPTANCE_ROLE
        ),
        "binary_sha256": binary_sha,
        "simulator_source_semantics_sha256": source_sha,
        "workflow_source_semantics_sha256": workflow_sha,
        "pooling_provenance_sha256": sha256_file(pool_path),
        "calibration_handoff_sha256": sha256_file(handoff_path),
        "raw_run_evidence": run_evidence,
        "raw_run_evidence_bundle_sha256": canonical_json_sha256(run_evidence),
        "pooled_cohort_identity": pooled_identity,
        "heldout_information_used_for_selection": False,
        "independent_final_holdout": False,
        "interpretation": (
            "development-validation balanced panel; availability-conditioned, "
            "not an independent final holdout"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--build-provenance", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify(args)
    except (CertificationFailure, OSError, ValueError) as error:
        print(json.dumps({
            "schema_version": 1,
            "artifact_role": "independent_global_calibration_certification",
            "status": "FAIL",
            "reason": str(error),
        }, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
