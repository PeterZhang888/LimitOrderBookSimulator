#!/usr/bin/env python3
"""Summarize an R4 calibration failure from its persistent JSON artifacts.

This is a read-only diagnostic utility.  It does not rerun candidates, alter
eligibility gates, or select a policy.  Outputs use only Python's standard
library so the script can run on a Seagull compute node without ``jq``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import re
import tempfile
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


CANDIDATE_FIELDS = (
    "block", "stage", "cluster_id", "candidate_index", "candidate_label",
    "eligible", "finite_selection_score", "finite_fit_wsmrmse",
    "two_sided_integrity_passed", "finite_boundary_adequacy_passed",
    "error_free", "fit_wsmrmse", "selection_score", "error_count",
    "two_sided_failure_count", "boundary_failure_count",
    "asset_boundary_failure_count", "run_boundary_failure_count",
    "failing_dates", "failing_seeds", "max_asset_event_ratio",
    "max_asset_quantity_ratio", "max_run_event_ratio",
    "max_run_quantity_ratio", "hawkes_activity_scale", "local_mm_enabled",
    "local_mm_interval_ms", "local_mm_quantity_multiplier",
    "local_mm_improvement_probability", "enabled", "threshold_bps",
    "depth_participation", "order_quantity", "multiplier", "controls_json",
    "errors_json",
    "source_path", "source_sha256",
)

DAY_SEED_FIELDS = (
    "block", "stage", "cluster_id", "candidate_index", "candidate_label",
    "date", "seed", "summary_path", "day_boundary_passed",
    "failure_count", "asset_event_failure_count",
    "asset_quantity_failure_count", "run_event_failure_count",
    "run_quantity_failure_count", "max_asset_event_ratio",
    "max_asset_event_symbol", "max_asset_quantity_ratio",
    "max_asset_quantity_symbol", "run_event_ratio", "run_quantity_ratio",
    "boundary_truncation_events", "background_event_count",
    "boundary_truncated_quantity", "background_removal_requested_quantity",
)

FAILURE_FIELDS = (
    "block", "stage", "cluster_id", "candidate_index", "candidate_label",
    "date", "seed", "summary_path", "scope", "symbol", "metric",
    "numerator", "denominator", "ratio", "maximum", "excess",
)

SYMBOL_FIELDS = (
    "symbol", "metric", "observation_count", "failure_observation_count",
    "max_ratio", "maximum_at_max", "excess_at_max", "block", "stage",
    "cluster_id", "candidate_index", "candidate_label", "date", "seed",
    "summary_path", "numerator_at_max", "denominator_at_max",
)

STAGE_FIELDS = (
    "block", "stage", "cluster_id", "status", "evaluated_candidates",
    "eligible_candidates", "promoted_candidates",
    "configured_ranked_survivor_count", "checkpoint_path", "checkpoint_sha256",
)


class DiagnosticError(RuntimeError):
    """Raised when persisted diagnostics are incomplete or inconsistent."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-root", required=True,
        help="R4 calibration directory containing calibration_progress.json",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="new directory for diagnostic_summary.json and CSV tables",
    )
    parser.add_argument(
        "--top-symbols", type=int, default=25,
        help="number of worst symbol/metric records embedded in JSON",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace this script's existing output files",
    )
    args = parser.parse_args(argv)
    if args.top_symbols <= 0:
        parser.error("--top-symbols must be positive")
    return args


def load_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise DiagnosticError(f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DiagnosticError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise DiagnosticError(f"{label} is not a JSON object: {path}")
    return value


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise DiagnosticError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True,
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_csv(
    path: pathlib.Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise DiagnosticError(f"refusing to overwrite {path}")
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
                writer.writerow({field: row.get(field, "") for field in fields})
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def finite_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def maximum_record(
    records: Iterable[Mapping[str, Any]], key: str,
) -> Mapping[str, Any] | None:
    valid = [record for record in records if finite_number(record.get(key)) is not None]
    if not valid:
        return None
    return max(valid, key=lambda record: float(record[key]))


def seed_from_path(path: object) -> str:
    match = re.search(r"(?:^|/)seed_([^/]+)(?:/|$)", str(path))
    return match.group(1) if match else ""


def day_evaluations(evaluation: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    raw = evaluation.get("training_day_evaluations")
    if isinstance(raw, list):
        result: list[tuple[str, Mapping[str, Any]]] = []
        for item in raw:
            if not isinstance(item, Mapping) or not isinstance(item.get("evaluation"), Mapping):
                raise DiagnosticError("malformed training_day_evaluations entry")
            result.append((str(item.get("date", "")), item["evaluation"]))
        return result
    return [("", evaluation)]


def verify_persisted_references(
    calibration_root: pathlib.Path,
    progress: Mapping[str, Any],
    checkpoints: Sequence[tuple[pathlib.Path, Mapping[str, Any]]],
) -> tuple[dict[str, int], set[pathlib.Path], set[pathlib.Path]]:
    candidate_references: dict[pathlib.Path, str] = {}
    checkpoint_references: dict[pathlib.Path, str] = {}
    events = progress.get("events")
    if not isinstance(events, list):
        raise DiagnosticError("progress checkpoint has no event list")
    if int(progress.get("event_count", -1)) != len(events):
        raise DiagnosticError("progress event_count disagrees with its event list")
    for event in events:
        if not isinstance(event, Mapping):
            raise DiagnosticError("progress checkpoint has a malformed event")
        kind = event.get("kind")
        if kind not in {"candidate_evaluation", "stage_checkpoint"}:
            continue
        path = pathlib.Path(str(event.get("path", ""))).expanduser().resolve()
        expected = str(event.get("sha256", ""))
        target = (
            candidate_references
            if kind == "candidate_evaluation"
            else checkpoint_references
        )
        previous = target.get(path)
        if previous is not None and previous != expected:
            raise DiagnosticError(f"conflicting {kind} hashes for {path}")
        target[path] = expected
    for _, checkpoint in checkpoints:
        for reference in checkpoint.get("candidate_evaluations", []):
            if not isinstance(reference, Mapping):
                raise DiagnosticError("stage checkpoint has a malformed candidate reference")
            path = pathlib.Path(str(reference.get("path", ""))).expanduser().resolve()
            expected = str(reference.get("sha256", ""))
            previous = candidate_references.get(path)
            if previous is not None and previous != expected:
                raise DiagnosticError(f"conflicting hashes for {path}")
            candidate_references[path] = expected
    all_references = {
        **candidate_references,
        **checkpoint_references,
    }
    for path, expected in all_references.items():
        if not path.is_file():
            raise DiagnosticError(f"referenced persistent diagnostic is missing: {path}")
        observed = sha256(path)
        if observed != expected:
            raise DiagnosticError(
                f"persistent diagnostic SHA-256 mismatch: {path} "
                f"expected={expected} observed={observed}"
            )
        try:
            path.relative_to(calibration_root)
        except ValueError as error:
            raise DiagnosticError(
                f"persistent diagnostic lies outside calibration root: {path}"
            ) from error
    return (
        {
            "referenced_candidate_files": len(candidate_references),
            "referenced_stage_checkpoint_files": len(checkpoint_references),
            "verified_hashes": len(all_references),
        },
        set(candidate_references),
        set(checkpoint_references),
    )


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    root = pathlib.Path(args.calibration_root).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    if not root.is_dir():
        raise DiagnosticError(f"calibration root is not a directory: {root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_path = root / "calibration_progress.json"
    failure_path = root / "calibration_failure.json"
    progress = load_object(progress_path, "progress checkpoint")
    failure = load_object(failure_path, "calibration failure")
    if progress.get("artifact_role") != "calibration_progress_checkpoint":
        raise DiagnosticError("wrong progress-checkpoint artifact role")
    if failure.get("artifact_role") != "calibration_failure":
        raise DiagnosticError("wrong terminal-failure artifact role")
    if failure.get("status") != "failed":
        raise DiagnosticError("terminal failure record is not marked failed")
    failure_progress_verified = False
    failure_progress = failure.get("progress_checkpoint")
    if failure_progress is not None:
        if not isinstance(failure_progress, Mapping):
            raise DiagnosticError("terminal failure has malformed progress provenance")
        recorded_progress_path = pathlib.Path(
            str(failure_progress.get("path", ""))
        ).expanduser().resolve()
        if recorded_progress_path != progress_path:
            raise DiagnosticError(
                "terminal failure references a different progress checkpoint: "
                f"{recorded_progress_path}"
            )
        expected_progress_hash = str(failure_progress.get("sha256", ""))
        observed_progress_hash = sha256(progress_path)
        if observed_progress_hash != expected_progress_hash:
            raise DiagnosticError(
                "terminal-failure progress SHA-256 mismatch: "
                f"expected={expected_progress_hash} observed={observed_progress_hash}"
            )
        if failure_progress.get("snapshot") != progress:
            raise DiagnosticError(
                "terminal-failure progress snapshot differs from the live checkpoint"
            )
        failure_progress_verified = True

    checkpoint_items: list[tuple[pathlib.Path, Mapping[str, Any]]] = []
    stage_rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("stage_checkpoint.json")):
        payload = load_object(path, "stage checkpoint")
        if payload.get("artifact_role") != "calibration_stage_checkpoint":
            raise DiagnosticError(f"wrong stage-checkpoint role: {path}")
        counts = payload.get("observed_counts")
        if not isinstance(counts, Mapping):
            raise DiagnosticError(f"stage checkpoint lacks counts: {path}")
        references = payload.get("candidate_evaluations")
        if not isinstance(references, list):
            raise DiagnosticError(f"stage checkpoint lacks candidate references: {path}")
        if int(counts.get("evaluated_candidates", -1)) != len(references):
            raise DiagnosticError(f"stage evaluated count disagrees: {path}")
        eligible = sum(
            reference.get("eligible") is True
            for reference in references if isinstance(reference, Mapping)
        )
        if int(counts.get("eligible_candidates", -1)) != eligible:
            raise DiagnosticError(f"stage eligible count disagrees: {path}")
        checkpoint_items.append((path, payload))
        stage_rows.append({
            "block": payload.get("block", ""),
            "stage": payload.get("stage", ""),
            "cluster_id": payload.get("cluster_id", ""),
            "status": payload.get("status", ""),
            "evaluated_candidates": counts.get("evaluated_candidates", ""),
            "eligible_candidates": counts.get("eligible_candidates", ""),
            "promoted_candidates": counts.get("promoted_candidates", ""),
            "configured_ranked_survivor_count": counts.get(
                "configured_ranked_survivor_count", ""
            ),
            "checkpoint_path": str(path),
            "checkpoint_sha256": sha256(path),
        })

    integrity, referenced_candidates, referenced_checkpoints = (
        verify_persisted_references(root, progress, checkpoint_items)
    )
    candidate_paths = sorted(root.rglob("candidate_evaluation.json"))
    if not candidate_paths:
        raise DiagnosticError(f"no candidate_evaluation.json files below {root}")

    candidate_rows: list[dict[str, Any]] = []
    day_seed_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    symbol_records: dict[tuple[str, str], dict[str, Any]] = {}

    for candidate_path in candidate_paths:
        payload = load_object(candidate_path, "candidate evaluation")
        if payload.get("artifact_role") != "calibration_candidate_evaluation":
            raise DiagnosticError(f"wrong candidate-evaluation role: {candidate_path}")
        candidate = payload.get("candidate")
        eligibility = payload.get("eligibility")
        evaluation = payload.get("evaluation")
        if not isinstance(candidate, Mapping) or not isinstance(eligibility, Mapping) \
                or not isinstance(evaluation, Mapping):
            raise DiagnosticError(f"malformed candidate diagnostic: {candidate_path}")
        predicates = eligibility.get("predicates")
        if not isinstance(predicates, Mapping):
            raise DiagnosticError(f"candidate eligibility predicates are missing: {candidate_path}")
        identity = {
            "block": str(payload.get("block", "")),
            "stage": str(payload.get("stage", "")),
            "cluster_id": "" if payload.get("cluster_id") is None else payload.get("cluster_id"),
            "candidate_index": int(payload.get("candidate_index", -1)),
            "candidate_label": str(candidate.get("label", "")),
        }
        boundary_failures_for_candidate: list[dict[str, Any]] = []
        run_rows_for_candidate: list[dict[str, Any]] = []
        failing_dates: set[str] = set()
        failing_seeds: set[str] = set()

        for trading_date, day_evaluation in day_evaluations(evaluation):
            adequacy = day_evaluation.get("finite_boundary_adequacy")
            if adequacy is None:
                continue
            if not isinstance(adequacy, Mapping):
                raise DiagnosticError(
                    f"finite-boundary report is malformed: {candidate_path} {trading_date}"
                )
            thresholds = adequacy.get("thresholds", {})
            if not isinstance(thresholds, Mapping):
                thresholds = {}
            raw_failures = adequacy.get("failures", [])
            if not isinstance(raw_failures, list):
                raise DiagnosticError(f"finite-boundary failures are malformed: {candidate_path}")
            failures_by_path: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for raw_failure in raw_failures:
                if not isinstance(raw_failure, Mapping):
                    raise DiagnosticError(f"malformed boundary failure: {candidate_path}")
                summary_path = str(raw_failure.get("summary_path", ""))
                failures_by_path[summary_path].append(raw_failure)
                seed = seed_from_path(summary_path)
                row = {
                    **identity,
                    "date": trading_date,
                    "seed": seed,
                    "summary_path": summary_path,
                    "scope": str(raw_failure.get("scope", "")),
                    "symbol": str(raw_failure.get("symbol", "")),
                    "metric": str(raw_failure.get("metric", "")),
                    "numerator": raw_failure.get("numerator", ""),
                    "denominator": raw_failure.get("denominator", ""),
                    "ratio": raw_failure.get("ratio", ""),
                    "maximum": raw_failure.get("maximum", ""),
                }
                ratio = finite_number(raw_failure.get("ratio"))
                maximum = finite_number(raw_failure.get("maximum"))
                row["excess"] = (
                    ratio - maximum if ratio is not None and maximum is not None else ""
                )
                failure_rows.append(row)
                boundary_failures_for_candidate.append(row)
                failing_dates.add(trading_date)
                if seed:
                    failing_seeds.add(seed)

            runs = adequacy.get("runs", [])
            if not isinstance(runs, list):
                raise DiagnosticError(f"finite-boundary runs are malformed: {candidate_path}")
            for run in runs:
                if not isinstance(run, Mapping):
                    raise DiagnosticError(f"malformed finite-boundary run: {candidate_path}")
                summary_path = str(run.get("summary_path", ""))
                seed = seed_from_path(summary_path)
                assets = run.get("assets", [])
                aggregate = run.get("aggregate", {})
                if not isinstance(assets, list) or not isinstance(aggregate, Mapping):
                    raise DiagnosticError(f"malformed run assets/aggregate: {candidate_path}")
                asset_mappings = [asset for asset in assets if isinstance(asset, Mapping)]
                worst_event = maximum_record(asset_mappings, "boundary_event_ratio")
                worst_quantity = maximum_record(asset_mappings, "boundary_quantity_ratio")
                run_failures = failures_by_path.get(summary_path, [])
                run_row = {
                    **identity,
                    "date": trading_date,
                    "seed": seed,
                    "summary_path": summary_path,
                    "day_boundary_passed": adequacy.get("passed") is True,
                    "failure_count": len(run_failures),
                    "asset_event_failure_count": sum(
                        failure.get("scope") == "asset_seed"
                        and failure.get("metric") == "boundary_event_ratio"
                        for failure in run_failures
                    ),
                    "asset_quantity_failure_count": sum(
                        failure.get("scope") == "asset_seed"
                        and failure.get("metric") == "boundary_quantity_ratio"
                        for failure in run_failures
                    ),
                    "run_event_failure_count": sum(
                        failure.get("scope") == "run_aggregate"
                        and failure.get("metric") == "boundary_event_ratio"
                        for failure in run_failures
                    ),
                    "run_quantity_failure_count": sum(
                        failure.get("scope") == "run_aggregate"
                        and failure.get("metric") == "boundary_quantity_ratio"
                        for failure in run_failures
                    ),
                    "max_asset_event_ratio": (
                        worst_event.get("boundary_event_ratio", "")
                        if worst_event else ""
                    ),
                    "max_asset_event_symbol": worst_event.get("symbol", "") if worst_event else "",
                    "max_asset_quantity_ratio": (
                        worst_quantity.get("boundary_quantity_ratio", "")
                        if worst_quantity else ""
                    ),
                    "max_asset_quantity_symbol": (
                        worst_quantity.get("symbol", "") if worst_quantity else ""
                    ),
                    "run_event_ratio": aggregate.get("boundary_event_ratio", ""),
                    "run_quantity_ratio": aggregate.get("boundary_quantity_ratio", ""),
                    "boundary_truncation_events": aggregate.get(
                        "boundary_truncation_events", ""
                    ),
                    "background_event_count": aggregate.get("background_event_count", ""),
                    "boundary_truncated_quantity": aggregate.get(
                        "boundary_truncated_quantity", ""
                    ),
                    "background_removal_requested_quantity": aggregate.get(
                        "background_removal_requested_quantity", ""
                    ),
                }
                day_seed_rows.append(run_row)
                run_rows_for_candidate.append(run_row)

                asset_thresholds = {
                    "boundary_event_ratio": thresholds.get("maximum_asset_event_ratio"),
                    "boundary_quantity_ratio": thresholds.get("maximum_asset_quantity_ratio"),
                }
                for asset in asset_mappings:
                    symbol = str(asset.get("symbol", ""))
                    for metric, numerator_key, denominator_key in (
                        (
                            "boundary_event_ratio", "boundary_truncation_events",
                            "background_event_count",
                        ),
                        (
                            "boundary_quantity_ratio", "boundary_truncated_quantity",
                            "background_removal_requested_quantity",
                        ),
                    ):
                        ratio = finite_number(asset.get(metric))
                        if ratio is None:
                            continue
                        maximum = finite_number(asset_thresholds[metric])
                        key = (symbol, metric)
                        record = symbol_records.setdefault(key, {
                            "symbol": symbol,
                            "metric": metric,
                            "observation_count": 0,
                            "failure_observation_count": 0,
                            "max_ratio": -math.inf,
                        })
                        record["observation_count"] += 1
                        if maximum is None or ratio > maximum + 1.0e-15:
                            record["failure_observation_count"] += 1
                        if ratio > float(record["max_ratio"]):
                            record.update({
                                "max_ratio": ratio,
                                "maximum_at_max": maximum if maximum is not None else "",
                                "excess_at_max": (
                                    ratio - maximum if maximum is not None else ""
                                ),
                                **identity,
                                "date": trading_date,
                                "seed": seed,
                                "summary_path": summary_path,
                                "numerator_at_max": asset.get(numerator_key, ""),
                                "denominator_at_max": asset.get(denominator_key, ""),
                            })

        event_asset_worst = maximum_record(run_rows_for_candidate, "max_asset_event_ratio")
        quantity_asset_worst = maximum_record(
            run_rows_for_candidate, "max_asset_quantity_ratio"
        )
        event_run_worst = maximum_record(run_rows_for_candidate, "run_event_ratio")
        quantity_run_worst = maximum_record(run_rows_for_candidate, "run_quantity_ratio")
        errors = eligibility.get("errors", evaluation.get("errors", []))
        if not isinstance(errors, list):
            errors = [str(errors)]
        row = {
            **identity,
            "eligible": eligibility.get("eligible") is True,
            "finite_selection_score": predicates.get("finite_selection_score") is True,
            "finite_fit_wsmrmse": predicates.get("finite_fit_wsmrmse") is True,
            "two_sided_integrity_passed": predicates.get(
                "two_sided_integrity_passed"
            ) is True,
            "finite_boundary_adequacy_passed": predicates.get(
                "finite_boundary_adequacy_passed"
            ) is True,
            "error_free": predicates.get("error_free") is True,
            "fit_wsmrmse": evaluation.get("fit_wsmrmse", ""),
            "selection_score": evaluation.get("selection_score", ""),
            "error_count": len(errors),
            "two_sided_failure_count": len(
                evaluation.get("two_sided_integrity_failures", [])
                if isinstance(evaluation.get("two_sided_integrity_failures", []), list)
                else []
            ),
            "boundary_failure_count": len(boundary_failures_for_candidate),
            "asset_boundary_failure_count": sum(
                failure["scope"] == "asset_seed"
                for failure in boundary_failures_for_candidate
            ),
            "run_boundary_failure_count": sum(
                failure["scope"] == "run_aggregate"
                for failure in boundary_failures_for_candidate
            ),
            "failing_dates": ";".join(sorted(failing_dates)),
            "failing_seeds": ";".join(sorted(failing_seeds)),
            "max_asset_event_ratio": (
                event_asset_worst.get("max_asset_event_ratio", "")
                if event_asset_worst else ""
            ),
            "max_asset_quantity_ratio": (
                quantity_asset_worst.get("max_asset_quantity_ratio", "")
                if quantity_asset_worst else ""
            ),
            "max_run_event_ratio": (
                event_run_worst.get("run_event_ratio", "")
                if event_run_worst else ""
            ),
            "max_run_quantity_ratio": (
                quantity_run_worst.get("run_quantity_ratio", "")
                if quantity_run_worst else ""
            ),
            "controls_json": json.dumps(candidate, sort_keys=True, separators=(",", ":")),
            "errors_json": json.dumps(errors, sort_keys=True, separators=(",", ":")),
            "source_path": str(candidate_path),
            "source_sha256": sha256(candidate_path),
        }
        for field in (
            "hawkes_activity_scale", "local_mm_enabled", "local_mm_interval_ms",
            "local_mm_quantity_multiplier", "local_mm_improvement_probability",
            "enabled", "threshold_bps", "depth_participation",
            "order_quantity", "multiplier",
        ):
            row[field] = candidate.get(field, "")
        candidate_rows.append(row)

    symbol_rows = list(symbol_records.values())
    symbol_rows.sort(key=lambda row: (
        str(row["metric"]), -float(row["max_ratio"]), str(row["symbol"]),
    ))
    candidate_rows.sort(key=lambda row: (
        str(row["block"]), str(row["stage"]), str(row["cluster_id"]),
        int(row["candidate_index"]),
    ))
    day_seed_rows.sort(key=lambda row: (
        str(row["block"]), str(row["stage"]), str(row["cluster_id"]),
        int(row["candidate_index"]), str(row["date"]), str(row["seed"]),
    ))
    def failure_sort_key(row: Mapping[str, Any]) -> tuple[str, float, str]:
        ratio = finite_number(row.get("ratio"))
        # Missing/non-finite ratios sort last.  Do not use truthiness here:
        # a legitimate ratio of zero must remain a finite value.
        descending_ratio = -ratio if ratio is not None else math.inf
        return str(row["metric"]), descending_ratio, str(row["symbol"])

    failure_rows.sort(key=failure_sort_key)
    stage_rows.sort(key=lambda row: (
        str(row["block"]), str(row["stage"]), str(row["cluster_id"]),
    ))

    output_paths = {
        "candidate_summary_csv": output_dir / "candidate_summary.csv",
        "day_seed_boundary_csv": output_dir / "day_seed_boundary.csv",
        "boundary_failures_csv": output_dir / "boundary_failures.csv",
        "symbol_worst_csv": output_dir / "symbol_worst.csv",
        "stage_summary_csv": output_dir / "stage_summary.csv",
    }
    atomic_csv(
        output_paths["candidate_summary_csv"], CANDIDATE_FIELDS, candidate_rows,
        overwrite=args.overwrite,
    )
    atomic_csv(
        output_paths["day_seed_boundary_csv"], DAY_SEED_FIELDS, day_seed_rows,
        overwrite=args.overwrite,
    )
    atomic_csv(
        output_paths["boundary_failures_csv"], FAILURE_FIELDS, failure_rows,
        overwrite=args.overwrite,
    )
    atomic_csv(
        output_paths["symbol_worst_csv"], SYMBOL_FIELDS, symbol_rows,
        overwrite=args.overwrite,
    )
    atomic_csv(
        output_paths["stage_summary_csv"], STAGE_FIELDS, stage_rows,
        overwrite=args.overwrite,
    )

    eligibility_counts = Counter(
        "eligible" if row["eligible"] else "ineligible" for row in candidate_rows
    )
    failed_predicate_counts = Counter()
    for row in candidate_rows:
        for predicate in (
            "finite_selection_score", "finite_fit_wsmrmse",
            "two_sided_integrity_passed", "finite_boundary_adequacy_passed",
            "error_free",
        ):
            if row[predicate] is not True:
                failed_predicate_counts[predicate] += 1
    failure_scope_counts = Counter(str(row["scope"]) for row in failure_rows)
    failure_date_counts = Counter(str(row["date"]) for row in failure_rows)
    failure_seed_counts = Counter(str(row["seed"]) for row in failure_rows)
    failure_symbol_counts = Counter(
        str(row["symbol"]) for row in failure_rows if row["symbol"]
    )

    worst_by_metric: dict[str, list[dict[str, Any]]] = {}
    for metric in ("boundary_event_ratio", "boundary_quantity_ratio"):
        rows = [row for row in symbol_rows if row["metric"] == metric]
        worst_by_metric[metric] = rows[:args.top_symbols]

    summary_path = output_dir / "diagnostic_summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": "r4_calibration_failure_diagnostic_summary",
        "calibration_root": str(root),
        "terminal_failure": failure,
        "progress": (
            {
                "path": str(progress_path),
                "status": progress.get("status"),
                "event_count": progress.get("event_count"),
                "last_event": progress.get("last_event"),
            }
        ),
        "integrity": {
            **integrity,
            "discovered_candidate_files": len(candidate_paths),
            "stage_checkpoint_files": len(checkpoint_items),
            "unreferenced_candidate_files": len(
                set(candidate_paths) - referenced_candidates
            ),
            "unreferenced_stage_checkpoint_files": len(
                {path for path, _ in checkpoint_items} - referenced_checkpoints
            ),
            "terminal_failure_progress_checkpoint_verified": (
                failure_progress_verified
            ),
        },
        "counts": {
            "candidates": len(candidate_rows),
            "eligibility": dict(sorted(eligibility_counts.items())),
            "failed_predicates": dict(sorted(failed_predicate_counts.items())),
            "day_seed_runs": len(day_seed_rows),
            "boundary_failures": len(failure_rows),
            "boundary_failures_by_scope": dict(sorted(failure_scope_counts.items())),
            "boundary_failures_by_date": dict(sorted(failure_date_counts.items())),
            "boundary_failures_by_seed": dict(sorted(failure_seed_counts.items())),
            "boundary_failures_by_symbol": dict(
                failure_symbol_counts.most_common(args.top_symbols)
            ),
        },
        "stages": stage_rows,
        "worst_symbols_by_metric": worst_by_metric,
        "artifacts": {},
    }
    for label, path in output_paths.items():
        summary["artifacts"][label] = {
            "path": str(path), "sha256": sha256(path),
        }
    atomic_json(summary_path, summary, overwrite=args.overwrite)
    return {
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "candidate_count": len(candidate_rows),
        "boundary_failure_count": len(failure_rows),
        "stage_checkpoint_count": len(stage_rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = summarize(args)
    except (DiagnosticError, OSError, ValueError, TypeError) as error:
        print(f"calibration diagnostic summarization failed: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
