#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Audit and select a passed R32 directional activity-regime pilot.

Only candidates that produced the driver's immutable passed handoff are
eligible.  Among multiple passed candidates, the script chooses the smallest
maximum pilot-threshold-normalized ACF error, then the smallest mean normalized
error, then the smaller activity scale.  Rejected candidates remain reported
as scientific outcomes; missing or inconsistent artifacts are hard errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from typing import Any


class GridAuditError(RuntimeError):
    """Raised when a candidate directory is incomplete or inconsistent."""


def finite_float(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise GridAuditError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise GridAuditError(f"{label} is not finite")
    return result


def read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        raise GridAuditError(f"missing JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GridAuditError(f"JSON artifact is not an object: {path}")
    return payload


def parse_candidate(value: str) -> tuple[float, pathlib.Path]:
    scale_text, separator, path_text = value.partition("=")
    if not separator or not path_text:
        raise argparse.ArgumentTypeError("candidate must be SCALE=/absolute/result/root")
    try:
        scale = float(scale_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid scale: {scale_text}") from error
    if not math.isfinite(scale) or not 0.5 <= scale <= 1.25:
        raise argparse.ArgumentTypeError("scale must be in [0.5,1.25]")
    path = pathlib.Path(path_text).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("candidate result root must be absolute")
    return scale, path.resolve()


def audit_candidate(scale: float, result_root: pathlib.Path) -> dict[str, Any]:
    adequacy = result_root / "full_training_adequacy"
    pilot = adequacy / "directional_pilot"
    decision = read_json(pilot / "pilot_decision.json")
    status = decision.get("status")
    if status not in {"passed", "rejected"}:
        raise GridAuditError(f"unexpected pilot status {status!r} in {result_root}")
    if decision.get("training_only") is not True:
        raise GridAuditError(f"pilot is not labelled training-only: {result_root}")

    expansion = read_json(adequacy / "full_universe_expansion_inputs.json")
    warm_start = expansion.get("training_refinement_warm_start")
    if not isinstance(warm_start, dict):
        raise GridAuditError(f"warm-start record is absent: {result_root}")
    scale_record = warm_start.get("coupling_scale")
    if not isinstance(scale_record, dict):
        raise GridAuditError(f"activity-scale record is absent: {result_root}")
    recorded_scale = finite_float(
        scale_record.get("global_coupling_scale"), label="recorded activity scale",
    )
    if not math.isclose(recorded_scale, scale, rel_tol=0.0, abs_tol=1.0e-12):
        raise GridAuditError(
            f"candidate label {scale:g} differs from recorded scale "
            f"{recorded_scale:g}: {result_root}"
        )

    thresholds = decision.get("thresholds", {}).get("maximum_acf_absolute_error")
    if not isinstance(thresholds, dict):
        raise GridAuditError(f"pilot ACF thresholds are absent: {result_root}")
    acf_path = pilot / "strict_diagnostics" / "absolute_return_acf_distribution.csv"
    if not acf_path.is_file():
        raise GridAuditError(f"missing ACF diagnostics: {acf_path}")
    normalized: list[float] = []
    raw_errors: list[float] = []
    with acf_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            statistic = str(row.get("statistic", ""))
            if statistic not in {"mean", "median", "p90"}:
                continue
            error = finite_float(row.get("absolute_error"), label="ACF absolute error")
            threshold = finite_float(
                thresholds.get(statistic), label=f"{statistic} pilot threshold",
            )
            if threshold <= 0.0:
                raise GridAuditError(f"nonpositive {statistic} pilot threshold")
            raw_errors.append(error)
            normalized.append(error / threshold)
    if len(normalized) != 9:
        raise GridAuditError(
            f"expected nine three-date ACF diagnostics, found {len(normalized)}: "
            f"{acf_path}"
        )

    handoff = pilot / "directional_pilot_handoff.json"
    eligible = status == "passed" and handoff.is_file()
    if (status == "passed") != handoff.is_file():
        raise GridAuditError(
            f"pilot status/handoff disagreement for scale {scale:g}: {result_root}"
        )
    return {
        "activity_scale": scale,
        "result_root": str(result_root),
        "status": status,
        "eligible_for_full_matrix": eligible,
        "maximum_threshold_normalized_acf_error": max(normalized),
        "mean_threshold_normalized_acf_error": sum(normalized) / len(normalized),
        "maximum_raw_acf_error": max(raw_errors),
        "failure_count": len(decision.get("failures", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", required=True, type=parse_candidate,
        metavar="SCALE=/ABSOLUTE/RESULT_ROOT",
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    scales = [scale for scale, _ in args.candidate]
    if len(scales) != len(set(scales)):
        raise GridAuditError("activity scales must be unique")
    records = [audit_candidate(scale, path) for scale, path in args.candidate]
    eligible = [record for record in records if record["eligible_for_full_matrix"]]
    selected = min(
        eligible,
        key=lambda record: (
            record["maximum_threshold_normalized_acf_error"],
            record["mean_threshold_normalized_acf_error"],
            record["activity_scale"],
        ),
        default=None,
    )
    payload = {
        "schema_version": 1,
        "status": "passed_candidate_selected" if selected else "no_pilot_passed",
        "selection_rule": (
            "passed handoff required; minimize maximum pilot-threshold-normalized "
            "ACF error, then mean normalized error, then activity scale"
        ),
        "candidates": sorted(records, key=lambda record: record["activity_scale"]),
        "selected": selected,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if selected else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GridAuditError, json.JSONDecodeError, OSError) as error:
        print(f"R32 pilot-grid audit failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
