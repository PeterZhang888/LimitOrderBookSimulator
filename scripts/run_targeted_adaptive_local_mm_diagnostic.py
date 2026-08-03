#!/usr/bin/env python3
"""Re-evaluate the two predeclared cluster-2 finalists with adaptive repair.

This is a training-only structural diagnostic.  It reuses the exact dated
configs, background policies, value policies and deterministic seeds recorded
by the earlier full-day confirmations.  The only changed controls are the two new
bounded local-MM spread-response parameters.  No held-out input is accepted or
read.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import calibrate_queue_reactive_model as calibration  # noqa: E402


BASE_CANDIDATES = (
    "value_value_5bps_20pct__vol_latent_scale_400_tail025__rechecks_2",
    "value_value_20bps_20pct__vol_latent_scale_400_tail025__rechecks_4",
)
ADAPTIVE_VARIANTS = (
    ("eta010_cap050", 0.1, 0.50),
    ("eta010_cap060", 0.1, 0.60),
    ("eta010_cap075", 0.1, 0.75),
    ("eta020_cap050", 0.2, 0.50),
    ("eta020_cap060", 0.2, 0.60),
    ("eta020_cap075", 0.2, 0.75),
    ("eta030_cap050", 0.3, 0.50),
    ("eta030_cap060", 0.3, 0.60),
    ("eta030_cap075", 0.3, 0.75),
    ("eta040_cap050", 0.4, 0.50),
    ("eta040_cap060", 0.4, 0.60),
    ("eta040_cap075", 0.4, 0.75),
    ("eta050_cap075", 0.5, 0.75),
    ("eta050_cap100", 0.5, 1.0),
    ("eta100_cap075", 1.0, 0.75),
    ("eta100_cap100", 1.0, 1.0),
)
FINE_VARIANTS = tuple(
    (
        f"eta{int(round(elasticity * 100)):03d}_cap{int(round(cap * 100)):03d}",
        elasticity,
        cap,
    )
    for elasticity in (0.35, 0.40, 0.45)
    for cap in (0.55, 0.60, 0.65)
)


def replace_option(command: list[str], option: str, value: str) -> None:
    if option in command:
        index = command.index(option)
        if index + 1 >= len(command):
            raise RuntimeError(f"recorded command has no value for {option}")
        command[index + 1] = value
        return
    insertion = command.index("--disable-shared-mm")
    command[insertion:insertion] = [option, value]


def load_confirmation_records(path: pathlib.Path) -> dict[str, Mapping[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("confirmation_candidates")
    if not isinstance(records, list):
        raise RuntimeError("cluster result lacks confirmation_candidates")
    by_id = {
        str(record.get("candidate_id")): record
        for record in records
        if isinstance(record, Mapping)
    }
    missing = [identifier for identifier in BASE_CANDIDATES if identifier not in by_id]
    if missing:
        raise RuntimeError(f"cluster result lacks required finalists: {missing}")
    return by_id


def run_diagnostic(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    executable = args.executable.resolve()
    if not executable.is_file():
        raise RuntimeError(f"simulator executable is missing: {executable}")

    configs = {
        day: calibration.read_config(path.resolve())
        for day, path in args.training_config
    }
    if len(configs) != 5:
        raise RuntimeError("exactly five unique training configs are required")

    source = load_confirmation_records(args.cluster_result.resolve())
    results: list[dict[str, object]] = []
    selected_base_candidates = tuple(args.base_candidate or BASE_CANDIDATES)
    selected_variants = FINE_VARIANTS if args.variant_set == "fine" else ADAPTIVE_VARIANTS
    for base_identifier in selected_base_candidates:
        base = source[base_identifier]
        original_runs = base.get("runs")
        if not isinstance(original_runs, Mapping):
            raise RuntimeError(f"{base_identifier} lacks recorded runs")
        for suffix, elasticity, cap in selected_variants:
            candidate_id = f"{base_identifier}__local_{suffix}"
            candidate_root = output_root / candidate_id
            run_records: dict[str, list[dict[str, object]]] = {}
            cluster_symbols: tuple[str, ...] | None = None
            for day in sorted(configs):
                source_runs = original_runs.get(day)
                if not isinstance(source_runs, list) or len(source_runs) != 3:
                    raise RuntimeError(
                        f"{base_identifier}:{day} must have exactly three source seeds"
                    )
                day_records: list[dict[str, object]] = []
                for source_run in source_runs:
                    if not isinstance(source_run, Mapping):
                        raise RuntimeError("recorded run is malformed")
                    command_value = source_run.get("command")
                    if not isinstance(command_value, list):
                        raise RuntimeError("recorded run lacks command")
                    command = [str(token) for token in command_value]
                    command[0] = str(executable)
                    replace_option(
                        command, "--local-mm-spread-elasticity", format(elasticity, ".17g")
                    )
                    replace_option(
                        command,
                        "--local-mm-max-improvement-probability",
                        format(cap, ".17g"),
                    )
                    seed = int(source_run["base_seed"])
                    run_dir = candidate_root / f"day_{day.replace('-', '')}" / f"base_seed_{seed}"
                    summary = run_dir / "fragmented_asset_summary.csv"
                    replace_option(command, "--asset-summary-csv", str(summary))
                    run_dir.mkdir(parents=True, exist_ok=False)
                    started = time.monotonic()
                    completed = subprocess.run(
                        command,
                        cwd=str(SCRIPT_DIR.parent),
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                    record = {
                        "base_seed": seed,
                        "session_date": day,
                        "command": command,
                        "return_code": completed.returncode,
                        "success": completed.returncode == 0 and summary.is_file(),
                        "summary_path": str(summary.resolve()),
                        "wall_seconds": time.monotonic() - started,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    }
                    calibration.write_json(run_dir / "run.json", record)
                    day_records.append(record)
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f"{candidate_id}:{day}:seed {seed} failed: {completed.stderr}"
                        )
                    universe_index = command.index("--universe-config") + 1
                    symbols = calibration.read_config(
                        pathlib.Path(command[universe_index])
                    ).symbols
                    if cluster_symbols is None:
                        cluster_symbols = symbols
                    elif symbols != cluster_symbols:
                        raise RuntimeError("cluster symbol order changed across dates")
                run_records[day] = day_records
            if cluster_symbols is None:
                raise RuntimeError("no cluster symbols were observed")
            score = calibration.score_runs(
                run_records_by_day=run_records,
                configs=configs,
                symbols_by_day={day: cluster_symbols for day in configs},
                duration=calibration.STAGE3_DURATION,
                metrics=calibration.STAGE2_METRICS,
            )
            candidate = {
                "candidate_id": candidate_id,
                "base_candidate_id": base_identifier,
                "local_mm_spread_elasticity": elasticity,
                "local_mm_max_improvement_probability": cap,
                "runs": run_records,
                "score": score,
            }
            candidate["strict_cluster_metric_gate"] = (
                calibration.cluster_confirmation_gate_audit(candidate)
            )
            calibration.write_json(candidate_root / "result.json", candidate)
            results.append(candidate)
            gate = candidate["strict_cluster_metric_gate"]
            print(
                candidate_id,
                "PASS" if gate["passed"] else "FAIL",
                "max_score=",
                max(
                    float(metric["score"])
                    for day_result in gate["date_results"]
                    for metric in day_result["metric_results"]
                ),
                flush=True,
            )
    payload = {
        "schema_version": 1,
        "role": "training_only_targeted_structural_diagnostic",
        "heldout_inputs_read": False,
        "base_candidates": list(selected_base_candidates),
        "adaptive_variants": [
            {"id": item[0], "spread_elasticity": item[1], "cap": item[2]}
            for item in selected_variants
        ],
        "results": results,
    }
    calibration.write_json(output_root / "diagnostic_result.json", payload)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=pathlib.Path, required=True)
    parser.add_argument("--cluster-result", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--base-candidate",
        action="append",
        choices=BASE_CANDIDATES,
        default=[],
        help="Restrict the diagnostic to one recorded finalist.",
    )
    parser.add_argument(
        "--variant-set", choices=("broad", "fine"), default="broad"
    )
    parser.add_argument(
        "--training-config",
        action="append",
        default=[],
        metavar="DATE=PATH",
        help="Repeat exactly five times; training dates only.",
    )
    args = parser.parse_args(argv)
    parsed: list[tuple[str, pathlib.Path]] = []
    for value in args.training_config:
        item = calibration.parse_dated_path(value, option="--training-config")
        parsed.append((item.day, item.path))
    if len(parsed) != 5 or len({day for day, _ in parsed}) != 5:
        parser.error("--training-config requires exactly five unique dates")
    args.training_config = parsed
    try:
        return run_diagnostic(args)
    except (OSError, ValueError, RuntimeError, calibration.CalibrationDriverError) as error:
        print(f"adaptive local-MM diagnostic failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
