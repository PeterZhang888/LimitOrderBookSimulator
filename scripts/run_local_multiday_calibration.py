#!/usr/bin/env python3
"""Pool extracted ITCH days, cluster the common universe, fit, and validate.

The raw archives are not read here.  This workstation stage consumes only the
derived outputs of ``run_local_itch_universe.py`` and invokes the rank-one
simulator directly, so it does not depend on a Slurm allocation or ``mpirun``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from datetime import date
from typing import Sequence


def compact(value: str) -> str:
    try:
        return date.fromisoformat(value).strftime("%Y%m%d")
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid ISO date {value!r}") from error


def completed_universe(root: pathlib.Path, trading_date: str) -> tuple[pathlib.Path, pathlib.Path]:
    compact_date = compact(trading_date)
    config = root / f"nasdaq_common_plus_qqq_{compact_date}.csv"
    targets = root / "empirical_data"
    # ``calibration_job_metadata.json`` is launcher provenance produced by
    # ``run_local_itch_universe.py``.  It is not an empirical input to either
    # pooling or calibration, and compact transferred bundles intentionally
    # omit it.  The downstream pooler and calibrator hash and validate the
    # configuration, marks, manifests, targets, and generated rates directly.
    if not config.is_file():
        raise FileNotFoundError(f"incomplete extracted universe: {config}")
    if not targets.is_dir():
        raise FileNotFoundError(f"incomplete extracted target root: {targets}")
    return config.resolve(), targets.resolve()


def run(command: Sequence[str], log: pathlib.Path) -> None:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(
        list(command), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command returned {completed.returncode}; full output: {log}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=str(pathlib.Path(__file__).resolve().parents[1]))
    parser.add_argument("--binary", required=True)
    parser.add_argument(
        "--build-provenance",
        help=("calibration_build_provenance.json for the exact Release binary; "
              "required with --require-certification-profile"),
    )
    parser.add_argument(
        "--require-certification-profile", action="store_true",
        help=("enforce the complete immutable thesis calibration and validation "
              "contract rather than producing a preliminary diagnostic"),
    )
    parser.add_argument(
        "--training-day", action="append", nargs=2, required=True,
        metavar=("DATE", "EXTRACTED_RESULT_ROOT"),
    )
    parser.add_argument(
        "--heldout", nargs=2, required=True,
        metavar=("DATE", "EXTRACTED_RESULT_ROOT"),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--validation-per-cluster", type=int, default=3)
    parser.add_argument("--cluster-seed", type=int, default=20200130)
    parser.add_argument("--stage1-duration", type=int, default=300)
    parser.add_argument("--stage2-duration", type=int, default=3600)
    parser.add_argument("--stage3-duration", type=int, default=23400)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--skip-marketwide-validation", action="store_true",
        help=("omit the required full-universe check; the result will be "
              "preliminary and cannot be a normal case-study handoff"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project = pathlib.Path(args.project_dir).expanduser().resolve()
    scripts = project / "scripts"
    binary = pathlib.Path(args.binary).expanduser().resolve()
    output = pathlib.Path(args.output_root).expanduser().resolve()
    if not binary.is_file():
        raise SystemExit(f"rank-one simulator is missing: {binary}")
    build_provenance = (
        pathlib.Path(args.build_provenance).expanduser().resolve()
        if args.build_provenance else None
    )
    if build_provenance is not None and not build_provenance.is_file():
        raise SystemExit(f"build provenance is missing: {build_provenance}")
    if args.require_certification_profile and build_provenance is None:
        raise SystemExit(
            "--build-provenance is required with --require-certification-profile"
        )
    if len(args.training_day) < 2:
        raise SystemExit("at least two --training-day inputs are required")
    training: list[tuple[str, pathlib.Path, pathlib.Path, pathlib.Path]] = []
    for raw_date, raw_root in args.training_day:
        compact(raw_date)
        root = pathlib.Path(raw_root).expanduser().resolve()
        config, targets = completed_universe(root, raw_date)
        training.append((raw_date, root, config, targets))
    training.sort(key=lambda item: item[0])
    if len({item[0] for item in training}) != len(training):
        raise SystemExit("duplicate --training-day dates")
    heldout_date, heldout_raw_root = args.heldout
    compact(heldout_date)
    heldout_root = pathlib.Path(heldout_raw_root).expanduser().resolve()
    heldout_config, heldout_targets = completed_universe(heldout_root, heldout_date)
    if training[-1][0] >= heldout_date:
        raise SystemExit("every training date must precede the held-out date")
    for name in ("clusters", "validation_per_cluster", "stage1_duration",
                 "stage2_duration", "stage3_duration"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not args.stage1_duration < args.stage2_duration < args.stage3_duration:
        raise SystemExit("require stage1-duration < stage2-duration < stage3-duration")

    output.mkdir(parents=True, exist_ok=True)
    pool_root = output / "pooled_training"
    pool_provenance = pool_root / "pooling_provenance.json"
    if not pool_provenance.is_file():
        command = [
            sys.executable, str(scripts / "pool_multiday_empirical_universe.py"),
        ]
        for training_date, _, config, targets in training:
            command.extend(["--training-day", training_date, str(config)])
            command.extend(["--training-target-root", training_date, str(targets)])
        command.extend([
            "--heldout-date", heldout_date,
            "--heldout-config", str(heldout_config),
            "--heldout-target-root", str(heldout_targets),
            "--output-root", str(pool_root),
            "--label", "five_2019_sessions",
            "--minimum-symbols", str(2 * args.clusters),
            "--activity-scale", "0.30", "--hawkes-beta", "10.0",
            "--balance-strength", "1.0",
            "--balance-directional-volume", "--balance-best-depth",
            "--quote-quantity-fraction", "0.5",
            "--minimum-quote-quantity", "10", "--maximum-quote-quantity", "1000",
            "--overwrite",
        ])
        if args.require_certification_profile:
            command.append("--require-certification-cohort")
        run(command, output / "pooling.log")

    pooled_config = pool_root / "pooled_training_universe.csv"
    heldout_common = pool_root / "heldout_common.csv"
    cluster_root = output / "liquidity_clusters"
    cluster_manifest = cluster_root / "cluster_manifest.json"
    if not cluster_manifest.is_file():
        run([
            sys.executable, str(scripts / "cluster_empirical_universe.py"),
            "--universe-config", str(pooled_config),
            "--output-dir", str(cluster_root),
            "--clusters", str(args.clusters),
            "--validation-per-cluster", str(args.validation_per_cluster),
            "--minimum-cluster-size", "6",
            "--seed", str(args.cluster_seed), "--overwrite",
        ], output / "clustering.log")

    calibration_root = output / "calibration"
    handoff = calibration_root / "calibration_handoff.json"
    preliminary = calibration_root / "preliminary_calibration_result.json"
    if not handoff.is_file() and not preliminary.is_file():
        command = [
            sys.executable, str(scripts / "calibrate_cluster_value_agents.py"),
            "--binary", str(binary),
            "--pooled-training-universe-config", str(pooled_config),
        ]
        if build_provenance is not None:
            command.extend(["--build-provenance", str(build_provenance)])
        if args.require_certification_profile:
            command.append("--require-certification-profile")
        for training_date, _, _, targets in training:
            common_config = pool_root / "training_days" / training_date / "universe_common.csv"
            command.extend([
                "--training-day", training_date, str(common_config), str(targets),
            ])
        command.extend([
            "--heldout-universe-config", str(heldout_common),
            "--cluster-assignments", str(cluster_root / "cluster_assignments.csv"),
            "--validation-sample", str(cluster_root / "validation_sample.csv"),
            "--cluster-manifest", str(cluster_manifest),
            "--pooling-provenance", str(pool_provenance),
            "--pooling-producer-project-root", str(project),
            "--heldout-date", heldout_date,
            "--heldout-target-root", str(heldout_targets),
            "--output-dir", str(calibration_root),
            "--stage1-duration", str(args.stage1_duration),
            "--stage2-duration", str(args.stage2_duration),
            "--stage3-duration", str(args.stage3_duration),
            "--session-duration", str(args.stage3_duration),
            "--stage1-top-candidates", "6",
            "--stage1-refinement-candidates", "32",
            "--stage2-top-candidates", "2",
            "--stage1-seeds", "1729",
            "--stage2-seeds", "1729", "7919",
            "--stage3-seeds", "1729", "7919", "1103", "6599", "2027",
            "--thresholds", "5", "8", "10", "15", "25", "40",
            "--depth-participations", "0.05", "0.1", "0.25", "0.5",
            "--hawkes-activity-scales", "0.30",
            "--local-mm-intervals-ms", "500", "1000", "2000",
            "--local-mm-quantity-multipliers", "0.5", "1.0", "2.0",
            "--local-mm-improvement-probabilities", "0", "0.25", "0.5", "1.0",
            "--shared-quote-multipliers", "0.5", "1.0", "2.0",
            "--shared-treatment-multiplier", "1.0",
            "--timeout-seconds", str(args.timeout_seconds), "--overwrite",
        ])
        if not args.skip_marketwide_validation:
            command.append("--marketwide-validation")
        run(command, output / "calibration.log")

    report = calibration_root / "cluster_value_agent_calibration_report.json"
    certified = handoff.is_file()
    result = {
        "status": "certified" if certified else "preliminary_not_certified",
        "training_dates": [item[0] for item in training],
        "heldout_date": heldout_date,
        "pooling_provenance": str(pool_provenance),
        "cluster_manifest": str(cluster_manifest),
        "calibration_report": str(report),
        "calibration_handoff": str(handoff) if certified else None,
        "preliminary_calibration_result": (
            None if certified else str(preliminary)
        ),
        "marketwide_validation": not args.skip_marketwide_validation,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
