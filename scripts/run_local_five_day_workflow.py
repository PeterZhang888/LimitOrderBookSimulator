#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Run the fixed five-session training and 2020 held-out workflow locally.

The program waits for exact-size public archives in ``--download-dir``, builds
each resumable empirical universe, then runs pooling, ten-cluster calibration,
stratified validation, and full-market held-out validation.  Raw ITCH files
remain outside the project and are never duplicated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from typing import Sequence


TRAINING = (
    ("2019-01-30", "01302019.NASDAQ_ITCH50.gz", 4_764_426_091),
    ("2019-03-27", "03272019.NASDAQ_ITCH50.gz", 5_510_131_732),
    ("2019-07-30", "07302019.NASDAQ_ITCH50.gz", 3_662_140_094),
    ("2019-10-30", "10302019.NASDAQ_ITCH50.gz", 3_872_931_242),
    ("2019-12-30", "12302019.NASDAQ_ITCH50.gz", 3_524_013_057),
)
HELDOUT = ("2020-01-30", "01302020.NASDAQ_ITCH50.gz", 5_597_158_940)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: Sequence[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(list(command), check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command returned {completed.returncode}: {' '.join(command)}"
        )


def wait_for_archive(path: pathlib.Path, expected_size: int,
                     poll_seconds: float, no_wait: bool) -> None:
    last_observed: int | None = None
    while True:
        observed = path.stat().st_size if path.is_file() else 0
        if observed == expected_size:
            print(f"archive ready: {path} ({observed} bytes)", flush=True)
            return
        if observed > expected_size:
            raise RuntimeError(
                f"archive is larger than the official size: {path} "
                f"({observed} > {expected_size})"
            )
        if no_wait:
            raise FileNotFoundError(
                f"archive is incomplete: {path} ({observed}/{expected_size} bytes)"
            )
        if observed != last_observed:
            print(
                f"waiting for archive: {path.name} "
                f"({observed}/{expected_size} bytes)",
                flush=True,
            )
            last_observed = observed
        time.sleep(poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=str(pathlib.Path(__file__).resolve().parents[1]))
    parser.add_argument("--download-dir", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument(
        "--fixed-symbols",
        help=(
            "newline-delimited predeclared universe supplied identically to all "
            "six startup-eligibility checks"
        ),
    )
    parser.add_argument("--symbols-per-batch", type=int, default=256)
    parser.add_argument("--concurrent-batches", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--skip-marketwide-validation", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project = pathlib.Path(args.project_dir).expanduser().resolve()
    downloads = pathlib.Path(args.download_dir).expanduser().resolve()
    work = pathlib.Path(args.work_root).expanduser().resolve()
    binary = pathlib.Path(args.binary).expanduser().resolve()
    fixed_symbols = (
        None if args.fixed_symbols is None
        else pathlib.Path(args.fixed_symbols).expanduser().resolve()
    )
    if not binary.is_file():
        raise SystemExit(f"simulator binary is missing: {binary}")
    if fixed_symbols is not None and not fixed_symbols.is_file():
        raise SystemExit(f"fixed-symbol file is missing: {fixed_symbols}")
    fixed_symbols_sha256 = (
        None if fixed_symbols is None else sha256_file(fixed_symbols)
    )
    if args.symbols_per_batch <= 0 or args.concurrent_batches <= 0:
        raise SystemExit("batch controls must be positive")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    work.mkdir(parents=True, exist_ok=True)
    extractor = project / "scripts" / "run_local_itch_universe.py"
    calibrator = project / "scripts" / "run_local_multiday_calibration.py"

    # The held-out archive is normally already present.  Re-running its driver
    # is cheap because a completed universe exits after validating its markers.
    all_days = (*TRAINING, HELDOUT)
    extracted_roots: dict[str, pathlib.Path] = {}
    for trading_date, filename, expected_size in all_days:
        if (
            fixed_symbols is not None
            and sha256_file(fixed_symbols) != fixed_symbols_sha256
        ):
            raise RuntimeError(
                "fixed-symbol declaration changed during the six-day workflow"
            )
        archive = downloads / filename
        wait_for_archive(archive, expected_size, args.poll_seconds, args.no_wait)
        result_root = work / f"itch_{trading_date.replace('-', '')}"
        extraction_command = [
            sys.executable, str(extractor),
            "--project-dir", str(project), "--itch", str(archive),
            "--date", trading_date, "--result-root", str(result_root),
            "--symbols-per-batch", str(args.symbols_per_batch),
            "--concurrent-batches", str(args.concurrent_batches),
            "--snapshot-ms", "1000", "--target-window-seconds", "300", "3600",
            "--progress-seconds", "60",
        ]
        if fixed_symbols is not None:
            extraction_command.extend(["--fixed-symbols", str(fixed_symbols)])
        run(extraction_command)
        if (
            fixed_symbols is not None
            and sha256_file(fixed_symbols) != fixed_symbols_sha256
        ):
            raise RuntimeError(
                "fixed-symbol declaration changed during the six-day workflow"
            )
        extracted_roots[trading_date] = result_root

    calibration_root = work / "five_day_calibration"
    command = [
        sys.executable, str(calibrator), "--project-dir", str(project),
        "--binary", str(binary),
    ]
    for trading_date, _, _ in TRAINING:
        command.extend([
            "--training-day", trading_date, str(extracted_roots[trading_date]),
        ])
    command.extend([
        "--heldout", HELDOUT[0], str(extracted_roots[HELDOUT[0]]),
        "--output-root", str(calibration_root),
        "--clusters", "10", "--validation-per-cluster", "3",
    ])
    if args.skip_marketwide_validation:
        command.append("--skip-marketwide-validation")
    run(command)
    print(json.dumps({
        "status": "complete",
        "work_root": str(work),
        "calibration_root": str(calibration_root),
        "marketwide_validation": not args.skip_marketwide_validation,
        "fixed_symbols": (
            None if fixed_symbols is None else {
                "path": str(fixed_symbols),
                "raw_sha256": fixed_symbols_sha256,
            }
        ),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
