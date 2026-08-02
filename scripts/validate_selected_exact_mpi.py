#!/usr/bin/env python3
"""Validate a selected coupled configuration against exact MPI rank counts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import subprocess
import time


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def normalized_summary(path: pathlib.Path) -> list[dict[str, str]]:
    ignored = {"owner_rank", "mpi_ranks", "wall_seconds"}
    return [
        {key: value for key, value in row.items() if key not in ignored}
        for row in read_csv(path)
    ]


def run(command: list[str], output_dir: pathlib.Path,
        timeout: float) -> float:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False, timeout=timeout,
    )
    elapsed = time.monotonic() - started
    with (output_dir / "run.log").open("w") as output:
        output.write("command=" + json.dumps(command) + "\n")
        output.write(f"external_wall_seconds={elapsed:.9f}\n")
        output.write(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with status {completed.returncode}; "
            f"see {output_dir / 'run.log'}"
        )
    return elapsed


def model_arguments(report: dict[str, object]) -> list[str]:
    selected = report["selected_parameters"]
    protocol = report["protocol"]
    assert isinstance(selected, dict) and isinstance(protocol, dict)
    fixed = protocol["fixed_value_parameters"]
    coupling = protocol["coupling_parameters"]
    assert isinstance(fixed, dict) and isinstance(coupling, dict)
    arguments = [
        "--enable-value-agent",
        "--value-threshold-bps", str(selected["threshold_bps"]),
        "--value-response-bps", str(selected["response_step_bps"]),
        "--value-base-quantity", str(selected["base_order_quantity"]),
        "--value-max-quantity", str(fixed["max_order_quantity"]),
        "--value-max-inventory", str(fixed["max_inventory"]),
        "--value-fundamental-volatility-bps",
        str(selected["volatility_bps_sqrt_second"]),
        "--value-interval-ms", str(fixed["decision_interval_ms"]),
        "--enable-etf-arbitrage",
        "--arbitrage-trigger-bps", str(coupling["arbitrage_trigger_bps"]),
        "--arbitrage-release-bps", str(coupling["arbitrage_release_bps"]),
        "--exposure-threshold", str(coupling["shared_mm_exposure_threshold"]),
        "--max-hedge-quantity", str(coupling["max_hedge_quantity"]),
    ]
    if protocol.get("shared_mm_cross_book_hedging"):
        arguments.append("--enable-shared-mm-hedging")
    return arguments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sequential-binary", default="build/sequential_multi_asset_lob")
    parser.add_argument("--exact-binary", default="build/exact_mpi_multi_asset_lob")
    parser.add_argument("--mpi-launcher", default="mpirun")
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output-dir", default="results/selected_exact_validation")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    if args.duration_seconds <= 0 or not args.ranks or min(args.ranks) <= 0:
        parser.error("duration and every rank count must be positive")

    report_path = pathlib.Path(args.report).resolve()
    config_path = pathlib.Path(args.config).resolve()
    sequential_binary = pathlib.Path(args.sequential_binary).resolve()
    exact_binary = pathlib.Path(args.exact_binary).resolve()
    output_root = pathlib.Path(args.output_dir).resolve()
    with report_path.open() as source:
        calibration = json.load(source)
    common = [
        "--duration-seconds", str(args.duration_seconds),
        "--seed", str(args.seed),
        "--book-config-file", str(config_path),
        *model_arguments(calibration),
    ]

    reference_dir = output_root / "sequential"
    reference_elapsed = run(
        [str(sequential_binary), *common, "--output-dir", str(reference_dir)],
        reference_dir, args.timeout_seconds,
    )
    reference_summary_path = reference_dir / "sequential_multi_asset_summary.csv"
    reference_trace_path = reference_dir / "sequential_multi_asset_state_trace.csv"
    reference_summary = normalized_summary(reference_summary_path)
    rank_results: list[dict[str, object]] = []
    for ranks in args.ranks:
        rank_dir = output_root / f"exact_mpi_{ranks}"
        command = [
            args.mpi_launcher, "--bind-to", "none", "--map-by",
            "slot:OVERSUBSCRIBE", "-n", str(ranks), str(exact_binary),
            *common, "--output-dir", str(rank_dir),
        ]
        elapsed = run(command, rank_dir, args.timeout_seconds)
        summary_path = rank_dir / "exact_mpi_multi_asset_summary.csv"
        trace_path = rank_dir / "exact_mpi_multi_asset_state_trace.csv"
        if normalized_summary(summary_path) != reference_summary:
            raise RuntimeError(f"{ranks}-rank summary differs from sequential reference")
        reference_bytes = reference_trace_path.read_bytes()
        if trace_path.read_bytes() != reference_bytes:
            raise RuntimeError(f"{ranks}-rank state trace differs from sequential reference")
        rank_results.append({
            "ranks": ranks,
            "external_wall_seconds": elapsed,
            "summary_sha256": sha256_file(summary_path),
            "state_trace_sha256": sha256_file(trace_path),
            "exact_equal": True,
        })

    evidence = {
        "configuration": str(config_path),
        "configuration_sha256": sha256_file(config_path),
        "calibration_report": str(report_path),
        "calibration_report_sha256": sha256_file(report_path),
        "duration_seconds": args.duration_seconds,
        "seed": args.seed,
        "sequential_external_wall_seconds": reference_elapsed,
        "sequential_summary_sha256": sha256_file(reference_summary_path),
        "sequential_state_trace_sha256": sha256_file(reference_trace_path),
        "rank_results": rank_results,
        "all_exact_equal": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_path = output_root / "selected_exact_mpi_validation.json"
    with evidence_path.open("w") as output:
        json.dump(evidence, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps({"all_exact_equal": True, "evidence": str(evidence_path)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
