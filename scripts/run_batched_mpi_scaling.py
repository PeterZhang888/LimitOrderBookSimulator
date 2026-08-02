#!/usr/bin/env python3
"""Run and validate the short batched multi-asset MPI timing matrix."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


RESULT_PREFIX = "batched_mpi_multi_asset "


def parse_result(stdout: str) -> dict[str, str]:
    line = next((row for row in stdout.splitlines() if row.startswith(RESULT_PREFIX)), None)
    if line is None:
        raise RuntimeError(f"benchmark output has no result line:\n{stdout}")
    fields: dict[str, str] = {}
    for match in re.finditer(r"([a-z_]+)=([^ ]+)", line):
        fields[match.group(1)] = match.group(2)
    required = {"ranks", "books", "wall_seconds", "state_hash", "processed_orders"}
    missing = required.difference(fields)
    if missing:
        raise RuntimeError(f"benchmark result is missing {sorted(missing)}")
    return fields


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", default="1,2,4")
    parser.add_argument("--books", type=int, default=101)
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--window-ms", type=float, default=1000.0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20200130)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--oversubscribe", action="store_true")
    args = parser.parse_args()

    ranks = [int(value) for value in args.ranks.split(",")]
    if not ranks or any(value <= 0 for value in ranks):
        raise SystemExit("--ranks must contain positive integers")
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")

    rows: list[dict[str, str]] = []
    expected_hash: str | None = None
    for repetition in range(1, args.repetitions + 1):
        for rank_count in ranks:
            command = [args.mpirun]
            if args.oversubscribe:
                command.append("--oversubscribe")
            command.extend(["--bind-to", "none"])
            command.extend(
                [
                    "-np",
                    str(rank_count),
                    str(args.executable),
                    "--base-config",
                    str(args.base_config),
                    "--books",
                    str(args.books),
                    "--duration-seconds",
                    str(args.duration_seconds),
                    "--window-ms",
                    str(args.window_ms),
                    "--seed",
                    str(args.seed),
                ]
            )
            completed = subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True,
            )
            result = parse_result(completed.stdout)
            if expected_hash is None:
                expected_hash = result["state_hash"]
            elif result["state_hash"] != expected_hash:
                raise RuntimeError(
                    "rank-invariance failure: "
                    f"expected {expected_hash}, observed {result['state_hash']} "
                    f"at ranks={rank_count}, repetition={repetition}"
                )
            result["repetition"] = str(repetition)
            rows.append(result)
            print(
                f"ranks={rank_count} repetition={repetition} "
                f"wall={result['wall_seconds']} hash={result['state_hash']}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
