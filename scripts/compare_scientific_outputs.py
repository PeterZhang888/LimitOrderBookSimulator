#!/usr/bin/env python3
"""Compare simulator CSV outputs field by field and report the first mismatch."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("treatment", type=Path)
    args = parser.parse_args()

    reference = rows(args.reference)
    treatment = rows(args.treatment)
    if len(reference) != len(treatment):
        raise SystemExit(
            f"row count differs: {len(reference)} != {len(treatment)}")
    for row_number, (left, right) in enumerate(
        zip(reference, treatment, strict=True), start=1
    ):
        if len(left) != len(right):
            raise SystemExit(
                f"column count differs at row {row_number}: "
                f"{len(left)} != {len(right)}"
            )
        for column_number, (a, b) in enumerate(
            zip(left, right, strict=True), start=1
        ):
            if a != b:
                raise SystemExit(
                    f"difference at row {row_number}, column {column_number}: "
                    f"{a!r} != {b!r}"
                )
    print(f"identical: {args.reference} and {args.treatment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
