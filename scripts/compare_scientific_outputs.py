#!/usr/bin/env python3
"""Compare simulator CSV outputs field by field and report the first mismatch."""

import argparse
import csv
import math
from pathlib import Path


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=0.0)
    parser.add_argument("--relative-tolerance", type=float, default=0.0)
    args = parser.parse_args()

    if args.absolute_tolerance < 0.0 or args.relative_tolerance < 0.0:
        raise SystemExit("numeric tolerances must be non-negative")

    reference = rows(args.reference)
    treatment = rows(args.treatment)
    if len(reference) != len(treatment):
        raise SystemExit(
            "row count differs: {} != {}".format(
                len(reference), len(treatment)
            )
        )
    tolerated_cells = 0
    maximum_absolute_difference = 0.0
    for row_number, (left, right) in enumerate(
        zip(reference, treatment), start=1
    ):
        if len(left) != len(right):
            raise SystemExit(
                "column count differs at row {}: {} != {}".format(
                    row_number, len(left), len(right)
                )
            )
        for column_number, (a, b) in enumerate(
            zip(left, right), start=1
        ):
            if a == b:
                continue
            try:
                left_number = float(a)
                right_number = float(b)
            except ValueError:
                left_number = None
                right_number = None
            if (
                left_number is not None
                and math.isfinite(left_number)
                and math.isfinite(right_number)
                and math.isclose(
                    left_number,
                    right_number,
                    rel_tol=args.relative_tolerance,
                    abs_tol=args.absolute_tolerance,
                )
            ):
                tolerated_cells += 1
                maximum_absolute_difference = max(
                    maximum_absolute_difference,
                    abs(left_number - right_number),
                )
                continue
            column_name = "unknown"
            if reference and column_number <= len(reference[0]):
                column_name = reference[0][column_number - 1]
            raise SystemExit(
                "difference at row {}, column {} ({}): {!r} != {!r}".format(
                    row_number, column_number, column_name, a, b
                )
            )
    if tolerated_cells:
        print(
            "equivalent within tolerance: {} and {}; "
            "tolerated_cells={}, maximum_absolute_difference={:.17g}".format(
                args.reference,
                args.treatment,
                tolerated_cells,
                maximum_absolute_difference,
            )
        )
    else:
        print("identical: {} and {}".format(args.reference, args.treatment))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
