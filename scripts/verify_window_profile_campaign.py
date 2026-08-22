#!/usr/bin/env python3
"""Apply the established scientific and placement gates to a profile run."""

import argparse
import pathlib

import summarize_layout_pair as layout_gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=pathlib.Path)
    parser.add_argument("control")
    parser.add_argument("treatment")
    parser.add_argument("control_ranks", type=int)
    parser.add_argument("control_threads", type=int)
    parser.add_argument("treatment_ranks", type=int)
    parser.add_argument("treatment_threads", type=int)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--expected-assets", type=int, required=True)
    parser.add_argument(
        "--expected-duration-seconds", type=int, required=True
    )
    args = parser.parse_args()
    if args.blocks < 1:
        raise SystemExit("--blocks must be positive")

    layout_gate.BLOCKS = tuple(range(1, args.blocks + 1))
    layouts = {
        args.control: (args.control_ranks, args.control_threads),
        args.treatment: (args.treatment_ranks, args.treatment_threads),
    }
    layout_gate.require_complete_order(
        args.result_root, args.control, args.treatment
    )
    rows = layout_gate.load_rows(args.result_root, layouts)
    for variant in (args.control, args.treatment):
        for block in layout_gate.BLOCKS:
            fields = rows[variant][block]
            for field in ("assets", "lobs"):
                if fields.get(field) != str(args.expected_assets):
                    raise SystemExit(
                        "{} block {} recorded {}={}, expected {}".format(
                            variant,
                            block,
                            field,
                            fields.get(field),
                            args.expected_assets,
                        )
                    )
            if fields.get("simulated_seconds") != str(
                args.expected_duration_seconds
            ):
                raise SystemExit(
                    "{} block {} recorded simulated_seconds={}, expected {}".format(
                        variant,
                        block,
                        fields.get("simulated_seconds"),
                        args.expected_duration_seconds,
                    )
                )
    diagnostic = layout_gate.require_equal_outputs(
        args.result_root, args.control, args.treatment
    )
    layout_gate.require_equal_scientific_fields(
        rows, args.control, args.treatment
    )
    layout_gate.require_equal_resources(
        args.result_root, args.control, args.treatment
    )

    print("configuration gate: PASS")
    print("CPU placement gate: PASS")
    print("per-asset outputs: identical across all {} runs".format(
        2 * args.blocks
    ))
    print(
        "derived metrics: numerically equivalent; differing_cells={}, "
        "maximum_scaled_difference={:.6f}".format(
            diagnostic["differing_cells"],
            diagnostic["maximum_scaled_difference"],
        )
    )


if __name__ == "__main__":
    main()
