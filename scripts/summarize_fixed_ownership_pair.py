#!/usr/bin/env python3
import csv
import math
import pathlib
import statistics
import sys

import summarize_layout_pair as base


def fail(message):
    raise SystemExit(message)


def load_rows(root, control, treatment):
    rows = {control: {}, treatment: {}}
    common = dict(base.COMMON_FIELDS)
    common.pop("openmp_schedule", None)
    common.pop("persistent_openmp_team", None)
    expected_by_variant = {
        control: {
            "ranks": "16",
            "worker_threads": "1",
            "openmp_schedule": "dynamic1",
            "persistent_openmp_team": "0",
            "persistent_fixed_book_ownership": "0",
            "thread_ownership_output": "0",
        },
        treatment: {
            "ranks": "1",
            "worker_threads": "16",
            "openmp_schedule": "weighted-static",
            "persistent_openmp_team": "0",
            "persistent_fixed_book_ownership": "1",
            "thread_ownership_output": "1",
        },
    }
    for variant in (control, treatment):
        expected = dict(common)
        expected.update(expected_by_variant[variant])
        for block in base.BLOCKS:
            path = root / variant / "block_{}".format(block) / "run_1.txt"
            if not path.is_file():
                fail("missing {}".format(path))
            fields = base.read_run(path)
            for name, value in expected.items():
                if fields.get(name) != value:
                    fail(
                        "{} recorded {}={}, expected {}".format(
                            path, name, fields.get(name), value
                        )
                    )
            rows[variant][block] = fields
    return rows


def validate_thread_ownership(root, treatment):
    reference_path = root / treatment / "block_1" / "thread_ownership.csv"
    reference = reference_path.read_bytes()
    for block in base.BLOCKS[1:]:
        candidate = (
            root / treatment / "block_{}".format(block)
            / "thread_ownership.csv"
        )
        if candidate.read_bytes() != reference:
            fail("thread ownership changed in block {}".format(block))
    with reference_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {
        "asset_id", "symbol", "thread_id", "assignment_weight"
    }:
        fail("invalid thread ownership columns")
    assets = set()
    counts = [0] * 16
    loads = [0.0] * 16
    for row in rows:
        asset = int(row["asset_id"])
        thread = int(row["thread_id"])
        weight = float(row["assignment_weight"])
        if asset in assets or asset < 0 or asset >= 1480:
            fail("invalid or duplicate asset in thread ownership: {}".format(asset))
        if thread < 0 or thread >= 16:
            fail("invalid thread in thread ownership: {}".format(thread))
        if not math.isfinite(weight) or weight <= 0.0:
            fail("invalid assignment weight for asset {}".format(asset))
        assets.add(asset)
        counts[thread] += 1
        loads[thread] += weight
    if assets != set(range(1480)) or any(count == 0 for count in counts):
        fail("thread ownership does not cover all 1,480 books and 16 threads")
    mean_load = sum(loads) / 16.0
    return {
        "books": len(rows),
        "minimum_books_per_thread": min(counts),
        "maximum_books_per_thread": max(counts),
        "predicted_max_mean_load_ratio": max(loads) / mean_load,
    }


def main():
    if len(sys.argv) != 4:
        fail(
            "usage: summarize_fixed_ownership_pair.py "
            "RESULT_ROOT CONTROL TREATMENT"
        )
    root = pathlib.Path(sys.argv[1])
    control = sys.argv[2]
    treatment = sys.argv[3]
    base.require_complete_order(root, control, treatment)
    rows = load_rows(root, control, treatment)
    metric_diagnostic = base.require_equal_outputs(root, control, treatment)
    base.LAYOUT_OR_TIMING_FIELDS.update({
        "openmp_schedule",
        "persistent_fixed_book_ownership",
        "thread_ownership_output",
        "predicted_thread_imbalance",
    })
    base.require_equal_scientific_fields(rows, control, treatment)
    base.require_equal_resources(root, control, treatment)
    ownership = validate_thread_ownership(root, treatment)

    summaries = []
    timings = {}
    raw_rows = []
    all_stable = True
    for variant in (control, treatment):
        values = [
            float(rows[variant][block]["execution_seconds"])
            for block in base.BLOCKS
        ]
        internal = [
            float(rows[variant][block]["wall_seconds"])
            for block in base.BLOCKS
        ]
        for block, value, wall in zip(base.BLOCKS, values, internal):
            if (
                not math.isfinite(value) or value <= 0.0
                or not math.isfinite(wall) or wall <= 0.0
                or value < wall
            ):
                fail("invalid timing for {} block {}".format(variant, block))
            raw_rows.append({
                "block": block,
                "position": base.expected_order(
                    block, control, treatment
                ).index(variant) + 1,
                "variant": variant,
                "execution_seconds": value,
                "internal_wall_seconds": wall,
            })
        timings[variant] = values
        ratio = max(values) / min(values)
        stable = ratio <= 1.15
        all_stable = all_stable and stable
        summaries.append({
            "variant": variant,
            "repetitions": len(values),
            "minimum_execution": min(values),
            "median_execution": statistics.median(values),
            "maximum_execution": max(values),
            "max_min_ratio": ratio,
            "stability": (
                "performance_repetitions_stable"
                if stable else "performance_repetitions_rejected"
            ),
            "median_internal_wall": statistics.median(internal),
        })

    with (root / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    raw_rows.sort(key=lambda row: (row["block"], row["position"]))
    with (root / "raw_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    ratios = [
        timings[treatment][index] / timings[control][index]
        for index in range(len(base.BLOCKS))
    ]
    paired = {
        "treatment": treatment,
        "control": control,
        "performance_status": "accepted" if all_stable else "timing_rejected",
        "paired_geometric_change_pct": 100.0
            * (base.geometric_mean(ratios) - 1.0),
        "first_six_paired_change_pct": 100.0
            * (base.geometric_mean(ratios[:6]) - 1.0),
        "minimum_paired_change_pct": 100.0 * (min(ratios) - 1.0),
        "maximum_paired_change_pct": 100.0 * (max(ratios) - 1.0),
    }
    with (root / "paired_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired))
        writer.writeheader()
        writer.writerow(paired)
    with (root / "thread_ownership_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ownership))
        writer.writeheader()
        writer.writerow(ownership)

    print("configuration gate: PASS")
    print("CPU placement gate: PASS")
    print("scientific-equivalence gate: PASS")
    print(
        "fixed ownership: {} books, 16 permanent thread buckets, "
        "{}--{} books per thread, predicted max/mean load={:.6f}".format(
            ownership["books"],
            ownership["minimum_books_per_thread"],
            ownership["maximum_books_per_thread"],
            ownership["predicted_max_mean_load_ratio"],
        )
    )
    print(
        "metric equivalence: differing_cells={}, maximum_scaled={:.6f}".format(
            metric_diagnostic["differing_cells"],
            metric_diagnostic["maximum_scaled_difference"],
        )
    )
    for summary in summaries:
        print(
            "{} median={:.9f} min={:.9f} max={:.9f} {}".format(
                summary["variant"],
                summary["median_execution"],
                summary["minimum_execution"],
                summary["maximum_execution"],
                summary["stability"],
            )
        )
    if all_stable:
        print(
            "{} vs {} paired change={:+.3f}%".format(
                treatment, control, paired["paired_geometric_change_pct"]
            )
        )
    else:
        print("timing gate: REJECTED")


if __name__ == "__main__":
    main()
