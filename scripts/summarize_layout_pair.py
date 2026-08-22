#!/usr/bin/env python3
import csv
import math
import pathlib
import statistics
import sys


BLOCKS = tuple(range(1, 8))
METRIC_ABSOLUTE_TOLERANCE = 5.0e-9
METRIC_RELATIVE_TOLERANCE = 1.0e-12
TOLERANT_METRIC_COLUMNS = {
    "mean_spread_bps",
    "shocked_mean_spread_bps",
    "unshocked_mean_spread_bps",
}
TOLERANT_RUN_FIELDS = {
    "peak_mean_spread_bps",
    "final_mean_spread_bps",
}
COMMON_FIELDS = {
    "partition": "cyclic",
    "openmp_schedule": "dynamic1",
    "openmp_window_only": "0",
    "persistent_openmp_team": "0",
    "persistent_fixed_book_ownership": "0",
    "thread_ownership_output": "0",
    "buffered_observations": "0",
    "persistent_risk_collective": "0",
    "nonblocking_risk_collective": "0",
    "boundary_wait_profile": "0",
    "risk_lookahead_max_windows": "0",
    "parallel_asset_initialization": "0",
    "parallel_boundary_reductions": "0",
    "parallel_metric_scans": "0",
    "fuse_metric_cluster_scans": "0",
    "openmp_enabled": "1",
    "shared_inventory_policy": "gross_pooled",
    "shared_quote_multiplier": "2.000000000",
}
SCIENTIFIC_FIELDS = (
    "assets",
    "lobs",
    "windows",
    "processed_orders",
    "trades",
    "risk_boundaries",
    "risk_collective_calls",
    "observation_collective_calls",
    "local_mm_refresh_boundaries",
    "shock_requested_quantity",
    "shock_executed_quantity",
    "final_shared_gross_exposure",
    "maximum_shared_gross_exposure",
    "shared_signed_mark_to_mid_pnl_usd",
    "shared_signed_liquidation_pnl_usd",
    "shared_terminal_liquidation_cost_usd",
    "shared_terminal_absolute_inventory",
    "shared_unliquidated_terminal_quantity",
    "shared_buy_quantity",
    "shared_sell_quantity",
    "shared_fill_count",
)
LAYOUT_OR_TIMING_FIELDS = {
    "ranks",
    "worker_threads",
    "predicted_partition_imbalance",
    "predicted_thread_imbalance",
    "wall_seconds",
    "execution_seconds",
    "max_initialization_seconds",
    "max_compute_seconds",
    "max_communication_seconds",
    "communication_fraction",
    "max_risk_collective_seconds",
    "max_risk_overlap_work_seconds",
    "max_risk_wait_after_overlap_seconds",
    "max_observation_collective_seconds",
    "max_terminal_collective_seconds",
    "max_boundary_wait_seconds",
    "min_compute_seconds",
    "mean_compute_seconds",
    "compute_imbalance",
    "min_orders_per_rank",
    "mean_orders_per_rank",
    "max_orders_per_rank",
    "min_books_per_rank",
    "mean_books_per_rank",
    "max_books_per_rank",
}


def usage():
    raise SystemExit(
        "usage: summarize_layout_pair.py RESULT_ROOT CONTROL TREATMENT "
        "CONTROL_RANKS CONTROL_THREADS TREATMENT_RANKS TREATMENT_THREADS"
    )


def read_run(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("lob_mpi ") or line.startswith("lob_openmp ")
    ]
    if len(lines) != 1:
        raise SystemExit("expected one completed simulator line in {}".format(path))
    fields = {}
    for item in lines[0].split()[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    return fields


def expected_order(block, control, treatment):
    if block % 2 == 1:
        return (control, treatment)
    return (treatment, control)


def require_complete_order(root, control, treatment):
    with (root / "run_order.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        recorded = list(csv.DictReader(handle))
    expected_runs = 2 * len(BLOCKS)
    if len(recorded) != expected_runs:
        raise SystemExit(
            "run_order.csv must contain {} treatment runs".format(
                expected_runs
            )
        )
    for block in BLOCKS:
        block_rows = [row for row in recorded if int(row["block"]) == block]
        block_rows.sort(key=lambda row: int(row["position"]))
        variants = tuple(row["variant"] for row in block_rows)
        if variants != expected_order(block, control, treatment):
            raise SystemExit("invalid treatment order in block {}".format(block))


def load_rows(root, layouts):
    rows = {variant: {} for variant in layouts}
    for variant, (expected_ranks, expected_threads) in layouts.items():
        for block in BLOCKS:
            run = root / variant / "block_{}".format(block) / "run_1.txt"
            if not run.is_file():
                raise SystemExit("missing {}".format(run))
            fields = read_run(run)
            expected = dict(COMMON_FIELDS)
            expected["ranks"] = str(expected_ranks)
            expected["worker_threads"] = str(expected_threads)
            for key, value in expected.items():
                if fields.get(key) != value:
                    raise SystemExit(
                        "{} recorded {}={}, expected {}".format(
                            run, key, fields.get(key), value
                        )
                    )
            rows[variant][block] = fields
    return rows


def require_reproducible_outputs_within_layout(root, control, treatment):
    for variant in (control, treatment):
        reference = root / variant / "block_1"
        for stem in ("metrics", "assets"):
            reference_file = reference / "{}_1.csv".format(stem)
            reference_bytes = reference_file.read_bytes()
            for block in BLOCKS[1:]:
                candidate = (
                    root
                    / variant
                    / "block_{}".format(block)
                    / "{}_1.csv".format(stem)
                )
                if candidate.read_bytes() != reference_bytes:
                    raise SystemExit(
                        "output is not reproducible within {}: {} versus {}".format(
                            variant, reference_file, candidate
                        )
                    )


def numeric_difference(left_text, right_text, context):
    try:
        left = float(left_text)
        right = float(right_text)
    except ValueError:
        raise SystemExit("nonnumeric value in {}".format(context))
    if not math.isfinite(left) or not math.isfinite(right):
        raise SystemExit("nonfinite value in {}".format(context))
    difference = abs(left - right)
    limit = METRIC_ABSOLUTE_TOLERANCE + (
        METRIC_RELATIVE_TOLERANCE * max(abs(left), abs(right))
    )
    return left, right, difference, limit


def require_equivalent_outputs_across_layouts(root, control, treatment):
    reference = root / control / "block_1"
    candidate = root / treatment / "block_1"

    reference_assets = reference / "assets_1.csv"
    candidate_assets = candidate / "assets_1.csv"
    if reference_assets.read_bytes() != candidate_assets.read_bytes():
        raise SystemExit(
            "per-asset scientific output differs: {} versus {}".format(
                reference_assets, candidate_assets
            )
        )

    reference_metrics = reference / "metrics_1.csv"
    candidate_metrics = candidate / "metrics_1.csv"
    with reference_metrics.open(newline="", encoding="utf-8") as handle:
        reference_rows = list(csv.reader(handle))
    with candidate_metrics.open(newline="", encoding="utf-8") as handle:
        candidate_rows = list(csv.reader(handle))
    if len(reference_rows) < 2 or len(candidate_rows) < 2:
        raise SystemExit("metric output must contain a header and data rows")
    if reference_rows[0] != candidate_rows[0]:
        raise SystemExit("metric headers differ across layouts")
    if len(reference_rows) != len(candidate_rows):
        raise SystemExit("metric row counts differ across layouts")

    header = reference_rows[0]
    if not header or any(not name for name in header):
        raise SystemExit("metric header contains an empty column name")
    if len(set(header)) != len(header):
        raise SystemExit("metric header contains duplicate column names")
    missing_tolerant_columns = TOLERANT_METRIC_COLUMNS.difference(header)
    if missing_tolerant_columns:
        raise SystemExit(
            "metric output is missing expected columns: {}".format(
                ", ".join(sorted(missing_tolerant_columns))
            )
        )
    differing_cells = 0
    compared_cells = 0
    above_tolerance = 0
    maximum_absolute = 0.0
    maximum_absolute_row = 0
    maximum_absolute_column = "none"
    maximum_absolute_left = ""
    maximum_absolute_right = ""
    maximum_scaled = 0.0
    maximum_scaled_row = 0
    maximum_scaled_column = "none"
    maximum_scaled_left = ""
    maximum_scaled_right = ""
    for row_number, (left_row, right_row) in enumerate(
        zip(reference_rows[1:], candidate_rows[1:]), start=2
    ):
        if len(left_row) != len(header) or len(right_row) != len(header):
            raise SystemExit("metric column count differs at row {}".format(row_number))
        for column_index, name in enumerate(header):
            left_text = left_row[column_index]
            right_text = right_row[column_index]
            compared_cells += 1
            left, right, difference, limit = numeric_difference(
                left_text,
                right_text,
                "metric row {}, column {}".format(row_number, name),
            )
            if left_text == right_text:
                continue
            differing_cells += 1
            if name not in TOLERANT_METRIC_COLUMNS:
                raise SystemExit(
                    "exact metric differs at row {}, column {}: {} versus {}".format(
                        row_number, name, left_text, right_text
                    )
                )
            scaled = difference / limit
            if difference > limit:
                above_tolerance += 1
            if difference > maximum_absolute:
                maximum_absolute = difference
                maximum_absolute_row = row_number
                maximum_absolute_column = name
                maximum_absolute_left = left_text
                maximum_absolute_right = right_text
            if scaled > maximum_scaled:
                maximum_scaled = scaled
                maximum_scaled_row = row_number
                maximum_scaled_column = name
                maximum_scaled_left = left_text
                maximum_scaled_right = right_text

    diagnostic = {
        "status": "pass" if above_tolerance == 0 else "fail",
        "compared_cells": compared_cells,
        "differing_cells": differing_cells,
        "cells_above_tolerance": above_tolerance,
        "absolute_tolerance": METRIC_ABSOLUTE_TOLERANCE,
        "relative_tolerance": METRIC_RELATIVE_TOLERANCE,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_absolute_row": maximum_absolute_row,
        "maximum_absolute_column": maximum_absolute_column,
        "maximum_absolute_reference_value": maximum_absolute_left,
        "maximum_absolute_treatment_value": maximum_absolute_right,
        "maximum_scaled_difference": maximum_scaled,
        "maximum_scaled_row": maximum_scaled_row,
        "maximum_scaled_column": maximum_scaled_column,
        "maximum_scaled_reference_value": maximum_scaled_left,
        "maximum_scaled_treatment_value": maximum_scaled_right,
    }
    with (root / "metric_equivalence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostic.keys()))
        writer.writeheader()
        writer.writerow(diagnostic)
    if above_tolerance:
        raise SystemExit(
            "metric equivalence failed: {} cells exceed tolerance; see {}".format(
                above_tolerance, root / "metric_equivalence.csv"
            )
        )
    return diagnostic


def require_equal_outputs(root, control, treatment):
    require_reproducible_outputs_within_layout(root, control, treatment)
    return require_equivalent_outputs_across_layouts(root, control, treatment)


def require_equal_scientific_fields(rows, control, treatment):
    reference = rows[control][1]
    for field in SCIENTIFIC_FIELDS:
        if field not in reference:
            raise SystemExit("missing required scientific field: {}".format(field))
    for variant in (control, treatment):
        for block in BLOCKS:
            candidate = rows[variant][block]
            if set(candidate) != set(reference):
                raise SystemExit(
                    "result fields differ for {} block {}".format(variant, block)
                )
            for field in SCIENTIFIC_FIELDS:
                if candidate.get(field) != reference[field]:
                    raise SystemExit(
                        "scientific field {} differs for {} block {}: {} versus {}".format(
                            field, variant, block, candidate.get(field), reference[field]
                        )
                    )
            for field, value in reference.items():
                if field in LAYOUT_OR_TIMING_FIELDS:
                    continue
                if field in TOLERANT_RUN_FIELDS:
                    _, _, difference, limit = numeric_difference(
                        value,
                        candidate[field],
                        "run field {} for {} block {}".format(
                            field, variant, block
                        ),
                    )
                    if difference > limit:
                        raise SystemExit(
                            "run field {} differs beyond tolerance for {} block {}: "
                            "{} versus {}".format(
                                field,
                                variant,
                                block,
                                candidate[field],
                                value,
                            )
                        )
                    continue
                if candidate[field] != value:
                    raise SystemExit(
                        "controlled field {} differs for {} block {}: {} versus {}".format(
                            field, variant, block, candidate[field], value
                        )
                    )


def expand_cpu_list(value):
    cpus = set()
    for part in value.split(","):
        if "-" in part:
            first, last = part.split("-", 1)
            cpus.update(range(int(first), int(last) + 1))
        else:
            cpus.add(int(part))
    return cpus


def read_placement(path):
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        host, rank, cpu_text = raw.split("|", 2)
        rows.append((host, int(rank), expand_cpu_list(cpu_text)))
    return rows


def occupied_cpus(rows):
    occupied = set()
    for host, _, cpus in rows:
        for cpu in cpus:
            occupied.add((host, cpu))
    return occupied


def require_equal_resources(root, control, treatment):
    references = {}
    for variant in (control, treatment):
        reference_file = root / variant / "block_1" / "cpu_placement.txt"
        reference_bytes = reference_file.read_bytes()
        references[variant] = read_placement(reference_file)
        for block in BLOCKS[1:]:
            candidate = (
                root / variant / "block_{}".format(block) / "cpu_placement.txt"
            )
            if candidate.read_bytes() != reference_bytes:
                raise SystemExit(
                    "CPU placement changed for {} block {}".format(variant, block)
                )
    control_cpus = occupied_cpus(references[control])
    treatment_cpus = occupied_cpus(references[treatment])
    if control_cpus != treatment_cpus:
        raise SystemExit(
            "control and treatment did not use the same nodes and physical cores"
        )


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def main():
    if len(sys.argv) != 8:
        usage()
    root = pathlib.Path(sys.argv[1])
    control = sys.argv[2]
    treatment = sys.argv[3]
    layouts = {
        control: (int(sys.argv[4]), int(sys.argv[5])),
        treatment: (int(sys.argv[6]), int(sys.argv[7])),
    }

    require_complete_order(root, control, treatment)
    rows = load_rows(root, layouts)
    metric_diagnostic = require_equal_outputs(root, control, treatment)
    require_equal_scientific_fields(rows, control, treatment)
    require_equal_resources(root, control, treatment)

    summaries = []
    timings = {}
    raw_rows = []
    all_stable = True
    for variant in (control, treatment):
        values = [
            float(rows[variant][block]["execution_seconds"]) for block in BLOCKS
        ]
        internal_values = [
            float(rows[variant][block]["wall_seconds"]) for block in BLOCKS
        ]
        for block, value, internal in zip(BLOCKS, values, internal_values):
            if (
                not math.isfinite(value)
                or value <= 0.0
                or not math.isfinite(internal)
                or internal <= 0.0
                or value < internal
            ):
                raise SystemExit(
                    "invalid timing for {} block {}: execution={} internal={}".format(
                        variant, block, value, internal
                    )
                )
            raw_rows.append(
                {
                    "block": block,
                    "position": expected_order(block, control, treatment).index(
                        variant
                    )
                    + 1,
                    "variant": variant,
                    "execution_seconds": value,
                    "internal_wall_seconds": internal,
                }
            )
        timings[variant] = values
        ratio = max(values) / min(values)
        stable = ratio <= 1.15
        all_stable = all_stable and stable
        summaries.append(
            {
                "variant": variant,
                "repetitions": len(values),
                "minimum_execution": min(values),
                "median_execution": statistics.median(values),
                "maximum_execution": max(values),
                "max_min_ratio": ratio,
                "stability": (
                    "performance_repetitions_stable"
                    if stable
                    else "performance_repetitions_rejected"
                ),
                "median_internal_wall": statistics.median(internal_values),
            }
        )

    with (root / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    raw_rows.sort(key=lambda row: (row["block"], row["position"]))
    with (root / "raw_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    ratios = [
        timings[treatment][index] / timings[control][index]
        for index in range(len(BLOCKS))
    ]
    paired_row = {
        "treatment": treatment,
        "control": control,
        "performance_status": (
            "accepted" if all_stable else "timing_rejected"
        ),
        "paired_geometric_change_pct": 100.0 * (geometric_mean(ratios) - 1.0),
        "first_six_paired_change_pct": 100.0
        * (geometric_mean(ratios[:6]) - 1.0),
        "minimum_paired_change_pct": 100.0 * (min(ratios) - 1.0),
        "maximum_paired_change_pct": 100.0 * (max(ratios) - 1.0),
    }
    with (root / "paired_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_row.keys()))
        writer.writeheader()
        writer.writerow(paired_row)

    print("configuration gate: PASS")
    print("CPU placement gate: PASS")
    print("per-asset outputs: identical across all 14 runs")
    print(
        "derived metrics: numerically equivalent; differing_cells={}, "
        "maximum_scaled_difference={:.6f}, row={}, column={}".format(
            metric_diagnostic["differing_cells"],
            metric_diagnostic["maximum_scaled_difference"],
            metric_diagnostic["maximum_scaled_row"],
            metric_diagnostic["maximum_scaled_column"],
        )
    )
    print("metric diagnostic: {}".format(root / "metric_equivalence.csv"))
    for summary in summaries:
        print(
            "{} median={:.9f} min={:.9f} max={:.9f} max/min={:.6f} {}".format(
                summary["variant"],
                summary["median_execution"],
                summary["minimum_execution"],
                summary["maximum_execution"],
                summary["max_min_ratio"],
                summary["stability"],
            )
        )
    if all_stable:
        print(
            "{} vs {} paired change={:+.3f}%".format(
                treatment, control, paired_row["paired_geometric_change_pct"]
            )
        )
        print(
            "first-six balanced-order sensitivity={:+.3f}%".format(
                paired_row["first_six_paired_change_pct"]
            )
        )
    else:
        print(
            "timing gate: REJECTED; the diagnostic ratios were written but "
            "must not be reported as performance estimates"
        )
    print("summary: {}".format(root / "comparison.csv"))
    print("paired comparison: {}".format(root / "paired_comparison.csv"))


if __name__ == "__main__":
    main()
