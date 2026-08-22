#!/usr/bin/env python3
import csv
import math
import pathlib
import statistics
import sys


BLOCKS = tuple(range(1, 8))
VARIANTS = ("all_phases", "window_only", "persistent")
EXPECTED_ORDERS = (
    ("all_phases", "window_only", "persistent"),
    ("window_only", "persistent", "all_phases"),
    ("persistent", "all_phases", "window_only"),
    ("all_phases", "persistent", "window_only"),
    ("persistent", "window_only", "all_phases"),
    ("window_only", "all_phases", "persistent"),
    ("all_phases", "window_only", "persistent"),
)
EXPECTED_TREATMENTS = {
    "all_phases": ("0", "0"),
    "window_only": ("1", "0"),
    "persistent": ("0", "1"),
}
COMMON_FIELDS = {
    "ranks": "32",
    "worker_threads": "2",
    "partition": "cyclic",
    "openmp_schedule": "dynamic1",
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
VARIABLE_FIELDS = {
    "openmp_window_only",
    "persistent_openmp_team",
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
}


def read_run(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("lob_mpi ")
    ]
    if len(lines) != 1:
        raise SystemExit(
            "expected one completed simulator line in {}".format(path)
        )
    fields = {}
    for item in lines[0].split()[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    return fields


def load_rows(root):
    rows = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        expected_window, expected_persistent = EXPECTED_TREATMENTS[variant]
        for block in BLOCKS:
            run = root / variant / "block_{}".format(block) / "run_1.txt"
            if not run.is_file():
                raise SystemExit("missing {}".format(run))
            fields = read_run(run)
            for key, expected in COMMON_FIELDS.items():
                if fields.get(key) != expected:
                    raise SystemExit(
                        "{} recorded {}={}, expected {}".format(
                            run, key, fields.get(key), expected
                        )
                    )
            treatment_fields = {
                "openmp_window_only": expected_window,
                "persistent_openmp_team": expected_persistent,
            }
            for key, expected in treatment_fields.items():
                if fields.get(key) != expected:
                    raise SystemExit(
                        "{} recorded {}={}, expected {}".format(
                            run, key, fields.get(key), expected
                        )
                    )
            rows[variant][block] = fields
    return rows


def require_equal_outputs(root):
    reference = root / "all_phases" / "block_1"
    for stem in ("metrics", "assets"):
        reference_bytes = (reference / "{}_1.csv".format(stem)).read_bytes()
        for variant in VARIANTS:
            for block in BLOCKS:
                candidate = (
                    root
                    / variant
                    / "block_{}".format(block)
                    / "{}_1.csv".format(stem)
                )
                if candidate.read_bytes() != reference_bytes:
                    raise SystemExit(
                        "scientific output differs: {} versus {}".format(
                            reference / "{}_1.csv".format(stem), candidate
                        )
                    )


def require_equal_placements(root):
    reference = (
        root / "all_phases" / "block_1" / "cpu_placement.txt"
    ).read_bytes()
    for variant in VARIANTS:
        for block in BLOCKS:
            candidate = (
                root
                / variant
                / "block_{}".format(block)
                / "cpu_placement.txt"
            )
            if candidate.read_bytes() != reference:
                raise SystemExit(
                    "CPU placement differs: {} versus {}".format(
                        root / "all_phases" / "block_1" / "cpu_placement.txt",
                        candidate,
                    )
                )


def require_equal_scientific_fields(rows):
    reference = rows["all_phases"][1]
    for field in SCIENTIFIC_FIELDS:
        if field not in reference:
            raise SystemExit("missing required scientific field: {}".format(field))
    for variant in VARIANTS:
        for block in BLOCKS:
            candidate = rows[variant][block]
            if set(candidate) != set(reference):
                raise SystemExit(
                    "result fields differ for {} block {}".format(variant, block)
                )
            for field in SCIENTIFIC_FIELDS:
                if field not in candidate:
                    raise SystemExit(
                        "missing scientific field {} for {} block {}".format(
                            field, variant, block
                        )
                    )
                if candidate[field] != reference[field]:
                    raise SystemExit(
                        "scientific field {} differs for {} block {}: {} versus {}".format(
                            field,
                            variant,
                            block,
                            candidate.get(field),
                            reference.get(field),
                        )
                    )
            for field, value in reference.items():
                if field in VARIABLE_FIELDS:
                    continue
                if candidate[field] != value:
                    raise SystemExit(
                        "controlled field {} differs for {} block {}: {} versus {}".format(
                            field, variant, block, candidate[field], value
                        )
                    )


def require_complete_order(root):
    order_file = root / "run_order.csv"
    with order_file.open(newline="", encoding="utf-8") as handle:
        recorded = list(csv.DictReader(handle))
    if len(recorded) != 21:
        raise SystemExit("run_order.csv must contain 21 treatment runs")
    for block in BLOCKS:
        block_rows = [row for row in recorded if int(row["block"]) == block]
        block_rows.sort(key=lambda row: int(row["position"]))
        variants = tuple(row["variant"] for row in block_rows)
        if variants != EXPECTED_ORDERS[block - 1]:
            raise SystemExit("invalid treatment order in block {}".format(block))


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_openmp_three_way.py RESULT_ROOT")
    root = pathlib.Path(sys.argv[1])
    require_complete_order(root)
    rows = load_rows(root)
    require_equal_outputs(root)
    require_equal_placements(root)
    require_equal_scientific_fields(rows)

    summaries = []
    walls = {}
    raw_rows = []
    for variant in VARIANTS:
        values = [
            float(rows[variant][block]["execution_seconds"])
            for block in BLOCKS
        ]
        internal_values = [
            float(rows[variant][block]["wall_seconds"])
            for block in BLOCKS
        ]
        for block, value, internal in zip(BLOCKS, values, internal_values):
            if (not math.isfinite(value) or value <= 0.0
                    or not math.isfinite(internal) or internal <= 0.0
                    or value < internal):
                raise SystemExit(
                    "invalid timing for {} block {}: execution={} internal={}".format(
                        variant, block, value, internal
                    )
                )
            position = EXPECTED_ORDERS[block - 1].index(variant) + 1
            raw_rows.append(
                {
                    "block": block,
                    "position": position,
                    "variant": variant,
                    "execution_seconds": value,
                    "internal_wall_seconds": internal,
                }
            )
        walls[variant] = values
        ratio = max(values) / min(values)
        if ratio > 1.5:
            raise SystemExit(
                "unstable timing for {}: max/min={:.6f}".format(variant, ratio)
            )
        summaries.append(
            {
                "variant": variant,
                "repetitions": len(values),
                "minimum_execution": min(values),
                "median_execution": statistics.median(values),
                "maximum_execution": max(values),
                "max_min_ratio": ratio,
                "median_internal_wall": statistics.median(internal_values),
            }
        )

    summary_file = root / "comparison.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    raw_file = root / "raw_results.csv"
    raw_rows.sort(key=lambda row: (row["block"], row["position"]))
    with raw_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    comparisons = (
        ("window_only", "all_phases"),
        ("persistent", "all_phases"),
        ("persistent", "window_only"),
    )
    paired_rows = []
    for treatment, control in comparisons:
        ratios = [
            walls[treatment][index] / walls[control][index]
            for index in range(len(BLOCKS))
        ]
        paired_rows.append(
            {
                "treatment": treatment,
                "control": control,
                "paired_geometric_change_pct": 100.0 * (geometric_mean(ratios) - 1.0),
                "first_six_paired_change_pct": 100.0 * (
                    geometric_mean(ratios[:6]) - 1.0
                ),
                "minimum_paired_change_pct": 100.0 * (min(ratios) - 1.0),
                "maximum_paired_change_pct": 100.0 * (max(ratios) - 1.0),
            }
        )

    paired_file = root / "paired_comparisons.csv"
    with paired_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)

    print("configuration gate: PASS")
    print("scientific outputs: identical across all 21 runs")
    for summary in summaries:
        print(
            "{} median={:.9f} min={:.9f} max={:.9f} max/min={:.6f}".format(
                summary["variant"],
                summary["median_execution"],
                summary["minimum_execution"],
                summary["maximum_execution"],
                summary["max_min_ratio"],
            )
        )
    for row in paired_rows:
        print(
            "{} vs {} paired change={:+.3f}%".format(
                row["treatment"],
                row["control"],
                row["paired_geometric_change_pct"],
            )
        )
        print(
            "{} vs {} first-six sensitivity={:+.3f}%".format(
                row["treatment"],
                row["control"],
                row["first_six_paired_change_pct"],
            )
        )
    print("raw results: {}".format(raw_file))
    print("summary: {}".format(summary_file))
    print("paired comparisons: {}".format(paired_file))


if __name__ == "__main__":
    main()
