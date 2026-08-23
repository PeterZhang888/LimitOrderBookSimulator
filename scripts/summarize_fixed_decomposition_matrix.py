#!/usr/bin/env python3
import csv
import math
import pathlib
import statistics
import sys

import summarize_layout_pair as base

HEALTH_THRESHOLD_MS = 2.0


def fail(message):
    raise SystemExit(message)


def layouts_for(total_cores):
    return [
        ("mpi_{}x{}".format(total_cores // threads, threads),
         total_cores // threads, threads)
        for threads in (1, 2, 4, 8, 16)
    ]


def validate_order(root, layouts, blocks):
    with (root / "run_order.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(blocks) * len(layouts):
        fail("run_order.csv does not contain the requested matrix")
    positions = {}
    labels = [layout[0] for layout in layouts]
    for block in blocks:
        block_rows = [row for row in rows if int(row["block"]) == block]
        block_rows.sort(key=lambda row: int(row["position"]))
        expected = [
            labels[((block - 1) % len(labels) + position) % len(labels)]
            for position in range(len(labels))
        ]
        if [row["variant"] for row in block_rows] != expected:
            fail("invalid treatment order in block {}".format(block))
        for row in block_rows:
            label = row["variant"]
            layout = next(item for item in layouts if item[0] == label)
            if (int(row["ranks"]), int(row["threads"])) != layout[1:]:
                fail("rank/thread metadata differs in block {}".format(block))
            positions[(block, label)] = int(row["position"])
    return positions


def load_runs(root, layouts, blocks):
    rows = {label: {} for label, _, _ in layouts}
    common = dict(base.COMMON_FIELDS)
    common.pop("openmp_schedule", None)
    common.pop("persistent_fixed_book_ownership", None)
    for label, ranks, threads in layouts:
        expected = dict(common)
        expected.update({
            "ranks": str(ranks),
            "assets": "1480",
            "lobs": "1480",
            "simulated_seconds": "23400",
            "windows": "23400",
            "worker_threads": str(threads),
            "openmp_schedule": (
                "dynamic1" if threads == 1 else "weighted-static"
            ),
            "persistent_fixed_book_ownership": (
                "0" if threads == 1 else "1"
            ),
            "thread_ownership_output": "0",
            "mpi_health_check_iterations": "100",
            "mpi_health_check_threshold_ms": "2.000000000",
        })
        for block in blocks:
            path = root / label / "block_{}".format(block) / "run_1.txt"
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
            health = float(fields["mpi_health_check_max_mean_ms"])
            execution = float(fields["execution_seconds"])
            if not math.isfinite(health) or health > HEALTH_THRESHOLD_MS:
                fail("{} failed the MPI health gate".format(path))
            if not math.isfinite(execution) or execution <= 0.0:
                fail("{} recorded an invalid execution time".format(path))
            rows[label][block] = fields
    return rows


def validate_attempts(root, layouts, blocks):
    path = root / "attempts.csv"
    if not path.is_file():
        fail("missing {}".format(path))
    with path.open(newline="", encoding="utf-8") as handle:
        attempt_rows = list(csv.DictReader(handle))
    expected_slots = {"cost_preparation"}
    for label, _, _ in layouts:
        for block in blocks:
            expected_slots.add("{}/block_{}".format(label, block))
    observed_accepted = set()
    attempts_by_slot = {}
    for row in attempt_rows:
        slot = row["slot"]
        if slot not in expected_slots:
            fail("attempt log contains unexpected slot {}".format(slot))
        attempt = int(row["attempt"])
        attempts_by_slot.setdefault(slot, []).append(attempt)
        health = float(row["health_mean_ms"])
        if not math.isfinite(health) or health < 0.0:
            fail("invalid health-check time for {}".format(slot))
        status = row["status"]
        result_path = pathlib.Path(row["result_path"])
        if not result_path.is_dir():
            fail("attempt result directory is missing: {}".format(result_path))
        if status == "accepted":
            if slot in observed_accepted:
                fail("multiple accepted attempts for {}".format(slot))
            execution = float(row["execution_seconds"])
            if (health > HEALTH_THRESHOLD_MS
                    or not math.isfinite(execution)
                    or execution <= 0.0):
                fail("accepted attempt violates a safeguard for {}".format(slot))
            observed_accepted.add(slot)
        elif status == "preflight_rejected":
            if health <= HEALTH_THRESHOLD_MS or row["execution_seconds"]:
                fail("invalid preflight rejection for {}".format(slot))
        elif status == "postrun_rejected":
            # Backward compatibility for result directories produced before
            # completed long runs were retained. New campaigns never write
            # this status.
            execution = float(row["execution_seconds"])
            if (health > HEALTH_THRESHOLD_MS
                    or not math.isfinite(execution)
                    or execution <= 0.0):
                fail("invalid post-run rejection for {}".format(slot))
        else:
            fail("unknown attempt status {}".format(status))
    if observed_accepted != expected_slots:
        fail("attempt log does not contain one accepted run for every slot")
    for slot, attempts in attempts_by_slot.items():
        if attempts != list(range(1, len(attempts) + 1)):
            fail("attempt numbers are not consecutive for {}".format(slot))
    return attempt_rows


def validate_outputs_and_resources(root, layouts, rows):
    control = layouts[0][0]
    base.LAYOUT_OR_TIMING_FIELDS.update({
        "openmp_schedule",
        "persistent_fixed_book_ownership",
        "predicted_thread_imbalance",
        "mpi_health_check_max_mean_ms",
    })
    diagnostics = []
    for treatment, _, _ in layouts[1:]:
        diagnostic = base.require_equal_outputs(root, control, treatment)
        diagnostic["treatment"] = treatment
        source = root / "metric_equivalence.csv"
        source.replace(root / "metric_equivalence_{}.csv".format(treatment))
        diagnostics.append(diagnostic)
        base.require_equal_scientific_fields(rows, control, treatment)
        base.require_equal_resources(root, control, treatment)
    return diagnostics


def validate_timing(value, wall, label, block):
    if (
        not math.isfinite(value) or value <= 0.0
        or not math.isfinite(wall) or wall <= 0.0
        or value < wall
    ):
        fail("invalid timing for {} block {}".format(label, block))


def main():
    if len(sys.argv) not in (3, 4):
        fail(
            "usage: summarize_fixed_decomposition_matrix.py "
            "RESULT_ROOT TOTAL_CORES [BLOCK_COUNT]"
        )
    root = pathlib.Path(sys.argv[1])
    total_cores = int(sys.argv[2])
    if total_cores not in (16, 32, 64):
        fail("TOTAL_CORES must be 16, 32, or 64")
    block_count = int(sys.argv[3]) if len(sys.argv) == 4 else 7
    if block_count < 1 or block_count > 7:
        fail("BLOCK_COUNT must be between 1 and 7")
    blocks = tuple(range(1, block_count + 1))
    base.BLOCKS = blocks
    layouts = layouts_for(total_cores)
    positions = validate_order(root, layouts, blocks)
    attempt_rows = validate_attempts(root, layouts, blocks)
    rows = load_runs(root, layouts, blocks)
    diagnostics = validate_outputs_and_resources(root, layouts, rows)

    summaries = []
    raw = []
    timings = {}
    stable_by_layout = {}
    for label, ranks, threads in layouts:
        values = []
        walls = []
        imbalances = []
        for block in blocks:
            fields = rows[label][block]
            value = float(fields["execution_seconds"])
            wall = float(fields["wall_seconds"])
            imbalance = float(fields["predicted_thread_imbalance"])
            validate_timing(value, wall, label, block)
            if not math.isfinite(imbalance) or imbalance < 1.0:
                fail(
                    "invalid predicted thread imbalance for {} block {}"
                    .format(label, block)
                )
            values.append(value)
            walls.append(wall)
            imbalances.append(imbalance)
            health = float(fields["mpi_health_check_max_mean_ms"])
            raw.append({
                "total_cores": total_cores,
                "block": block,
                "position": positions[(block, label)],
                "variant": label,
                "ranks": ranks,
                "threads": threads,
                "execution_seconds": value,
                "internal_wall_seconds": wall,
                "predicted_max_thread_imbalance": imbalance,
                "mpi_health_check_max_mean_ms": health,
            })
        ratio = max(values) / min(values)
        stable = ratio <= 1.15
        timings[label] = values
        stable_by_layout[label] = stable
        summaries.append({
            "total_cores": total_cores,
            "variant": label,
            "ranks": ranks,
            "threads": threads,
            "repetitions": len(values),
            "minimum_execution": min(values),
            "median_execution": statistics.median(values),
            "maximum_execution": max(values),
            "max_min_ratio": ratio,
            "stability": (
                "performance_repetitions_stable"
                if stable else "performance_variability_warning"
            ),
            "median_internal_wall": statistics.median(walls),
            "maximum_predicted_thread_imbalance": max(imbalances),
            "maximum_mpi_health_check_mean_ms": max(
                float(rows[label][block]["mpi_health_check_max_mean_ms"])
                for block in blocks
            ),
        })

    raw.sort(key=lambda row: (row["block"], row["position"]))
    with (root / "raw_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0].keys()))
        writer.writeheader()
        writer.writerows(raw)
    with (root / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    control = layouts[0][0]
    paired_rows = []
    for treatment, ranks, threads in layouts[1:]:
        ratios = [
            timings[treatment][index] / timings[control][index]
            for index in range(len(blocks))
        ]
        stable_comparison = (
            stable_by_layout[control] and stable_by_layout[treatment]
        )
        paired_rows.append({
            "total_cores": total_cores,
            "treatment": treatment,
            "ranks": ranks,
            "threads": threads,
            "control": control,
            "performance_status": (
                "accepted"
                if stable_comparison else "timing_variability_warning"
            ),
            "paired_geometric_change_pct": 100.0
                * (base.geometric_mean(ratios) - 1.0),
            "first_five_balanced_change_pct": 100.0
                * (base.geometric_mean(ratios[:min(5, len(ratios))]) - 1.0),
            "minimum_paired_change_pct": 100.0 * (min(ratios) - 1.0),
            "maximum_paired_change_pct": 100.0 * (max(ratios) - 1.0),
        })
    with (root / "paired_comparisons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)

    print("configuration gate: PASS")
    print("CPU placement gate: PASS")
    print("scientific-equivalence gate: PASS")
    print("MPI health gate: PASS ({} attempts)".format(
        len(attempt_rows)
    ))
    for summary in summaries:
        print(
            "{} median={:.9f} min={:.9f} max={:.9f} {}"
            .format(
                summary["variant"],
                summary["median_execution"],
                summary["minimum_execution"],
                summary["maximum_execution"],
                summary["stability"],
            )
        )
    for row in paired_rows:
        if row["performance_status"] == "accepted":
            print(
                "{} vs {} paired change={:+.3f}%".format(
                    row["treatment"], row["control"],
                    row["paired_geometric_change_pct"],
                )
            )
        else:
            print("{} timing variability: WARNING".format(row["treatment"]))
    print("metric comparisons passed: {}".format(len(diagnostics)))
    if not all(stable_by_layout.values()):
        print(
            "TIMING VARIABILITY WARNING: one or more layouts have a "
            "maximum-to-minimum ratio above 1.15; all repetitions and "
            "median results were retained"
        )


if __name__ == "__main__":
    main()
