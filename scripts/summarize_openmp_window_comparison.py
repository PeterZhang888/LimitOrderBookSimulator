#!/usr/bin/env python3
import csv
import pathlib
import statistics
import sys


def read_run(path):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.startswith("lob_mpi ")]
    if len(lines) != 1:
        raise SystemExit("expected one completed simulator line in {}".format(path))
    fields = {}
    for item in lines[0].split()[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    return fields


def rows_for(root, label):
    rows = []
    for repetition in range(1, 8):
        run = root / label / "run_{}.txt".format(repetition)
        if not run.is_file():
            raise SystemExit("missing {}".format(run))
        fields = read_run(run)
        expected = {
            "ranks": "32",
            "worker_threads": "2",
            "partition": "cyclic",
            "openmp_schedule": "dynamic1",
            "persistent_openmp_team": "0",
        }
        for key, value in expected.items():
            if fields.get(key) != value:
                raise SystemExit("{} recorded {}={}, expected {}".format(
                    run, key, fields.get(key), value))
        rows.append(fields)
    return rows


def require_equal_outputs(root):
    for repetition in range(1, 8):
        for stem in ("metrics", "assets"):
            control = root / "all_phases" / "{}_{}.csv".format(stem, repetition)
            treatment = root / "window_only" / "{}_{}.csv".format(stem, repetition)
            if control.read_bytes() != treatment.read_bytes():
                raise SystemExit("scientific output differs: {} versus {}".format(
                    control, treatment))


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_openmp_window_comparison.py RESULT_ROOT")
    root = pathlib.Path(sys.argv[1])
    control = rows_for(root, "all_phases")
    treatment = rows_for(root, "window_only")
    require_equal_outputs(root)

    summaries = []
    for label, rows in (("all_phases", control), ("window_only", treatment)):
        walls = [float(row["wall_seconds"]) for row in rows]
        summaries.append({
            "variant": label,
            "repetitions": len(walls),
            "minimum_wall": min(walls),
            "median_wall": statistics.median(walls),
            "maximum_wall": max(walls),
            "max_min_ratio": max(walls) / min(walls),
        })

    output = root / "comparison.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    control_median = summaries[0]["median_wall"]
    treatment_median = summaries[1]["median_wall"]
    change = 100.0 * (treatment_median / control_median - 1.0)
    print("scientific outputs: identical")
    print("control median: {:.9f}".format(control_median))
    print("window-only median: {:.9f}".format(treatment_median))
    print("change: {:+.3f}%".format(change))
    print("summary: {}".format(output))


if __name__ == "__main__":
    main()
