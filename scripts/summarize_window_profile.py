#!/usr/bin/env python3
"""Summarize matched per-window phase profiles.

The input may be either raw rank/window rows or one row per window that was
aggregated after the run.  Raw rows avoid adding profiling collectives to the
timed simulator.  Their required columns are::

    window_index,start_time_seconds,end_time_seconds,rank,
    total_window_seconds,<phase>_seconds,...

Aggregated rows require::

    window_index,start_time_seconds,end_time_seconds,
    window_total_max_seconds,window_total_mean_seconds,
    <phase>_max_seconds,<phase>_mean_seconds,...

Aggregated ``<phase>_critical_seconds`` columns are optional.  For raw input,
the script derives them from the rank with the largest total time in each
window.  Per-window order and trade counts are checked when present, but are
not required because full-session scientific equivalence is checked by the
experiment launcher.

``risk_collective_seconds`` and ``global_metrics_collective_seconds`` are the
elapsed collective-call times experienced by each rank.  They include rank
arrival waiting and MPI progress as well as data movement, so they must not be
described as network time alone.

For raw MPI input, the phase decomposition uses the slowest observed rank in
each window.  This is a diagnostic proxy for work on the window's slow path,
not an exact reconstruction of the full-session critical path.
"""

import argparse
import csv
import glob
import math
import pathlib
import statistics


ORDER_COLUMNS = (
    "window_processed_orders",
    "window_orders",
    "processed_orders",
    "orders",
)
TRADE_COLUMNS = (
    "window_trade_count",
    "window_trades",
    "trades",
    "trade_count",
)
RAW_PHASES = (
    "event_processing",
    "risk_local",
    "risk_collective",
    "asset_moments",
    "return_panel",
    "local_market_maker",
    "risk_finalize",
    "global_metrics_local",
    "global_metrics_collective",
    "global_metrics_write",
    "fundamental",
    "shared_market_maker",
    "news_value_agent",
    "periodic_value_agent",
    "other",
)
RAW_REQUIRED = (
    "window_index",
    "start_time_seconds",
    "end_time_seconds",
    "rank",
    "total_window_seconds",
)
AGGREGATE_REQUIRED = (
    "window_index",
    "start_time_seconds",
    "end_time_seconds",
    "window_total_max_seconds",
    "window_total_mean_seconds",
)


def fail(message):
    raise SystemExit(message)


def finite_number(text, context, nonnegative=False):
    try:
        value = float(text)
    except (TypeError, ValueError):
        fail("{} is not numeric: {!r}".format(context, text))
    if not math.isfinite(value):
        fail("{} is not finite: {!r}".format(context, text))
    if nonnegative and value < 0.0:
        fail("{} is negative: {!r}".format(context, text))
    return value


def integer_value(text, context):
    value = finite_number(text, context, nonnegative=True)
    rounded = int(value)
    if value != float(rounded):
        fail("{} is not an integer: {!r}".format(context, text))
    return rounded


def percentile(values, probability):
    """Linearly interpolated percentile on the sorted sample."""
    if not values:
        fail("cannot calculate a percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def select_column(fieldnames, candidates, label, required=True):
    present = [name for name in candidates if name in fieldnames]
    if len(present) > 1 or (required and len(present) != 1):
        fail(
            "expected {} {} column from {}; found {}".format(
                "exactly one" if required else "at most one",
                label, ", ".join(candidates), ", ".join(present) or "none"
            )
        )
    return present[0] if present else None


def read_csv_files(paths):
    rows = []
    reference_header = None
    for path in paths:
        if not path.is_file():
            fail("profile CSV does not exist: {}".format(path))
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                fail("profile CSV has no header: {}".format(path))
            if any(not name for name in reader.fieldnames):
                fail("profile CSV contains an empty column name: {}".format(path))
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                fail("profile CSV contains duplicate column names: {}".format(path))
            if reference_header is None:
                reference_header = list(reader.fieldnames)
            elif list(reader.fieldnames) != reference_header:
                fail("profile CSV headers differ: {}".format(path))
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    fail("too many fields at {}:{}".format(path, line_number))
                row["__source"] = "{}:{}".format(path, line_number)
                rows.append(row)
    if not rows:
        fail("profile input contains no data rows")
    return reference_header, rows


def expand_input_paths(values, label):
    expanded = []
    for value in values:
        pattern = str(value)
        if glob.has_magic(pattern):
            matches = [pathlib.Path(item) for item in sorted(glob.glob(pattern))]
            if not matches:
                fail("{} pattern matched no files: {}".format(label, pattern))
            expanded.extend(matches)
        else:
            expanded.append(pathlib.Path(pattern))
    unique = []
    seen = set()
    for path in expanded:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def raw_phase_names(fieldnames):
    missing = [phase for phase in RAW_PHASES if phase + "_seconds" not in fieldnames]
    if missing:
        fail("raw profile is missing phase columns: {}".format(", ".join(missing)))
    return RAW_PHASES


def closure_limit(total, parts):
    return 1.0e-9 + 1.0e-6 * max(abs(total), abs(parts))


def require_closed(total, parts, context):
    difference = abs(total - parts)
    if difference > closure_limit(total, parts):
        fail(
            "phase times do not close to total in {}: total={:.17g}, "
            "phase_sum={:.17g}, difference={:.17g}".format(
                context, total, parts, difference
            )
        )


def require_one_second_interval(window, start_time, end_time, context):
    expected_start = float(window - 1)
    expected_end = float(window)
    tolerance = 1.0e-12
    if (
        abs(start_time - expected_start) > tolerance
        or abs(end_time - expected_end) > tolerance
    ):
        fail(
            "{} does not describe expected one-second interval [{}, {}): "
            "found [{:.17g}, {:.17g})".format(
                context, window - 1, window, start_time, end_time
            )
        )


def aggregate_phase_names(fieldnames):
    phases = []
    for name in fieldnames:
        if not name.endswith("_max_seconds"):
            continue
        phase = name[:-len("_max_seconds")]
        if phase == "window_total":
            continue
        mean_name = phase + "_mean_seconds"
        if mean_name not in fieldnames:
            fail("aggregate profile is missing {}".format(mean_name))
        phases.append(phase)
    if not phases:
        fail("aggregate profile contains no <phase>_max_seconds columns")
    critical = [phase + "_critical_seconds" in fieldnames for phase in phases]
    if any(critical) and not all(critical):
        fail("critical-rank phase columns must be present for every phase or none")
    return tuple(phases), all(critical)


def normalize_raw(label, fieldnames, source_rows):
    for name in RAW_REQUIRED:
        if name not in fieldnames:
            fail("raw profile is missing {}".format(name))
    order_column = select_column(
        fieldnames, ORDER_COLUMNS, "order-count", required=False
    )
    trade_column = select_column(
        fieldnames, TRADE_COLUMNS, "trade-count", required=False
    )
    if (order_column is None) != (trade_column is None):
        fail("order and trade columns must either both be present or both be absent")
    phases = raw_phase_names(fieldnames)

    grouped = {}
    for source in source_rows:
        context = source["__source"]
        window = integer_value(source["window_index"], context + " window_index")
        start_time = finite_number(
            source["start_time_seconds"], context + " start_time_seconds"
        )
        end_time = finite_number(
            source["end_time_seconds"], context + " end_time_seconds"
        )
        if end_time <= start_time:
            fail("window end must be after its start in {}".format(context))
        require_one_second_interval(window, start_time, end_time, context)
        rank = integer_value(source["rank"], context + " rank")
        key = window
        if key not in grouped:
            grouped[key] = {
                "start_time": start_time,
                "end_time": end_time,
                "ranks": {},
            }
        elif (
            grouped[key]["start_time"] != start_time
            or grouped[key]["end_time"] != end_time
        ):
            fail("time interval differs within window {}".format(window))
        if rank in grouped[key]["ranks"]:
            fail("duplicate rank {} in window {}".format(rank, window))
        phase_values = {}
        for phase in phases:
            phase_values[phase] = finite_number(
                source[phase + "_seconds"],
                "{} {}_seconds".format(context, phase),
                nonnegative=True,
            )
        total = finite_number(
                source["total_window_seconds"],
                context + " total_window_seconds",
                nonnegative=True,
            )
        require_closed(total, sum(phase_values.values()), context)
        grouped[key]["ranks"][rank] = {
            "total": total,
            "orders": (
                integer_value(source[order_column], context + " orders")
                if order_column is not None else None
            ),
            "trades": (
                integer_value(source[trade_column], context + " trades")
                if trade_column is not None else None
            ),
            "phases": phase_values,
        }

    expected_ranks = None
    normalized = []
    expected_window = 1
    previous_end = None
    for window in sorted(grouped):
        group = grouped[window]
        if window != expected_window:
            fail(
                "window indices must be contiguous from 1; expected {}, found {}".format(
                    expected_window, window
                )
            )
        if previous_end is not None and group["start_time"] != previous_end:
            fail("time intervals are not continuous before window {}".format(window))
        expected_window += 1
        previous_end = group["end_time"]
        ranks = tuple(sorted(group["ranks"]))
        if expected_ranks is None:
            expected_ranks = ranks
        elif ranks != expected_ranks:
            fail("rank set differs in window {}".format(window))
        values = group["ranks"]
        critical_rank = min(ranks, key=lambda rank: (-values[rank]["total"], rank))
        totals = [values[rank]["total"] for rank in ranks]
        phase_max = {}
        phase_min = {}
        phase_mean = {}
        phase_critical = {}
        for phase in phases:
            phase_sample = [values[rank]["phases"][phase] for rank in ranks]
            phase_max[phase] = max(phase_sample)
            phase_min[phase] = min(phase_sample)
            phase_mean[phase] = statistics.mean(phase_sample)
            phase_critical[phase] = values[critical_rank]["phases"][phase]
        normalized.append(
            {
                "window": window,
                "start_time": group["start_time"],
                "time": group["end_time"],
                "rank_count": len(ranks),
                "critical_rank": critical_rank,
                "total_max": max(totals),
                "total_mean": statistics.mean(totals),
                "orders": (
                    sum(values[rank]["orders"] for rank in ranks)
                    if order_column is not None else None
                ),
                "trades": (
                    sum(values[rank]["trades"] for rank in ranks)
                    if trade_column is not None else None
                ),
                "phase_max": phase_max,
                "phase_min": phase_min,
                "phase_mean": phase_mean,
                "phase_critical": phase_critical,
            }
        )
    return {
        "label": label,
        "mode": "raw_rank_rows",
        "phases": phases,
        "has_min": True,
        "has_critical": True,
        "has_counts": order_column is not None,
        "rows": normalized,
    }


def normalize_aggregate(label, fieldnames, source_rows):
    for name in AGGREGATE_REQUIRED:
        if name not in fieldnames:
            fail("aggregate profile is missing {}".format(name))
    order_column = select_column(
        fieldnames, ORDER_COLUMNS, "order-count", required=False
    )
    trade_column = select_column(
        fieldnames, TRADE_COLUMNS, "trade-count", required=False
    )
    if (order_column is None) != (trade_column is None):
        fail("order and trade columns must either both be present or both be absent")
    phases, has_critical = aggregate_phase_names(fieldnames)
    has_min = all(phase + "_min_seconds" in fieldnames for phase in phases)
    if any(phase + "_min_seconds" in fieldnames for phase in phases) and not has_min:
        fail("rank-minimum phase columns must be present for every phase or none")
    has_rank_count = "rank_count" in fieldnames
    has_critical_rank = "critical_rank" in fieldnames

    normalized = []
    seen = set()
    for source in source_rows:
        context = source["__source"]
        window = integer_value(source["window_index"], context + " window_index")
        if window in seen:
            fail("duplicate aggregate row for window {}".format(window))
        seen.add(window)
        phase_max = {}
        phase_min = {}
        phase_mean = {}
        phase_critical = {}
        for phase in phases:
            phase_max[phase] = finite_number(
                source[phase + "_max_seconds"],
                "{} {} max".format(context, phase),
                nonnegative=True,
            )
            phase_mean[phase] = finite_number(
                source[phase + "_mean_seconds"],
                "{} {} mean".format(context, phase),
                nonnegative=True,
            )
            if has_min:
                phase_min[phase] = finite_number(
                    source[phase + "_min_seconds"],
                    "{} {} min".format(context, phase),
                    nonnegative=True,
                )
            if has_critical:
                phase_critical[phase] = finite_number(
                    source[phase + "_critical_seconds"],
                    "{} {} critical".format(context, phase),
                    nonnegative=True,
                )
        start_time = finite_number(
            source["start_time_seconds"], context + " start_time_seconds"
        )
        end_time = finite_number(
            source["end_time_seconds"], context + " end_time_seconds"
        )
        if end_time <= start_time:
            fail("window end must be after its start in {}".format(context))
        require_one_second_interval(window, start_time, end_time, context)
        total_max = finite_number(
            source["window_total_max_seconds"],
            context + " window total max",
            nonnegative=True,
        )
        if has_critical:
            require_closed(total_max, sum(phase_critical.values()), context)
        normalized.append(
            {
                "window": window,
                "start_time": start_time,
                "time": end_time,
                "rank_count": integer_value(
                    source["rank_count"], context + " rank_count"
                ) if has_rank_count else 0,
                "critical_rank": integer_value(
                    source["critical_rank"], context + " critical_rank"
                ) if has_critical_rank else -1,
                "total_max": total_max,
                "total_mean": finite_number(
                    source["window_total_mean_seconds"],
                    context + " window total mean",
                    nonnegative=True,
                ),
                "orders": (
                    integer_value(source[order_column], context + " orders")
                    if order_column is not None else None
                ),
                "trades": (
                    integer_value(source[trade_column], context + " trades")
                    if trade_column is not None else None
                ),
                "phase_max": phase_max,
                "phase_min": phase_min,
                "phase_mean": phase_mean,
                "phase_critical": phase_critical,
            }
        )
    normalized.sort(key=lambda row: row["window"])
    previous_end = None
    for expected_window, row in enumerate(normalized, start=1):
        if row["window"] != expected_window:
            fail(
                "window indices must be contiguous from 1; expected {}, found {}".format(
                    expected_window, row["window"]
                )
            )
        if previous_end is not None and row["start_time"] != previous_end:
            fail("time intervals are not continuous before window {}".format(
                row["window"]
            ))
        previous_end = row["time"]
    return {
        "label": label,
        "mode": "aggregate_window_rows",
        "phases": phases,
        "has_min": has_min,
        "has_critical": has_critical,
        "has_counts": order_column is not None,
        "rows": normalized,
    }


def load_profile(label, paths):
    fieldnames, rows = read_csv_files(paths)
    if "rank" in fieldnames and "total_window_seconds" in fieldnames:
        return normalize_raw(label, fieldnames, rows)
    return normalize_aggregate(label, fieldnames, rows)


def run_runtime(path):
    if path is None:
        return None
    if not path.is_file():
        fail("run log does not exist: {}".format(path))
    completed = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("lob_mpi ") or line.startswith("lob_openmp "):
            completed.append(line)
    if len(completed) != 1:
        fail("expected one completed simulator line in {}".format(path))
    fields = {}
    for item in completed[0].split()[1:]:
        if "=" in item:
            name, value = item.split("=", 1)
            fields[name] = value
    if "execution_seconds" not in fields:
        fail("run log does not contain execution_seconds: {}".format(path))
    return finite_number(
        fields["execution_seconds"], "{} execution_seconds".format(path), True
    )


def ensure_matched_workload(control, treatment):
    if control["phases"] != treatment["phases"]:
        fail("control and treatment phase columns differ")
    control_rows = {row["window"]: row for row in control["rows"]}
    treatment_rows = {row["window"]: row for row in treatment["rows"]}
    if set(control_rows) != set(treatment_rows):
        fail("control and treatment window sets differ")
    if control["has_counts"] != treatment["has_counts"]:
        fail("per-window counts are present for only one layout")
    for window in sorted(control_rows):
        left = control_rows[window]
        right = treatment_rows[window]
        if (
            left["start_time"] != right["start_time"]
            or left["time"] != right["time"]
        ):
            fail("time interval differs for window {}".format(window))
        if control["has_counts"] and left["orders"] != right["orders"]:
            fail("processed-order count differs for window {}".format(window))
        if control["has_counts"] and left["trades"] != right["trades"]:
            fail("trade count differs for window {}".format(window))


def decomposition_basis(profile):
    return "critical_rank" if profile["has_critical"] else "phase_max_proxy"


def phase_values(profile, phase, kind):
    return [row["phase_" + kind][phase] for row in profile["rows"]]


def decomposition_values(profile, phase):
    kind = "critical" if profile["has_critical"] else "max"
    return phase_values(profile, phase, kind)


def phase_summary(profile, block):
    window_total = sum(row["total_max"] for row in profile["rows"])
    basis_totals = {
        phase: sum(decomposition_values(profile, phase))
        for phase in profile["phases"]
    }
    named_total = sum(basis_totals.values())
    result = []
    for phase in profile["phases"]:
        maxima = phase_values(profile, phase, "max")
        minima = (
            phase_values(profile, phase, "min") if profile["has_min"] else []
        )
        means = phase_values(profile, phase, "mean")
        critical = (
            phase_values(profile, phase, "critical")
            if profile["has_critical"] else []
        )
        basis_total = basis_totals[phase]
        result.append(
            {
                "block": block,
                "variant": profile["label"],
                "input_mode": profile["mode"],
                "decomposition_basis": decomposition_basis(profile),
                "phase": phase,
                "windows": len(profile["rows"]),
                "total_max_seconds": sum(maxima),
                "total_min_seconds": sum(minima) if minima else "",
                "total_mean_seconds": sum(means),
                "total_critical_seconds": sum(critical) if critical else "",
                "decomposition_seconds": basis_total,
                "share_of_named_phases_pct": (
                    100.0 * basis_total / named_total if named_total > 0.0 else 0.0
                ),
                "share_of_window_total_pct": (
                    100.0 * basis_total / window_total if window_total > 0.0 else 0.0
                ),
                "mean_max_seconds": statistics.mean(maxima),
                "median_max_seconds": statistics.median(maxima),
                "p95_max_seconds": percentile(maxima, 0.95),
                "p99_max_seconds": percentile(maxima, 0.99),
                "maximum_max_seconds": max(maxima),
                "mean_min_seconds": statistics.mean(minima) if minima else "",
                "median_min_seconds": statistics.median(minima) if minima else "",
                "p95_min_seconds": percentile(minima, 0.95) if minima else "",
                "p99_min_seconds": percentile(minima, 0.99) if minima else "",
                "maximum_min_seconds": max(minima) if minima else "",
                "mean_mean_seconds": statistics.mean(means),
                "median_mean_seconds": statistics.median(means),
                "p95_mean_seconds": percentile(means, 0.95),
                "p99_mean_seconds": percentile(means, 0.99),
                "maximum_mean_seconds": max(means),
                "mean_critical_seconds": (
                    statistics.mean(critical) if critical else ""
                ),
                "median_critical_seconds": (
                    statistics.median(critical) if critical else ""
                ),
                "p95_critical_seconds": (
                    percentile(critical, 0.95) if critical else ""
                ),
                "p99_critical_seconds": (
                    percentile(critical, 0.99) if critical else ""
                ),
                "maximum_critical_seconds": max(critical) if critical else "",
            }
        )
    return result


def decomposition_summary(profile, runtime, block):
    total_maxima = [row["total_max"] for row in profile["rows"]]
    total_means = [row["total_mean"] for row in profile["rows"]]
    window_total = sum(total_maxima)
    named_total = sum(
        sum(decomposition_values(profile, phase)) for phase in profile["phases"]
    )
    within_residual = window_total - named_total
    if abs(within_residual) <= closure_limit(window_total, named_total):
        within_residual = 0.0
    result = {
        "block": block,
        "variant": profile["label"],
        "input_mode": profile["mode"],
        "decomposition_basis": decomposition_basis(profile),
        "rank_count": profile["rows"][0]["rank_count"],
        "windows": len(profile["rows"]),
        "processed_orders": (
            sum(row["orders"] for row in profile["rows"])
            if profile["has_counts"] else ""
        ),
        "trades": (
            sum(row["trades"] for row in profile["rows"])
            if profile["has_counts"] else ""
        ),
        "sum_window_total_max_seconds": window_total,
        "sum_window_total_mean_seconds": sum(total_means),
        "mean_window_total_max_seconds": statistics.mean(total_maxima),
        "median_window_total_max_seconds": statistics.median(total_maxima),
        "p95_window_total_max_seconds": percentile(total_maxima, 0.95),
        "p99_window_total_max_seconds": percentile(total_maxima, 0.99),
        "sum_named_phase_seconds": named_total,
        "within_window_residual_seconds": within_residual,
        "within_window_residual_pct": (
            100.0 * within_residual / window_total
            if window_total > 0.0 else 0.0
        ),
        "execution_seconds": runtime if runtime is not None else "",
        "runtime_minus_profiled_windows_seconds": (
            runtime - window_total if runtime is not None else ""
        ),
        "runtime_minus_named_phases_seconds": (
            runtime - named_total if runtime is not None else ""
        ),
    }
    return result


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0.0 else ""


def geometric_mean(values):
    if not values or any(value <= 0.0 for value in values):
        return ""
    return math.exp(statistics.mean(math.log(value) for value in values))


def phase_ratios(control, treatment, block):
    control_window = sum(row["total_max"] for row in control["rows"])
    treatment_window = sum(row["total_max"] for row in treatment["rows"])
    total_excess = treatment_window - control_window
    rows = []
    for phase in ("window_total",) + control["phases"]:
        if phase == "window_total":
            control_values = [row["total_max"] for row in control["rows"]]
            treatment_values = [row["total_max"] for row in treatment["rows"]]
        else:
            control_values = decomposition_values(control, phase)
            treatment_values = decomposition_values(treatment, phase)
        control_total = sum(control_values)
        treatment_total = sum(treatment_values)
        phase_ratio = ratio(treatment_total, control_total)
        excess = treatment_total - control_total
        rows.append(
            {
                "block": block,
                "phase": phase,
                "control": control["label"],
                "treatment": treatment["label"],
                "control_seconds": control_total,
                "treatment_seconds": treatment_total,
                "treatment_control_ratio": phase_ratio,
                "change_pct": (
                    100.0 * (phase_ratio - 1.0) if phase_ratio != "" else ""
                ),
                "excess_seconds": excess,
                "share_of_window_time_excess_pct": (
                    100.0 * excess / total_excess
                    if phase != "window_total" and total_excess != 0.0 else ""
                ),
                "control_median_seconds": statistics.median(control_values),
                "treatment_median_seconds": statistics.median(treatment_values),
                "control_p95_seconds": percentile(control_values, 0.95),
                "treatment_p95_seconds": percentile(treatment_values, 0.95),
                "control_p99_seconds": percentile(control_values, 0.99),
                "treatment_p99_seconds": percentile(treatment_values, 0.99),
            }
        )
    return rows


def dominant_phase(profile, row):
    kind = "phase_critical" if profile["has_critical"] else "phase_max"
    return max(profile["phases"], key=lambda phase: row[kind][phase])


def window_comparison(control, treatment, block):
    control_rows = {row["window"]: row for row in control["rows"]}
    treatment_rows = {row["window"]: row for row in treatment["rows"]}
    result = []
    for window in sorted(control_rows):
        left = control_rows[window]
        right = treatment_rows[window]
        value_ratio = ratio(right["total_max"], left["total_max"])
        result.append(
            {
                "block": block,
                "window_index": window,
                "start_time_seconds": left["start_time"],
                "end_time_seconds": left["time"],
                "processed_orders": left["orders"] if control["has_counts"] else "",
                "trades": left["trades"] if control["has_counts"] else "",
                "control_total_max_seconds": left["total_max"],
                "treatment_total_max_seconds": right["total_max"],
                "treatment_control_ratio": value_ratio,
                "excess_seconds": right["total_max"] - left["total_max"],
                "control_dominant_phase": dominant_phase(control, left),
                "treatment_dominant_phase": dominant_phase(treatment, right),
            }
        )
    return result


def combine_profiles(profiles):
    reference = profiles[0]
    for profile in profiles[1:]:
        for field in ("label", "mode", "phases", "has_min", "has_critical",
                      "has_counts"):
            if profile[field] != reference[field]:
                fail("profile structure changes between matched blocks: {}".format(
                    field
                ))
        if profile["rows"][0]["rank_count"] != reference["rows"][0]["rank_count"]:
            fail("rank count changes between matched blocks")
    rows = []
    for profile in profiles:
        rows.extend(profile["rows"])
    return {
        "label": reference["label"],
        "mode": reference["mode"],
        "phases": reference["phases"],
        "has_min": reference["has_min"],
        "has_critical": reference["has_critical"],
        "has_counts": reference["has_counts"],
        "rows": rows,
    }


def slowest_windows(profile, block, limit=20):
    ordered = sorted(
        profile["rows"],
        key=lambda row: (-row["total_max"], row["window"]),
    )
    result = []
    for position, row in enumerate(ordered[:limit], start=1):
        result.append(
            {
                "block": block,
                "variant": profile["label"],
                "position": position,
                "window_index": row["window"],
                "start_time_seconds": row["start_time"],
                "end_time_seconds": row["time"],
                "critical_rank": row["critical_rank"],
                "window_total_max_seconds": row["total_max"],
                "window_total_mean_seconds": row["total_mean"],
                "dominant_phase": dominant_phase(profile, row),
            }
        )
    return result


def block_pair_summary(block, control, treatment, control_runtime,
                       treatment_runtime):
    control_total = sum(row["total_max"] for row in control["rows"])
    treatment_total = sum(row["total_max"] for row in treatment["rows"])
    profile_ratio = ratio(treatment_total, control_total)
    execution_ratio = (
        ratio(treatment_runtime, control_runtime)
        if control_runtime is not None and treatment_runtime is not None else ""
    )
    return {
        "block": block,
        "control": control["label"],
        "treatment": treatment["label"],
        "control_profiled_window_seconds": control_total,
        "treatment_profiled_window_seconds": treatment_total,
        "profiled_window_ratio": profile_ratio,
        "profiled_window_change_pct": (
            100.0 * (profile_ratio - 1.0) if profile_ratio != "" else ""
        ),
        "control_execution_seconds": (
            control_runtime if control_runtime is not None else ""
        ),
        "treatment_execution_seconds": (
            treatment_runtime if treatment_runtime is not None else ""
        ),
        "execution_ratio": execution_ratio,
        "execution_change_pct": (
            100.0 * (execution_ratio - 1.0) if execution_ratio != "" else ""
        ),
    }


def write_rows(path, rows):
    if not rows:
        fail("cannot write empty output: {}".format(path))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Summarize matched MPI and OpenMP per-window phase profiles."
    )
    parser.add_argument("output_directory", type=pathlib.Path)
    parser.add_argument("--control-label", required=True)
    parser.add_argument("--control-ranks", type=int)
    parser.add_argument(
        "--control-csv", action="append", required=True, type=pathlib.Path
    )
    parser.add_argument("--control-run", action="append", type=pathlib.Path)
    parser.add_argument("--treatment-label", required=True)
    parser.add_argument("--treatment-ranks", type=int)
    parser.add_argument(
        "--treatment-csv", action="append", required=True, type=pathlib.Path
    )
    parser.add_argument("--treatment-run", action="append", type=pathlib.Path)
    parser.add_argument("--expected-windows", type=int)
    return parser.parse_args()


def align_optional_runs(values, block_count, label):
    if values is None:
        return [None] * block_count
    if len(values) != block_count:
        fail(
            "expected {} {} run logs, found {}".format(
                block_count, label, len(values)
            )
        )
    return values


def main():
    args = parse_arguments()
    if args.control_label == args.treatment_label:
        fail("control and treatment labels must differ")
    if len(args.control_csv) != len(args.treatment_csv):
        fail("control and treatment must contain the same number of blocks")
    if (args.control_ranks is None) != (args.treatment_ranks is None):
        fail("--control-ranks and --treatment-ranks must be used together")
    if args.control_ranks is not None and (
        args.control_ranks < 1 or args.treatment_ranks < 1
    ):
        fail("expected rank counts must be positive")
    if args.expected_windows is not None and args.expected_windows < 1:
        fail("--expected-windows must be positive")
    block_count = len(args.control_csv)
    control_run_paths = align_optional_runs(
        args.control_run, block_count, "control"
    )
    treatment_run_paths = align_optional_runs(
        args.treatment_run, block_count, "treatment"
    )

    controls = []
    treatments = []
    control_runtimes = []
    treatment_runtimes = []
    for index in range(block_count):
        control_paths = expand_input_paths(
            [args.control_csv[index]], "control block {} CSV".format(index + 1)
        )
        treatment_paths = expand_input_paths(
            [args.treatment_csv[index]],
            "treatment block {} CSV".format(index + 1),
        )
        control = load_profile(args.control_label, control_paths)
        treatment = load_profile(args.treatment_label, treatment_paths)
        ensure_matched_workload(control, treatment)
        if args.control_ranks is not None:
            observed_control_ranks = control["rows"][0]["rank_count"]
            observed_treatment_ranks = treatment["rows"][0]["rank_count"]
            if observed_control_ranks != args.control_ranks:
                fail(
                    "control block {} contains {} ranks; expected {}".format(
                        index + 1, observed_control_ranks, args.control_ranks
                    )
                )
            if observed_treatment_ranks != args.treatment_ranks:
                fail(
                    "treatment block {} contains {} ranks; expected {}".format(
                        index + 1,
                        observed_treatment_ranks,
                        args.treatment_ranks,
                    )
                )
        if args.expected_windows is not None:
            for profile_name, profile in (
                ("control", control), ("treatment", treatment)
            ):
                observed_windows = len(profile["rows"])
                observed_end = profile["rows"][-1]["time"]
                if observed_windows != args.expected_windows:
                    fail(
                        "{} block {} contains {} windows; expected {}".format(
                            profile_name,
                            index + 1,
                            observed_windows,
                            args.expected_windows,
                        )
                    )
                if observed_end != float(args.expected_windows):
                    fail(
                        "{} block {} ends at {}; expected {}".format(
                            profile_name,
                            index + 1,
                            observed_end,
                            args.expected_windows,
                        )
                    )
        controls.append(control)
        treatments.append(treatment)
        control_runtimes.append(run_runtime(control_run_paths[index]))
        treatment_runtimes.append(run_runtime(treatment_run_paths[index]))

    args.output_directory.mkdir(parents=True, exist_ok=True)
    summaries = []
    decompositions = []
    ratios = []
    windows = []
    slowest = []
    block_rows = []
    for index in range(block_count):
        block = index + 1
        control = controls[index]
        treatment = treatments[index]
        control_runtime = control_runtimes[index]
        treatment_runtime = treatment_runtimes[index]
        summaries.extend(phase_summary(control, block))
        summaries.extend(phase_summary(treatment, block))
        decompositions.append(
            decomposition_summary(control, control_runtime, block)
        )
        decompositions.append(
            decomposition_summary(treatment, treatment_runtime, block)
        )
        ratios.extend(phase_ratios(control, treatment, block))
        windows.extend(window_comparison(control, treatment, block))
        slowest.extend(slowest_windows(control, block))
        slowest.extend(slowest_windows(treatment, block))
        block_rows.append(
            block_pair_summary(
                block, control, treatment, control_runtime, treatment_runtime
            )
        )

    combined_control = combine_profiles(controls)
    combined_treatment = combine_profiles(treatments)
    combined_control_runtime = (
        sum(control_runtimes) if all(
            value is not None for value in control_runtimes
        ) else None
    )
    combined_treatment_runtime = (
        sum(treatment_runtimes) if all(
            value is not None for value in treatment_runtimes
        ) else None
    )
    summaries.extend(phase_summary(combined_control, "combined"))
    summaries.extend(phase_summary(combined_treatment, "combined"))
    decompositions.append(
        decomposition_summary(
            combined_control, combined_control_runtime, "combined"
        )
    )
    decompositions.append(
        decomposition_summary(
            combined_treatment, combined_treatment_runtime, "combined"
        )
    )
    combined_ratios = phase_ratios(
        combined_control, combined_treatment, "combined"
    )
    ratios.extend(combined_ratios)
    block_rows.append(
        block_pair_summary(
            "combined",
            combined_control,
            combined_treatment,
            combined_control_runtime,
            combined_treatment_runtime,
        )
    )
    block_profile_ratios = [
        row["profiled_window_ratio"] for row in block_rows[:-1]
    ]
    block_execution_ratios = [
        row["execution_ratio"] for row in block_rows[:-1]
        if row["execution_ratio"] != ""
    ]
    paired_profile_ratio = geometric_mean(block_profile_ratios)
    paired_execution_ratio = (
        geometric_mean(block_execution_ratios)
        if len(block_execution_ratios) == block_count else ""
    )
    paired_row = {
        "blocks": block_count,
        "control": combined_control["label"],
        "treatment": combined_treatment["label"],
        "paired_geometric_profiled_window_ratio": paired_profile_ratio,
        "paired_geometric_profiled_window_change_pct": (
            100.0 * (paired_profile_ratio - 1.0)
            if paired_profile_ratio != "" else ""
        ),
        "minimum_block_profiled_window_change_pct": min(
            100.0 * (value - 1.0) for value in block_profile_ratios
        ),
        "maximum_block_profiled_window_change_pct": max(
            100.0 * (value - 1.0) for value in block_profile_ratios
        ),
        "paired_geometric_execution_ratio": paired_execution_ratio,
        "paired_geometric_execution_change_pct": (
            100.0 * (paired_execution_ratio - 1.0)
            if paired_execution_ratio != "" else ""
        ),
    }
    write_rows(args.output_directory / "phase_summary.csv", summaries)
    write_rows(args.output_directory / "decomposition.csv", decompositions)
    write_rows(args.output_directory / "phase_ratios.csv", ratios)
    write_rows(args.output_directory / "window_comparison.csv", windows)
    write_rows(args.output_directory / "slowest_windows.csv", slowest)
    write_rows(args.output_directory / "block_summary.csv", block_rows)
    write_rows(args.output_directory / "paired_summary.csv", [paired_row])

    total_ratio = combined_ratios[0]["treatment_control_ratio"]
    print(
        "window alignment and phase-closure gates: PASS for {} block(s)".format(
            block_count
        )
    )
    if combined_control["has_counts"]:
        print(
            "{} windows, {} orders, {} trades".format(
                len(combined_control["rows"]),
                decompositions[-2]["processed_orders"],
                decompositions[-2]["trades"],
            )
        )
    else:
        print(
            "{} profiled windows across blocks; per-window counts absent "
            "(full-session equivalence must be checked separately)".format(
                len(combined_control["rows"])
            )
        )
    print(
        "profiled window time: {}={:.9f}s, {}={:.9f}s, ratio={:.6f}".format(
            combined_control["label"],
            decompositions[-2]["sum_window_total_max_seconds"],
            combined_treatment["label"],
            decompositions[-1]["sum_window_total_max_seconds"],
            total_ratio,
        )
    )
    print(
        "paired geometric profiled-window change across blocks: {:+.3f}%".format(
            paired_row["paired_geometric_profiled_window_change_pct"]
        )
    )
    print(
        "slowest-rank per-window decomposition is a diagnostic proxy, "
        "not an exact full-session critical path"
    )
    print(
        "collective elapsed time includes rank-arrival waiting and MPI "
        "progress; OpenMP phase time includes scheduling and its implicit "
        "end-of-loop barrier"
    )
    ranked_excess = sorted(
        combined_ratios[1:], key=lambda row: row["excess_seconds"], reverse=True
    )
    print("largest treatment-minus-control phase contributions:")
    for row in ranked_excess[:5]:
        print(
            "  {}: {:+.9f}s ({})".format(
                row["phase"],
                row["excess_seconds"],
                (
                    "{:+.3f}% of window-time difference".format(
                        row["share_of_window_time_excess_pct"]
                    )
                    if row["share_of_window_time_excess_pct"] != ""
                    else "share unavailable"
                ),
            )
        )
    print("phase summary: {}".format(args.output_directory / "phase_summary.csv"))
    print("decomposition: {}".format(args.output_directory / "decomposition.csv"))
    print("phase ratios: {}".format(args.output_directory / "phase_ratios.csv"))
    print("block summary: {}".format(args.output_directory / "block_summary.csv"))
    print("paired summary: {}".format(args.output_directory / "paired_summary.csv"))
    print("slowest windows: {}".format(args.output_directory / "slowest_windows.csv"))
    print(
        "window comparison: {}".format(
            args.output_directory / "window_comparison.csv"
        )
    )


if __name__ == "__main__":
    main()
