#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Describe full-session liquidity effects across the ten frozen clusters.

The primary financial endpoint remains the 30-minute market-wide time series
written by ``analyze_fragmented_shared_liquidity_case.py``.  This script uses
per-asset fixed-clock summaries to show whether the paired effect is larger in
some predeclared liquidity clusters.  Because each summary averages the full
session, including the pre-shock half, these estimates are descriptive and
diluted relative to the primary post-shock endpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict
from typing import Mapping, Sequence


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_fragmented_shared_liquidity_case as primary  # noqa: E402


class ClusterAnalysisError(ValueError):
    """Raised for incomplete, mixed or tampered cluster evidence."""


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))
    except OSError as error:
        raise ClusterAnalysisError(f"cannot read {path}: {error}") from error


def require_artifact(row: Mapping[str, str], field: str) -> pathlib.Path:
    text = row.get(field, "").strip()
    if not text:
        raise ClusterAnalysisError(f"raw row lacks {field}")
    path = pathlib.Path(text).resolve()
    if not path.is_file():
        raise ClusterAnalysisError(f"missing {field}: {path}")
    expected = row.get(f"{field}_sha256", "")
    if primary.sha256_file(path) != expected:
        raise ClusterAnalysisError(f"SHA-256 mismatch for {field}: {path}")
    return path


def load_universe(path: pathlib.Path) -> dict[str, float]:
    rows = read_csv(path)
    if not rows:
        raise ClusterAnalysisError("universe configuration is empty")
    midpoints: dict[str, float] = {}
    for number, row in enumerate(rows, start=2):
        symbol = row.get("symbol", "").strip()
        if not symbol or symbol in midpoints:
            raise ClusterAnalysisError(
                f"invalid or duplicate universe symbol at line {number}"
            )
        try:
            bid = float(row["initial_best_bid_ticks"])
            ask = float(row["initial_best_ask_ticks"])
        except (KeyError, ValueError) as error:
            raise ClusterAnalysisError(
                f"invalid opening midpoint for {symbol}"
            ) from error
        midpoint = 0.5 * (bid + ask)
        if not math.isfinite(midpoint) or midpoint <= 0.0:
            raise ClusterAnalysisError(f"non-positive opening midpoint for {symbol}")
        midpoints[symbol] = midpoint
    return midpoints


def load_clusters(path: pathlib.Path, expected: set[str]) -> dict[str, int]:
    rows = read_csv(path)
    clusters: dict[str, int] = {}
    for number, row in enumerate(rows, start=2):
        symbol = row.get("symbol", "").strip()
        try:
            cluster = int(row.get("cluster_id", ""))
        except ValueError as error:
            raise ClusterAnalysisError(
                f"invalid cluster at {path}:{number}"
            ) from error
        if symbol in clusters or cluster not in range(10):
            raise ClusterAnalysisError(
                f"duplicate symbol or out-of-range cluster at {path}:{number}"
            )
        clusters[symbol] = cluster
    if set(clusters) != expected or set(clusters.values()) != set(range(10)):
        raise ClusterAnalysisError(
            "cluster assignments do not cover the exact universe and clusters 0--9"
        )
    return clusters


def target_mask(row: Mapping[str, str], expected: set[str]) -> set[str]:
    path = require_artifact(row, "shock_targets_csv")
    targets: set[str] = set()
    observed: set[str] = set()
    for number, item in enumerate(read_csv(path), start=2):
        symbol = item.get("symbol", "").strip()
        if symbol in observed:
            raise ClusterAnalysisError(f"duplicate target row at {path}:{number}")
        observed.add(symbol)
        try:
            selected = int(item.get("is_shock_target", ""))
        except ValueError as error:
            raise ClusterAnalysisError(
                f"invalid target flag at {path}:{number}"
            ) from error
        if selected not in {0, 1}:
            raise ClusterAnalysisError(f"invalid target flag at {path}:{number}")
        if selected:
            targets.add(symbol)
    if observed != expected:
        raise ClusterAnalysisError(f"target mask does not cover the universe: {path}")
    return targets


def cluster_snapshot(
    row: Mapping[str, str], midpoints: Mapping[str, float],
    clusters: Mapping[str, int], targets: set[str],
) -> dict[int, dict[str, float]]:
    path = require_artifact(row, "asset_summary_csv")
    grouped: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"depth": [], "spread_bps": []}
    )
    observed: set[str] = set()
    for number, item in enumerate(read_csv(path), start=2):
        symbol = item.get("symbol", "").strip()
        if symbol in observed or symbol not in midpoints:
            raise ClusterAnalysisError(
                f"duplicate or unexpected asset at {path}:{number}"
            )
        observed.add(symbol)
        if symbol in targets:
            continue
        try:
            depth = float(item["mean_bid_depth"]) + float(item["mean_ask_depth"])
            spread_ticks = float(item["mean_spread_ticks"])
        except (KeyError, ValueError) as error:
            raise ClusterAnalysisError(
                f"invalid liquidity moment at {path}:{number}"
            ) from error
        if not all(math.isfinite(value) for value in (depth, spread_ticks)):
            raise ClusterAnalysisError(f"non-finite liquidity moment for {symbol}")
        cluster = clusters[symbol]
        grouped[cluster]["depth"].append(depth)
        grouped[cluster]["spread_bps"].append(
            10_000.0 * spread_ticks / midpoints[symbol]
        )
    if observed != set(midpoints) or set(grouped) != set(range(10)):
        raise ClusterAnalysisError(f"asset summary is incomplete: {path}")
    return {
        cluster: {
            "symbol_count": float(len(values["depth"])),
            "mean_top_depth": statistics.fmean(values["depth"]),
            "mean_spread_bps": statistics.fmean(values["spread_bps"]),
        }
        for cluster, values in grouped.items()
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-raw", type=pathlib.Path, required=True)
    parser.add_argument("--uncoupled-raw", type=pathlib.Path, required=True)
    parser.add_argument("--shared-off-raw", type=pathlib.Path, required=True)
    parser.add_argument("--universe-config", type=pathlib.Path, required=True)
    parser.add_argument("--cluster-assignments", type=pathlib.Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    if args.rank <= 0:
        raise SystemExit("--rank must be positive")

    try:
        global_cases = primary.read_raw(args.global_raw.resolve(), "global", args.rank)
        uncoupled_cases = primary.read_raw(
            args.uncoupled_raw.resolve(), "uncoupled", args.rank
        )
        off_cases = primary.read_raw(args.shared_off_raw.resolve(), "off", args.rank)
        metadata = primary.ensure_common_metadata(
            [*global_cases, *uncoupled_cases, *off_cases]
        )
        cluster_path = args.cluster_assignments.resolve()
        if not cluster_path.is_file():
            raise ClusterAnalysisError(f"missing cluster assignments: {cluster_path}")
        if primary.sha256_file(cluster_path) != metadata["shock_cluster_sha256"]:
            raise ClusterAnalysisError(
                "cluster assignments differ from the hash-bound shock mapping"
            )
        universe_path = args.universe_config.resolve()
        if primary.sha256_file(universe_path) != metadata["input_config_sha256"]:
            raise ClusterAnalysisError(
                "universe configuration differs from the raw campaign input"
            )
        midpoints = load_universe(universe_path)
        if len(midpoints) != 1480:
            raise ClusterAnalysisError(
                f"expected the certified 1,480-symbol universe, found {len(midpoints)}"
            )
        clusters = load_clusters(cluster_path, set(midpoints))

        global_index = primary.unique_cases(global_cases, "global")
        uncoupled_index = primary.unique_off_cases(uncoupled_cases)
        off_index = primary.unique_off_cases(off_cases)
        risks = sorted(
            {primary.canonical_risk(case.risk_limit) for case in global_cases},
            key=float,
        )
        seeds = sorted({case.seed for case in global_cases})
        if not seeds:
            raise ClusterAnalysisError("no paired seeds")

        snapshot_cache: dict[tuple[str, int, bool], dict[int, dict[str, float]]] = {}
        canonical_targets: set[str] | None = None

        def snapshot(case: primary.RawCase) -> dict[int, dict[str, float]]:
            nonlocal canonical_targets
            key = (str(case.metrics_path), case.seed, case.shock)
            if key not in snapshot_cache:
                targets = target_mask(case.row, set(midpoints))
                if canonical_targets is None:
                    canonical_targets = targets
                elif targets != canonical_targets:
                    raise ClusterAnalysisError("target mask differs across paired paths")
                snapshot_cache[key] = cluster_snapshot(
                    case.row, midpoints, clusters, targets
                )
            return snapshot_cache[key]

        per_seed: list[dict[str, object]] = []
        for risk in risks:
            for seed in seeds:
                cases = {
                    "global_control": global_index.get((risk, seed, False)),
                    "global_shock": global_index.get((risk, seed, True)),
                    "uncoupled_control": uncoupled_index.get((seed, False)),
                    "uncoupled_shock": uncoupled_index.get((seed, True)),
                    "off_control": off_index.get((seed, False)),
                    "off_shock": off_index.get((seed, True)),
                }
                if any(value is None for value in cases.values()):
                    raise ClusterAnalysisError(
                        f"incomplete pair for risk={risk}, seed={seed}"
                    )
                values = {name: snapshot(case) for name, case in cases.items() if case}
                for cluster in range(10):
                    gc = values["global_control"][cluster]
                    gs = values["global_shock"][cluster]
                    uc = values["uncoupled_control"][cluster]
                    us = values["uncoupled_shock"][cluster]
                    oc = values["off_control"][cluster]
                    os = values["off_shock"][cluster]
                    global_depth = gc["mean_top_depth"] - gs["mean_top_depth"]
                    uncoupled_depth = uc["mean_top_depth"] - us["mean_top_depth"]
                    global_spread = gs["mean_spread_bps"] - gc["mean_spread_bps"]
                    uncoupled_spread = us["mean_spread_bps"] - uc["mean_spread_bps"]
                    per_seed.append({
                        "risk_limit_per_asset": risk,
                        "seed": seed,
                        "cluster_id": cluster,
                        "non_target_symbol_count": int(gc["symbol_count"]),
                        "baseline_uncoupled_top_depth": uc["mean_top_depth"],
                        "baseline_uncoupled_spread_bps": uc["mean_spread_bps"],
                        "global_depth_deterioration": global_depth,
                        "uncoupled_depth_deterioration": uncoupled_depth,
                        "depth_difference_in_differences": (
                            global_depth - uncoupled_depth
                        ),
                        "global_spread_deterioration_bps": global_spread,
                        "uncoupled_spread_deterioration_bps": uncoupled_spread,
                        "spread_difference_in_differences_bps": (
                            global_spread - uncoupled_spread
                        ),
                        "shared_off_depth_deterioration": (
                            oc["mean_top_depth"] - os["mean_top_depth"]
                        ),
                        "shared_off_spread_deterioration_bps": (
                            os["mean_spread_bps"] - oc["mean_spread_bps"]
                        ),
                    })

        summaries: list[dict[str, object]] = []
        for risk in risks:
            for cluster in range(10):
                rows = [
                    row for row in per_seed
                    if row["risk_limit_per_asset"] == risk
                    and row["cluster_id"] == cluster
                ]
                record: dict[str, object] = {
                    "risk_limit_per_asset": risk,
                    "cluster_id": cluster,
                    "seed_count": len(rows),
                    "non_target_symbol_count": rows[0]["non_target_symbol_count"],
                    "mean_baseline_uncoupled_top_depth": statistics.fmean(
                        float(row["baseline_uncoupled_top_depth"]) for row in rows
                    ),
                    "mean_baseline_uncoupled_spread_bps": statistics.fmean(
                        float(row["baseline_uncoupled_spread_bps"]) for row in rows
                    ),
                }
                for field in (
                    "depth_difference_in_differences",
                    "spread_difference_in_differences_bps",
                ):
                    for label, value in summarize(
                        [float(row[field]) for row in rows]
                    ).items():
                        record[f"{field}_{label}"] = value
                summaries.append(record)

        output = args.output_dir.resolve()
        if output.exists() and any(output.iterdir()):
            raise ClusterAnalysisError(
                f"refusing non-empty output directory: {output}"
            )
        output.mkdir(parents=True, exist_ok=True)
        primary.atomic_csv(
            output / "cluster_liquidity_effects_by_seed.csv",
            list(per_seed[0]), per_seed,
        )
        primary.atomic_csv(
            output / "cluster_liquidity_effect_summary.csv",
            list(summaries[0]), summaries,
        )
        manifest = {
            "schema_version": 1,
            "analysis_role": "descriptive_full_session_cluster_heterogeneity",
            "primary_endpoint": False,
            "interpretation": (
                "Full-session fixed-clock effects in non-target books; diluted "
                "by the pre-shock half and not a replacement for the 30-minute "
                "market-wide endpoint."
            ),
            "cluster_count": 10,
            "seed_count": len(seeds),
            "target_symbol_count": len(canonical_targets or set()),
            "cluster_assignments": str(cluster_path),
            "cluster_assignments_sha256": primary.sha256_file(cluster_path),
            "outputs": {
                "by_seed": "cluster_liquidity_effects_by_seed.csv",
                "summary": "cluster_liquidity_effect_summary.csv",
            },
        }
        primary.atomic_json(output / "cluster_analysis_manifest.json", manifest)
        print(json.dumps({"output_dir": str(output), **manifest}, sort_keys=True))
    except (primary.AnalysisError, ClusterAnalysisError) as error:
        print(f"cluster-liquidity analysis failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
