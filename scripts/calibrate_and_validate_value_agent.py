#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Fit four-book value-agent parameters and validate them chronologically.

This runner implements *diagonal weighted standardized moment matching*, not
an unconstrained black-box fit.  The primary selection residual for asset ``a``
and moment ``m`` is

``(mean_sim - empirical_target) / empirical_scale``.

The empirical target scale is the five-minute delete-block jackknife standard
error emitted by the ITCH extractor.  Repeated simulation seeds additionally
produce a Monte-Carlo standard error; the report retains the secondary
uncertainty-aware diagnostic with denominator
``sqrt(empirical_scale**2 + mc_se**2)``.

The script enforces a chronological split: it selects parameters only from
training-day targets, freezes them, and subsequently evaluates the *same agent
configuration* on the held-out opening state while retaining training-day
background inputs.  Held-out targets are opened only after selection and cannot
change the parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import pathlib
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


SYMBOLS = ("QQQ", "AAPL", "MSFT", "AMZN")
METRICS = (
    "mean_spread_ticks",
    "mean_bid_depth",
    "mean_ask_depth",
    "mid_move_rate",
    "return_variance",
    "return_kurtosis",
    "absolute_return_acf1",
)


@dataclass(frozen=True)
class TargetMoment:
    """One empirical moment and its predeclared diagonal WMM weight."""

    target: float
    empirical_scale: float
    weight: float


@dataclass(frozen=True)
class MomentEstimate:
    """A seed-averaged simulated moment used in the WMM objective."""

    symbol: str
    metric: str
    target: float
    empirical_scale: float
    weight: float
    simulated_mean: float
    simulated_sample_sd: float
    simulated_mean_se: float
    combined_scale: float
    empirical_standardized_residual: float
    combined_uncertainty_residual: float
    objective_residual: float
    weighted_squared_residual: float
    seed_count: int


def compact(date: str) -> str:
    result = date.replace("-", "")
    if len(result) != 8 or not result.isdigit():
        raise ValueError(f"invalid ISO date: {date}")
    return result


def csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_backgrounds(training_config: pathlib.Path,
                                heldout_config: pathlib.Path,
                                training_date: str,
                                heldout_date: str) -> None:
    """Reject accidental refitting of nuisance parameters on validation data."""
    training_rows = {row["symbol"]: row for row in csv_rows(training_config)}
    heldout_rows = {row["symbol"]: row for row in csv_rows(heldout_config)}
    if set(training_rows) != set(SYMBOLS) or set(heldout_rows) != set(SYMBOLS):
        raise ValueError("training and held-out configs must contain exactly four symbols")
    frozen_fields = (
        "data_dir", "hawkes_rates_file", "basket_weight",
        "market_maker_quote_quantity", "target_spread_ticks",
        "quote_improvement_probability",
    )
    training_token = compact(training_date)
    heldout_token = compact(heldout_date)
    for symbol in SYMBOLS:
        for field in frozen_fields:
            if training_rows[symbol].get(field) != heldout_rows[symbol].get(field):
                raise ValueError(
                    f"held-out config refits {symbol} field {field}; "
                    "background/quoting inputs must be frozen from training"
                )
        for field in ("data_dir", "hawkes_rates_file"):
            value = heldout_rows[symbol][field]
            if training_token not in value or heldout_token in value:
                raise ValueError(
                    f"held-out {symbol} {field} does not point exclusively "
                    f"to training date {training_date}: {value}"
                )


def load_targets(
    root: pathlib.Path,
    date: str,
    *,
    window_seconds: int | None = None,
) -> dict[str, dict[str, TargetMoment]]:
    """Load full-session or explicitly matched empirical-prefix targets.

    A 300-second simulation is a preliminary structural screen, not a cheap
    estimate of a full-day loss.  It must therefore use the first 300 seconds
    of the same ITCH session.  ``window_seconds=None`` denotes the established
    full-session target artifact.
    """
    if window_seconds is not None and window_seconds <= 0:
        raise ValueError("target window seconds must be positive")
    date_value = compact(date)
    result: dict[str, dict[str, TargetMoment]] = {}
    for symbol in SYMBOLS:
        lower = symbol.lower()
        suffix = (
            "" if window_seconds is None
            else f"_window_{window_seconds}s"
        )
        path = root / f"itch_{date_value}_{lower}" / (
            f"market_targets_{lower}_{date_value}{suffix}.csv"
        )
        if not path.is_file():
            if window_seconds is None:
                raise FileNotFoundError(f"missing full-session target file: {path}")
            raise FileNotFoundError(
                f"missing {window_seconds}-second matched target file: {path}; "
                "rerun the ITCH extractor with --target-window-seconds "
                f"{window_seconds}"
            )
        values: dict[str, TargetMoment] = {}
        for row in csv_rows(path):
            name = row["name"]
            target = float(row["target"])
            scale = float(row["scale"])
            weight = float(row.get("weight", "1.0"))
            if (not math.isfinite(target) or not math.isfinite(scale)
                    or not math.isfinite(weight) or scale <= 0.0
                    or weight <= 0.0):
                raise ValueError(
                    f"{path} contains an invalid target, scale, or positive weight "
                    f"for {name}"
                )
            values[name] = TargetMoment(target, scale, weight)
        missing = set(METRICS).difference(values)
        if missing:
            raise ValueError(f"{path} is missing target metrics: {sorted(missing)}")
        result[symbol] = values
    return result


def summary_rows(summary_path: pathlib.Path) -> dict[str, dict[str, str]]:
    rows = {row["symbol"]: row for row in csv_rows(summary_path)}
    if set(rows) != set(SYMBOLS):
        raise ValueError(f"{summary_path} does not contain exactly {SYMBOLS}")
    for symbol, row in rows.items():
        if ("sample_count" in row and "expected_sample_count" in row
                and row["sample_count"] != row["expected_sample_count"]):
            raise ValueError(
                f"{summary_path} has incomplete fixed-clock samples for {symbol}"
            )
    return rows


def weighted_moment_loss(
    summary_paths: Sequence[pathlib.Path],
    targets: Mapping[str, Mapping[str, TargetMoment]],
    *,
    residual_cap: float | None = None,
    uncertainty_mode: str = "empirical",
) -> tuple[float, list[MomentEstimate]]:
    """Return a diagonal WMM loss and its complete auditable decomposition.

    ``residual_cap`` is allowed only for the inexpensive screening stage.  The
    final training and held-out losses must be uncapped, so a poor high-order
    moment remains visible in the thesis table rather than being hidden by a
    numerical convenience.

    The default objective uses the empirical jackknife scale only.  This is
    deliberate: placing a parameter-dependent Monte-Carlo standard error in a
    selection denominator can reward a noisy candidate.  The ``combined`` mode
    is retained for a secondary uncertainty-aware diagnostic in the report.
    """
    if not summary_paths:
        raise ValueError("weighted moment matching requires at least one summary")
    if residual_cap is not None and (
            not math.isfinite(residual_cap) or residual_cap <= 0.0):
        raise ValueError("residual_cap must be finite and positive when supplied")
    if uncertainty_mode not in {"empirical", "combined"}:
        raise ValueError("uncertainty_mode must be empirical or combined")

    summaries = [summary_rows(path) for path in summary_paths]
    for rows in summaries:
        if any(rows[symbol].get("structurally_valid") != "1" for symbol in SYMBOLS):
            return math.inf, []

    total_weight = 0.0
    weighted_squared = 0.0
    estimates: list[MomentEstimate] = []
    for symbol in SYMBOLS:
        for metric in METRICS:
            target = targets[symbol][metric]
            values = [float(rows[symbol][metric]) for rows in summaries]
            if not all(math.isfinite(value) for value in values):
                return math.inf, []
            simulated_mean = statistics.fmean(values)
            simulated_sample_sd = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
            simulated_mean_se = simulated_sample_sd / math.sqrt(len(values))
            combined_scale = math.hypot(
                target.empirical_scale, simulated_mean_se,
            )
            if not math.isfinite(combined_scale) or combined_scale <= 0.0:
                return math.inf, []
            empirical_standardized = (
                (simulated_mean - target.target) / target.empirical_scale
            )
            combined_uncertainty = (
                (simulated_mean - target.target) / combined_scale
            )
            standardized = (
                empirical_standardized if uncertainty_mode == "empirical"
                else combined_uncertainty
            )
            objective_residual = (
                max(-residual_cap, min(residual_cap, standardized))
                if residual_cap is not None else standardized
            )
            contribution = target.weight * objective_residual * objective_residual
            estimates.append(MomentEstimate(
                symbol=symbol,
                metric=metric,
                target=target.target,
                empirical_scale=target.empirical_scale,
                weight=target.weight,
                simulated_mean=simulated_mean,
                simulated_sample_sd=simulated_sample_sd,
                simulated_mean_se=simulated_mean_se,
                combined_scale=combined_scale,
                empirical_standardized_residual=empirical_standardized,
                combined_uncertainty_residual=combined_uncertainty,
                objective_residual=objective_residual,
                weighted_squared_residual=contribution,
                seed_count=len(values),
            ))
            total_weight += target.weight
            weighted_squared += contribution
    if total_weight <= 0.0:
        return math.inf, []
    return math.sqrt(weighted_squared / total_weight), estimates


def standardized_rmse(summary_path: pathlib.Path,
                      targets: Mapping[str, Mapping[str, TargetMoment]]) -> float:
    """Legacy-compatible one-seed empirical-scale WMM screening score."""
    return weighted_moment_loss([summary_path], targets)[0]


@dataclass(frozen=True)
class Candidate:
    threshold_bps: float
    response_step_bps: float
    base_order_quantity: int
    volatility_bps_sqrt_second: float


def candidate_arguments(candidate: Candidate, args: argparse.Namespace) -> list[str]:
    return [
        "--enable-value-agent",
        "--value-threshold-bps", str(candidate.threshold_bps),
        "--value-response-bps", str(candidate.response_step_bps),
        "--value-base-quantity", str(candidate.base_order_quantity),
        "--value-max-quantity", str(args.max_order_quantity),
        "--value-max-inventory", str(args.max_inventory),
        "--value-fundamental-volatility-bps",
        str(candidate.volatility_bps_sqrt_second),
        "--value-interval-ms", str(args.decision_interval_ms),
    ]


def run_model(binary: pathlib.Path, config: pathlib.Path, output_dir: pathlib.Path,
              duration: int, seed: int, candidate: Candidate,
              args: argparse.Namespace, coupling_mode: str) -> tuple[pathlib.Path, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "--duration-seconds", str(duration),
        "--seed", str(seed),
        "--book-config-file", str(config),
        "--output-dir", str(output_dir),
        *candidate_arguments(candidate, args),
    ]
    if coupling_mode in {"etf", "etf_shared_mm_hedging"}:
        command.extend([
            "--enable-etf-arbitrage",
            "--arbitrage-trigger-bps", str(args.arbitrage_trigger_bps),
            "--arbitrage-release-bps", str(args.arbitrage_release_bps),
        ])
    if coupling_mode in {"shared_mm_hedging", "etf_shared_mm_hedging"}:
        command.extend([
            "--exposure-threshold", str(args.coupled_exposure_threshold),
            "--max-hedge-quantity", str(args.max_hedge_quantity),
            "--enable-shared-mm-hedging",
        ])
    else:
        command.extend(["--exposure-threshold", "1000000000000000"])
    started = time.monotonic()
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout_seconds,
        check=False,
    )
    wall_seconds = time.monotonic() - started
    with (output_dir / "run.log").open("w") as output:
        output.write("command=" + json.dumps(command) + "\n")
        output.write(f"wall_seconds_external={wall_seconds:.9f}\n")
        output.write(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"simulator failed with status {completed.returncode}; "
            f"see {output_dir / 'run.log'}"
        )
    return output_dir / "sequential_multi_asset_summary.csv", wall_seconds


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def evaluate_replicates(
    *,
    binary: pathlib.Path,
    config: pathlib.Path,
    output_dir: pathlib.Path,
    duration: int,
    seeds: Sequence[int],
    candidate: Candidate,
    args: argparse.Namespace,
    coupling_mode: str,
    targets: Mapping[str, Mapping[str, TargetMoment]],
) -> dict[str, object]:
    """Run common random-number replications and evaluate their mean moments."""
    summaries: list[pathlib.Path] = []
    wall_seconds: list[float | None] = []
    errors: list[str] = []
    for seed in seeds:
        try:
            summary, elapsed = run_model(
                binary, config, output_dir / f"seed_{seed}", duration, seed,
                candidate, args, coupling_mode,
            )
            summaries.append(summary)
            wall_seconds.append(elapsed)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            wall_seconds.append(None)
            errors.append(str(exc))

    if errors or len(summaries) != len(seeds):
        return {
            "fit_wsmrmse": math.inf,
            "combined_uncertainty_wsmrmse": math.inf,
            "seed_wall_seconds": wall_seconds,
            "summary_paths": [str(path) for path in summaries],
            "errors": errors,
            "moment_estimates": [],
        }

    fit_score, estimates = weighted_moment_loss(summaries, targets)
    combined_score = math.inf
    if len(summaries) >= 2 and math.isfinite(fit_score):
        combined_score, _ = weighted_moment_loss(
            summaries, targets, uncertainty_mode="combined",
        )
    return {
        "fit_wsmrmse": fit_score,
        "combined_uncertainty_wsmrmse": combined_score,
        "seed_wall_seconds": wall_seconds,
        "summary_paths": [str(path) for path in summaries],
        "errors": errors,
        "moment_estimates": [asdict(estimate) for estimate in estimates],
    }


def evaluation_for_report(evaluation: Mapping[str, object]) -> dict[str, object]:
    """Convert an in-memory evaluation into strict JSON without NaN/Infinity."""
    fit_score = float(evaluation["fit_wsmrmse"])
    combined_score = float(evaluation["combined_uncertainty_wsmrmse"])
    return {
        "fit_wsmrmse": finite_or_none(fit_score),
        "combined_uncertainty_wsmrmse": finite_or_none(combined_score),
        "seed_count": len(evaluation["summary_paths"]),
        "seed_wall_seconds": evaluation["seed_wall_seconds"],
        "summary_paths": evaluation["summary_paths"],
        "errors": evaluation["errors"],
        "moment_estimates": evaluation["moment_estimates"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build/sequential_multi_asset_lob")
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--heldout-config", required=True)
    parser.add_argument("--training-date", required=True)
    parser.add_argument("--heldout-date", required=True)
    parser.add_argument("--target-root", default="data")
    parser.add_argument("--output-dir", default="results/value_agent_heldout")
    parser.add_argument(
        "--stage1-duration", type=int, default=300,
        help="short structural screen in seconds; matched to its ITCH prefix",
    )
    parser.add_argument(
        "--stage2-duration", type=int, default=3_600,
        help="intermediate refinement horizon in seconds; matched to its ITCH prefix",
    )
    parser.add_argument(
        "--stage3-duration", type=int, default=23_400,
        help="full-session selection and held-out horizon in seconds",
    )
    parser.add_argument(
        "--session-duration", type=int, default=23_400,
        help="regular-session duration represented by the unsuffixed target CSV",
    )
    # Compatibility aliases for the previous two-stage command line.  They
    # intentionally override only stage 1 and stage 3; stage 2 remains an
    # explicit intermediate filter in the revised protocol.
    parser.add_argument("--screen-duration", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--full-duration", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--stage1-top-candidates", type=int, default=12)
    parser.add_argument("--stage2-top-candidates", type=int, default=4)
    parser.add_argument("--top-candidates", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument(
        "--stage1-seeds", type=int, nargs="+", default=[1729],
        help="common random-number seeds for the short candidate screen",
    )
    parser.add_argument(
        "--stage2-seeds", type=int, nargs="+", default=[1729, 7919],
        help="common random-number seeds for the intermediate refinement",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[1729, 7919, 1103, 6599, 2027],
        help="full-day common random-number seeds for stage 3 and held-out validation",
    )
    parser.add_argument("--screen-seeds", type=int, nargs="+", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[5.0, 10.0, 20.0])
    parser.add_argument("--response-steps", type=float, nargs="+", default=[2.5, 5.0])
    parser.add_argument("--base-quantities", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--volatilities", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--max-order-quantity", type=int, default=1_000)
    parser.add_argument("--max-inventory", type=int, default=2_000_000)
    parser.add_argument("--decision-interval-ms", type=float, default=1000.0)
    parser.add_argument("--arbitrage-trigger-bps", type=float, default=5.0)
    parser.add_argument("--arbitrage-release-bps", type=float, default=2.5)
    parser.add_argument("--coupled-exposure-threshold", type=float, default=500.0)
    parser.add_argument("--max-hedge-quantity", type=int, default=1_000)
    parser.add_argument(
        "--coupling-mode",
        choices=("local", "etf", "shared_mm_hedging", "etf_shared_mm_hedging"),
        default="etf",
        help=("agent configuration used identically for screening, selection, "
              "training evaluation, and held-out validation"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()

    if args.screen_duration is not None:
        args.stage1_duration = args.screen_duration
    if args.full_duration is not None:
        args.stage3_duration = args.full_duration
    if args.top_candidates is not None:
        args.stage2_top_candidates = args.top_candidates
    if args.screen_seeds is not None:
        args.stage1_seeds = args.screen_seeds

    if args.training_date >= args.heldout_date:
        parser.error("training date must be earlier than held-out date")
    if (args.stage1_duration <= 0 or args.stage2_duration <= 0
            or args.stage3_duration <= 0 or args.session_duration <= 0):
        parser.error("durations must be positive")
    if not (args.stage1_duration < args.stage2_duration < args.stage3_duration):
        parser.error("require stage1-duration < stage2-duration < stage3-duration")
    if args.stage3_duration != args.session_duration:
        parser.error(
            "stage3-duration must equal session-duration so selection and held-out "
            "validation use full-session targets"
        )
    if (args.stage1_top_candidates <= 0 or args.stage2_top_candidates <= 0
            or not args.seeds or not args.stage1_seeds or not args.stage2_seeds):
        parser.error("candidate counts and each seed list must be positive/non-empty")
    all_seed_lists = (args.stage1_seeds, args.stage2_seeds, args.seeds)
    if any(len(set(values)) != len(values) for values in all_seed_lists):
        parser.error("each stage's seed list must contain unique values")
    if len(args.stage2_seeds) < 2 or len(args.seeds) < 2:
        parser.error("stages 2 and 3 require at least two seeds for Monte-Carlo uncertainty")

    binary = pathlib.Path(args.binary).resolve()
    training_config = pathlib.Path(args.training_config).resolve()
    heldout_config = pathlib.Path(args.heldout_config).resolve()
    output_root = pathlib.Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    validate_frozen_backgrounds(
        training_config, heldout_config,
        args.training_date, args.heldout_date,
    )
    target_root = pathlib.Path(args.target_root)
    stage1_targets = load_targets(
        target_root, args.training_date, window_seconds=args.stage1_duration,
    )
    stage2_targets = load_targets(
        target_root, args.training_date, window_seconds=args.stage2_duration,
    )
    stage3_targets = load_targets(target_root, args.training_date)

    candidates = [
        Candidate(*values)
        for values in itertools.product(
            args.thresholds,
            args.response_steps,
            args.base_quantities,
            args.volatilities,
        )
    ]
    stage1_screening: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        evaluation = evaluate_replicates(
            binary=binary,
            config=training_config,
            output_dir=output_root / "stage1_screen" / f"candidate_{index:03d}",
            duration=args.stage1_duration,
            seeds=args.stage1_seeds,
            candidate=candidate,
            args=args,
            coupling_mode=args.coupling_mode,
            targets=stage1_targets,
        )
        stage1_screening.append({
            "candidate_index": index,
            "candidate": candidate,
            "evaluation": evaluation,
        })
    stage1_screening.sort(
        key=lambda item: float(
            item["evaluation"]["fit_wsmrmse"]  # type: ignore[index]
        )
    )
    eligible_stage1 = [
        item for item in stage1_screening
        if math.isfinite(float(item["evaluation"]["fit_wsmrmse"]))  # type: ignore[index]
    ]
    if not eligible_stage1:
        raise RuntimeError("no structurally valid stage-1 training candidate")
    stage1_finalists = eligible_stage1[
        : min(args.stage1_top_candidates, len(eligible_stage1))
    ]

    stage2_refinement: list[dict[str, object]] = []
    for finalist in stage1_finalists:
        index = int(finalist["candidate_index"])
        candidate = finalist["candidate"]
        assert isinstance(candidate, Candidate)
        evaluation = evaluate_replicates(
            binary=binary,
            config=training_config,
            output_dir=output_root / "stage2_refinement" / f"candidate_{index:03d}",
            duration=args.stage2_duration,
            seeds=args.stage2_seeds,
            candidate=candidate,
            args=args,
            coupling_mode=args.coupling_mode,
            targets=stage2_targets,
        )
        stage2_refinement.append({
            "candidate_index": index,
            "candidate": candidate,
            "evaluation": evaluation,
        })
    stage2_refinement.sort(
        key=lambda item: float(
            item["evaluation"]["fit_wsmrmse"]  # type: ignore[index]
        )
    )
    eligible_stage2 = [
        item for item in stage2_refinement
        if math.isfinite(float(item["evaluation"]["fit_wsmrmse"]))  # type: ignore[index]
    ]
    if not eligible_stage2:
        raise RuntimeError("every stage-2 training finalist failed")
    stage2_finalists = eligible_stage2[
        : min(args.stage2_top_candidates, len(eligible_stage2))
    ]

    stage3_training: list[dict[str, object]] = []
    for finalist in stage2_finalists:
        index = int(finalist["candidate_index"])
        candidate = finalist["candidate"]
        assert isinstance(candidate, Candidate)
        evaluation = evaluate_replicates(
            binary=binary,
            config=training_config,
            output_dir=output_root / "stage3_full_training" / f"candidate_{index:03d}",
            duration=args.stage3_duration,
            seeds=args.seeds,
            candidate=candidate,
            args=args,
            coupling_mode=args.coupling_mode,
            targets=stage3_targets,
        )
        stage3_training.append({
            "candidate_index": index,
            "candidate": candidate,
            "evaluation": evaluation,
        })
    stage3_training.sort(
        key=lambda item: float(
            item["evaluation"]["fit_wsmrmse"]  # type: ignore[index]
        )
    )
    if not math.isfinite(float(stage3_training[0]["evaluation"]["fit_wsmrmse"])):  # type: ignore[index]
        raise RuntimeError("every stage-3 full-session training finalist failed")
    selected = stage3_training[0]
    selected_candidate = selected["candidate"]
    assert isinstance(selected_candidate, Candidate)

    # Selection is now complete.  Only from this point onward is the held-out
    # target file opened, and selected_candidate is never mutated.
    heldout_targets = load_targets(target_root, args.heldout_date)
    # Selection already used these exact full-day runs, seeds, inputs and
    # coupling mode.  Reusing them avoids wasting an extra five full paths and
    # ensures that the reported training fit is exactly the selected fit.
    selected_training_evaluation = selected["evaluation"]
    assert isinstance(selected_training_evaluation, dict)
    heldout_evaluation = evaluate_replicates(
        binary=binary,
        config=heldout_config,
        output_dir=output_root / "heldout_validation",
        duration=args.stage3_duration,
        seeds=args.seeds,
        candidate=selected_candidate,
        args=args,
        coupling_mode=args.coupling_mode,
        targets=heldout_targets,
    )

    report = {
        "protocol": {
            "training_date": args.training_date,
            "heldout_date": args.heldout_date,
            "screen_duration_seconds": args.stage1_duration,
            "full_duration_seconds": args.stage3_duration,
            "three_stage_protocol": {
                "stage1": {
                    "purpose": "short structural screen",
                    "duration_seconds": args.stage1_duration,
                    "empirical_target": "matching first-session prefix",
                    "seeds": args.stage1_seeds,
                    "candidate_count": len(candidates),
                    "survivor_count": len(stage1_finalists),
                },
                "stage2": {
                    "purpose": "intermediate multi-seed refinement",
                    "duration_seconds": args.stage2_duration,
                    "empirical_target": "matching first-session prefix",
                    "seeds": args.stage2_seeds,
                    "candidate_count": len(stage1_finalists),
                    "survivor_count": len(stage2_finalists),
                },
                "stage3": {
                    "purpose": "full-session multi-seed selection",
                    "duration_seconds": args.stage3_duration,
                    "empirical_target": "full-session target",
                    "seeds": args.seeds,
                    "candidate_count": len(stage2_finalists),
                },
            },
            "seeds": args.seeds,
            "selection_uses_heldout_targets": False,
            "heldout_background_inputs": "frozen training-day distributions and Hawkes rates",
            "heldout_opening_state": "held-out day 09:30 BBO and midpoint fundamental",
            "basket_weights": "frozen 2019-09-30 QQQ three-stock proxy filed 2019-12-20",
            "coupling_mode": args.coupling_mode,
            "selection_and_validation_use_identical_agent_configuration": True,
            "training_config_sha256": sha256_file(training_config),
            "heldout_config_sha256": sha256_file(heldout_config),
            "simulator_binary_sha256": sha256_file(binary),
            "fixed_value_parameters": {
                "max_order_quantity": args.max_order_quantity,
                "max_inventory": args.max_inventory,
                "decision_interval_ms": args.decision_interval_ms,
            },
            "coupling_parameters": {
                "arbitrage_trigger_bps": args.arbitrage_trigger_bps,
                "arbitrage_release_bps": args.arbitrage_release_bps,
                "shared_mm_exposure_threshold": args.coupled_exposure_threshold,
                "max_hedge_quantity": args.max_hedge_quantity,
            },
        },
        "moment_matching": {
            "name": "diagonal_weighted_standardized_simulated_moment_matching",
            "selection_loss": "fit_wsmrmse",
            "selection_formula": (
                "sqrt(sum(weight * ((simulated_mean - empirical_target) / "
                "empirical_scale)^2) / sum(weight))"
            ),
            "mc_se_definition": (
                "sample standard deviation across independent seeds divided by "
                "sqrt(seed_count)"
            ),
            "combined_uncertainty_diagnostic": (
                "sqrt(sum(weight * ((simulated_mean - empirical_target) / "
                "hypot(empirical_scale, simulation_mc_se))^2) / sum(weight))"
            ),
            "residual_cap": None,
            "target_scale_column": "scale",
            "target_weight_column": "weight",
            "moments": list(METRICS),
        },
        "grid_size": len(candidates),
        "stage1_screening": [
            {
                "candidate_index": item["candidate_index"],
                "candidate": asdict(item["candidate"]),
                "evaluation": evaluation_for_report(item["evaluation"]),
            }
            for item in stage1_screening
        ],
        "stage2_refinement": [
            {
                "candidate_index": item["candidate_index"],
                "candidate": asdict(item["candidate"]),
                "evaluation": evaluation_for_report(item["evaluation"]),
            }
            for item in stage2_refinement
        ],
        "stage3_full_training": [
            {
                "candidate_index": item["candidate_index"],
                "candidate": asdict(item["candidate"]),
                "evaluation": evaluation_for_report(item["evaluation"]),
            }
            for item in stage3_training
        ],
        # Compatibility names used by earlier table-generation scripts.
        "screening": [
            {
                "candidate_index": item["candidate_index"],
                "candidate": asdict(item["candidate"]),
                "evaluation": evaluation_for_report(item["evaluation"]),
            }
            for item in stage1_screening
        ],
        "full_training": [
            {
                "candidate_index": item["candidate_index"],
                "candidate": asdict(item["candidate"]),
                "evaluation": evaluation_for_report(item["evaluation"]),
            }
            for item in stage3_training
        ],
        "selected_parameters": asdict(selected_candidate),
        "selected_training_evaluation": evaluation_for_report(
            selected_training_evaluation
        ),
        "heldout_evaluation": evaluation_for_report(heldout_evaluation),
        # These names remain for compatibility with prior table-generation
        # scripts.  They now hold a seed-averaged weighted-moment score rather
        # than an arithmetic mean of seed-specific RMSE values.
        "selected_training_score": finite_or_none(float(
            selected_training_evaluation["fit_wsmrmse"]
        )),
        "coupled_training_wall_seconds": selected_training_evaluation[
            "seed_wall_seconds"
        ],
        "coupled_training_mean_score": finite_or_none(float(
            selected_training_evaluation["fit_wsmrmse"]
        )),
        "heldout_wall_seconds": heldout_evaluation["seed_wall_seconds"],
        "heldout_mean_score": finite_or_none(float(
            heldout_evaluation["fit_wsmrmse"]
        )),
    }
    with (output_root / "value_agent_calibration_report.json").open("w") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps({
        "selected_parameters": report["selected_parameters"],
        "training_score": report["coupled_training_mean_score"],
        "heldout_score": report["heldout_mean_score"],
        "report": str(output_root / "value_agent_calibration_report.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
