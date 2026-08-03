#!/usr/bin/env python3
"""Validate that the shared-dealer treatment is active when the shock arrives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


class PreflightError(RuntimeError):
    """Raised when the causal treatment is not operational."""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise PreflightError(f"empty CSV: {path}")
    return rows


def finite(row: dict[str, str], field: str, *, source: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise PreflightError(f"{source} lacks numeric field {field}") from error
    if not math.isfinite(value):
        raise PreflightError(f"{source} has non-finite {field}")
    return value


def observation_at(
    rows: list[dict[str, str]], time_seconds: float, source: Path,
) -> dict[str, str]:
    matches = [
        row for row in rows
        if math.isclose(
            finite(row, "time_seconds", source=source),
            time_seconds,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ]
    if len(matches) != 1:
        raise PreflightError(
            f"{source} has {len(matches)} observations at t={time_seconds:g}"
        )
    return matches[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_inventory_adverse_targets(
    run: dict[str, str], *, source: Path,
) -> dict[str, int] | None:
    """Verify every realized stress side against left-limit dealer inventory."""
    if run.get("requested_shock_inventory_adverse") != "1":
        return None
    path = Path(run.get("shock_targets_csv", ""))
    if not path.is_file():
        raise PreflightError(
            f"inventory-adverse path lacks a shock-target manifest: {path}"
        )
    expected_hash = run.get("shock_targets_csv_sha256", "")
    if not expected_hash or sha256_file(path) != expected_hash:
        raise PreflightError(f"shock-target manifest SHA-256 mismatch: {path}")
    rows = read_csv(path)
    required = {
        "asset_id", "is_shock_target", "shock_enabled",
        "requested_quantity", "requested_sell_quantity",
        "requested_buy_quantity", "shock_side",
        "pre_shock_shared_inventory", "direction_rule",
    }
    missing = required.difference(rows[0])
    if missing:
        raise PreflightError(
            f"shock-target manifest lacks fields {sorted(missing)}: {path}"
        )
    target_count = 0
    buy_count = 0
    sell_count = 0
    requested_total = 0
    for line_number, row in enumerate(rows, start=2):
        target = row.get("is_shock_target") == "1"
        if row.get("shock_enabled") != "1":
            raise PreflightError(
                f"shock-target manifest is disabled at {path}:{line_number}"
            )
        try:
            quantity = int(row.get("requested_quantity", ""))
            buy_quantity = int(row.get("requested_buy_quantity", ""))
            sell_quantity = int(row.get("requested_sell_quantity", ""))
            inventory = int(row.get("pre_shock_shared_inventory", ""))
        except (TypeError, ValueError) as error:
            raise PreflightError(
                f"invalid integer in shock-target manifest {path}:{line_number}"
            ) from error
        side = row.get("shock_side", "")
        if not target:
            if quantity != 0 or buy_quantity != 0 or sell_quantity != 0:
                raise PreflightError(
                    f"non-target has a positive dose at {path}:{line_number}"
                )
            continue
        target_count += 1
        requested_total += quantity
        if quantity <= 0 or buy_quantity + sell_quantity != quantity:
            raise PreflightError(
                f"invalid target dose at {path}:{line_number}"
            )
        if row.get("direction_rule") != "inventory_adverse":
            raise PreflightError(
                f"wrong direction rule at {path}:{line_number}"
            )
        expected_side = "buy" if inventory < 0 else "sell"
        if side != expected_side:
            raise PreflightError(
                f"shock side is not inventory-adverse at {path}:{line_number}"
            )
        if side == "buy":
            buy_count += 1
            if buy_quantity != quantity or sell_quantity != 0:
                raise PreflightError(
                    f"buy-side quantity mismatch at {path}:{line_number}"
                )
        else:
            sell_count += 1
            if sell_quantity != quantity or buy_quantity != 0:
                raise PreflightError(
                    f"sell-side quantity mismatch at {path}:{line_number}"
                )
    declared_total = finite(run, "shock_requested_quantity", source=source)
    if requested_total != declared_total:
        raise PreflightError(
            f"shock-target total {requested_total} differs from raw result "
            f"{declared_total:g}"
        )
    if target_count <= 0 or target_count != buy_count + sell_count:
        raise PreflightError("inventory-adverse shock has no valid targets")
    return {
        "target_count": target_count,
        "buy_target_count": buy_count,
        "sell_target_count": sell_count,
        "requested_quantity": requested_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--shock-time-seconds", type=float, required=True)
    parser.add_argument("--metrics-interval-seconds", type=float, default=1.0)
    parser.add_argument("--lookback-seconds", type=float, default=60.0)
    parser.add_argument(
        "--minimum-quote-scale",
        type=float,
        default=0.05,
        help=(
            "numerical participation floor; the median pre-shock quote "
            "scale must be strictly above this value"
        ),
    )
    parser.add_argument("--minimum-active-asset-fraction", type=float, default=0.99)
    parser.add_argument(
        "--minimum-two-sided-active-asset-fraction", type=float, default=1.0,
        help=(
            "minimum fraction of books for which the dealer requests both a "
            "bid and an ask throughout the pre-shock lookback"
        ),
    )
    parser.add_argument(
        "--minimum-resting-two-sided-active-asset-fraction",
        type=float,
        default=1.0,
        help=(
            "minimum fraction retaining both requested quotes after immediate "
            "execution at each pre-shock observation"
        ),
    )
    parser.add_argument(
        "--minimum-economic-quote-scale", type=float, default=0.25,
        help="minimum median risk-increasing capacity multiplier before shock",
    )
    parser.add_argument(
        "--maximum-pre-shock-utilization", type=float, default=0.85,
        help="maximum allowed utilization immediately before the shock",
    )
    parser.add_argument(
        "--minimum-target-bid-participation", type=float, default=0.0,
        help=(
            "legacy fixed-sell gate for the shared-dealer fraction of "
            "target-book best-bid depth; use zero for a mixed-side "
            "inventory-adverse intervention and gate realized absorption"
        ),
    )
    parser.add_argument(
        "--minimum-marketwide-bbo-participation", type=float, default=0.05,
        help=(
            "minimum shared-dealer fraction of aggregate best-bid and "
            "best-ask depth throughout the pre-shock lookback"
        ),
    )
    parser.add_argument(
        "--minimum-inventory-asset-fraction", type=float, default=0.25,
        help="minimum fraction of books with nonzero shared-dealer inventory",
    )
    parser.add_argument(
        "--minimum-shock-absorption-fraction", type=float, default=0.05,
        help="minimum fraction of executed stress quantity filled by the dealer",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 0.0 < args.minimum_quote_scale <= 1.0:
        raise PreflightError("minimum quote scale must lie in (0,1]")
    if not 0.0 < args.minimum_active_asset_fraction <= 1.0:
        raise PreflightError("minimum active fraction must lie in (0,1]")
    if not 0.0 < args.minimum_two_sided_active_asset_fraction <= 1.0:
        raise PreflightError(
            "minimum two-sided active fraction must lie in (0,1]"
        )
    if not 0.0 < args.minimum_resting_two_sided_active_asset_fraction <= 1.0:
        raise PreflightError(
            "minimum resting two-sided active fraction must lie in (0,1]"
        )
    for label, value in (
        ("minimum economic quote scale", args.minimum_economic_quote_scale),
        ("maximum pre-shock utilization", args.maximum_pre_shock_utilization),
        ("minimum market-wide BBO participation",
         args.minimum_marketwide_bbo_participation),
        ("minimum inventory asset fraction", args.minimum_inventory_asset_fraction),
        ("minimum shock absorption fraction", args.minimum_shock_absorption_fraction),
    ):
        if not 0.0 < value <= 1.0:
            raise PreflightError(f"{label} must lie in (0,1]")
    if not 0.0 <= args.minimum_target_bid_participation <= 1.0:
        raise PreflightError(
            "minimum target bid participation must lie in [0,1]"
        )
    if not 0.0 < args.metrics_interval_seconds <= args.shock_time_seconds:
        raise PreflightError(
            "metrics interval must be positive and no larger than shock time"
        )
    if not 0.0 < args.lookback_seconds <= args.shock_time_seconds:
        raise PreflightError(
            "lookback must be positive and no larger than shock time"
        )
    # The simulator records the boundary state before processing events whose
    # timestamp equals that boundary.  Therefore the observation at t_s is
    # the left-limit market state immediately before the shock is injected.
    pre_shock_time = args.shock_time_seconds

    runs = read_csv(args.raw)
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in runs:
        if row.get("shared_mm_mode") != "global":
            raise PreflightError("mechanism preflight must contain global mode only")
        key = (row.get("risk_limit_per_asset", ""), row.get("shock_mode", ""))
        if not key[0] or key[1] not in {"on", "off"}:
            raise PreflightError(f"invalid treatment key {key}")
        if key in indexed:
            raise PreflightError(f"duplicate treatment path {key}")
        indexed[key] = row

    risks = sorted({key[0] for key in indexed}, key=float)
    diagnostics: list[dict[str, object]] = []
    failures: list[str] = []
    quote_scale_responses: list[float] = []
    for risk in risks:
        pair = {mode: indexed.get((risk, mode)) for mode in ("off", "on")}
        if pair["off"] is None or pair["on"] is None:
            raise PreflightError(f"capacity {risk} lacks a matched shock/control pair")
        observations: dict[str, dict[str, str]] = {}
        first_post_observations: dict[str, dict[str, str]] = {}
        lookback_scales: dict[str, list[float]] = {}
        lookback_requested_two_sided: dict[str, list[float]] = {}
        lookback_resting_two_sided: dict[str, list[float]] = {}
        lookback_bbo_participation: dict[str, list[float]] = {}
        for mode, run in pair.items():
            assert run is not None
            metrics_path = Path(run.get("metrics_csv", ""))
            if not metrics_path.is_file():
                raise PreflightError(f"missing metrics for capacity {risk}/{mode}")
            metric_rows = read_csv(metrics_path)
            observations[mode] = observation_at(
                metric_rows, pre_shock_time, metrics_path,
            )
            first_post_observations[mode] = observation_at(
                metric_rows,
                pre_shock_time + args.metrics_interval_seconds,
                metrics_path,
            )
            lower = args.shock_time_seconds - args.lookback_seconds
            selected = [
                row for row in metric_rows
                if lower < finite(
                    row, "time_seconds", source=metrics_path
                ) <= args.shock_time_seconds
            ]
            expected_count = round(
                args.lookback_seconds / args.metrics_interval_seconds
            )
            if len(selected) != expected_count:
                raise PreflightError(
                    f"{metrics_path} has {len(selected)} lookback observations; "
                    f"expected {expected_count}"
                )
            lookback_scales[mode] = [
                finite(row, "shared_quote_scale", source=metrics_path)
                for row in selected
            ]
            lookback_requested_two_sided[mode] = [
                finite(
                    row,
                    "shared_requested_two_sided_asset_fraction",
                    source=metrics_path,
                )
                for row in selected
            ]
            lookback_resting_two_sided[mode] = [
                finite(
                    row,
                    "shared_two_sided_active_asset_fraction",
                    source=metrics_path,
                )
                for row in selected
            ]
            lookback_bbo_participation[mode] = [
                finite(
                    row, "shared_bbo_depth_participation",
                    source=metrics_path,
                )
                for row in selected
            ]

        shock_observation = observations["on"]
        control_observation = observations["off"]
        shock_first_post = first_post_observations["on"]
        control_first_post = first_post_observations["off"]
        fields = (
            "shared_quote_scale",
            "shared_requested_quote_depth",
            "shared_risk_reducing_requested_quote_depth",
            "shared_risk_increasing_requested_quote_depth",
            "shared_resting_quote_depth",
            "shared_risk_reducing_resting_quote_depth",
            "shared_risk_increasing_resting_quote_depth",
            "shared_active_asset_fraction",
            "shared_two_sided_active_asset_fraction",
            "shared_utilization",
            "shocked_bid_top_depth",
            "shocked_shared_bid_resting_depth",
            "shocked_shared_bid_participation",
            "shared_nonzero_inventory_asset_fraction",
            "mean_absolute_shared_inventory",
            "mean_absolute_shocked_shared_inventory",
            "shared_requested_active_asset_fraction",
            "shared_requested_two_sided_asset_fraction",
            "shared_best_bid_depth",
            "shared_best_ask_depth",
            "shared_at_best_bid_asset_fraction",
            "shared_at_best_ask_asset_fraction",
            "shared_bbo_depth_participation",
        )
        shock_values = {
            field: finite(shock_observation, field, source=args.raw)
            for field in fields
        }
        control_values = {
            field: finite(control_observation, field, source=args.raw)
            for field in fields
        }
        for field in fields:
            if not math.isclose(
                shock_values[field], control_values[field],
                rel_tol=0.0, abs_tol=1.0e-9,
            ):
                failures.append(
                    f"capacity {risk}: shock/control differ before intervention "
                    f"for {field}"
                )

        executed = finite(pair["on"], "shock_executed_quantity", source=args.raw)  # type: ignore[arg-type]
        absorbed = finite(pair["on"], "shock_shared_mm_quantity", source=args.raw)  # type: ignore[arg-type]
        direction_audit = audit_inventory_adverse_targets(
            pair["on"], source=args.raw,  # type: ignore[arg-type]
        )
        fill_ownership = {
            "shared_dealer": absorbed,
            "local_market_maker": finite(
                pair["on"], "shock_local_mm_quantity", source=args.raw  # type: ignore[arg-type]
            ),
            "value_agent": finite(
                pair["on"], "shock_value_agent_quantity", source=args.raw  # type: ignore[arg-type]
            ),
            "background": finite(
                pair["on"], "shock_background_quantity", source=args.raw  # type: ignore[arg-type]
            ),
            "other": finite(
                pair["on"], "shock_other_quantity", source=args.raw  # type: ignore[arg-type]
            ),
        }
        if not math.isclose(
            sum(fill_ownership.values()), executed,
            rel_tol=0.0, abs_tol=1.0e-9,
        ):
            failures.append(
                f"capacity {risk}: shock fill ownership does not sum to "
                "executed quantity"
            )
        median_quote_scale = statistics.median(lookback_scales["on"])
        control_median_quote_scale = statistics.median(
            lookback_scales["off"]
        )
        if not math.isclose(
            median_quote_scale, control_median_quote_scale,
            rel_tol=0.0, abs_tol=1.0e-12,
        ):
            failures.append(
                f"capacity {risk}: shock/control median pre-shock quote "
                "scales differ"
            )
        if median_quote_scale <= args.minimum_quote_scale:
            failures.append(
                f"capacity {risk}: median pre-shock quote scale "
                f"{median_quote_scale:.6g} is not above the numerical floor "
                f"{args.minimum_quote_scale:.6g}"
            )
        if median_quote_scale < args.minimum_economic_quote_scale:
            failures.append(
                f"capacity {risk}: median pre-shock quote scale "
                f"{median_quote_scale:.6g} is below the economic activity "
                f"gate {args.minimum_economic_quote_scale:.6g}"
            )
        if (shock_values["shared_utilization"]
                > args.maximum_pre_shock_utilization):
            failures.append(
                f"capacity {risk}: pre-shock utilization "
                f"{shock_values['shared_utilization']:.6g} exceeds the "
                f"headroom gate {args.maximum_pre_shock_utilization:.6g}"
            )
        if shock_values["shared_active_asset_fraction"] < args.minimum_active_asset_fraction:
            failures.append(
                f"capacity {risk}: active-asset fraction "
                f"{shock_values['shared_active_asset_fraction']:.6g} is below "
                f"{args.minimum_active_asset_fraction:.6g}"
            )
        minimum_requested_two_sided = min(
            lookback_requested_two_sided["on"]
        )
        control_minimum_requested_two_sided = min(
            lookback_requested_two_sided["off"]
        )
        if not math.isclose(
            minimum_requested_two_sided,
            control_minimum_requested_two_sided,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            failures.append(
                f"capacity {risk}: shock/control minimum pre-shock "
                "two-sided dealer-policy coverage differs"
            )
        if (minimum_requested_two_sided
                < args.minimum_two_sided_active_asset_fraction):
            failures.append(
                f"capacity {risk}: minimum pre-shock two-sided requested-asset "
                f"fraction {minimum_requested_two_sided:.6g} is below "
                f"{args.minimum_two_sided_active_asset_fraction:.6g}"
            )
        minimum_resting_two_sided = min(
            lookback_resting_two_sided["on"]
        )
        control_minimum_resting_two_sided = min(
            lookback_resting_two_sided["off"]
        )
        if not math.isclose(
            minimum_resting_two_sided,
            control_minimum_resting_two_sided,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            failures.append(
                f"capacity {risk}: shock/control minimum pre-shock "
                "two-sided resting coverage differs"
            )
        if (minimum_resting_two_sided
                < args.minimum_resting_two_sided_active_asset_fraction):
            failures.append(
                f"capacity {risk}: minimum pre-shock two-sided resting-asset "
                f"fraction {minimum_resting_two_sided:.6g} "
                "is below "
                f"{args.minimum_resting_two_sided_active_asset_fraction:.6g}"
            )
        if shock_values["shared_resting_quote_depth"] <= 0.0:
            failures.append(f"capacity {risk}: no standing shared quote before shock")
        if shock_values["shared_risk_increasing_resting_quote_depth"] <= 0.0:
            failures.append(
                f"capacity {risk}: no standing risk-increasing quote before shock"
            )
        if (shock_values["shocked_shared_bid_participation"]
                < args.minimum_target_bid_participation):
            failures.append(
                f"capacity {risk}: shared-dealer target-bid participation "
                f"{shock_values['shocked_shared_bid_participation']:.6g} is "
                f"below {args.minimum_target_bid_participation:.6g}"
            )
        minimum_bbo_participation = min(lookback_bbo_participation["on"])
        control_minimum_bbo_participation = min(
            lookback_bbo_participation["off"]
        )
        if not math.isclose(
            minimum_bbo_participation,
            control_minimum_bbo_participation,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            failures.append(
                f"capacity {risk}: shock/control minimum pre-shock "
                "market-wide BBO participation differs"
            )
        if (minimum_bbo_participation
                < args.minimum_marketwide_bbo_participation):
            failures.append(
                f"capacity {risk}: minimum market-wide BBO participation "
                f"{minimum_bbo_participation:.6g} is below "
                f"{args.minimum_marketwide_bbo_participation:.6g}"
            )
        if (shock_values["shared_nonzero_inventory_asset_fraction"]
                < args.minimum_inventory_asset_fraction):
            failures.append(
                f"capacity {risk}: nonzero-inventory asset fraction "
                f"{shock_values['shared_nonzero_inventory_asset_fraction']:.6g} "
                f"is below {args.minimum_inventory_asset_fraction:.6g}"
            )
        if executed <= 0.0:
            failures.append(f"capacity {risk}: shock executed no quantity")
        if absorbed <= 0.0:
            failures.append(f"capacity {risk}: shared dealer absorbed none of the shock")
        absorption_fraction = absorbed / executed if executed else 0.0
        if absorption_fraction < args.minimum_shock_absorption_fraction:
            failures.append(
                f"capacity {risk}: shock absorption fraction "
                f"{absorption_fraction:.6g} is below "
                f"{args.minimum_shock_absorption_fraction:.6g}"
            )

        immediate_gross_delta = finite(
            shock_first_post, "shared_gross_exposure", source=args.raw,
        ) - finite(
            control_first_post, "shared_gross_exposure", source=args.raw,
        )
        immediate_utilization_delta = finite(
            shock_first_post, "shared_utilization", source=args.raw,
        ) - finite(
            control_first_post, "shared_utilization", source=args.raw,
        )
        immediate_quote_scale_delta = finite(
            shock_first_post, "shared_quote_scale", source=args.raw,
        ) - finite(
            control_first_post, "shared_quote_scale", source=args.raw,
        )
        quote_scale_responses.append(immediate_quote_scale_delta)
        if immediate_gross_delta <= 0.0:
            failures.append(
                f"capacity {risk}: shock did not increase shared gross "
                f"exposure relative to control ({immediate_gross_delta:.6g})"
            )
        if immediate_utilization_delta <= 0.0:
            failures.append(
                f"capacity {risk}: shock did not increase shared utilization "
                f"relative to control ({immediate_utilization_delta:.6g})"
            )

        diagnostics.append({
            "risk_limit_per_asset": float(risk),
            "pre_shock_quote_scale": shock_values["shared_quote_scale"],
            "median_pre_shock_quote_scale": median_quote_scale,
            "pre_shock_active_asset_fraction": shock_values[
                "shared_active_asset_fraction"
            ],
            "pre_shock_two_sided_active_asset_fraction": shock_values[
                "shared_two_sided_active_asset_fraction"
            ],
            "minimum_lookback_two_sided_requested_asset_fraction": (
                minimum_requested_two_sided
            ),
            "minimum_lookback_two_sided_resting_asset_fraction": (
                minimum_resting_two_sided
            ),
            "pre_shock_utilization": shock_values["shared_utilization"],
            "pre_shock_target_bid_depth": shock_values["shocked_bid_top_depth"],
            "pre_shock_target_shared_bid_depth": shock_values[
                "shocked_shared_bid_resting_depth"
            ],
            "pre_shock_target_bid_participation": shock_values[
                "shocked_shared_bid_participation"
            ],
            "pre_shock_shared_best_bid_depth": shock_values[
                "shared_best_bid_depth"
            ],
            "pre_shock_shared_best_ask_depth": shock_values[
                "shared_best_ask_depth"
            ],
            "pre_shock_shared_at_best_bid_asset_fraction": shock_values[
                "shared_at_best_bid_asset_fraction"
            ],
            "pre_shock_shared_at_best_ask_asset_fraction": shock_values[
                "shared_at_best_ask_asset_fraction"
            ],
            "minimum_lookback_marketwide_bbo_depth_participation": (
                minimum_bbo_participation
            ),
            "pre_shock_nonzero_inventory_asset_fraction": shock_values[
                "shared_nonzero_inventory_asset_fraction"
            ],
            "pre_shock_mean_absolute_inventory": shock_values[
                "mean_absolute_shared_inventory"
            ],
            "pre_shock_mean_absolute_target_inventory": shock_values[
                "mean_absolute_shocked_shared_inventory"
            ],
            "pre_shock_requested_quote_depth": shock_values[
                "shared_requested_quote_depth"
            ],
            "pre_shock_risk_reducing_quote_depth": shock_values[
                "shared_risk_reducing_requested_quote_depth"
            ],
            "pre_shock_risk_increasing_quote_depth": shock_values[
                "shared_risk_increasing_requested_quote_depth"
            ],
            "pre_shock_resting_quote_depth": shock_values[
                "shared_resting_quote_depth"
            ],
            "pre_shock_risk_reducing_resting_quote_depth": shock_values[
                "shared_risk_reducing_resting_quote_depth"
            ],
            "pre_shock_risk_increasing_resting_quote_depth": shock_values[
                "shared_risk_increasing_resting_quote_depth"
            ],
            "shock_executed_quantity": executed,
            "shock_shared_dealer_quantity": absorbed,
            "shock_fill_ownership": fill_ownership,
            "immediate_shock_minus_control_gross_exposure": (
                immediate_gross_delta
            ),
            "immediate_shock_minus_control_utilization": (
                immediate_utilization_delta
            ),
            "immediate_shock_minus_control_quote_scale": (
                immediate_quote_scale_delta
            ),
            "shock_absorption_fraction": absorption_fraction,
            "shock_direction_audit": direction_audit,
        })

    if quote_scale_responses and min(quote_scale_responses) >= 0.0:
        failures.append(
            "no capacity treatment reduced the shared quote scale immediately "
            "after the shock"
        )

    payload = {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "purpose": "shared_dealer_causal_treatment_preflight",
        "shock_time_seconds": args.shock_time_seconds,
        "pre_shock_observation_time_seconds": pre_shock_time,
        "lookback_seconds": args.lookback_seconds,
        "minimum_quote_scale": args.minimum_quote_scale,
        "quote_scale_gate": "median_strictly_above_numerical_floor",
        "minimum_active_asset_fraction": args.minimum_active_asset_fraction,
        "minimum_two_sided_active_asset_fraction": (
            args.minimum_two_sided_active_asset_fraction
        ),
        "minimum_resting_two_sided_active_asset_fraction": (
            args.minimum_resting_two_sided_active_asset_fraction
        ),
        "minimum_economic_quote_scale": args.minimum_economic_quote_scale,
        "maximum_pre_shock_utilization": args.maximum_pre_shock_utilization,
        "minimum_target_bid_participation": args.minimum_target_bid_participation,
        "minimum_marketwide_bbo_participation": (
            args.minimum_marketwide_bbo_participation
        ),
        "minimum_inventory_asset_fraction": args.minimum_inventory_asset_fraction,
        "minimum_shock_absorption_fraction": args.minimum_shock_absorption_fraction,
        "diagnostics": diagnostics,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise PreflightError(
            "shared-dealer mechanism preflight failed; the financial matrix "
            "was not authorized"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreflightError as error:
        raise SystemExit(f"shared-dealer preflight failed: {error}") from error
