#!/usr/bin/env python3
"""
Estimate stylised-fact moments from an empirical or simulated LOB time-series.

Expected snapshot CSV columns, with flexible names:
    time_seconds or time_ns
    mid_price or mid or mid_ticks
    spread or spread_ticks
    best_bid_depth
    best_ask_depth
    volume, executed_volume, aggressive_volume, or market_order_executed_quantity

Output format is compatible with ../empirical_moments.csv:
    moment_name,target,weight,scale_floor
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

MOMENT_DEFAULTS = {
    "return_mean": (0.5, 1e-6),
    "return_variance": (3.0, 1e-8),
    "return_skewness": (0.5, 0.05),
    "return_kurtosis": (1.5, 1.0),
    "return_acf_lag1": (2.0, 0.01),
    "abs_return_acf_lag1": (1.5, 0.02),
    "abs_return_acf_lag2": (1.0, 0.02),
    "abs_return_acf_lag5": (1.0, 0.02),
    "abs_return_acf_lag10": (1.0, 0.02),
    "abs_return_acf_lag25": (0.75, 0.02),
    "abs_return_acf_lag50": (0.75, 0.02),
    "abs_return_acf_lag100": (0.5, 0.02),
    "mean_spread": (2.5, 100.0),
    "spread_p90": (1.5, 100.0),
    "spread_p95": (1.0, 100.0),
    "volume_volatility_corr": (1.0, 0.05),
    "price_diffusion_10s": (3.0, 1000.0),
    "mean_best_bid_depth": (2.0, 100.0),
    "mean_best_ask_depth": (2.0, 100.0),
    "mid_move_rate": (3.0, 0.02),
    "zero_mid_change_ratio": (2.0, 0.02),
    "market_order_best_removal_rate": (2.0, 0.002),
    "cancel_best_removal_rate": (2.0, 0.002),
    "limit_order_rate": (1.0, 1.0),
    "market_order_rate": (1.0, 0.5),
    "cancel_rate": (1.0, 1.0),
}


def choose_column(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def acf(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if lag <= 0 or x.size <= lag:
        return 0.0
    x = x - x.mean()
    den = float(np.dot(x, x))
    if den <= 1e-30:
        return 0.0
    return float(np.dot(x[lag:], x[:-lag]) / den)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x = np.asarray(x[:n], dtype=float)
    y = np.asarray(y[:n], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.std(x) <= 1e-30 or np.std(y) <= 1e-30:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def estimate_from_snapshots(df: pd.DataFrame, sample_interval_seconds: float | None = None) -> dict[str, float]:
    mid_col = choose_column(df, ["mid_price", "mid", "mid_ticks"])
    spread_col = choose_column(df, ["spread", "spread_ticks"])
    bid_depth_col = choose_column(df, ["best_bid_depth", "bid_depth", "mean_best_bid_depth"])
    ask_depth_col = choose_column(df, ["best_ask_depth", "ask_depth", "mean_best_ask_depth"])
    vol_col = choose_column(df, ["volume", "executed_volume", "aggressive_volume", "market_order_executed_quantity"])
    time_col = choose_column(df, ["time_seconds", "seconds", "timestamp_seconds"])
    time_ns_col = choose_column(df, ["time_ns", "timestamp_ns"])

    if mid_col is None:
        raise ValueError("Could not find a mid-price column. Expected mid_price, mid, or mid_ticks.")

    mid = pd.to_numeric(df[mid_col], errors="coerce").dropna().to_numpy(dtype=float)
    mid = mid[mid > 0]
    if mid.size < 3:
        raise ValueError("Not enough positive mid-price observations.")

    returns = np.diff(np.log(mid))
    returns = returns[np.isfinite(returns)]
    abs_returns = np.abs(returns)
    mid_changes = np.diff(mid)

    if sample_interval_seconds is None:
        if time_col is not None:
            t = pd.to_numeric(df[time_col], errors="coerce").dropna().to_numpy(dtype=float)
            if t.size >= 2:
                sample_interval_seconds = float(np.nanmedian(np.diff(t)))
        elif time_ns_col is not None:
            t = pd.to_numeric(df[time_ns_col], errors="coerce").dropna().to_numpy(dtype=float) / 1e9
            if t.size >= 2:
                sample_interval_seconds = float(np.nanmedian(np.diff(t)))
    if sample_interval_seconds is None or not np.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0:
        sample_interval_seconds = 1.0

    diffusion_lag = max(1, int(round(10.0 / sample_interval_seconds)))
    price_diffusion_10s = float(np.var(mid[diffusion_lag:] - mid[:-diffusion_lag], ddof=1)) if mid.size > diffusion_lag + 1 else 0.0

    spread = pd.to_numeric(df[spread_col], errors="coerce").dropna().to_numpy(dtype=float) if spread_col else np.array([], dtype=float)
    bid_depth = pd.to_numeric(df[bid_depth_col], errors="coerce").dropna().to_numpy(dtype=float) if bid_depth_col else np.array([], dtype=float)
    ask_depth = pd.to_numeric(df[ask_depth_col], errors="coerce").dropna().to_numpy(dtype=float) if ask_depth_col else np.array([], dtype=float)
    volume = pd.to_numeric(df[vol_col], errors="coerce").fillna(0).to_numpy(dtype=float) if vol_col else np.zeros_like(mid)
    if volume.size > abs_returns.size:
        volume = volume[1:]

    result = {
        "return_mean": float(np.mean(returns)) if returns.size else 0.0,
        "return_variance": float(np.var(returns, ddof=1)) if returns.size > 1 else 0.0,
        "return_skewness": float(pd.Series(returns).skew()) if returns.size > 2 else 0.0,
        "return_kurtosis": float(pd.Series(returns).kurtosis() + 3.0) if returns.size > 3 else 0.0,
        "return_acf_lag1": acf(returns, 1),
        "abs_return_acf_lag1": acf(abs_returns, 1),
        "abs_return_acf_lag2": acf(abs_returns, 2),
        "abs_return_acf_lag5": acf(abs_returns, 5),
        "abs_return_acf_lag10": acf(abs_returns, 10),
        "abs_return_acf_lag25": acf(abs_returns, 25),
        "abs_return_acf_lag50": acf(abs_returns, 50),
        "abs_return_acf_lag100": acf(abs_returns, 100),
        "mean_spread": float(np.mean(spread)) if spread.size else 0.0,
        "spread_p90": float(np.quantile(spread, 0.90)) if spread.size else 0.0,
        "spread_p95": float(np.quantile(spread, 0.95)) if spread.size else 0.0,
        "volume_volatility_corr": safe_corr(volume, abs_returns),
        "price_diffusion_10s": price_diffusion_10s,
        "mean_best_bid_depth": float(np.mean(bid_depth)) if bid_depth.size else 0.0,
        "mean_best_ask_depth": float(np.mean(ask_depth)) if ask_depth.size else 0.0,
        "mid_move_rate": float(np.mean(np.abs(mid_changes) > 1e-12)) if mid_changes.size else 0.0,
        "zero_mid_change_ratio": float(np.mean(np.abs(mid_changes) <= 1e-12)) if mid_changes.size else 0.0,
    }
    return result


def add_event_moments(result: dict[str, float], event_csv: Path | None, duration_seconds: float | None) -> dict[str, float]:
    defaults = {
        "market_order_best_removal_rate": 0.0,
        "cancel_best_removal_rate": 0.0,
        "limit_order_rate": 0.0,
        "market_order_rate": 0.0,
        "cancel_rate": 0.0,
    }
    result.update(defaults)
    if event_csv is None:
        return result
    events = pd.read_csv(event_csv)
    if events.empty:
        return result

    action_col = choose_column(events, ["order_action", "event_type", "hawkes_event_type"])
    removed_col = choose_column(events, ["best_level_removed"])
    time_col = choose_column(events, ["time_seconds", "seconds", "timestamp_seconds"])
    if duration_seconds is None:
        if time_col is not None:
            t = pd.to_numeric(events[time_col], errors="coerce").dropna()
            if len(t) >= 2:
                duration_seconds = float(t.max() - t.min())
    if duration_seconds is None or duration_seconds <= 0:
        duration_seconds = 1.0

    if action_col is not None:
        actions = events[action_col].astype(str).str.lower()
        is_limit = actions.str.contains("limit")
        is_market = actions.str.contains("market")
        is_cancel = actions.str.contains("cancel")
        result["limit_order_rate"] = float(is_limit.sum()) / duration_seconds
        result["market_order_rate"] = float(is_market.sum()) / duration_seconds
        result["cancel_rate"] = float(is_cancel.sum()) / duration_seconds
        if removed_col is not None:
            removed = pd.to_numeric(events[removed_col], errors="coerce").fillna(0).astype(float) > 0
            result["market_order_best_removal_rate"] = float((removed & is_market).sum()) / max(1, int(is_market.sum()))
            result["cancel_best_removal_rate"] = float((removed & is_cancel).sum()) / max(1, int(is_cancel.sum()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", required=True, help="CSV containing mid/spread/depth/volume snapshots")
    parser.add_argument("--events", default=None, help="Optional event/applied-order CSV for event rates and best-removal rates")
    parser.add_argument("--output", default="empirical_moments_from_data.csv")
    parser.add_argument("--sample-interval-seconds", type=float, default=None)
    parser.add_argument("--duration-seconds", type=float, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.snapshots)
    moments = estimate_from_snapshots(df, args.sample_interval_seconds)
    moments = add_event_moments(moments, Path(args.events) if args.events else None, args.duration_seconds)

    rows = []
    for name, (weight, floor) in MOMENT_DEFAULTS.items():
        rows.append({
            "moment_name": name,
            "target": moments.get(name, 0.0),
            "weight": weight,
            "scale_floor": floor,
        })
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
