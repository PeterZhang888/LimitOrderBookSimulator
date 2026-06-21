import glob
import os
import sys

import pandas as pd


RESULT_DIRECTORY = "mc_full_day_results"
PATH_PATTERN = os.path.join(
    RESULT_DIRECTORY,
    "autonomous_mm_path_*.csv",
)

SUMMARY_FILE = os.path.join(
    RESULT_DIRECTORY,
    "autonomous_mm_summary.csv",
)

ALL_PATHS_FILE = os.path.join(
    RESULT_DIRECTORY,
    "autonomous_mm_all_paths.csv",
)

PARAMETER_SUMMARY_FILE = os.path.join(
    RESULT_DIRECTORY,
    "autonomous_mm_parameter_summary.csv",
)


def safe_std(series: pd.Series) -> float:
    """
    Return sample standard deviation.

    For a single path, pandas returns NaN. In that case,
    report zero.
    """
    value = series.std()

    if pd.isna(value):
        return 0.0

    return float(value)


def check_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> None:
    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)

        raise ValueError(
            "The strategy CSV files are missing these columns: "
            f"{missing_text}"
        )


def main() -> None:
    files = glob.glob(PATH_PATTERN)

    if not files:
        raise FileNotFoundError(
            "No autonomous market-maker path files were found.\n"
            f"Expected files matching: {PATH_PATTERN}"
        )

    dataframes = []

    for filename in files:
        try:
            dataframe = pd.read_csv(filename)
        except Exception as error:
            raise RuntimeError(
                f"Could not read {filename}: {error}"
            ) from error

        if dataframe.empty:
            print(
                f"Warning: ignoring empty file {filename}",
                file=sys.stderr,
            )
            continue

        dataframes.append(dataframe)

    if not dataframes:
        raise RuntimeError(
            "All autonomous market-maker CSV files were empty."
        )

    df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    required_columns = [
        "path_id",
        "seed",
        "pnl",
        "inventory",
        "cash",
        "max_drawdown",
        "max_abs_inventory",
        "fills",
        "traded_quantity",
        "final_norm_mid",
        "total_loss",
        "mean_spread",
        "median_spread",
        "p90_spread",
        "p95_spread",
        "mid_move_rate",
        "base_half_spread_ticks",
        "base_order_quantity",
        "max_inventory",
        "inventory_risk_strength",
        "volatility_strength",
        "signal_strength",
        "aggressive_inventory_ratio",
        "refresh_interval_ms",
        "final_volatility_ticks",
        "final_signal",
        "defensive_actions",
        "aggressive_actions",
    ]

    check_required_columns(
        df,
        required_columns,
    )

    # Put paths in numerical order.
    df = df.sort_values(
        by="path_id",
    ).reset_index(
        drop=True,
    )

    # Avoid accidentally summarising duplicate path files.
    duplicated_paths = df[
        df.duplicated(
            subset=["path_id"],
            keep=False,
        )
    ]

    if not duplicated_paths.empty:
        duplicate_ids = sorted(
            duplicated_paths["path_id"]
            .astype(int)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate path IDs were found: "
            f"{duplicate_ids}"
        )

    metrics = [
        "pnl",
        "inventory",
        "cash",
        "max_drawdown",
        "max_abs_inventory",
        "fills",
        "traded_quantity",
        "final_norm_mid",
        "total_loss",
        "mean_spread",
        "median_spread",
        "p90_spread",
        "p95_spread",
        "mid_move_rate",
        "final_volatility_ticks",
        "final_signal",
        "defensive_actions",
        "aggressive_actions",
    ]

    summary_rows = []

    for metric in metrics:
        series = pd.to_numeric(
            df[metric],
            errors="raise",
        )

        summary_rows.append({
            "metric": metric,
            "mean": float(series.mean()),
            "std": safe_std(series),
            "min": float(series.min()),
            "max": float(series.max()),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    mean_pnl = float(
        df["pnl"].mean()
    )

    std_pnl = safe_std(
        df["pnl"]
    )

    mean_drawdown = float(
        df["max_drawdown"].mean()
    )

    mean_max_inventory = float(
        df["max_abs_inventory"].mean()
    )

    mean_fills = float(
        df["fills"].mean()
    )

    mean_traded_quantity = float(
        df["traded_quantity"].mean()
    )

    mean_defensive_actions = float(
        df["defensive_actions"].mean()
    )

    mean_aggressive_actions = float(
        df["aggressive_actions"].mean()
    )

    profitable_path_ratio = float(
        (df["pnl"] > 0.0).mean()
    )

    if mean_traded_quantity > 0.0:
        mean_pnl_per_1000_shares = (
            1000.0
            * mean_pnl
            / mean_traded_quantity
        )
    else:
        mean_pnl_per_1000_shares = 0.0

    # Risk-adjusted strategy utility.
    #
    # Higher is better:
    # - rewards mean P&L;
    # - penalises P&L instability;
    # - penalises drawdown;
    # - penalises inventory exposure.
    strategy_score = (
        mean_pnl
        - 0.5 * std_pnl
        - 0.1 * mean_drawdown
        - 0.01 * mean_max_inventory
    )

    parameter_row = {
        "base_half_spread_ticks":
            int(df["base_half_spread_ticks"].iloc[0]),

        "base_order_quantity":
            int(df["base_order_quantity"].iloc[0]),

        "max_inventory":
            int(df["max_inventory"].iloc[0]),

        "inventory_risk_strength":
            float(df["inventory_risk_strength"].iloc[0]),

        "volatility_strength":
            float(df["volatility_strength"].iloc[0]),

        "signal_strength":
            float(df["signal_strength"].iloc[0]),

        "aggressive_inventory_ratio":
            float(df["aggressive_inventory_ratio"].iloc[0]),

        "refresh_interval_ms":
            int(df["refresh_interval_ms"].iloc[0]),

        "paths":
            int(len(df)),

        "mean_pnl":
            mean_pnl,

        "std_pnl":
            std_pnl,

        "min_pnl":
            float(df["pnl"].min()),

        "max_pnl":
            float(df["pnl"].max()),

        "profitable_path_ratio":
            profitable_path_ratio,

        "mean_pnl_per_1000_shares":
            mean_pnl_per_1000_shares,

        "mean_max_drawdown":
            mean_drawdown,

        "mean_max_abs_inventory":
            mean_max_inventory,

        "mean_final_inventory":
            float(df["inventory"].mean()),

        "mean_fills":
            mean_fills,

        "mean_traded_quantity":
            mean_traded_quantity,

        "mean_final_volatility_ticks":
            float(
                df["final_volatility_ticks"].mean()
            ),

        "mean_final_signal":
            float(
                df["final_signal"].mean()
            ),

        "mean_defensive_actions":
            mean_defensive_actions,

        "mean_aggressive_actions":
            mean_aggressive_actions,

        "mean_final_norm_mid":
            float(df["final_norm_mid"].mean()),

        "std_final_norm_mid":
            safe_std(df["final_norm_mid"]),

        "mean_total_loss":
            float(df["total_loss"].mean()),

        "mean_spread":
            float(df["mean_spread"].mean()),

        "mean_p90_spread":
            float(df["p90_spread"].mean()),

        "mean_p95_spread":
            float(df["p95_spread"].mean()),

        "mean_mid_move_rate":
            float(df["mid_move_rate"].mean()),

        "strategy_score":
            strategy_score,
    }

    parameter_summary = pd.DataFrame(
        [parameter_row]
    )

    os.makedirs(
        RESULT_DIRECTORY,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    df.to_csv(
        ALL_PATHS_FILE,
        index=False,
    )

    parameter_summary.to_csv(
        PARAMETER_SUMMARY_FILE,
        index=False,
    )

    print("\nAutonomous MM path files:")
    print(len(files))

    print("\nValid paths summarised:")
    print(len(df))

    print("\nAutonomous MM summary:")
    print(
        summary.to_string(
            index=False
        )
    )

    print("\nParameter summary:")
    print(
        parameter_summary.to_string(
            index=False
        )
    )

    print("\nStrategy interpretation:")
    print(f"Mean P&L: {mean_pnl:.6f}")
    print(f"P&L standard deviation: {std_pnl:.6f}")
    print(
        "Profitable path ratio: "
        f"{profitable_path_ratio:.2%}"
    )
    print(
        "Mean P&L per 1,000 shares: "
        f"{mean_pnl_per_1000_shares:.6f}"
    )
    print(f"Strategy score: {strategy_score:.6f}")

    print("\nSaved:")
    print(SUMMARY_FILE)
    print(ALL_PATHS_FILE)
    print(PARAMETER_SUMMARY_FILE)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"\nError: {error}",
            file=sys.stderr,
        )
        sys.exit(1)
