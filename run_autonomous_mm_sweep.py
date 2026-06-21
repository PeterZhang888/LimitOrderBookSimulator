import csv
import glob
import os
import re
import shutil
import subprocess
from pathlib import Path

# ============================================================
# Autonomous Market Maker parameter sweep
#
# Usage from your thesis folder:
#   python3 run_autonomous_mm_sweep.py
#
# This script:
#   1. Edits the constexpr autonomous MM parameters in SimulationRunner.cpp
#   2. Rebuilds the C++ project
#   3. Runs mpirun -np 10 ./calibrate 10
#   4. Collects autonomous_mm_path_*.csv
#   5. Writes sweep_results/autonomous_mm_sweep_summary.csv
# ============================================================

SIMULATION_RUNNER = Path("SimulationRunner.cpp")
RESULTS_DIR = Path("mc_full_day_results")
SWEEP_DIR = Path("sweep_results")
BACKUP_FILE = Path("SimulationRunner.cpp.before_autonomous_mm_sweep")

NUM_RANKS = 10
NUM_PATHS = 10
BASE_SEED = 12345

# Search only 3 parameters first. Keep max_inventory and refresh fixed.
QUOTE_HALF_SPREAD_GRID = [4,5,6]
ORDER_QUANTITY_GRID = [2, 5, 10]
INVENTORY_SKEW_GRID = [1.0]

FIXED_MAX_INVENTORY = 500
FIXED_REFRESH_INTERVAL_NS = "1000LL * 1000000LL"

# Strategy score. Higher is better.
# Conservative: rewards mean P&L, penalises P&L volatility, drawdown, and inventory risk.
def strategy_score(mean_pnl, std_pnl, mean_drawdown, mean_max_abs_inventory):
    return (
        mean_pnl
        - 0.5 * std_pnl
        - 0.1 * mean_drawdown
        - 0.01 * mean_max_abs_inventory
    )


def run_command(command, log_file=None):
    print("\n$ " + " ".join(command), flush=True)

    if log_file is None:
        subprocess.run(command, check=True)
        return

    with open(log_file, "w") as out:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            out.write(line)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def replace_constant(text, name, value):
    pattern = rf"constexpr\s+[^;]+\s+{name}\s*=\s*[^;]+;"
    replacement = None

    if isinstance(value, int):
        replacement = f"constexpr int {name} = {value};"
    elif isinstance(value, float):
        replacement = f"constexpr double {name} = {value};"
    elif isinstance(value, str):
        replacement = f"constexpr std::int64_t {name} = {value};"
    else:
        raise TypeError(f"Unsupported value type for {name}: {type(value)}")

    new_text, count = re.subn(pattern, replacement, text)

    if count != 1:
        raise RuntimeError(
            f"Could not replace exactly one definition of {name}. Replaced {count}."
        )

    return new_text


def set_strategy_parameters(
    quote_half_spread_ticks,
    order_quantity,
    max_inventory,
    inventory_skew_strength,
    refresh_interval_ns,
):
    text = SIMULATION_RUNNER.read_text()

    text = replace_constant(
        text,
        "AUTONOMOUS_MM_QUOTE_HALF_SPREAD_TICKS",
        quote_half_spread_ticks,
    )
    text = replace_constant(
        text,
        "AUTONOMOUS_MM_ORDER_QUANTITY",
        order_quantity,
    )
    text = replace_constant(
        text,
        "AUTONOMOUS_MM_MAX_INVENTORY",
        max_inventory,
    )
    text = replace_constant(
        text,
        "AUTONOMOUS_MM_INVENTORY_SKEW_STRENGTH",
        float(inventory_skew_strength),
    )
    text = replace_constant(
        text,
        "AUTONOMOUS_MM_REFRESH_INTERVAL_NS",
        refresh_interval_ns,
    )

    SIMULATION_RUNNER.write_text(text)


def clean_previous_path_csvs():
    RESULTS_DIR.mkdir(exist_ok=True)
    for file in RESULTS_DIR.glob("autonomous_mm_path_*.csv"):
        file.unlink()


def read_path_results():
    files = sorted(RESULTS_DIR.glob("autonomous_mm_path_*.csv"))

    if not files:
        raise RuntimeError("No autonomous_mm_path_*.csv files were produced.")

    rows = []
    for file in files:
        with open(file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    if not rows:
        raise RuntimeError("Path CSV files were found but contained no rows.")

    return rows, files


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / (len(values) - 1)) ** 0.5


def summarise_rows(rows):
    pnl = [float(r["pnl"]) for r in rows]
    max_drawdown = [float(r["max_drawdown"]) for r in rows]
    max_abs_inventory = [float(r["max_abs_inventory"]) for r in rows]
    fills = [float(r["fills"]) for r in rows]
    traded_quantity = [float(r["traded_quantity"]) for r in rows]
    final_norm_mid = [float(r["final_norm_mid"]) for r in rows]
    total_loss = [float(r["total_loss"]) for r in rows]
    mean_spread = [float(r["mean_spread"]) for r in rows]
    p90_spread = [float(r["p90_spread"]) for r in rows]
    p95_spread = [float(r["p95_spread"]) for r in rows]
    mid_move_rate = [float(r["mid_move_rate"]) for r in rows]

    mean_pnl = mean(pnl)
    std_pnl = std(pnl)
    mean_drawdown = mean(max_drawdown)
    mean_inventory = mean(max_abs_inventory)

    return {
        "paths": len(rows),
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "min_pnl": min(pnl),
        "max_pnl": max(pnl),
        "mean_max_drawdown": mean_drawdown,
        "mean_max_abs_inventory": mean_inventory,
        "mean_fills": mean(fills),
        "mean_traded_quantity": mean(traded_quantity),
        "mean_final_norm_mid": mean(final_norm_mid),
        "mean_total_loss": mean(total_loss),
        "mean_spread": mean(mean_spread),
        "mean_p90_spread": mean(p90_spread),
        "mean_p95_spread": mean(p95_spread),
        "mean_mid_move_rate": mean(mid_move_rate),
        "strategy_score": strategy_score(
            mean_pnl,
            std_pnl,
            mean_drawdown,
            mean_inventory,
        ),
    }


def main():
    if not SIMULATION_RUNNER.exists():
        raise FileNotFoundError("SimulationRunner.cpp not found. Run this from the thesis folder.")

    SWEEP_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    if not BACKUP_FILE.exists():
        shutil.copy2(SIMULATION_RUNNER, BACKUP_FILE)
        print(f"Backup created: {BACKUP_FILE}")
    else:
        print(f"Backup already exists: {BACKUP_FILE}")

    summary_file = SWEEP_DIR / "autonomous_mm_sweep_summary.csv"

    fieldnames = [
        "config_id",
        "quote_half_spread_ticks",
        "order_quantity",
        "max_inventory",
        "inventory_skew_strength",
        "refresh_interval_ms",
        "paths",
        "mean_pnl",
        "std_pnl",
        "min_pnl",
        "max_pnl",
        "mean_max_drawdown",
        "mean_max_abs_inventory",
        "mean_fills",
        "mean_traded_quantity",
        "mean_final_norm_mid",
        "mean_total_loss",
        "mean_spread",
        "mean_p90_spread",
        "mean_p95_spread",
        "mean_mid_move_rate",
        "strategy_score",
    ]

    completed_rows = []
    config_id = 0

    try:
        for quote_half_spread_ticks in QUOTE_HALF_SPREAD_GRID:
            for order_quantity in ORDER_QUANTITY_GRID:
                for inventory_skew_strength in INVENTORY_SKEW_GRID:
                    print("\n" + "=" * 72)
                    print(
                        "Config",
                        config_id,
                        "quote_half_spread_ticks=",
                        quote_half_spread_ticks,
                        "order_quantity=",
                        order_quantity,
                        "inventory_skew_strength=",
                        inventory_skew_strength,
                    )
                    print("=" * 72)

                    set_strategy_parameters(
                        quote_half_spread_ticks=quote_half_spread_ticks,
                        order_quantity=order_quantity,
                        max_inventory=FIXED_MAX_INVENTORY,
                        inventory_skew_strength=inventory_skew_strength,
                        refresh_interval_ns=FIXED_REFRESH_INTERVAL_NS,
                    )

                    clean_previous_path_csvs()

                    config_dir = SWEEP_DIR / f"config_{config_id:03d}"
                    config_dir.mkdir(exist_ok=True)

                    run_command(["make", "clean"], log_file=config_dir / "make_clean.log")
                    run_command(["make"], log_file=config_dir / "make.log")
                    run_command(
                        [
                            "mpirun",
                            "-np",
                            str(NUM_RANKS),
                            "./calibrate",
                            str(NUM_PATHS),
                        ],
                        log_file=config_dir / "run.log",
                    )

                    rows, files = read_path_results()

                    for file in files:
                        shutil.copy2(file, config_dir / file.name)

                    stats = summarise_rows(rows)

                    row = {
                        "config_id": config_id,
                        "quote_half_spread_ticks": quote_half_spread_ticks,
                        "order_quantity": order_quantity,
                        "max_inventory": FIXED_MAX_INVENTORY,
                        "inventory_skew_strength": inventory_skew_strength,
                        "refresh_interval_ms": 1000,
                    }
                    row.update(stats)

                    completed_rows.append(row)

                    with open(summary_file, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(completed_rows)

                    print("\nCurrent config summary:")
                    print(row)
                    print(f"\nUpdated sweep summary: {summary_file}")

                    config_id += 1

    finally:
        # Restore the original SimulationRunner.cpp so the sweep script does not leave
        # the source tree in a random parameter configuration.
        shutil.copy2(BACKUP_FILE, SIMULATION_RUNNER)
        print(f"\nRestored original SimulationRunner.cpp from {BACKUP_FILE}")

    # Print top configurations.
    completed_rows = sorted(
        completed_rows,
        key=lambda r: float(r["strategy_score"]),
        reverse=True,
    )

    print("\nTop 10 configurations by strategy_score:")
    for row in completed_rows[:10]:
        print(
            f"config={row['config_id']} "
            f"spread={row['quote_half_spread_ticks']} "
            f"qty={row['order_quantity']} "
            f"skew={row['inventory_skew_strength']} "
            f"mean_pnl={row['mean_pnl']:.4f} "
            f"std_pnl={row['std_pnl']:.4f} "
            f"drawdown={row['mean_max_drawdown']:.4f} "
            f"inventory={row['mean_max_abs_inventory']:.4f} "
            f"score={row['strategy_score']:.4f}"
        )


if __name__ == "__main__":
    main()

