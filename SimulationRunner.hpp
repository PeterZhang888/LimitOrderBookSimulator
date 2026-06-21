#ifndef SIMULATION_RUNNER_HPP
#define SIMULATION_RUNNER_HPP

#include "SimulationConfig.hpp"

// ============================================================
// CalibrationResult
//
// One object stores:
//   1. the parameter configuration identity,
//   2. the parameter values used,
//   3. the simulated summary statistics,
//   4. the final calibration loss.
//
// During MPI calibration, each rank keeps only its local top 10
// CalibrationResult objects.
// ============================================================

struct CalibrationResult {
    // ========================================================
    // 1. MPI / calibration identity
    // ========================================================

    int rank = 0;

    int calibration_id = 0;

    // ========================================================
    // 2. Loss
    // ========================================================

    double total_loss = 0.0;

    // ========================================================
    // 3. Hawkes parameters
    // ========================================================

    double hawkes_beta = 0.0;

    double limit_branching = 0.0;

    double market_branching = 0.0;

    double cancel_branching = 0.0;

    // ========================================================
    // 4. Market maker parameters
    // ========================================================

    int num_market_maker_agents = 0;

    int market_maker_order_quantity = 0;

    int market_maker_min_spread_ticks = 0;

    // ========================================================
    // 5. Momentum agent parameters
    // ========================================================

    int num_momentum_agents = 0;

    int momentum_order_quantity = 0;

    double momentum_threshold_ticks = 0.0;

    // ========================================================
    // 6. Institutional agent parameters
    // ========================================================

    int num_institutional_agents = 0;

    int institutional_parent_quantity = 0;

    int institutional_child_quantity = 0;

    int institutional_interval_seconds = 0;

    // ========================================================
    // 7. Spread statistics
    // ========================================================

    double mean_spread = 0.0;

    double median_spread = 0.0;

    double p90_spread = 0.0;

    double p95_spread = 0.0;

    // ========================================================
    // 8. Mid-price dynamics statistics
    // ========================================================

    double std_mid_change = 0.0;

    double zero_mid_change_ratio = 0.0;

    double mid_move_rate = 0.0;

    double final_norm_mid = 0.0;

    double mean_mid_change = 0.0;
};

// ============================================================
// run_single_simulation
//
// This function should contain the logic from your previous
// single-run main.cpp.
//
// It should:
//   1. read all parameters from SimulationConfig,
//   2. build the initial book,
//   3. create Hawkes background order flow,
//   4. create all agent populations,
//   5. run the simulation,
//   6. compute summary statistics,
//   7. compute total calibration loss,
//   8. return CalibrationResult.
//
// Important:
//   During MPI calibration, config.write_detailed_outputs should
//   normally be false, so this function should NOT write millions
//   of event/time-series files.
// ============================================================

CalibrationResult run_single_simulation(
    const SimulationConfig& config
);

#endif
