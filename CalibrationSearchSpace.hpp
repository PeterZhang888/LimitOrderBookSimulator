#ifndef CALIBRATION_SEARCH_SPACE_HPP
#define CALIBRATION_SEARCH_SPACE_HPP

#include "SimulationConfig.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

struct CalibrationSearchSpace {
    // ========================================================
    // 0. Repetitions per parameter configuration
    // ========================================================
    //
    // Each real parameter configuration is run 3 times with
    // different seeds. This makes the calibration less dependent
    // on one lucky or unlucky random path.

    std::vector<int> repeat_id_values = {
        0,
        1,
        2
    };

    // ========================================================
    // 1. Hawkes process
    // ========================================================
    //
    // In M2, Hawkes parameters are still NOT searched here.
    // The fitted full 6x6 MLE Hawkes model is loaded inside
    // SimulationRunner.cpp from:
    //
    // hawkes_mle_fixed_beta_lbfgsb_eta010/fixed_beta_mle_params_flat.csv
    //
    // Therefore the scalar Hawkes fields below are placeholders
    // only, kept so that SimulationConfig remains valid.

    const double fitted_hawkes_beta_placeholder = 10.0;
    const double fitted_limit_branching_placeholder = 0.0;
    const double fitted_market_branching_placeholder = 0.0;
    const double fitted_cancel_branching_placeholder = 0.0;

    // ========================================================
    // 2. Market maker grid
    // ========================================================
    //
    // Previous M1 result:
    // mean_spread was acceptable, but mid_move_rate was too low.
    //
    // Therefore M2 keeps the market-maker region near the best
    // M1 configurations, while slightly weakening passive depth.
    //
    // We do NOT expand this grid too much, because the main
    // problem is not only passive liquidity. The main problem is
    // that aggressive flow is not strong enough to move the best
    // bid/ask frequently.

    std::vector<int> num_market_maker_agents_values = {
        2
    };

    std::vector<int> market_maker_order_quantity_values = {
        50,
        75,
        100
    };

    std::vector<int> market_maker_min_spread_ticks_values = {
        2
    };

    std::vector<int> market_maker_interval_ms_values = {
        1000,
        1500
    };

    std::vector<double> quote_improvement_probability_values = {
        0.00,
        0.03
    };

    // ========================================================
    // 3. Momentum agent grid
    // ========================================================
    //
    // M1 kept momentum mostly fixed.
    //
    // M2 searches momentum more actively because the mid-price
    // does not move often enough. Lower threshold means the
    // momentum agent reacts more easily. Larger order quantity
    // means it can consume more best-level liquidity.

    std::vector<int> num_momentum_agents_values = {
        3,
        5
    };

    std::vector<int> momentum_order_quantity_values = {
        300,
        500
    };

    std::vector<double> momentum_threshold_ticks_values = {
        0.25,
        0.5
    };

    // ========================================================
    // 4. Institutional agent grid
    // ========================================================
    //
    // Keep the mixed buy/sell institutional structure explicit.
    // Search child size more aggressively, because larger child
    // orders are more likely to consume depth and move the quote.

    std::vector<int> num_buy_institutional_agents_values = {
        4
    };

    std::vector<int> num_sell_institutional_agents_values = {
        2
    };

    std::vector<int> buy_institutional_parent_quantity_values = {
        330'000
    };

    std::vector<int> sell_institutional_parent_quantity_values = {
        350'000
    };

    std::vector<int> institutional_child_quantity_values = {
        1000,
        1500,
        2000
    };

    std::vector<int> institutional_interval_seconds_values = {
        1
    };

    std::vector<int> institutional_start_spacing_seconds_values = {
        150
    };

    std::vector<int> institutional_duration_seconds_values = {
        500
    };

    // ========================================================
    // 5. Limit-order placement grid
    // ========================================================
    //
    // Keep this fixed in M2. The current target is mid-price
    // movement, not limit-order distance calibration.

    std::vector<int> limit_distance_shift_ticks_values = {
        0
    };

    // ========================================================
    // 6. Market-order quantity scale grid
    // ========================================================
    //
    // This is important for M2.
    //
    // If Hawkes-generated market orders are too small, they may
    // hit the book but fail to remove the whole best level.
    // In that case, the best bid/ask stays unchanged and the
    // mid-price does not move.
    //
    // Increasing this scale directly tests whether aggressive
    // Hawkes flow is too weak.

    std::vector<double> market_order_quantity_scale_values = {
        1.0,
        1.5,
        2.0
    };

    // ========================================================
    // Total number of configurations
    // ========================================================
    //
    // Total M2 paths:
    //
    // repeats                         = 3
    // market maker grid               = 1 * 3 * 1 * 2 * 2 = 12
    // momentum grid                   = 2 * 2 * 2 = 8
    // institutional child grid        = 3
    // market order quantity scale     = 3
    //
    // total = 3 * 12 * 8 * 3 * 3 = 2592

    std::size_t total_configs() const {
        std::size_t n = 1;

        n *= repeat_id_values.size();

        n *= num_market_maker_agents_values.size();
        n *= market_maker_order_quantity_values.size();
        n *= market_maker_min_spread_ticks_values.size();
        n *= market_maker_interval_ms_values.size();
        n *= quote_improvement_probability_values.size();

        n *= num_momentum_agents_values.size();
        n *= momentum_order_quantity_values.size();
        n *= momentum_threshold_ticks_values.size();

        n *= num_buy_institutional_agents_values.size();
        n *= num_sell_institutional_agents_values.size();
        n *= buy_institutional_parent_quantity_values.size();
        n *= sell_institutional_parent_quantity_values.size();
        n *= institutional_child_quantity_values.size();
        n *= institutional_interval_seconds_values.size();
        n *= institutional_start_spacing_seconds_values.size();
        n *= institutional_duration_seconds_values.size();

        n *= limit_distance_shift_ticks_values.size();
        n *= market_order_quantity_scale_values.size();

        return n;
    }

    // ========================================================
    // Convert config index into one concrete SimulationConfig
    // ========================================================

    SimulationConfig make_config(std::size_t index) const {
        if (index >= total_configs()) {
            throw std::out_of_range(
                "Calibration config index out of range."
            );
        }

        const std::size_t original_index = index;

        SimulationConfig config;

        config.calibration_id =
            static_cast<int>(original_index);

        auto pick = [&index](const auto& values) {
            const std::size_t n = values.size();
            const std::size_t chosen = index % n;
            index /= n;
            return values[chosen];
        };

        // ====================================================
        // Repetition id
        // ====================================================
        //
        // repeat_id is not written into SimulationConfig unless
        // SimulationConfig has such a field.
        //
        // Instead, it is used in:
        // 1. calibration_name
        // 2. seed
        //
        // Because repeat_id is picked first, paths are arranged as:
        //
        // path 0 = parameter config 0, repeat 0
        // path 1 = parameter config 0, repeat 1
        // path 2 = parameter config 0, repeat 2
        // path 3 = parameter config 1, repeat 0
        // ...

        const int repeat_id =
            pick(repeat_id_values);

        const std::size_t parameter_index = index;

        config.calibration_name =
            "m2_mid_move_search_param_"
            + std::to_string(parameter_index)
            + "_rep_"
            + std::to_string(repeat_id)
            + "_path_"
            + std::to_string(original_index);

        // ====================================================
        // Hawkes placeholders
        // ====================================================
        //
        // These are NOT used to build the M2 Hawkes process.
        // SimulationRunner.cpp loads the fitted full 6x6 MLE
        // Hawkes parameters from CSV and overwrites the result
        // reporting fields.

        config.hawkes_beta =
            fitted_hawkes_beta_placeholder;

        config.limit_branching =
            fitted_limit_branching_placeholder;

        config.market_branching =
            fitted_market_branching_placeholder;

        config.cancel_branching =
            fitted_cancel_branching_placeholder;

        // ====================================================
        // Market makers
        // ====================================================

        config.num_market_maker_agents =
            pick(num_market_maker_agents_values);

        config.market_maker_order_quantity =
            pick(market_maker_order_quantity_values);

        config.market_maker_min_spread_ticks =
            pick(market_maker_min_spread_ticks_values);

        const int market_maker_interval_ms =
            pick(market_maker_interval_ms_values);

        config.market_maker_interval_ns =
            static_cast<std::int64_t>(
                market_maker_interval_ms
            )
            * 1'000'000LL;

        config.quote_improvement_probability =
            pick(quote_improvement_probability_values);

        config.market_maker_num_levels = 5;
        config.market_maker_level_spacing_ticks = 1;

        config.heterogeneous_market_makers = true;

        config.market_maker_num_levels_step = -1;
        config.market_maker_order_quantity_step = -25;
        config.market_maker_min_spread_ticks_step = 1;

        config.market_maker_min_num_levels = 1;
        config.market_maker_min_order_quantity = 25;

        // ====================================================
        // Momentum agents
        // ====================================================

        config.num_momentum_agents =
            pick(num_momentum_agents_values);

        config.momentum_order_quantity =
            pick(momentum_order_quantity_values);

        config.momentum_threshold_ticks =
            pick(momentum_threshold_ticks_values);

        config.momentum_interval_ns =
            1LL * 1'000'000'000LL;

        config.momentum_base_lookback_ns =
            2LL * 1'000'000'000LL;

        config.momentum_lookback_step_ns =
            3LL * 1'000'000'000LL;

        config.heterogeneous_momentum_agents = true;

        config.momentum_order_quantity_step = 0;
        config.momentum_threshold_step_ticks = 0.0;

        // ====================================================
        // Institutional agents
        // ====================================================

        config.use_mixed_institutional_sides = true;

        config.num_buy_institutional_agents =
            pick(num_buy_institutional_agents_values);

        config.num_sell_institutional_agents =
            pick(num_sell_institutional_agents_values);

        config.num_institutional_agents =
            config.num_buy_institutional_agents
            + config.num_sell_institutional_agents;

        config.buy_institutional_parent_quantity =
            pick(buy_institutional_parent_quantity_values);

        config.sell_institutional_parent_quantity =
            pick(sell_institutional_parent_quantity_values);

        config.institutional_parent_quantity =
            config.buy_institutional_parent_quantity;

        config.institutional_child_quantity =
            pick(institutional_child_quantity_values);

        const int institutional_interval_seconds =
            pick(institutional_interval_seconds_values);

        const int institutional_start_spacing_seconds =
            pick(institutional_start_spacing_seconds_values);

        const int institutional_duration_seconds =
            pick(institutional_duration_seconds_values);

        config.institutional_interval_ns =
            static_cast<std::int64_t>(
                institutional_interval_seconds
            )
            * 1'000'000'000LL;

        config.institutional_start_spacing_ns =
            static_cast<std::int64_t>(
                institutional_start_spacing_seconds
            )
            * 1'000'000'000LL;

        config.institutional_duration_ns =
            static_cast<std::int64_t>(
                institutional_duration_seconds
            )
            * 1'000'000'000LL;

        config.institutional_base_start_time_ns = 0LL;

        config.institutional_side = 1;

        config.heterogeneous_institutional_agents = true;

        config.institutional_parent_quantity_step = 0;
        config.institutional_child_quantity_step = 0;

        // ====================================================
        // Limit-order placement
        // ====================================================

        config.limit_distance_shift_ticks =
            pick(limit_distance_shift_ticks_values);

        // ====================================================
        // Market-order quantity scale
        // ====================================================

        config.market_order_quantity_scale =
            pick(market_order_quantity_scale_values);

        // ====================================================
        // Seed
        // ====================================================
        //
        // Same parameter configuration gets 3 independent seeds.
        //
        // Example:
        //
        // parameter config 0:
        //   repeat 0 seed = 12345
        //   repeat 1 seed = 12346
        //   repeat 2 seed = 12347
        //
        // parameter config 1:
        //   repeat 0 seed = 12445
        //   repeat 1 seed = 12446
        //   repeat 2 seed = 12447

        config.seed =
            12345ULL
            + static_cast<std::uint64_t>(
                parameter_index
            )
            * 100ULL
            + static_cast<std::uint64_t>(
                repeat_id
            );

        // ====================================================
        // No detailed output during MPI grid search
        // ====================================================

        config.write_detailed_outputs = false;

        return config;
    }
};

#endif
