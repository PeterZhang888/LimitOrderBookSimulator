#ifndef SIMULATION_CONFIG_HPP
#define SIMULATION_CONFIG_HPP

#include <cstdint>
#include <string>

struct SimulationConfig {
    int calibration_id = 0;
    std::string calibration_name = "default";

    std::uint64_t seed = 12353;

    std::int64_t start_time_ns = 0;
    std::int64_t end_time_ns = 1000LL * 1000000000LL;
    std::int64_t sample_interval_ns = 100000000LL;

    int tick_size = 100;

    double empirical_rate_limit_buy = 47.037011;
    double empirical_rate_limit_sell = 47.321285;
    double empirical_rate_market_buy = 0.889317;
    double empirical_rate_market_sell = 0.893212;
    double empirical_rate_cancel_bid = 45.980008;
    double empirical_rate_cancel_ask = 46.627447;

    double hawkes_beta = 5.0;
    double limit_branching = 0.30;
    double market_branching = 0.45;
    double cancel_branching = 0.08;

    bool use_hardcoded_initial_book = true;
    double initial_depth_scale = 1.0;

    std::string limit_buy_quantity_file = "limit_buy_quantity_distribution.txt";
    std::string limit_sell_quantity_file = "limit_sell_quantity_distribution.txt";
    std::string market_buy_quantity_file = "market_buy_quantity_distribution.txt";
    std::string market_sell_quantity_file = "market_sell_quantity_distribution.txt";
    std::string cancel_bid_quantity_file = "cancel_bid_quantity_distribution.txt";
    std::string cancel_ask_quantity_file = "cancel_ask_quantity_distribution.txt";

    std::string limit_buy_distance_file = "limit_buy_distance_distribution.txt";
    std::string limit_sell_distance_file = "limit_sell_distance_distribution.txt";
    std::string cancel_bid_distance_file = "cancel_bid_distance_distribution.txt";
    std::string cancel_ask_distance_file = "cancel_ask_distance_distribution.txt";

    std::string quantity_column_name = "quantity";
    std::string distance_column_name = "distance_ticks";

    int num_market_maker_agents = 5;
    int num_momentum_agents = 3;
    int num_institutional_agents = 6;

    std::int64_t market_maker_interval_ns = 750LL * 1000000LL;

    int market_maker_num_levels = 5;
    int market_maker_order_quantity = 250;
    int market_maker_level_spacing_ticks = 1;
    int market_maker_min_spread_ticks = 1;

    bool heterogeneous_market_makers = true;

    int market_maker_num_levels_step = -1;
    int market_maker_order_quantity_step = -25;
    int market_maker_min_spread_ticks_step = 1;

    int market_maker_min_num_levels = 1;
    int market_maker_min_order_quantity = 25;

    std::int64_t momentum_interval_ns = 1LL * 1000000000LL;
    std::int64_t momentum_base_lookback_ns = 2LL * 1000000000LL;
    std::int64_t momentum_lookback_step_ns = 3LL * 1000000000LL;

    int momentum_order_quantity = 300;
    int momentum_order_quantity_step = 0;

    double momentum_threshold_ticks = 0.5;
    double momentum_threshold_step_ticks = 0.0;

    bool heterogeneous_momentum_agents = true;

    std::int64_t institutional_interval_ns = 1LL * 1000000000LL;

    int institutional_side = 1;

    bool use_mixed_institutional_sides = true;

    int num_buy_institutional_agents = 4;
    int num_sell_institutional_agents = 2;

    int buy_institutional_parent_quantity = 450000;
    int sell_institutional_parent_quantity = 225000;

    int institutional_parent_quantity = 300000;
    int institutional_parent_quantity_step = 0;

    int institutional_child_quantity = 1200;
    int institutional_child_quantity_step = 0;

    std::int64_t institutional_base_start_time_ns = 0LL;
    std::int64_t institutional_start_spacing_ns = 150LL * 1000000000LL;
    std::int64_t institutional_duration_ns = 500LL * 1000000000LL;

    bool heterogeneous_institutional_agents = true;

    int limit_distance_shift_ticks = 0;
    double market_order_quantity_scale = 1.0;

    double quote_improvement_probability = 0.10;

    bool write_detailed_outputs = false;

    std::string simulated_timeseries_file = "simulated_book_timeseries.csv";
    std::string simulated_events_file = "simulated_events.csv";
    std::string simulated_summary_file = "simulated_summary.csv";

    double target_mean_spread = 135.956404;
    double target_median_spread = 100.0;
    double target_p90_spread = 200.0;
    double target_p95_spread = 200.0;

    double target_mean_mid_change = 1.4;
    double target_std_mid_change = 66.317266;
    double target_zero_mid_change_ratio = 0.5058;
    double target_mid_move_rate = 0.4942;
    double target_final_norm_mid = 14000.0;

    double weight_mean_spread = 1.0;
    double weight_median_spread = 1.0;
    double weight_p90_spread = 1.0;
    double weight_p95_spread = 1.0;

    double weight_mean_mid_change = 2.0;
    double weight_std_mid_change = 1.0;
    double weight_zero_mid_change_ratio = 1.0;
    double weight_mid_move_rate = 1.0;
    double weight_final_norm_mid = 3.0;
};

#endif
