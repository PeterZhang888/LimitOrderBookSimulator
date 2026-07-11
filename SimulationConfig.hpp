#pragma once

#include <array>
#include <cstdint>
#include <string>

struct SimulationConfig {
    int calibration_id = 0;
    std::uint64_t seed = 12345ULL;

    int tick_size = 100; // one cent in 1e-4 price ticks.
    std::int64_t start_time_ns = 0;
    std::int64_t end_time_ns = 3600LL * 1000LL * 1000LL * 1000LL;
    std::int64_t sample_interval_ns = 1LL * 1000LL * 1000LL * 1000LL;
    double initial_depth_scale = 1.0;

    // Background Hawkes agent parameters.
    std::array<double, 6> hawkes_mu{{12.0, 12.0, 2.2, 2.2, 18.0, 18.0}};
    std::array<std::array<double, 6>, 6> hawkes_alpha{};
    double hawkes_beta = 8.0;

    // Background event application.
    double quote_improvement_probability = 0.02;
    int limit_distance_shift_ticks = 0;
    double market_order_quantity_scale = 1.0;
    double cancel_quantity_scale = 1.0;

    std::string quantity_column_name = "quantity";
    std::string distance_column_name = "distance";
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

    // Agent 1: passive background market maker.
    int num_market_maker_agents = 2;
    int market_maker_num_levels = 1;
    int market_maker_order_quantity = 75;
    int market_maker_level_spacing_ticks = 1;
    int market_maker_min_spread_ticks = 2;
    std::int64_t market_maker_interval_ns = 1000LL * 1000LL * 1000LL;
    double market_maker_quote_skip_probability = 0.10;
    double market_maker_recovery_quote_skip_probability = 0.02;

    // Agent 2: momentum/order-flow/depth agent.
    int num_momentum_agents = 3;
    std::int64_t momentum_interval_ns = 1000LL * 1000LL * 1000LL;
    std::int64_t momentum_base_lookback_ns = 5LL * 1000LL * 1000LL * 1000LL;
    int momentum_order_quantity = 300;
    double momentum_threshold_ticks = 0.25;
    double momentum_order_flow_imbalance_threshold = 0.20;
    double momentum_depth_imbalance_threshold = 0.35;
    double momentum_strong_depth_imbalance_threshold = 0.55;

    // Agent 3: large institutional agent.
    int num_buy_institutional_agents = 2;
    int num_sell_institutional_agents = 2;
    int buy_institutional_parent_quantity = 50000;
    int sell_institutional_parent_quantity = 50000;
    int institutional_child_quantity = 1000;
    double institutional_participation_cap = 0.20;
    std::int64_t institutional_interval_ns = 5LL * 1000LL * 1000LL * 1000LL;
    std::int64_t institutional_base_start_time_ns = 300LL * 1000LL * 1000LL * 1000LL;
    std::int64_t institutional_start_spacing_ns = 90LL * 1000LL * 1000LL * 1000LL;
    std::int64_t institutional_duration_ns = 1200LL * 1000LL * 1000LL * 1000LL;
};
