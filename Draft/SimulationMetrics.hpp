#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct SimulationMetrics {
    int calibration_id = 0;
    std::uint64_t seed = 0;
    double duration_seconds = 0.0;
    double sample_interval_seconds = 1.0;

    std::vector<double> mid_prices;
    std::vector<double> spreads;
    std::vector<double> best_bid_depths;
    std::vector<double> best_ask_depths;
    std::vector<double> aggressive_volume_by_sample;

    std::array<double, 6> raw_hawkes_counts{{0,0,0,0,0,0}};
    double raw_hawkes_event_count = 0.0;

    double limit_order_count = 0.0;
    double limit_buy_count = 0.0;
    double limit_sell_count = 0.0;
    double market_order_count = 0.0;
    double market_buy_count = 0.0;
    double market_sell_count = 0.0;
    double market_order_executed_quantity = 0.0;
    double market_order_best_removal_count = 0.0;
    double market_order_best_removal_rate = 0.0;
    double cancel_count = 0.0;
    double cancel_bid_count = 0.0;
    double cancel_ask_count = 0.0;
    double cancel_best_removal_count = 0.0;
    double cancel_best_removal_rate = 0.0;

    double market_maker_order_count = 0.0;
    double market_maker_cancel_count = 0.0;
    double momentum_market_order_count = 0.0;
    double institutional_market_order_count = 0.0;
    double strategic_market_order_executed_quantity = 0.0;

    double total_message_count = 0.0;

    double hawkes_generation_wall_seconds = 0.0;
    double order_book_application_wall_seconds = 0.0;
    double total_wall_seconds = 0.0;
};
