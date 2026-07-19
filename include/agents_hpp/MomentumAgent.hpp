#pragma once

#include "LimitOrderBook.hpp"

#include <cstdint>
#include <deque>

class MomentumAgent {
public:
    MomentumAgent(
        std::int64_t lookback_ns,
        int order_quantity,
        double threshold_ticks,
        int tick_size,
        double order_flow_imbalance_threshold,
        double depth_imbalance_threshold,
        double strong_depth_imbalance_threshold
    );

    void record_mid_price(const LimitOrderBook& book, std::int64_t current_time_ns);
    int wake_up(LimitOrderBook& book, std::int64_t current_time_ns);

private:
    struct MarketRecord {
        std::int64_t time_ns = 0;
        double mid_price = 0.0;
        std::uint64_t aggressive_buy_quantity = 0;
        std::uint64_t aggressive_sell_quantity = 0;
        int best_bid_depth = 0;
        int best_ask_depth = 0;
    };

    int agent_index_ = 0;
    std::int64_t lookback_ns_ = 0;
    int order_quantity_ = 0;
    double threshold_ticks_ = 0.0;
    int tick_size_ = 100;
    double order_flow_imbalance_threshold_ = 0.20;
    double depth_imbalance_threshold_ = 0.35;
    double strong_depth_imbalance_threshold_ = 0.55;
    std::deque<MarketRecord> history_;

    static int number_of_agents_;
};
