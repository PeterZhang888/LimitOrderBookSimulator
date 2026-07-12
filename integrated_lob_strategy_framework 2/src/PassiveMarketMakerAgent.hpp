#pragma once

#include "LimitOrderBook.hpp"

#include <cstdint>
#include <random>

class PassiveMarketMakerAgent {
public:
    PassiveMarketMakerAgent(
        int tick_size,
        int num_levels,
        int order_quantity,
        int level_spacing_ticks,
        int min_spread_price,
        double quote_skip_probability,
        double recovery_quote_skip_probability,
        std::uint64_t seed
    );

    std::size_t wake_up(LimitOrderBook& book, std::uint64_t& next_order_id, std::int64_t current_time_ns);

private:
    int agent_index_ = 0;
    int tick_size_ = 100;
    int num_levels_ = 1;
    int order_quantity_ = 100;
    int level_spacing_ticks_ = 1;
    int min_spread_price_ = 200;
    double quote_skip_probability_ = 0.0;
    double recovery_quote_skip_probability_ = 0.0;
    int inventory_ = 0;
    double cash_ = 0.0;
    std::mt19937_64 rng_;

    static int number_of_agents_;
};
