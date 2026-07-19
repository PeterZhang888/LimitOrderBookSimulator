#ifndef MARKET_MAKER_AGENT_HPP
#define MARKET_MAKER_AGENT_HPP

#include "LimitOrderBook.hpp"

#include <cstdint>

class MarketMakerAgent {
public:
    MarketMakerAgent(
        int tick_size,
        int num_levels,
        int order_quantity,
        int level_spacing_ticks,
        int min_spread_price
    );

    MarketMakerAgent(
        const MarketMakerAgent&
    ) = delete;

    MarketMakerAgent& operator=(
        const MarketMakerAgent&
    ) = delete;

    MarketMakerAgent(
        MarketMakerAgent&&
    ) = default;

    MarketMakerAgent& operator=(
        MarketMakerAgent&&
    ) = default;

    void wake_up(
        LimitOrderBook& book,
        std::uint64_t& next_order_id,
        std::int64_t current_time_ns
    );

    int agent_index() const;

    static int number_of_agents();

private:
    int agent_index_;

    int tick_size_;
    int num_levels_;
    int order_quantity_;
    int level_spacing_ticks_;
    int min_spread_price_;

    static int number_of_agents_;
};

#endif
