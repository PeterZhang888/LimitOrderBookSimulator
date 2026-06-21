#include "MarketMakerAgent.hpp"

#include "Order.hpp"

#include <algorithm>
#include <cstdint>

int MarketMakerAgent::number_of_agents_ = 0;

MarketMakerAgent::MarketMakerAgent(
    int tick_size,
    int num_levels,
    int order_quantity,
    int level_spacing_ticks,
    int min_spread_price
)
    : agent_index_(number_of_agents_),
      tick_size_(tick_size),
      num_levels_(std::max(1, num_levels)),
      order_quantity_(std::max(1, order_quantity)),
      level_spacing_ticks_(
          std::max(1, level_spacing_ticks)
      ),
      min_spread_price_(min_spread_price)
{
    ++number_of_agents_;
}

void MarketMakerAgent::wake_up(
    LimitOrderBook& book,
    std::uint64_t& next_order_id,
    std::int64_t current_time_ns
) {
    if (!book.has_bid() || !book.has_ask()) {
        return;
    }

    if (tick_size_ <= 0) {
        return;
    }

    const int best_bid = book.best_bid();
    const int best_ask = book.best_ask();

    const int current_spread =
        best_ask - best_bid;

    if (current_spread <= 0) {
        return;
    }

    /*
     * Do not add more market-maker orders when the spread
     * is already at the configured minimum.
     */
    if (current_spread <= min_spread_price_) {
        return;
    }

    /*
     * Level 0 is placed at the current best bid and ask.
     * Higher levels are placed behind the current best prices.
     *
     * No order is placed inside the spread.
     */
    for (int level = 0; level < num_levels_; ++level) {
        const int distance_price =
            level
            * level_spacing_ticks_
            * tick_size_;

        const int bid_price =
            best_bid - distance_price;

        const int ask_price =
            best_ask + distance_price;

        if (bid_price <= 0) {
            continue;
        }

        if (bid_price >= ask_price) {
            continue;
        }

        book.add_limit_order(Order{
            next_order_id++,
            current_time_ns,
            OrderType::Limit,
            Side::Buy,
            order_quantity_,
            bid_price
        });

        book.add_limit_order(Order{
            next_order_id++,
            current_time_ns,
            OrderType::Limit,
            Side::Sell,
            order_quantity_,
            ask_price
        });
    }
}

int MarketMakerAgent::agent_index() const {
    return agent_index_;
}

int MarketMakerAgent::number_of_agents() {
    return number_of_agents_;
}
