#include "MomentumAgent.hpp"

#include <cmath>

int MomentumAgent::number_of_agents_ = 0;

MomentumAgent::MomentumAgent(
    std::int64_t lookback_ns,
    int order_quantity,
    double threshold_ticks,
    int tick_size
)
    : agent_index_(number_of_agents_++),
      lookback_ns_(lookback_ns),
      order_quantity_(order_quantity),
      threshold_ticks_(threshold_ticks),
      tick_size_(tick_size)
{
}

void MomentumAgent::record_mid_price(
    const LimitOrderBook& book,
    std::int64_t current_time_ns
) {
    if (!book.has_bid() || !book.has_ask()) {
        return;
    }

    mid_history_.push_back(MidRecord{
        current_time_ns,
        book.mid_price()
    });

    std::int64_t keep_after = current_time_ns - 2 * lookback_ns_;

    while (!mid_history_.empty() &&
           mid_history_.front().time_ns < keep_after) {
        mid_history_.pop_front();
    }
}

void MomentumAgent::wake_up(
    LimitOrderBook& book,
    std::int64_t current_time_ns
) {
    if (!book.has_bid() || !book.has_ask()) {
        return;
    }

    if (mid_history_.empty()) {
        return;
    }

    double current_mid = book.mid_price();

    std::int64_t target_time_ns = current_time_ns - lookback_ns_;

    bool found_past = false;
    double past_mid = mid_history_.front().mid_price;

    for (const auto& record : mid_history_) {
        if (record.time_ns <= target_time_ns) {
            past_mid = record.mid_price;
            found_past = true;
        } else {
            break;
        }
    }

    if (!found_past) {
        return;
    }

    double mid_change_ticks =
        (current_mid - past_mid) / static_cast<double>(tick_size_);

    if (mid_change_ticks > threshold_ticks_) {
        // Upward momentum: consume ask liquidity.
        book.submit_market_order(
            Side::Buy,
            order_quantity_
        );
    }
    else if (mid_change_ticks < -threshold_ticks_) {
        // Downward momentum: consume bid liquidity.
        book.submit_market_order(
            Side::Sell,
            order_quantity_
        );
    }
}

int MomentumAgent::agent_index() const {
    return agent_index_;
}

int MomentumAgent::number_of_agents() {
    return number_of_agents_;
}
