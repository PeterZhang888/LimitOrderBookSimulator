#include "MomentumAgent.hpp"

#include <algorithm>
#include <cmath>

int MomentumAgent::number_of_agents_ = 0;

namespace {
int sign_from_threshold(double value, double threshold) {
    if (value > threshold) return 1;
    if (value < -threshold) return -1;
    return 0;
}
}

MomentumAgent::MomentumAgent(
    std::int64_t lookback_ns,
    int order_quantity,
    double threshold_ticks,
    int tick_size,
    double order_flow_imbalance_threshold,
    double depth_imbalance_threshold,
    double strong_depth_imbalance_threshold
)
    : agent_index_(number_of_agents_++),
      lookback_ns_(std::max<std::int64_t>(1, lookback_ns)),
      order_quantity_(std::max(1, order_quantity)),
      threshold_ticks_(threshold_ticks),
      tick_size_(std::max(1, tick_size)),
      order_flow_imbalance_threshold_(std::clamp(order_flow_imbalance_threshold, 0.0, 1.0)),
      depth_imbalance_threshold_(std::clamp(depth_imbalance_threshold, 0.0, 1.0)),
      strong_depth_imbalance_threshold_(std::clamp(strong_depth_imbalance_threshold, 0.0, 1.0)) {}

void MomentumAgent::record_mid_price(const LimitOrderBook& book, std::int64_t current_time_ns) {
    if (!book.has_bid() || !book.has_ask()) return;
    history_.push_back(MarketRecord{
        current_time_ns,
        book.mid_price(),
        book.cumulative_aggressive_buy_quantity(),
        book.cumulative_aggressive_sell_quantity(),
        book.quantity_at_best_bid(),
        book.quantity_at_best_ask()
    });

    const std::int64_t keep_after = current_time_ns - 2 * lookback_ns_;
    while (!history_.empty() && history_.front().time_ns < keep_after) history_.pop_front();
}

int MomentumAgent::wake_up(LimitOrderBook& book, std::int64_t current_time_ns) {
    if (!book.has_bid() || !book.has_ask() || history_.empty()) return 0;
    const std::int64_t target_time = current_time_ns - lookback_ns_;
    bool found = false;
    MarketRecord past = history_.front();
    for (const MarketRecord& record : history_) {
        if (record.time_ns <= target_time) {
            past = record;
            found = true;
        } else break;
    }
    if (!found) return 0;

    const double mid_change_ticks = (book.mid_price() - past.mid_price) / static_cast<double>(tick_size_);
    const int mid_signal = sign_from_threshold(mid_change_ticks, threshold_ticks_);

    const std::uint64_t current_buy = book.cumulative_aggressive_buy_quantity();
    const std::uint64_t current_sell = book.cumulative_aggressive_sell_quantity();
    const std::uint64_t recent_buy = current_buy >= past.aggressive_buy_quantity ? current_buy - past.aggressive_buy_quantity : 0ULL;
    const std::uint64_t recent_sell = current_sell >= past.aggressive_sell_quantity ? current_sell - past.aggressive_sell_quantity : 0ULL;
    const std::uint64_t recent_total = recent_buy + recent_sell;

    int flow_signal = 0;
    if (recent_total >= static_cast<std::uint64_t>(std::max(1, order_quantity_ / 2))) {
        const double flow_imbalance = (static_cast<double>(recent_buy) - static_cast<double>(recent_sell)) / static_cast<double>(recent_total);
        flow_signal = sign_from_threshold(flow_imbalance, order_flow_imbalance_threshold_);
    }

    const int bid_depth = book.quantity_at_best_bid();
    const int ask_depth = book.quantity_at_best_ask();
    const int total_depth = bid_depth + ask_depth;
    int depth_signal = 0;
    if (total_depth > 0) {
        const double depth_imbalance = (static_cast<double>(bid_depth) - static_cast<double>(ask_depth)) / static_cast<double>(total_depth);
        depth_signal = sign_from_threshold(depth_imbalance, depth_imbalance_threshold_);
        if (depth_signal == 0) depth_signal = sign_from_threshold(depth_imbalance, strong_depth_imbalance_threshold_);
    }

    int direction = 0;
    if (mid_signal != 0) {
        if (flow_signal != 0 && flow_signal != mid_signal) return 0;
        direction = mid_signal;
    } else if (flow_signal != 0) {
        direction = flow_signal;
    } else if (depth_signal != 0) {
        direction = depth_signal;
    }

    if (direction == 0) return 0;
    int quantity = order_quantity_;
    if (mid_signal == 0 && flow_signal == 0 && depth_signal != 0) quantity = std::max(1, order_quantity_ / 2);

    if (direction > 0) return book.submit_market_order(Side::Buy, quantity, current_time_ns);
    return book.submit_market_order(Side::Sell, quantity, current_time_ns);
}
