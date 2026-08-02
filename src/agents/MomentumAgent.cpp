#include "agents/MomentumAgent.hpp"

#include <algorithm>
#include <cmath>

namespace dlob {
namespace {
int sign_from_threshold(double value, double threshold) {
    if (value > threshold) return 1;
    if (value < -threshold) return -1;
    return 0;
}
} // namespace

MomentumAgent::MomentumAgent(int owner_id,
                             const MomentumAgentConfig& config,
                             std::int64_t simulation_start_ns,
                             std::uint64_t seed)
    : owner_id_(owner_id), config_(config), rng_(seed) {
    next_wake_ns_ = safe_add_time(simulation_start_ns, rng_.exponential_wait_ns(config_.wake_rate_per_second));
}

void MomentumAgent::generate_orders(const MarketState& current,
                                    const MarketState* past,
                                    std::int64_t window_start_ns,
                                    std::int64_t window_end_ns,
                                    OrderMessageBuilder& builder,
                                    std::vector<OrderMessage>& out) {
    const bool valid_book = current.best_bid_ticks > 0 && current.best_ask_ticks > 0;

    while (next_wake_ns_ < window_end_ns) {
        const std::int64_t decision_time = std::max(next_wake_ns_, window_start_ns);
        if (valid_book && past != nullptr) {
            const double mid_change_ticks = (current.mid_price_ticks - past->mid_price_ticks)
                / static_cast<double>(std::max(1, config_.tick_size));
            const int mid_signal = sign_from_threshold(mid_change_ticks, config_.threshold_ticks);

            const std::uint64_t recent_buy = current.cumulative_aggressive_buy >= past->cumulative_aggressive_buy
                ? current.cumulative_aggressive_buy - past->cumulative_aggressive_buy : 0ULL;
            const std::uint64_t recent_sell = current.cumulative_aggressive_sell >= past->cumulative_aggressive_sell
                ? current.cumulative_aggressive_sell - past->cumulative_aggressive_sell : 0ULL;
            const std::uint64_t recent_total = recent_buy + recent_sell;

            int flow_signal = 0;
            if (recent_total > 0) {
                const double imbalance = (static_cast<double>(recent_buy) - static_cast<double>(recent_sell))
                    / static_cast<double>(recent_total);
                flow_signal = sign_from_threshold(imbalance, config_.order_flow_imbalance_threshold);
            }

            int depth_signal = 0;
            const int total_depth = current.best_bid_depth + current.best_ask_depth;
            if (total_depth > 0) {
                const double imbalance = (static_cast<double>(current.best_bid_depth) - current.best_ask_depth)
                    / static_cast<double>(total_depth);
                depth_signal = sign_from_threshold(imbalance, config_.depth_imbalance_threshold);
                if (depth_signal == 0) {
                    depth_signal = sign_from_threshold(imbalance, config_.strong_depth_imbalance_threshold);
                }
            }

            int direction = 0;
            if (mid_signal != 0) {
                if (flow_signal == 0 || flow_signal == mid_signal) direction = mid_signal;
            } else if (flow_signal != 0) {
                direction = flow_signal;
            } else {
                direction = depth_signal;
            }

            if (direction != 0) {
                int quantity = std::max(1, static_cast<int>(std::llround(
                    config_.order_quantity * (0.75 + 0.50 * rng_.uniform01()))));
                if (mid_signal == 0 && flow_signal == 0) quantity = std::max(1, quantity / 2);
                builder.emit(out, AgentKind::Momentum, owner_id_, OrderAction::Market,
                             direction > 0 ? Side::Buy : Side::Sell,
                             quantity, 0, 0, decision_time, rng_);
            }
        }
        next_wake_ns_ = safe_add_time(next_wake_ns_, rng_.exponential_wait_ns(config_.wake_rate_per_second));
    }
}

void MomentumAgent::apply_report(const AgentReport& report) {
    if (report.owner_id != owner_id_) return;
    update_cash_inventory(report, inventory_, cash_ticks_);
}

} // namespace dlob
