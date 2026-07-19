#include "agents/InformedTraderAgent.hpp"
#include <algorithm>
#include <cmath>
namespace dlob {
InformedTraderAgent::InformedTraderAgent(int owner_id,const InformedTraderConfig& config,std::int64_t simulation_start_ns,std::uint64_t seed): owner_id_(owner_id), config_(config), rng_(seed) {
    next_wake_ns_ = safe_add_time(simulation_start_ns, rng_.exponential_wait_ns(config_.wake_rate_per_second));
}

void InformedTraderAgent::generate_orders(const MarketState& current,std::int64_t window_start_ns,std::int64_t window_end_ns,OrderMessageBuilder& builder,std::vector<OrderMessage>& out) {
    if (current.best_bid_ticks <= 0 || current.best_ask_ticks <= 0) return;
    while (next_wake_ns_ < window_end_ns) {
        const std::int64_t decision_time = std::max(next_wake_ns_, window_start_ns);
        const double private_signal = current.fundamental_value_ticks + config_.signal_noise_ticks * config_.tick_size * rng_.normal01();
        const double mispricing_ticks = (private_signal - current.mid_price_ticks)/ static_cast<double>(std::max(1, config_.tick_size));
        if (std::abs(mispricing_ticks) > config_.trade_threshold_ticks) {
            const Side side = mispricing_ticks > 0.0 ? Side::Buy : Side::Sell;
            const bool risk_allowed = side == Side::Buy
                ? inventory_ < config_.max_abs_inventory
                : inventory_ > -config_.max_abs_inventory;
            if (risk_allowed) {
                const double strength = std::min(3.0,
                    std::abs(mispricing_ticks) / std::max(1e-9, config_.trade_threshold_ticks));
                const int quantity = std::max(1, static_cast<int>(std::llround(
                    config_.base_quantity * (0.50 + 0.50 * strength) * (0.75 + 0.50 * rng_.uniform01()))));
                builder.emit(out, AgentKind::Informed, owner_id_, OrderAction::Market,side, quantity, 0, 0, decision_time, rng_);
            }
        }
        next_wake_ns_ = safe_add_time(next_wake_ns_, rng_.exponential_wait_ns(config_.wake_rate_per_second));
    }
}

void InformedTraderAgent::apply_report(const AgentReport& report) {
    if (report.owner_id != owner_id_) return;
    update_cash_inventory(report, inventory_, cash_ticks_);
}

}
