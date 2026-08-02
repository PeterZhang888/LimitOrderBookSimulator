#include "agents/LargeInstitutionalAgent.hpp"

#include "common/EmpiricalDistribution.hpp"
#include "common/DataPaths.hpp"

#include <algorithm>
#include <cmath>

namespace dlob {
namespace {
struct InstitutionalDistributions {
    EmpiricalDistribution market_buy_quantity;
    EmpiricalDistribution market_sell_quantity;

    InstitutionalDistributions() {
        market_buy_quantity.set_fallback(50, 1'000);
        market_sell_quantity.set_fallback(50, 1'000);
        market_buy_quantity.load_from_csv(resolve_data_file("market_buy_quantity_distribution.txt"), "quantity");
        market_sell_quantity.load_from_csv(resolve_data_file("market_sell_quantity_distribution.txt"), "quantity");
    }
};

InstitutionalDistributions& distributions() {
    static InstitutionalDistributions value;
    return value;
}

int sample_child_quantity(Side side, FastRng& rng) {
    auto& d = distributions();
    return side == Side::Buy ? d.market_buy_quantity.sample(rng) : d.market_sell_quantity.sample(rng);
}
} // namespace

LargeInstitutionalAgent::LargeInstitutionalAgent(int owner_id,
                                                 const LargeInstitutionalConfig& config,
                                                 std::uint64_t seed)
    : owner_id_(owner_id), config_(config), rng_(seed) {
    next_wake_ns_ = safe_add_time(config_.start_time_ns, rng_.exponential_wait_ns(config_.wake_rate_per_second));
}

int LargeInstitutionalAgent::remaining_quantity() const {
    return std::max(0, config_.parent_quantity - executed_quantity_ - outstanding_quantity_);
}

void LargeInstitutionalAgent::generate_orders(const MarketState& current,
                                              std::int64_t window_start_ns,
                                              std::int64_t window_end_ns,
                                              OrderMessageBuilder& builder,
                                              std::vector<OrderMessage>& out) {
    while (next_wake_ns_ < window_end_ns && remaining_quantity() > 0) {
        const std::int64_t decision_time = std::max(next_wake_ns_, window_start_ns);
        if (decision_time >= config_.start_time_ns && decision_time <= config_.end_time_ns) {
            const int available = config_.side == Side::Buy ? current.best_ask_depth : current.best_bid_depth;
            const int participation_limit = available > 0
                ? std::max(1, static_cast<int>(std::floor(config_.participation_cap * available)))
                : 0;
            if (participation_limit > 0) {
                const int sampled = sample_child_quantity(config_.side, rng_);
                const int quantity = std::min({config_.child_quantity, sampled,
                                               participation_limit, remaining_quantity()});
                if (quantity > 0) {
                    builder.emit(out, AgentKind::Institutional, owner_id_, OrderAction::Market,
                                 config_.side, quantity, 0, 0, decision_time, rng_);
                    outstanding_quantity_ += quantity;
                }
            }
        }
        next_wake_ns_ = safe_add_time(next_wake_ns_, rng_.exponential_wait_ns(config_.wake_rate_per_second));
    }
}

void LargeInstitutionalAgent::apply_report(const AgentReport& report) {
    if (report.owner_id != owner_id_) return;
    if (report.kind == ReportKind::OrderResult && report.action == OrderAction::Market) {
        outstanding_quantity_ = std::max(0, outstanding_quantity_ - report.requested_quantity);
    } else if (report.kind == ReportKind::Fill) {
        executed_quantity_ = std::min(config_.parent_quantity, executed_quantity_ + report.fill_quantity);
        update_cash_inventory(report, inventory_, cash_ticks_);
    }
}

} // namespace dlob
