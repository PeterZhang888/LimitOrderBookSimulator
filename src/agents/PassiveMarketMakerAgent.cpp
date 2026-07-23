#include "agents/PassiveMarketMakerAgent.hpp"

#include "common/EmpiricalDistribution.hpp"
#include "common/DataPaths.hpp"

#include <algorithm>
#include <cmath>

namespace dlob {
namespace {
constexpr int max_inventory_skew_ticks = 3;
constexpr int empirical_reference_quantity = 100;

struct MarketMakerDistributions {
    EmpiricalDistribution limit_buy_quantity;
    EmpiricalDistribution limit_sell_quantity;

    MarketMakerDistributions() {
        limit_buy_quantity.set_fallback(25, 200);
        limit_sell_quantity.set_fallback(25, 200);
        limit_buy_quantity.load_from_csv(resolve_data_file("limit_buy_quantity_distribution.txt"), "quantity");
        limit_sell_quantity.load_from_csv(resolve_data_file("limit_sell_quantity_distribution.txt"), "quantity");
    }
};

MarketMakerDistributions& distributions() {
    static MarketMakerDistributions value;
    return value;
}

int floor_to_tick(int price, int tick_size) {
    return (price / tick_size) * tick_size;
}

int ceil_to_tick(int price, int tick_size) {
    return ((price + tick_size - 1) / tick_size) * tick_size;
}
} // namespace

PassiveMarketMakerAgent::PassiveMarketMakerAgent(int owner_id,
                                                 const PassiveMarketMakerConfig& config,
                                                 std::int64_t simulation_start_ns,
                                                 std::uint64_t seed)
    : owner_id_(owner_id), config_(config), next_wake_ns_(simulation_start_ns), rng_(seed) {}

int PassiveMarketMakerAgent::inventory_skew_ticks() const {
    const double units = static_cast<double>(inventory_) / static_cast<double>(std::max(1, config_.order_quantity));
    return std::clamp(static_cast<int>(std::llround(0.25 * units)),
                      -max_inventory_skew_ticks, max_inventory_skew_ticks);
}

int PassiveMarketMakerAgent::sample_quantity(Side side) {
    auto& d = distributions();
    const int empirical = side == Side::Buy
        ? d.limit_buy_quantity.sample(rng_)
        : d.limit_sell_quantity.sample(rng_);
    const double scale = static_cast<double>(std::max(1, config_.order_quantity))
        / static_cast<double>(empirical_reference_quantity);
    int quantity = std::max(1, static_cast<int>(std::llround(scale * empirical)));
    quantity = std::min(quantity, std::max(1, 4 * config_.order_quantity));

    if (inventory_ > 0) {
        if (side == Side::Buy) quantity = std::max(1, quantity / 2);
        else quantity = std::min(2 * config_.order_quantity, quantity + inventory_ / 4);
    } else if (inventory_ < 0) {
        if (side == Side::Sell) quantity = std::max(1, quantity / 2);
        else quantity = std::min(2 * config_.order_quantity, quantity + std::abs(inventory_) / 4);
    }
    return std::max(1, quantity);
}

void PassiveMarketMakerAgent::generate_orders(const MarketState& current,
                                               std::int64_t window_start_ns,
                                               std::int64_t window_end_ns,
                                               OrderMessageBuilder& builder,
                                               std::vector<OrderMessage>& out) {
    const bool any_quote = current.best_bid_ticks > 0 || current.best_ask_ticks > 0;

    while (next_wake_ns_ < window_end_ns) {
        const std::int64_t decision_time = std::max(next_wake_ns_, window_start_ns);
        const bool complete_book = current.best_bid_ticks > 0 && current.best_ask_ticks > 0;
        if (!any_quote) {
            next_wake_ns_ = safe_add_time(next_wake_ns_, std::max<std::int64_t>(1, config_.quote_interval_ns));
            continue;
        }
        const double skip_probability = complete_book
            ? config_.quote_skip_probability
            : config_.recovery_quote_skip_probability;

        if (rng_.uniform01() >= std::clamp(skip_probability, 0.0, 0.95)) {
            const std::int64_t common_latency = builder.sample_latency_ns(AgentKind::MarketMaker, rng_);
            builder.emit(out, AgentKind::MarketMaker, owner_id_, OrderAction::CancelOwner,
                         Side::Buy, 0, 0, 0, decision_time, rng_, common_latency);

            const int tick = std::max(1, config_.tick_size);
            const int minimum_spread = std::max(2 * tick, config_.min_spread_ticks * tick);
            const int skew = inventory_skew_ticks();
            int primary_bid = 0;
            int primary_ask = 0;

            if (complete_book) {
                const int spread = current.best_ask_ticks - current.best_bid_ticks;
                const int reservation = static_cast<int>(std::llround(current.mid_price_ticks)) - skew * tick;
                primary_bid = current.best_bid_ticks;
                primary_ask = current.best_ask_ticks;
                if (spread > minimum_spread) {
                    const int half = std::max(tick, minimum_spread / 2);
                    primary_bid = std::max(current.best_bid_ticks + tick, reservation - half);
                    primary_ask = std::min(current.best_ask_ticks - tick, reservation + half);
                }
                primary_bid = std::min(primary_bid, current.best_ask_ticks - tick);
                primary_ask = std::max(primary_ask, current.best_bid_ticks + tick);
            } else if (current.best_bid_ticks > 0) {
                primary_bid = current.best_bid_ticks;
                primary_ask = current.best_bid_ticks + minimum_spread;
            } else {
                primary_bid = current.best_ask_ticks - minimum_spread;
                primary_ask = current.best_ask_ticks;
            }

            primary_bid = floor_to_tick(primary_bid, tick);
            primary_ask = ceil_to_tick(primary_ask, tick);
            const int spacing = std::max(1, config_.level_spacing_ticks) * tick;
            if (primary_bid > 0 && primary_ask > primary_bid) {
                for (int level = 0; level < std::max(1, config_.num_levels); ++level) {
                    const int bid = primary_bid - level * spacing;
                    const int ask = primary_ask + level * spacing;
                    if (bid <= 0 || ask <= bid) continue;
                    builder.emit(out, AgentKind::MarketMaker, owner_id_, OrderAction::Limit,
                                 Side::Buy, sample_quantity(Side::Buy), bid, 0,
                                 decision_time, rng_, common_latency);
                    builder.emit(out, AgentKind::MarketMaker, owner_id_, OrderAction::Limit,
                                 Side::Sell, sample_quantity(Side::Sell), ask, 0,
                                 decision_time, rng_, common_latency);
                }
            }
        }
        next_wake_ns_ = safe_add_time(next_wake_ns_, std::max<std::int64_t>(1, config_.quote_interval_ns));
    }
}

void PassiveMarketMakerAgent::apply_report(const AgentReport& report) {
    if (report.owner_id != owner_id_) return;
    update_cash_inventory(report, inventory_, cash_ticks_);
}

} // namespace dlob
