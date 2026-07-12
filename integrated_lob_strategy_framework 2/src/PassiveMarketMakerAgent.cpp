#include "PassiveMarketMakerAgent.hpp"

#include "EmpiricalDistribution.hpp"
#include "Order.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

int PassiveMarketMakerAgent::number_of_agents_ = 0;

namespace {
constexpr int MARKET_MAKER_OWNER_ID_BASE = 1000000;
constexpr int MAX_INVENTORY_SKEW_TICKS = 3;
constexpr int EMPIRICAL_REFERENCE_QUANTITY = 100;

int owner_id_for_agent(int agent_index) { return MARKET_MAKER_OWNER_ID_BASE + agent_index; }

int round_to_tick(double price, int tick_size) {
    return static_cast<int>(std::llround(price / static_cast<double>(tick_size))) * tick_size;
}
int floor_to_tick(double price, int tick_size) {
    return static_cast<int>(std::floor(price / static_cast<double>(tick_size))) * tick_size;
}
int ceil_to_tick(double price, int tick_size) {
    return static_cast<int>(std::ceil(price / static_cast<double>(tick_size))) * tick_size;
}

struct MMDistributions {
    EmpiricalDistribution limit_buy_quantity;
    EmpiricalDistribution limit_sell_quantity;
    MMDistributions() {
        limit_buy_quantity.set_fallback(25, 200);
        limit_sell_quantity.set_fallback(25, 200);
        limit_buy_quantity.load_from_csv("limit_buy_quantity_distribution.txt", "quantity");
        limit_sell_quantity.load_from_csv("limit_sell_quantity_distribution.txt", "quantity");
    }
};

MMDistributions& mm_distributions() {
    static MMDistributions d;
    return d;
}

void update_inventory_from_reports(LimitOrderBook& book, int owner_id, int& inventory, double& cash) {
    std::vector<ExecutionReport> reports = book.get_and_clear_execution_reports(owner_id);
    for (const ExecutionReport& report : reports) {
        const double notional = static_cast<double>(report.quantity) * static_cast<double>(report.price_ticks);
        if (report.side == Side::Buy) {
            inventory += report.quantity;
            cash -= notional;
        } else {
            inventory -= report.quantity;
            cash += notional;
        }
    }
}

int inventory_skew_ticks(int inventory, int base_quantity) {
    if (base_quantity <= 0) return 0;
    const double units = static_cast<double>(inventory) / static_cast<double>(base_quantity);
    const int skew = static_cast<int>(std::llround(0.25 * units));
    return std::clamp(skew, -MAX_INVENTORY_SKEW_TICKS, MAX_INVENTORY_SKEW_TICKS);
}

int sample_empirical_quantity(Side side, int base_quantity, int inventory, std::mt19937_64& rng) {
    auto& d = mm_distributions();
    const int empirical = side == Side::Buy ? d.limit_buy_quantity.sample(rng) : d.limit_sell_quantity.sample(rng);
    const double scale = static_cast<double>(std::max(1, base_quantity)) / static_cast<double>(EMPIRICAL_REFERENCE_QUANTITY);
    int quantity = std::max(1, static_cast<int>(std::llround(scale * empirical)));
    quantity = std::min(quantity, std::max(1, base_quantity * 4));

    if (inventory != 0) {
        const int units = std::min(std::abs(inventory) / std::max(1, base_quantity), 4);
        if (inventory > 0) {
            if (side == Side::Buy) quantity = std::max(1, quantity / (1 + units));
            else quantity = std::min(quantity + units * base_quantity / 2, base_quantity * 2);
        } else {
            if (side == Side::Buy) quantity = std::min(quantity + units * base_quantity / 2, base_quantity * 2);
            else quantity = std::max(1, quantity / (1 + units));
        }
    }
    return std::max(1, quantity);
}
}

PassiveMarketMakerAgent::PassiveMarketMakerAgent(
    int tick_size,
    int num_levels,
    int order_quantity,
    int level_spacing_ticks,
    int min_spread_price,
    double quote_skip_probability,
    double recovery_quote_skip_probability,
    std::uint64_t seed
)
    : agent_index_(number_of_agents_++),
      tick_size_(std::max(1, tick_size)),
      num_levels_(std::max(1, num_levels)),
      order_quantity_(std::max(1, order_quantity)),
      level_spacing_ticks_(std::max(1, level_spacing_ticks)),
      min_spread_price_(std::max(tick_size_, min_spread_price)),
      quote_skip_probability_(std::clamp(quote_skip_probability, 0.0, 0.95)),
      recovery_quote_skip_probability_(std::clamp(recovery_quote_skip_probability, 0.0, 0.95)),
      rng_(seed + static_cast<std::uint64_t>(agent_index_) * 1009ULL) {}

std::size_t PassiveMarketMakerAgent::wake_up(LimitOrderBook& book, std::uint64_t& next_order_id, std::int64_t current_time_ns) {
    const int owner_id = owner_id_for_agent(agent_index_);
    update_inventory_from_reports(book, owner_id, inventory_, cash_);

    const bool has_bid = book.has_bid();
    const bool has_ask = book.has_ask();
    if (!has_bid && !has_ask) return 0;

    const double skip = (has_bid && has_ask) ? quote_skip_probability_ : recovery_quote_skip_probability_;
    if (skip > 0.0) {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        if (u(rng_) < skip) return 0;
    }

    const int half_spread_ticks = std::max(1, static_cast<int>(std::ceil(static_cast<double>(min_spread_price_) / (2.0 * tick_size_))));
    const int skew_ticks = inventory_skew_ticks(inventory_, order_quantity_);

    int primary_bid_price = 0;
    int primary_ask_price = 0;

    if (has_bid && has_ask) {
        const int best_bid = book.best_bid();
        const int best_ask = book.best_ask();
        const int current_spread = best_ask - best_bid;
        if (current_spread <= 0) return 0;
        const double mid = 0.5 * static_cast<double>(best_bid + best_ask);
        const int reservation = round_to_tick(mid - static_cast<double>(skew_ticks * tick_size_), tick_size_);

        primary_bid_price = best_bid;
        primary_ask_price = best_ask;
        if (current_spread > min_spread_price_) {
            primary_bid_price = std::max(reservation - half_spread_ticks * tick_size_, best_bid + tick_size_);
            primary_ask_price = std::min(reservation + half_spread_ticks * tick_size_, best_ask - tick_size_);
        }
        primary_bid_price = std::min(primary_bid_price, best_ask - tick_size_);
        primary_ask_price = std::max(primary_ask_price, best_bid + tick_size_);
    } else if (has_bid) {
        primary_bid_price = book.best_bid();
        primary_ask_price = book.best_bid() + min_spread_price_;
    } else {
        primary_bid_price = book.best_ask() - min_spread_price_;
        primary_ask_price = book.best_ask();
    }

    primary_bid_price = floor_to_tick(primary_bid_price, tick_size_);
    primary_ask_price = ceil_to_tick(primary_ask_price, tick_size_);
    if (primary_bid_price <= 0 || primary_bid_price >= primary_ask_price) return 0;

    // Academic design choice: process fills first, verify replacement prices, then cancel stale quotes.
    const std::size_t cancelled = book.cancel_orders_by_owner(owner_id);

    const int spacing = std::max(1, level_spacing_ticks_) * tick_size_;
    for (int level = 0; level < num_levels_; ++level) {
        const int bid_price = primary_bid_price - level * spacing;
        const int ask_price = primary_ask_price + level * spacing;
        if (bid_price <= 0 || bid_price >= ask_price) continue;

        book.add_limit_order(Order{next_order_id++, current_time_ns, OrderType::Limit, Side::Buy,
                                   sample_empirical_quantity(Side::Buy, order_quantity_, inventory_, rng_), bid_price, owner_id});
        book.add_limit_order(Order{next_order_id++, current_time_ns, OrderType::Limit, Side::Sell,
                                   sample_empirical_quantity(Side::Sell, order_quantity_, inventory_, rng_), ask_price, owner_id});
    }
    return cancelled;
}
