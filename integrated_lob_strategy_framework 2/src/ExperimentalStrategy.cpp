#include "ExperimentalStrategy.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace {
int round_to_tick(double price_ticks, int tick_size) {
    const int tick = std::max(1, tick_size);
    return static_cast<int>(std::llround(price_ticks / static_cast<double>(tick))) * tick;
}

StrategyQuote passive_quote_from_center(
    const MarketState& state,
    double center_ticks,
    double spread_ticks,
    int quantity
) {
    StrategyQuote q;
    if (quantity <= 0 || !std::isfinite(center_ticks) || !std::isfinite(spread_ticks)) return q;
    if (state.best_bid_ticks <= 0 || state.best_ask_ticks <= 0 || state.best_ask_ticks <= state.best_bid_ticks) return q;

    const int tick = std::max(1, state.tick_size);
    const double half = 0.5 * std::max(static_cast<double>(tick), spread_ticks);
    int bid = round_to_tick(center_ticks - half, tick);
    int ask = round_to_tick(center_ticks + half, tick);

    // Keep the strategy passive. Crossing orders would make this a taker strategy rather than a market maker.
    bid = std::min(bid, state.best_ask_ticks - tick);
    ask = std::max(ask, state.best_bid_ticks + tick);
    if (bid >= ask) {
        bid = state.best_bid_ticks;
        ask = state.best_ask_ticks;
    }
    if (bid <= 0 || ask <= 0 || bid >= ask) return q;

    q.should_quote = true;
    q.bid_price_ticks = bid;
    q.ask_price_ticks = ask;
    q.bid_volume = quantity;
    q.ask_volume = quantity;
    return q;
}
}

ExperimentalMarketMaker::ExperimentalMarketMaker(
    int strategy_id,
    double hyperparameter,
    int owner_id,
    int base_order_size,
    int base_spread_ticks
) : strategy_id_(strategy_id),
    hyperparameter_(hyperparameter),
    owner_id_(owner_id),
    base_order_size_(std::max(1, base_order_size)),
    base_spread_ticks_(std::max(1, base_spread_ticks)) {}

void ExperimentalMarketMaker::process_execution_reports(const std::vector<ExecutionReport>& reports) {
    for (const ExecutionReport& r : reports) {
        if (r.quantity <= 0 || r.price_ticks <= 0) continue;
        if (r.side == Side::Buy) {
            // Our resting bid was filled: we bought inventory and paid cash.
            inventory_ += r.quantity;
            cash_ticks_ -= static_cast<double>(r.price_ticks) * static_cast<double>(r.quantity);
        } else {
            // Our resting ask was filled: we sold inventory and received cash.
            inventory_ -= r.quantity;
            cash_ticks_ += static_cast<double>(r.price_ticks) * static_cast<double>(r.quantity);
        }
        total_traded_volume_ += static_cast<double>(r.quantity);
        gross_notional_ticks_ += static_cast<double>(r.price_ticks) * static_cast<double>(r.quantity);
    }
}

double ExperimentalMarketMaker::mark_to_market(double mid_price_ticks) const {
    return cash_ticks_ + static_cast<double>(inventory_) * mid_price_ticks;
}

InventoryLinearSkewStrategy::InventoryLinearSkewStrategy(double gamma, int owner_id, int base_order_size, int base_spread_ticks)
    : ExperimentalMarketMaker(1, gamma, owner_id, base_order_size, base_spread_ticks) {}

StrategyQuote InventoryLinearSkewStrategy::get_quote(const MarketState& state) {
    const double gamma = hyperparameter_;
    const double skew = -gamma * static_cast<double>(inventory_) * static_cast<double>(state.tick_size);
    const double center = state.mid_price_ticks + skew;
    const double spread = static_cast<double>(base_spread_ticks_ * state.tick_size);
    return passive_quote_from_center(state, center, spread, base_order_size_);
}

QueueAwareFillHazardStrategy::QueueAwareFillHazardStrategy(double eta, int owner_id, int base_order_size, int base_spread_ticks)
    : ExperimentalMarketMaker(2, eta, owner_id, base_order_size, base_spread_ticks) {}

StrategyQuote QueueAwareFillHazardStrategy::get_quote(const MarketState& state) {
    const double bid_hazard = 1.0 / (static_cast<double>(std::max(0, state.bid_queue_depth)) + 1.0);
    const double ask_hazard = 1.0 / (static_cast<double>(std::max(0, state.ask_queue_depth)) + 1.0);
    const double fill_hazard = 0.5 * (bid_hazard + ask_hazard);
    const double eta = hyperparameter_;
    const double base = static_cast<double>(base_spread_ticks_ * state.tick_size);
    const double spread = base * (1.0 + eta * (1.0 - fill_hazard));
    return passive_quote_from_center(state, state.mid_price_ticks, spread, base_order_size_);
}

HawkesSignalHazardStrategy::HawkesSignalHazardStrategy(double theta, int owner_id, int base_order_size, int base_spread_ticks)
    : ExperimentalMarketMaker(3, theta, owner_id, base_order_size, base_spread_ticks) {}

StrategyQuote HawkesSignalHazardStrategy::get_quote(const MarketState& state) {
    const double theta = hyperparameter_;
    const double avg = std::max(1e-9, state.hawkes_average_intensity);
    const double ratio = std::max(0.0, state.hawkes_total_intensity) / avg;
    const double base = static_cast<double>(base_spread_ticks_ * state.tick_size);
    const double spread = base * (1.0 + theta * ratio);
    return passive_quote_from_center(state, state.mid_price_ticks, spread, base_order_size_);
}

std::unique_ptr<ExperimentalMarketMaker> make_strategy(
    int strategy_id,
    double hyperparameter,
    int owner_id,
    int base_order_size,
    int base_spread_ticks
) {
    switch (strategy_id) {
        case 1:
            return std::make_unique<InventoryLinearSkewStrategy>(hyperparameter, owner_id, base_order_size, base_spread_ticks);
        case 2:
            return std::make_unique<QueueAwareFillHazardStrategy>(hyperparameter, owner_id, base_order_size, base_spread_ticks);
        case 3:
            return std::make_unique<HawkesSignalHazardStrategy>(hyperparameter, owner_id, base_order_size, base_spread_ticks);
        default:
            throw std::invalid_argument("Unknown strategy_id. Use 1, 2 or 3.");
    }
}

std::vector<double> hyperparameter_grid(int strategy_id) {
    switch (strategy_id) {
        case 1: return {0.001, 0.005, 0.010, 0.025, 0.050, 0.100};
        case 2: return {0.50, 1.00, 2.00, 3.50, 5.00};
        case 3: return {0.10, 0.25, 0.50, 1.00, 1.50, 2.00};
        default: throw std::invalid_argument("Unknown strategy_id. Use 1, 2 or 3.");
    }
}

std::string strategy_name(int strategy_id) {
    switch (strategy_id) {
        case 1: return "inventory_linear_skew";
        case 2: return "queue_aware_fill_hazard";
        case 3: return "hawkes_signal_hazard";
        default: return "unknown";
    }
}
