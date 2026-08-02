// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#include "agents/EtfArbitrageAgent.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace dlob {
namespace {

std::int64_t checked_arrival(std::int64_t time_ns, std::int64_t latency_ns) {
    if (latency_ns <= 0
        || time_ns > std::numeric_limits<std::int64_t>::max() - latency_ns) {
        throw std::invalid_argument("invalid ETF-arbitrage order latency");
    }
    return time_ns + latency_ns;
}

double usable_mid(const MarketState& state) {
    if (state.best_bid_ticks <= 0 || state.best_ask_ticks <= state.best_bid_ticks
        || !std::isfinite(state.mid_price_ticks) || state.mid_price_ticks <= 0.0) {
        throw std::logic_error("ETF arbitrage requires a valid two-sided state");
    }
    return state.mid_price_ticks;
}

void checked_accumulate(std::int64_t& target, std::int64_t change) {
    if ((change > 0 && target > std::numeric_limits<std::int64_t>::max() - change)
        || (change < 0
            && target < std::numeric_limits<std::int64_t>::min() - change)) {
        throw std::overflow_error("ETF-arbitrage accounting overflows int64");
    }
    target += change;
}

} // namespace

EtfArbitrageAgent::EtfArbitrageAgent(
    EtfArbitrageConfig config,
    std::vector<MultiAssetBookConfig> books)
    : config_(config), books_(std::move(books)),
      normalized_weights_(books_.size(), 0.0),
      inventory_by_book_(books_.size(), 0),
      cash_by_book_(books_.size(), 0) {
    if (!config_.enabled) return;
    if (books_.size() < 2U
        || config_.etf_book_id >= static_cast<BookId>(books_.size())
        || !std::isfinite(config_.trigger_bps) || config_.trigger_bps <= 0.0
        || !std::isfinite(config_.release_bps) || config_.release_bps < 0.0
        || config_.release_bps >= config_.trigger_bps
        || config_.etf_order_quantity <= 0
        || config_.max_component_quantity <= 0
        || config_.decision_interval_ns <= 0
        || config_.order_latency_ns <= 0) {
        throw std::invalid_argument("invalid ETF-arbitrage configuration");
    }
    double total_weight = 0.0;
    for (std::size_t index = 0; index < books_.size(); ++index) {
        if (index == static_cast<std::size_t>(config_.etf_book_id)) continue;
        total_weight += books_[index].basket_weight;
    }
    if (!std::isfinite(total_weight) || total_weight <= 0.0) {
        throw std::invalid_argument("ETF arbitrage needs positive component weights");
    }
    for (std::size_t index = 0; index < books_.size(); ++index) {
        if (index != static_cast<std::size_t>(config_.etf_book_id)) {
            normalized_weights_[index] = books_[index].basket_weight / total_weight;
        }
    }
}

std::vector<OrderMessage> EtfArbitrageAgent::make_orders(
    const std::vector<MarketState>& states,
    std::int64_t decision_time_ns,
    std::uint64_t decision_sequence) {
    if (!config_.enabled) return {};
    if (states.size() != books_.size() || decision_sequence == 0) {
        throw std::invalid_argument("ETF-arbitrage state vector does not match books");
    }
    const std::size_t etf_index = static_cast<std::size_t>(config_.etf_book_id);
    const double etf_reference = books_[etf_index].fundamental_price_ticks;
    const double etf_normalized = usable_mid(states[etf_index]) / etf_reference;
    double basket_normalized = 0.0;
    for (std::size_t index = 0; index < books_.size(); ++index) {
        if (index == etf_index) continue;
        basket_normalized += normalized_weights_[index]
            * usable_mid(states[index]) / books_[index].fundamental_price_ticks;
    }
    last_deviation_bps_ = 10'000.0 * (etf_normalized - basket_normalized);

    int signal = 0;
    if (last_deviation_bps_ > config_.trigger_bps) signal = 1;
    if (last_deviation_bps_ < -config_.trigger_bps) signal = -1;
    if (signal == 0) {
        if (std::abs(last_deviation_bps_) <= config_.release_bps) last_signal_ = 0;
        return {};
    }
    if (signal == last_signal_) return {};
    last_signal_ = signal;

    const std::int64_t arrival_time = checked_arrival(
        decision_time_ns, config_.order_latency_ns);
    std::vector<OrderMessage> orders;
    orders.reserve(books_.size());
    auto make_order = [&](BookId book_id, Side side, std::int32_t quantity,
                          std::uint32_t child) {
        OrderMessage order;
        order.generated_time_ns = decision_time_ns;
        order.arrival_time_ns = arrival_time;
        order.sequence = stable_sequence(etf_arbitrage_entity,
                                         decision_sequence, child);
        order.tie_breaker = stable_sequence(etf_arbitrage_entity,
                                            decision_sequence, child + 10'000U);
        order.source_rank = 0;
        order.owner_id = etf_arbitrage_owner_id;
        order.agent_kind = AgentKind::Arbitrage;
        order.action = OrderAction::Market;
        order.side = side;
        order.quantity = quantity;
        order.book_id = book_id;
        orders.push_back(order);
    };

    // Positive signal means the ETF is expensive relative to the proxy basket:
    // sell ETF and buy the components.  Negative signal reverses every leg.
    make_order(config_.etf_book_id,
               signal > 0 ? Side::Sell : Side::Buy,
               config_.etf_order_quantity, 0);
    const double etf_notional = static_cast<double>(config_.etf_order_quantity)
        * etf_reference;
    std::uint32_t child = 1;
    for (std::size_t index = 0; index < books_.size(); ++index) {
        if (index == etf_index) continue;
        const double ideal = etf_notional * normalized_weights_[index]
            / books_[index].fundamental_price_ticks;
        const auto rounded = static_cast<std::int64_t>(std::llround(ideal));
        const std::int64_t bounded = std::clamp<std::int64_t>(
            rounded, 1, config_.max_component_quantity);
        make_order(static_cast<BookId>(index),
                   signal > 0 ? Side::Buy : Side::Sell,
                   static_cast<std::int32_t>(bounded), child++);
    }
    return orders;
}

void EtfArbitrageAgent::on_trade(const TradeExecution& trade) {
    if (!config_.enabled) return;
    const bool buyer = trade.buyer_owner_id == etf_arbitrage_owner_id;
    const bool seller = trade.seller_owner_id == etf_arbitrage_owner_id;
    if (!buyer && !seller) return;
    if (trade.book_id >= books_.size()
        || trade.quantity <= 0 || trade.price_ticks <= 0) {
        throw std::invalid_argument("invalid ETF-arbitrage trade accounting input");
    }
    const std::int64_t quantity = trade.quantity;
    const std::int64_t notional = quantity
        * static_cast<std::int64_t>(trade.price_ticks);
    const std::size_t index = static_cast<std::size_t>(trade.book_id);
    if (buyer) {
        checked_accumulate(inventory_by_book_[index], quantity);
        checked_accumulate(cash_by_book_[index], -notional);
    }
    if (seller) {
        checked_accumulate(inventory_by_book_[index], -quantity);
        checked_accumulate(cash_by_book_[index], notional);
    }
}

std::int64_t EtfArbitrageAgent::inventory(BookId book_id) const {
    if (book_id >= inventory_by_book_.size()) {
        throw std::out_of_range("ETF-arbitrage inventory book is not configured");
    }
    return inventory_by_book_[static_cast<std::size_t>(book_id)];
}

std::int64_t EtfArbitrageAgent::cash_ticks(BookId book_id) const {
    if (book_id >= cash_by_book_.size()) {
        throw std::out_of_range("ETF-arbitrage cash book is not configured");
    }
    return cash_by_book_[static_cast<std::size_t>(book_id)];
}

std::int64_t EtfArbitrageAgent::total_cash_ticks() const {
    std::int64_t total = 0;
    for (std::int64_t cash : cash_by_book_) checked_accumulate(total, cash);
    return total;
}

} // namespace dlob
