#include "agents/SharedMarketMakerAgent.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace dlob {
namespace {

std::uint64_t mix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::int64_t checked_add_time(std::int64_t time_ns, std::int64_t latency_ns) {
    if (latency_ns <= 0) {
        throw std::invalid_argument("shared market-maker latency must be positive");
    }
    if (time_ns > std::numeric_limits<std::int64_t>::max() - latency_ns) {
        throw std::overflow_error("shared market-maker event time overflow");
    }
    return time_ns + latency_ns;
}

} // namespace

SharedMarketMakerAgent::SharedMarketMakerAgent(SharedMarketMakerConfig config)
    : config_(std::move(config)) {
    if (config_.logical_owner_id <= 0) {
        throw std::invalid_argument("shared market maker requires a positive logical owner id");
    }
    if (config_.quote_quantity <= 0 || config_.quote_levels <= 0
        || config_.quote_levels > 10'000
        || config_.quote_quantity_growth <= 0
        || config_.quote_quantity_growth > 10
        || config_.max_quote_quantity_per_level < config_.quote_quantity
        || config_.quote_half_spread_ticks <= 0
        || config_.price_tick_size <= 0 || config_.hedge_lot_size <= 0
        || config_.max_hedge_quantity <= 0) {
        throw std::invalid_argument("shared market-maker sizes and tick parameters must be positive");
    }
    if (config_.order_latency_ns <= 0 || config_.report_latency_ns <= 0
        || config_.reaction_latency_ns <= 0 || config_.network_latency_ns <= 0) {
        throw std::invalid_argument("shared market-maker causal latencies must all be positive");
    }
    if (!std::isfinite(config_.exposure_threshold) || config_.exposure_threshold < 0.0) {
        throw std::invalid_argument("shared market-maker exposure threshold must be finite and non-negative");
    }
    if (config_.books.empty()) {
        throw std::invalid_argument("shared market maker needs at least one configured book");
    }

    for (const auto& book : config_.books) {
        if (!std::isfinite(book.beta) || book.beta == 0.0) {
            throw std::invalid_argument("every shared market-maker book needs a finite, non-zero beta");
        }
        if (book.quote_quantity < 0
            || book.quote_quantity > config_.max_quote_quantity_per_level
            || book.target_spread_ticks <= 0
            || book.target_spread_ticks > 100'000) {
            throw std::invalid_argument("invalid per-book market-maker quote configuration");
        }
        if (config_.books.size() >= 2 && book.hedge_routes.empty()
            && book.hedge_book_id == book.book_id) {
            throw std::invalid_argument("a cross-asset hedge book must differ from its source book");
        }
        const auto [inventory_it, inserted] = inventory_by_book_.emplace(book.book_id, 0);
        (void)inventory_it;
        if (!inserted) {
            throw std::invalid_argument("duplicate shared market-maker book id");
        }
        cash_by_book_.emplace(book.book_id, 0);
    }
    for (const auto& book : config_.books) {
        const std::vector<SharedMarketMakerHedgeRoute> routes =
            book.hedge_routes.empty()
                ? std::vector<SharedMarketMakerHedgeRoute>{{book.hedge_book_id, 1.0}}
                : book.hedge_routes;
        double total_weight = 0.0;
        for (const SharedMarketMakerHedgeRoute& route : routes) {
            if (inventory_by_book_.find(route.book_id) == inventory_by_book_.end()) {
                throw std::invalid_argument(
                    "shared market-maker hedge book is not configured");
            }
            if (config_.books.size() >= 2 && route.book_id == book.book_id) {
                throw std::invalid_argument(
                    "a cross-asset hedge route cannot target its source book");
            }
            if (!std::isfinite(route.weight) || route.weight <= 0.0) {
                throw std::invalid_argument(
                    "shared market-maker hedge weights must be positive");
            }
            total_weight += route.weight;
        }
        if (!std::isfinite(total_weight) || total_weight <= 0.0) {
            throw std::invalid_argument(
                "shared market-maker hedge routes need positive total weight");
        }
    }
}

const SharedMarketMakerBookConfig* SharedMarketMakerAgent::find_book(BookId book_id) const {
    const auto it = std::find_if(config_.books.begin(), config_.books.end(),
                                 [book_id](const auto& book) {
                                     return book.book_id == book_id;
                                 });
    return it == config_.books.end() ? nullptr : &*it;
}

std::uint64_t SharedMarketMakerAgent::next_order_sequence() {
    if (next_local_sequence_ > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("shared market-maker logical order sequence exhausted");
    }
    const auto owner = static_cast<std::uint64_t>(
        static_cast<std::uint32_t>(config_.logical_owner_id));
    return (owner << 32U) | next_local_sequence_++;
}

OrderMessage SharedMarketMakerAgent::make_message(BookId book_id,
                                                   OrderAction action,
                                                   Side side,
                                                   std::int32_t quantity,
                                                   std::int32_t price_ticks,
                                                   std::int64_t generated_time_ns,
                                                   std::int64_t arrival_time_ns) {
    OrderMessage message;
    message.book_id = book_id;
    message.generated_time_ns = generated_time_ns;
    message.arrival_time_ns = arrival_time_ns;
    message.sequence = next_order_sequence();
    message.tie_breaker = mix64(message.sequence
                                ^ (static_cast<std::uint64_t>(book_id) << 17U)
                                ^ 0xd1b54a32d192ed03ULL);
    message.source_rank = config_.message_source_rank;
    message.owner_id = config_.logical_owner_id;
    message.agent_kind = AgentKind::MarketMaker;
    message.action = action;
    message.side = side;
    message.quantity = quantity;
    message.price_ticks = price_ticks;
    return message;
}

std::vector<OrderMessage> SharedMarketMakerAgent::make_quotes(
    BookId book_id,
    const MarketState& state,
    std::int64_t decision_time_ns) {
    const SharedMarketMakerBookConfig* book_config = find_book(book_id);
    if (book_config == nullptr) {
        throw std::invalid_argument("cannot quote an unconfigured book");
    }

    const std::int64_t arrival_time_ns = checked_add_time(
        decision_time_ns, config_.order_latency_ns);
    std::vector<OrderMessage> messages;
    messages.reserve(static_cast<std::size_t>(1)
                     + static_cast<std::size_t>(2)
                         * static_cast<std::size_t>(config_.quote_levels));
    messages.push_back(make_message(book_id, OrderAction::CancelOwner, Side::Buy,
                                    0, 0, decision_time_ns, arrival_time_ns));

    const std::int64_t tick = config_.price_tick_size;
    const std::int64_t target_spread = tick
        * static_cast<std::int64_t>(book_config->target_spread_ticks);
    std::int64_t bid = 0;
    std::int64_t ask = 0;
    if (state.best_bid_ticks > 0 && state.best_ask_ticks > state.best_bid_ticks) {
        const std::int64_t observed_spread
            = static_cast<std::int64_t>(state.best_ask_ticks) - state.best_bid_ticks;
        bid = state.best_bid_ticks;
        ask = state.best_ask_ticks;

        // Collapse a wide gap to this book's empirical target bracket in one
        // decision.  AAPL and AMZN, for example, should not inherit QQQ's
        // one-cent inside spread merely because the same logical maker quotes
        // them.
        // Prefer the last execution as the continuity anchor: using the raw
        // midpoint after one side is depleted would turn a liquidity gap into
        // an artificial price jump.  The bracket is clamped inside the live
        // spread, so both limits remain passive.
        if (observed_spread >= target_spread + 2 * tick) {
            double reference = state.last_trade_price_ticks > 0
                ? static_cast<double>(state.last_trade_price_ticks)
                : state.mid_price_ticks;
            if (!std::isfinite(reference) || reference <= 0.0) {
                reference = (static_cast<double>(state.best_bid_ticks)
                             + static_cast<double>(state.best_ask_ticks)) / 2.0;
            }
            const double preferred = (reference - static_cast<double>(target_spread) / 2.0)
                / static_cast<double>(tick);
            const std::int64_t preferred_bid = static_cast<std::int64_t>(
                std::llround(preferred)) * tick;
            bid = std::clamp<std::int64_t>(
                preferred_bid,
                static_cast<std::int64_t>(state.best_bid_ticks),
                static_cast<std::int64_t>(state.best_ask_ticks) - target_spread);
            ask = bid + target_spread;
        }
    } else if (state.best_bid_ticks > 0) {
        const std::int64_t center = state.last_trade_price_ticks > 0
            ? state.last_trade_price_ticks
            : static_cast<std::int64_t>(state.best_bid_ticks) + tick;
        bid = center - tick;
        ask = center + tick;
    } else if (state.best_ask_ticks > 0) {
        const std::int64_t center = state.last_trade_price_ticks > 0
            ? state.last_trade_price_ticks
            : static_cast<std::int64_t>(state.best_ask_ticks) - tick;
        bid = center - tick;
        ask = center + tick;
    } else if (state.last_trade_price_ticks > 2 * tick
               || state.fundamental_value_ticks > static_cast<double>(2 * tick)) {
        const double recovery_reference = state.last_trade_price_ticks > 2 * tick
            ? static_cast<double>(state.last_trade_price_ticks)
            : state.fundamental_value_ticks;
        const auto center = static_cast<std::int64_t>(
            std::llround(recovery_reference / static_cast<double>(tick))) * tick;
        bid = center - tick;
        ask = center + tick;
    }

    if (bid <= 0 || ask <= bid
        || bid > std::numeric_limits<std::int32_t>::max()
        || ask > std::numeric_limits<std::int32_t>::max()) {
        return messages;
    }
    std::int64_t level_quantity = book_config->quote_quantity > 0
        ? book_config->quote_quantity : config_.quote_quantity;
    for (std::int32_t level = 0; level < config_.quote_levels; ++level) {
        const std::int64_t level_bid = bid - static_cast<std::int64_t>(level) * tick;
        const std::int64_t level_ask = ask + static_cast<std::int64_t>(level) * tick;
        if (level_bid <= 0 || level_ask <= level_bid
            || level_ask > std::numeric_limits<std::int32_t>::max()) {
            continue;
        }
        messages.push_back(make_message(
            book_id, OrderAction::Limit, Side::Buy,
            static_cast<std::int32_t>(level_quantity),
            static_cast<std::int32_t>(level_bid), decision_time_ns, arrival_time_ns));
        messages.push_back(make_message(
            book_id, OrderAction::Limit, Side::Sell,
            static_cast<std::int32_t>(level_quantity),
            static_cast<std::int32_t>(level_ask), decision_time_ns, arrival_time_ns));
        level_quantity = std::min<std::int64_t>(
            config_.max_quote_quantity_per_level,
            level_quantity * static_cast<std::int64_t>(config_.quote_quantity_growth));
    }
    return messages;
}

void SharedMarketMakerAgent::reconcile_pending(std::uint64_t order_sequence,
                                                std::int64_t signed_fill_quantity) {
    auto pending = pending_hedges_.find(order_sequence);
    if (pending == pending_hedges_.end() || signed_fill_quantity == 0) return;
    if ((pending->second.signed_quantity > 0) != (signed_fill_quantity > 0)) return;

    const std::int64_t remaining = pending->second.signed_quantity;
    if (std::abs(signed_fill_quantity) >= std::abs(remaining)) {
        pending_hedges_.erase(pending);
    } else {
        pending->second.signed_quantity -= signed_fill_quantity;
    }
}

void SharedMarketMakerAgent::apply_own_fill(BookId book_id,
                                             Side side,
                                             std::int32_t quantity,
                                             std::int32_t price_ticks,
                                             std::uint64_t order_sequence) {
    if (quantity <= 0 || price_ticks <= 0) return;
    const std::int64_t signed_quantity = side == Side::Buy
        ? static_cast<std::int64_t>(quantity)
        : -static_cast<std::int64_t>(quantity);
    const std::int64_t notional = static_cast<std::int64_t>(quantity)
        * static_cast<std::int64_t>(price_ticks);
    inventory_by_book_[book_id] += signed_quantity;
    cash_by_book_[book_id] += side == Side::Buy ? -notional : notional;
    reconcile_pending(order_sequence, signed_quantity);
}

std::vector<OrderMessage> SharedMarketMakerAgent::on_trade(
    const TradeExecution& execution,
    bool allow_cross_book_reaction) {
    const bool is_buyer = execution.buyer_owner_id == config_.logical_owner_id;
    const bool is_seller = execution.seller_owner_id == config_.logical_owner_id;
    if (!is_buyer && !is_seller) return {};
    if (execution.quantity <= 0 || execution.price_ticks <= 0) return {};

    if (is_buyer) {
        apply_own_fill(execution.book_id, Side::Buy, execution.quantity,
                       execution.price_ticks, execution.buyer_order_sequence);
    }
    if (is_seller) {
        apply_own_fill(execution.book_id, Side::Sell, execution.quantity,
                       execution.price_ticks, execution.seller_order_sequence);
    }

    // Hedge fills must update cash, inventory, and pending-quantity accounting,
    // but must not recursively launch another hedge.  Otherwise a single
    // source fill can create an artificial cross-book ping-pong chain.
    if (!allow_cross_book_reaction || !config_.enable_cross_book_hedging) return {};

    const auto* source = find_book(execution.book_id);
    if (source == nullptr) return {};
    // The exact one-book QQQ baseline uses the same accounting and quoting
    // implementation, but has no second venue in which a causal reaction can
    // occur.  Its own fills therefore stop at the accounting update above.
    if (config_.books.size() == 1) return {};
    const double exposure = projected_beta_exposure();
    if (std::abs(exposure) <= config_.exposure_threshold) return {};

    const std::vector<SharedMarketMakerHedgeRoute> routes =
        source->hedge_routes.empty()
            ? std::vector<SharedMarketMakerHedgeRoute>{{source->hedge_book_id, 1.0}}
            : source->hedge_routes;
    double total_weight = 0.0;
    for (const SharedMarketMakerHedgeRoute& route : routes) {
        total_weight += route.weight;
    }
    const std::int64_t report_time = checked_add_time(
        execution.timestamp_ns, config_.report_latency_ns);
    const std::int64_t decision_time = checked_add_time(
        report_time, config_.reaction_latency_ns);
    const std::int64_t arrival_time = checked_add_time(
        decision_time, config_.network_latency_ns);
    std::vector<OrderMessage> hedges;
    hedges.reserve(routes.size());
    for (const SharedMarketMakerHedgeRoute& route : routes) {
        const auto* hedge_book = find_book(route.book_id);
        if (hedge_book == nullptr || hedge_book->book_id == execution.book_id) {
            throw std::logic_error("invalid cross-asset hedge route");
        }
        const double allocation = route.weight / total_weight;
        const double ideal_signed_quantity =
            -exposure * allocation / hedge_book->beta;
        if (!std::isfinite(ideal_signed_quantity)
            || ideal_signed_quantity == 0.0) {
            continue;
        }
        const std::int64_t lot = config_.hedge_lot_size;
        const double ideal_lots = std::abs(ideal_signed_quantity)
            / static_cast<double>(lot);
        std::int64_t lots = std::max<std::int64_t>(
            1, static_cast<std::int64_t>(std::llround(ideal_lots)));
        std::int64_t quantity = lots * lot;
        quantity = std::min<std::int64_t>(
            quantity, config_.max_hedge_quantity);
        if (quantity <= 0) continue;
        const Side side = ideal_signed_quantity > 0.0 ? Side::Buy : Side::Sell;
        OrderMessage hedge = make_message(
            hedge_book->book_id, OrderAction::Market, side,
            static_cast<std::int32_t>(quantity), 0, decision_time, arrival_time);
        pending_hedges_.emplace(
            hedge.sequence,
            PendingHedge{hedge.book_id,
                         side == Side::Buy ? quantity : -quantity});
        hedges.push_back(std::move(hedge));
    }
    return hedges;
}

bool SharedMarketMakerAgent::complete_order(std::uint64_t sequence) {
    return pending_hedges_.erase(sequence) != 0;
}

std::int64_t SharedMarketMakerAgent::inventory(BookId book_id) const {
    const auto it = inventory_by_book_.find(book_id);
    return it == inventory_by_book_.end() ? 0 : it->second;
}

std::int64_t SharedMarketMakerAgent::cash_ticks(BookId book_id) const {
    const auto it = cash_by_book_.find(book_id);
    return it == cash_by_book_.end() ? 0 : it->second;
}

std::int64_t SharedMarketMakerAgent::total_cash_ticks() const {
    std::int64_t total = 0;
    for (const auto& [book_id, cash] : cash_by_book_) {
        (void)book_id;
        total += cash;
    }
    return total;
}

std::int64_t SharedMarketMakerAgent::projected_inventory(BookId book_id) const {
    std::int64_t result = inventory(book_id);
    for (const auto& [sequence, pending] : pending_hedges_) {
        (void)sequence;
        if (pending.book_id == book_id) result += pending.signed_quantity;
    }
    return result;
}

double SharedMarketMakerAgent::beta_exposure() const {
    double exposure = 0.0;
    for (const auto& book : config_.books) {
        exposure += book.beta * static_cast<double>(inventory(book.book_id));
    }
    return exposure;
}

double SharedMarketMakerAgent::projected_beta_exposure() const {
    double exposure = 0.0;
    for (const auto& book : config_.books) {
        exposure += book.beta * static_cast<double>(projected_inventory(book.book_id));
    }
    return exposure;
}

} // namespace dlob
