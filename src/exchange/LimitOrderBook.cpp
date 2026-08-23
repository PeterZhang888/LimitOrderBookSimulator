#include "exchange/LimitOrderBook.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>

namespace dlob {

namespace {

int checked_depth(std::int64_t total, const char* description) {
    if (total < 0 || total > std::numeric_limits<int>::max()) {
        throw std::overflow_error(description);
    }
    return static_cast<int>(total);
}

} // namespace

LimitOrderBook::LimitOrderBook(int tick_size, BookId book_id)
    : tick_size_(std::max(1, tick_size)), book_id_(book_id) {}

void LimitOrderBook::seed_default_book(double depth_scale) {
    const int bid_prices[] = {2203400,2203300,2203200,2203100,2203000,2202900,2202800,2202600,2202500,2202400};
    const int bid_qty[]    = {1623,1723,2100,1100,1200,200,564,500,700,200};
    const int ask_prices[] = {2203700,2203800,2203900,2204000,2204100,2204200,2204300,2204400,2204500,2204600};
    const int ask_qty[]    = {823,823,1823,1923,1923,1223,823,200,823,823};

    std::uint64_t id = 1;
    for (int i = 0; i < 10; ++i) {
        const int quantity = std::max(1, static_cast<int>(std::llround(depth_scale * bid_qty[i])));
        bids_[bid_prices[i]].push_back(RestingOrder{id++, 0, Side::Buy, quantity, bid_prices[i], 0});
        owner_bid_prices_[0].insert(bid_prices[i]);
        total_bid_quantity_ += quantity;
        background_bid_quantity_ += quantity;
    }
    for (int i = 0; i < 10; ++i) {
        const int quantity = std::max(1, static_cast<int>(std::llround(depth_scale * ask_qty[i])));
        asks_[ask_prices[i]].push_back(RestingOrder{id++, 0, Side::Sell, quantity, ask_prices[i], 0});
        owner_ask_prices_[0].insert(ask_prices[i]);
        total_ask_quantity_ += quantity;
        background_ask_quantity_ += quantity;
    }
}

void LimitOrderBook::seed_calibrated_book(
    int best_bid_ticks,
    int best_ask_ticks,
    int best_bid_depth,
    int best_ask_depth,
    double depth_scale) {
    if (best_bid_ticks <= 0 || best_ask_ticks <= best_bid_ticks
        || best_bid_depth <= 0 || best_ask_depth <= 0
        || !std::isfinite(depth_scale) || depth_scale <= 0.0) {
        throw std::invalid_argument("invalid calibrated opening book");
    }
    if (!bids_.empty() || !asks_.empty()) {
        throw std::logic_error("cannot seed a non-empty limit order book");
    }

    // The observed BBO determines the first level.  Deeper deterministic
    // levels provide finite support for market orders without pretending that
    // the full opening depth curve was calibrated from one snapshot.
    constexpr double depth_shape[] = {1.0, 1.0, 1.25, 1.5, 1.5,
                                      1.0, 0.75, 0.5, 0.5, 0.25};
    static_assert(std::size(depth_shape)
                  == static_cast<std::size_t>(reduced_background_depth_levels));
    std::uint64_t id = 1;
    for (std::size_t level = 0; level < std::size(depth_shape); ++level) {
        const std::int64_t raw_price = static_cast<std::int64_t>(best_bid_ticks)
            - static_cast<std::int64_t>(level) * tick_size_;
        if (raw_price <= 0) break;
        const int price = static_cast<int>(raw_price);
        const int quantity = std::max(1, static_cast<int>(std::llround(
            depth_scale * static_cast<double>(best_bid_depth) * depth_shape[level])));
        bids_[price].push_back(RestingOrder{id++, 0, Side::Buy, quantity, price, 0});
        owner_bid_prices_[0].insert(price);
        total_bid_quantity_ += quantity;
        background_bid_quantity_ += quantity;
    }
    for (std::size_t level = 0; level < std::size(depth_shape); ++level) {
        const std::int64_t raw_price = static_cast<std::int64_t>(best_ask_ticks)
            + static_cast<std::int64_t>(level) * tick_size_;
        if (raw_price > std::numeric_limits<int>::max()) break;
        const int price = static_cast<int>(raw_price);
        if (price <= best_bid_ticks) break;
        const int quantity = std::max(1, static_cast<int>(std::llround(
            depth_scale * static_cast<double>(best_ask_depth) * depth_shape[level])));
        asks_[price].push_back(RestingOrder{id++, 0, Side::Sell, quantity, price, 0});
        owner_ask_prices_[0].insert(price);
        total_ask_quantity_ += quantity;
        background_ask_quantity_ += quantity;
    }
}

bool LimitOrderBook::has_bid() const { return !bids_.empty(); }
bool LimitOrderBook::has_ask() const { return !asks_.empty(); }
int LimitOrderBook::best_bid() const { return has_bid() ? bids_.begin()->first : 0; }
int LimitOrderBook::best_ask() const { return has_ask() ? asks_.begin()->first : 0; }

double LimitOrderBook::mid_price() const {
    return has_bid() && has_ask()
        ? 0.5 * (static_cast<double>(best_bid())
                 + static_cast<double>(best_ask()))
        : 0.0;
}

int LimitOrderBook::best_bid_depth() const {
    if (!has_bid()) return 0;
    std::int64_t total = 0;
    for (const RestingOrder& order : bids_.begin()->second) {
        const int quantity = std::max(0, order.quantity);
        if (total > std::numeric_limits<std::int64_t>::max() - quantity) {
            throw std::overflow_error("best bid depth exceeds int64 range");
        }
        total += quantity;
    }
    return checked_depth(total, "best bid depth exceeds int32 range");
}

int LimitOrderBook::best_ask_depth() const {
    if (!has_ask()) return 0;
    std::int64_t total = 0;
    for (const RestingOrder& order : asks_.begin()->second) {
        const int quantity = std::max(0, order.quantity);
        if (total > std::numeric_limits<std::int64_t>::max() - quantity) {
            throw std::overflow_error("best ask depth exceeds int64 range");
        }
        total += quantity;
    }
    return checked_depth(total, "best ask depth exceeds int32 range");
}

int LimitOrderBook::background_best_bid_depth() const {
    const auto owner = owner_bid_prices_.find(0);
    if (owner == owner_bid_prices_.end()) return 0;
    // A strategic maker may improve ahead of the anonymous queue.  The
    // background cancellation response must still observe owner zero's own
    // leading represented level rather than incorrectly reading zero at the
    // market BBO.  Skip stale index entries defensively.
    for (auto price = owner->second.rbegin(); price != owner->second.rend(); ++price) {
        const auto level = bids_.find(*price);
        if (level == bids_.end()) continue;
        std::int64_t total = 0;
        for (const RestingOrder& order : level->second) {
            if (order.owner_id != 0) continue;
            const int quantity = std::max(0, order.quantity);
            if (total > std::numeric_limits<std::int64_t>::max() - quantity) {
                throw std::overflow_error(
                    "background best bid depth exceeds int64 range");
            }
            total += quantity;
        }
        if (total > 0) {
            return checked_depth(
                total, "background best bid depth exceeds int32 range");
        }
    }
    return 0;
}

int LimitOrderBook::background_best_ask_depth() const {
    const auto owner = owner_ask_prices_.find(0);
    if (owner == owner_ask_prices_.end()) return 0;
    for (const int price : owner->second) {
        const auto level = asks_.find(price);
        if (level == asks_.end()) continue;
        std::int64_t total = 0;
        for (const RestingOrder& order : level->second) {
            if (order.owner_id != 0) continue;
            const int quantity = std::max(0, order.quantity);
            if (total > std::numeric_limits<std::int64_t>::max() - quantity) {
                throw std::overflow_error(
                    "background best ask depth exceeds int64 range");
            }
            total += quantity;
        }
        if (total > 0) {
            return checked_depth(
                total, "background best ask depth exceeds int32 range");
        }
    }
    return 0;
}

std::int64_t LimitOrderBook::total_bid_depth() const {
    return total_bid_quantity_;
}

std::int64_t LimitOrderBook::total_ask_depth() const {
    return total_ask_quantity_;
}

std::int64_t LimitOrderBook::total_background_bid_depth() const {
    return background_bid_quantity_;
}

std::int64_t LimitOrderBook::total_background_ask_depth() const {
    return background_ask_quantity_;
}

std::int64_t LimitOrderBook::owner_resting_depth(
    std::int32_t owner_id, Side side) const {
    const auto& owner_prices = side == Side::Buy
        ? owner_bid_prices_ : owner_ask_prices_;
    const auto indexed = owner_prices.find(owner_id);
    if (indexed == owner_prices.end()) return 0;
    std::int64_t total = 0;
    const auto accumulate = [&](const auto& levels) {
        for (const int price : indexed->second) {
            const auto level = levels.find(price);
            if (level == levels.end()) continue;
            for (const RestingOrder& order : level->second) {
                if (order.owner_id != owner_id || order.quantity <= 0) continue;
                if (total > std::numeric_limits<std::int64_t>::max()
                        - order.quantity) {
                    throw std::overflow_error("owner resting depth overflow");
                }
                total += order.quantity;
            }
        }
    };
    if (side == Side::Buy) accumulate(bids_);
    else accumulate(asks_);
    return total;
}

std::vector<OwnerRestingQuote>
LimitOrderBook::owner_resting_quotes(std::int32_t owner_id) const {
    std::vector<OwnerRestingQuote> result;
    const auto append = [&](Side side, const auto& owner_prices,
                            const auto& levels) {
        const auto indexed = owner_prices.find(owner_id);
        if (indexed == owner_prices.end()) return;
        for (const int price : indexed->second) {
            const auto level = levels.find(price);
            if (level == levels.end()) continue;
            std::int64_t quantity = 0;
            for (const RestingOrder& order : level->second) {
                if (order.owner_id != owner_id || order.quantity <= 0) continue;
                if (quantity > std::numeric_limits<std::int64_t>::max()
                        - order.quantity) {
                    throw std::overflow_error(
                        "owner quote quantity overflow");
                }
                quantity += order.quantity;
            }
            if (quantity > 0) {
                result.push_back(OwnerRestingQuote{
                    side, price, quantity});
            }
        }
    };
    append(Side::Buy, owner_bid_prices_, bids_);
    append(Side::Sell, owner_ask_prices_, asks_);
    return result;
}

void LimitOrderBook::record_fill(std::int32_t owner_id,
                                             std::uint64_t order_sequence,
                                             OrderAction action,
                                             Side side,
                                             int quantity,
                                             int price_ticks,
                                             std::int64_t timestamp_ns) {
    if (owner_id <= 0 || quantity <= 0) return;
    AgentReport report;
    report.timestamp_ns = timestamp_ns;
    report.owner_id = owner_id;
    report.order_sequence = order_sequence;
    report.kind = ReportKind::Fill;
    report.action = action;
    report.side = side;
    report.fill_quantity = quantity;
    report.fill_price_ticks = price_ticks;
    report.book_id = book_id_;
    reports_.push_back(report);
}

void LimitOrderBook::record_order_result(const OrderMessage& message,
                                                     const ApplyResult& result) {
    if (message.owner_id <= 0 || message.agent_kind != AgentKind::Institutional) return;
    AgentReport report;
    report.timestamp_ns = message.arrival_time_ns;
    report.owner_id = message.owner_id;
    report.order_sequence = message.sequence;
    report.kind = ReportKind::OrderResult;
    report.action = message.action;
    report.side = message.side;
    report.requested_quantity = result.requested_quantity;
    report.executed_quantity = result.executed_quantity;
    report.resting_quantity = result.resting_quantity;
    report.cancelled_quantity = result.cancelled_quantity;
    report.book_id = book_id_;
    reports_.push_back(report);
}

void LimitOrderBook::record_trade(std::int64_t timestamp_ns,
                                              int price_ticks,
                                              int quantity,
                                              std::int32_t buyer_owner_id,
                                              std::int32_t seller_owner_id,
                                              std::uint64_t buyer_order_sequence,
                                              std::uint64_t seller_order_sequence,
                                              Side aggressor_side,
                                              OrderAction aggressor_action) {
    if (quantity <= 0) return;
    TradeExecution trade;
    trade.book_id = book_id_;
    trade.timestamp_ns = timestamp_ns;
    trade.trade_sequence = next_trade_sequence_++;
    trade.price_ticks = price_ticks;
    trade.quantity = quantity;
    trade.buyer_owner_id = buyer_owner_id;
    trade.seller_owner_id = seller_owner_id;
    trade.buyer_order_sequence = buyer_order_sequence;
    trade.seller_order_sequence = seller_order_sequence;
    trade.aggressor_side = aggressor_side;
    trade.aggressor_action = aggressor_action;
    trades_.push_back(trade);
}

int LimitOrderBook::execute_buy(int quantity,
                                           std::int64_t timestamp_ns,
                                           int limit_price_ticks,
                                           bool has_limit,
                                           std::int32_t aggressor_owner,
                                           std::uint64_t aggressor_order_id,
                                           OrderAction aggressor_action,
                                           bool preserve_background_reserve,
                                           int* boundary_truncated_quantity) {
    int remaining = std::max(0, quantity);
    int executed = 0;
    bool boundary_stop = false;
    const auto reachable_without_reserve = [&]() {
        std::int64_t reachable = 0;
        for (const auto& [price, queue] : asks_) {
            if (has_limit && price > limit_price_ticks) break;
            for (const RestingOrder& order : queue) {
                reachable += std::max(0, order.quantity);
                if (reachable >= remaining) return remaining;
            }
        }
        return static_cast<int>(reachable);
    };
    while (remaining > 0 && !asks_.empty() && !boundary_stop) {
        auto level = asks_.begin();
        const int price = level->first;
        if (has_limit && price > limit_price_ticks) break;
        auto& queue = level->second;
        while (remaining > 0 && !queue.empty()) {
            RestingOrder& resting = queue.front();
            int available = resting.quantity;
            if (preserve_background_reserve) {
                const std::int64_t removable = std::max<std::int64_t>(
                    0, total_ask_quantity_ - 1);
                if (removable == 0) {
                    if (boundary_truncated_quantity != nullptr) {
                        *boundary_truncated_quantity +=
                            reachable_without_reserve();
                    }
                    boundary_stop = true;
                    break;
                }
                available = static_cast<int>(std::min<std::int64_t>(
                    available, removable));
            }
            const int fill = std::min(remaining, available);
            remaining -= fill;
            resting.quantity -= fill;
            executed += fill;
            total_ask_quantity_ -= fill;
            if (resting.owner_id == 0) background_ask_quantity_ -= fill;
            last_trade_price_ticks_ = price;
            record_trade(timestamp_ns, price, fill,
                         aggressor_owner, resting.owner_id,
                         aggressor_order_id, resting.order_id,
                         Side::Buy, aggressor_action);
            record_fill(resting.owner_id, resting.order_id, OrderAction::Limit,
                        resting.side, fill, price, timestamp_ns);
            record_fill(aggressor_owner, aggressor_order_id, aggressor_action,
                        Side::Buy, fill, price, timestamp_ns);
            if (resting.quantity <= 0) {
                const std::int32_t resting_owner = resting.owner_id;
                queue.pop_front();
                const bool owner_remains = std::any_of(
                    queue.begin(), queue.end(),
                    [resting_owner](const RestingOrder& order) {
                        return order.owner_id == resting_owner && order.quantity > 0;
                    });
                if (!owner_remains) {
                    auto owner = owner_ask_prices_.find(resting_owner);
                    if (owner != owner_ask_prices_.end()) {
                        owner->second.erase(price);
                        if (owner->second.empty()) owner_ask_prices_.erase(owner);
                    }
                }
            }
        }
        if (queue.empty()) asks_.erase(level);
    }
    return executed;
}

int LimitOrderBook::execute_sell(int quantity,
                                            std::int64_t timestamp_ns,
                                            int limit_price_ticks,
                                            bool has_limit,
                                            std::int32_t aggressor_owner,
                                            std::uint64_t aggressor_order_id,
                                            OrderAction aggressor_action,
                                            bool preserve_background_reserve,
                                            int* boundary_truncated_quantity) {
    int remaining = std::max(0, quantity);
    int executed = 0;
    bool boundary_stop = false;
    const auto reachable_without_reserve = [&]() {
        std::int64_t reachable = 0;
        for (const auto& [price, queue] : bids_) {
            if (has_limit && price < limit_price_ticks) break;
            for (const RestingOrder& order : queue) {
                reachable += std::max(0, order.quantity);
                if (reachable >= remaining) return remaining;
            }
        }
        return static_cast<int>(reachable);
    };
    while (remaining > 0 && !bids_.empty() && !boundary_stop) {
        auto level = bids_.begin();
        const int price = level->first;
        if (has_limit && price < limit_price_ticks) break;
        auto& queue = level->second;
        while (remaining > 0 && !queue.empty()) {
            RestingOrder& resting = queue.front();
            int available = resting.quantity;
            if (preserve_background_reserve) {
                const std::int64_t removable = std::max<std::int64_t>(
                    0, total_bid_quantity_ - 1);
                if (removable == 0) {
                    if (boundary_truncated_quantity != nullptr) {
                        *boundary_truncated_quantity +=
                            reachable_without_reserve();
                    }
                    boundary_stop = true;
                    break;
                }
                available = static_cast<int>(std::min<std::int64_t>(
                    available, removable));
            }
            const int fill = std::min(remaining, available);
            remaining -= fill;
            resting.quantity -= fill;
            executed += fill;
            total_bid_quantity_ -= fill;
            if (resting.owner_id == 0) background_bid_quantity_ -= fill;
            last_trade_price_ticks_ = price;
            record_trade(timestamp_ns, price, fill,
                         resting.owner_id, aggressor_owner,
                         resting.order_id, aggressor_order_id,
                         Side::Sell, aggressor_action);
            record_fill(resting.owner_id, resting.order_id, OrderAction::Limit,
                        resting.side, fill, price, timestamp_ns);
            record_fill(aggressor_owner, aggressor_order_id, aggressor_action,
                        Side::Sell, fill, price, timestamp_ns);
            if (resting.quantity <= 0) {
                const std::int32_t resting_owner = resting.owner_id;
                queue.pop_front();
                const bool owner_remains = std::any_of(
                    queue.begin(), queue.end(),
                    [resting_owner](const RestingOrder& order) {
                        return order.owner_id == resting_owner && order.quantity > 0;
                    });
                if (!owner_remains) {
                    auto owner = owner_bid_prices_.find(resting_owner);
                    if (owner != owner_bid_prices_.end()) {
                        owner->second.erase(price);
                        if (owner->second.empty()) owner_bid_prices_.erase(owner);
                    }
                }
            }
        }
        if (queue.empty()) bids_.erase(level);
    }
    return executed;
}

ApplyResult LimitOrderBook::add_limit(const OrderMessage& message) {
    ApplyResult result;
    result.requested_quantity = std::max(0, message.quantity);
    if (message.quantity <= 0 || message.price_ticks <= 0) return result;

    // The reconstructed book is an explicit moving ten-level reduction.
    // Empirical tail marks are still sampled and counted, but additions and
    // cancellations outside the same band must both be non-mutating.  The
    // band is expressed in distance from the contemporaneous BBO, not fixed
    // opening prices, so price discovery and recentering remain possible.
    if (message.owner_id == 0
        && message.agent_kind == AgentKind::Background
        && message.distance_ticks >= reduced_background_depth_levels) {
        return result;
    }

    int remaining = message.quantity;
    if (message.side == Side::Buy && has_ask() && message.price_ticks >= best_ask()) {
        result.executed_quantity = execute_buy(remaining, message.arrival_time_ns,
                                               message.price_ticks, true, message.owner_id,
                                               message.sequence, OrderAction::Limit);
        cumulative_aggressive_buy_ += static_cast<std::uint64_t>(result.executed_quantity);
        remaining -= result.executed_quantity;
    } else if (message.side == Side::Sell && has_bid() && message.price_ticks <= best_bid()) {
        result.executed_quantity = execute_sell(remaining, message.arrival_time_ns,
                                                message.price_ticks, true, message.owner_id,
                                                message.sequence, OrderAction::Limit);
        cumulative_aggressive_sell_ += static_cast<std::uint64_t>(result.executed_quantity);
        remaining -= result.executed_quantity;
    }

    if (remaining > 0) {
        std::int64_t& side_total = message.side == Side::Buy
            ? total_bid_quantity_ : total_ask_quantity_;
        if (side_total > std::numeric_limits<std::int64_t>::max() - remaining) {
            throw std::overflow_error("displayed book quantity overflow");
        }
        side_total += remaining;
        if (message.owner_id == 0) {
            std::int64_t& background_total = message.side == Side::Buy
                ? background_bid_quantity_ : background_ask_quantity_;
            if (background_total
                > std::numeric_limits<std::int64_t>::max() - remaining) {
                throw std::overflow_error("background book quantity overflow");
            }
            background_total += remaining;
        }
        RestingOrder order{message.sequence, message.owner_id, message.side, remaining,
                           message.price_ticks, message.arrival_time_ns};
        if (message.side == Side::Buy) {
            auto& queue = bids_[message.price_ticks];
            // Background order references are intentionally absent from the
            // reduced Hawkes model.  Merge adjacent background flow at one
            // price into a single FIFO segment; strategic/maker orders retain
            // individual price-time priority.  This keeps a full-universe
            // session from storing millions of indistinguishable owner-0
            // deque entries without changing aggregate level semantics.
            if (message.owner_id == 0 && !queue.empty()
                && queue.back().owner_id == 0
                && remaining <= std::numeric_limits<int>::max()
                    - queue.back().quantity) {
                queue.back().quantity += remaining;
            } else {
                queue.push_back(order);
            }
            owner_bid_prices_[message.owner_id].insert(message.price_ticks);
        } else {
            auto& queue = asks_[message.price_ticks];
            if (message.owner_id == 0 && !queue.empty()
                && queue.back().owner_id == 0
                && remaining <= std::numeric_limits<int>::max()
                    - queue.back().quantity) {
                queue.back().quantity += remaining;
            } else {
                queue.push_back(order);
            }
            owner_ask_prices_[message.owner_id].insert(message.price_ticks);
        }
        result.resting_quantity = remaining;
    }
    return result;
}

ApplyResult LimitOrderBook::add_conserved_limit(
    const OrderMessage& message) {
    ApplyResult result;
    result.requested_quantity = std::max(0, message.quantity);
    if (message.quantity <= 0 || message.price_ticks <= 0) {
        return result;
    }
    if (message.owner_id != 0
        || message.agent_kind != AgentKind::Background) {
        throw std::invalid_argument(
            "liquidity-conserving revision must remain anonymous background flow");
    }

    // This operation repositions part of the already displayed owner-zero
    // flow.  It must never create quantity, cross the opposite BBO, or worsen
    // its own contemporaneous BBO if the state changed during modeled order
    // latency.  Resolve and validate the donor/target pair before withdrawing
    // anything, so an invalid target cannot partially mutate the book.
    const std::optional<int> donor_price = nearest_background_price(message.side);
    if (!donor_price.has_value()) return result;

    std::int64_t safe_price = message.price_ticks;
    if (message.side == Side::Buy) {
        safe_price = std::max<std::int64_t>(
            safe_price, *donor_price);
    } else {
        safe_price = std::min<std::int64_t>(
            safe_price, *donor_price);
    }
    if (message.side == Side::Buy && has_ask()) {
        safe_price = std::min<std::int64_t>(
            safe_price, static_cast<std::int64_t>(best_ask()) - tick_size_);
    } else if (message.side == Side::Sell && has_bid()) {
        safe_price = std::max<std::int64_t>(
            safe_price, static_cast<std::int64_t>(best_bid()) + tick_size_);
    }
    if (safe_price <= 0
        || safe_price > std::numeric_limits<std::int32_t>::max()) {
        throw std::logic_error(
            "no valid non-crossing price for conserved quote revision");
    }

    const BackgroundWithdrawal withdrawal = withdraw_background_nearest(
        message.side, message.quantity);
    if (withdrawal.quantity <= 0) return result;

    OrderMessage resting = message;
    resting.action = OrderAction::Limit;
    resting.owner_id = 0;
    resting.agent_kind = AgentKind::Background;
    resting.quantity = withdrawal.quantity;
    resting.price_ticks = static_cast<int>(safe_price);
    // ConservedLimit is an endogenous relocation within represented support,
    // not a newly sampled empirical tail mark.  Carrying the incoming
    // distance into add_limit could incorrectly filter the repost after the
    // donor had already been withdrawn.
    resting.distance_ticks = 0;
    result = add_limit(resting);
    if (result.executed_quantity != 0
        || result.resting_quantity != withdrawal.quantity) {
        throw std::logic_error(
            "liquidity-conserving quote became aggressive or failed to rest");
    }
    result.requested_quantity = message.quantity;
    return result;
}

ApplyResult LimitOrderBook::submit_market(const OrderMessage& message) {
    ApplyResult result;
    result.requested_quantity = std::max(0, message.quantity);
    if (message.quantity <= 0) return result;
    // The reduced six-type Hawkes generator has no ITCH order-reference IDs.
    // Its removal mark is therefore an aggregate price-level flow event: it
    // may consume several background queue entries at the contemporaneous
    // best price, but must not become an unlimited order that walks the book.
    // A value order
    // carries its perceived fundamental in price_ticks and may consume levels
    // only up to that price; this permits stabilising reversion without
    // overshooting the fundamental in a sparse reduced book.  Both removal
    // policies stop at the final displayed share on the consumed side.  The
    // protected unit may be anonymous or maker-owned; the boundary preserves
    // a two-sided book without reserving one empirical share when independently
    // supplied liquidity is present.  Explicit institutional stress orders
    // retain normal multi-level execution.
    const bool background_execution = message.agent_kind == AgentKind::Background;
    const bool value_execution = message.agent_kind == AgentKind::Value;
    const bool preserve_background_reserve =
        background_execution || value_execution;
    if (message.side == Side::Buy) {
        const int price_limit = value_execution && message.price_ticks > 0
            ? message.price_ticks : best_ask();
        result.executed_quantity = execute_buy(message.quantity, message.arrival_time_ns,
                                               price_limit,
                                               background_execution || value_execution,
                                               message.owner_id, message.sequence,
                                               OrderAction::Market,
                                               preserve_background_reserve,
                                               &result.boundary_truncated_quantity);
        cumulative_aggressive_buy_ += static_cast<std::uint64_t>(result.executed_quantity);
    } else {
        const int price_limit = value_execution && message.price_ticks > 0
            ? message.price_ticks : best_bid();
        result.executed_quantity = execute_sell(message.quantity, message.arrival_time_ns,
                                                price_limit,
                                                background_execution || value_execution,
                                                message.owner_id, message.sequence,
                                                OrderAction::Market,
                                                preserve_background_reserve,
                                                &result.boundary_truncated_quantity);
        cumulative_aggressive_sell_ += static_cast<std::uint64_t>(result.executed_quantity);
    }
    return result;
}

int LimitOrderBook::cancel_owner(std::int32_t owner_id) {
    if (owner_id <= 0) return 0;
    std::int64_t owner_quantity = 0;
    const auto count_side = [&](const auto& side, const auto& owner_prices) {
        const auto indexed = owner_prices.find(owner_id);
        if (indexed == owner_prices.end()) return;
        for (const int price : indexed->second) {
            const auto level = side.find(price);
            if (level == side.end()) continue;
            for (const RestingOrder& order : level->second) {
                if (order.owner_id != owner_id || order.quantity <= 0) continue;
                if (owner_quantity
                    > std::numeric_limits<int>::max() - order.quantity) {
                    throw std::overflow_error(
                        "cancelled owner quantity exceeds int32 range");
                }
                owner_quantity += order.quantity;
            }
        }
    };
    count_side(bids_, owner_bid_prices_);
    count_side(asks_, owner_ask_prices_);

    int cancelled_quantity = 0;
    auto cancel_side = [&](auto& side, auto& owner_prices,
                           std::int64_t& side_total) {
        auto indexed = owner_prices.find(owner_id);
        if (indexed == owner_prices.end()) return;
        const std::vector<int> prices(indexed->second.begin(), indexed->second.end());
        for (const int price : prices) {
            auto level = side.find(price);
            if (level == side.end()) continue;
            auto& queue = level->second;
            for (auto order = queue.begin(); order != queue.end();) {
                if (order->owner_id == owner_id) {
                    const int quantity = std::max(0, order->quantity);
                    cancelled_quantity += quantity;
                    side_total -= std::max(0, order->quantity);
                    order = queue.erase(order);
                } else {
                    ++order;
                }
            }
            if (queue.empty()) side.erase(level);
        }
        owner_prices.erase(indexed);
    };
    cancel_side(bids_, owner_bid_prices_, total_bid_quantity_);
    cancel_side(asks_, owner_ask_prices_, total_ask_quantity_);
    if (cancelled_quantity != owner_quantity) {
        throw std::logic_error("owner cancellation index is inconsistent");
    }
    return cancelled_quantity;
}

std::optional<int> LimitOrderBook::nearest_background_price(
    Side side) const {
    const auto& owner_prices = side == Side::Buy
        ? owner_bid_prices_ : owner_ask_prices_;
    const auto owner = owner_prices.find(0);
    if (owner == owner_prices.end() || owner->second.empty()) {
        return std::nullopt;
    }
    return side == Side::Buy
        ? std::optional<int>{*owner->second.rbegin()}
        : std::optional<int>{*owner->second.begin()};
}

LimitOrderBook::BackgroundWithdrawal
LimitOrderBook::withdraw_background_nearest(Side side,
                                                        int quantity) {
    if (quantity <= 0) return {};
    auto withdraw_side = [&](auto& levels, auto& owner_prices,
                             bool highest_price) {
        auto owner = owner_prices.find(0);
        if (owner == owner_prices.end() || owner->second.empty()) {
            return BackgroundWithdrawal{};
        }
        const int price = highest_price
            ? *owner->second.rbegin() : *owner->second.begin();
        auto level = levels.find(price);
        if (level == levels.end()) {
            throw std::logic_error("background price index is inconsistent");
        }
        auto& queue = level->second;
        int remaining = quantity;
        int withdrawn = 0;
        // Requote the least time-prioritized aggregate background quantity.
        // Strategic and shared-maker orders at the same price are untouched.
        for (auto order = queue.rbegin();
             order != queue.rend() && remaining > 0;) {
            if (order->owner_id != 0) {
                ++order;
                continue;
            }
            const int remove = std::min(remaining, order->quantity);
            order->quantity -= remove;
            remaining -= remove;
            withdrawn += remove;
            if (order->quantity <= 0) {
                order = std::deque<RestingOrder>::reverse_iterator(
                    queue.erase(std::next(order).base()));
            } else {
                ++order;
            }
        }
        const bool background_remains = std::any_of(
            queue.begin(), queue.end(), [](const RestingOrder& order) {
                return order.owner_id == 0 && order.quantity > 0;
            });
        if (!background_remains) {
            owner->second.erase(price);
            if (owner->second.empty()) owner_prices.erase(owner);
        }
        if (queue.empty()) levels.erase(level);
        if (side == Side::Buy) {
            total_bid_quantity_ -= withdrawn;
            background_bid_quantity_ -= withdrawn;
        } else {
            total_ask_quantity_ -= withdrawn;
            background_ask_quantity_ -= withdrawn;
        }
        return BackgroundWithdrawal{withdrawn, price};
    };
    return side == Side::Buy
        ? withdraw_side(bids_, owner_bid_prices_, true)
        : withdraw_side(asks_, owner_ask_prices_, false);
}

int LimitOrderBook::cancel_at_distance(
    const OrderMessage& message,
    int* boundary_truncated_quantity) {
    if (message.quantity <= 0 || message.distance_ticks < 0
        || (message.owner_id == 0
            && message.agent_kind == AgentKind::Background
            && message.distance_ticks >= reduced_background_depth_levels)) {
        return 0;
    }
    int cancelled = 0;
    auto nearest_owned_price = [](const std::set<int>& prices,
                                  int target) -> std::optional<int> {
        if (prices.empty()) return std::nullopt;
        if (target <= *prices.begin()) return *prices.begin();
        if (target >= *prices.rbegin()) return *prices.rbegin();
        const auto upper = prices.lower_bound(target);
        if (upper == prices.begin()) return *upper;
        if (upper == prices.end()) return *prices.rbegin();
        const int lower = *std::prev(upper);
        const std::int64_t upper_distance = std::abs(
            static_cast<std::int64_t>(*upper) - target);
        const std::int64_t lower_distance = std::abs(
            static_cast<std::int64_t>(lower) - target);
        return lower_distance <= upper_distance ? lower : *upper;
    };
    auto cancel_indexed = [&](auto& side, auto& owner_prices, int target) {
        while (true) {
            auto owner = owner_prices.find(message.owner_id);
            if (owner == owner_prices.end() || owner->second.empty()) return 0;
            // The six-type background generator retains the empirical distance
            // mark but not the source order reference.  Within the declared
            // ten-level reduced support, map anonymous cancellation flow to the
            // nearest represented anonymous level.  Marks outside that support
            // are filtered above.  Strategic owners retain exact-price
            // cancellation semantics so their orders cannot jump levels.
            const std::optional<int> selected =
                message.owner_id == 0
                    && message.agent_kind == AgentKind::Background
                ? nearest_owned_price(owner->second, target)
                : (owner->second.contains(target)
                    ? std::optional<int>{target} : std::nullopt);
            if (!selected.has_value()) return 0;
            auto level = side.find(*selected);
            if (level == side.end()) {
                owner->second.erase(*selected);
                if (owner->second.empty()) owner_prices.erase(owner);
                continue;
            }
            auto& queue = level->second;
            const bool contains_owner = std::any_of(
                queue.begin(), queue.end(),
                [&](const RestingOrder& order) {
                    return order.owner_id == message.owner_id && order.quantity > 0;
                });
            if (!contains_owner) {
                owner->second.erase(*selected);
                if (owner->second.empty()) owner_prices.erase(owner);
                continue;
            }
            int removed_total = 0;
            std::int64_t available_at_selected_level = 0;
            for (const RestingOrder& order : queue) {
                if (order.owner_id == message.owner_id && order.quantity > 0) {
                    available_at_selected_level = std::min<std::int64_t>(
                        message.quantity,
                        available_at_selected_level + order.quantity);
                }
            }
            const int unconstrained_removal = static_cast<int>(
                std::min<std::int64_t>(
                    message.quantity, available_at_selected_level));
            int remaining = unconstrained_removal;
            if (message.owner_id == 0
                && message.agent_kind == AgentKind::Background) {
                const std::int64_t background_total = message.side == Side::Buy
                    ? background_bid_quantity_ : background_ask_quantity_;
                const std::int64_t displayed_total = message.side == Side::Buy
                    ? total_bid_quantity_ : total_ask_quantity_;
                const bool background_is_entire_side =
                    displayed_total <= background_total;
                const int reserve_constrained_removal = static_cast<int>(
                    std::min<std::int64_t>(
                        unconstrained_removal,
                        std::max<std::int64_t>(
                            0, background_total
                                - (background_is_entire_side ? 1 : 0))));
                if (boundary_truncated_quantity != nullptr) {
                    *boundary_truncated_quantity +=
                        unconstrained_removal - reserve_constrained_removal;
                }
                remaining = reserve_constrained_removal;
            }
            // Without source order IDs, cancellation is a reduced aggregate
            // removal at one represented price level.  It may span several
            // background entries at that price, but never another level.
            for (auto order = queue.rbegin();
                 order != queue.rend() && remaining > 0;) {
                if (order->owner_id != message.owner_id) {
                    ++order;
                    continue;
                }
                const int remove = std::min(remaining, order->quantity);
                order->quantity -= remove;
                remaining -= remove;
                removed_total += remove;
                if (order->quantity <= 0) {
                    order = std::deque<RestingOrder>::reverse_iterator(
                        queue.erase(std::next(order).base()));
                } else {
                    ++order;
                }
            }
            const bool owner_remains = std::any_of(
                queue.begin(), queue.end(),
                [&](const RestingOrder& order) {
                    return order.owner_id == message.owner_id && order.quantity > 0;
            });
            if (!owner_remains) {
                owner->second.erase(*selected);
                if (owner->second.empty()) owner_prices.erase(owner);
            }
            if (queue.empty()) side.erase(level);
            return removed_total;
        }
    };
    if (message.side == Side::Buy) {
        if (!has_bid()) return 0;
        const std::int64_t raw_target = static_cast<std::int64_t>(best_bid())
            - static_cast<std::int64_t>(message.distance_ticks) * tick_size_;
        const int target = static_cast<int>(std::clamp<std::int64_t>(
            raw_target, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()));
        cancelled = cancel_indexed(bids_, owner_bid_prices_, target);
    } else {
        if (!has_ask()) return 0;
        const std::int64_t raw_target = static_cast<std::int64_t>(best_ask())
            + static_cast<std::int64_t>(message.distance_ticks) * tick_size_;
        const int target = static_cast<int>(std::clamp<std::int64_t>(
            raw_target, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()));
        cancelled = cancel_indexed(asks_, owner_ask_prices_, target);
    }
    if (message.side == Side::Buy) {
        total_bid_quantity_ -= cancelled;
        if (message.owner_id == 0) background_bid_quantity_ -= cancelled;
    } else {
        total_ask_quantity_ -= cancelled;
        if (message.owner_id == 0) background_ask_quantity_ -= cancelled;
    }
    return cancelled;
}

ApplyResult LimitOrderBook::apply(const OrderMessage& message) {
    ApplyResult result;
    switch (message.action) {
        case OrderAction::Limit:
            result = add_limit(message);
            break;
        case OrderAction::Market:
            result = submit_market(message);
            break;
        case OrderAction::CancelOwner:
            result.cancelled_quantity = cancel_owner(message.owner_id);
            break;
        case OrderAction::CancelAtDistance:
            result.requested_quantity = std::max(0, message.quantity);
            result.cancelled_quantity = cancel_at_distance(
                message, &result.boundary_truncated_quantity);
            break;
        case OrderAction::ConservedLimit:
            result = add_conserved_limit(message);
            break;
    }
    if (total_bid_quantity_ < 0 || total_ask_quantity_ < 0
        || background_bid_quantity_ < 0 || background_ask_quantity_ < 0
        || background_bid_quantity_ > total_bid_quantity_
        || background_ask_quantity_ > total_ask_quantity_
        || (total_bid_quantity_ > 0) != has_bid()
        || (total_ask_quantity_ > 0) != has_ask()) {
        throw std::logic_error("displayed-depth accounting invariant failed");
    }
    record_order_result(message, result);
    return result;
}

MarketState LimitOrderBook::state(std::int64_t time_ns,
                                             double fundamental_value_ticks) const {
    MarketState state;
    state.exchange_time_ns = time_ns;
    state.best_bid_ticks = best_bid();
    state.best_ask_ticks = best_ask();
    state.best_bid_depth = best_bid_depth();
    state.best_ask_depth = best_ask_depth();
    state.background_best_bid_depth = background_best_bid_depth();
    state.background_best_ask_depth = background_best_ask_depth();
    state.total_background_bid_depth = total_background_bid_depth();
    state.total_background_ask_depth = total_background_ask_depth();
    state.last_trade_price_ticks = last_trade_price_ticks_;
    state.mid_price_ticks = mid_price();
    state.fundamental_value_ticks = fundamental_value_ticks;
    state.cumulative_aggressive_buy = cumulative_aggressive_buy_;
    state.cumulative_aggressive_sell = cumulative_aggressive_sell_;
    state.book_id = book_id_;
    return state;
}

MarketState LimitOrderBook::state_excluding_owner(
    std::int64_t time_ns,
    double fundamental_value_ticks,
    std::int32_t excluded_owner_id) const {
    MarketState result = state(time_ns, fundamental_value_ticks);
    if (excluded_owner_id <= 0) return result;

    const auto external_best = [excluded_owner_id](const auto& levels) {
        std::pair<int, int> best{0, 0};
        for (const auto& [price, queue] : levels) {
            std::int64_t depth = 0;
            for (const RestingOrder& order : queue) {
                if (order.owner_id == excluded_owner_id || order.quantity <= 0) {
                    continue;
                }
                if (depth > std::numeric_limits<std::int64_t>::max()
                        - order.quantity) {
                    throw std::overflow_error(
                        "external best-level depth overflow");
                }
                depth += order.quantity;
            }
            if (depth > 0) {
                best.first = price;
                best.second = checked_depth(
                    depth, "external best-level depth exceeds int32 range");
                break;
            }
        }
        return best;
    };
    const auto bid = external_best(bids_);
    const auto ask = external_best(asks_);
    result.best_bid_ticks = bid.first;
    result.best_bid_depth = bid.second;
    result.best_ask_ticks = ask.first;
    result.best_ask_depth = ask.second;
    result.mid_price_ticks = bid.first > 0 && ask.first > bid.first
        ? 0.5 * (static_cast<double>(bid.first)
                 + static_cast<double>(ask.first))
        : 0.0;
    return result;
}

std::vector<AgentReport> LimitOrderBook::take_reports() {
    std::vector<AgentReport> output;
    output.swap(reports_);
    return output;
}

TerminalLiquidationPreview
LimitOrderBook::preview_terminal_liquidation(
    std::int64_t signed_inventory,
    std::int32_t excluded_owner_id,
    int fallback_distance_ticks,
    double reference_value_ticks) const {
    if (fallback_distance_ticks < 0) {
        throw std::invalid_argument(
            "terminal liquidation fallback distance must be non-negative");
    }

    TerminalLiquidationPreview preview;
    if (signed_inventory == 0) return preview;
    if (signed_inventory == std::numeric_limits<std::int64_t>::min()) {
        throw std::overflow_error(
            "terminal liquidation inventory magnitude overflow");
    }
    const bool sell_long = signed_inventory > 0;
    const std::int64_t requested = sell_long
        ? signed_inventory : -signed_inventory;
    preview.requested_quantity = requested;
    std::int64_t remaining = requested;
    int last_external_price = 0;

    const auto consume = [&](const auto& levels) {
        for (const auto& [price, queue] : levels) {
            if (remaining == 0) break;
            std::int64_t external_at_level = 0;
            for (const RestingOrder& order : queue) {
                if (order.owner_id == excluded_owner_id || order.quantity <= 0) {
                    continue;
                }
                if (external_at_level
                    > std::numeric_limits<std::int64_t>::max() - order.quantity) {
                    throw std::overflow_error(
                        "terminal liquidation level depth overflow");
                }
                external_at_level += order.quantity;
            }
            if (external_at_level == 0) continue;
            last_external_price = price;
            const std::int64_t fill = std::min(remaining, external_at_level);
            preview.signed_cash_change_ticks +=
                static_cast<long double>(sell_long ? price : -price)
                * static_cast<long double>(fill);
            preview.displayed_filled_quantity += fill;
            remaining -= fill;
        }
    };

    if (sell_long) {
        consume(bids_);
    } else {
        consume(asks_);
    }

    preview.unliquidated_quantity = remaining;
    if (remaining == 0) return preview;

    std::int64_t reference = last_external_price;
    if (reference > 0) {
        preview.fallback_from_external_quote = true;
    } else {
        // No external depth exists on the liquidation side. Use the external
        // quote on the opposite side when available; never use a quote owned
        // by the dealer whose inventory is being valued.
        const auto external_best_price = [excluded_owner_id](const auto& levels) {
            for (const auto& [price, queue] : levels) {
                const bool has_external = std::any_of(
                    queue.begin(), queue.end(),
                    [excluded_owner_id](const RestingOrder& order) {
                        return order.owner_id != excluded_owner_id
                            && order.quantity > 0;
                    });
                if (has_external) return price;
            }
            return 0;
        };
        const int opposite_external = sell_long
            ? external_best_price(asks_) : external_best_price(bids_);
        if (opposite_external > 0) {
            reference = sell_long
                ? std::max<std::int64_t>(
                    tick_size_,
                    static_cast<std::int64_t>(opposite_external) - tick_size_)
                : std::min<std::int64_t>(
                    std::numeric_limits<int>::max(),
                    static_cast<std::int64_t>(opposite_external) + tick_size_);
            preview.fallback_from_external_quote = true;
        } else {
            if (!std::isfinite(reference_value_ticks)
                || reference_value_ticks <= 0.0) {
                throw std::invalid_argument(
                    "terminal liquidation requires a positive reference value");
            }
            reference = static_cast<std::int64_t>(std::llround(std::clamp(
                reference_value_ticks,
                static_cast<double>(tick_size_),
                static_cast<double>(std::numeric_limits<int>::max()))));
            preview.fallback_from_reference_value = true;
        }
    }
    const std::int64_t distance =
        static_cast<std::int64_t>(fallback_distance_ticks) * tick_size_;
    const std::int64_t fallback = sell_long
        ? std::max<std::int64_t>(tick_size_, reference - distance)
        : std::min<std::int64_t>(
            std::numeric_limits<int>::max(), reference + distance);
    preview.fallback_price_ticks = static_cast<int>(fallback);
    preview.signed_cash_change_ticks +=
        static_cast<long double>(sell_long ? fallback : -fallback)
        * static_cast<long double>(remaining);
    return preview;
}

std::vector<TradeExecution> LimitOrderBook::take_trades() {
    std::vector<TradeExecution> output;
    output.swap(trades_);
    return output;
}

} // namespace dlob
