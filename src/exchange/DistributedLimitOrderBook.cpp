#include "exchange/DistributedLimitOrderBook.hpp"

#include <algorithm>
#include <cmath>

namespace dlob {

DistributedLimitOrderBook::DistributedLimitOrderBook(int tick_size)
    : tick_size_(std::max(1, tick_size)) {}

void DistributedLimitOrderBook::seed_default_book(double depth_scale) {
    const int bid_prices[] = {2203400,2203300,2203200,2203100,2203000,2202900,2202800,2202600,2202500,2202400};
    const int bid_qty[]    = {1623,1723,2100,1100,1200,200,564,500,700,200};
    const int ask_prices[] = {2203700,2203800,2203900,2204000,2204100,2204200,2204300,2204400,2204500,2204600};
    const int ask_qty[]    = {823,823,1823,1923,1923,1223,823,200,823,823};

    std::uint64_t id = 1;
    for (int i = 0; i < 10; ++i) {
        const int quantity = std::max(1, static_cast<int>(std::llround(depth_scale * bid_qty[i])));
        bids_[bid_prices[i]].push_back(RestingOrder{id++, 0, Side::Buy, quantity, bid_prices[i], 0});
    }
    for (int i = 0; i < 10; ++i) {
        const int quantity = std::max(1, static_cast<int>(std::llround(depth_scale * ask_qty[i])));
        asks_[ask_prices[i]].push_back(RestingOrder{id++, 0, Side::Sell, quantity, ask_prices[i], 0});
    }
}

bool DistributedLimitOrderBook::has_bid() const { return !bids_.empty(); }
bool DistributedLimitOrderBook::has_ask() const { return !asks_.empty(); }
int DistributedLimitOrderBook::best_bid() const { return has_bid() ? bids_.begin()->first : 0; }
int DistributedLimitOrderBook::best_ask() const { return has_ask() ? asks_.begin()->first : 0; }

double DistributedLimitOrderBook::mid_price() const {
    return has_bid() && has_ask() ? 0.5 * static_cast<double>(best_bid() + best_ask()) : 0.0;
}

int DistributedLimitOrderBook::best_bid_depth() const {
    if (!has_bid()) return 0;
    int total = 0;
    for (const RestingOrder& order : bids_.begin()->second) total += std::max(0, order.quantity);
    return total;
}

int DistributedLimitOrderBook::best_ask_depth() const {
    if (!has_ask()) return 0;
    int total = 0;
    for (const RestingOrder& order : asks_.begin()->second) total += std::max(0, order.quantity);
    return total;
}

void DistributedLimitOrderBook::record_fill(std::int32_t owner_id,
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
    reports_.push_back(report);
}

void DistributedLimitOrderBook::record_order_result(const OrderMessage& message,
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
    reports_.push_back(report);
}

int DistributedLimitOrderBook::execute_buy(int quantity,
                                           std::int64_t timestamp_ns,
                                           int limit_price_ticks,
                                           bool has_limit,
                                           std::int32_t aggressor_owner,
                                           std::uint64_t aggressor_order_id,
                                           OrderAction aggressor_action) {
    int remaining = std::max(0, quantity);
    int executed = 0;
    while (remaining > 0 && !asks_.empty()) {
        auto level = asks_.begin();
        const int price = level->first;
        if (has_limit && price > limit_price_ticks) break;
        auto& queue = level->second;
        while (remaining > 0 && !queue.empty()) {
            RestingOrder& resting = queue.front();
            const int fill = std::min(remaining, resting.quantity);
            remaining -= fill;
            resting.quantity -= fill;
            executed += fill;
            last_trade_price_ticks_ = price;
            record_fill(resting.owner_id, resting.order_id, OrderAction::Limit,
                        resting.side, fill, price, timestamp_ns);
            record_fill(aggressor_owner, aggressor_order_id, aggressor_action,
                        Side::Buy, fill, price, timestamp_ns);
            if (resting.quantity <= 0) queue.pop_front();
        }
        if (queue.empty()) asks_.erase(level);
    }
    return executed;
}

int DistributedLimitOrderBook::execute_sell(int quantity,
                                            std::int64_t timestamp_ns,
                                            int limit_price_ticks,
                                            bool has_limit,
                                            std::int32_t aggressor_owner,
                                            std::uint64_t aggressor_order_id,
                                            OrderAction aggressor_action) {
    int remaining = std::max(0, quantity);
    int executed = 0;
    while (remaining > 0 && !bids_.empty()) {
        auto level = bids_.begin();
        const int price = level->first;
        if (has_limit && price < limit_price_ticks) break;
        auto& queue = level->second;
        while (remaining > 0 && !queue.empty()) {
            RestingOrder& resting = queue.front();
            const int fill = std::min(remaining, resting.quantity);
            remaining -= fill;
            resting.quantity -= fill;
            executed += fill;
            last_trade_price_ticks_ = price;
            record_fill(resting.owner_id, resting.order_id, OrderAction::Limit,
                        resting.side, fill, price, timestamp_ns);
            record_fill(aggressor_owner, aggressor_order_id, aggressor_action,
                        Side::Sell, fill, price, timestamp_ns);
            if (resting.quantity <= 0) queue.pop_front();
        }
        if (queue.empty()) bids_.erase(level);
    }
    return executed;
}

ApplyResult DistributedLimitOrderBook::add_limit(const OrderMessage& message) {
    ApplyResult result;
    result.requested_quantity = std::max(0, message.quantity);
    if (message.quantity <= 0 || message.price_ticks <= 0) return result;

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
        RestingOrder order{message.sequence, message.owner_id, message.side, remaining,
                           message.price_ticks, message.arrival_time_ns};
        if (message.side == Side::Buy) bids_[message.price_ticks].push_back(order);
        else asks_[message.price_ticks].push_back(order);
        result.resting_quantity = remaining;
    }
    return result;
}

ApplyResult DistributedLimitOrderBook::submit_market(const OrderMessage& message) {
    ApplyResult result;
    result.requested_quantity = std::max(0, message.quantity);
    if (message.quantity <= 0) return result;
    if (message.side == Side::Buy) {
        result.executed_quantity = execute_buy(message.quantity, message.arrival_time_ns,
                                               0, false, message.owner_id, message.sequence,
                                               OrderAction::Market);
        cumulative_aggressive_buy_ += static_cast<std::uint64_t>(result.executed_quantity);
    } else {
        result.executed_quantity = execute_sell(message.quantity, message.arrival_time_ns,
                                                0, false, message.owner_id, message.sequence,
                                                OrderAction::Market);
        cumulative_aggressive_sell_ += static_cast<std::uint64_t>(result.executed_quantity);
    }
    return result;
}

int DistributedLimitOrderBook::cancel_owner(std::int32_t owner_id) {
    if (owner_id <= 0) return 0;
    int cancelled_quantity = 0;
    auto cancel_side = [&](auto& side) {
        for (auto level = side.begin(); level != side.end();) {
            auto& queue = level->second;
            for (auto order = queue.begin(); order != queue.end();) {
                if (order->owner_id == owner_id) {
                    cancelled_quantity += std::max(0, order->quantity);
                    order = queue.erase(order);
                } else {
                    ++order;
                }
            }
            if (queue.empty()) level = side.erase(level);
            else ++level;
        }
    };
    cancel_side(bids_);
    cancel_side(asks_);
    return cancelled_quantity;
}

int DistributedLimitOrderBook::cancel_at_distance(const OrderMessage& message) {
    if (message.quantity <= 0 || message.distance_ticks < 0) return 0;
    int cancelled = 0;
    if (message.side == Side::Buy) {
        if (!has_bid()) return 0;
        const int target = best_bid() - message.distance_ticks * tick_size_;
        auto it = bids_.find(target);
        if (it == bids_.end()) return 0;
        int remaining = message.quantity;
        auto& queue = it->second;
        while (remaining > 0 && !queue.empty()) {
            RestingOrder& order = queue.back();
            const int remove = std::min(remaining, order.quantity);
            order.quantity -= remove;
            remaining -= remove;
            cancelled += remove;
            if (order.quantity <= 0) queue.pop_back();
        }
        if (queue.empty()) bids_.erase(it);
    } else {
        if (!has_ask()) return 0;
        const int target = best_ask() + message.distance_ticks * tick_size_;
        auto it = asks_.find(target);
        if (it == asks_.end()) return 0;
        int remaining = message.quantity;
        auto& queue = it->second;
        while (remaining > 0 && !queue.empty()) {
            RestingOrder& order = queue.back();
            const int remove = std::min(remaining, order.quantity);
            order.quantity -= remove;
            remaining -= remove;
            cancelled += remove;
            if (order.quantity <= 0) queue.pop_back();
        }
        if (queue.empty()) asks_.erase(it);
    }
    return cancelled;
}

ApplyResult DistributedLimitOrderBook::apply(const OrderMessage& message) {
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
            result.cancelled_quantity = cancel_at_distance(message);
            break;
    }
    record_order_result(message, result);
    return result;
}

MarketState DistributedLimitOrderBook::state(std::int64_t time_ns,
                                             double fundamental_value_ticks) const {
    MarketState state;
    state.exchange_time_ns = time_ns;
    state.best_bid_ticks = best_bid();
    state.best_ask_ticks = best_ask();
    state.best_bid_depth = best_bid_depth();
    state.best_ask_depth = best_ask_depth();
    state.last_trade_price_ticks = last_trade_price_ticks_;
    state.mid_price_ticks = mid_price();
    state.fundamental_value_ticks = fundamental_value_ticks;
    state.cumulative_aggressive_buy = cumulative_aggressive_buy_;
    state.cumulative_aggressive_sell = cumulative_aggressive_sell_;
    return state;
}

std::vector<AgentReport> DistributedLimitOrderBook::take_reports() {
    std::vector<AgentReport> output;
    output.swap(reports_);
    return output;
}

} // namespace dlob
