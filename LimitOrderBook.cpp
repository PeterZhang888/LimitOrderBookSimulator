#include "LimitOrderBook.hpp"

#include <algorithm>
#include <cmath>

LimitOrderBook::LimitOrderBook(int tick_size)
    : tick_size_(std::max(1, tick_size)) {}

bool LimitOrderBook::has_bid() const { return !bids_.empty(); }
bool LimitOrderBook::has_ask() const { return !asks_.empty(); }
int LimitOrderBook::best_bid() const { return bids_.empty() ? 0 : bids_.begin()->first; }
int LimitOrderBook::best_ask() const { return asks_.empty() ? 0 : asks_.begin()->first; }

double LimitOrderBook::mid_price() const {
    if (!has_bid() || !has_ask()) return 0.0;
    return 0.5 * static_cast<double>(best_bid() + best_ask());
}

int LimitOrderBook::spread_ticks() const {
    if (!has_bid() || !has_ask()) return 0;
    return best_ask() - best_bid();
}

int LimitOrderBook::quantity_at_price(Side side, int price_ticks) const {
    int total = 0;
    if (side == Side::Buy) {
        auto it = bids_.find(price_ticks);
        if (it == bids_.end()) return 0;
        for (const auto& order : it->second) total += std::max(0, order.quantity);
    } else {
        auto it = asks_.find(price_ticks);
        if (it == asks_.end()) return 0;
        for (const auto& order : it->second) total += std::max(0, order.quantity);
    }
    return total;
}

int LimitOrderBook::quantity_at_best_bid() const { return has_bid() ? quantity_at_price(Side::Buy, best_bid()) : 0; }
int LimitOrderBook::quantity_at_best_ask() const { return has_ask() ? quantity_at_price(Side::Sell, best_ask()) : 0; }

void LimitOrderBook::record_fill(const Order& resting, int quantity, int price_ticks, std::int64_t timestamp_ns) {
    if (resting.owner_id == 0 || quantity <= 0) return;
    execution_reports_[resting.owner_id].push_back(ExecutionReport{resting.order_id, resting.side, quantity, price_ticks, timestamp_ns});
}

int LimitOrderBook::execute_against_asks(int quantity, std::int64_t timestamp_ns, int limit_price_ticks, bool has_limit) {
    int remaining = std::max(0, quantity);
    int executed = 0;
    while (remaining > 0 && !asks_.empty()) {
        auto level_it = asks_.begin();
        const int price = level_it->first;
        if (has_limit && price > limit_price_ticks) break;
        auto& queue = level_it->second;
        while (remaining > 0 && !queue.empty()) {
            Order& resting = queue.front();
            const int fill = std::min(remaining, resting.quantity);
            remaining -= fill;
            executed += fill;
            resting.quantity -= fill;
            record_fill(resting, fill, price, timestamp_ns);
            if (resting.quantity <= 0) queue.pop_front();
        }
        if (queue.empty()) asks_.erase(level_it);
    }
    return executed;
}

int LimitOrderBook::execute_against_bids(int quantity, std::int64_t timestamp_ns, int limit_price_ticks, bool has_limit) {
    int remaining = std::max(0, quantity);
    int executed = 0;
    while (remaining > 0 && !bids_.empty()) {
        auto level_it = bids_.begin();
        const int price = level_it->first;
        if (has_limit && price < limit_price_ticks) break;
        auto& queue = level_it->second;
        while (remaining > 0 && !queue.empty()) {
            Order& resting = queue.front();
            const int fill = std::min(remaining, resting.quantity);
            remaining -= fill;
            executed += fill;
            resting.quantity -= fill;
            record_fill(resting, fill, price, timestamp_ns);
            if (resting.quantity <= 0) queue.pop_front();
        }
        if (queue.empty()) bids_.erase(level_it);
    }
    return executed;
}

void LimitOrderBook::add_limit_order(const Order& raw_order) {
    Order order = raw_order;
    if (order.quantity <= 0 || order.price_ticks <= 0) return;

    // Price-time rule: if a limit order crosses, it trades first and rests only remaining quantity.
    if (order.side == Side::Buy && has_ask() && order.price_ticks >= best_ask()) {
        const int executed = execute_against_asks(order.quantity, order.timestamp_ns, order.price_ticks, true);
        cumulative_aggressive_buy_quantity_ += static_cast<std::uint64_t>(std::max(0, executed));
        order.quantity -= executed;
    } else if (order.side == Side::Sell && has_bid() && order.price_ticks <= best_bid()) {
        const int executed = execute_against_bids(order.quantity, order.timestamp_ns, order.price_ticks, true);
        cumulative_aggressive_sell_quantity_ += static_cast<std::uint64_t>(std::max(0, executed));
        order.quantity -= executed;
    }

    if (order.quantity <= 0) return;
    if (order.side == Side::Buy) bids_[order.price_ticks].push_back(order);
    else asks_[order.price_ticks].push_back(order);
}

int LimitOrderBook::submit_market_order(Side side, int quantity, std::int64_t timestamp_ns) {
    if (quantity <= 0) return 0;
    int executed = 0;
    if (side == Side::Buy) {
        executed = execute_against_asks(quantity, timestamp_ns);
        cumulative_aggressive_buy_quantity_ += static_cast<std::uint64_t>(std::max(0, executed));
    } else {
        executed = execute_against_bids(quantity, timestamp_ns);
        cumulative_aggressive_sell_quantity_ += static_cast<std::uint64_t>(std::max(0, executed));
    }
    return executed;
}

int LimitOrderBook::cancel_at_distance(Side side, int distance_ticks, int quantity, int tick_size) {
    if (quantity <= 0 || distance_ticks < 0) return 0;
    const int tick = std::max(1, tick_size);
    int target_price = 0;
    if (side == Side::Buy) {
        if (!has_bid()) return 0;
        target_price = best_bid() - distance_ticks * tick;
        auto it = bids_.find(target_price);
        if (it == bids_.end()) return 0;
        int remaining = quantity;
        int cancelled = 0;
        auto& queue = it->second;
        while (remaining > 0 && !queue.empty()) {
            Order& order = queue.back(); // cancellations are anonymous; remove from back to preserve front priority.
            const int remove = std::min(remaining, order.quantity);
            order.quantity -= remove;
            remaining -= remove;
            cancelled += remove;
            if (order.quantity <= 0) queue.pop_back();
        }
        if (queue.empty()) bids_.erase(it);
        return cancelled;
    }

    if (!has_ask()) return 0;
    target_price = best_ask() + distance_ticks * tick;
    auto it = asks_.find(target_price);
    if (it == asks_.end()) return 0;
    int remaining = quantity;
    int cancelled = 0;
    auto& queue = it->second;
    while (remaining > 0 && !queue.empty()) {
        Order& order = queue.back();
        const int remove = std::min(remaining, order.quantity);
        order.quantity -= remove;
        remaining -= remove;
        cancelled += remove;
        if (order.quantity <= 0) queue.pop_back();
    }
    if (queue.empty()) asks_.erase(it);
    return cancelled;
}

std::size_t LimitOrderBook::cancel_orders_by_owner(int owner_id) {
    if (owner_id == 0) return 0;
    std::size_t cancelled = 0;
    auto cancel_from_book = [&](auto& book_side) {
        for (auto level_it = book_side.begin(); level_it != book_side.end();) {
            auto& queue = level_it->second;
            for (auto order_it = queue.begin(); order_it != queue.end();) {
                if (order_it->owner_id == owner_id) {
                    order_it = queue.erase(order_it);
                    ++cancelled;
                } else {
                    ++order_it;
                }
            }
            if (queue.empty()) level_it = book_side.erase(level_it);
            else ++level_it;
        }
    };
    cancel_from_book(bids_);
    cancel_from_book(asks_);
    return cancelled;
}

std::vector<ExecutionReport> LimitOrderBook::get_and_clear_execution_reports(int owner_id) {
    std::vector<ExecutionReport> reports;
    auto it = execution_reports_.find(owner_id);
    if (it == execution_reports_.end()) return reports;
    reports.swap(it->second);
    execution_reports_.erase(it);
    return reports;
}

std::uint64_t LimitOrderBook::cumulative_aggressive_buy_quantity() const { return cumulative_aggressive_buy_quantity_; }
std::uint64_t LimitOrderBook::cumulative_aggressive_sell_quantity() const { return cumulative_aggressive_sell_quantity_; }
