#pragma once

#include "Order.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

class LimitOrderBook {
public:
    explicit LimitOrderBook(int tick_size = 100);

    bool has_bid() const;
    bool has_ask() const;
    int best_bid() const;
    int best_ask() const;
    double mid_price() const;
    int spread_ticks() const;
    int quantity_at_best_bid() const;
    int quantity_at_best_ask() const;
    int quantity_at_price(Side side, int price_ticks) const;

    void add_limit_order(const Order& order);
    int submit_market_order(Side side, int quantity, std::int64_t timestamp_ns = 0);
    int cancel_at_distance(Side side, int distance_ticks, int quantity, int tick_size);
    std::size_t cancel_orders_by_owner(int owner_id);

    std::vector<ExecutionReport> get_and_clear_execution_reports(int owner_id);

    std::uint64_t cumulative_aggressive_buy_quantity() const;
    std::uint64_t cumulative_aggressive_sell_quantity() const;

private:
    using BidMap = std::map<int, std::deque<Order>, std::greater<int>>;
    using AskMap = std::map<int, std::deque<Order>>;

    int tick_size_ = 100;
    BidMap bids_;
    AskMap asks_;
    std::unordered_map<int, std::vector<ExecutionReport>> execution_reports_;
    std::uint64_t cumulative_aggressive_buy_quantity_ = 0;
    std::uint64_t cumulative_aggressive_sell_quantity_ = 0;

    int execute_against_asks(int quantity, std::int64_t timestamp_ns, int limit_price_ticks = 0, bool has_limit = false);
    int execute_against_bids(int quantity, std::int64_t timestamp_ns, int limit_price_ticks = 0, bool has_limit = false);
    void record_fill(const Order& resting, int quantity, int price_ticks, std::int64_t timestamp_ns);
};
