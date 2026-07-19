#pragma once

#include "common/DistributedTypes.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <vector>

namespace dlob {

struct ApplyResult {
    int requested_quantity = 0;
    int executed_quantity = 0;
    int resting_quantity = 0;
    int cancelled_quantity = 0;
};

class DistributedLimitOrderBook {
public:
    explicit DistributedLimitOrderBook(int tick_size);

    void seed_default_book(double depth_scale = 1.0);
    ApplyResult apply(const OrderMessage& message);

    MarketState state(std::int64_t time_ns, double fundamental_value_ticks) const;
    std::vector<AgentReport> take_reports();

    bool has_bid() const;
    bool has_ask() const;
    int best_bid() const;
    int best_ask() const;
    int best_bid_depth() const;
    int best_ask_depth() const;
    double mid_price() const;

private:
    struct RestingOrder {
        std::uint64_t order_id = 0;
        std::int32_t owner_id = 0;
        Side side = Side::Buy;
        int quantity = 0;
        int price_ticks = 0;
        std::int64_t timestamp_ns = 0;
    };

    using BidMap = std::map<int, std::deque<RestingOrder>, std::greater<int>>;
    using AskMap = std::map<int, std::deque<RestingOrder>>;

    int execute_buy(int quantity,
                    std::int64_t timestamp_ns,
                    int limit_price_ticks,
                    bool has_limit,
                    std::int32_t aggressor_owner,
                    std::uint64_t aggressor_order_id,
                    OrderAction aggressor_action);
    int execute_sell(int quantity,
                     std::int64_t timestamp_ns,
                     int limit_price_ticks,
                     bool has_limit,
                     std::int32_t aggressor_owner,
                     std::uint64_t aggressor_order_id,
                     OrderAction aggressor_action);
    ApplyResult add_limit(const OrderMessage& message);
    ApplyResult submit_market(const OrderMessage& message);
    int cancel_owner(std::int32_t owner_id);
    int cancel_at_distance(const OrderMessage& message);
    void record_fill(std::int32_t owner_id,
                     std::uint64_t order_sequence,
                     OrderAction action,
                     Side side,
                     int quantity,
                     int price_ticks,
                     std::int64_t timestamp_ns);
    void record_order_result(const OrderMessage& message, const ApplyResult& result);

    int tick_size_ = 1;
    BidMap bids_;
    AskMap asks_;
    std::vector<AgentReport> reports_;
    std::uint64_t cumulative_aggressive_buy_ = 0;
    std::uint64_t cumulative_aggressive_sell_ = 0;
    int last_trade_price_ticks_ = 0;
};

} // namespace dlob
