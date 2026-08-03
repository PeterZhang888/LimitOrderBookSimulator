#pragma once

#include "common/DistributedTypes.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <vector>

namespace dlob {

struct ApplyResult {
    int requested_quantity = 0;
    int executed_quantity = 0;
    int resting_quantity = 0;
    int cancelled_quantity = 0;
    int boundary_truncated_quantity = 0;
};

// Aggregated resting quantity owned by one participant at one price.  The
// simulator uses this read-only view to avoid cancelling and recreating an
// unchanged market-maker quote.  Preserving an unchanged order is essential
// in a price--time-priority book: a gratuitous refresh would otherwise send
// the participant to the back of the queue every decision window.
struct OwnerRestingQuote {
    Side side = Side::Buy;
    int price_ticks = 0;
    std::int64_t quantity = 0;
};

class DistributedLimitOrderBook {
public:
    explicit DistributedLimitOrderBook(int tick_size, BookId book_id = 0);

    void seed_default_book(double depth_scale = 1.0);
    void seed_calibrated_book(int best_bid_ticks,
                              int best_ask_ticks,
                              int best_bid_depth,
                              int best_ask_depth,
                              double depth_scale = 1.0);
    ApplyResult apply(const OrderMessage& message);

    MarketState state(std::int64_t time_ns, double fundamental_value_ticks) const;
    std::vector<AgentReport> take_reports();
    std::vector<TradeExecution> take_trades();

    BookId book_id() const noexcept { return book_id_; }

    bool has_bid() const;
    bool has_ask() const;
    int best_bid() const;
    int best_ask() const;
    int best_bid_depth() const;
    int best_ask_depth() const;
    int background_best_bid_depth() const;
    int background_best_ask_depth() const;
    std::int64_t total_bid_depth() const;
    std::int64_t total_ask_depth() const;
    std::int64_t total_background_bid_depth() const;
    std::int64_t total_background_ask_depth() const;
    std::int64_t owner_resting_depth(std::int32_t owner_id, Side side) const;
    std::vector<OwnerRestingQuote> owner_resting_quotes(
        std::int32_t owner_id) const;
    double mid_price() const;

private:
    struct BackgroundWithdrawal {
        int quantity = 0;
        int donor_price_ticks = 0;
    };

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
                    OrderAction aggressor_action,
                    bool preserve_background_reserve = false,
                    int* boundary_truncated_quantity = nullptr);
    int execute_sell(int quantity,
                     std::int64_t timestamp_ns,
                     int limit_price_ticks,
                     bool has_limit,
                     std::int32_t aggressor_owner,
                     std::uint64_t aggressor_order_id,
                     OrderAction aggressor_action,
                     bool preserve_background_reserve = false,
                     int* boundary_truncated_quantity = nullptr);
    ApplyResult add_limit(const OrderMessage& message);
    ApplyResult add_conserved_limit(const OrderMessage& message);
    ApplyResult submit_market(const OrderMessage& message);
    int cancel_owner(std::int32_t owner_id);
    std::optional<int> nearest_background_price(Side side) const;
    BackgroundWithdrawal withdraw_background_nearest(Side side, int quantity);
    int cancel_at_distance(const OrderMessage& message,
                           int* boundary_truncated_quantity = nullptr);
    void record_fill(std::int32_t owner_id,
                     std::uint64_t order_sequence,
                     OrderAction action,
                     Side side,
                     int quantity,
                     int price_ticks,
                     std::int64_t timestamp_ns);
    void record_order_result(const OrderMessage& message, const ApplyResult& result);
    void record_trade(std::int64_t timestamp_ns,
                      int price_ticks,
                      int quantity,
                      std::int32_t buyer_owner_id,
                      std::int32_t seller_owner_id,
                      std::uint64_t buyer_order_sequence,
                      std::uint64_t seller_order_sequence,
                      Side aggressor_side,
                      OrderAction aggressor_action);

    int tick_size_ = 1;
    BookId book_id_ = 0;
    BidMap bids_;
    AskMap asks_;
    std::map<std::int32_t, std::set<int>> owner_bid_prices_;
    std::map<std::int32_t, std::set<int>> owner_ask_prices_;
    std::vector<AgentReport> reports_;
    std::vector<TradeExecution> trades_;
    std::uint64_t next_trade_sequence_ = 1;
    std::uint64_t cumulative_aggressive_buy_ = 0;
    std::uint64_t cumulative_aggressive_sell_ = 0;
    std::int64_t total_bid_quantity_ = 0;
    std::int64_t total_ask_quantity_ = 0;
    std::int64_t background_bid_quantity_ = 0;
    std::int64_t background_ask_quantity_ = 0;
    int last_trade_price_ticks_ = 0;
};

} // namespace dlob
