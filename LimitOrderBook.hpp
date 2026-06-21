#ifndef LIMIT_ORDER_BOOK_HPP
#define LIMIT_ORDER_BOOK_HPP

#include "Order.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class LimitOrderBook {
public:
    LimitOrderBook() = default;

    // Add a limit order. If the order crosses the opposite
    // best price, it executes immediately as a marketable
    // limit order. Any remaining quantity rests in the book.
    void add_limit_order(
        const Order& order
    );

    // Submit an aggressive market order.
    //
    // Side::Buy consumes the ask book.
    // Side::Sell consumes the bid book.
    //
    // Returns the total executed quantity.
    int submit_market_order(
        Side side,
        int quantity,
        std::int64_t timestamp_ns = 0
    );

    void cancel_from_side(
        Side side,
        int quantity
    );

    void cancel_at_distance(
        Side side,
        int distance_ticks,
        int quantity,
        int tick_size
    );

    bool cancel_order(
        std::uint64_t order_id
    );

    std::size_t cancel_orders_by_owner(
        int owner_id
    );

    // ========================================================
    // Execution-report access
    // ========================================================

    const std::vector<ExecutionReport>&
    execution_reports() const;

    void clear_execution_reports();

    std::vector<ExecutionReport>
    get_and_clear_execution_reports();

    std::vector<ExecutionReport>
    get_and_clear_execution_reports(
        int owner_id
    );

    // ========================================================
    // Book state
    // ========================================================

    bool has_bid() const;
    bool has_ask() const;

    int best_bid() const;
    int best_ask() const;

    int spread() const;
    double mid_price() const;

    int quantity_at_best_bid() const;
    int quantity_at_best_ask() const;

    // ========================================================
    // Aggressive order-flow counters
    // ========================================================

    // Total quantity executed by incoming aggressive buys:
    // market buys and marketable buy limit orders.
    std::uint64_t
    cumulative_aggressive_buy_quantity() const;

    // Total quantity executed by incoming aggressive sells:
    // market sells and marketable sell limit orders.
    std::uint64_t
    cumulative_aggressive_sell_quantity() const;

    void print_book() const;

private:
    using OrderQueue = std::list<Order>;

    using BidBook = std::map<
        int,
        OrderQueue,
        std::greater<int>
    >;

    using AskBook = std::map<
        int,
        OrderQueue,
        std::less<int>
    >;

    struct OrderLocation {
        Side side = Side::Buy;
        int price_ticks = 0;
        OrderQueue::iterator iterator;
    };

    BidBook bids_;
    AskBook asks_;

    // Order ID -> position in the book.
    std::unordered_map<
        std::uint64_t,
        OrderLocation
    > order_index_;

    // Owner ID -> all active order IDs belonging to the owner.
    std::unordered_map<
        int,
        std::unordered_set<std::uint64_t>
    > owner_order_ids_;

    std::vector<ExecutionReport>
        execution_reports_;

    std::uint64_t
        cumulative_aggressive_buy_quantity_ = 0;

    std::uint64_t
        cumulative_aggressive_sell_quantity_ = 0;

    void remove_order_from_tracking(
        const Order& order
    );

    void record_execution_report(
        std::int64_t timestamp_ns,
        const Order& order,
        int executed_quantity,
        int execution_price_ticks,
        bool liquidity_provider
    );

    void record_aggressive_execution(
        Side aggressive_side,
        int executed_quantity
    );

    int cancel_at_price(
        Side side,
        int price_ticks,
        int quantity
    );

    static int queue_quantity(
        const OrderQueue& queue
    );
};

#endif
