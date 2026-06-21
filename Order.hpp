#ifndef ORDER_HPP
#define ORDER_HPP

#include <cstdint>

enum class OrderType {
    Limit,
    Market
};

enum class Side {
    Buy,
    Sell
};

struct Order {
    std::uint64_t id = 0;
    std::int64_t timestamp_ns = 0;

    OrderType type = OrderType::Limit;
    Side side = Side::Buy;

    int quantity = 0;
    int price_ticks = 0;

    // -1 means background market flow or untracked order.
    // Non-negative values identify agent-owned orders.
    int owner_id = -1;
};

struct ExecutionReport {
    std::int64_t timestamp_ns = 0;

    // Owner of the executed order. Reports are created only
    // for orders with owner_id >= 0.
    int owner_id = -1;

    // Side of the owner-owned order that was executed.
    // If side == Buy, the owner bought and inventory increases.
    // If side == Sell, the owner sold and inventory decreases.
    Side side = Side::Buy;

    int quantity = 0;
    int price_ticks = 0;

    // True when the owner's resting order was hit.
    // False when the owner's incoming marketable order executed.
    bool liquidity_provider = true;
};

#endif

