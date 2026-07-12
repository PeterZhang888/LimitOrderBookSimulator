#pragma once

#include <cstdint>

// Six event types used by the Hawkes background agent.
enum class EventType : int {
    LimitBuy = 0,
    LimitSell = 1,
    MarketBuy = 2,
    MarketSell = 3,
    CancelBid = 4,
    CancelAsk = 5
};

constexpr int NUM_EVENT_TYPES = 6;

inline const char* event_type_to_string(EventType type) {
    switch (type) {
        case EventType::LimitBuy: return "limit_buy";
        case EventType::LimitSell: return "limit_sell";
        case EventType::MarketBuy: return "market_buy";
        case EventType::MarketSell: return "market_sell";
        case EventType::CancelBid: return "cancel_bid";
        case EventType::CancelAsk: return "cancel_ask";
        default: return "unknown";
    }
}

enum class Side : int {
    Buy = 1,
    Sell = -1
};

enum class OrderType : int {
    Limit = 0,
    Market = 1
};

struct Order {
    std::uint64_t order_id = 0;
    std::int64_t timestamp_ns = 0;
    OrderType type = OrderType::Limit;
    Side side = Side::Buy;
    int quantity = 0;
    int price_ticks = 0;
    int owner_id = 0; // zero is background/non-strategic ownership.
};

struct ExecutionReport {
    std::uint64_t order_id = 0;
    Side side = Side::Buy;       // side of the resting owner that received the fill
    int quantity = 0;
    int price_ticks = 0;
    std::int64_t timestamp_ns = 0;
};

struct HawkesEvent {
    std::int64_t time_ns = 0;
    EventType type = EventType::LimitBuy;
};
