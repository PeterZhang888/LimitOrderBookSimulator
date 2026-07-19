#pragma once

#include <cstdint>
#include <type_traits>

namespace dlob {

enum class Side : std::int32_t { Buy = 1, Sell = -1 };

enum class OrderAction : std::int32_t {
    Limit = 1,
    Market = 2,
    CancelOwner = 3,
    CancelAtDistance = 4
};

enum class AgentKind : std::int32_t {
    Background = 0,
    MarketMaker = 1,
    Momentum = 2,
    Institutional = 3,
    Informed = 4
};

enum class HawkesEventType : std::int32_t {
    LimitBuy = 0,
    LimitSell = 1,
    MarketBuy = 2,
    MarketSell = 3,
    CancelBid = 4,
    CancelAsk = 5
};

enum class ReportKind : std::int32_t {
    Fill = 1,
    OrderResult = 2
};

struct MarketState {
    std::int64_t exchange_time_ns = 0;
    std::int32_t best_bid_ticks = 0;
    std::int32_t best_ask_ticks = 0;
    std::int32_t best_bid_depth = 0;
    std::int32_t best_ask_depth = 0;
    std::int32_t last_trade_price_ticks = 0;
    double mid_price_ticks = 0.0;
    double fundamental_value_ticks = 0.0;
    std::uint64_t cumulative_aggressive_buy = 0;
    std::uint64_t cumulative_aggressive_sell = 0;
};

struct OrderMessage {
    std::int64_t generated_time_ns = 0;
    std::int64_t arrival_time_ns = 0;
    std::uint64_t sequence = 0;
    std::uint64_t tie_breaker = 0;
    std::int32_t source_rank = 0;
    std::int32_t owner_id = 0;
    AgentKind agent_kind = AgentKind::Background;
    OrderAction action = OrderAction::Limit;
    Side side = Side::Buy;
    std::int32_t quantity = 0;
    std::int32_t price_ticks = 0;
    std::int32_t distance_ticks = 0;
};

// One report type carries both fill notifications and terminal results for an
// incoming message. The latter releases outstanding institutional quantity even
// when a market order is only partially filled.
struct AgentReport {
    std::int64_t timestamp_ns = 0;
    std::int32_t owner_id = 0;
    std::uint64_t order_sequence = 0;
    ReportKind kind = ReportKind::OrderResult;
    OrderAction action = OrderAction::Limit;
    Side side = Side::Buy;
    std::int32_t requested_quantity = 0;
    std::int32_t executed_quantity = 0;
    std::int32_t resting_quantity = 0;
    std::int32_t cancelled_quantity = 0;
    std::int32_t fill_quantity = 0;
    std::int32_t fill_price_ticks = 0;
};

struct HawkesEvent {
    std::int64_t time_ns = 0;
    HawkesEventType type = HawkesEventType::LimitBuy;
};

inline int owner_rank(std::int32_t owner_id) {
    return owner_id > 0 ? owner_id / 1'000'000 : 0;
}

inline int owner_local_index(std::int32_t owner_id) {
    return owner_id > 0 ? owner_id % 1'000'000 - 1 : -1;
}

inline std::int32_t make_owner_id(int rank, int local_index) {
    return rank * 1'000'000 + local_index + 1;
}

static_assert(std::is_trivially_copyable_v<MarketState>);
static_assert(std::is_trivially_copyable_v<OrderMessage>);
static_assert(std::is_trivially_copyable_v<AgentReport>);

} // namespace dlob
