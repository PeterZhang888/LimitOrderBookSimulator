#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

namespace dlob {

// Zero is the legacy/default single-book identifier.  Explicit identifiers are
// used by the multi-asset simulator without changing single-book callers.
using BookId = std::uint32_t;

// The reduced background model retains the contemporaneous BBO and nine
// further price levels on each side.  ITCH marks outside this moving band are
// still generated and counted but do not mutate the represented book.
inline constexpr int reduced_background_depth_levels = 10;

enum class Side : std::int32_t { Buy = 1, Sell = -1 };

enum class OrderAction : std::int32_t {
    Limit = 1,
    Market = 2,
    CancelOwner = 3,
    CancelAtDistance = 4,
    // Fragmented-calibration action that revises the location of anonymous
    // aggregate ITCH liquidity without creating displayed quantity.  It is
    // intentionally distinct from the additive shared market maker used in
    // the systemic-liquidity experiment.
    ConservedLimit = 5
};

enum class AgentKind : std::int32_t {
    Background = 0,
    MarketMaker = 1,
    Momentum = 2,
    Institutional = 3,
    Informed = 4,
    Arbitrage = 5,
    Value = 6
};

enum class WorkerRole : std::int32_t {
    Exchange = 0,
    MarketMaker = 1,
    Momentum = 2,
    Informed = 3,
    Institutional = 4,
    Mixed = 5,
    AllLocal = 6
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

inline constexpr std::int64_t no_wake_time = std::numeric_limits<std::int64_t>::max();

struct MarketState {
    std::int64_t exchange_time_ns = 0;
    std::int32_t best_bid_ticks = 0;
    std::int32_t best_ask_ticks = 0;
    std::int32_t best_bid_depth = 0;
    std::int32_t best_ask_depth = 0;
    // Anonymous owner-zero depth at owner zero's own leading represented
    // price level.  This may be one or more ticks behind the market BBO when
    // a strategic maker improves the quote.  Queue-reactive background
    // cancellation must not respond to liquidity supplied by an experimental
    // agent or collapse to zero merely because that agent leads the market.
    std::int32_t background_best_bid_depth = 0;
    std::int32_t background_best_ask_depth = 0;
    // Full owner-zero side quantities are future-relevant state for the
    // queue-reactive anonymous cancellation rule.  The separate same-side
    // reflecting boundary protects the final *displayed* share, irrespective
    // of owner; independently supplied maker liquidity may therefore replace
    // the final anonymous share without being relabelled as empirical flow.
    std::int64_t total_background_bid_depth = 0;
    std::int64_t total_background_ask_depth = 0;
    std::int32_t last_trade_price_ticks = 0;
    double mid_price_ticks = 0.0;
    double fundamental_value_ticks = 0.0;
    std::uint64_t cumulative_aggressive_buy = 0;
    std::uint64_t cumulative_aggressive_sell = 0;
    BookId book_id = 0;

    friend constexpr bool operator==(const MarketState&, const MarketState&) = default;
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
    BookId book_id = 0;
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
    BookId book_id = 0;
};

// Canonical, fixed-width representation of one price-time-priority match.  A
// single incoming order can produce several executions, one for each resting
// order it matches.  Buyer/seller fields make the record independent of which
// party happened to be the aggressor, which is useful to cross-asset agents.
struct TradeExecution {
    BookId book_id = 0;
    std::int64_t timestamp_ns = 0;
    std::uint64_t trade_sequence = 0;
    std::int32_t price_ticks = 0;
    std::int32_t quantity = 0;
    std::int32_t buyer_owner_id = 0;
    std::int32_t seller_owner_id = 0;
    std::uint64_t buyer_order_sequence = 0;
    std::uint64_t seller_order_sequence = 0;
    Side aggressor_side = Side::Buy;
    OrderAction aggressor_action = OrderAction::Market;
};

struct HawkesEvent {
    std::int64_t time_ns = 0;
    HawkesEventType type = HawkesEventType::LimitBuy;
};

struct SharedMarketSnapshotSlot {
    std::uint64_t version = 0;
    MarketState state{};
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

inline const char* worker_role_name(WorkerRole role) {
    switch (role) {
        case WorkerRole::Exchange: return "exchange";
        case WorkerRole::MarketMaker: return "market_maker";
        case WorkerRole::Momentum: return "momentum";
        case WorkerRole::Informed: return "informed";
        case WorkerRole::Institutional: return "institutional";
        case WorkerRole::Mixed: return "mixed";
        case WorkerRole::AllLocal: return "all_local";
    }
    return "unknown";
}

static_assert(std::is_trivially_copyable_v<MarketState>);
static_assert(std::is_trivially_copyable_v<OrderMessage>);
static_assert(std::is_trivially_copyable_v<AgentReport>);
static_assert(std::is_trivially_copyable_v<TradeExecution>);
static_assert(std::is_trivially_copyable_v<SharedMarketSnapshotSlot>);

} // namespace dlob
