#pragma once

#include "common/DistributedTypes.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace dlob {

struct SharedMarketMakerHedgeRoute {
    BookId book_id = 0;
    double weight = 1.0;
};

// A book's beta converts one unit of inventory into units of the common risk
// factor.  hedge_book_id remains the one-route compatibility field; when
// hedge_routes is non-empty, risk is allocated across every listed destination
// by normalized positive weights.
struct SharedMarketMakerBookConfig {
    BookId book_id = 0;
    double beta = 1.0;
    BookId hedge_book_id = 0;
    // Zero quantity inherits the global quantity.  The spread is measured in
    // minimum price ticks and lets empirically different books retain their
    // characteristic inside spread.
    std::int32_t quote_quantity = 0;
    std::int32_t target_spread_ticks = 1;
    std::vector<SharedMarketMakerHedgeRoute> hedge_routes;
};

struct SharedMarketMakerConfig {
    // This is a logical simulation identifier.  It must not be derived from an
    // MPI rank, so the same agent owns its orders under every partitioning.
    std::int32_t logical_owner_id = 900'001;
    std::int32_t message_source_rank = 0;

    std::int32_t quote_quantity = 100;
    std::int32_t quote_levels = 3;
    // Deeper levels carry more size so rare large market orders see a depth
    // curve instead of an unbounded gap.  Level k uses quantity*growth^k,
    // capped by max_quote_quantity_per_level.
    std::int32_t quote_quantity_growth = 3;
    std::int32_t max_quote_quantity_per_level = 10'000;
    // Prices in this code base use fixed-point price units (normally 100 units
    // per cent for ITCH prices).  The half-spread uses those same units.
    std::int32_t quote_half_spread_ticks = 100;
    std::int32_t price_tick_size = 100;
    std::int64_t order_latency_ns = 5'000;

    double exposure_threshold = 100.0;
    bool enable_cross_book_hedging = true;
    std::int32_t hedge_lot_size = 1;
    std::int32_t max_hedge_quantity = 100'000;
    std::int64_t report_latency_ns = 1'000;
    std::int64_t reaction_latency_ns = 1'000;
    std::int64_t network_latency_ns = 5'000;

    std::vector<SharedMarketMakerBookConfig> books;
};

// Deterministic shared-capital market maker used by the sequential reference
// model and, unchanged, by an MPI event engine.  It never touches a LOB
// directly: callers enqueue the returned OrderMessages in their event queue.
class SharedMarketMakerAgent {
public:
    explicit SharedMarketMakerAgent(SharedMarketMakerConfig config);

    // Cancel-and-replace one passive bid and ask in the selected book.  The
    // Returned orders are CancelOwner followed by bid/ask limits at each level.
    std::vector<OrderMessage> make_quotes(BookId book_id,
                                          const MarketState& state,
                                          std::int64_t decision_time_ns);

    // Consume exactly one canonical execution.  Trades not involving this
    // logical owner are ignored.  An own fill can create one weighted batch of
    // market hedges in other books.  One-book mode only accounts for the fill
    // and always returns an empty vector.
    std::vector<OrderMessage> on_trade(
        const TradeExecution& execution,
        bool allow_cross_book_reaction = true);

    // Release any still-projected quantity after the exchange reports that a
    // hedge order is terminal (including partial and zero-fill market orders).
    // Returns true only when sequence identified an outstanding hedge.
    bool complete_order(std::uint64_t sequence);

    std::int32_t logical_owner_id() const { return config_.logical_owner_id; }
    std::int64_t inventory(BookId book_id) const;
    std::int64_t cash_ticks(BookId book_id) const;
    std::int64_t total_cash_ticks() const;
    double beta_exposure() const;
    double projected_beta_exposure() const;

private:
    struct PendingHedge {
        BookId book_id = 0;
        // Positive is an expected buy; negative is an expected sell.
        std::int64_t signed_quantity = 0;
    };

    const SharedMarketMakerBookConfig* find_book(BookId book_id) const;
    OrderMessage make_message(BookId book_id,
                              OrderAction action,
                              Side side,
                              std::int32_t quantity,
                              std::int32_t price_ticks,
                              std::int64_t generated_time_ns,
                              std::int64_t arrival_time_ns);
    void apply_own_fill(BookId book_id,
                        Side side,
                        std::int32_t quantity,
                        std::int32_t price_ticks,
                        std::uint64_t order_sequence);
    void reconcile_pending(std::uint64_t order_sequence,
                           std::int64_t signed_fill_quantity);
    std::int64_t projected_inventory(BookId book_id) const;
    std::uint64_t next_order_sequence();

    SharedMarketMakerConfig config_;
    std::map<BookId, std::int64_t> inventory_by_book_;
    std::map<BookId, std::int64_t> cash_by_book_;
    std::map<std::uint64_t, PendingHedge> pending_hedges_;
    std::uint64_t next_local_sequence_ = 1;
};

} // namespace dlob
