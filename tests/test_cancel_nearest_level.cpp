#include "exchange/DistributedLimitOrderBook.hpp"

#include <cassert>
#include <iostream>

namespace {

dlob::OrderMessage limit(dlob::BookId book,
                         dlob::Side side,
                         int price,
                         int quantity,
                         std::uint64_t sequence,
                         std::int32_t owner_id = 1) {
    dlob::OrderMessage message;
    message.book_id = book;
    message.arrival_time_ns = static_cast<std::int64_t>(sequence);
    message.sequence = sequence;
    message.owner_id = owner_id;
    message.action = dlob::OrderAction::Limit;
    message.side = side;
    message.price_ticks = price;
    message.quantity = quantity;
    return message;
}

dlob::OrderMessage cancel(dlob::BookId book,
                          dlob::Side side,
                          int distance,
                          int quantity,
                          std::uint64_t sequence,
                          std::int32_t owner_id = 1,
                          dlob::AgentKind kind = dlob::AgentKind::MarketMaker) {
    dlob::OrderMessage message;
    message.book_id = book;
    message.arrival_time_ns = static_cast<std::int64_t>(sequence);
    message.sequence = sequence;
    message.owner_id = owner_id;
    message.agent_kind = kind;
    message.action = dlob::OrderAction::CancelAtDistance;
    message.side = side;
    message.distance_ticks = distance;
    message.quantity = quantity;
    return message;
}

dlob::OrderMessage market(dlob::BookId book,
                          dlob::Side side,
                          int quantity,
                          std::uint64_t sequence) {
    dlob::OrderMessage message;
    message.book_id = book;
    message.arrival_time_ns = static_cast<std::int64_t>(sequence);
    message.sequence = sequence;
    message.action = dlob::OrderAction::Market;
    message.side = side;
    message.quantity = quantity;
    return message;
}

} // namespace

int main() {
    using namespace dlob;
    constexpr BookId id = 3;
    DistributedLimitOrderBook book(100, id);

    book.apply(limit(id, Side::Buy, 1'000, 10, 1));
    book.apply(limit(id, Side::Buy, 900, 10, 2));
    const ApplyResult bid_cancel = book.apply(cancel(id, Side::Buy, 5, 10, 3));
    // The sampled target is 500, below the represented owner support
    // [900,1000].  A reduced book must not project this tail mark onto 900.
    assert(bid_cancel.cancelled_quantity == 0);
    // A background execution represents one ITCH E/C message and therefore
    // consumes at most one price level.  The second message leaves the final
    // displayed share, independent of that share's owner identity.
    assert(book.apply(market(id, Side::Sell, 20, 4)).executed_quantity == 10);
    assert(book.apply(market(id, Side::Sell, 10, 40)).executed_quantity == 9);

    book.apply(limit(id, Side::Sell, 1'100, 10, 5));
    book.apply(limit(id, Side::Sell, 1'200, 10, 6));
    const ApplyResult ask_cancel = book.apply(cancel(id, Side::Sell, 5, 10, 7));
    // Likewise, target 1600 lies above represented ask support [1100,1200].
    assert(ask_cancel.cancelled_quantity == 0);
    assert(book.apply(market(id, Side::Buy, 20, 8)).executed_quantity == 10);
    assert(book.apply(market(id, Side::Buy, 10, 80)).executed_quantity == 9);

    // Ownership filtering is part of modeled support.  Owner 1 has support
    // only at 900, so a target at 1000 is not projected through owner 2's
    // quote onto owner 1's furthest represented order.
    DistributedLimitOrderBook owner_filtered(100, id);
    owner_filtered.apply(limit(id, Side::Buy, 1'000, 10, 20, 2));
    owner_filtered.apply(limit(id, Side::Buy, 900, 10, 21, 1));
    const ApplyResult filtered_cancel = owner_filtered.apply(
        cancel(id, Side::Buy, 0, 10, 22));
    assert(filtered_cancel.cancelled_quantity == 0);
    assert(owner_filtered.apply(market(id, Side::Sell, 20, 23)).executed_quantity == 10);
    assert(owner_filtered.apply(market(id, Side::Sell, 10, 24)).executed_quantity == 9);

    // Anonymous ITCH cancellations retain a distance mark but no source order
    // reference.  Remove the represented level at distance five, then prove
    // that the same retained mark is mapped to the nearest surviving owner-zero
    // level.  Levels four and six are equidistant; the deterministic lower-price
    // tie break selects level six, whose seeded quantity is 75.
    DistributedLimitOrderBook background_nearest(100, id);
    background_nearest.seed_calibrated_book(10'000, 10'200, 100, 100, 1.0);
    const std::int64_t initial_background_bid =
        background_nearest.total_background_bid_depth();
    const ApplyResult exact_background_cancel = background_nearest.apply(
        cancel(id, Side::Buy, 5, 100, 30, 0, AgentKind::Background));
    assert(exact_background_cancel.cancelled_quantity == 100);
    assert(exact_background_cancel.boundary_truncated_quantity == 0);
    const std::int64_t after_exact_cancel =
        background_nearest.total_background_bid_depth();
    assert(after_exact_cancel == initial_background_bid - 100);

    const ApplyResult missing_background_cancel = background_nearest.apply(
        cancel(id, Side::Buy, 5, 100, 31, 0, AgentKind::Background));
    assert(missing_background_cancel.cancelled_quantity == 75);
    assert(missing_background_cancel.boundary_truncated_quantity == 0);
    assert(background_nearest.total_background_bid_depth()
           == after_exact_cancel - 75);

    // The final-share guard remains active when a retained mark is projected
    // onto the only surviving anonymous level, not only for an exact hit.
    DistributedLimitOrderBook final_share(100, id);
    final_share.seed_calibrated_book(10'000, 10'200, 1, 1, 1.0);
    std::uint64_t drain_sequence = 40;
    while (final_share.total_background_bid_depth() > 1) {
        const int depth = final_share.background_best_bid_depth();
        assert(depth > 0);
        assert(final_share.apply(
            market(id, Side::Sell, depth, drain_sequence++)).executed_quantity
            > 0);
    }
    const ApplyResult reflected = final_share.apply(
        cancel(id, Side::Buy, 1, 1, drain_sequence++,
               0, AgentKind::Background));
    assert(reflected.cancelled_quantity == 0);
    assert(reflected.boundary_truncated_quantity == 1);

    // Marks outside the declared ten-level state are counted upstream but do
    // not mutate the represented book and cannot activate its boundary.
    const ApplyResult tail = final_share.apply(
        cancel(id, Side::Buy, reduced_background_depth_levels, 1,
               drain_sequence++,
               0, AgentKind::Background));
    assert(tail.cancelled_quantity == 0);
    assert(tail.boundary_truncated_quantity == 0);

    std::cout << "nearest reduced-level cancellation tests passed\n";
    return 0;
}
