#include "exchange/LimitOrderBook.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>

int main() {
    using namespace dlob;

    LimitOrderBook book(100);
    book.seed_default_book(1.0);
    const int ask_before = book.best_ask_depth();

    OrderMessage informed;
    informed.arrival_time_ns = 1000;
    informed.generated_time_ns = 900;
    informed.sequence = 1;
    informed.owner_id = make_owner_id(1, 0);
    informed.source_rank = 1;
    informed.agent_kind = AgentKind::Informed;
    informed.action = OrderAction::Market;
    informed.side = Side::Buy;
    informed.quantity = 100;

    const ApplyResult informed_result = book.apply(informed);
    assert(informed_result.executed_quantity == 100);
    assert(book.best_ask_depth() == ask_before - 100);
    auto reports = book.take_reports();
    assert(reports.size() == 1);
    assert(reports[0].kind == ReportKind::Fill);
    assert(reports[0].fill_quantity == 100);
    assert(reports[0].side == Side::Buy);

    OrderMessage institution = informed;
    institution.sequence = 2;
    institution.owner_id = make_owner_id(2, 0);
    institution.source_rank = 2;
    institution.agent_kind = AgentKind::Institutional;
    institution.quantity = 100000;
    const ApplyResult institutional_result = book.apply(institution);
    assert(institutional_result.requested_quantity == 100000);
    assert(institutional_result.executed_quantity > 0);
    reports = book.take_reports();
    const bool has_fill = std::any_of(reports.begin(), reports.end(), [](const AgentReport& report) {
        return report.kind == ReportKind::Fill;
    });
    const bool has_result = std::any_of(reports.begin(), reports.end(), [](const AgentReport& report) {
        return report.kind == ReportKind::OrderResult && report.requested_quantity == 100000;
    });
    assert(has_fill);
    assert(has_result);

    LimitOrderBook cancel_book(100);
    cancel_book.seed_default_book(1.0);
    OrderMessage quote;
    quote.arrival_time_ns = 2000;
    quote.generated_time_ns = 1900;
    quote.sequence = 3;
    quote.owner_id = make_owner_id(3, 0);
    quote.source_rank = 3;
    quote.agent_kind = AgentKind::MarketMaker;
    quote.action = OrderAction::Limit;
    quote.side = Side::Buy;
    quote.quantity = 50;
    quote.price_ticks = cancel_book.best_bid();
    cancel_book.apply(quote);
    const int depth_with_quote = cancel_book.best_bid_depth();

    OrderMessage cancel = quote;
    cancel.sequence = 4;
    cancel.action = OrderAction::CancelOwner;
    cancel.quantity = 0;
    const ApplyResult cancel_result = cancel_book.apply(cancel);
    assert(cancel_result.cancelled_quantity == 50);
    assert(cancel_book.best_bid_depth() == depth_with_quote - 50);

    // An ITCH-derived background execution represents one E/C message and may
    // consume only the contemporaneous best price level.  A strategic market
    // order remains able to walk through multiple levels.
    LimitOrderBook background_book(100);
    background_book.seed_calibrated_book(10'000, 10'200, 10, 10, 1.0);
    OrderMessage second_background_ask;
    second_background_ask.arrival_time_ns = 2'900;
    second_background_ask.generated_time_ns = 2'900;
    second_background_ask.sequence = 50;
    second_background_ask.owner_id = 0;
    second_background_ask.agent_kind = AgentKind::Background;
    second_background_ask.action = OrderAction::Limit;
    second_background_ask.side = Side::Sell;
    second_background_ask.quantity = 7;
    second_background_ask.price_ticks = 10'200;
    background_book.apply(second_background_ask);
    OrderMessage background_market;
    background_market.arrival_time_ns = 3000;
    background_market.generated_time_ns = 3000;
    background_market.sequence = 5;
    background_market.agent_kind = AgentKind::Background;
    background_market.action = OrderAction::Market;
    background_market.side = Side::Buy;
    background_market.quantity = 15;
    const ApplyResult background_result = background_book.apply(background_market);
    assert(background_result.executed_quantity == 15);
    assert(background_book.best_ask() == 10'200);
    assert(background_book.best_ask_depth() == 2);

    LimitOrderBook background_cancel_book(100);
    background_cancel_book.seed_calibrated_book(10'000, 10'200, 10, 10, 1.0);
    OrderMessage second_background_bid = second_background_ask;
    second_background_bid.sequence = 51;
    second_background_bid.side = Side::Buy;
    second_background_bid.quantity = 7;
    second_background_bid.price_ticks = 10'000;
    background_cancel_book.apply(second_background_bid);
    OrderMessage background_cancel = second_background_bid;
    background_cancel.sequence = 52;
    background_cancel.action = OrderAction::CancelAtDistance;
    background_cancel.quantity = 15;
    background_cancel.distance_ticks = 0;
    const ApplyResult background_cancel_result =
        background_cancel_book.apply(background_cancel);
    assert(background_cancel_result.cancelled_quantity == 15);
    // Although the request exceeds some selected-level queues, the global
    // reserve does not change the five/ten/etc. shares actually reachable at
    // that one level.  Such ordinary level exhaustion is not a boundary
    // truncation and must not contaminate its diagnostic counter.
    assert(background_cancel_result.boundary_truncated_quantity == 0);
    assert(background_cancel_book.best_bid_depth() == 2);
    background_cancel.distance_ticks = 20;
    const ApplyResult absent_cancel_result =
        background_cancel_book.apply(background_cancel);
    assert(absent_cancel_result.cancelled_quantity == 0);
    assert(background_cancel_book.best_bid_depth() == 2);

    LimitOrderBook selected_level_exhaustion_book(100);
    selected_level_exhaustion_book.seed_calibrated_book(
        10'000, 10'200, 5, 5, 1.0);
    OrderMessage oversized_level_cancel = background_cancel;
    oversized_level_cancel.sequence = 520;
    oversized_level_cancel.quantity = 100;
    oversized_level_cancel.distance_ticks = 0;
    const ApplyResult selected_level_exhaustion =
        selected_level_exhaustion_book.apply(oversized_level_cancel);
    assert(selected_level_exhaustion.cancelled_quantity == 5);
    assert(selected_level_exhaustion.boundary_truncated_quantity == 0);

    // A reduced finite-support book must treat unrepresented outer-tail adds
    // and cancels symmetrically.  An outside add is ignored, while an
    // inside-spread add remains a valid revision of represented liquidity.
    LimitOrderBook finite_support_book(100);
    finite_support_book.seed_calibrated_book(10'000, 10'500, 10, 10, 1.0);
    const std::int64_t finite_bid_total = finite_support_book.total_bid_depth();
    OrderMessage tail_add = second_background_bid;
    tail_add.sequence = 53;
    tail_add.action = OrderAction::Limit;
    tail_add.quantity = 7;
    tail_add.distance_ticks = reduced_background_depth_levels;
    tail_add.price_ticks = 8'000;
    const ApplyResult tail_add_result = finite_support_book.apply(tail_add);
    assert(tail_add_result.requested_quantity == 7);
    assert(tail_add_result.resting_quantity == 0);
    assert(finite_support_book.total_bid_depth() == finite_bid_total);
    tail_add.sequence += 1;
    tail_add.distance_ticks = reduced_background_depth_levels - 1;
    tail_add.price_ticks = 9'100;
    const ApplyResult retained_add_result = finite_support_book.apply(tail_add);
    assert(retained_add_result.resting_quantity == 7);
    assert(finite_support_book.best_bid() == 10'000);
    assert(finite_support_book.total_bid_depth() == finite_bid_total + 7);
    tail_add.sequence += 1;
    tail_add.distance_ticks = 0;
    tail_add.price_ticks = 10'100;
    const ApplyResult inside_add_result = finite_support_book.apply(tail_add);
    assert(inside_add_result.resting_quantity == 7);
    assert(finite_support_book.best_bid() == 10'100);
    assert(finite_support_book.total_bid_depth() == finite_bid_total + 14);

    LimitOrderBook value_book(100);
    value_book.seed_calibrated_book(10'000, 10'200, 10, 10, 1.0);
    OrderMessage value_market = background_market;
    value_market.sequence = 6;
    value_market.agent_kind = AgentKind::Value;
    value_market.quantity = 25;
    value_market.price_ticks = 10'300;
    const ApplyResult value_result = value_book.apply(value_market);
    assert(value_result.executed_quantity == 20);
    const std::vector<TradeExecution> value_buy_trades =
        value_book.take_trades();
    assert(value_buy_trades.size() == 2U);
    int value_buy_executed = 0;
    for (const TradeExecution& trade : value_buy_trades) {
        assert(trade.aggressor_side == Side::Buy);
        assert(trade.aggressor_action == OrderAction::Market);
        assert(trade.price_ticks <= value_market.price_ticks);
        value_buy_executed += trade.quantity;
    }
    assert(value_buy_executed == value_result.executed_quantity);
    assert(value_buy_trades[0].price_ticks == 10'200);
    assert(value_buy_trades[1].price_ticks == 10'300);
    assert(value_book.best_ask() == 10'400);
    assert(value_book.best_ask_depth() == 13);

    // The perceived-fundamental protection is symmetric: a value sell may
    // consume bids at or above its price floor, but never a cheaper level.
    LimitOrderBook value_sell_book(100);
    value_sell_book.seed_calibrated_book(10'000, 10'200, 10, 10, 1.0);
    OrderMessage value_sell_market = value_market;
    value_sell_market.sequence = 61;
    value_sell_market.side = Side::Sell;
    value_sell_market.price_ticks = 9'900;
    const ApplyResult value_sell_result =
        value_sell_book.apply(value_sell_market);
    assert(value_sell_result.executed_quantity == 20);
    const std::vector<TradeExecution> value_sell_trades =
        value_sell_book.take_trades();
    assert(value_sell_trades.size() == 2U);
    int value_sell_executed = 0;
    for (const TradeExecution& trade : value_sell_trades) {
        assert(trade.aggressor_side == Side::Sell);
        assert(trade.aggressor_action == OrderAction::Market);
        assert(trade.price_ticks >= value_sell_market.price_ticks);
        value_sell_executed += trade.quantity;
    }
    assert(value_sell_executed == value_sell_result.executed_quantity);
    assert(value_sell_trades[0].price_ticks == 10'000);
    assert(value_sell_trades[1].price_ticks == 9'900);
    assert(value_sell_book.best_bid() == 9'800);
    assert(value_sell_book.best_bid_depth() == 13);

    LimitOrderBook strategic_book(100);
    strategic_book.seed_calibrated_book(10'000, 10'200, 10, 10, 1.0);
    OrderMessage strategic_market = background_market;
    strategic_market.sequence = 7;
    strategic_market.agent_kind = AgentKind::Institutional;
    const ApplyResult strategic_result = strategic_book.apply(strategic_market);
    assert(strategic_result.executed_quantity == 15);
    assert(strategic_book.best_ask() == 10'300);
    assert(strategic_book.best_ask_depth() == 5);

    // The calibration-only local maker identifies and repositions part of
    // owner-zero ITCH flow; it does not add a second source of liquidity.
    // Releasing, moving and withdrawing that quote must preserve exact total
    // displayed quantity on every refresh.
    LimitOrderBook conserved_book(100);
    conserved_book.seed_calibrated_book(10'000, 10'500, 10, 12, 1.0);
    const std::int64_t original_bid_total = conserved_book.total_bid_depth();
    const std::int64_t original_ask_total = conserved_book.total_ask_depth();
    for (int refresh = 0; refresh < 32; ++refresh) {
        OrderMessage conserved_bid;
        conserved_bid.sequence = 100 + static_cast<std::uint64_t>(refresh) * 2U;
        conserved_bid.owner_id = 0;
        conserved_bid.agent_kind = AgentKind::Background;
        conserved_bid.action = OrderAction::ConservedLimit;
        conserved_bid.side = Side::Buy;
        conserved_bid.quantity = 7;
        conserved_bid.price_ticks = refresh % 2 == 0 ? 10'100 : 10'200;
        const ApplyResult bid_result = conserved_book.apply(conserved_bid);
        assert(bid_result.executed_quantity == 0);
        assert(bid_result.resting_quantity == 7);

        OrderMessage conserved_ask = conserved_bid;
        conserved_ask.sequence += 1;
        conserved_ask.side = Side::Sell;
        conserved_ask.quantity = 9;
        conserved_ask.price_ticks = refresh % 2 == 0 ? 10'400 : 10'300;
        const ApplyResult ask_result = conserved_book.apply(conserved_ask);
        assert(ask_result.executed_quantity == 0);
        assert(ask_result.resting_quantity == 9);
        assert(conserved_book.total_bid_depth() == original_bid_total);
        assert(conserved_book.total_ask_depth() == original_ask_total);
        assert(conserved_book.best_bid() < conserved_book.best_ask());
    }

    // A request larger than the represented background pool is filled only
    // from what already exists; no synthetic remainder may appear.
    OrderMessage oversized;
    oversized.sequence = 1000;
    oversized.action = OrderAction::ConservedLimit;
    oversized.side = Side::Buy;
    oversized.quantity = 1'000'000;
    oversized.price_ticks = 10'200;
    const ApplyResult oversized_result = conserved_book.apply(oversized);
    assert(oversized_result.resting_quantity > 0);
    assert(oversized_result.resting_quantity
           < oversized_result.requested_quantity);
    assert(conserved_book.total_bid_depth() == original_bid_total);
    assert(conserved_book.total_ask_depth() == original_ask_total);

    // ConservedLimit is an endogenous represented-book relocation.  A stale
    // or malformed incoming empirical distance must not filter the repost
    // after its donor has been withdrawn.
    OrderMessage tail_labeled_revision = oversized;
    tail_labeled_revision.sequence = 1001;
    tail_labeled_revision.quantity = 7;
    tail_labeled_revision.distance_ticks = reduced_background_depth_levels;
    const ApplyResult tail_labeled_result =
        conserved_book.apply(tail_labeled_revision);
    assert(tail_labeled_result.resting_quantity == 7);
    assert(conserved_book.total_bid_depth() == original_bid_total);
    assert(conserved_book.total_ask_depth() == original_ask_total);

    // Validate the non-crossing repost price before any withdrawal.  This
    // deliberately constructs a tick-size-100 book whose near-INT_MAX spread
    // is narrower than one tick, making the conserved sell target impossible.
    LimitOrderBook invalid_revision_book(100);
    OrderMessage extreme_bid;
    extreme_bid.sequence = 1100;
    extreme_bid.owner_id = 0;
    extreme_bid.agent_kind = AgentKind::Background;
    extreme_bid.action = OrderAction::Limit;
    extreme_bid.side = Side::Buy;
    extreme_bid.quantity = 10;
    extreme_bid.price_ticks = std::numeric_limits<int>::max() - 50;
    invalid_revision_book.apply(extreme_bid);
    OrderMessage extreme_ask = extreme_bid;
    extreme_ask.sequence += 1;
    extreme_ask.side = Side::Sell;
    extreme_ask.price_ticks = std::numeric_limits<int>::max();
    invalid_revision_book.apply(extreme_ask);
    const std::int64_t extreme_bid_total =
        invalid_revision_book.total_background_bid_depth();
    const std::int64_t extreme_ask_total =
        invalid_revision_book.total_background_ask_depth();
    OrderMessage invalid_revision = extreme_ask;
    invalid_revision.sequence += 1;
    invalid_revision.action = OrderAction::ConservedLimit;
    invalid_revision.quantity = 5;
    bool invalid_revision_rejected = false;
    try {
        invalid_revision_book.apply(invalid_revision);
    } catch (const std::logic_error&) {
        invalid_revision_rejected = true;
    }
    assert(invalid_revision_rejected);
    assert(invalid_revision_book.total_background_bid_depth()
           == extreme_bid_total);
    assert(invalid_revision_book.total_background_ask_depth()
           == extreme_ask_total);
    assert(invalid_revision_book.best_bid()
           == std::numeric_limits<int>::max() - 50);
    assert(invalid_revision_book.best_ask()
           == std::numeric_limits<int>::max());

    // A delayed revision observes a newer same-side BBO at arrival.  It may
    // retain or improve that BBO, but must never move the donor backwards to
    // the stale decision-time price.
    LimitOrderBook delayed_book(100);
    delayed_book.seed_calibrated_book(10'000, 10'500, 10, 12, 1.0);
    OrderMessage newer_bid;
    newer_bid.sequence = 2000;
    newer_bid.agent_kind = AgentKind::Background;
    newer_bid.action = OrderAction::Limit;
    newer_bid.side = Side::Buy;
    newer_bid.quantity = 5;
    newer_bid.price_ticks = 10'200;
    delayed_book.apply(newer_bid);
    OrderMessage newer_ask = newer_bid;
    newer_ask.sequence += 1;
    newer_ask.side = Side::Sell;
    newer_ask.price_ticks = 10'300;
    delayed_book.apply(newer_ask);
    const std::int64_t delayed_bid_total = delayed_book.total_bid_depth();
    const std::int64_t delayed_ask_total = delayed_book.total_ask_depth();

    OrderMessage stale_bid = newer_bid;
    stale_bid.sequence += 2;
    stale_bid.action = OrderAction::ConservedLimit;
    stale_bid.price_ticks = 10'100;
    delayed_book.apply(stale_bid);
    assert(delayed_book.best_bid() == 10'200);
    OrderMessage stale_ask = newer_ask;
    stale_ask.sequence += 2;
    stale_ask.action = OrderAction::ConservedLimit;
    stale_ask.price_ticks = 10'400;
    delayed_book.apply(stale_ask);
    assert(delayed_book.best_ask() == 10'300);
    assert(delayed_book.total_bid_depth() == delayed_bid_total);
    assert(delayed_book.total_ask_depth() == delayed_ask_total);

    LimitOrderBook no_donor_book(100);
    OrderMessage no_donor = stale_bid;
    no_donor.sequence = 3000;
    const ApplyResult no_donor_result = no_donor_book.apply(no_donor);
    assert(no_donor_result.requested_quantity == no_donor.quantity);
    assert(no_donor_result.resting_quantity == 0);
    assert(no_donor_book.total_bid_depth() == 0);
    assert(no_donor_book.total_ask_depth() == 0);

    // Background and stabilising value-flow removals use a same-side
    // reflecting boundary: they may not consume the final represented share.
    // This preserves exact two-sidedness without injecting new liquidity or
    // transferring quantity across the spread.  Every curtailed share is
    // returned explicitly in the ApplyResult for audit aggregation.
    LimitOrderBook bid_boundary_book(100);
    bid_boundary_book.seed_calibrated_book(10'000, 10'200, 2, 2, 1.0);
    OrderMessage drain_bid;
    drain_bid.sequence = 4000;
    drain_bid.owner_id = make_owner_id(4, 0);
    drain_bid.agent_kind = AgentKind::Institutional;
    drain_bid.action = OrderAction::Market;
    drain_bid.side = Side::Sell;
    drain_bid.quantity = static_cast<int>(
        bid_boundary_book.total_bid_depth() - 1);
    const ApplyResult drain_bid_result = bid_boundary_book.apply(drain_bid);
    assert(drain_bid_result.executed_quantity == drain_bid.quantity);
    assert(bid_boundary_book.total_bid_depth() == 1);

    OrderMessage reflected_market = drain_bid;
    reflected_market.sequence += 1;
    reflected_market.owner_id = 0;
    reflected_market.agent_kind = AgentKind::Background;
    reflected_market.quantity = 100;
    const ApplyResult reflected_market_result =
        bid_boundary_book.apply(reflected_market);
    assert(reflected_market_result.executed_quantity == 0);
    assert(reflected_market_result.boundary_truncated_quantity == 1);
    assert(bid_boundary_book.total_bid_depth() == 1);
    assert(bid_boundary_book.has_bid());

    OrderMessage reflected_cancel = reflected_market;
    reflected_cancel.sequence += 1;
    reflected_cancel.action = OrderAction::CancelAtDistance;
    reflected_cancel.side = Side::Buy;
    reflected_cancel.distance_ticks = 0;
    const ApplyResult reflected_cancel_result =
        bid_boundary_book.apply(reflected_cancel);
    assert(reflected_cancel_result.cancelled_quantity == 0);
    assert(reflected_cancel_result.boundary_truncated_quantity == 1);
    assert(bid_boundary_book.total_bid_depth() == 1);

    // A sampled tail mark is already outside the reduced support, so it is a
    // non-mutating event rather than a boundary intervention.
    reflected_cancel.sequence += 1;
    reflected_cancel.distance_ticks = reduced_background_depth_levels;
    const ApplyResult tail_cancel_at_boundary =
        bid_boundary_book.apply(reflected_cancel);
    assert(tail_cancel_at_boundary.cancelled_quantity == 0);
    assert(tail_cancel_at_boundary.boundary_truncated_quantity == 0);
    assert(bid_boundary_book.total_bid_depth() == 1);

    LimitOrderBook ask_boundary_book(100);
    ask_boundary_book.seed_calibrated_book(10'000, 10'200, 2, 2, 1.0);
    OrderMessage drain_ask = drain_bid;
    drain_ask.sequence = 4100;
    drain_ask.side = Side::Buy;
    drain_ask.quantity = static_cast<int>(
        ask_boundary_book.total_ask_depth() - 1);
    assert(ask_boundary_book.apply(drain_ask).executed_quantity
           == drain_ask.quantity);
    assert(ask_boundary_book.total_ask_depth() == 1);
    OrderMessage reflected_ask_cancel = reflected_cancel;
    reflected_ask_cancel.sequence = 4101;
    reflected_ask_cancel.side = Side::Sell;
    reflected_ask_cancel.distance_ticks = 0;
    const ApplyResult reflected_ask_result =
        ask_boundary_book.apply(reflected_ask_cancel);
    assert(reflected_ask_result.cancelled_quantity == 0);
    assert(reflected_ask_result.boundary_truncated_quantity == 1);
    assert(ask_boundary_book.total_ask_depth() == 1);

    // The reserve belongs to the displayed side, not permanently to anonymous
    // background liquidity.  A maker quote can therefore replace the final
    // owner-zero share as the protected displayed unit.
    LimitOrderBook shared_market_boundary_book(100);
    shared_market_boundary_book.seed_calibrated_book(
        10'000, 10'200, 2, 2, 1.0);
    OrderMessage drain_background_bid = drain_bid;
    drain_background_bid.sequence = 4200;
    drain_background_bid.quantity = static_cast<int>(
        shared_market_boundary_book.total_background_bid_depth() - 1);
    assert(shared_market_boundary_book.apply(drain_background_bid)
               .executed_quantity == drain_background_bid.quantity);
    assert(shared_market_boundary_book.total_background_bid_depth() == 1);

    OrderMessage temporary_shared_bid;
    temporary_shared_bid.sequence = 4201;
    temporary_shared_bid.owner_id = make_owner_id(5, 0);
    temporary_shared_bid.agent_kind = AgentKind::MarketMaker;
    temporary_shared_bid.action = OrderAction::Limit;
    temporary_shared_bid.side = Side::Buy;
    temporary_shared_bid.quantity = 50;
    temporary_shared_bid.price_ticks =
        shared_market_boundary_book.best_bid();
    shared_market_boundary_book.apply(temporary_shared_bid);
    assert(shared_market_boundary_book.total_bid_depth() == 51);
    assert(shared_market_boundary_book.total_background_bid_depth() == 1);

    OrderMessage guarded_background_market = reflected_market;
    guarded_background_market.sequence = 4202;
    const ApplyResult guarded_market_result =
        shared_market_boundary_book.apply(guarded_background_market);
    assert(guarded_market_result.executed_quantity == 50);
    assert(guarded_market_result.boundary_truncated_quantity == 1);
    assert(shared_market_boundary_book.total_bid_depth() == 1);
    assert(shared_market_boundary_book.total_background_bid_depth() == 0);

    OrderMessage withdraw_shared_bid = temporary_shared_bid;
    withdraw_shared_bid.sequence = 4203;
    withdraw_shared_bid.action = OrderAction::CancelOwner;
    withdraw_shared_bid.quantity = 0;
    assert(shared_market_boundary_book.apply(withdraw_shared_bid)
               .cancelled_quantity == 1);
    const MarketState after_shared_withdrawal =
        shared_market_boundary_book.state(4203, 10'100.0);
    assert(after_shared_withdrawal.best_bid_depth == 0);
    assert(after_shared_withdrawal.background_best_bid_depth == 0);
    assert(after_shared_withdrawal.best_bid_ticks == 0);
    assert(after_shared_withdrawal.best_ask_ticks > 0);

    // A background-only cancellation can remove all anonymous quantity when
    // independently owned maker liquidity keeps the displayed side present.
    LimitOrderBook shared_cancel_boundary_book(100);
    OrderMessage compact_background_bid;
    compact_background_bid.sequence = 4300;
    compact_background_bid.owner_id = 0;
    compact_background_bid.agent_kind = AgentKind::Background;
    compact_background_bid.action = OrderAction::Limit;
    compact_background_bid.side = Side::Buy;
    compact_background_bid.quantity = 2;
    compact_background_bid.price_ticks = 10'000;
    shared_cancel_boundary_book.apply(compact_background_bid);
    OrderMessage compact_background_ask = compact_background_bid;
    compact_background_ask.sequence += 1;
    compact_background_ask.side = Side::Sell;
    compact_background_ask.price_ticks = 10'200;
    shared_cancel_boundary_book.apply(compact_background_ask);
    assert(shared_cancel_boundary_book.total_background_bid_depth() == 2);
    temporary_shared_bid.sequence = 4302;
    temporary_shared_bid.price_ticks = shared_cancel_boundary_book.best_bid();
    shared_cancel_boundary_book.apply(temporary_shared_bid);
    OrderMessage guarded_background_cancel = reflected_cancel;
    guarded_background_cancel.sequence = 4303;
    guarded_background_cancel.quantity = 100;
    guarded_background_cancel.distance_ticks = 0;
    const ApplyResult guarded_cancel_result =
        shared_cancel_boundary_book.apply(guarded_background_cancel);
    assert(guarded_cancel_result.cancelled_quantity == 2);
    assert(guarded_cancel_result.boundary_truncated_quantity == 0);
    assert(shared_cancel_boundary_book.total_background_bid_depth() == 0);
    withdraw_shared_bid.sequence = 4304;
    assert(shared_cancel_boundary_book.apply(withdraw_shared_bid)
               .cancelled_quantity == 50);
    assert(shared_cancel_boundary_book.total_bid_depth() == 0);
    assert(shared_cancel_boundary_book.background_best_bid_depth() == 0);

    std::cout << "LOB and report tests passed\n";
    return 0;
}
