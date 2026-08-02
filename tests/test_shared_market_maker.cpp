// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#include "agents/SharedMarketMakerAgent.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace {

dlob::SharedMarketMakerConfig config(std::int32_t source_rank = 0) {
    dlob::SharedMarketMakerConfig value;
    value.logical_owner_id = 77'001;
    value.message_source_rank = source_rank;
    value.quote_quantity = 25;
    value.quote_levels = 1;
    value.quote_quantity_growth = 1;
    value.quote_half_spread_ticks = 100;
    value.price_tick_size = 100;
    value.order_latency_ns = 11;
    value.exposure_threshold = 50.0;
    value.hedge_lot_size = 1;
    value.max_hedge_quantity = 1'000;
    value.report_latency_ns = 2;
    value.reaction_latency_ns = 3;
    value.network_latency_ns = 5;
    value.books = {
        {10, 1.0, 20},
        {20, 1.0, 10},
    };
    return value;
}

} // namespace

int main() {
    using namespace dlob;

    SharedMarketMakerAgent maker(config());
    MarketState state;
    state.book_id = 10;
    state.best_bid_ticks = 9'900;
    state.best_ask_ticks = 10'100;
    state.mid_price_ticks = 10'000.0;

    const auto quotes = maker.make_quotes(10, state, 100);
    assert(quotes.size() == 3);
    assert(quotes[0].action == OrderAction::CancelOwner);
    assert(quotes[1].action == OrderAction::Limit);
    assert(quotes[1].side == Side::Buy);
    assert(quotes[1].price_ticks == state.best_bid_ticks);
    assert(quotes[2].action == OrderAction::Limit);
    assert(quotes[2].side == Side::Sell);
    assert(quotes[2].price_ticks == state.best_ask_ticks);
    for (const auto& quote : quotes) {
        assert(quote.book_id == 10);
        assert(quote.owner_id == maker.logical_owner_id());
        assert(quote.arrival_time_ns == 111);
        assert(quote.arrival_time_ns > quote.generated_time_ns);
    }

    MarketState wide_state = state;
    wide_state.best_ask_ticks = 10'200;
    wide_state.mid_price_ticks = 10'050.0;
    const auto inside_quotes = maker.make_quotes(10, wide_state, 120);
    assert(inside_quotes.size() == 3);
    assert(inside_quotes[1].price_ticks == wide_state.best_bid_ticks + 100);
    assert(inside_quotes[2].price_ticks == wide_state.best_ask_ticks - 100);
    assert(inside_quotes[1].price_ticks < inside_quotes[2].price_ticks);

    MarketState very_wide_state = state;
    very_wide_state.best_ask_ticks = 10'900;
    very_wide_state.mid_price_ticks = 10'400.0;
    const auto collapsed_quotes = maker.make_quotes(10, very_wide_state, 130);
    assert(collapsed_quotes.size() == 3);
    assert(collapsed_quotes[1].price_ticks == 10'400);
    assert(collapsed_quotes[2].price_ticks == 10'500);

    MarketState depleted_bid_state = state;
    depleted_bid_state.best_bid_ticks = 9'000;
    depleted_bid_state.best_ask_ticks = 10'900;
    depleted_bid_state.last_trade_price_ticks = 10'800;
    depleted_bid_state.mid_price_ticks = 9'950.0;
    const auto continuity_quotes = maker.make_quotes(10, depleted_bid_state, 135);
    assert(continuity_quotes.size() == 3);
    assert(continuity_quotes[1].price_ticks == 10'800);
    assert(continuity_quotes[2].price_ticks == 10'900);

    MarketState one_tick_state = state;
    one_tick_state.best_ask_ticks = 10'000;
    one_tick_state.mid_price_ticks = 9'950.0;
    const auto one_tick_quotes = maker.make_quotes(10, one_tick_state, 140);
    assert(one_tick_quotes.size() == 3);
    assert(one_tick_quotes[1].price_ticks == one_tick_state.best_bid_ticks);
    assert(one_tick_quotes[2].price_ticks == one_tick_state.best_ask_ticks);

    MarketState bid_only_state;
    bid_only_state.book_id = 10;
    bid_only_state.best_bid_ticks = 9'900;
    bid_only_state.fundamental_value_ticks = 10'000.0;
    const auto bid_recovery = maker.make_quotes(10, bid_only_state, 160);
    assert(bid_recovery.size() == 3);
    assert(bid_recovery[1].price_ticks == 9'900);
    assert(bid_recovery[2].price_ticks == 10'100);

    MarketState empty_state;
    empty_state.book_id = 10;
    empty_state.fundamental_value_ticks = 10'000.0;
    const auto empty_recovery = maker.make_quotes(10, empty_state, 180);
    assert(empty_recovery.size() == 3);
    assert(empty_recovery[1].price_ticks == 9'900);
    assert(empty_recovery[2].price_ticks == 10'100);

    empty_state.last_trade_price_ticks = 10'500;
    const auto trade_anchored_recovery = maker.make_quotes(10, empty_state, 200);
    assert(trade_anchored_recovery.size() == 3);
    assert(trade_anchored_recovery[1].price_ticks == 10'400);
    assert(trade_anchored_recovery[2].price_ticks == 10'600);

    // The logical sequence and tie breaker do not depend on physical rank.
    SharedMarketMakerAgent other_partition(config(7));
    const auto partition_quotes = other_partition.make_quotes(10, state, 100);
    assert(partition_quotes.size() == quotes.size());
    for (std::size_t i = 0; i < quotes.size(); ++i) {
        assert(partition_quotes[i].sequence == quotes[i].sequence);
        assert(partition_quotes[i].tie_breaker == quotes[i].tie_breaker);
        assert(partition_quotes[i].owner_id == quotes[i].owner_id);
    }

    auto depth_curve_config = config();
    depth_curve_config.quote_levels = 3;
    depth_curve_config.quote_quantity_growth = 3;
    SharedMarketMakerAgent depth_curve_maker(depth_curve_config);
    const auto depth_curve = depth_curve_maker.make_quotes(10, state, 210);
    assert(depth_curve.size() == 7);
    assert(depth_curve[1].quantity == 25 && depth_curve[2].quantity == 25);
    assert(depth_curve[3].quantity == 75 && depth_curve[4].quantity == 75);
    assert(depth_curve[5].quantity == 225 && depth_curve[6].quantity == 225);

    TradeExecution irrelevant;
    irrelevant.book_id = 10;
    irrelevant.timestamp_ns = 200;
    irrelevant.price_ticks = 10'000;
    irrelevant.quantity = 1'000;
    irrelevant.buyer_owner_id = 123;
    irrelevant.seller_owner_id = 456;
    assert(maker.on_trade(irrelevant).empty());
    assert(maker.inventory(10) == 0);

    // ETF-like source exposure is split across all configured component
    // routes.  Beta converts component shares to common-factor exposure.
    auto weighted_config = config();
    weighted_config.books = {
        {10, 1.0, 20, 0, 1, {{20, 3.0}, {30, 1.0}}},
        {20, 1.0, 10},
        {30, 2.0, 10},
    };
    SharedMarketMakerAgent weighted_maker(weighted_config);
    TradeExecution weighted_fill;
    weighted_fill.book_id = 10;
    weighted_fill.timestamp_ns = 900;
    weighted_fill.price_ticks = 10'000;
    weighted_fill.quantity = 120;
    weighted_fill.buyer_owner_id = weighted_maker.logical_owner_id();
    weighted_fill.seller_owner_id = 123;
    const auto weighted_hedges = weighted_maker.on_trade(weighted_fill);
    assert(weighted_hedges.size() == 2);
    assert(weighted_hedges[0].book_id == 20);
    assert(weighted_hedges[0].side == Side::Sell);
    assert(weighted_hedges[0].quantity == 90);
    assert(weighted_hedges[1].book_id == 30);
    assert(weighted_hedges[1].side == Side::Sell);
    assert(weighted_hedges[1].quantity == 15);
    assert(std::abs(weighted_maker.projected_beta_exposure()) < 1e-12);

    TradeExecution own_fill;
    own_fill.book_id = 10;
    own_fill.timestamp_ns = 1'000;
    own_fill.trade_sequence = 1;
    own_fill.price_ticks = 10'000;
    own_fill.quantity = 120;
    own_fill.buyer_owner_id = maker.logical_owner_id();
    own_fill.seller_owner_id = 123;
    own_fill.buyer_order_sequence = quotes[1].sequence;
    own_fill.seller_order_sequence = 99;
    const auto hedges = maker.on_trade(own_fill);
    assert(hedges.size() == 1);
    const auto& hedge = hedges.front();
    assert(maker.inventory(10) == 120);
    assert(maker.cash_ticks(10) == -1'200'000);
    assert(std::abs(maker.beta_exposure() - 120.0) < 1e-12);
    assert(hedge.book_id == 20);
    assert(hedge.action == OrderAction::Market);
    assert(hedge.side == Side::Sell);
    assert(hedge.quantity == 120);
    assert(hedge.generated_time_ns == 1'005);
    assert(hedge.arrival_time_ns == 1'010);
    assert(std::abs(maker.projected_beta_exposure()) < 1e-12);

    // Filling the hedge transfers projected inventory into actual inventory;
    // it must not recursively create another cross-book order.
    TradeExecution hedge_fill;
    hedge_fill.book_id = 20;
    hedge_fill.timestamp_ns = hedge.arrival_time_ns;
    hedge_fill.trade_sequence = 2;
    hedge_fill.price_ticks = 9'950;
    hedge_fill.quantity = hedge.quantity;
    hedge_fill.buyer_owner_id = 321;
    hedge_fill.seller_owner_id = maker.logical_owner_id();
    hedge_fill.buyer_order_sequence = 100;
    hedge_fill.seller_order_sequence = hedge.sequence;
    assert(maker.on_trade(hedge_fill).empty());
    assert(maker.inventory(20) == -120);
    assert(maker.cash_ticks(20) == 1'194'000);
    assert(maker.total_cash_ticks() == -6'000);
    assert(std::abs(maker.beta_exposure()) < 1e-12);
    assert(std::abs(maker.projected_beta_exposure()) < 1e-12);

    // A partial terminal market hedge must not leave unexecuted quantity in
    // projected exposure forever.
    SharedMarketMakerAgent partial_maker(config());
    own_fill.buyer_owner_id = partial_maker.logical_owner_id();
    const auto partial_hedges = partial_maker.on_trade(own_fill);
    assert(partial_hedges.size() == 1);
    const auto& partial_hedge = partial_hedges.front();
    hedge_fill.seller_owner_id = partial_maker.logical_owner_id();
    hedge_fill.seller_order_sequence = partial_hedge.sequence;
    hedge_fill.quantity = 20;
    assert(partial_maker.on_trade(hedge_fill).empty());
    assert(std::abs(partial_maker.projected_beta_exposure()) < 1e-12);
    assert(partial_maker.complete_order(partial_hedge.sequence));
    assert(!partial_maker.complete_order(partial_hedge.sequence));
    assert(std::abs(partial_maker.projected_beta_exposure() - 100.0) < 1e-12);

    // A fill exactly at the configured threshold does not trigger a hedge.
    SharedMarketMakerAgent threshold_maker(config());
    TradeExecution threshold_fill = own_fill;
    threshold_fill.quantity = 50;
    threshold_fill.buyer_owner_id = threshold_maker.logical_owner_id();
    assert(threshold_maker.on_trade(threshold_fill).empty());

    // A fill report marked as originating from a hedge still updates the
    // balance sheet, but cannot start a second hedge even when exposure is
    // above the normal reaction threshold.
    SharedMarketMakerAgent no_feedback_maker(config());
    TradeExecution no_feedback_fill = own_fill;
    no_feedback_fill.buyer_owner_id = no_feedback_maker.logical_owner_id();
    assert(no_feedback_maker.on_trade(no_feedback_fill, false).empty());
    assert(no_feedback_maker.inventory(10) == 120);
    assert(std::abs(no_feedback_maker.projected_beta_exposure() - 120.0) < 1e-12);

    // One-book mode is the exact local-only QQQ baseline: it shares quote and
    // accounting code, permits the self route, and never emits a cross-book
    // reaction.
    auto local_config = config();
    local_config.books = {{10, 1.0, 10}};
    SharedMarketMakerAgent local_maker(local_config);
    const auto local_quotes = local_maker.make_quotes(10, state, 500);
    assert(local_quotes.size() == 3);
    TradeExecution local_fill = own_fill;
    local_fill.buyer_owner_id = local_maker.logical_owner_id();
    assert(local_maker.on_trade(local_fill).empty());
    assert(local_maker.inventory(10) == 120);
    assert(local_maker.cash_ticks(10) == -1'200'000);
    assert(std::abs(local_maker.beta_exposure() - 120.0) < 1e-12);

    std::cout << "shared market-maker tests passed\n";
    return 0;
}
