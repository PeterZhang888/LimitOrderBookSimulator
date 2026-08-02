#include "agents/FundamentalValueAgent.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main() {
    dlob::FundamentalValueConfig config;
    config.enabled = true;
    config.threshold_bps = 10.0;
    config.response_step_bps = 5.0;
    config.base_order_quantity = 20;
    config.max_order_quantity = 100;
    config.max_abs_inventory = 120;
    config.fundamental_volatility_bps_sqrt_second = 0.0;
    config.decision_interval_ns = 1'000'000'000;
    config.order_latency_ns = 5'000;

    dlob::FundamentalValueAgent agent(config, 2, 10'000.0, 12345);
    dlob::MarketState state;
    state.book_id = 2;
    state.best_bid_ticks = 9'979;
    state.best_ask_ticks = 9'981;
    state.mid_price_ticks = 9'980.0; // 20 bps cheap

    const auto buy = agent.make_order(state, 1'000'000'000, 1);
    assert(buy.has_value());
    assert(buy->agent_kind == dlob::AgentKind::Value);
    assert(buy->action == dlob::OrderAction::Market);
    assert(buy->side == dlob::Side::Buy);
    assert(buy->quantity == 60); // floor((20-10)/5)+1 = 3 lots
    assert(buy->book_id == 2);
    assert(buy->owner_id == dlob::fundamental_value_owner_id(2));
    assert(buy->arrival_time_ns == 1'000'005'000);

    dlob::TradeExecution fill;
    fill.book_id = 2;
    fill.price_ticks = 9'981;
    fill.quantity = 60;
    fill.buyer_owner_id = buy->owner_id;
    fill.seller_owner_id = 42;
    agent.on_trade(fill);
    assert(agent.inventory() == 60);
    assert(agent.cash_ticks() == -598'860);

    state.best_bid_ticks = 10'039;
    state.best_ask_ticks = 10'041;
    state.mid_price_ticks = 10'040.0; // 40 bps expensive
    const auto sell = agent.make_order(state, 2'000'000'000, 2);
    assert(sell.has_value());
    assert(sell->side == dlob::Side::Sell);
    assert(sell->quantity == 100);

    state.best_bid_ticks = 9'994;
    state.best_ask_ticks = 9'996;
    state.mid_price_ticks = 9'995.0; // 5 bps cheap, inside threshold
    assert(!agent.make_order(state, 3'000'000'000, 3).has_value());

    dlob::FundamentalValueConfig stochastic = config;
    stochastic.fundamental_volatility_bps_sqrt_second = 0.5;
    dlob::FundamentalValueAgent first(stochastic, 0, 10'000.0, 99);
    dlob::FundamentalValueAgent second(stochastic, 0, 10'000.0, 99);
    state.book_id = 0;
    (void)first.make_order(state, 1'000'000'000, 1);
    (void)second.make_order(state, 1'000'000'000, 1);
    assert(first.fundamental_value_ticks() == second.fundamental_value_ticks());
    assert(first.fundamental_value_ticks() != 10'000.0);

    dlob::FundamentalValueConfig overlapping = config;
    overlapping.order_latency_ns = overlapping.decision_interval_ns;
    bool rejected = false;
    try {
        dlob::FundamentalValueAgent invalid(overlapping, 0, 10'000.0, 1);
        (void)invalid;
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);

    std::cout << "fundamental value agent tests passed\n";
    return 0;
}
