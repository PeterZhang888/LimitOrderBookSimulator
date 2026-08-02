#include "agents/AgentPopulation.hpp"
#include "exchange/EventOrdering.hpp"

#include <cassert>
#include <cstdint>

int main() {
    using namespace dlob;

    PopulationConfig config;
    config.market_makers = 2;
    config.momentum_traders = 8;
    config.informed_traders = 6;
    config.institutional_traders = 4;
    config.simulation_start_ns = 0;
    config.simulation_end_ns = 10'000'000'000LL;
    config.market_maker_interval_ns = 20'000'000LL;
    config.market_maker_batch_horizon_ns = 100'000'000LL;

    AgentPopulation mm(1, 5, config);
    AgentPopulation momentum(2, 5, config);
    AgentPopulation informed(3, 5, config);
    AgentPopulation institutional(4, 5, config);

    assert(mm.role() == WorkerRole::MarketMaker);
    assert(momentum.role() == WorkerRole::Momentum);
    assert(informed.role() == WorkerRole::Informed);
    assert(institutional.role() == WorkerRole::Institutional);

    assert(mm.local_summary().market_makers == 2);
    assert(mm.local_summary().momentum == 0);
    assert(momentum.local_summary().momentum == 8);
    assert(informed.local_summary().informed == 6);
    assert(institutional.local_summary().institutional == 4);

    MarketState state;
    state.exchange_time_ns = 0;
    state.best_bid_ticks = 2'203'400;
    state.best_ask_ticks = 2'203'700;
    state.best_bid_depth = 1'000;
    state.best_ask_depth = 1'000;
    state.mid_price_ticks = 2'203'550.0;
    state.fundamental_value_ticks = 2'203'550.0;

    mm.observe_market(state);
    const std::int64_t wake = mm.next_wake_time();
    assert(wake == 0);
    const std::int64_t cutoff = wake + mm.batch_horizon_ns();
    const auto orders = mm.generate_due_orders(wake, cutoff);
    assert(!orders.empty());
    for (const auto& order : orders) assert(order.agent_kind == AgentKind::MarketMaker);
    for (std::size_t i = 1; i < orders.size(); ++i) {
        assert(!order_before(orders[i], orders[i - 1]));
    }
    assert(mm.next_wake_time() > cutoff);

    return 0;
}
