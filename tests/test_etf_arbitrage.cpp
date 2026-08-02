// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#include "agents/EtfArbitrageAgent.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

dlob::MarketState state(dlob::BookId id, std::int32_t mid) {
    dlob::MarketState value;
    value.book_id = id;
    value.best_bid_ticks = mid - 50;
    value.best_ask_ticks = mid + 50;
    value.mid_price_ticks = static_cast<double>(mid);
    return value;
}

} // namespace

int main() {
    using namespace dlob;

    std::vector<MultiAssetBookConfig> books(3);
    books[0].symbol = "ETF";
    books[0].fundamental_price_ticks = 10'000.0;
    books[1].symbol = "A";
    books[1].fundamental_price_ticks = 10'000.0;
    books[1].basket_weight = 1.0;
    books[2].symbol = "B";
    books[2].fundamental_price_ticks = 10'000.0;
    books[2].basket_weight = 1.0;

    EtfArbitrageConfig config;
    config.enabled = true;
    config.etf_book_id = 0;
    config.trigger_bps = 5.0;
    config.release_bps = 2.0;
    config.etf_order_quantity = 100;
    config.max_component_quantity = 1'000;
    config.order_latency_ns = 7;

    EtfArbitrageAgent agent(config, books);
    std::vector<MarketState> expensive{
        state(0, 10'100), state(1, 10'000), state(2, 10'000)};
    const auto sell_etf = agent.make_orders(expensive, 100, 1);
    assert(sell_etf.size() == 3);
    assert(std::abs(agent.last_deviation_bps() - 100.0) < 1e-12);
    assert(sell_etf[0].book_id == 0 && sell_etf[0].side == Side::Sell);
    assert(sell_etf[0].quantity == 100);
    assert(sell_etf[1].book_id == 1 && sell_etf[1].side == Side::Buy);
    assert(sell_etf[2].book_id == 2 && sell_etf[2].side == Side::Buy);
    assert(sell_etf[1].quantity == 50 && sell_etf[2].quantity == 50);
    for (const OrderMessage& order : sell_etf) {
        assert(order.agent_kind == AgentKind::Arbitrage);
        assert(order.action == OrderAction::Market);
        assert(order.owner_id == etf_arbitrage_owner_id);
        assert(order.generated_time_ns == 100);
        assert(order.arrival_time_ns == 107);
    }
    TradeExecution etf_fill;
    etf_fill.book_id = 0;
    etf_fill.price_ticks = 10'100;
    etf_fill.quantity = 100;
    etf_fill.buyer_owner_id = 10;
    etf_fill.seller_owner_id = etf_arbitrage_owner_id;
    agent.on_trade(etf_fill);
    assert(agent.inventory(0) == -100);
    assert(agent.cash_ticks(0) == 1'010'000);

    TradeExecution component_fill;
    component_fill.book_id = 1;
    component_fill.price_ticks = 10'000;
    component_fill.quantity = 50;
    component_fill.buyer_owner_id = etf_arbitrage_owner_id;
    component_fill.seller_owner_id = 11;
    agent.on_trade(component_fill);
    assert(agent.inventory(1) == 50);
    assert(agent.cash_ticks(1) == -500'000);
    assert(agent.total_cash_ticks() == 510'000);

    // Hysteresis prevents repeated orders while the same dislocation persists.
    assert(agent.make_orders(expensive, 200, 2).empty());

    // A return inside the release band rearms the agent.
    std::vector<MarketState> neutral{
        state(0, 10'000), state(1, 10'000), state(2, 10'000)};
    assert(agent.make_orders(neutral, 300, 3).empty());

    std::vector<MarketState> cheap{
        state(0, 9'900), state(1, 10'000), state(2, 10'000)};
    const auto buy_etf = agent.make_orders(cheap, 400, 4);
    assert(buy_etf.size() == 3);
    assert(std::abs(agent.last_deviation_bps() + 100.0) < 1e-12);
    assert(buy_etf[0].side == Side::Buy);
    assert(buy_etf[1].side == Side::Sell);
    assert(buy_etf[2].side == Side::Sell);

    std::cout << "ETF arbitrage signal, hysteresis, and basket legs passed\n";
    return 0;
}
