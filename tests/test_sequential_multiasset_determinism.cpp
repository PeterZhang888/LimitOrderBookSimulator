#include "simulation/SequentialMultiAssetSimulator.hpp"

#include <array>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

void assert_state_equal(const dlob::MarketState& left,
                        const dlob::MarketState& right) {
    assert(left.exchange_time_ns == right.exchange_time_ns);
    assert(left.best_bid_ticks == right.best_bid_ticks);
    assert(left.best_ask_ticks == right.best_ask_ticks);
    assert(left.best_bid_depth == right.best_bid_depth);
    assert(left.best_ask_depth == right.best_ask_depth);
    assert(left.background_best_bid_depth == right.background_best_bid_depth);
    assert(left.background_best_ask_depth == right.background_best_ask_depth);
    assert(left.total_background_bid_depth == right.total_background_bid_depth);
    assert(left.total_background_ask_depth == right.total_background_ask_depth);
    assert(left.last_trade_price_ticks == right.last_trade_price_ticks);
    assert(left.mid_price_ticks == right.mid_price_ticks);
    assert(left.fundamental_value_ticks == right.fundamental_value_ticks);
    assert(left.cumulative_aggressive_buy == right.cumulative_aggressive_buy);
    assert(left.cumulative_aggressive_sell == right.cumulative_aggressive_sell);
    assert(left.book_id == right.book_id);
}

void assert_record_equal(const dlob::calibration::SimulationRecord& left,
                         const dlob::calibration::SimulationRecord& right) {
    assert(left.event_counts == right.event_counts);
    assert(left.owner_cancel_messages == right.owner_cancel_messages);
    assert(left.quantity_samples == right.quantity_samples);
    assert(left.state_trace == right.state_trace);
    assert(left.market.mean_spread_ticks == right.market.mean_spread_ticks);
    assert(left.market.mean_bid_depth == right.market.mean_bid_depth);
    assert(left.market.mean_ask_depth == right.market.mean_ask_depth);
    assert(left.market.mid_move_rate == right.market.mid_move_rate);
    assert(left.market.return_variance == right.market.return_variance);
    assert(left.market.return_kurtosis == right.market.return_kurtosis);
    assert(left.market.absolute_return_acf1 == right.market.absolute_return_acf1);
    assert(left.market.snapshots == right.market.snapshots);
}

void assert_result_equal(const dlob::SequentialMultiAssetResult& left,
                         const dlob::SequentialMultiAssetResult& right) {
    assert(left.combined_trade_count == right.combined_trade_count);
    assert(left.combined_trade_hash == right.combined_trade_hash);
    assert(left.processed_events == right.processed_events);
    assert(left.cross_book_reaction_events == right.cross_book_reaction_events);
    assert(left.hedge_order_events == right.hedge_order_events);
    assert(left.liquidity_shock_events == right.liquidity_shock_events);
    assert(left.value_decision_events == right.value_decision_events);
    assert(left.value_order_events == right.value_order_events);
    assert(left.market_maker_cash_ticks == right.market_maker_cash_ticks);
    assert(left.arbitrage_cash_ticks == right.arbitrage_cash_ticks);
    assert(left.books.size() == right.books.size());
    for (std::size_t index = 0; index < left.books.size(); ++index) {
        const auto& a = left.books[index];
        const auto& b = right.books[index];
        assert(a.book_id == b.book_id);
        assert_state_equal(a.final_state, b.final_state);
        assert(a.market_maker_inventory == b.market_maker_inventory);
        assert(a.market_maker_cash_ticks == b.market_maker_cash_ticks);
        assert(a.arbitrage_inventory == b.arbitrage_inventory);
        assert(a.arbitrage_cash_ticks == b.arbitrage_cash_ticks);
        assert(a.value_agent_inventory == b.value_agent_inventory);
        assert(a.value_agent_cash_ticks == b.value_agent_cash_ticks);
        assert(a.final_fundamental_value_ticks
               == b.final_fundamental_value_ticks);
        assert(a.processed_events == b.processed_events);
        assert(a.submitted_orders == b.submitted_orders);
        assert(a.trade_count == b.trade_count);
        assert(a.trade_hash == b.trade_hash);
        assert_record_equal(a.calibration_record, b.calibration_record);
    }
}

dlob::SequentialMultiAssetResult run_once(int books, const std::string& suffix) {
    dlob::SequentialMultiAssetConfig config;
    config.duration_seconds = 10;
    config.book_count = books;
    config.seed = 0x1234'5678'9abc'def0ULL;
    config.initial_depth_scale = 1.0;
    config.market_maker_exposure_threshold = 0.0;
    config.enable_shared_market_maker_hedging = true;
    config.fundamental_value.enabled = true;
    config.fundamental_value.threshold_bps = 0.1;
    config.fundamental_value.fundamental_volatility_bps_sqrt_second = 0.5;
    config.output_dir = (std::filesystem::temp_directory_path()
                         / ("dlob_sequential_determinism_" + suffix)).string();
    dlob::SequentialMultiAssetSimulator simulator(config);
    return simulator.run();
}

} // namespace

int main() {
    const auto single = run_once(1, "single");
    assert(single.books.size() == 1);
    assert(single.cross_book_reaction_events == 0);
    assert(single.hedge_order_events == 0);
    assert(single.books[0].calibration_record.market.snapshots == 10);
    assert(single.combined_trade_hash == single.books[0].trade_hash);
    assert(single.combined_trade_count == single.books[0].trade_count);

    const auto first = run_once(2, "first");
    const auto second = run_once(2, "second");
    assert_result_equal(first, second);
    assert(first.books.size() == 2);
    assert(first.books[0].calibration_record.market.snapshots == 10);
    assert(first.books[1].calibration_record.market.snapshots == 10);
    assert(first.cross_book_reaction_events > 0);
    assert(first.hedge_order_events > 0);

    std::cout << "sequential multi-asset determinism tests passed; hash=0x"
              << std::hex << first.combined_trade_hash << std::dec << '\n';
    return 0;
}
