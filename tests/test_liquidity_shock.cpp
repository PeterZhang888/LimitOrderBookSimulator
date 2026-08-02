// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#include "simulation/SequentialMultiAssetSimulator.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

std::filesystem::path source_root() {
    std::filesystem::path root = std::filesystem::path(__FILE__).parent_path()
        .parent_path();
    if (!std::filesystem::exists(root / "data")) {
        root = std::filesystem::current_path();
    }
    return std::filesystem::absolute(root);
}

void assert_same_model_result(const dlob::SequentialMultiAssetResult& left,
                              const dlob::SequentialMultiAssetResult& right) {
    assert(left.structurally_valid == right.structurally_valid);
    assert(left.combined_trade_count == right.combined_trade_count);
    assert(left.combined_trade_hash == right.combined_trade_hash);
    assert(left.processed_events == right.processed_events);
    assert(left.cross_book_reaction_events == right.cross_book_reaction_events);
    assert(left.hedge_order_events == right.hedge_order_events);
    assert(left.liquidity_shock_events == right.liquidity_shock_events);
    assert(left.market_maker_cash_ticks == right.market_maker_cash_ticks);
    assert(left.books.size() == right.books.size());
    for (std::size_t index = 0; index < left.books.size(); ++index) {
        const dlob::MultiAssetBookSummary& a = left.books[index];
        const dlob::MultiAssetBookSummary& b = right.books[index];
        assert(a.book_id == b.book_id);
        assert(a.final_state.best_bid_ticks == b.final_state.best_bid_ticks);
        assert(a.final_state.best_ask_ticks == b.final_state.best_ask_ticks);
        assert(a.final_state.best_bid_depth == b.final_state.best_bid_depth);
        assert(a.final_state.best_ask_depth == b.final_state.best_ask_depth);
        assert(a.final_state.background_best_bid_depth
               == b.final_state.background_best_bid_depth);
        assert(a.final_state.background_best_ask_depth
               == b.final_state.background_best_ask_depth);
        assert(a.final_state.total_background_bid_depth
               == b.final_state.total_background_bid_depth);
        assert(a.final_state.total_background_ask_depth
               == b.final_state.total_background_ask_depth);
        assert(a.final_state.last_trade_price_ticks
               == b.final_state.last_trade_price_ticks);
        assert(a.market_maker_inventory == b.market_maker_inventory);
        assert(a.market_maker_cash_ticks == b.market_maker_cash_ticks);
        assert(a.processed_events == b.processed_events);
        assert(a.submitted_orders == b.submitted_orders);
        assert(a.trade_count == b.trade_count);
        assert(a.trade_hash == b.trade_hash);
        assert(a.calibration_record.event_counts
               == b.calibration_record.event_counts);
        assert(a.calibration_record.market.snapshots
               == b.calibration_record.market.snapshots);
    }
}

dlob::SequentialMultiAssetResult run(bool with_shock,
                                     const std::string& suffix) {
    dlob::SequentialMultiAssetConfig config;
    config.duration_seconds = 2;
    config.book_count = 2;
    config.seed = 0x19d5'2f8a'7721'cc40ULL;
    config.data_dir = (source_root() / "data").string();
    config.market_maker_exposure_threshold = 0.0;
    config.enable_shared_market_maker_hedging = true;
    config.output_dir = (std::filesystem::temp_directory_path()
                         / ("dlob_liquidity_shock_" + suffix)).string();
    if (with_shock) {
        config.liquidity_shock = dlob::LiquidityShockConfig{
            20'000'000LL, dlob::BookId{1}, dlob::Side::Sell, 5'000};
    }
    dlob::SequentialMultiAssetSimulator simulator(config);
    return simulator.run();
}

} // namespace

int main() {
    const dlob::SequentialMultiAssetResult control = run(false, "control");
    const dlob::SequentialMultiAssetResult first = run(true, "first");
    const dlob::SequentialMultiAssetResult second = run(true, "second");

    assert_same_model_result(first, second);
    assert(control.liquidity_shock_events == 0);
    assert(first.liquidity_shock_events == 1);
    assert(first.cross_book_reaction_events > 0);
    assert(first.hedge_order_events > 0);

    // Book 1 receives the exogenous sell.  A shared-maker fill creates a
    // deterministic hedge in book 0, proving that the intervention propagates
    // across books rather than remaining a local tape perturbation.
    assert(first.books.at(1).trade_hash != control.books.at(1).trade_hash);
    assert(first.books.at(0).trade_hash != control.books.at(0).trade_hash);

    std::cout << "deterministic liquidity shock passed; shock_events="
              << first.liquidity_shock_events
              << " cross_reactions=" << first.cross_book_reaction_events
              << " hedge_orders=" << first.hedge_order_events << '\n';
    return 0;
}
