#include "simulation/ExactMpiMultiAssetSimulator.hpp"
#include "simulation/SequentialMultiAssetSimulator.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void compare_state(const dlob::MarketState& expected,
                   const dlob::MarketState& actual,
                   dlob::BookId book_id) {
    const std::string prefix = "book " + std::to_string(book_id) + " final state: ";
    require(expected.exchange_time_ns == actual.exchange_time_ns,
            prefix + "exchange_time_ns differs");
    require(expected.best_bid_ticks == actual.best_bid_ticks,
            prefix + "best_bid_ticks differs");
    require(expected.best_ask_ticks == actual.best_ask_ticks,
            prefix + "best_ask_ticks differs");
    require(expected.best_bid_depth == actual.best_bid_depth,
            prefix + "best_bid_depth differs");
    require(expected.best_ask_depth == actual.best_ask_depth,
            prefix + "best_ask_depth differs");
    require(expected.background_best_bid_depth
                == actual.background_best_bid_depth,
            prefix + "background_best_bid_depth differs");
    require(expected.background_best_ask_depth
                == actual.background_best_ask_depth,
            prefix + "background_best_ask_depth differs");
    require(expected.total_background_bid_depth
                == actual.total_background_bid_depth,
            prefix + "total_background_bid_depth differs");
    require(expected.total_background_ask_depth
                == actual.total_background_ask_depth,
            prefix + "total_background_ask_depth differs");
    require(expected.last_trade_price_ticks == actual.last_trade_price_ticks,
            prefix + "last_trade_price_ticks differs");
    require(expected.mid_price_ticks == actual.mid_price_ticks,
            prefix + "mid_price_ticks differs");
    require(expected.fundamental_value_ticks == actual.fundamental_value_ticks,
            prefix + "fundamental_value_ticks differs");
    require(expected.cumulative_aggressive_buy == actual.cumulative_aggressive_buy,
            prefix + "cumulative_aggressive_buy differs");
    require(expected.cumulative_aggressive_sell == actual.cumulative_aggressive_sell,
            prefix + "cumulative_aggressive_sell differs");
    require(expected.book_id == actual.book_id, prefix + "book_id differs");
}

void compare_market_features(
    const dlob::calibration::MarketFeatureSummary& expected,
    const dlob::calibration::MarketFeatureSummary& actual,
    dlob::BookId book_id) {
    const std::string prefix = "book " + std::to_string(book_id)
        + " market features: ";
    require(expected.mean_spread_ticks == actual.mean_spread_ticks,
            prefix + "mean_spread_ticks differs");
    require(expected.mean_bid_depth == actual.mean_bid_depth,
            prefix + "mean_bid_depth differs");
    require(expected.mean_ask_depth == actual.mean_ask_depth,
            prefix + "mean_ask_depth differs");
    require(expected.mid_move_rate == actual.mid_move_rate,
            prefix + "mid_move_rate differs");
    require(expected.return_variance == actual.return_variance,
            prefix + "return_variance differs");
    require(expected.return_kurtosis == actual.return_kurtosis,
            prefix + "return_kurtosis differs");
    require(expected.absolute_return_acf1 == actual.absolute_return_acf1,
            prefix + "absolute_return_acf1 differs");
    require(expected.snapshots == actual.snapshots,
            prefix + "snapshot count differs");
}

void compare_results(const dlob::SequentialMultiAssetResult& expected,
                     const dlob::SequentialMultiAssetResult& actual) {
    require(expected.combined_trade_count == actual.combined_trade_count,
            "combined trade count differs");
    require(expected.combined_trade_hash == actual.combined_trade_hash,
            "combined trade hash differs");
    require(expected.processed_events == actual.processed_events,
            "global processed-event count differs");
    require(expected.cross_book_reaction_events
                == actual.cross_book_reaction_events,
            "cross-book reaction-event count differs");
    require(expected.hedge_order_events == actual.hedge_order_events,
            "hedge-order event count differs");
    require(expected.liquidity_shock_events == actual.liquidity_shock_events,
            "liquidity-shock event count differs");
    require(expected.arbitrage_decision_events
                == actual.arbitrage_decision_events,
            "arbitrage decision-event count differs");
    require(expected.arbitrage_order_events == actual.arbitrage_order_events,
            "arbitrage order-event count differs");
    require(expected.value_decision_events == actual.value_decision_events,
            "value decision-event count differs");
    require(expected.value_order_events == actual.value_order_events,
            "value order-event count differs");
    require(expected.market_maker_cash_ticks == actual.market_maker_cash_ticks,
            "shared market-maker total cash differs");
    require(expected.arbitrage_cash_ticks == actual.arbitrage_cash_ticks,
            "ETF-arbitrage total cash differs");
    require(expected.structurally_valid == actual.structurally_valid,
            "global structural-validity flag differs");
    require(expected.books.size() == actual.books.size(), "book count differs");

    for (std::size_t index = 0; index < expected.books.size(); ++index) {
        const dlob::MultiAssetBookSummary& reference = expected.books[index];
        const dlob::MultiAssetBookSummary& parallel = actual.books[index];
        require(reference.book_id == parallel.book_id, "book ordering differs");
        require(reference.symbol == parallel.symbol, "book symbol differs");
        const dlob::BookId book_id = reference.book_id;
        compare_state(reference.final_state, parallel.final_state, book_id);
        require(reference.market_maker_inventory
                    == parallel.market_maker_inventory,
                "book " + std::to_string(book_id)
                    + " shared market-maker inventory differs");
        require(reference.market_maker_cash_ticks
                    == parallel.market_maker_cash_ticks,
                "book " + std::to_string(book_id)
                    + " shared market-maker cash differs");
        require(reference.arbitrage_inventory
                    == parallel.arbitrage_inventory,
                "book " + std::to_string(book_id)
                    + " ETF-arbitrage inventory differs");
        require(reference.arbitrage_cash_ticks
                    == parallel.arbitrage_cash_ticks,
                "book " + std::to_string(book_id)
                    + " ETF-arbitrage cash differs");
        require(reference.value_agent_inventory
                    == parallel.value_agent_inventory,
                "book " + std::to_string(book_id)
                    + " value-agent inventory differs");
        require(reference.value_agent_cash_ticks
                    == parallel.value_agent_cash_ticks,
                "book " + std::to_string(book_id)
                    + " value-agent cash differs");
        require(reference.final_fundamental_value_ticks
                    == parallel.final_fundamental_value_ticks,
                "book " + std::to_string(book_id)
                    + " final fundamental differs");
        require(reference.processed_events == parallel.processed_events,
                "book " + std::to_string(book_id)
                    + " processed-event count differs");
        require(reference.submitted_orders == parallel.submitted_orders,
                "book " + std::to_string(book_id)
                    + " submitted-order count differs");
        require(reference.trade_count == parallel.trade_count,
                "book " + std::to_string(book_id) + " trade count differs");
        require(reference.trade_hash == parallel.trade_hash,
                "book " + std::to_string(book_id) + " trade hash differs");
        require(reference.expected_sample_count == parallel.expected_sample_count,
                "book " + std::to_string(book_id)
                    + " expected sample count differs");
        require(reference.structurally_valid == parallel.structurally_valid,
                "book " + std::to_string(book_id)
                    + " structural-validity flag differs");
        require(reference.calibration_record.event_counts
                    == parallel.calibration_record.event_counts,
                "book " + std::to_string(book_id)
                    + " empirical event counts differ");
        require(reference.calibration_record.owner_cancel_messages
                    == parallel.calibration_record.owner_cancel_messages,
                "book " + std::to_string(book_id)
                    + " owner-cancel count differs");
        require(reference.calibration_record.quantity_samples
                    == parallel.calibration_record.quantity_samples,
                "book " + std::to_string(book_id)
                    + " recorder reservoirs differ");
        require(reference.calibration_record.state_trace
                    == parallel.calibration_record.state_trace,
                "book " + std::to_string(book_id)
                    + " recorder state traces differ");
        compare_market_features(reference.calibration_record.market,
                                parallel.calibration_record.market,
                                book_id);
    }
}

std::filesystem::path source_root() {
    std::filesystem::path root = std::filesystem::path(__FILE__).parent_path()
        .parent_path();
    if (!std::filesystem::exists(root / "data")) {
        root = std::filesystem::current_path();
    }
    return std::filesystem::absolute(root);
}

} // namespace

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    dlob::SequentialMultiAssetConfig config;
    config.duration_seconds = 5;
    config.book_count = 3;
    config.seed = 0x34a1'12bc'9876'fed0ULL;
    config.data_dir = (source_root() / "data").string();
    config.book_configs.resize(3);
    for (std::size_t index = 0; index < config.book_configs.size(); ++index) {
        dlob::MultiAssetBookConfig& book = config.book_configs[index];
        book.symbol = "TEST_" + std::to_string(index);
        book.data_dir = config.data_dir;
        book.fundamental_price_ticks = index == 0 ? 2'203'550.0 : 2'200'000.0;
        book.beta = 1.0;
        book.basket_weight = index == 0 ? 0.0 : 1.0;
        book.target_spread_ticks = 1;
    }
    config.etf_arbitrage.enabled = true;
    config.etf_arbitrage.etf_book_id = 0;
    config.etf_arbitrage.trigger_bps = 1.0;
    config.etf_arbitrage.release_bps = 0.5;
    config.etf_arbitrage.etf_order_quantity = 25;
    config.etf_arbitrage.decision_interval_ns = 100'000'000;
    config.fundamental_value.enabled = true;
    config.fundamental_value.threshold_bps = 0.1;
    config.fundamental_value.response_step_bps = 0.1;
    config.fundamental_value.base_order_quantity = 10;
    config.fundamental_value.max_order_quantity = 50;
    config.fundamental_value.max_abs_inventory = 10'000;
    config.fundamental_value.fundamental_volatility_bps_sqrt_second = 0.5;
    config.fundamental_value.decision_interval_ns = 500'000'000;
    config.market_maker_exposure_threshold = 0.0;
    config.enable_shared_market_maker_hedging = true;
    config.liquidity_shock = dlob::LiquidityShockConfig{
        20'000'000LL, dlob::BookId{1}, dlob::Side::Sell, 5'000};
    config.output_dir = (std::filesystem::temp_directory_path()
                         / ("dlob_exact_mpi_" + std::to_string(world_size))).string();

    dlob::SequentialMultiAssetResult reference;
    int reference_ok = 1;
    std::string reference_error;
    if (rank == 0) {
        try {
            dlob::SequentialMultiAssetConfig sequential_config = config;
            sequential_config.output_dir = (std::filesystem::temp_directory_path()
                / ("dlob_exact_reference_" + std::to_string(world_size))).string();
            dlob::SequentialMultiAssetSimulator simulator(sequential_config);
            reference = simulator.run();
        } catch (const std::exception& error) {
            reference_ok = 0;
            reference_error = error.what();
        }
    }
    MPI_Bcast(&reference_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (reference_ok == 0) {
        if (rank == 0) {
            std::cerr << "sequential reference setup failed: "
                      << reference_error << '\n';
        }
        MPI_Finalize();
        return 1;
    }

    try {
        dlob::ExactMpiMultiAssetSimulator simulator(MPI_COMM_WORLD, config);
        const dlob::ExactMpiMultiAssetResult parallel = simulator.run();

        int expected_local_books = 0;
        for (int index = 0; index < config.book_count; ++index) {
            if (dlob::ExactMpiMultiAssetSimulator::owner_rank(
                    static_cast<dlob::BookId>(index), world_size) == rank) {
                ++expected_local_books;
            }
        }
        const int local_layout_ok = parallel.local_book_count == expected_local_books
            ? 1 : 0;
        int layout_ok = 0;
        MPI_Allreduce(&local_layout_ok, &layout_ok, 1, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD);

        int equivalent = layout_ok;
        std::string comparison_error;
        if (rank == 0 && equivalent != 0) {
            try {
                compare_results(reference, parallel.model);
                require(reference.cross_book_reaction_events > 0,
                        "test configuration did not exercise cross-book reactions");
                require(reference.hedge_order_events > 0,
                        "test configuration did not exercise hedge orders");
                require(reference.liquidity_shock_events == 1,
                        "test configuration did not exercise the liquidity shock");
                require(reference.arbitrage_decision_events > 0,
                        "test configuration did not exercise arbitrage decisions");
                require(reference.arbitrage_order_events > 0,
                        "test configuration did not exercise arbitrage orders");
                require(reference.value_decision_events > 0,
                        "test configuration did not exercise value decisions");
                require(reference.value_order_events > 0,
                        "test configuration did not exercise value orders");
            } catch (const std::exception& error) {
                equivalent = 0;
                comparison_error = error.what();
            }
        }
        MPI_Bcast(&equivalent, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (equivalent == 0) {
            if (rank == 0) {
                if (comparison_error.empty()) {
                    comparison_error = "book ownership layout differs";
                }
                std::cerr << "exact MPI equivalence failed with " << world_size
                          << " rank(s): " << comparison_error << '\n';
            }
            MPI_Finalize();
            return 1;
        }

        if (rank == 0) {
            std::cout << "exact MPI equivalence passed with " << world_size
                      << " rank(s), 3 books, " << reference.processed_events
                      << " events, trade_hash=0x" << std::hex
                      << reference.combined_trade_hash << std::dec << '\n';
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_exact_mpi_equivalence rank " << rank << ": "
                  << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        return 1;
    }
}
