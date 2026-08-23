#include "simulation/DistributedMarketSimulator.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

[[noreturn]] void fail(const std::string& message, int rank) {
    std::cerr << "rank " << rank << ": " << message << '\n';
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void require(bool condition, const std::string& message, int rank) {
    if (!condition) fail(message, rank);
}

dlob::SimulationConfig make_config(const std::filesystem::path& root) {
    dlob::SimulationConfig config;
    config.asset_count = 17;
    config.duration_seconds = 3;
    config.decision_window_ns = 1'000'000'000LL;
    config.global_metrics_interval_ns = 1'000'000'000LL;
    config.asset_summary_interval_ns = 1'000'000'000LL;
    config.buffer_global_observations = false;
    config.use_persistent_risk_collective = false;
    config.background_model = "queue-reactive-v1";
    config.partition_mode = dlob::PartitionMode::Cyclic;
    config.enable_local_market_makers = false;
    config.enable_value_agents = false;
    config.enable_shared_market_maker = true;
    config.enable_global_shared_capacity = true;
    config.shared_quote_levels = 1;
    config.shared_quote_quantity = 10;
    config.shared_global_risk_limit_per_asset = 1'000.0;
    config.enable_shock = true;
    config.shock_time_ns = 2'000'000'000LL;
    config.shock_target_count = 3;
    config.shock_reference_bid_depth_multiple = 2.0;
    config.shock_inventory_adverse = true;
    config.asset_summary_csv = (root / "assets.csv").string();

    const std::filesystem::path quantity = root / "quantity.csv";
    const std::filesystem::path distance = root / "distance.csv";
    for (int index = 0; index < config.asset_count; ++index) {
        dlob::MultiAssetBookConfig book;
        book.symbol = "MPI" + std::to_string(index);
        book.fundamental_price_ticks = 10'000.0 + index;
        book.fundamental_move_probability_per_second = 0.0;
        book.initial_best_bid_ticks = 9'900 + index;
        book.initial_best_ask_ticks = 10'100 + index;
        book.initial_best_bid_depth = 100 + index;
        book.initial_best_ask_depth = 120 + index;
        book.beta = 1.0;
        book.market_maker_quote_quantity = 10;
        book.target_spread_ticks = 2;
        config.asset_configs.push_back(std::move(book));

        dlob::BackgroundHawkesConfig background;
        background.activity_scale = 1.0;
        background.mu.fill(1.0e-12);
        for (auto& row : background.alpha) row.fill(0.0);
        background.seed = 7'000U + static_cast<std::uint64_t>(index);
        background.tick_size = 100;
        background.target_spread_ticks = 2;
        background.limit_buy_quantity_file = quantity.string();
        background.limit_sell_quantity_file = quantity.string();
        background.market_buy_quantity_file = quantity.string();
        background.market_sell_quantity_file = quantity.string();
        background.cancel_bid_quantity_file = quantity.string();
        background.cancel_ask_quantity_file = quantity.string();
        background.limit_buy_distance_file = distance.string();
        background.limit_sell_distance_file = distance.string();
        background.cancel_bid_distance_file = distance.string();
        background.cancel_ask_distance_file = distance.string();
        config.background_configs.push_back(std::move(background));
    }
    return config;
}

} // namespace

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return EXIT_FAILURE;
    int rank = 0;
    int world_size = 0;
    require(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS,
            "MPI_Comm_rank failed", rank);
    require(MPI_Comm_size(MPI_COMM_WORLD, &world_size) == MPI_SUCCESS,
            "MPI_Comm_size failed", rank);
    require(argc == 2, "expected one output-directory argument", rank);
    require(world_size == 1 || world_size == 2 || world_size == 4,
            "expected 1, 2, or 4 MPI ranks", rank);

    const std::filesystem::path root(argv[1]);
    if (rank == 0) {
        std::filesystem::create_directories(root);
        std::ofstream quantity(root / "quantity.csv");
        require(static_cast<bool>(quantity),
                "could not create quantity distribution", rank);
        quantity << "quantity,count\n1,1\n";
        std::ofstream distance(root / "distance.csv");
        require(static_cast<bool>(distance),
                "could not create distance distribution", rank);
        distance << "distance_ticks,count\n1,1\n";
    }
    require(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS,
            "input-file barrier failed", rank);

    const dlob::SimulationResult result =
        dlob::DistributedMarketSimulator(
            MPI_COMM_WORLD, make_config(root)).run();

    if (rank == 0) {
        const std::uint64_t minimum = static_cast<std::uint64_t>(17 / world_size);
        const std::uint64_t maximum = static_cast<std::uint64_t>(
            (17 + world_size - 1) / world_size);
        require(result.world_size == world_size,
                "reported MPI size is incorrect", rank);
        require(result.asset_count == 17, "asset count is incorrect", rank);
        require(result.lob_count == 17U, "LOB count is incorrect", rank);
        require(result.windows == 3U, "window count is incorrect", rank);
        require(result.min_books_per_rank == minimum,
                "minimum books per rank is incorrect", rank);
        require(result.max_books_per_rank == maximum,
                "maximum books per rank is incorrect", rank);
        require(std::abs(result.mean_books_per_rank
                         - 17.0 / static_cast<double>(world_size)) < 1.0e-12,
                "mean books per rank is incorrect", rank);
        require(result.processed_orders > 0U, "no orders were processed", rank);
        require(result.trades > 0U, "no trades were produced", rank);
        require(result.shared_buy_quantity + result.shared_sell_quantity > 0U,
                "shared maker has no executed quantity", rank);
        require(result.shared_fill_count > 0U,
                "shared maker has no fills", rank);
        require(result.shared_terminal_absolute_inventory > 0U,
                "shared maker has no terminal inventory", rank);
        require(result.risk_boundaries == 4U,
                "risk-boundary count is incorrect", rank);
        require(result.risk_collective_calls == 4U,
                "risk collective-call count is incorrect", rank);
        require(result.observation_collective_calls == 4U,
                "observation collective-call count is incorrect", rank);
        require(result.terminal_collective_calls == 7U,
                "terminal collective-call count is incorrect", rank);
        require(result.collective_calls == 17U,
                "total collective-call count is incorrect", rank);

        std::ofstream canonical(root / "canonical.txt");
        require(static_cast<bool>(canonical),
                "could not create canonical output", rank);
        canonical << std::setprecision(17)
                  << "assets=" << result.asset_count << '\n'
                  << "lobs=" << result.lob_count << '\n'
                  << "windows=" << result.windows << '\n'
                  << "processed_orders=" << result.processed_orders << '\n'
                  << "trades=" << result.trades << '\n'
                  << "collective_calls=" << result.collective_calls << '\n'
                  << "risk_collective_calls="
                  << result.risk_collective_calls << '\n'
                  << "observation_collective_calls="
                  << result.observation_collective_calls << '\n'
                  << "terminal_collective_calls="
                  << result.terminal_collective_calls << '\n'
                  << "shared_buy_quantity=" << result.shared_buy_quantity << '\n'
                  << "shared_sell_quantity=" << result.shared_sell_quantity << '\n'
                  << "shared_fill_count=" << result.shared_fill_count << '\n'
                  << "terminal_absolute_inventory="
                  << result.shared_terminal_absolute_inventory << '\n'
                  << "final_gross_exposure="
                  << result.final_shared_gross_exposure << '\n'
                  << "mark_to_mid_pnl="
                  << result.shared_signed_mark_to_mid_pnl_usd << '\n'
                  << "liquidation_pnl="
                  << result.shared_signed_liquidation_pnl_usd << '\n'
                  << "fallback_quantity="
                  << result.terminal_fallback_quantity << '\n';
    }

    require(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS,
            "final output barrier failed", rank);
    return MPI_Finalize() == MPI_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
