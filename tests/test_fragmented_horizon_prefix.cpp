#include "simulation/FragmentedMpiSimulator.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> lines(const std::filesystem::path& path) {
    std::ifstream input(path);
    assert(input);
    std::vector<std::string> result;
    std::string line;
    while (std::getline(input, line)) result.push_back(line);
    return result;
}

dlob::FragmentedMpiConfig make_config(
    int duration, const std::filesystem::path& metrics,
    const std::filesystem::path& quantity,
    const std::filesystem::path& distance) {
    dlob::FragmentedMpiConfig config;
    config.asset_count = 1;
    config.duration_seconds = duration;
    config.stochastic_baseline_normalization_horizon_ns = 4'000'000'000LL;
    config.decision_window_ns = 1'000'000'000LL;
    config.global_metrics_interval_ns = 1'000'000'000LL;
    config.tick_size = 100;
    config.seed = 20200130;
    config.background_model = "queue-reactive-v1";
    config.enable_local_market_makers = false;
    config.enable_value_agents = false;
    config.enable_shared_market_maker = true;
    config.enable_global_shared_capacity = true;
    config.metrics_csv = metrics.string();

    dlob::MultiAssetBookConfig book;
    book.symbol = "PREFIX";
    book.fundamental_price_ticks = 10'000.0;
    book.fundamental_move_probability_per_second = 0.0;
    book.initial_best_bid_ticks = 9'900;
    book.initial_best_ask_ticks = 10'100;
    book.initial_best_bid_depth = 100;
    book.initial_best_ask_depth = 100;
    book.beta = 1.0;
    book.market_maker_quote_quantity = 50;
    book.target_spread_ticks = 2;
    book.fundamental_log_variance_persistence = 0.7;
    book.fundamental_log_variance_std = 1.0;
    book.fundamental_order_flow_coupling = 0.8;
    config.asset_configs.push_back(book);

    dlob::BackgroundHawkesConfig background;
    background.activity_scale = 1.0;
    background.mu.fill(2.0);
    for (auto& row : background.alpha) row.fill(0.0);
    background.seed = 1234;
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
    config.background_configs.push_back(background);
    return config;
}

} // namespace

int main(int argc, char** argv) {
    assert(MPI_Init(&argc, &argv) == MPI_SUCCESS);
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "dlob_horizon_prefix";
    std::filesystem::create_directories(root);
    const auto quantity = root / "quantity.csv";
    const auto distance = root / "distance.csv";
    {
        std::ofstream output(quantity);
        output << "quantity,count\n10,1\n";
    }
    {
        std::ofstream output(distance);
        output << "distance_ticks,count\n1,1\n";
    }
    const auto short_metrics = root / "short.csv";
    const auto full_metrics = root / "full.csv";
    const auto short_result = dlob::FragmentedMpiSimulator(
        MPI_COMM_WORLD, make_config(2, short_metrics, quantity, distance)).run();
    const auto full_result = dlob::FragmentedMpiSimulator(
        MPI_COMM_WORLD, make_config(4, full_metrics, quantity, distance)).run();
    assert(short_result.asset_count == 1);
    assert(full_result.asset_count == 1);
    const std::vector<std::string> shortened = lines(short_metrics);
    const std::vector<std::string> full = lines(full_metrics);
    assert(shortened.size() == 4U);
    assert(full.size() == 6U);
    for (std::size_t index = 0; index < shortened.size(); ++index) {
        assert(shortened[index] == full[index]);
    }

    // The treatment amendments must be observational when the shared dealer
    // is disabled.  This is the source-level ordinary-market isolation check:
    // a full-day-equivalent horizon, relative quote/capacity sizing, and very
    // different dealer limits cannot alter a no-dealer state.
    auto baseline_off = make_config(
        4, root / "baseline_off.csv", quantity, distance);
    baseline_off.enable_shared_market_maker = false;
    baseline_off.stochastic_baseline_normalization_horizon_ns = 0;
    const dlob::FragmentedMpiResult baseline_result =
        dlob::FragmentedMpiSimulator(
            MPI_COMM_WORLD, std::move(baseline_off)).run();
    auto amended_off = make_config(
        4, root / "amended_off.csv", quantity, distance);
    amended_off.enable_shared_market_maker = false;
    amended_off.shared_quote_relative_to_asset = true;
    amended_off.shared_capacity_relative_to_asset = true;
    amended_off.shared_quote_multiplier = 7.0;
    amended_off.shared_global_risk_limit_per_asset = 10'000.0;
    amended_off.shared_local_inventory_scale = 5'000.0;
    const dlob::FragmentedMpiResult amended_result =
        dlob::FragmentedMpiSimulator(
            MPI_COMM_WORLD, std::move(amended_off)).run();
    assert(baseline_result.state_hash == amended_result.state_hash);
    assert(lines(root / "baseline_off.csv")
           == lines(root / "amended_off.csv"));
    assert(MPI_Finalize() == MPI_SUCCESS);
    return 0;
}
