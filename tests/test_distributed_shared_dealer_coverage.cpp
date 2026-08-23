#include "simulation/DistributedMarketSimulator.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Row = std::unordered_map<std::string, std::string>;

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream input(line);
    std::string field;
    while (std::getline(input, field, ',')) fields.push_back(field);
    return fields;
}

std::vector<Row> read_rows(const std::filesystem::path& path) {
    std::ifstream input(path);
    assert(input);
    std::string line;
    assert(std::getline(input, line));
    const std::vector<std::string> fields = split(line);
    std::vector<Row> rows;
    while (std::getline(input, line)) {
        const std::vector<std::string> values = split(line);
        assert(values.size() == fields.size());
        Row row;
        for (std::size_t index = 0; index < fields.size(); ++index) {
            row.emplace(fields[index], values[index]);
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

double value(const Row& row, const char* field) {
    const auto found = row.find(field);
    assert(found != row.end());
    return std::stod(found->second);
}

} // namespace

int main(int argc, char** argv) {
    const int init_status = MPI_Init(&argc, &argv);
    assert(init_status == MPI_SUCCESS);

    const std::filesystem::path root =
        std::filesystem::temp_directory_path()
        / "dlob_shared_dealer_coverage";
    std::filesystem::create_directories(root);
    const std::filesystem::path quantity = root / "quantity.csv";
    const std::filesystem::path distance = root / "distance.csv";
    const std::filesystem::path metrics = root / "metrics.csv";
    const std::filesystem::path shock_targets = root / "shock_targets.csv";
    {
        std::ofstream output(quantity);
        assert(output);
        output << "quantity,count\n1,1\n";
    }
    {
        std::ofstream output(distance);
        assert(output);
        output << "distance_ticks,count\n1,1\n";
    }

    dlob::SimulationConfig config;
    // Exercise the exact empirical cohort size used by the financial study.
    // This prevents an implementation that quotes only a small hard-coded
    // subset from satisfying the source-only acceptance test.
    config.asset_count = 1'480;
    config.duration_seconds = 4;
    config.decision_window_ns = 1'000'000'000LL;
    config.global_metrics_interval_ns = 1'000'000'000LL;
    config.tick_size = 100;
    config.seed = 20200130;
    config.background_model = "queue-reactive-v1";
    config.enable_local_market_makers = false;
    config.enable_value_agents = false;
    config.enable_shared_market_maker = true;
    config.enable_global_shared_capacity = true;
    config.shared_quote_relative_to_asset = true;
    config.shared_capacity_relative_to_asset = true;
    config.shared_quote_multiplier = 2.0;
    config.shared_quote_levels = 1;
    config.shared_local_inventory_scale = 800.0;
    config.shared_global_risk_limit_per_asset = 1600.0;
    config.shared_capacity_threshold = 0.5;
    config.shared_minimum_quote_scale = 0.05;
    config.enable_shock = true;
    config.shock_time_ns = 2'000'000'000LL;
    config.shock_asset_fraction = 0.10;
    config.shock_target_seed = 314159;
    config.shock_reference_bid_depth_multiple = 3.0;
    config.shock_inventory_adverse = true;
    config.metrics_csv = metrics.string();
    config.shock_targets_csv = shock_targets.string();

    for (int index = 0; index < config.asset_count; ++index) {
        dlob::MultiAssetBookConfig book;
        book.symbol = "TEST" + std::to_string(index);
        book.fundamental_price_ticks = 10'000.0;
        book.fundamental_move_probability_per_second = 0.0;
        book.initial_best_bid_ticks = 9'900;
        book.initial_best_ask_ticks = 10'100;
        book.initial_best_bid_depth = 100;
        book.initial_best_ask_depth = 100;
        book.beta = 1.0;
        book.market_maker_quote_quantity = 100;
        book.target_spread_ticks = 2;
        config.asset_configs.push_back(std::move(book));

        dlob::BackgroundHawkesConfig background;
        background.activity_scale = 1.0;
        background.mu.fill(1.0e-12);
        for (auto& row : background.alpha) row.fill(0.0);
        background.seed = 1000U + static_cast<std::uint64_t>(index);
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

    const dlob::SimulationResult result =
        dlob::DistributedMarketSimulator(MPI_COMM_WORLD, std::move(config)).run();
    assert(result.asset_count == 1'480);
    assert(result.shock_target_assets == 148U);
    // All 148 target books receive exactly 3 x their held-out opening bid
    // depth, independent of the dealer treatment.
    assert(result.shock_requested_quantity == 44'400U);
    // The opening anonymous queue is followed by the dealer's persistent BBO
    // quote, so this stress must reach and execute against the dealer.
    assert(result.shock_shared_mm_quantity == 29'600U);
    assert(result.shock_background_quantity == 14'800U);
    assert(result.shock_local_mm_quantity == 0U);
    assert(result.shock_value_agent_quantity == 0U);
    assert(result.shock_other_quantity == 0U);
    assert(result.shock_executed_quantity
           == result.shock_shared_mm_quantity
            + result.shock_background_quantity
            + result.shock_local_mm_quantity
            + result.shock_value_agent_quantity
            + result.shock_other_quantity);
    assert(result.shared_buy_quantity == 29'600U);
    assert(result.shared_sell_quantity == 0U);
    assert(result.shared_fill_count == 148U);
    assert(result.shared_terminal_absolute_inventory == 29'600U);
    assert(result.terminal_fallback_asset_count <= 1'480U);
    assert(result.terminal_fallback_quantity
           == result.shared_unliquidated_terminal_quantity);
    assert(result.terminal_fallback_from_external_quote
               + result.terminal_fallback_from_reference_value
           == result.terminal_fallback_quantity);
    assert(std::isfinite(result.shared_signed_mark_to_mid_pnl_usd));
    assert(std::isfinite(result.shared_signed_liquidation_pnl_usd));
    assert(result.shared_terminal_liquidation_cost_usd >= 0.0);
    // 2,960 opening quotes, 148 shock orders, and 148 depleted-bid top-ups.
    // Any unconditional one-second cancel/recreate cycle would make this
    // count larger and would silently forfeit price--time priority.
    assert(result.processed_orders == 3'256U);

    const std::vector<Row> rows = read_rows(metrics);
    const auto at_shock = std::find_if(
        rows.begin(), rows.end(), [](const Row& row) {
            return std::abs(value(row, "time_seconds") - 2.0) < 1.0e-12;
        });
    assert(at_shock != rows.end());
    assert(value(*at_shock, "shared_requested_active_asset_fraction") == 1.0);
    assert(value(*at_shock, "shared_requested_two_sided_asset_fraction") == 1.0);
    assert(value(*at_shock, "shared_active_asset_fraction") == 1.0);
    assert(value(*at_shock, "shared_two_sided_active_asset_fraction") == 1.0);
    assert(value(*at_shock, "shared_quote_scale") == 1.0);
    assert(value(*at_shock, "shared_at_best_bid_asset_fraction") == 1.0);
    assert(value(*at_shock, "shared_at_best_ask_asset_fraction") == 1.0);
    assert(value(*at_shock, "shared_best_bid_depth") == 296'000.0);
    assert(value(*at_shock, "shared_best_ask_depth") == 296'000.0);
    assert(std::abs(
        value(*at_shock, "shared_bbo_depth_participation") - 2.0 / 3.0)
        < 1.0e-9);

    const std::vector<Row> target_rows = read_rows(shock_targets);
    assert(target_rows.size() == 1'480U);
    std::size_t target_count = 0;
    for (const Row& row : target_rows) {
        if (value(row, "is_shock_target") == 0.0) continue;
        ++target_count;
        assert(value(row, "requested_quantity") == 300.0);
        assert(value(row, "requested_sell_quantity") == 300.0);
        assert(value(row, "requested_buy_quantity") == 0.0);
        assert(row.at("shock_side") == "sell");
        assert(value(row, "pre_shock_shared_inventory") == 0.0);
        assert(row.at("direction_rule") == "inventory_adverse");
    }
    assert(target_count == 148U);

    const int finalize_status = MPI_Finalize();
    assert(finalize_status == MPI_SUCCESS);
    return 0;
}
