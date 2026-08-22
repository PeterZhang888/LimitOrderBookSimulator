#include "simulation/DistributedMarketSimulator.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::string contents(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    assert(input);
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

dlob::SimulationConfig make_config(
    const std::filesystem::path& root,
    const std::string& label) {
    dlob::SimulationConfig config;
    config.asset_count = 64;
    config.duration_seconds = 4;
    config.decision_window_ns = 1'000'000'000LL;
    config.global_metrics_interval_ns = 1'000'000'000LL;
    config.asset_summary_interval_ns = 1'000'000'000LL;
    config.return_panel_interval_ns = 1'000'000'000LL;
    config.background_model = "queue-reactive-v1";
    config.enable_local_market_makers = true;
    config.enable_value_agents = false;
    config.enable_shared_market_maker = true;
    config.shared_global_risk_limit_per_asset = 1'600.0;
    config.shared_local_inventory_scale = 800.0;
    config.enable_shock = true;
    config.shock_time_ns = 2'000'000'000LL;
    config.shock_target_count = 6;
    config.shock_inventory_adverse = true;
    config.metrics_csv = (root / (label + "_metrics.csv")).string();
    config.cluster_metrics_csv =
        (root / (label + "_clusters.csv")).string();
    config.return_panel_prefix =
        (root / (label + "_twice_midpoint")).string();

    const std::filesystem::path quantity = root / "quantity.csv";
    const std::filesystem::path distance = root / "distance.csv";
    for (int index = 0; index < config.asset_count; ++index) {
        dlob::MultiAssetBookConfig book;
        book.symbol = "ASSET" + std::to_string(index);
        book.fundamental_price_ticks = 10'000.0;
        book.fundamental_move_probability_per_second = 0.0;
        book.initial_best_bid_ticks = 9'900;
        book.initial_best_ask_ticks = 10'100;
        book.initial_best_bid_depth = 100 + index;
        book.initial_best_ask_depth = 100 + index;
        book.beta = 1.0;
        book.market_maker_quote_quantity = 100;
        book.target_spread_ticks = 2;
        config.asset_configs.push_back(std::move(book));
        config.shock_cluster_ids.push_back(index % 4);

        dlob::BackgroundHawkesConfig background;
        background.activity_scale = 1.0;
        background.mu.fill(1.0e-12);
        for (auto& row : background.alpha) row.fill(0.0);
        background.seed = 1'000U + static_cast<std::uint64_t>(index);
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
    assert(MPI_Init(&argc, &argv) == MPI_SUCCESS);
    const std::filesystem::path root =
        std::filesystem::temp_directory_path()
        / "dlob_distributed_optimization_equivalence";
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "quantity.csv");
        assert(output);
        output << "quantity,count\n1,1\n";
    }
    {
        std::ofstream output(root / "distance.csv");
        assert(output);
        output << "distance_ticks,count\n1,1\n";
    }

    dlob::SimulationConfig baseline = make_config(root, "baseline");
    baseline.worker_threads = 4;
    const dlob::SimulationResult reference =
        dlob::DistributedMarketSimulator(MPI_COMM_WORLD, std::move(baseline)).run();

    const auto verify = [&](const std::string& label,
                            dlob::SimulationConfig candidate) {
        const dlob::SimulationResult result =
            dlob::DistributedMarketSimulator(
                MPI_COMM_WORLD, std::move(candidate)).run();
        assert(result.processed_orders == reference.processed_orders);
        assert(result.trades == reference.trades);
        assert(result.shock_executed_quantity
               == reference.shock_executed_quantity);
        assert(contents(root / (label + "_metrics.csv"))
               == contents(root / "baseline_metrics.csv"));
        assert(contents(root / (label + "_clusters.csv"))
               == contents(root / "baseline_clusters.csv"));
        assert(contents(root / (label + "_twice_midpoint.rank00000.csv"))
               == contents(root / "baseline_twice_midpoint.rank00000.csv"));
    };

    const std::string return_panel = contents(
        root / "baseline_twice_midpoint.rank00000.csv");
    assert(return_panel.starts_with("time_seconds,ASSET0,ASSET1,"));
    assert(return_panel.find("0.000000000,20000,20000,")
           != std::string::npos);

    dlob::SimulationConfig guided = make_config(root, "guided");
    guided.worker_threads = 4;
    guided.openmp_schedule = dlob::OpenMpSchedule::Guided;
    verify("guided", std::move(guided));

    dlob::SimulationConfig weighted_static = make_config(root, "static");
    weighted_static.worker_threads = 4;
    weighted_static.openmp_schedule = dlob::OpenMpSchedule::Static;
    verify("static", std::move(weighted_static));

    dlob::SimulationConfig measured_static =
        make_config(root, "weighted_static");
    measured_static.worker_threads = 4;
    measured_static.openmp_schedule =
        dlob::OpenMpSchedule::WeightedStatic;
    measured_static.realized_partition_costs.resize(64);
    for (std::size_t index = 0;
         index < measured_static.realized_partition_costs.size(); ++index) {
        measured_static.realized_partition_costs[index] =
            static_cast<double>(1U + index % 11U);
    }
    verify("weighted_static", std::move(measured_static));

    dlob::SimulationConfig persistent = make_config(root, "persistent");
    persistent.worker_threads = 4;
    persistent.persistent_openmp_team = true;
    verify("persistent", std::move(persistent));

    dlob::SimulationConfig window_only = make_config(root, "window_only");
    window_only.worker_threads = 4;
    window_only.openmp_window_only = true;
    verify("window_only", std::move(window_only));

    dlob::SimulationConfig initialization = make_config(root, "init");
    initialization.worker_threads = 4;
    initialization.parallel_asset_initialization = true;
    verify("init", std::move(initialization));

    dlob::SimulationConfig reductions = make_config(root, "reductions");
    reductions.worker_threads = 4;
    reductions.parallel_boundary_reductions = true;
    verify("reductions", std::move(reductions));

    dlob::SimulationConfig metrics = make_config(root, "parallel_metrics");
    metrics.worker_threads = 4;
    metrics.parallel_metric_scans = true;
    verify("parallel_metrics", std::move(metrics));

    dlob::SimulationConfig fused = make_config(root, "fused_metrics");
    fused.worker_threads = 4;
    fused.parallel_metric_scans = true;
    fused.fuse_metric_cluster_scans = true;
    verify("fused_metrics", std::move(fused));

    dlob::SimulationConfig realized = make_config(root, "realized_lpt");
    realized.worker_threads = 4;
    realized.partition_mode = dlob::PartitionMode::RealizedCostLpt;
    realized.realized_partition_costs.resize(64);
    for (std::size_t index = 0;
         index < realized.realized_partition_costs.size(); ++index) {
        realized.realized_partition_costs[index] =
            static_cast<double>(64U - index);
    }
    verify("realized_lpt", std::move(realized));

    dlob::SimulationConfig profiled = make_config(root, "profiled");
    profiled.worker_threads = 4;
    profiled.asset_work_csv = (root / "asset_work.csv").string();
    profiled.boundary_arrival_csv =
        (root / "boundary_arrivals.csv").string();
    verify("profiled", std::move(profiled));
    const std::string asset_work = contents(root / "asset_work.csv");
    const std::string arrivals = contents(root / "boundary_arrivals.csv");
    assert(asset_work.starts_with(
        "asset_id,symbol,cluster_id,is_shock_target,owner_rank,"
        "processed_orders,"));
    assert(arrivals.starts_with(
        "boundary_index,time_seconds,rank,arrival_seconds,"
        "work_interval_seconds,work_interval_spread_seconds,"));

    dlob::SimulationConfig combined = make_config(root, "combined");
    combined.worker_threads = 4;
    combined.partition_mode = dlob::PartitionMode::RealizedCostLpt;
    combined.realized_partition_costs.assign(64, 1.0);
    combined.persistent_openmp_team = true;
    combined.parallel_asset_initialization = true;
    combined.parallel_boundary_reductions = true;
    combined.parallel_metric_scans = true;
    combined.fuse_metric_cluster_scans = true;
    verify("combined", std::move(combined));

    assert(MPI_Finalize() == MPI_SUCCESS);
    return 0;
}
