#include "simulation/DistributedSimulator.hpp"

#include "agents/AgentPopulation.hpp"
#include "common/PerformanceMetrics.hpp"
#include "exchange/BackgroundHawkesAgent.hpp"
#include "exchange/DistributedLimitOrderBook.hpp"
#include "mpi/AsyncExchangeLoop.hpp"
#include "mpi/EventDrivenExchangeLoop.hpp"
#include "mpi/SharedMarketSnapshot.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace dlob::simulation {
namespace {

void check_mpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) throw std::runtime_error(std::string(operation) + " failed");
}

bool create_output_directory(MPI_Comm communicator,
                             const std::filesystem::path& output_dir,
                             int rank) {
    int success = 1;
    if (rank == 0) {
        std::error_code error;
        std::filesystem::create_directories(output_dir, error);
        if (error) success = 0;
    }
    check_mpi(MPI_Bcast(&success, 1, MPI_INT, 0, communicator),
              "MPI_Bcast(output directory status)");
    return success == 1;
}

void configure_data_paths(BackgroundHawkesConfig& config,
                          const std::filesystem::path& data_dir) {
    config.limit_buy_quantity_file = (data_dir / "limit_buy_quantity_distribution.txt").string();
    config.limit_sell_quantity_file = (data_dir / "limit_sell_quantity_distribution.txt").string();
    config.market_buy_quantity_file = (data_dir / "market_buy_quantity_distribution.txt").string();
    config.market_sell_quantity_file = (data_dir / "market_sell_quantity_distribution.txt").string();
    config.cancel_bid_quantity_file = (data_dir / "cancel_bid_quantity_distribution.txt").string();
    config.cancel_ask_quantity_file = (data_dir / "cancel_ask_quantity_distribution.txt").string();
    config.limit_buy_distance_file = (data_dir / "limit_buy_distance_distribution.txt").string();
    config.limit_sell_distance_file = (data_dir / "limit_sell_distance_distribution.txt").string();
    config.cancel_bid_distance_file = (data_dir / "cancel_bid_distance_distribution.txt").string();
    config.cancel_ask_distance_file = (data_dir / "cancel_ask_distance_distribution.txt").string();
}

const char* communication_mode_name(CommunicationMode mode) {
    return mode == CommunicationMode::EventDrivenBatched
        ? "event_driven_batched" : "fixed_window_legacy";
}

void write_summary(const SimulatorResult& result,
                   const SimulatorConfig& config,
                   int world_size,
                   const PopulationSummary& population,
                   bool shared_snapshot_enabled) {
    const auto path = config.output_directory / "simulation_summary.csv";
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write " + path.string());
    out << "ranks,duration_seconds,communication_mode,sync_window_us,"
           "shared_snapshot,fixed_hawkes_activity_scale,"
           "market_makers,momentum,informed,institutional,total_agents,"
           "generated_strategic,processed_strategic,processed_background,"
           "pending_end,peak_pending,activations,wall_seconds,terminated_early,"
           "final_simulated_time_ns,termination_reason,structurally_valid,mean_spread_ticks,mean_bid_depth,mean_ask_depth,mid_move_rate,"
           "return_variance,return_kurtosis,absolute_return_acf1\n";
    out << std::setprecision(12)
        << world_size << ',' << config.duration_seconds << ','
        << communication_mode_name(config.communication_mode) << ','
        << config.sync_window_us << ',' << (shared_snapshot_enabled ? 1 : 0) << ','
        << fixed_background_hawkes_activity_scale << ','
        << population.market_makers << ',' << population.momentum << ','
        << population.informed << ',' << population.institutional << ','
        << population.total() << ',' << result.generated_strategic << ','
        << result.processed_strategic << ',' << result.processed_background << ','
        << result.pending_end << ',' << result.peak_pending << ',' << result.activations << ','
        << result.wall_seconds << ',' << (result.terminated_early ? 1 : 0) << ','
        << result.final_simulated_time_ns << ',' << result.termination_reason << ','
        << (result.structurally_valid ? 1 : 0) << ','
        << result.record.market.mean_spread_ticks << ','
        << result.record.market.mean_bid_depth << ','
        << result.record.market.mean_ask_depth << ','
        << result.record.market.mid_move_rate << ','
        << result.record.market.return_variance << ','
        << result.record.market.return_kurtosis << ','
        << result.record.market.absolute_return_acf1 << '\n';
}

PopulationConfig make_population_config(const SimulatorConfig& config,
                                        std::int64_t start_ns,
                                        std::int64_t end_ns) {
    PopulationConfig p;
    p.market_makers = config.parameters.market_makers;
    p.momentum_traders = config.parameters.momentum_traders;
    p.informed_traders = config.parameters.informed_traders;
    p.institutional_traders = config.parameters.institutional_traders;
    p.population_scale = 1;
    p.tick_size = config.tick_size;
    p.simulation_start_ns = start_ns;
    p.simulation_end_ns = end_ns;
    p.seed = config.seed;
    p.market_maker_order_quantity = config.parameters.market_maker_order_quantity;
    p.market_maker_min_spread_ticks = config.parameters.market_maker_min_spread_ticks;
    p.market_maker_interval_ns = static_cast<std::int64_t>(std::llround(
        std::max(0.001, config.parameters.market_maker_interval_ms) * 1'000'000.0));
    p.market_maker_quote_skip_probability = config.parameters.market_maker_quote_skip_probability;
    p.momentum_rate_per_second = config.parameters.momentum_rate_per_second;
    p.momentum_threshold_ticks = config.parameters.momentum_threshold_ticks;
    p.momentum_order_quantity = config.parameters.momentum_order_quantity;
    p.informed_rate_per_second = config.parameters.informed_rate_per_second;
    p.informed_signal_noise_ticks = config.parameters.informed_signal_noise_ticks;
    p.informed_trade_threshold_ticks = config.parameters.informed_trade_threshold_ticks;
    p.informed_base_quantity = config.parameters.informed_base_quantity;
    p.institutional_rate_per_second = config.parameters.institutional_rate_per_second;
    p.institutional_participation_cap = config.parameters.institutional_participation_cap;
    p.market_maker_batch_horizon_ns =
        static_cast<std::int64_t>(std::max(0, config.market_maker_batch_horizon_us)) * 1'000LL;
    p.momentum_batch_horizon_ns =
        static_cast<std::int64_t>(std::max(0, config.momentum_batch_horizon_us)) * 1'000LL;
    p.informed_batch_horizon_ns =
        static_cast<std::int64_t>(std::max(0, config.informed_batch_horizon_us)) * 1'000LL;
    p.institutional_batch_horizon_ns =
        static_cast<std::int64_t>(std::max(0, config.institutional_batch_horizon_us)) * 1'000LL;
    return p;
}

SimulatorResult run_fixed_window_legacy(MPI_Comm communicator,
                                        const SimulatorConfig& config,
                                        int rank,
                                        int world_size,
                                        AgentPopulation& population,
                                        DistributedLimitOrderBook& book,
                                        BackgroundHawkesAgent& background,
                                        const std::vector<HawkesEvent>& hawkes_events,
                                        PerformanceMetrics& metrics,
                                        calibration::SimulationRecorder& recorder,
                                        std::int64_t start_ns,
                                        std::int64_t end_ns) {
    const std::int64_t window_ns = static_cast<std::int64_t>(config.sync_window_us) * 1'000LL;
    AsyncExchangeLoop loop(communicator, rank, world_size, book, background, hawkes_events,
                           metrics, rank == 0 ? &recorder : nullptr);

    std::mt19937_64 fundamental_rng(config.seed + 0xABCDEFULL);
    std::normal_distribution<double> fundamental_shock(0.0, 0.03);
    double fundamental_value = 2'203'550.0;
    std::uint64_t window_index = 0;
    std::int64_t next_sample_ns = 1'000'000'000LL;

    for (std::int64_t window_start = start_ns;
         window_start < end_ns;
         window_start += window_ns, ++window_index) {
        metrics.increment_windows();
        const std::int64_t window_end = std::min(end_ns, window_start + window_ns);

        if (rank == 0) {
            const double duration_ms = static_cast<double>(window_end - window_start) / 1e6;
            fundamental_value += fundamental_shock(fundamental_rng)
                * std::sqrt(std::max(0.0, duration_ms)) * config.tick_size;
            const MarketState opening = book.state(window_start, fundamental_value);
            ExchangeWindowResult window_result = loop.run_exchange_window(
                window_index, opening, window_end, fundamental_value, {});
            while (window_end >= next_sample_ns && next_sample_ns <= end_ns) {
                recorder.observe_state(window_result.closing_state);
                next_sample_ns += 1'000'000'000LL;
            }
        } else {
            const MarketState opening = loop.receive_market_state(window_index);
            std::vector<OrderMessage> local_orders;
            {
                ScopedStageTimer timer(metrics, TimingStage::AgentObserveAndGenerate);
                population.observe_market(opening);
                local_orders = population.generate_orders(window_start, window_end);
            }
            metrics.counters().strategic_orders_generated += local_orders.size();
            loop.post_order_batch(window_index, local_orders);
            std::vector<AgentReport> reports = loop.receive_report_batch(window_index);
            population.apply_reports(reports);
        }
    }

    SimulatorResult result;
    if (rank == 0) {
        result.processed_strategic = metrics.counters().strategic_orders_processed;
        result.processed_background = metrics.counters().background_orders_processed;
        result.pending_end = loop.pending_order_count();
        result.peak_pending = loop.peak_pending_order_count();
        result.final_simulated_time_ns = end_ns;
        result.record = recorder.finalize();
    }
    return result;
}

} // namespace

SimulatorResult run_distributed_simulator(MPI_Comm communicator,
                                          const SimulatorConfig& config) {
    int rank = 0;
    int world_size = 1;
    check_mpi(MPI_Comm_rank(communicator, &rank), "MPI_Comm_rank(simulator)");
    check_mpi(MPI_Comm_size(communicator, &world_size), "MPI_Comm_size(simulator)");
    if (config.duration_seconds <= 0) {
        throw std::invalid_argument("Simulator duration must be positive");
    }
    if (config.communication_mode == CommunicationMode::FixedWindowLegacy
        && config.sync_window_us <= 0) {
        throw std::invalid_argument("Legacy synchronization window must be positive");
    }
    if (config.write_files && !create_output_directory(communicator, config.output_directory, rank)) {
        throw std::runtime_error("Could not create simulator output directory");
    }

#if defined(_WIN32)
    _putenv_s("LOB_DATA_DIR", config.data_directory.string().c_str());
#else
    setenv("LOB_DATA_DIR", config.data_directory.string().c_str(), 1);
#endif

    const std::int64_t start_ns = 0;
    const std::int64_t end_ns = static_cast<std::int64_t>(config.duration_seconds) * 1'000'000'000LL;
    const PopulationConfig population_config = make_population_config(config, start_ns, end_ns);

    AgentPopulation population(rank, world_size, population_config);
    const PopulationSummary local_population = population.local_summary();
    const PopulationSummary global_population = population.global_summary();

    PerformanceMetrics metrics(rank, world_size, rank == 0, population.is_worker());
    metrics.set_local_population(local_population.market_makers, local_population.momentum,
                                 local_population.informed, local_population.institutional);

    DistributedLimitOrderBook book(config.tick_size);
    BackgroundHawkesConfig hawkes_config;
    hawkes_config.seed = config.seed;
    hawkes_config.tick_size = config.tick_size;
    configure_data_paths(hawkes_config, config.data_directory);
    BackgroundHawkesAgent background(hawkes_config);

    std::vector<HawkesEvent> hawkes_events;
    if (rank == 0) {
        book.seed_default_book(1.0);
        ScopedStageTimer timer(metrics, TimingStage::HawkesGeneration);
        hawkes_events = background.simulate(start_ns, end_ns);
    }

    calibration::SimulationRecorder recorder(config.seed, config.reservoir_capacity, config.tick_size);
    SharedMarketSnapshot shared_snapshot(
        communicator, rank, world_size,
        config.communication_mode == CommunicationMode::EventDrivenBatched
            && config.use_shared_market_snapshot);

    check_mpi(MPI_Barrier(communicator), "MPI_Barrier(simulator start)");
    const double wall_start = MPI_Wtime();

    SimulatorResult result;
    if (config.communication_mode == CommunicationMode::EventDrivenBatched) {
        EventDrivenExchangeLoop loop(communicator, rank, world_size, book, background,
                                     hawkes_events, population, metrics,
                                     rank == 0 ? &recorder : nullptr,
                                     shared_snapshot, end_ns, config.tick_size, config.seed,
                                     config.max_wall_seconds);
        const EventDrivenRunResult event_result = loop.run();
        if (rank == 0) {
            result.processed_strategic = metrics.counters().strategic_orders_processed;
            result.processed_background = metrics.counters().background_orders_processed;
            result.pending_end = event_result.pending_orders;
            result.peak_pending = event_result.peak_pending_orders;
            result.activations = event_result.activations;
            result.terminated_early = event_result.terminated_early;
            result.final_simulated_time_ns = event_result.final_time_ns;
            result.termination_reason = event_result.termination_reason;
            result.record = recorder.finalize();
        }
    } else {
        result = run_fixed_window_legacy(communicator, config, rank, world_size,
                                         population, book, background, hawkes_events,
                                         metrics, recorder, start_ns, end_ns);
    }

    check_mpi(MPI_Barrier(communicator), "MPI_Barrier(simulator end)");
    const double wall_seconds = MPI_Wtime() - wall_start;

    const unsigned long long local_generated =
        static_cast<unsigned long long>(metrics.counters().strategic_orders_generated);
    unsigned long long global_generated = 0;
    check_mpi(MPI_Reduce(&local_generated, &global_generated, 1,
                         MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, communicator),
              "MPI_Reduce(generated orders)");

    if (rank == 0) {
        result.generated_strategic = static_cast<std::uint64_t>(global_generated);
        result.wall_seconds = wall_seconds;
        const std::int64_t closing_time = result.final_simulated_time_ns > 0
            ? result.final_simulated_time_ns : end_ns;
        const MarketState closing = book.state(closing_time, 0.0);
        result.structurally_valid =
            !result.terminated_early
            && result.generated_strategic == result.processed_strategic + result.pending_end
            && result.record.market.snapshots > 0
            && closing.best_bid_ticks > 0
            && closing.best_ask_ticks > closing.best_bid_ticks;
    }

    if (config.write_files) {
        metrics.write_rank_csv(config.output_directory);
        const std::vector<RankProfile> profiles = gather_rank_profiles(
            metrics.profile(), rank, world_size, communicator);
        if (rank == 0) {
            write_combined_timing_csv(config.output_directory, profiles);
            write_timing_summary_csv(config.output_directory, profiles);
            write_rank_counters_csv(config.output_directory, profiles);
            write_summary(result, config, world_size, global_population,
                          shared_snapshot.enabled());
        }
    }
    return result;
}

} // namespace dlob::simulation
