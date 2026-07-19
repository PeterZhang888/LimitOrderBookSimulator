#include "agents/AgentPopulation.hpp"
#include "common/PerformanceMetrics.hpp"
#include "common/RunConfig.hpp"
#include "exchange/BackgroundHawkesAgent.hpp"
#include "exchange/DistributedLimitOrderBook.hpp"
#include "exchange/EventOrdering.hpp"
#include "mpi/MpiCompat.hpp"
#include "mpi/MpiTransport.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace dlob;

void check_mpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) throw std::runtime_error(std::string(operation) + " failed");
}

bool create_output_directory(const std::filesystem::path& output_dir,
                             int rank) {
    int success = 1;
    if (rank == 0) {
        std::error_code error;
        std::filesystem::create_directories(output_dir, error);
        if (error) {
            std::cerr << "Could not create output directory " << output_dir
                      << ": " << error.message() << '\n';
            success = 0;
        }
    }
    check_mpi(MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD),
              "MPI_Bcast(output directory status)");
    return success == 1;
}

struct MarketSummary {
    std::uint64_t valid_snapshots = 0;
    double spread_sum = 0.0;
    double bid_depth_sum = 0.0;
    double ask_depth_sum = 0.0;

    void add(const MarketState& state) {
        if (state.best_bid_ticks <= 0 || state.best_ask_ticks <= state.best_bid_ticks) return;
        spread_sum += static_cast<double>(state.best_ask_ticks - state.best_bid_ticks);
        bid_depth_sum += static_cast<double>(state.best_bid_depth);
        ask_depth_sum += static_cast<double>(state.best_ask_depth);
        ++valid_snapshots;
    }

    double mean_spread() const {
        return valid_snapshots > 0 ? spread_sum / static_cast<double>(valid_snapshots) : 0.0;
    }
    double mean_bid_depth() const {
        return valid_snapshots > 0 ? bid_depth_sum / static_cast<double>(valid_snapshots) : 0.0;
    }
    double mean_ask_depth() const {
        return valid_snapshots > 0 ? ask_depth_sum / static_cast<double>(valid_snapshots) : 0.0;
    }
};

void write_simulation_summary(const std::filesystem::path& output_dir,
                              const RunConfig& run,
                              int world_size,
                              const PopulationSummary& population,
                              const RankCounters& exchange_counters,
                              std::size_t pending_end,
                              double wall_seconds,
                              const MarketSummary& market,
                              const std::vector<RankProfile>& profiles) {
    const std::filesystem::path file = output_dir / "simulation_summary.csv";
    std::ofstream output(file);
    if (!output) throw std::runtime_error("Could not open simulation summary: " + file.string());

    double max_agent_seconds = 0.0;
    double max_communication_seconds = 0.0;
    double sum_communication_seconds = 0.0;
    std::uint64_t generated_global = 0;
    for (const RankProfile& profile : profiles) {
        max_agent_seconds = std::max(max_agent_seconds,
            profile.seconds[static_cast<std::size_t>(TimingStage::AgentObserveAndGenerate)]);
        const double communication =
            profile.seconds[static_cast<std::size_t>(TimingStage::BroadcastMarketState)]
            + profile.seconds[static_cast<std::size_t>(TimingStage::GatherOrderCounts)]
            + profile.seconds[static_cast<std::size_t>(TimingStage::GatherOrderPayload)]
            + profile.seconds[static_cast<std::size_t>(TimingStage::ScatterReportCounts)]
            + profile.seconds[static_cast<std::size_t>(TimingStage::ScatterReportPayload)];
        max_communication_seconds = std::max(max_communication_seconds, communication);
        sum_communication_seconds += communication;
        generated_global += profile.counters.strategic_orders_generated;
    }

    const std::uint64_t total_processed = exchange_counters.strategic_orders_processed
        + exchange_counters.background_orders_processed;
    const double throughput = wall_seconds > 0.0
        ? static_cast<double>(total_processed) / wall_seconds : 0.0;
    const double mean_communication = profiles.empty() ? 0.0
        : sum_communication_seconds / static_cast<double>(profiles.size());

    output << "profile,ranks,agent_workers,duration_seconds,sync_window_us,population_scale,"
              "market_makers,momentum,informed,institutional,total_agents,generated_strategic,"
              "processed_strategic,processed_background,fill_reports,order_result_reports,"
              "pending_end,peak_pending,wall_seconds,throughput_events_per_second,"
              "max_agent_seconds,max_communication_seconds,mean_communication_seconds,"
              "mean_spread,mean_best_bid_depth,mean_best_ask_depth\n";
    output << std::setprecision(12)
           << run.profile << ',' << world_size << ',' << (world_size == 1 ? 1 : world_size - 1) << ','
           << run.duration_seconds << ',' << run.sync_window_us << ',' << run.population_scale << ','
           << population.market_makers << ',' << population.momentum << ','
           << population.informed << ',' << population.institutional << ',' << population.total() << ','
           << generated_global << ',' << exchange_counters.strategic_orders_processed << ','
           << exchange_counters.background_orders_processed << ',' << exchange_counters.fill_reports << ','
           << exchange_counters.order_result_reports << ',' << pending_end << ','
           << exchange_counters.peak_pending_orders << ',' << wall_seconds << ',' << throughput << ','
           << max_agent_seconds << ',' << max_communication_seconds << ',' << mean_communication << ','
           << market.mean_spread() << ',' << market.mean_bid_depth() << ','
           << market.mean_ask_depth() << '\n';
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            dlob::print_usage(std::cout, argv[0]);
            return 0;
        }
    }

    const int init_status = MPI_Init(&argc, &argv);
    if (init_status != MPI_SUCCESS) {
        std::cerr << "MPI_Init failed\n";
        return 1;
    }

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const double initialization_start = MPI_Wtime();

    try {
        const RunConfig run = parse_run_config(argc, argv);
        if (run.expected_ranks > 0 && world_size != run.expected_ranks) {
            if (rank == 0) {
                std::cerr << "Expected " << run.expected_ranks << " MPI ranks but received "
                          << world_size << ". Refusing to run.\n";
            }
            MPI_Finalize();
            return 2;
        }

        const std::filesystem::path output_dir(run.output_dir);
        if (!create_output_directory(output_dir, rank)) {
            MPI_Finalize();
            return 3;
        }

        const std::int64_t start_time_ns = 0;
        const std::int64_t end_time_ns = static_cast<std::int64_t>(run.duration_seconds) * 1'000'000'000LL;
        const std::int64_t window_ns = static_cast<std::int64_t>(run.sync_window_us) * 1'000LL;

        PopulationConfig population_config;
        population_config.market_makers = run.market_makers;
        population_config.momentum_traders = run.momentum_traders;
        population_config.informed_traders = run.informed_traders;
        population_config.institutional_traders = run.institutional_traders;
        population_config.population_scale = run.population_scale;
        population_config.simulation_start_ns = start_time_ns;
        population_config.simulation_end_ns = end_time_ns;
        population_config.seed = run.seed;

        AgentPopulation population(rank, world_size, population_config);
        const PopulationSummary local_population = population.local_summary();
        const PopulationSummary global_population = population.global_summary();

        PerformanceMetrics metrics(rank, world_size, rank == 0, population.is_worker());
        metrics.set_local_population(local_population.market_makers, local_population.momentum,
                                     local_population.informed, local_population.institutional);

        DistributedLimitOrderBook book(population_config.tick_size);
        BackgroundHawkesConfig hawkes_config;
        hawkes_config.seed = run.seed;
        hawkes_config.tick_size = population_config.tick_size;
        BackgroundHawkesAgent background(hawkes_config);

        std::vector<HawkesEvent> hawkes_events;
        if (rank == 0) {
            book.seed_default_book(1.0);
            ScopedStageTimer timer(metrics, TimingStage::HawkesGeneration);
            hawkes_events = background.simulate(start_time_ns, end_time_ns);
        }
        metrics.set(TimingStage::Initialization, MPI_Wtime() - initialization_start);

        if (rank == 0) {
            std::cout << "Distributed Hawkes + heterogeneous-agent LOB simulator\n"
                      << "Profile: " << run.profile << '\n'
                      << "MPI ranks: " << world_size << " (1 exchange + "
                      << (world_size == 1 ? 1 : world_size - 1) << " worker(s))\n"
                      << "Duration: " << run.duration_seconds << " seconds\n"
                      << "Synchronization window: " << run.sync_window_us << " us\n"
                      << "Strategic agents: " << global_population.total() << '\n'
                      << "  market makers: " << global_population.market_makers << '\n'
                      << "  momentum: " << global_population.momentum << '\n'
                      << "  informed: " << global_population.informed << '\n'
                      << "  institutional: " << global_population.institutional << '\n'
                      << "Output directory: " << output_dir << '\n';
            if (world_size == 1) {
                std::cout << "Single-process correctness mode: not an MPI scaling result.\n";
            }
        }

        std::size_t next_hawkes = 0;
        std::optional<OrderMessage> cached_background_message;
        std::mt19937_64 fundamental_rng(run.seed + 0xABCDEFULL);
        std::normal_distribution<double> fundamental_shock(0.0, 0.03);
        double fundamental_value = 2'203'550.0;

        std::vector<OrderMessage> pending_orders;
        pending_orders.reserve(100'000);
        std::uint64_t background_sequence = 1;
        MarketSummary market_summary;

        {
            const double start = MPI_Wtime();
            check_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(start)");
            metrics.add(TimingStage::StartBarrier, MPI_Wtime() - start);
        }
        const double wall_start = MPI_Wtime();

        for (std::int64_t window_start = start_time_ns;
             window_start < end_time_ns;
             window_start += window_ns) {
            metrics.increment_windows();
            const std::int64_t window_end = std::min(end_time_ns, window_start + window_ns);
            MarketState snapshot{};

            if (rank == 0) {
                ScopedStageTimer timer(metrics, TimingStage::MarketStateBuild);
                const double duration_ms = static_cast<double>(window_end - window_start) / 1e6;
                fundamental_value += fundamental_shock(fundamental_rng)
                    * std::sqrt(std::max(0.0, duration_ms)) * population_config.tick_size;
                snapshot = book.state(window_start, fundamental_value);
            }

            {
                ScopedStageTimer timer(metrics, TimingStage::BroadcastMarketState);
                check_mpi(MPI_Bcast(&snapshot, static_cast<int>(sizeof(MarketState)),
                                    MPI_BYTE, 0, MPI_COMM_WORLD),
                          "MPI_Bcast(market state)");
            }

            std::vector<OrderMessage> local_orders;
            {
                ScopedStageTimer timer(metrics, TimingStage::AgentObserveAndGenerate);
                population.observe_market(snapshot);
                local_orders = population.generate_orders(window_start, window_end);
            }
            metrics.counters().strategic_orders_generated += local_orders.size();

            std::vector<OrderMessage> gathered = gather_orders(local_orders, rank, world_size, metrics);
            std::vector<AgentReport> reports;

            if (rank == 0) {
                {
                    ScopedStageTimer timer(metrics, TimingStage::QueuePartition);
                    pending_orders.insert(pending_orders.end(), gathered.begin(), gathered.end());
                    metrics.counters().peak_pending_orders = std::max<std::uint64_t>(
                        metrics.counters().peak_pending_orders,
                        static_cast<std::uint64_t>(pending_orders.size()));
                }

                std::vector<OrderMessage> due;
                {
                    ScopedStageTimer timer(metrics, TimingStage::QueuePartition);
                    std::vector<OrderMessage> future;
                    due.reserve(pending_orders.size());
                    future.reserve(pending_orders.size());
                    for (const OrderMessage& message : pending_orders) {
                        if (message.arrival_time_ns < window_end) due.push_back(message);
                        else future.push_back(message);
                    }
                    pending_orders.swap(future);
                }

                {
                    ScopedStageTimer timer(metrics, TimingStage::EventSort);
                    std::sort(due.begin(), due.end(), order_before);
                }

                {
                    ScopedStageTimer timer(metrics, TimingStage::MatchingEngine);
                    std::size_t order_index = 0;
                    while (order_index < due.size()
                           || cached_background_message.has_value()
                           || (next_hawkes < hawkes_events.size()
                               && hawkes_events[next_hawkes].time_ns < window_end)) {
                        const bool have_order = order_index < due.size();
                        const bool have_hawkes_event = next_hawkes < hawkes_events.size()
                            && hawkes_events[next_hawkes].time_ns < window_end;

                        if (!cached_background_message.has_value() && have_hawkes_event
                            && (!have_order
                                || hawkes_events[next_hawkes].time_ns <= due[order_index].arrival_time_ns)) {
                            const HawkesEvent& event = hawkes_events[next_hawkes];
                            const MarketState event_state = book.state(event.time_ns, fundamental_value);
                            cached_background_message = background.make_order(
                                event, event_state, background_sequence++);
                        }

                        const bool background_due = cached_background_message.has_value();
                        if (background_due
                            && (!have_order
                                || order_before(*cached_background_message, due[order_index]))) {
                            book.apply(*cached_background_message);
                            cached_background_message.reset();
                            ++next_hawkes;
                            ++metrics.counters().background_orders_processed;
                        } else if (have_order) {
                            book.apply(due[order_index++]);
                            ++metrics.counters().strategic_orders_processed;
                        } else if (have_hawkes_event) {
                            // This path is reached only when no strategic order was available
                            // to trigger construction of the next background message.
                            const HawkesEvent& event = hawkes_events[next_hawkes];
                            const MarketState event_state = book.state(event.time_ns, fundamental_value);
                            cached_background_message = background.make_order(
                                event, event_state, background_sequence++);
                        } else {
                            break;
                        }
                    }
                }

                reports = book.take_reports();
                metrics.counters().reports_created += reports.size();
                for (const AgentReport& report : reports) {
                    if (report.kind == ReportKind::Fill) ++metrics.counters().fill_reports;
                    else ++metrics.counters().order_result_reports;
                }
                market_summary.add(book.state(window_end, fundamental_value));
            }

            std::vector<AgentReport> local_reports = scatter_reports(
                reports, rank, world_size, metrics);
            {
                ScopedStageTimer timer(metrics, TimingStage::ApplyReports);
                population.apply_reports(local_reports);
            }
        }

        {
            const double start = MPI_Wtime();
            check_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(end)");
            metrics.add(TimingStage::EndBarrier, MPI_Wtime() - start);
        }
        const double total_wall_seconds = MPI_Wtime() - wall_start;
        metrics.set(TimingStage::TotalWall, total_wall_seconds);

        // Every rank writes a unique file; no shared-file write contention occurs.
        metrics.write_rank_csv(output_dir);
        const std::vector<RankProfile> profiles = gather_rank_profiles(
            metrics.profile(), rank, world_size);

        if (rank == 0) {
            write_combined_timing_csv(output_dir, profiles);
            write_timing_summary_csv(output_dir, profiles);
            write_rank_counters_csv(output_dir, profiles);
            write_simulation_summary(output_dir, run, world_size, global_population,
                                     metrics.counters(), pending_orders.size(),
                                     total_wall_seconds, market_summary, profiles);

            const std::uint64_t total_processed = metrics.counters().strategic_orders_processed
                + metrics.counters().background_orders_processed;
            const double throughput = total_wall_seconds > 0.0
                ? static_cast<double>(total_processed) / total_wall_seconds : 0.0;
            std::cout << std::fixed << std::setprecision(6)
                      << "Completed.\n"
                      << "Wall seconds: " << total_wall_seconds << '\n'
                      << "Processed strategic messages: "
                      << metrics.counters().strategic_orders_processed << '\n'
                      << "Processed Hawkes messages: "
                      << metrics.counters().background_orders_processed << '\n'
                      << "Pending messages after horizon: " << pending_orders.size() << '\n'
                      << "Peak pending queue: " << metrics.counters().peak_pending_orders << '\n'
                      << "Throughput: " << throughput << " events/s\n"
                      << "Mean spread: " << market_summary.mean_spread() << '\n'
                      << "Timing files written to: " << output_dir << '\n';
        }

        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Rank " << rank << " error: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        return 1;
    }
}
