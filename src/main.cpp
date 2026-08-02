#include "calibration/CalibrationParameters.hpp"
#include "common/RunConfig.hpp"
#include "mpi/MpiCompat.hpp"
#include "simulation/DistributedSimulator.hpp"

#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "MPI_Init failed\n";
        return 1;
    }
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    try {
        const dlob::RunConfig run = dlob::parse_run_config(argc, argv);
        if (run.expected_ranks > 0 && run.expected_ranks != world_size) {
            if (rank == 0) {
                std::cerr << "Expected " << run.expected_ranks << " ranks, received "
                          << world_size << '\n';
            }
            MPI_Finalize();
            return 2;
        }

        dlob::simulation::SimulatorConfig config;
        config.duration_seconds = run.duration_seconds;
        config.sync_window_us = run.sync_window_us;
        config.seed = run.seed;
        config.output_directory = run.output_dir;
        config.write_files = true;
        config.data_directory = run.data_dir;
        config.parameters.market_makers = run.market_makers;
        config.parameters.momentum_traders = run.momentum_traders * run.population_scale;
        config.parameters.informed_traders = run.informed_traders * run.population_scale;
        config.parameters.institutional_traders = run.institutional_traders * run.population_scale;
        config.parameters.market_maker_interval_ms = run.market_maker_interval_ms;
        config.parameters.market_maker_min_spread_ticks = run.market_maker_min_spread_ticks;
        config.parameters.momentum_rate_per_second = run.momentum_rate_per_second;
        config.parameters.momentum_threshold_ticks = run.momentum_threshold_ticks;
        config.parameters.informed_rate_per_second = run.informed_rate_per_second;
        config.parameters.informed_signal_precision = run.informed_signal_precision;
        config.parameters.informed_signal_noise_ticks = 1.0 / run.informed_signal_precision;
        config.parameters.institutional_rate_per_second = run.institutional_rate_per_second;
        config.parameters.institutional_participation_cap = run.institutional_participation_cap;
        config.communication_mode = run.event_driven
            ? dlob::simulation::CommunicationMode::EventDrivenBatched
            : dlob::simulation::CommunicationMode::FixedWindowLegacy;
        config.use_shared_market_snapshot = run.use_shared_snapshot;
        config.market_maker_batch_horizon_us = run.market_maker_batch_horizon_us;
        config.momentum_batch_horizon_us = run.momentum_batch_horizon_us;
        config.informed_batch_horizon_us = run.informed_batch_horizon_us;
        config.institutional_batch_horizon_us = run.institutional_batch_horizon_us;
        config.max_wall_seconds = run.max_wall_seconds;

        const dlob::simulation::SimulatorResult result =
            dlob::simulation::run_distributed_simulator(MPI_COMM_WORLD, config);
        if (rank == 0) {
            const std::uint64_t total = result.processed_strategic + result.processed_background;
            const double throughput = result.wall_seconds > 0.0
                ? static_cast<double>(total) / result.wall_seconds : 0.0;
            std::cout << std::fixed << std::setprecision(6)
                      << "Distributed Hawkes + heterogeneous-agent LOB simulator\n"
                      << "Communication mode: " << (run.event_driven ? "event_driven_batched" : "fixed_window_legacy") << '\n'
                      << "MPI ranks: " << world_size << '\n'
                      << "Fixed Hawkes activity scale: "
                      << dlob::calibration::fixed_hawkes_activity_scale << '\n'
                      << "Completed.\n"
                      << "Wall seconds: " << result.wall_seconds << '\n'
                      << "Processed strategic messages: " << result.processed_strategic << '\n'
                      << "Processed Hawkes messages: " << result.processed_background << '\n'
                      << "Pending messages after horizon: " << result.pending_end << '\n'
                      << "Peak pending queue: " << result.peak_pending << '\n'
                      << "Terminated early: " << (result.terminated_early ? "yes" : "no") << '\n'
                      << "Termination reason: " << result.termination_reason << '\n'
                      << "Final simulated time (ns): " << result.final_simulated_time_ns << '\n'
                      << "Throughput: " << throughput << " events/s\n"
                      << "Mean spread: " << result.record.market.mean_spread_ticks << " ticks\n"
                      << "Results: " << std::filesystem::path(run.output_dir) << '\n';
        }
        MPI_Finalize();
        return result.structurally_valid || rank != 0 ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "Rank " << rank << " error: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        return 1;
    }
}
