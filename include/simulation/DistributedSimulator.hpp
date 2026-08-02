// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#pragma once

#include "calibration/CalibrationParameters.hpp"
#include "calibration/SimulationRecorder.hpp"
#include "mpi/MpiCompat.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace dlob::simulation {

enum class CommunicationMode {
    EventDrivenBatched,
    FixedWindowLegacy
};

struct SimulatorConfig {
    int duration_seconds = 23'400;
    int sync_window_us = 10'000; // legacy benchmark mode only
    int tick_size = 100;
    std::uint64_t seed = 12345;
    calibration::PhysicalParameters parameters{};
    std::filesystem::path data_directory = "data";
    bool write_files = false;
    std::filesystem::path output_directory;
    std::size_t reservoir_capacity = 8192;

    CommunicationMode communication_mode = CommunicationMode::EventDrivenBatched;
    bool use_shared_market_snapshot = true;

    int market_maker_batch_horizon_us = 100'000;
    int momentum_batch_horizon_us = 250'000;
    int informed_batch_horizon_us = 250'000;
    int institutional_batch_horizon_us = 1'000'000;

    // Optional safety guard for expensive calibration candidates. Zero disables
    // the guard. The event-driven exchange sends an orderly STOP packet to all
    // worker ranks when the limit is reached.
    double max_wall_seconds = 0.0;
};

struct SimulatorResult {
    bool structurally_valid = false;
    std::uint64_t generated_strategic = 0;
    std::uint64_t processed_strategic = 0;
    std::uint64_t processed_background = 0;
    std::uint64_t pending_end = 0;
    std::uint64_t peak_pending = 0;
    std::uint64_t activations = 0;
    double wall_seconds = 0.0;
    bool terminated_early = false;
    std::int64_t final_simulated_time_ns = 0;
    std::string termination_reason;
    calibration::SimulationRecord record{};
};

SimulatorResult run_distributed_simulator(MPI_Comm communicator,
                                          const SimulatorConfig& config);

} // namespace dlob::simulation
