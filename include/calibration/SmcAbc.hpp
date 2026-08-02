#pragma once

#include "calibration/CalibrationParameters.hpp"
#include "mpi/MpiCompat.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace dlob::calibration {

struct SmcAbcConfig {
    std::size_t particles_per_wave = 32;
    int max_waves = 2;
    double tolerance_quantile = 0.50;
    int ranks_per_simulation = 5;
    int duration_seconds = 60;
    int sync_window_us = 10'000;
    int replicates_per_particle = 1;
    std::size_t reservoir_capacity = 8192;
    std::uint64_t base_seed = 12345;
    std::size_t max_attempt_multiplier = 100;
    double minimum_relative_epsilon_improvement = 0.005;
    double minimum_acceptance_rate = 0.0005;
    double final_epsilon = 0.0;
    std::filesystem::path parameter_space_file = "calibration/parameter_space.csv";
    std::filesystem::path data_directory = "data";
    std::filesystem::path market_targets_file;
    std::filesystem::path output_directory = "results/legacy_smc_abc";
};

struct Particle {
    UnitParameters theta{};
    double distance = 0.0;
    double weight = 0.0;
    std::uint64_t task_id = 0;
    std::int64_t ancestor = -1;
    std::uint64_t seed = 0;

    // Per-proposal diagnostics returned by the distributed forward model.
    std::array<double, 6> quantity_ks{};
    double event_proportion_l1 = 0.0;
    double market_component = 0.0;
    double mean_forward_wall_seconds = 0.0;
    int valid_replicates = 0;
};

SmcAbcConfig parse_smc_abc_config(int argc, char** argv);
void print_smc_abc_usage(const char* program_name);
void run_smc_abc(MPI_Comm world, const SmcAbcConfig& config);

// Exposed for unit tests.
double weighted_quantile(std::vector<double> values,
                         std::vector<double> weights,
                         double probability);
std::array<double, parameter_count> diagonal_kernel_variance(
    const std::vector<Particle>& particles);
std::vector<double> normalize_log_weights(const std::vector<double>& log_weights);
double effective_sample_size(const std::vector<double>& weights);

} // namespace dlob::calibration
