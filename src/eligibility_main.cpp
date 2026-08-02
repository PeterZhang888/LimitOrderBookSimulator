#include "calibration/CalibrationParameters.hpp"
#include "calibration/EmpiricalTargets.hpp"
#include "mpi/MpiCompat.hpp"
#include "simulation/DistributedSimulator.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
    std::string stage = "A";
    int duration_seconds = 300;
    int expected_ranks = 0;
    std::uint64_t seed = 12345;
    std::filesystem::path parameter_space = "calibration/parameter_space.csv";
    std::filesystem::path data_dir = "data";
    std::filesystem::path market_targets;
    std::filesystem::path output_dir = "results/eligibility_eval";
    dlob::calibration::UnitParameters unit{};
    bool have_unit = false;
    bool shared_snapshot = true;
    double max_wall_seconds = 600.0;
    double eligibility_threshold = std::numeric_limits<double>::infinity();
    int market_maker_batch_horizon_us = 100'000;
    int momentum_batch_horizon_us = 250'000;
    int informed_batch_horizon_us = 250'000;
    int institutional_batch_horizon_us = 1'000'000;
};

std::string require_value(int& index, int argc, char** argv, const std::string& option) {
    if (index + 1 >= argc) throw std::invalid_argument("Missing value after " + option);
    return argv[++index];
}

int parse_int(const std::string& value, const std::string& option) {
    int out = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), out);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
        throw std::invalid_argument("Invalid integer for " + option + ": " + value);
    }
    return out;
}

std::uint64_t parse_u64(const std::string& value, const std::string& option) {
    std::uint64_t out = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), out);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
        throw std::invalid_argument("Invalid integer for " + option + ": " + value);
    }
    return out;
}

double parse_double(const std::string& value, const std::string& option) {
    std::size_t used = 0;
    const double out = std::stod(value, &used);
    if (used != value.size()) throw std::invalid_argument("Invalid number for " + option);
    return out;
}

dlob::calibration::UnitParameters parse_unit(const std::string& text) {
    dlob::calibration::UnitParameters out{};
    std::stringstream stream(text);
    std::string cell;
    std::size_t index = 0;
    while (std::getline(stream, cell, ',')) {
        if (index >= out.size()) throw std::invalid_argument("Too many unit parameters");
        out[index++] = std::stod(cell);
    }
    if (index != out.size()) {
        throw std::invalid_argument("--unit-params requires exactly 8 comma-separated values");
    }
    return out;
}

Config parse_config(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--stage") {
            config.stage = require_value(i, argc, argv, arg);
            if (config.stage == "A") config.duration_seconds = 300;
            else if (config.stage == "B") config.duration_seconds = 1800;
            else if (config.stage == "C") config.duration_seconds = 23400;
            else throw std::invalid_argument("--stage must be A, B, or C");
        } else if (arg == "--duration-seconds") {
            config.duration_seconds = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--expected-ranks") {
            config.expected_ranks = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--seed") {
            config.seed = parse_u64(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--parameter-space") {
            config.parameter_space = require_value(i, argc, argv, arg);
        } else if (arg == "--data-dir") {
            config.data_dir = require_value(i, argc, argv, arg);
        } else if (arg == "--market-targets") {
            config.market_targets = require_value(i, argc, argv, arg);
        } else if (arg == "--output-dir") {
            config.output_dir = require_value(i, argc, argv, arg);
        } else if (arg == "--unit-params") {
            config.unit = parse_unit(require_value(i, argc, argv, arg));
            config.have_unit = true;
        } else if (arg == "--max-wall-seconds") {
            config.max_wall_seconds = parse_double(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--eligibility-threshold") {
            config.eligibility_threshold = parse_double(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--mm-batch-horizon-us") {
            config.market_maker_batch_horizon_us = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--momentum-batch-horizon-us") {
            config.momentum_batch_horizon_us = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--informed-batch-horizon-us") {
            config.informed_batch_horizon_us = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--institutional-batch-horizon-us") {
            config.institutional_batch_horizon_us = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--shared-snapshot") {
            config.shared_snapshot = true;
        } else if (arg == "--no-shared-snapshot") {
            config.shared_snapshot = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "eligibility_evaluate --stage A|B|C --unit-params u1,...,u8 [options]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }
    if (!config.have_unit) throw std::invalid_argument("--unit-params is required");
    if (config.duration_seconds <= 0) throw std::invalid_argument("Duration must be positive");
    if (config.max_wall_seconds < 0.0
        || config.market_maker_batch_horizon_us < 0
        || config.momentum_batch_horizon_us < 0
        || config.informed_batch_horizon_us < 0
        || config.institutional_batch_horizon_us < 0) {
        throw std::invalid_argument("Wall-time and batching controls must be non-negative");
    }
    return config;
}

void write_result(const Config& config,
                  const dlob::calibration::PhysicalParameters& p,
                  const dlob::simulation::SimulatorResult& result,
                  const dlob::calibration::DistanceBreakdown& distance,
                  bool eligible,
                  const std::string& reason) {
    std::filesystem::create_directories(config.output_dir);
    const auto path = config.output_dir / "eligibility_result.csv";
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write " + path.string());
    out << "stage,seed,duration_seconds,eligible,rejection_reason,distance,"
           "event_proportion_l1,market_component,eligibility_threshold,threshold_applied,"
           "wall_seconds,terminated_early,termination_reason,final_simulated_time_ns,"
           "mm_batch_horizon_us,momentum_batch_horizon_us,informed_batch_horizon_us,institutional_batch_horizon_us,"
           "structurally_valid,processed_strategic,processed_background,pending_end,peak_pending,activations,"
           "mean_spread_ticks,mean_bid_depth,mean_ask_depth,mid_move_rate,"
           "return_variance,return_kurtosis,absolute_return_acf1";
    for (const char* name : dlob::calibration::parameter_names()) out << ',' << name;
    for (const char* name : dlob::calibration::parameter_names()) out << ",u_" << name;
    out << '\n' << std::setprecision(17);
    out << config.stage << ',' << config.seed << ',' << config.duration_seconds << ','
        << (eligible ? 1 : 0) << ',' << reason << ',' << distance.total << ','
        << distance.event_proportion_l1 << ',' << distance.market_component << ','
        << config.eligibility_threshold << ','
        << (std::isfinite(config.eligibility_threshold) ? 1 : 0) << ','
        << result.wall_seconds << ',' << (result.terminated_early ? 1 : 0) << ','
        << result.termination_reason << ',' << result.final_simulated_time_ns << ','
        << config.market_maker_batch_horizon_us << ','
        << config.momentum_batch_horizon_us << ','
        << config.informed_batch_horizon_us << ','
        << config.institutional_batch_horizon_us << ','
        << (result.structurally_valid ? 1 : 0) << ','
        << result.processed_strategic << ',' << result.processed_background << ','
        << result.pending_end << ',' << result.peak_pending << ',' << result.activations << ','
        << result.record.market.mean_spread_ticks << ','
        << result.record.market.mean_bid_depth << ','
        << result.record.market.mean_ask_depth << ','
        << result.record.market.mid_move_rate << ','
        << result.record.market.return_variance << ','
        << result.record.market.return_kurtosis << ','
        << result.record.market.absolute_return_acf1 << ','
        << p.market_maker_interval_ms << ','
        << p.market_maker_min_spread_ticks << ','
        << p.momentum_rate_per_second << ','
        << p.momentum_threshold_ticks << ','
        << p.informed_rate_per_second << ','
        << p.informed_signal_precision << ','
        << p.institutional_rate_per_second << ','
        << p.institutional_participation_cap;
    for (double value : config.unit) out << ',' << value;
    out << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    try {
        const Config config = parse_config(argc, argv);
        if (config.expected_ranks > 0 && config.expected_ranks != world_size) {
            throw std::runtime_error("MPI world size does not match --expected-ranks");
        }
        const dlob::calibration::ParameterSpace space =
            dlob::calibration::ParameterSpace::load_csv(config.parameter_space);
        const auto physical = space.decode(config.unit);

        dlob::simulation::SimulatorConfig simulator;
        simulator.duration_seconds = config.duration_seconds;
        simulator.seed = config.seed;
        simulator.parameters = physical;
        simulator.data_directory = config.data_dir;
        simulator.output_directory = config.output_dir / "simulation";
        simulator.write_files = true;
        simulator.communication_mode = dlob::simulation::CommunicationMode::EventDrivenBatched;
        simulator.use_shared_market_snapshot = config.shared_snapshot;
        simulator.max_wall_seconds = config.max_wall_seconds;
        simulator.market_maker_batch_horizon_us = config.market_maker_batch_horizon_us;
        simulator.momentum_batch_horizon_us = config.momentum_batch_horizon_us;
        simulator.informed_batch_horizon_us = config.informed_batch_horizon_us;
        simulator.institutional_batch_horizon_us = config.institutional_batch_horizon_us;

        const auto result = dlob::simulation::run_distributed_simulator(MPI_COMM_WORLD, simulator);
        if (rank == 0) {
            const auto targets = dlob::calibration::EmpiricalTargets::load(
                config.data_dir, config.market_targets);
            const auto distance = targets.distance(result.record);
            const std::uint64_t total_events =
                result.processed_strategic + result.processed_background;

            bool eligible = true;
            std::string reason = "accepted";
            auto reject = [&](const std::string& value) {
                if (eligible) {
                    eligible = false;
                    reason = value;
                }
            };
            if (result.terminated_early) reject(
                result.termination_reason.empty() ? "terminated_early" : result.termination_reason);
            if (!result.structurally_valid) reject("invalid_book_or_message_accounting");
            if (total_events < 10) reject("negligible_event_count");
            if (result.record.market.mean_spread_ticks <= 0.0
                || result.record.market.mean_spread_ticks > 50.0) reject("extreme_spread");
            if (result.record.market.mean_bid_depth <= 0.0
                || result.record.market.mean_ask_depth <= 0.0
                || result.record.market.mean_bid_depth > 1e7
                || result.record.market.mean_ask_depth > 1e7) reject("excessive_or_invalid_depth");
            if (config.duration_seconds >= 300
                && result.record.market.mid_move_rate < 1e-5) reject("negligible_price_movement");
            if (config.max_wall_seconds > 0.0
                && result.wall_seconds > config.max_wall_seconds) reject("exploding_runtime");
            if (!std::isfinite(distance.total)) reject("non_finite_distance");
            if (std::isfinite(config.eligibility_threshold)
                && distance.total > config.eligibility_threshold) {
                reject("distance_above_eligibility_threshold");
            }

            write_result(config, physical, result, distance, eligible, reason);
            std::cout << "stage=" << config.stage
                      << " eligible=" << (eligible ? 1 : 0)
                      << " reason=" << reason
                      << " distance=" << distance.total
                      << " wall_seconds=" << result.wall_seconds << '\n';
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Rank " << rank << " eligibility error: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        return 1;
    }
}
