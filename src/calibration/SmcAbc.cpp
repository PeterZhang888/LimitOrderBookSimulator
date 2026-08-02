#include "calibration/SmcAbc.hpp"

#include "calibration/EmpiricalTargets.hpp"
#include "simulation/DistributedSimulator.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <type_traits>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dlob::calibration {
namespace {

constexpr int task_tag = 7301;
constexpr int result_tag = 7302;
constexpr int command_run = 1;
constexpr int command_end_wave = 2;

struct ParticleTask {
    int command = command_run;
    int wave = 0;
    std::uint64_t task_id = 0;
    std::uint64_t seed = 0;
    std::int64_t ancestor = -1;
    std::array<double, parameter_count> theta{};
};

struct ParticleResult {
    std::uint64_t task_id = 0;
    int valid = 0;
    int valid_replicates = 0;
    double distance = std::numeric_limits<double>::infinity();
    double mean_wall_seconds = 0.0;
    std::array<double, empirical_event_bucket_count> mean_ks{};
    double mean_event_l1 = 0.0;
    double mean_market_component = 0.0;
};

static_assert(std::is_trivially_copyable_v<ParticleTask>);
static_assert(std::is_trivially_copyable_v<ParticleResult>);

struct ProposalContext {
    Particle particle;
};

struct WaveOutcome {
    std::vector<Particle> particles;
    std::size_t attempts = 0;
    std::size_t structurally_invalid = 0;
    double sum_forward_wall_seconds = 0.0;
};

void check_mpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) throw std::runtime_error(std::string(operation) + " failed");
}

int checked_bytes(std::size_t size, const char* label) {
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(label) + " exceeds MPI int count");
    }
    return static_cast<int>(size);
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

template <typename Integer>
Integer parse_integer(std::string_view text, const char* option) {
    Integer value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument(std::string("Invalid value for ") + option);
    }
    return value;
}

double parse_double(const std::string& text, const char* option) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument(std::string("Invalid value for ") + option);
    }
    return value;
}

std::string require_value(int& index, int argc, char** argv, const char* option) {
    if (index + 1 >= argc) throw std::invalid_argument(std::string("Missing value after ") + option);
    return argv[++index];
}

double ordinary_quantile(std::vector<double> values, double probability) {
    if (values.empty()) throw std::invalid_argument("Cannot compute a quantile of an empty vector");
    std::sort(values.begin(), values.end());
    const double p = std::clamp(probability, 0.0, 1.0);
    const double location = p * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(location));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(location));
    const double fraction = location - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

std::vector<double> particle_weights(const std::vector<Particle>& particles) {
    std::vector<double> weights;
    weights.reserve(particles.size());
    for (const Particle& particle : particles) weights.push_back(particle.weight);
    return weights;
}

UnitParameters propose_theta(int wave,
                             const std::vector<Particle>& previous,
                             const std::array<double, parameter_count>& variance,
                             std::discrete_distribution<std::size_t>* ancestor_distribution,
                             std::mt19937_64& rng,
                             std::int64_t& ancestor) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    if (wave == 0) {
        UnitParameters theta{};
        for (double& value : theta) value = uniform(rng);
        ancestor = -1;
        return theta;
    }

    std::normal_distribution<double> normal(0.0, 1.0);
    for (;;) {
        if (ancestor_distribution == nullptr) {
            throw std::logic_error("Missing ancestor distribution for SMC wave");
        }
        const std::size_t selected = (*ancestor_distribution)(rng);
        UnitParameters theta = previous[selected].theta;
        bool inside = true;
        for (std::size_t dimension = 0; dimension < parameter_count; ++dimension) {
            theta[dimension] += std::sqrt(variance[dimension]) * normal(rng);
            if (theta[dimension] < 0.0 || theta[dimension] > 1.0) {
                inside = false;
                break;
            }
        }
        if (inside) {
            ancestor = static_cast<std::int64_t>(selected);
            return theta;
        }
    }
}

ParticleTask make_task(int wave,
                       std::uint64_t task_id,
                       const std::vector<Particle>& previous,
                       const std::array<double, parameter_count>& variance,
                       std::discrete_distribution<std::size_t>* ancestor_distribution,
                       std::mt19937_64& rng,
                       std::uint64_t base_seed,
                       ProposalContext& context) {
    ParticleTask task;
    task.command = command_run;
    task.wave = wave;
    task.task_id = task_id;
    task.seed = splitmix64(base_seed ^ (task_id * 0x9e3779b97f4a7c15ULL)
                           ^ static_cast<std::uint64_t>(wave + 1));
    task.theta = propose_theta(wave, previous, variance, ancestor_distribution,
                               rng, task.ancestor);
    context.particle.theta = task.theta;
    context.particle.task_id = task.task_id;
    context.particle.ancestor = task.ancestor;
    context.particle.seed = task.seed;
    return task;
}

void send_task(int leader, const ParticleTask& task, MPI_Comm world) {
    check_mpi(MPI_Send(&task, checked_bytes(sizeof(task), "particle task"), MPI_BYTE,
                       leader, task_tag, world), "MPI_Send(particle task)");
}

ParticleResult evaluate_task(const ParticleTask& task,
                             MPI_Comm group,
                             int group_rank,
                             const SmcAbcConfig& config,
                             const ParameterSpace& space,
                             const EmpiricalTargets* targets) {
    std::vector<double> distances;
    std::array<double, empirical_event_bucket_count> ks_sum{};
    double event_sum = 0.0;
    double market_sum = 0.0;
    double wall_sum = 0.0;
    int valid_replicates = 0;

    for (int repeat = 0; repeat < config.replicates_per_particle; ++repeat) {
        simulation::SimulatorConfig simulator;
        simulator.duration_seconds = config.duration_seconds;
        simulator.sync_window_us = config.sync_window_us;
        simulator.communication_mode = simulation::CommunicationMode::EventDrivenBatched;
        simulator.use_shared_market_snapshot = true;
        simulator.seed = splitmix64(task.seed ^ static_cast<std::uint64_t>(repeat + 1));
        simulator.parameters = space.decode(task.theta);
        simulator.data_directory = config.data_directory;
        simulator.write_files = false;
        simulator.reservoir_capacity = config.reservoir_capacity;

        const simulation::SimulatorResult result =
            simulation::run_distributed_simulator(group, simulator);
        if (group_rank == 0) {
            wall_sum += result.wall_seconds;
            if (result.structurally_valid) {
                const DistanceBreakdown breakdown = targets->distance(result.record);
                distances.push_back(breakdown.total);
                for (std::size_t i = 0; i < empirical_event_bucket_count; ++i) {
                    ks_sum[i] += breakdown.quantity_ks[i];
                }
                event_sum += breakdown.event_proportion_l1;
                market_sum += breakdown.market_component;
                ++valid_replicates;
            }
        }
    }

    ParticleResult output;
    output.task_id = task.task_id;
    if (group_rank == 0) {
        output.valid_replicates = valid_replicates;
        output.valid = valid_replicates == config.replicates_per_particle ? 1 : 0;
        output.mean_wall_seconds = wall_sum / static_cast<double>(config.replicates_per_particle);
        if (output.valid != 0) {
            output.distance = ordinary_quantile(distances, 0.50);
            for (std::size_t i = 0; i < empirical_event_bucket_count; ++i) {
                output.mean_ks[i] = ks_sum[i] / static_cast<double>(valid_replicates);
            }
            output.mean_event_l1 = event_sum / static_cast<double>(valid_replicates);
            output.mean_market_component = market_sum / static_cast<double>(valid_replicates);
        }
    }
    return output;
}

void worker_wave(MPI_Comm world,
                 MPI_Comm group,
                 int group_rank,
                 const SmcAbcConfig& config,
                 const ParameterSpace& space,
                 const EmpiricalTargets* targets) {
    for (;;) {
        ParticleTask task;
        if (group_rank == 0) {
            check_mpi(MPI_Recv(&task, checked_bytes(sizeof(task), "particle task"), MPI_BYTE,
                               0, task_tag, world, MPI_STATUS_IGNORE),
                      "MPI_Recv(particle task)");
        }
        check_mpi(MPI_Bcast(&task, checked_bytes(sizeof(task), "particle task"), MPI_BYTE,
                            0, group), "MPI_Bcast(particle task)");
        if (task.command == command_end_wave) break;
        const ParticleResult result = evaluate_task(task, group, group_rank, config, space, targets);
        if (group_rank == 0) {
            check_mpi(MPI_Send(&result, checked_bytes(sizeof(result), "particle result"), MPI_BYTE,
                               0, result_tag, world), "MPI_Send(particle result)");
        }
    }
}

WaveOutcome master_wave(MPI_Comm world,
                        int wave,
                        double epsilon,
                        const std::vector<int>& leaders,
                        const SmcAbcConfig& config,
                        const std::vector<Particle>& previous,
                        const std::array<double, parameter_count>& variance,
                        std::mt19937_64& rng,
                        std::uint64_t& next_task_id) {
    WaveOutcome outcome;
    outcome.particles.reserve(config.particles_per_wave + leaders.size());
    const std::size_t maximum_attempts = std::max(
        config.particles_per_wave,
        config.particles_per_wave * config.max_attempt_multiplier);

    std::optional<std::discrete_distribution<std::size_t>> ancestor_distribution;
    if (wave > 0) {
        const std::vector<double> weights = particle_weights(previous);
        ancestor_distribution.emplace(weights.begin(), weights.end());
    }

    std::unordered_map<std::uint64_t, ProposalContext> in_flight_context;
    std::size_t in_flight = 0;
    std::size_t ended = 0;

    auto dispatch = [&](int leader) -> bool {
        if (outcome.attempts >= maximum_attempts) return false;
        ProposalContext context;
        ParticleTask task = make_task(
            wave, next_task_id++, previous, variance,
            ancestor_distribution ? &*ancestor_distribution : nullptr,
            rng, config.base_seed, context);
        in_flight_context.emplace(task.task_id, context);
        send_task(leader, task, world);
        ++outcome.attempts;
        ++in_flight;
        return true;
    };

    for (int leader : leaders) {
        if (!dispatch(leader)) {
            ParticleTask end;
            end.command = command_end_wave;
            send_task(leader, end, world);
            ++ended;
        }
    }

    while (in_flight > 0) {
        ParticleResult result;
        MPI_Status status{};
        check_mpi(MPI_Recv(&result, checked_bytes(sizeof(result), "particle result"), MPI_BYTE,
                           MPI_ANY_SOURCE, result_tag, world, &status),
                  "MPI_Recv(particle result)");
        --in_flight;
        const auto found = in_flight_context.find(result.task_id);
        if (found == in_flight_context.end()) throw std::runtime_error("Unknown particle result task id");
        ProposalContext context = found->second;
        in_flight_context.erase(found);
        outcome.sum_forward_wall_seconds += result.mean_wall_seconds;

        if (result.valid == 0 || !std::isfinite(result.distance)) {
            ++outcome.structurally_invalid;
        } else if (result.distance <= epsilon) {
            context.particle.distance = result.distance;
            context.particle.quantity_ks = result.mean_ks;
            context.particle.event_proportion_l1 = result.mean_event_l1;
            context.particle.market_component = result.mean_market_component;
            context.particle.mean_forward_wall_seconds = result.mean_wall_seconds;
            context.particle.valid_replicates = result.valid_replicates;
            outcome.particles.push_back(context.particle);
        }

        const bool need_more = outcome.particles.size() < config.particles_per_wave;
        if (need_more && dispatch(status.MPI_SOURCE)) {
            continue;
        }
        ParticleTask end;
        end.command = command_end_wave;
        send_task(status.MPI_SOURCE, end, world);
        ++ended;
    }

    if (ended != leaders.size()) {
        throw std::runtime_error("Not every simulation group received wave-end command");
    }
    if (outcome.particles.size() < config.particles_per_wave) {
        throw std::runtime_error(
            "SMC wave exhausted max attempts before obtaining the requested accepted population. "
            "Increase --max-attempt-multiplier or use a less aggressive tolerance quantile.");
    }
    std::sort(outcome.particles.begin(), outcome.particles.end(),
              [](const Particle& a, const Particle& b) { return a.task_id < b.task_id; });
    outcome.particles.resize(config.particles_per_wave);
    return outcome;
}

void broadcast_vector(std::vector<double>& values, int root, MPI_Comm world) {
    unsigned long long size = static_cast<unsigned long long>(values.size());
    check_mpi(MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG_LONG, root, world),
              "MPI_Bcast(vector size)");
    if (values.size() != static_cast<std::size_t>(size)) values.resize(static_cast<std::size_t>(size));
    if (size > 0) {
        check_mpi(MPI_Bcast(values.data(), checked_bytes(values.size(), "double vector") ,
                            MPI_DOUBLE, root, world), "MPI_Bcast(double vector)");
    }
}

std::vector<double> distributed_importance_weights(
    MPI_Comm world,
    int rank,
    int world_size,
    int wave,
    const std::vector<Particle>& previous_on_root,
    const std::vector<Particle>& current_on_root,
    const std::array<double, parameter_count>& variance_on_root) {
    int compute = wave > 0 ? 1 : 0;
    check_mpi(MPI_Bcast(&compute, 1, MPI_INT, 0, world), "MPI_Bcast(weight phase)");
    if (compute == 0) return {};

    std::vector<double> previous_theta;
    std::vector<double> previous_weights;
    std::vector<double> current_theta;
    std::vector<double> variance(parameter_count);
    if (rank == 0) {
        previous_theta.reserve(previous_on_root.size() * parameter_count);
        previous_weights.reserve(previous_on_root.size());
        current_theta.reserve(current_on_root.size() * parameter_count);
        for (const Particle& p : previous_on_root) {
            previous_theta.insert(previous_theta.end(), p.theta.begin(), p.theta.end());
            previous_weights.push_back(p.weight);
        }
        for (const Particle& p : current_on_root) {
            current_theta.insert(current_theta.end(), p.theta.begin(), p.theta.end());
        }
        std::copy(variance_on_root.begin(), variance_on_root.end(), variance.begin());
    }
    broadcast_vector(previous_theta, 0, world);
    broadcast_vector(previous_weights, 0, world);
    broadcast_vector(current_theta, 0, world);
    broadcast_vector(variance, 0, world);

    const std::size_t n = previous_weights.size();
    if (current_theta.size() != n * parameter_count
        || previous_theta.size() != n * parameter_count) {
        throw std::runtime_error("Weight phase particle-array size mismatch");
    }

    const std::size_t begin = n * static_cast<std::size_t>(rank)
        / static_cast<std::size_t>(world_size);
    const std::size_t end = n * static_cast<std::size_t>(rank + 1)
        / static_cast<std::size_t>(world_size);
    std::vector<double> local_log_weights(end - begin);
    constexpr double log_two_pi = 1.83787706640934548356;

    std::vector<double> terms(n);
    for (std::size_t i = begin; i < end; ++i) {
        double max_term = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < n; ++j) {
            double log_kernel = 0.0;
            for (std::size_t d = 0; d < parameter_count; ++d) {
                const double var = std::max(variance[d], 1e-12);
                const double difference = current_theta[i * parameter_count + d]
                    - previous_theta[j * parameter_count + d];
                log_kernel += -0.5 * (log_two_pi + std::log(var)
                                      + difference * difference / var);
            }
            const double term = std::log(std::max(previous_weights[j], 1e-300)) + log_kernel;
            terms[j] = term;
            max_term = std::max(max_term, term);
        }
        double sum = 0.0;
        for (double term : terms) sum += std::exp(term - max_term);
        const double log_denominator = max_term + std::log(std::max(sum, 1e-300));
        local_log_weights[i - begin] = -log_denominator; // uniform unit-cube prior
    }

    std::vector<int> counts(static_cast<std::size_t>(world_size));
    std::vector<int> displacements(static_cast<std::size_t>(world_size));
    for (int r = 0; r < world_size; ++r) {
        const std::size_t r_begin = n * static_cast<std::size_t>(r)
            / static_cast<std::size_t>(world_size);
        const std::size_t r_end = n * static_cast<std::size_t>(r + 1)
            / static_cast<std::size_t>(world_size);
        counts[static_cast<std::size_t>(r)] = checked_bytes(r_end - r_begin, "weight chunk");
        displacements[static_cast<std::size_t>(r)] = checked_bytes(r_begin, "weight displacement");
    }
    std::vector<double> all_log_weights;
    if (rank == 0) all_log_weights.resize(n);
    check_mpi(MPI_Gatherv(local_log_weights.data(), checked_bytes(local_log_weights.size(), "local weights"),
                          MPI_DOUBLE,
                          rank == 0 ? all_log_weights.data() : nullptr,
                          counts.data(), displacements.data(), MPI_DOUBLE,
                          0, world), "MPI_Gatherv(log weights)");
    return rank == 0 ? normalize_log_weights(all_log_weights) : std::vector<double>{};
}

void atomic_write(const std::filesystem::path& final_path,
                  const std::function<void(std::ostream&)>& writer) {
    const std::filesystem::path temporary = final_path.string() + ".tmp";
    {
        std::ofstream output(temporary);
        if (!output) throw std::runtime_error("Cannot write " + temporary.string());
        writer(output);
    }
    std::error_code error;
    std::filesystem::rename(temporary, final_path, error);
    if (error) {
        std::filesystem::remove(final_path, error);
        error.clear();
        std::filesystem::rename(temporary, final_path, error);
        if (error) throw std::runtime_error("Cannot atomically replace " + final_path.string());
    }
}

void write_particles(const std::filesystem::path& output_directory,
                     int wave,
                     double epsilon,
                     const std::vector<Particle>& particles,
                     const ParameterSpace& space) {
    const auto path = output_directory /
        ("wave_" + std::string(wave < 10 ? "00" : wave < 100 ? "0" : "")
         + std::to_string(wave) + "_particles.csv");
    atomic_write(path, [&](std::ostream& output) {
        output << "wave,epsilon,particle_id,task_id,ancestor,seed,distance,weight,"
                  "valid_replicates,mean_forward_wall_seconds,"
                  "ks_limit_buy,ks_limit_sell,ks_market_buy,ks_market_sell,"
                  "ks_cancel_bid,ks_cancel_ask,event_proportion_l1,market_component";
        for (const char* name : parameter_names()) output << ',' << name;
        for (const char* name : parameter_names()) output << ",u_" << name;
        output << '\n' << std::setprecision(17);
        for (std::size_t i = 0; i < particles.size(); ++i) {
            const Particle& particle = particles[i];
            const PhysicalParameters p = space.decode(particle.theta);
            output << wave << ',' << epsilon << ',' << i << ',' << particle.task_id << ','
                   << particle.ancestor << ',' << particle.seed << ',' << particle.distance << ','
                   << particle.weight << ',' << particle.valid_replicates << ','
                   << particle.mean_forward_wall_seconds;
            for (double value : particle.quantity_ks) output << ',' << value;
            output << ',' << particle.event_proportion_l1
                   << ',' << particle.market_component << ','
                   << p.market_maker_interval_ms << ','
                   << p.market_maker_min_spread_ticks << ','
                   << p.momentum_rate_per_second << ','
                   << p.momentum_threshold_ticks << ','
                   << p.informed_rate_per_second << ','
                   << p.informed_signal_precision << ','
                   << p.institutional_rate_per_second << ','
                   << p.institutional_participation_cap;
            for (double value : particle.theta) output << ',' << value;
            output << '\n';
        }
    });
}

std::vector<std::pair<double, double>> weighted_mean_sd(const std::vector<Particle>& particles,
                                                        const ParameterSpace& space) {
    std::vector<std::pair<double, double>> output(parameter_count);
    std::vector<std::vector<double>> values(parameter_count);
    for (auto& vector : values) vector.reserve(particles.size());
    for (const Particle& particle : particles) {
        const PhysicalParameters p = space.decode(particle.theta);
        const std::array<double, parameter_count> physical{
            p.market_maker_interval_ms,
            static_cast<double>(p.market_maker_min_spread_ticks),
            p.momentum_rate_per_second,
            p.momentum_threshold_ticks,
            p.informed_rate_per_second,
            p.informed_signal_precision,
            p.institutional_rate_per_second,
            p.institutional_participation_cap
        };
        for (std::size_t d = 0; d < parameter_count; ++d) values[d].push_back(physical[d]);
    }
    const std::vector<double> weights = particle_weights(particles);
    for (std::size_t d = 0; d < parameter_count; ++d) {
        double mean = 0.0;
        for (std::size_t i = 0; i < particles.size(); ++i) mean += weights[i] * values[d][i];
        double variance = 0.0;
        for (std::size_t i = 0; i < particles.size(); ++i) {
            const double difference = values[d][i] - mean;
            variance += weights[i] * difference * difference;
        }
        output[d] = {mean, std::sqrt(std::max(0.0, variance))};
    }
    return output;
}

void write_posterior_summary(const std::filesystem::path& output_directory,
                             int wave,
                             const std::vector<Particle>& particles,
                             const ParameterSpace& space) {
    const std::vector<double> weights = particle_weights(particles);
    std::vector<std::vector<double>> values(parameter_count);
    for (auto& v : values) v.reserve(particles.size());
    for (const Particle& particle : particles) {
        const PhysicalParameters p = space.decode(particle.theta);
        const std::array<double, parameter_count> physical{
            p.market_maker_interval_ms,
            static_cast<double>(p.market_maker_min_spread_ticks),
            p.momentum_rate_per_second,
            p.momentum_threshold_ticks,
            p.informed_rate_per_second,
            p.informed_signal_precision,
            p.institutional_rate_per_second,
            p.institutional_participation_cap
        };
        for (std::size_t d = 0; d < parameter_count; ++d) values[d].push_back(physical[d]);
    }
    const auto means = weighted_mean_sd(particles, space);
    const auto path = output_directory / "posterior_summary.csv";
    atomic_write(path, [&](std::ostream& output) {
        output << "wave,parameter,weighted_mean,weighted_sd,q025,q50,q975\n";
        output << std::setprecision(17);
        for (std::size_t d = 0; d < parameter_count; ++d) {
            output << wave << ',' << parameter_names()[d] << ',' << means[d].first << ','
                   << means[d].second << ','
                   << weighted_quantile(values[d], weights, 0.025) << ','
                   << weighted_quantile(values[d], weights, 0.50) << ','
                   << weighted_quantile(values[d], weights, 0.975) << '\n';
        }
    });
}


std::array<double, parameter_count> physical_vector(
    const Particle& particle,
    const ParameterSpace& space) {
    const PhysicalParameters p = space.decode(particle.theta);
    return {
        p.market_maker_interval_ms,
        static_cast<double>(p.market_maker_min_spread_ticks),
        p.momentum_rate_per_second,
        p.momentum_threshold_ticks,
        p.informed_rate_per_second,
        p.informed_signal_precision,
        p.institutional_rate_per_second,
        p.institutional_participation_cap
    };
}

void write_posterior_correlations(const std::filesystem::path& output_directory,
                                  int wave,
                                  const std::vector<Particle>& particles,
                                  const ParameterSpace& space) {
    std::array<double, parameter_count> mean{};
    for (const Particle& particle : particles) {
        const auto values = physical_vector(particle, space);
        for (std::size_t d = 0; d < parameter_count; ++d) {
            mean[d] += particle.weight * values[d];
        }
    }
    std::array<double, parameter_count> variance{};
    std::array<std::array<double, parameter_count>, parameter_count> covariance{};
    for (const Particle& particle : particles) {
        const auto values = physical_vector(particle, space);
        for (std::size_t i = 0; i < parameter_count; ++i) {
            const double di = values[i] - mean[i];
            variance[i] += particle.weight * di * di;
            for (std::size_t j = 0; j < parameter_count; ++j) {
                covariance[i][j] += particle.weight * di * (values[j] - mean[j]);
            }
        }
    }
    const auto path = output_directory / "posterior_correlations.csv";
    atomic_write(path, [&](std::ostream& output) {
        output << "wave,parameter_i,parameter_j,correlation\n" << std::setprecision(17);
        for (std::size_t i = 0; i < parameter_count; ++i) {
            for (std::size_t j = 0; j < parameter_count; ++j) {
                const double denominator = std::sqrt(std::max(0.0, variance[i] * variance[j]));
                const double correlation = denominator > 0.0 ? covariance[i][j] / denominator : 0.0;
                output << wave << ',' << parameter_names()[i] << ',' << parameter_names()[j]
                       << ',' << correlation << '\n';
            }
        }
    });
}

void write_run_manifest(const std::filesystem::path& output_directory,
                        const SmcAbcConfig& config,
                        const ParameterSpace& space,
                        int world_size,
                        int group_count) {
    const auto path = output_directory / "run_manifest.txt";
    atomic_write(path, [&](std::ostream& output) {
        output << std::setprecision(17)
               << "method=adaptive_smc_abc_beaumont_importance_weights\n"
               << "outer_parallelism=dynamic_master_worker_map_reduce\n"
               << "inner_parallelism=distributed_interacting_lob_mpi\n"
               << "world_ranks=" << world_size << '\n'
               << "simulation_groups=" << group_count << '\n'
               << "ranks_per_simulation=" << config.ranks_per_simulation << '\n'
               << "particles_per_wave=" << config.particles_per_wave << '\n'
               << "max_waves=" << config.max_waves << '\n'
               << "tolerance_quantile=" << config.tolerance_quantile << '\n'
               << "duration_seconds=" << config.duration_seconds << '\n'
               << "sync_window_us=" << config.sync_window_us << '\n'
               << "replicates_per_particle=" << config.replicates_per_particle << '\n'
               << "reservoir_capacity=" << config.reservoir_capacity << '\n'
               << "base_seed=" << config.base_seed << '\n'
               << "max_attempt_multiplier=" << config.max_attempt_multiplier << '\n'
               << "minimum_relative_epsilon_improvement="
               << config.minimum_relative_epsilon_improvement << '\n'
               << "minimum_acceptance_rate=" << config.minimum_acceptance_rate << '\n'
               << "final_epsilon=" << config.final_epsilon << '\n'
               << "fixed_hawkes_activity_scale=" << fixed_hawkes_activity_scale << '\n'
               << "parameter_space_file=" << config.parameter_space_file.string() << '\n'
               << "data_directory=" << config.data_directory.string() << '\n'
               << "market_targets_file=" << config.market_targets_file.string() << '\n'
               << "parameter_count=" << parameter_count << '\n';
        for (std::size_t i = 0; i < parameter_count; ++i) {
            const ParameterSpec& spec = space.specs()[i];
            output << "parameter." << i << ".name=" << spec.name << '\n'
                   << "parameter." << i << ".lower=" << spec.lower << '\n'
                   << "parameter." << i << ".upper=" << spec.upper << '\n'
                   << "parameter." << i << ".type="
                   << (spec.kind == ParameterKind::Integer ? "integer" : "continuous")
                   << '\n';
        }
    });
}

void append_diagnostics(const std::filesystem::path& output_directory,
                        int wave,
                        double epsilon,
                        double epsilon_next,
                        const WaveOutcome& outcome,
                        const std::vector<Particle>& particles,
                        double wave_wall_seconds) {
    const auto path = output_directory / "wave_diagnostics.csv";
    const bool new_file = !std::filesystem::exists(path);
    std::ofstream output(path, std::ios::app);
    if (!output) throw std::runtime_error("Cannot write diagnostics");
    if (new_file) {
        output << "wave,particles,epsilon_used,epsilon_next,attempts,accepted,acceptance_rate,"
                  "structurally_invalid,ess,distance_q10,distance_q50,distance_q90,"
                  "wave_wall_seconds,sum_forward_wall_seconds,fixed_hawkes_activity_scale\n";
    }
    std::vector<double> distances;
    distances.reserve(particles.size());
    for (const Particle& p : particles) distances.push_back(p.distance);
    const double rate = outcome.attempts > 0
        ? static_cast<double>(particles.size()) / static_cast<double>(outcome.attempts) : 0.0;
    output << std::setprecision(17)
           << wave << ',' << particles.size() << ',' << epsilon << ',' << epsilon_next << ','
           << outcome.attempts << ',' << particles.size() << ',' << rate << ','
           << outcome.structurally_invalid << ',' << effective_sample_size(particle_weights(particles)) << ','
           << ordinary_quantile(distances, 0.10) << ',' << ordinary_quantile(distances, 0.50) << ','
           << ordinary_quantile(distances, 0.90) << ',' << wave_wall_seconds << ','
           << outcome.sum_forward_wall_seconds << ',' << fixed_hawkes_activity_scale << '\n';
}

} // namespace

SmcAbcConfig parse_smc_abc_config(int argc, char** argv) {
    SmcAbcConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--particles") config.particles_per_wave = parse_integer<std::size_t>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        else if (arg == "--waves") config.max_waves = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        else if (arg == "--tolerance-quantile") config.tolerance_quantile = parse_double(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        else if (arg == "--ranks-per-simulation") config.ranks_per_simulation = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        else if (arg == "--duration-seconds") config.duration_seconds = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        else if (arg == "--sync-window-us") config.sync_window_us = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        else if (arg == "--replicates") config.replicates_per_particle = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        else if (arg == "--reservoir-capacity") config.reservoir_capacity = parse_integer<std::size_t>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        else if (arg == "--base-seed") config.base_seed = parse_integer<std::uint64_t>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        else if (arg == "--max-attempt-multiplier") config.max_attempt_multiplier = parse_integer<std::size_t>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        else if (arg == "--min-epsilon-improvement") config.minimum_relative_epsilon_improvement = parse_double(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        else if (arg == "--min-acceptance-rate") config.minimum_acceptance_rate = parse_double(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        else if (arg == "--final-epsilon") config.final_epsilon = parse_double(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        else if (arg == "--parameter-space") config.parameter_space_file = require_value(i, argc, argv, arg.c_str());
        else if (arg == "--data-dir") config.data_directory = require_value(i, argc, argv, arg.c_str());
        else if (arg == "--market-targets") config.market_targets_file = require_value(i, argc, argv, arg.c_str());
        else if (arg == "--output-dir") config.output_directory = require_value(i, argc, argv, arg.c_str());
        else if (arg == "--help" || arg == "-h") {
            print_smc_abc_usage(argv[0]);
            std::exit(0);
        } else throw std::invalid_argument("Unknown SMC-ABC option: " + arg);
    }
    if (config.particles_per_wave == 0 || config.max_waves <= 0
        || config.ranks_per_simulation <= 0 || config.duration_seconds <= 0
        || config.sync_window_us <= 0 || config.replicates_per_particle <= 0) {
        throw std::invalid_argument("Particle count, waves, ranks, duration, window and replicates must be positive");
    }
    if (!(config.tolerance_quantile > 0.0 && config.tolerance_quantile < 1.0)) {
        throw std::invalid_argument("--tolerance-quantile must lie strictly between 0 and 1");
    }
    return config;
}

void print_smc_abc_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "  --particles N                 accepted particles per wave (default 32; legacy comparison only)\n"
              << "  --waves N                     maximum SMC waves\n"
              << "  --tolerance-quantile Q        adaptive next-wave quantile\n"
              << "  --ranks-per-simulation N      MPI ranks in each interacting LOB group\n"
              << "  --duration-seconds N          23400 for a full QQQ day\n"
              << "  --replicates N                simulator draws per ABC proposal\n"
              << "  --parameter-space FILE\n  --data-dir DIR\n  --market-targets FILE\n"
              << "  --output-dir DIR\n";
}

double weighted_quantile(std::vector<double> values,
                         std::vector<double> weights,
                         double probability) {
    if (values.empty() || values.size() != weights.size()) {
        throw std::invalid_argument("weighted_quantile requires equal non-empty vectors");
    }
    std::vector<std::size_t> order(values.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return values[a] < values[b];
    });
    const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (!(sum > 0.0)) throw std::invalid_argument("weighted_quantile weights must have positive sum");
    const double threshold = std::clamp(probability, 0.0, 1.0) * sum;
    double cumulative = 0.0;
    for (std::size_t index : order) {
        cumulative += std::max(0.0, weights[index]);
        if (cumulative >= threshold) return values[index];
    }
    return values[order.back()];
}

std::array<double, parameter_count> diagonal_kernel_variance(
    const std::vector<Particle>& particles) {
    if (particles.empty()) throw std::invalid_argument("Cannot calculate kernel variance from no particles");
    std::array<double, parameter_count> mean{};
    std::array<double, parameter_count> variance{};
    double weight_sum = 0.0;
    for (const Particle& particle : particles) {
        weight_sum += particle.weight;
        for (std::size_t d = 0; d < parameter_count; ++d) mean[d] += particle.weight * particle.theta[d];
    }
    for (double& value : mean) value /= std::max(weight_sum, 1e-300);
    for (const Particle& particle : particles) {
        for (std::size_t d = 0; d < parameter_count; ++d) {
            const double difference = particle.theta[d] - mean[d];
            variance[d] += particle.weight * difference * difference;
        }
    }
    for (double& value : variance) {
        value = std::max(1e-6, 2.0 * value / std::max(weight_sum, 1e-300));
    }
    return variance;
}

std::vector<double> normalize_log_weights(const std::vector<double>& log_weights) {
    if (log_weights.empty()) return {};
    const double maximum = *std::max_element(log_weights.begin(), log_weights.end());
    std::vector<double> weights(log_weights.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < log_weights.size(); ++i) {
        weights[i] = std::exp(log_weights[i] - maximum);
        sum += weights[i];
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) throw std::runtime_error("Failed to normalize importance weights");
    for (double& weight : weights) weight /= sum;
    return weights;
}

double effective_sample_size(const std::vector<double>& weights) {
    double sum2 = 0.0;
    for (double weight : weights) sum2 += weight * weight;
    return sum2 > 0.0 ? 1.0 / sum2 : 0.0;
}

void run_smc_abc(MPI_Comm world, const SmcAbcConfig& config) {
    int rank = 0;
    int world_size = 1;
    check_mpi(MPI_Comm_rank(world, &rank), "MPI_Comm_rank(SMC)");
    check_mpi(MPI_Comm_size(world, &world_size), "MPI_Comm_size(SMC)");

    int valid_layout = world_size > 1
        && (world_size - 1) % config.ranks_per_simulation == 0 ? 1 : 0;
    if (rank == 0 && valid_layout == 0) {
        std::cerr << "SMC-ABC requires world_size = 1 + G * ranks_per_simulation. Received "
                  << world_size << " ranks with ranks_per_simulation="
                  << config.ranks_per_simulation << ".\n";
    }
    check_mpi(MPI_Bcast(&valid_layout, 1, MPI_INT, 0, world), "MPI_Bcast(layout)");
    if (valid_layout == 0) throw std::runtime_error("Invalid hierarchical MPI layout");

    const int group_count = (world_size - 1) / config.ranks_per_simulation;
    const int color = rank == 0 ? MPI_UNDEFINED : (rank - 1) / config.ranks_per_simulation;
    MPI_Comm group = MPI_COMM_NULL;
    check_mpi(MPI_Comm_split(world, color, rank, &group), "MPI_Comm_split(simulation groups)");
    int group_rank = -1;
    if (rank != 0) check_mpi(MPI_Comm_rank(group, &group_rank), "MPI_Comm_rank(group)");

    const ParameterSpace space = ParameterSpace::load_csv(config.parameter_space_file);
    std::optional<EmpiricalTargets> targets;
    if (rank != 0 && group_rank == 0) {
        targets = EmpiricalTargets::load(config.data_directory, config.market_targets_file);
    }

    std::vector<int> leaders;
    if (rank == 0) {
        leaders.reserve(static_cast<std::size_t>(group_count));
        for (int g = 0; g < group_count; ++g) {
            leaders.push_back(1 + g * config.ranks_per_simulation);
        }
        std::filesystem::create_directories(config.output_directory);
        write_run_manifest(config.output_directory, config, space, world_size, group_count);
        std::cout << "Adaptive hierarchical SMC-ABC\n"
                  << "particles_per_wave=" << config.particles_per_wave << '\n'
                  << "simulation_groups=" << group_count << '\n'
                  << "ranks_per_simulation=" << config.ranks_per_simulation << '\n'
                  << "full_day_seconds=" << config.duration_seconds << '\n'
                  << "fixed_hawkes_activity_scale=" << fixed_hawkes_activity_scale << '\n';
    }

    std::vector<Particle> previous;
    double epsilon = std::numeric_limits<double>::infinity();
    std::uint64_t next_task_id = 1;
    std::mt19937_64 proposal_rng(config.base_seed);
    bool continue_sampling = true;

    for (int wave = 0; wave < config.max_waves && continue_sampling; ++wave) {
        const double wave_start = MPI_Wtime();
        std::array<double, parameter_count> kernel_variance{};
        if (rank == 0) {
            if (wave == 0) kernel_variance.fill(1.0 / 12.0);
            else kernel_variance = diagonal_kernel_variance(previous);
        }

        WaveOutcome outcome;
        if (rank == 0) {
            outcome = master_wave(world, wave, epsilon, leaders, config,
                                  previous, kernel_variance, proposal_rng, next_task_id);
        } else {
            worker_wave(world, group, group_rank, config, space,
                        group_rank == 0 ? &*targets : nullptr);
        }

        check_mpi(MPI_Barrier(world), "MPI_Barrier(after forward map)");

        std::vector<double> weights = distributed_importance_weights(
            world, rank, world_size, wave, previous,
            rank == 0 ? outcome.particles : std::vector<Particle>{}, kernel_variance);

        if (rank == 0) {
            if (wave == 0) {
                const double uniform = 1.0 / static_cast<double>(outcome.particles.size());
                for (Particle& particle : outcome.particles) particle.weight = uniform;
            } else {
                if (weights.size() != outcome.particles.size()) {
                    throw std::runtime_error("Importance-weight result size mismatch");
                }
                for (std::size_t i = 0; i < weights.size(); ++i) outcome.particles[i].weight = weights[i];
            }

            std::vector<double> distances;
            distances.reserve(outcome.particles.size());
            for (const Particle& particle : outcome.particles) distances.push_back(particle.distance);
            const double epsilon_next = ordinary_quantile(distances, config.tolerance_quantile);
            const double wave_wall = MPI_Wtime() - wave_start;

            write_particles(config.output_directory, wave, epsilon, outcome.particles, space);
            write_posterior_summary(config.output_directory, wave, outcome.particles, space);
            write_posterior_correlations(config.output_directory, wave, outcome.particles, space);
            append_diagnostics(config.output_directory, wave, epsilon, epsilon_next,
                               outcome, outcome.particles, wave_wall);

            const double acceptance_rate = static_cast<double>(outcome.particles.size())
                / static_cast<double>(outcome.attempts);
            const double relative_improvement = std::isfinite(epsilon) && epsilon > 0.0
                ? (epsilon - epsilon_next) / epsilon
                : std::numeric_limits<double>::infinity();

            std::cout << "wave=" << wave
                      << " accepted=" << outcome.particles.size()
                      << " attempts=" << outcome.attempts
                      << " acceptance_rate=" << acceptance_rate
                      << " epsilon_used=" << epsilon
                      << " epsilon_next=" << epsilon_next
                      << " ESS=" << effective_sample_size(particle_weights(outcome.particles))
                      << " wall_seconds=" << wave_wall << '\n';

            previous = std::move(outcome.particles);
            continue_sampling = wave + 1 < config.max_waves;
            if (config.final_epsilon > 0.0 && epsilon_next <= config.final_epsilon) {
                continue_sampling = false;
            }
            if (wave > 0 && relative_improvement < config.minimum_relative_epsilon_improvement) {
                continue_sampling = false;
            }
            if (wave > 0 && acceptance_rate < config.minimum_acceptance_rate) {
                continue_sampling = false;
            }
            epsilon = epsilon_next;
        }

        int continue_value = rank == 0 && continue_sampling ? 1 : 0;
        check_mpi(MPI_Bcast(&continue_value, 1, MPI_INT, 0, world),
                  "MPI_Bcast(continue sampling)");
        continue_sampling = continue_value != 0;
        check_mpi(MPI_Barrier(world), "MPI_Barrier(end wave)");
    }

    if (rank != 0) check_mpi(MPI_Comm_free(&group), "MPI_Comm_free(group)");
}

} // namespace dlob::calibration
