#include "CalibrationSearchSpace.hpp"
#include "FullSimulator.hpp"
#include "LossFunction.hpp"
#include "MomentEstimator.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::string join_csv_values(const std::vector<double>& values) {
    std::ostringstream ss;
    ss << std::setprecision(12);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) ss << ",";
        ss << values[i];
    }
    return ss.str();
}

std::vector<double> average_moments(const std::vector<std::vector<NamedMoment>>& repeats) {
    const auto& names = standard_moment_names();
    std::vector<double> avg(names.size(), 0.0);
    if (repeats.empty()) return avg;
    for (const auto& moments : repeats) {
        for (std::size_t i = 0; i < names.size(); ++i) {
            avg[i] += get_moment_value(moments, names[i], 0.0);
        }
    }
    for (double& x : avg) x /= static_cast<double>(repeats.size());
    return avg;
}

std::vector<NamedMoment> named_from_values(const std::vector<double>& values) {
    const auto& names = standard_moment_names();
    std::vector<NamedMoment> result;
    result.reserve(names.size());
    for (std::size_t i = 0; i < names.size(); ++i) {
        result.push_back(NamedMoment{names[i], i < values.size() ? values[i] : 0.0});
    }
    return result;
}

void write_result_header(std::ofstream& out, const ParameterVector& p) {
    out << "candidate_id,loss";
    for (const auto& name : p.names) out << "," << name;
    for (const auto& name : standard_moment_names()) out << "," << name;
    out << "\n";
}

void append_result_row(std::ofstream& out, int candidate_id, double loss, const ParameterVector& p, const std::vector<double>& moments) {
    out << candidate_id << "," << std::setprecision(12) << loss;
    for (double v : p.values) out << "," << v;
    for (double v : moments) out << "," << v;
    out << "\n";
}

struct LocalBest {
    double loss = std::numeric_limits<double>::infinity();
    int candidate_id = -1;
    std::vector<double> param_values;
    std::vector<double> moments;
    std::vector<NamedMoment> loss_terms;
};
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int n_candidates = argc >= 2 ? std::max(1, std::atoi(argv[1])) : 100;
    const int repeats = argc >= 3 ? std::max(1, std::atoi(argv[2])) : 3;
    const int duration_seconds = argc >= 4 ? std::max(60, std::atoi(argv[3])) : 3600;
    const std::uint64_t base_seed = argc >= 5 ? static_cast<std::uint64_t>(std::strtoull(argv[4], nullptr, 10)) : 12345ULL;
    const std::string target_file = argc >= 6 ? argv[5] : std::string("empirical_moments.csv");

    std::vector<LossTarget> targets;
    try {
        targets = load_loss_targets(target_file);
    } catch (const std::exception& ex) {
        if (rank == 0) std::cerr << "Error: " << ex.what() << "\n";
        MPI_Finalize();
        return 1;
    }

    CalibrationSearchSpace search_space(n_candidates);
    const std::int64_t end_time_ns = static_cast<std::int64_t>(duration_seconds) * 1000LL * 1000LL * 1000LL;

    if (rank == 0) {
        std::filesystem::create_directories("results");
        std::cout << "Full Hawkes-background ABM calibration started.\n"
                  << "MPI ranks: " << world_size << "\n"
                  << "Candidates: " << n_candidates << "\n"
                  << "Repeats per candidate: " << repeats << "\n"
                  << "Duration seconds per simulation: " << duration_seconds << "\n"
                  << "Target file: " << target_file << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    LocalBest best;
    const std::string tmp_file = "results/tmp_rank_" + std::to_string(rank) + ".csv";
    std::ofstream local_out(tmp_file);
    write_result_header(local_out, search_space.parameter_vector(0));

    for (int candidate_id = rank; candidate_id < n_candidates; candidate_id += world_size) {
        std::vector<std::vector<NamedMoment>> repeat_moments;
        repeat_moments.reserve(static_cast<std::size_t>(repeats));

        for (int repeat = 0; repeat < repeats; ++repeat) {
            SimulationConfig config = search_space.make_config(candidate_id, repeat, base_seed, end_time_ns);
            SimulationMetrics metrics = run_full_simulation(config);
            repeat_moments.push_back(estimate_moments(metrics));
        }

        const std::vector<double> avg_values = average_moments(repeat_moments);
        const std::vector<NamedMoment> avg_named = named_from_values(avg_values);
        const LossBreakdown loss = compute_loss(avg_named, targets);
        const ParameterVector p = search_space.parameter_vector(candidate_id);
        append_result_row(local_out, candidate_id, loss.total_loss, p, avg_values);

        if (loss.total_loss < best.loss) {
            best.loss = loss.total_loss;
            best.candidate_id = candidate_id;
            best.param_values = p.values;
            best.moments = avg_values;
            best.loss_terms = loss.loss_terms;
        }

        std::cout << "rank " << rank << " candidate " << candidate_id << " loss " << loss.total_loss << "\n";
    }
    local_out.close();

    struct { double value; int rank; } local_min, global_min;
    local_min.value = best.loss;
    local_min.rank = rank;
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    int winning_rank = 0;
    if (rank == 0) {
        winning_rank = global_min.rank;
    }
    MPI_Bcast(&winning_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    if (rank == 0) {
        std::vector<std::string> all_lines;
        bool wrote_header = false;
        std::ofstream merged("results/calibration_results.csv");
        for (int r = 0; r < world_size; ++r) {
            const std::string file = "results/tmp_rank_" + std::to_string(r) + ".csv";
            std::ifstream in(file);
            std::string line;
            bool first = true;
            while (std::getline(in, line)) {
                if (first) {
                    if (!wrote_header) {
                        merged << line << "\n";
                        wrote_header = true;
                    }
                    first = false;
                    continue;
                }
                if (!line.empty()) merged << line << "\n";
            }
            in.close();
            std::remove(file.c_str());
        }
        merged.close();
    }

    // The winning rank writes a compact best payload, then rank 0 reads it after a barrier.
    if (rank == winning_rank) {
        std::ofstream out("results/best_payload.csv");
        out << "candidate_id,loss\n" << best.candidate_id << "," << std::setprecision(12) << best.loss << "\n";
        out << "parameters\n" << join_csv_values(best.param_values) << "\n";
        out << "moments\n" << join_csv_values(best.moments) << "\n";
        out << "loss_terms\n";
        for (std::size_t i = 0; i < best.loss_terms.size(); ++i) {
            if (i > 0) out << ",";
            out << best.loss_terms[i].value;
        }
        out << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        const double elapsed = end - start;
        std::cout << "Total calibration time with " << world_size << " ranks: " << std::fixed << std::setprecision(3) << elapsed << " seconds\n";

        std::ofstream timing("results/timing_ranks_" + std::to_string(world_size) + ".txt");
        timing << "Total calibration time with " << world_size << " ranks: " << std::fixed << std::setprecision(3) << elapsed << " seconds\n";
        timing << "mpi_ranks," << world_size << "\n";
        timing << "candidates," << n_candidates << "\n";
        timing << "repeats," << repeats << "\n";
        timing << "duration_seconds," << duration_seconds << "\n";
        timing.close();

        std::ifstream payload("results/best_payload.csv");
        std::string line;
        std::vector<std::string> lines;
        while (std::getline(payload, line)) lines.push_back(line);
        payload.close();

        const ParameterVector p0 = search_space.parameter_vector(0);
        std::ofstream best_config("results/best_config.txt");
        best_config << "BEST FULL SIMULATOR CONFIGURATION\n";
        best_config << "=================================\n";
        if (lines.size() >= 2) best_config << lines[1] << "\n\n";
        best_config << "Parameters:\n";
        if (lines.size() >= 4) {
            std::stringstream ss(lines[3]);
            std::string cell;
            std::size_t i = 0;
            while (std::getline(ss, cell, ',') && i < p0.names.size()) {
                best_config << p0.names[i] << ": " << cell << "\n";
                ++i;
            }
        }
        best_config.close();

        std::ofstream detail("results/best_loss_detail.csv");
        detail << "name,value\n";
        if (lines.size() >= 6) {
            std::stringstream ss(lines[5]);
            std::string cell;
            std::size_t i = 0;
            for (const auto& name : standard_moment_names()) {
                if (!std::getline(ss, cell, ',')) break;
                detail << name << "," << cell << "\n";
                ++i;
            }
        }
        detail << "\nloss_term,value\n";
        if (lines.size() >= 8) {
            std::stringstream ss(lines[7]);
            std::string cell;
            std::size_t i = 0;
            for (const LossTarget& t : targets) {
                if (!std::getline(ss, cell, ',')) break;
                detail << "loss_" << t.name << "," << cell << "\n";
                ++i;
            }
        }
        detail.close();
        std::remove("results/best_payload.csv");
    }

    MPI_Finalize();
    return 0;
}
