#include "ExperimentRunner.hpp"
#include "ExperimentalStrategy.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::vector<double> read_top3(int strategy_id) {
    const std::string file = "configs/top3_strategy_" + std::to_string(strategy_id) + ".txt";
    std::ifstream in(file);
    if (!in.is_open()) throw std::runtime_error("Cannot open " + file + ". Run tuner first.");
    std::vector<double> hp;
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (first) { first = false; continue; }
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        if (std::getline(ss, cell, ',')) hp.push_back(std::stod(cell));
    }
    return hp;
}

std::string param_label(int strategy_id) {
    if (strategy_id == 1) return "Gamma";
    if (strategy_id == 2) return "Eta";
    if (strategy_id == 3) return "Theta";
    return "Hyperparameter";
}
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    const int strategy_id = argc >= 2 ? std::atoi(argv[1]) : 1;
    const unsigned long long base_seed = argc >= 3 ? std::strtoull(argv[2], nullptr, 10) : 12345ULL;

    std::vector<double> candidates;
    try { candidates = read_top3(strategy_id); }
    catch (const std::exception& e) {
        if (rank == 0) std::cerr << e.what() << '\n';
        MPI_Finalize();
        return 1;
    }
    if (candidates.empty()) {
        if (rank == 0) std::cerr << "No validation candidates found.\n";
        MPI_Finalize();
        return 1;
    }

    constexpr int first_day = 801;
    constexpr int last_day = 1000;
    const int n_days = last_day - first_day + 1;
    const int total_jobs = static_cast<int>(candidates.size()) * n_days;

    std::vector<double> local_sum(candidates.size(), 0.0), local_sumsq(candidates.size(), 0.0);
    std::vector<int> local_count(candidates.size(), 0);

    for (int job = rank; job < total_jobs; job += world) {
        const int h = job / n_days;
        const int day = first_day + (job % n_days);
        StrategyDayResult r = run_strategy_day(strategy_id, candidates[static_cast<std::size_t>(h)], day, rank, base_seed, false, "");
        local_sum[static_cast<std::size_t>(h)] += r.final_pnl;
        local_sumsq[static_cast<std::size_t>(h)] += r.final_pnl * r.final_pnl;
        local_count[static_cast<std::size_t>(h)] += 1;
    }

    std::vector<double> global_sum(candidates.size()), global_sumsq(candidates.size());
    std::vector<int> global_count(candidates.size());
    MPI_Reduce(local_sum.data(), global_sum.data(), static_cast<int>(candidates.size()), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_sumsq.data(), global_sumsq.data(), static_cast<int>(candidates.size()), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_count.data(), global_count.data(), static_cast<int>(candidates.size()), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::filesystem::create_directories("results");
        std::filesystem::create_directories("configs");
        struct Row { double hp; double mean; double sharpe; double max_dd; int count; };
        std::vector<Row> rows;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            const int n = std::max(1, global_count[i]);
            const double m = global_sum[i] / static_cast<double>(n);
            double var = 0.0;
            if (n > 1) var = std::max(0.0, (global_sumsq[i] - static_cast<double>(n) * m * m) / static_cast<double>(n - 1));
            const double sd = std::sqrt(var);
            const double sharpe = sd > 1e-12 ? std::sqrt(252.0) * m / sd : 0.0;
            rows.push_back(Row{candidates[i], m, sharpe, 0.0, n});
        }
        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.sharpe > b.sharpe; });

        const std::string metrics_file = "results/validation_metrics_strategy_" + std::to_string(strategy_id) + ".csv";
        std::ofstream out(metrics_file);
        out << param_label(strategy_id) << ",Mean_PnL_Val,Sharpe_Val,Max_Drawdown_Val,Days\n" << std::setprecision(12);
        for (const Row& r : rows) out << r.hp << ',' << r.mean << ',' << r.sharpe << ',' << r.max_dd << ',' << r.count << '\n';
        out.close();

        const std::string best_file = "configs/best_config_strategy_" + std::to_string(strategy_id) + ".txt";
        std::ofstream best(best_file);
        best << "strategy_id," << strategy_id << "\n";
        best << "strategy_name," << strategy_name(strategy_id) << "\n";
        best << "hyperparameter," << rows.front().hp << "\n";
        best << "validation_mean_pnl," << rows.front().mean << "\n";
        best << "validation_sharpe," << rows.front().sharpe << "\n";
        best.close();

        std::cout << "Validation complete. Best config written to " << best_file << "\n";
    }

    MPI_Finalize();
    return 0;
}
