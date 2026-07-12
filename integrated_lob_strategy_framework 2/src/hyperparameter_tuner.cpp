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
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {
struct LocalAccum {
    double sum = 0.0;
    double sumsq = 0.0;
    int count = 0;
};

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

    std::vector<double> grid;
    try { grid = hyperparameter_grid(strategy_id); }
    catch (const std::exception& e) {
        if (rank == 0) std::cerr << e.what() << '\n';
        MPI_Finalize();
        return 1;
    }

    constexpr int first_day = 1;
    constexpr int last_day = 800;
    const int n_days = last_day - first_day + 1;
    const int total_jobs = static_cast<int>(grid.size()) * n_days;
    std::vector<LocalAccum> local(grid.size());

    std::filesystem::create_directories("results");
    std::filesystem::create_directories("configs");

    for (int job = rank; job < total_jobs; job += world) {
        const int h = job / n_days;
        const int day = first_day + (job % n_days);
        const StrategyDayResult r = run_strategy_day(strategy_id, grid[static_cast<std::size_t>(h)], day, rank, base_seed, false, "");
        local[static_cast<std::size_t>(h)].sum += r.final_pnl;
        local[static_cast<std::size_t>(h)].sumsq += r.final_pnl * r.final_pnl;
        local[static_cast<std::size_t>(h)].count += 1;
    }

    std::vector<double> local_sum(grid.size()), local_sumsq(grid.size()), global_sum(grid.size()), global_sumsq(grid.size());
    std::vector<int> local_count(grid.size()), global_count(grid.size());
    for (std::size_t i = 0; i < grid.size(); ++i) {
        local_sum[i] = local[i].sum;
        local_sumsq[i] = local[i].sumsq;
        local_count[i] = local[i].count;
    }

    MPI_Reduce(local_sum.data(), global_sum.data(), static_cast<int>(grid.size()), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_sumsq.data(), global_sumsq.data(), static_cast<int>(grid.size()), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_count.data(), global_count.data(), static_cast<int>(grid.size()), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        struct Row { double hp; double mean; double sd; int count; };
        std::vector<Row> rows;
        for (std::size_t i = 0; i < grid.size(); ++i) {
            const int n = std::max(1, global_count[i]);
            const double m = global_sum[i] / static_cast<double>(n);
            double var = 0.0;
            if (n > 1) var = std::max(0.0, (global_sumsq[i] - static_cast<double>(n) * m * m) / static_cast<double>(n - 1));
            rows.push_back(Row{grid[i], m, std::sqrt(var), n});
        }
        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.mean > b.mean; });

        const std::string metrics_file = "results/training_metrics_strategy_" + std::to_string(strategy_id) + ".csv";
        std::ofstream out(metrics_file);
        out << param_label(strategy_id) << ",Mean_PnL_Train,Std_PnL_Train,Days\n";
        out << std::setprecision(12);
        for (const Row& r : rows) out << r.hp << ',' << r.mean << ',' << r.sd << ',' << r.count << '\n';
        out.close();

        const std::string top_file = "configs/top3_strategy_" + std::to_string(strategy_id) + ".txt";
        std::ofstream top(top_file);
        top << "hyperparameter,mean_pnl_train,std_pnl_train\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(3, rows.size()); ++i) top << rows[i].hp << ',' << rows[i].mean << ',' << rows[i].sd << '\n';
        top.close();

        std::cout << "Training complete for strategy " << strategy_id << " (" << strategy_name(strategy_id) << ").\n";
        std::cout << "Wrote " << metrics_file << " and " << top_file << "\n";
    }

    MPI_Finalize();
    return 0;
}
