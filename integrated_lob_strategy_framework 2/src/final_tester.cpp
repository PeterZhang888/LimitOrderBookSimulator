#include "ExperimentRunner.hpp"
#include "ExperimentalStrategy.hpp"

#include <mpi.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {
double read_best_hyperparameter(int strategy_id) {
    const std::string file = "configs/best_config_strategy_" + std::to_string(strategy_id) + ".txt";
    std::ifstream in(file);
    if (!in.is_open()) throw std::runtime_error("Cannot open " + file + ". Run validator first.");
    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string key, value;
        if (std::getline(ss, key, ',') && std::getline(ss, value)) {
            if (key == "hyperparameter") return std::stod(value);
        }
    }
    throw std::runtime_error("No hyperparameter row found in " + file);
}

void write_replica_tmp(const ReplicaSummary& r, int world_size) {
    std::filesystem::create_directories("results/tmp_test");
    const std::string file = "results/tmp_test/replica_" + std::to_string(r.replica_id) + "_" + std::to_string(world_size) + ".csv";
    std::ofstream out(file);
    out << "Replica_ID,Final_PnL,Final_Inventory,Avg_Spread,Total_Traded_Volume,Sharpe_Ratio,Max_Drawdown,Mean_Abs_Position\n";
    out << std::setprecision(12)
        << r.replica_id << ',' << r.final_pnl << ',' << r.final_inventory << ','
        << r.avg_spread << ',' << r.total_traded_volume << ',' << r.sharpe_ratio << ','
        << r.max_drawdown << ',' << r.mean_abs_position << '\n';
}
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    const int strategy_id = argc >= 2 ? std::atoi(argv[1]) : 1;
    const unsigned long long base_seed = argc >= 3 ? std::strtoull(argv[2], nullptr, 10) : 12345ULL;

    double hyperparameter = 0.0;
    if (rank == 0) {
        try { hyperparameter = read_best_hyperparameter(strategy_id); }
        catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::filesystem::create_directories("results");
        std::filesystem::create_directories("logs");
        std::filesystem::create_directories("results/tmp_test");
    }
    MPI_Bcast(&hyperparameter, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    const double start = MPI_Wtime();

    // Rank 0 is master. For world_size == 1, rank 0 also runs one replica so debugging still works.
    if (rank > 0 || world == 1) {
        const int replica_id = rank;
        ReplicaSummary summary = run_test_replica(strategy_id, hyperparameter, replica_id, base_seed, 1001, 1300, true);
        write_replica_tmp(summary, world);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    if (rank == 0) {
        const std::string final_file = "results/final_test_" + std::to_string(strategy_id) + "_" + std::to_string(world) + "ranks.csv";
        std::ofstream out(final_file);
        out << "Replica_ID,Final_PnL,Final_Inventory,Avg_Spread,Total_Traded_Volume,Sharpe_Ratio,Max_Drawdown,Mean_Abs_Position\n";
        const int first_replica = world == 1 ? 0 : 1;
        for (int r = first_replica; r < world; ++r) {
            const std::string tmp = "results/tmp_test/replica_" + std::to_string(r) + "_" + std::to_string(world) + ".csv";
            std::ifstream in(tmp);
            std::string line;
            bool first_line = true;
            while (std::getline(in, line)) {
                if (first_line) { first_line = false; continue; }
                if (!line.empty()) out << line << '\n';
            }
            in.close();
            std::filesystem::remove(tmp);
        }
        out.close();
        std::filesystem::remove_all("results/tmp_test");

        std::ofstream timing("results/final_test_timing_" + std::to_string(strategy_id) + "_" + std::to_string(world) + "ranks.txt");
        timing << "strategy_id," << strategy_id << "\n";
        timing << "strategy_name," << strategy_name(strategy_id) << "\n";
        timing << "mpi_ranks," << world << "\n";
        timing << "worker_replicas," << (world == 1 ? 1 : world - 1) << "\n";
        timing << "test_days,300\n";
        timing << "duration_seconds_per_day,23400\n";
        timing << "elapsed_seconds," << std::setprecision(12) << (end - start) << "\n";
        timing.close();

        std::cout << "Final test complete. Wrote " << final_file << "\n";
    }

    MPI_Finalize();
    return 0;
}
