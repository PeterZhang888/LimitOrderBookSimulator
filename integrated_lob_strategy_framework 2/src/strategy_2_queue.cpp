#include "ExperimentRunner.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    const double eta = argc >= 2 ? std::atof(argv[1]) : 1.0;
    const int day_id = argc >= 3 ? std::atoi(argv[2]) : 1;
    const int replica_id = argc >= 4 ? std::atoi(argv[3]) : 0;
    const unsigned long long base_seed = argc >= 5 ? std::strtoull(argv[4], nullptr, 10) : 12345ULL;
    std::filesystem::create_directories("results");
    StrategyDayResult r = run_strategy_day(2, eta, day_id, replica_id, base_seed, false, "");
    std::cout << std::setprecision(12)
              << "strategy_id,day_id,replica_id,eta,pnl,inventory,avg_spread,total_volume,sharpe\n"
              << r.strategy_id << ',' << r.day_id << ',' << r.replica_id << ',' << r.hyperparameter << ','
              << r.final_pnl << ',' << r.final_inventory << ',' << r.avg_spread << ','
              << r.total_traded_volume << ',' << r.sharpe_ratio << '\n';
    return 0;
}
