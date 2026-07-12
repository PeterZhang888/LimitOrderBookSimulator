#pragma once

#include "ExperimentalStrategy.hpp"
#include "SimulationConfig.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct ReplicaSummary {
    int replica_id = 0;
    int strategy_id = 0;
    double hyperparameter = 0.0;
    double final_pnl = 0.0;
    int final_inventory = 0;
    double avg_spread = 0.0;
    double total_traded_volume = 0.0;
    double sharpe_ratio = 0.0;
    double max_drawdown = 0.0;
    double mean_abs_position = 0.0;
};

SimulationConfig calibrated_background_config(std::uint64_t seed, int day_id, std::int64_t duration_seconds = 23400);

StrategyDayResult run_strategy_day(
    int strategy_id,
    double hyperparameter,
    int day_id,
    int replica_id,
    std::uint64_t base_seed,
    bool log_quotes,
    const std::string& quote_log_path
);

ReplicaSummary run_test_replica(
    int strategy_id,
    double hyperparameter,
    int replica_id,
    std::uint64_t base_seed,
    int first_day,
    int last_day,
    bool enable_rank1_quote_log
);

double mean(const std::vector<double>& x);
double sample_stddev(const std::vector<double>& x);
double annualized_sharpe_from_daily_pnl(const std::vector<double>& daily_pnl);
double max_drawdown_from_cumulative_pnl(const std::vector<double>& daily_pnl);
